"""CUDA-accelerated, run-safe recalibration and evaluation for Rieth TEP.

The scientific model parameters and mask generation remain the NumPy reference
implementation.  Only repeated posterior filtering is batched on CUDA.  This
module also shares an identical posterior calculation across methods after
verifying that their independently fitted monitor parameters match exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import numpy as np

from .baselines import CalibratedPointMonitor, _mask_only_context_features
from .datasets import SensorSequence
from .experiment import _config_dict, _monitoring_metrics, _sequence_summary
from .imputation import (
    imputation_error,
    linear_interpolate_offline,
    locf_impute,
    median_impute,
)
from .missingness import (
    MissingnessConfig,
    apply_mask,
    gap_lengths,
    generate_observed_mask,
    scenario_cycle,
    severity_labels,
)
from .model import MAPIC
from .monitor import ScoreResult, _quadratic_at_mean
from .multirun import FittedMultiRunSuite
from .state_space import FilterResult
from .torch_backend import score_incomplete_batch
from .utils import EPS


T = TypeVar("T")


@dataclass
class CUDAExecutionTracker:
    """Auditable aggregate of CUDA kernel calls made by one experiment."""

    calls: int = 0
    sequences: int = 0
    time_steps: int = 0
    cuda_elapsed_ms: float = 0.0
    fallback_sequences: int = 0
    factorization_failures: int = 0
    device: str | None = None
    device_name: str | None = None
    dtype: str | None = None

    def update(self, result: dict[str, object]) -> None:
        batch_size = int(result["batch_size"])
        sequence_length = int(result["sequence_length"])
        self.calls += 1
        self.sequences += batch_size
        self.time_steps += batch_size * sequence_length
        self.cuda_elapsed_ms += float(result["elapsed_ms"])
        self.fallback_sequences += int(
            result.get("fallback_sequence_count", 0)
        )
        self.factorization_failures += int(
            result.get("factorization_failure_count", 0)
        )
        self.device = str(result["device"])
        self.device_name = str(result["device_name"])
        self.dtype = str(result["dtype"])

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": "pytorch_cuda_compact_batch",
            "device": self.device,
            "device_name": self.device_name,
            "dtype": self.dtype,
            "kernel_calls": self.calls,
            "sequences": self.sequences,
            "time_steps": self.time_steps,
            "cuda_event_elapsed_seconds": self.cuda_elapsed_ms / 1000.0,
            "fallback_sequences": self.fallback_sequences,
            "factorization_failures": self.factorization_failures,
            "parameter_fitting_backend": "numpy_cpu",
            "mask_generation_backend": "numpy_cpu",
            "tf32": False,
        }


def cuda_runtime_metadata(device: str = "cuda") -> dict[str, object]:
    """Return explicit PyTorch/CUDA/GPU provenance or fail without CUDA."""
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("a CUDA-enabled PyTorch installation is required") from error
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")
    properties = torch.cuda.get_device_properties(target)
    return {
        "backend": "pytorch_cuda_compact_batch",
        "requested_device": device,
        "resolved_device": str(target),
        "device_name": torch.cuda.get_device_name(target),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float64",
        "tf32": False,
        "deterministic_masks": True,
    }


def _chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError("CUDA batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _all_monitors(suite: FittedMultiRunSuite) -> tuple[object, ...]:
    return (
        suite.proposed.monitor,
        suite.integrated.monitor,
        suite.mask_only.monitor,
        *(baseline.monitor for baseline in suite.baselines.values()),
    )


def verify_shared_monitor_parameters(suite: FittedMultiRunSuite) -> None:
    """Prove that one posterior path can be reused by every fitted method."""
    monitors = _all_monitors(suite)
    reference = monitors[0]
    filter_names = (
        "location_",
        "scale_",
        "intercept_",
        "transition_",
        "innovation_covariance_",
        "initial_covariance_",
        "exogenous_coefficients_",
    )
    monitor_names = (
        "baseline_intercept_",
        "baseline_coefficients_",
        "t2_matrix_",
        "spe_matrix_",
        "combined_matrix_",
        "component_scales_",
        "score_center_",
        "score_scale_",
    )
    for candidate in monitors[1:]:
        for name in filter_names:
            left = np.asarray(getattr(reference.filter_model, name))
            right = np.asarray(getattr(candidate.filter_model, name))
            if not np.array_equal(left, right):
                raise ValueError(
                    f"fitted monitor filters differ at {name}; CUDA sharing is invalid"
                )
        for name in monitor_names:
            left = np.asarray(getattr(reference, name))
            right = np.asarray(getattr(candidate, name))
            if not np.array_equal(left, right):
                raise ValueError(
                    f"fitted monitor parameters differ at {name}; CUDA sharing is invalid"
                )


def _score_from_components(
    monitor: object,
    *,
    mean_z: np.ndarray,
    reconstructed: np.ndarray,
    normalized_innovation: np.ndarray,
    mask: np.ndarray,
    t2_location: np.ndarray,
    spe_location: np.ndarray,
    t2_uncertainty: np.ndarray,
    spe_uncertainty: np.ndarray,
    location_energy: np.ndarray,
    uncertainty_energy: np.ndarray,
    posterior_trace: np.ndarray,
    integrate_uncertainty: bool,
    elapsed_start: int = 0,
) -> ScoreResult:
    t2 = t2_location + t2_uncertainty if integrate_uncertainty else t2_location
    spe = spe_location + spe_uncertainty if integrate_uncertainty else spe_location
    combined = (
        location_energy + uncertainty_energy
        if integrate_uncertainty
        else location_energy
    )
    instantaneous = (
        np.log1p(combined) - float(getattr(monitor, "score_center_"))
    ) / float(getattr(monitor, "score_scale_"))
    total_energy = location_energy + uncertainty_energy
    uncertainty_fraction = np.divide(
        uncertainty_energy,
        total_energy,
        out=np.zeros_like(uncertainty_energy),
        where=total_energy > EPS,
    )
    longest_gap = gap_lengths(mask).max(axis=1)
    context_features = np.column_stack(
        [
            (~mask).mean(axis=1),
            np.log1p(longest_gap),
            np.log1p(posterior_trace),
            np.log1p(uncertainty_energy),
            np.log1p(elapsed_start + np.arange(mask.shape[0], dtype=float)),
        ]
    )
    compact_filter = FilterResult(
        mean_z=mean_z,
        covariance_z=np.empty((0, mean_z.shape[1], mean_z.shape[1])),
        reconstructed=reconstructed,
        normalized_innovation=normalized_innovation,
        observed_mask=mask,
    )
    return ScoreResult(
        fused=instantaneous,
        t2=t2,
        spe=spe,
        t2_location=t2_location,
        t2_uncertainty=t2_uncertainty,
        spe_location=spe_location,
        spe_uncertainty=spe_uncertainty,
        innovation=normalized_innovation,
        context=severity_labels(mask),
        posterior_trace=posterior_trace,
        location_energy=location_energy,
        uncertainty_energy=uncertainty_energy,
        uncertainty_fraction=uncertainty_fraction,
        context_features=context_features,
        filter_result=compact_filter,
    )


def _score_complete(monitor: object, x: np.ndarray) -> ScoreResult:
    """Exact fast path for fully observed inputs; posterior covariance is zero."""
    values = np.asarray(x, dtype=float)
    filter_model = getattr(monitor, "filter_model")
    mean_z = filter_model.standardize(values)
    baseline = np.broadcast_to(
        np.asarray(getattr(monitor, "baseline_intercept_")), mean_z.shape
    )
    monitoring_mean = mean_z - baseline
    t2_location = _quadratic_at_mean(
        monitoring_mean, np.asarray(getattr(monitor, "t2_matrix_"))
    )
    spe_location = _quadratic_at_mean(
        monitoring_mean, np.asarray(getattr(monitor, "spe_matrix_"))
    )
    location_energy = _quadratic_at_mean(
        monitoring_mean, np.asarray(getattr(monitor, "combined_matrix_"))
    )
    zeros = np.zeros(values.shape[0], dtype=float)
    mask = np.ones_like(values, dtype=bool)
    return _score_from_components(
        monitor,
        mean_z=mean_z,
        reconstructed=values.copy(),
        normalized_innovation=zeros,
        mask=mask,
        t2_location=t2_location,
        spe_location=spe_location,
        t2_uncertainty=zeros,
        spe_uncertainty=zeros,
        location_energy=location_energy,
        uncertainty_energy=zeros,
        posterior_trace=zeros,
        integrate_uncertainty=False,
    )


def _scores_from_cuda_result(
    monitor: object,
    result: dict[str, object],
    masks: np.ndarray,
    *,
    integrate_uncertainty: bool,
) -> list[ScoreResult]:
    fields = {
        name: np.asarray(result[name])
        for name in (
            "mean_z",
            "reconstructed",
            "normalized_innovation",
            "t2_location",
            "spe_location",
            "t2_uncertainty",
            "spe_uncertainty",
            "location_energy",
            "uncertainty_energy",
            "posterior_trace",
        )
    }
    scores: list[ScoreResult] = []
    for index, mask in enumerate(masks):
        scores.append(
            _score_from_components(
                monitor,
                mean_z=fields["mean_z"][index],
                reconstructed=fields["reconstructed"][index],
                normalized_innovation=fields["normalized_innovation"][index],
                mask=mask,
                t2_location=fields["t2_location"][index],
                spe_location=fields["spe_location"][index],
                t2_uncertainty=fields["t2_uncertainty"][index],
                spe_uncertainty=fields["spe_uncertainty"][index],
                location_energy=fields["location_energy"][index],
                uncertainty_energy=fields["uncertainty_energy"][index],
                posterior_trace=fields["posterior_trace"][index],
                integrate_uncertainty=integrate_uncertainty,
            )
        )
    return scores


@dataclass(frozen=True)
class _MaskTask:
    run_index: int
    run: SensorSequence
    mechanism: str
    mask: np.ndarray


def _validate_runs(runs: Sequence[SensorSequence], label: str) -> None:
    if not runs:
        raise ValueError(f"{label} requires at least one run")
    dimensions = {run.x.shape[1] for run in runs}
    lengths = {run.x.shape[0] for run in runs}
    if len(dimensions) != 1 or len(lengths) != 1:
        raise ValueError(f"{label} runs must have a common shape")
    for run in runs:
        if run.fault.any():
            raise ValueError(f"{label} run {run.name} is not healthy")
        if run.covariates is not None:
            raise NotImplementedError(
                "the CUDA multi-run path currently supports Rieth runs without covariates"
            )
        if not np.isfinite(run.x).all():
            raise ValueError(f"{label} run {run.name} contains non-finite truth")
        mask = (
            np.ones_like(run.x, dtype=bool)
            if run.observed_mask is None
            else np.asarray(run.observed_mask, dtype=bool)
        )
        if mask.shape != run.x.shape or not mask.all():
            raise ValueError(
                f"{label} run {run.name} must be completely observed before stress masks"
            )


def _cuda_task_scores(
    monitor: object,
    tasks: Sequence[_MaskTask],
    *,
    device: str,
    batch_size: int,
    tracker: CUDAExecutionTracker,
) -> Iterator[tuple[_MaskTask, ScoreResult, ScoreResult]]:
    for batch in _chunks(tasks, batch_size):
        values = np.stack([apply_mask(task.run.x, task.mask) for task in batch])
        masks = np.stack([task.mask for task in batch])
        raw = score_incomplete_batch(
            monitor,
            values,
            masks,
            device=device,
            dtype="float64",
        )
        tracker.update(raw)
        location_scores = _scores_from_cuda_result(
            monitor,
            raw,
            masks,
            integrate_uncertainty=False,
        )
        integrated_scores = _scores_from_cuda_result(
            monitor,
            raw,
            masks,
            integrate_uncertainty=True,
        )
        yield from zip(batch, location_scores, integrated_scores)


def _core_decision(
    model: MAPIC,
    score: ScoreResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, unknown, persistence = model._decision_statistic(score)
    alarm = (persistence > model.calibrator.threshold_) & ~unknown
    return alarm, unknown, persistence


def _baseline_decision(
    baseline: CalibratedPointMonitor,
    score: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _mask_only_context_features(mask)
    _, _, unknown, persistence = baseline._decision_statistic(
        score, features, mask
    )
    alarm = (persistence > baseline.calibrator.threshold_) & ~unknown
    return alarm, unknown, persistence


def _baseline_completed(
    baseline: CalibratedPointMonitor,
    run: SensorSequence,
    mask: np.ndarray,
) -> np.ndarray:
    incomplete = apply_mask(run.x, mask)
    if baseline.method == "median":
        return median_impute(incomplete, baseline.reference_, mask)
    if baseline.method == "locf":
        return locf_impute(incomplete, baseline.reference_, mask)
    if baseline.method == "linear_offline":
        return linear_interpolate_offline(incomplete, baseline.reference_, mask)
    raise ValueError("kalman_mean uses the shared posterior reconstruction")


def _calibration_diagnostics(
    records: Sequence[tuple[str, str, np.ndarray]],
    calibrator: object,
) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for mechanism, run_name, statistic in records:
        grouped.setdefault((run_name, mechanism), []).append(
            float(np.max(statistic))
        )
    run_names = sorted({run_name for run_name, _ in grouped})
    mechanisms = sorted({mechanism for _, mechanism in grouped})
    mechanism_envelopes = {
        mechanism: np.asarray(
            [max(grouped[(run_name, mechanism)]) for run_name in run_names]
        )
        for mechanism in mechanisms
    }
    family = np.asarray(
        [
            max(
                max(grouped[(run_name, mechanism)])
                for mechanism in mechanisms
            )
            for run_name in run_names
        ]
    )
    order = np.argsort(family, kind="mergesort")
    sorted_family = family[order]
    alpha = float(getattr(calibrator, "alpha"))
    rank = min(
        int(np.ceil((len(run_names) + 1) * (1.0 - alpha))),
        len(run_names),
    )
    driver_run = run_names[int(order[rank - 1])]
    driver_mechanism = max(
        mechanisms,
        key=lambda mechanism: max(grouped[(driver_run, mechanism)]),
    )
    driver_values = grouped[(driver_run, driver_mechanism)]
    driver_repeat = int(np.argmax(driver_values)) + 1
    selected_orders = sorted(
        {
            1,
            rank,
            len(run_names),
            *(
                value
                for value in (90, 95, 96, 99, 100)
                if value <= len(run_names)
            ),
        }
    )
    mechanism_thresholds = dict(
        getattr(calibrator, "mechanism_thresholds_")
    )
    base_threshold = mechanism_thresholds.get("base_observed")
    pooled = float(getattr(calibrator, "threshold_"))
    return {
        "healthy_run_count": len(run_names),
        "conformal_rank_one_based": rank,
        "family_order_statistics": {
            str(position): float(sorted_family[position - 1])
            for position in selected_orders
        },
        "family_minimum": float(sorted_family[0]),
        "family_maximum": float(sorted_family[-1]),
        "pooled_threshold": pooled,
        "mechanism_thresholds": mechanism_thresholds,
        "pooled_to_base_ratio": (
            pooled / float(base_threshold)
            if base_threshold is not None and float(base_threshold) != 0.0
            else None
        ),
        "rank_driver": {
            "run": driver_run,
            "mechanism": driver_mechanism,
            "repeat_one_based": driver_repeat,
            "score": float(sorted_family[rank - 1]),
        },
        "mechanism_family_ranges": {
            mechanism: {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
            for mechanism, values in mechanism_envelopes.items()
        },
    }


def _method_models(
    suite: FittedMultiRunSuite,
) -> dict[str, object]:
    return {
        "UD-MAPIC": suite.proposed,
        "MAPIC_integrated_expectation": suite.integrated,
        "UD-MAPIC_mask_only_context": suite.mask_only,
        **suite.baselines,
    }


def recalibrate_multirun_suite_cuda(
    suite: FittedMultiRunSuite,
    calibration_runs: Iterable[SensorSequence],
    *,
    device: str = "cuda",
    batch_size: int = 32,
    progress: Callable[[str], None] | None = None,
) -> tuple[FittedMultiRunSuite, dict[str, object]]:
    """Replace only conformal calibrators using shared batched CUDA scores."""
    runs = tuple(calibration_runs)
    _validate_runs(runs, "CUDA calibration")
    verify_shared_monitor_parameters(suite)
    if not suite.design_run_names:
        raise ValueError("warm-start suite has no design-run provenance")

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    old_summaries = {
        name: dict(model.calibration_summary_)
        for name, model in _method_models(suite).items()
    }
    records: dict[str, list[tuple[str, str, np.ndarray]]] = {
        name: [] for name in _method_models(suite)
    }
    reference_monitor = suite.proposed.monitor
    announce(f"CUDA recalibration: {len(runs)} base-observed runs")
    for run in runs:
        score = _score_complete(reference_monitor, run.x)
        for name, model in (
            ("UD-MAPIC", suite.proposed),
            ("MAPIC_integrated_expectation", suite.integrated),
            ("UD-MAPIC_mask_only_context", suite.mask_only),
        ):
            _, _, persistence = _core_decision(model, score)
            records[name].append(("base_observed", run.name, persistence))
        for method, baseline in suite.baselines.items():
            _, _, persistence = _baseline_decision(
                baseline, score.fused, np.ones_like(run.x, dtype=bool)
            )
            records[method].append(("base_observed", run.name, persistence))

    configs = scenario_cycle(
        suite.config.scenarios, suite.config.calibration_repeats
    )
    tasks: list[_MaskTask] = []
    for run_index, run in enumerate(runs):
        for config_index, scenario in enumerate(configs):
            seed = (
                suite.config.seed
                + 1000003
                + 10000019 * run_index
                + 1013 * (config_index + 1)
            )
            mask = generate_observed_mask(run.x, scenario, seed=seed)
            tasks.append(
                _MaskTask(
                    run_index=run_index,
                    run=run,
                    mechanism=f"base+{scenario.identifier()}",
                    mask=mask,
                )
            )
    tracker = CUDAExecutionTracker()
    announce(
        f"CUDA recalibration: {len(tasks)} augmented sequences "
        f"in batches of {batch_size}"
    )
    scored_tasks = _cuda_task_scores(
        reference_monitor,
        tasks,
        device=device,
        batch_size=batch_size,
        tracker=tracker,
    )
    for completed, (task, location_score, integrated_score) in enumerate(
        scored_tasks,
        start=1,
    ):
        for name, model, score in (
            ("UD-MAPIC", suite.proposed, location_score),
            (
                "MAPIC_integrated_expectation",
                suite.integrated,
                integrated_score,
            ),
            (
                "UD-MAPIC_mask_only_context",
                suite.mask_only,
                location_score,
            ),
        ):
            _, _, persistence = _core_decision(model, score)
            records[name].append((task.mechanism, task.run.name, persistence))
        for method, baseline in suite.baselines.items():
            if method == "kalman_mean":
                baseline_score = location_score.fused
            else:
                completed_values = _baseline_completed(
                    baseline, task.run, task.mask
                )
                baseline_score = _score_complete(
                    reference_monitor, completed_values
                ).fused
            _, _, persistence = _baseline_decision(
                baseline, baseline_score, task.mask
            )
            records[method].append(
                (task.mechanism, task.run.name, persistence)
            )
        if completed % max(batch_size * 5, 1) == 0:
            announce(f"CUDA recalibration progress: {completed}/{len(tasks)}")

    diagnostics: dict[str, object] = {}
    for name, model in _method_models(suite).items():
        model.calibrator.fit_run_clusters(records[name])
        model.phase1_calibration_run_count_ = len(runs)
        after = _calibration_diagnostics(records[name], model.calibrator)
        before_threshold = float(old_summaries[name]["threshold"])
        after_threshold = float(model.calibrator.threshold_)
        diagnostics[name] = {
            "before": old_summaries[name],
            "after": after,
            "threshold_change": {
                "absolute": after_threshold - before_threshold,
                "relative": (
                    after_threshold / before_threshold - 1.0
                    if before_threshold != 0.0
                    else None
                ),
            },
        }
    updated = replace(
        suite,
        calibration_run_names=tuple(run.name for run in runs),
    )
    return updated, {
        "cuda_execution": tracker.as_dict(),
        "calibration_diagnostics": diagnostics,
        "scientific_change": "healthy calibration runs increased; design fit frozen",
    }


def validate_cuda_equivalence(
    suite: FittedMultiRunSuite,
    run: SensorSequence,
    scenario: MissingnessConfig,
    *,
    seed: int = 9187,
    device: str = "cuda",
    absolute_tolerance: float = 2e-6,
    relative_tolerance: float = 2e-6,
) -> dict[str, object]:
    """Fail unless compact CUDA scores reproduce the NumPy decision path."""
    verify_shared_monitor_parameters(suite)
    if run.covariates is not None or not np.isfinite(run.x).all():
        raise ValueError("CUDA validation requires a complete run without covariates")
    base_mask = (
        np.ones_like(run.x, dtype=bool)
        if run.observed_mask is None
        else np.asarray(run.observed_mask, dtype=bool)
    )
    mask = base_mask & generate_observed_mask(run.x, scenario, seed)
    incomplete = apply_mask(run.x, mask)
    cpu_location = suite.proposed.monitor.score(incomplete, mask)
    cpu_integrated = suite.integrated.monitor.score(incomplete, mask)
    stacked_masks = np.stack([mask, mask])
    raw = score_incomplete_batch(
        suite.proposed.monitor,
        np.stack([incomplete, incomplete]),
        stacked_masks,
        device=device,
        dtype="float64",
    )
    cuda_location = _scores_from_cuda_result(
        suite.proposed.monitor,
        raw,
        stacked_masks,
        integrate_uncertainty=False,
    )
    cuda_integrated = _scores_from_cuda_result(
        suite.proposed.monitor,
        raw,
        stacked_masks,
        integrate_uncertainty=True,
    )
    comparisons = {
        "reconstructed": (
            cpu_location.filter_result.reconstructed,
            cuda_location[0].filter_result.reconstructed,
        ),
        "location_energy": (
            cpu_location.location_energy,
            cuda_location[0].location_energy,
        ),
        "uncertainty_energy": (
            cpu_location.uncertainty_energy,
            cuda_location[0].uncertainty_energy,
        ),
        "posterior_trace": (
            cpu_location.posterior_trace,
            cuda_location[0].posterior_trace,
        ),
        "location_fused": (cpu_location.fused, cuda_location[0].fused),
        "integrated_fused": (cpu_integrated.fused, cuda_integrated[0].fused),
    }
    errors: dict[str, object] = {}
    for name, (reference, candidate) in comparisons.items():
        difference = np.asarray(candidate) - np.asarray(reference)
        errors[name] = {
            "maximum_absolute": float(np.max(np.abs(difference))),
            "root_mean_square": float(np.sqrt(np.mean(difference**2))),
            "within_tolerance": bool(
                np.allclose(
                    candidate,
                    reference,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
            ),
        }
    duplicate_error = max(
        float(np.max(np.abs(cuda_location[0].fused - cuda_location[1].fused))),
        float(
            np.max(
                np.abs(
                    cuda_location[0].filter_result.reconstructed
                    - cuda_location[1].filter_result.reconstructed
                )
            )
        ),
    )
    cpu_alarm, cpu_unknown, _ = _core_decision(
        suite.proposed, cpu_location
    )
    cuda_alarm, cuda_unknown, _ = _core_decision(
        suite.proposed, cuda_location[0]
    )
    integrated_cpu_alarm, integrated_cpu_unknown, _ = _core_decision(
        suite.integrated, cpu_integrated
    )
    integrated_cuda_alarm, integrated_cuda_unknown, _ = _core_decision(
        suite.integrated, cuda_integrated[0]
    )
    decisions_equal = bool(
        np.array_equal(cpu_alarm, cuda_alarm)
        and np.array_equal(cpu_unknown, cuda_unknown)
        and np.array_equal(integrated_cpu_alarm, integrated_cuda_alarm)
        and np.array_equal(integrated_cpu_unknown, integrated_cuda_unknown)
    )
    passed = bool(
        all(bool(item["within_tolerance"]) for item in errors.values())
        and duplicate_error == 0.0
        and decisions_equal
        and int(raw.get("fallback_sequence_count", 0)) == 0
    )
    result = {
        "passed": passed,
        "run": run.name,
        "scenario": asdict(scenario),
        "seed": seed,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "array_errors": errors,
        "batch_duplicate_maximum_absolute": duplicate_error,
        "alarm_and_unknown_exact_match": decisions_equal,
        "cuda_execution": {
            key: raw[key]
            for key in (
                "device",
                "device_name",
                "dtype",
                "elapsed_ms",
                "batch_size",
                "sequence_length",
                "factorization_failure_count",
                "fallback_sequence_count",
            )
        },
    }
    if not passed:
        raise RuntimeError(f"CPU/CUDA equivalence validation failed: {result}")
    return result


def _post_fault_ratio(
    persistence: np.ndarray,
    unknown: np.ndarray,
    fault: np.ndarray,
    threshold: float,
) -> tuple[float | None, float | None]:
    eligible = np.asarray(fault, dtype=bool) & ~np.asarray(unknown, dtype=bool)
    if not eligible.any():
        return None, None
    maximum = float(np.max(persistence[eligible]))
    ratio = maximum / threshold if threshold != 0.0 else None
    return maximum, ratio


def evaluate_multirun_stress_cuda(
    suite: FittedMultiRunSuite,
    test: SensorSequence,
    scenarios: Iterable[MissingnessConfig],
    seeds: Iterable[int] = (11, 29, 47),
    *,
    device: str = "cuda",
    batch_size: int = 32,
    warning_horizon: int = 0,
    refractory: int = 0,
    tracker: CUDAExecutionTracker | None = None,
    analysis_status: str = "exploratory_development",
) -> dict[str, object]:
    """Evaluate one fault run with shared compact CUDA posterior filtering."""
    if analysis_status not in {"exploratory_development", "confirmatory"}:
        raise ValueError("analysis status must be exploratory_development or confirmatory")
    if not np.isfinite(test.x).all():
        raise ValueError("controlled stress testing requires complete test truth")
    if test.x.shape[1] != suite.proposed.n_features_in_:
        raise ValueError("test feature count differs from the fitted suite")
    if test.covariates is not None:
        raise NotImplementedError("CUDA Rieth evaluation does not use covariates")
    verify_shared_monitor_parameters(suite)
    test_base_mask = (
        np.ones_like(test.x, dtype=bool)
        if test.observed_mask is None
        else np.asarray(test.observed_mask, dtype=bool)
    )
    if test_base_mask.shape != test.x.shape or not test_base_mask.all():
        raise ValueError("CUDA stress evaluation requires complete test truth")
    reference_monitor = suite.proposed.monitor
    oracle_score = _score_complete(reference_monitor, test.x)
    oracle_alarm, oracle_unknown, oracle_persistence = _core_decision(
        suite.proposed, oracle_score
    )
    oracle_maximum, oracle_ratio = _post_fault_ratio(
        oracle_persistence,
        oracle_unknown,
        test.fault,
        float(suite.proposed.calibrator.threshold_),
    )

    scenario_values = tuple(scenarios)
    seed_values = tuple(int(seed) for seed in seeds)
    if not scenario_values or not seed_values:
        raise ValueError("scenarios and seeds must both be nonempty")
    task_specs: list[tuple[MissingnessConfig, int, _MaskTask]] = []
    for scenario in scenario_values:
        for seed in seed_values:
            mask = test_base_mask & generate_observed_mask(
                test.x, scenario, int(seed)
            )
            task_specs.append(
                (
                    scenario,
                    int(seed),
                    _MaskTask(
                        run_index=len(task_specs),
                        run=test,
                        mechanism=scenario.identifier(),
                        mask=mask,
                    ),
                )
            )
    task_lookup = {
        task.run_index: (scenario, seed)
        for scenario, seed, task in task_specs
    }
    active_tracker = CUDAExecutionTracker() if tracker is None else tracker
    scored_tasks = _cuda_task_scores(
        reference_monitor,
        [task for _, _, task in task_specs],
        device=device,
        batch_size=batch_size,
        tracker=active_tracker,
    )
    records: list[dict[str, object]] = []
    for task, location_score, integrated_score in scored_tasks:
        scenario, seed = task_lookup[task.run_index]
        mask = task.mask
        artificially_hidden = ~mask
        imputation_evaluation_mask = ~artificially_hidden
        mapic_alarm, mapic_unknown, mapic_persistence = _core_decision(
            suite.proposed, location_score
        )
        integrated_alarm, integrated_unknown, integrated_persistence = (
            _core_decision(suite.integrated, integrated_score)
        )
        mask_alarm, mask_unknown, mask_persistence = _core_decision(
            suite.mask_only, location_score
        )

        def core_entry(
            model: MAPIC,
            alarm: np.ndarray,
            unknown: np.ndarray,
            persistence: np.ndarray,
            *,
            include_imputation: bool = False,
        ) -> dict[str, object]:
            maximum, ratio = _post_fault_ratio(
                persistence,
                unknown,
                test.fault,
                float(model.calibrator.threshold_),
            )
            entry: dict[str, object] = {
                "monitoring": _monitoring_metrics(
                    alarm,
                    test,
                    warning_horizon,
                    refractory,
                    decision_mask=~unknown,
                ),
                "unknown_rate": float(unknown.mean()),
                "post_fault_max_persistence": maximum,
                "post_fault_max_to_threshold_ratio": ratio,
            }
            if include_imputation:
                entry["imputation"] = imputation_error(
                    test.x,
                    location_score.filter_result.reconstructed,
                    imputation_evaluation_mask,
                    scale=reference_monitor.filter_model.scale_,
                )
                entry["imputation_scope"] = (
                    "originally observed, artificially hidden cells only; "
                    "residuals normalized by Phase-I feature scale"
                )
            return entry

        methods: dict[str, object] = {
            "same_chart_base_observed_input": {
                "monitoring": _monitoring_metrics(
                    oracle_alarm,
                    test,
                    warning_horizon,
                    refractory,
                    decision_mask=~oracle_unknown,
                ),
                "unknown_rate": float(oracle_unknown.mean()),
                "post_fault_max_persistence": oracle_maximum,
                "post_fault_max_to_threshold_ratio": oracle_ratio,
            },
            "UD-MAPIC": core_entry(
                suite.proposed,
                mapic_alarm,
                mapic_unknown,
                mapic_persistence,
                include_imputation=True,
            ),
            "MAPIC_integrated_expectation": core_entry(
                suite.integrated,
                integrated_alarm,
                integrated_unknown,
                integrated_persistence,
            ),
            "UD-MAPIC_mask_only_context": core_entry(
                suite.mask_only,
                mask_alarm,
                mask_unknown,
                mask_persistence,
            ),
        }
        for method, baseline in suite.baselines.items():
            if method == "kalman_mean":
                baseline_score = location_score.fused
                reconstructed = location_score.filter_result.reconstructed
            else:
                reconstructed = _baseline_completed(baseline, test, mask)
                baseline_score = _score_complete(
                    reference_monitor, reconstructed
                ).fused
            alarm, unknown, persistence = _baseline_decision(
                baseline, baseline_score, mask
            )
            maximum, ratio = _post_fault_ratio(
                persistence,
                unknown,
                test.fault,
                float(baseline.calibrator.threshold_),
            )
            methods[method] = {
                "monitoring": _monitoring_metrics(
                    alarm,
                    test,
                    warning_horizon,
                    refractory,
                    decision_mask=~unknown,
                ),
                "imputation": imputation_error(
                    test.x,
                    reconstructed,
                    imputation_evaluation_mask,
                    scale=reference_monitor.filter_model.scale_,
                ),
                "unknown_rate": float(unknown.mean()),
                "post_fault_max_persistence": maximum,
                "post_fault_max_to_threshold_ratio": ratio,
                "imputation_scope": (
                    "originally observed, artificially hidden cells only; "
                    "residuals normalized by Phase-I feature scale"
                ),
                "causal": method != "linear_offline",
            }
        records.append(
            {
                "mechanism": scenario.mechanism,
                "scenario_id": scenario.identifier(),
                "scenario": asdict(scenario),
                "target_missing_rate": scenario.rate,
                "realized_missing_rate": float((~mask).mean()),
                "natural_missing_rate": 0.0,
                "realized_artificial_missing_rate_on_observed_cells": float(
                    artificially_hidden.mean()
                ),
                "seed": seed,
                "methods": methods,
            }
        )
    return {
        "schema_version": 1,
        "method": "UD-MAPIC",
        "configuration": _config_dict(suite.config),
        "evaluation_protocol": {
            "warning_horizon_samples": int(warning_horizon),
            "refractory_samples": int(refractory),
            "calibration_unit": "healthy_run_maximum",
        },
        "test": _sequence_summary(test),
        "test_metadata": {
            key: value
            for key, value in test.metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "records": records,
        "execution_backend": {
            "backend": "pytorch_cuda_compact_batch",
            "device": device,
            "dtype": "float64",
        },
        "analysis_status": analysis_status,
    }
