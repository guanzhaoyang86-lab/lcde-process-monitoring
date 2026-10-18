"""Frozen CRIS-v9 four-tank study orchestration.

This module contains deterministic, side-effect-light building blocks.  The
CLI owns artifact writes and the confirmation claim; these functions own the
scientific unit, role, scoring, and gate semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .baselines import (
    BaselineThreshold,
    EnergyCUSUMBaseline,
    PCAT2QBaseline,
    PredictiveResidualCUSUMBaseline,
)
from .evaluation import EpisodeOutcome, clopper_pearson_upper, nearest_rank
from .model import CRISConfig, CRISModel, CRISTrainingConfig
from .monitor import (
    MonitorConfig,
    PointCalibrator,
    TrajectoryThreshold,
    apply_threshold,
    calibrate_trajectory_threshold,
    scan_innovations,
)
from .simulator import CRISBatch, simulate_batch


CHARTER_SCHEMA = "cris_v9_four_tank_study_charter"
CHARTER_VERSION = 3
FAULT_KINDS = (
    "sensor_bias",
    "intermittent_pump",
    "valve_stiction",
    "dependence_shift",
)
NEGATIVE_KINDS = ("healthy", "covariate_shift_only")
CANDIDATE_IDS = ("cris-k1-equal", "cris-k2-equal", "cris-k2-location")
ROLE_TO_KIND = MappingProxyType(
    {
        "healthy_model_fit": "healthy",
        "healthy_point_reference": "healthy",
        "healthy_threshold_calibration": "healthy",
        "development_healthy": "healthy",
        "development_sensor_bias": "sensor_bias",
        "development_intermittent_pump": "intermittent_pump",
        "development_valve_stiction": "valve_stiction",
        "development_dependence_shift": "dependence_shift",
        "development_covariate_shift_only": "covariate_shift_only",
        "confirmation_healthy": "healthy",
        "confirmation_sensor_bias": "sensor_bias",
        "confirmation_intermittent_pump": "intermittent_pump",
        "confirmation_valve_stiction": "valve_stiction",
        "confirmation_dependence_shift": "dependence_shift",
        "confirmation_covariate_shift_only": "covariate_shift_only",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_study_charter(path: str | Path) -> Mapping[str, Any]:
    """Authenticate and validate the frozen v3 study charter."""

    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = source.with_suffix(".sha256")
    fields = sidecar.read_text(encoding="ascii").strip().split("  ")
    if fields != [digest, source.name]:
        raise ValueError("four-tank charter SHA-256 sidecar mismatch")
    document = json.loads(raw.decode("utf-8"))
    if (
        document.get("schema") != CHARTER_SCHEMA
        or document.get("schema_version") != CHARTER_VERSION
        or document.get("protocol_id")
        != "cris-v9-four-tank-development-and-confirmation-v3"
        or document.get("state") != "PREREGISTERED_BEFORE_CONFIRMATION_GENERATION"
    ):
        raise ValueError("wrong CRIS-v9 four-tank charter identity")

    method = document.get("method", {})
    fixed_method = {
        "context_length": 60,
        "prediction_horizons": [1, 5, 15],
        "student_t_df": 5.0,
        "gru_layers": 2,
        "gru_hidden_width": 64,
        "training_epochs": 40,
        "training_batch_size": 1024,
        "training_seed": 20260915,
        "gradient_clip_norm": 1.0,
        "inference_batch_size": 2048,
        "minimum_scale": 0.0001,
        "rolling_scales": [4, 16, 64],
        "power_betting_eta": 0.5,
        "trajectory_alpha": 0.05,
        "alarm_comparison": "strictly_greater_than",
        "reset_gap": 64,
        "hmm_initial_state": [1.0, 0.0],
    }
    if any(method.get(name) != value for name, value in fixed_method.items()):
        raise ValueError("charter method differs from the frozen CRIS-v9 settings")
    if method.get("hmm_modes") != [
        {"rho": 0.02, "lambda": 0.2},
        {"rho": 0.1, "lambda": 0.5},
    ]:
        raise ValueError("charter HMM modes differ from the frozen settings")
    if method.get("unknown_semantics") != (
        "missing components hold their own HMM state; available enabled components "
        "continue; combination denominator remains fixed; 64 consecutive missing "
        "samples reset a component"
    ):
        raise ValueError("charter UNKNOWN semantics are not frozen")

    candidates = method.get("candidates", ())
    if tuple(item.get("id") for item in candidates) != CANDIDATE_IDS:
        raise ValueError("charter candidate identities or order differ")
    expected_candidates = (
        (1, (1.0 / 3.0,) * 3),
        (2, (1.0 / 3.0,) * 3),
        (2, (0.5, 0.25, 0.25)),
    )
    for item, (top_k, weights) in zip(candidates, expected_candidates, strict=True):
        if item.get("top_k") != top_k or not np.allclose(
            item.get("moment_weights"), weights, rtol=0.0, atol=0.0
        ):
            raise ValueError("charter candidate definition differs")

    all_seeds: list[int] = []
    for section_name in ("development_seed_roles", "confirmation_seed_roles"):
        section = document.get(section_name, {})
        expected_roles = {
            role for role in ROLE_TO_KIND if role.startswith(section_name.split("_")[0])
        }
        if section_name == "development_seed_roles":
            expected_roles.update(
                {"healthy_model_fit", "healthy_point_reference", "healthy_threshold_calibration"}
            )
        if set(section) != expected_roles:
            raise ValueError(f"{section_name} role names differ from the contract")
        for role, spec in section.items():
            first = spec.get("first")
            last = spec.get("last")
            count = spec.get("count")
            if (
                type(first) is not int
                or type(last) is not int
                or type(count) is not int
                or first < 0
                or last - first + 1 != count
                or count != 200
                and role != "healthy_model_fit"
            ):
                raise ValueError(f"invalid seed interval for {role}")
            if role == "healthy_model_fit" and count != 400:
                raise ValueError("healthy_model_fit must contain 400 runs")
            all_seeds.extend(range(first, last + 1))
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("study seed roles overlap")
    if document["selection"].get("confirmation_unlock") != (
        "FORBIDDEN until the selected candidate passes every development_go_gate"
    ):
        raise ValueError("confirmation unlock rule is not frozen")
    return MappingProxyType(document)


def seeds_for_role(charter: Mapping[str, Any], role: str) -> tuple[int, ...]:
    if role not in ROLE_TO_KIND:
        raise ValueError(f"unknown study role: {role}")
    section = (
        "confirmation_seed_roles"
        if role.startswith("confirmation_")
        else "development_seed_roles"
    )
    spec = charter[section][role]
    values = tuple(range(int(spec["first"]), int(spec["last"]) + 1))
    if len(values) != int(spec["count"]):
        raise ValueError("charter seed interval count mismatch")
    return values


def simulate_role(
    charter: Mapping[str, Any],
    role: str,
    *,
    confirmation_authorized: bool = False,
) -> CRISBatch:
    if role.startswith("confirmation_") and not confirmation_authorized:
        raise PermissionError("confirmation generation requires a committed global claim")
    simulator = charter["simulator"]
    seeds = seeds_for_role(charter, role)
    kind = ROLE_TO_KIND[role]
    return simulate_batch(
        (kind,) * len(seeds),
        seeds,
        T=int(simulator["trajectory_length"]),
        onset=int(simulator["fault_onset_index"]),
        dt=float(simulator["sample_period_seconds"]),
    )


def model_arrays(batch: CRISBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw simulator output to six targets and two exogenous covariates."""

    commands = batch.c[:, :, 2:]
    targets = np.concatenate((batch.x, commands), axis=2)
    covariates = batch.c[:, :, :2]
    command_mask = np.ones(commands.shape, dtype=np.bool_)
    mask = np.concatenate((batch.observed_mask, command_mask), axis=2)
    return targets, covariates, mask


def fit_model(
    batch: CRISBatch,
    charter: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> CRISModel:
    targets, covariates, mask = model_arrays(batch)
    method = charter["method"]
    return CRISModel.fit(
        targets,
        covariates,
        mask,
        config=CRISConfig(
            target_dim=targets.shape[2],
            covariate_dim=covariates.shape[2],
            minimum_scale=float(method["minimum_scale"]),
        ),
        training_config=CRISTrainingConfig(
            epochs=int(method["training_epochs"]),
            batch_size=int(method["training_batch_size"]),
            learning_rate=float(method["optimizer"]["learning_rate"]),
            weight_decay=float(method["optimizer"]["weight_decay"]),
            gradient_clip_norm=float(method["gradient_clip_norm"]),
            seed=int(method["training_seed"]),
            device=device,
        ),
        metadata={
            "study": charter["protocol_id"],
            "role": "healthy_model_fit",
            "target_channels": charter["simulator"]["model_target_channels"],
            "exogenous_covariates": charter["simulator"]["model_exogenous_covariates"],
        },
    )


def innovation_runs(
    model: CRISModel,
    batch: CRISBatch,
    charter: Mapping[str, Any],
    *,
    device: str = "cuda",
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, ...]:
    targets, covariates, mask = model_arrays(batch)
    batch_size = int(charter["method"]["inference_batch_size"])
    output: list[np.ndarray] = []
    for index in range(batch.run_count):
        output.append(
            model.innovations(
                targets[index],
                covariates[index],
                mask[index],
                device=device,
                batch_size=batch_size,
            )
        )
        if progress is not None:
            progress(index + 1, batch.run_count)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    top_k: int
    moment_weights: tuple[float, float, float]
    order: int

    @property
    def monitor_config(self) -> MonitorConfig:
        return MonitorConfig(top_k=self.top_k, moment_weights=self.moment_weights)


def candidate_specs(charter: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(
            candidate_id=str(item["id"]),
            top_k=int(item["top_k"]),
            moment_weights=tuple(float(value) for value in item["moment_weights"]),
            order=index,
        )
        for index, item in enumerate(charter["method"]["candidates"])
    )


def scan_runs(
    runs: Sequence[np.ndarray],
    calibrator: PointCalibrator,
    threshold: TrajectoryThreshold | None = None,
) -> tuple[np.ndarray, ...]:
    return tuple(
        scan_innovations(run, calibrator, threshold=threshold).score
        for run in runs
    )


def _negative_metrics(score_paths: Sequence[np.ndarray], threshold: float) -> dict[str, Any]:
    alarms = [apply_threshold(path, threshold) for path in score_paths]
    count = sum(bool(np.any(alarm)) for alarm in alarms)
    runs = len(alarms)
    return {
        "runs": runs,
        "alarm_runs": count,
        "alarm_rate": count / runs,
        "one_sided_cp95_upper": clopper_pearson_upper(count, runs),
    }


def summarize_paths(
    score_paths: Sequence[np.ndarray],
    batch: CRISBatch,
    threshold: TrajectoryThreshold | float,
    charter: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one condition using independent runs and episode matching."""

    if len(score_paths) != batch.run_count:
        raise ValueError("score path count does not match the trajectory batch")
    cutoff = threshold.value if isinstance(threshold, TrajectoryThreshold) else float(threshold)
    kind_set = set(batch.fault_kind)
    if len(kind_set) != 1:
        raise ValueError("one summary batch must contain exactly one condition")
    kind = batch.fault_kind[0]
    if kind in NEGATIVE_KINDS:
        result = _negative_metrics(score_paths, cutoff)
        result["condition"] = kind
        return result

    grace = (
        int(charter["episode_evaluation"]["intermittent_right_grace_samples"])
        if kind == "intermittent_pump"
        else 0
    )
    outcomes = tuple(
        EpisodeOutcome(
            run_id=int(batch.seed[index]),
            condition=kind,
            alarm=apply_threshold(score_paths[index], cutoff),
            fault_mask=batch.fault_mask[index],
            grace_samples=grace,
        )
        for index in range(batch.run_count)
    )
    detections = sum(item.detected for item in outcomes)
    delays = tuple(item.delay for item in outcomes)
    relative_delays = tuple(
        item.delay / (item.alarm.size - item.first_affected_index) for item in outcomes
    )
    pre_effect = sum(item.pre_effect_false_alarm for item in outcomes)
    inactive = sum(item.inactive_gap_alarm_onsets for item in outcomes)
    return {
        "condition": kind,
        "runs": len(outcomes),
        "detected_runs": detections,
        "missed_runs": len(outcomes) - detections,
        "recall": detections / len(outcomes),
        "median_delay_samples": float(np.median(np.asarray(delays, dtype=np.float64))),
        "nearest_rank_p90_delay_samples": nearest_rank(delays, 0.90),
        "median_relative_delay": float(
            np.median(np.asarray(relative_delays, dtype=np.float64))
        ),
        "pre_effect_false_alarm_runs": pre_effect,
        "pre_effect_false_alarm_rate": pre_effect / len(outcomes),
        "inactive_gap_alarm_onsets": inactive,
        "matching_grace_samples": grace,
    }


@dataclass(frozen=True, slots=True)
class CandidateDevelopment:
    spec: CandidateSpec
    calibrator: PointCalibrator
    threshold: TrajectoryThreshold
    summaries: Mapping[str, Mapping[str, Any]]

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.spec),
            "threshold": {
                "value": self.threshold.value,
                "alpha": self.threshold.alpha,
                "calibration_run_count": self.threshold.calibration_run_count,
                "ascending_rank": self.threshold.ascending_rank,
            },
            "summaries": {name: dict(value) for name, value in self.summaries.items()},
        }


def evaluate_candidate(
    spec: CandidateSpec,
    point_reference: Sequence[np.ndarray],
    threshold_innovations: Sequence[np.ndarray],
    development: Mapping[str, tuple[CRISBatch, Sequence[np.ndarray]]],
    charter: Mapping[str, Any],
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> CandidateDevelopment:
    config = spec.monitor_config
    calibrator = PointCalibrator.fit(point_reference, config)
    threshold_paths: list[np.ndarray] = []
    for index, run in enumerate(threshold_innovations):
        threshold_paths.append(scan_innovations(run, calibrator).score)
        if progress is not None:
            progress("threshold", index + 1, len(threshold_innovations))
    threshold = calibrate_trajectory_threshold(
        threshold_paths, alpha=float(charter["method"]["trajectory_alpha"])
    )
    summaries: dict[str, Mapping[str, Any]] = {}
    for role, (batch, innovations) in development.items():
        paths: list[np.ndarray] = []
        for index, run in enumerate(innovations):
            paths.append(scan_innovations(run, calibrator).score)
            if progress is not None:
                progress(role, index + 1, len(innovations))
        summaries[batch.fault_kind[0]] = summarize_paths(paths, batch, threshold, charter)
    return CandidateDevelopment(
        spec=spec,
        calibrator=calibrator,
        threshold=threshold,
        summaries=MappingProxyType(summaries),
    )


@dataclass(frozen=True, slots=True)
class BaselineDevelopment:
    baseline: Any
    threshold: BaselineThreshold
    summaries: Mapping[str, Mapping[str, Any]]

    def to_document(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.name,
            "threshold": {
                "value": self.threshold.threshold,
                "alpha": self.threshold.alpha,
                "calibration_run_count": int(
                    self.threshold.calibration_maxima.size
                ),
                "ascending_rank": self.threshold.rank,
            },
            "summaries": {name: dict(value) for name, value in self.summaries.items()},
        }


def evaluate_baselines(
    fit_batch: CRISBatch,
    point_reference_innovations: Sequence[np.ndarray],
    threshold_batch: CRISBatch,
    threshold_innovations: Sequence[np.ndarray],
    evaluation: Mapping[str, tuple[CRISBatch, Sequence[np.ndarray]]],
    charter: Mapping[str, Any],
) -> tuple[BaselineDevelopment, ...]:
    """Fit and fairly calibrate all three pre-registered baselines."""

    fit_targets, _fit_covariates, _fit_mask = model_arrays(fit_batch)
    threshold_targets, _threshold_covariates, _threshold_mask = model_arrays(
        threshold_batch
    )
    alpha = float(charter["method"]["trajectory_alpha"])
    baselines_and_calibration = (
        (
            PCAT2QBaseline.fit(tuple(fit_targets)),
            tuple(threshold_targets),
            "targets",
        ),
        (
            EnergyCUSUMBaseline.fit(tuple(fit_targets)),
            tuple(threshold_targets),
            "targets",
        ),
        (
            PredictiveResidualCUSUMBaseline.fit(point_reference_innovations),
            tuple(threshold_innovations),
            "innovations",
        ),
    )
    results: list[BaselineDevelopment] = []
    for baseline, calibration_values, input_kind in baselines_and_calibration:
        threshold = baseline.calibrate_trajectory_threshold(
            calibration_values, alpha=alpha
        )
        summaries: dict[str, Mapping[str, Any]] = {}
        for _role, (batch, innovations) in evaluation.items():
            if input_kind == "targets":
                values, _covariates, _mask = model_arrays(batch)
                paths = tuple(baseline.score(run).path for run in values)
            else:
                paths = tuple(baseline.score(run).path for run in innovations)
            summaries[batch.fault_kind[0]] = summarize_paths(
                paths, batch, threshold.threshold, charter
            )
        results.append(
            BaselineDevelopment(
                baseline=baseline,
                threshold=threshold,
                summaries=MappingProxyType(summaries),
            )
        )
    return tuple(results)


def negative_controls_pass(
    result: CandidateDevelopment, charter: Mapping[str, Any]
) -> bool:
    gate = charter["development_go_gates"]
    return (
        result.summaries["healthy"]["alarm_rate"] <= float(gate["healthy_alarm_rate_max"])
        and result.summaries["covariate_shift_only"]["alarm_rate"]
        <= float(gate["covariate_shift_only_alarm_rate_max"])
    )


def development_go(result: CandidateDevelopment, charter: Mapping[str, Any]) -> bool:
    if not negative_controls_pass(result, charter):
        return False
    gate = charter["development_go_gates"]
    for kind in FAULT_KINDS:
        required = (
            float(gate["intermittent_pump_recall_min"])
            if kind == "intermittent_pump"
            else float(gate["each_persistent_fault_recall_min"])
        )
        if result.summaries[kind]["recall"] < required:
            return False
        if (
            result.summaries[kind]["median_relative_delay"]
            > float(gate["each_fault_median_relative_delay_max"])
        ):
            return False
    return True


def _selection_key(result: CandidateDevelopment) -> tuple[float, int, float, int]:
    minimum_recall = min(result.summaries[kind]["recall"] for kind in FAULT_KINDS)
    pre_effect = sum(
        int(result.summaries[kind]["pre_effect_false_alarm_runs"])
        for kind in FAULT_KINDS
    )
    worst_delay = max(
        float(result.summaries[kind]["median_relative_delay"]) for kind in FAULT_KINDS
    )
    return (-minimum_recall, pre_effect, worst_delay, result.spec.order)


def select_candidate(
    results: Sequence[CandidateDevelopment],
    charter: Mapping[str, Any],
) -> CandidateDevelopment | None:
    if tuple(item.spec.candidate_id for item in results) != CANDIDATE_IDS:
        raise ValueError("candidate results must appear once in frozen order")
    eligible = [item for item in results if negative_controls_pass(item, charter)]
    return None if not eligible else min(eligible, key=_selection_key)


def diagnostic_best(results: Sequence[CandidateDevelopment]) -> CandidateDevelopment:
    if not results:
        raise ValueError("at least one candidate result is required")
    return min(results, key=_selection_key)


__all__ = [
    "CANDIDATE_IDS",
    "CHARTER_SCHEMA",
    "CHARTER_VERSION",
    "FAULT_KINDS",
    "NEGATIVE_KINDS",
    "ROLE_TO_KIND",
    "BaselineDevelopment",
    "CandidateDevelopment",
    "CandidateSpec",
    "candidate_specs",
    "development_go",
    "diagnostic_best",
    "evaluate_candidate",
    "evaluate_baselines",
    "fit_model",
    "innovation_runs",
    "load_study_charter",
    "model_arrays",
    "negative_controls_pass",
    "scan_runs",
    "seeds_for_role",
    "select_candidate",
    "sha256_file",
    "simulate_role",
    "summarize_paths",
]
