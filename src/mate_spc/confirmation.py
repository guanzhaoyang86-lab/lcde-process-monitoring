"""Frozen retrospective confirmation protocol for the batch-distillation archive."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable

import numpy as np

from .baselines import CalibratedPointMonitor
from .datasets import SensorSequence
from .experiment import evaluate_natural_missingness
from .model import MAPIC, MAPICConfig


def slice_sequence(
    sequence: SensorSequence,
    start: int,
    stop: int,
    *,
    name: str | None = None,
) -> SensorSequence:
    """Slice every row-aligned field and metadata array without resetting time."""
    n_rows = sequence.x.shape[0]
    if not 0 <= start < stop <= n_rows:
        raise ValueError("slice bounds must satisfy 0 <= start < stop <= n_rows")
    section = slice(start, stop)
    metadata: dict[str, object] = {}
    for key, value in sequence.metadata.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n_rows:
            metadata[key] = value[section].copy()
        else:
            metadata[key] = value
    return replace(
        sequence,
        name=sequence.name if name is None else name,
        x=sequence.x[section].copy(),
        fault=sequence.fault[section].copy(),
        timestamps=(
            None
            if sequence.timestamps is None
            else sequence.timestamps[section].copy()
        ),
        observed_mask=(
            None
            if sequence.observed_mask is None
            else sequence.observed_mask[section].copy()
        ),
        covariates=(
            None
            if sequence.covariates is None
            else sequence.covariates[section].copy()
        ),
        metadata=metadata,
    )


def temporal_confirmation_split(
    normal: SensorSequence,
    train_fraction: float = 0.65,
    purge_samples: int = 600,
) -> tuple[SensorSequence, SensorSequence, dict[str, int | float]]:
    """Make a chronological design/purge/calibration split from one healthy run."""
    if normal.fault.any():
        raise ValueError("the confirmation split requires an entirely healthy run")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    if isinstance(purge_samples, bool) or int(purge_samples) != purge_samples:
        raise ValueError("purge_samples must be an integer")
    purge_samples = int(purge_samples)
    if purge_samples < 0:
        raise ValueError("purge_samples must be nonnegative")
    n_rows = normal.x.shape[0]
    usable = n_rows - purge_samples
    if usable < 4:
        raise ValueError("too few rows remain after the purge interval")
    train_stop = int(np.floor(usable * train_fraction))
    calibration_start = train_stop + purge_samples
    if train_stop < 2 or n_rows - calibration_start < 2:
        raise ValueError("the requested split leaves fewer than two rows in a partition")
    train = slice_sequence(
        normal,
        0,
        train_stop,
        name=f"{normal.name}::design[0:{train_stop}]",
    )
    calibration = slice_sequence(
        normal,
        calibration_start,
        n_rows,
        name=f"{normal.name}::calibration[{calibration_start}:{n_rows}]",
    )
    protocol: dict[str, int | float] = {
        "train_fraction_of_nonpurged_rows": float(train_fraction),
        "design_start": 0,
        "design_stop_exclusive": train_stop,
        "purge_start": train_stop,
        "purge_stop_exclusive": calibration_start,
        "purge_samples": purge_samples,
        "calibration_start": calibration_start,
        "calibration_stop_exclusive": n_rows,
    }
    return train, calibration, protocol


def _released_values(sequence: SensorSequence) -> np.ndarray:
    if "released_values" not in sequence.metadata:
        raise ValueError("batch sequence metadata lacks released_values")
    values = np.asarray(sequence.metadata["released_values"], dtype=float)
    if values.shape != sequence.x.shape or not np.isfinite(values).all():
        raise ValueError("released_values must be finite and aligned with x")
    return values


def _validate_aligned(
    normal: SensorSequence,
    tests: tuple[SensorSequence, ...],
) -> None:
    if normal.observed_mask is None:
        raise ValueError("normal run has no observed mask")
    test_names = [test.name for test in tests]
    if any(test is normal or test.name == normal.name for test in tests):
        raise ValueError("a test run must not be the normal run or share its name")
    if len(set(test_names)) != len(test_names):
        raise ValueError("retrospective test run names must be unique")
    for test in tests:
        if test.observed_mask is None:
            raise ValueError(f"test run {test.name!r} has no observed mask")
        if test.feature_names != normal.feature_names:
            raise ValueError("normal and test feature schemas differ")
        if test.covariate_names != normal.covariate_names:
            raise ValueError("normal and test covariate schemas differ")
        if not np.isclose(
            test.sample_period_seconds,
            normal.sample_period_seconds,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("normal and test sample periods differ")


def _aggregate_visible_results(
    evaluations: dict[str, dict[str, dict[str, object]]],
    row_counts: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    method_names = tuple(next(iter(evaluations.values()))["methods"])
    aggregate: dict[str, dict[str, float | int]] = {}
    total_rows = sum(row_counts.values())
    for method in method_names:
        reports = [
            run["methods"][method]["visible_effect_monitoring"]
            for run in evaluations.values()
        ]
        event_count = int(sum(int(report["event_count"]) for report in reports))
        detected = int(sum(int(report["events_detected"]) for report in reports))
        false_episodes = int(
            sum(int(report["false_alarm_episodes"]) for report in reports)
        )
        normal_hours = float(
            sum(float(report["monitored_normal_hours"]) for report in reports)
        )
        coverage = sum(
            float(evaluations[name]["methods"][method]["unknown_rate"])
            * row_counts[name]
            for name in evaluations
        )
        aggregate[method] = {
            "events_detected": detected,
            "event_count": event_count,
            "event_recall": detected / event_count if event_count else float("nan"),
            "false_alarm_episodes": false_episodes,
            "false_alarm_episodes_per_100h": (
                100.0 * false_episodes / normal_hours
                if normal_hours > 0.0
                else float("nan")
            ),
            "monitored_normal_hours": normal_hours,
            "decision_coverage": 1.0 - coverage / total_rows,
        }
    return aggregate


def run_batch_confirmation(
    normal: SensorSequence,
    tests: Iterable[SensorSequence],
    config: MAPICConfig,
    *,
    train_fraction: float = 0.65,
    purge_samples: int = 600,
    baseline_methods: tuple[str, ...] = (
        "median",
        "locf",
        "kalman_mean",
        "linear_offline",
    ),
) -> dict[str, object]:
    """Fit once on the frozen healthy split and score every retrospective test run."""
    test_runs = tuple(tests)
    if not test_runs:
        raise ValueError("at least one retrospective test run is required")
    _validate_aligned(normal, test_runs)
    train, calibration, split = temporal_confirmation_split(
        normal,
        train_fraction=train_fraction,
        purge_samples=purge_samples,
    )
    design_elapsed_start = int(split["design_start"])
    calibration_elapsed_start = int(split["calibration_start"])
    ud_config = replace(config, integrate_uncertainty=False)
    integrated_config = replace(config, integrate_uncertainty=True)
    train_release = _released_values(train)
    calibration_release = _released_values(calibration)
    ud_model = MAPIC(ud_config).fit(
        train_release,
        calibration_release,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
        train_elapsed_start=design_elapsed_start,
        calibration_elapsed_start=calibration_elapsed_start,
    )
    integrated_model = MAPIC(integrated_config).fit(
        train_release,
        calibration_release,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
        train_elapsed_start=design_elapsed_start,
        calibration_elapsed_start=calibration_elapsed_start,
    )
    mask_only_config = replace(
        config,
        integrate_uncertainty=False,
        use_posterior_quality_context=False,
    )
    mask_only_model = MAPIC(mask_only_config).fit(
        train_release,
        calibration_release,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
        train_elapsed_start=design_elapsed_start,
        calibration_elapsed_start=calibration_elapsed_start,
    )
    baselines = {
        method: CalibratedPointMonitor(method, ud_config).fit(
            train_release,
            calibration_release,
            train.covariates,
            calibration.covariates,
            train.observed_mask,
            calibration.observed_mask,
            train_elapsed_start=design_elapsed_start,
            calibration_elapsed_start=calibration_elapsed_start,
        )
        for method in baseline_methods
    }
    models: dict[str, MAPIC | CalibratedPointMonitor] = {
        "UD-MAPIC": ud_model,
        "MAPIC_integrated_expectation": integrated_model,
        "UD-MAPIC_mask_only_context": mask_only_model,
        **baselines,
    }
    evaluations: dict[str, dict[str, dict[str, object]]] = {}
    row_counts: dict[str, int] = {}
    for test in test_runs:
        row_counts[test.name] = int(test.x.shape[0])
        evaluations[test.name] = {
            "methods": {
                name: {
                    **evaluate_natural_missingness(model, test),
                    "causal": name != "linear_offline",
                }
                for name, model in models.items()
            }
        }
    return {
        "method": "UD-MAPIC",
        "configuration": asdict(ud_config),
        "split": split,
        "elapsed_context": {
            "design_start_sample": design_elapsed_start,
            "calibration_start_sample": calibration_elapsed_start,
            "held_out_runs_start_sample": 0,
        },
        "design_rows": int(train.x.shape[0]),
        "calibration_rows": int(calibration.x.shape[0]),
        "purged_rows": int(purge_samples),
        "calibration_base_blocks": int(calibration.x.shape[0] // config.block_size),
        "phase1_interpolated_rows_excluded": {
            "design": int((~train.observed_mask).any(axis=1).sum()),
            "calibration": int((~calibration.observed_mask).any(axis=1).sum()),
        },
        "calibration_summaries": {
            name: model.calibration_summary_ for name, model in models.items()
        },
        "method_metadata": {
            name: {"causal": name != "linear_offline"} for name in models
        },
        "test_runs": evaluations,
        "aggregate_visible_effect": _aggregate_visible_results(evaluations, row_counts),
    }
