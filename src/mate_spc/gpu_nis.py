"""Auditable CUDA scoring helpers for complete-run normalized innovations.

This module deliberately calls :func:`score_incomplete_batch` even though its
inputs are fully observed.  The complete-input shortcut used by the monitoring
pipeline does not calculate innovations and therefore must not be used for a
NIS experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from .datasets import SensorSequence
from .gpu_multirun import CUDAExecutionTracker, cuda_runtime_metadata
from .torch_backend import score_incomplete_batch


RIETH_FEATURE_COUNT = 52
DEFAULT_NEGATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CompleteRunNISResult:
    """Normalized innovations and auditable CUDA execution provenance."""

    by_run: dict[str, np.ndarray]
    execution: dict[str, object]
    runtime_environment: dict[str, object]
    sequence_length: int
    feature_count: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _validate_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and nonnegative")
    return tolerance


def _complete_mask(run: SensorSequence) -> np.ndarray:
    values = np.asarray(run.x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != RIETH_FEATURE_COUNT:
        raise ValueError(
            f"run {run.name!r} must have shape (time, {RIETH_FEATURE_COUNT})"
        )
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"run {run.name!r} must contain complete finite data")
    if run.covariates is not None:
        raise ValueError(f"run {run.name!r} must not contain exogenous variables")
    mask = (
        np.ones_like(values, dtype=bool)
        if run.observed_mask is None
        else np.asarray(run.observed_mask, dtype=bool)
    )
    if mask.shape != values.shape or not mask.all():
        raise ValueError(f"run {run.name!r} must be fully observed")
    return mask


def _validate_runs(
    runs: Iterable[SensorSequence],
) -> tuple[tuple[SensorSequence, ...], int]:
    normalized = tuple(runs)
    if not normalized:
        raise ValueError("at least one complete run is required")
    names = tuple(str(run.name) for run in normalized)
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("complete run names must be nonempty and unique")
    lengths: set[int] = set()
    for run in normalized:
        _complete_mask(run)
        lengths.add(int(run.x.shape[0]))
    if len(lengths) != 1:
        raise ValueError("all complete runs in a CUDA batch must have equal length")
    return normalized, lengths.pop()


def _validated_nis(
    value: object,
    expected_shape: tuple[int, int],
    negative_tolerance: float,
) -> np.ndarray:
    nis = np.asarray(value, dtype=np.float64)
    if nis.shape != expected_shape:
        raise RuntimeError(
            f"CUDA NIS has shape {nis.shape}, expected {expected_shape}"
        )
    if not np.isfinite(nis).all():
        raise RuntimeError("CUDA NIS contains non-finite values")
    minimum = float(np.min(nis))
    if minimum < -negative_tolerance:
        raise RuntimeError(
            "CUDA NIS contains a materially negative value: "
            f"{minimum:.6g} < {-negative_tolerance:.6g}"
        )
    return np.maximum(nis, 0.0)


def score_complete_runs_cuda(
    monitor: object,
    runs: Iterable[SensorSequence],
    *,
    device: str = "cuda",
    batch_size: int = 32,
    negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE,
) -> CompleteRunNISResult:
    """Compute normalized innovation/NIS arrays for equal-length complete runs."""
    batch_size = _positive_integer(batch_size, "batch_size")
    negative_tolerance = _validate_negative_tolerance(negative_tolerance)
    normalized, sequence_length = _validate_runs(runs)
    runtime = cuda_runtime_metadata(device)
    tracker = CUDAExecutionTracker()
    by_run: dict[str, np.ndarray] = {}

    for start in range(0, len(normalized), batch_size):
        chunk = normalized[start : start + batch_size]
        values = np.stack(
            [np.asarray(run.x, dtype=np.float64) for run in chunk], axis=0
        )
        masks = np.ones_like(values, dtype=bool)
        raw = score_incomplete_batch(
            monitor,
            values,
            masks,
            device=device,
            dtype="float64",
        )
        tracker.update(raw)
        nis = _validated_nis(
            raw.get("nis"),
            (len(chunk), sequence_length),
            negative_tolerance,
        )
        for index, run in enumerate(chunk):
            by_run[str(run.name)] = nis[index].copy()

    execution = tracker.as_dict()
    expected_steps = len(normalized) * sequence_length
    if (
        int(execution["sequences"]) != len(normalized)
        or int(execution["time_steps"]) != expected_steps
    ):
        raise RuntimeError("CUDA execution tracker does not match the scored runs")
    if int(execution["fallback_sequences"]) != 0:
        raise RuntimeError("CUDA NIS scoring required fallback sequence replays")
    if int(execution["factorization_failures"]) != 0:
        raise RuntimeError("CUDA NIS scoring encountered factorization failures")

    return CompleteRunNISResult(
        by_run=by_run,
        execution=execution,
        runtime_environment=runtime,
        sequence_length=sequence_length,
        feature_count=RIETH_FEATURE_COUNT,
    )


def validate_healthy_run_nis_cuda(
    monitor: object,
    run: SensorSequence,
    *,
    device: str = "cuda",
    tolerance: float = 1e-10,
    negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE,
) -> dict[str, object]:
    """Fail unless one healthy run has matching CPU/CUDA NIS and exact batching."""
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    negative_tolerance = _validate_negative_tolerance(negative_tolerance)
    mask = _complete_mask(run)
    if np.asarray(run.fault, dtype=bool).any():
        raise ValueError("CPU/CUDA NIS validation requires a healthy run")

    filter_model = getattr(monitor, "filter_model", None)
    if filter_model is None:
        raise ValueError("monitor does not contain a filter_model")
    cpu = np.asarray(
        filter_model.filter(run.x, mask).normalized_innovation,
        dtype=np.float64,
    )
    if cpu.shape != (run.x.shape[0],) or not np.isfinite(cpu).all():
        raise RuntimeError("CPU reference NIS is invalid")
    if float(np.min(cpu)) < -negative_tolerance:
        raise RuntimeError("CPU reference NIS contains a materially negative value")
    cpu = np.maximum(cpu, 0.0)

    values = np.stack([run.x, run.x]).astype(np.float64, copy=False)
    masks = np.stack([mask, mask])
    runtime = cuda_runtime_metadata(device)
    raw = score_incomplete_batch(
        monitor,
        values,
        masks,
        device=device,
        dtype="float64",
    )
    tracker = CUDAExecutionTracker()
    tracker.update(raw)
    gpu = _validated_nis(
        raw.get("nis"),
        (2, run.x.shape[0]),
        negative_tolerance,
    )
    difference = gpu[0] - cpu
    maximum_absolute = float(np.max(np.abs(difference)))
    root_mean_square = float(np.sqrt(np.mean(difference**2)))
    within_tolerance = bool(
        np.allclose(gpu[0], cpu, atol=tolerance, rtol=tolerance)
    )
    duplicate_maximum_absolute = float(np.max(np.abs(gpu[0] - gpu[1])))
    duplicate_exact = bool(np.array_equal(gpu[0], gpu[1]))
    passed = bool(
        within_tolerance
        and duplicate_exact
        and int(raw.get("fallback_sequence_count", 0)) == 0
        and int(raw.get("factorization_failure_count", 0)) == 0
    )
    result: dict[str, object] = {
        "passed": passed,
        "run": str(run.name),
        "sequence_length": int(run.x.shape[0]),
        "feature_count": int(run.x.shape[1]),
        "absolute_tolerance": tolerance,
        "relative_tolerance": tolerance,
        "maximum_absolute": maximum_absolute,
        "root_mean_square": root_mean_square,
        "within_tolerance": within_tolerance,
        "batch_duplicate_exact": duplicate_exact,
        "batch_duplicate_maximum_absolute": duplicate_maximum_absolute,
        "runtime_environment": runtime,
        "cuda_execution": tracker.as_dict(),
    }
    if not passed:
        raise RuntimeError(f"CPU/CUDA NIS validation failed: {result}")
    return result


__all__ = [
    "CompleteRunNISResult",
    "score_complete_runs_cuda",
    "validate_healthy_run_nis_cuda",
]
