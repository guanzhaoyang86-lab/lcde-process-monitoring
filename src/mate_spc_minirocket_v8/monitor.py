"""Run-relative log-odds monitoring for MiniROCKET v8.

All sequential arithmetic in this module is NumPy float64 on CPU, as fixed by
the contingent method charter.  The module has no data-loading side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np


PREFIX_ENDPOINTS = np.arange(128, 157, 4, dtype=np.int64)
MONITOR_START = 160
MONITOR_STOP = 500
MONITOR_ENDPOINTS = np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64)
PREFIX_COUNT = 8
MONITOR_LENGTH = 340
COMPLETE_SCORE_LENGTH = PREFIX_COUNT + MONITOR_LENGTH
FIT_HEALTHY_RUN_COUNT = 60
INNER_HEALTHY_RUN_COUNT = 20
MISS_DELAY = 340
CUSUM_DRIFT = 0.25
INTERMITTENT_FIRST_ENDPOINT = 175
INTERMITTENT_WINDOW = 16
INTERMITTENT_TOP_R = 4
INTERMITTENT_COEFFICIENT = 0.5
CALIBRATION_SCALE_FLOOR = 1e-3
MINIMUM_DISTINCT_MAXIMA = 18
REPLAY_ERROR_FLOOR = 1e-6

if PREFIX_ENDPOINTS.tolist() != [128, 132, 136, 140, 144, 148, 152, 156]:
    raise RuntimeError("v8 prefix endpoints changed")
if MONITOR_ENDPOINTS.size != MONITOR_LENGTH or MONITOR_ENDPOINTS[-1] != 499:
    raise RuntimeError("v8 monitor endpoints changed")


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{name} must be an integer (boolean values are invalid)")
    return int(value)


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number (boolean values are invalid)")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    last_dimension: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype.kind not in {"i", "u", "f"} or value.dtype.kind == "b":
        raise ValueError(f"{name} must have a real numeric dtype")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if last_dimension is not None and (value.ndim < 1 or value.shape[-1] != last_dimension):
        raise ValueError(f"{name} must end in dimension {last_dimension}")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _readonly(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def compute_log_odds(logits: np.ndarray) -> np.ndarray:
    """Return ``logsumexp(F3,F9,F15) - H`` in CPU float64."""

    values = _float_array(logits, name="logits", last_dimension=4)
    faults = values[..., 1:]
    maximum = np.max(faults, axis=-1)
    result = maximum + np.log(
        np.sum(np.exp(faults - maximum[..., None]), axis=-1, dtype=np.float64)
    ) - values[..., 0]
    if not np.isfinite(result).all():
        raise FloatingPointError("log-odds calculation produced nonfinite values")
    return np.asarray(result, dtype=np.float64)


def _run_relative_values(q: np.ndarray, prefix_q: np.ndarray) -> np.ndarray:
    dense = _float_array(q, name="q")
    prefix = _float_array(prefix_q, name="prefix_q")
    if dense.ndim == 1:
        if dense.shape != (MONITOR_LENGTH,) or prefix.shape != (PREFIX_COUNT,):
            raise ValueError("q/prefix_q must have shapes (340,) and (8,)")
        baseline = float(np.median(prefix))
        result = dense - baseline
    elif dense.ndim == 2:
        if dense.shape[0] < 1 or dense.shape[1] != MONITOR_LENGTH:
            raise ValueError("q must have shape (n,340) with n >= 1")
        if prefix.shape != (dense.shape[0], PREFIX_COUNT):
            raise ValueError("prefix_q must have aligned shape (n,8)")
        baseline = np.median(prefix, axis=1)
        result = dense - baseline[:, None]
    else:
        raise ValueError("q must be one- or two-dimensional")
    if not np.isfinite(result).all():
        raise FloatingPointError("run-relative evidence produced nonfinite values")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MonitorCalibration:
    """Fit-healthy robust center and scale of run-relative evidence."""

    center: float
    scale: float
    pool_size: int = FIT_HEALTHY_RUN_COUNT * MONITOR_LENGTH

    def __post_init__(self) -> None:
        center = _strict_float(self.center, name="center")
        scale = _strict_float(self.scale, name="scale")
        pool_size = _strict_int(self.pool_size, name="pool_size")
        if scale < CALIBRATION_SCALE_FLOOR:
            raise ValueError("monitor scale violates the frozen floor")
        if pool_size != FIT_HEALTHY_RUN_COUNT * MONITOR_LENGTH:
            raise ValueError("monitor calibration pool size must be exactly 20,400")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "pool_size", pool_size)

    def to_document(self) -> dict[str, float | int]:
        return {
            "center": self.center,
            "scale": self.scale,
            "pool_size": self.pool_size,
            "scale_floor": CALIBRATION_SCALE_FLOOR,
        }


def fit_monitor_calibration(fit_healthy_q: np.ndarray) -> MonitorCalibration:
    """Fit calibration from 60 complete fit-healthy q sequences only.

    Columns 0..7 are prefix endpoints 128..156; columns 8..347 are dense
    monitor endpoints 160..499.
    """

    complete = _float_array(
        fit_healthy_q,
        name="fit_healthy_q",
        shape=(FIT_HEALTHY_RUN_COUNT, COMPLETE_SCORE_LENGTH),
    )
    prefix = complete[:, :PREFIX_COUNT]
    dense = complete[:, PREFIX_COUNT:]
    relative = _run_relative_values(dense, prefix)
    pooled = relative.reshape(-1)
    center = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - center)))
    scale = max(1.4826 * mad, CALIBRATION_SCALE_FLOOR)
    if not math.isfinite(center) or not math.isfinite(scale):
        raise FloatingPointError("monitor calibration produced nonfinite state")
    return MonitorCalibration(center=center, scale=scale)


def scan_run(
    q: np.ndarray, prefix_q: np.ndarray, calibration: MonitorCalibration
) -> np.ndarray:
    """Compute the exact CUSUM/intermittent combined path for one run."""

    if not isinstance(calibration, MonitorCalibration):
        raise TypeError("calibration must be MonitorCalibration")
    relative = _run_relative_values(q, prefix_q)
    if relative.ndim != 1:
        raise ValueError("scan_run accepts exactly one run")
    u = (relative - calibration.center) / calibration.scale
    if not np.isfinite(u).all():
        raise FloatingPointError("standardized run evidence is nonfinite")
    path = np.empty(MONITOR_LENGTH, dtype=np.float64)
    cusum = 0.0
    first_intermittent_index = INTERMITTENT_FIRST_ENDPOINT - MONITOR_START
    for index in range(MONITOR_LENGTH):
        cusum = max(0.0, cusum + float(u[index]) - CUSUM_DRIFT)
        if index < first_intermittent_index:
            path[index] = cusum
            continue
        positive_window = np.maximum(
            u[index - INTERMITTENT_WINDOW + 1 : index + 1], 0.0
        )
        largest = np.partition(
            positive_window, positive_window.size - INTERMITTENT_TOP_R
        )[-INTERMITTENT_TOP_R:]
        intermittent = INTERMITTENT_COEFFICIENT * float(
            np.sum(largest, dtype=np.float64)
        )
        path[index] = max(cusum, intermittent)
    if not np.isfinite(path).all() or np.any(path < 0.0):
        raise FloatingPointError("sequential scan produced invalid scores")
    return path


@dataclass(frozen=True, slots=True)
class SequentialScoreBatch:
    """Aligned combined paths and run maxima."""

    paths: np.ndarray
    run_maxima: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        paths = _float_array(self.paths, name="paths")
        if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] != MONITOR_LENGTH:
            raise ValueError("paths must have shape (n,340) with n >= 1")
        if np.any(paths < 0.0):
            raise ValueError("paths must be nonnegative")
        maxima = np.max(paths, axis=1)
        object.__setattr__(self, "paths", _readonly(paths, dtype=np.dtype(np.float64)))
        object.__setattr__(
            self, "run_maxima", _readonly(maxima, dtype=np.dtype(np.float64))
        )

    @property
    def endpoints(self) -> np.ndarray:
        return _readonly(MONITOR_ENDPOINTS, dtype=np.dtype(np.int64))


def scan_batch(
    q: np.ndarray, prefix_q: np.ndarray, calibration: MonitorCalibration
) -> SequentialScoreBatch:
    """Scan aligned batches while preserving input run order."""

    dense = _float_array(q, name="q")
    prefix = _float_array(prefix_q, name="prefix_q")
    if dense.ndim != 2 or dense.shape[0] < 1 or dense.shape[1] != MONITOR_LENGTH:
        raise ValueError("q must have shape (n,340) with n >= 1")
    if prefix.shape != (dense.shape[0], PREFIX_COUNT):
        raise ValueError("prefix_q must have aligned shape (n,8)")
    paths = np.empty_like(dense, dtype=np.float64)
    for index in range(dense.shape[0]):
        paths[index] = scan_run(dense[index], prefix[index], calibration)
    return SequentialScoreBatch(paths=paths)


@dataclass(frozen=True, slots=True)
class InnerThreshold:
    """Strict rank-20 threshold with replay-separation evidence."""

    threshold: float
    rank19: float
    distinct_count: int
    repeat_error: float

    def __post_init__(self) -> None:
        threshold = _strict_float(self.threshold, name="threshold")
        rank19 = _strict_float(self.rank19, name="rank19")
        distinct = _strict_int(self.distinct_count, name="distinct_count")
        error = _strict_float(self.repeat_error, name="repeat_error")
        if threshold < 0.0 or rank19 < 0.0 or rank19 > threshold:
            raise ValueError("threshold order must satisfy 0 <= rank19 <= threshold")
        if not MINIMUM_DISTINCT_MAXIMA <= distinct <= INNER_HEALTHY_RUN_COUNT:
            raise ValueError("inner maxima fail the minimum distinct-count gate")
        if error < 0.0:
            raise ValueError("repeat_error must be nonnegative")
        tolerance = 100.0 * max(error, REPLAY_ERROR_FLOOR)
        if threshold - rank19 <= tolerance:
            raise ValueError("rank20-rank19 fails the replay-separation gate")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rank19", rank19)
        object.__setattr__(self, "distinct_count", distinct)
        object.__setattr__(self, "repeat_error", error)

    @property
    def top_gap(self) -> float:
        return self.threshold - self.rank19

    @property
    def numeric_tolerance(self) -> float:
        return 100.0 * max(self.repeat_error, REPLAY_ERROR_FLOOR)

    def to_document(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "rank19": self.rank19,
            "rank20_minus_rank19": self.top_gap,
            "distinct_count": self.distinct_count,
            "minimum_distinct_count": MINIMUM_DISTINCT_MAXIMA,
            "repeat_error": self.repeat_error,
            "numeric_tolerance": self.numeric_tolerance,
            "comparison": "strictly_greater_than",
        }


def inner_threshold(
    maxima: Sequence[float] | np.ndarray, repeat_error: float
) -> InnerThreshold:
    """Fit the exact maximum-of-20 threshold or fail closed."""

    raw = np.asarray(maxima)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise ValueError("maxima must have a real numeric dtype")
    values = np.asarray(maxima, dtype=np.float64)
    if values.shape != (INNER_HEALTHY_RUN_COUNT,) or not np.isfinite(values).all():
        raise ValueError("maxima must contain exactly 20 finite values")
    if np.any(values < 0.0):
        raise ValueError("maxima must be nonnegative")
    error = _strict_float(repeat_error, name="repeat_error")
    ordered = np.sort(values, kind="stable")
    return InnerThreshold(
        threshold=float(ordered[-1]),
        rank19=float(ordered[-2]),
        distinct_count=int(np.unique(values).size),
        repeat_error=error,
    )


@dataclass(frozen=True, slots=True)
class ThresholdDecisions:
    """Strict-crossing decisions for one aligned score batch."""

    alarms: np.ndarray
    alarm_endpoints: np.ndarray
    delays: np.ndarray
    run_maxima: np.ndarray
    threshold: float

    def __post_init__(self) -> None:
        alarms = np.asarray(self.alarms)
        endpoints = np.asarray(self.alarm_endpoints)
        delays = np.asarray(self.delays)
        maxima = np.asarray(self.run_maxima)
        count = alarms.size if alarms.ndim == 1 else -1
        if count < 1 or alarms.dtype != np.dtype(np.bool_):
            raise ValueError("alarms must be a nonempty one-dimensional bool array")
        if endpoints.dtype != np.dtype(np.int64) or endpoints.shape != (count,):
            raise ValueError("alarm_endpoints must be aligned int64")
        if delays.dtype != np.dtype(np.int64) or delays.shape != (count,):
            raise ValueError("delays must be aligned int64")
        if maxima.dtype != np.dtype(np.float64) or maxima.shape != (count,):
            raise ValueError("run_maxima must be aligned float64")
        threshold = _strict_float(self.threshold, name="threshold")
        if threshold < 0.0 or not np.isfinite(maxima).all() or np.any(maxima < 0.0):
            raise ValueError("threshold and run maxima must be finite and nonnegative")
        if np.any(endpoints[alarms] < MONITOR_START) or np.any(
            endpoints[alarms] >= MONITOR_STOP
        ):
            raise ValueError("alarmed endpoints must lie in 160..499")
        if np.any(endpoints[~alarms] != -1):
            raise ValueError("misses must use endpoint -1")
        expected_delays = np.where(alarms, endpoints - MONITOR_START, MISS_DELAY)
        if not np.array_equal(delays, expected_delays):
            raise ValueError("delays are inconsistent with alarm endpoints")
        if not np.array_equal(alarms, maxima > threshold):
            raise ValueError("alarms are inconsistent with strict thresholding")
        for name, value, dtype in (
            ("alarms", alarms, np.dtype(np.bool_)),
            ("alarm_endpoints", endpoints, np.dtype(np.int64)),
            ("delays", delays, np.dtype(np.int64)),
            ("run_maxima", maxima, np.dtype(np.float64)),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        object.__setattr__(self, "threshold", threshold)


def apply_threshold(
    scores: SequentialScoreBatch | np.ndarray, threshold: InnerThreshold
) -> ThresholdDecisions:
    """Apply strict ``S[t] > h`` and return the earliest eligible crossing."""

    if not isinstance(threshold, InnerThreshold):
        raise TypeError("threshold must be InnerThreshold")
    if isinstance(scores, SequentialScoreBatch):
        paths = scores.paths
    else:
        paths = _float_array(scores, name="scores")
        if paths.ndim != 2 or paths.shape[0] < 1 or paths.shape[1] != MONITOR_LENGTH:
            raise ValueError("scores must have shape (n,340) with n >= 1")
        if np.any(paths < 0.0):
            raise ValueError("scores must be nonnegative")
    crossings = paths > threshold.threshold
    alarms = np.any(crossings, axis=1).astype(np.bool_)
    first = np.argmax(crossings, axis=1)
    endpoints = np.where(alarms, MONITOR_START + first, -1).astype(np.int64)
    delays = np.where(alarms, first, MISS_DELAY).astype(np.int64)
    maxima = np.max(paths, axis=1).astype(np.float64)
    return ThresholdDecisions(
        alarms=alarms,
        alarm_endpoints=endpoints,
        delays=delays,
        run_maxima=maxima,
        threshold=threshold.threshold,
    )


__all__ = [
    "CALIBRATION_SCALE_FLOOR",
    "COMPLETE_SCORE_LENGTH",
    "CUSUM_DRIFT",
    "FIT_HEALTHY_RUN_COUNT",
    "INNER_HEALTHY_RUN_COUNT",
    "INTERMITTENT_COEFFICIENT",
    "INTERMITTENT_FIRST_ENDPOINT",
    "INTERMITTENT_TOP_R",
    "INTERMITTENT_WINDOW",
    "InnerThreshold",
    "MINIMUM_DISTINCT_MAXIMA",
    "MISS_DELAY",
    "MONITOR_ENDPOINTS",
    "MONITOR_LENGTH",
    "MONITOR_START",
    "MONITOR_STOP",
    "MonitorCalibration",
    "PREFIX_COUNT",
    "PREFIX_ENDPOINTS",
    "REPLAY_ERROR_FLOOR",
    "SequentialScoreBatch",
    "ThresholdDecisions",
    "apply_threshold",
    "compute_log_odds",
    "fit_monitor_calibration",
    "inner_threshold",
    "scan_batch",
    "scan_run",
]
