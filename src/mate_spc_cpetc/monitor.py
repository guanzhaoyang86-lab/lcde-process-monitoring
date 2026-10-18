"""Empirical-tail calibration and causal top-tail scan for CP-ETC-Scan v5.

The module intentionally operates on already-computed monitor-endpoint logits.  It
does not know how data were loaded or how the three expert models were trained.
Every public numerical input is required to be finite, exact ``float64`` so that
calibration and semantic replay cannot silently change precision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


FAULT_IDS = (3, 9, 15)
EXPERT_COUNT = len(FAULT_IDS)
MONITOR_START = 160
MONITOR_STOP = 500
MONITOR_STRIDE = 4
MONITOR_ENDPOINTS = tuple(range(MONITOR_START, MONITOR_STOP, MONITOR_STRIDE))
MONITOR_COUNT = len(MONITOR_ENDPOINTS)
SCAN_WINDOW = 8
SCAN_TOP_K = 2
CALIBRATION_RUNS = 20
MISS_DELAY = MONITOR_STOP - MONITOR_START
NONDEGENERATE_DISTINCT_MINIMUM = 19


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_logits(
    value: Any,
    *,
    name: str,
    exact_runs: int | None = None,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(value)
    single = array.ndim == 2
    if single:
        array = array[None, :, :]
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or tuple(array.shape[1:]) != (MONITOR_COUNT, EXPERT_COUNT)
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be finite exact float64 with shape "
            f"[runs,{MONITOR_COUNT},{EXPERT_COUNT}] or "
            f"[{MONITOR_COUNT},{EXPERT_COUNT}]"
        )
    if exact_runs is not None and array.shape[0] != exact_runs:
        raise ValueError(f"{name} must contain exactly {exact_runs} runs")
    return array, single


def _validate_pvalues(value: Any, *, name: str = "pvalues") -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (MONITOR_COUNT, EXPERT_COUNT)
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
        or np.any(array > 1.0)
    ):
        raise ValueError(
            f"{name} must be finite exact float64 in (0,1] with shape "
            f"[{MONITOR_COUNT},{EXPERT_COUNT}]"
        )
    return array


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ExpertTailCalibrator:
    """Per-expert empirical survival maps fit on healthy monitor endpoints.

    For an expert with ``N`` calibration logits, the tail probability is exactly
    ``(1 + number(calibration_logit >= query_logit)) / (N + 1)``.  The left-sided
    search used below is deliberate: every calibration value tied with the query
    is included in the tail count.
    """

    sorted_logits: np.ndarray
    calibration_run_count: int
    endpoint_count: int = MONITOR_COUNT
    fault_ids: tuple[int, ...] = FAULT_IDS

    def __post_init__(self) -> None:
        sorted_logits = np.asarray(self.sorted_logits)
        run_count = _plain_int(self.calibration_run_count, name="calibration_run_count")
        endpoint_count = _plain_int(self.endpoint_count, name="endpoint_count")
        fault_ids = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        expected_samples = run_count * endpoint_count
        if (
            run_count < 1
            or endpoint_count != MONITOR_COUNT
            or fault_ids != FAULT_IDS
            or sorted_logits.shape != (EXPERT_COUNT, expected_samples)
            or sorted_logits.dtype != np.dtype(np.float64)
            or not np.isfinite(sorted_logits).all()
            or np.any(np.diff(sorted_logits, axis=1) < 0.0)
        ):
            raise ValueError("empirical tail calibration state is invalid")
        object.__setattr__(
            self,
            "sorted_logits",
            _readonly_copy(sorted_logits, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "calibration_run_count", run_count)
        object.__setattr__(self, "endpoint_count", endpoint_count)
        object.__setattr__(self, "fault_ids", fault_ids)

    @property
    def samples_per_expert(self) -> int:
        return int(self.sorted_logits.shape[1])

    @property
    def minimum_pvalue(self) -> float:
        return 1.0 / float(self.samples_per_expert + 1)

    @classmethod
    def fit(cls, calibration_healthy_logits: Any) -> "ExpertTailCalibrator":
        values, _ = _validate_logits(
            calibration_healthy_logits,
            name="calibration_healthy_logits",
        )
        flattened = values.reshape(-1, EXPERT_COUNT)
        sorted_logits = np.sort(flattened, axis=0).T.astype(np.float64, copy=False)
        return cls(sorted_logits, int(values.shape[0]))

    def pvalues(self, logits: Any) -> np.ndarray:
        values, single = _validate_logits(logits, name="logits")
        flattened = values.reshape(-1, EXPERT_COUNT)
        result = np.empty_like(flattened, dtype=np.float64)
        denominator = float(self.samples_per_expert + 1)
        for expert_index in range(EXPERT_COUNT):
            ordered = self.sorted_logits[expert_index]
            # side="left" includes all values equal to the query in the >= count.
            first_equal = np.searchsorted(
                ordered,
                flattened[:, expert_index],
                side="left",
            )
            greater_or_equal = self.samples_per_expert - first_equal
            result[:, expert_index] = (1.0 + greater_or_equal) / denominator
        restored = result.reshape(values.shape)
        if not np.isfinite(restored).all():
            raise FloatingPointError("empirical tail calibration produced nonfinite p-values")
        output = restored[0] if single else restored
        return _readonly_copy(output, dtype=np.dtype(np.float64))


@dataclass(frozen=True)
class CPETCMonitorPath:
    """Fully calibrated causal monitor path for one 85-endpoint run."""

    pvalues: np.ndarray
    fused_pvalues: np.ndarray
    evidence: np.ndarray
    scan_scores: np.ndarray
    winning_expert_indices: np.ndarray
    eligible: np.ndarray
    run_maximum: float
    fault_ids: tuple[int, ...] = FAULT_IDS

    def __post_init__(self) -> None:
        pvalues = _validate_pvalues(self.pvalues)
        fused = np.asarray(self.fused_pvalues)
        evidence = np.asarray(self.evidence)
        scan = np.asarray(self.scan_scores)
        winners = np.asarray(self.winning_expert_indices)
        eligible = np.asarray(self.eligible)
        maximum = _finite_float(self.run_maximum, name="run_maximum")
        fault_ids = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        if (
            fused.shape != (MONITOR_COUNT,)
            or evidence.shape != (MONITOR_COUNT,)
            or scan.shape != (MONITOR_COUNT,)
            or winners.shape != (MONITOR_COUNT,)
            or eligible.shape != (MONITOR_COUNT,)
            or fused.dtype != np.dtype(np.float64)
            or evidence.dtype != np.dtype(np.float64)
            or scan.dtype != np.dtype(np.float64)
            or winners.dtype != np.dtype(np.int64)
            or eligible.dtype != np.dtype(np.bool_)
            or not np.isfinite(fused).all()
            or not np.isfinite(evidence).all()
            or not np.isfinite(scan).all()
            or np.any(fused <= 0.0)
            or np.any(fused > 1.0)
            or np.any(evidence < 0.0)
            or np.any(scan < 0.0)
            or np.any(winners < 0)
            or np.any(winners >= EXPERT_COUNT)
            or fault_ids != FAULT_IDS
        ):
            raise ValueError("monitor path arrays have invalid shape, dtype, or values")

        expected_fused = np.minimum(1.0, EXPERT_COUNT * np.min(pvalues, axis=1))
        expected_evidence = -np.log(expected_fused)
        expected_winners = np.argmin(pvalues, axis=1).astype(np.int64)
        expected_eligible = np.arange(MONITOR_COUNT) >= SCAN_TOP_K - 1
        expected_scan = _top_tail_scan(expected_evidence)
        expected_maximum = float(np.max(expected_scan[expected_eligible]))
        if (
            not np.array_equal(fused, expected_fused)
            or not np.array_equal(evidence, expected_evidence)
            or not np.array_equal(winners, expected_winners)
            or not np.array_equal(eligible, expected_eligible)
            or not np.array_equal(scan, expected_scan)
            or maximum != expected_maximum
        ):
            raise ValueError("monitor path is internally inconsistent")

        for name, value, dtype in (
            ("pvalues", pvalues, np.dtype(np.float64)),
            ("fused_pvalues", fused, np.dtype(np.float64)),
            ("evidence", evidence, np.dtype(np.float64)),
            ("scan_scores", scan, np.dtype(np.float64)),
            ("winning_expert_indices", winners, np.dtype(np.int64)),
            ("eligible", eligible, np.dtype(np.bool_)),
        ):
            object.__setattr__(self, name, _readonly_copy(value, dtype=dtype))
        object.__setattr__(self, "run_maximum", maximum)
        object.__setattr__(self, "fault_ids", fault_ids)


def _top_tail_scan(evidence: np.ndarray) -> np.ndarray:
    if (
        evidence.shape != (MONITOR_COUNT,)
        or evidence.dtype != np.dtype(np.float64)
        or not np.isfinite(evidence).all()
        or np.any(evidence < 0.0)
    ):
        raise ValueError("evidence must be a finite nonnegative float64 monitor vector")
    scan = np.zeros(MONITOR_COUNT, dtype=np.float64)
    for endpoint_index in range(SCAN_TOP_K - 1, MONITOR_COUNT):
        start = max(0, endpoint_index - SCAN_WINDOW + 1)
        current = evidence[start : endpoint_index + 1]
        top = np.partition(current, current.size - SCAN_TOP_K)[-SCAN_TOP_K:]
        scan[endpoint_index] = float(np.sum(top, dtype=np.float64))
    return scan


def monitor_path_from_pvalues(pvalues: Any) -> CPETCMonitorPath:
    """Apply Bonferroni fusion and the fixed latest-8/top-2 causal scan."""

    values = _validate_pvalues(pvalues)
    fused = np.minimum(1.0, EXPERT_COUNT * np.min(values, axis=1)).astype(
        np.float64, copy=False
    )
    evidence = (-np.log(fused)).astype(np.float64, copy=False)
    winners = np.argmin(values, axis=1).astype(np.int64)
    eligible = (np.arange(MONITOR_COUNT) >= SCAN_TOP_K - 1).astype(np.bool_)
    scan = _top_tail_scan(evidence)
    maximum = float(np.max(scan[eligible]))
    return CPETCMonitorPath(
        values,
        fused,
        evidence,
        scan,
        winners,
        eligible,
        maximum,
    )


def fused_monitor_path(
    logits: Any,
    calibrator: ExpertTailCalibrator,
) -> CPETCMonitorPath:
    """Calibrate and monitor one run of three-expert endpoint logits."""

    if not isinstance(calibrator, ExpertTailCalibrator):
        raise TypeError("calibrator must be ExpertTailCalibrator")
    values, single = _validate_logits(logits, name="logits")
    if not single:
        raise ValueError("fused_monitor_path accepts exactly one monitor run")
    return monitor_path_from_pvalues(calibrator.pvalues(values[0]))


def leave_one_out_run_maxima(calibration_healthy_logits: Any) -> np.ndarray:
    """Return 20 run maxima, fitting each ECDF after fully excluding that run."""

    values, _ = _validate_logits(
        calibration_healthy_logits,
        name="calibration_healthy_logits",
        exact_runs=CALIBRATION_RUNS,
    )
    maxima = np.empty(CALIBRATION_RUNS, dtype=np.float64)
    for run_index in range(CALIBRATION_RUNS):
        keep = np.arange(CALIBRATION_RUNS) != run_index
        calibrator = ExpertTailCalibrator.fit(values[keep])
        maxima[run_index] = fused_monitor_path(values[run_index], calibrator).run_maximum
    if not np.isfinite(maxima).all():
        raise FloatingPointError("leave-one-out calibration produced nonfinite run maxima")
    return _readonly_copy(maxima, dtype=np.dtype(np.float64))


@dataclass(frozen=True)
class LOOThresholdCalibration:
    """Maximum/rank-20 threshold from leave-one-run-out healthy scores."""

    threshold: float
    run_maxima: np.ndarray
    rank19: float
    distinct_maxima: int

    def __post_init__(self) -> None:
        threshold = _finite_float(self.threshold, name="threshold")
        rank19 = _finite_float(self.rank19, name="rank19")
        maxima = np.asarray(self.run_maxima)
        distinct = _plain_int(self.distinct_maxima, name="distinct_maxima")
        if (
            maxima.shape != (CALIBRATION_RUNS,)
            or maxima.dtype != np.dtype(np.float64)
            or not np.isfinite(maxima).all()
            or threshold != float(np.max(maxima))
            or rank19 != float(np.sort(maxima)[-2])
            or distinct != int(np.unique(maxima).size)
        ):
            raise ValueError("leave-one-out threshold calibration is inconsistent")
        object.__setattr__(
            self,
            "run_maxima",
            _readonly_copy(maxima, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rank19", rank19)
        object.__setattr__(self, "distinct_maxima", distinct)

    @property
    def adjacent_tail_gap(self) -> float:
        return self.threshold - self.rank19

    @property
    def numerical_tolerance(self) -> float:
        scale = max(1.0, abs(self.threshold), abs(self.rank19))
        return 100.0 * np.finfo(np.float64).eps * scale

    @property
    def nondegenerate(self) -> bool:
        return (
            self.distinct_maxima >= NONDEGENERATE_DISTINCT_MINIMUM
            and self.adjacent_tail_gap > self.numerical_tolerance
        )

    def require_nondegenerate(self) -> None:
        if not self.nondegenerate:
            raise ValueError(
                "leave-one-out threshold is degenerate: require at least "
                f"{NONDEGENERATE_DISTINCT_MINIMUM} distinct maxima and a positive top gap"
            )


def fit_loo_threshold(
    calibration_healthy_logits: Any,
    *,
    require_nondegenerate: bool = True,
) -> LOOThresholdCalibration:
    """Fit the strict threshold as the maximum of 20 LOO healthy run maxima."""

    if type(require_nondegenerate) is not bool:
        raise ValueError("require_nondegenerate must be a bool")
    maxima = leave_one_out_run_maxima(calibration_healthy_logits)
    ordered = np.sort(maxima)
    result = LOOThresholdCalibration(
        float(ordered[-1]),
        maxima,
        float(ordered[-2]),
        int(np.unique(maxima).size),
    )
    if require_nondegenerate:
        result.require_nondegenerate()
    return result


def first_strict_alarm(path: CPETCMonitorPath, threshold: float) -> int | None:
    """Return the first endpoint position with scan score strictly above threshold."""

    if not isinstance(path, CPETCMonitorPath):
        raise TypeError("path must be CPETCMonitorPath")
    boundary = _finite_float(threshold, name="threshold")
    indices = np.flatnonzero(path.eligible & (path.scan_scores > boundary))
    return None if indices.size == 0 else int(indices[0])


def detection_delay(first_alarm_endpoint: int | None) -> int:
    """Return delay from endpoint 160, using the frozen miss penalty of 340."""

    if first_alarm_endpoint is None:
        return MISS_DELAY
    endpoint = _plain_int(first_alarm_endpoint, name="first_alarm_endpoint")
    if endpoint not in MONITOR_ENDPOINTS:
        raise ValueError("first_alarm_endpoint must be a frozen monitor endpoint")
    return endpoint - MONITOR_START


@dataclass(frozen=True)
class AlarmDecision:
    alarm: bool
    first_alarm_position: int | None
    first_alarm_endpoint: int | None
    attributed_fault: int | None
    delay: int
    run_maximum: float
    threshold: float

    def __post_init__(self) -> None:
        if type(self.alarm) is not bool:
            raise ValueError("alarm must be a bool")
        maximum = _finite_float(self.run_maximum, name="run_maximum")
        threshold = _finite_float(self.threshold, name="threshold")
        delay = _plain_int(self.delay, name="delay")
        if self.alarm:
            position = _plain_int(self.first_alarm_position, name="first_alarm_position")
            endpoint = _plain_int(self.first_alarm_endpoint, name="first_alarm_endpoint")
            fault = _plain_int(self.attributed_fault, name="attributed_fault")
            if (
                position < SCAN_TOP_K - 1
                or position >= MONITOR_COUNT
                or endpoint != MONITOR_ENDPOINTS[position]
                or fault not in FAULT_IDS
                or delay != detection_delay(endpoint)
                or maximum <= threshold
            ):
                raise ValueError("alarming decision fields are inconsistent")
        elif (
            self.first_alarm_position is not None
            or self.first_alarm_endpoint is not None
            or self.attributed_fault is not None
            or delay != MISS_DELAY
            or maximum > threshold
        ):
            raise ValueError("non-alarming decision fields are inconsistent")
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "run_maximum", maximum)
        object.__setattr__(self, "threshold", threshold)


def apply_strict_threshold(path: CPETCMonitorPath, threshold: float) -> AlarmDecision:
    """Apply ``S > h`` and attribute an alarm to the current smallest-p expert."""

    if not isinstance(path, CPETCMonitorPath):
        raise TypeError("path must be CPETCMonitorPath")
    boundary = _finite_float(threshold, name="threshold")
    position = first_strict_alarm(path, boundary)
    endpoint = None if position is None else MONITOR_ENDPOINTS[position]
    fault = (
        None
        if position is None
        else path.fault_ids[int(path.winning_expert_indices[position])]
    )
    return AlarmDecision(
        position is not None,
        position,
        endpoint,
        fault,
        detection_delay(endpoint),
        path.run_maximum,
        boundary,
    )


__all__ = [
    "CALIBRATION_RUNS",
    "EXPERT_COUNT",
    "FAULT_IDS",
    "MISS_DELAY",
    "MONITOR_COUNT",
    "MONITOR_ENDPOINTS",
    "MONITOR_START",
    "MONITOR_STOP",
    "MONITOR_STRIDE",
    "NONDEGENERATE_DISTINCT_MINIMUM",
    "SCAN_TOP_K",
    "SCAN_WINDOW",
    "AlarmDecision",
    "CPETCMonitorPath",
    "ExpertTailCalibrator",
    "LOOThresholdCalibration",
    "apply_strict_threshold",
    "detection_delay",
    "first_strict_alarm",
    "fit_loo_threshold",
    "fused_monitor_path",
    "leave_one_out_run_maxima",
    "monitor_path_from_pvalues",
]
