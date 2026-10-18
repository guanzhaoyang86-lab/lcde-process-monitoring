"""Run-level maxT monitor for the frozen three-expert CF-GLR v4 method."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


MONITOR_SCHEMA = "mate_spc_cf_glr_monitor_standardizer_v1"
MONITOR_SCHEMA_VERSION = 1
FAULT_IDS = (3, 9, 15)
RUN_LENGTH = 500
MONITOR_START = 160
CALIBRATION_RUNS = 20
CALIBRATION_RANK = 20
MAD_CONSISTENCY = 1.4826
MINIMUM_SCALE = 1.0e-12


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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
        or tuple(array.shape[1:]) != (RUN_LENGTH, len(FAULT_IDS))
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must be finite exact float64 with shape [runs,500,3] or [500,3]"
        )
    if exact_runs is not None and array.shape[0] != exact_runs:
        raise ValueError(f"{name} must contain exactly {exact_runs} runs")
    return array, single


def _json_float_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) not in (int, float) for item in value)
    ):
        raise ValueError(f"{name} must be a JSON numeric list of length {length}")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FittedExpertStandardizer:
    """Per-expert median/MAD fit only on fitting-fold healthy point logits."""

    center: np.ndarray
    scale: np.ndarray
    scale_floor: float
    fault_ids: tuple[int, ...] = FAULT_IDS
    monitor_start: int = MONITOR_START

    def __post_init__(self) -> None:
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        floor = _finite_float(self.scale_floor, name="scale_floor")
        faults = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        start = _plain_int(self.monitor_start, name="monitor_start")
        if faults != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        if start != MONITOR_START:
            raise ValueError(f"monitor_start must be exactly {MONITOR_START}")
        if (
            center.shape != (len(FAULT_IDS),)
            or scale.shape != (len(FAULT_IDS),)
            or center.dtype != np.dtype(np.float64)
            or scale.dtype != np.dtype(np.float64)
            or not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
            or floor <= 0.0
            or np.any(scale < floor)
        ):
            raise ValueError("standardizer center/scale are invalid exact-float64 vectors")
        center_copy = np.array(center, dtype=np.float64, copy=True, order="C")
        scale_copy = np.array(scale, dtype=np.float64, copy=True, order="C")
        center_copy.setflags(write=False)
        scale_copy.setflags(write=False)
        object.__setattr__(self, "center", center_copy)
        object.__setattr__(self, "scale", scale_copy)
        object.__setattr__(self, "scale_floor", floor)
        object.__setattr__(self, "fault_ids", faults)
        object.__setattr__(self, "monitor_start", start)

    @classmethod
    def fit(cls, fit_healthy_logits: Any) -> "FittedExpertStandardizer":
        """Fit across all monitored point logits, never calibration/test logits."""

        values, _ = _validate_logits(fit_healthy_logits, name="fit_healthy_logits")
        monitored = values[:, MONITOR_START:, :].reshape(-1, len(FAULT_IDS))
        center = np.median(monitored, axis=0).astype(np.float64, copy=False)
        raw_scale = (
            MAD_CONSISTENCY * np.median(np.abs(monitored - center[None, :]), axis=0)
        ).astype(np.float64, copy=False)
        positive = raw_scale[raw_scale > MINIMUM_SCALE]
        if positive.size == 0:
            raise ValueError("all fit-healthy expert logits have degenerate zero MAD")
        floor = max(
            MINIMUM_SCALE,
            float(np.quantile(positive, 0.10, method="linear")),
        )
        scale = np.maximum(raw_scale, floor).astype(np.float64, copy=False)
        return cls(center, scale, floor)

    def transform(self, logits: Any) -> np.ndarray:
        values, single = _validate_logits(logits, name="logits")
        standardized = (values - self.center[None, None, :]) / self.scale[None, None, :]
        if not np.isfinite(standardized).all():
            raise FloatingPointError("expert standardization produced nonfinite values")
        output = np.array(
            standardized[0] if single else standardized,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        output.setflags(write=False)
        return output

    def to_document(self) -> dict[str, object]:
        return {
            "schema": MONITOR_SCHEMA,
            "schema_version": MONITOR_SCHEMA_VERSION,
            "fault_ids": list(self.fault_ids),
            "monitor_start": self.monitor_start,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "scale_floor": self.scale_floor,
            "fit_scope": "fit_fold_healthy_point_logits_only",
            "location": "per_expert_median_over_monitored_points",
            "scale_method": "1.4826_MAD_with_shared_q10_positive_floor",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "FittedExpertStandardizer":
        required = {
            "schema",
            "schema_version",
            "fault_ids",
            "monitor_start",
            "center",
            "scale",
            "scale_floor",
            "fit_scope",
            "location",
            "scale_method",
        }
        if not isinstance(document, Mapping) or set(document) != required:
            raise ValueError("monitor standardizer document keys are invalid")
        if (
            document["schema"] != MONITOR_SCHEMA
            or type(document["schema_version"]) is not int
            or document["schema_version"] != MONITOR_SCHEMA_VERSION
            or document["fault_ids"] != list(FAULT_IDS)
            or type(document["monitor_start"]) is not int
            or document["monitor_start"] != MONITOR_START
            or document["fit_scope"] != "fit_fold_healthy_point_logits_only"
            or document["location"] != "per_expert_median_over_monitored_points"
            or document["scale_method"]
            != "1.4826_MAD_with_shared_q10_positive_floor"
        ):
            raise ValueError("monitor standardizer document metadata are invalid")
        return cls(
            _json_float_vector(document["center"], name="center", length=3),
            _json_float_vector(document["scale"], name="scale", length=3),
            _finite_float(document["scale_floor"], name="scale_floor"),
        )


@dataclass(frozen=True)
class CFGLRMonitorPath:
    """Dense standardized, union, and two-point-persistent monitor path."""

    standardized_logits: np.ndarray
    union_score: np.ndarray
    persistent_score: np.ndarray
    winning_expert_indices: np.ndarray
    eligible: np.ndarray
    run_maximum: float
    fault_ids: tuple[int, ...] = FAULT_IDS

    def __post_init__(self) -> None:
        standardized = np.asarray(self.standardized_logits)
        union = np.asarray(self.union_score)
        persistent = np.asarray(self.persistent_score)
        winners = np.asarray(self.winning_expert_indices)
        eligible = np.asarray(self.eligible)
        faults = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        maximum = _finite_float(self.run_maximum, name="run_maximum")
        if faults != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        if (
            standardized.shape != (RUN_LENGTH, len(FAULT_IDS))
            or union.shape != (RUN_LENGTH,)
            or persistent.shape != (RUN_LENGTH,)
            or winners.shape != (RUN_LENGTH,)
            or eligible.shape != (RUN_LENGTH,)
            or standardized.dtype != np.dtype(np.float64)
            or union.dtype != np.dtype(np.float64)
            or persistent.dtype != np.dtype(np.float64)
            or winners.dtype != np.dtype(np.int64)
            or eligible.dtype != np.dtype(np.bool_)
            or not np.isfinite(standardized).all()
            or not np.isfinite(union).all()
            or not np.isfinite(persistent).all()
            or np.any(winners < 0)
            or np.any(winners >= len(FAULT_IDS))
        ):
            raise ValueError("monitor path arrays have invalid shape, dtype, or values")
        expected_union = np.max(standardized, axis=1)
        expected_persistent = expected_union.copy()
        expected_persistent[1:] = np.minimum(expected_union[:-1], expected_union[1:])
        expected_eligible = np.arange(RUN_LENGTH) >= MONITOR_START + 1
        expected_winners = np.argmax(standardized, axis=1).astype(np.int64)
        expected_maximum = float(np.max(expected_persistent[expected_eligible]))
        if (
            not np.array_equal(union, expected_union)
            or not np.array_equal(persistent, expected_persistent)
            or not np.array_equal(eligible, expected_eligible)
            or not np.array_equal(winners, expected_winners)
            or maximum != expected_maximum
        ):
            raise ValueError("monitor path is internally inconsistent")
        for name, source in (
            ("standardized_logits", standardized),
            ("union_score", union),
            ("persistent_score", persistent),
            ("winning_expert_indices", winners),
            ("eligible", eligible),
        ):
            copied = np.array(source, copy=True, order="C")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "run_maximum", maximum)
        object.__setattr__(self, "fault_ids", faults)


def fused_monitor_path(
    logits: Any,
    standardizer: FittedExpertStandardizer,
) -> CFGLRMonitorPath:
    """Fuse three expert logits before time-wise two-point persistence."""

    if not isinstance(standardizer, FittedExpertStandardizer):
        raise TypeError("standardizer must be FittedExpertStandardizer")
    values, single = _validate_logits(logits, name="logits")
    if not single:
        raise ValueError("fused_monitor_path accepts exactly one [500,3] run")
    standardized = standardizer.transform(values[0])
    union = np.max(standardized, axis=1).astype(np.float64, copy=False)
    persistent = union.copy()
    persistent[1:] = np.minimum(union[:-1], union[1:])
    winners = np.argmax(standardized, axis=1).astype(np.int64)
    eligible = (np.arange(RUN_LENGTH) >= MONITOR_START + 1).astype(np.bool_)
    maximum = float(np.max(persistent[eligible]))
    return CFGLRMonitorPath(
        standardized,
        union,
        persistent,
        winners,
        eligible,
        maximum,
    )


@dataclass(frozen=True)
class Rank20Calibration:
    """Independent 20-run maximum/rank-20 strict-threshold calibration."""

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
            raise ValueError("rank-20 calibration values are inconsistent")
        copied = np.array(maxima, dtype=np.float64, copy=True, order="C")
        copied.setflags(write=False)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rank19", rank19)
        object.__setattr__(self, "run_maxima", copied)
        object.__setattr__(self, "distinct_maxima", distinct)

    @property
    def adjacent_tail_gap(self) -> float:
        return self.threshold - self.rank19

    @property
    def nondegenerate(self) -> bool:
        return self.distinct_maxima >= 19 and self.adjacent_tail_gap > 0.0

    def to_document(self) -> dict[str, object]:
        return {
            "schema": "mate_spc_cf_glr_rank20_calibration_v1",
            "calibration_run_count": CALIBRATION_RUNS,
            "rank": CALIBRATION_RANK,
            "comparison": "strictly_greater_than",
            "threshold": self.threshold,
            "rank19": self.rank19,
            "adjacent_tail_gap": self.adjacent_tail_gap,
            "distinct_maxima": self.distinct_maxima,
            "run_maxima": self.run_maxima.tolist(),
            "run_level_exchangeable_upper_bound": 1.0 / 21.0,
        }


def fit_rank20_threshold(
    calibration_healthy_logits: Any,
    standardizer: FittedExpertStandardizer,
) -> Rank20Calibration:
    """Fit rank 20 / maximum from exactly 20 independent healthy runs."""

    if not isinstance(standardizer, FittedExpertStandardizer):
        raise TypeError("standardizer must be FittedExpertStandardizer")
    values, _ = _validate_logits(
        calibration_healthy_logits,
        name="calibration_healthy_logits",
        exact_runs=CALIBRATION_RUNS,
    )
    maxima = np.asarray(
        [fused_monitor_path(run, standardizer).run_maximum for run in values],
        dtype=np.float64,
    )
    ordered = np.sort(maxima)
    return Rank20Calibration(
        float(ordered[CALIBRATION_RANK - 1]),
        maxima,
        float(ordered[CALIBRATION_RANK - 2]),
        int(np.unique(maxima).size),
    )


def first_strict_alarm(path: CFGLRMonitorPath, threshold: float) -> int | None:
    """Return the first eligible index with persistent score strictly above threshold."""

    if not isinstance(path, CFGLRMonitorPath):
        raise TypeError("path must be CFGLRMonitorPath")
    boundary = _finite_float(threshold, name="threshold")
    indices = np.flatnonzero(path.eligible & (path.persistent_score > boundary))
    return None if indices.size == 0 else int(indices[0])


@dataclass(frozen=True)
class AlarmDecision:
    alarm: bool
    first_alarm_index: int | None
    attributed_fault: int | None
    run_maximum: float
    threshold: float

    def __post_init__(self) -> None:
        if type(self.alarm) is not bool:
            raise ValueError("alarm must be a bool")
        maximum = _finite_float(self.run_maximum, name="run_maximum")
        threshold = _finite_float(self.threshold, name="threshold")
        if self.alarm:
            attributed_fault = _plain_int(
                self.attributed_fault, name="attributed_fault"
            )
            if (
                type(self.first_alarm_index) is not int
                or self.first_alarm_index < MONITOR_START + 1
                or self.first_alarm_index >= RUN_LENGTH
                or attributed_fault not in FAULT_IDS
                or maximum <= threshold
            ):
                raise ValueError("alarming decision fields are inconsistent")
            object.__setattr__(self, "attributed_fault", attributed_fault)
        elif self.first_alarm_index is not None or self.attributed_fault is not None:
            raise ValueError("non-alarming decision must have null index and attribution")
        object.__setattr__(self, "run_maximum", maximum)
        object.__setattr__(self, "threshold", threshold)


def apply_strict_threshold(
    path: CFGLRMonitorPath,
    threshold: float,
) -> AlarmDecision:
    """Apply the strict rank threshold and attribute to the current winning expert."""

    boundary = _finite_float(threshold, name="threshold")
    index = first_strict_alarm(path, boundary)
    fault = None if index is None else path.fault_ids[int(path.winning_expert_indices[index])]
    return AlarmDecision(
        index is not None,
        index,
        fault,
        path.run_maximum,
        boundary,
    )


__all__ = [
    "AlarmDecision",
    "CALIBRATION_RANK",
    "CALIBRATION_RUNS",
    "CFGLRMonitorPath",
    "FAULT_IDS",
    "FittedExpertStandardizer",
    "MONITOR_SCHEMA",
    "MONITOR_SCHEMA_VERSION",
    "MONITOR_START",
    "RUN_LENGTH",
    "Rank20Calibration",
    "apply_strict_threshold",
    "first_strict_alarm",
    "fit_rank20_threshold",
    "fused_monitor_path",
]
