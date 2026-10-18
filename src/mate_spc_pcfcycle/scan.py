"""Unbounded run-relative sequential scan for PCF-CycleRank-Scan v6.

The monitor deliberately operates on raw detector logits.  It contains no
empirical CDF, clipping, or other finite evidence ceiling.  All threshold and
scan arithmetic is float64 on the CPU after authoritative CUDA inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


PREFIX_ENDPOINTS = tuple(range(128, 160, 4))
DENSE_ENDPOINTS = tuple(range(160, 500))
SUSTAINED_WINDOWS = (4, 8, 16, 32, 64)
INTERMITTENT_WINDOW = 16
INTERMITTENT_TOP_K = 4
DEFAULT_DRIFT = 0.25
DEFAULT_SCALE_FLOOR = 1.0e-3
MAD_CONSISTENCY = 1.4826
CALIBRATOR_SCHEMA = "pcf_cyclerank_run_relative_logit_calibrator_v2"
ALLOWED_FIT_RUN_COUNTS = (60, 100)


def _finite_matrix(value: Any, *, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != columns or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (runs, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _frozen_drift(value: Any) -> float:
    try:
        drift = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("drift must be the frozen value 0.25") from exc
    if not math.isfinite(drift) or drift != DEFAULT_DRIFT:
        raise ValueError("drift must be the frozen value 0.25")
    return drift


@dataclass(frozen=True)
class RunRelativeLogitCalibrator:
    """Training-healthy robust scale for same-run baseline-relative logits."""

    center: float
    scale: float
    fit_run_count: int
    drift: float = DEFAULT_DRIFT
    prefix_endpoint_count: int = len(PREFIX_ENDPOINTS)
    monitor_endpoint_count: int = len(DENSE_ENDPOINTS)

    def __post_init__(self) -> None:
        values = (self.center, self.scale, self.drift)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("calibrator values must be finite")
        if float(self.scale) < DEFAULT_SCALE_FLOOR:
            raise ValueError("calibrator scale is below the frozen floor")
        _frozen_drift(self.drift)
        if isinstance(self.fit_run_count, (bool, np.bool_)) or not isinstance(
            self.fit_run_count, (int, np.integer)
        ):
            raise ValueError("fit_run_count must be an integer")
        if int(self.fit_run_count) not in ALLOWED_FIT_RUN_COUNTS:
            raise ValueError("fit_run_count must be exactly 60 or 100")
        object.__setattr__(self, "fit_run_count", int(self.fit_run_count))
        if self.prefix_endpoint_count != len(PREFIX_ENDPOINTS):
            raise ValueError("prefix endpoint count changed")
        if self.monitor_endpoint_count != len(DENSE_ENDPOINTS):
            raise ValueError("monitor endpoint count changed")

    @classmethod
    def fit(
        cls, healthy_logits: Any, *, expected_run_count: int = 60
    ) -> "RunRelativeLogitCalibrator":
        """Fit one scale using model-fit healthy paths only.

        Columns must be the eight frozen prefix logits followed by the 340
        dense monitoring logits.  Each run's prefix median is removed before
        the pooled healthy MAD is estimated.
        """

        if isinstance(expected_run_count, (bool, np.bool_)) or not isinstance(
            expected_run_count, (int, np.integer)
        ):
            raise ValueError("expected_run_count must be an integer")
        fit_run_count = int(expected_run_count)
        if fit_run_count not in ALLOWED_FIT_RUN_COUNTS:
            raise ValueError("expected_run_count must be exactly 60 or 100")
        columns = len(PREFIX_ENDPOINTS) + len(DENSE_ENDPOINTS)
        logits = _finite_matrix(healthy_logits, name="healthy_logits", columns=columns)
        if logits.shape[0] != fit_run_count:
            raise ValueError(
                f"healthy_logits must contain exactly {fit_run_count} fit-health runs"
            )
        prefix = np.median(logits[:, : len(PREFIX_ENDPOINTS)], axis=1)
        residual = logits[:, len(PREFIX_ENDPOINTS) :] - prefix[:, None]
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        scale = max(MAD_CONSISTENCY * mad, DEFAULT_SCALE_FLOOR)
        if not math.isfinite(scale):
            raise FloatingPointError("healthy logit scale is nonfinite")
        return cls(center=center, scale=scale, fit_run_count=fit_run_count)

    def transform(self, logits: Any) -> np.ndarray:
        columns = len(PREFIX_ENDPOINTS) + len(DENSE_ENDPOINTS)
        matrix = _finite_matrix(logits, name="logits", columns=columns)
        prefix = np.median(matrix[:, : len(PREFIX_ENDPOINTS)], axis=1)
        residual = matrix[:, len(PREFIX_ENDPOINTS) :] - prefix[:, None]
        result = (residual - self.center) / self.scale
        if not np.isfinite(result).all():
            raise FloatingPointError("run-relative logits are nonfinite")
        return np.asarray(result, dtype=np.float64)

    def to_document(self) -> dict[str, object]:
        return {
            "schema": CALIBRATOR_SCHEMA,
            "center": float(self.center),
            "scale": float(self.scale),
            "drift": float(self.drift),
            "fit_run_count": self.fit_run_count,
            "scale_floor": DEFAULT_SCALE_FLOOR,
            "mad_consistency": MAD_CONSISTENCY,
            "prefix_endpoints": list(PREFIX_ENDPOINTS),
            "dense_endpoints": [DENSE_ENDPOINTS[0], DENSE_ENDPOINTS[-1], 1],
            "fit_scope": "model_fit_healthy_paths_only",
        }

    @classmethod
    def from_document(cls, value: Any) -> "RunRelativeLogitCalibrator":
        if not isinstance(value, dict):
            raise ValueError("calibrator document must be a mapping")
        expected = {
            "schema",
            "center",
            "scale",
            "drift",
            "fit_run_count",
            "scale_floor",
            "mad_consistency",
            "prefix_endpoints",
            "dense_endpoints",
            "fit_scope",
        }
        if set(value) != expected:
            raise ValueError("calibrator document fields changed")
        if value["schema"] != CALIBRATOR_SCHEMA:
            raise ValueError("calibrator schema changed")
        if value["prefix_endpoints"] != list(PREFIX_ENDPOINTS):
            raise ValueError("calibrator prefix endpoints changed")
        if value["dense_endpoints"] != [160, 499, 1]:
            raise ValueError("calibrator dense endpoints changed")
        if value["fit_scope"] != "model_fit_healthy_paths_only":
            raise ValueError("calibrator fit scope changed")
        if float(value["scale_floor"]) != DEFAULT_SCALE_FLOOR:
            raise ValueError("calibrator scale floor changed")
        if float(value["mad_consistency"]) != MAD_CONSISTENCY:
            raise ValueError("calibrator MAD consistency changed")
        return cls(
            center=float(value["center"]),
            scale=float(value["scale"]),
            fit_run_count=value["fit_run_count"],
            drift=float(value["drift"]),
        )


@dataclass(frozen=True)
class ScanPath:
    """Aligned raw and two-sample-persistent scan paths for one run."""

    raw_endpoints: np.ndarray
    raw_scores: np.ndarray
    endpoints: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        raw_endpoints = np.asarray(self.raw_endpoints, dtype=np.int64)
        raw_scores = np.asarray(self.raw_scores, dtype=np.float64)
        endpoints = np.asarray(self.endpoints, dtype=np.int64)
        scores = np.asarray(self.scores, dtype=np.float64)
        if raw_endpoints.shape != (337,) or raw_scores.shape != (337,):
            raise ValueError("raw scan geometry changed")
        if endpoints.shape != (336,) or scores.shape != (336,):
            raise ValueError("persistent scan geometry changed")
        if not np.array_equal(raw_endpoints, np.arange(163, 500)):
            raise ValueError("raw scan endpoints changed")
        if not np.array_equal(endpoints, np.arange(164, 500)):
            raise ValueError("persistent scan endpoints changed")
        if not np.isfinite(raw_scores).all() or not np.isfinite(scores).all():
            raise ValueError("scan scores must be finite")
        for array in (raw_endpoints, raw_scores, endpoints, scores):
            array = np.array(array, copy=True)
            array.setflags(write=False)
        object.__setattr__(self, "raw_endpoints", np.array(raw_endpoints, copy=True))
        object.__setattr__(self, "raw_scores", np.array(raw_scores, copy=True))
        object.__setattr__(self, "endpoints", np.array(endpoints, copy=True))
        object.__setattr__(self, "scores", np.array(scores, copy=True))
        self.raw_endpoints.setflags(write=False)
        self.raw_scores.setflags(write=False)
        self.endpoints.setflags(write=False)
        self.scores.setflags(write=False)

    @property
    def run_maximum(self) -> float:
        return float(np.max(self.scores))

    def first_exceedance(self, threshold: float) -> int | None:
        limit = float(threshold)
        if not math.isfinite(limit):
            raise ValueError("threshold must be finite")
        positions = np.flatnonzero(self.scores > limit)
        return None if positions.size == 0 else int(self.endpoints[int(positions[0])])


def multiscale_unbounded_scan(
    run_relative_logits: Any, *, drift: float = DEFAULT_DRIFT
) -> ScanPath:
    """Compute sustained and intermittent raw-logit branches for one run."""

    evidence = np.asarray(run_relative_logits, dtype=np.float64)
    if evidence.shape != (len(DENSE_ENDPOINTS),) or not np.isfinite(evidence).all():
        raise ValueError("run_relative_logits must be finite shape (340,)")
    drift_value = _frozen_drift(drift)
    shifted = evidence - drift_value
    prefix_sum = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(shifted)))
    raw_values: list[float] = []
    raw_endpoints: list[int] = []
    for index in range(3, evidence.size):
        branches: list[float] = []
        length = index + 1
        for window in SUSTAINED_WINDOWS:
            if length >= window:
                total = prefix_sum[length] - prefix_sum[length - window]
                branches.append(float(total / math.sqrt(window)))
        if length >= INTERMITTENT_WINDOW:
            trailing = np.maximum(shifted[length - INTERMITTENT_WINDOW : length], 0.0)
            top = np.partition(trailing, -INTERMITTENT_TOP_K)[-INTERMITTENT_TOP_K:]
            branches.append(float(np.sum(top) / math.sqrt(INTERMITTENT_TOP_K)))
        raw_values.append(max(branches))
        raw_endpoints.append(DENSE_ENDPOINTS[index])
    raw = np.asarray(raw_values, dtype=np.float64)
    persistent = np.minimum(raw[:-1], raw[1:])
    if not np.isfinite(raw).all() or not np.isfinite(persistent).all():
        raise FloatingPointError("unbounded scan became nonfinite")
    return ScanPath(
        raw_endpoints=np.asarray(raw_endpoints, dtype=np.int64),
        raw_scores=raw,
        endpoints=np.asarray(raw_endpoints[1:], dtype=np.int64),
        scores=persistent,
    )


def scan_many(run_relative_logits: Any, *, drift: float = DEFAULT_DRIFT) -> tuple[ScanPath, ...]:
    drift_value = _frozen_drift(drift)
    matrix = _finite_matrix(
        run_relative_logits, name="run_relative_logits", columns=len(DENSE_ENDPOINTS)
    )
    return tuple(multiscale_unbounded_scan(row, drift=drift_value) for row in matrix)


@dataclass(frozen=True)
class ThresholdAudit:
    threshold: float
    rank19: float
    top_gap: float
    distinct_count: int
    tolerance: float
    valid: bool

    def to_document(self) -> dict[str, object]:
        return {
            "schema": "pcf_cyclerank_inner_threshold_audit_v1",
            "calibration_run_count": 20,
            "ascending_rank": 20,
            "comparison": "strictly_greater_than",
            "threshold": self.threshold,
            "rank19": self.rank19,
            "rank20_minus_rank19": self.top_gap,
            "distinct_count": self.distinct_count,
            "minimum_distinct_count": 18,
            "numeric_tolerance": self.tolerance,
            "valid": self.valid,
        }


def validate_inner_threshold(
    run_maxima: Sequence[float] | np.ndarray, *, repeat_error: float
) -> ThresholdAudit:
    maxima = np.asarray(run_maxima, dtype=np.float64)
    if maxima.shape != (20,) or not np.isfinite(maxima).all():
        raise ValueError("inner run maxima must be 20 finite values")
    error = float(repeat_error)
    if not math.isfinite(error) or error < 0.0:
        raise ValueError("repeat_error must be finite and nonnegative")
    ordered = np.sort(maxima)
    threshold = float(ordered[-1])
    rank19 = float(ordered[-2])
    gap = threshold - rank19
    tolerance = 100.0 * max(error, 1.0e-7)
    distinct = int(np.unique(maxima).size)
    return ThresholdAudit(
        threshold=threshold,
        rank19=rank19,
        top_gap=gap,
        distinct_count=distinct,
        tolerance=tolerance,
        valid=bool(distinct >= 18 and gap > tolerance),
    )


__all__ = [
    "ALLOWED_FIT_RUN_COUNTS",
    "CALIBRATOR_SCHEMA",
    "DEFAULT_DRIFT",
    "DENSE_ENDPOINTS",
    "INTERMITTENT_TOP_K",
    "INTERMITTENT_WINDOW",
    "PREFIX_ENDPOINTS",
    "RunRelativeLogitCalibrator",
    "SUSTAINED_WINDOWS",
    "ScanPath",
    "ThresholdAudit",
    "multiscale_unbounded_scan",
    "scan_many",
    "validate_inner_threshold",
]
