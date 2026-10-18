"""Training-only robust multi-scale top-tail monitor for MSCD-TCN v3.

The fitted object owns both normalization layers required by the frozen scan:

1. endpoint-wise robust standardization of detector evidence;
2. branch-wise robust standardization of the trailing top-tail statistic.

All fit operations consume training healthy paths only.  Transform operations
are strictly causal and never update fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Mapping, Sequence

# Preserve the Windows CUDA/OpenMP load order used by the numeric backend.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # array-only metadata use remains possible
    _torch = None
else:  # pragma: no cover - host-runtime dependent
    try:
        import torch as _torch
    except (ImportError, OSError):
        _torch = None

import numpy as np


MAD_CONSISTENCY = 1.4826
DEFAULT_WINDOWS = (16, 32, 64, 128)
DEFAULT_TOP_FRACTION = 0.125
DEFAULT_PERSISTENCE = 2
DEFAULT_SCALE_FLOOR = 1.0e-6
DEFAULT_FLOOR_QUANTILE = 0.10
QUANTILE_METHOD = "linear"
STANDARDIZER_SCHEMA = "mstcn_robust_pointwise_standardizer_v2"
MONITOR_SCHEMA = "mstcn_fitted_top_tail_monitor_v1"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _fraction(value: Any, *, name: str) -> float:
    result = _finite_positive(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be at most one")
    return result


def _matrix(value: Any, *, name: str, minimum_rows: int = 2) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < minimum_rows or array.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape (at_least_{minimum_rows}_rows, positive_time)"
        )
    if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite exact float64")
    return array


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size < 1 or (length is not None and array.size != length):
        suffix = "" if length is None else f" of length {length}"
        raise ValueError(f"{name} must be a one-dimensional vector{suffix}")
    if array.dtype != np.dtype(np.float64) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite exact float64")
    return array


def _dense_endpoints(value: Any, *, length: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size < 1
        or raw.dtype.kind not in "iu"
        or (length is not None and raw.size != length)
    ):
        raise ValueError("endpoints must be an aligned one-dimensional integer vector")
    endpoints = np.asarray(raw, dtype=np.int64)
    if np.any(np.diff(endpoints) != 1):
        raise ValueError("endpoints must be dense consecutive integers")
    return endpoints


def _shared_mad_floor(raw_scales: np.ndarray, *, minimum_scale: float) -> float:
    requested = _finite_positive(minimum_scale, name="minimum_scale")
    quantile = float(
        np.quantile(raw_scales, DEFAULT_FLOOR_QUANTILE, method=QUANTILE_METHOD)
    )
    return max(requested, quantile)


def _json_float_vector(value: Any, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a nonempty JSON number array")
    return np.asarray(value, dtype=np.float64)


def _json_int_vector(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a nonempty JSON integer array")
    return np.asarray(value, dtype=np.int64)


@dataclass(frozen=True)
class RobustPointwiseStandardizer:
    """Per-endpoint median/MAD standardizer fit on healthy training paths."""

    center: np.ndarray
    scale: np.ndarray
    scale_floor: float
    floor_quantile: float = DEFAULT_FLOOR_QUANTILE
    quantile_method: str = QUANTILE_METHOD

    def __post_init__(self) -> None:
        center = _vector(self.center, name="center")
        scale = _vector(self.scale, name="scale", length=center.size)
        floor = _finite_positive(self.scale_floor, name="scale_floor")
        if np.any(scale < floor):
            raise ValueError("every pointwise scale must be at least scale_floor")
        if self.floor_quantile != DEFAULT_FLOOR_QUANTILE:
            raise ValueError("pointwise floor quantile changed")
        if self.quantile_method != QUANTILE_METHOD:
            raise ValueError("pointwise quantile method changed")
        center_copy = np.array(center, dtype=np.float64, copy=True, order="C")
        scale_copy = np.array(scale, dtype=np.float64, copy=True, order="C")
        center_copy.setflags(write=False)
        scale_copy.setflags(write=False)
        object.__setattr__(self, "center", center_copy)
        object.__setattr__(self, "scale", scale_copy)
        object.__setattr__(self, "scale_floor", floor)

    @classmethod
    def fit(
        cls,
        healthy_paths: Any,
        *,
        minimum_scale: float = DEFAULT_SCALE_FLOOR,
    ) -> "RobustPointwiseStandardizer":
        values = _matrix(healthy_paths, name="healthy_paths")
        center = np.median(values, axis=0)
        raw_scale = MAD_CONSISTENCY * np.median(
            np.abs(values - center[None, :]), axis=0
        )
        floor = _shared_mad_floor(raw_scale, minimum_scale=minimum_scale)
        scale = np.maximum(raw_scale, floor)
        return cls(
            np.asarray(center, dtype=np.float64),
            np.asarray(scale, dtype=np.float64),
            floor,
        )

    @property
    def time_length(self) -> int:
        return int(self.center.size)

    def transform(self, paths: Any) -> np.ndarray:
        values = np.asarray(paths)
        one_dimensional = values.ndim == 1
        if one_dimensional:
            values = values[None, :]
        checked = _matrix(values, name="paths", minimum_rows=1)
        if checked.shape[1] != self.time_length:
            raise ValueError("paths time dimension differs from fitted standardizer")
        result = np.array(
            (checked - self.center[None, :]) / self.scale[None, :],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("standardized evidence became nonfinite")
        return result[0] if one_dimensional else result

    def to_document(self) -> dict[str, object]:
        return {
            "schema": STANDARDIZER_SCHEMA,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "scale_floor": self.scale_floor,
            "floor_quantile": self.floor_quantile,
            "quantile_method": self.quantile_method,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RobustPointwiseStandardizer":
        if not isinstance(document, Mapping) or set(document) != {
            "schema",
            "center",
            "scale",
            "scale_floor",
            "floor_quantile",
            "quantile_method",
        }:
            raise ValueError("standardizer document fields changed")
        if document["schema"] != STANDARDIZER_SCHEMA:
            raise ValueError("standardizer schema changed")
        return cls(
            _json_float_vector(document["center"], name="center"),
            _json_float_vector(document["scale"], name="scale"),
            document["scale_floor"],
            document["floor_quantile"],
            document["quantile_method"],
        )


def _validate_windows(windows: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(_positive_int(value, name="scan window") for value in windows)
    if parsed != tuple(sorted(set(parsed))):
        raise ValueError("scan windows must be unique and increasing")
    return parsed


def _top_tail_branches(
    standardized_paths: np.ndarray,
    *,
    windows: tuple[int, ...],
    top_fraction: float,
) -> dict[int, np.ndarray]:
    """Return branch arrays shaped (runs, time), with invalid prefix NaN."""

    run_count, time_length = standardized_paths.shape
    branches: dict[int, np.ndarray] = {}
    for window in windows:
        result = np.full((run_count, time_length), np.nan, dtype=np.float64)
        if window <= time_length:
            trailing = np.lib.stride_tricks.sliding_window_view(
                standardized_paths, window_shape=window, axis=1
            )
            count = int(math.ceil(top_fraction * window))
            split = window - count
            top = np.partition(trailing, split, axis=2)[:, :, split:]
            result[:, window - 1 :] = np.mean(top, axis=2)
        branches[window] = result
    return branches


@dataclass(frozen=True)
class ScanPath:
    """Aligned raw grid statistic ``G`` and two-point persistent path ``P``."""

    endpoints: np.ndarray
    grid_values: np.ndarray
    persistent_values: np.ndarray
    winning_windows: np.ndarray
    winning_tail_counts: np.ndarray

    def __post_init__(self) -> None:
        endpoints = _dense_endpoints(self.endpoints)
        length = endpoints.size
        grid = np.asarray(self.grid_values)
        persistent = np.asarray(self.persistent_values)
        windows = np.asarray(self.winning_windows)
        counts = np.asarray(self.winning_tail_counts)
        if (
            grid.shape != (length,)
            or persistent.shape != (length,)
            or windows.shape != (length,)
            or counts.shape != (length,)
            or grid.dtype != np.dtype(np.float64)
            or persistent.dtype != np.dtype(np.float64)
            or windows.dtype.kind not in "iu"
            or counts.dtype.kind not in "iu"
        ):
            raise ValueError("ScanPath arrays are not aligned with required dtypes")
        for name, array in (("grid", grid), ("persistent", persistent)):
            if np.isnan(array).any() or np.isposinf(array).any():
                raise ValueError(f"{name} scan values cannot contain NaN/+inf")
        expected = np.full(length, -np.inf, dtype=np.float64)
        if length > 1:
            expected[1:] = np.minimum(grid[:-1], grid[1:])
        if not np.array_equal(persistent, expected):
            raise ValueError("persistent path must equal min(G[t-1], G[t])")
        for name, array in (
            ("endpoints", endpoints),
            ("grid_values", grid),
            ("persistent_values", persistent),
            ("winning_windows", windows),
            ("winning_tail_counts", counts),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    @property
    def values(self) -> np.ndarray:
        """Compatibility alias: operational thresholding always uses ``P``."""

        return self.persistent_values

    @property
    def run_maximum(self) -> float:
        valid = self.persistent_values[np.isfinite(self.persistent_values)]
        if valid.size == 0:
            raise ValueError("persistent scan path has no legal value")
        return float(np.max(valid))


@dataclass(frozen=True)
class FittedTopTailMonitor:
    """One candidate-specific (single rho) frozen monitoring transform."""

    endpoints: np.ndarray
    pointwise: RobustPointwiseStandardizer
    windows: tuple[int, ...]
    top_fraction: float
    branch_center: np.ndarray
    branch_scale: np.ndarray
    branch_scale_floor: float
    persistence: int = DEFAULT_PERSISTENCE

    def __post_init__(self) -> None:
        endpoints = _dense_endpoints(self.endpoints)
        if not isinstance(self.pointwise, RobustPointwiseStandardizer):
            raise TypeError("pointwise must be RobustPointwiseStandardizer")
        if endpoints.size != self.pointwise.time_length:
            raise ValueError("monitor endpoints and pointwise standardizer differ")
        windows = _validate_windows(self.windows)
        if windows[-1] > endpoints.size:
            raise ValueError("scan window exceeds path length")
        rho = _fraction(self.top_fraction, name="top_fraction")
        center = _vector(self.branch_center, name="branch_center", length=len(windows))
        scale = _vector(self.branch_scale, name="branch_scale", length=len(windows))
        floor = _finite_positive(self.branch_scale_floor, name="branch_scale_floor")
        if np.any(scale < floor):
            raise ValueError("every branch scale must be at least branch_scale_floor")
        persistence = _positive_int(self.persistence, name="persistence")
        if persistence != DEFAULT_PERSISTENCE:
            raise ValueError("frozen scan requires exactly two-point persistence")
        for name, array in (
            ("endpoints", endpoints),
            ("branch_center", center),
            ("branch_scale", scale),
        ):
            copied = np.array(array, copy=True, order="C")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "top_fraction", rho)
        object.__setattr__(self, "branch_scale_floor", floor)
        object.__setattr__(self, "persistence", persistence)

    @classmethod
    def fit(
        cls,
        healthy_evidence_paths: Any,
        endpoints: Sequence[int] | np.ndarray,
        *,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        top_fraction: float = DEFAULT_TOP_FRACTION,
        minimum_scale: float = DEFAULT_SCALE_FLOOR,
    ) -> "FittedTopTailMonitor":
        evidence = _matrix(healthy_evidence_paths, name="healthy_evidence_paths")
        endpoint_array = _dense_endpoints(endpoints, length=evidence.shape[1])
        parsed_windows = _validate_windows(windows)
        if parsed_windows[-1] > evidence.shape[1]:
            raise ValueError("scan window exceeds evidence path length")
        rho = _fraction(top_fraction, name="top_fraction")
        pointwise = RobustPointwiseStandardizer.fit(
            evidence, minimum_scale=minimum_scale
        )
        standardized = pointwise.transform(evidence)
        branches = _top_tail_branches(
            standardized, windows=parsed_windows, top_fraction=rho
        )
        centers: list[float] = []
        raw_scales: list[float] = []
        for window in parsed_windows:
            legal = branches[window][:, window - 1 :].reshape(-1)
            center = float(np.median(legal))
            scale = float(MAD_CONSISTENCY * np.median(np.abs(legal - center)))
            centers.append(center)
            raw_scales.append(scale)
        raw_scale_array = np.asarray(raw_scales, dtype=np.float64)
        branch_floor = _shared_mad_floor(
            raw_scale_array, minimum_scale=minimum_scale
        )
        scales = np.maximum(raw_scale_array, branch_floor)
        return cls(
            endpoint_array,
            pointwise,
            parsed_windows,
            rho,
            np.asarray(centers, dtype=np.float64),
            np.asarray(scales, dtype=np.float64),
            branch_floor,
        )

    def transform(self, evidence_path: Any) -> ScanPath:
        evidence = _vector(
            evidence_path, name="evidence_path", length=self.endpoints.size
        )
        standardized = self.pointwise.transform(evidence)[None, :]
        branches = _top_tail_branches(
            standardized, windows=self.windows, top_fraction=self.top_fraction
        )
        length = self.endpoints.size
        grid = np.full(length, -np.inf, dtype=np.float64)
        winning_windows = np.zeros(length, dtype=np.int64)
        winning_counts = np.zeros(length, dtype=np.int64)
        for index, window in enumerate(self.windows):
            raw = branches[window][0]
            legal = np.arange(length) >= window - 1
            standardized_branch = np.full(length, -np.inf, dtype=np.float64)
            standardized_branch[legal] = (
                raw[legal] - self.branch_center[index]
            ) / self.branch_scale[index]
            replace = standardized_branch > grid
            grid[replace] = standardized_branch[replace]
            winning_windows[replace] = window
            winning_counts[replace] = int(math.ceil(self.top_fraction * window))
        persistent = np.full(length, -np.inf, dtype=np.float64)
        persistent[1:] = np.minimum(grid[:-1], grid[1:])
        return ScanPath(
            self.endpoints,
            grid,
            persistent,
            winning_windows,
            winning_counts,
        )

    def transform_many(self, evidence_paths: Any) -> tuple[ScanPath, ...]:
        values = _matrix(evidence_paths, name="evidence_paths", minimum_rows=1)
        if values.shape[1] != self.endpoints.size:
            raise ValueError("evidence paths differ from fitted monitor length")
        return tuple(self.transform(values[index]) for index in range(values.shape[0]))

    def training_run_maxima(self, healthy_evidence_paths: Any) -> np.ndarray:
        paths = self.transform_many(healthy_evidence_paths)
        maxima = np.asarray([path.run_maximum for path in paths], dtype=np.float64)
        maxima.setflags(write=False)
        return maxima

    def to_document(self) -> dict[str, object]:
        return {
            "schema": MONITOR_SCHEMA,
            "endpoints": self.endpoints.tolist(),
            "pointwise": self.pointwise.to_document(),
            "windows": list(self.windows),
            "top_fraction": self.top_fraction,
            "branch_center": self.branch_center.tolist(),
            "branch_scale": self.branch_scale.tolist(),
            "branch_scale_floor": self.branch_scale_floor,
            "branch_floor_quantile": DEFAULT_FLOOR_QUANTILE,
            "quantile_method": QUANTILE_METHOD,
            "persistence": self.persistence,
            "threshold_comparison": "strictly_greater_than",
            "run_maximum_source": "max_two_point_persistent_path",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "FittedTopTailMonitor":
        expected = {
            "schema",
            "endpoints",
            "pointwise",
            "windows",
            "top_fraction",
            "branch_center",
            "branch_scale",
            "branch_scale_floor",
            "branch_floor_quantile",
            "quantile_method",
            "persistence",
            "threshold_comparison",
            "run_maximum_source",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise ValueError("fitted monitor document fields changed")
        if (
            document["schema"] != MONITOR_SCHEMA
            or document["branch_floor_quantile"] != DEFAULT_FLOOR_QUANTILE
            or document["quantile_method"] != QUANTILE_METHOD
            or document["threshold_comparison"] != "strictly_greater_than"
            or document["run_maximum_source"] != "max_two_point_persistent_path"
        ):
            raise ValueError("fitted monitor document contract changed")
        return cls(
            _json_int_vector(document["endpoints"], name="endpoints"),
            RobustPointwiseStandardizer.from_document(document["pointwise"]),
            tuple(document["windows"]),
            document["top_fraction"],
            _json_float_vector(document["branch_center"], name="branch_center"),
            _json_float_vector(document["branch_scale"], name="branch_scale"),
            document["branch_scale_floor"],
            document["persistence"],
        )


def multiscale_top_tail_scan(
    evidence_path: Any,
    monitor: FittedTopTailMonitor,
) -> ScanPath:
    """Apply a fitted single-rho monitor; this function never fits state."""

    if not isinstance(monitor, FittedTopTailMonitor):
        raise TypeError("monitor must be FittedTopTailMonitor")
    return monitor.transform(evidence_path)


def strict_order_statistic(values: Any, *, rank: int) -> float:
    """Return a one-based ascending order statistic, used later with strict ``>``."""

    array = _vector(values, name="values")
    chosen_rank = _positive_int(rank, name="rank")
    if chosen_rank > array.size:
        raise ValueError("rank exceeds the number of values")
    return float(np.sort(array)[chosen_rank - 1])


def first_persistent_exceedance(path: ScanPath, threshold: float) -> int | None:
    """Return the first endpoint whose frozen two-point path ``P`` exceeds h."""

    if not isinstance(path, ScanPath):
        raise TypeError("path must be ScanPath")
    if isinstance(threshold, (bool, np.bool_)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    indices = np.flatnonzero(path.persistent_values > float(threshold))
    return None if indices.size == 0 else int(path.endpoints[int(indices[0])])


def causal_fault_attribution(
    attribution_logits: Any,
    endpoints: Sequence[int] | np.ndarray,
    alarm_endpoint: int | None,
    *,
    fault_ids: Sequence[int] = (3, 9, 15),
) -> int | None:
    """Attribute an alarm using fault-head logits available through that endpoint."""

    if alarm_endpoint is None:
        return None
    logits = np.asarray(attribution_logits)
    endpoint_array = _dense_endpoints(endpoints)
    ids = tuple(_positive_int(value, name="fault id") for value in fault_ids)
    if (
        logits.ndim != 2
        or logits.shape != (endpoint_array.size, len(ids))
        or logits.dtype != np.dtype(np.float64)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("attribution logits/endpoints are malformed")
    eligible = endpoint_array <= int(alarm_endpoint)
    if not np.any(endpoint_array == int(alarm_endpoint)):
        raise ValueError("alarm endpoint is absent from the causal path")
    cumulative = logits[eligible].sum(axis=0)
    return ids[int(np.argmax(cumulative))]


__all__ = [
    "DEFAULT_PERSISTENCE",
    "DEFAULT_TOP_FRACTION",
    "DEFAULT_WINDOWS",
    "FittedTopTailMonitor",
    "RobustPointwiseStandardizer",
    "ScanPath",
    "causal_fault_attribution",
    "first_persistent_exceedance",
    "multiscale_top_tail_scan",
    "strict_order_statistic",
]
