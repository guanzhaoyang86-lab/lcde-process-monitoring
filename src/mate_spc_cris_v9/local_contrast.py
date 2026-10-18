"""Causal local-contrast dual-energy finite-window scan.

LC-DE uses a known-healthy run prefix as a local physical reference.  Four
raw-process contrasts (level energy, mean, variance, and difference energy)
are complemented by one predictive-innovation energy contrast.  Healthy-only
component calibration puts the five contrasts on a common robust scale; a
bounded CUSUM scan then accumulates only recent evidence, preventing the
months-long baseline drift observed with an unbounded global CUSUM.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np


WARMUP_ROWS = 300
REFERENCE_ROWS = 200
MAD_FACTOR = 1.4826
SCALE_FLOOR = 0.05
EPSILON = 1.0e-8
Z_CLIP = 20.0
COMPONENT_NAMES = (
    "raw_level_energy",
    "raw_mean_shift",
    "raw_log_variance_ratio",
    "raw_log_difference_energy_ratio",
    "predictive_log_energy_ratio",
)
Mode = Literal["dual", "raw_only", "predictive_only"]


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must be a real numeric matrix")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _innovations(value: Any, *, length: int, dimension: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError("innovations must be a real numeric cube")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if result.ndim != 3 or result.shape[0] != length or result.shape[2] != dimension:
        raise ValueError("innovations must have aligned shape (T,H,D)")
    if result.shape[1] < 1 or np.isinf(result).any():
        raise ValueError("innovations must have a horizon and may not contain infinity")
    return result


def _top_k_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    count = min(top_k, values.shape[1])
    selected = np.partition(values, values.shape[1] - count, axis=1)[:, -count:]
    return np.mean(selected, axis=1, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LocalContrastConfig:
    current_window: int = 20
    channel_top_k: int = 2
    allowance: float = 1.0
    scan_horizon: int = 60
    warmup_rows: int = WARMUP_ROWS
    reference_rows: int = REFERENCE_ROWS

    def __post_init__(self) -> None:
        window = _positive_integer(self.current_window, name="current_window")
        top_k = _positive_integer(self.channel_top_k, name="channel_top_k")
        horizon = _positive_integer(self.scan_horizon, name="scan_horizon")
        warmup = _positive_integer(self.warmup_rows, name="warmup_rows")
        reference = _positive_integer(self.reference_rows, name="reference_rows")
        allowance = _finite_real(self.allowance, name="allowance")
        if window < 2 or window >= warmup:
            raise ValueError("current_window must lie in [2,warmup_rows)")
        if window >= reference or reference >= warmup:
            raise ValueError(
                "current_window < reference_rows < warmup_rows is required"
            )
        if allowance < 0.0:
            raise ValueError("allowance must be nonnegative")
        object.__setattr__(self, "current_window", window)
        object.__setattr__(self, "channel_top_k", top_k)
        object.__setattr__(self, "scan_horizon", horizon)
        object.__setattr__(self, "warmup_rows", warmup)
        object.__setattr__(self, "reference_rows", reference)
        object.__setattr__(self, "allowance", allowance)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "current_window": self.current_window,
            "channel_top_k": self.channel_top_k,
            "allowance": self.allowance,
            "scan_horizon": self.scan_horizon,
            "warmup_rows": self.warmup_rows,
            "reference_rows": self.reference_rows,
        }


def local_contrast_components(
    x: Any,
    innovations: Any,
    config: LocalContrastConfig,
) -> np.ndarray:
    """Return five causal nonnegative contrast components in ``(T,5)`` form."""

    if not isinstance(config, LocalContrastConfig):
        raise TypeError("config must be LocalContrastConfig")
    values = _matrix(x, name="x")
    cube = _innovations(
        innovations, length=values.shape[0], dimension=values.shape[1]
    )
    if values.shape[0] <= config.warmup_rows:
        raise ValueError("run must extend beyond the healthy warmup prefix")
    if config.channel_top_k > values.shape[1]:
        raise ValueError("channel_top_k exceeds the feature dimension")

    reference = values[: config.reference_rows]
    reference_level = float(np.mean(np.square(reference), dtype=np.float64))
    reference_variance = np.var(reference, axis=0, ddof=1, dtype=np.float64)
    reference_difference = np.mean(
        np.square(np.diff(reference, axis=0)), axis=0, dtype=np.float64
    )
    finite_innovation = np.isfinite(cube)
    innovation_count = np.sum(finite_innovation, axis=(1, 2), dtype=np.int64)
    innovation_energy = np.full(values.shape[0], np.nan, dtype=np.float64)
    available = innovation_count > 0
    innovation_energy[available] = np.sum(
        np.square(np.where(finite_innovation, cube, 0.0)),
        axis=(1, 2),
        dtype=np.float64,
    )[available] / innovation_count[available]
    reference_prediction = innovation_energy[: config.reference_rows]
    reference_prediction = reference_prediction[np.isfinite(reference_prediction)]
    if reference_prediction.size < 2:
        raise ValueError("warmup prefix has too few predictive innovations")
    reference_prediction_energy = float(np.mean(reference_prediction))

    output = np.full((values.shape[0], len(COMPONENT_NAMES)), np.nan, dtype=np.float64)
    window = config.current_window
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window, axis=0
    )
    # NumPy aligns this as (time, channel, window).
    means = np.mean(windows, axis=2, dtype=np.float64)
    variances = np.var(windows, axis=2, ddof=1, dtype=np.float64)
    differences = np.mean(
        np.square(np.diff(windows, axis=2)), axis=2, dtype=np.float64
    )
    levels = np.mean(np.square(windows), axis=(1, 2), dtype=np.float64)

    start = config.reference_rows
    aligned = slice(start - window + 1, values.shape[0] - window + 1)
    selected_means = means[aligned]
    selected_variances = variances[aligned]
    selected_differences = differences[aligned]
    selected_levels = levels[aligned]
    if selected_levels.shape[0] != values.shape[0] - start:
        raise RuntimeError("local contrast alignment failed")

    output[start:, 0] = np.abs(
        np.log((selected_levels + EPSILON) / (reference_level + EPSILON))
    )
    output[start:, 1] = _top_k_mean(
        np.square(selected_means), config.channel_top_k
    )
    output[start:, 2] = _top_k_mean(
        np.square(
            np.log(
                (selected_variances + EPSILON)
                / (reference_variance[None, :] + EPSILON)
            )
        ),
        config.channel_top_k,
    )
    output[start:, 3] = _top_k_mean(
        np.square(
            np.log(
                (selected_differences + EPSILON)
                / (reference_difference[None, :] + EPSILON)
            )
        ),
        config.channel_top_k,
    )

    predictive_window = np.lib.stride_tricks.sliding_window_view(
        innovation_energy, window
    )
    selected_prediction = np.mean(
        predictive_window[aligned], axis=1, dtype=np.float64
    )
    output[start:, 4] = np.abs(
        np.log(
            (selected_prediction + EPSILON)
            / (reference_prediction_energy + EPSILON)
        )
    )
    if not np.isfinite(output[start:]).all():
        raise ValueError("local contrast components are not finite after warmup")
    return output


def _mode_indices(mode: Mode) -> np.ndarray:
    if mode == "dual":
        return np.arange(5, dtype=np.int64)
    if mode == "raw_only":
        return np.arange(4, dtype=np.int64)
    if mode == "predictive_only":
        return np.asarray([4], dtype=np.int64)
    raise ValueError("mode must be dual, raw_only, or predictive_only")


def _finite_window_cusum(increments: np.ndarray, horizon: int) -> np.ndarray:
    result = np.zeros(increments.size, dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(increments, dtype=np.float64)]
    for index in range(increments.size):
        earliest = max(0, index + 1 - horizon)
        minimum = float(np.min(cumulative[earliest : index + 1]))
        result[index] = max(0.0, float(cumulative[index + 1] - minimum))
    return result


@dataclass(frozen=True, slots=True)
class LocalContrastDualEnergyMonitor:
    config: LocalContrastConfig
    component_center: np.ndarray
    component_scale: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.config, LocalContrastConfig):
            raise TypeError("config must be LocalContrastConfig")
        center = np.asarray(self.component_center)
        scale = np.asarray(self.component_scale)
        if center.dtype != np.dtype(np.float64) or center.shape != (5,):
            raise ValueError("component_center must be float64 shape (5,)")
        if scale.dtype != np.dtype(np.float64) or scale.shape != (5,):
            raise ValueError("component_scale must be float64 shape (5,)")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("component calibration must be finite")
        if np.any(scale < SCALE_FLOOR):
            raise ValueError("component_scale violates the frozen floor")
        center_copy = np.array(center, dtype=np.float64, copy=True)
        scale_copy = np.array(scale, dtype=np.float64, copy=True)
        center_copy.setflags(write=False)
        scale_copy.setflags(write=False)
        object.__setattr__(self, "component_center", center_copy)
        object.__setattr__(self, "component_scale", scale_copy)

    @classmethod
    def fit(
        cls,
        healthy_x: Any,
        healthy_innovations: Any,
        config: LocalContrastConfig,
    ) -> "LocalContrastDualEnergyMonitor":
        components = local_contrast_components(
            healthy_x, healthy_innovations, config
        )[config.warmup_rows :]
        center = np.median(components, axis=0)
        mad = np.median(np.abs(components - center[None, :]), axis=0)
        scale = np.maximum(MAD_FACTOR * mad, SCALE_FLOOR)
        return cls(
            config=config,
            component_center=np.asarray(center, dtype=np.float64),
            component_scale=np.asarray(scale, dtype=np.float64),
        )

    def score(
        self,
        x: Any,
        innovations: Any,
        *,
        mode: Mode = "dual",
    ) -> np.ndarray:
        components = local_contrast_components(x, innovations, self.config)
        indices = _mode_indices(mode)
        commissioning = components[
            self.config.reference_rows : self.config.warmup_rows, indices
        ]
        local_center = np.median(commissioning, axis=0)
        local_mad = np.median(
            np.abs(commissioning - local_center[None, :]), axis=0
        )
        # A fraction of the independent healthy scale prevents a nearly
        # constant commissioning prefix from producing an unstable divisor.
        local_scale = np.maximum(
            MAD_FACTOR * local_mad,
            np.maximum(0.25 * self.component_scale[indices], SCALE_FLOOR),
        )
        standardized = np.maximum(
            0.0,
            (components[self.config.warmup_rows :, indices]
             - local_center[None, :])
            / local_scale[None, :],
        )
        standardized = np.clip(standardized, 0.0, Z_CLIP)
        count = min(2, standardized.shape[1])
        point = _top_k_mean(standardized, count)
        increments = point - self.config.allowance
        scan = _finite_window_cusum(increments, self.config.scan_horizon)
        output = np.zeros(components.shape[0], dtype=np.float64)
        output[self.config.warmup_rows :] = scan
        return output


__all__ = [
    "COMPONENT_NAMES",
    "LocalContrastConfig",
    "LocalContrastDualEnergyMonitor",
    "REFERENCE_ROWS",
    "WARMUP_ROWS",
    "local_contrast_components",
]
