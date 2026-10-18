"""Strictly causal mechanism features for the frozen CF-MS-GLR v4 design.

The Extended Tennessee Eastman 52-vector does not contain a measurement of
D-feed temperature. IDV(3) and IDV(9) perturb that latent temperature, while
XMEAS(2) flow is tightly controlled. The fault-3 and fault-9 experts therefore
retain broad downstream evidence by reducing channelwise statistics over all
52 baseline-normalized process variables.

Every nonzero feature at endpoint ``t`` uses only the commissioning baseline
``[0, 128)`` and samples with index at most ``t``.  Rows before endpoint 128
are canonical zeros.  In particular, the prefix-negative endpoints
``range(128, 160, 4)`` are populated without looking beyond their endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


RUN_LENGTH = 500
PROCESS_FEATURE_COUNT = 52
BASELINE_STOP = 128
MONITOR_START = 160
PREFIX_ENDPOINTS = tuple(range(BASELINE_STOP, MONITOR_START, 4))
FAULT_IDS = (3, 9, 15)

# The process vector is [XMEAS(1:41), XMV(1:11)].
D_FEED_XMEAS_INDEX = 1  # XMEAS(2), observable D-feed flow response
D_FEED_XMV_INDEX = 41  # XMV(1), retained as provenance metadata
F15_XMV11_INDEX = 51
F15_SEPARATOR_TEMP_INDEX = 10  # XMEAS(11)
F15_COOLING_OUTLET_INDEX = 21  # XMEAS(22)

BROAD_PROCESS_WINDOWS = (8, 16, 32, 64)
BROAD_PROCESS_CHANNELS = tuple(range(PROCESS_FEATURE_COUNT))
BROAD_TOP_K = 5
F15_WINDOWS = (16, 32, 64, 128)
F15_LAGS = (1, 2, 4, 8)
MAD_CONSISTENCY = 1.4826
NUMERIC_SCALE_FLOOR = 1.0e-6
STANDARDIZED_CLIP = 1.0e12
DWELL_DELTA_EPSILON = 0.05
ARX_RIDGE = 1.0e-6
FEATURE_SCHEMA = "cf_glr_mechanism_features_v2"

BROAD_STATISTIC_ORDER = (
    "top5mean_abs_robust_location",
    "top5mean_robust_variance",
    "top5mean_delta_energy",
    "top5mean_abs_lag1_autocovariance",
    "top5mean_haar_energy",
)
F15_STATISTIC_ORDER = (
    "dwell_ratio",
    "slip_impulse_after_dwell4",
    "normalized_loop_area_u_y1",
    "normalized_loop_area_u_y2",
    "maxlag_corr_asymmetry_u_y1",
    "maxlag_corr_asymmetry_u_y2",
    "robust_ARX_residual_energy_proxy",
)


def _broad_process_feature_names(fault_id: int) -> tuple[str, ...]:
    return tuple(
        f"fault{fault_id}.{statistic}[window={width}]"
        for width in BROAD_PROCESS_WINDOWS
        for statistic in BROAD_STATISTIC_ORDER
    )


F3_FEATURE_NAMES = _broad_process_feature_names(3)
F9_FEATURE_NAMES = _broad_process_feature_names(9)
F15_FEATURE_NAMES = tuple(
    f"fault15.{statistic}[window={width}]"
    for width in F15_WINDOWS
    for statistic in F15_STATISTIC_ORDER
)
F3_FEATURE_COUNT = len(F3_FEATURE_NAMES)
F9_FEATURE_COUNT = len(F9_FEATURE_NAMES)
F15_FEATURE_COUNT = len(F15_FEATURE_NAMES)
EXPERT_FEATURE_COUNTS = MappingProxyType(
    {3: F3_FEATURE_COUNT, 9: F9_FEATURE_COUNT, 15: F15_FEATURE_COUNT}
)

if (F3_FEATURE_COUNT, F9_FEATURE_COUNT, F15_FEATURE_COUNT) != (20, 20, 28):
    raise RuntimeError("CF-MS-GLR feature geometry changed")

FEATURE_METADATA = MappingProxyType(
    {
        "feature_schema": FEATURE_SCHEMA,
        "run_shape": (RUN_LENGTH, PROCESS_FEATURE_COUNT),
        "commissioning_baseline_half_open": (0, BASELINE_STOP),
        "prefix_endpoints": PREFIX_ENDPOINTS,
        "monitor_start": MONITOR_START,
        "causality": "fixed_0_128_prefix_then_samples_at_or_before_endpoint",
        "prebaseline_rows": "canonical_zero",
        "normalization": "per_run_baseline_median_MAD_floor_then_asinh",
        "disturbance_observability": (
            "D_feed_temperature_IDV_is_unmeasured;XMEAS2_flow_is_tightly_controlled;"
            "F3_F9_use_broad_downstream_response"
        ),
        "broad_process_channel_indices": BROAD_PROCESS_CHANNELS,
        "broad_process_windows": BROAD_PROCESS_WINDOWS,
        "broad_process_top_k": BROAD_TOP_K,
        "broad_process_reduction": "mean_of_five_largest_channel_magnitudes",
        "broad_process_statistic_order": BROAD_STATISTIC_ORDER,
        "f15_indices_u_y1_y2": (
            F15_XMV11_INDEX,
            F15_SEPARATOR_TEMP_INDEX,
            F15_COOLING_OUTLET_INDEX,
        ),
        "f15_windows": F15_WINDOWS,
        "f15_lags": F15_LAGS,
        "f15_arx_proxy": (
            "baseline_ridge_ARX1_y[t]~1+y[t-1]+u[t-1];"
            "median_of_mean_two_squared_robust_residuals_in_trailing_window"
        ),
        "expert_feature_counts": (F3_FEATURE_COUNT, F9_FEATURE_COUNT, F15_FEATURE_COUNT),
    }
)


def _validated_run(value: Any, *, name: str = "run") -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT):
        raise ValueError(f"{name} must have shape ({RUN_LENGTH}, {PROCESS_FEATURE_COUNT})")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _stable_median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=None)
    count = int(ordered.size)
    if count == 0:
        raise ValueError("cannot take a median of an empty vector")
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    # Dividing before adding avoids overflow for two large values of one sign.
    return float(ordered[middle - 1] * 0.5 + ordered[middle] * 0.5)


def _safe_difference(left: np.ndarray, right: float | np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(left, dtype=np.float64) - right
    return np.nan_to_num(
        result,
        copy=False,
        nan=0.0,
        posinf=STANDARDIZED_CLIP,
        neginf=-STANDARDIZED_CLIP,
    )


def _baseline_standardize(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values)
    if vector.shape != (RUN_LENGTH,) or vector.dtype != np.dtype(np.float64):
        raise ValueError("series must be exact float64 with shape (500,)")
    baseline = vector[:BASELINE_STOP]
    center = _stable_median(baseline)
    baseline_deviation = np.abs(_safe_difference(baseline, center))
    scale = max(MAD_CONSISTENCY * _stable_median(baseline_deviation), NUMERIC_SCALE_FLOOR)
    difference = _safe_difference(vector, center)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = difference / np.float64(scale)
    standardized = np.nan_to_num(
        standardized,
        copy=False,
        nan=0.0,
        posinf=STANDARDIZED_CLIP,
        neginf=-STANDARDIZED_CLIP,
    )
    standardized = np.clip(standardized, -STANDARDIZED_CLIP, STANDARDIZED_CLIP)
    result = np.arcsinh(standardized).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("baseline standardization is nonfinite")
    return result


def _mean_top_channel_magnitudes(values: np.ndarray) -> float:
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    if magnitudes.shape != (PROCESS_FEATURE_COUNT,) or not np.isfinite(magnitudes).all():
        raise ValueError("channel statistic must be a finite float64 52-vector")
    split = PROCESS_FEATURE_COUNT - BROAD_TOP_K
    largest = np.partition(magnitudes, split)[split:]
    return float(np.mean(largest))


def _broad_process_features(series: np.ndarray) -> np.ndarray:
    if series.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT):
        raise ValueError("normalized process path must have shape (500, 52)")
    features = np.zeros((RUN_LENGTH, F3_FEATURE_COUNT), dtype=np.float64)
    for endpoint in range(BASELINE_STOP, RUN_LENGTH):
        column = 0
        for width in BROAD_PROCESS_WINDOWS:
            start = endpoint - width + 1
            segment = series[start : endpoint + 1]
            location = np.median(segment, axis=0)
            deviations = segment - location[None, :]
            robust_scale = MAD_CONSISTENCY * np.median(np.abs(deviations), axis=0)
            robust_variance = np.square(robust_scale)
            increments = np.diff(segment, axis=0)
            delta_energy = np.mean(np.square(increments), axis=0)
            lag1_autocovariance = np.mean(
                deviations[:-1] * deviations[1:], axis=0
            )
            half = width // 2
            haar = np.mean(segment[half:], axis=0) - np.mean(segment[:half], axis=0)
            features[endpoint, column : column + 5] = (
                _mean_top_channel_magnitudes(location),
                _mean_top_channel_magnitudes(robust_variance),
                _mean_top_channel_magnitudes(delta_energy),
                _mean_top_channel_magnitudes(lag1_autocovariance),
                _mean_top_channel_magnitudes(np.square(haar)),
            )
            column += 5
    if not np.isfinite(features).all():
        raise FloatingPointError("broad-process expert features are nonfinite")
    return features


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("correlation vectors must have equal one-dimensional support")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator <= NUMERIC_SCALE_FLOOR:
        return 0.0
    value = float(np.dot(left_centered, right_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _normalized_loop_area(actuator: np.ndarray, response: np.ndarray) -> float:
    actuator_centered = actuator - _stable_median(actuator)
    response_centered = response - _stable_median(response)
    oriented_steps = (
        actuator_centered[:-1] * response_centered[1:]
        - actuator_centered[1:] * response_centered[:-1]
    )
    actuator_rms = float(np.sqrt(np.mean(np.square(actuator_centered))))
    response_rms = float(np.sqrt(np.mean(np.square(response_centered))))
    denominator = max(
        float(actuator.size - 1) * actuator_rms * response_rms,
        NUMERIC_SCALE_FLOOR,
    )
    return float(abs(np.sum(oriented_steps)) / denominator)


def _maxlag_correlation_asymmetry(actuator: np.ndarray, response: np.ndarray) -> float:
    asymmetries = []
    for lag in F15_LAGS:
        forward = _safe_correlation(actuator[:-lag], response[lag:])
        reverse = _safe_correlation(response[:-lag], actuator[lag:])
        asymmetries.append(abs(forward - reverse))
    return float(max(asymmetries))


def _fit_baseline_arx_residual(actuator: np.ndarray, response: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        (
            np.ones(BASELINE_STOP - 1, dtype=np.float64),
            response[: BASELINE_STOP - 1],
            actuator[: BASELINE_STOP - 1],
        )
    )
    target = response[1:BASELINE_STOP]
    penalty = np.diag(np.array((0.0, ARX_RIDGE, ARX_RIDGE), dtype=np.float64))
    gram = design.T @ design + penalty
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram, rcond=1.0e-12) @ rhs

    residual = np.zeros(RUN_LENGTH, dtype=np.float64)
    residual[1:] = response[1:] - (
        coefficients[0]
        + coefficients[1] * response[:-1]
        + coefficients[2] * actuator[:-1]
    )
    baseline_residual = residual[1:BASELINE_STOP]
    residual_center = _stable_median(baseline_residual)
    residual_deviation = _safe_difference(residual, residual_center)
    residual_scale = max(
        MAD_CONSISTENCY
        * _stable_median(np.abs(_safe_difference(baseline_residual, residual_center))),
        NUMERIC_SCALE_FLOOR,
    )
    standardized = np.clip(
        residual_deviation / np.float64(residual_scale),
        -STANDARDIZED_CLIP,
        STANDARDIZED_CLIP,
    )
    result = np.arcsinh(standardized).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("ARX residual standardization is nonfinite")
    return result


def _fault15_features(
    actuator: np.ndarray,
    separator_temperature: np.ndarray,
    cooling_outlet: np.ndarray,
) -> np.ndarray:
    features = np.zeros((RUN_LENGTH, F15_FEATURE_COUNT), dtype=np.float64)
    actuator_delta = np.zeros(RUN_LENGTH, dtype=np.float64)
    actuator_delta[1:] = actuator[1:] - actuator[:-1]
    dwell = np.abs(actuator_delta) <= DWELL_DELTA_EPSILON
    release_impulse = np.zeros(RUN_LENGTH, dtype=np.float64)
    for index in range(4, RUN_LENGTH):
        if bool(np.all(dwell[index - 4 : index])) and not bool(dwell[index]):
            release_impulse[index] = abs(actuator_delta[index])

    residual_y1 = _fit_baseline_arx_residual(actuator, separator_temperature)
    residual_y2 = _fit_baseline_arx_residual(actuator, cooling_outlet)
    joint_residual_energy = 0.5 * (
        np.square(residual_y1) + np.square(residual_y2)
    )

    for endpoint in range(BASELINE_STOP, RUN_LENGTH):
        column = 0
        for width in F15_WINDOWS:
            start = endpoint - width + 1
            stop = endpoint + 1
            u = actuator[start:stop]
            y1 = separator_temperature[start:stop]
            y2 = cooling_outlet[start:stop]
            internal_delta = dwell[start + 1 : stop]
            dwell_ratio = float(np.mean(internal_delta))
            slip_impulse = float(np.max(release_impulse[start + 1 : stop]))
            features[endpoint, column : column + 7] = (
                dwell_ratio,
                slip_impulse,
                _normalized_loop_area(u, y1),
                _normalized_loop_area(u, y2),
                _maxlag_correlation_asymmetry(u, y1),
                _maxlag_correlation_asymmetry(u, y2),
                _stable_median(joint_residual_energy[start:stop]),
            )
            column += 7
    if not np.isfinite(features).all():
        raise FloatingPointError("fault-15 expert features are nonfinite")
    return features


def _readonly_feature_matrix(value: np.ndarray, *, columns: int, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    if source.shape != (RUN_LENGTH, columns) or not np.isfinite(source).all():
        raise ValueError(f"{name} must be finite with shape ({RUN_LENGTH}, {columns})")
    result = np.array(source, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ExpertFeaturePaths:
    """Read-only per-run causal feature paths for the three specialists."""

    fault3: np.ndarray
    fault9: np.ndarray
    fault15: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fault3",
            _readonly_feature_matrix(self.fault3, columns=F3_FEATURE_COUNT, name="fault3"),
        )
        object.__setattr__(
            self,
            "fault9",
            _readonly_feature_matrix(self.fault9, columns=F9_FEATURE_COUNT, name="fault9"),
        )
        object.__setattr__(
            self,
            "fault15",
            _readonly_feature_matrix(self.fault15, columns=F15_FEATURE_COUNT, name="fault15"),
        )

    def for_fault(self, fault_id: int) -> np.ndarray:
        if type(fault_id) is not int:
            raise ValueError("fault_id must be one of 3, 9, or 15")
        if fault_id == 3:
            return self.fault3
        if fault_id == 9:
            return self.fault9
        if fault_id == 15:
            return self.fault15
        raise ValueError("fault_id must be one of 3, 9, or 15")

    @property
    def by_fault(self) -> Mapping[int, np.ndarray]:
        return MappingProxyType({3: self.fault3, 9: self.fault9, 15: self.fault15})


def expert_feature_names(fault_id: int) -> tuple[str, ...]:
    if type(fault_id) is not int:
        raise ValueError("fault_id must be one of 3, 9, or 15")
    if fault_id == 3:
        return F3_FEATURE_NAMES
    if fault_id == 9:
        return F9_FEATURE_NAMES
    if fault_id == 15:
        return F15_FEATURE_NAMES
    raise ValueError("fault_id must be one of 3, 9, or 15")


def extract_expert_feature_paths(run: Any) -> ExpertFeaturePaths:
    """Transform one exact-float64 ``(500, 52)`` run into three expert paths."""

    values = _validated_run(run)
    normalized_process = np.column_stack(
        [
            _baseline_standardize(values[:, channel])
            for channel in BROAD_PROCESS_CHANNELS
        ]
    )
    broad_features = _broad_process_features(normalized_process)

    actuator = normalized_process[:, F15_XMV11_INDEX]
    separator_temperature = normalized_process[:, F15_SEPARATOR_TEMP_INDEX]
    cooling_outlet = normalized_process[:, F15_COOLING_OUTLET_INDEX]
    fault15 = _fault15_features(actuator, separator_temperature, cooling_outlet)

    return ExpertFeaturePaths(
        fault3=broad_features,
        fault9=broad_features,
        fault15=fault15,
    )


__all__ = [
    "ARX_RIDGE",
    "BASELINE_STOP",
    "BROAD_PROCESS_CHANNELS",
    "BROAD_PROCESS_WINDOWS",
    "BROAD_STATISTIC_ORDER",
    "BROAD_TOP_K",
    "D_FEED_XMEAS_INDEX",
    "D_FEED_XMV_INDEX",
    "DWELL_DELTA_EPSILON",
    "EXPERT_FEATURE_COUNTS",
    "ExpertFeaturePaths",
    "F15_COOLING_OUTLET_INDEX",
    "F15_FEATURE_COUNT",
    "F15_FEATURE_NAMES",
    "F15_LAGS",
    "F15_SEPARATOR_TEMP_INDEX",
    "F15_STATISTIC_ORDER",
    "F15_WINDOWS",
    "F15_XMV11_INDEX",
    "F3_FEATURE_COUNT",
    "F3_FEATURE_NAMES",
    "F9_FEATURE_COUNT",
    "F9_FEATURE_NAMES",
    "FAULT_IDS",
    "FEATURE_METADATA",
    "FEATURE_SCHEMA",
    "MONITOR_START",
    "PREFIX_ENDPOINTS",
    "PROCESS_FEATURE_COUNT",
    "RUN_LENGTH",
    "expert_feature_names",
    "extract_expert_feature_paths",
]
