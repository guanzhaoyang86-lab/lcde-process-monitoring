"""Causal PCF-CycleRank-Scan feature construction.

This module is deliberately data-loader free.  It accepts only in-memory,
finite, exact-float64 500-by-52 runs and cannot open an HDF5 archive.  Every
fitted quantity is estimated from the explicitly selected 60 development-fit
or 100 final-refit healthy commissioning prefixes ``[0, 128)``.  An endpoint
feature uses the same-run fixed reference
``[0, 128)`` and the trailing current window ``[t-127, t]`` only.

The 604-dimensional vector ``B`` has a frozen ordering:

* 52 channel-major blocks of 11 current-minus-reference summaries (572);
* two 12-value XMV11 loop blocks (24), paired with XMEAS11 and XMEAS22;
* eight current-window healthy-subspace statistics (8).

The loop block order is recorded by :data:`LOOP_FEATURE_NAMES`.  For each
pair it is: current dwell, dwell delta, current stick-slip, stick-slip delta,
current reversal, reversal delta, current oriented area, oriented-area delta,
lag-asymmetry delta, log reference-fitted ARX residual ratio, Welch-coherence
delta, and actuator-PSD-concentration delta.  "Delta" means current minus the
fixed reference.  This exact definition removes any implicit feature search.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CANDIDATE_NAME = "pcf-cyclerank-scan-v6-fixed"
RUN_LENGTH = 500
PROCESS_FEATURE_COUNT = 52
REFERENCE_STOP = 128
WINDOW_LENGTH = 128
FIT_HEALTHY_RUN_COUNT = 60
FINAL_REFIT_HEALTHY_RUN_COUNT = 100
ALLOWED_FIT_RUN_COUNTS = (FIT_HEALTHY_RUN_COUNT, FINAL_REFIT_HEALTHY_RUN_COUNT)
MONITOR_START = 160
MONITOR_STOP = 497
MONITOR_STRIDE = 4
MAD_CONSISTENCY = 1.4826
NORMALIZATION_FLOOR = 1.0e-6
NORMALIZATION_CLIP = 10.0
SUMMARY_EPSILON = 1.0e-8
SPECTRAL_EPSILON = 1.0e-12
PCA_RANK = 16
DPCA_LAG = 4
DPCA_RANK = 16
SFA_TAIL_COUNT = 4
VAR_ORDER = 8
SUBSPACE_RIDGE = 1.0e-6

MONITOR_ENDPOINTS = np.arange(
    MONITOR_START, MONITOR_STOP, MONITOR_STRIDE, dtype=np.int64
)
MONITOR_ENDPOINTS.setflags(write=False)

# Positive rFFT bins for a length-128 window are 1..64 inclusive.  These six
# half-open intervals are contiguous, disjoint, and cover all positive bins.
RFFT_BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 65))

CHANNEL_SUMMARY_NAMES = (
    "delta_mean",
    "delta_log_variance",
    "delta_log_diff_energy",
    *(f"delta_log_rfft_share[{lo}:{hi}]" for lo, hi in RFFT_BANDS),
    "delta_normalized_spectral_entropy",
    "delta_normalized_spectral_centroid",
)
if len(CHANNEL_SUMMARY_NAMES) != 11:  # pragma: no cover - import invariant
    raise RuntimeError("channel summary geometry changed")

# Zero-based Extended TEP channels requested by the frozen candidate.
LOOP_PAIRS = (
    (51, 10, "xmv11_xmeas11"),
    (51, 21, "xmv11_xmeas22"),
)
_LOOP_ORDER = (
    "current_dwell",
    "delta_dwell",
    "current_stick_slip",
    "delta_stick_slip",
    "current_reversal",
    "delta_reversal",
    "current_oriented_area",
    "delta_oriented_area",
    "delta_lag_asymmetry",
    "log_arx_residual_ratio",
    "delta_welch_coherence",
    "delta_actuator_psd_concentration",
)
LOOP_FEATURE_NAMES = tuple(
    f"{pair_name}.{metric}"
    for _, _, pair_name in LOOP_PAIRS
    for metric in _LOOP_ORDER
)
if len(LOOP_FEATURE_NAMES) != 24:  # pragma: no cover - import invariant
    raise RuntimeError("loop summary geometry changed")

HEALTH_FEATURE_NAMES = (
    "pca_t2",
    "pca_q",
    "lag4_dpca_t2",
    "lag4_dpca_q",
    "sfa_slow",
    "sfa_fast",
    "var8_state_mahalanobis",
    "var8_residual_mahalanobis",
)

A_SHAPE = (PROCESS_FEATURE_COUNT, WINDOW_LENGTH)
CHANNEL_FEATURE_COUNT = PROCESS_FEATURE_COUNT * len(CHANNEL_SUMMARY_NAMES)
LOOP_FEATURE_COUNT = len(LOOP_FEATURE_NAMES)
HEALTH_FEATURE_COUNT = len(HEALTH_FEATURE_NAMES)
B_FEATURE_COUNT = CHANNEL_FEATURE_COUNT + LOOP_FEATURE_COUNT + HEALTH_FEATURE_COUNT
if B_FEATURE_COUNT != 604:  # pragma: no cover - import invariant
    raise RuntimeError("PCF-CycleRank B geometry changed")
B_FIT_ROW_COUNT = FIT_HEALTHY_RUN_COUNT * MONITOR_ENDPOINTS.size
FINAL_B_FIT_ROW_COUNT = FINAL_REFIT_HEALTHY_RUN_COUNT * MONITOR_ENDPOINTS.size
FEATURE_STATE_SCHEMA = "mate_spc_pcfcycle_feature_state"
FEATURE_STATE_SCHEMA_VERSION = 2


def _validated_run(value: Any, *, name: str = "run") -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT):
        raise ValueError(
            f"{name} must have shape ({RUN_LENGTH}, {PROCESS_FEATURE_COUNT})"
        )
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_fit_prefix(value: Any, *, name: str) -> np.ndarray:
    """Validate only identity and permitted prefix values; never inspect suffix."""

    array = np.asarray(value)
    if array.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT):
        raise ValueError(
            f"{name} must have shape ({RUN_LENGTH}, {PROCESS_FEATURE_COUNT})"
        )
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    if not np.isfinite(array[:REFERENCE_STOP]).all():
        raise ValueError(f"{name} prefix must contain only finite values")
    return array


def _readonly(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float64) or source.shape != shape:
        raise ValueError(f"{name} must be exact float64 with shape {shape}")
    if not np.isfinite(source).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(source, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _ordered_runs(
    runs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    *,
    expected_run_count: int,
) -> tuple[np.ndarray, ...]:
    expected_count = _validated_fit_run_count(expected_run_count)
    if isinstance(runs, Mapping):
        if not runs:
            raise ValueError("fit_healthy_runs must be nonempty")
        values = tuple(
            _validated_fit_prefix(runs[key], name=f"fit_healthy_runs[{key!r}]")
            for key in sorted(runs)
        )
    elif isinstance(runs, Sequence) and not isinstance(runs, (str, bytes)):
        values = tuple(
            _validated_fit_prefix(value, name=f"fit_healthy_runs[{index}]")
            for index, value in enumerate(runs)
        )
    else:
        raise ValueError("fit_healthy_runs must be a mapping or sequence")
    if len(values) != expected_count:
        raise ValueError(
            f"fit_healthy_runs must contain exactly {expected_count} runs"
        )
    return values


def _validated_fit_run_count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"expected_run_count must be one of {ALLOWED_FIT_RUN_COUNTS}")
    result = int(value)
    if result not in ALLOWED_FIT_RUN_COUNTS:
        raise ValueError(f"expected_run_count must be one of {ALLOWED_FIT_RUN_COUNTS}")
    return result


@dataclass(frozen=True)
class HealthyPrefixNormalizer:
    """Pooled fit-prefix median/MAD transform with a fixed z clip."""

    center: np.ndarray
    scale: np.ndarray
    fit_run_count: int = FIT_HEALTHY_RUN_COUNT

    def __post_init__(self) -> None:
        center = _readonly(
            self.center, shape=(PROCESS_FEATURE_COUNT,), name="center"
        )
        scale = _readonly(self.scale, shape=(PROCESS_FEATURE_COUNT,), name="scale")
        if np.any(scale < NORMALIZATION_FLOOR):
            raise ValueError("scale must respect the normalization floor")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self, "fit_run_count", _validated_fit_run_count(self.fit_run_count)
        )

    @classmethod
    def fit(
        cls,
        fit_healthy_runs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
        *,
        expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
    ) -> "HealthyPrefixNormalizer":
        count = _validated_fit_run_count(expected_run_count)
        runs = _ordered_runs(fit_healthy_runs, expected_run_count=count)
        # No row at or after REFERENCE_STOP is read to fit either quantity.
        pooled = np.concatenate(
            [run[:REFERENCE_STOP] for run in runs], axis=0, dtype=np.float64
        )
        center = np.median(pooled, axis=0)
        scale = MAD_CONSISTENCY * np.median(np.abs(pooled - center), axis=0)
        scale = np.maximum(scale, NORMALIZATION_FLOOR)
        return cls(center.astype(np.float64), scale.astype(np.float64), count)

    def transform_run(self, run: Any) -> np.ndarray:
        values = _validated_run(run)
        result = np.clip(
            (values - self.center) / self.scale,
            -NORMALIZATION_CLIP,
            NORMALIZATION_CLIP,
        )
        if not np.isfinite(result).all():  # pragma: no cover - guarded inputs
            raise FloatingPointError("normalization produced nonfinite values")
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "candidate": CANDIDATE_NAME,
                "fit_run_count": self.fit_run_count,
                "fit_rows_half_open": (0, REFERENCE_STOP),
                "fit_population": "pooled_fit_healthy_prefix_values_only",
                "center": "per_channel_median",
                "scale": "max(1.4826_per_channel_MAD,1e-6)",
                "clip": (-NORMALIZATION_CLIP, NORMALIZATION_CLIP),
            }
        )


def _canonical_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    """Remove the arbitrary sign of real symmetric eigensolvers."""

    result = np.array(vectors, dtype=np.float64, copy=True)
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _descending_eigensystem(
    covariance: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values, kind="stable")[::-1][:rank]
    selected_values = np.maximum(values[order], SUBSPACE_RIDGE)
    selected_vectors = _canonical_eigenvectors(vectors[:, order])
    return selected_values.astype(np.float64), selected_vectors.astype(np.float64)


def _lagged_rows(windows: np.ndarray, order: int, *, include_current: bool) -> np.ndarray:
    # windows: [batch, time, channel].  The feature order is newest to oldest.
    stop = windows.shape[1]
    if include_current:
        pieces = [windows[:, order - lag : stop - lag] for lag in range(order + 1)]
    else:
        pieces = [windows[:, order - lag : stop - lag] for lag in range(1, order + 1)]
    return np.concatenate(pieces, axis=2)


@dataclass(frozen=True)
class HealthySubspaceModel:
    """Eight strictly fit-prefix health coordinates.

    PCA and lag-4 DPCA use fixed rank 16.  SFA is fitted inside the whitened
    PCA-16 space and reports derivative energy in the four slowest and four
    fastest directions.  VAR(8) is ridge fitted; its state statistic is a
    diagonal-covariance Mahalanobis distance (explicitly avoiding an unstable
    416-by-416 inverse), while residual Mahalanobis uses the fitted 52-by-52
    ridge covariance.  No statistic is refitted on an evaluated window.
    """

    pca_center: np.ndarray
    pca_components: np.ndarray
    pca_variances: np.ndarray
    dpca_center: np.ndarray
    dpca_components: np.ndarray
    dpca_variances: np.ndarray
    sfa_directions: np.ndarray
    sfa_delta_variances: np.ndarray
    var_state_center: np.ndarray
    var_state_scale: np.ndarray
    var_target_center: np.ndarray
    var_coefficients: np.ndarray
    var_residual_precision: np.ndarray
    fit_run_count: int = FIT_HEALTHY_RUN_COUNT

    def __post_init__(self) -> None:
        shapes = {
            "pca_center": (PROCESS_FEATURE_COUNT,),
            "pca_components": (PROCESS_FEATURE_COUNT, PCA_RANK),
            "pca_variances": (PCA_RANK,),
            "dpca_center": ((DPCA_LAG + 1) * PROCESS_FEATURE_COUNT,),
            "dpca_components": (
                (DPCA_LAG + 1) * PROCESS_FEATURE_COUNT,
                DPCA_RANK,
            ),
            "dpca_variances": (DPCA_RANK,),
            "sfa_directions": (PCA_RANK, PCA_RANK),
            "sfa_delta_variances": (PCA_RANK,),
            "var_state_center": (VAR_ORDER * PROCESS_FEATURE_COUNT,),
            "var_state_scale": (VAR_ORDER * PROCESS_FEATURE_COUNT,),
            "var_target_center": (PROCESS_FEATURE_COUNT,),
            "var_coefficients": (
                VAR_ORDER * PROCESS_FEATURE_COUNT,
                PROCESS_FEATURE_COUNT,
            ),
            "var_residual_precision": (
                PROCESS_FEATURE_COUNT,
                PROCESS_FEATURE_COUNT,
            ),
        }
        for name, shape in shapes.items():
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), shape=shape, name=name),
            )
        if np.any(self.pca_variances <= 0.0) or np.any(self.dpca_variances <= 0.0):
            raise ValueError("subspace variances must be positive")
        if np.any(self.sfa_delta_variances <= 0.0) or np.any(self.var_state_scale <= 0.0):
            raise ValueError("SFA/VAR scales must be positive")
        object.__setattr__(
            self, "fit_run_count", _validated_fit_run_count(self.fit_run_count)
        )

    @classmethod
    def fit(
        cls,
        normalized_prefixes: Any,
        *,
        expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
    ) -> "HealthySubspaceModel":
        count = _validated_fit_run_count(expected_run_count)
        prefixes = np.asarray(normalized_prefixes)
        expected = (count, REFERENCE_STOP, PROCESS_FEATURE_COUNT)
        if prefixes.shape != expected or prefixes.dtype != np.dtype(np.float64):
            raise ValueError(f"normalized_prefixes must be exact float64 with shape {expected}")
        if not np.isfinite(prefixes).all():
            raise ValueError("normalized_prefixes must contain only finite values")

        flat = prefixes.reshape(-1, PROCESS_FEATURE_COUNT)
        pca_center = flat.mean(axis=0)
        pca_centered = flat - pca_center
        pca_cov = (pca_centered.T @ pca_centered) / float(flat.shape[0])
        pca_values, pca_components = _descending_eigensystem(pca_cov, PCA_RANK)

        dpca_rows = _lagged_rows(prefixes, DPCA_LAG, include_current=True).reshape(
            -1, (DPCA_LAG + 1) * PROCESS_FEATURE_COUNT
        )
        dpca_center = dpca_rows.mean(axis=0)
        dpca_centered = dpca_rows - dpca_center
        dpca_cov = (dpca_centered.T @ dpca_centered) / float(dpca_rows.shape[0])
        dpca_values, dpca_components = _descending_eigensystem(dpca_cov, DPCA_RANK)

        whitened = ((prefixes - pca_center) @ pca_components) / np.sqrt(pca_values)
        differences = np.diff(whitened, axis=1).reshape(-1, PCA_RANK)
        sfa_cov = (differences.T @ differences) / float(differences.shape[0])
        sfa_values, sfa_vectors = np.linalg.eigh(sfa_cov)
        sfa_order = np.argsort(sfa_values, kind="stable")
        sfa_values = np.maximum(sfa_values[sfa_order], SUBSPACE_RIDGE)
        sfa_vectors = _canonical_eigenvectors(sfa_vectors[:, sfa_order])

        var_states = _lagged_rows(prefixes, VAR_ORDER, include_current=False).reshape(
            -1, VAR_ORDER * PROCESS_FEATURE_COUNT
        )
        var_targets = prefixes[:, VAR_ORDER:, :].reshape(-1, PROCESS_FEATURE_COUNT)
        state_center = var_states.mean(axis=0)
        target_center = var_targets.mean(axis=0)
        centered_states = var_states - state_center
        centered_targets = var_targets - target_center
        observation_count = float(var_states.shape[0])
        gram = (centered_states.T @ centered_states) / observation_count
        gram.flat[:: gram.shape[0] + 1] += SUBSPACE_RIDGE
        cross = (centered_states.T @ centered_targets) / observation_count
        coefficients = np.linalg.solve(gram, cross)
        residuals = centered_targets - centered_states @ coefficients
        residual_cov = (residuals.T @ residuals) / observation_count
        residual_cov.flat[:: residual_cov.shape[0] + 1] += SUBSPACE_RIDGE
        residual_precision = np.linalg.inv(residual_cov)
        residual_precision = 0.5 * (residual_precision + residual_precision.T)
        state_scale = np.sqrt(np.maximum(np.var(var_states, axis=0), SUBSPACE_RIDGE))

        values = (
            pca_center,
            pca_components,
            pca_values,
            dpca_center,
            dpca_components,
            dpca_values,
            sfa_vectors,
            sfa_values,
            state_center,
            state_scale,
            target_center,
            coefficients,
            residual_precision,
        )
        if not all(np.isfinite(value).all() for value in values):
            raise FloatingPointError("healthy-subspace fitting produced nonfinite values")
        return cls(*values, fit_run_count=count)

    def statistics_many(self, windows: Any) -> np.ndarray:
        array = np.asarray(windows)
        expected_tail = (WINDOW_LENGTH, PROCESS_FEATURE_COUNT)
        if (
            array.ndim != 3
            or array.shape[1:] != expected_tail
            or array.dtype != np.dtype(np.float64)
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                "windows must be finite exact float64 with shape "
                f"(batch, {WINDOW_LENGTH}, {PROCESS_FEATURE_COUNT})"
            )
        batch = array.shape[0]

        centered = array - self.pca_center
        scores = centered @ self.pca_components
        pca_t2 = np.mean(
            np.sum(np.square(scores) / self.pca_variances, axis=2), axis=1
        )
        reconstructed = scores @ self.pca_components.T
        pca_q = np.mean(np.sum(np.square(centered - reconstructed), axis=2), axis=1)

        dpca = _lagged_rows(array, DPCA_LAG, include_current=True)
        dpca_centered = dpca - self.dpca_center
        dpca_scores = dpca_centered @ self.dpca_components
        dpca_t2 = np.mean(
            np.sum(np.square(dpca_scores) / self.dpca_variances, axis=2), axis=1
        )
        dpca_reconstructed = dpca_scores @ self.dpca_components.T
        dpca_q = np.mean(
            np.sum(np.square(dpca_centered - dpca_reconstructed), axis=2), axis=1
        )

        whitened = scores / np.sqrt(self.pca_variances)
        sfa_delta = np.diff(whitened, axis=1) @ self.sfa_directions
        standardized_sfa = np.square(sfa_delta) / self.sfa_delta_variances
        slow = np.mean(standardized_sfa[:, :, :SFA_TAIL_COUNT], axis=(1, 2))
        fast = np.mean(standardized_sfa[:, :, -SFA_TAIL_COUNT:], axis=(1, 2))

        var_states = _lagged_rows(array, VAR_ORDER, include_current=False)
        state_centered = var_states - self.var_state_center
        state_mahalanobis = np.mean(
            np.sum(np.square(state_centered / self.var_state_scale), axis=2), axis=1
        )
        targets = array[:, VAR_ORDER:, :] - self.var_target_center
        residuals = targets - state_centered @ self.var_coefficients
        residual_flat = residuals.reshape(-1, PROCESS_FEATURE_COUNT)
        residual_quadratic = np.einsum(
            "bi,ij,bj->b",
            residual_flat,
            self.var_residual_precision,
            residual_flat,
            optimize=True,
        ).reshape(batch, WINDOW_LENGTH - VAR_ORDER)
        residual_mahalanobis = residual_quadratic.mean(axis=1)

        result = np.column_stack(
            (
                pca_t2,
                pca_q,
                dpca_t2,
                dpca_q,
                slow,
                fast,
                state_mahalanobis,
                residual_mahalanobis,
            )
        )
        if result.shape != (batch, HEALTH_FEATURE_COUNT) or not np.isfinite(result).all():
            raise FloatingPointError("healthy-subspace statistics are invalid")
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    def statistics(self, window: Any) -> np.ndarray:
        array = np.asarray(window)
        if array.shape != (WINDOW_LENGTH, PROCESS_FEATURE_COUNT):
            raise ValueError(
                f"window must have shape ({WINDOW_LENGTH}, {PROCESS_FEATURE_COUNT})"
            )
        return self.statistics_many(array[None, ...])[0]

    def statistics_path(self, normalized_run: Any, endpoints: Any) -> np.ndarray:
        """Exact window statistics with every pointwise quadratic computed once."""

        run = _validated_run(normalized_run, name="normalized_run")
        endpoint_array = _validated_endpoints(endpoints)
        starts = endpoint_array - WINDOW_LENGTH + 1

        centered = run - self.pca_center
        scores = centered @ self.pca_components
        pca_t2_path = np.sum(np.square(scores) / self.pca_variances, axis=1)
        reconstruction = scores @ self.pca_components.T
        pca_q_path = np.sum(np.square(centered - reconstruction), axis=1)

        dpca = _lagged_rows(run[None, ...], DPCA_LAG, include_current=True)[0]
        dpca_centered = dpca - self.dpca_center
        dpca_scores = dpca_centered @ self.dpca_components
        dpca_t2_path = np.sum(
            np.square(dpca_scores) / self.dpca_variances, axis=1
        )
        dpca_reconstruction = dpca_scores @ self.dpca_components.T
        dpca_q_path = np.sum(
            np.square(dpca_centered - dpca_reconstruction), axis=1
        )

        whitened = scores / np.sqrt(self.pca_variances)
        sfa_delta = np.diff(whitened, axis=0) @ self.sfa_directions
        standardized_sfa = np.square(sfa_delta) / self.sfa_delta_variances
        slow_path = np.mean(standardized_sfa[:, :SFA_TAIL_COUNT], axis=1)
        fast_path = np.mean(standardized_sfa[:, -SFA_TAIL_COUNT:], axis=1)

        var_states = _lagged_rows(run[None, ...], VAR_ORDER, include_current=False)[0]
        state_centered = var_states - self.var_state_center
        state_path = np.sum(
            np.square(state_centered / self.var_state_scale), axis=1
        )
        targets = run[VAR_ORDER:] - self.var_target_center
        residuals = targets - state_centered @ self.var_coefficients
        residual_path = np.einsum(
            "bi,ij,bj->b",
            residuals,
            self.var_residual_precision,
            residuals,
            optimize=True,
        )

        def interval_mean(
            values: np.ndarray, first: np.ndarray, last: np.ndarray, offset: int
        ) -> np.ndarray:
            # values[0] corresponds to the global row ``offset``.
            prefix = np.concatenate(
                (np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64))
            )
            left = first - offset
            right = last - offset + 1
            return (prefix[right] - prefix[left]) / (right - left)

        result = np.column_stack(
            (
                interval_mean(pca_t2_path, starts, endpoint_array, 0),
                interval_mean(pca_q_path, starts, endpoint_array, 0),
                interval_mean(
                    dpca_t2_path, starts + DPCA_LAG, endpoint_array, DPCA_LAG
                ),
                interval_mean(
                    dpca_q_path, starts + DPCA_LAG, endpoint_array, DPCA_LAG
                ),
                interval_mean(slow_path, starts + 1, endpoint_array, 1),
                interval_mean(fast_path, starts + 1, endpoint_array, 1),
                interval_mean(
                    state_path, starts + VAR_ORDER, endpoint_array, VAR_ORDER
                ),
                interval_mean(
                    residual_path, starts + VAR_ORDER, endpoint_array, VAR_ORDER
                ),
            )
        )
        if result.shape != (endpoint_array.size, HEALTH_FEATURE_COUNT):
            raise RuntimeError("healthy-subspace path geometry changed")
        if not np.isfinite(result).all():
            raise FloatingPointError("healthy-subspace path is nonfinite")
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result


def _window_spectrum(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = windows - windows.mean(axis=-1, keepdims=True)
    positive_power = np.square(np.abs(np.fft.rfft(centered, axis=-1)[..., 1:]))
    probability = (positive_power + SPECTRAL_EPSILON) / (
        positive_power.sum(axis=-1, keepdims=True)
        + positive_power.shape[-1] * SPECTRAL_EPSILON
    )
    band_shares = np.stack(
        [probability[..., lo - 1 : hi - 1].sum(axis=-1) for lo, hi in RFFT_BANDS],
        axis=-1,
    )
    entropy = -np.sum(probability * np.log(probability), axis=-1) / np.log(
        float(probability.shape[-1])
    )
    bins = np.arange(1, probability.shape[-1] + 1, dtype=np.float64)
    centroid = np.sum(probability * bins, axis=-1) / float(probability.shape[-1])
    return band_shares, entropy, centroid


def paired_channel_summaries(current: Any, reference: Any) -> np.ndarray:
    """Return channel-major ``[endpoint, 52, 11]`` summary deltas."""

    current_array = np.asarray(current)
    reference_array = np.asarray(reference)
    if (
        current_array.ndim != 3
        or current_array.shape[1:] != A_SHAPE
        or current_array.dtype != np.dtype(np.float64)
        or not np.isfinite(current_array).all()
    ):
        raise ValueError("current must be finite exact float64 [endpoint,52,128]")
    if (
        reference_array.shape != A_SHAPE
        or reference_array.dtype != np.dtype(np.float64)
        or not np.isfinite(reference_array).all()
    ):
        raise ValueError("reference must be finite exact float64 [52,128]")
    def describe(values: np.ndarray) -> tuple[np.ndarray, ...]:
        mean = values.mean(axis=-1)
        log_variance = np.log(np.var(values, axis=-1) + SUMMARY_EPSILON)
        log_diff_energy = np.log(
            np.mean(np.square(np.diff(values, axis=-1)), axis=-1) + SUMMARY_EPSILON
        )
        shares, entropy, centroid = _window_spectrum(values)
        return (
            mean,
            log_variance,
            log_diff_energy,
            *tuple(np.log(shares[..., index] + SUMMARY_EPSILON) for index in range(6)),
            entropy,
            centroid,
        )

    current_stats = describe(current_array)
    # Reference summaries are calculated once, then broadcast over endpoints.
    reference_stats = describe(reference_array[None, ...])
    result = np.stack(
        [left - right for left, right in zip(current_stats, reference_stats)], axis=-1
    )
    if result.shape != (current_array.shape[0], PROCESS_FEATURE_COUNT, 11):
        raise RuntimeError("channel summary geometry changed")
    if not np.isfinite(result).all():
        raise FloatingPointError("channel summaries are nonfinite")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _safe_row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.mean(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.mean(np.square(left_centered), axis=1)
        * np.mean(np.square(right_centered), axis=1)
    )
    return numerator / np.maximum(denominator, SUMMARY_EPSILON)


def _welch_coherence(actuator: np.ndarray, response: np.ndarray) -> np.ndarray:
    segment_length = 32
    starts = range(0, WINDOW_LENGTH - segment_length + 1, 16)
    taper = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(segment_length, dtype=np.float64)
        / float(segment_length)
    )
    actuator_segments = np.stack(
        [actuator[:, start : start + segment_length] for start in starts], axis=1
    )
    response_segments = np.stack(
        [response[:, start : start + segment_length] for start in starts], axis=1
    )
    actuator_segments = (
        actuator_segments - actuator_segments.mean(axis=2, keepdims=True)
    ) * taper
    response_segments = (
        response_segments - response_segments.mean(axis=2, keepdims=True)
    ) * taper
    ux = np.fft.rfft(actuator_segments, axis=2)[..., 1:]
    yy = np.fft.rfft(response_segments, axis=2)[..., 1:]
    pxx = np.mean(np.square(np.abs(ux)), axis=1)
    pyy = np.mean(np.square(np.abs(yy)), axis=1)
    pxy = np.mean(ux * np.conjugate(yy), axis=1)
    coherence = np.square(np.abs(pxy)) / np.maximum(
        pxx * pyy, SPECTRAL_EPSILON
    )
    return np.mean(np.clip(coherence, 0.0, 1.0), axis=1)


def _actuator_psd_concentration(actuator: np.ndarray) -> np.ndarray:
    centered = actuator - actuator.mean(axis=1, keepdims=True)
    power = np.square(np.abs(np.fft.rfft(centered, axis=1)[:, 1:]))
    probability = (power + SPECTRAL_EPSILON) / (
        power.sum(axis=1, keepdims=True) + power.shape[1] * SPECTRAL_EPSILON
    )
    return probability.max(axis=1)


def _loop_metrics(
    windows: np.ndarray, actuator_index: int, response_index: int
) -> dict[str, np.ndarray]:
    actuator = windows[:, actuator_index, :]
    response = windows[:, response_index, :]
    actuator_diff = np.diff(actuator, axis=1)
    response_diff = np.diff(response, axis=1)
    dwell = np.mean(np.abs(actuator_diff) <= 0.05, axis=1)
    stick_slip = np.mean(
        (np.abs(actuator_diff[:, 1:]) >= 0.50)
        & (np.abs(actuator_diff[:, :-1]) <= 0.05),
        axis=1,
    )
    reversal = np.mean(actuator_diff[:, 1:] * actuator_diff[:, :-1] < 0.0, axis=1)
    oriented_area = np.mean(
        actuator_diff * response[:, :-1] - response_diff * actuator[:, :-1],
        axis=1,
    )
    lag_asymmetry = _safe_row_correlation(
        actuator_diff[:, :-1], response_diff[:, 1:]
    ) - _safe_row_correlation(response_diff[:, :-1], actuator_diff[:, 1:])
    coherence = _welch_coherence(actuator, response)
    concentration = _actuator_psd_concentration(actuator)
    return {
        "dwell": dwell,
        "stick_slip": stick_slip,
        "reversal": reversal,
        "oriented_area": oriented_area,
        "lag_asymmetry": lag_asymmetry,
        "coherence": coherence,
        "concentration": concentration,
    }


def _reference_arx_residual_ratio(
    current: np.ndarray,
    reference: np.ndarray,
    actuator_index: int,
    response_index: int,
) -> np.ndarray:
    ref_u = reference[actuator_index]
    ref_y = reference[response_index]
    design = np.column_stack(
        (np.ones(WINDOW_LENGTH - 1, dtype=np.float64), ref_y[:-1], ref_u[:-1])
    )
    gram = design.T @ design
    gram.flat[::4] += SUBSPACE_RIDGE
    coefficients = np.linalg.solve(gram, design.T @ ref_y[1:])
    reference_residual = ref_y[1:] - design @ coefficients
    reference_mse = np.mean(np.square(reference_residual))

    current_u = current[:, actuator_index, :]
    current_y = current[:, response_index, :]
    prediction = (
        coefficients[0]
        + coefficients[1] * current_y[:, :-1]
        + coefficients[2] * current_u[:, :-1]
    )
    current_mse = np.mean(np.square(current_y[:, 1:] - prediction), axis=1)
    return np.log(
        (current_mse + SUMMARY_EPSILON) / (reference_mse + SUMMARY_EPSILON)
    )


def loop_summaries(current: Any, reference: Any) -> np.ndarray:
    """Return the documented exact 24-dimensional XMV11 loop branch."""

    current_array = np.asarray(current)
    reference_array = np.asarray(reference)
    if (
        current_array.ndim != 3
        or current_array.shape[1:] != A_SHAPE
        or current_array.dtype != np.dtype(np.float64)
        or not np.isfinite(current_array).all()
    ):
        raise ValueError("current must be finite exact float64 [endpoint,52,128]")
    if (
        reference_array.shape != A_SHAPE
        or reference_array.dtype != np.dtype(np.float64)
        or not np.isfinite(reference_array).all()
    ):
        raise ValueError("reference must be finite exact float64 [52,128]")
    pieces: list[np.ndarray] = []
    for actuator, response, _ in LOOP_PAIRS:
        current_metrics = _loop_metrics(current_array, actuator, response)
        # The same-run reference is fixed, so never repeat its FFT/Welch work.
        reference_metrics = _loop_metrics(reference_array[None, ...], actuator, response)
        pieces.extend(
            (
                current_metrics["dwell"],
                current_metrics["dwell"] - reference_metrics["dwell"],
                current_metrics["stick_slip"],
                current_metrics["stick_slip"] - reference_metrics["stick_slip"],
                current_metrics["reversal"],
                current_metrics["reversal"] - reference_metrics["reversal"],
                current_metrics["oriented_area"],
                current_metrics["oriented_area"] - reference_metrics["oriented_area"],
                current_metrics["lag_asymmetry"] - reference_metrics["lag_asymmetry"],
                _reference_arx_residual_ratio(
                    current_array, reference_array, actuator, response
                ),
                current_metrics["coherence"] - reference_metrics["coherence"],
                current_metrics["concentration"]
                - reference_metrics["concentration"],
            )
        )
    result = np.column_stack(pieces)
    if result.shape != (current_array.shape[0], LOOP_FEATURE_COUNT):
        raise RuntimeError("loop summary geometry changed")
    if not np.isfinite(result).all():
        raise FloatingPointError("loop summaries are nonfinite")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validated_endpoints(endpoints: Any) -> np.ndarray:
    array = np.asarray(endpoints)
    if array.ndim != 1 or array.dtype != np.dtype(np.int64) or array.size == 0:
        raise ValueError("endpoints must be a nonempty exact int64 vector")
    if np.any(array < WINDOW_LENGTH - 1) or np.any(array >= RUN_LENGTH):
        raise ValueError("endpoints cannot form complete causal windows")
    if np.any(np.diff(array) <= 0):
        raise ValueError("endpoints must be strictly increasing")
    return array


@dataclass(frozen=True)
class PCFCycleFeatures:
    endpoints: np.ndarray
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        endpoints = np.asarray(self.endpoints)
        if endpoints.ndim != 1 or endpoints.dtype != np.dtype(np.int64):
            raise ValueError("endpoints must be exact int64")
        endpoint_copy = np.array(endpoints, dtype=np.int64, copy=True)
        endpoint_copy.setflags(write=False)
        object.__setattr__(self, "endpoints", endpoint_copy)
        object.__setattr__(
            self,
            "a",
            _readonly(
                self.a,
                shape=(endpoint_copy.size, *A_SHAPE),
                name="a",
            ),
        )
        object.__setattr__(
            self,
            "b",
            _readonly(
                self.b,
                shape=(endpoint_copy.size, B_FEATURE_COUNT),
                name="b",
            ),
        )


@dataclass(frozen=True)
class BFeatureStandardizer:
    """Mandatory robust standardizer for the heterogeneous 604-vector B.

    ``fit`` accepts exactly either 5,100 development-fit rows (60x85) or 8,500
    final-refit rows (100x85), selected by an explicit restricted count.
    Per-coordinate median and 1.4826*MAD are fitted once; transform clips to
    [-10, 10].
    """

    center: np.ndarray
    scale: np.ndarray
    fit_run_count: int = FIT_HEALTHY_RUN_COUNT

    def __post_init__(self) -> None:
        center = _readonly(self.center, shape=(B_FEATURE_COUNT,), name="b_center")
        scale = _readonly(self.scale, shape=(B_FEATURE_COUNT,), name="b_scale")
        if np.any(scale < NORMALIZATION_FLOOR):
            raise ValueError("b_scale must respect the normalization floor")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self, "fit_run_count", _validated_fit_run_count(self.fit_run_count)
        )

    @classmethod
    def fit(
        cls,
        fit_healthy_b_rows: Any,
        *,
        expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
    ) -> "BFeatureStandardizer":
        count = _validated_fit_run_count(expected_run_count)
        rows = np.asarray(fit_healthy_b_rows)
        expected = (count * MONITOR_ENDPOINTS.size, B_FEATURE_COUNT)
        if rows.shape != expected or rows.dtype != np.dtype(np.float64):
            raise ValueError(
                f"fit_healthy_b_rows must be exact float64 with shape {expected}"
            )
        if not np.isfinite(rows).all():
            raise ValueError("fit_healthy_b_rows must contain only finite values")
        center = np.median(rows, axis=0)
        scale = MAD_CONSISTENCY * np.median(np.abs(rows - center), axis=0)
        scale = np.maximum(scale, NORMALIZATION_FLOOR)
        return cls(center.astype(np.float64), scale.astype(np.float64), count)

    def transform(self, raw_b: Any) -> np.ndarray:
        values = np.asarray(raw_b)
        if (
            values.dtype != np.dtype(np.float64)
            or values.ndim not in (1, 2)
            or values.shape[-1] != B_FEATURE_COUNT
        ):
            raise ValueError("raw_b must be exact float64 with final dimension 604")
        if not np.isfinite(values).all():
            raise ValueError("raw_b must contain only finite values")
        result = np.clip(
            (values - self.center) / self.scale,
            -NORMALIZATION_CLIP,
            NORMALIZATION_CLIP,
        )
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    def inverse_transform(self, standardized_b: Any) -> np.ndarray:
        values = np.asarray(standardized_b)
        if (
            values.dtype != np.dtype(np.float64)
            or values.ndim not in (1, 2)
            or values.shape[-1] != B_FEATURE_COUNT
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "standardized_b must be finite exact float64 with final dimension 604"
            )
        result = values * self.scale + self.center
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "candidate": CANDIDATE_NAME,
                "mandatory_before_neural_stage": True,
                "fit_run_count": self.fit_run_count,
                "fit_rows": self.fit_run_count * MONITOR_ENDPOINTS.size,
                "fit_identity": (
                    f"{self.fit_run_count}_fit_healthy_runs_x_85_monitor_endpoints"
                ),
                "center": "per_coordinate_median",
                "scale": "max(1.4826_per_coordinate_MAD,1e-6)",
                "clip": (-NORMALIZATION_CLIP, NORMALIZATION_CLIP),
            }
        )


@dataclass(frozen=True)
class PCFCycleFeatureExtractor:
    normalizer: HealthyPrefixNormalizer
    health_model: HealthySubspaceModel

    def __post_init__(self) -> None:
        if not isinstance(self.normalizer, HealthyPrefixNormalizer):
            raise ValueError("normalizer must be a HealthyPrefixNormalizer")
        if not isinstance(self.health_model, HealthySubspaceModel):
            raise ValueError("health_model must be a HealthySubspaceModel")
        if self.normalizer.fit_run_count != self.health_model.fit_run_count:
            raise ValueError("normalizer and health_model fit counts must match")

    @classmethod
    def fit(
        cls,
        fit_healthy_runs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
        *,
        expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
    ) -> "PCFCycleFeatureExtractor":
        count = _validated_fit_run_count(expected_run_count)
        runs = _ordered_runs(fit_healthy_runs, expected_run_count=count)
        normalizer = HealthyPrefixNormalizer.fit(runs, expected_run_count=count)
        prefixes = np.stack(
            [
                np.clip(
                    (run[:REFERENCE_STOP] - normalizer.center) / normalizer.scale,
                    -NORMALIZATION_CLIP,
                    NORMALIZATION_CLIP,
                )
                for run in runs
            ],
            axis=0,
        )
        health_model = HealthySubspaceModel.fit(prefixes, expected_run_count=count)
        return cls(normalizer=normalizer, health_model=health_model)

    def extract_run(
        self,
        run: Any,
        *,
        endpoints: np.ndarray = MONITOR_ENDPOINTS,
    ) -> PCFCycleFeatures:
        endpoint_array = _validated_endpoints(endpoints)
        normalized = self.normalizer.transform_run(run)
        reference = np.array(normalized[:REFERENCE_STOP].T, copy=True, order="C")
        current = np.stack(
            [
                normalized[endpoint - WINDOW_LENGTH + 1 : endpoint + 1].T
                for endpoint in endpoint_array
            ],
            axis=0,
        )
        a = current - reference[None, :, :]
        channel = paired_channel_summaries(current, reference).reshape(
            endpoint_array.size, CHANNEL_FEATURE_COUNT
        )
        loops = loop_summaries(current, reference)
        health = self.health_model.statistics_path(normalized, endpoint_array)
        b = np.concatenate((channel, loops, health), axis=1)
        if a.shape != (endpoint_array.size, *A_SHAPE) or b.shape != (
            endpoint_array.size,
            B_FEATURE_COUNT,
        ):
            raise RuntimeError("PCF-CycleRank feature geometry changed")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError("PCF-CycleRank features are nonfinite")
        return PCFCycleFeatures(endpoint_array, a, b)

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "candidate": CANDIDATE_NAME,
                "input_contract": "finite_exact_float64_500x52",
                "reference_half_open": (0, REFERENCE_STOP),
                "current_window": "endpoint_minus_127_through_endpoint_inclusive",
                "monitor_endpoints": tuple(int(value) for value in MONITOR_ENDPOINTS),
                "a_shape_per_endpoint": A_SHAPE,
                "b_feature_count": B_FEATURE_COUNT,
                "b_blocks": (CHANNEL_FEATURE_COUNT, LOOP_FEATURE_COUNT, HEALTH_FEATURE_COUNT),
                "b_output": "raw_requires_mandatory_fit_healthy_BFeatureStandardizer",
                "channel_summary_names": CHANNEL_SUMMARY_NAMES,
                "rfft_positive_bin_bands": RFFT_BANDS,
                "loop_pairs": LOOP_PAIRS,
                "loop_feature_names": LOOP_FEATURE_NAMES,
                "health_feature_names": HEALTH_FEATURE_NAMES,
                "fit_run_count": self.normalizer.fit_run_count,
                "fit_access": (
                    f"{self.normalizer.fit_run_count}_fit_healthy_prefixes_0_128_only"
                ),
                "causality": "fixed_reference_and_rows_at_or_before_endpoint",
            }
        )


def fit_pcfcycle_features(
    fit_healthy_runs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    *,
    expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
) -> PCFCycleFeatureExtractor:
    """Public fit entry point; equivalent to ``PCFCycleFeatureExtractor.fit``."""

    return PCFCycleFeatureExtractor.fit(
        fit_healthy_runs, expected_run_count=expected_run_count
    )


def extract_pcfcycle_features(
    fitted: PCFCycleFeatureExtractor,
    run: Any,
    *,
    endpoints: np.ndarray = MONITOR_ENDPOINTS,
) -> PCFCycleFeatures:
    """Public extraction entry point for an already-fitted feature map."""

    if not isinstance(fitted, PCFCycleFeatureExtractor):
        raise ValueError("fitted must be a PCFCycleFeatureExtractor")
    return fitted.extract_run(run, endpoints=endpoints)


def fit_b_feature_standardizer(
    fitted: PCFCycleFeatureExtractor,
    fit_healthy_runs: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    *,
    expected_run_count: int = FIT_HEALTHY_RUN_COUNT,
) -> BFeatureStandardizer:
    """Fit mandatory B scaling from all permitted healthy monitor paths."""

    if not isinstance(fitted, PCFCycleFeatureExtractor):
        raise ValueError("fitted must be a PCFCycleFeatureExtractor")
    count = _validated_fit_run_count(expected_run_count)
    if fitted.normalizer.fit_run_count != count:
        raise ValueError("extractor fit count must equal expected_run_count")
    runs = _ordered_runs(fit_healthy_runs, expected_run_count=count)
    rows = np.concatenate(
        [fitted.extract_run(run, endpoints=MONITOR_ENDPOINTS).b for run in runs], axis=0
    )
    if rows.shape != (count * MONITOR_ENDPOINTS.size, B_FEATURE_COUNT):  # pragma: no cover
        raise RuntimeError("fit-healthy B row geometry changed")
    return BFeatureStandardizer.fit(rows, expected_run_count=count)


_HEALTH_STATE_NAMES = (
    "pca_center",
    "pca_components",
    "pca_variances",
    "dpca_center",
    "dpca_components",
    "dpca_variances",
    "sfa_directions",
    "sfa_delta_variances",
    "var_state_center",
    "var_state_scale",
    "var_target_center",
    "var_coefficients",
    "var_residual_precision",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def save_pcfcycle_feature_state(
    path: str | os.PathLike[str],
    fitted: PCFCycleFeatureExtractor,
    b_standardizer: BFeatureStandardizer | None,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save the complete feature state as a pickle-free NPZ."""

    if not isinstance(fitted, PCFCycleFeatureExtractor):
        raise ValueError("fitted must be a PCFCycleFeatureExtractor")
    if b_standardizer is not None and not isinstance(
        b_standardizer, BFeatureStandardizer
    ):
        raise ValueError("b_standardizer must be BFeatureStandardizer or None")
    if (
        b_standardizer is not None
        and b_standardizer.fit_run_count != fitted.normalizer.fit_run_count
    ):
        raise ValueError("feature and B-standardizer fit counts must match")
    metadata_document = {} if metadata is None else dict(metadata)
    metadata_json = _canonical_json(metadata_document)
    payload: dict[str, np.ndarray] = {
        "schema": np.array(FEATURE_STATE_SCHEMA),
        "schema_version": np.array(FEATURE_STATE_SCHEMA_VERSION, dtype=np.int64),
        "metadata_json": np.array(metadata_json),
        "has_b_standardizer": np.array(
            1 if b_standardizer is not None else 0, dtype=np.int64
        ),
        "fit_run_count": np.array(
            fitted.normalizer.fit_run_count, dtype=np.int64
        ),
        "normalizer_center": np.array(fitted.normalizer.center, copy=True),
        "normalizer_scale": np.array(fitted.normalizer.scale, copy=True),
    }
    for name in _HEALTH_STATE_NAMES:
        payload[f"health_{name}"] = np.array(
            getattr(fitted.health_model, name), copy=True
        )
    if b_standardizer is not None:
        payload["b_fit_run_count"] = np.array(
            b_standardizer.fit_run_count, dtype=np.int64
        )
        payload["b_center"] = np.array(b_standardizer.center, copy=True)
        payload["b_scale"] = np.array(b_standardizer.scale, copy=True)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".npz",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _scalar_text(array: np.ndarray, *, name: str, limit: int) -> str:
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    result = str(array.item())
    if len(result) > limit:
        raise ValueError(f"{name} exceeds its size limit")
    return result


def load_pcfcycle_feature_state(
    path: str | os.PathLike[str],
) -> tuple[PCFCycleFeatureExtractor, BFeatureStandardizer | None, dict[str, object]]:
    """Load and structurally validate a pickle-free feature-state NPZ."""

    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise ValueError("object arrays are forbidden in PCF-Cycle feature state")
    base_keys = {
        "schema",
        "schema_version",
        "metadata_json",
        "has_b_standardizer",
        "fit_run_count",
        "normalizer_center",
        "normalizer_scale",
        *(f"health_{name}" for name in _HEALTH_STATE_NAMES),
    }
    for scalar_name in ("schema_version", "has_b_standardizer", "fit_run_count"):
        value = arrays.get(scalar_name)
        if value is None or value.shape != () or value.dtype != np.dtype(np.int64):
            raise ValueError(f"{scalar_name} must be scalar int64")
    has_b = int(arrays["has_b_standardizer"].item())
    if has_b not in (0, 1):
        raise ValueError("has_b_standardizer must be zero or one")
    expected_keys = base_keys | (
        {"b_fit_run_count", "b_center", "b_scale"} if has_b else set()
    )
    if set(arrays) != expected_keys:
        raise ValueError("feature-state NPZ contains extra or missing arrays")
    if _scalar_text(arrays["schema"], name="schema", limit=128) != FEATURE_STATE_SCHEMA:
        raise ValueError("feature-state schema is invalid")
    if int(arrays["schema_version"].item()) != FEATURE_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported feature-state schema version")
    fit_run_count = _validated_fit_run_count(int(arrays["fit_run_count"].item()))
    if has_b:
        b_count_array = arrays["b_fit_run_count"]
        if b_count_array.shape != () or b_count_array.dtype != np.dtype(np.int64):
            raise ValueError("b_fit_run_count must be scalar int64")
        b_fit_run_count = _validated_fit_run_count(int(b_count_array.item()))
        if b_fit_run_count != fit_run_count:
            raise ValueError("feature and B-standardizer fit counts must match")
    metadata_json = _scalar_text(
        arrays["metadata_json"], name="metadata_json", limit=1_048_576
    )
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("metadata_json is invalid") from error
    if not isinstance(metadata, dict) or _canonical_json(metadata) != metadata_json:
        raise ValueError("metadata_json must be a canonical object")

    normalizer = HealthyPrefixNormalizer(
        arrays["normalizer_center"], arrays["normalizer_scale"], fit_run_count
    )
    health_model = HealthySubspaceModel(
        **{name: arrays[f"health_{name}"] for name in _HEALTH_STATE_NAMES},
        fit_run_count=fit_run_count,
    )
    fitted = PCFCycleFeatureExtractor(normalizer, health_model)
    standardizer = (
        BFeatureStandardizer(
            arrays["b_center"], arrays["b_scale"], b_fit_run_count
        )
        if has_b
        else None
    )
    return fitted, standardizer, metadata


__all__ = [
    "A_SHAPE",
    "ALLOWED_FIT_RUN_COUNTS",
    "BFeatureStandardizer",
    "B_FIT_ROW_COUNT",
    "B_FEATURE_COUNT",
    "CANDIDATE_NAME",
    "CHANNEL_FEATURE_COUNT",
    "CHANNEL_SUMMARY_NAMES",
    "FIT_HEALTHY_RUN_COUNT",
    "FEATURE_STATE_SCHEMA",
    "FEATURE_STATE_SCHEMA_VERSION",
    "FINAL_B_FIT_ROW_COUNT",
    "FINAL_REFIT_HEALTHY_RUN_COUNT",
    "HEALTH_FEATURE_COUNT",
    "HEALTH_FEATURE_NAMES",
    "HealthyPrefixNormalizer",
    "HealthySubspaceModel",
    "LOOP_FEATURE_COUNT",
    "LOOP_FEATURE_NAMES",
    "LOOP_PAIRS",
    "MONITOR_ENDPOINTS",
    "PCFCycleFeatureExtractor",
    "PCFCycleFeatures",
    "REFERENCE_STOP",
    "RFFT_BANDS",
    "RUN_LENGTH",
    "WINDOW_LENGTH",
    "extract_pcfcycle_features",
    "fit_b_feature_standardizer",
    "fit_pcfcycle_features",
    "load_pcfcycle_feature_state",
    "loop_summaries",
    "paired_channel_summaries",
    "save_pcfcycle_feature_state",
]
