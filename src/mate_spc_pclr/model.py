"""Causal PCLR successor model for 500-by-52 Tennessee Eastman runs.

The module is intentionally data-source agnostic: it accepts in-memory run
matrices and never opens a dataset.  Feature construction is NumPy-only.  The
three one-vs-healthy density-ratio heads are fitted with deterministic float64
CUDA AdamW, while a separate NumPy implementation is the inference reference.

The frozen sequence contract is:

* per-run baseline ``[0:128]`` for robust centering and scaling;
* causal VARX lags ``(1, 2, 4, 8, 16, 32)``;
* monitoring interval ``[160:500]``;
* three-point same-component Page-CUSUM persistence; and
* fault heads ``(3, 9, 15)`` versus healthy runs.

No positive hard clipping is used by the asinh transform, raw logits,
standardized logits, or Page-CUSUM state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Hashable, Iterator, Mapping, Sequence
import zipfile

import numpy as np


FEATURE_COUNT = 52
RUN_LENGTH = 500
BASELINE_START = 0
BASELINE_STOP = 128
MONITOR_START = 160
MONITOR_STOP = 500
CAUSAL_LAGS = (1, 2, 4, 8, 16, 32)
MULTISCALE_WINDOWS = (1, 4, 16, 32)
FAULT_IDS = (3, 9, 15)
HEALTHY_ID = 0
PERSISTENCE = 3
ROBUST_SCALE_FACTOR = 1.4826

# Zero-based (XMV, XMEAS) positions in the fixed 52-channel layout.
CONTROL_PAIRS = (
    (41, 1),
    (42, 2),
    (43, 0),
    (44, 3),
    (45, 4),
    (46, 9),
    (47, 13),
    (48, 16),
    (49, 18),
    (50, 20),
    (51, 21),
)

MODEL_SCHEMA = "mate_spc_pclr"
MODEL_SCHEMA_VERSION = 1
MAX_MODEL_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024 * 1024


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _readonly_float(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional float64 array")
    array.setflags(write=False)
    return array


def _readonly_int(value: Any, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer {ndim}-dimensional array")
    array = np.array(raw, dtype=np.int64, copy=True, order="C")
    array.setflags(write=False)
    return array


def _validate_run(value: Any, *, name: str = "run") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (RUN_LENGTH, FEATURE_COUNT):
        raise ValueError(f"{name} must have shape ({RUN_LENGTH}, {FEATURE_COUNT})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _named_runs(
    runs: Mapping[str, Any] | Sequence[Any],
    *,
    label: str,
) -> tuple[tuple[str, np.ndarray], ...]:
    if isinstance(runs, Mapping):
        if not runs:
            raise ValueError(f"{label} must contain at least one run")
        if any(not isinstance(name, str) or not name for name in runs):
            raise ValueError(f"{label} names must be nonempty strings")
        return tuple(
            (name, _validate_run(runs[name], name=f"{label} {name!r}"))
            for name in sorted(runs)
        )
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence) or not runs:
        raise ValueError(f"{label} must be a nonempty mapping or sequence")
    return tuple(
        (f"run_{index:06d}", _validate_run(run, name=f"{label}[{index}]"))
        for index, run in enumerate(runs)
    )


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and be strictly positive")
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class PCLRConfig:
    """Validated immutable feature-extraction contract."""

    baseline_start: int = BASELINE_START
    baseline_stop: int = BASELINE_STOP
    monitor_start: int = MONITOR_START
    monitor_stop: int = MONITOR_STOP
    causal_lags: tuple[int, ...] = CAUSAL_LAGS
    multiscale_windows: tuple[int, ...] = MULTISCALE_WINDOWS
    control_pairs: tuple[tuple[int, int], ...] = CONTROL_PAIRS
    var_ridge_alpha: float = 1.0e-2
    mad_floor_quantile: float = 0.10
    numeric_scale_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if (self.baseline_start, self.baseline_stop) != (BASELINE_START, BASELINE_STOP):
            raise ValueError("the baseline slice is frozen at [0:128]")
        if (self.monitor_start, self.monitor_stop) != (MONITOR_START, MONITOR_STOP):
            raise ValueError("the monitoring slice is frozen at [160:500]")
        if tuple(self.causal_lags) != CAUSAL_LAGS:
            raise ValueError(f"causal_lags must be exactly {CAUSAL_LAGS}")
        if tuple(self.multiscale_windows) != MULTISCALE_WINDOWS:
            raise ValueError(f"multiscale_windows must be exactly {MULTISCALE_WINDOWS}")
        pairs = tuple(tuple(int(v) for v in pair) for pair in self.control_pairs)
        if pairs != CONTROL_PAIRS:
            raise ValueError("control_pairs do not match the frozen 11-pair contract")
        object.__setattr__(self, "causal_lags", CAUSAL_LAGS)
        object.__setattr__(self, "multiscale_windows", MULTISCALE_WINDOWS)
        object.__setattr__(self, "control_pairs", CONTROL_PAIRS)
        object.__setattr__(
            self,
            "var_ridge_alpha",
            _positive_float(self.var_ridge_alpha, name="var_ridge_alpha"),
        )
        quantile = float(self.mad_floor_quantile)
        if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("mad_floor_quantile must lie in [0, 1]")
        object.__setattr__(self, "mad_floor_quantile", quantile)
        object.__setattr__(
            self,
            "numeric_scale_floor",
            _positive_float(self.numeric_scale_floor, name="numeric_scale_floor"),
        )


def pclr_feature_names() -> tuple[str, ...]:
    """Return the deterministic 548-coordinate feature mapping."""

    names: list[str] = []
    for window in MULTISCALE_WINDOWS:
        names.extend(
            f"residual_signed_mean_w{window:02d}_x{channel:02d}"
            for channel in range(FEATURE_COUNT)
        )
        names.extend(
            f"residual_rms_w{window:02d}_x{channel:02d}"
            for channel in range(FEATURE_COUNT)
        )
        for controller, response in CONTROL_PAIRS:
            prefix = f"loop_u{controller:02d}_y{response:02d}_w{window:02d}"
            names.extend(
                (
                    f"{prefix}_directed_innovation",
                    f"{prefix}_directed_innovation_rms",
                    f"{prefix}_signed_hysteresis",
                )
            )
    result = tuple(names)
    expected = len(MULTISCALE_WINDOWS) * (
        2 * FEATURE_COUNT + 3 * len(CONTROL_PAIRS)
    )
    if len(result) != expected or len(set(result)) != len(result):
        raise RuntimeError("internal PCLR feature mapping is invalid")
    return result


PCLR_FEATURE_NAMES = pclr_feature_names()
PCLR_FEATURE_COUNT = len(PCLR_FEATURE_NAMES)


def _run_location_scale(run: np.ndarray, mad_floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = run[BASELINE_START:BASELINE_STOP]
    center = np.median(baseline, axis=0)
    mad = ROBUST_SCALE_FACTOR * np.median(np.abs(baseline - center), axis=0)
    scale = np.maximum(mad, mad_floor)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise FloatingPointError("per-run robust location/scale is invalid")
    return center, scale


def _lagged_design(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stop = BASELINE_STOP
    start = max(CAUSAL_LAGS)
    target = transformed[start:stop]
    columns = [np.ones((stop - start, 1), dtype=np.float64)]
    columns.extend(transformed[start - lag : stop - lag] for lag in CAUSAL_LAGS)
    return np.concatenate(columns, axis=1), target


def _rolling_mean_full(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean aligned to its causal right endpoint."""

    if values.ndim == 1:
        matrix = values[:, None]
        squeeze = True
    elif values.ndim == 2:
        matrix = values
        squeeze = False
    else:
        raise ValueError("rolling input must be one- or two-dimensional")
    result = np.full(matrix.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(matrix)
    safe = np.where(finite, matrix, 0.0)
    cumulative = np.vstack(
        (np.zeros((1, matrix.shape[1]), dtype=np.float64), np.cumsum(safe, axis=0))
    )
    counts = np.vstack(
        (
            np.zeros((1, matrix.shape[1]), dtype=np.int64),
            np.cumsum(finite.astype(np.int64), axis=0),
        )
    )
    window_sum = cumulative[window:] - cumulative[:-window]
    window_count = counts[window:] - counts[:-window]
    valid = window_count == window
    aligned = result[window - 1 :]
    np.divide(window_sum, float(window), out=aligned, where=valid)
    return result[:, 0] if squeeze else result


@dataclass(frozen=True)
class PCLRFeaturePath:
    """Detailed causal transform for one run."""

    values: np.ndarray
    local_indices: np.ndarray
    prefix_values: np.ndarray
    prefix_local_indices: np.ndarray
    normalized: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    baseline_center: np.ndarray
    baseline_scale: np.ndarray

    def __post_init__(self) -> None:
        values = _readonly_float(self.values, ndim=2, name="values")
        indices = _readonly_int(self.local_indices, ndim=1, name="local_indices")
        prefix_values = _readonly_float(
            self.prefix_values, ndim=2, name="prefix_values"
        )
        prefix_indices = _readonly_int(
            self.prefix_local_indices, ndim=1, name="prefix_local_indices"
        )
        normalized = _readonly_float(self.normalized, ndim=2, name="normalized")
        predictions = np.array(self.predictions, dtype=np.float64, copy=True, order="C")
        residuals = np.array(self.residuals, dtype=np.float64, copy=True, order="C")
        center = _readonly_float(self.baseline_center, ndim=1, name="baseline_center")
        scale = _readonly_float(self.baseline_scale, ndim=1, name="baseline_scale")
        if values.shape != (MONITOR_STOP - MONITOR_START, PCLR_FEATURE_COUNT):
            raise ValueError("values have an invalid monitoring shape")
        if indices.shape != (MONITOR_STOP - MONITOR_START,) or not np.array_equal(
            indices, np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64)
        ):
            raise ValueError("local_indices must be exactly 160 through 499")
        if prefix_values.shape != (MONITOR_START - BASELINE_STOP, PCLR_FEATURE_COUNT):
            raise ValueError("prefix_values have an invalid [128:160] shape")
        if prefix_indices.shape != (MONITOR_START - BASELINE_STOP,) or not np.array_equal(
            prefix_indices, np.arange(BASELINE_STOP, MONITOR_START, dtype=np.int64)
        ):
            raise ValueError("prefix_local_indices must be exactly 128 through 159")
        if normalized.shape != (RUN_LENGTH, FEATURE_COUNT):
            raise ValueError("normalized must have shape (500, 52)")
        if predictions.shape != normalized.shape or residuals.shape != normalized.shape:
            raise ValueError("predictions and residuals must have shape (500, 52)")
        valid_start = max(CAUSAL_LAGS)
        if not np.isnan(predictions[:valid_start]).all() or not np.isnan(
            residuals[:valid_start]
        ).all():
            raise ValueError("prediction/residual warm-up must be NaN")
        if not np.isfinite(predictions[valid_start:]).all() or not np.isfinite(
            residuals[valid_start:]
        ).all():
            raise ValueError("prediction/residual path must be finite after warm-up")
        if center.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
            raise ValueError("baseline center/scale must have shape (52,)")
        if np.any(scale <= 0.0):
            raise ValueError("baseline_scale must be strictly positive")
        predictions.setflags(write=False)
        residuals.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "local_indices", indices)
        object.__setattr__(self, "prefix_values", prefix_values)
        object.__setattr__(self, "prefix_local_indices", prefix_indices)
        object.__setattr__(self, "normalized", normalized)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "baseline_center", center)
        object.__setattr__(self, "baseline_scale", scale)


@dataclass(frozen=True)
class PCLRFeatureModel:
    """Learned MAD floors and closed-form causal ridge-VARX predictor."""

    mad_floor: np.ndarray
    var_coefficients: np.ndarray
    config: PCLRConfig = PCLRConfig()

    def __post_init__(self) -> None:
        floor = _readonly_float(self.mad_floor, ndim=1, name="mad_floor")
        coefficients = _readonly_float(
            self.var_coefficients, ndim=2, name="var_coefficients"
        )
        if floor.shape != (FEATURE_COUNT,) or np.any(floor <= 0.0):
            raise ValueError("mad_floor must be a strictly positive length-52 vector")
        expected_rows = 1 + FEATURE_COUNT * len(CAUSAL_LAGS)
        if coefficients.shape != (expected_rows, FEATURE_COUNT):
            raise ValueError(
                f"var_coefficients must have shape ({expected_rows}, {FEATURE_COUNT})"
            )
        if not isinstance(self.config, PCLRConfig):
            raise TypeError("config must be a PCLRConfig")
        object.__setattr__(self, "mad_floor", floor)
        object.__setattr__(self, "var_coefficients", coefficients)

    @classmethod
    def fit(
        cls,
        development_runs: Mapping[str, Any] | Sequence[Any],
        *,
        config: PCLRConfig | None = None,
    ) -> "PCLRFeatureModel":
        """Fit floors and the VARX using only each development run's baseline."""

        actual_config = PCLRConfig() if config is None else config
        if not isinstance(actual_config, PCLRConfig):
            raise TypeError("config must be a PCLRConfig")
        records = _named_runs(development_runs, label="development_runs")
        baselines = np.stack(
            [run[BASELINE_START:BASELINE_STOP] for _, run in records], axis=0
        )
        centers = np.median(baselines, axis=1)
        run_mads = ROBUST_SCALE_FACTOR * np.median(
            np.abs(baselines - centers[:, None, :]), axis=1
        )
        learned_floor = np.quantile(
            run_mads, actual_config.mad_floor_quantile, axis=0
        )
        # A unit-aware numerical safeguard is needed only when a channel is
        # constant in at least the requested quantile of all baselines.
        typical_magnitude = np.median(np.abs(baselines), axis=(0, 1))
        numerical = np.maximum(
            actual_config.numeric_scale_floor,
            np.sqrt(np.finfo(np.float64).eps) * np.maximum(typical_magnitude, 1.0),
        )
        learned_floor = np.maximum(learned_floor, numerical)

        gram_size = 1 + FEATURE_COUNT * len(CAUSAL_LAGS)
        gram = np.zeros((gram_size, gram_size), dtype=np.float64)
        right = np.zeros((gram_size, FEATURE_COUNT), dtype=np.float64)
        for _, run in records:
            center, scale = _run_location_scale(run, learned_floor)
            transformed = np.arcsinh((run - center) / scale)
            design, target = _lagged_design(transformed)
            # Normalizing inside each run makes the closed-form objective an
            # equal mean over runs and an equal mean over baseline time points.
            weight = 1.0 / math.sqrt(float(design.shape[0] * len(records)))
            weighted_design = design * weight
            weighted_target = target * weight
            gram += weighted_design.T @ weighted_design
            right += weighted_design.T @ weighted_target
        penalty = np.eye(gram_size, dtype=np.float64) * actual_config.var_ridge_alpha
        penalty[0, 0] = 0.0  # never penalize the intercept
        coefficients = np.linalg.solve(gram + penalty, right)
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("closed-form VARX coefficients are non-finite")
        return cls(learned_floor, coefficients, actual_config)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return PCLR_FEATURE_NAMES

    @property
    def feature_count(self) -> int:
        return PCLR_FEATURE_COUNT

    def normalize_run(self, run: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = _validate_run(run)
        center, scale = _run_location_scale(matrix, self.mad_floor)
        # arcsinh is robust but unbounded; deliberately do not hard-clip it.
        transformed = np.arcsinh((matrix - center) / scale)
        if not np.isfinite(transformed).all():
            raise FloatingPointError("asinh normalization produced non-finite values")
        return transformed, center, scale

    def transform_run_detailed(self, run: Any) -> PCLRFeaturePath:
        normalized, center, scale = self.normalize_run(run)
        predictions = np.full_like(normalized, np.nan)
        residuals = np.full_like(normalized, np.nan)
        start = max(CAUSAL_LAGS)
        design_columns = [np.ones((RUN_LENGTH - start, 1), dtype=np.float64)]
        design_columns.extend(
            normalized[start - lag : RUN_LENGTH - lag] for lag in CAUSAL_LAGS
        )
        design = np.concatenate(design_columns, axis=1)
        predictions[start:] = design @ self.var_coefficients
        residuals[start:] = normalized[start:] - predictions[start:]

        differences = np.full_like(normalized, np.nan)
        differences[1:] = normalized[1:] - normalized[:-1]
        columns: list[np.ndarray] = []
        causal_feature_interval = slice(BASELINE_STOP, MONITOR_STOP)
        for window in MULTISCALE_WINDOWS:
            signed = _rolling_mean_full(residuals, window)
            magnitude = np.sqrt(_rolling_mean_full(residuals * residuals, window))
            columns.extend(
                (
                    signed[causal_feature_interval],
                    magnitude[causal_feature_interval],
                )
            )
            loop_columns: list[np.ndarray] = []
            for controller, response in CONTROL_PAIRS:
                directed = differences[:, controller] * residuals[:, response]
                hysteresis = np.full(RUN_LENGTH, np.nan, dtype=np.float64)
                hysteresis[1:] = (
                    normalized[:-1, controller] * normalized[1:, response]
                    - normalized[1:, controller] * normalized[:-1, response]
                )
                loop_columns.extend(
                    (
                        _rolling_mean_full(directed, window)[causal_feature_interval, None],
                        np.sqrt(_rolling_mean_full(directed * directed, window))[
                            causal_feature_interval, None
                        ],
                        _rolling_mean_full(hysteresis, window)[
                            causal_feature_interval, None
                        ],
                    )
                )
            columns.append(np.concatenate(loop_columns, axis=1))
        causal_features = np.concatenate(columns, axis=1)
        expected_causal_shape = (MONITOR_STOP - BASELINE_STOP, PCLR_FEATURE_COUNT)
        if causal_features.shape != expected_causal_shape:
            raise RuntimeError("internal PCLR feature shape mismatch")
        if not np.isfinite(causal_features).all():
            raise FloatingPointError("PCLR monitoring features are non-finite")
        prefix_count = MONITOR_START - BASELINE_STOP
        prefix_features = causal_features[:prefix_count]
        features = causal_features[prefix_count:]
        return PCLRFeaturePath(
            values=features,
            local_indices=np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64),
            prefix_values=prefix_features,
            prefix_local_indices=np.arange(
                BASELINE_STOP, MONITOR_START, dtype=np.int64
            ),
            normalized=normalized,
            predictions=predictions,
            residuals=residuals,
            baseline_center=center,
            baseline_scale=scale,
        )

    def transform_run(self, run: Any) -> np.ndarray:
        """Return only the 340-by-548 monitoring feature matrix."""

        return self.transform_run_detailed(run).values.copy()

    def transform_runs(
        self, runs: Mapping[str, Any] | Sequence[Any]
    ) -> dict[str, np.ndarray]:
        return {
            name: self.transform_run(run)
            for name, run in _named_runs(runs, label="runs")
        }

    def transform_training_runs(
        self, runs: Mapping[str, Any] | Sequence[Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return monitoring and true healthy-prefix features for training."""

        monitoring: dict[str, np.ndarray] = {}
        prefixes: dict[str, np.ndarray] = {}
        for name, run in _named_runs(runs, label="runs"):
            path = self.transform_run_detailed(run)
            monitoring[name] = path.values.copy()
            prefixes[name] = path.prefix_values.copy()
        return monitoring, prefixes


def _head_vector(value: Any, *, name: str, positive: bool) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        raw = np.repeat(raw.reshape(1), len(FAULT_IDS))
    if raw.shape != (len(FAULT_IDS),) or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite scalar or length-3 vector")
    if positive and np.any(raw <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not positive and np.any(raw < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.array(raw, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class PCLRHeads:
    """Three unbounded density-ratio logits and healthy-only normalization."""

    weights: np.ndarray
    bias: np.ndarray
    healthy_logit_center: np.ndarray
    healthy_logit_scale: np.ndarray
    kappa: np.ndarray
    fault_ids: tuple[int, int, int] = FAULT_IDS
    threshold: float | None = None

    def __post_init__(self) -> None:
        weights = _readonly_float(self.weights, ndim=2, name="weights")
        bias = _readonly_float(self.bias, ndim=1, name="bias")
        center = _readonly_float(
            self.healthy_logit_center, ndim=1, name="healthy_logit_center"
        )
        scale = _readonly_float(
            self.healthy_logit_scale, ndim=1, name="healthy_logit_scale"
        )
        if weights.shape[0] != len(FAULT_IDS) or weights.shape[1] < 1:
            raise ValueError("weights must have shape (3, positive_feature_count)")
        if bias.shape != (3,) or center.shape != (3,) or scale.shape != (3,):
            raise ValueError("bias and healthy normalization must have shape (3,)")
        if np.any(scale <= 0.0):
            raise ValueError("healthy_logit_scale must be strictly positive")
        kappa = _head_vector(self.kappa, name="kappa", positive=False)
        kappa.setflags(write=False)
        fault_ids = tuple(int(value) for value in self.fault_ids)
        if fault_ids != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        threshold = self.threshold
        if threshold is not None:
            threshold = _nonnegative_float(threshold, name="threshold")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "healthy_logit_center", center)
        object.__setattr__(self, "healthy_logit_scale", scale)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "fault_ids", fault_ids)
        object.__setattr__(self, "threshold", threshold)

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[1])

    def with_kappa(self, value: Any) -> "PCLRHeads":
        return replace(self, kappa=_head_vector(value, name="kappa", positive=False))

    def with_threshold(self, value: float | None) -> "PCLRHeads":
        return replace(self, threshold=value)

    def logits_numpy(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError(f"features must have shape (time, {self.feature_count})")
        if not np.isfinite(matrix).all():
            raise ValueError("features must contain only finite values")
        # This affine map is deliberately not clipped on its positive side.
        logits = matrix @ self.weights.T + self.bias
        if not np.isfinite(logits).all():
            raise FloatingPointError("raw density-ratio logits are non-finite")
        return np.asarray(logits, dtype=np.float64)

    def standardized_logits_numpy(self, features: Any) -> np.ndarray:
        raw = self.logits_numpy(features)
        result = (raw - self.healthy_logit_center) / self.healthy_logit_scale
        if not np.isfinite(result).all():
            raise FloatingPointError("standardized density-ratio logits are non-finite")
        return np.asarray(result, dtype=np.float64)


def logits_torch_float64(
    features: Any,
    heads: PCLRHeads,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Independent PyTorch float64 affine scoring path for parity audits."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != heads.feature_count:
        raise ValueError(f"features must have shape (time, {heads.feature_count})")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required for the torch scoring path") from error
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")
    tensor = torch.as_tensor(matrix, dtype=torch.float64, device=target)
    # The dataclass owns read-only NumPy arrays; copy before exposing storage to
    # PyTorch so no undefined-write warning can mask the parity audit.
    weights = torch.as_tensor(
        np.array(heads.weights, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    bias = torch.as_tensor(
        np.array(heads.bias, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    with torch.no_grad():
        result = tensor @ weights.T + bias
    return np.asarray(result.cpu().numpy(), dtype=np.float64)


def page_cusum_torch_float64(
    raw_logits: Any,
    heads: PCLRHeads,
    *,
    local_indices: Any | None = None,
    threshold: float | None = None,
    device: str = "cuda",
) -> PCLRMonitoringPath:
    """Independent torch-float64 Page path used by parity audits.

    The recurrence remains on ``device`` through standardization, the Page
    reset, the same-component three-point persistence minimum, and the strict
    threshold comparison.  Arrays are copied to the host only to construct the
    common immutable result object.
    """

    raw = np.asarray(raw_logits, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(FAULT_IDS) or not np.isfinite(raw).all():
        raise ValueError("raw_logits must be a finite matrix with shape (time, 3)")
    length = int(raw.shape[0])
    if local_indices is None:
        indices = np.arange(length, dtype=np.int64)
    else:
        indices = np.asarray(local_indices)
        if indices.shape != (length,) or indices.dtype.kind not in "iu":
            raise ValueError("local_indices must be an integer vector matching logits")
        indices = np.array(indices, dtype=np.int64, copy=True)
    actual_threshold = heads.threshold if threshold is None else threshold
    if actual_threshold is not None:
        actual_threshold = _nonnegative_float(actual_threshold, name="threshold")

    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required for the torch scoring path") from error
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")

    raw_tensor = torch.as_tensor(raw, dtype=torch.float64, device=target)
    center = torch.as_tensor(
        np.array(heads.healthy_logit_center, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    scale = torch.as_tensor(
        np.array(heads.healthy_logit_scale, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    kappa = torch.as_tensor(
        np.array(heads.kappa, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    with torch.no_grad(), _cuda_numeric_contract(torch):
        standardized_tensor = (raw_tensor - center) / scale
        state = torch.zeros(len(FAULT_IDS), dtype=torch.float64, device=target)
        states: list[Any] = []
        zero = torch.zeros_like(state)
        for time in range(length):
            # The lower reset is the Page recursion, not a positive-side clip.
            state = torch.maximum(zero, state + standardized_tensor[time] - kappa)
            states.append(state)
        if states:
            cusum_tensor = torch.stack(states, dim=0)
        else:
            cusum_tensor = torch.empty(
                (0, len(FAULT_IDS)), dtype=torch.float64, device=target
            )
        persistent_tensor = torch.zeros_like(cusum_tensor)
        for time in range(PERSISTENCE - 1, length):
            persistent_tensor[time] = torch.amin(
                cusum_tensor[time - PERSISTENCE + 1 : time + 1], dim=0
            )
        envelope_tensor = torch.amax(persistent_tensor, dim=1)
        eligible_tensor = torch.arange(length, device=target) >= PERSISTENCE - 1
        if actual_threshold is None:
            alarm_tensor = torch.zeros(length, dtype=torch.bool, device=target)
        else:
            alarm_tensor = eligible_tensor & (envelope_tensor > actual_threshold)
        previous_tensor = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=target),
                alarm_tensor[:-1],
            )
        )
        episode_tensor = alarm_tensor & ~previous_tensor

    def host(value: Any, *, dtype: Any) -> np.ndarray:
        return np.asarray(value.detach().cpu().numpy(), dtype=dtype)

    return PCLRMonitoringPath(
        local_indices=indices,
        raw_logits=raw,
        standardized_logits=host(standardized_tensor, dtype=np.float64),
        cusum=host(cusum_tensor, dtype=np.float64),
        component_persistence=host(persistent_tensor, dtype=np.float64),
        envelope=host(envelope_tensor, dtype=np.float64),
        eligible=host(eligible_tensor, dtype=bool),
        alarm=host(alarm_tensor, dtype=bool),
        episode_start=host(episode_tensor, dtype=bool),
        threshold=actual_threshold,
    )


def assert_numpy_torch_logit_parity(
    features: Any,
    heads: PCLRHeads,
    *,
    device: str = "cuda",
    atol: float = 1.0e-10,
) -> float:
    """Raise unless the full independent NumPy/torch monitor paths agree.

    The historical name is retained for API compatibility, but this check is
    deliberately stronger than affine-logit parity: it also compares healthy
    standardization, the Page path, persistence envelope, eligibility, strict
    alarms, and episode starts.
    """

    tolerance = _nonnegative_float(atol, name="atol")
    reference_logits = heads.logits_numpy(features)
    candidate_logits = logits_torch_float64(features, heads, device=device)
    reference = page_cusum_numpy(reference_logits, heads)
    candidate = page_cusum_torch_float64(candidate_logits, heads, device=device)
    maximum = 0.0
    for name in (
        "raw_logits",
        "standardized_logits",
        "cusum",
        "component_persistence",
        "envelope",
    ):
        difference = np.asarray(getattr(reference, name)) - np.asarray(
            getattr(candidate, name)
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(difference), initial=0.0)),
        )
    if maximum > tolerance:
        raise AssertionError(
            f"NumPy/torch full-monitor parity failed: {maximum:.3e} > {tolerance:.3e}"
        )
    for name in ("local_indices", "eligible", "alarm", "episode_start"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise AssertionError(f"NumPy/torch {name} parity failed")
    if reference.threshold != candidate.threshold:
        raise AssertionError("NumPy/torch threshold parity failed")
    return maximum


@dataclass(frozen=True)
class PCLRMonitoringPath:
    """Unclipped three-head Page-CUSUM path for one monitoring interval."""

    local_indices: np.ndarray
    raw_logits: np.ndarray
    standardized_logits: np.ndarray
    cusum: np.ndarray
    component_persistence: np.ndarray
    envelope: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    episode_start: np.ndarray
    threshold: float | None

    def __post_init__(self) -> None:
        indices = _readonly_int(self.local_indices, ndim=1, name="local_indices")
        raw = _readonly_float(self.raw_logits, ndim=2, name="raw_logits")
        standardized = _readonly_float(
            self.standardized_logits, ndim=2, name="standardized_logits"
        )
        cusum = _readonly_float(self.cusum, ndim=2, name="cusum")
        persistent = _readonly_float(
            self.component_persistence, ndim=2, name="component_persistence"
        )
        envelope = _readonly_float(self.envelope, ndim=1, name="envelope")
        length = indices.size
        if raw.shape != (length, 3) or standardized.shape != raw.shape:
            raise ValueError("logit paths must have shape (time, 3)")
        if cusum.shape != raw.shape or persistent.shape != raw.shape:
            raise ValueError("CUSUM paths must have shape (time, 3)")
        if envelope.shape != (length,):
            raise ValueError("envelope must have shape (time,)")
        if length > 1 and np.any(np.diff(indices) != 1):
            raise ValueError("local_indices must be consecutive")
        boolean_arrays: dict[str, np.ndarray] = {}
        for name in ("eligible", "alarm", "episode_start"):
            value = np.asarray(getattr(self, name))
            if value.shape != (length,) or value.dtype.kind != "b":
                raise ValueError(f"{name} must be a boolean vector")
            copied = np.array(value, dtype=bool, copy=True)
            copied.setflags(write=False)
            boolean_arrays[name] = copied
        if np.any(cusum < 0.0) or np.any(persistent < 0.0):
            raise ValueError("CUSUM paths must be nonnegative")
        object.__setattr__(self, "local_indices", indices)
        object.__setattr__(self, "raw_logits", raw)
        object.__setattr__(self, "standardized_logits", standardized)
        object.__setattr__(self, "cusum", cusum)
        object.__setattr__(self, "component_persistence", persistent)
        object.__setattr__(self, "envelope", envelope)
        for name, value in boolean_arrays.items():
            object.__setattr__(self, name, value)

    @property
    def run_max(self) -> float:
        return float(np.max(self.envelope, initial=0.0))

    @property
    def component_max(self) -> np.ndarray:
        return np.max(self.component_persistence, axis=0)

    @property
    def raw_logit_component_max(self) -> np.ndarray:
        return np.max(self.raw_logits, axis=0)


@dataclass(frozen=True)
class PCLRMonitoringBatch:
    """Immutable batched counterpart of :class:`PCLRMonitoringPath`."""

    local_indices: np.ndarray
    raw_logits: np.ndarray
    standardized_logits: np.ndarray
    cusum: np.ndarray
    component_persistence: np.ndarray
    envelope: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    episode_start: np.ndarray
    threshold: float | None

    def __post_init__(self) -> None:
        indices = _readonly_int(self.local_indices, ndim=2, name="local_indices")
        raw = _readonly_float(self.raw_logits, ndim=3, name="raw_logits")
        standardized = _readonly_float(
            self.standardized_logits, ndim=3, name="standardized_logits"
        )
        cusum = _readonly_float(self.cusum, ndim=3, name="cusum")
        persistent = _readonly_float(
            self.component_persistence, ndim=3, name="component_persistence"
        )
        envelope = _readonly_float(self.envelope, ndim=2, name="envelope")
        run_count, length = indices.shape
        expected = (run_count, length, len(FAULT_IDS))
        if raw.shape != expected or standardized.shape != raw.shape:
            raise ValueError("batched logit paths must have shape (run, time, 3)")
        if cusum.shape != raw.shape or persistent.shape != raw.shape:
            raise ValueError("batched CUSUM paths must have shape (run, time, 3)")
        if envelope.shape != (run_count, length):
            raise ValueError("batched envelope must have shape (run, time)")
        if length > 1 and np.any(np.diff(indices, axis=1) != 1):
            raise ValueError("each row of local_indices must be consecutive")
        boolean_arrays: dict[str, np.ndarray] = {}
        for name in ("eligible", "alarm", "episode_start"):
            value = np.asarray(getattr(self, name))
            if value.shape != (run_count, length) or value.dtype.kind != "b":
                raise ValueError(f"{name} must be a boolean (run, time) matrix")
            copied = np.array(value, dtype=bool, copy=True)
            copied.setflags(write=False)
            boolean_arrays[name] = copied
        if np.any(cusum < 0.0) or np.any(persistent < 0.0):
            raise ValueError("batched CUSUM paths must be nonnegative")
        object.__setattr__(self, "local_indices", indices)
        object.__setattr__(self, "raw_logits", raw)
        object.__setattr__(self, "standardized_logits", standardized)
        object.__setattr__(self, "cusum", cusum)
        object.__setattr__(self, "component_persistence", persistent)
        object.__setattr__(self, "envelope", envelope)
        for name, value in boolean_arrays.items():
            object.__setattr__(self, name, value)

    @property
    def run_max(self) -> np.ndarray:
        if self.envelope.shape[1] == 0:
            return np.zeros(self.envelope.shape[0], dtype=np.float64)
        return np.max(self.envelope, axis=1)

    @property
    def component_max(self) -> np.ndarray:
        if self.component_persistence.shape[1] == 0:
            return np.zeros(
                (self.component_persistence.shape[0], len(FAULT_IDS)),
                dtype=np.float64,
            )
        return np.max(self.component_persistence, axis=1)

    def run(self, index: int) -> PCLRMonitoringPath:
        """Return one run without changing any monitoring semantics."""

        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("run index must be an integer")
        selected = int(index)
        if not 0 <= selected < self.raw_logits.shape[0]:
            raise IndexError("run index is out of range")
        return PCLRMonitoringPath(
            local_indices=self.local_indices[selected],
            raw_logits=self.raw_logits[selected],
            standardized_logits=self.standardized_logits[selected],
            cusum=self.cusum[selected],
            component_persistence=self.component_persistence[selected],
            envelope=self.envelope[selected],
            eligible=self.eligible[selected],
            alarm=self.alarm[selected],
            episode_start=self.episode_start[selected],
            threshold=self.threshold,
        )


def page_cusum_numpy(
    raw_logits: Any,
    heads: PCLRHeads,
    *,
    local_indices: Any | None = None,
    threshold: float | None = None,
) -> PCLRMonitoringPath:
    """Apply healthy-normalized, unceiled Page-CUSUM with persistence three."""

    raw = np.asarray(raw_logits, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("raw_logits must be a finite matrix with shape (time, 3)")
    length = raw.shape[0]
    if local_indices is None:
        indices = np.arange(length, dtype=np.int64)
    else:
        indices = np.asarray(local_indices)
        if indices.shape != (length,) or indices.dtype.kind not in "iu":
            raise ValueError("local_indices must be an integer vector matching logits")
        indices = np.array(indices, dtype=np.int64, copy=True)
    standardized = (raw - heads.healthy_logit_center) / heads.healthy_logit_scale
    cusum = np.zeros_like(standardized)
    state = np.zeros(3, dtype=np.float64)
    for time in range(length):
        # No ceiling: arbitrarily persistent positive evidence can keep growing.
        state = np.maximum(0.0, state + standardized[time] - heads.kappa)
        cusum[time] = state
    persistent = np.zeros_like(cusum)
    for time in range(PERSISTENCE - 1, length):
        persistent[time] = np.min(cusum[time - PERSISTENCE + 1 : time + 1], axis=0)
    envelope = np.max(persistent, axis=1)
    eligible = np.arange(length) >= PERSISTENCE - 1
    actual_threshold = heads.threshold if threshold is None else threshold
    if actual_threshold is None:
        alarm = np.zeros(length, dtype=bool)
    else:
        actual_threshold = _nonnegative_float(actual_threshold, name="threshold")
        alarm = eligible & (envelope > actual_threshold)
    previous = np.concatenate((np.array([False]), alarm[:-1]))
    episode_start = alarm & ~previous
    return PCLRMonitoringPath(
        local_indices=indices,
        raw_logits=raw,
        standardized_logits=standardized,
        cusum=cusum,
        component_persistence=persistent,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm,
        episode_start=episode_start,
        threshold=actual_threshold,
    )


def monitor_batch_torch_float64(
    features: Any,
    heads: PCLRHeads,
    *,
    local_indices: Any | None = None,
    threshold: float | None = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> PCLRMonitoringBatch:
    """Score independent runs with batched torch float64 Page-CUSUM.

    ``features`` has shape ``(run, time, feature)``.  Work is chunked only
    along the independent-run dimension; the recurrence stays sequential in
    time and is vectorized over every run and all three heads in a chunk.  This
    is exactly the lower-reset Page recurrence in :func:`page_cusum_numpy` and
    introduces no positive clip or ceiling.
    """

    matrix = np.asarray(features, dtype=np.float64)
    if (
        matrix.ndim != 3
        or matrix.shape[0] < 1
        or matrix.shape[2] != heads.feature_count
    ):
        raise ValueError(
            f"features must have shape (run>=1, time, {heads.feature_count})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    size = _positive_integer(batch_size, name="batch_size")
    run_count, length, _ = matrix.shape
    if local_indices is None:
        indices = np.broadcast_to(
            np.arange(length, dtype=np.int64), (run_count, length)
        ).copy()
    else:
        raw_indices = np.asarray(local_indices)
        if raw_indices.dtype.kind not in "iu":
            raise ValueError("local_indices must contain integers")
        if raw_indices.shape == (length,):
            indices = np.broadcast_to(raw_indices, (run_count, length)).astype(
                np.int64, copy=True
            )
        elif raw_indices.shape == (run_count, length):
            indices = np.array(raw_indices, dtype=np.int64, copy=True)
        else:
            raise ValueError(
                "local_indices must have shape (time,) or (run, time)"
            )
    if length > 1 and np.any(np.diff(indices, axis=1) != 1):
        raise ValueError("each row of local_indices must be consecutive")
    actual_threshold = heads.threshold if threshold is None else threshold
    if actual_threshold is not None:
        actual_threshold = _nonnegative_float(actual_threshold, name="threshold")

    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required for the torch scoring path") from error
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")

    weights = torch.as_tensor(
        np.array(heads.weights, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    bias = torch.as_tensor(
        np.array(heads.bias, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    center = torch.as_tensor(
        np.array(heads.healthy_logit_center, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    scale = torch.as_tensor(
        np.array(heads.healthy_logit_scale, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )
    kappa = torch.as_tensor(
        np.array(heads.kappa, dtype=np.float64, copy=True),
        dtype=torch.float64,
        device=target,
    )

    component_shape = (run_count, length, len(FAULT_IDS))
    raw_host = np.empty(component_shape, dtype=np.float64)
    standardized_host = np.empty_like(raw_host)
    cusum_host = np.empty_like(raw_host)
    persistent_host = np.empty_like(raw_host)
    envelope_host = np.empty((run_count, length), dtype=np.float64)
    eligible_host = np.empty((run_count, length), dtype=bool)
    alarm_host = np.empty((run_count, length), dtype=bool)
    episode_host = np.empty((run_count, length), dtype=bool)

    def host(value: Any, *, dtype: Any) -> np.ndarray:
        return np.asarray(value.detach().cpu().numpy(), dtype=dtype)

    with torch.no_grad(), _cuda_numeric_contract(torch):
        for start in range(0, run_count, size):
            stop = min(start + size, run_count)
            tensor = torch.as_tensor(
                matrix[start:stop], dtype=torch.float64, device=target
            )
            raw_tensor = tensor @ weights.T + bias
            standardized_tensor = (raw_tensor - center) / scale
            state = torch.zeros(
                (stop - start, len(FAULT_IDS)),
                dtype=torch.float64,
                device=target,
            )
            zero = torch.zeros_like(state)
            states: list[Any] = []
            for time in range(length):
                state = torch.maximum(
                    zero, state + standardized_tensor[:, time, :] - kappa
                )
                states.append(state)
            if states:
                cusum_tensor = torch.stack(states, dim=1)
            else:
                cusum_tensor = torch.empty(
                    (stop - start, 0, len(FAULT_IDS)),
                    dtype=torch.float64,
                    device=target,
                )
            persistent_tensor = torch.zeros_like(cusum_tensor)
            if length >= PERSISTENCE:
                persistent_tensor[:, PERSISTENCE - 1 :, :] = torch.amin(
                    torch.stack(
                        [
                            cusum_tensor[:, offset : length - PERSISTENCE + offset + 1, :]
                            for offset in range(PERSISTENCE)
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
            envelope_tensor = torch.amax(persistent_tensor, dim=2)
            eligible_tensor = (
                torch.arange(length, device=target).unsqueeze(0)
                >= PERSISTENCE - 1
            ).expand(stop - start, -1)
            if actual_threshold is None:
                alarm_tensor = torch.zeros(
                    (stop - start, length), dtype=torch.bool, device=target
                )
            else:
                alarm_tensor = eligible_tensor & (
                    envelope_tensor > actual_threshold
                )
            previous_tensor = torch.cat(
                (
                    torch.zeros(
                        (stop - start, 1), dtype=torch.bool, device=target
                    ),
                    alarm_tensor[:, :-1],
                ),
                dim=1,
            )
            episode_tensor = alarm_tensor & ~previous_tensor

            raw_host[start:stop] = host(raw_tensor, dtype=np.float64)
            standardized_host[start:stop] = host(
                standardized_tensor, dtype=np.float64
            )
            cusum_host[start:stop] = host(cusum_tensor, dtype=np.float64)
            persistent_host[start:stop] = host(
                persistent_tensor, dtype=np.float64
            )
            envelope_host[start:stop] = host(envelope_tensor, dtype=np.float64)
            eligible_host[start:stop] = host(eligible_tensor, dtype=bool)
            alarm_host[start:stop] = host(alarm_tensor, dtype=bool)
            episode_host[start:stop] = host(episode_tensor, dtype=bool)

    return PCLRMonitoringBatch(
        local_indices=indices,
        raw_logits=raw_host,
        standardized_logits=standardized_host,
        cusum=cusum_host,
        component_persistence=persistent_host,
        envelope=envelope_host,
        eligible=eligible_host,
        alarm=alarm_host,
        episode_start=episode_host,
        threshold=actual_threshold,
    )


def first_episode_detection(
    alarm_or_path: Any,
    *,
    onset: int = MONITOR_START,
    local_indices: Any | None = None,
) -> int | None:
    """Return the first new alarm episode at/after onset.

    An episode that began before ``onset`` and merely continues through it does
    not count as a detection.
    """

    if isinstance(alarm_or_path, PCLRMonitoringPath):
        alarm = alarm_or_path.alarm
        indices = alarm_or_path.local_indices
    else:
        raw = np.asarray(alarm_or_path)
        if raw.ndim != 1 or raw.dtype.kind != "b":
            raise ValueError("alarm must be a one-dimensional boolean vector")
        alarm = np.array(raw, dtype=bool, copy=True)
        if local_indices is None:
            indices = np.arange(alarm.size, dtype=np.int64)
        else:
            indices = np.asarray(local_indices)
            if indices.shape != alarm.shape or indices.dtype.kind not in "iu":
                raise ValueError("local_indices must be an integer vector matching alarm")
            indices = np.array(indices, dtype=np.int64, copy=True)
    if indices.size > 1 and np.any(np.diff(indices) != 1):
        raise ValueError("local_indices must be consecutive")
    previous = np.concatenate((np.array([False]), alarm[:-1]))
    starts = alarm & ~previous & (indices >= int(onset))
    selected = indices[starts]
    return None if selected.size == 0 else int(selected[0])


@dataclass(frozen=True)
class PCLRRunDiagnostics:
    """Top-tail and maximum statistics needed for identifiability checks."""

    run_max: float
    component_max: np.ndarray
    raw_logit_component_max: np.ndarray
    component_top_tail_mean: np.ndarray
    raw_logit_top_tail_mean: np.ndarray


def _top_tail_mean(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(math.ceil(values.shape[0] * fraction)))
    partitioned = np.partition(values, values.shape[0] - count, axis=0)
    return np.mean(partitioned[-count:], axis=0)


def run_score_diagnostics(
    path: PCLRMonitoringPath,
    *,
    top_fraction: float = 0.25,
) -> PCLRRunDiagnostics:
    """Expose per-run, per-component maxima and top-tail means."""

    fraction = float(top_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    return PCLRRunDiagnostics(
        run_max=path.run_max,
        component_max=path.component_max,
        raw_logit_component_max=path.raw_logit_component_max,
        component_top_tail_mean=_top_tail_mean(path.component_persistence, fraction),
        raw_logit_top_tail_mean=_top_tail_mean(path.raw_logits, fraction),
    )


def diagnostics_by_run(
    paths: Mapping[str, PCLRMonitoringPath],
    *,
    top_fraction: float = 0.25,
) -> dict[str, PCLRRunDiagnostics]:
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("paths must be a nonempty mapping")
    return {
        name: run_score_diagnostics(paths[name], top_fraction=top_fraction)
        for name in sorted(paths)
    }


def _validate_training_features(
    features_by_run: Mapping[str, Any], labels_by_run: Mapping[str, Any]
) -> tuple[tuple[str, int, np.ndarray], ...]:
    if not isinstance(features_by_run, Mapping) or not features_by_run:
        raise ValueError("features_by_run must be a nonempty mapping")
    if not isinstance(labels_by_run, Mapping) or set(features_by_run) != set(labels_by_run):
        raise ValueError("features_by_run and labels_by_run must have identical keys")
    records: list[tuple[str, int, np.ndarray]] = []
    feature_count: int | None = None
    counts = {label: 0 for label in (HEALTHY_ID, *FAULT_IDS)}
    for name in sorted(features_by_run):
        if not isinstance(name, str) or not name:
            raise ValueError("training run names must be nonempty strings")
        raw_label = labels_by_run[name]
        if isinstance(raw_label, (bool, np.bool_)) or not isinstance(
            raw_label, (int, np.integer)
        ):
            raise ValueError(f"label for {name!r} must be an integer")
        label = int(raw_label)
        if label not in counts:
            raise ValueError(f"unsupported label {label} for {name!r}")
        matrix = np.asarray(features_by_run[name], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 1:
            raise ValueError(f"features for {name!r} must have shape (time>=4, feature>0)")
        if not np.isfinite(matrix).all():
            raise ValueError(f"features for {name!r} must be finite")
        if feature_count is None:
            feature_count = int(matrix.shape[1])
        if matrix.shape[1] != feature_count:
            raise ValueError("all training runs must share one feature count")
        counts[label] += 1
        records.append((name, label, np.array(matrix, dtype=np.float64, copy=True)))
    if any(counts[label] == 0 for label in counts):
        raise ValueError(f"training requires healthy and faults 3/9/15; counts={counts}")
    return tuple(records)


def _validate_training_prefix_features(
    prefix_features_by_run: Mapping[str, Any] | None,
    records: Sequence[tuple[str, int, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not isinstance(prefix_features_by_run, Mapping) or set(
        prefix_features_by_run
    ) != {name for name, _, _ in records}:
        raise ValueError(
            "prefix_features_by_run must exactly match all training runs when "
            "prefix_rank_weight is positive"
        )
    feature_count = int(records[0][2].shape[1])
    expected_shape = (MONITOR_START - BASELINE_STOP, feature_count)
    validated: dict[str, np.ndarray] = {}
    for name, _, _ in records:
        matrix = np.asarray(prefix_features_by_run[name], dtype=np.float64)
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(
                f"prefix features for {name!r} must be finite with shape {expected_shape}"
            )
        validated[name] = np.array(matrix, dtype=np.float64, copy=True, order="C")
    return validated


def _fit_healthy_normalization(
    features_by_run: Sequence[np.ndarray], weights: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.concatenate([matrix @ weights.T + bias for matrix in features_by_run], axis=0)
    center = np.median(pooled, axis=0)
    mad = ROBUST_SCALE_FACTOR * np.median(np.abs(pooled - center), axis=0)
    sample_sd = np.std(pooled, axis=0, ddof=1)
    fallback = ~np.isfinite(mad) | (mad <= 0.0)
    scale = np.where(fallback, sample_sd, mad)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise FloatingPointError(
            "healthy monitoring logits are degenerate; density-ratio heads are not identifiable"
        )
    return center, scale, fallback


@contextmanager
def _cuda_numeric_contract(torch: Any) -> Iterator[None]:
    previous_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn = bool(torch.backends.cudnn.allow_tf32)
    previous_deterministic = bool(torch.backends.cudnn.deterministic)
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


def _stack_training_features_torch(
    records: Sequence[tuple[str, int, np.ndarray]],
    prefix_records: Mapping[str, np.ndarray] | None,
    *,
    torch: Any,
    target: Any,
) -> dict[int, tuple[tuple[Any, Any | None], ...]]:
    """Stack equal-length runs without changing per-run loss weighting."""

    result: dict[int, tuple[tuple[Any, Any | None], ...]] = {}
    for label in (HEALTHY_ID, *FAULT_IDS):
        grouped: dict[int, list[tuple[str, np.ndarray]]] = {}
        for name, run_label, matrix in records:
            if run_label == label:
                grouped.setdefault(int(matrix.shape[0]), []).append((name, matrix))
        stacks: list[tuple[Any, Any | None]] = []
        for length in sorted(grouped):
            members = grouped[length]
            post = torch.as_tensor(
                np.stack([matrix for _, matrix in members], axis=0),
                dtype=torch.float64,
                device=target,
            )
            prefix = (
                torch.as_tensor(
                    np.stack([prefix_records[name] for name, _ in members], axis=0),
                    dtype=torch.float64,
                    device=target,
                )
                if prefix_records is not None
                else None
            )
            stacks.append((post, prefix))
        result[label] = tuple(stacks)
    return result


def _pclr_training_objective_batched(
    torch: Any,
    by_label: Mapping[int, Sequence[tuple[Any, Any | None]]],
    weights: Any,
    bias: Any,
    *,
    ridge: float,
    point_weight: float,
    bag_weight: float,
    prefix_rank_weight: float,
    rank_margin: float,
    top_fraction: float,
) -> tuple[Any, Any]:
    """Vectorized loss with the original time/run/class equal weighting."""

    logits_by_label: dict[int, list[tuple[Any, Any | None]]] = {}
    for label in (HEALTHY_ID, *FAULT_IDS):
        logits_by_label[label] = [
            (
                post @ weights.T + bias,
                prefix @ weights.T + bias if prefix is not None else None,
            )
            for post, prefix in by_label[label]
        ]

    def top_bag(logits: Any) -> Any:
        count = max(1, int(math.ceil(logits.shape[1] * top_fraction)))
        return torch.topk(
            logits, count, dim=1, largest=True, sorted=False
        ).values.mean(dim=1)

    negative_point = torch.cat(
        [
            torch.nn.functional.softplus(post).mean(dim=1)
            for post, _ in logits_by_label[HEALTHY_ID]
        ],
        dim=0,
    ).mean(dim=0)
    negative_bag = torch.cat(
        [
            torch.nn.functional.softplus(top_bag(post))
            for post, _ in logits_by_label[HEALTHY_ID]
        ],
        dim=0,
    ).mean(dim=0)

    positive_points: list[Any] = []
    positive_bags: list[Any] = []
    rankings: list[Any] = []
    for head, fault_id in enumerate(FAULT_IDS):
        groups = logits_by_label[fault_id]
        positive_points.append(
            torch.cat(
                [
                    torch.nn.functional.softplus(-post[:, :, head]).mean(dim=1)
                    for post, _ in groups
                ],
                dim=0,
            ).mean()
        )
        positive_bags.append(
            torch.cat(
                [
                    torch.nn.functional.softplus(-top_bag(post[:, :, head]))
                    for post, _ in groups
                ],
                dim=0,
            ).mean()
        )
        if prefix_rank_weight <= 0.0:
            rankings.append(
                torch.zeros((), dtype=torch.float64, device=weights.device)
            )
        else:
            ranking_groups: list[Any] = []
            for post, prefix in groups:
                if prefix is None:
                    raise RuntimeError("prefix tensor missing from ranked objective")
                post_tail = top_bag(post[:, :, head])
                prefix_mean = prefix[:, :, head].mean(dim=1)
                ranking_groups.append(
                    torch.nn.functional.softplus(
                        rank_margin - (post_tail - prefix_mean)
                    )
                )
            rankings.append(torch.cat(ranking_groups, dim=0).mean())

    positive_point = torch.stack(positive_points)
    positive_bag = torch.stack(positive_bags)
    ranking = torch.stack(rankings)
    point = 0.5 * (negative_point + positive_point)
    bag = 0.5 * (negative_bag + positive_bag)
    penalty = 0.5 * ridge * torch.sum(weights**2, dim=1)
    head_losses = (
        point_weight * point
        + bag_weight * bag
        + prefix_rank_weight * ranking
        + penalty
    )
    return head_losses.mean(), head_losses


def fit_pclr_heads_cuda(
    features_by_run: Mapping[str, Any],
    labels_by_run: Mapping[str, Any],
    *,
    prefix_features_by_run: Mapping[str, Any] | None = None,
    ridge_lambda: float = 1.0e-3,
    kappa: float | Sequence[float] | np.ndarray = 0.25,
    device: str = "cuda",
    epochs: int = 120,
    learning_rate: float = 2.0e-2,
    point_weight: float = 1.0,
    bag_weight: float = 0.25,
    prefix_rank_weight: float = 0.10,
    rank_margin: float = 0.25,
    top_fraction: float = 0.25,
    seed: int = 1729,
) -> tuple[PCLRHeads, dict[str, object]]:
    """Fit three CUDA float64 one-vs-healthy ridge-logistic heads.

    Each loss first averages time within a run, then runs within a class, and
    finally the healthy/target-fault classes.  The optional losses are a top-25%
    bag logistic term and a within-fault-run post-versus-true-prefix ranking
    term.  The prefix is the causal, certified-healthy local interval
    ``[128:160]``; the monitoring logits remain local ``[160:500]``.
    Other fault types are never silently treated as healthy negatives.
    """

    records = _validate_training_features(features_by_run, labels_by_run)
    ridge = _positive_float(ridge_lambda, name="ridge_lambda")
    epochs = _positive_integer(epochs, name="epochs")
    learning_rate = _positive_float(learning_rate, name="learning_rate")
    point_weight = _nonnegative_float(point_weight, name="point_weight")
    bag_weight = _nonnegative_float(bag_weight, name="bag_weight")
    prefix_rank_weight = _nonnegative_float(
        prefix_rank_weight, name="prefix_rank_weight"
    )
    if point_weight + bag_weight + prefix_rank_weight <= 0.0:
        raise ValueError("at least one training loss weight must be positive")
    rank_margin = _nonnegative_float(rank_margin, name="rank_margin")
    top_fraction = float(top_fraction)
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must lie in (0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    kappa_array = _head_vector(kappa, name="kappa", positive=False)
    prefix_records = (
        _validate_training_prefix_features(prefix_features_by_run, records)
        if prefix_rank_weight > 0.0
        else None
    )

    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("a CUDA-enabled PyTorch installation is required") from error
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    by_label = _stack_training_features_torch(
        records,
        prefix_records,
        torch=torch,
        target=target,
    )
    feature_count = records[0][2].shape[1]
    weights = torch.zeros(
        (len(FAULT_IDS), feature_count),
        dtype=torch.float64,
        device=target,
        requires_grad=True,
    )
    bias = torch.zeros(3, dtype=torch.float64, device=target, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weights, bias], lr=learning_rate, weight_decay=0.0, amsgrad=True
    )

    def objective() -> tuple[Any, list[Any]]:
        loss, head_losses = _pclr_training_objective_batched(
            torch,
            by_label,
            weights,
            bias,
            ridge=ridge,
            point_weight=point_weight,
            bag_weight=bag_weight,
            prefix_rank_weight=prefix_rank_weight,
            rank_margin=rank_margin,
            top_fraction=top_fraction,
        )
        return loss, list(torch.unbind(head_losses))

    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats(target)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _cuda_numeric_contract(torch):
        begin.record()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("PCLR ridge-logistic objective became non-finite")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        end.record()
        end.synchronize()
        elapsed_seconds = float(begin.elapsed_time(end)) / 1000.0
        final_loss, final_heads = objective()
    learned_weights = np.asarray(weights.detach().cpu().numpy(), dtype=np.float64)
    learned_bias = np.asarray(bias.detach().cpu().numpy(), dtype=np.float64)
    healthy_features = [
        matrix for _, label, matrix in records if label == HEALTHY_ID
    ]
    center, scale, fallback = _fit_healthy_normalization(
        healthy_features, learned_weights, learned_bias
    )
    heads = PCLRHeads(
        weights=learned_weights,
        bias=learned_bias,
        healthy_logit_center=center,
        healthy_logit_scale=scale,
        kappa=kappa_array,
    )
    properties = torch.cuda.get_device_properties(target)
    metadata: dict[str, object] = {
        "backend": "pytorch_cuda_float64_deterministic_adamw",
        "requested_device": str(device),
        "resolved_device": str(target),
        "device_name": str(torch.cuda.get_device_name(target)),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "dtype": "float64",
        "tf32": False,
        "fault_ids": list(FAULT_IDS),
        "healthy_id": HEALTHY_ID,
        "feature_count": int(feature_count),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "ridge_lambda": ridge,
        "seed": int(seed),
        "initial_objective": losses[0],
        "final_objective": float(final_loss.detach().cpu()),
        "final_head_objectives": [float(value.detach().cpu()) for value in final_heads],
        "loss_weighting": "equal_heads_equal_classes_equal_runs_equal_times",
        "cuda_training_vectorization": "label_length_stacked_batched_float64",
        "point_weight": point_weight,
        "bag_weight": bag_weight,
        "prefix_rank_weight": prefix_rank_weight,
        "prefix_rank_contract": "causal_local_128_160_mean_vs_post_160_500_top_quartile",
        "rank_margin": rank_margin,
        "top_fraction": top_fraction,
        "healthy_logit_normalization": "development_healthy_monitoring_median_1.4826MAD",
        "healthy_scale_sample_sd_fallback": [bool(value) for value in fallback],
        "cuda_elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
    }
    del by_label, weights, bias
    return heads, metadata


@dataclass(frozen=True)
class SequenceGroupSplit:
    fold: int
    training_run_names: tuple[str, ...]
    validation_run_names: tuple[str, ...]


def sequence_group_folds(
    labels_by_run: Mapping[str, Any],
    groups_by_run: Mapping[str, Hashable],
    *,
    n_splits: int,
) -> dict[str, int]:
    """Assign whole sequence groups to deterministic, label-aware folds."""

    n_splits = _positive_integer(n_splits, name="n_splits")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if not isinstance(labels_by_run, Mapping) or not labels_by_run:
        raise ValueError("labels_by_run must be a nonempty mapping")
    if not isinstance(groups_by_run, Mapping) or set(groups_by_run) != set(labels_by_run):
        raise ValueError("groups_by_run and labels_by_run must have identical keys")
    group_members: dict[Hashable, list[str]] = {}
    for name in sorted(labels_by_run):
        group = groups_by_run[name]
        try:
            hash(group)
        except TypeError as error:
            raise ValueError(f"group for {name!r} must be hashable") from error
        group_members.setdefault(group, []).append(name)
    if len(group_members) < n_splits:
        raise ValueError("the number of distinct groups must be at least n_splits")

    # Groups with the same label composition are round-robined together.  This
    # improves fold balance without ever splitting a sequence group.
    buckets: dict[tuple[tuple[int, int], ...], list[Hashable]] = {}
    expected_signature = tuple((label, 1) for label in (HEALTHY_ID, *FAULT_IDS))
    for group, names in group_members.items():
        counts: dict[int, int] = {}
        for name in names:
            raw = labels_by_run[name]
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
                raise ValueError(f"label for {name!r} must be an integer")
            label = int(raw)
            if label not in (HEALTHY_ID, *FAULT_IDS):
                raise ValueError(f"unsupported PCLR label {label} for {name!r}")
            counts[label] = counts.get(label, 0) + 1
        signature = tuple(sorted(counts.items()))
        if signature != expected_signature:
            raise ValueError(
                "each CRN group must contain exactly one healthy run and one "
                f"run for each fault {FAULT_IDS}; group {group!r} has {signature}"
            )
        buckets.setdefault(signature, []).append(group)
    group_fold: dict[Hashable, int] = {}
    offset = 0
    for signature in sorted(buckets, key=repr):
        groups = sorted(buckets[signature], key=lambda value: (type(value).__name__, repr(value)))
        for index, group in enumerate(groups):
            group_fold[group] = (offset + index) % n_splits
        offset = (offset + len(groups)) % n_splits
    return {name: group_fold[groups_by_run[name]] for name in sorted(labels_by_run)}


def sequence_group_splits(
    labels_by_run: Mapping[str, Any],
    groups_by_run: Mapping[str, Hashable],
    *,
    n_splits: int,
) -> tuple[SequenceGroupSplit, ...]:
    folds = sequence_group_folds(labels_by_run, groups_by_run, n_splits=n_splits)
    result: list[SequenceGroupSplit] = []
    for fold in range(n_splits):
        validation = tuple(name for name in sorted(folds) if folds[name] == fold)
        training = tuple(name for name in sorted(folds) if folds[name] != fold)
        training_groups = {groups_by_run[name] for name in training}
        validation_groups = {groups_by_run[name] for name in validation}
        if training_groups & validation_groups:
            raise RuntimeError("internal sequence-group leakage")
        result.append(SequenceGroupSplit(fold, training, validation))
    return tuple(result)


@dataclass(frozen=True)
class PCLRHeadFoldFit:
    ridge_lambda: float
    fold: int
    training_run_names: tuple[str, ...]
    validation_run_names: tuple[str, ...]
    heads: PCLRHeads
    metadata: dict[str, object]


def fit_pclr_heads_group_cv_cuda(
    features_by_run: Mapping[str, Any],
    labels_by_run: Mapping[str, Any],
    groups_by_run: Mapping[str, Hashable],
    *,
    ridge_lambdas: Sequence[float],
    n_splits: int,
    **fit_kwargs: Any,
) -> tuple[PCLRHeadFoldFit, ...]:
    """Deprecated precomputed-feature CV entry point.

    PCLR feature construction estimates a MAD floor and VARX coefficients.
    Consequently one global feature matrix cannot be reused across CV folds
    without leaking validation baselines into training.  Use
    :func:`fit_pclr_group_cv_cuda`, which starts from raw runs and refits the
    feature model within every training fold.
    """

    raise RuntimeError(
        "precomputed-feature PCLR CV is not fold-safe; use "
        "fit_pclr_group_cv_cuda with raw run matrices"
    )


def _validated_group_cv_inputs(
    raw_runs_by_run: Mapping[str, Any],
    labels_by_run: Mapping[str, Any],
    groups_by_run: Mapping[str, Hashable],
    *,
    n_splits: int,
) -> tuple[dict[str, np.ndarray], tuple[SequenceGroupSplit, ...]]:
    if not isinstance(raw_runs_by_run, Mapping) or not raw_runs_by_run:
        raise ValueError("raw_runs_by_run must be a nonempty mapping")
    validated_runs = dict(_named_runs(raw_runs_by_run, label="raw_runs_by_run"))
    if not isinstance(labels_by_run, Mapping) or set(labels_by_run) != set(validated_runs):
        raise ValueError("raw_runs_by_run and labels_by_run must have identical keys")
    if not isinstance(groups_by_run, Mapping) or set(groups_by_run) != set(validated_runs):
        raise ValueError("raw_runs_by_run and groups_by_run must have identical keys")
    splits = sequence_group_splits(labels_by_run, groups_by_run, n_splits=n_splits)
    return validated_runs, splits


@dataclass(frozen=True)
class PCLRModelFoldFit:
    """One fold-local feature model, fitted heads, and held-out logits."""

    ridge_lambda: float
    fold: int
    training_run_names: tuple[str, ...]
    validation_run_names: tuple[str, ...]
    feature_model: PCLRFeatureModel
    heads: PCLRHeads
    validation_raw_logits: Mapping[str, np.ndarray]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        ridge = _positive_float(self.ridge_lambda, name="ridge_lambda")
        training = tuple(self.training_run_names)
        validation = tuple(self.validation_run_names)
        if not training or not validation or set(training) & set(validation):
            raise ValueError("fold training/validation names must be nonempty and disjoint")
        if not isinstance(self.feature_model, PCLRFeatureModel):
            raise TypeError("feature_model must be a PCLRFeatureModel")
        if not isinstance(self.heads, PCLRHeads):
            raise TypeError("heads must be PCLRHeads")
        if self.heads.feature_count != self.feature_model.feature_count:
            raise ValueError("fold feature-model and head feature counts differ")
        if not isinstance(self.validation_raw_logits, Mapping) or set(
            self.validation_raw_logits
        ) != set(validation):
            raise ValueError("validation_raw_logits must exactly match validation names")
        copied_logits: dict[str, np.ndarray] = {}
        expected_shape = (MONITOR_STOP - MONITOR_START, len(FAULT_IDS))
        for name in sorted(validation):
            logits = _readonly_float(
                self.validation_raw_logits[name], ndim=2, name=f"validation logits {name!r}"
            )
            if logits.shape != expected_shape:
                raise ValueError(
                    f"validation logits for {name!r} must have shape {expected_shape}"
                )
            copied_logits[name] = logits
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "ridge_lambda", ridge)
        object.__setattr__(self, "training_run_names", training)
        object.__setattr__(self, "validation_run_names", validation)
        object.__setattr__(self, "validation_raw_logits", copied_logits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def model(self) -> "PCLRModel":
        return PCLRModel(self.feature_model, self.heads)


def fit_pclr_group_cv_cuda(
    raw_runs_by_run: Mapping[str, Any],
    labels_by_run: Mapping[str, Any],
    groups_by_run: Mapping[str, Hashable],
    *,
    ridge_lambdas: Sequence[float],
    n_splits: int = 5,
    feature_config: PCLRConfig | None = None,
    **fit_kwargs: Any,
) -> tuple[PCLRModelFoldFit, ...]:
    """Fit fold-safe PCLR candidates directly from raw 500-by-52 runs.

    For each CRN fold, the MAD floors and VARX coefficients are fitted from
    training groups only.  The density-ratio heads and their healthy
    center/MAD are then fitted from those same training runs only.  Validation
    runs are transformed only after both fitted stages are fixed.  Held-out
    raw logits are returned for candidate evaluation without retaining the
    much larger 548-coordinate feature matrices.
    """

    if "ridge_lambda" in fit_kwargs:
        raise ValueError("pass ridge candidates only through ridge_lambdas")
    if isinstance(ridge_lambdas, (str, bytes)) or not isinstance(
        ridge_lambdas, Sequence
    ):
        raise TypeError("ridge_lambdas must be a nonempty sequence")
    lambdas = tuple(_positive_float(value, name="ridge lambda") for value in ridge_lambdas)
    if not lambdas or len(lambdas) != len(set(lambdas)):
        raise ValueError("ridge_lambdas must be nonempty and unique")
    actual_config = PCLRConfig() if feature_config is None else feature_config
    if not isinstance(actual_config, PCLRConfig):
        raise TypeError("feature_config must be a PCLRConfig")
    validated_runs, splits = _validated_group_cv_inputs(
        raw_runs_by_run,
        labels_by_run,
        groups_by_run,
        n_splits=n_splits,
    )

    results: list[PCLRModelFoldFit] = []
    for split in splits:
        training_runs = {
            name: validated_runs[name] for name in split.training_run_names
        }
        validation_runs = {
            name: validated_runs[name] for name in split.validation_run_names
        }
        feature_model = PCLRFeatureModel.fit(training_runs, config=actual_config)
        training_features, training_prefix_features = (
            feature_model.transform_training_runs(training_runs)
        )
        validation_features = feature_model.transform_runs(validation_runs)
        training_labels = {
            name: labels_by_run[name] for name in split.training_run_names
        }
        training_groups = {groups_by_run[name] for name in split.training_run_names}
        validation_groups = {groups_by_run[name] for name in split.validation_run_names}
        if training_groups & validation_groups:
            raise RuntimeError("internal CRN group leakage after fold preparation")
        healthy_fit_names = tuple(
            name
            for name in split.training_run_names
            if int(labels_by_run[name]) == HEALTHY_ID
        )
        for ridge in lambdas:
            heads, head_metadata = fit_pclr_heads_cuda(
                training_features,
                training_labels,
                prefix_features_by_run=training_prefix_features,
                ridge_lambda=ridge,
                **fit_kwargs,
            )
            validation_logits = {
                name: heads.logits_numpy(validation_features[name])
                for name in split.validation_run_names
            }
            metadata = dict(head_metadata)
            metadata.update(
                {
                    "fold_safe_preprocessing": True,
                    "feature_fit_run_names": list(split.training_run_names),
                    "head_fit_run_names": list(split.training_run_names),
                    "healthy_normalization_fit_run_names": list(healthy_fit_names),
                    "validation_transform_run_names": list(split.validation_run_names),
                    "training_group_count": len(training_groups),
                    "validation_group_count": len(validation_groups),
                }
            )
            results.append(
                PCLRModelFoldFit(
                    ridge_lambda=ridge,
                    fold=split.fold,
                    training_run_names=split.training_run_names,
                    validation_run_names=split.validation_run_names,
                    feature_model=feature_model,
                    heads=heads,
                    validation_raw_logits=validation_logits,
                    metadata=metadata,
                )
            )
        del training_features, training_prefix_features, validation_features
    return tuple(results)


@dataclass(frozen=True)
class PCLRModel:
    """Complete feature, density-ratio, and monitoring model."""

    feature_model: PCLRFeatureModel
    heads: PCLRHeads

    def __post_init__(self) -> None:
        if not isinstance(self.feature_model, PCLRFeatureModel):
            raise TypeError("feature_model must be a PCLRFeatureModel")
        if not isinstance(self.heads, PCLRHeads):
            raise TypeError("heads must be PCLRHeads")
        if self.heads.feature_count != self.feature_model.feature_count:
            raise ValueError("head and feature-model feature counts differ")

    def score_run_numpy(self, run: Any) -> PCLRMonitoringPath:
        path = self.feature_model.transform_run_detailed(run)
        logits = self.heads.logits_numpy(path.values)
        return page_cusum_numpy(logits, self.heads, local_indices=path.local_indices)

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_pclr_model(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "PCLRModel":
        model, _ = load_pclr_model(path)
        return model


def _atomic_npz(path: str | os.PathLike[str], payload: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def save_pclr_model(
    model: PCLRModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically serialize numeric arrays and canonical JSON without pickle."""

    if not isinstance(model, PCLRModel):
        raise TypeError("model must be a PCLRModel")
    metadata_dict = {} if metadata is None else dict(metadata)
    threshold_present = model.heads.threshold is not None
    config_payload = asdict(model.feature_model.config)
    payload = {
        "schema": np.array(MODEL_SCHEMA, dtype=np.str_),
        "schema_version": np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
        "config_json": np.array(_canonical_json(config_payload), dtype=np.str_),
        "metadata_json": np.array(_canonical_json(metadata_dict), dtype=np.str_),
        "mad_floor": np.asarray(model.feature_model.mad_floor, dtype=np.float64),
        "var_coefficients": np.asarray(
            model.feature_model.var_coefficients, dtype=np.float64
        ),
        "weights": np.asarray(model.heads.weights, dtype=np.float64),
        "bias": np.asarray(model.heads.bias, dtype=np.float64),
        "healthy_logit_center": np.asarray(
            model.heads.healthy_logit_center, dtype=np.float64
        ),
        "healthy_logit_scale": np.asarray(
            model.heads.healthy_logit_scale, dtype=np.float64
        ),
        "kappa": np.asarray(model.heads.kappa, dtype=np.float64),
        "fault_ids": np.asarray(model.heads.fault_ids, dtype=np.int64),
        "threshold_present": np.array(int(threshold_present), dtype=np.int64),
        "threshold": np.array(
            0.0 if model.heads.threshold is None else model.heads.threshold,
            dtype=np.float64,
        ),
    }
    return _atomic_npz(path, payload)


def load_pclr_model(
    path: str | os.PathLike[str],
) -> tuple[PCLRModel, dict[str, Any]]:
    """Load a strict PCLR NPZ with ``allow_pickle=False``."""

    expected = {
        "schema",
        "schema_version",
        "config_json",
        "metadata_json",
        "mad_floor",
        "var_coefficients",
        "weights",
        "bias",
        "healthy_logit_center",
        "healthy_logit_scale",
        "kappa",
        "fault_ids",
        "threshold_present",
        "threshold",
    }
    archive_path = Path(path)
    expected_members = {f"{name}.npy" for name in expected}
    try:
        with zipfile.ZipFile(archive_path, mode="r") as zipped:
            infos = zipped.infolist()
            member_names = [info.filename for info in infos]
            if (
                len(member_names) != len(expected_members)
                or set(member_names) != expected_members
            ):
                raise ValueError("PCLR ZIP members do not match the schema exactly")
            if any(info.flag_bits & 0x1 for info in infos):
                raise ValueError("encrypted PCLR archives are not supported")
            if sum(info.file_size for info in infos) > MAX_MODEL_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError("PCLR archive exceeds the uncompressed size limit")
    except zipfile.BadZipFile as error:
        raise ValueError("PCLR model must be a valid NPZ archive") from error

    with np.load(archive_path, allow_pickle=False) as archive:
        if len(archive.files) != len(expected) or set(archive.files) != expected:
            raise ValueError("PCLR archive fields do not match the schema")
        arrays = {name: archive[name] for name in expected}
        scalar_text = ((), "U")
        required_specs = {
            "schema": scalar_text,
            "schema_version": ((), np.dtype(np.int64)),
            "config_json": scalar_text,
            "metadata_json": scalar_text,
            "mad_floor": ((FEATURE_COUNT,), np.dtype(np.float64)),
            "var_coefficients": (
                (1 + FEATURE_COUNT * len(CAUSAL_LAGS), FEATURE_COUNT),
                np.dtype(np.float64),
            ),
            "weights": ((len(FAULT_IDS), PCLR_FEATURE_COUNT), np.dtype(np.float64)),
            "bias": ((len(FAULT_IDS),), np.dtype(np.float64)),
            "healthy_logit_center": ((len(FAULT_IDS),), np.dtype(np.float64)),
            "healthy_logit_scale": ((len(FAULT_IDS),), np.dtype(np.float64)),
            "kappa": ((len(FAULT_IDS),), np.dtype(np.float64)),
            "fault_ids": ((len(FAULT_IDS),), np.dtype(np.int64)),
            "threshold_present": ((), np.dtype(np.int64)),
            "threshold": ((), np.dtype(np.float64)),
        }
        for name, (shape, dtype_contract) in required_specs.items():
            array = arrays[name]
            dtype_ok = (
                array.dtype.kind == dtype_contract
                if isinstance(dtype_contract, str)
                else array.dtype == dtype_contract
            )
            if array.shape != shape or not dtype_ok:
                raise ValueError(
                    f"PCLR archive field {name!r} has an invalid shape or dtype"
                )
        if str(arrays["schema"].item()) != MODEL_SCHEMA:
            raise ValueError("PCLR archive schema is invalid")
        if int(arrays["schema_version"].item()) != MODEL_SCHEMA_VERSION:
            raise ValueError("PCLR archive schema version is unsupported")
        config_text = str(arrays["config_json"].item())
        metadata_text = str(arrays["metadata_json"].item())
        if len(config_text) > 65536 or len(metadata_text) > 1048576:
            raise ValueError("PCLR archive JSON field exceeds its size limit")
        try:
            config_raw = json.loads(config_text)
            metadata = json.loads(metadata_text)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("PCLR archive JSON is invalid") from error
        if not isinstance(config_raw, dict) or not isinstance(metadata, dict):
            raise ValueError("PCLR archive JSON fields must contain objects")
        try:
            if (
                _canonical_json(config_raw) != config_text
                or _canonical_json(metadata) != metadata_text
            ):
                raise ValueError("PCLR archive JSON fields must be canonical and finite")
        except (TypeError, ValueError) as error:
            raise ValueError("PCLR archive JSON fields must be canonical and finite") from error
        # JSON converts tuples to lists; explicitly restore the frozen nested fields.
        config_raw["causal_lags"] = tuple(config_raw["causal_lags"])
        config_raw["multiscale_windows"] = tuple(config_raw["multiscale_windows"])
        config_raw["control_pairs"] = tuple(
            tuple(pair) for pair in config_raw["control_pairs"]
        )
        config = PCLRConfig(**config_raw)
        present = int(arrays["threshold_present"].item())
        if present not in (0, 1):
            raise ValueError("threshold_present must be zero or one")
        stored_threshold = float(arrays["threshold"].item())
        if not present and stored_threshold != 0.0:
            raise ValueError("an absent threshold must use the canonical zero sentinel")
        threshold = stored_threshold if present else None
        feature_model = PCLRFeatureModel(
            mad_floor=np.array(arrays["mad_floor"], dtype=np.float64, copy=True),
            var_coefficients=np.array(
                arrays["var_coefficients"], dtype=np.float64, copy=True
            ),
            config=config,
        )
        heads = PCLRHeads(
            weights=np.array(arrays["weights"], dtype=np.float64, copy=True),
            bias=np.array(arrays["bias"], dtype=np.float64, copy=True),
            healthy_logit_center=np.array(
                arrays["healthy_logit_center"], dtype=np.float64, copy=True
            ),
            healthy_logit_scale=np.array(
                arrays["healthy_logit_scale"], dtype=np.float64, copy=True
            ),
            kappa=np.array(arrays["kappa"], dtype=np.float64, copy=True),
            fault_ids=tuple(int(value) for value in arrays["fault_ids"].tolist()),
            threshold=threshold,
        )
    return PCLRModel(feature_model, heads), metadata


__all__ = [
    "BASELINE_START",
    "BASELINE_STOP",
    "CAUSAL_LAGS",
    "CONTROL_PAIRS",
    "FAULT_IDS",
    "FEATURE_COUNT",
    "HEALTHY_ID",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "MONITOR_START",
    "MONITOR_STOP",
    "MULTISCALE_WINDOWS",
    "PCLRConfig",
    "PCLRFeatureModel",
    "PCLRFeaturePath",
    "PCLRHeadFoldFit",
    "PCLRHeads",
    "PCLRModel",
    "PCLRModelFoldFit",
    "PCLRMonitoringBatch",
    "PCLRMonitoringPath",
    "PCLRRunDiagnostics",
    "PCLR_FEATURE_COUNT",
    "PCLR_FEATURE_NAMES",
    "PERSISTENCE",
    "RUN_LENGTH",
    "SequenceGroupSplit",
    "assert_numpy_torch_logit_parity",
    "diagnostics_by_run",
    "first_episode_detection",
    "fit_pclr_heads_cuda",
    "fit_pclr_heads_group_cv_cuda",
    "fit_pclr_group_cv_cuda",
    "load_pclr_model",
    "logits_torch_float64",
    "monitor_batch_torch_float64",
    "page_cusum_numpy",
    "page_cusum_torch_float64",
    "pclr_feature_names",
    "run_score_diagnostics",
    "save_pclr_model",
    "sequence_group_folds",
    "sequence_group_splits",
]
