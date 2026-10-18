"""Fit-only preprocessing and the frozen linear head for MiniROCKET v8.

This module contains no file or dataset loading.  Every public production
entrypoint validates the exact shapes fixed by the v8 charter so that a caller
cannot silently change the experiment while reusing these routines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import random
from numbers import Integral
from typing import Any

# The pinned Windows CUDA environment requires PyTorch's DLLs to be loaded
# before NumPy. Keep this order.
try:
    import torch
except ImportError as error:  # pragma: no cover - environment preflight owns this
    raise RuntimeError("PyTorch is required for MiniROCKET v8") from error

import numpy as np


CANDIDATE_ID = "gpu-minirocket-logodds-cusum-v8-fixed"
GLOBAL_SEED = 20260912
CHANNEL_COUNT = 52
ROWS_PER_RUN = 500
FIT_RUNS_PER_CLASS = 60
FINAL_REFIT_RUNS_PER_CLASS = 100
WINDOW_LENGTH = 128
FEATURE_COUNT = 9_996
CLASS_COUNT = 4
CLASS_ORDER: tuple[str, ...] = ("H", "F3", "F9", "F15")
CLASS_LABELS: tuple[int, ...] = (0, 1, 2, 3)
CLASS_EXAMPLE_COUNTS: tuple[int, ...] = (6_540, 4_620, 4_620, 4_620)
TRAINING_EXAMPLE_COUNT = sum(CLASS_EXAMPLE_COUNTS)
FINAL_REFIT_CLASS_EXAMPLE_COUNTS: tuple[int, ...] = (10_900, 7_700, 7_700, 7_700)
FINAL_REFIT_TRAINING_EXAMPLE_COUNT = sum(FINAL_REFIT_CLASS_EXAMPLE_COUNTS)
CLASS_WEIGHTS: tuple[float, ...] = (85.0 / 109.0, 85.0 / 77.0, 85.0 / 77.0, 85.0 / 77.0)
NORMALIZATION_SCALE_FLOOR = 1e-6
FEATURE_STD_FLOOR = 1e-6
NORMALIZATION_CLIP = 10.0
L2_COEFFICIENT = 1e-4
EXTRACTION_BATCH_SIZE = 128

HEALTHY_ENDPOINTS = tuple(range(160, 497, 4))
PREFIX_ENDPOINTS = tuple(range(128, 157, 4))
MATURE_FAULT_ENDPOINTS = tuple(range(192, 497, 4))

if len(HEALTHY_ENDPOINTS) != 85:
    raise RuntimeError("v8 healthy endpoint count changed")
if len(PREFIX_ENDPOINTS) != 8:
    raise RuntimeError("v8 prefix endpoint count changed")
if len(MATURE_FAULT_ENDPOINTS) != 77:
    raise RuntimeError("v8 mature-fault endpoint count changed")
if CLASS_COUNT * FEATURE_COUNT + CLASS_COUNT != 39_988:
    raise RuntimeError("v8 linear-head parameter count changed")


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{name} must be an integer (boolean values are invalid)")
    return int(value)


def _numeric_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype.kind not in {"i", "u", "f"} or value.dtype.kind == "b":
        raise ValueError(f"{name} must have a real numeric dtype")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _owned_readonly(value: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def set_determinism(seed: int = GLOBAL_SEED) -> None:
    """Configure the fixed Python/NumPy/PyTorch deterministic state.

    ``CUBLAS_WORKSPACE_CONFIG`` is set before PyTorch is imported.  A caller
    that initialized CUDA earlier is responsible for rejecting an incompatible
    process during its preflight.
    """

    fixed_seed = _strict_int(seed, name="seed")
    if fixed_seed != GLOBAL_SEED:
        raise ValueError(f"seed must be the frozen value {GLOBAL_SEED}")
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with the v8 protocol")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment preflight owns this
        raise RuntimeError("PyTorch is required for MiniROCKET v8") from error
    torch.manual_seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True, slots=True)
class RobustNormalizer:
    """Per-channel fit-healthy median and MAD scale."""

    location: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        location = _numeric_array(
            self.location, name="location", shape=(CHANNEL_COUNT,)
        )
        scale = _numeric_array(self.scale, name="scale", shape=(CHANNEL_COUNT,))
        if location.dtype != np.dtype(np.float64) or scale.dtype != np.dtype(np.float64):
            raise ValueError("normalizer arrays must be float64")
        if np.any(scale < NORMALIZATION_SCALE_FLOOR):
            raise ValueError("normalizer scales violate the frozen floor")
        object.__setattr__(
            self, "location", _owned_readonly(location, dtype=np.dtype(np.float64))
        )
        object.__setattr__(
            self, "scale", _owned_readonly(scale, dtype=np.dtype(np.float64))
        )


def robust_fit_normalizer(fit_healthy_runs: np.ndarray) -> RobustNormalizer:
    """Fit the exact 60-run, all-row robust channel normalization."""

    return _robust_fit_normalizer(
        fit_healthy_runs, expected_run_count=FIT_RUNS_PER_CLASS
    )


def robust_fit_normalizer_final_refit(
    fit_healthy_runs: np.ndarray,
) -> RobustNormalizer:
    """Fit the post-GO 100-run robust channel normalization."""

    return _robust_fit_normalizer(
        fit_healthy_runs, expected_run_count=FINAL_REFIT_RUNS_PER_CLASS
    )


def _robust_fit_normalizer(
    fit_healthy_runs: np.ndarray, *, expected_run_count: int
) -> RobustNormalizer:

    runs = _numeric_array(
        fit_healthy_runs,
        name="fit_healthy_runs",
        shape=(expected_run_count, ROWS_PER_RUN, CHANNEL_COUNT),
    )
    values = np.asarray(runs, dtype=np.float64)
    location = np.median(values, axis=(0, 1))
    mad = np.median(np.abs(values - location[None, None, :]), axis=(0, 1))
    scale = np.maximum(1.4826 * mad, NORMALIZATION_SCALE_FLOOR)
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise FloatingPointError("robust normalization produced nonfinite state")
    return RobustNormalizer(location=location, scale=scale)


def normalize_runs(runs: np.ndarray, state: RobustNormalizer) -> np.ndarray:
    """Normalize complete 500x52 runs, clip, and cast to float32."""

    if not isinstance(state, RobustNormalizer):
        raise TypeError("state must be RobustNormalizer")
    values = _numeric_array(runs, name="runs", ndim=3)
    if values.shape[0] < 1 or values.shape[1:] != (ROWS_PER_RUN, CHANNEL_COUNT):
        raise ValueError("runs must have shape (n, 500, 52) with n >= 1")
    normalized = (
        np.asarray(values, dtype=np.float64) - state.location[None, None, :]
    ) / state.scale[None, None, :]
    np.clip(normalized, -NORMALIZATION_CLIP, NORMALIZATION_CLIP, out=normalized)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("run normalization produced nonfinite values")
    return np.asarray(normalized, dtype=np.float32, order="C")


@dataclass(frozen=True, slots=True)
class TrainingWindows:
    """Exact ordered fold-training windows and their provenance."""

    windows: np.ndarray
    labels: np.ndarray
    source_idv: np.ndarray
    simulation_run: np.ndarray
    endpoints: np.ndarray
    fit_run_ids: np.ndarray

    def __post_init__(self) -> None:
        fit_run_ids = _validate_fit_run_ids(self.fit_run_ids)
        example_count = 340 * fit_run_ids.size
        windows = _numeric_array(
            self.windows,
            name="windows",
            shape=(example_count, CHANNEL_COUNT, WINDOW_LENGTH),
        )
        if windows.dtype != np.dtype(np.float32):
            raise ValueError("training windows must be float32")
        if np.any(np.abs(windows) > NORMALIZATION_CLIP):
            raise ValueError("training windows must already be clipped to [-10,10]")
        labels = np.asarray(self.labels)
        sources = np.asarray(self.source_idv)
        runs = np.asarray(self.simulation_run)
        endpoints = np.asarray(self.endpoints)
        for name, value in (
            ("labels", labels),
            ("source_idv", sources),
            ("simulation_run", runs),
            ("endpoints", endpoints),
        ):
            if value.dtype != np.dtype(np.int64) or value.shape != (
                example_count,
            ):
                raise ValueError(f"{name} must be aligned int64 metadata")
        expected = _training_metadata(fit_run_ids)
        supplied = (labels, sources, runs, endpoints)
        if any(
            not np.array_equal(actual, frozen)
            for actual, frozen in zip(supplied, expected)
        ):
            raise ValueError(
                "training metadata violates the frozen class/source/run/endpoint order"
            )
        for name, value in (
            ("windows", windows),
            ("labels", labels),
            ("source_idv", sources),
            ("simulation_run", runs),
            ("endpoints", endpoints),
            ("fit_run_ids", fit_run_ids),
        ):
            dtype = np.dtype(np.float32) if name == "windows" else np.dtype(np.int64)
            object.__setattr__(self, name, _owned_readonly(value, dtype=dtype))


def _validate_normalized_fit_runs(
    value: np.ndarray, *, name: str, expected_run_count: int
) -> np.ndarray:
    result = _numeric_array(
        value,
        name=name,
        shape=(expected_run_count, ROWS_PER_RUN, CHANNEL_COUNT),
    )
    if result.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be normalized float32")
    if np.any(np.abs(result) > NORMALIZATION_CLIP):
        raise ValueError(f"{name} must be clipped to [-10,10]")
    return result


def _validate_fit_run_ids(
    fit_run_ids: np.ndarray, *, expected_count: int | None = None
) -> np.ndarray:
    allowed_counts = (FIT_RUNS_PER_CLASS, FINAL_REFIT_RUNS_PER_CLASS)
    count = fit_run_ids.size if isinstance(fit_run_ids, np.ndarray) and fit_run_ids.ndim == 1 else -1
    if (
        not isinstance(fit_run_ids, np.ndarray)
        or fit_run_ids.dtype != np.dtype(np.int64)
        or count not in allowed_counts
    ):
        raise ValueError("fit_run_ids must be NumPy int64 with shape (60,) or (100,)")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"fit_run_ids must contain exactly {expected_count} runs")
    if np.any((fit_run_ids < 1) | (fit_run_ids > 100)):
        raise ValueError("fit_run_ids must lie in 1..100")
    if not np.all(fit_run_ids[1:] > fit_run_ids[:-1]):
        raise ValueError("fit_run_ids must be unique and strictly ascending")
    return fit_run_ids


def _training_metadata(
    fit_run_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build provenance without allocating the 0.5-GiB window tensor."""

    run_ids = _validate_fit_run_ids(fit_run_ids)
    labels: list[int] = []
    sources: list[int] = []
    runs: list[int] = []
    endpoints: list[int] = []

    def append(source: int, label: int, selected: tuple[int, ...]) -> None:
        for run in run_ids.tolist():
            for endpoint in selected:
                labels.append(label)
                sources.append(source)
                runs.append(run)
                endpoints.append(endpoint)

    append(1, 0, HEALTHY_ENDPOINTS)
    append(3, 0, PREFIX_ENDPOINTS)
    append(9, 0, PREFIX_ENDPOINTS)
    append(15, 0, PREFIX_ENDPOINTS)
    append(3, 1, MATURE_FAULT_ENDPOINTS)
    append(9, 2, MATURE_FAULT_ENDPOINTS)
    append(15, 3, MATURE_FAULT_ENDPOINTS)
    result = tuple(
        np.asarray(value, dtype=np.int64)
        for value in (labels, sources, runs, endpoints)
    )
    if any(value.shape != (340 * run_ids.size,) for value in result):
        raise RuntimeError("v8 training metadata count changed")
    return result  # type: ignore[return-value]


def build_training_windows(
    healthy: np.ndarray,
    f3: np.ndarray,
    f9: np.ndarray,
    f15: np.ndarray,
    *,
    fit_run_ids: np.ndarray,
) -> TrainingWindows:
    """Build all 20,400 causal windows in the charter's stable order.

    Each input must contain the 60 normalized fit runs in ascending run-ID
    order. ``fit_run_ids`` binds every row to its real Mode2 simulation-run
    identity; positional 1..60 identifiers are deliberately forbidden.
    """

    return _build_training_windows(
        healthy,
        f3,
        f9,
        f15,
        fit_run_ids=fit_run_ids,
        expected_run_count=FIT_RUNS_PER_CLASS,
    )


def build_final_refit_training_windows(
    healthy: np.ndarray,
    f3: np.ndarray,
    f9: np.ndarray,
    f15: np.ndarray,
    *,
    fit_run_ids: np.ndarray,
) -> TrainingWindows:
    """Build the exact 34,000 post-GO refit windows from Run1..100."""

    run_ids = _validate_fit_run_ids(
        fit_run_ids, expected_count=FINAL_REFIT_RUNS_PER_CLASS
    )
    if not np.array_equal(run_ids, np.arange(1, 101, dtype=np.int64)):
        raise ValueError("final-refit fit_run_ids must be exactly Run1..100")
    return _build_training_windows(
        healthy,
        f3,
        f9,
        f15,
        fit_run_ids=run_ids,
        expected_run_count=FINAL_REFIT_RUNS_PER_CLASS,
    )


def _build_training_windows(
    healthy: np.ndarray,
    f3: np.ndarray,
    f9: np.ndarray,
    f15: np.ndarray,
    *,
    fit_run_ids: np.ndarray,
    expected_run_count: int,
) -> TrainingWindows:
    arrays = {
        1: _validate_normalized_fit_runs(
            healthy, name="healthy", expected_run_count=expected_run_count
        ),
        3: _validate_normalized_fit_runs(
            f3, name="f3", expected_run_count=expected_run_count
        ),
        9: _validate_normalized_fit_runs(
            f9, name="f9", expected_run_count=expected_run_count
        ),
        15: _validate_normalized_fit_runs(
            f15, name="f15", expected_run_count=expected_run_count
        ),
    }
    fit_run_ids = _validate_fit_run_ids(
        fit_run_ids, expected_count=expected_run_count
    )
    example_count = 340 * expected_run_count
    windows = np.empty(
        (example_count, CHANNEL_COUNT, WINDOW_LENGTH), dtype=np.float32
    )
    labels, sources, run_numbers, endpoints = _training_metadata(fit_run_ids)
    cursor = 0

    def append(source: int, label: int, selected_endpoints: tuple[int, ...]) -> None:
        nonlocal cursor
        values = arrays[source]
        for run_index in range(expected_run_count):
            for endpoint in selected_endpoints:
                windows[cursor] = values[
                    run_index, endpoint - WINDOW_LENGTH + 1 : endpoint + 1, :
                ].T
                if (
                    labels[cursor] != label
                    or sources[cursor] != source
                    or run_numbers[cursor] != fit_run_ids[run_index]
                    or endpoints[cursor] != endpoint
                ):
                    raise RuntimeError("training metadata/window ordering diverged")
                cursor += 1

    append(1, 0, HEALTHY_ENDPOINTS)
    append(3, 0, PREFIX_ENDPOINTS)
    append(9, 0, PREFIX_ENDPOINTS)
    append(15, 0, PREFIX_ENDPOINTS)
    append(3, 1, MATURE_FAULT_ENDPOINTS)
    append(9, 2, MATURE_FAULT_ENDPOINTS)
    append(15, 3, MATURE_FAULT_ENDPOINTS)
    if cursor != example_count:
        raise RuntimeError("v8 training-window construction count changed")
    return TrainingWindows(
        windows=windows,
        labels=labels,
        source_idv=sources,
        simulation_run=run_numbers,
        endpoints=endpoints,
        fit_run_ids=fit_run_ids,
    )


@dataclass(frozen=True, slots=True)
class FeatureStandardizer:
    """Float64 population moments fitted on all fold-training features."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = _numeric_array(self.mean, name="mean", shape=(FEATURE_COUNT,))
        scale = _numeric_array(self.scale, name="scale", shape=(FEATURE_COUNT,))
        if mean.dtype != np.dtype(np.float64) or scale.dtype != np.dtype(np.float64):
            raise ValueError("feature standardizer arrays must be float64")
        if np.any(scale < FEATURE_STD_FLOOR):
            raise ValueError("feature standardizer scales violate the frozen floor")
        object.__setattr__(self, "mean", _owned_readonly(mean, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "scale", _owned_readonly(scale, dtype=np.dtype(np.float64)))


def _validate_features(
    features: np.ndarray, *, name: str, exact_training_count: bool
) -> np.ndarray:
    values = _numeric_array(features, name=name, ndim=2)
    required_rows = TRAINING_EXAMPLE_COUNT if exact_training_count else None
    if values.shape[1:] != (FEATURE_COUNT,) or values.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n, {FEATURE_COUNT}) with n >= 1")
    if required_rows is not None and values.shape[0] != required_rows:
        raise ValueError(
            f"{name} must have exactly {TRAINING_EXAMPLE_COUNT} training rows"
        )
    return values


def fit_feature_standardizer(features: np.ndarray) -> FeatureStandardizer:
    """Fit float64 two-pass population moments without a giant float64 copy."""

    values = _validate_features(
        features, name="features", exact_training_count=True
    )
    mean, scale = _fit_feature_moments(values)
    return FeatureStandardizer(mean=mean, scale=scale)


def fit_feature_standardizer_final_refit(
    features: np.ndarray,
) -> FeatureStandardizer:
    """Fit feature moments on exactly 34,000 post-GO refit examples."""

    values = _validate_features(
        features, name="features", exact_training_count=False
    )
    if values.shape[0] != FINAL_REFIT_TRAINING_EXAMPLE_COUNT:
        raise ValueError(
            f"features must have exactly {FINAL_REFIT_TRAINING_EXAMPLE_COUNT} "
            "final-refit rows"
        )
    mean, scale = _fit_feature_moments(values)
    return FeatureStandardizer(mean=mean, scale=scale)


def _fit_feature_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generic two-pass float64 moments used by the fixed public wrapper."""

    data = _numeric_array(values, name="values", ndim=2)
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("values must have a nonempty two-dimensional shape")
    mean = np.mean(data, axis=0, dtype=np.float64)
    squared = np.zeros(data.shape[1], dtype=np.float64)
    for start in range(0, data.shape[0], 128):
        block = np.asarray(data[start : start + 128], dtype=np.float64)
        delta = block - mean[None, :]
        squared += np.sum(delta * delta, axis=0, dtype=np.float64)
    variance = squared / float(data.shape[0])
    scale = np.maximum(np.sqrt(variance), FEATURE_STD_FLOOR)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise FloatingPointError("feature standardization produced nonfinite state")
    return mean, scale


def standardize_features(
    features: np.ndarray, state: FeatureStandardizer
) -> np.ndarray:
    """Apply fit-fold-only feature state and return contiguous float32."""

    if not isinstance(state, FeatureStandardizer):
        raise TypeError("state must be FeatureStandardizer")
    values = _validate_features(
        features, name="features", exact_training_count=False
    )
    result = np.empty(values.shape, dtype=np.float32)
    for start in range(0, values.shape[0], 128):
        block = np.asarray(values[start : start + 128], dtype=np.float64)
        standardized = (block - state.mean[None, :]) / state.scale[None, :]
        if not np.isfinite(standardized).all():
            raise FloatingPointError("feature standardization produced nonfinite values")
        result[start : start + block.shape[0]] = standardized.astype(np.float32)
    return result


@dataclass(frozen=True, slots=True)
class LinearHeadState:
    """Portable frozen parameters and optimization acceptance evidence."""

    weight: np.ndarray
    bias: np.ndarray
    initial_objective: float
    final_objective: float
    final_max_abs_gradient: float
    source_device: str

    def __post_init__(self) -> None:
        weight = _numeric_array(
            self.weight, name="weight", shape=(CLASS_COUNT, FEATURE_COUNT)
        )
        bias = _numeric_array(self.bias, name="bias", shape=(CLASS_COUNT,))
        if weight.dtype != np.dtype(np.float32) or bias.dtype != np.dtype(np.float32):
            raise ValueError("linear-head arrays must be float32")
        numeric = (
            float(self.initial_objective),
            float(self.final_objective),
            float(self.final_max_abs_gradient),
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("linear-head optimization evidence must be finite")
        if numeric[1] > numeric[0] + 1e-10:
            raise ValueError("final objective exceeds the zero-initialization objective")
        if numeric[2] < 0.0:
            raise ValueError("gradient magnitude must be nonnegative")
        if not isinstance(self.source_device, str) or not self.source_device:
            raise ValueError("source_device must be a nonempty string")
        object.__setattr__(self, "weight", _owned_readonly(weight, dtype=np.dtype(np.float32)))
        object.__setattr__(self, "bias", _owned_readonly(bias, dtype=np.dtype(np.float32)))
        object.__setattr__(self, "initial_objective", numeric[0])
        object.__setattr__(self, "final_objective", numeric[1])
        object.__setattr__(self, "final_max_abs_gradient", numeric[2])


def _validate_training_labels(labels: np.ndarray) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
        raise ValueError("labels must be a NumPy int64 array")
    if labels.shape != (TRAINING_EXAMPLE_COUNT,):
        raise ValueError(f"labels must have shape ({TRAINING_EXAMPLE_COUNT},)")
    expected = np.repeat(np.arange(CLASS_COUNT, dtype=np.int64), CLASS_EXAMPLE_COUNTS)
    if not np.array_equal(labels, expected):
        raise ValueError("labels violate the frozen class ordering/counts")
    return labels


def _validate_final_refit_labels(labels: np.ndarray) -> np.ndarray:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
        raise ValueError("labels must be a NumPy int64 array")
    if labels.shape != (FINAL_REFIT_TRAINING_EXAMPLE_COUNT,):
        raise ValueError(
            f"labels must have shape ({FINAL_REFIT_TRAINING_EXAMPLE_COUNT},)"
        )
    expected = np.repeat(
        np.arange(CLASS_COUNT, dtype=np.int64), FINAL_REFIT_CLASS_EXAMPLE_COUNTS
    )
    if not np.array_equal(labels, expected):
        raise ValueError("labels violate the frozen final-refit ordering/counts")
    return labels


def _torch_linear_fit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: Any,
    enforce_frozen_shape: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, float, str]:
    """Internal optimizer implementation; toy shapes are used only by tests."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - environment preflight owns this
        raise RuntimeError("PyTorch is required for the v8 linear head") from error

    values = _numeric_array(features, name="features", ndim=2)
    if values.dtype != np.dtype(np.float32):
        raise ValueError("features must be standardized float32")
    if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
        raise ValueError("labels must be a NumPy int64 array")
    if labels.shape != (values.shape[0],):
        raise ValueError("labels must align with feature rows")
    if enforce_frozen_shape:
        _validate_features(values, name="features", exact_training_count=True)
        _validate_training_labels(labels)
    else:
        if values.shape[0] < CLASS_COUNT or values.shape[1] < 1:
            raise ValueError("test optimization input is too small")
        if np.any((labels < 0) | (labels >= CLASS_COUNT)):
            raise ValueError("labels must be in 0..3")
        if set(labels.tolist()) != set(CLASS_LABELS):
            raise ValueError("all four classes must be represented")

    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must identify CPU or CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    x = torch.as_tensor(
        np.array(values, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32, device=target)
    y = torch.as_tensor(
        np.array(labels, dtype=np.int64, order="C", copy=True),
        dtype=torch.int64, device=target)
    weight = torch.nn.Parameter(
        torch.zeros((CLASS_COUNT, values.shape[1]), dtype=torch.float32, device=target)
    )
    bias = torch.nn.Parameter(torch.zeros(CLASS_COUNT, dtype=torch.float32, device=target))
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=target)

    def objective() -> Any:
        logits = functional.linear(x, weight, bias)
        cross_entropy = functional.cross_entropy(
            logits, y, weight=class_weights, reduction="mean"
        )
        return cross_entropy + (L2_COEFFICIENT / 2.0) * torch.sum(weight * weight)

    with torch.no_grad():
        initial = float(objective().detach().cpu().item())
    if not math.isfinite(initial):
        raise FloatingPointError("zero-initialization objective is nonfinite")
    optimizer = torch.optim.LBFGS(
        (weight, bias),
        lr=1.0,
        max_iter=100,
        max_eval=125,
        history_size=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("LBFGS objective became nonfinite")
        loss.backward()
        for parameter in (weight, bias):
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError("LBFGS gradient became nonfinite")
        return loss

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_loss = objective()
    if not bool(torch.isfinite(final_loss).item()):
        raise FloatingPointError("final LBFGS objective is nonfinite")
    final_loss.backward()
    gradients = (weight.grad, bias.grad)
    if any(gradient is None for gradient in gradients):
        raise FloatingPointError("final LBFGS gradient is missing")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
        raise FloatingPointError("final LBFGS gradient is nonfinite")
    if not bool(torch.isfinite(weight).all().item()) or not bool(torch.isfinite(bias).all().item()):
        raise FloatingPointError("final LBFGS parameter is nonfinite")
    final = float(final_loss.detach().cpu().item())
    max_gradient = max(
        float(torch.max(torch.abs(gradient)).detach().cpu().item())
        for gradient in gradients
        if gradient is not None
    )
    if final > initial + 1e-10:
        raise FloatingPointError("final objective exceeds zero initialization")
    return (
        weight.detach().cpu().numpy().astype(np.float32, copy=True),
        bias.detach().cpu().numpy().astype(np.float32, copy=True),
        initial,
        final,
        max_gradient,
        str(target),
    )


def fit_linear_head(
    features: np.ndarray, labels: np.ndarray, device: Any = "cuda"
) -> LinearHeadState:
    """Fit the exact zero-initialized full-batch weighted LBFGS head."""

    result = _torch_linear_fit(
        features, labels, device=device, enforce_frozen_shape=True
    )
    return LinearHeadState(*result)


def fit_linear_head_final_refit(
    features: np.ndarray, labels: np.ndarray, device: Any = "cuda"
) -> LinearHeadState:
    """Fit the unchanged head on exactly 34,000 post-GO refit examples."""

    values = _validate_features(
        features, name="features", exact_training_count=False
    )
    if values.shape[0] != FINAL_REFIT_TRAINING_EXAMPLE_COUNT:
        raise ValueError(
            f"features must have exactly {FINAL_REFIT_TRAINING_EXAMPLE_COUNT} "
            "final-refit rows"
        )
    _validate_final_refit_labels(labels)
    result = _torch_linear_fit(
        values, labels, device=device, enforce_frozen_shape=False
    )
    return LinearHeadState(*result)


def linear_logits(
    features: np.ndarray,
    state: LinearHeadState,
    device: Any = "cuda",
    batch_size: int = EXTRACTION_BATCH_SIZE,
) -> np.ndarray:
    """Score standardized MiniROCKET features and return CPU float64 logits."""

    if not isinstance(state, LinearHeadState):
        raise TypeError("state must be LinearHeadState")
    values = _validate_features(
        features, name="features", exact_training_count=False
    )
    if values.dtype != np.dtype(np.float32):
        raise ValueError("features must be standardized float32")
    size = _strict_int(batch_size, name="batch_size")
    if size != EXTRACTION_BATCH_SIZE:
        raise ValueError(f"batch_size must be the frozen value {EXTRACTION_BATCH_SIZE}")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyTorch is required for v8 scoring") from error
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must identify CPU or CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    # Frozen states loaded from the strict NPZ persistence layer are deliberately
    # read-only.  PyTorch warns when ``as_tensor`` aliases such an array because
    # writes through the tensor would be undefined.  Materialize owned writable
    # buffers before the device transfer; this is deterministic and keeps
    # ``-W error`` production preflight clean.
    weight = torch.as_tensor(
        np.array(state.weight, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32,
        device=target,
    )
    bias = torch.as_tensor(
        np.array(state.bias, dtype=np.float32, order="C", copy=True),
        dtype=torch.float32,
        device=target,
    )
    output = np.empty((values.shape[0], CLASS_COUNT), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, values.shape[0], size):
            stop = min(start + size, values.shape[0])
            x = torch.as_tensor(
                np.array(values[start:stop], dtype=np.float32, order="C", copy=True),
                dtype=torch.float32,
                device=target,
            )
            logits = functional.linear(x, weight, bias)
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("linear scoring produced nonfinite logits")
            output[start:stop] = logits.detach().cpu().numpy().astype(np.float64)
    return output


__all__ = [
    "CANDIDATE_ID",
    "CHANNEL_COUNT",
    "CLASS_COUNT",
    "CLASS_EXAMPLE_COUNTS",
    "CLASS_ORDER",
    "CLASS_WEIGHTS",
    "EXTRACTION_BATCH_SIZE",
    "FEATURE_COUNT",
    "FINAL_REFIT_CLASS_EXAMPLE_COUNTS",
    "FINAL_REFIT_RUNS_PER_CLASS",
    "FINAL_REFIT_TRAINING_EXAMPLE_COUNT",
    "FIT_RUNS_PER_CLASS",
    "FeatureStandardizer",
    "GLOBAL_SEED",
    "HEALTHY_ENDPOINTS",
    "LinearHeadState",
    "MATURE_FAULT_ENDPOINTS",
    "PREFIX_ENDPOINTS",
    "ROWS_PER_RUN",
    "RobustNormalizer",
    "TRAINING_EXAMPLE_COUNT",
    "TrainingWindows",
    "WINDOW_LENGTH",
    "build_training_windows",
    "build_final_refit_training_windows",
    "fit_feature_standardizer",
    "fit_feature_standardizer_final_refit",
    "fit_linear_head",
    "fit_linear_head_final_refit",
    "linear_logits",
    "normalize_runs",
    "robust_fit_normalizer",
    "robust_fit_normalizer_final_refit",
    "set_determinism",
    "standardize_features",
]
