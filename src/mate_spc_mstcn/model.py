"""Multi-Scale Causal Dynamics TCN prototype.

This module contains no dataset loader and no experiment protocol.  It is a
standalone technical prototype intended for later, separately governed use on
already-consumed development data.  The implementation provides:

* strictly endpoint-causal windows;
* fixed-prefix robust preprocessing with variance/covariance and loop-dynamic
  channels;
* a dilated residual TCN endpoint state plus trailing multi-scale mean/RMS
  pooling of the pointwise projection, preserving exact whole-run/window
  equivalence for the frozen 128-sample window;
* fixed-epoch, class-balanced CUDA float64 training without validation or
  early-stopping leakage;
* safe NPZ serialization and CPU/CUDA inference parity checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# On the project's Windows CUDA runtime, Torch must own the OpenMP/DLL load
# order.  Direct import of this numeric module therefore preloads Torch before
# NumPy.  If NumPy is already present, do not attempt a dangerous late load;
# numeric entry points fail closed with an actionable error.  Package-level
# metadata imports stay Torch-free through the lazy __init__ module.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # pragma: no cover - subprocess regression test
    _torch = None
else:
    try:  # pragma: no cover - runtime dependent
        import torch as _torch
    except (ImportError, OSError):  # pragma: no cover - runtime dependent
        _torch = None

import numpy as np

from .features import (
    BASELINE_STOP,
    CLASS_IDS,
    DYNAMIC_FEATURE_COUNT,
    FEATURE_SCHEMA,
    MONITOR_START,
    PROCESS_FEATURE_COUNT,
    RUN_LENGTH,
    RobustRunTransform,
)


MODEL_SCHEMA = "mate_spc_mstcn_detection_mil_model"
MODEL_SCHEMA_VERSION = 2
DEFAULT_WINDOW_LENGTH = 128
DEFAULT_CHANNELS = 32
DEFAULT_DILATIONS = (1, 2, 4, 8, 16)
DEFAULT_POOL_SCALES = (8, 32, 96)
DEFAULT_HIDDEN = 64
DEFAULT_STRIDE = 4
DEFAULT_MIL_TOP_FRACTION = 0.125
DEFAULT_PARITY_ATOL = 1.0e-9
QUICK_DIAGNOSTIC_TRAINING = MappingProxyType(
    {
        "epochs": 8,
        "batch_size": 8,
        "examples_per_class_per_epoch": 16,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "device": "cuda",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _plain_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _plain_class_id(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _fraction(value: Any, *, name: str) -> float:
    result = _finite_positive(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must be no larger than one")
    return result


@dataclass(frozen=True)
class MSTCNConfig:
    """Serializable architecture, causal sampling, and MIL geometry."""

    feature_schema: str = FEATURE_SCHEMA
    input_features: int = DYNAMIC_FEATURE_COUNT
    window_length: int = DEFAULT_WINDOW_LENGTH
    channels: int = DEFAULT_CHANNELS
    kernel_size: int = 3
    dilations: tuple[int, ...] = DEFAULT_DILATIONS
    pool_scales: tuple[int, ...] = DEFAULT_POOL_SCALES
    hidden_size: int = DEFAULT_HIDDEN
    class_ids: tuple[int, ...] = CLASS_IDS
    monitor_start: int = MONITOR_START
    endpoint_stride: int = DEFAULT_STRIDE
    mil_top_fraction: float = DEFAULT_MIL_TOP_FRACTION

    def __post_init__(self) -> None:
        if not isinstance(self.feature_schema, str) or self.feature_schema != FEATURE_SCHEMA:
            raise ValueError(f"feature_schema must be {FEATURE_SCHEMA!r}")
        integer_fields = (
            "input_features",
            "window_length",
            "channels",
            "kernel_size",
            "hidden_size",
            "monitor_start",
            "endpoint_stride",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _plain_positive_int(getattr(self, name), name=name))
        dilations = tuple(_plain_positive_int(value, name="dilation") for value in self.dilations)
        pools = tuple(_plain_positive_int(value, name="pool scale") for value in self.pool_scales)
        classes = tuple(
            _plain_class_id(value, name="class id") for value in self.class_ids
        )
        top_fraction = _fraction(self.mil_top_fraction, name="mil_top_fraction")
        if self.input_features != DYNAMIC_FEATURE_COUNT:
            raise ValueError(f"input_features must be {DYNAMIC_FEATURE_COUNT}")
        if (
            self.kernel_size < 2
            or not dilations
            or len(set(dilations)) != len(dilations)
            or tuple(sorted(dilations)) != dilations
        ):
            raise ValueError("kernel_size and dilations do not define a valid causal TCN")
        if (
            not pools
            or tuple(sorted(pools)) != pools
            or len(set(pools)) != len(pools)
            or pools[-1] > self.window_length
        ):
            raise ValueError("pool_scales must be increasing and no larger than window_length")
        if classes != CLASS_IDS:
            raise ValueError(f"class_ids must be exactly {CLASS_IDS}")
        if self.window_length > self.monitor_start + 1:
            raise ValueError("the first monitoring endpoint must contain one complete window")
        if self.monitor_start < BASELINE_STOP:
            raise ValueError("monitor_start must not overlap the commissioning prefix")
        if top_fraction not in (0.125, 0.25):
            raise ValueError("mil_top_fraction must be the frozen q=1/8 or q=1/4")
        object.__setattr__(self, "dilations", dilations)
        object.__setattr__(self, "pool_scales", pools)
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "mil_top_fraction", top_fraction)

    @property
    def receptive_field(self) -> int:
        # Two causal convolutions per residual block.
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


@dataclass(frozen=True)
class MSTCNTrainingConfig:
    """Complete optimization configuration recorded with a fitted model."""

    epochs: int = 30
    batch_size: int = 8
    examples_per_class_per_epoch: int = 100
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "examples_per_class_per_epoch"):
            object.__setattr__(
                self, name, _plain_positive_int(getattr(self, name), name=name)
            )
        if self.batch_size != 8:
            raise ValueError("batch_size is frozen at eight runs (two per class)")
        if self.examples_per_class_per_epoch % 2:
            raise ValueError("examples_per_class_per_epoch must be even")
        for name in ("learning_rate", "gradient_clip_norm"):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name=name)
            )
        for name in ("weight_decay",):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name=name)
            )
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")


_CACHED_NETWORK_CLASS: Any | None = None


def _network_class() -> Any:
    global _CACHED_NETWORK_CLASS
    if _CACHED_NETWORK_CLASS is not None:
        return _CACHED_NETWORK_CLASS
    try:
        from torch import nn
        import torch
        import torch.nn.functional as functional
    except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for MSCD-TCN") from error

    class CausalResidualBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.activation = nn.GELU()

        def _conv(self, layer: Any, values: Any) -> Any:
            return layer(functional.pad(values, (self.left_padding, 0)))

        def forward(self, values: Any) -> Any:
            residual = values
            hidden = self.activation(self._conv(self.conv1, values))
            hidden = self._conv(self.conv2, hidden)
            return self.activation(residual + hidden)

    class DetectionAttributionTCN(nn.Module):
        def __init__(self, config: MSTCNConfig):
            super().__init__()
            self.config = config
            self.input_projection = nn.Conv1d(config.input_features, config.channels, 1)
            self.blocks = nn.ModuleList(
                [
                    CausalResidualBlock(config.channels, config.kernel_size, dilation)
                    for dilation in config.dilations
                ]
            )
            pooled_width = config.channels * (1 + 2 * len(config.pool_scales))
            self.hidden = nn.Linear(pooled_width, config.hidden_size)
            self.activation = nn.GELU()
            self.detection_head = nn.Linear(config.hidden_size, 1)
            self.attribution_head = nn.Linear(
                config.hidden_size, len(config.class_ids) - 1
            )

        def encode_sequence(self, values: Any) -> tuple[Any, Any]:
            if (
                values.ndim != 3
                or values.shape[1] < self.config.window_length
                or values.shape[2] != self.config.input_features
            ):
                raise ValueError(
                    "MSCD-TCN input must have shape "
                    f"(batch, time>={self.config.window_length}, "
                    f"{self.config.input_features})"
                )
            projected = self.input_projection(values.transpose(1, 2))
            hidden = projected
            for block in self.blocks:
                hidden = block(hidden)
            return projected, hidden

        def forward_endpoints(self, values: Any, endpoints: Any) -> tuple[Any, Any]:
            projected, hidden = self.encode_sequence(values)
            if (
                endpoints.ndim != 1
                or endpoints.numel() < 1
                or endpoints.dtype != torch.int64
                or int(torch.min(endpoints)) < self.config.window_length - 1
                or int(torch.max(endpoints)) >= values.shape[1]
            ):
                raise ValueError("endpoints must name complete causal windows")
            # Final TCN state has receptive field 125 <= window length 128.
            # Multi-scale moments use the pointwise projection, whose support is
            # exactly the named trailing scale. This makes whole-sequence and
            # independently materialized-window evaluation mathematically
            # equivalent while avoiding repeated TCN convolution.
            pieces = [hidden.index_select(2, endpoints).transpose(1, 2)]
            prefix = functional.pad(torch.cumsum(projected, dim=2), (1, 0))
            square_prefix = functional.pad(
                torch.cumsum(torch.square(projected), dim=2), (1, 0)
            )
            ends = endpoints + 1
            for scale in self.config.pool_scales:
                starts = ends - scale
                mean = (
                    prefix.index_select(2, ends)
                    - prefix.index_select(2, starts)
                ) / float(scale)
                square_mean = (
                    square_prefix.index_select(2, ends)
                    - square_prefix.index_select(2, starts)
                ) / float(scale)
                pieces.append(mean.transpose(1, 2))
                pieces.append(
                    torch.sqrt(torch.clamp(square_mean, min=0.0) + 1.0e-12).transpose(
                        1, 2
                    )
                )
            pooled = torch.cat(pieces, dim=2)
            representation = self.activation(self.hidden(pooled))
            detection = self.detection_head(representation).squeeze(2)
            attribution = self.attribution_head(representation)
            return detection, attribution

        def forward(self, values: Any) -> tuple[Any, Any]:
            endpoint = torch.tensor(
                [values.shape[1] - 1], dtype=torch.int64, device=values.device
            )
            detection, attribution = self.forward_endpoints(values, endpoint)
            return detection[:, 0], attribution[:, 0, :]

    _CACHED_NETWORK_CLASS = DetectionAttributionTCN
    return DetectionAttributionTCN


def _torch_module() -> Any:
    if _torch is None:
        raise RuntimeError(
            "MSCD-TCN requires PyTorch to be imported before NumPy in this "
            "Windows process; start a fresh process and import "
            "mate_spc_mstcn.model first"
        )
    return _torch


def _configure_determinism(torch: Any) -> None:
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):  # pragma: no cover - version dependent
        pass
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_mstcn_network(
    config: MSTCNConfig | None = None,
    *,
    seed: int = 1729,
    device: str = "cpu",
) -> Any:
    """Build a deterministically initialized float64 causal network."""

    torch = _torch_module()
    actual = MSTCNConfig() if config is None else config
    if not isinstance(actual, MSTCNConfig):
        raise TypeError("config must be MSTCNConfig")
    seed_value = _plain_class_id(seed, name="seed")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    _configure_determinism(torch)
    return _network_class()(actual).to(device=target, dtype=torch.float64)


def _numpy_state_dict(network: Any) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, value in network.state_dict().items():
        array = np.array(value.detach().cpu().numpy(), dtype=np.float64, copy=True, order="C")
        if not np.isfinite(array).all():
            raise FloatingPointError(f"network parameter {name!r} is nonfinite")
        array.setflags(write=False)
        state[str(name)] = array
    return state


def _load_numpy_state(network: Any, state: Mapping[str, np.ndarray]) -> None:
    torch = _torch_module()
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("serialized MSCD-TCN state keys differ from the architecture")
    tensors: dict[str, Any] = {}
    for name, expected_value in expected.items():
        array = np.asarray(state[name])
        if (
            array.dtype != np.dtype(np.float64)
            or tuple(array.shape) != tuple(expected_value.shape)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid MSCD-TCN parameter {name!r}")
        # Model state arrays are intentionally read-only. torch.tensor makes a
        # writable owned copy and avoids PyTorch's undefined-behaviour warning
        # for non-writable NumPy buffers.
        tensors[name] = torch.tensor(
            array, device=expected_value.device, dtype=expected_value.dtype
        )
    network.load_state_dict(tensors, strict=True)


def causal_windows(
    features: Any,
    endpoints: Sequence[int] | np.ndarray,
    *,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> np.ndarray:
    """Materialize windows whose final row is exactly each named endpoint."""

    values = np.asarray(features)
    length = _plain_positive_int(window_length, name="window_length")
    if values.shape != (RUN_LENGTH, DYNAMIC_FEATURE_COUNT):
        raise ValueError(
            f"features must have shape ({RUN_LENGTH}, {DYNAMIC_FEATURE_COUNT})"
        )
    if values.dtype != np.dtype(np.float64) or not np.isfinite(values).all():
        raise ValueError("features must be finite exact float64")
    raw_endpoints = np.asarray(endpoints)
    if raw_endpoints.ndim != 1 or raw_endpoints.dtype.kind not in "iu":
        raise ValueError("endpoints must be a one-dimensional integer array")
    ends = np.asarray(raw_endpoints, dtype=np.int64)
    if ends.size == 0 or np.any(ends < length - 1) or np.any(ends >= RUN_LENGTH):
        raise ValueError("each endpoint must contain one complete in-run causal window")
    offsets = np.arange(length, dtype=np.int64)
    indices = ends[:, None] - length + 1 + offsets[None, :]
    windows = np.array(values[indices], dtype=np.float64, copy=True, order="C")
    windows.setflags(write=False)
    return windows


@dataclass(frozen=True)
class CausalTrainingIndex:
    """Run-level classes plus one shared causal full-monitoring MIL bag."""

    run_names: tuple[str, ...]
    class_indices: np.ndarray
    mil_endpoints: np.ndarray
    config: MSTCNConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, MSTCNConfig):
            raise TypeError("config must be MSTCNConfig")
        names = tuple(str(name) for name in self.run_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("run_names must be nonempty and unique")
        raw_classes = np.asarray(self.class_indices)
        raw_endpoints = np.asarray(self.mil_endpoints)
        if raw_classes.ndim != 1 or raw_classes.dtype.kind not in "iu":
            raise ValueError("class_indices must be a one-dimensional integer array")
        if raw_endpoints.ndim != 1 or raw_endpoints.dtype.kind not in "iu":
            raise ValueError("mil_endpoints must be a one-dimensional integer array")
        classes = np.array(raw_classes, dtype=np.int64, copy=True)
        endpoints = np.array(raw_endpoints, dtype=np.int64, copy=True)
        if classes.shape != (len(names),):
            raise ValueError("class_indices must align one-to-one with run_names")
        if np.any(classes < 0) or np.any(classes >= len(CLASS_IDS)):
            raise ValueError("training class index is out of bounds")
        expected = np.arange(
            self.config.monitor_start,
            RUN_LENGTH,
            self.config.endpoint_stride,
            dtype=np.int64,
        )
        if not np.array_equal(endpoints, expected):
            raise ValueError("mil_endpoints must be the frozen full monitoring bag")
        classes.setflags(write=False)
        endpoints.setflags(write=False)
        object.__setattr__(self, "run_names", names)
        object.__setattr__(self, "class_indices", classes)
        object.__setattr__(self, "mil_endpoints", endpoints)


def build_training_index(
    runs: Mapping[str, np.ndarray],
    labels: Mapping[str, int],
    *,
    config: MSTCNConfig | None = None,
) -> CausalTrainingIndex:
    """Build run labels and the shared 160:500:4 bag.

    Crucially, the returned object has no per-endpoint class label. A fault
    label belongs to its run/bag only, which permits intermittent Fault 9/15
    evidence rather than forcing every post-onset endpoint to be positive.
    """

    actual = MSTCNConfig() if config is None else config
    if not isinstance(actual, MSTCNConfig):
        raise TypeError("config must be MSTCNConfig")
    if not isinstance(runs, Mapping) or not runs or set(runs) != set(labels):
        raise ValueError("runs and labels must be aligned nonempty mappings")
    names = tuple(sorted(runs))
    classes: list[int] = []
    present: set[int] = set()
    for name in names:
        values = np.asarray(runs[name])
        if (
            values.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT)
            or values.dtype != np.dtype(np.float64)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"runs[{name!r}] must be finite exact float64 with shape (500, 52)"
            )
        label = _plain_class_id(labels[name], name=f"labels[{name!r}]")
        if label not in actual.class_ids:
            raise ValueError(f"unsupported class label {label}")
        present.add(label)
        classes.append(actual.class_ids.index(label))
    if present != set(actual.class_ids):
        raise ValueError("training mappings must contain every class 0/3/9/15")
    return CausalTrainingIndex(
        names,
        np.asarray(classes, dtype=np.int64),
        np.arange(
            actual.monitor_start,
            RUN_LENGTH,
            actual.endpoint_stride,
            dtype=np.int64,
        ),
        actual,
    )


def _copy_state(state: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a nonempty mapping")
    copied: dict[str, np.ndarray] = {}
    for raw_name, value in state.items():
        if not isinstance(raw_name, str) or not raw_name or raw_name in copied:
            raise ValueError("state parameter names must be unique nonempty strings")
        source = np.asarray(value)
        if source.dtype != np.dtype(np.float64) or not np.isfinite(source).all():
            raise ValueError(
                f"state parameter {raw_name!r} must be finite exact float64"
            )
        array = np.array(source, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
        copied[raw_name] = array
    return MappingProxyType(copied)


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


@dataclass(frozen=True)
class MSTCNWindowScores:
    detection_logits: np.ndarray
    attribution_logits: np.ndarray

    def __post_init__(self) -> None:
        detection = np.asarray(self.detection_logits)
        attribution = np.asarray(self.attribution_logits)
        if (
            detection.ndim != 1
            or attribution.shape != (detection.size, len(CLASS_IDS) - 1)
            or detection.dtype != np.dtype(np.float64)
            or attribution.dtype != np.dtype(np.float64)
            or not np.isfinite(detection).all()
            or not np.isfinite(attribution).all()
        ):
            raise ValueError(
                "window scores must be aligned finite exact-float64 arrays"
            )
        detection_copy = np.array(detection, copy=True, order="C")
        attribution_copy = np.array(attribution, copy=True, order="C")
        detection_copy.setflags(write=False)
        attribution_copy.setflags(write=False)
        object.__setattr__(self, "detection_logits", detection_copy)
        object.__setattr__(self, "attribution_logits", attribution_copy)


@dataclass(frozen=True)
class MSTCNPath:
    endpoints: np.ndarray
    detection_logits: np.ndarray
    attribution_logits: np.ndarray
    detection_probabilities: np.ndarray = field(init=False)
    attribution_probabilities: np.ndarray = field(init=False)
    class_probabilities: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        raw_endpoints = np.asarray(self.endpoints)
        detection = np.asarray(self.detection_logits)
        attribution = np.asarray(self.attribution_logits)
        if raw_endpoints.ndim != 1 or raw_endpoints.dtype.kind not in "iu":
            raise ValueError("endpoints must be a one-dimensional integer array")
        endpoints = np.array(raw_endpoints, dtype=np.int64, copy=True)
        if (
            detection.shape != (endpoints.size,)
            or attribution.shape != (endpoints.size, len(CLASS_IDS) - 1)
            or detection.dtype != np.dtype(np.float64)
            or attribution.dtype != np.dtype(np.float64)
            or not np.isfinite(detection).all()
            or not np.isfinite(attribution).all()
        ):
            raise ValueError("MSTCNPath logits are not aligned finite float64 arrays")
        detection_copy = np.array(detection, copy=True, order="C")
        attribution_copy = np.array(attribution, copy=True, order="C")
        detection_probability = _sigmoid_numpy(detection_copy)
        shifted = attribution_copy - np.max(
            attribution_copy, axis=1, keepdims=True
        )
        exponent = np.exp(shifted)
        attribution_probability = exponent / exponent.sum(axis=1, keepdims=True)
        class_probability = np.column_stack(
            (
                1.0 - detection_probability,
                detection_probability[:, None] * attribution_probability,
            )
        )
        if not np.allclose(
            class_probability.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise FloatingPointError("joint MSCD-TCN probabilities do not sum to one")
        for array in (
            endpoints,
            detection_copy,
            attribution_copy,
            detection_probability,
            attribution_probability,
            class_probability,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "detection_logits", detection_copy)
        object.__setattr__(self, "attribution_logits", attribution_copy)
        object.__setattr__(self, "detection_probabilities", detection_probability)
        object.__setattr__(
            self, "attribution_probabilities", attribution_probability
        )
        object.__setattr__(self, "class_probabilities", class_probability)

    @property
    def detection_evidence(self) -> np.ndarray:
        """Fault-vs-healthy log-odds evidence used by the sequential scan."""

        return self.detection_logits

    @property
    def probabilities(self) -> np.ndarray:
        """Joint healthy/3/9/15 probabilities for reporting."""

        return self.class_probabilities


@dataclass(frozen=True)
class MSTCNModel:
    transform: RobustRunTransform
    config: MSTCNConfig
    state: Mapping[str, np.ndarray]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.transform, RobustRunTransform):
            raise TypeError("transform must be RobustRunTransform")
        if not isinstance(self.config, MSTCNConfig):
            raise TypeError("config must be MSTCNConfig")
        seed = _plain_class_id(self.seed, name="seed")
        object.__setattr__(self, "state", _copy_state(self.state))
        object.__setattr__(self, "seed", seed)

    @classmethod
    def from_untrained(
        cls,
        transform: RobustRunTransform,
        *,
        config: MSTCNConfig | None = None,
        seed: int = 1729,
    ) -> "MSTCNModel":
        actual = MSTCNConfig() if config is None else config
        network = build_mstcn_network(actual, seed=seed, device="cpu")
        return cls(transform, actual, _numpy_state_dict(network), int(seed))

    def _network(self, *, device: str) -> Any:
        network = build_mstcn_network(self.config, seed=self.seed, device=device)
        _load_numpy_state(network, self.state)
        network.eval()
        return network

    def predict_feature_windows(
        self,
        windows: Any,
        *,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> MSTCNWindowScores:
        torch = _torch_module()
        batch = _plain_positive_int(batch_size, name="batch_size")
        values = np.asarray(windows)
        expected_tail = (self.config.window_length, self.config.input_features)
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or tuple(values.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"windows must have shape (n>=1, {expected_tail[0]}, "
                f"{expected_tail[1]})"
            )
        if values.dtype != np.dtype(np.float64) or not np.isfinite(values).all():
            raise ValueError("windows must be finite exact float64")
        network = self._network(device=device)
        target = torch.device(device)
        detection_parts: list[np.ndarray] = []
        attribution_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, values.shape[0], batch):
                tensor = torch.tensor(
                    values[start : start + batch], device=target, dtype=torch.float64
                )
                detection, attribution = network(tensor)
                detection_parts.append(detection.detach().cpu().numpy())
                attribution_parts.append(attribution.detach().cpu().numpy())
        return MSTCNWindowScores(
            np.concatenate(detection_parts, axis=0).astype(np.float64, copy=False),
            np.concatenate(attribution_parts, axis=0).astype(np.float64, copy=False),
        )

    def predict_feature_runs(
        self,
        features: Any,
        endpoints: Sequence[int] | np.ndarray,
        *,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> tuple[MSTCNWindowScores, ...]:
        """Efficiently score complete causal sequences once per run batch."""

        torch = _torch_module()
        batch = _plain_positive_int(batch_size, name="batch_size")
        values = np.asarray(features)
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1:] != (RUN_LENGTH, self.config.input_features)
            or values.dtype != np.dtype(np.float64)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"features must be finite exact float64 (runs>=1, {RUN_LENGTH}, "
                f"{self.config.input_features})"
            )
        raw_endpoints = np.asarray(endpoints)
        if raw_endpoints.ndim != 1 or raw_endpoints.dtype.kind not in "iu":
            raise ValueError("endpoints must be a one-dimensional integer array")
        chosen = np.asarray(raw_endpoints, dtype=np.int64)
        if (
            chosen.size == 0
            or np.any(chosen < self.config.window_length - 1)
            or np.any(chosen >= RUN_LENGTH)
        ):
            raise ValueError("endpoints must name complete causal windows")
        network = self._network(device=device)
        target = torch.device(device)
        endpoint_tensor = torch.tensor(
            chosen, device=target, dtype=torch.int64
        )
        detection_batches: list[np.ndarray] = []
        attribution_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, values.shape[0], batch):
                tensor = torch.tensor(
                    values[start : start + batch], device=target, dtype=torch.float64
                )
                detection, attribution = network.forward_endpoints(
                    tensor, endpoint_tensor
                )
                detection_batches.append(detection.detach().cpu().numpy())
                attribution_batches.append(attribution.detach().cpu().numpy())
        all_detection = np.concatenate(detection_batches, axis=0)
        all_attribution = np.concatenate(attribution_batches, axis=0)
        return tuple(
            MSTCNWindowScores(all_detection[index], all_attribution[index])
            for index in range(values.shape[0])
        )

    def score_run(
        self,
        run: Any,
        *,
        endpoints: Sequence[int] | np.ndarray | None = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> MSTCNPath:
        features = self.transform.transform_run(run)
        if endpoints is None:
            chosen = np.arange(self.config.monitor_start, RUN_LENGTH, dtype=np.int64)
        else:
            chosen = np.asarray(endpoints)
        scores = self.predict_feature_runs(
            features[None, :, :],
            chosen,
            device=device,
            batch_size=batch_size,
        )[0]
        return MSTCNPath(
            np.asarray(chosen),
            scores.detection_logits,
            scores.attribution_logits,
        )

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        return save_mstcn_model(self, path, metadata=metadata)


@dataclass(frozen=True)
class MSTCNTrainingResult:
    model: MSTCNModel
    loss_history: tuple[float, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.model, MSTCNModel):
            raise TypeError("model must be MSTCNModel")
        losses = tuple(float(value) for value in self.loss_history)
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("loss_history must contain finite values")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata = dict(self.metadata)
        _canonical_json(metadata)
        object.__setattr__(self, "loss_history", losses)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


def _balanced_run_batches(
    class_indices: np.ndarray,
    *,
    class_count: int,
    maximum_per_class: int,
    seed: int,
) -> np.ndarray:
    """Return batches containing exactly two runs from each of four classes."""

    rng = np.random.default_rng(seed)
    available = [
        np.flatnonzero(class_indices == class_index)
        for class_index in range(class_count)
    ]
    if any(values.size == 0 for values in available):
        raise ValueError("every class must have training runs")
    target = min(
        _plain_positive_int(maximum_per_class, name="maximum_per_class"),
        *(len(values) for values in available),
    )
    target -= target % 2
    if target < 2:
        raise ValueError("each class needs at least two runs for a balanced batch")
    columns: list[np.ndarray] = []
    for values in available:
        chosen = rng.choice(values, size=target, replace=False)
        rng.shuffle(chosen)
        columns.append(chosen.reshape(-1, 2))
    batches = np.concatenate(columns, axis=1).astype(np.int64, copy=False)
    rng.shuffle(batches, axis=0)
    return batches


def mstcn_mil_loss_torch(
    detection_logits: Any,
    attribution_logits: Any,
    class_indices: Any,
    *,
    top_fraction: float,
) -> tuple[Any, Mapping[str, Any]]:
    """Frozen class-balanced full-bag MIL objective.

    Each input batch must contain exactly two healthy, Fault 3, Fault 9, and
    Fault 15 runs. Only the top-q run evidence is supervised; individual
    endpoints never receive positive labels.
    """

    torch = _torch_module()
    import torch.nn.functional as functional

    if (
        not torch.is_tensor(detection_logits)
        or detection_logits.ndim != 2
        or detection_logits.shape[0] != 8
        or detection_logits.shape[1] < 1
        or not torch.is_floating_point(detection_logits)
        or not bool(torch.isfinite(detection_logits).all())
    ):
        raise ValueError("detection_logits must be finite floating (8, endpoints)")
    if (
        not torch.is_tensor(attribution_logits)
        or attribution_logits.shape
        != (
            detection_logits.shape[0],
            detection_logits.shape[1],
            len(CLASS_IDS) - 1,
        )
        or attribution_logits.dtype != detection_logits.dtype
        or attribution_logits.device != detection_logits.device
        or not bool(torch.isfinite(attribution_logits).all())
    ):
        raise ValueError("attribution_logits must be aligned with detection logits")
    if (
        not torch.is_tensor(class_indices)
        or class_indices.shape != (8,)
        or class_indices.dtype != torch.int64
        or class_indices.device != detection_logits.device
    ):
        raise ValueError("class_indices must be int64 shape (8,) on the same device")
    counts = torch.bincount(class_indices, minlength=len(CLASS_IDS))
    if counts.shape[0] != len(CLASS_IDS) or not bool(torch.all(counts == 2)):
        raise ValueError("each MIL batch must contain exactly two runs per class")
    fraction = _fraction(top_fraction, name="top_fraction")
    if fraction not in (0.125, 0.25):
        raise ValueError("top_fraction must be q=1/8 or q=1/4")
    top_count = int(math.ceil(detection_logits.shape[1] * fraction))
    top_values, top_indices = torch.topk(
        detection_logits,
        k=top_count,
        dim=1,
        largest=True,
        sorted=False,
    )
    run_evidence = top_values.mean(dim=1)
    selected_attribution = torch.gather(
        attribution_logits,
        1,
        top_indices[:, :, None].expand(-1, -1, len(CLASS_IDS) - 1),
    )
    run_attribution = selected_attribution.mean(dim=1)

    healthy = class_indices == 0
    negative_term = 0.5 * functional.softplus(run_evidence[healthy]).mean()
    positive_terms: list[Any] = []
    attribution_terms: list[Any] = []
    for class_index in range(1, len(CLASS_IDS)):
        selected = class_indices == class_index
        positive_terms.append(
            functional.softplus(-run_evidence[selected]).mean()
        )
        targets = torch.full(
            (int(selected.sum()),),
            class_index - 1,
            dtype=torch.int64,
            device=detection_logits.device,
        )
        attribution_terms.append(
            functional.cross_entropy(run_attribution[selected], targets)
        )
    positive_term = torch.stack(positive_terms).sum() / 6.0
    attribution_term = 0.25 * torch.stack(attribution_terms).mean()
    total = negative_term + positive_term + attribution_term
    terms: Mapping[str, Any] = MappingProxyType(
        {
            "healthy_detection": negative_term,
            "fault_detection": positive_term,
            "fault_attribution": attribution_term,
            "run_evidence": run_evidence,
            "run_attribution_logits": run_attribution,
            "top_count": top_count,
        }
    )
    return total, terms


def fit_mstcn_cuda(
    runs: Mapping[str, np.ndarray],
    labels: Mapping[str, int],
    *,
    config: MSTCNConfig | None = None,
    seed: int = 1729,
    epochs: int = 30,
    batch_size: int = 8,
    examples_per_class_per_epoch: int = 100,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    gradient_clip_norm: float = 1.0,
    device: str = "cuda",
) -> MSTCNTrainingResult:
    """Fit q-specific binary detection plus auxiliary attribution on CUDA.

    All runs use the same causal endpoints 160,164,...,496. Supervision is
    strictly at run/bag level. There is no endpoint loss, transition exclusion,
    ranking loss, validation, early stopping, or threshold-data access.
    """

    torch = _torch_module()
    actual = MSTCNConfig() if config is None else config
    if not isinstance(actual, MSTCNConfig):
        raise TypeError("config must be MSTCNConfig")
    seed_value = _plain_class_id(seed, name="seed")
    training = MSTCNTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        examples_per_class_per_epoch=examples_per_class_per_epoch,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        device=device,
    )
    target = torch.device(training.device)
    if target.type != "cuda":
        raise ValueError("fit_mstcn_cuda requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    _configure_determinism(torch)

    index = build_training_index(runs, labels, config=actual)
    ordered_runs = {name: np.asarray(runs[name]) for name in index.run_names}
    transform = RobustRunTransform.fit(ordered_runs)
    features = np.stack(
        [transform.transform_run(ordered_runs[name]) for name in index.run_names],
        axis=0,
    )
    # Keep the complete feature corpus in host memory. Only each frozen
    # eight-run bag batch is transferred to CUDA.
    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    endpoint_tensor = torch.tensor(
        index.mil_endpoints, device=target, dtype=torch.int64
    )
    network = build_mstcn_network(
        actual, seed=seed_value, device=training.device
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    losses: list[float] = []

    for epoch in range(training.epochs):
        batches = _balanced_run_batches(
            index.class_indices,
            class_count=len(actual.class_ids),
            maximum_per_class=training.examples_per_class_per_epoch,
            seed=seed_value + epoch,
        )
        network.train()
        total_loss = 0.0
        for selected in batches:
            run_features = feature_tensor[selected].to(
                device=target, dtype=torch.float64
            )
            targets = torch.tensor(
                index.class_indices[selected], device=target, dtype=torch.int64
            )
            optimizer.zero_grad(set_to_none=True)
            detection, attribution = network.forward_endpoints(
                run_features, endpoint_tensor
            )
            loss, _ = mstcn_mil_loss_torch(
                detection,
                attribution,
                targets,
                top_fraction=actual.mil_top_fraction,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("MSCD-TCN training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=training.gradient_clip_norm
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        epoch_loss = total_loss / float(len(batches))
        if not math.isfinite(epoch_loss):
            raise FloatingPointError("MSCD-TCN training loss became nonfinite")
        losses.append(epoch_loss)

    model = MSTCNModel(
        transform=transform,
        config=actual,
        state=_numpy_state_dict(network),
        seed=seed_value,
    )
    class_run_counts = {
        str(class_id): int(np.count_nonzero(index.class_indices == class_index))
        for class_index, class_id in enumerate(actual.class_ids)
    }
    metadata: Mapping[str, object] = MappingProxyType(
        {
            "method": "Detection-first Multi-Scale Causal Dynamics TCN",
            "objective": "q-specific full-monitoring run-level MIL",
            "mil_top_fraction_q": actual.mil_top_fraction,
            "positive_endpoint_labels_used": False,
            "detection_target": "fault_vs_healthy",
            "attribution_target": "fault_3_vs_9_vs_15_fault_runs_only",
            "loss_formula": (
                "0.5*mean_H softplus(A) + "
                "(1/6)*sum_k mean_Fk softplus(-A) + "
                "0.25*(1/3)*sum_k mean_Fk CE(C,k)"
            ),
            "architecture_pooling": (
                "TCN endpoint state plus 8/32/96 mean-RMS of pointwise projection"
            ),
            "requested_device": training.device,
            "dtype": "float64",
            "seed": seed_value,
            "model_config": _config_document(actual),
            "training_config": asdict(training),
            "training_run_names": list(index.run_names),
            "training_run_count_by_class": class_run_counts,
            "mil_endpoints_half_open_stride": [
                actual.monitor_start,
                RUN_LENGTH,
                actual.endpoint_stride,
            ],
            "mil_endpoint_count": int(index.mil_endpoints.size),
            "dense_inference_endpoints_half_open_stride": [
                actual.monitor_start,
                RUN_LENGTH,
                1,
            ],
            "batch_balance": "eight_runs_exactly_two_per_class",
            "validation_or_early_stopping_used": False,
            "window_contract": (
                "each output ends at its named endpoint and contains no future row"
            ),
            "preprocessor_fit_scope": "training_run_commissioning_prefixes_only",
        }
    )
    return MSTCNTrainingResult(model, tuple(losses), metadata)


def assert_cpu_cuda_inference_parity(
    model: MSTCNModel,
    run: Any,
    *,
    endpoints: Sequence[int] | np.ndarray | None = None,
    batch_size: int = 64,
    atol: float = DEFAULT_PARITY_ATOL,
) -> float:
    """Return maximum dual-head logit error or raise on parity failure."""

    if not isinstance(model, MSTCNModel):
        raise TypeError("model must be MSTCNModel")
    tolerance = _finite_nonnegative(atol, name="atol")
    torch = _torch_module()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cpu = model.score_run(run, endpoints=endpoints, device="cpu", batch_size=batch_size)
    cuda = model.score_run(run, endpoints=endpoints, device="cuda", batch_size=batch_size)
    if not np.array_equal(cpu.endpoints, cuda.endpoints):
        raise AssertionError("CPU/CUDA endpoints differ")
    detection_error = float(
        np.max(
            np.abs(cpu.detection_logits - cuda.detection_logits),
            initial=0.0,
        )
    )
    attribution_error = float(
        np.max(
            np.abs(cpu.attribution_logits - cuda.attribution_logits),
            initial=0.0,
        )
    )
    error = max(detection_error, attribution_error)
    if error > tolerance:
        raise AssertionError(
            f"CPU/CUDA MSCD-TCN dual-head parity failed: "
            f"{error:.3e} > {tolerance:.3e}"
        )
    return error


def assert_window_sequence_inference_parity(
    model: MSTCNModel,
    run: Any,
    *,
    endpoints: Sequence[int] | np.ndarray,
    device: str = "cpu",
    batch_size: int = 64,
    atol: float = 1.0e-10,
) -> float:
    """Verify efficient whole-sequence logits match independent 128 windows."""

    if not isinstance(model, MSTCNModel):
        raise TypeError("model must be MSTCNModel")
    tolerance = _finite_nonnegative(atol, name="atol")
    features = model.transform.transform_run(run)
    chosen = np.asarray(endpoints)
    windows = causal_windows(
        features, chosen, window_length=model.config.window_length
    )
    window_scores = model.predict_feature_windows(
        windows, device=device, batch_size=batch_size
    )
    sequence_scores = model.predict_feature_runs(
        features[None, :, :],
        chosen,
        device=device,
        batch_size=1,
    )[0]
    detection_error = float(
        np.max(
            np.abs(
                window_scores.detection_logits
                - sequence_scores.detection_logits
            ),
            initial=0.0,
        )
    )
    attribution_error = float(
        np.max(
            np.abs(
                window_scores.attribution_logits
                - sequence_scores.attribution_logits
            ),
            initial=0.0,
        )
    )
    error = max(detection_error, attribution_error)
    if error > tolerance:
        raise AssertionError(
            f"window/sequence MSCD-TCN parity failed: "
            f"{error:.3e} > {tolerance:.3e}"
        )
    return error


def _config_document(config: MSTCNConfig) -> dict[str, object]:
    return asdict(config)


def _config_from_document(document: Mapping[str, Any]) -> MSTCNConfig:
    expected = set(asdict(MSTCNConfig()))
    if set(document) != expected:
        raise ValueError("serialized MSTCN config fields differ from schema")
    values = dict(document)
    for name in ("dilations", "pool_scales", "class_ids"):
        if not isinstance(values[name], list):
            raise ValueError(f"serialized config {name} must be a JSON array")
        values[name] = tuple(values[name])
    return MSTCNConfig(**values)


def save_mstcn_model(
    model: MSTCNModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write a pickle-free, manifest-bound NPZ model."""

    if not isinstance(model, MSTCNModel):
        raise TypeError("model must be MSTCNModel")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    metadata_document = dict(metadata or {})
    metadata_json = _canonical_json(metadata_document)
    if len(metadata_json) > 1_048_576:
        raise ValueError("metadata exceeds the canonical JSON size limit")
    manifest: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {}
    for index, name in enumerate(sorted(model.state)):
        key = f"state_{index:03d}"
        value = np.asarray(model.state[name])
        manifest.append(
            {
                "name": name,
                "storage_key": key,
                "shape": list(value.shape),
                "dtype": "float64",
            }
        )
        payload[key] = np.asarray(value, dtype=np.float64)
    payload.update(
        {
            "schema": np.array(MODEL_SCHEMA, dtype=np.str_),
            "schema_version": np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
            "config_json": np.array(
                _canonical_json(_config_document(model.config)), dtype=np.str_
            ),
            "metadata_json": np.array(metadata_json, dtype=np.str_),
            "state_manifest_json": np.array(
                _canonical_json(manifest), dtype=np.str_
            ),
            "scale_floor": np.asarray(model.transform.scale_floor, dtype=np.float64),
            "floor_quantile": np.array(
                model.transform.floor_quantile, dtype=np.float64
            ),
            "seed": np.array(model.seed, dtype=np.int64),
            "dtype": np.array("float64", dtype=np.str_),
        }
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
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
    value = str(array.item())
    if len(value) > limit:
        raise ValueError(f"{name} exceeds its size limit")
    return value


def load_mstcn_model(
    path: str | os.PathLike[str],
) -> tuple[MSTCNModel, dict[str, object]]:
    """Load and structurally validate a safe MSCD-TCN NPZ model."""

    source = Path(path)
    base_keys = {
        "schema",
        "schema_version",
        "config_json",
        "metadata_json",
        "state_manifest_json",
        "scale_floor",
        "floor_quantile",
        "seed",
        "dtype",
    }
    with np.load(source, allow_pickle=False) as archive:
        files = set(archive.files)
        if not base_keys <= files:
            raise ValueError("MSTCN model is missing required arrays")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("object arrays are forbidden in MSTCN models")
    schema = _scalar_text(arrays["schema"], name="schema", limit=128)
    dtype_name = _scalar_text(arrays["dtype"], name="dtype", limit=32)
    config_json = _scalar_text(arrays["config_json"], name="config_json", limit=65536)
    metadata_json = _scalar_text(arrays["metadata_json"], name="metadata_json", limit=1048576)
    manifest_json = _scalar_text(
        arrays["state_manifest_json"], name="state_manifest_json", limit=1048576
    )
    if schema != MODEL_SCHEMA or dtype_name != "float64":
        raise ValueError("MSTCN schema or dtype changed")
    if arrays["schema_version"].shape != () or arrays["schema_version"].dtype != np.dtype(np.int64):
        raise ValueError("schema_version must be scalar int64")
    if int(arrays["schema_version"].item()) != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported MSTCN schema version")
    try:
        config_raw = json.loads(config_json)
        metadata = json.loads(metadata_json)
        manifest = json.loads(manifest_json)
    except json.JSONDecodeError as error:
        raise ValueError("MSTCN JSON document is invalid") from error
    if (
        not isinstance(config_raw, dict)
        or not isinstance(metadata, dict)
        or not isinstance(manifest, list)
    ):
        raise ValueError("MSTCN JSON document has the wrong type")
    if (
        _canonical_json(config_raw) != config_json
        or _canonical_json(metadata) != metadata_json
        or _canonical_json(manifest) != manifest_json
    ):
        raise ValueError("MSTCN JSON documents must use canonical encoding")
    config = _config_from_document(config_raw)
    if len(manifest) == 0 or len(manifest) > 256:
        raise ValueError("MSTCN state manifest length is invalid")
    expected_storage = {f"state_{index:03d}" for index in range(len(manifest))}
    if files != base_keys | expected_storage:
        raise ValueError("MSTCN NPZ contains extra or missing state arrays")
    state: dict[str, np.ndarray] = {}
    for index, raw in enumerate(manifest):
        key = f"state_{index:03d}"
        if not isinstance(raw, dict) or set(raw) != {"name", "storage_key", "shape", "dtype"}:
            raise ValueError("MSTCN state manifest row is invalid")
        name = raw["name"]
        shape = raw["shape"]
        if (
            not isinstance(name, str)
            or not name
            or raw["storage_key"] != key
            or raw["dtype"] != "float64"
            or name in state
            or not isinstance(shape, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in shape
            )
        ):
            raise ValueError("MSTCN state manifest identity is invalid")
        array = arrays[key]
        if (
            tuple(shape) != array.shape
            or array.dtype != np.dtype(np.float64)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"MSTCN state array {name!r} is invalid")
        state[name] = array
    floor = arrays["scale_floor"]
    quantile = arrays["floor_quantile"]
    seed = arrays["seed"]
    if quantile.shape != () or quantile.dtype != np.dtype(np.float64):
        raise ValueError("floor_quantile must be scalar float64")
    if seed.shape != () or seed.dtype != np.dtype(np.int64):
        raise ValueError("seed must be scalar int64")
    transform = RobustRunTransform(floor, float(quantile.item()))
    model = MSTCNModel(transform, config, state, int(seed.item()))
    # Constructing once validates every state name and tensor shape against the
    # exact serialized architecture, rather than deferring it to first use.
    model._network(device="cpu")
    return model, dict(metadata)


__all__ = [
    "CausalTrainingIndex",
    "DEFAULT_PARITY_ATOL",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "MSTCNConfig",
    "MSTCNModel",
    "MSTCNPath",
    "MSTCNTrainingConfig",
    "MSTCNTrainingResult",
    "MSTCNWindowScores",
    "QUICK_DIAGNOSTIC_TRAINING",
    "assert_cpu_cuda_inference_parity",
    "assert_window_sequence_inference_parity",
    "build_mstcn_network",
    "build_training_index",
    "causal_windows",
    "fit_mstcn_cuda",
    "load_mstcn_model",
    "mstcn_mil_loss_torch",
    "save_mstcn_model",
]
