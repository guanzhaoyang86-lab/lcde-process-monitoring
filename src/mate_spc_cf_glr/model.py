"""Low-parameter one-vs-healthy experts for the frozen CF-GLR v4 method.

The module deliberately contains no data loader or split logic.  Callers provide
the causal expert feature matrices produced by ``features.py``.  Fitting is
CUDA/float64 only; a fitted model can be scored on CPU or CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping

# The supported Windows environment requires Torch to own the DLL/OpenMP load
# order.  Package users must import this numeric module before importing NumPy.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # pragma: no cover - subprocess/runtime dependent
    _torch = None
else:
    try:  # pragma: no cover - runtime dependent
        import torch as _torch
    except (ImportError, OSError):  # pragma: no cover - runtime dependent
        _torch = None

import numpy as np


MODEL_SCHEMA = "mate_spc_cf_glr_three_expert_model"
MODEL_SCHEMA_VERSION = 1
FEATURE_SCHEMA = "cf_glr_mechanism_features_v2"
FAULT_IDS = (3, 9, 15)
EXPERT_FEATURE_DIMS = (20, 20, 28)
RUN_LENGTH = 500
PREFIX_START = 128
MONITOR_START = 160
ENDPOINT_STEP = 4
MIL_TOP_FRACTION = 0.10
F15_HIDDEN_SIZE = 16
LINEAR_SEED = 20260909
F15_ENSEMBLE_SEEDS = (20260909, 20260910, 20260911)
DEFAULT_PARITY_ATOL = 1.0e-9
FEATURE_MAD_CONSISTENCY = 1.4826
FEATURE_SCALE_FLOOR = 1.0e-6


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_int(value: Any, *, name: str) -> int:
    result = _plain_int(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class CFGLRConfig:
    """Frozen feature geometry and expert architecture."""

    feature_schema: str = FEATURE_SCHEMA
    fault_ids: tuple[int, ...] = FAULT_IDS
    expert_feature_dims: tuple[int, ...] = EXPERT_FEATURE_DIMS
    run_length: int = RUN_LENGTH
    prefix_start: int = PREFIX_START
    monitor_start: int = MONITOR_START
    endpoint_step: int = ENDPOINT_STEP
    mil_top_fraction: float = MIL_TOP_FRACTION
    f15_hidden_size: int = F15_HIDDEN_SIZE
    linear_seed: int = LINEAR_SEED
    f15_ensemble_seeds: tuple[int, ...] = F15_ENSEMBLE_SEEDS

    def __post_init__(self) -> None:
        if self.feature_schema != FEATURE_SCHEMA:
            raise ValueError(f"feature_schema must be exactly {FEATURE_SCHEMA!r}")
        faults = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        dims = tuple(
            _positive_int(value, name="expert feature dimension")
            for value in self.expert_feature_dims
        )
        seeds = tuple(_plain_int(value, name="ensemble seed") for value in self.f15_ensemble_seeds)
        if faults != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        if dims != EXPERT_FEATURE_DIMS:
            raise ValueError(f"expert_feature_dims must be exactly {EXPERT_FEATURE_DIMS}")
        if seeds != F15_ENSEMBLE_SEEDS:
            raise ValueError(f"f15_ensemble_seeds must be exactly {F15_ENSEMBLE_SEEDS}")
        integer_names = (
            "run_length",
            "prefix_start",
            "monitor_start",
            "endpoint_step",
            "f15_hidden_size",
            "linear_seed",
        )
        for name in integer_names:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        fraction = _positive_float(self.mil_top_fraction, name="mil_top_fraction")
        if self.run_length != RUN_LENGTH:
            raise ValueError(f"run_length must be {RUN_LENGTH}")
        if self.prefix_start != PREFIX_START or self.monitor_start != MONITOR_START:
            raise ValueError("the frozen prefix and monitoring boundaries are 128 and 160")
        if self.endpoint_step != ENDPOINT_STEP:
            raise ValueError(f"endpoint_step must be {ENDPOINT_STEP}")
        if fraction != MIL_TOP_FRACTION:
            raise ValueError(f"mil_top_fraction must be exactly {MIL_TOP_FRACTION}")
        if self.f15_hidden_size != F15_HIDDEN_SIZE:
            raise ValueError(f"f15_hidden_size must be {F15_HIDDEN_SIZE}")
        if self.linear_seed != LINEAR_SEED:
            raise ValueError(f"linear_seed must be {LINEAR_SEED}")
        object.__setattr__(self, "fault_ids", faults)
        object.__setattr__(self, "expert_feature_dims", dims)
        object.__setattr__(self, "f15_ensemble_seeds", seeds)
        object.__setattr__(self, "mil_top_fraction", fraction)

    @property
    def main_endpoints(self) -> tuple[int, ...]:
        return tuple(range(self.monitor_start, self.run_length, self.endpoint_step))

    @property
    def prefix_endpoints(self) -> tuple[int, ...]:
        return tuple(range(self.prefix_start, self.monitor_start, self.endpoint_step))

    @property
    def top_count(self) -> int:
        return int(math.ceil(len(self.main_endpoints) * self.mil_top_fraction))

    def feature_dim(self, fault_id: int) -> int:
        parsed = _plain_int(fault_id, name="fault_id")
        try:
            return self.expert_feature_dims[self.fault_ids.index(parsed)]
        except ValueError as error:
            raise ValueError(f"fault_id must be one of {self.fault_ids}") from error


@dataclass(frozen=True)
class CFGLRTrainingConfig:
    """Immutable optimization contract for the single v4 candidate."""

    epochs: int = 80
    learning_rate: float = 1.0e-3
    l2_coefficient: float = 1.0e-4
    prefix_negative_weight: float = 0.1
    device: str = "cuda"
    dtype: str = "float64"
    optimizer: str = "AdamW"

    def __post_init__(self) -> None:
        if _positive_int(self.epochs, name="epochs") != 80:
            raise ValueError("epochs are frozen at 80")
        if _positive_float(self.learning_rate, name="learning_rate") != 1.0e-3:
            raise ValueError("learning_rate is frozen at 1e-3")
        if _positive_float(self.l2_coefficient, name="l2_coefficient") != 1.0e-4:
            raise ValueError("l2_coefficient is frozen at 1e-4")
        if (
            _positive_float(self.prefix_negative_weight, name="prefix_negative_weight")
            != 0.1
        ):
            raise ValueError("prefix_negative_weight is frozen at 0.1")
        if self.device != "cuda" or self.dtype != "float64" or self.optimizer != "AdamW":
            raise ValueError("training is frozen to CUDA float64 AdamW")


def _torch_module() -> Any:
    if _torch is None:
        raise RuntimeError(
            "CF-GLR numerics require PyTorch to be imported before NumPy in this "
            "Windows process; start a fresh process and import "
            "mate_spc_cf_glr.model first"
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


_CACHED_NETWORK_CLASSES: tuple[Any, Any] | None = None


def _network_classes() -> tuple[Any, Any]:
    global _CACHED_NETWORK_CLASSES
    if _CACHED_NETWORK_CLASSES is not None:
        return _CACHED_NETWORK_CLASSES
    torch = _torch_module()
    nn = torch.nn

    class LinearExpert(nn.Module):
        def __init__(self, input_features: int):
            super().__init__()
            self.head = nn.Linear(input_features, 1)

        def forward(self, values: Any) -> Any:
            return self.head(values).squeeze(-1)

    class Fault15Expert(nn.Module):
        def __init__(self, input_features: int, hidden_size: int):
            super().__init__()
            self.input = nn.Linear(input_features, hidden_size)
            self.activation = nn.GELU()
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            return self.output(self.activation(self.input(values))).squeeze(-1)

    _CACHED_NETWORK_CLASSES = (LinearExpert, Fault15Expert)
    return _CACHED_NETWORK_CLASSES


def _member_seeds(fault_id: int, config: CFGLRConfig) -> tuple[int, ...]:
    return config.f15_ensemble_seeds if fault_id == 15 else (config.linear_seed,)


def build_expert_network(
    fault_id: int,
    config: CFGLRConfig | None = None,
    *,
    seed: int | None = None,
    device: str = "cpu",
) -> Any:
    """Build one deterministically initialized expert member."""

    torch = _torch_module()
    actual = CFGLRConfig() if config is None else config
    if not isinstance(actual, CFGLRConfig):
        raise TypeError("config must be CFGLRConfig")
    parsed_fault = _plain_int(fault_id, name="fault_id")
    if parsed_fault not in actual.fault_ids:
        raise ValueError(f"fault_id must be one of {actual.fault_ids}")
    allowed_seeds = _member_seeds(parsed_fault, actual)
    actual_seed = allowed_seeds[0] if seed is None else _plain_int(seed, name="seed")
    if actual_seed not in allowed_seeds:
        raise ValueError(f"seed for fault {parsed_fault} must be one of {allowed_seeds}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    _configure_determinism(torch)
    linear_class, f15_class = _network_classes()
    if parsed_fault == 15:
        network = f15_class(actual.feature_dim(parsed_fault), actual.f15_hidden_size)
    else:
        network = linear_class(actual.feature_dim(parsed_fault))
    return network.to(device=target, dtype=torch.float64)


def _numpy_state_dict(network: Any) -> Mapping[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for name, value in network.state_dict().items():
        array = np.array(value.detach().cpu().numpy(), dtype=np.float64, copy=True, order="C")
        if not np.isfinite(array).all():
            raise FloatingPointError(f"network parameter {name!r} is nonfinite")
        array.setflags(write=False)
        copied[str(name)] = array
    return MappingProxyType(copied)


def _load_numpy_state(network: Any, state: Mapping[str, np.ndarray]) -> None:
    torch = _torch_module()
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("serialized expert state keys differ from the architecture")
    tensors: dict[str, Any] = {}
    for name, expected_value in expected.items():
        array = np.asarray(state[name])
        if (
            array.dtype != np.dtype(np.float64)
            or tuple(array.shape) != tuple(expected_value.shape)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid expert parameter {name!r}")
        tensors[name] = torch.tensor(
            array, device=expected_value.device, dtype=expected_value.dtype
        )
    network.load_state_dict(tensors, strict=True)


def _copy_state(state: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for raw_name, raw_value in state.items():
        name = str(raw_name)
        if not name or name in result:
            raise ValueError("state parameter names must be nonempty and unique")
        source = np.asarray(raw_value)
        if source.dtype != np.dtype(np.float64) or not np.isfinite(source).all():
            raise ValueError(f"state parameter {name!r} must be finite exact float64")
        value = np.array(source, dtype=np.float64, copy=True, order="C")
        value.setflags(write=False)
        result[name] = value
    if not result:
        raise ValueError("expert state must not be empty")
    return MappingProxyType(result)


def _normalize_fault_mapping(
    value: Mapping[Any, Any], *, name: str
) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[int, Any] = {}
    for raw_fault_id, item in value.items():
        fault_id = _plain_int(raw_fault_id, name=f"{name} fault id")
        if fault_id in result:
            raise ValueError(f"{name} contains duplicate fault ids")
        result[fault_id] = item
    if set(result) != set(FAULT_IDS):
        raise ValueError(f"{name} must map exactly the fault ids {FAULT_IDS}")
    return result


def _copy_standardizer_vector(
    value: Any,
    *,
    fault_id: int,
    dimension: int,
    kind: str,
) -> np.ndarray:
    array = np.asarray(value)
    invalid_scale = kind == "scale" and (
        np.any(array <= 0.0) or np.any(array < FEATURE_SCALE_FLOOR)
    )
    if (
        array.shape != (dimension,)
        or array.dtype != np.dtype(np.float64)
        or not np.isfinite(array).all()
        or invalid_scale
    ):
        raise ValueError(
            f"fault {fault_id} feature {kind} must be a finite exact-float64 "
            f"vector of length {dimension}"
        )
    copied = np.array(array, dtype=np.float64, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class ExpertLogitPaths:
    """Dense logits in canonical fault order, shape ``[runs, 500, 3]``."""

    logits: np.ndarray
    fault_ids: tuple[int, ...] = FAULT_IDS

    def __post_init__(self) -> None:
        values = np.asarray(self.logits)
        faults = tuple(_plain_int(value, name="fault id") for value in self.fault_ids)
        if faults != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        if (
            values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1:] != (RUN_LENGTH, len(FAULT_IDS))
            or values.dtype != np.dtype(np.float64)
            or not np.isfinite(values).all()
        ):
            raise ValueError("logits must be finite exact float64 with shape [runs,500,3]")
        copied = np.array(values, dtype=np.float64, copy=True, order="C")
        copied.setflags(write=False)
        object.__setattr__(self, "logits", copied)
        object.__setattr__(self, "fault_ids", faults)

    def for_fault(self, fault_id: int) -> np.ndarray:
        parsed = _plain_int(fault_id, name="fault_id")
        try:
            index = self.fault_ids.index(parsed)
        except ValueError as error:
            raise ValueError(f"fault_id must be one of {self.fault_ids}") from error
        result = self.logits[:, :, index]
        result.setflags(write=False)
        return result


def _validate_feature_batches(
    features: Mapping[int, Any],
    config: CFGLRConfig,
    *,
    label: str,
) -> tuple[dict[int, np.ndarray], int]:
    normalized = _normalize_fault_mapping(features, name=label)
    result: dict[int, np.ndarray] = {}
    run_count: int | None = None
    for fault_id in config.fault_ids:
        value = np.asarray(normalized[fault_id])
        expected_tail = (config.run_length, config.feature_dim(fault_id))
        if (
            value.ndim != 3
            or value.shape[0] < 1
            or tuple(value.shape[1:]) != expected_tail
            or value.dtype != np.dtype(np.float64)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"{label}[{fault_id}] must be finite exact float64 with shape "
                f"[runs,{expected_tail[0]},{expected_tail[1]}]"
            )
        if run_count is None:
            run_count = int(value.shape[0])
        elif value.shape[0] != run_count:
            raise ValueError(f"all {label} expert matrices must have the same run count")
        result[fault_id] = value
    assert run_count is not None
    return result, run_count


@dataclass(frozen=True)
class CFGLRModel:
    """Immutable three-expert model with NumPy-owned state."""

    config: CFGLRConfig
    states: Mapping[int, tuple[Mapping[str, np.ndarray], ...]]
    feature_centers: Mapping[int, np.ndarray]
    feature_scales: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.config, CFGLRConfig):
            raise TypeError("config must be CFGLRConfig")
        raw_states = _normalize_fault_mapping(self.states, name="states")
        raw_centers = _normalize_fault_mapping(
            self.feature_centers, name="feature_centers"
        )
        raw_scales = _normalize_fault_mapping(self.feature_scales, name="feature_scales")
        copied: dict[int, tuple[Mapping[str, np.ndarray], ...]] = {}
        centers: dict[int, np.ndarray] = {}
        scales: dict[int, np.ndarray] = {}
        for fault_id in self.config.fault_ids:
            members = tuple(raw_states[fault_id])
            seeds = _member_seeds(fault_id, self.config)
            if len(members) != len(seeds):
                raise ValueError(f"fault {fault_id} requires {len(seeds)} state member(s)")
            checked: list[Mapping[str, np.ndarray]] = []
            for seed, member in zip(seeds, members, strict=True):
                state = _copy_state(member)
                network = build_expert_network(
                    fault_id, self.config, seed=seed, device="cpu"
                )
                _load_numpy_state(network, state)
                checked.append(state)
            copied[fault_id] = tuple(checked)
            dimension = self.config.feature_dim(fault_id)
            centers[fault_id] = _copy_standardizer_vector(
                raw_centers[fault_id],
                fault_id=fault_id,
                dimension=dimension,
                kind="center",
            )
            scales[fault_id] = _copy_standardizer_vector(
                raw_scales[fault_id],
                fault_id=fault_id,
                dimension=dimension,
                kind="scale",
            )
        object.__setattr__(self, "states", MappingProxyType(copied))
        object.__setattr__(self, "feature_centers", MappingProxyType(centers))
        object.__setattr__(self, "feature_scales", MappingProxyType(scales))

    def score_runs(
        self,
        features: Mapping[int, Any],
        *,
        device: str = "cpu",
        batch_size: int = 65536,
    ) -> ExpertLogitPaths:
        """Score supplied expert feature matrices without temporal look-ahead."""

        torch = _torch_module()
        matrices, run_count = _validate_feature_batches(
            features, self.config, label="features"
        )
        size = _positive_int(batch_size, name="batch_size")
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        output = np.empty((run_count, self.config.run_length, len(FAULT_IDS)), np.float64)
        for fault_index, fault_id in enumerate(self.config.fault_ids):
            standardized = (
                matrices[fault_id] - self.feature_centers[fault_id][None, None, :]
            ) / self.feature_scales[fault_id][None, None, :]
            if not np.isfinite(standardized).all():
                raise FloatingPointError("feature standardization produced nonfinite values")
            flat = standardized.reshape(-1, self.config.feature_dim(fault_id))
            member_outputs: list[np.ndarray] = []
            for seed, state in zip(
                _member_seeds(fault_id, self.config), self.states[fault_id], strict=True
            ):
                network = build_expert_network(
                    fault_id, self.config, seed=seed, device=str(target)
                )
                _load_numpy_state(network, state)
                network.eval()
                pieces: list[np.ndarray] = []
                with torch.inference_mode():
                    for start in range(0, flat.shape[0], size):
                        tensor = torch.tensor(
                            flat[start : start + size],
                            device=target,
                            dtype=torch.float64,
                        )
                        pieces.append(
                            np.asarray(network(tensor).cpu().numpy(), dtype=np.float64)
                        )
                member_outputs.append(np.concatenate(pieces))
            averaged = np.mean(np.stack(member_outputs, axis=0), axis=0, dtype=np.float64)
            output[:, :, fault_index] = averaged.reshape(run_count, self.config.run_length)
        if not np.isfinite(output).all():
            raise FloatingPointError("expert inference produced nonfinite logits")
        return ExpertLogitPaths(output, self.config.fault_ids)

    def score_run(
        self,
        features: Mapping[int, Any],
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        """Score one run and return a read-only ``[500,3]`` logit matrix."""

        normalized = _normalize_fault_mapping(features, name="features")
        batched: dict[int, np.ndarray] = {}
        for fault_id in self.config.fault_ids:
            value = np.asarray(normalized[fault_id])
            if value.shape != (self.config.run_length, self.config.feature_dim(fault_id)):
                raise ValueError(f"run features for fault {fault_id} have the wrong shape")
            if value.dtype != np.dtype(np.float64) or not np.isfinite(value).all():
                raise ValueError("run features must be finite exact float64")
            batched[fault_id] = value[None, :, :]
        result = self.score_runs(batched, device=device).logits[0]
        result.setflags(write=False)
        return result

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        return save_cf_glr_model(self, path, metadata=metadata)


def initialize_cf_glr_model(config: CFGLRConfig | None = None) -> CFGLRModel:
    """Return the deterministic initialized state; useful for structural tests."""

    actual = CFGLRConfig() if config is None else config
    states: dict[int, tuple[Mapping[str, np.ndarray], ...]] = {}
    centers: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    for fault_id in actual.fault_ids:
        states[fault_id] = tuple(
            _numpy_state_dict(
                build_expert_network(fault_id, actual, seed=seed, device="cpu")
            )
            for seed in _member_seeds(fault_id, actual)
        )
        centers[fault_id] = np.zeros(actual.feature_dim(fault_id), dtype=np.float64)
        scales[fault_id] = np.ones(actual.feature_dim(fault_id), dtype=np.float64)
    return CFGLRModel(actual, states, centers, scales)


def binary_mil_loss_torch(
    healthy_logits: Any,
    fault_logits: Any,
    prefix_logits: Any,
    parameters: Iterable[Any],
    *,
    top_fraction: float = MIL_TOP_FRACTION,
    prefix_negative_weight: float = 0.1,
    l2_coefficient: float = 1.0e-4,
) -> tuple[Any, Mapping[str, Any]]:
    """Balanced binary bag loss used independently by every expert."""

    torch = _torch_module()
    functional = torch.nn.functional
    if (
        healthy_logits.ndim != 2
        or fault_logits.ndim != 2
        or prefix_logits.ndim not in (1, 2)
        or healthy_logits.shape[0] < 1
        or fault_logits.shape[0] < 1
        or healthy_logits.shape[1] != fault_logits.shape[1]
        or healthy_logits.shape[1] < 1
    ):
        raise ValueError("MIL logits have incompatible shapes")
    tensors = (healthy_logits, fault_logits, prefix_logits)
    if any(
        value.dtype != torch.float64 or not bool(torch.isfinite(value).all())
        for value in tensors
    ):
        raise ValueError("MIL logits must be finite torch.float64 tensors")
    fraction = _positive_float(top_fraction, name="top_fraction")
    if fraction != MIL_TOP_FRACTION:
        raise ValueError(f"top_fraction must be exactly {MIL_TOP_FRACTION}")
    prefix_weight = _positive_float(
        prefix_negative_weight, name="prefix_negative_weight"
    )
    coefficient = _positive_float(l2_coefficient, name="l2_coefficient")
    if prefix_weight != 0.1 or coefficient != 1.0e-4:
        raise ValueError("the prefix and L2 loss weights are frozen at 0.1 and 1e-4")
    top_count = int(math.ceil(healthy_logits.shape[1] * fraction))
    healthy_bag = torch.topk(healthy_logits, top_count, dim=1).values.mean(dim=1)
    fault_bag = torch.topk(fault_logits, top_count, dim=1).values.mean(dim=1)
    healthy_loss = functional.binary_cross_entropy_with_logits(
        healthy_bag, torch.zeros_like(healthy_bag)
    )
    fault_loss = functional.binary_cross_entropy_with_logits(
        fault_bag, torch.ones_like(fault_bag)
    )
    prefix_flat = prefix_logits.reshape(-1)
    prefix_loss = functional.binary_cross_entropy_with_logits(
        prefix_flat, torch.zeros_like(prefix_flat)
    )
    parameter_values = tuple(parameters)
    if not parameter_values:
        raise ValueError("parameters must not be empty")
    l2 = sum(torch.sum(torch.square(value)) for value in parameter_values)
    loss = (
        0.5 * healthy_loss
        + 0.5 * fault_loss
        + prefix_weight * prefix_loss
        + coefficient * l2
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("MIL loss is nonfinite")
    return loss, MappingProxyType(
        {
            "healthy_bag_loss": healthy_loss,
            "fault_bag_loss": fault_loss,
            "prefix_negative_loss": prefix_loss,
            "l2_squared_norm": l2,
            "top_count": top_count,
        }
    )


@dataclass(frozen=True)
class CFGLRTrainingResult:
    model: CFGLRModel
    final_losses: Mapping[str, float]
    training: CFGLRTrainingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.model, CFGLRModel):
            raise TypeError("model must be CFGLRModel")
        if not isinstance(self.training, CFGLRTrainingConfig):
            raise TypeError("training must be CFGLRTrainingConfig")
        copied: dict[str, float] = {}
        for name, value in self.final_losses.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("final losses must be finite")
            copied[str(name)] = number
        if not copied:
            raise ValueError("final_losses must not be empty")
        object.__setattr__(self, "final_losses", MappingProxyType(copied))


def _fit_feature_standardizers_validated(
    healthy: Mapping[int, np.ndarray],
    faults: Mapping[int, np.ndarray],
    config: CFGLRConfig,
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    endpoints = np.asarray(
        config.prefix_endpoints + config.main_endpoints, dtype=np.int64
    )
    centers: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    for fault_id in config.fault_ids:
        if healthy[fault_id].shape[0] != faults[fault_id].shape[0]:
            raise ValueError(
                f"fault {fault_id} feature standardization requires balanced "
                "healthy and target-fault run counts"
            )
        selected = np.concatenate(
            (
                healthy[fault_id][:, endpoints, :],
                faults[fault_id][:, endpoints, :],
            ),
            axis=0,
        ).reshape(-1, config.feature_dim(fault_id))
        center = np.median(selected, axis=0).astype(np.float64, copy=False)
        raw_scale = (
            FEATURE_MAD_CONSISTENCY
            * np.median(np.abs(selected - center[None, :]), axis=0)
        ).astype(np.float64, copy=False)
        scale = np.maximum(raw_scale, FEATURE_SCALE_FLOOR).astype(
            np.float64, copy=False
        )
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise FloatingPointError("feature standardizer fit produced nonfinite values")
        center = np.array(center, dtype=np.float64, copy=True, order="C")
        scale = np.array(scale, dtype=np.float64, copy=True, order="C")
        center.setflags(write=False)
        scale.setflags(write=False)
        centers[fault_id] = center
        scales[fault_id] = scale
    return MappingProxyType(centers), MappingProxyType(scales)


def fit_feature_standardizers(
    healthy_features: Mapping[int, Any],
    fault_features: Mapping[int, Any],
    *,
    config: CFGLRConfig | None = None,
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray]]:
    """Fit per-expert median/MAD transforms from balanced fit runs only."""

    actual = CFGLRConfig() if config is None else config
    if not isinstance(actual, CFGLRConfig):
        raise TypeError("config must be CFGLRConfig")
    healthy, healthy_count = _validate_feature_batches(
        healthy_features, actual, label="healthy_features"
    )
    faults, fault_count = _validate_feature_batches(
        fault_features, actual, label="fault_features"
    )
    if healthy_count != fault_count:
        raise ValueError("healthy and target-fault fit run counts must be balanced")
    return _fit_feature_standardizers_validated(healthy, faults, actual)


def fit_cf_glr_cuda(
    healthy_features: Mapping[int, Any],
    fault_features: Mapping[int, Any],
    *,
    config: CFGLRConfig | None = None,
    training: CFGLRTrainingConfig | None = None,
) -> CFGLRTrainingResult:
    """Fit the single frozen candidate using only caller-supplied fit matrices."""

    torch = _torch_module()
    actual = CFGLRConfig() if config is None else config
    train_config = CFGLRTrainingConfig() if training is None else training
    if not isinstance(actual, CFGLRConfig) or not isinstance(
        train_config, CFGLRTrainingConfig
    ):
        raise TypeError("config and training have invalid types")
    if not torch.cuda.is_available():
        raise RuntimeError("fit_cf_glr_cuda requires an available CUDA device")
    healthy, healthy_count = _validate_feature_batches(
        healthy_features, actual, label="healthy_features"
    )
    faults, fault_count = _validate_feature_batches(
        fault_features, actual, label="fault_features"
    )
    if healthy_count != fault_count:
        raise ValueError("healthy and target-fault fit run counts must be balanced")
    feature_centers, feature_scales = _fit_feature_standardizers_validated(
        healthy, faults, actual
    )
    target = torch.device("cuda")
    main_endpoints = torch.tensor(actual.main_endpoints, device=target, dtype=torch.int64)
    prefix_endpoints = torch.tensor(
        actual.prefix_endpoints, device=target, dtype=torch.int64
    )
    states: dict[int, tuple[Mapping[str, np.ndarray], ...]] = {}
    losses: dict[str, float] = {}
    for fault_id in actual.fault_ids:
        standardized_healthy = (
            healthy[fault_id] - feature_centers[fault_id][None, None, :]
        ) / feature_scales[fault_id][None, None, :]
        standardized_fault = (
            faults[fault_id] - feature_centers[fault_id][None, None, :]
        ) / feature_scales[fault_id][None, None, :]
        healthy_tensor = torch.tensor(
            standardized_healthy, device=target, dtype=torch.float64
        )
        fault_tensor = torch.tensor(
            standardized_fault, device=target, dtype=torch.float64
        )
        member_states: list[Mapping[str, np.ndarray]] = []
        for member_index, seed in enumerate(_member_seeds(fault_id, actual)):
            network = build_expert_network(
                fault_id, actual, seed=seed, device="cuda"
            )
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=train_config.learning_rate, weight_decay=0.0
            )
            final_loss: Any | None = None
            for _ in range(train_config.epochs):
                network.train()
                optimizer.zero_grad(set_to_none=True)
                healthy_main = network(healthy_tensor.index_select(1, main_endpoints))
                fault_main = network(fault_tensor.index_select(1, main_endpoints))
                prefix = torch.cat(
                    (
                        network(healthy_tensor.index_select(1, prefix_endpoints)),
                        network(fault_tensor.index_select(1, prefix_endpoints)),
                    ),
                    dim=0,
                )
                final_loss, _ = binary_mil_loss_torch(
                    healthy_main,
                    fault_main,
                    prefix,
                    network.parameters(),
                    top_fraction=actual.mil_top_fraction,
                    prefix_negative_weight=train_config.prefix_negative_weight,
                    l2_coefficient=train_config.l2_coefficient,
                )
                final_loss.backward()
                optimizer.step()
            assert final_loss is not None
            loss_number = float(final_loss.detach().cpu())
            if not math.isfinite(loss_number):
                raise FloatingPointError("training produced a nonfinite final loss")
            losses[f"fault_{fault_id}_member_{member_index}"] = loss_number
            member_states.append(_numpy_state_dict(network))
        states[fault_id] = tuple(member_states)
    model = CFGLRModel(actual, states, feature_centers, feature_scales)
    return CFGLRTrainingResult(model, losses, train_config)


def assert_cpu_cuda_inference_parity(
    model: CFGLRModel,
    features: Mapping[int, Any],
    *,
    atol: float = DEFAULT_PARITY_ATOL,
) -> float:
    """Require dense CPU/CUDA logits to agree within the frozen tolerance."""

    torch = _torch_module()
    if not isinstance(model, CFGLRModel):
        raise TypeError("model must be CFGLRModel")
    tolerance = _positive_float(atol, name="atol")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    cpu = model.score_runs(features, device="cpu").logits
    cuda = model.score_runs(features, device="cuda").logits
    error = float(np.max(np.abs(cpu - cuda)))
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"CPU/CUDA expert-logit parity failed: max error {error:.17g} > {tolerance:.17g}"
        )
    return error


def _config_document(config: CFGLRConfig) -> dict[str, object]:
    return asdict(config)


def _config_from_document(document: Mapping[str, Any]) -> CFGLRConfig:
    required = set(asdict(CFGLRConfig()))
    if set(document) != required:
        raise ValueError("serialized config keys do not match the frozen schema")
    values = dict(document)
    for name in ("fault_ids", "expert_feature_dims", "f15_ensemble_seeds"):
        if not isinstance(values[name], list):
            raise ValueError(f"serialized {name} must be a JSON list")
        values[name] = tuple(values[name])
    return CFGLRConfig(**values)


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_cf_glr_model(
    model: CFGLRModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save a pickle-free, manifest-validated NPZ model."""

    if not isinstance(model, CFGLRModel):
        raise TypeError("model must be CFGLRModel")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_value: Mapping[str, object] = {} if metadata is None else metadata
    if not isinstance(metadata_value, Mapping):
        raise TypeError("metadata must be a mapping")
    metadata_json = _canonical_json(dict(metadata_value))
    payload: dict[str, np.ndarray] = {
        "schema": np.array(MODEL_SCHEMA, dtype=np.str_),
        "schema_version": np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
        "dtype": np.array("float64", dtype=np.str_),
        "config_json": np.array(_canonical_json(_config_document(model.config)), dtype=np.str_),
        "metadata_json": np.array(metadata_json, dtype=np.str_),
    }
    state_entries: list[dict[str, object]] = []
    state_index = 0
    for fault_id in model.config.fault_ids:
        for member_index, state in enumerate(model.states[fault_id]):
            for name in sorted(state):
                archive_key = f"state_{state_index:04d}"
                array = np.asarray(state[name])
                payload[archive_key] = np.asarray(array, dtype=np.float64)
                state_entries.append(
                    {
                        "archive_key": archive_key,
                        "fault_id": fault_id,
                        "member_index": member_index,
                        "name": name,
                        "shape": list(array.shape),
                        "sha256": _array_sha256(array),
                    }
                )
                state_index += 1
    standardizer_entries: list[dict[str, object]] = []
    for fault_id in model.config.fault_ids:
        for kind, array in (
            ("center", model.feature_centers[fault_id]),
            ("scale", model.feature_scales[fault_id]),
        ):
            archive_key = f"feature_{kind}_f{fault_id:02d}"
            payload[archive_key] = np.asarray(array, dtype=np.float64)
            standardizer_entries.append(
                {
                    "archive_key": archive_key,
                    "fault_id": fault_id,
                    "kind": kind,
                    "shape": list(array.shape),
                    "sha256": _array_sha256(array),
                }
            )
    manifest = {
        "schema": "mate_spc_cf_glr_model_manifest_v1",
        "state_entries": state_entries,
        "feature_standardizer_entries": standardizer_entries,
        "feature_standardization": {
            "fit_scope": (
                "balanced_fit_fold_healthy_plus_target_fault_prefix_and_main_endpoints"
            ),
            "location": "per_expert_feature_median",
            "scale": "1.4826_MAD",
            "scale_floor": FEATURE_SCALE_FLOOR,
            "prefix_half_open_step": [PREFIX_START, MONITOR_START, ENDPOINT_STEP],
            "main_half_open_step": [MONITOR_START, RUN_LENGTH, ENDPOINT_STEP],
        },
    }
    payload["manifest_json"] = np.array(_canonical_json(manifest), dtype=np.str_)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz.tmp", prefix=f".{target.name}.",
            dir=target.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def _scalar_text(array: np.ndarray, *, name: str, limit: int) -> str:
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode value")
    value = str(array.item())
    if len(value) > limit:
        raise ValueError(f"{name} exceeds its safety limit")
    return value


def load_cf_glr_model(
    path: str | os.PathLike[str],
) -> tuple[CFGLRModel, Mapping[str, object]]:
    """Load and fully validate a pickle-free CF-GLR model."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        fixed = {
            "schema",
            "schema_version",
            "dtype",
            "config_json",
            "metadata_json",
            "manifest_json",
        }
        if not fixed.issubset(archive.files):
            raise ValueError("serialized model is missing required fields")
        schema = _scalar_text(archive["schema"], name="schema", limit=128)
        dtype_name = _scalar_text(archive["dtype"], name="dtype", limit=32)
        if schema != MODEL_SCHEMA or dtype_name != "float64":
            raise ValueError("serialized model schema or dtype is invalid")
        version = archive["schema_version"]
        if version.shape != () or version.dtype != np.dtype(np.int64):
            raise ValueError("schema_version must be scalar int64")
        if int(version.item()) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported serialized model version")
        try:
            config_doc = json.loads(
                _scalar_text(archive["config_json"], name="config_json", limit=100_000)
            )
            metadata = json.loads(
                _scalar_text(archive["metadata_json"], name="metadata_json", limit=1_000_000)
            )
            manifest = json.loads(
                _scalar_text(archive["manifest_json"], name="manifest_json", limit=1_000_000)
            )
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("serialized model JSON is invalid") from error
        if not isinstance(config_doc, dict) or not isinstance(metadata, dict):
            raise ValueError("config and metadata JSON must be objects")
        expected_standardization = {
            "fit_scope": (
                "balanced_fit_fold_healthy_plus_target_fault_prefix_and_main_endpoints"
            ),
            "location": "per_expert_feature_median",
            "scale": "1.4826_MAD",
            "scale_floor": FEATURE_SCALE_FLOOR,
            "prefix_half_open_step": [PREFIX_START, MONITOR_START, ENDPOINT_STEP],
            "main_half_open_step": [MONITOR_START, RUN_LENGTH, ENDPOINT_STEP],
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {
                "schema",
                "state_entries",
                "feature_standardizer_entries",
                "feature_standardization",
            }
            or manifest.get("schema") != "mate_spc_cf_glr_model_manifest_v1"
            or not isinstance(manifest.get("state_entries"), list)
            or not isinstance(manifest.get("feature_standardizer_entries"), list)
            or manifest.get("feature_standardization") != expected_standardization
        ):
            raise ValueError("serialized model manifest is invalid")
        entries = manifest["state_entries"]
        archive_keys: set[str] = set()
        state_builder: dict[int, dict[int, dict[str, np.ndarray]]] = {
            fault_id: {} for fault_id in FAULT_IDS
        }
        for raw_entry in entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {
                "archive_key", "fault_id", "member_index", "name", "shape", "sha256"
            }:
                raise ValueError("state manifest entry is malformed")
            key = raw_entry["archive_key"]
            fault_id = raw_entry["fault_id"]
            member_index = raw_entry["member_index"]
            name = raw_entry["name"]
            shape = raw_entry["shape"]
            digest = raw_entry["sha256"]
            if (
                not isinstance(key, str)
                or not key.startswith("state_")
                or key in archive_keys
                or key not in archive.files
                or type(fault_id) is not int
                or fault_id not in FAULT_IDS
                or type(member_index) is not int
                or member_index < 0
                or not isinstance(name, str)
                or not name
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError("state manifest entry has invalid values")
            array = np.asarray(archive[key])
            if (
                array.dtype != np.dtype(np.float64)
                or list(array.shape) != shape
                or not np.isfinite(array).all()
                or _array_sha256(array) != digest
            ):
                raise ValueError(f"serialized parameter {key!r} failed validation")
            archive_keys.add(key)
            member = state_builder[fault_id].setdefault(member_index, {})
            if name in member:
                raise ValueError("duplicate state parameter in manifest")
            member[name] = np.array(array, dtype=np.float64, copy=True, order="C")
        standardizer_builder: dict[int, dict[str, np.ndarray]] = {
            fault_id: {} for fault_id in FAULT_IDS
        }
        for raw_entry in manifest["feature_standardizer_entries"]:
            if not isinstance(raw_entry, dict) or set(raw_entry) != {
                "archive_key", "fault_id", "kind", "shape", "sha256"
            }:
                raise ValueError("feature-standardizer manifest entry is malformed")
            key = raw_entry["archive_key"]
            fault_id = raw_entry["fault_id"]
            kind = raw_entry["kind"]
            shape = raw_entry["shape"]
            digest = raw_entry["sha256"]
            expected_key = (
                f"feature_{kind}_f{fault_id:02d}"
                if type(fault_id) is int and kind in ("center", "scale")
                else None
            )
            if (
                not isinstance(key, str)
                or key != expected_key
                or key in archive_keys
                or key not in archive.files
                or type(fault_id) is not int
                or fault_id not in FAULT_IDS
                or kind not in ("center", "scale")
                or not isinstance(shape, list)
                or any(type(value) is not int or value < 0 for value in shape)
                or not isinstance(digest, str)
                or len(digest) != 64
                or kind in standardizer_builder[fault_id]
            ):
                raise ValueError("feature-standardizer manifest values are invalid")
            array = np.asarray(archive[key])
            if (
                array.dtype != np.dtype(np.float64)
                or list(array.shape) != shape
                or not np.isfinite(array).all()
                or _array_sha256(array) != digest
            ):
                raise ValueError(
                    f"serialized feature standardizer {key!r} failed validation"
                )
            archive_keys.add(key)
            standardizer_builder[fault_id][kind] = np.array(
                array, dtype=np.float64, copy=True, order="C"
            )
        if set(archive.files) != fixed | archive_keys:
            raise ValueError("serialized model contains unmanifested fields")
    config = _config_from_document(config_doc)
    states: dict[int, tuple[Mapping[str, np.ndarray], ...]] = {}
    for fault_id in config.fault_ids:
        expected_count = len(_member_seeds(fault_id, config))
        members = state_builder[fault_id]
        if set(members) != set(range(expected_count)):
            raise ValueError(f"serialized fault {fault_id} member indices are invalid")
        states[fault_id] = tuple(members[index] for index in range(expected_count))
    centers: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    for fault_id in config.fault_ids:
        if set(standardizer_builder[fault_id]) != {"center", "scale"}:
            raise ValueError(f"fault {fault_id} feature standardizer is incomplete")
        centers[fault_id] = standardizer_builder[fault_id]["center"]
        scales[fault_id] = standardizer_builder[fault_id]["scale"]
    model = CFGLRModel(config, states, centers, scales)
    return model, MappingProxyType(dict(metadata))


__all__ = [
    "CFGLRConfig",
    "CFGLRModel",
    "CFGLRTrainingConfig",
    "CFGLRTrainingResult",
    "DEFAULT_PARITY_ATOL",
    "ENDPOINT_STEP",
    "EXPERT_FEATURE_DIMS",
    "ExpertLogitPaths",
    "F15_ENSEMBLE_SEEDS",
    "FAULT_IDS",
    "FEATURE_MAD_CONSISTENCY",
    "FEATURE_SCALE_FLOOR",
    "FEATURE_SCHEMA",
    "MIL_TOP_FRACTION",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "MONITOR_START",
    "PREFIX_START",
    "RUN_LENGTH",
    "assert_cpu_cuda_inference_parity",
    "binary_mil_loss_torch",
    "build_expert_network",
    "fit_cf_glr_cuda",
    "fit_feature_standardizers",
    "initialize_cf_glr_model",
    "load_cf_glr_model",
    "save_cf_glr_model",
]
