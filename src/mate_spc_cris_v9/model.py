"""Causal recurrent conditional Student-t predictor for CRIS-v9.

The model is deliberately data-source agnostic.  It consumes one or more
already-audited runs and never opens experiment data itself.  Its uncertainty
head is a *strictly positive diagonal scale* for each horizon and target.  It
does not implement, and must not be described as implementing, a low-rank or
full covariance matrix.

Alignment is explicit: the prediction stored at ``mu[t, j]`` for horizon
``horizons[j]`` is computed from the context ending at ``t - horizons[j]``.
Consequently it can only depend on inputs whose index is no larger than
``t - horizons[j]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_SCHEMA = "mate_spc_cris_v9_conditional_student_t_gru"
MODEL_SCHEMA_VERSION = 1
CONTEXT_LENGTH = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FORECAST_HORIZONS = (1, 5, 15)
STUDENT_T_NU = 5.0
ADAMW_LEARNING_RATE = 3.0e-4
ADAMW_WEIGHT_DECAY = 1.0e-4
FLOAT_DTYPE = "float32"
DIAGONAL_SCALE_SEMANTICS = "strictly_positive_diagonal_scale"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _positive_int(value: Any, *, name: str) -> int:
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


def _readonly_float32(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float32, order="C", copy=True)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_float64(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _atomic_npz(path: str | os.PathLike[str], payload: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("CRIS-v9 model path must end in .npz")
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


@dataclass(frozen=True)
class CRISConfig:
    """Frozen architecture with configurable observed-data dimensions."""

    target_dim: int
    covariate_dim: int
    delta_time_dim: int = 1
    context_length: int = CONTEXT_LENGTH
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    horizons: tuple[int, ...] = FORECAST_HORIZONS
    degrees_of_freedom: float = STUDENT_T_NU
    minimum_scale: float = 1.0e-4

    def __post_init__(self) -> None:
        for name in (
            "target_dim",
            "covariate_dim",
            "delta_time_dim",
            "context_length",
            "hidden_size",
            "num_layers",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        horizons = tuple(_positive_int(value, name="horizon") for value in self.horizons)
        if horizons != FORECAST_HORIZONS:
            raise ValueError(f"horizons must be exactly {FORECAST_HORIZONS}")
        if self.context_length != CONTEXT_LENGTH:
            raise ValueError(f"context_length must be exactly {CONTEXT_LENGTH}")
        if self.hidden_size != HIDDEN_SIZE or self.num_layers != NUM_LAYERS:
            raise ValueError("CRIS-v9 requires a two-layer GRU with hidden size 64")
        nu = _finite_positive(self.degrees_of_freedom, name="degrees_of_freedom")
        if nu != STUDENT_T_NU:
            raise ValueError(f"degrees_of_freedom must be exactly {STUDENT_T_NU:g}")
        minimum_scale = _finite_positive(self.minimum_scale, name="minimum_scale")
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "degrees_of_freedom", nu)
        object.__setattr__(self, "minimum_scale", minimum_scale)

    @property
    def input_dim(self) -> int:
        # x, covariates, observation mask, and elapsed-time features.
        return self.target_dim + self.covariate_dim + self.target_dim + self.delta_time_dim

    @property
    def output_dim(self) -> int:
        # Each horizon/target pair emits mu and log_scale.
        return len(self.horizons) * self.target_dim * 2

    def architecture_spec(self) -> dict[str, object]:
        return {
            "model": "two_layer_causal_gru_conditional_student_t",
            "input_order": ["target_history", "covariates", "target_mask", "delta_t"],
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "covariate_dim": self.covariate_dim,
            "delta_time_dim": self.delta_time_dim,
            "context_length": self.context_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "horizons": list(self.horizons),
            "degrees_of_freedom": self.degrees_of_freedom,
            "distribution_head": {
                "parameters": ["mu", "log_scale"],
                "scale_semantics": DIAGONAL_SCALE_SEMANTICS,
                "low_rank_covariance": False,
            },
            "dtype": FLOAT_DTYPE,
        }


@dataclass(frozen=True)
class CRISTrainingConfig:
    """Deterministic, fixed-epoch AdamW training contract."""

    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = ADAMW_LEARNING_RATE
    weight_decay: float = ADAMW_WEIGHT_DECAY
    gradient_clip_norm: float = 1.0
    seed: int = 20260915
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "epochs", _positive_int(self.epochs, name="epochs"))
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, name="batch_size"))
        object.__setattr__(self, "seed", _positive_int(self.seed, name="seed"))
        learning_rate = _finite_positive(self.learning_rate, name="learning_rate")
        weight_decay = _finite_positive(self.weight_decay, name="weight_decay")
        gradient_clip_norm = _finite_positive(
            self.gradient_clip_norm, name="gradient_clip_norm"
        )
        if learning_rate != ADAMW_LEARNING_RATE:
            raise ValueError(f"learning_rate is frozen at {ADAMW_LEARNING_RATE:g}")
        if weight_decay != ADAMW_WEIGHT_DECAY:
            raise ValueError(f"weight_decay is frozen at {ADAMW_WEIGHT_DECAY:g}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "gradient_clip_norm", gradient_clip_norm)


@dataclass(frozen=True)
class CRISNormalization:
    """Training-only normalization constants bound to the fitted state."""

    target_center: np.ndarray
    target_scale: np.ndarray
    covariate_center: np.ndarray
    covariate_scale: np.ndarray
    delta_log_center: np.ndarray
    delta_log_scale: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "target_center",
            "target_scale",
            "covariate_center",
            "covariate_scale",
            "delta_log_center",
            "delta_log_scale",
        ):
            object.__setattr__(
                self, name, _readonly_float32(getattr(self, name), ndim=1, name=name)
            )
        if np.any(self.target_scale <= 0.0):
            raise ValueError("target_scale must be strictly positive")
        if np.any(self.covariate_scale <= 0.0):
            raise ValueError("covariate_scale must be strictly positive")
        if np.any(self.delta_log_scale <= 0.0):
            raise ValueError("delta_log_scale must be strictly positive")
        if self.target_center.shape != self.target_scale.shape:
            raise ValueError("target normalization center and scale shapes must match")
        if self.covariate_center.shape != self.covariate_scale.shape:
            raise ValueError("covariate normalization center and scale shapes must match")
        if self.delta_log_center.shape != self.delta_log_scale.shape:
            raise ValueError("delta_t normalization center and scale shapes must match")


@dataclass(frozen=True)
class DeviceAttestation:
    """Runtime facts captured when the fitted weights were produced."""

    requested_device: str
    source_device: str
    torch_version: str
    cuda_runtime: str
    cuda_device_name: str
    dtype: str = FLOAT_DTYPE
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        for name in (
            "requested_device",
            "source_device",
            "torch_version",
            "cuda_runtime",
            "cuda_device_name",
            "dtype",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text")
        if not self.requested_device or not self.source_device or not self.torch_version:
            raise ValueError("device and torch attestation fields must be nonempty")
        if self.dtype != FLOAT_DTYPE or self.deterministic_algorithms is not True:
            raise ValueError("CRIS-v9 attestation requires deterministic float32 training")
        if self.source_device != "cpu" and not self.source_device.startswith("cuda:"):
            raise ValueError("source_device must identify cpu or a concrete CUDA index")
        if self.source_device == "cpu":
            if self.requested_device != "cpu" or self.cuda_runtime or self.cuda_device_name:
                raise ValueError("CPU attestation fields are internally inconsistent")
        else:
            if not (
                self.requested_device == "cuda"
                or self.requested_device.startswith("cuda:")
            ):
                raise ValueError("CUDA source requires a CUDA requested_device")
            if not self.cuda_runtime or not self.cuda_device_name:
                raise ValueError("CUDA attestation requires runtime and device name")
            try:
                source_index = int(self.source_device.removeprefix("cuda:"))
            except ValueError as error:
                raise ValueError("source CUDA device index must be an integer") from error
            if source_index < 0 or self.source_device != f"cuda:{source_index}":
                raise ValueError("source_device must use a canonical nonnegative CUDA index")
            if self.requested_device.startswith("cuda:"):
                try:
                    requested_index = int(self.requested_device.removeprefix("cuda:"))
                except ValueError as error:
                    raise ValueError("CUDA device indices must be integers") from error
                if (
                    requested_index < 0
                    or self.requested_device != f"cuda:{requested_index}"
                    or requested_index != source_index
                ):
                    raise ValueError("requested and source CUDA indices disagree")


@dataclass(frozen=True)
class PredictionResult:
    """Physically scaled, horizon-aligned diagonal distribution parameters."""

    mu: np.ndarray
    scale: np.ndarray
    valid: np.ndarray
    horizons: tuple[int, ...]
    source_device: str

    def __post_init__(self) -> None:
        mu = np.array(self.mu, dtype=np.float32, order="C", copy=True)
        scale = np.array(self.scale, dtype=np.float32, order="C", copy=True)
        valid = np.array(self.valid, dtype=bool, order="C", copy=True)
        horizons = tuple(_positive_int(value, name="horizon") for value in self.horizons)
        if mu.ndim != 3 or scale.shape != mu.shape or valid.shape != mu.shape:
            raise ValueError("mu, scale, and valid must have the same (T,H,D) shape")
        if mu.shape[1] != len(horizons):
            raise ValueError("prediction horizon dimension does not match horizons")
        if np.any(~np.isfinite(mu[valid])) or np.any(~np.isfinite(scale[valid])):
            raise ValueError("valid predictions must be finite")
        if np.any(scale[valid] <= 0.0):
            raise ValueError("valid diagonal scales must be strictly positive")
        if np.any(np.isfinite(mu[~valid])) or np.any(np.isfinite(scale[~valid])):
            raise ValueError("invalid predictions must be NaN")
        if self.source_device != "cpu" and not self.source_device.startswith("cuda:"):
            raise ValueError("source_device must identify CPU or a concrete CUDA index")
        mu.setflags(write=False)
        scale.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "horizons", horizons)

    @property
    def log_scale(self) -> np.ndarray:
        result = np.full(self.scale.shape, np.nan, dtype=np.float32)
        result[self.valid] = np.log(self.scale[self.valid])
        result.setflags(write=False)
        return result


_CACHED_NETWORK_CLASS: Any | None = None


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:  # pragma: no cover - runtime dependent
        raise RuntimeError("PyTorch is required for CRIS-v9") from error
    return torch


def _network_class() -> Any:
    global _CACHED_NETWORK_CLASS
    if _CACHED_NETWORK_CLASS is not None:
        return _CACHED_NETWORK_CLASS
    torch = _require_torch()
    nn = torch.nn

    class ConditionalStudentTGRU(nn.Module):
        def __init__(self, config: CRISConfig):
            super().__init__()
            self.config = config
            self.gru = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
            self.head = nn.Linear(config.hidden_size, config.output_dim)
            with torch.no_grad():
                bias = self.head.bias.view(len(config.horizons), config.target_dim, 2)
                bias[..., 0].zero_()
                bias[..., 1].fill_(-0.5)

        def forward(self, windows: Any) -> tuple[Any, Any]:
            hidden, _ = self.gru(windows)
            parameters = self.head(hidden[:, -1]).view(
                windows.shape[0], len(self.config.horizons), self.config.target_dim, 2
            )
            return parameters[..., 0], parameters[..., 1]

    _CACHED_NETWORK_CLASS = ConditionalStudentTGRU
    return ConditionalStudentTGRU


def _expected_state_shapes(config: CRISConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(config.num_layers):
        input_size = config.input_dim if layer == 0 else hidden
        shapes[f"gru.weight_ih_l{layer}"] = (3 * hidden, input_size)
        shapes[f"gru.weight_hh_l{layer}"] = (3 * hidden, hidden)
        shapes[f"gru.bias_ih_l{layer}"] = (3 * hidden,)
        shapes[f"gru.bias_hh_l{layer}"] = (3 * hidden,)
    shapes["head.weight"] = (config.output_dim, hidden)
    shapes["head.bias"] = (config.output_dim,)
    return shapes


def _resolve_device(torch: Any, requested: str) -> tuple[Any, str]:
    try:
        target = torch.device(requested)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid torch device {requested!r}") from error
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = torch.cuda.current_device() if target.index is None else int(target.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is not available")
        concrete = torch.device(f"cuda:{index}")
        return concrete, str(concrete)
    if target.type != "cpu":
        raise ValueError("CRIS-v9 supports only CPU and CUDA devices")
    return torch.device("cpu"), "cpu"


def _prepare_workspace_for_requested_device(requested: Any) -> bool:
    """Bind the CUBLAS workspace before any CUDA availability query/context."""

    if not isinstance(requested, str) or not requested:
        raise ValueError("device must be a nonempty string")
    configured_now = False
    if requested == "cuda" or requested.startswith("cuda:"):
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8"):
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen CRIS-v9 value :4096:8"
            )
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            configured_now = True
    return configured_now


def _reject_late_cuda_workspace_binding(torch: Any, configured_now: bool) -> None:
    if configured_now and torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was already initialized before CUBLAS_WORKSPACE_CONFIG was bound"
        )


def _configure_deterministic_runtime(torch: Any, device: Any, *, seed: int | None) -> None:
    """Apply the same deterministic contract for fit and fresh-process inference."""

    if device.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in (None, ":4096:8"):
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen CRIS-v9 value :4096:8"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _as_run_list(
    values: Any,
    *,
    feature_dim: int,
    name: str,
    dtype: Any,
) -> list[np.ndarray]:
    if isinstance(values, np.ndarray):
        raw_runs: Sequence[Any]
        if values.ndim == 2:
            raw_runs = (values,)
        elif values.ndim == 3:
            raw_runs = tuple(values[index] for index in range(values.shape[0]))
        else:
            raise ValueError(f"{name} must have shape (T,D), (N,T,D), or be a sequence")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_runs = values
    else:
        raise ValueError(f"{name} must be an ndarray or sequence of ndarrays")
    if not raw_runs:
        raise ValueError(f"{name} must contain at least one run")
    runs: list[np.ndarray] = []
    for index, value in enumerate(raw_runs):
        array = np.array(value, dtype=dtype, order="C", copy=True)
        if array.ndim != 2 or array.shape[1] != feature_dim:
            raise ValueError(f"{name}[{index}] must have shape (T,{feature_dim})")
        runs.append(array)
    return runs


def _as_delta_runs(values: Any, *, config: CRISConfig) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = np.array(values, dtype=np.float32, copy=True)[:, None]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, np.ndarray)):
        converted: list[np.ndarray] = []
        for value in values:
            array = np.asarray(value)
            if array.ndim == 1:
                array = array[:, None]
            converted.append(array)
        values = converted
    return _as_run_list(
        values, feature_dim=config.delta_time_dim, name="delta_t", dtype=np.float32
    )


def _mask_runs(values: Any, *, config: CRISConfig) -> list[np.ndarray]:
    raw = _as_run_list(values, feature_dim=config.target_dim, name="mask", dtype=np.float32)
    result: list[np.ndarray] = []
    for index, array in enumerate(raw):
        if not np.isfinite(array).all() or np.any((array != 0.0) & (array != 1.0)):
            raise ValueError(f"mask[{index}] must contain only zero and one")
        result.append(np.array(array, dtype=bool, order="C", copy=True))
    return result


def _coerce_inputs(
    x: Any,
    c: Any,
    mask: Any | None,
    delta_t: Any | None,
    *,
    config: CRISConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    x_runs = _as_run_list(x, feature_dim=config.target_dim, name="x", dtype=np.float32)
    c_runs = _as_run_list(
        c, feature_dim=config.covariate_dim, name="c", dtype=np.float32
    )
    if mask is None:
        mask_runs = [np.array(np.isfinite(run), dtype=bool, order="C", copy=True) for run in x_runs]
    else:
        mask_runs = _mask_runs(mask, config=config)
    if delta_t is None:
        delta_runs = [
            np.zeros((run.shape[0], config.delta_time_dim), dtype=np.float32)
            for run in x_runs
        ]
    else:
        delta_runs = _as_delta_runs(delta_t, config=config)
    count = len(x_runs)
    if len(c_runs) != count or len(mask_runs) != count or len(delta_runs) != count:
        raise ValueError("x, c, mask, and delta_t must contain the same number of runs")
    minimum_length = config.context_length + max(config.horizons)
    for index, (x_run, c_run, mask_run, delta_run) in enumerate(
        zip(x_runs, c_runs, mask_runs, delta_runs, strict=True)
    ):
        length = x_run.shape[0]
        if c_run.shape[0] != length or mask_run.shape[0] != length or delta_run.shape[0] != length:
            raise ValueError(f"run {index} inputs do not have the same time length")
        if length < minimum_length:
            raise ValueError(f"run {index} is shorter than context_length + max(horizons)")
        if not np.isfinite(c_run).all():
            raise ValueError(f"c[{index}] must be finite")
        if not np.isfinite(delta_run).all() or np.any(delta_run < 0.0):
            raise ValueError(f"delta_t[{index}] must be finite and nonnegative")
        if np.any(~np.isfinite(x_run) & mask_run):
            raise ValueError(f"x[{index}] has a nonfinite value marked as observed")
    return x_runs, c_runs, mask_runs, delta_runs


def _fit_normalization(
    x_runs: Sequence[np.ndarray],
    c_runs: Sequence[np.ndarray],
    mask_runs: Sequence[np.ndarray],
    delta_runs: Sequence[np.ndarray],
) -> CRISNormalization:
    target_dim = x_runs[0].shape[1]
    target_center = np.empty(target_dim, dtype=np.float32)
    target_scale = np.empty(target_dim, dtype=np.float32)
    for feature in range(target_dim):
        observed = np.concatenate(
            [run[:, feature][mask[:, feature]] for run, mask in zip(x_runs, mask_runs, strict=True)]
        )
        if observed.size == 0:
            raise ValueError(f"target feature {feature} is never observed")
        target_center[feature] = np.mean(observed, dtype=np.float64)
        target_scale[feature] = max(float(np.std(observed, dtype=np.float64)), 1.0e-6)
    covariates = np.concatenate(c_runs, axis=0).astype(np.float64, copy=False)
    covariate_center = np.mean(covariates, axis=0)
    covariate_scale = np.maximum(np.std(covariates, axis=0), 1.0e-6)
    delta_logs = np.log1p(np.concatenate(delta_runs, axis=0).astype(np.float64, copy=False))
    delta_log_center = np.mean(delta_logs, axis=0)
    delta_log_scale = np.maximum(np.std(delta_logs, axis=0), 1.0e-6)
    return CRISNormalization(
        target_center=target_center,
        target_scale=target_scale,
        covariate_center=covariate_center,
        covariate_scale=covariate_scale,
        delta_log_center=delta_log_center,
        delta_log_scale=delta_log_scale,
    )


def _normalized_features(
    x: np.ndarray,
    c: np.ndarray,
    mask: np.ndarray,
    delta_t: np.ndarray,
    normalization: CRISNormalization,
) -> np.ndarray:
    filled_x = np.where(mask, x, normalization.target_center[None, :])
    x_part = (filled_x - normalization.target_center[None, :]) / normalization.target_scale[
        None, :
    ]
    c_part = (c - normalization.covariate_center[None, :]) / normalization.covariate_scale[
        None, :
    ]
    delta_log = np.log1p(delta_t)
    delta_part = (delta_log - normalization.delta_log_center[None, :]) / (
        normalization.delta_log_scale[None, :]
    )
    # Concatenation creates owned, writable C-contiguous memory even when every
    # caller array is read-only.
    return np.array(
        np.concatenate((x_part, c_part, mask.astype(np.float32), delta_part), axis=1),
        dtype=np.float32,
        order="C",
        copy=True,
    )


def _student_t_nll(
    torch: Any,
    target: Any,
    mu: Any,
    log_scale: Any,
    observed: Any,
    *,
    config: CRISConfig,
) -> Any:
    bounded_log_scale = torch.clamp(log_scale, min=-8.0, max=5.0)
    scale = torch.exp(bounded_log_scale) + config.minimum_scale
    standardized = (target - mu) / scale
    per_element = torch.log(scale) + torch.log1p(
        standardized.square() / config.degrees_of_freedom
    ) * ((config.degrees_of_freedom + 1.0) / 2.0)
    weights = observed.to(dtype=per_element.dtype)
    return (per_element * weights).sum() / torch.clamp(weights.sum(), min=1.0)


def _state_to_numpy(network: Any) -> dict[str, np.ndarray]:
    return {
        name: np.array(tensor.detach().cpu().numpy(), dtype=np.float32, order="C", copy=True)
        for name, tensor in network.state_dict().items()
    }


def _load_network(model: "CRISModel", *, device: Any) -> Any:
    torch = _require_torch()
    network = _network_class()(model.config).to(device=device, dtype=torch.float32)
    state = {
        name: torch.from_numpy(np.array(value, dtype=np.float32, order="C", copy=True))
        for name, value in model.state_dict.items()
    }
    network.load_state_dict(state, strict=True)
    network.eval()
    return network


@dataclass(frozen=True)
class CRISModel:
    """Fitted CRIS-v9 conditional predictor with safe portable state."""

    config: CRISConfig
    training_config: CRISTrainingConfig
    normalization: CRISNormalization
    state_dict: Mapping[str, np.ndarray]
    loss_history: np.ndarray
    attestation: DeviceAttestation
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.config, CRISConfig):
            raise ValueError("config must be CRISConfig")
        if not isinstance(self.training_config, CRISTrainingConfig):
            raise ValueError("training_config must be CRISTrainingConfig")
        if not isinstance(self.normalization, CRISNormalization):
            raise ValueError("normalization must be CRISNormalization")
        if not isinstance(self.attestation, DeviceAttestation):
            raise ValueError("attestation must be DeviceAttestation")
        if self.attestation.requested_device != self.training_config.device:
            raise ValueError("training_config.device and device attestation disagree")
        expected = _expected_state_shapes(self.config)
        if set(self.state_dict) != set(expected):
            raise ValueError("state_dict keys do not match the CRIS-v9 architecture")
        owned_state: dict[str, np.ndarray] = {}
        for name in sorted(expected):
            raw = np.asarray(self.state_dict[name])
            if raw.dtype != np.dtype(np.float32) or raw.shape != expected[name]:
                raise ValueError(f"state tensor {name!r} has wrong dtype or shape")
            value = _readonly_float32(raw, ndim=len(expected[name]), name=f"state_dict[{name!r}]")
            owned_state[name] = value
        history = _readonly_float64(self.loss_history, ndim=1, name="loss_history")
        if history.size != self.training_config.epochs:
            raise ValueError("loss_history length must equal the fixed epoch count")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        metadata = json.loads(_canonical_json(dict(self.metadata)))
        object.__setattr__(self, "state_dict", MappingProxyType(owned_state))
        object.__setattr__(self, "loss_history", history)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        if self.normalization.target_center.shape != (self.config.target_dim,):
            raise ValueError("target normalization dimension does not match config")
        if self.normalization.covariate_center.shape != (self.config.covariate_dim,):
            raise ValueError("covariate normalization dimension does not match config")
        if self.normalization.delta_log_center.shape != (self.config.delta_time_dim,):
            raise ValueError("delta_t normalization dimension does not match config")

    @property
    def source_device(self) -> str:
        return self.attestation.source_device

    @classmethod
    def fit(
        cls,
        x: Any,
        c: Any,
        mask: Any | None = None,
        delta_t: Any | None = None,
        *,
        config: CRISConfig,
        training_config: CRISTrainingConfig | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "CRISModel":
        """Fit on one run, a stacked ``(N,T,D)`` array, or run sequences.

        Each run supplies ``x[T,target_dim]``, ``c[T,covariate_dim]``, an
        optional target observation ``mask[T,target_dim]``, and optional
        ``delta_t[T,delta_time_dim]``.  No validation split, early stopping, or
        data-dependent epoch selection occurs: exactly ``epochs`` passes run.
        """

        actual_training = training_config or CRISTrainingConfig()
        x_runs, c_runs, mask_runs, delta_runs = _coerce_inputs(
            x, c, mask, delta_t, config=config
        )
        normalization = _fit_normalization(x_runs, c_runs, mask_runs, delta_runs)
        feature_runs = [
            _normalized_features(x_run, c_run, mask_run, delta_run, normalization)
            for x_run, c_run, mask_run, delta_run in zip(
                x_runs, c_runs, mask_runs, delta_runs, strict=True
            )
        ]
        max_horizon = max(config.horizons)
        sample_index: list[tuple[int, int]] = []
        for run_index, run in enumerate(x_runs):
            sample_index.extend(
                (run_index, anchor)
                for anchor in range(config.context_length - 1, run.shape[0] - max_horizon)
            )
        if not sample_index:
            raise ValueError("no complete causal training windows are available")

        configured_now = _prepare_workspace_for_requested_device(actual_training.device)
        torch = _require_torch()
        _reject_late_cuda_workspace_binding(torch, configured_now)
        target_device, source_device = _resolve_device(torch, actual_training.device)
        _configure_deterministic_runtime(
            torch, target_device, seed=actual_training.seed
        )

        network = _network_class()(config).to(device=target_device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=actual_training.learning_rate,
            weight_decay=actual_training.weight_decay,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(actual_training.seed)
        losses: list[float] = []
        network.train()
        for _epoch in range(actual_training.epochs):
            order = torch.randperm(len(sample_index), generator=generator).tolist()
            epoch_loss_sum = 0.0
            epoch_weight = 0
            for offset in range(0, len(order), actual_training.batch_size):
                chosen = [sample_index[order[position]] for position in range(
                    offset, min(offset + actual_training.batch_size, len(order))
                )]
                windows = np.stack(
                    [
                        feature_runs[run_index][
                            anchor - config.context_length + 1 : anchor + 1
                        ]
                        for run_index, anchor in chosen
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
                target_values = np.stack(
                    [
                        np.stack(
                            [x_runs[run_index][anchor + horizon] for horizon in config.horizons]
                        )
                        for run_index, anchor in chosen
                    ],
                    axis=0,
                )
                target_observed = np.stack(
                    [
                        np.stack(
                            [
                                mask_runs[run_index][anchor + horizon]
                                for horizon in config.horizons
                            ]
                        )
                        for run_index, anchor in chosen
                    ],
                    axis=0,
                )
                observed_count = int(np.count_nonzero(target_observed))
                if observed_count == 0:
                    continue
                normalized_target = (
                    target_values - normalization.target_center[None, None, :]
                ) / normalization.target_scale[None, None, :]
                normalized_target = np.where(target_observed, normalized_target, 0.0)

                # Every conversion owns writable memory, including when the
                # original user arrays were read-only views.
                window_tensor = torch.from_numpy(
                    np.array(windows, dtype=np.float32, order="C", copy=True)
                ).to(device=target_device)
                target_tensor = torch.from_numpy(
                    np.array(normalized_target, dtype=np.float32, order="C", copy=True)
                ).to(device=target_device)
                observed_tensor = torch.from_numpy(
                    np.array(target_observed, dtype=bool, order="C", copy=True)
                ).to(device=target_device)
                optimizer.zero_grad(set_to_none=True)
                mu, log_scale = network(window_tensor)
                loss = _student_t_nll(
                    torch,
                    target_tensor,
                    mu,
                    log_scale,
                    observed_tensor,
                    config=config,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), max_norm=actual_training.gradient_clip_norm
                )
                optimizer.step()
                epoch_loss_sum += float(loss.detach().cpu()) * observed_count
                epoch_weight += observed_count
            if epoch_weight == 0:
                raise ValueError("no observed future targets are available for training")
            losses.append(epoch_loss_sum / epoch_weight)

        if target_device.type == "cuda":
            cuda_runtime = "" if torch.version.cuda is None else str(torch.version.cuda)
            cuda_device_name = str(torch.cuda.get_device_name(target_device))
        else:
            cuda_runtime = ""
            cuda_device_name = ""
        attestation = DeviceAttestation(
            requested_device=actual_training.device,
            source_device=source_device,
            torch_version=str(torch.__version__),
            cuda_runtime=cuda_runtime,
            cuda_device_name=cuda_device_name,
        )
        return cls(
            config=config,
            training_config=actual_training,
            normalization=normalization,
            state_dict=_state_to_numpy(network),
            loss_history=np.asarray(losses, dtype=np.float64),
            attestation=attestation,
            metadata={} if metadata is None else metadata,
        )

    def predict(
        self,
        x: Any,
        c: Any,
        mask: Any | None = None,
        delta_t: Any | None = None,
        *,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> PredictionResult:
        """Return ``(T,H,D)`` parameters with strict causal horizon alignment."""

        batch_size = _positive_int(batch_size, name="batch_size")
        x_runs, c_runs, mask_runs, delta_runs = _coerce_inputs(
            x, c, mask, delta_t, config=self.config
        )
        if len(x_runs) != 1:
            raise ValueError("predict accepts exactly one run")
        features = _normalized_features(
            x_runs[0], c_runs[0], mask_runs[0], delta_runs[0], self.normalization
        )
        # Configure CUBLAS before the first CUDA context can be created, then
        # re-enable deterministic algorithms even in a fresh load-only process.
        configured_now = _prepare_workspace_for_requested_device(device)
        torch = _require_torch()
        _reject_late_cuda_workspace_binding(torch, configured_now)
        target_device, source_device = _resolve_device(torch, device)
        _configure_deterministic_runtime(torch, target_device, seed=None)
        network = _load_network(self, device=target_device)
        anchors = np.arange(
            self.config.context_length - 1, features.shape[0], dtype=np.int64
        )
        output_mu = np.full(
            (features.shape[0], len(self.config.horizons), self.config.target_dim),
            np.nan,
            dtype=np.float32,
        )
        output_scale = np.full_like(output_mu, np.nan)
        valid = np.zeros(output_mu.shape, dtype=bool)
        with torch.inference_mode():
            for offset in range(0, anchors.size, batch_size):
                batch_anchors = anchors[offset : offset + batch_size]
                windows = np.stack(
                    [
                        features[
                            anchor - self.config.context_length + 1 : anchor + 1
                        ]
                        for anchor in batch_anchors
                    ],
                    axis=0,
                )
                tensor = torch.from_numpy(
                    np.array(windows, dtype=np.float32, order="C", copy=True)
                ).to(device=target_device)
                normalized_mu, normalized_log_scale = network(tensor)
                normalized_scale = torch.exp(
                    torch.clamp(normalized_log_scale, min=-8.0, max=5.0)
                ) + self.config.minimum_scale
                batch_mu = normalized_mu.detach().cpu().numpy()
                batch_scale = normalized_scale.detach().cpu().numpy()
                batch_mu = (
                    batch_mu * self.normalization.target_scale[None, None, :]
                    + self.normalization.target_center[None, None, :]
                )
                batch_scale = (
                    batch_scale * self.normalization.target_scale[None, None, :]
                )
                for local_index, anchor in enumerate(batch_anchors):
                    for horizon_index, horizon in enumerate(self.config.horizons):
                        target_index = int(anchor) + horizon
                        if target_index >= features.shape[0]:
                            continue
                        output_mu[target_index, horizon_index] = batch_mu[
                            local_index, horizon_index
                        ]
                        output_scale[target_index, horizon_index] = batch_scale[
                            local_index, horizon_index
                        ]
                        valid[target_index, horizon_index] = True
        return PredictionResult(
            mu=output_mu,
            scale=output_scale,
            valid=valid,
            horizons=self.config.horizons,
            source_device=source_device,
        )

    def innovations(
        self,
        x: Any,
        c: Any | None = None,
        mask: Any | None = None,
        delta_t: Any | None = None,
        *,
        prediction: PredictionResult | None = None,
        device: str = "cpu",
        batch_size: int = 512,
    ) -> np.ndarray:
        """Return unit-variance Student-t innovations in aligned ``(T,H,D)`` form."""

        x_array = np.array(x, dtype=np.float32, order="C", copy=True)
        if x_array.ndim != 2 or x_array.shape[1] != self.config.target_dim:
            raise ValueError(f"x must have shape (T,{self.config.target_dim})")
        if prediction is None:
            if c is None:
                raise ValueError("c is required when prediction is not supplied")
            prediction = self.predict(
                x_array,
                c,
                mask,
                delta_t,
                device=device,
                batch_size=batch_size,
            )
        if prediction.mu.shape != (
            x_array.shape[0],
            len(self.config.horizons),
            self.config.target_dim,
        ) or prediction.horizons != self.config.horizons:
            raise ValueError("prediction is not aligned with x and this model")
        if mask is None:
            observed = np.isfinite(x_array)
        else:
            mask_runs = _mask_runs(mask, config=self.config)
            if len(mask_runs) != 1 or mask_runs[0].shape != x_array.shape:
                raise ValueError("mask is not aligned with x")
            observed = mask_runs[0]
            if np.any(~np.isfinite(x_array) & observed):
                raise ValueError("x has a nonfinite value marked as observed")
        usable = prediction.valid & observed[:, None, :]
        result = np.full(prediction.mu.shape, np.nan, dtype=np.float32)
        factor = math.sqrt(
            (self.config.degrees_of_freedom - 2.0) / self.config.degrees_of_freedom
        )
        broadcast_x = np.broadcast_to(x_array[:, None, :], prediction.mu.shape)
        result[usable] = (
            factor
            * (broadcast_x[usable] - prediction.mu[usable])
            / prediction.scale[usable]
        )
        result.setflags(write=False)
        return result

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        """Atomically save architecture, normalization, weights, and attestation."""

        combined_metadata = dict(self.metadata)
        if metadata is not None:
            combined_metadata.update(dict(metadata))
        state_manifest: list[dict[str, object]] = []
        payload: dict[str, np.ndarray] = {
            "schema": np.asarray(MODEL_SCHEMA),
            "schema_version": np.asarray(MODEL_SCHEMA_VERSION, dtype=np.int64),
            "config_json": np.asarray(_canonical_json(asdict(self.config))),
            "architecture_json": np.asarray(_canonical_json(self.config.architecture_spec())),
            "training_json": np.asarray(_canonical_json(asdict(self.training_config))),
            "attestation_json": np.asarray(_canonical_json(asdict(self.attestation))),
            "metadata_json": np.asarray(_canonical_json(combined_metadata)),
            "target_center": np.array(self.normalization.target_center, copy=True),
            "target_scale": np.array(self.normalization.target_scale, copy=True),
            "covariate_center": np.array(self.normalization.covariate_center, copy=True),
            "covariate_scale": np.array(self.normalization.covariate_scale, copy=True),
            "delta_log_center": np.array(self.normalization.delta_log_center, copy=True),
            "delta_log_scale": np.array(self.normalization.delta_log_scale, copy=True),
            "loss_history": np.array(self.loss_history, copy=True),
        }
        for index, name in enumerate(sorted(self.state_dict)):
            storage_key = f"state_{index:03d}"
            value = np.array(self.state_dict[name], dtype=np.float32, order="C", copy=True)
            payload[storage_key] = value
            state_manifest.append(
                {
                    "name": name,
                    "storage_key": storage_key,
                    "shape": list(value.shape),
                    "dtype": "float32",
                }
            )
        payload["state_manifest_json"] = np.asarray(_canonical_json(state_manifest))
        return _atomic_npz(path, payload)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "CRISModel":
        """Load a no-pickle NPZ and reject every unexpected field or tensor."""

        target = Path(path)
        with np.load(target, allow_pickle=False) as archive:
            fixed_keys = {
                "schema",
                "schema_version",
                "config_json",
                "architecture_json",
                "training_json",
                "attestation_json",
                "metadata_json",
                "target_center",
                "target_scale",
                "covariate_center",
                "covariate_scale",
                "delta_log_center",
                "delta_log_scale",
                "loss_history",
                "state_manifest_json",
            }

            def scalar_text(name: str) -> str:
                value = archive[name]
                if value.shape != () or value.dtype.kind not in "US":
                    raise ValueError(f"{name} must be a scalar text array")
                return str(value.item())

            if scalar_text("schema") != MODEL_SCHEMA:
                raise ValueError("wrong CRIS-v9 model schema")
            version = archive["schema_version"]
            if version.shape != () or version.dtype != np.dtype(np.int64):
                raise ValueError("schema_version must be scalar int64")
            if int(version) != MODEL_SCHEMA_VERSION:
                raise ValueError("unsupported CRIS-v9 model schema version")
            try:
                config_payload = json.loads(scalar_text("config_json"))
                training_payload = json.loads(scalar_text("training_json"))
                attestation_payload = json.loads(scalar_text("attestation_json"))
                metadata_payload = json.loads(scalar_text("metadata_json"))
                architecture_payload = json.loads(scalar_text("architecture_json"))
                manifest = json.loads(scalar_text("state_manifest_json"))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError("invalid JSON in CRIS-v9 model archive") from error
            config_payload["horizons"] = tuple(config_payload["horizons"])
            config = CRISConfig(**config_payload)
            if architecture_payload != config.architecture_spec():
                raise ValueError("architecture_json does not match config_json")
            training = CRISTrainingConfig(**training_payload)
            attestation = DeviceAttestation(**attestation_payload)
            if not isinstance(metadata_payload, dict) or not isinstance(manifest, list):
                raise ValueError("metadata and state manifest have wrong JSON types")
            expected_shapes = _expected_state_shapes(config)
            if len(manifest) != len(expected_shapes):
                raise ValueError("state manifest tensor count is wrong")
            state: dict[str, np.ndarray] = {}
            storage_keys: set[str] = set()
            for item in manifest:
                if not isinstance(item, dict) or set(item) != {
                    "name",
                    "storage_key",
                    "shape",
                    "dtype",
                }:
                    raise ValueError("invalid state manifest entry")
                name = item["name"]
                storage_key = item["storage_key"]
                if (
                    not isinstance(name, str)
                    or name not in expected_shapes
                    or name in state
                    or not isinstance(storage_key, str)
                    or storage_key in storage_keys
                    or item["dtype"] != "float32"
                    or item["shape"] != list(expected_shapes[name])
                ):
                    raise ValueError("state manifest does not match architecture")
                value = archive[storage_key]
                if value.dtype != np.dtype(np.float32) or value.shape != expected_shapes[name]:
                    raise ValueError(f"state tensor {name!r} has wrong dtype or shape")
                state[name] = np.array(value, dtype=np.float32, order="C", copy=True)
                storage_keys.add(storage_key)
            if set(archive.files) != fixed_keys | storage_keys:
                raise ValueError("CRIS-v9 model archive has missing or unexpected fields")
            normalization_shapes = {
                "target_center": (config.target_dim,),
                "target_scale": (config.target_dim,),
                "covariate_center": (config.covariate_dim,),
                "covariate_scale": (config.covariate_dim,),
                "delta_log_center": (config.delta_time_dim,),
                "delta_log_scale": (config.delta_time_dim,),
            }
            normalization_values: dict[str, np.ndarray] = {}
            for name, shape in normalization_shapes.items():
                value = archive[name]
                if value.dtype != np.dtype(np.float32) or value.shape != shape:
                    raise ValueError(f"{name} has wrong dtype or shape")
                normalization_values[name] = np.array(
                    value, dtype=np.float32, order="C", copy=True
                )
            loss_value = archive["loss_history"]
            if loss_value.dtype != np.dtype(np.float64) or loss_value.shape != (
                training.epochs,
            ):
                raise ValueError("loss_history has wrong dtype or shape")
            normalization = CRISNormalization(**normalization_values)
            loss_history = np.array(loss_value, dtype=np.float64, copy=True)
        return cls(
            config=config,
            training_config=training,
            normalization=normalization,
            state_dict=state,
            loss_history=loss_history,
            attestation=attestation,
            metadata=metadata_payload,
        )


__all__ = [
    "ADAMW_LEARNING_RATE",
    "ADAMW_WEIGHT_DECAY",
    "CONTEXT_LENGTH",
    "CRISConfig",
    "CRISModel",
    "CRISNormalization",
    "CRISTrainingConfig",
    "DIAGONAL_SCALE_SEMANTICS",
    "DeviceAttestation",
    "FORECAST_HORIZONS",
    "HIDDEN_SIZE",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "NUM_LAYERS",
    "PredictionResult",
    "STUDENT_T_NU",
]
