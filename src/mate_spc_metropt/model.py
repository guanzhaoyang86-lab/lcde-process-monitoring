"""CARED-BC model and monitoring primitives for one-minute MetroPT data.

The module is deliberately data-source agnostic.  It accepts already audited
one-minute arrays and never opens a MetroPT file.  PyTorch is imported lazily,
so the robust statistics and conformal calibration remain usable without a
torch installation.  All learned artifacts are serialized as numeric arrays
plus canonical JSON in NPZ files opened with ``allow_pickle=False``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ANALOG_NAMES = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)
SAFE_CONTROL_NAMES = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Pressure_switch",
    "Caudal_impulses",
)
PHYSICAL_FEATURE_NAMES = (
    "TP2_minus_TP3",
    "TP3_minus_Reservoirs",
    "TP3_one_minute_slope",
    "Reservoirs_one_minute_slope",
    "pressure_rise_per_motor_current",
    "ten_minute_loaded_duty_fraction",
    "DV_pressure_discharge_pulse_amplitude",
    "DV_pressure_discharge_pulse_duration",
    "Caudal_impulses_per_loaded_minute",
    "Oil_temperature_five_minute_slope",
)

ANALOG_COUNT = 7
SAFE_CONTROL_COUNT = 6
STATE_CONTROL_COUNT = 5
INPUT_FEATURE_COUNT = ANALOG_COUNT + SAFE_CONTROL_COUNT
INPUT_WINDOW_MINUTES = 10
FORECAST_HORIZONS = (1, 3, 5)
FORECAST_HEAD_COUNT = len(FORECAST_HORIZONS)
PREDICTION_RESIDUAL_COUNT = FORECAST_HEAD_COUNT * ANALOG_COUNT
PHYSICAL_FEATURE_COUNT = len(PHYSICAL_FEATURE_NAMES)
MONITORING_RESIDUAL_COUNT = PREDICTION_RESIDUAL_COUNT + PHYSICAL_FEATURE_COUNT

TCN_CHANNELS = 64
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = (1, 2, 4, 8)
TCN_LATENT = 32
TCN_DROPOUT = 0.1
ENSEMBLE_SEEDS = (17, 29, 43)

ADAMW_LEARNING_RATE = 1.0e-3
ADAMW_WEIGHT_DECAY = 1.0e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
TRAIN_BATCH_SIZE = 1024
HUBER_DELTA = 1.0

STATE_SHRINKAGE_COUNT = 500
PAGE_KAPPA = 0.5
RANK_Z_LOWER = -4.0
RANK_Z_UPPER = 8.0
DEFAULT_GAP_SECONDS = 60.0
MINIMUM_CALIBRATION_BLOCKS = 39
STRICT_COMPARISON = "strictly_greater_than"

ENSEMBLE_SCHEMA = "mate_spc_metropt_cared_ensemble"
SCORER_SCHEMA = "mate_spc_metropt_cared_scorer"
SCHEMA_VERSION = 1


def cared_architecture_spec() -> dict[str, object]:
    """Return the frozen neural architecture and optimization contract."""
    return {
        "sampling_interval_minutes": 1,
        "input_window_minutes": INPUT_WINDOW_MINUTES,
        "forecast_horizons_minutes": list(FORECAST_HORIZONS),
        "analog_names": list(ANALOG_NAMES),
        "safe_control_names": list(SAFE_CONTROL_NAMES),
        "excluded_direct_alarm_inputs": ["LPS", "Oil_level"],
        "tcn": {
            "strictly_causal": True,
            "residual_blocks": len(TCN_DILATIONS),
            "channels": TCN_CHANNELS,
            "kernel_size": TCN_KERNEL_SIZE,
            "dilations": list(TCN_DILATIONS),
            "latent_size": TCN_LATENT,
            "dropout": TCN_DROPOUT,
            "output_shape": [FORECAST_HEAD_COUNT, ANALOG_COUNT],
        },
        "ensemble_seeds": list(ENSEMBLE_SEEDS),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": ADAMW_LEARNING_RATE,
            "weight_decay": ADAMW_WEIGHT_DECAY,
            "maximum_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "batch_size": TRAIN_BATCH_SIZE,
        },
        "loss": {"name": "Huber", "delta": HUBER_DELTA},
    }


def _readonly_float(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_int(value: Any, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer {ndim}-dimensional array")
    array = np.array(raw, dtype=np.int64, copy=True, order="C")
    array.setflags(write=False)
    return array


def _reset_vector(value: Any | None, length: int, *, name: str = "reset_before") -> np.ndarray:
    """Return a defensive boolean reset vector with explicit type checking."""

    if value is None:
        return np.zeros(length, dtype=bool)
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector with shape (T,)")
    return np.array(raw, dtype=bool, copy=True)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


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


@dataclass(frozen=True)
class RobustScaler:
    """Feature-wise median and robust-scale transform."""

    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        center = _readonly_float(self.center, ndim=1, name="center")
        scale = _readonly_float(self.scale, ndim=1, name="scale")
        if center.shape != scale.shape or center.size < 1 or np.any(scale <= 0.0):
            raise ValueError("center and scale must be same-length nonempty vectors")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(cls, values: Any, *, scale_floor: float = 1.0e-8) -> "RobustScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or array.shape[-1] < 1 or not np.isfinite(array).all():
            raise ValueError("values must be a finite array with features on the last axis")
        floor = float(scale_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be finite and positive")
        flat = array.reshape(-1, array.shape[-1])
        center = np.median(flat, axis=0)
        mad = 1.4826 * np.median(np.abs(flat - center), axis=0)
        standard = np.std(flat, axis=0, ddof=0)
        scale = np.where(mad > floor, mad, np.where(standard > floor, standard, 1.0))
        return cls(center=center, scale=scale)

    def transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.center.shape or not np.isfinite(array).all():
            raise ValueError("values have incompatible features or nonfinite entries")
        return np.asarray((array - self.center) / self.scale, dtype=np.float64)

    def inverse_transform(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.center.shape or not np.isfinite(array).all():
            raise ValueError("values have incompatible features or nonfinite entries")
        return np.asarray(array * self.scale + self.center, dtype=np.float64)


@dataclass(frozen=True)
class WindowBatch:
    """Causal 10-minute inputs and future 1/3/5-minute analog targets."""

    inputs: np.ndarray
    targets: np.ndarray
    anchor_indices: np.ndarray
    target_indices: np.ndarray
    anchor_timestamps: np.ndarray
    target_timestamps: np.ndarray

    def __post_init__(self) -> None:
        inputs = _readonly_float(self.inputs, ndim=3, name="inputs")
        targets = _readonly_float(self.targets, ndim=3, name="targets")
        anchors = _readonly_int(self.anchor_indices, ndim=1, name="anchor_indices")
        target_indices = _readonly_int(self.target_indices, ndim=2, name="target_indices")
        anchor_ts = np.array(self.anchor_timestamps, dtype="datetime64[ns]", copy=True)
        target_ts = np.array(self.target_timestamps, dtype="datetime64[ns]", copy=True)
        count = inputs.shape[0]
        if inputs.shape[1:] != (INPUT_WINDOW_MINUTES, INPUT_FEATURE_COUNT):
            raise ValueError("inputs must have shape (N, 10, 13)")
        if targets.shape != (count, FORECAST_HEAD_COUNT, ANALOG_COUNT):
            raise ValueError("targets must have shape (N, 3, 7)")
        if anchors.shape != (count,) or target_indices.shape != (count, FORECAST_HEAD_COUNT):
            raise ValueError("window index shapes are inconsistent")
        if anchor_ts.shape != (count,) or target_ts.shape != (count, FORECAST_HEAD_COUNT):
            raise ValueError("window timestamp shapes are inconsistent")
        if np.isnat(anchor_ts).any() or np.isnat(target_ts).any():
            raise ValueError("window timestamps must not contain NaT")
        anchor_ts.setflags(write=False)
        target_ts.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "target_indices", target_indices)
        object.__setattr__(self, "anchor_timestamps", anchor_ts)
        object.__setattr__(self, "target_timestamps", target_ts)

    def subset(self, indices: Any) -> "WindowBatch":
        selected = np.asarray(indices)
        return WindowBatch(
            self.inputs[selected],
            self.targets[selected],
            self.anchor_indices[selected],
            self.target_indices[selected],
            self.anchor_timestamps[selected],
            self.target_timestamps[selected],
        )


@dataclass(frozen=True)
class InferenceWindowBatch:
    """Causal model inputs available at issue time, with no future labels.

    Unlike :class:`WindowBatch`, membership in this batch depends only on the
    input history through each anchor.  It is therefore the required
    representation for online inference and monitoring.
    """

    inputs: np.ndarray
    anchor_indices: np.ndarray
    anchor_timestamps: np.ndarray

    def __post_init__(self) -> None:
        inputs = _readonly_float(self.inputs, ndim=3, name="inputs")
        anchors = _readonly_int(self.anchor_indices, ndim=1, name="anchor_indices")
        anchor_ts = np.array(self.anchor_timestamps, dtype="datetime64[ns]", copy=True)
        count = inputs.shape[0]
        if inputs.shape[1:] != (INPUT_WINDOW_MINUTES, INPUT_FEATURE_COUNT):
            raise ValueError("inputs must have shape (N, 10, 13)")
        if anchors.shape != (count,) or anchor_ts.shape != (count,):
            raise ValueError("inference-window index shapes are inconsistent")
        if np.isnat(anchor_ts).any():
            raise ValueError("anchor timestamps must not contain NaT")
        if count > 1 and np.any(np.diff(anchors) <= 0):
            raise ValueError("inference anchors must be strictly increasing")
        anchor_ts.setflags(write=False)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "anchor_timestamps", anchor_ts)

    def subset(self, indices: Any) -> "InferenceWindowBatch":
        selected = np.asarray(indices)
        return InferenceWindowBatch(
            self.inputs[selected],
            self.anchor_indices[selected],
            self.anchor_timestamps[selected],
        )


def build_causal_windows(
    analog: Any,
    controls: Any,
    timestamps: Any,
    *,
    eligible: Any | None = None,
    reset_before: Any | None = None,
    window_minutes: int = INPUT_WINDOW_MINUTES,
    horizons: Sequence[int] = FORECAST_HORIZONS,
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> WindowBatch:
    """Build labelled training windows without crossing a reset, gap, or day.

    This interface intentionally requires all future targets and is for model
    fitting and validation.  Monitoring must use
    :func:`build_causal_input_windows`, whose membership has no future-label
    dependency.
    """
    if int(window_minutes) != INPUT_WINDOW_MINUTES or tuple(horizons) != FORECAST_HORIZONS:
        raise ValueError("CARED-BC requires window=10 and horizons=(1, 3, 5)")
    x = np.asarray(analog, dtype=np.float64)
    u = np.asarray(controls, dtype=np.float64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if x.ndim != 2 or x.shape[1] != ANALOG_COUNT:
        raise ValueError("analog must have shape (T, 7)")
    if u.ndim != 2 or u.shape != (x.shape[0], SAFE_CONTROL_COUNT):
        raise ValueError("controls must have shape (T, 6)")
    if ts.shape != (x.shape[0],) or np.isnat(ts).any():
        raise ValueError("timestamps must have shape (T,) and contain no NaT")
    if ts.size > 1 and np.any(np.diff(ts.astype(np.int64)) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if eligible is None:
        ok = np.ones(x.shape[0], dtype=bool)
    else:
        ok = np.asarray(eligible)
        if ok.shape != (x.shape[0],) or ok.dtype.kind != "b":
            raise ValueError("eligible must be a boolean vector with shape (T,)")
        ok = np.array(ok, dtype=bool, copy=True)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(u).all(axis=1)
    ok &= finite
    resets = _reset_vector(reset_before, x.shape[0])
    gap_limit = float(maximum_gap_seconds)
    if not np.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("maximum_gap_seconds must be finite and positive")
    days = ts.astype("datetime64[D]")
    seconds = ts.astype(np.int64) / 1.0e9
    combined = np.concatenate((x, u), axis=1)
    input_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    anchors: list[int] = []
    target_rows_index: list[list[int]] = []
    start_anchor = INPUT_WINDOW_MINUTES - 1
    last_horizon = FORECAST_HORIZONS[-1]
    for anchor in range(start_anchor, x.shape[0] - last_horizon):
        begin = anchor - INPUT_WINDOW_MINUTES + 1
        end = anchor + last_horizon
        span = slice(begin, end + 1)
        if not np.all(ok[span]) or np.any(days[span] != days[anchor]):
            continue
        if np.any(resets[begin + 1 : end + 1]):
            continue
        if end > begin and np.any(np.diff(seconds[span]) > gap_limit):
            continue
        indices = [anchor + horizon for horizon in FORECAST_HORIZONS]
        input_rows.append(combined[begin : anchor + 1])
        target_rows.append(x[indices])
        anchors.append(anchor)
        target_rows_index.append(indices)
    if not anchors:
        raise ValueError("no complete causal windows remain after gap/day filtering")
    anchor_array = np.asarray(anchors, dtype=np.int64)
    target_index_array = np.asarray(target_rows_index, dtype=np.int64)
    return WindowBatch(
        np.asarray(input_rows, dtype=np.float64),
        np.asarray(target_rows, dtype=np.float64),
        anchor_array,
        target_index_array,
        ts[anchor_array],
        ts[target_index_array],
    )


def build_causal_input_windows(
    analog: Any,
    controls: Any,
    timestamps: Any,
    *,
    eligible: Any | None = None,
    reset_before: Any | None = None,
    window_minutes: int = INPUT_WINDOW_MINUTES,
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> InferenceWindowBatch:
    """Build online-available input windows without inspecting future rows."""

    if int(window_minutes) != INPUT_WINDOW_MINUTES:
        raise ValueError("CARED-BC requires window=10")
    x = np.asarray(analog, dtype=np.float64)
    u = np.asarray(controls, dtype=np.float64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if x.ndim != 2 or x.shape[1] != ANALOG_COUNT:
        raise ValueError("analog must have shape (T, 7)")
    if u.ndim != 2 or u.shape != (x.shape[0], SAFE_CONTROL_COUNT):
        raise ValueError("controls must have shape (T, 6)")
    if ts.shape != (x.shape[0],) or np.isnat(ts).any():
        raise ValueError("timestamps must have shape (T,) and contain no NaT")
    if ts.size > 1 and np.any(np.diff(ts.astype(np.int64)) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if eligible is None:
        ok = np.ones(x.shape[0], dtype=bool)
    else:
        raw_ok = np.asarray(eligible)
        if raw_ok.shape != (x.shape[0],) or raw_ok.dtype.kind != "b":
            raise ValueError("eligible must be a boolean vector with shape (T,)")
        ok = np.array(raw_ok, dtype=bool, copy=True)
    ok &= np.isfinite(x).all(axis=1) & np.isfinite(u).all(axis=1)
    resets = _reset_vector(reset_before, x.shape[0])
    gap_limit = float(maximum_gap_seconds)
    if not np.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("maximum_gap_seconds must be finite and positive")
    days = ts.astype("datetime64[D]")
    seconds = ts.astype(np.int64) / 1.0e9
    combined = np.concatenate((x, u), axis=1)
    input_rows: list[np.ndarray] = []
    anchors: list[int] = []
    for anchor in range(INPUT_WINDOW_MINUTES - 1, x.shape[0]):
        begin = anchor - INPUT_WINDOW_MINUTES + 1
        span = slice(begin, anchor + 1)
        if not np.all(ok[span]) or np.any(days[span] != days[anchor]):
            continue
        if np.any(resets[begin + 1 : anchor + 1]):
            continue
        if anchor > begin and np.any(np.diff(seconds[span]) > gap_limit):
            continue
        input_rows.append(combined[span])
        anchors.append(anchor)
    if not anchors:
        raise ValueError("no complete causal input windows remain after gap/day filtering")
    anchor_array = np.asarray(anchors, dtype=np.int64)
    return InferenceWindowBatch(
        np.asarray(input_rows, dtype=np.float64),
        anchor_array,
        ts[anchor_array],
    )


@dataclass(frozen=True)
class CausalTCNConfig:
    input_features: int = INPUT_FEATURE_COUNT
    output_analog: int = ANALOG_COUNT
    horizons: tuple[int, int, int] = FORECAST_HORIZONS
    channels: int = TCN_CHANNELS
    kernel_size: int = TCN_KERNEL_SIZE
    dilations: tuple[int, int, int, int] = TCN_DILATIONS
    latent_size: int = TCN_LATENT
    dropout: float = TCN_DROPOUT

    def __post_init__(self) -> None:
        expected = (
            INPUT_FEATURE_COUNT,
            ANALOG_COUNT,
            FORECAST_HORIZONS,
            TCN_CHANNELS,
            TCN_KERNEL_SIZE,
            TCN_DILATIONS,
            TCN_LATENT,
            TCN_DROPOUT,
        )
        observed = (
            self.input_features,
            self.output_analog,
            tuple(self.horizons),
            self.channels,
            self.kernel_size,
            tuple(self.dilations),
            self.latent_size,
            self.dropout,
        )
        if observed != expected:
            raise ValueError("CausalTCNConfig is protocol-frozen and cannot be changed")


_CACHED_TCN_CLASS: Any | None = None


def _tcn_class() -> Any:
    global _CACHED_TCN_CLASS
    if _CACHED_TCN_CLASS is not None:
        return _CACHED_TCN_CLASS
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required for the CARED-BC TCN") from error

    class CausalResidualBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def _conv(self, layer: Any, values: Any) -> Any:
            return layer(functional.pad(values, (self.left_padding, 0)))

        def forward(self, values: Any) -> Any:
            residual = values
            result = self.dropout(self.activation(self._conv(self.conv1, values)))
            result = self.dropout(self.activation(self._conv(self.conv2, result)))
            return residual + result

    class CausalTCN(nn.Module):
        def __init__(self, config: CausalTCNConfig):
            super().__init__()
            self.config = config
            self.input_projection = nn.Conv1d(config.input_features, config.channels, 1)
            self.blocks = nn.ModuleList(
                [
                    CausalResidualBlock(
                        config.channels, config.kernel_size, dilation, config.dropout
                    )
                    for dilation in config.dilations
                ]
            )
            self.latent = nn.Linear(config.channels, config.latent_size)
            self.activation = nn.GELU()
            self.output = nn.Linear(
                config.latent_size, len(config.horizons) * config.output_analog
            )

        def forward_sequence(self, values: Any) -> Any:
            if values.ndim != 3 or values.shape[-1] != self.config.input_features:
                raise ValueError("TCN input must have shape (batch, time, 13)")
            result = self.input_projection(values.transpose(1, 2))
            for block in self.blocks:
                result = block(result)
            result = result.transpose(1, 2)
            result = self.output(self.activation(self.latent(result)))
            return result.reshape(
                values.shape[0], values.shape[1], len(self.config.horizons), -1
            )

        def forward(self, values: Any) -> Any:
            return self.forward_sequence(values)[:, -1]

    _CACHED_TCN_CLASS = CausalTCN
    return CausalTCN


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "float64":
        return torch.float64
    if dtype == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float64' or 'float32'")


def build_causal_tcn(
    *, seed: int = ENSEMBLE_SEEDS[0], device: str = "cpu", dtype: str = "float64"
) -> Any:
    """Construct one deterministically initialized, protocol-frozen TCN."""
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _tcn_class()(CausalTCNConfig())
    return model.to(device=torch.device(device), dtype=_torch_dtype(torch, dtype))


def _numpy_state_dict(model: Any) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        array = np.array(value.detach().cpu().numpy(), copy=True, order="C")
        array.setflags(write=False)
        state[name] = array
    return state


def _load_numpy_state(model: Any, state: Mapping[str, np.ndarray]) -> None:
    import torch

    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("serialized TCN state keys do not match the architecture")
    tensors: dict[str, Any] = {}
    for name, expected_value in expected.items():
        array = np.array(state[name], copy=True, order="C")
        if tuple(array.shape) != tuple(expected_value.shape) or not np.isfinite(array).all():
            raise ValueError(f"invalid TCN parameter {name!r}")
        tensors[name] = torch.as_tensor(
            array, device=expected_value.device, dtype=expected_value.dtype
        )
    model.load_state_dict(tensors, strict=True)


def _validate_inputs(inputs: Any) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1:] != (INPUT_WINDOW_MINUTES, INPUT_FEATURE_COUNT)
        or not np.isfinite(values).all()
    ):
        raise ValueError("inputs must have finite shape (positive_N, 10, 13)")
    return np.array(values, dtype=np.float64, copy=True, order="C")


@dataclass(frozen=True)
class CaredEnsemble:
    """Deterministic TCN members and their training-only scalers."""

    seeds: tuple[int, ...]
    input_scaler: RobustScaler
    target_scaler: RobustScaler
    states: tuple[Mapping[str, np.ndarray], ...]
    dtype: str = "float64"

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds) or len(seeds) != len(self.states):
            raise ValueError("seeds and states must be nonempty, unique, and aligned")
        if self.input_scaler.center.shape != (INPUT_FEATURE_COUNT,):
            raise ValueError("input_scaler must contain 13 features")
        if self.target_scaler.center.shape != (ANALOG_COUNT,):
            raise ValueError("target_scaler must contain 7 features")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        copied_states: list[dict[str, np.ndarray]] = []
        for raw_state in self.states:
            copied: dict[str, np.ndarray] = {}
            for name, value in raw_state.items():
                array = np.array(value, copy=True, order="C")
                if array.dtype.kind != "f" or not np.isfinite(array).all():
                    raise ValueError(f"TCN parameter {name!r} must be finite floating point")
                array.setflags(write=False)
                copied[str(name)] = array
            copied_states.append(copied)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "states", tuple(copied_states))

    @classmethod
    def from_untrained(
        cls,
        input_scaler: RobustScaler,
        target_scaler: RobustScaler,
        *,
        seeds: Sequence[int] = ENSEMBLE_SEEDS,
        dtype: str = "float64",
    ) -> "CaredEnsemble":
        states = tuple(
            _numpy_state_dict(build_causal_tcn(seed=int(seed), dtype=dtype)) for seed in seeds
        )
        return cls(tuple(int(seed) for seed in seeds), input_scaler, target_scaler, states, dtype)

    def predict(
        self,
        inputs: Any,
        *,
        device: str = "cpu",
        batch_size: int = TRAIN_BATCH_SIZE,
    ) -> np.ndarray:
        """Return the member-wise median forecast in original analog units."""
        import torch

        values = _validate_inputs(inputs)
        batch = int(batch_size)
        if batch < 1:
            raise ValueError("batch_size must be positive")
        scaled = self.input_scaler.transform(values)
        torch_dtype = _torch_dtype(torch, self.dtype)
        target_device = torch.device(device)
        member_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for seed, state in zip(self.seeds, self.states):
                model = build_causal_tcn(seed=seed, device=device, dtype=self.dtype)
                _load_numpy_state(model, state)
                model.eval()
                parts: list[np.ndarray] = []
                for begin in range(0, scaled.shape[0], batch):
                    tensor = torch.as_tensor(
                        scaled[begin : begin + batch], device=target_device, dtype=torch_dtype
                    )
                    parts.append(model(tensor).detach().cpu().numpy())
                member_predictions.append(np.concatenate(parts, axis=0))
        standardized = np.median(np.stack(member_predictions, axis=0), axis=0)
        return self.target_scaler.inverse_transform(standardized)

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        document = {
            "schema": ENSEMBLE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "architecture": cared_architecture_spec(),
            "seeds": list(self.seeds),
            "dtype": self.dtype,
            "parameter_names": [sorted(state) for state in self.states],
            "metadata": dict(metadata or {}),
        }
        payload: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(_canonical_json(document)),
            "input_center": self.input_scaler.center,
            "input_scale": self.input_scaler.scale,
            "target_center": self.target_scaler.center,
            "target_scale": self.target_scaler.scale,
        }
        for member, state in enumerate(self.states):
            for name, value in state.items():
                payload[f"member_{member}__{name}"] = np.asarray(value)
        return _atomic_npz(path, payload)

    @classmethod
    def load_npz(
        cls, path: str | os.PathLike[str]
    ) -> tuple["CaredEnsemble", dict[str, object]]:
        with np.load(path, allow_pickle=False) as archive:
            document = json.loads(str(archive["metadata_json"].item()))
            if (
                document.get("schema") != ENSEMBLE_SCHEMA
                or document.get("schema_version") != SCHEMA_VERSION
                or document.get("architecture") != cared_architecture_spec()
            ):
                raise ValueError("unsupported or altered CARED ensemble schema")
            seeds = tuple(int(value) for value in document["seeds"])
            if seeds != ENSEMBLE_SEEDS:
                raise ValueError("serialized ensemble seeds violate the frozen protocol")
            names_by_member = document["parameter_names"]
            if len(names_by_member) != len(seeds):
                raise ValueError("serialized ensemble member metadata are inconsistent")
            states = tuple(
                {
                    str(name): np.array(archive[f"member_{member}__{name}"], copy=True)
                    for name in names
                }
                for member, names in enumerate(names_by_member)
            )
            ensemble = cls(
                seeds,
                RobustScaler(archive["input_center"], archive["input_scale"]),
                RobustScaler(archive["target_center"], archive["target_scale"]),
                states,
                str(document["dtype"]),
            )
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("serialized ensemble metadata must be an object")
        return ensemble, dict(metadata)


@dataclass(frozen=True)
class EnsembleMemberHistory:
    seed: int
    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    best_epoch_one_based: int
    best_validation_loss: float
    stopped_early: bool


@dataclass(frozen=True)
class CaredTrainingResult:
    ensemble: CaredEnsemble
    histories: tuple[EnsembleMemberHistory, ...]
    ensemble_validation_huber_loss: float
    persistence_validation_huber_loss: float

    @property
    def relative_improvement_over_persistence(self) -> float:
        baseline = self.persistence_validation_huber_loss
        if baseline <= 0.0:
            return 0.0
        return float((baseline - self.ensemble_validation_huber_loss) / baseline)


def huber_loss_numpy(prediction: Any, target: Any, *, delta: float = HUBER_DELTA) -> float:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or not (
        np.isfinite(prediction_array).all() and np.isfinite(target_array).all()
    ):
        raise ValueError("prediction and target must have equal finite shapes")
    difference = np.abs(prediction_array - target_array)
    value = np.where(
        difference <= delta,
        0.5 * difference * difference,
        delta * (difference - 0.5 * delta),
    )
    return float(np.mean(value))


def persistence_huber_loss(batch: WindowBatch, scaler: RobustScaler | None = None) -> float:
    """Huber loss for repeating the most recent analog observation."""
    if not isinstance(batch, WindowBatch):
        raise TypeError("batch must be a WindowBatch")
    scaler = scaler or RobustScaler.fit(batch.inputs[:, :, :ANALOG_COUNT])
    if scaler.center.shape != (ANALOG_COUNT,):
        raise ValueError("persistence scaler must contain seven analog features")
    prediction = np.repeat(
        batch.inputs[:, -1:, :ANALOG_COUNT], FORECAST_HEAD_COUNT, axis=1
    )
    return huber_loss_numpy(scaler.transform(prediction), scaler.transform(batch.targets))


def deterministic_state_balanced_indices(inputs: Any, *, seed: int) -> np.ndarray:
    """Return one deterministic epoch with equal sample counts per observed state."""
    values = _validate_inputs(inputs)
    codes = control_state_codes(values[:, :, ANALOG_COUNT:])
    unique, counts = np.unique(codes, return_counts=True)
    target = int(np.max(counts))
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for code in unique.tolist():
        available = np.flatnonzero(codes == code)
        parts.append(rng.choice(available, size=target, replace=available.size < target))
    result = np.concatenate(parts).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def train_cared_ensemble(
    training: WindowBatch,
    validation: WindowBatch,
    *,
    device: str = "cuda",
    dtype: str = "float64",
    seeds: Sequence[int] = ENSEMBLE_SEEDS,
    learning_rate: float = ADAMW_LEARNING_RATE,
    weight_decay: float = ADAMW_WEIGHT_DECAY,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = TRAIN_BATCH_SIZE,
) -> CaredTrainingResult:
    """Fit deterministic TCN members using training and healthy validation windows."""
    import torch
    import torch.nn.functional as functional

    if not isinstance(training, WindowBatch) or not isinstance(validation, WindowBatch):
        raise TypeError("training and validation must be WindowBatch instances")
    seeds_tuple = tuple(int(seed) for seed in seeds)
    if not seeds_tuple or len(set(seeds_tuple)) != len(seeds_tuple):
        raise ValueError("seeds must be a nonempty unique integer sequence")
    epochs = int(max_epochs)
    wait_limit = int(patience)
    batch = int(batch_size)
    if epochs < 1 or wait_limit < 1 or batch < 1:
        raise ValueError("max_epochs, patience, and batch_size must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid AdamW learning_rate or weight_decay")

    input_scaler = RobustScaler.fit(training.inputs)
    target_scaler = RobustScaler.fit(training.targets)
    train_x = input_scaler.transform(training.inputs)
    train_y = target_scaler.transform(training.targets)
    validation_x = input_scaler.transform(validation.inputs)
    validation_y = target_scaler.transform(validation.targets)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    torch_dtype = _torch_dtype(torch, dtype)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    saved_states: list[dict[str, np.ndarray]] = []
    histories: list[EnsembleMemberHistory] = []
    validation_tensor_x = torch.as_tensor(validation_x, device=target_device, dtype=torch_dtype)
    validation_tensor_y = torch.as_tensor(validation_y, device=target_device, dtype=torch_dtype)
    for seed in seeds_tuple:
        model = build_causal_tcn(seed=seed, device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, np.ndarray] | None = None
        train_history: list[float] = []
        validation_history: list[float] = []
        without_improvement = 0
        for epoch in range(epochs):
            model.train()
            order = deterministic_state_balanced_indices(
                training.inputs, seed=seed + epoch
            )
            weighted_loss = 0.0
            seen = 0
            for begin in range(0, order.size, batch):
                selected = order[begin : begin + batch]
                xb = torch.as_tensor(train_x[selected], device=target_device, dtype=torch_dtype)
                yb = torch.as_tensor(train_y[selected], device=target_device, dtype=torch_dtype)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                loss = functional.huber_loss(
                    prediction, yb, reduction="mean", delta=HUBER_DELTA
                )
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.detach().cpu()) * selected.size
                seen += selected.size
            train_history.append(weighted_loss / float(seen))
            model.eval()
            with torch.no_grad():
                value = functional.huber_loss(
                    model(validation_tensor_x),
                    validation_tensor_y,
                    reduction="mean",
                    delta=HUBER_DELTA,
                )
            validation_loss = float(value.detach().cpu())
            validation_history.append(validation_loss)
            if validation_loss < best_loss - 1.0e-12:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = _numpy_state_dict(model)
                without_improvement = 0
            else:
                without_improvement += 1
            if without_improvement >= wait_limit:
                break
        if best_state is None:
            raise RuntimeError("TCN training failed to produce a finite validation loss")
        saved_states.append(best_state)
        histories.append(
            EnsembleMemberHistory(
                seed=seed,
                training_loss=tuple(train_history),
                validation_loss=tuple(validation_history),
                best_epoch_one_based=best_epoch,
                best_validation_loss=best_loss,
                stopped_early=len(train_history) < epochs,
            )
        )

    ensemble = CaredEnsemble(
        seeds_tuple, input_scaler, target_scaler, tuple(saved_states), dtype
    )
    prediction = ensemble.predict(validation.inputs, device=device, batch_size=batch)
    ensemble_loss = huber_loss_numpy(
        target_scaler.transform(prediction), target_scaler.transform(validation.targets)
    )
    baseline_loss = persistence_huber_loss(validation, target_scaler)
    return CaredTrainingResult(
        ensemble=ensemble,
        histories=tuple(histories),
        ensemble_validation_huber_loss=ensemble_loss,
        persistence_validation_huber_loss=baseline_loss,
    )


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.tolist():
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def physical_causal_features(inputs: Any) -> np.ndarray:
    """Compute the ten pre-registered physical features from past data only."""
    values = _validate_inputs(inputs)
    last = values[:, -1]
    previous = values[:, -2]
    tp2 = last[:, 0]
    tp3 = last[:, 1]
    reservoirs = last[:, 4]
    motor = values[:, :, 6]
    comp = values[:, :, ANALOG_COUNT] > 0.5
    dv_electric = values[:, :, ANALOG_COUNT + 1] > 0.5
    caudal = values[:, :, ANALOG_COUNT + 5]
    loaded_minutes = np.sum(comp, axis=1).astype(np.float64)
    motor_denominator = np.maximum(np.mean(np.abs(motor), axis=1), 1.0e-8)
    oil_begin = values[:, -6, 5]
    result = np.column_stack(
        (
            tp2 - tp3,
            tp3 - reservoirs,
            tp3 - previous[:, 1],
            reservoirs - previous[:, 4],
            (tp3 - values[:, 0, 1]) / motor_denominator,
            np.mean(comp, axis=1),
            np.ptp(values[:, :, 3], axis=1),
            np.asarray(
                [_longest_true_run(row) for row in dv_electric], dtype=np.float64
            )
            / float(INPUT_WINDOW_MINUTES),
            np.sum(np.maximum(caudal, 0.0), axis=1) / np.maximum(loaded_minutes, 1.0),
            (last[:, 5] - oil_begin) / 5.0,
        )
    )
    if result.shape != (values.shape[0], PHYSICAL_FEATURE_COUNT):
        raise AssertionError("physical feature dimension changed")
    return np.asarray(result, dtype=np.float64)


def prediction_residuals(prediction: Any, targets: Any) -> np.ndarray:
    """Return horizon-major target-minus-prediction residuals with 21 columns."""
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or predicted.ndim != 3
        or predicted.shape[1:] != (FORECAST_HEAD_COUNT, ANALOG_COUNT)
        or not (np.isfinite(predicted).all() and np.isfinite(observed).all())
    ):
        raise ValueError("prediction and targets must have finite shape (N, 3, 7)")
    return np.asarray((observed - predicted).reshape(predicted.shape[0], -1), dtype=np.float64)


def build_monitoring_residuals(
    prediction: Any, targets: Any, inputs: Any
) -> np.ndarray:
    """Concatenate 21 forecast residuals and ten causal physical features."""
    residual = prediction_residuals(prediction, targets)
    physical = physical_causal_features(inputs)
    if residual.shape[0] != physical.shape[0]:
        raise ValueError("prediction residuals and input windows must align")
    result = np.concatenate((residual, physical), axis=1)
    if result.shape[1] != MONITORING_RESIDUAL_COUNT:
        raise AssertionError("monitoring residual dimension changed")
    return result


@dataclass(frozen=True)
class AlignedResidualBatch:
    """Three issued forecasts aligned to one observed score minute."""

    residuals: np.ndarray
    prediction_residuals: np.ndarray
    physical_features: np.ndarray
    score_indices: np.ndarray
    score_timestamps: np.ndarray
    state_codes: np.ndarray
    source_window_rows: np.ndarray
    reset_before: np.ndarray

    def __post_init__(self) -> None:
        residuals = _readonly_float(self.residuals, ndim=2, name="residuals")
        prediction = _readonly_float(
            self.prediction_residuals, ndim=2, name="prediction_residuals"
        )
        physical = _readonly_float(
            self.physical_features, ndim=2, name="physical_features"
        )
        indices = _readonly_int(self.score_indices, ndim=1, name="score_indices")
        states = _readonly_int(self.state_codes, ndim=1, name="state_codes")
        source_rows = _readonly_int(
            self.source_window_rows, ndim=2, name="source_window_rows"
        )
        reset_before = _reset_vector(self.reset_before, residuals.shape[0])
        timestamps = np.array(self.score_timestamps, dtype="datetime64[ns]", copy=True)
        count = residuals.shape[0]
        if (
            residuals.shape != (count, MONITORING_RESIDUAL_COUNT)
            or prediction.shape != (count, PREDICTION_RESIDUAL_COUNT)
            or physical.shape != (count, PHYSICAL_FEATURE_COUNT)
            or indices.shape != (count,)
            or states.shape != (count,)
            or source_rows.shape != (count, FORECAST_HEAD_COUNT)
            or reset_before.shape != (count,)
            or timestamps.shape != (count,)
            or np.isnat(timestamps).any()
        ):
            raise ValueError("aligned residual arrays are inconsistent")
        timestamps.setflags(write=False)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "prediction_residuals", prediction)
        object.__setattr__(self, "physical_features", physical)
        object.__setattr__(self, "score_indices", indices)
        object.__setattr__(self, "score_timestamps", timestamps)
        object.__setattr__(self, "state_codes", states)
        object.__setattr__(self, "source_window_rows", source_rows)
        reset_before.setflags(write=False)
        object.__setattr__(self, "reset_before", reset_before)


def _project_reset_before(reset_before: np.ndarray, score_indices: np.ndarray) -> np.ndarray:
    """Project raw minute resets onto a possibly sparse scored timeline."""

    projected = np.zeros(score_indices.size, dtype=bool)
    if score_indices.size == 0:
        return projected
    projected[0] = True
    for row in range(1, score_indices.size):
        previous = int(score_indices[row - 1])
        current = int(score_indices[row])
        projected[row] = bool(np.any(reset_before[previous + 1 : current + 1]))
    return projected


def align_forecasts_to_score_times(
    windows: WindowBatch | InferenceWindowBatch,
    predictions: Any,
    analog: Any,
    controls: Any,
    timestamps: Any,
    *,
    eligible: Any | None = None,
    reset_before: Any | None = None,
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> AlignedResidualBatch:
    """Align issued forecasts to x_t and causal physics ending at t.

    Passing an :class:`InferenceWindowBatch` dispatches to the online-safe
    implementation.  ``WindowBatch`` remains accepted only for compatibility
    with training diagnostics; because its membership requires future labels,
    it must not be used for production monitoring.
    """
    if isinstance(windows, InferenceWindowBatch):
        return align_inference_forecasts_to_score_times(
            windows,
            predictions,
            analog,
            controls,
            timestamps,
            eligible=eligible,
            reset_before=reset_before,
            maximum_gap_seconds=maximum_gap_seconds,
        )
    if not isinstance(windows, WindowBatch):
        raise TypeError("windows must be a WindowBatch or InferenceWindowBatch")
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != windows.targets.shape or not np.isfinite(predicted).all():
        raise ValueError("predictions must have finite shape matching windows.targets")
    x = np.asarray(analog, dtype=np.float64)
    u = np.asarray(controls, dtype=np.float64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if x.ndim != 2 or x.shape[1] != ANALOG_COUNT:
        raise ValueError("analog must have shape (T, 7)")
    if u.shape != (x.shape[0], SAFE_CONTROL_COUNT):
        raise ValueError("controls must have shape (T, 6)")
    if ts.shape != (x.shape[0],) or np.isnat(ts).any():
        raise ValueError("timestamps must have shape (T,) and contain no NaT")
    if eligible is None:
        ok = np.ones(x.shape[0], dtype=bool)
    else:
        raw_ok = np.asarray(eligible)
        if raw_ok.shape != (x.shape[0],) or raw_ok.dtype.kind != "b":
            raise ValueError("eligible must be a boolean vector with shape (T,)")
        ok = np.array(raw_ok, dtype=bool, copy=True)
    ok &= np.isfinite(x).all(axis=1) & np.isfinite(u).all(axis=1)
    resets = _reset_vector(reset_before, x.shape[0])
    if np.any(windows.target_indices < 0) or np.any(windows.target_indices >= x.shape[0]):
        raise ValueError("window target index lies outside the supplied minute arrays")
    if not np.array_equal(ts[windows.target_indices], windows.target_timestamps):
        raise ValueError("window target timestamps do not match the supplied timeline")
    gap_limit = float(maximum_gap_seconds)
    if not np.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("maximum_gap_seconds must be finite and positive")

    row_by_target: list[dict[int, int]] = []
    for head in range(FORECAST_HEAD_COUNT):
        mapping: dict[int, int] = {}
        for row, target in enumerate(windows.target_indices[:, head].tolist()):
            if target in mapping:
                raise ValueError("a forecast horizon contains duplicate target timestamps")
            mapping[int(target)] = row
        row_by_target.append(mapping)
    common = sorted(set.intersection(*(set(mapping) for mapping in row_by_target)))
    score_indices: list[int] = []
    source_rows: list[list[int]] = []
    forecast_rows: list[np.ndarray] = []
    physical_windows: list[np.ndarray] = []
    seconds = ts.astype(np.int64) / 1.0e9
    days = ts.astype("datetime64[D]")
    for target in common:
        begin = target - INPUT_WINDOW_MINUTES + 1
        if begin < 0 or np.any(days[begin : target + 1] != days[target]):
            continue
        if np.any(np.diff(seconds[begin : target + 1]) > gap_limit):
            continue
        if not np.all(ok[begin : target + 1]):
            continue
        if np.any(resets[begin + 1 : target + 1]):
            continue
        rows = [row_by_target[head][target] for head in range(FORECAST_HEAD_COUNT)]
        residual_by_head = [x[target] - predicted[row, head] for head, row in enumerate(rows)]
        score_indices.append(target)
        source_rows.append(rows)
        forecast_rows.append(np.concatenate(residual_by_head))
        physical_windows.append(np.concatenate((x[begin : target + 1], u[begin : target + 1]), axis=1))
    if not score_indices:
        raise ValueError("no common causal score timestamps remain after alignment")
    physical = physical_causal_features(np.asarray(physical_windows))
    forecast = np.asarray(forecast_rows, dtype=np.float64)
    indices = np.asarray(score_indices, dtype=np.int64)
    states = control_state_codes(u[indices])
    return AlignedResidualBatch(
        residuals=np.concatenate((forecast, physical), axis=1),
        prediction_residuals=forecast,
        physical_features=physical,
        score_indices=indices,
        score_timestamps=ts[indices],
        state_codes=states,
        source_window_rows=np.asarray(source_rows, dtype=np.int64),
        reset_before=_project_reset_before(resets, indices),
    )


def align_inference_forecasts_to_score_times(
    windows: InferenceWindowBatch,
    predictions: Any,
    analog: Any,
    controls: Any,
    timestamps: Any,
    *,
    eligible: Any | None = None,
    reset_before: Any | None = None,
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> AlignedResidualBatch:
    """Align online-issued h=1/3/5 predictions without future dependence.

    For a score at minute ``t``, this function uses only the observation at t,
    causal physical history ending at t, and predictions already issued from
    input windows ending at t-1, t-3, and t-5.  Neither construction of those
    issue windows nor membership of the score at t inspects rows after t.
    """

    if not isinstance(windows, InferenceWindowBatch):
        raise TypeError("windows must be an InferenceWindowBatch")
    predicted = np.asarray(predictions, dtype=np.float64)
    expected_shape = (windows.inputs.shape[0], FORECAST_HEAD_COUNT, ANALOG_COUNT)
    if predicted.shape != expected_shape or not np.isfinite(predicted).all():
        raise ValueError(f"predictions must have finite shape {expected_shape}")
    x = np.asarray(analog, dtype=np.float64)
    u = np.asarray(controls, dtype=np.float64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if x.ndim != 2 or x.shape[1] != ANALOG_COUNT:
        raise ValueError("analog must have shape (T, 7)")
    if u.shape != (x.shape[0], SAFE_CONTROL_COUNT):
        raise ValueError("controls must have shape (T, 6)")
    if ts.shape != (x.shape[0],) or np.isnat(ts).any():
        raise ValueError("timestamps must have shape (T,) and contain no NaT")
    values_ns = ts.astype(np.int64)
    if values_ns.size > 1 and np.any(np.diff(values_ns) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if np.any(windows.anchor_indices < 0) or np.any(windows.anchor_indices >= x.shape[0]):
        raise ValueError("inference anchor index lies outside supplied minute arrays")
    if not np.array_equal(ts[windows.anchor_indices], windows.anchor_timestamps):
        raise ValueError("inference anchor timestamps do not match supplied timeline")
    if eligible is None:
        ok = np.ones(x.shape[0], dtype=bool)
    else:
        raw_ok = np.asarray(eligible)
        if raw_ok.shape != (x.shape[0],) or raw_ok.dtype.kind != "b":
            raise ValueError("eligible must be a boolean vector with shape (T,)")
        ok = np.array(raw_ok, dtype=bool, copy=True)
    ok &= np.isfinite(x).all(axis=1) & np.isfinite(u).all(axis=1)
    resets = _reset_vector(reset_before, x.shape[0])
    gap_limit = float(maximum_gap_seconds)
    if not np.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("maximum_gap_seconds must be finite and positive")

    index_by_timestamp = {int(value): index for index, value in enumerate(values_ns.tolist())}
    row_by_target: list[dict[int, int]] = []
    minute_ns = 60_000_000_000
    anchor_ns = windows.anchor_timestamps.astype(np.int64)
    for head, horizon in enumerate(FORECAST_HORIZONS):
        mapping: dict[int, int] = {}
        offset = int(horizon) * minute_ns
        for row, issued_ns in enumerate(anchor_ns.tolist()):
            target = index_by_timestamp.get(int(issued_ns) + offset)
            if target is not None:
                if target in mapping:
                    raise ValueError("a forecast horizon contains duplicate target timestamps")
                mapping[target] = row
        row_by_target.append(mapping)

    common = sorted(set.intersection(*(set(mapping) for mapping in row_by_target)))
    score_indices: list[int] = []
    source_rows: list[list[int]] = []
    forecast_rows: list[np.ndarray] = []
    physical_windows: list[np.ndarray] = []
    seconds = values_ns / 1.0e9
    days = ts.astype("datetime64[D]")
    for target in common:
        begin = target - INPUT_WINDOW_MINUTES + 1
        if begin < 0 or not np.all(ok[begin : target + 1]):
            continue
        if np.any(days[begin : target + 1] != days[target]):
            continue
        if np.any(np.diff(seconds[begin : target + 1]) > gap_limit):
            continue
        if np.any(resets[begin + 1 : target + 1]):
            continue
        rows = [row_by_target[head][target] for head in range(FORECAST_HEAD_COUNT)]
        valid_issuance = True
        for row in rows:
            anchor = int(windows.anchor_indices[row])
            if anchor >= target or np.any(days[anchor : target + 1] != days[target]):
                valid_issuance = False
                break
            if np.any(np.diff(seconds[anchor : target + 1]) > gap_limit):
                valid_issuance = False
                break
            if np.any(resets[anchor + 1 : target + 1]):
                valid_issuance = False
                break
        if not valid_issuance:
            continue
        residual_by_head = [x[target] - predicted[row, head] for head, row in enumerate(rows)]
        score_indices.append(target)
        source_rows.append(rows)
        forecast_rows.append(np.concatenate(residual_by_head))
        physical_windows.append(
            np.concatenate((x[begin : target + 1], u[begin : target + 1]), axis=1)
        )
    if not score_indices:
        raise ValueError("no common causal score timestamps remain after online alignment")
    physical = physical_causal_features(np.asarray(physical_windows))
    forecast = np.asarray(forecast_rows, dtype=np.float64)
    indices = np.asarray(score_indices, dtype=np.int64)
    states = control_state_codes(u[indices])
    return AlignedResidualBatch(
        residuals=np.concatenate((forecast, physical), axis=1),
        prediction_residuals=forecast,
        physical_features=physical,
        score_indices=indices,
        score_timestamps=ts[indices],
        state_codes=states,
        source_window_rows=np.asarray(source_rows, dtype=np.int64),
        reset_before=_project_reset_before(resets, indices),
    )


def control_state_bits(controls: Any) -> np.ndarray:
    """Threshold the five safe binary controls at the current minute."""
    values = np.asarray(controls, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[1:] != (INPUT_WINDOW_MINUTES, SAFE_CONTROL_COUNT):
            raise ValueError("control windows must have shape (N, 10, 6)")
        values = values[:, -1]
    if values.ndim != 2 or values.shape[1] != SAFE_CONTROL_COUNT:
        raise ValueError("controls must have shape (N, 6) or (N, 10, 6)")
    if not np.isfinite(values).all():
        raise ValueError("controls must be finite")
    return np.asarray(values[:, :STATE_CONTROL_COUNT] > 0.5, dtype=np.uint8)


def control_state_codes(controls: Any) -> np.ndarray:
    bits = control_state_bits(controls)
    weights = (1 << np.arange(STATE_CONTROL_COUNT, dtype=np.uint8)).astype(np.uint8)
    return np.asarray(bits @ weights, dtype=np.int64)


def _state_bit_tuple(code: int) -> tuple[int, ...]:
    """Return bits in the frozen control-column order for lexical tie breaks."""

    value = int(code)
    if value < 0 or value >= 2**STATE_CONTROL_COUNT:
        raise ValueError("state code must lie in [0, 31]")
    return tuple((value >> bit) & 1 for bit in range(STATE_CONTROL_COUNT))


def _nearest_state_code(code: int, candidates: Any) -> int:
    """Choose minimum Hamming distance, then five-bit tuple lexical order."""

    raw = np.asarray(candidates)
    if raw.ndim != 1 or raw.size < 1 or raw.dtype.kind not in "iu":
        raise ValueError("candidate state codes must be a nonempty integer vector")
    candidate_values = [int(value) for value in raw.tolist()]
    if any(value < 0 or value >= 2**STATE_CONTROL_COUNT for value in candidate_values):
        raise ValueError("candidate state code must lie in [0, 31]")
    return min(
        candidate_values,
        key=lambda candidate: (
            (int(code) ^ candidate).bit_count(),
            _state_bit_tuple(candidate),
        ),
    )


def _validate_residual_matrix(
    values: Any, *, name: str, minimum_rows: int = 1
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < minimum_rows or array.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape (at_least_{minimum_rows}, positive_features)"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _validate_state_codes(values: Any, *, count: int) -> np.ndarray:
    codes = _readonly_int(values, ndim=1, name="state_codes")
    if codes.shape != (count,) or np.any((codes < 0) | (codes >= 2**STATE_CONTROL_COUNT)):
        raise ValueError("state_codes must have shape (N,) and values in [0, 31]")
    return codes


def _huber_location(values: np.ndarray, *, tuning: float = 1.345) -> np.ndarray:
    """Coordinate-wise Huber M-location with a frozen robust scale."""
    location = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - location), axis=0)
    standard = np.std(values, axis=0, ddof=0)
    scale = np.where(mad > 1.0e-8, mad, np.where(standard > 1.0e-8, standard, 1.0))
    for _ in range(50):
        standardized = (values - location) / scale
        absolute = np.abs(standardized)
        weights = np.ones_like(absolute)
        large = absolute > tuning
        weights[large] = tuning / absolute[large]
        updated = np.sum(weights * values, axis=0) / np.maximum(
            np.sum(weights, axis=0), 1.0e-12
        )
        if np.max(np.abs(updated - location) / scale) <= 1.0e-10:
            location = updated
            break
        location = updated
    return np.asarray(location, dtype=np.float64)


def _ledoit_wolf_covariance(centered: np.ndarray) -> np.ndarray:
    """Fit the Ledoit-Wolf identity-target covariance without sklearn."""
    count, features = centered.shape
    empirical = centered.T @ centered / float(count)
    mu = float(np.trace(empirical) / features)
    delta = float(np.sum((empirical - mu * np.eye(features)) ** 2) / features)
    squared = centered * centered
    beta = float(
        np.sum((squared.T @ squared) / float(count) - empirical * empirical)
        / float(features * count)
    )
    beta = min(max(beta, 0.0), delta)
    shrinkage = 0.0 if delta <= 1.0e-30 else beta / delta
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: features + 1] += shrinkage * mu
    floor = max(1.0e-8, 1.0e-8 * max(mu, 1.0))
    covariance = 0.5 * (covariance + covariance.T) + floor * np.eye(features)
    return covariance


def _robust_center_covariance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = _huber_location(values)
    covariance = _ledoit_wolf_covariance(values - center)
    return center, covariance


@dataclass(frozen=True)
class StateConditionalReference:
    """Robust state centers and shrinkage covariance/precision matrices."""

    state_codes: np.ndarray
    counts: np.ndarray
    centers: np.ndarray
    covariances: np.ndarray
    precisions: np.ndarray
    global_center: np.ndarray
    global_covariance: np.ndarray
    global_precision: np.ndarray
    shrinkage_count: int = STATE_SHRINKAGE_COUNT

    def __post_init__(self) -> None:
        codes = _readonly_int(self.state_codes, ndim=1, name="state_codes")
        counts = _readonly_int(self.counts, ndim=1, name="counts")
        centers = _readonly_float(self.centers, ndim=2, name="centers")
        covariances = _readonly_float(self.covariances, ndim=3, name="covariances")
        precisions = _readonly_float(self.precisions, ndim=3, name="precisions")
        global_center = _readonly_float(self.global_center, ndim=1, name="global_center")
        global_covariance = _readonly_float(
            self.global_covariance, ndim=2, name="global_covariance"
        )
        global_precision = _readonly_float(
            self.global_precision, ndim=2, name="global_precision"
        )
        states = codes.size
        features = global_center.size
        if (
            counts.shape != (states,)
            or centers.shape != (states, features)
            or covariances.shape != (states, features, features)
            or precisions.shape != (states, features, features)
            or global_covariance.shape != (features, features)
            or global_precision.shape != (features, features)
            or np.any(counts < 1)
            or len(set(codes.tolist())) != states
        ):
            raise ValueError("state reference arrays are inconsistent")
        object.__setattr__(self, "state_codes", codes)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "covariances", covariances)
        object.__setattr__(self, "precisions", precisions)
        object.__setattr__(self, "global_center", global_center)
        object.__setattr__(self, "global_covariance", global_covariance)
        object.__setattr__(self, "global_precision", global_precision)

    def row_for_state(self, code: int) -> int:
        found = np.flatnonzero(self.state_codes == int(code))
        if found.size:
            return int(found[0])
        destination = _nearest_state_code(code, self.state_codes)
        return int(np.flatnonzero(self.state_codes == destination)[0])


def fit_state_conditional_reference(
    healthy_residuals: Any,
    state_codes: Any,
    *,
    shrinkage_count: int = STATE_SHRINKAGE_COUNT,
) -> StateConditionalReference:
    residuals = _validate_residual_matrix(
        healthy_residuals, name="healthy_residuals", minimum_rows=2
    )
    codes = _validate_state_codes(state_codes, count=residuals.shape[0])
    shrink = int(shrinkage_count)
    if shrink < 1:
        raise ValueError("shrinkage_count must be positive")
    global_center, global_covariance = _robust_center_covariance(residuals)
    global_precision = np.linalg.pinv(global_covariance, hermitian=True)
    raw_unique, raw_counts = np.unique(codes, return_counts=True)
    frequent = raw_unique[raw_counts >= shrink]
    if frequent.size == 0:
        raise ValueError(
            "no control state reaches the required shrinkage_count; "
            "state-conditional reference fit fails closed"
        )
    pooled_codes = np.empty_like(codes)
    for code in raw_unique.tolist():
        if np.any(frequent == code):
            destination = int(code)
        else:
            destination = _nearest_state_code(code, frequent)
        pooled_codes[codes == code] = destination
    unique = np.asarray(sorted(frequent.tolist()), dtype=np.int64)
    counts: list[int] = []
    centers: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    precisions: list[np.ndarray] = []
    for code in unique.tolist():
        local = residuals[pooled_codes == code]
        local_center, local_covariance = _robust_center_covariance(local)
        count = local.shape[0]
        counts.append(count)
        centers.append(local_center)
        covariances.append(local_covariance)
        precisions.append(np.linalg.pinv(local_covariance, hermitian=True))
    return StateConditionalReference(
        state_codes=unique,
        counts=np.asarray(counts, dtype=np.int64),
        centers=np.asarray(centers),
        covariances=np.asarray(covariances),
        precisions=np.asarray(precisions),
        global_center=global_center,
        global_covariance=global_covariance,
        global_precision=global_precision,
        shrinkage_count=shrink,
    )


def _state_parameters(
    reference: StateConditionalReference, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.empty((codes.size, reference.global_center.size), dtype=np.float64)
    precisions = np.empty(
        (codes.size, reference.global_center.size, reference.global_center.size),
        dtype=np.float64,
    )
    for code in np.unique(codes).tolist():
        selected = codes == code
        row = reference.row_for_state(code)
        if row is None:
            centers[selected] = reference.global_center
            precisions[selected] = reference.global_precision
        else:
            centers[selected] = reference.centers[row]
            precisions[selected] = reference.precisions[row]
    return centers, precisions


def state_conditional_energy(
    residuals: Any, state_codes: Any, reference: StateConditionalReference
) -> np.ndarray:
    values = _validate_residual_matrix(residuals, name="residuals")
    if values.shape[1] != reference.global_center.size:
        raise ValueError("residual feature count does not match the reference")
    codes = _validate_state_codes(state_codes, count=values.shape[0])
    centers, precisions = _state_parameters(reference, codes)
    centered = values - centers
    score = np.einsum("ni,nij,nj->n", centered, precisions, centered, optimize=True)
    return np.maximum(np.asarray(score, dtype=np.float64), 0.0)


@dataclass(frozen=True)
class DirectionalGLRModel:
    """F1-derived positive directions, shrunk to one global direction."""

    state_codes: np.ndarray
    fault_counts: np.ndarray
    directions: np.ndarray
    global_direction: np.ndarray
    shrinkage_count: int = STATE_SHRINKAGE_COUNT

    def __post_init__(self) -> None:
        codes = _readonly_int(self.state_codes, ndim=1, name="state_codes")
        counts = _readonly_int(self.fault_counts, ndim=1, name="fault_counts")
        directions = _readonly_float(self.directions, ndim=2, name="directions")
        global_direction = _readonly_float(
            self.global_direction, ndim=1, name="global_direction"
        )
        if (
            counts.shape != (codes.size,)
            or directions.shape != (codes.size, global_direction.size)
            or np.any(counts < 0)
            or len(set(codes.tolist())) != codes.size
        ):
            raise ValueError("direction model arrays are inconsistent")
        object.__setattr__(self, "state_codes", codes)
        object.__setattr__(self, "fault_counts", counts)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "global_direction", global_direction)

    def direction_for_state(self, code: int) -> np.ndarray:
        found = np.flatnonzero(self.state_codes == int(code))
        if found.size:
            return self.directions[int(found[0])]
        destination = _nearest_state_code(code, self.state_codes)
        return self.directions[int(np.flatnonzero(self.state_codes == destination)[0])]


def fit_f1_directional_glr(
    reference: StateConditionalReference,
    development_healthy_residuals: Any,
    development_healthy_state_codes: Any,
    f1_residuals: Any,
    f1_state_codes: Any,
    *,
    shrinkage_count: int = STATE_SHRINKAGE_COUNT,
) -> DirectionalGLRModel:
    healthy = _validate_residual_matrix(
        development_healthy_residuals,
        name="development_healthy_residuals",
        minimum_rows=2,
    )
    if healthy.shape[1] != reference.global_center.size:
        raise ValueError("development healthy feature count does not match the reference")
    healthy_codes = _validate_state_codes(
        development_healthy_state_codes, count=healthy.shape[0]
    )
    fault = _validate_residual_matrix(
        f1_residuals, name="f1_residuals", minimum_rows=2
    )
    if fault.shape[1] != reference.global_center.size:
        raise ValueError("F1 residual feature count does not match the reference")
    codes = _validate_state_codes(f1_state_codes, count=fault.shape[0])
    shrink = int(shrinkage_count)
    if shrink < 1:
        raise ValueError("shrinkage_count must be positive")
    global_fault_center = _huber_location(fault)
    global_healthy_center = _huber_location(healthy)
    global_direction = global_fault_center - global_healthy_center
    mapped_healthy = np.asarray(
        [reference.state_codes[reference.row_for_state(code)] for code in healthy_codes]
    )
    mapped_fault = np.asarray(
        [reference.state_codes[reference.row_for_state(code)] for code in codes]
    )
    state_codes = np.asarray(reference.state_codes)
    counts: list[int] = []
    directions: list[np.ndarray] = []
    for row, code in enumerate(state_codes.tolist()):
        selected_fault = fault[mapped_fault == code]
        selected_healthy = healthy[mapped_healthy == code]
        count = selected_fault.shape[0]
        if count and selected_healthy.shape[0]:
            local_direction = _huber_location(selected_fault) - _huber_location(
                selected_healthy
            )
            rho = count / float(count + shrink)
            direction = rho * local_direction + (1.0 - rho) * global_direction
        else:
            direction = global_direction
        counts.append(count)
        directions.append(direction)
    return DirectionalGLRModel(
        state_codes=state_codes,
        fault_counts=np.asarray(counts, dtype=np.int64),
        directions=np.asarray(directions),
        global_direction=global_direction,
        shrinkage_count=shrink,
    )


def directional_glr_scores(
    residuals: Any,
    state_codes: Any,
    reference: StateConditionalReference,
    direction_model: DirectionalGLRModel,
) -> np.ndarray:
    """Compute the positive one-sided F1 directional GLR score."""
    values = _validate_residual_matrix(residuals, name="residuals")
    if values.shape[1] != reference.global_center.size:
        raise ValueError("residual feature count does not match the reference")
    if direction_model.global_direction.size != values.shape[1]:
        raise ValueError("direction feature count does not match residuals")
    codes = _validate_state_codes(state_codes, count=values.shape[0])
    centers, precisions = _state_parameters(reference, codes)
    centered = values - centers
    result = np.zeros(values.shape[0], dtype=np.float64)
    for code in np.unique(codes).tolist():
        selected = np.flatnonzero(codes == code)
        direction = np.asarray(direction_model.direction_for_state(code))
        precision = precisions[selected[0]]
        denominator_squared = float(direction @ precision @ direction)
        if denominator_squared <= 1.0e-18:
            continue
        numerator = centered[selected] @ (precision @ direction)
        result[selected] = np.maximum(numerator / math.sqrt(denominator_squared), 0.0)
    return result


def _normal_ppf(probability: Any) -> np.ndarray:
    """Vectorized Acklam inverse standard-normal CDF approximation."""
    p = np.asarray(probability, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0) | ~np.isfinite(p)):
        raise ValueError("normal probabilities must be finite and strictly between zero and one")
    a = np.asarray(
        [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    )
    b = np.asarray(
        [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    )
    c = np.asarray(
        [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    )
    d = np.asarray(
        [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416]
    )
    low = 0.02425
    high = 1.0 - low
    result = np.empty_like(p)
    lower = p < low
    upper = p > high
    middle = ~(lower | upper)
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        result[lower] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        result[upper] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        result[middle] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return result


@dataclass(frozen=True)
class RankNormalTransform:
    """Development-healthy empirical rank transform for the two channels."""

    sorted_reference: np.ndarray
    lower_clip: float = RANK_Z_LOWER
    upper_clip: float = RANK_Z_UPPER

    def __post_init__(self) -> None:
        reference = _readonly_float(
            self.sorted_reference, ndim=2, name="sorted_reference"
        )
        if reference.shape[0] < 2 or reference.shape[1] != 2:
            raise ValueError("sorted_reference must have shape (at_least_two, 2)")
        if np.any(np.diff(reference, axis=0) < 0.0):
            raise ValueError("each rank reference column must be sorted")
        if not self.lower_clip < self.upper_clip:
            raise ValueError("rank-normal clipping limits are invalid")
        object.__setattr__(self, "sorted_reference", reference)

    @classmethod
    def fit(cls, healthy_channel_scores: Any) -> "RankNormalTransform":
        values = _validate_residual_matrix(
            healthy_channel_scores, name="healthy_channel_scores", minimum_rows=2
        )
        if values.shape[1] != 2:
            raise ValueError("healthy_channel_scores must have two columns")
        return cls(np.sort(values, axis=0))

    def transform(self, channel_scores: Any) -> np.ndarray:
        values = np.asarray(channel_scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("channel_scores must have finite shape (N, 2)")
        count = self.sorted_reference.shape[0]
        transformed = np.empty_like(values)
        for channel in range(2):
            # Midranks are essential for atomic score distributions.  In
            # particular, directional GLR has a large exact-zero atom; using
            # the right insertion rank turns healthy zeros into positive
            # evidence and creates deterministic Page-CUSUM drift.
            left = np.searchsorted(
                self.sorted_reference[:, channel], values[:, channel], side="left"
            )
            right = np.searchsorted(
                self.sorted_reference[:, channel], values[:, channel], side="right"
            )
            rank = 0.5 * (left.astype(np.float64) + right.astype(np.float64))
            probability = (rank + 0.5) / float(count + 1)
            probability = np.clip(
                probability,
                0.5 / float(count + 1),
                (count + 0.5) / float(count + 1),
            )
            transformed[:, channel] = _normal_ppf(probability)
        return np.clip(transformed, self.lower_clip, self.upper_clip)


@dataclass(frozen=True)
class CUSUMPath:
    standardized_evidence: np.ndarray
    cusum: np.ndarray
    envelope: np.ndarray
    reset: np.ndarray
    alarm: np.ndarray
    threshold: float | None


def dual_page_cusum(
    standardized_evidence: Any,
    timestamps: Any,
    *,
    kappa: float = PAGE_KAPPA,
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS,
    reset_before: Any | None = None,
    threshold: float | None = None,
) -> CUSUMPath:
    """Run two Page CUSUMs with mandatory gap and local-day resets."""
    evidence = np.asarray(standardized_evidence, dtype=np.float64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if evidence.ndim != 2 or evidence.shape[1] != 2 or not np.isfinite(evidence).all():
        raise ValueError("standardized_evidence must have finite shape (N, 2)")
    if ts.shape != (evidence.shape[0],) or np.isnat(ts).any():
        raise ValueError("timestamps must align with evidence and contain no NaT")
    explicit_reset = _reset_vector(reset_before, evidence.shape[0])
    allowance = float(kappa)
    gap_limit = float(maximum_gap_seconds)
    if not np.isfinite(allowance) or allowance < 0.0:
        raise ValueError("kappa must be finite and nonnegative")
    if not np.isfinite(gap_limit) or gap_limit <= 0.0:
        raise ValueError("maximum_gap_seconds must be finite and positive")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
        raise ValueError("threshold must be finite and nonnegative")
    values_ns = ts.astype(np.int64)
    if values_ns.size > 1 and np.any(np.diff(values_ns) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    days = ts.astype("datetime64[D]")
    reset = np.zeros(evidence.shape[0], dtype=bool)
    cusum = np.zeros_like(evidence)
    state = np.zeros(2, dtype=np.float64)
    for index in range(evidence.shape[0]):
        should_reset = bool(index == 0 or explicit_reset[index])
        if index:
            gap_seconds = (values_ns[index] - values_ns[index - 1]) / 1.0e9
            should_reset = bool(
                should_reset
                or days[index] != days[index - 1]
                or gap_seconds > gap_limit
            )
        if should_reset:
            state.fill(0.0)
            reset[index] = True
        state = np.maximum(0.0, state + evidence[index] - allowance)
        cusum[index] = state
    envelope = np.max(cusum, axis=1)
    alarm = np.zeros(evidence.shape[0], dtype=bool)
    if threshold is not None:
        alarm = envelope > float(threshold)
    return CUSUMPath(
        standardized_evidence=np.array(evidence, copy=True),
        cusum=cusum,
        envelope=envelope,
        reset=reset,
        alarm=alarm,
        threshold=None if threshold is None else float(threshold),
    )


@dataclass(frozen=True)
class DailyMaxima:
    dates: np.ndarray
    maxima: np.ndarray
    counts: np.ndarray


def daily_maxima(path: CUSUMPath, timestamps: Any) -> DailyMaxima:
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if ts.shape != path.envelope.shape or np.isnat(ts).any():
        raise ValueError("timestamps must align with the CUSUM path")
    dates = ts.astype("datetime64[D]")
    unique = np.unique(dates)
    maxima = np.asarray([np.max(path.envelope[dates == day]) for day in unique])
    counts = np.asarray([np.count_nonzero(dates == day) for day in unique], dtype=np.int64)
    return DailyMaxima(unique, maxima, counts)


def block_conformal_rank(block_count: int, *, alpha: float = 0.05) -> int:
    count = int(block_count)
    level = float(alpha)
    if count < 1 or not 0.0 < level < 1.0:
        raise ValueError("block_count must be positive and alpha must lie in (0, 1)")
    rank = int(math.ceil((count + 1) * (1.0 - level)))
    if rank > count:
        raise ValueError("too few blocks for a finite split-conformal threshold")
    return rank


@dataclass(frozen=True)
class BlockConformalCalibration:
    block_count: int
    alpha: float
    rank_one_based: int
    threshold: float
    upper_alarm_probability_bound: float
    comparison: str = STRICT_COMPARISON

    def alarm(self, maxima: Any) -> np.ndarray:
        values = np.asarray(maxima, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("maxima must be finite")
        return values > self.threshold


def calibrate_block_conformal(
    block_maxima: Any,
    *,
    alpha: float = 0.05,
    minimum_blocks: int = MINIMUM_CALIBRATION_BLOCKS,
) -> BlockConformalCalibration:
    values = np.asarray(block_maxima, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("block_maxima must be a finite one-dimensional array")
    if values.size < int(minimum_blocks):
        raise ValueError(f"at least {int(minimum_blocks)} calibration blocks are required")
    rank = block_conformal_rank(values.size, alpha=alpha)
    threshold = float(np.sort(values)[rank - 1])
    bound = float((values.size + 1 - rank) / float(values.size + 1))
    return BlockConformalCalibration(
        block_count=int(values.size),
        alpha=float(alpha),
        rank_one_based=rank,
        threshold=threshold,
        upper_alarm_probability_bound=bound,
    )


@dataclass(frozen=True)
class CaredScorePath:
    energy: np.ndarray
    directional_glr: np.ndarray
    standardized_evidence: np.ndarray
    cusum: np.ndarray
    envelope: np.ndarray
    reset: np.ndarray
    alarm: np.ndarray
    threshold: float | None


@dataclass(frozen=True)
class CaredBCScorer:
    """Frozen state reference, F1 direction, rank map, and Page monitor."""

    reference: StateConditionalReference
    direction: DirectionalGLRModel
    rank_normalizer: RankNormalTransform
    kappa: float = PAGE_KAPPA
    maximum_gap_seconds: float = DEFAULT_GAP_SECONDS
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.reference.global_center.size != self.direction.global_direction.size:
            raise ValueError("reference and direction feature dimensions differ")
        if not np.array_equal(self.reference.state_codes, self.direction.state_codes):
            raise ValueError("reference and direction state codes must be identical")
        if not np.isfinite(self.kappa) or self.kappa < 0.0:
            raise ValueError("kappa must be finite and nonnegative")
        if not np.isfinite(self.maximum_gap_seconds) or self.maximum_gap_seconds <= 0.0:
            raise ValueError("maximum_gap_seconds must be finite and positive")
        if self.threshold is not None and (
            not np.isfinite(self.threshold) or self.threshold < 0.0
        ):
            raise ValueError("threshold must be finite and nonnegative")

    def with_threshold(self, threshold: float | None) -> "CaredBCScorer":
        return replace(self, threshold=threshold)

    def score(
        self,
        residuals: Any,
        state_codes: Any,
        timestamps: Any,
        *,
        reset_before: Any | None = None,
    ) -> CaredScorePath:
        energy = state_conditional_energy(residuals, state_codes, self.reference)
        directional = directional_glr_scores(
            residuals, state_codes, self.reference, self.direction
        )
        evidence = self.rank_normalizer.transform(np.column_stack((energy, directional)))
        page = dual_page_cusum(
            evidence,
            timestamps,
            kappa=self.kappa,
            maximum_gap_seconds=self.maximum_gap_seconds,
            reset_before=reset_before,
            threshold=self.threshold,
        )
        return CaredScorePath(
            energy=energy,
            directional_glr=directional,
            standardized_evidence=evidence,
            cusum=page.cusum,
            envelope=page.envelope,
            reset=page.reset,
            alarm=page.alarm,
            threshold=page.threshold,
        )

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        document = {
            "schema": SCORER_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "feature_count": int(self.reference.global_center.size),
            "kappa": float(self.kappa),
            "maximum_gap_seconds": float(self.maximum_gap_seconds),
            "threshold": self.threshold,
            "reference_shrinkage_count": int(self.reference.shrinkage_count),
            "direction_shrinkage_count": int(self.direction.shrinkage_count),
            "rank_lower_clip": float(self.rank_normalizer.lower_clip),
            "rank_upper_clip": float(self.rank_normalizer.upper_clip),
            "metadata": dict(metadata or {}),
        }
        payload = {
            "metadata_json": np.asarray(_canonical_json(document)),
            "reference_state_codes": self.reference.state_codes,
            "reference_counts": self.reference.counts,
            "reference_centers": self.reference.centers,
            "reference_covariances": self.reference.covariances,
            "reference_precisions": self.reference.precisions,
            "reference_global_center": self.reference.global_center,
            "reference_global_covariance": self.reference.global_covariance,
            "reference_global_precision": self.reference.global_precision,
            "direction_state_codes": self.direction.state_codes,
            "direction_fault_counts": self.direction.fault_counts,
            "directions": self.direction.directions,
            "global_direction": self.direction.global_direction,
            "rank_sorted_reference": self.rank_normalizer.sorted_reference,
        }
        return _atomic_npz(path, payload)

    @classmethod
    def load_npz(
        cls, path: str | os.PathLike[str]
    ) -> tuple["CaredBCScorer", dict[str, object]]:
        with np.load(path, allow_pickle=False) as archive:
            document = json.loads(str(archive["metadata_json"].item()))
            if (
                document.get("schema") != SCORER_SCHEMA
                or document.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError("unsupported CARED scorer schema")
            frozen_metadata = {
                "feature_count": MONITORING_RESIDUAL_COUNT,
                "kappa": PAGE_KAPPA,
                "maximum_gap_seconds": DEFAULT_GAP_SECONDS,
                "reference_shrinkage_count": STATE_SHRINKAGE_COUNT,
                "direction_shrinkage_count": STATE_SHRINKAGE_COUNT,
                "rank_lower_clip": RANK_Z_LOWER,
                "rank_upper_clip": RANK_Z_UPPER,
            }
            if any(document.get(name) != value for name, value in frozen_metadata.items()):
                raise ValueError("serialized scorer metadata violate the frozen protocol")
            reference = StateConditionalReference(
                archive["reference_state_codes"],
                archive["reference_counts"],
                archive["reference_centers"],
                archive["reference_covariances"],
                archive["reference_precisions"],
                archive["reference_global_center"],
                archive["reference_global_covariance"],
                archive["reference_global_precision"],
                int(document["reference_shrinkage_count"]),
            )
            direction = DirectionalGLRModel(
                archive["direction_state_codes"],
                archive["direction_fault_counts"],
                archive["directions"],
                archive["global_direction"],
                int(document["direction_shrinkage_count"]),
            )
            if reference.global_center.size != MONITORING_RESIDUAL_COUNT:
                raise ValueError("serialized scorer feature count violates the frozen protocol")
            if not np.array_equal(reference.state_codes, direction.state_codes):
                raise ValueError(
                    "serialized reference and direction state codes are inconsistent"
                )
            normalizer = RankNormalTransform(
                archive["rank_sorted_reference"],
                float(document["rank_lower_clip"]),
                float(document["rank_upper_clip"]),
            )
            scorer = cls(
                reference,
                direction,
                normalizer,
                float(document["kappa"]),
                float(document["maximum_gap_seconds"]),
                None if document["threshold"] is None else float(document["threshold"]),
            )
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("serialized scorer metadata must be an object")
        return scorer, dict(metadata)


def fit_cared_scorer(
    fit_residuals: Any,
    fit_state_codes: Any,
    development_healthy_residuals: Any,
    development_healthy_state_codes: Any,
    f1_residuals: Any,
    f1_state_codes: Any,
    *,
    shrinkage_count: int = STATE_SHRINKAGE_COUNT,
) -> CaredBCScorer:
    """Fit covariance on fit, then direction/rank only on dev healthy and F1."""
    fit_values = _validate_residual_matrix(
        fit_residuals, name="fit_residuals", minimum_rows=2
    )
    fit_codes = _validate_state_codes(fit_state_codes, count=fit_values.shape[0])
    healthy = _validate_residual_matrix(
        development_healthy_residuals,
        name="development_healthy_residuals",
        minimum_rows=2,
    )
    healthy_codes = _validate_state_codes(
        development_healthy_state_codes, count=healthy.shape[0]
    )
    if healthy.shape[1] != fit_values.shape[1]:
        raise ValueError("fit and development healthy feature counts differ")
    reference = fit_state_conditional_reference(
        fit_values, fit_codes, shrinkage_count=shrinkage_count
    )
    direction = fit_f1_directional_glr(
        reference,
        healthy,
        healthy_codes,
        f1_residuals,
        f1_state_codes,
        shrinkage_count=shrinkage_count,
    )
    energy = state_conditional_energy(healthy, healthy_codes, reference)
    directional = directional_glr_scores(healthy, healthy_codes, reference, direction)
    normalizer = RankNormalTransform.fit(np.column_stack((energy, directional)))
    return CaredBCScorer(reference, direction, normalizer)


def calibrate_cared_scorer(
    scorer: CaredBCScorer,
    calibration_residuals: Any,
    calibration_state_codes: Any,
    calibration_timestamps: Any,
    *,
    alpha: float = 0.05,
    minimum_blocks: int = MINIMUM_CALIBRATION_BLOCKS,
) -> tuple[CaredBCScorer, BlockConformalCalibration, DailyMaxima]:
    """Replay calibration days and bind their block-conformal threshold."""
    if scorer.threshold is not None:
        raise ValueError("scorer is already calibrated")
    path = scorer.score(
        calibration_residuals, calibration_state_codes, calibration_timestamps
    )
    blocks = daily_maxima(
        CUSUMPath(
            path.standardized_evidence,
            path.cusum,
            path.envelope,
            path.reset,
            path.alarm,
            None,
        ),
        calibration_timestamps,
    )
    calibration = calibrate_block_conformal(
        blocks.maxima, alpha=alpha, minimum_blocks=minimum_blocks
    )
    return scorer.with_threshold(calibration.threshold), calibration, blocks
