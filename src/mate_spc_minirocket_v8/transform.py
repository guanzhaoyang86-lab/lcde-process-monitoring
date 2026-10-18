"""Strict adapter around the vendored tsai v1.0.1 MiniROCKET transform."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import random
import uuid
from typing import Any

# Preserve the required Windows DLL load order.
import torch
import numpy as np

from vendor.tsai_v1_0_1_minirocket import MiniRocketFeaturesPlus

from .model import (
    CHANNEL_COUNT,
    EXTRACTION_BATCH_SIZE,
    FEATURE_COUNT,
    NORMALIZATION_CLIP,
    TRAINING_EXAMPLE_COUNT,
    TrainingWindows,
    WINDOW_LENGTH,
)


REQUESTED_FEATURE_COUNT = 10_000
MAX_DILATIONS_PER_KERNEL = 32
KERNEL_LENGTH = 9
KERNEL_COUNT = 84
MAX_NUM_CHANNELS = 9
ADD_LSAZ = False
BIAS_FIT_CHUNK_SIZE = 1_260
STATE_SCHEMA = "gpu_minirocket_logodds_v8_transform_state"
EXPECTED_DILATIONS = np.arange(1, 16, dtype=np.int32)
EXPECTED_PADDING = np.arange(4, 61, 4, dtype=np.int64)
EXPECTED_FEATURES_PER_DILATION = np.asarray(
    [30, 19, 12, 12, 8, 4, 8, 4, 4, 3, 3, 3, 3, 3, 3], dtype=np.int32
)
EXPECTED_STATE_NAMES = (
    "kernels",
    *(f"channel_combinations_{index}" for index in range(15)),
    *(f"biases_{index}" for index in range(15)),
    "prefit",
)


def _configuration() -> dict[str, int | bool]:
    return {
        "c_in": CHANNEL_COUNT,
        "seq_len": WINDOW_LENGTH,
        "num_features": REQUESTED_FEATURE_COUNT,
        "max_dilations_per_kernel": MAX_DILATIONS_PER_KERNEL,
        "kernel_size": KERNEL_LENGTH,
        "max_num_channels": MAX_NUM_CHANNELS,
        "max_num_kernels": KERNEL_COUNT,
        "add_lsaz": ADD_LSAZ,
    }


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, order="C", copy=True)
    result.setflags(write=False)
    return result


def _strict_structure_array(
    value: object, *, name: str, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class MiniRocketTransformState:
    """No-pickle portable state sufficient to reproduce every PPV feature."""

    state_items: tuple[tuple[str, np.ndarray], ...]
    indices: np.ndarray
    dilations: np.ndarray
    padding: np.ndarray
    num_features_per_dilation: np.ndarray
    num_channels_per_combination: np.ndarray
    source_device: str

    def __post_init__(self) -> None:
        if type(self.state_items) is not tuple:
            raise ValueError("state_items must be a tuple")
        if any(type(item) is not tuple or len(item) != 2 for item in self.state_items):
            raise ValueError("each state item must be a (name,array) tuple")
        names = tuple(item[0] for item in self.state_items)
        if names != EXPECTED_STATE_NAMES:
            raise ValueError("MiniROCKET state_dict names/order changed")
        arrays = {name: value for name, value in self.state_items}
        kernels = _strict_structure_array(
            arrays["kernels"],
            name="kernels",
            dtype=np.dtype(np.float32),
            shape=(CHANNEL_COUNT * KERNEL_COUNT, 1, KERNEL_LENGTH),
        )
        prefit = _strict_structure_array(
            arrays["prefit"],
            name="prefit",
            dtype=np.dtype(np.bool_),
            shape=(1,),
        )
        if not bool(prefit[0]):
            raise ValueError("MiniROCKET state must have fitted biases")

        indices = _strict_structure_array(
            self.indices,
            name="indices",
            dtype=np.dtype(np.int64),
            shape=(KERNEL_COUNT, 1, 3),
        )
        expected_indices = np.asarray(
            list(itertools.combinations(range(KERNEL_LENGTH), 3)), dtype=np.int64
        )[:, None, :]
        if not np.array_equal(indices, expected_indices):
            raise ValueError("MiniROCKET does not contain all 84 frozen kernels")
        base_kernels = -np.ones((KERNEL_COUNT, KERNEL_LENGTH), dtype=np.float32)
        for row, positions in enumerate(expected_indices[:, 0, :]):
            base_kernels[row, positions] = 2.0
        expected_kernels = np.tile(base_kernels, (CHANNEL_COUNT, 1))[:, None, :]
        if not np.array_equal(kernels, expected_kernels):
            raise ValueError("MiniROCKET kernel coefficients/order changed")

        dilations = _strict_structure_array(
            self.dilations,
            name="dilations",
            dtype=np.dtype(np.int32),
            shape=(15,),
        )
        padding = _strict_structure_array(
            self.padding,
            name="padding",
            dtype=np.dtype(np.int64),
            shape=(15,),
        )
        features_per_dilation = _strict_structure_array(
            self.num_features_per_dilation,
            name="num_features_per_dilation",
            dtype=np.dtype(np.int32),
            shape=(15,),
        )
        channels_per_combination = _strict_structure_array(
            self.num_channels_per_combination,
            name="num_channels_per_combination",
            dtype=np.dtype(np.int32),
            shape=(15 * KERNEL_COUNT,),
        )
        if not np.array_equal(dilations, EXPECTED_DILATIONS):
            raise ValueError("MiniROCKET dilation allocation changed")
        if not np.array_equal(padding, EXPECTED_PADDING):
            raise ValueError("MiniROCKET padding allocation changed")
        if not np.array_equal(features_per_dilation, EXPECTED_FEATURES_PER_DILATION):
            raise ValueError("MiniROCKET feature allocation changed")
        if int(features_per_dilation.sum()) * KERNEL_COUNT != FEATURE_COUNT:
            raise ValueError("MiniROCKET effective feature count changed")
        if np.any((channels_per_combination < 1) | (channels_per_combination > 9)):
            raise ValueError("MiniROCKET channel subset sizes must be in 1..9")

        for index, features_this_dilation in enumerate(features_per_dilation.tolist()):
            channel_name = f"channel_combinations_{index}"
            channel_mask = _strict_structure_array(
                arrays[channel_name],
                name=channel_name,
                dtype=np.dtype(np.float32),
                shape=(1, CHANNEL_COUNT, KERNEL_COUNT, 1),
            )
            if not np.all((channel_mask == 0.0) | (channel_mask == 1.0)):
                raise ValueError(f"{channel_name} must be a binary mask")
            actual_counts = channel_mask.sum(axis=(0, 1, 3), dtype=np.float32).astype(
                np.int32
            )
            expected_counts = channels_per_combination[
                index * KERNEL_COUNT : (index + 1) * KERNEL_COUNT
            ]
            if not np.array_equal(actual_counts, expected_counts):
                raise ValueError(f"{channel_name} disagrees with channel subset sizes")
            bias_name = f"biases_{index}"
            _strict_structure_array(
                arrays[bias_name],
                name=bias_name,
                dtype=np.dtype(np.float32),
                shape=(KERNEL_COUNT, features_this_dilation),
            )
        if not isinstance(self.source_device, str) or not self.source_device:
            raise ValueError("source_device must be a nonempty string")

        object.__setattr__(
            self,
            "state_items",
            tuple((name, _readonly(arrays[name])) for name in EXPECTED_STATE_NAMES),
        )
        for name, value in (
            ("indices", indices),
            ("dilations", dilations),
            ("padding", padding),
            ("num_features_per_dilation", features_per_dilation),
            ("num_channels_per_combination", channels_per_combination),
        ):
            object.__setattr__(self, name, _readonly(value))

    def state_dict_numpy(self) -> dict[str, np.ndarray]:
        return {name: np.array(value, copy=True) for name, value in self.state_items}

    @classmethod
    def from_model(cls, model: MiniRocketFeaturesPlus) -> "MiniRocketTransformState":
        if not isinstance(model, MiniRocketFeaturesPlus):
            raise TypeError("model must be the fixed vendored MiniRocketFeaturesPlus")
        state_items = tuple(
            (name, tensor.detach().cpu().numpy().copy())
            for name, tensor in model.state_dict().items()
        )
        return cls(
            state_items=state_items,
            indices=model.indices.detach().cpu().numpy().astype(np.int64, copy=True),
            dilations=np.asarray(model.dilations, dtype=np.int32),
            padding=np.asarray(model.padding, dtype=np.int64),
            num_features_per_dilation=np.asarray(
                model.num_features_per_dilation, dtype=np.int32
            ),
            num_channels_per_combination=np.asarray(
                model.num_channels_per_combination, dtype=np.int32
            ),
            source_device=str(model.kernels.device),
        )


def _validate_windows(
    windows: np.ndarray, *, exact_training_count: bool
) -> np.ndarray:
    if (
        not isinstance(windows, np.ndarray)
        or windows.dtype != np.dtype(np.float32)
        or windows.ndim != 3
        or windows.shape[0] < 1
        or windows.shape[1:] != (CHANNEL_COUNT, WINDOW_LENGTH)
    ):
        raise ValueError("windows must be float32 with shape (n,52,128), n >= 1")
    if exact_training_count and windows.shape[0] != TRAINING_EXAMPLE_COUNT:
        raise ValueError("bias fitting requires exactly 20,400 ordered windows")
    if not np.isfinite(windows).all() or np.any(np.abs(windows) > NORMALIZATION_CLIP):
        raise ValueError("windows must be finite and clipped to [-10,10]")
    return windows


def _target_device(device: Any) -> torch.device:
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must identify CPU or CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return target


def _new_model(
    device: Any, *, preserve_rng_state: bool
) -> MiniRocketFeaturesPlus:
    target = _target_device(device)
    python_state = random.getstate() if preserve_rng_state else None
    numpy_state = np.random.get_state() if preserve_rng_state else None
    torch_state = torch.random.get_rng_state().clone() if preserve_rng_state else None
    cuda_states = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if preserve_rng_state and torch.cuda.is_initialized()
        else None
    )
    try:
        model = MiniRocketFeaturesPlus(**_configuration()).to(target).eval()
    finally:
        if preserve_rng_state:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
    if (
        model.num_features != FEATURE_COUNT
        or model.num_kernels != KERNEL_COUNT
        or model.num_dilations != 15
    ):
        raise RuntimeError("vendored MiniROCKET structure differs from the v8 charter")
    return model


def _fit_transform_impl(
    windows: np.ndarray, *, device: Any, enforce_frozen_count: bool
) -> MiniRocketTransformState:
    values = _validate_windows(windows, exact_training_count=enforce_frozen_count)
    # The orchestrator seeds once before fold 0.  Each fit deliberately consumes
    # the continuous global stream; no fold-local reset is permitted.
    model = _new_model(device, preserve_rng_state=False)
    model.fit(values, chunksize=BIAS_FIT_CHUNK_SIZE)
    state = MiniRocketTransformState.from_model(model)
    if not bool(dict(state.state_items)["prefit"][0]):
        raise FloatingPointError("MiniROCKET bias fit did not complete")
    return state


def fit_minirocket_transform(
    training: TrainingWindows, device: Any = "cuda"
) -> MiniRocketTransformState:
    """Fit biases on exactly the 20,400 ordered fit-fold windows."""

    if not isinstance(training, TrainingWindows):
        raise TypeError("training must be TrainingWindows")
    return _fit_transform_impl(
        training.windows, device=device, enforce_frozen_count=True
    )


def fit_minirocket_transform_final_refit(
    training: TrainingWindows, device: Any = "cuda"
) -> MiniRocketTransformState:
    """Fit biases on the exact 34,000-window GO-only refit batch."""

    if not isinstance(training, TrainingWindows):
        raise TypeError("training must be TrainingWindows")
    if training.windows.shape[0] != 34_000 or training.fit_run_ids.size != 100:
        raise ValueError("final-refit training must contain 100 runs and 34,000 windows")
    return _fit_transform_impl(
        training.windows, device=device, enforce_frozen_count=False
    )


def _model_from_state(
    state: MiniRocketTransformState, *, device: Any
) -> MiniRocketFeaturesPlus:
    if not isinstance(state, MiniRocketTransformState):
        raise TypeError("state must be MiniRocketTransformState")
    target = _target_device(device)
    # Reconstruction is a scoring operation.  The upstream constructor creates
    # temporary random channel masks, so snapshot/restore all RNG streams before
    # loading the serialized masks to make scoring RNG-neutral.
    model = _new_model(target, preserve_rng_state=True)
    if not np.array_equal(np.asarray(model.dilations, dtype=np.int32), state.dilations):
        raise RuntimeError("restored MiniROCKET dilations differ")
    if not np.array_equal(
        np.asarray(model.num_features_per_dilation, dtype=np.int32),
        state.num_features_per_dilation,
    ):
        raise RuntimeError("restored MiniROCKET feature allocation differs")
    tensors = {
        name: torch.as_tensor(np.array(value, copy=True), device=target)
        for name, value in state.state_items
    }
    model.load_state_dict(tensors, strict=True)
    model.indices = torch.as_tensor(
        np.array(state.indices, copy=True), dtype=torch.int64, device=target
    )
    model.dilations = np.array(state.dilations, copy=True)
    model.padding = np.array(state.padding, copy=True).tolist()
    model.num_features_per_dilation = np.array(
        state.num_features_per_dilation, copy=True
    )
    model.num_channels_per_combination = np.array(
        state.num_channels_per_combination, copy=True
    )
    model.fitting = False
    model.eval()
    if not bool(model.prefit.detach().cpu().item()):
        raise RuntimeError("restored MiniROCKET state is not fitted")
    return model


def transform_windows(
    windows: np.ndarray,
    state: MiniRocketTransformState,
    device: Any = "cuda",
    batch_size: int = EXTRACTION_BATCH_SIZE,
) -> np.ndarray:
    """Extract all PPV features in deterministic batches without dropping rows."""

    values = _validate_windows(windows, exact_training_count=False)
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise ValueError("batch_size must be an integer")
    if int(batch_size) != EXTRACTION_BATCH_SIZE:
        raise ValueError(f"batch_size must be the frozen value {EXTRACTION_BATCH_SIZE}")
    target = _target_device(device)
    model = _model_from_state(state, device=target)
    output = np.empty((values.shape[0], FEATURE_COUNT), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, values.shape[0], EXTRACTION_BATCH_SIZE):
            stop = min(start + EXTRACTION_BATCH_SIZE, values.shape[0])
            tensor = torch.as_tensor(
                np.array(values[start:stop], dtype=np.float32, order="C", copy=True),
                dtype=torch.float32,
                device=target,
            )
            features = model(tensor)
            if features.shape != (stop - start, FEATURE_COUNT):
                raise RuntimeError("MiniROCKET returned an unexpected feature shape")
            if not bool(torch.isfinite(features).all().item()):
                raise FloatingPointError("MiniROCKET returned nonfinite features")
            output[start:stop] = features.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
    return output


def save_minirocket_state_npz(
    path: str | os.PathLike[str], state: MiniRocketTransformState
) -> None:
    """Atomically save the complete transform state without pickle."""

    if not isinstance(state, MiniRocketTransformState):
        raise TypeError("state must be MiniRocketTransformState")
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError("state destination directory does not exist")
    metadata = {
        "schema": STATE_SCHEMA,
        "configuration": _configuration(),
        "effective_feature_count": FEATURE_COUNT,
        "state_names": list(EXPECTED_STATE_NAMES),
        "source_device": state.source_device,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
        "indices": state.indices,
        "dilations": state.dilations,
        "padding": state.padding,
        "num_features_per_dilation": state.num_features_per_dilation,
        "num_channels_per_combination": state.num_channels_per_combination,
    }
    arrays.update({f"state__{name}": value for name, value in state.state_items})
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_minirocket_state_npz(
    path: str | os.PathLike[str],
) -> MiniRocketTransformState:
    """Load a no-pickle transform archive and revalidate every structure."""

    source = Path(path)
    expected_members = {
        "metadata_json",
        "indices",
        "dilations",
        "padding",
        "num_features_per_dilation",
        "num_channels_per_combination",
        *(f"state__{name}" for name in EXPECTED_STATE_NAMES),
    }
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != expected_members:
            raise ValueError("MiniROCKET state archive members changed")
        metadata_value = archive["metadata_json"]
        if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
            raise ValueError("state metadata must be a scalar string")
        raw_metadata = metadata_value.item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("state metadata is invalid JSON") from error
        expected_metadata = {
            "schema": STATE_SCHEMA,
            "configuration": _configuration(),
            "effective_feature_count": FEATURE_COUNT,
            "state_names": list(EXPECTED_STATE_NAMES),
            "source_device": metadata.get("source_device") if isinstance(metadata, dict) else None,
        }
        if metadata != expected_metadata:
            raise ValueError("state metadata differs from the frozen schema/configuration")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    return MiniRocketTransformState(
        state_items=tuple(
            (name, arrays[f"state__{name}"]) for name in EXPECTED_STATE_NAMES
        ),
        indices=arrays["indices"],
        dilations=arrays["dilations"],
        padding=arrays["padding"],
        num_features_per_dilation=arrays["num_features_per_dilation"],
        num_channels_per_combination=arrays["num_channels_per_combination"],
        source_device=metadata["source_device"],
    )


__all__ = [
    "ADD_LSAZ",
    "BIAS_FIT_CHUNK_SIZE",
    "EXPECTED_DILATIONS",
    "EXPECTED_FEATURES_PER_DILATION",
    "EXPECTED_PADDING",
    "EXPECTED_STATE_NAMES",
    "KERNEL_COUNT",
    "KERNEL_LENGTH",
    "MAX_DILATIONS_PER_KERNEL",
    "MAX_NUM_CHANNELS",
    "MiniRocketTransformState",
    "REQUESTED_FEATURE_COUNT",
    "STATE_SCHEMA",
    "fit_minirocket_transform",
    "fit_minirocket_transform_final_refit",
    "load_minirocket_state_npz",
    "save_minirocket_state_npz",
    "transform_windows",
]
