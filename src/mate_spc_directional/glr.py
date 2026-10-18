"""Auditable directional density-ratio Page-CUSUM primitives.

This module is the independent NumPy reference for the directional monitor.
It deliberately has no PyTorch dependency; CUDA fitting and scoring live in
``mate_spc_directional.gpu`` and import PyTorch only inside called functions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FAULT_IDS = (3, 9, 15)
NORMAL_ID = 0
TRAIN_START = 219
TRAIN_STOP = 500
PERSISTENCE_WINDOW = 3
Z_CLIP = 8.0
ROBUST_SCALE_FACTOR = 1.4826
MODEL_SCHEMA = "mate_spc_directional_glr"
MODEL_SCHEMA_VERSION = 1


def _head_vector(
    value: float | Sequence[float] | np.ndarray,
    *,
    name: str,
    positive: bool,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), len(FAULT_IDS))
    if array.shape != (len(FAULT_IDS),) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite scalar or length-{len(FAULT_IDS)}")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not positive and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.array(array, dtype=np.float64, copy=True)


def normalize_kappa(
    value: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Normalize a scalar/per-head Page reference allowance."""
    return _head_vector(value, name="kappa", positive=False)


def _readonly_float_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional float array")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DirectionalGLRModel:
    """Frozen three-head density-ratio model and Page-CUSUM parameters."""

    fault_ids: tuple[int, int, int]
    weights: np.ndarray
    bias: np.ndarray
    normal_logit_center: np.ndarray
    normal_logit_scale: np.ndarray
    kappa: np.ndarray
    threshold: float | None = None

    def __post_init__(self) -> None:
        fault_ids = tuple(int(value) for value in self.fault_ids)
        if fault_ids != FAULT_IDS:
            raise ValueError(f"fault_ids must be exactly {FAULT_IDS}")
        weights = _readonly_float_array(self.weights, ndim=2, name="weights")
        if weights.shape[0] != len(FAULT_IDS) or weights.shape[1] < 1:
            raise ValueError("weights must have shape (3, positive_feature_count)")
        bias = _readonly_float_array(self.bias, ndim=1, name="bias")
        center = _readonly_float_array(
            self.normal_logit_center,
            ndim=1,
            name="normal_logit_center",
        )
        scale = _readonly_float_array(
            self.normal_logit_scale,
            ndim=1,
            name="normal_logit_scale",
        )
        kappa = normalize_kappa(self.kappa)
        for name, array in (
            ("bias", bias),
            ("normal_logit_center", center),
            ("normal_logit_scale", scale),
        ):
            if array.shape != (len(FAULT_IDS),):
                raise ValueError(f"{name} must have shape (3,)")
        if np.any(scale <= 0.0):
            raise ValueError("normal_logit_scale must be strictly positive")
        kappa.setflags(write=False)
        threshold = self.threshold
        if threshold is not None:
            threshold = float(threshold)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError("threshold must be finite and nonnegative")
        object.__setattr__(self, "fault_ids", fault_ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "normal_logit_center", center)
        object.__setattr__(self, "normal_logit_scale", scale)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "threshold", threshold)

    @property
    def feature_count(self) -> int:
        return int(self.weights.shape[1])

    @property
    def head_count(self) -> int:
        return len(self.fault_ids)

    def with_kappa(
        self,
        value: float | Sequence[float] | np.ndarray,
    ) -> "DirectionalGLRModel":
        return replace(self, kappa=normalize_kappa(value))

    def with_threshold(self, threshold: float | None) -> "DirectionalGLRModel":
        return replace(self, threshold=threshold)

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_directional_glr_model(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "DirectionalGLRModel":
        model, _ = load_directional_glr_model(path)
        return model


@dataclass(frozen=True)
class DirectionalGLRPath:
    """All causal arrays for one run, including the final family decision."""

    fault_ids: tuple[int, int, int]
    logits: np.ndarray
    standardized_logits: np.ndarray
    clipped_evidence: np.ndarray
    cusum: np.ndarray
    component_persistence: np.ndarray
    envelope: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    valid_start: int
    threshold: float | None

    @property
    def z_score(self) -> np.ndarray:
        return self.standardized_logits

    @property
    def persistence(self) -> np.ndarray:
        return self.envelope

    @property
    def component_score(self) -> np.ndarray:
        return self.cusum

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "logits": self.logits,
            "standardized_logits": self.standardized_logits,
            "clipped_evidence": self.clipped_evidence,
            "cusum": self.cusum,
            "component_persistence": self.component_persistence,
            "envelope": self.envelope,
            "eligible": self.eligible,
            "alarm": self.alarm,
        }


def _validate_nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validate_feature_matrix(
    values: Any,
    *,
    feature_count: int | None,
    minimum_length: int,
    finite_start: int,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < minimum_length or matrix.shape[1] < 1:
        raise ValueError(f"{label} must have shape (time, positive_feature_count)")
    if feature_count is not None and matrix.shape[1] != feature_count:
        raise ValueError(
            f"{label} has {matrix.shape[1]} features; expected {feature_count}"
        )
    if finite_start >= matrix.shape[0]:
        raise ValueError(f"{label} is too short for finite_start={finite_start}")
    if np.isinf(matrix).any():
        raise ValueError(f"{label} must not contain positive or negative infinity")
    if not np.isfinite(matrix[finite_start:]).all():
        raise ValueError(f"{label} must be finite from index {finite_start}")
    return np.array(matrix, dtype=np.float64, copy=True, order="C")


def validate_feature_mapping(
    features_by_run: Mapping[str, np.ndarray],
    *,
    feature_count: int | None = None,
    minimum_length: int,
    finite_start: int,
    require_equal_length: bool = False,
    label: str = "features",
) -> tuple[tuple[str, np.ndarray], ...]:
    """Validate named feature runs; NaN is allowed only before ``finite_start``."""
    minimum_length = _validate_nonnegative_integer(
        minimum_length,
        name="minimum_length",
    )
    finite_start = _validate_nonnegative_integer(finite_start, name="finite_start")
    if not isinstance(features_by_run, Mapping) or not features_by_run:
        raise ValueError(f"{label} must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in features_by_run):
        raise ValueError(f"{label} run names must be nonempty strings")
    records: list[tuple[str, np.ndarray]] = []
    inferred_count = feature_count
    for name in sorted(features_by_run):
        matrix = _validate_feature_matrix(
            features_by_run[name],
            feature_count=inferred_count,
            minimum_length=minimum_length,
            finite_start=finite_start,
            label=f"{label} run {name!r}",
        )
        if inferred_count is None:
            inferred_count = int(matrix.shape[1])
        records.append((name, matrix))
    if require_equal_length and len({matrix.shape[0] for _, matrix in records}) != 1:
        raise ValueError(f"{label} runs must have equal sequence length")
    return tuple(records)


def validate_labeled_feature_mapping(
    features_by_run: Mapping[str, np.ndarray],
    labels_by_run: Mapping[str, int],
    *,
    require_balanced: bool = True,
) -> tuple[tuple[str, int, np.ndarray], ...]:
    """Validate the fixed normal/3/9/15 training task and time interval."""
    if not isinstance(labels_by_run, Mapping):
        raise TypeError("labels_by_run must be a mapping")
    if set(features_by_run) != set(labels_by_run):
        raise ValueError("features_by_run and labels_by_run must have identical keys")
    features = validate_feature_mapping(
        features_by_run,
        minimum_length=TRAIN_STOP,
        finite_start=TRAIN_START,
        label="GLR training features",
    )
    records: list[tuple[str, int, np.ndarray]] = []
    counts = {label: 0 for label in (NORMAL_ID, *FAULT_IDS)}
    for name, matrix in features:
        raw_label = labels_by_run[name]
        if isinstance(raw_label, (bool, np.bool_)) or not isinstance(
            raw_label,
            (int, np.integer),
        ):
            raise ValueError(f"label for run {name!r} must be an integer")
        run_label = int(raw_label)
        if run_label not in counts:
            raise ValueError(f"unsupported label {run_label} for run {name!r}")
        counts[run_label] += 1
        records.append((name, run_label, matrix))
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"training requires normal and all fault classes; counts={counts}")
    if require_balanced and len(set(counts.values())) != 1:
        raise ValueError(f"training run counts must be class-balanced; counts={counts}")
    feature_counts = {matrix.shape[1] for _, _, matrix in records}
    if len(feature_counts) != 1:
        raise ValueError("all training runs must have the same feature count")
    return tuple(records)


def page_cusum_numpy(evidence: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Independent vectorized Page recursion using cumsum minus running minimum."""
    values = np.asarray(evidence, dtype=np.float64)
    allowance = normalize_kappa(kappa)
    if values.ndim != 2 or values.shape[1] != len(FAULT_IDS):
        raise ValueError("evidence must have shape (time, 3)")
    if values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError("evidence must be nonempty and finite")
    partial = np.cumsum(values - allowance, axis=0, dtype=np.float64)
    augmented = np.vstack((np.zeros((1, len(FAULT_IDS))), partial))
    running_minimum = np.minimum.accumulate(augmented, axis=0)[1:]
    return np.maximum(partial - running_minimum, 0.0)


def _path_from_eligible_logits(
    model: DirectionalGLRModel,
    eligible_logits: np.ndarray,
    *,
    length: int,
    valid_start: int,
) -> DirectionalGLRPath:
    z_valid = (
        eligible_logits - model.normal_logit_center
    ) / model.normal_logit_scale
    clipped_valid = np.clip(z_valid, -Z_CLIP, Z_CLIP)
    cusum_valid = page_cusum_numpy(clipped_valid, model.kappa)
    shape = (length, len(FAULT_IDS))
    logits = np.full(shape, np.nan, dtype=np.float64)
    standardized = np.full(shape, np.nan, dtype=np.float64)
    clipped = np.full(shape, np.nan, dtype=np.float64)
    cusum = np.full(shape, np.nan, dtype=np.float64)
    logits[valid_start:] = eligible_logits
    standardized[valid_start:] = z_valid
    clipped[valid_start:] = clipped_valid
    cusum[valid_start:] = cusum_valid

    first = valid_start + PERSISTENCE_WINDOW - 1
    component_persistence = np.full(shape, -np.inf, dtype=np.float64)
    for index in range(first, length):
        component_persistence[index] = np.min(
            cusum[index - PERSISTENCE_WINDOW + 1 : index + 1],
            axis=0,
        )
    envelope = np.full(length, -np.inf, dtype=np.float64)
    envelope[first:] = np.max(component_persistence[first:], axis=1)
    eligible = np.zeros(length, dtype=bool)
    eligible[first:] = True
    alarm = np.zeros(length, dtype=bool)
    if model.threshold is not None:
        alarm = eligible & (envelope > model.threshold)
    return DirectionalGLRPath(
        fault_ids=FAULT_IDS,
        logits=logits,
        standardized_logits=standardized,
        clipped_evidence=clipped,
        cusum=cusum,
        component_persistence=component_persistence,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm,
        valid_start=valid_start,
        threshold=model.threshold,
    )


def score_directional_glr_numpy(
    model: DirectionalGLRModel,
    features: np.ndarray,
    *,
    valid_start: int,
) -> DirectionalGLRPath:
    """Score one run with an independent NumPy implementation."""
    if not isinstance(model, DirectionalGLRModel):
        raise TypeError("model must be a DirectionalGLRModel")
    valid_start = _validate_nonnegative_integer(valid_start, name="valid_start")
    matrix = _validate_feature_matrix(
        features,
        feature_count=model.feature_count,
        minimum_length=valid_start + PERSISTENCE_WINDOW,
        finite_start=valid_start,
        label="directional scoring features",
    )
    eligible_logits = matrix[valid_start:] @ model.weights.T + model.bias
    if not np.isfinite(eligible_logits).all():
        raise FloatingPointError("directional logits are non-finite")
    return _path_from_eligible_logits(
        model,
        eligible_logits,
        length=matrix.shape[0],
        valid_start=valid_start,
    )


def score_directional_glr_runs_numpy(
    model: DirectionalGLRModel,
    features_by_run: Mapping[str, np.ndarray],
    *,
    valid_start: int,
) -> dict[str, DirectionalGLRPath]:
    """Score independent named runs without ever joining their Page states."""
    valid_start = _validate_nonnegative_integer(valid_start, name="valid_start")
    records = validate_feature_mapping(
        features_by_run,
        feature_count=model.feature_count,
        minimum_length=valid_start + PERSISTENCE_WINDOW,
        finite_start=valid_start,
        label="directional scoring features",
    )
    return {
        name: score_directional_glr_numpy(model, matrix, valid_start=valid_start)
        for name, matrix in records
    }


def _finite_maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise AssertionError(f"array shape mismatch: {left.shape} != {right.shape}")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise AssertionError("NaN locations differ")
    if not np.array_equal(np.isposinf(left), np.isposinf(right)):
        raise AssertionError("positive-infinity locations differ")
    if not np.array_equal(np.isneginf(left), np.isneginf(right)):
        raise AssertionError("negative-infinity locations differ")
    finite = np.isfinite(left)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def compare_directional_paths(
    reference: DirectionalGLRPath,
    candidate: DirectionalGLRPath,
    *,
    atol: float = 1e-10,
) -> dict[str, float | bool]:
    """Assert numeric path agreement and exact eligibility/alarm decisions."""
    if not np.isfinite(float(atol)) or float(atol) < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    if reference.fault_ids != candidate.fault_ids:
        raise AssertionError("fault head order differs")
    errors: dict[str, float] = {}
    for name in (
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
        "envelope",
    ):
        errors[name] = _finite_maximum_error(
            np.asarray(getattr(reference, name)),
            np.asarray(getattr(candidate, name)),
        )
        if errors[name] > float(atol):
            raise AssertionError(
                f"{name} maximum absolute error {errors[name]:.3e} exceeds {atol:.3e}"
            )
    eligible_equal = bool(np.array_equal(reference.eligible, candidate.eligible))
    alarm_equal = bool(np.array_equal(reference.alarm, candidate.alarm))
    if not eligible_equal or not alarm_equal:
        raise AssertionError("eligibility or alarm decisions differ")
    return {
        **{f"{name}_max_abs_error": value for name, value in errors.items()},
        "eligible_exact": eligible_equal,
        "alarm_exact": alarm_equal,
    }


def conformal_run_threshold(
    envelopes_by_run: Mapping[str, np.ndarray],
    *,
    alpha: float = 0.05,
    required_count: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the finite run-level conformal order statistic and audit metadata."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not isinstance(envelopes_by_run, Mapping) or not envelopes_by_run:
        raise ValueError("envelopes_by_run must be a nonempty mapping")
    if required_count is not None:
        required_count = _validate_nonnegative_integer(
            required_count,
            name="required_count",
        )
        if required_count < 1 or len(envelopes_by_run) != required_count:
            raise ValueError(
                f"expected exactly {required_count} calibration runs; "
                f"received {len(envelopes_by_run)}"
            )
    maxima: list[tuple[str, float]] = []
    for name in sorted(envelopes_by_run):
        if not isinstance(name, str) or not name:
            raise ValueError("calibration run names must be nonempty strings")
        values = np.asarray(envelopes_by_run[name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"envelope for {name!r} must be one-dimensional")
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size == 0:
            raise ValueError(f"envelope for {name!r} has no eligible score")
        first = int(finite[0])
        if not np.isneginf(values[:first]).all() or not np.isfinite(values[first:]).all():
            raise ValueError(
                f"envelope for {name!r} must be leading -inf then finite values"
            )
        maxima.append((name, float(np.max(values[first:]))))
    count = len(maxima)
    rank = int(ceil((count + 1) * (1.0 - float(alpha))))
    if rank > count:
        raise ValueError("finite conformal threshold is unavailable at this alpha")
    ordered = sorted(maxima, key=lambda item: (item[1], item[0]))
    driver_name, threshold = ordered[rank - 1]
    return threshold, {
        "alpha": float(alpha),
        "healthy_run_count": count,
        "rank_one_based": rank,
        "threshold": threshold,
        "rank_driver_run": driver_name,
        "comparison": "strictly_greater_than",
        "run_maxima": [
            {"run": name, "maximum": value} for name, value in maxima
        ],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON safe")


def save_directional_glr_model(
    model: DirectionalGLRModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save numeric arrays plus JSON metadata, never Python pickle."""
    if not isinstance(model, DirectionalGLRModel):
        raise TypeError("model must be a DirectionalGLRModel")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("directional model path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    supplied = {} if metadata is None else dict(metadata)
    metadata_text = json.dumps(
        supplied,
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    threshold = (
        np.empty(0, dtype=np.float64)
        if model.threshold is None
        else np.asarray([model.threshold], dtype=np.float64)
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(
                handle,
                schema=np.asarray(MODEL_SCHEMA),
                schema_version=np.asarray([MODEL_SCHEMA_VERSION], dtype=np.int64),
                fault_ids=np.asarray(model.fault_ids, dtype=np.int64),
                weights=model.weights,
                bias=model.bias,
                normal_logit_center=model.normal_logit_center,
                normal_logit_scale=model.normal_logit_scale,
                kappa=model.kappa,
                threshold=threshold,
                metadata_json=np.asarray(metadata_text),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def load_directional_glr_model(
    path: str | os.PathLike[str],
) -> tuple[DirectionalGLRModel, dict[str, Any]]:
    """Load the strict numeric/JSON schema with ``allow_pickle=False``."""
    target = Path(path)
    required = {
        "schema",
        "schema_version",
        "fault_ids",
        "weights",
        "bias",
        "normal_logit_center",
        "normal_logit_scale",
        "kappa",
        "threshold",
        "metadata_json",
    }
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != required:
            missing = sorted(required - set(archive.files))
            extra = sorted(set(archive.files) - required)
            raise ValueError(f"invalid model archive keys; missing={missing}, extra={extra}")
        schema_array = np.asarray(archive["schema"])
        version = np.asarray(archive["schema_version"])
        metadata_array = np.asarray(archive["metadata_json"])
        if schema_array.shape != () or str(schema_array.item()) != MODEL_SCHEMA:
            raise ValueError("unsupported directional model schema")
        if version.shape != (1,) or int(version[0]) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported directional model schema version")
        if metadata_array.shape != () or metadata_array.dtype.kind not in "US":
            raise ValueError("metadata_json must be a scalar Unicode string")
        threshold_array = np.asarray(archive["threshold"], dtype=np.float64)
        if threshold_array.shape not in ((0,), (1,)):
            raise ValueError("threshold archive field must contain zero or one value")
        model = DirectionalGLRModel(
            fault_ids=tuple(int(value) for value in archive["fault_ids"].tolist()),
            weights=archive["weights"],
            bias=archive["bias"],
            normal_logit_center=archive["normal_logit_center"],
            normal_logit_scale=archive["normal_logit_scale"],
            kappa=archive["kappa"],
            threshold=(
                None if threshold_array.size == 0 else float(threshold_array[0])
            ),
        )
        metadata = json.loads(str(metadata_array.item()))
    if not isinstance(metadata, dict):
        raise ValueError("model metadata must decode to a JSON object")
    return model, metadata


__all__ = [
    "FAULT_IDS",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "NORMAL_ID",
    "PERSISTENCE_WINDOW",
    "TRAIN_START",
    "TRAIN_STOP",
    "Z_CLIP",
    "DirectionalGLRModel",
    "DirectionalGLRPath",
    "compare_directional_paths",
    "conformal_run_threshold",
    "load_directional_glr_model",
    "normalize_kappa",
    "page_cusum_numpy",
    "save_directional_glr_model",
    "score_directional_glr_numpy",
    "score_directional_glr_runs_numpy",
    "validate_feature_mapping",
    "validate_labeled_feature_mapping",
]
