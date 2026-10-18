"""Causal hybrid directional and variance-sensitive energy features.

The transformer wraps an already-fitted :class:`DirectionalFeatureTransformer`.
It preserves every retained directional coordinate and appends independently
scaled rolling mean-square features.  Fitting this class uses healthy reference
runs only and never refits or otherwise mutates the wrapped transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np

from mate_spc_directional.features import DirectionalFeatureTransformer


FEATURE_COUNT = 52
PAST_ORDER = 5
RESIDUAL_DIMENSION = 204
ENERGY_WINDOWS = (5, 20, 60)
ROBUST_SCALE_FACTOR = 1.4826
OUTPUT_CLIP = 10.0
VALID_START = 63
ENERGY_RAW_FEATURE_COUNT = (RESIDUAL_DIMENSION + FEATURE_COUNT) * len(ENERGY_WINDOWS)
HYBRID_TRANSFORMER_SCHEMA = "mate_spc_hybrid_features"
HYBRID_TRANSFORMER_SCHEMA_VERSION = 1


def _energy_feature_names() -> tuple[str, ...]:
    residual = tuple(
        f"residual_ewma_z_r{coordinate:03d}_rolling_mean_square_w{window:02d}"
        for window in ENERGY_WINDOWS
        for coordinate in range(RESIDUAL_DIMENSION)
    )
    raw_difference = tuple(
        f"raw_z_x{coordinate:02d}_diff_rolling_mean_square_w{window:02d}"
        for window in ENERGY_WINDOWS
        for coordinate in range(FEATURE_COUNT)
    )
    names = residual + raw_difference
    if len(names) != ENERGY_RAW_FEATURE_COUNT or len(set(names)) != len(names):
        raise RuntimeError("internal hybrid energy-feature names are invalid")
    return names


ENERGY_FEATURE_NAMES = _energy_feature_names()


def _readonly_float64(values: object, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool(values: object, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind != "b":
        raise ValueError(f"{label} must have boolean dtype")
    result = np.array(array, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _validate_run(values: object, *, minimum_length: int = VALID_START + 1) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise ValueError(f"each run must have shape (time, {FEATURE_COUNT})")
    if matrix.shape[0] < minimum_length:
        raise ValueError(f"each run must contain at least {minimum_length} time samples")
    if not np.isfinite(matrix).all():
        raise ValueError("each run must contain only finite values")
    return matrix


def _materialize_runs(
    runs: Iterable[np.ndarray | tuple[str, np.ndarray]],
) -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    for item in runs:
        values: object
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError("named runs must be (name, matrix) pairs")
            values = item[1]
        else:
            values = item
        matrices.append(_validate_run(values))
    if not matrices:
        raise ValueError("at least one independent healthy reference run is required")
    return tuple(matrices)


def _rolling_mean_square(values: np.ndarray, window: int) -> np.ndarray:
    squared = values * values
    cumulative = np.vstack(
        (np.zeros((1, squared.shape[1]), dtype=np.float64), np.cumsum(squared, axis=0))
    )
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def _location_scale(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("energy-scaler fitting requires at least two finite rows")
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = ROBUST_SCALE_FACTOR * mad
    fallback = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(fallback):
        sample_sd = np.std(matrix, axis=0, ddof=1)
        scale = np.where(fallback, sample_sd, scale)
    keep = np.isfinite(center) & np.isfinite(scale) & (scale > 0.0)
    if not np.any(keep):
        raise ValueError("all hybrid energy features are degenerate")
    return center, scale, fallback, keep


def _validate_directional(transformer: DirectionalFeatureTransformer) -> None:
    if not isinstance(transformer, DirectionalFeatureTransformer):
        raise TypeError("directional must be a DirectionalFeatureTransformer")
    if not transformer.fitted_:
        raise RuntimeError("directional feature transformer must already be fitted")
    required = (
        transformer.residual_center_,
        transformer.residual_scale_,
        transformer.residual_scale_fallback_,
        transformer.feature_center_,
        transformer.feature_scale_,
        transformer.feature_scale_fallback_,
        transformer.keep_mask_,
    )
    if any(value is None for value in required):
        raise RuntimeError("fitted directional transformer state is incomplete")
    if transformer.valid_start_ != VALID_START:
        raise ValueError("directional transformer valid_start must be 63")
    if transformer.residual_center_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("directional residual center must have shape (204,)")
    if transformer.residual_scale_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("directional residual scale must have shape (204,)")
    if np.any(transformer.residual_scale_ <= 0.0):
        raise ValueError("directional residual scale must be strictly positive")
    if transformer.residual_scale_fallback_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("directional residual fallback must have shape (204,)")
    if transformer.residual_scale_fallback_.dtype.kind != "b":
        raise ValueError("directional residual fallback must have boolean dtype")
    if (
        transformer.keep_mask_.shape != (transformer.RAW_FEATURE_COUNT,)
        or transformer.keep_mask_.dtype.kind != "b"
        or not np.any(transformer.keep_mask_)
    ):
        raise ValueError("directional keep mask has an invalid shape")
    retained_count = int(np.sum(transformer.keep_mask_))
    for label, values in (
        ("feature center", transformer.feature_center_),
        ("feature scale", transformer.feature_scale_),
        ("feature fallback", transformer.feature_scale_fallback_),
    ):
        if values.shape != (retained_count,):
            raise ValueError(f"directional {label} has an invalid shape")
    if np.any(transformer.feature_scale_ <= 0.0):
        raise ValueError("directional feature scale must be strictly positive")
    if transformer.feature_scale_fallback_.dtype.kind != "b":
        raise ValueError("directional feature fallback must have boolean dtype")
    expected_names = tuple(
        name
        for name, retained in zip(transformer.RAW_FEATURE_NAMES, transformer.keep_mask_)
        if retained
    )
    if transformer.feature_names_ != expected_names:
        raise ValueError("directional retained feature names do not match its keep mask")


@dataclass
class HybridFeatureTransformer:
    """Append variance-sensitive causal energies to retained directional features."""

    directional: DirectionalFeatureTransformer
    energy_center_: np.ndarray | None = field(default=None, init=False, repr=False)
    energy_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    energy_scale_fallback_: np.ndarray | None = field(default=None, init=False, repr=False)
    energy_keep_mask_: np.ndarray | None = field(default=None, init=False, repr=False)
    energy_feature_names_: tuple[str, ...] = field(default=(), init=False)
    dropped_energy_feature_names_: tuple[str, ...] = field(default=(), init=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False)
    fitted_: bool = field(default=False, init=False)

    ENERGY_WINDOWS: ClassVar[tuple[int, ...]] = ENERGY_WINDOWS
    ENERGY_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ENERGY_FEATURE_NAMES
    ENERGY_RAW_FEATURE_COUNT: ClassVar[int] = ENERGY_RAW_FEATURE_COUNT
    OUTPUT_CLIP: ClassVar[float] = OUTPUT_CLIP
    VALID_START: ClassVar[int] = VALID_START

    def __post_init__(self) -> None:
        _validate_directional(self.directional)

    @property
    def valid_start_(self) -> int:
        return self.VALID_START

    @property
    def directional_feature_count_(self) -> int:
        return self.directional.output_feature_count_

    @property
    def energy_feature_count_(self) -> int:
        if not self.fitted_:
            raise RuntimeError("hybrid feature transformer has not been fitted")
        return len(self.energy_feature_names_)

    @property
    def output_feature_count_(self) -> int:
        if not self.fitted_:
            raise RuntimeError("hybrid feature transformer has not been fitted")
        return len(self.feature_names_)

    def _raw_energy_features(self, values: np.ndarray) -> np.ndarray:
        _validate_directional(self.directional)
        matrix = _validate_run(values)
        length = matrix.shape[0]

        residual = self.directional.signed_residual_ewma(matrix)[PAST_ORDER - 1 :]
        residual_z = (
            residual - self.directional.residual_center_
        ) / self.directional.residual_scale_
        raw_z = (matrix - self.directional.raw_center) / self.directional.raw_scale
        raw_differences = np.diff(raw_z, axis=0)

        parts: list[np.ndarray] = []
        for window in ENERGY_WINDOWS:
            path = np.full((length, RESIDUAL_DIMENSION), np.nan, dtype=np.float64)
            start = PAST_ORDER - 1 + window - 1
            path[start:] = _rolling_mean_square(residual_z, window)
            parts.append(path)
        for window in ENERGY_WINDOWS:
            path = np.full((length, FEATURE_COUNT), np.nan, dtype=np.float64)
            path[window:] = _rolling_mean_square(raw_differences, window)
            parts.append(path)

        combined = np.column_stack(parts)
        if combined.shape != (length, ENERGY_RAW_FEATURE_COUNT):
            raise RuntimeError("internal hybrid energy-feature shape mismatch")
        combined[:VALID_START] = np.nan
        if not np.isfinite(combined[VALID_START:]).all():
            raise RuntimeError("hybrid energy-feature calculation produced non-finite values")
        return combined

    def fit(
        self,
        healthy_reference_runs: Iterable[np.ndarray | tuple[str, np.ndarray]],
    ) -> "HybridFeatureTransformer":
        """Fit only the appended-energy scaler on independent healthy runs."""
        _validate_directional(self.directional)
        self.fitted_ = False
        matrices = _materialize_runs(healthy_reference_runs)
        raw_rows = np.concatenate(
            [self._raw_energy_features(matrix)[VALID_START:] for matrix in matrices],
            axis=0,
        )
        center, scale, fallback, keep = _location_scale(raw_rows)
        self.energy_keep_mask_ = _readonly_bool(keep, label="energy_keep_mask")
        self.energy_center_ = _readonly_float64(center[keep], label="energy_center")
        self.energy_scale_ = _readonly_float64(scale[keep], label="energy_scale")
        self.energy_scale_fallback_ = _readonly_bool(
            fallback[keep], label="energy_scale_fallback"
        )
        self.energy_feature_names_ = tuple(
            name for name, retained in zip(ENERGY_FEATURE_NAMES, keep) if retained
        )
        self.dropped_energy_feature_names_ = tuple(
            name for name, retained in zip(ENERGY_FEATURE_NAMES, keep) if not retained
        )
        self.feature_names_ = self.directional.feature_names_ + self.energy_feature_names_
        if len(set(self.feature_names_)) != len(self.feature_names_):
            raise RuntimeError("hybrid output feature names are not unique")
        self.fitted_ = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform one run, resetting all EWMA and rolling state at its boundary."""
        if (
            not self.fitted_
            or self.energy_keep_mask_ is None
            or self.energy_center_ is None
            or self.energy_scale_ is None
        ):
            raise RuntimeError("hybrid feature transformer has not been fitted")
        directional = self.directional.transform(values)
        raw_energy = self._raw_energy_features(values)
        energy = np.full(
            (raw_energy.shape[0], self.energy_feature_count_), np.nan, dtype=np.float64
        )
        retained = raw_energy[VALID_START:, self.energy_keep_mask_]
        standardized = (retained - self.energy_center_) / self.energy_scale_
        energy[VALID_START:] = np.clip(standardized, -OUTPUT_CLIP, OUTPUT_CLIP)
        output = np.column_stack((directional, energy))
        if output.shape != (raw_energy.shape[0], self.output_feature_count_):
            raise RuntimeError("internal hybrid output-feature shape mismatch")
        if not np.isnan(output[:VALID_START]).all():
            raise RuntimeError("hybrid features before valid_start must be NaN")
        if not np.isfinite(output[VALID_START:]).all():
            raise RuntimeError("scaled hybrid features are non-finite")
        return output

    def transform_runs(self, runs: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
        return tuple(self.transform(values) for values in runs)

    def transform_named_runs(
        self,
        runs: Iterable[tuple[str, np.ndarray]],
    ) -> tuple[tuple[str, np.ndarray], ...]:
        transformed: list[tuple[str, np.ndarray]] = []
        for item in runs:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("named runs must be (name, matrix) pairs")
            name, values = item
            if not isinstance(name, str) or not name:
                raise ValueError("run names must be nonempty strings")
            transformed.append((name, self.transform(values)))
        return tuple(transformed)

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_hybrid_feature_transformer(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "HybridFeatureTransformer":
        transformer, _ = load_hybrid_feature_transformer(path)
        return transformer


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON safe")


_ARCHIVE_KEYS = {
    "schema",
    "schema_version",
    "energy_windows",
    "valid_start",
    "energy_feature_names",
    "directional_raw_center",
    "directional_raw_scale",
    "directional_past_mean",
    "directional_residual_projection",
    "directional_cva_rank",
    "directional_residual_center",
    "directional_residual_scale",
    "directional_residual_scale_fallback",
    "directional_feature_center",
    "directional_feature_scale",
    "directional_feature_scale_fallback",
    "directional_keep_mask",
    "energy_center",
    "energy_scale",
    "energy_scale_fallback",
    "energy_keep_mask",
    "metadata_json",
}


def save_hybrid_feature_transformer(
    transformer: HybridFeatureTransformer,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save all hybrid and embedded directional state without pickle."""
    if not isinstance(transformer, HybridFeatureTransformer):
        raise TypeError("transformer must be a HybridFeatureTransformer")
    if not transformer.fitted_:
        raise RuntimeError("hybrid feature transformer has not been fitted")
    _validate_directional(transformer.directional)
    required = (
        transformer.energy_center_,
        transformer.energy_scale_,
        transformer.energy_scale_fallback_,
        transformer.energy_keep_mask_,
    )
    if any(value is None for value in required):
        raise RuntimeError("fitted hybrid transformer state is incomplete")

    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("hybrid transformer path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(
        {} if metadata is None else dict(metadata),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    directional = transformer.directional
    rank = -1 if directional.cva_rank is None else directional.cva_rank
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
                schema=np.asarray(HYBRID_TRANSFORMER_SCHEMA),
                schema_version=np.asarray([HYBRID_TRANSFORMER_SCHEMA_VERSION], dtype=np.int64),
                energy_windows=np.asarray(ENERGY_WINDOWS, dtype=np.int64),
                valid_start=np.asarray([VALID_START], dtype=np.int64),
                energy_feature_names=np.asarray(ENERGY_FEATURE_NAMES),
                directional_raw_center=directional.raw_center,
                directional_raw_scale=directional.raw_scale,
                directional_past_mean=directional.past_mean,
                directional_residual_projection=directional.residual_projection,
                directional_cva_rank=np.asarray([rank], dtype=np.int64),
                directional_residual_center=directional.residual_center_,
                directional_residual_scale=directional.residual_scale_,
                directional_residual_scale_fallback=directional.residual_scale_fallback_,
                directional_feature_center=directional.feature_center_,
                directional_feature_scale=directional.feature_scale_,
                directional_feature_scale_fallback=directional.feature_scale_fallback_,
                directional_keep_mask=directional.keep_mask_,
                energy_center=transformer.energy_center_,
                energy_scale=transformer.energy_scale_,
                energy_scale_fallback=transformer.energy_scale_fallback_,
                energy_keep_mask=transformer.energy_keep_mask_,
                metadata_json=np.asarray(metadata_text),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def _scalar_text(array: np.ndarray, *, label: str) -> str:
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{label} must be a scalar Unicode string")
    return str(array.item())


def _vector_int(array: np.ndarray, *, label: str, length: int) -> np.ndarray:
    if array.shape != (length,) or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be an integer vector with shape ({length},)")
    return np.asarray(array, dtype=np.int64)


def load_hybrid_feature_transformer(
    path: str | os.PathLike[str],
) -> tuple[HybridFeatureTransformer, dict[str, Any]]:
    """Load a strict self-contained hybrid archive with pickle disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != _ARCHIVE_KEYS:
            missing = sorted(_ARCHIVE_KEYS - set(archive.files))
            extra = sorted(set(archive.files) - _ARCHIVE_KEYS)
            raise ValueError(f"invalid hybrid archive keys; missing={missing}, extra={extra}")
        schema = _scalar_text(np.asarray(archive["schema"]), label="schema")
        version = _vector_int(
            np.asarray(archive["schema_version"]), label="schema_version", length=1
        )
        windows = _vector_int(
            np.asarray(archive["energy_windows"]),
            label="energy_windows",
            length=len(ENERGY_WINDOWS),
        )
        valid_start = _vector_int(
            np.asarray(archive["valid_start"]), label="valid_start", length=1
        )
        archived_names = np.asarray(archive["energy_feature_names"])
        metadata_text = _scalar_text(
            np.asarray(archive["metadata_json"]), label="metadata_json"
        )
        rank = _vector_int(
            np.asarray(archive["directional_cva_rank"]),
            label="directional_cva_rank",
            length=1,
        )
        arrays = {name: np.asarray(archive[name]) for name in _ARCHIVE_KEYS if name not in {
            "schema",
            "schema_version",
            "energy_windows",
            "valid_start",
            "energy_feature_names",
            "directional_cva_rank",
            "metadata_json",
        }}

    if schema != HYBRID_TRANSFORMER_SCHEMA:
        raise ValueError("unsupported hybrid transformer schema")
    if int(version[0]) != HYBRID_TRANSFORMER_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid transformer schema version")
    if tuple(int(item) for item in windows) != ENERGY_WINDOWS:
        raise ValueError("archived energy windows do not match the frozen design")
    if int(valid_start[0]) != VALID_START:
        raise ValueError("archived valid_start does not match the frozen design")
    if archived_names.dtype.kind not in "US" or archived_names.shape != (
        ENERGY_RAW_FEATURE_COUNT,
    ):
        raise ValueError("archived energy feature names have an invalid representation")
    if tuple(str(item) for item in archived_names) != ENERGY_FEATURE_NAMES:
        raise ValueError("archived energy feature names do not match the frozen design")

    rank_value = int(rank[0])
    if rank_value not in (-1, 56):
        raise ValueError("invalid archived CVA rank")
    directional = DirectionalFeatureTransformer(
        raw_center=arrays["directional_raw_center"],
        raw_scale=arrays["directional_raw_scale"],
        past_mean=arrays["directional_past_mean"],
        residual_projection=arrays["directional_residual_projection"],
        cva_rank=None if rank_value == -1 else rank_value,
    )
    directional.residual_center_ = _readonly_float64(
        arrays["directional_residual_center"], label="directional_residual_center"
    )
    directional.residual_scale_ = _readonly_float64(
        arrays["directional_residual_scale"], label="directional_residual_scale"
    )
    directional.residual_scale_fallback_ = _readonly_bool(
        arrays["directional_residual_scale_fallback"],
        label="directional_residual_scale_fallback",
    )
    directional.feature_center_ = _readonly_float64(
        arrays["directional_feature_center"], label="directional_feature_center"
    )
    directional.feature_scale_ = _readonly_float64(
        arrays["directional_feature_scale"], label="directional_feature_scale"
    )
    directional.feature_scale_fallback_ = _readonly_bool(
        arrays["directional_feature_scale_fallback"],
        label="directional_feature_scale_fallback",
    )
    directional.keep_mask_ = _readonly_bool(
        arrays["directional_keep_mask"], label="directional_keep_mask"
    )
    if directional.residual_center_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("archived directional residual center has an invalid shape")
    if directional.residual_scale_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("archived directional residual scale has an invalid shape")
    if directional.residual_scale_fallback_.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("archived directional residual fallback has an invalid shape")
    if directional.keep_mask_.shape != (directional.RAW_FEATURE_COUNT,):
        raise ValueError("archived directional keep mask has an invalid shape")
    retained_directional = int(np.sum(directional.keep_mask_))
    for label, values in (
        ("directional_feature_center", directional.feature_center_),
        ("directional_feature_scale", directional.feature_scale_),
        ("directional_feature_fallback", directional.feature_scale_fallback_),
    ):
        if values.shape != (retained_directional,):
            raise ValueError(f"archived {label} has an invalid shape")
    directional.feature_names_ = tuple(
        name
        for name, retained in zip(directional.RAW_FEATURE_NAMES, directional.keep_mask_)
        if retained
    )
    directional.dropped_feature_names_ = tuple(
        name
        for name, retained in zip(directional.RAW_FEATURE_NAMES, directional.keep_mask_)
        if not retained
    )
    directional.fitted_ = True
    _validate_directional(directional)

    energy_keep = _readonly_bool(arrays["energy_keep_mask"], label="energy_keep_mask")
    energy_center = _readonly_float64(arrays["energy_center"], label="energy_center")
    energy_scale = _readonly_float64(arrays["energy_scale"], label="energy_scale")
    energy_fallback = _readonly_bool(
        arrays["energy_scale_fallback"], label="energy_scale_fallback"
    )
    if energy_keep.shape != (ENERGY_RAW_FEATURE_COUNT,) or not np.any(energy_keep):
        raise ValueError("archived energy keep mask is invalid")
    retained_energy = int(np.sum(energy_keep))
    for label, values in (
        ("energy_center", energy_center),
        ("energy_scale", energy_scale),
        ("energy_scale_fallback", energy_fallback),
    ):
        if values.shape != (retained_energy,):
            raise ValueError(f"archived {label} has an invalid shape")
    if np.any(energy_scale <= 0.0):
        raise ValueError("archived energy scale must be strictly positive")

    try:
        metadata_value = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid hybrid transformer metadata JSON") from error
    if not isinstance(metadata_value, dict):
        raise ValueError("hybrid transformer metadata must decode to a JSON object")

    hybrid = HybridFeatureTransformer(directional)
    hybrid.energy_keep_mask_ = energy_keep
    hybrid.energy_center_ = energy_center
    hybrid.energy_scale_ = energy_scale
    hybrid.energy_scale_fallback_ = energy_fallback
    hybrid.energy_feature_names_ = tuple(
        name for name, retained in zip(ENERGY_FEATURE_NAMES, energy_keep) if retained
    )
    hybrid.dropped_energy_feature_names_ = tuple(
        name for name, retained in zip(ENERGY_FEATURE_NAMES, energy_keep) if not retained
    )
    hybrid.feature_names_ = directional.feature_names_ + hybrid.energy_feature_names_
    hybrid.fitted_ = True
    return hybrid, metadata_value


__all__ = [
    "ENERGY_FEATURE_NAMES",
    "ENERGY_RAW_FEATURE_COUNT",
    "ENERGY_WINDOWS",
    "HYBRID_TRANSFORMER_SCHEMA",
    "HYBRID_TRANSFORMER_SCHEMA_VERSION",
    "HybridFeatureTransformer",
    "load_hybrid_feature_transformer",
    "save_hybrid_feature_transformer",
]
