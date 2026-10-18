"""Leakage-safe directional features for the 52-channel Rieth TEP layout.

The control-pair indices in :data:`CONTROL_PAIRS` are zero based and assume the
fixed 52-column Rieth layout used by this repository.  In every pair the first
index is an XMV (controller output) and the second is the associated XMEAS
(process response).  The final ``(51, 21)`` pair is an explicitly frozen
physical auxiliary response for XMV 11, not an inferred controller pairing.

All temporal operations are performed on one run at a time.  In particular,
the CVA past block, EWMA state, differences, and rolling windows never cross a
run boundary.  Run names can be carried by :meth:`transform_named_runs`, but
they never participate in a numerical calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


FEATURE_COUNT = 52
PAST_ORDER = 5
EWMA_LAMBDA = 0.2
RESIDUAL_DIMENSION = 204
ROBUST_SCALE_FACTOR = 1.4826
OUTPUT_CLIP = 10.0
TRANSFORMER_SCHEMA = "mate_spc_directional_features"
TRANSFORMER_SCHEMA_VERSION = 1

RESIDUAL_WINDOWS = (1, 5, 20, 60)
CONTROL_WINDOWS = (20, 60)
CONTROL_LAGS = (0, 1, 2, 4, 8)

# Zero-based (XMV, XMEAS) positions in the repository's fixed 52-column array.
CONTROL_PAIRS = (
    (41, 1),
    (42, 2),
    (43, 0),
    (44, 3),
    (45, 4),
    (46, 9),
    (47, 11),
    (48, 14),
    (49, 18),
    (50, 20),
    (51, 16),
    (51, 21),
)

_CONTROL_STATISTIC_NAMES = (
    "diff_corr_u_leads_y_lag0",
    "diff_corr_u_leads_y_lag1",
    "diff_corr_u_leads_y_lag2",
    "diff_corr_u_leads_y_lag4",
    "diff_corr_u_leads_y_lag8",
    "normalized_directed_hysteresis_area",
    "normalized_hysteresis_area_abs",
    "log_diff_energy_u",
    "log_diff_energy_y",
    "diff_autocorr_maxabs_u",
    "diff_autocorr_maxabs_y",
)


def _readonly_float64(values: object, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_array(values: object, *, dtype: object) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validate_run(values: object, *, minimum_length: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise ValueError(f"each run must have shape (time, {FEATURE_COUNT})")
    if matrix.shape[0] < minimum_length:
        raise ValueError(
            f"each run must contain at least {minimum_length} time samples"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("each run must contain only finite values")
    return matrix


def _materialize_runs(
    runs: Iterable[np.ndarray | tuple[str, np.ndarray]],
    *,
    minimum_length: int,
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
        matrices.append(_validate_run(values, minimum_length=minimum_length))
    if not matrices:
        raise ValueError("at least one independent design run is required")
    return tuple(matrices)


def _location_scale(
    values: np.ndarray,
    *,
    allow_drop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return median, MAD/SD scale, SD-fallback mask, and retained mask."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("scaler fitting requires at least two finite rows")
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = ROBUST_SCALE_FACTOR * mad
    fallback = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(fallback):
        sample_sd = np.std(matrix, axis=0, ddof=1)
        scale = np.where(fallback, sample_sd, scale)
    keep = np.isfinite(center) & np.isfinite(scale) & (scale > 0.0)
    if not allow_drop and not np.all(keep):
        indices = np.flatnonzero(~keep).tolist()
        raise ValueError(
            "design residual scale remains degenerate at coordinates "
            f"{indices}"
        )
    if allow_drop and not np.any(keep):
        raise ValueError("all combined directional features are degenerate")
    return center, scale, fallback, keep


def _past_blocks(standardized: np.ndarray) -> np.ndarray:
    length = standardized.shape[0]
    return np.concatenate(
        [
            standardized[PAST_ORDER - 1 - lag : length - lag]
            for lag in range(PAST_ORDER)
        ],
        axis=1,
    )


def _row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Finite row-wise Pearson correlation, with zero for a constant side."""
    left_zero = left - np.mean(left, axis=1, keepdims=True)
    right_zero = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.sum(left_zero * right_zero, axis=1)
    denominator = np.sqrt(
        np.sum(left_zero * left_zero, axis=1)
        * np.sum(right_zero * right_zero, axis=1)
    )
    result = np.zeros(left.shape[0], dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return np.clip(result, -1.0, 1.0)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling sums over an already-valid two-dimensional causal path."""
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0))
    )
    return cumulative[window:] - cumulative[:-window]


def _pair_window_features(
    standardized: np.ndarray,
    *,
    controller_index: int,
    response_index: int,
    window: int,
) -> np.ndarray:
    """Vectorized causal features for one controller-response window family."""
    controller = sliding_window_view(standardized[:, controller_index], window)
    response = sliding_window_view(standardized[:, response_index], window)
    diff_controller = np.diff(controller, axis=1)
    diff_response = np.diff(response, axis=1)

    columns: list[np.ndarray] = []
    for lag in CONTROL_LAGS:
        if lag == 0:
            leading = diff_controller
            trailing = diff_response
        else:
            # Positive lag encodes controller change leading the response change.
            leading = diff_controller[:, :-lag]
            trailing = diff_response[:, lag:]
        columns.append(_row_correlation(leading, trailing))

    controller_zero = controller - np.mean(controller, axis=1, keepdims=True)
    response_zero = response - np.mean(response, axis=1, keepdims=True)
    directed_area = 0.5 * np.sum(
        controller_zero[:, :-1] * response_zero[:, 1:]
        - controller_zero[:, 1:] * response_zero[:, :-1],
        axis=1,
    )
    area_scale = np.sqrt(
        np.sum(controller_zero * controller_zero, axis=1)
        * np.sum(response_zero * response_zero, axis=1)
    )
    normalized_area = np.zeros(directed_area.shape, dtype=np.float64)
    np.divide(
        directed_area,
        area_scale,
        out=normalized_area,
        where=area_scale > 0.0,
    )
    columns.extend((normalized_area, np.abs(normalized_area)))
    columns.extend(
        (
            np.log1p(np.mean(diff_controller * diff_controller, axis=1)),
            np.log1p(np.mean(diff_response * diff_response, axis=1)),
        )
    )

    maximum_lag = min(20, (window - 2) // 2)
    for differences in (diff_controller, diff_response):
        autocorrelations = [
            np.abs(_row_correlation(differences[:, :-lag], differences[:, lag:]))
            for lag in range(2, maximum_lag + 1)
        ]
        columns.append(np.max(np.column_stack(autocorrelations), axis=1))
    result = np.column_stack(columns)
    if result.shape[1] != len(_CONTROL_STATISTIC_NAMES):
        raise RuntimeError("internal control-feature count mismatch")
    if not np.isfinite(result).all():
        raise RuntimeError("control-feature calculation produced non-finite values")
    return result


def _raw_feature_names() -> tuple[str, ...]:
    residual = tuple(
        f"residual_ewma_r{coordinate:03d}_rolling_sum_w{window:02d}"
        for window in RESIDUAL_WINDOWS
        for coordinate in range(RESIDUAL_DIMENSION)
    )
    control = tuple(
        f"control_u{controller:02d}_y{response:02d}_w{window:02d}_{statistic}"
        for controller, response in CONTROL_PAIRS
        for window in CONTROL_WINDOWS
        for statistic in _CONTROL_STATISTIC_NAMES
    )
    names = residual + control
    if len(names) != 1080 or len(set(names)) != len(names):
        raise RuntimeError("internal directional-feature names are invalid")
    return names


@dataclass
class DirectionalFeatureTransformer:
    """Fit design-only scaling and transform one causal run into features.

    Parameters are copied from a *frozen* p=5 CVA model.  ``residual_projection``
    is the residual slice ``projection[:, rank:]`` and must therefore have shape
    ``(260, 204)`` for the frozen Stage-A rank of 56.

    The raw feature family has 1,080 coordinates: 816 signed residual rolling
    sums plus 264 controller-response statistics.  Final median/MAD scaling uses
    sample-SD fallback, drops any still-degenerate raw feature, and clips retained
    standardized values to ``[-10, 10]``.  The first valid row is zero-based 63.
    """

    raw_center: np.ndarray
    raw_scale: np.ndarray
    past_mean: np.ndarray
    residual_projection: np.ndarray
    cva_rank: int | None = None

    residual_center_: np.ndarray | None = field(default=None, init=False, repr=False)
    residual_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    residual_scale_fallback_: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    feature_center_: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_scale_fallback_: np.ndarray | None = field(
        default=None, init=False, repr=False
    )
    keep_mask_: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_names_: tuple[str, ...] = field(default=(), init=False)
    dropped_feature_names_: tuple[str, ...] = field(default=(), init=False)
    fitted_: bool = field(default=False, init=False)

    FEATURE_COUNT: ClassVar[int] = FEATURE_COUNT
    PAST_ORDER: ClassVar[int] = PAST_ORDER
    RESIDUAL_DIMENSION: ClassVar[int] = RESIDUAL_DIMENSION
    EWMA_LAMBDA: ClassVar[float] = EWMA_LAMBDA
    VALID_START: ClassVar[int] = PAST_ORDER - 1 + max(RESIDUAL_WINDOWS) - 1
    RAW_FEATURE_COUNT: ClassVar[int] = 1080
    OUTPUT_CLIP: ClassVar[float] = OUTPUT_CLIP
    CONTROL_PAIR_ASSUMPTION: ClassVar[str] = (
        "zero-based (XMV, XMEAS) indices in the fixed 52-column Rieth layout; "
        "(51, 21) is a frozen physical auxiliary pair"
    )
    RAW_FEATURE_NAMES: ClassVar[tuple[str, ...]] = _raw_feature_names()

    def __post_init__(self) -> None:
        self.raw_center = _readonly_float64(self.raw_center, label="raw_center")
        self.raw_scale = _readonly_float64(self.raw_scale, label="raw_scale")
        self.past_mean = _readonly_float64(self.past_mean, label="past_mean")
        self.residual_projection = _readonly_float64(
            self.residual_projection, label="residual_projection"
        )
        if self.raw_center.shape != (FEATURE_COUNT,):
            raise ValueError("raw_center must have shape (52,)")
        if self.raw_scale.shape != (FEATURE_COUNT,) or np.any(self.raw_scale <= 0.0):
            raise ValueError("raw_scale must have shape (52,) and be strictly positive")
        past_dimension = FEATURE_COUNT * PAST_ORDER
        if self.past_mean.shape != (past_dimension,):
            raise ValueError("past_mean must have shape (260,)")
        if self.residual_projection.shape != (past_dimension, RESIDUAL_DIMENSION):
            raise ValueError("residual_projection must have shape (260, 204)")
        if self.cva_rank is not None:
            if (
                isinstance(self.cva_rank, (bool, np.bool_))
                or not isinstance(self.cva_rank, (int, np.integer))
                or int(self.cva_rank) != past_dimension - RESIDUAL_DIMENSION
            ):
                raise ValueError("cva_rank must be 56 for a 204-dimensional residual")
            self.cva_rank = int(self.cva_rank)

    @classmethod
    def from_cva_monitor(cls, monitor: object) -> "DirectionalFeatureTransformer":
        """Copy required arrays from a compatible frozen CVA monitor.

        The monitor is deliberately accessed by attributes instead of imported,
        keeping this NumPy feature package independent of ``mate_spc``.
        """
        required = ("raw_center_", "raw_scale_", "past_mean_", "projection_", "rank_")
        missing = [name for name in required if not hasattr(monitor, name)]
        if missing:
            raise ValueError(f"CVA monitor is missing frozen parameters: {missing}")
        rank = getattr(monitor, "rank_")
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
            raise ValueError("CVA rank must be an integer")
        projection = np.asarray(getattr(monitor, "projection_"), dtype=np.float64)
        if projection.ndim != 2 or not 0 <= int(rank) < projection.shape[1]:
            raise ValueError("CVA projection and rank are inconsistent")
        return cls(
            raw_center=np.asarray(getattr(monitor, "raw_center_")),
            raw_scale=np.asarray(getattr(monitor, "raw_scale_")),
            past_mean=np.asarray(getattr(monitor, "past_mean_")),
            residual_projection=projection[:, int(rank) :],
            cva_rank=int(rank),
        )

    @property
    def valid_start_(self) -> int:
        return self.VALID_START

    @property
    def output_feature_count_(self) -> int:
        if not self.fitted_:
            raise RuntimeError("directional feature transformer has not been fitted")
        return len(self.feature_names_)

    def signed_residual_ewma(self, values: np.ndarray) -> np.ndarray:
        """Return the signed residual EWMA path, reset to zero for this run."""
        matrix = _validate_run(values, minimum_length=PAST_ORDER)
        standardized = (matrix - self.raw_center) / self.raw_scale
        past = _past_blocks(standardized)
        coordinates = (past - self.past_mean) @ self.residual_projection
        smoothed = np.empty_like(coordinates)
        previous = np.zeros(RESIDUAL_DIMENSION, dtype=np.float64)
        for index, coordinate in enumerate(coordinates):
            previous = EWMA_LAMBDA * coordinate + (1.0 - EWMA_LAMBDA) * previous
            smoothed[index] = previous
        result = np.full(
            (matrix.shape[0], RESIDUAL_DIMENSION), np.nan, dtype=np.float64
        )
        result[PAST_ORDER - 1 :] = smoothed
        return result

    def _raw_features(self, values: np.ndarray) -> np.ndarray:
        if self.residual_center_ is None or self.residual_scale_ is None:
            raise RuntimeError("design residual scaling has not been fitted")
        matrix = _validate_run(values, minimum_length=self.VALID_START + 1)
        standardized = (matrix - self.raw_center) / self.raw_scale
        residual = self.signed_residual_ewma(matrix)[PAST_ORDER - 1 :]
        residual_z = (residual - self.residual_center_) / self.residual_scale_

        parts: list[np.ndarray] = []
        length = matrix.shape[0]
        for window in RESIDUAL_WINDOWS:
            path = np.full((length, RESIDUAL_DIMENSION), np.nan, dtype=np.float64)
            sums = _rolling_sum(residual_z, window) / np.sqrt(float(window))
            start = PAST_ORDER - 1 + window - 1
            path[start:] = sums
            parts.append(path)

        for controller, response in CONTROL_PAIRS:
            for window in CONTROL_WINDOWS:
                path = np.full(
                    (length, len(_CONTROL_STATISTIC_NAMES)),
                    np.nan,
                    dtype=np.float64,
                )
                path[window - 1 :] = _pair_window_features(
                    standardized,
                    controller_index=controller,
                    response_index=response,
                    window=window,
                )
                parts.append(path)
        combined = np.column_stack(parts)
        if combined.shape != (length, self.RAW_FEATURE_COUNT):
            raise RuntimeError("internal raw directional-feature shape mismatch")
        combined[: self.VALID_START] = np.nan
        if not np.isfinite(combined[self.VALID_START :]).all():
            raise RuntimeError("directional-feature calculation produced non-finite values")
        return combined

    def fit(
        self,
        design_runs: Iterable[np.ndarray | tuple[str, np.ndarray]],
    ) -> "DirectionalFeatureTransformer":
        """Fit residual and final feature scales on independent healthy runs."""
        self.fitted_ = False
        matrices = _materialize_runs(
            design_runs, minimum_length=self.VALID_START + 1
        )
        residual_rows = np.concatenate(
            [run[PAST_ORDER - 1 :] for run in map(self.signed_residual_ewma, matrices)],
            axis=0,
        )
        residual_center, residual_scale, residual_fallback, _ = _location_scale(
            residual_rows, allow_drop=False
        )
        self.residual_center_ = _readonly_array(residual_center, dtype=np.float64)
        self.residual_scale_ = _readonly_array(residual_scale, dtype=np.float64)
        self.residual_scale_fallback_ = _readonly_array(
            residual_fallback, dtype=bool
        )

        raw_rows = np.concatenate(
            [self._raw_features(matrix)[self.VALID_START :] for matrix in matrices],
            axis=0,
        )
        center, scale, fallback, keep = _location_scale(raw_rows, allow_drop=True)
        self.keep_mask_ = _readonly_array(keep, dtype=bool)
        self.feature_center_ = _readonly_array(center[keep], dtype=np.float64)
        self.feature_scale_ = _readonly_array(scale[keep], dtype=np.float64)
        self.feature_scale_fallback_ = _readonly_array(fallback[keep], dtype=bool)
        self.feature_names_ = tuple(
            name for name, retained in zip(self.RAW_FEATURE_NAMES, keep) if retained
        )
        self.dropped_feature_names_ = tuple(
            name for name, retained in zip(self.RAW_FEATURE_NAMES, keep) if not retained
        )
        self.fitted_ = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform one run; rows 0..62 are NaN and rows 63 onward are finite."""
        if (
            not self.fitted_
            or self.keep_mask_ is None
            or self.feature_center_ is None
            or self.feature_scale_ is None
        ):
            raise RuntimeError("directional feature transformer has not been fitted")
        raw = self._raw_features(values)
        output = np.full(
            (raw.shape[0], self.output_feature_count_), np.nan, dtype=np.float64
        )
        retained = raw[self.VALID_START :, self.keep_mask_]
        standardized = (retained - self.feature_center_) / self.feature_scale_
        output[self.VALID_START :] = np.clip(
            standardized, -self.OUTPUT_CLIP, self.OUTPUT_CLIP
        )
        if not np.isfinite(output[self.VALID_START :]).all():
            raise RuntimeError("scaled directional features are non-finite")
        return output

    def transform_runs(self, runs: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
        """Transform a list of independent runs without joining boundaries."""
        return tuple(self.transform(values) for values in runs)

    def transform_named_runs(
        self,
        runs: Iterable[tuple[str, np.ndarray]],
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Carry run names through unchanged; names cannot affect feature values."""
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
        """Atomically save fitted numeric state without Python pickle."""
        return save_directional_feature_transformer(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "DirectionalFeatureTransformer":
        """Load a strict, pickle-free fitted transformer archive."""
        transformer, _ = load_directional_feature_transformer(path)
        return transformer


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


def save_directional_feature_transformer(
    transformer: DirectionalFeatureTransformer,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a fitted transformer as arrays plus JSON metadata."""
    if not isinstance(transformer, DirectionalFeatureTransformer):
        raise TypeError("transformer must be a DirectionalFeatureTransformer")
    if not transformer.fitted_:
        raise RuntimeError("directional feature transformer has not been fitted")
    required_state = (
        transformer.residual_center_,
        transformer.residual_scale_,
        transformer.residual_scale_fallback_,
        transformer.feature_center_,
        transformer.feature_scale_,
        transformer.feature_scale_fallback_,
        transformer.keep_mask_,
    )
    if any(value is None for value in required_state):
        raise RuntimeError("fitted transformer state is incomplete")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("directional transformer path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(
        {} if metadata is None else dict(metadata),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    rank = -1 if transformer.cva_rank is None else transformer.cva_rank
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
                schema=np.asarray(TRANSFORMER_SCHEMA),
                schema_version=np.asarray(
                    [TRANSFORMER_SCHEMA_VERSION], dtype=np.int64
                ),
                raw_center=transformer.raw_center,
                raw_scale=transformer.raw_scale,
                past_mean=transformer.past_mean,
                residual_projection=transformer.residual_projection,
                cva_rank=np.asarray([rank], dtype=np.int64),
                residual_center=transformer.residual_center_,
                residual_scale=transformer.residual_scale_,
                residual_scale_fallback=transformer.residual_scale_fallback_,
                feature_center=transformer.feature_center_,
                feature_scale=transformer.feature_scale_,
                feature_scale_fallback=transformer.feature_scale_fallback_,
                keep_mask=transformer.keep_mask_,
                metadata_json=np.asarray(metadata_text),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def load_directional_feature_transformer(
    path: str | os.PathLike[str],
) -> tuple[DirectionalFeatureTransformer, dict[str, Any]]:
    """Load and validate the strict transformer schema with pickle disabled."""
    required = {
        "schema",
        "schema_version",
        "raw_center",
        "raw_scale",
        "past_mean",
        "residual_projection",
        "cva_rank",
        "residual_center",
        "residual_scale",
        "residual_scale_fallback",
        "feature_center",
        "feature_scale",
        "feature_scale_fallback",
        "keep_mask",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != required:
            missing = sorted(required - set(archive.files))
            extra = sorted(set(archive.files) - required)
            raise ValueError(
                f"invalid transformer archive keys; missing={missing}, extra={extra}"
            )
        schema = np.asarray(archive["schema"])
        version = np.asarray(archive["schema_version"])
        rank_array = np.asarray(archive["cva_rank"])
        metadata_array = np.asarray(archive["metadata_json"])
        if schema.shape != () or str(schema.item()) != TRANSFORMER_SCHEMA:
            raise ValueError("unsupported directional transformer schema")
        if version.shape != (1,) or int(version[0]) != TRANSFORMER_SCHEMA_VERSION:
            raise ValueError("unsupported directional transformer schema version")
        if rank_array.shape != (1,):
            raise ValueError("cva_rank archive field must have shape (1,)")
        rank_value = int(rank_array[0])
        expected_rank = FEATURE_COUNT * PAST_ORDER - RESIDUAL_DIMENSION
        if rank_value not in (-1, expected_rank):
            raise ValueError("invalid archived CVA rank")
        if metadata_array.shape != () or metadata_array.dtype.kind not in "US":
            raise ValueError("metadata_json must be a scalar Unicode string")
        transformer = DirectionalFeatureTransformer(
            raw_center=archive["raw_center"],
            raw_scale=archive["raw_scale"],
            past_mean=archive["past_mean"],
            residual_projection=archive["residual_projection"],
            cva_rank=None if rank_value == -1 else rank_value,
        )
        residual_center = _readonly_float64(
            archive["residual_center"], label="residual_center"
        )
        residual_scale = _readonly_float64(
            archive["residual_scale"], label="residual_scale"
        )
        residual_fallback = _readonly_array(
            archive["residual_scale_fallback"], dtype=bool
        )
        feature_center = _readonly_float64(
            archive["feature_center"], label="feature_center"
        )
        feature_scale = _readonly_float64(
            archive["feature_scale"], label="feature_scale"
        )
        feature_fallback = _readonly_array(
            archive["feature_scale_fallback"], dtype=bool
        )
        keep_mask = _readonly_array(archive["keep_mask"], dtype=bool)
        metadata_text = str(metadata_array.item())

    if residual_center.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("archived residual_center must have shape (204,)")
    if residual_scale.shape != (RESIDUAL_DIMENSION,) or np.any(residual_scale <= 0):
        raise ValueError("archived residual_scale must be positive with shape (204,)")
    if residual_fallback.shape != (RESIDUAL_DIMENSION,):
        raise ValueError("archived residual fallback mask must have shape (204,)")
    if keep_mask.shape != (transformer.RAW_FEATURE_COUNT,) or not np.any(keep_mask):
        raise ValueError("archived keep mask is invalid")
    retained = int(np.sum(keep_mask))
    for name, values in (
        ("feature_center", feature_center),
        ("feature_scale", feature_scale),
        ("feature fallback mask", feature_fallback),
    ):
        if values.shape != (retained,):
            raise ValueError(f"archived {name} has an invalid shape")
    if np.any(feature_scale <= 0):
        raise ValueError("archived feature_scale must be strictly positive")
    try:
        metadata_value = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid transformer metadata JSON") from error
    if not isinstance(metadata_value, dict):
        raise ValueError("transformer metadata must decode to a JSON object")

    transformer.residual_center_ = residual_center
    transformer.residual_scale_ = residual_scale
    transformer.residual_scale_fallback_ = residual_fallback
    transformer.feature_center_ = feature_center
    transformer.feature_scale_ = feature_scale
    transformer.feature_scale_fallback_ = feature_fallback
    transformer.keep_mask_ = keep_mask
    transformer.feature_names_ = tuple(
        name
        for name, retained_flag in zip(transformer.RAW_FEATURE_NAMES, keep_mask)
        if retained_flag
    )
    transformer.dropped_feature_names_ = tuple(
        name
        for name, retained_flag in zip(transformer.RAW_FEATURE_NAMES, keep_mask)
        if not retained_flag
    )
    transformer.fitted_ = True
    return transformer, metadata_value


__all__ = [
    "CONTROL_LAGS",
    "CONTROL_PAIRS",
    "CONTROL_WINDOWS",
    "EWMA_LAMBDA",
    "OUTPUT_CLIP",
    "RESIDUAL_DIMENSION",
    "RESIDUAL_WINDOWS",
    "TRANSFORMER_SCHEMA",
    "TRANSFORMER_SCHEMA_VERSION",
    "DirectionalFeatureTransformer",
    "load_directional_feature_transformer",
    "save_directional_feature_transformer",
]
