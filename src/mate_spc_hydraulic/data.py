"""Safe, causal data preparation for UCI Hydraulic Systems (UCI 447).

The public loader deliberately requires an explicit list of zero-based cycle
indices.  Sensor files are streamed one at a time and only the requested
cycles are retained, so callers cannot accidentally materialize a held-out
partition by loading the complete data set.

All constants in this module are fixed by the prospective
``hydraulic_padr_pac_v1`` analysis protocol.  Apart from NumPy, only the Python
standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np


EXPECTED_CYCLES = 2205
CYCLE_SECONDS = 60
ROBUST_SCALE_FACTOR = 1.4826
SCALE_FLOOR_FRACTION = 0.1
MINIMUM_SCALE = 1.0e-6
RESIDUAL_CLIP = 8.0
RAW_CHANNEL_COUNT = 44
OUTPUT_CHANNEL_COUNT = 88

TEMPLATE_SCHEMA = "mate_spc_hydraulic_phasewise_template"
TEMPLATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SensorSpec:
    """One official sensor file and its within-cycle sampling rate."""

    name: str
    rate_hz: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("sensor name must be a nonempty string")
        if not isinstance(self.rate_hz, int) or isinstance(self.rate_hz, bool):
            raise TypeError("sensor rate must be an integer")
        if self.rate_hz <= 0:
            raise ValueError("sensor rate must be strictly positive")

    @property
    def filename(self) -> str:
        return f"{self.name}.txt"

    @property
    def columns_per_cycle(self) -> int:
        return CYCLE_SECONDS * self.rate_hz

    @property
    def output_channel_count(self) -> int:
        return 1 if self.rate_hz == 1 else 4

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        if self.rate_hz == 1:
            return (f"{self.name}_observed_value",)
        return tuple(
            f"{self.name}_{statistic}"
            for statistic in ("mean", "population_std", "minimum", "maximum")
        )


# Official sensor order is fixed here and must not be inferred from directory
# enumeration.  In particular, profile.txt and documentation files are never
# measurement dependencies of this loader.
SENSOR_SPECS: tuple[SensorSpec, ...] = (
    *(SensorSpec(f"PS{index}", 100) for index in range(1, 7)),
    SensorSpec("EPS1", 100),
    SensorSpec("FS1", 10),
    SensorSpec("FS2", 10),
    *(SensorSpec(f"TS{index}", 1) for index in range(1, 5)),
    SensorSpec("VS1", 1),
    SensorSpec("CE", 1),
    SensorSpec("CP", 1),
    SensorSpec("SE", 1),
)

RAW_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for spec in SENSOR_SPECS for name in spec.output_feature_names
)
LEVEL_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"phase_residual__{name}" for name in RAW_FEATURE_NAMES
)
DIFFERENCE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"phase_residual_difference__{name}" for name in RAW_FEATURE_NAMES
)
OUTPUT_FEATURE_NAMES = LEVEL_FEATURE_NAMES + DIFFERENCE_FEATURE_NAMES

if len(SENSOR_SPECS) != 17 or len({spec.name for spec in SENSOR_SPECS}) != 17:
    raise RuntimeError("internal Hydraulic sensor specification is invalid")
if len(RAW_FEATURE_NAMES) != RAW_CHANNEL_COUNT:
    raise RuntimeError("internal Hydraulic raw-channel specification is invalid")
if len(OUTPUT_FEATURE_NAMES) != OUTPUT_CHANNEL_COUNT:
    raise RuntimeError("internal Hydraulic output-channel specification is invalid")


def _validate_cycle_indices(
    cycle_indices: Iterable[int], *, expected_cycles: int
) -> tuple[int, ...]:
    if isinstance(cycle_indices, (str, bytes)):
        raise TypeError("cycle_indices must be an iterable of zero-based integers")
    try:
        raw_indices = tuple(cycle_indices)
    except TypeError as error:
        raise TypeError(
            "cycle_indices must be an iterable of zero-based integers"
        ) from error
    if not raw_indices:
        raise ValueError("at least one cycle index must be requested")

    indices: list[int] = []
    for value in raw_indices:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("cycle indices must be zero-based integers")
        index = int(value)
        if index < 0 or index >= expected_cycles:
            raise IndexError(
                f"cycle index {index} is outside the valid range "
                f"[0, {expected_cycles - 1}]"
            )
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("cycle_indices must not contain duplicates")
    return tuple(indices)


def _checked_dataset_directory(dataset_dir: str | os.PathLike[str]) -> Path:
    root = Path(dataset_dir)
    if root.is_symlink():
        raise ValueError("Hydraulic dataset directory must not be a symlink")
    if not root.exists():
        raise FileNotFoundError(f"Hydraulic dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Hydraulic dataset path is not a directory: {root}")
    return root.resolve(strict=True)


def _checked_sensor_path(root: Path, spec: SensorSpec) -> Path:
    path = root / spec.filename
    if path.is_symlink():
        raise ValueError(f"Hydraulic sensor file must not be a symlink: {spec.filename}")
    if not path.exists():
        raise FileNotFoundError(f"missing required Hydraulic sensor file: {spec.filename}")
    if not path.is_file():
        raise ValueError(f"Hydraulic sensor path is not a regular file: {spec.filename}")
    # The fixed filename has no directory components.  This resolved-parent
    # check is defense in depth against unusual reparse-point behavior.
    resolved = path.resolve(strict=True)
    if resolved.parent != root:
        raise ValueError(f"Hydraulic sensor file escapes the dataset directory: {spec.filename}")
    return resolved


def _validate_row_structure(
    text: str,
    *,
    path: Path,
    row_index: int,
    expected_columns: int,
) -> None:
    tokens = text.split()
    if len(tokens) != expected_columns:
        raise ValueError(
            f"{path.name} row {row_index + 1} must contain exactly "
            f"{expected_columns} whitespace-delimited columns; found {len(tokens)}"
        )


def _parse_numeric_row(
    text: str,
    *,
    path: Path,
    row_index: int,
    expected_columns: int,
) -> np.ndarray:
    _validate_row_structure(
        text,
        path=path,
        row_index=row_index,
        expected_columns=expected_columns,
    )
    try:
        # NumPy performs the string-to-float conversion in compiled code.  It
        # is used only after this row's index has passed the access guard.
        values = np.asarray(text.split(), dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            f"{path.name} row {row_index + 1} contains a non-numeric value"
        ) from error
    if values.shape != (expected_columns,):
        raise ValueError(
            f"{path.name} row {row_index + 1} contains a non-numeric value"
        )
    if not np.isfinite(values).all():
        raise ValueError(
            f"{path.name} row {row_index + 1} must contain only finite values"
        )
    return values


def _aggregate_sensor_cycle(values: np.ndarray, spec: SensorSpec) -> np.ndarray:
    samples = values.reshape(CYCLE_SECONDS, spec.rate_hz)
    if spec.rate_hz == 1:
        return samples
    return np.column_stack(
        (
            np.mean(samples, axis=1),
            np.std(samples, axis=1, ddof=0),
            np.min(samples, axis=1),
            np.max(samples, axis=1),
        )
    )


def _read_sensor_cycles(
    path: Path,
    spec: SensorSpec,
    cycle_indices: Sequence[int],
    *,
    expected_cycles: int = EXPECTED_CYCLES,
) -> np.ndarray:
    """Byte-scan one sensor file and parse only explicitly selected cycles."""

    if path.is_symlink():
        raise ValueError(f"Hydraulic sensor file must not be a symlink: {path.name}")
    indices = _validate_cycle_indices(cycle_indices, expected_cycles=expected_cycles)
    output = np.empty(
        (len(indices), CYCLE_SECONDS, spec.output_channel_count), dtype=np.float64
    )
    output_positions = {cycle_index: position for position, cycle_index in enumerate(indices)}
    row_count = 0
    try:
        handle = path.open("rb")
    except OSError:
        raise
    with handle:
        for row_index, raw_line in enumerate(handle):
            row_count = row_index + 1
            if row_count > expected_cycles:
                raise ValueError(
                    f"{path.name} must contain exactly {expected_cycles} rows; found more"
                )
            position = output_positions.get(row_index)
            if position is not None:
                try:
                    text = raw_line.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"{path.name} row {row_index + 1} must be valid UTF-8"
                    ) from error
                values = _parse_numeric_row(
                    text,
                    path=path,
                    row_index=row_index,
                    expected_columns=spec.columns_per_cycle,
                )
                output[position] = _aggregate_sensor_cycle(values, spec)
    if row_count != expected_cycles:
        raise ValueError(
            f"{path.name} must contain exactly {expected_cycles} rows; found {row_count}"
        )
    if not np.isfinite(output).all():
        raise RuntimeError(f"internal aggregation of {path.name} produced non-finite values")
    return output


def _load_cycles_from_specs(
    dataset_dir: str | os.PathLike[str],
    cycle_indices: Iterable[int],
    *,
    specs: Sequence[SensorSpec],
    expected_cycles: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Testable implementation used by the fixed public loader."""

    root = _checked_dataset_directory(dataset_dir)
    indices = _validate_cycle_indices(cycle_indices, expected_cycles=expected_cycles)
    if not specs:
        raise ValueError("at least one sensor specification is required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("sensor specifications must have unique names")

    total_channels = sum(spec.output_channel_count for spec in specs)
    output = np.empty(
        (len(indices), CYCLE_SECONDS, total_channels), dtype=np.float64
    )
    start = 0
    for spec in specs:
        path = _checked_sensor_path(root, spec)
        stop = start + spec.output_channel_count
        # No full-dataset or multi-sensor measurement matrix is constructed:
        # one file is parsed, aggregated into its final output slice, and then
        # released before the next sensor file is opened.
        output[:, :, start:stop] = _read_sensor_cycles(
            path,
            spec,
            indices,
            expected_cycles=expected_cycles,
        )
        start = stop
    if not np.isfinite(output).all():
        raise RuntimeError("internal Hydraulic aggregation produced non-finite values")
    metadata: dict[str, object] = {
        "schema": "mate_spc_hydraulic_loaded_cycles",
        "schema_version": 1,
        "cycle_index_base": 0,
        "expected_dataset_cycles": expected_cycles,
        "parsed_cycle_indices": list(indices),
        "parsed_cycle_count": len(indices),
        "unrequested_rows_byte_scanned_for_boundaries": True,
        "unrequested_rows_decoded": False,
        "unrequested_numeric_values_parsed": False,
        "all_row_boundaries_counted": True,
        "requested_row_column_counts_validated": True,
        "parsed_values_numeric_and_finite": True,
        "sensor_rates_hz": {spec.name: spec.rate_hz for spec in specs},
        "feature_names": [
            name for spec in specs for name in spec.output_feature_names
        ],
        "output_shape": list(output.shape),
        "dtype": "float64",
    }
    return output, metadata


def load_hydraulic_cycles(
    dataset_dir: str | os.PathLike[str], cycle_indices: Iterable[int]
) -> tuple[np.ndarray, dict[str, object]]:
    """Load selected cycles and access metadata.

    The first return value is a ``(cycles, 60, 44)`` float64 array.  The second
    is JSON-safe metadata that explicitly records the numerically parsed cycle
    indices.

    ``cycle_indices`` are zero based, must be explicit and unique, and are
    returned in exactly the requested order.  Every official sensor file is
    byte-scanned to count exactly 2,205 row boundaries.  UTF-8 decoding,
    fixed-column validation, numeric conversion, and finite-value validation
    are intentionally restricted to requested cycles.  Unrequested rows
    (including held-out confirmation) are never decoded or inspected based on
    their content.  Files are streamed sequentially to bound memory use.

    This function never reads ``profile.txt``; labels and split membership must
    come from the separately audited and frozen protocol artifacts.
    """

    output, metadata = _load_cycles_from_specs(
        dataset_dir,
        cycle_indices,
        specs=SENSOR_SPECS,
        expected_cycles=EXPECTED_CYCLES,
    )
    if output.shape[2] != RAW_CHANNEL_COUNT:
        raise RuntimeError("internal Hydraulic channel-count mismatch")
    if metadata["feature_names"] != list(RAW_FEATURE_NAMES):
        raise RuntimeError("internal Hydraulic feature-name mismatch")
    return output, metadata


def _as_raw_cycles(values: object, *, require_nonempty: bool) -> np.ndarray:
    cycles = np.asarray(values, dtype=np.float64)
    if cycles.ndim != 3 or cycles.shape[1:] != (CYCLE_SECONDS, RAW_CHANNEL_COUNT):
        raise ValueError(
            f"Hydraulic raw cycles must have shape (cycles, {CYCLE_SECONDS}, "
            f"{RAW_CHANNEL_COUNT})"
        )
    if require_nonempty and cycles.shape[0] == 0:
        raise ValueError("at least one healthy development cycle is required")
    if not np.isfinite(cycles).all():
        raise ValueError("Hydraulic raw cycles must contain only finite values")
    return cycles


def _readonly_float64(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HydraulicPhasewiseTemplate:
    """Healthy-development-only phase median/MAD reference."""

    phase_center: np.ndarray
    phase_scale: np.ndarray
    global_center: np.ndarray
    global_robust_scale: np.ndarray
    scale_floor: np.ndarray

    def __post_init__(self) -> None:
        fields = {
            "phase_center": (self.phase_center, (CYCLE_SECONDS, RAW_CHANNEL_COUNT)),
            "phase_scale": (self.phase_scale, (CYCLE_SECONDS, RAW_CHANNEL_COUNT)),
            "global_center": (self.global_center, (RAW_CHANNEL_COUNT,)),
            "global_robust_scale": (
                self.global_robust_scale,
                (RAW_CHANNEL_COUNT,),
            ),
            "scale_floor": (self.scale_floor, (RAW_CHANNEL_COUNT,)),
        }
        for name, (value, shape) in fields.items():
            array = _readonly_float64(value, name=name)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, array)
        if np.any(self.phase_scale <= 0.0):
            raise ValueError("phase_scale must be strictly positive")
        if np.any(self.global_robust_scale < 0.0):
            raise ValueError("global_robust_scale must be nonnegative")
        if np.any(self.scale_floor < MINIMUM_SCALE):
            raise ValueError(f"scale_floor must be at least {MINIMUM_SCALE}")
        if np.any(self.phase_scale < self.scale_floor[None, :]):
            raise ValueError("phase_scale must not be below the channel scale floor")

    @classmethod
    def fit(cls, healthy_development_cycles: object) -> "HydraulicPhasewiseTemplate":
        """Fit the frozen reference using healthy development cycles only."""

        cycles = _as_raw_cycles(healthy_development_cycles, require_nonempty=True)
        phase_center = np.median(cycles, axis=0)
        phase_mad = np.median(np.abs(cycles - phase_center[None, :, :]), axis=0)
        phase_robust_scale = ROBUST_SCALE_FACTOR * phase_mad

        flattened = cycles.reshape(-1, RAW_CHANNEL_COUNT)
        global_center = np.median(flattened, axis=0)
        global_mad = np.median(np.abs(flattened - global_center[None, :]), axis=0)
        global_robust_scale = ROBUST_SCALE_FACTOR * global_mad
        scale_floor = np.maximum(
            MINIMUM_SCALE, SCALE_FLOOR_FRACTION * global_robust_scale
        )
        phase_scale = np.maximum(phase_robust_scale, scale_floor[None, :])
        return cls(
            phase_center=phase_center,
            phase_scale=phase_scale,
            global_center=global_center,
            global_robust_scale=global_robust_scale,
            scale_floor=scale_floor,
        )

    def transform(self, raw_cycles: object) -> np.ndarray:
        """Return clipped level residuals and their causal first differences."""

        cycles = _as_raw_cycles(raw_cycles, require_nonempty=False)
        residual = np.clip(
            (cycles - self.phase_center[None, :, :])
            / self.phase_scale[None, :, :],
            -RESIDUAL_CLIP,
            RESIDUAL_CLIP,
        )
        difference = np.empty_like(residual)
        difference[:, 0, :] = 0.0
        difference[:, 1:, :] = residual[:, 1:, :] - residual[:, :-1, :]
        output = np.concatenate((residual, difference), axis=2)
        if output.shape != (cycles.shape[0], CYCLE_SECONDS, OUTPUT_CHANNEL_COUNT):
            raise RuntimeError("internal Hydraulic feature shape mismatch")
        if not np.isfinite(output).all():
            raise RuntimeError("Hydraulic phasewise transformation produced non-finite values")
        return output

    def save_npz(self, path: str | os.PathLike[str]) -> Path:
        return save_phasewise_template(self, path)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "HydraulicPhasewiseTemplate":
        return load_phasewise_template(path)


_TEMPLATE_ARCHIVE_KEYS = {
    "schema",
    "schema_version",
    "raw_feature_names",
    "phase_center",
    "phase_scale",
    "global_center",
    "global_robust_scale",
    "scale_floor",
}


def save_phasewise_template(
    template: HydraulicPhasewiseTemplate, path: str | os.PathLike[str]
) -> Path:
    """Atomically save a template as a pickle-free, exact-schema NPZ."""

    if not isinstance(template, HydraulicPhasewiseTemplate):
        raise TypeError("template must be a HydraulicPhasewiseTemplate")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("Hydraulic template path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
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
                schema=np.asarray(TEMPLATE_SCHEMA),
                schema_version=np.asarray([TEMPLATE_SCHEMA_VERSION], dtype=np.int64),
                raw_feature_names=np.asarray(RAW_FEATURE_NAMES),
                phase_center=template.phase_center,
                phase_scale=template.phase_scale,
                global_center=template.global_center,
                global_robust_scale=template.global_robust_scale,
                scale_floor=template.scale_floor,
            )
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return target


def _archive_scalar_text(array: np.ndarray, *, name: str) -> str:
    value = np.asarray(array)
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"archived {name} must be a scalar string")
    return str(value.item())


def _archive_numeric_field(
    archive: np.lib.npyio.NpzFile,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Validate an archive array before permitting float64 conversion."""

    try:
        raw = np.asarray(archive[name])
    except (TypeError, ValueError) as error:
        # With allow_pickle=False, NumPy raises while materializing an object
        # array.  Translate that into a field-specific validation failure.
        raise ValueError(
            f"archived {name} must have a real numeric, non-boolean dtype"
        ) from error
    if raw.dtype.kind not in "fiu":
        raise ValueError(
            f"archived {name} must have a real numeric, non-boolean dtype"
        )
    if raw.shape != expected_shape:
        raise ValueError(f"archived {name} must have shape {expected_shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"archived {name} must contain only finite values")
    return np.array(raw, dtype=np.float64, copy=True)


def load_phasewise_template(
    path: str | os.PathLike[str],
) -> HydraulicPhasewiseTemplate:
    """Load and strictly validate a pickle-free Hydraulic template."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("Hydraulic template file must not be a symlink")
    if not source.is_file():
        raise FileNotFoundError(f"Hydraulic template file does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != _TEMPLATE_ARCHIVE_KEYS:
            missing = sorted(_TEMPLATE_ARCHIVE_KEYS - keys)
            extra = sorted(keys - _TEMPLATE_ARCHIVE_KEYS)
            raise ValueError(
                f"Hydraulic template archive keys mismatch; missing={missing}, extra={extra}"
            )
        schema = _archive_scalar_text(archive["schema"], name="schema")
        version = np.asarray(archive["schema_version"])
        names = np.asarray(archive["raw_feature_names"])
        expected_shapes = {
            "phase_center": (CYCLE_SECONDS, RAW_CHANNEL_COUNT),
            "phase_scale": (CYCLE_SECONDS, RAW_CHANNEL_COUNT),
            "global_center": (RAW_CHANNEL_COUNT,),
            "global_robust_scale": (RAW_CHANNEL_COUNT,),
            "scale_floor": (RAW_CHANNEL_COUNT,),
        }
        values = {
            name: _archive_numeric_field(
                archive, name=name, expected_shape=expected_shape
            )
            for name, expected_shape in expected_shapes.items()
        }
    if schema != TEMPLATE_SCHEMA:
        raise ValueError("Hydraulic template archive has an unknown schema")
    if version.shape != (1,) or version.dtype.kind not in "iu":
        raise ValueError("archived schema_version must be one integer")
    if int(version[0]) != TEMPLATE_SCHEMA_VERSION:
        raise ValueError("unsupported Hydraulic template schema version")
    if names.ndim != 1 or names.dtype.kind not in "US":
        raise ValueError("archived raw_feature_names must be a string vector")
    if tuple(str(value) for value in names.tolist()) != RAW_FEATURE_NAMES:
        raise ValueError("archived Hydraulic raw feature names do not match the protocol")
    return HydraulicPhasewiseTemplate(**values)


def fit_phasewise_template(healthy_development_cycles: object) -> HydraulicPhasewiseTemplate:
    """Functional alias for :meth:`HydraulicPhasewiseTemplate.fit`."""

    return HydraulicPhasewiseTemplate.fit(healthy_development_cycles)


__all__ = [
    "CYCLE_SECONDS",
    "DIFFERENCE_FEATURE_NAMES",
    "EXPECTED_CYCLES",
    "HydraulicPhasewiseTemplate",
    "LEVEL_FEATURE_NAMES",
    "MINIMUM_SCALE",
    "OUTPUT_CHANNEL_COUNT",
    "OUTPUT_FEATURE_NAMES",
    "RAW_CHANNEL_COUNT",
    "RAW_FEATURE_NAMES",
    "RESIDUAL_CLIP",
    "ROBUST_SCALE_FACTOR",
    "SCALE_FLOOR_FRACTION",
    "SENSOR_SPECS",
    "SensorSpec",
    "TEMPLATE_SCHEMA",
    "TEMPLATE_SCHEMA_VERSION",
    "fit_phasewise_template",
    "load_hydraulic_cycles",
    "load_phasewise_template",
    "save_phasewise_template",
]
