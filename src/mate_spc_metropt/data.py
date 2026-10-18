"""Phase-isolated data access for the MetroPT CARED-BC experiment.

The public entry points deliberately separate a timestamp-only audit from
sensor access.  :func:`audit_metropt_timestamps` is allowed to scan the whole
CSV, but asks pandas for the timestamp column only.  Formal experiment runners
must use :func:`load_governed_metropt_partition`, which authenticates a durable
phase intent before delegating to the low-level interval loader.

Direct alarm channels are required in the source schema so that a different
CSV cannot silently be substituted, but they are never requested from the CSV
parser and never exposed by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMESTAMP_COLUMN = "timestamp"

ANALOG_COLUMNS: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)

SAFE_CONTROL_COLUMNS: tuple[str, ...] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "Pressure_switch",
    "Caudal_impulses",
)

STATE_CONTROL_COLUMNS: tuple[str, ...] = SAFE_CONTROL_COLUMNS[:-1]
IMPULSE_COLUMN = SAFE_CONTROL_COLUMNS[-1]
DIRECT_ALARM_COLUMNS: tuple[str, ...] = ("LPS", "Oil_level")
APPROVED_NUMERIC_COLUMNS: tuple[str, ...] = ANALOG_COLUMNS + SAFE_CONTROL_COLUMNS

# The UCI export commonly contains this inert row-number field.  No other
# unrecognised column is accepted, because it could be an undeclared signal.
OPTIONAL_INDEX_COLUMNS: tuple[str, ...] = ("Unnamed: 0", "index")
RESAMPLE_FREQUENCY = "1min"
CONTROL_FFILL_LIMIT_MINUTES = 5
NOMINAL_RAW_SAMPLES_PER_MINUTE = 6
MINIMUM_COVERAGE_FRACTION = 0.5
SCHEMA_VERSION = 1


class MetroPTDataError(ValueError):
    """Raised when a source or interval violates the frozen access contract."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validated_source_path(csv_path: str | Path) -> Path:
    supplied = Path(csv_path)
    if supplied.is_symlink():
        raise MetroPTDataError("MetroPT source must not be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MetroPTDataError(f"MetroPT source does not exist: {supplied}") from exc
    if not resolved.is_file():
        raise MetroPTDataError(f"MetroPT source is not a regular file: {resolved}")
    return resolved


def _read_and_validate_header(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except StopIteration as exc:
        raise MetroPTDataError("MetroPT CSV is empty") from exc
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        raise MetroPTDataError("MetroPT CSV header cannot be read safely") from exc

    if not header:
        raise MetroPTDataError("MetroPT CSV contains no field names")
    # The published MetroPT CSV writes its row-index header as the first empty
    # field; pandas deterministically exposes that field as Unnamed: 0.
    # No signal field (or any other position) may be unnamed.
    if header[0] == "":
        header[0] = "Unnamed: 0"
    if any(not name or "\x00" in name for name in header):
        raise MetroPTDataError("MetroPT CSV contains an empty or invalid field name")
    if any(name != name.strip() for name in header):
        raise MetroPTDataError("MetroPT CSV field names must not contain outer whitespace")
    if len(set(header)) != len(header):
        raise MetroPTDataError("MetroPT CSV contains duplicate field names")

    required = {TIMESTAMP_COLUMN, *APPROVED_NUMERIC_COLUMNS, *DIRECT_ALARM_COLUMNS}
    missing = sorted(required.difference(header))
    if missing:
        raise MetroPTDataError(f"MetroPT CSV is missing required fields: {missing}")

    allowed = required.union(OPTIONAL_INDEX_COLUMNS)
    unexpected = sorted(set(header).difference(allowed))
    if unexpected:
        raise MetroPTDataError(f"MetroPT CSV contains undeclared fields: {unexpected}")
    return tuple(header)


def _parse_timestamp_values(values: Sequence[Any], *, context: str) -> pd.DatetimeIndex:
    text = pd.Series(values, dtype="string")
    if text.isna().any() or text.str.strip().eq("").any():
        raise MetroPTDataError(f"{context} contains a missing timestamp")
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(text, errors="raise"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetroPTDataError(f"{context} contains an invalid timestamp") from exc
    if parsed.hasnans:
        raise MetroPTDataError(f"{context} contains NaT")
    if parsed.tz is not None:
        raise MetroPTDataError(f"{context} timestamps must be timezone-naive")
    return parsed


def _coerce_boundary(value: str | np.datetime64 | pd.Timestamp, *, name: str) -> pd.Timestamp:
    try:
        boundary = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MetroPTDataError(f"{name} is not a valid timestamp") from exc
    if pd.isna(boundary):
        raise MetroPTDataError(f"{name} must not be NaT")
    if boundary.tz is not None:
        raise MetroPTDataError(f"{name} must be timezone-naive")
    if boundary.second or boundary.microsecond or boundary.nanosecond:
        raise MetroPTDataError(f"{name} must be aligned to a complete minute")
    return boundary


def _validate_chunksize(chunksize: int) -> int:
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize <= 0:
        raise MetroPTDataError("chunksize must be a positive integer")
    return chunksize


@dataclass(frozen=True)
class MetroPTTimestampAudit:
    """Attestation produced without parsing any MetroPT sensor value."""

    source_path: str
    source_size_bytes: int
    source_mtime_ns: int
    source_sha256: str
    header: tuple[str, ...]
    row_count: int
    first_timestamp: str
    last_timestamp: str
    minimum_interval_seconds: float | None
    modal_interval_seconds: float | None
    modal_interval_count: int
    maximum_interval_seconds: float | None
    chunks_scanned: int
    partition_row_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = dict(self.partition_row_counts)
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in counts.items()
        ):
            raise MetroPTDataError("partition row counts must be non-negative integers")
        object.__setattr__(self, "partition_row_counts", counts)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "metropt_cared_bc_timestamp_audit",
            "schema_version": SCHEMA_VERSION,
            "source_path": self.source_path,
            "source_size_bytes": self.source_size_bytes,
            "source_mtime_ns": self.source_mtime_ns,
            "source_sha256": self.source_sha256,
            "header": list(self.header),
            "timestamp_column": TIMESTAMP_COLUMN,
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "strictly_increasing": True,
            "duplicate_timestamp_count": 0,
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "modal_interval_seconds": self.modal_interval_seconds,
            "modal_interval_count": self.modal_interval_count,
            "maximum_interval_seconds": self.maximum_interval_seconds,
            "chunks_scanned": self.chunks_scanned,
            "partition_row_counts": dict(self.partition_row_counts),
            "timestamp_values_parsed": True,
            "sensor_numeric_values_parsed": False,
            "numeric_columns_parsed": [],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._body())

    def as_dict(self) -> dict[str, Any]:
        result = self._body()
        result["audit_fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True)
class MetroPTMinuteInterval:
    """One-minute CARED-BC input with explicit observation eligibility."""

    timestamps: np.ndarray
    analog: np.ndarray
    controls: np.ndarray
    analog_observed_mask: np.ndarray
    control_observed_mask: np.ndarray
    control_imputed_mask: np.ndarray
    bin_observed_mask: np.ndarray
    coverage_fraction: np.ndarray
    eligible: np.ndarray
    reset_before: np.ndarray
    source_row_count_per_bin: np.ndarray
    access_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        analog = np.asarray(self.analog, dtype=np.float64)
        controls = np.asarray(self.controls, dtype=np.float64)
        analog_mask = np.asarray(self.analog_observed_mask, dtype=bool)
        control_mask = np.asarray(self.control_observed_mask, dtype=bool)
        imputed_mask = np.asarray(self.control_imputed_mask, dtype=bool)
        bin_mask = np.asarray(self.bin_observed_mask, dtype=bool)
        coverage = np.asarray(self.coverage_fraction, dtype=np.float64)
        eligible = np.asarray(self.eligible, dtype=bool)
        reset_before = np.asarray(self.reset_before, dtype=bool)
        row_count = np.asarray(self.source_row_count_per_bin, dtype=np.int64)

        if timestamps.ndim != 1 or timestamps.size == 0:
            raise MetroPTDataError("minute timestamps must be a non-empty vector")
        length = timestamps.size
        if analog.shape != (length, len(ANALOG_COLUMNS)):
            raise MetroPTDataError("analog matrix has an invalid shape")
        if controls.shape != (length, len(SAFE_CONTROL_COLUMNS)):
            raise MetroPTDataError("control matrix has an invalid shape")
        if analog_mask.shape != analog.shape:
            raise MetroPTDataError("analog observation mask has an invalid shape")
        if control_mask.shape != controls.shape or imputed_mask.shape != controls.shape:
            raise MetroPTDataError("control mask has an invalid shape")
        for name, value in (
            ("bin_observed_mask", bin_mask),
            ("coverage_fraction", coverage),
            ("eligible", eligible),
            ("reset_before", reset_before),
            ("source_row_count_per_bin", row_count),
        ):
            if value.shape != (length,):
                raise MetroPTDataError(f"{name} has an invalid shape")
        if length > 1:
            delta_ns = np.diff(timestamps.astype(np.int64))
            if not np.all(delta_ns == 60_000_000_000):
                raise MetroPTDataError("minute timestamps do not form a complete one-minute grid")
        if not np.isfinite(analog).all():
            raise MetroPTDataError("analog output must be finite; missing values use the mask")
        if np.isinf(controls).any():
            raise MetroPTDataError("control output must not contain infinity")
        if (row_count < 0).any() or not np.array_equal(bin_mask, row_count > 0):
            raise MetroPTDataError("bin observation mask is inconsistent with row counts")
        expected_coverage = np.minimum(
            row_count.astype(np.float64) / float(NOMINAL_RAW_SAMPLES_PER_MINUTE), 1.0
        )
        if not np.array_equal(coverage, expected_coverage):
            raise MetroPTDataError("coverage fraction is inconsistent with row counts")
        if np.any(imputed_mask & control_mask):
            raise MetroPTDataError("a control cannot be both observed and imputed")
        expected_eligible = (
            bin_mask
            & (coverage >= MINIMUM_COVERAGE_FRACTION)
            & analog_mask.all(axis=1)
            & np.isfinite(controls).all(axis=1)
        )
        if not np.array_equal(eligible, expected_eligible):
            raise MetroPTDataError("eligibility is inconsistent with values and masks")

        for value in (
            timestamps,
            analog,
            controls,
            analog_mask,
            control_mask,
            imputed_mask,
            bin_mask,
            coverage,
            eligible,
            reset_before,
            row_count,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "analog", analog)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "analog_observed_mask", analog_mask)
        object.__setattr__(self, "control_observed_mask", control_mask)
        object.__setattr__(self, "control_imputed_mask", imputed_mask)
        object.__setattr__(self, "bin_observed_mask", bin_mask)
        object.__setattr__(self, "coverage_fraction", coverage)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "reset_before", reset_before)
        object.__setattr__(self, "source_row_count_per_bin", row_count)
        object.__setattr__(self, "access_metadata", dict(self.access_metadata))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return APPROVED_NUMERIC_COLUMNS

    @property
    def features(self) -> np.ndarray:
        """Return the frozen 13-column model order without copying."""

        features = np.concatenate((self.analog, self.controls), axis=1)
        features.setflags(write=False)
        return features


def audit_metropt_timestamps(
    csv_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> MetroPTTimestampAudit:
    """Audit schema, identity, order and cadence using timestamps only.

    The source file is hashed byte-for-byte, which is permitted by the staged
    protocol, but no sensor or alarm field is requested or converted.
    """

    chunk_rows = _validate_chunksize(chunksize)
    path = _validated_source_path(csv_path)
    header = _read_and_validate_header(path)
    before = path.stat()
    source_sha256 = _sha256_file(path)

    # Governance is imported lazily to keep the timestamp/data primitives
    # usable without a module-import cycle.  Counting frozen partitions asks
    # pandas only for timestamps and therefore does not expand numeric access.
    from .governance import PARTITIONS

    partition_bounds_ns = {
        partition.name: (
            int(pd.Timestamp(partition.start).value),
            int(pd.Timestamp(partition.end).value),
        )
        for partition in PARTITIONS
    }
    partition_row_counts = {partition.name: 0 for partition in PARTITIONS}

    row_count = 0
    chunks_scanned = 0
    first_ns: int | None = None
    last_ns: int | None = None
    minimum_delta_ns: int | None = None
    maximum_delta_ns: int | None = None
    delta_counts: dict[int, int] = {}

    try:
        reader = pd.read_csv(
            path,
            usecols=[TIMESTAMP_COLUMN],
            dtype={TIMESTAMP_COLUMN: "string"},
            keep_default_na=False,
            chunksize=chunk_rows,
        )
        for chunk_index, chunk in enumerate(reader):
            chunks_scanned += 1
            parsed = _parse_timestamp_values(
                chunk[TIMESTAMP_COLUMN], context=f"timestamp chunk {chunk_index}"
            )
            values_ns = parsed.asi8
            if values_ns.size == 0:
                continue
            for name, (start_ns, end_ns) in partition_bounds_ns.items():
                partition_row_counts[name] += int(
                    np.count_nonzero((values_ns >= start_ns) & (values_ns < end_ns))
                )
            if last_ns is not None and int(values_ns[0]) <= last_ns:
                raise MetroPTDataError("timestamps are duplicated or not strictly increasing")
            deltas = np.diff(values_ns)
            if (deltas <= 0).any():
                raise MetroPTDataError("timestamps are duplicated or not strictly increasing")
            if last_ns is not None:
                deltas = np.concatenate(
                    (np.asarray([int(values_ns[0]) - last_ns], dtype=np.int64), deltas)
                )
            for delta in deltas:
                delta_value = int(delta)
                delta_counts[delta_value] = delta_counts.get(delta_value, 0) + 1
                minimum_delta_ns = (
                    delta_value
                    if minimum_delta_ns is None
                    else min(minimum_delta_ns, delta_value)
                )
                maximum_delta_ns = (
                    delta_value
                    if maximum_delta_ns is None
                    else max(maximum_delta_ns, delta_value)
                )
            if first_ns is None:
                first_ns = int(values_ns[0])
            last_ns = int(values_ns[-1])
            row_count += int(values_ns.size)
    except MetroPTDataError:
        raise
    except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise MetroPTDataError("timestamp-only MetroPT scan failed") from exc

    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MetroPTDataError("MetroPT source changed during timestamp audit")
    if row_count == 0 or first_ns is None or last_ns is None:
        raise MetroPTDataError("MetroPT CSV contains no data rows")

    modal_delta_ns: int | None = None
    modal_count = 0
    if delta_counts:
        modal_delta_ns, modal_count = min(
            delta_counts.items(), key=lambda item: (-item[1], item[0])
        )

    return MetroPTTimestampAudit(
        source_path=str(path),
        source_size_bytes=int(before.st_size),
        source_mtime_ns=int(before.st_mtime_ns),
        source_sha256=source_sha256,
        header=header,
        row_count=row_count,
        first_timestamp=pd.Timestamp(first_ns).isoformat(),
        last_timestamp=pd.Timestamp(last_ns).isoformat(),
        minimum_interval_seconds=(
            None if minimum_delta_ns is None else minimum_delta_ns / 1_000_000_000.0
        ),
        modal_interval_seconds=(
            None if modal_delta_ns is None else modal_delta_ns / 1_000_000_000.0
        ),
        modal_interval_count=modal_count,
        maximum_interval_seconds=(
            None if maximum_delta_ns is None else maximum_delta_ns / 1_000_000_000.0
        ),
        chunks_scanned=chunks_scanned,
        partition_row_counts=partition_row_counts,
    )


def _authenticate_audit(path: Path, audit: MetroPTTimestampAudit) -> None:
    if not isinstance(audit, MetroPTTimestampAudit):
        raise MetroPTDataError("audit must be a MetroPTTimestampAudit")
    if audit.source_path != str(path):
        raise MetroPTDataError("timestamp audit belongs to a different source")
    header = _read_and_validate_header(path)
    if header != audit.header:
        raise MetroPTDataError("MetroPT header changed after timestamp audit")
    stat = path.stat()
    if int(stat.st_size) != audit.source_size_bytes or int(stat.st_mtime_ns) != audit.source_mtime_ns:
        raise MetroPTDataError("MetroPT source metadata changed after timestamp audit")
    if _sha256_file(path) != audit.source_sha256:
        raise MetroPTDataError("MetroPT source hash changed after timestamp audit")


def _resample_selected_rows(
    timestamps: np.ndarray,
    numeric_values: np.ndarray,
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    access_metadata: Mapping[str, Any],
) -> MetroPTMinuteInterval:
    frame = pd.DataFrame(numeric_values, columns=APPROVED_NUMERIC_COLUMNS)
    frame.index = pd.DatetimeIndex(timestamps)

    state_values = frame.loc[:, STATE_CONTROL_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isin(state_values, (0.0, 1.0)).all():
        raise MetroPTDataError("state controls must be exactly binary (0 or 1)")
    impulse = frame.loc[:, IMPULSE_COLUMN].to_numpy(dtype=np.float64)
    if (impulse < 0.0).any():
        raise MetroPTDataError("Caudal_impulses must be non-negative")

    grid = pd.date_range(start=start, end=stop, inclusive="left", freq=RESAMPLE_FREQUENCY)
    if grid.empty:
        raise MetroPTDataError("requested interval has no complete minute")

    resample_options = {
        "rule": RESAMPLE_FREQUENCY,
        "closed": "left",
        "label": "left",
        "origin": start,
    }
    analog_frame = (
        frame.loc[:, ANALOG_COLUMNS]
        .resample(**resample_options)
        .median()
        .reindex(grid)
    )
    state_frame = (
        frame.loc[:, STATE_CONTROL_COLUMNS]
        .resample(**resample_options)
        .last()
        .reindex(grid)
    )
    impulse_frame = (
        frame.loc[:, [IMPULSE_COLUMN]]
        .resample(**resample_options)
        .sum(min_count=1)
        .reindex(grid)
    )
    row_count_frame = (
        pd.Series(np.ones(frame.shape[0], dtype=np.int64), index=frame.index)
        .resample(**resample_options)
        .sum()
        .reindex(grid, fill_value=0)
        .fillna(0)
        .astype(np.int64)
    )

    # Preserve acquisition discontinuities before resampling.  A gap can be
    # longer than 60 seconds even when its endpoints fall in adjacent minute
    # bins (for example 00:00:01 -> 00:01:59); comparing the completed minute
    # grid alone would lose that reset.  ``reset_before[i]`` means that no
    # stateful calculation or model window may carry state from a row before
    # minute i into minute i.  The first minute is an interval boundary too.
    reset_before = np.zeros(grid.size, dtype=bool)
    reset_before[0] = True
    raw_ns = np.asarray(timestamps, dtype="datetime64[ns]").astype(np.int64)
    raw_gap_right = np.flatnonzero(np.diff(raw_ns) > 60_000_000_000) + 1
    if raw_gap_right.size:
        right_minutes = pd.DatetimeIndex(timestamps[raw_gap_right]).floor(RESAMPLE_FREQUENCY)
        positions = grid.get_indexer(right_minutes)
        if np.any(positions < 0):
            raise MetroPTDataError("raw gap endpoint does not map to the requested minute grid")
        reset_before[positions] = True

    analog_observed = analog_frame.notna().to_numpy(dtype=bool)
    state_observed = state_frame.notna()
    impulse_observed = impulse_frame.notna()

    # Only persistent state variables are carried forward.  An impulse count is
    # an interval quantity; carrying it into a missing minute would invent
    # events.  Pandas ffill is causal and ``limit`` counts consecutive bins.
    reset_groups = pd.Series(np.cumsum(reset_before), index=grid)
    state_filled = state_frame.groupby(reset_groups, sort=False).ffill(
        limit=CONTROL_FFILL_LIMIT_MINUTES
    )
    state_imputed = state_filled.notna() & ~state_observed
    controls_frame = pd.concat((state_filled, impulse_frame), axis=1).loc[
        :, SAFE_CONTROL_COLUMNS
    ]
    control_observed_frame = pd.concat((state_observed, impulse_observed), axis=1).loc[
        :, SAFE_CONTROL_COLUMNS
    ]
    control_imputed_frame = pd.concat(
        (
            state_imputed,
            pd.DataFrame(False, index=grid, columns=[IMPULSE_COLUMN]),
        ),
        axis=1,
    ).loc[:, SAFE_CONTROL_COLUMNS]

    analog = analog_frame.fillna(0.0).to_numpy(dtype=np.float64)
    controls = controls_frame.to_numpy(dtype=np.float64)
    control_observed = control_observed_frame.to_numpy(dtype=bool)
    control_imputed = control_imputed_frame.to_numpy(dtype=bool)
    row_count = row_count_frame.to_numpy(dtype=np.int64)
    bin_observed = row_count > 0
    coverage = np.minimum(
        row_count.astype(np.float64) / float(NOMINAL_RAW_SAMPLES_PER_MINUTE), 1.0
    )
    eligible = (
        bin_observed
        & (coverage >= MINIMUM_COVERAGE_FRACTION)
        & analog_observed.all(axis=1)
        & np.isfinite(controls).all(axis=1)
    )

    return MetroPTMinuteInterval(
        timestamps=grid.to_numpy(dtype="datetime64[ns]"),
        analog=analog,
        controls=controls,
        analog_observed_mask=analog_observed,
        control_observed_mask=control_observed,
        control_imputed_mask=control_imputed,
        bin_observed_mask=bin_observed,
        coverage_fraction=coverage,
        eligible=eligible,
        reset_before=reset_before,
        source_row_count_per_bin=row_count,
        access_metadata=access_metadata,
    )


def load_metropt_interval(
    csv_path: str | Path,
    start: str | np.datetime64 | pd.Timestamp,
    stop: str | np.datetime64 | pd.Timestamp,
    *,
    chunksize: int = 100_000,
    timestamp_audit: MetroPTTimestampAudit | None = None,
) -> MetroPTMinuteInterval:
    """Low-level load and resample of one right-open MetroPT interval.

    Sensor strings in rows outside ``[start, stop)`` are never passed to a
    numeric parser.  The complete source is still timestamp-audited, so an
    out-of-order or duplicate timestamp anywhere fails closed.  Formal staged
    experiments must call :func:`load_governed_metropt_partition` instead.
    """

    chunk_rows = _validate_chunksize(chunksize)
    interval_start = _coerce_boundary(start, name="start")
    interval_stop = _coerce_boundary(stop, name="stop")
    if interval_start >= interval_stop:
        raise MetroPTDataError("start must be earlier than stop")

    path = _validated_source_path(csv_path)
    audit = timestamp_audit or audit_metropt_timestamps(path, chunksize=chunk_rows)
    _authenticate_audit(path, audit)

    selected_timestamp_parts: list[np.ndarray] = []
    selected_numeric_parts: list[np.ndarray] = []
    scanned_rows = 0
    chunks_scanned = 0

    try:
        reader = pd.read_csv(
            path,
            usecols=[TIMESTAMP_COLUMN, *APPROVED_NUMERIC_COLUMNS],
            dtype="string",
            keep_default_na=False,
            chunksize=chunk_rows,
        )
        for chunk_index, chunk in enumerate(reader):
            chunks_scanned += 1
            parsed_timestamps = _parse_timestamp_values(
                chunk[TIMESTAMP_COLUMN], context=f"data chunk {chunk_index}"
            )
            scanned_rows += int(chunk.shape[0])
            selected = (parsed_timestamps >= interval_start) & (
                parsed_timestamps < interval_stop
            )
            if not selected.any():
                continue

            # This conversion occurs only after the time selection.  Values in
            # development-external rows remain pandas strings for their entire
            # lifetime and are discarded with the chunk.
            selected_strings = chunk.loc[selected, APPROVED_NUMERIC_COLUMNS]
            normalized = selected_strings.apply(
                lambda column: column.str.strip().str.lower()
            )
            nonfinite_tokens = {
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
            }
            if normalized.isin(nonfinite_tokens).to_numpy(dtype=bool).any():
                raise MetroPTDataError(
                    "requested interval contains a non-finite approved field"
                )
            try:
                selected_numeric = selected_strings.apply(
                    pd.to_numeric, errors="raise"
                ).to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MetroPTDataError(
                    "requested interval contains a non-numeric approved field"
                ) from exc
            if not np.isfinite(selected_numeric).all():
                raise MetroPTDataError(
                    "requested interval contains a non-finite approved field"
                )
            selected_timestamp_parts.append(
                parsed_timestamps[selected].to_numpy(dtype="datetime64[ns]")
            )
            selected_numeric_parts.append(selected_numeric)
    except MetroPTDataError:
        raise
    except (KeyError, OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
        raise MetroPTDataError("chunked MetroPT interval scan failed") from exc

    if scanned_rows != audit.row_count:
        raise MetroPTDataError("data scan row count does not match timestamp audit")
    _authenticate_audit(path, audit)
    if not selected_timestamp_parts:
        raise MetroPTDataError("requested interval contains no source rows")

    timestamps = np.concatenate(selected_timestamp_parts)
    numeric_values = np.concatenate(selected_numeric_parts, axis=0)
    timestamp_ns = timestamps.astype("datetime64[ns]").astype(np.int64)
    if timestamp_ns.size > 1 and (np.diff(timestamp_ns) <= 0).any():
        raise MetroPTDataError("selected timestamps are duplicated or not strictly increasing")

    access_body: dict[str, Any] = {
        "schema": "metropt_cared_bc_interval_access",
        "schema_version": SCHEMA_VERSION,
        "source_path": str(path),
        "source_size_bytes": audit.source_size_bytes,
        "source_sha256": audit.source_sha256,
        "timestamp_audit_fingerprint": audit.fingerprint,
        "requested_interval": {
            "start_inclusive": interval_start.isoformat(),
            "stop_exclusive": interval_stop.isoformat(),
            "notation": "[start, stop)",
        },
        "approved_numeric_columns": list(APPROVED_NUMERIC_COLUMNS),
        "excluded_direct_alarm_columns": list(DIRECT_ALARM_COLUMNS),
        "numeric_columns_parsed": list(APPROVED_NUMERIC_COLUMNS),
        "requested_source_row_count": int(timestamps.size),
        "full_source_row_count": audit.row_count,
        "chunks_scanned": chunks_scanned,
        "full_file_timestamp_values_scanned": True,
        "unrequested_rows_byte_scanned": True,
        "unrequested_numeric_values_parsed": False,
        "direct_alarm_values_parsed": False,
        "resampling": {
            "frequency": RESAMPLE_FREQUENCY,
            "analog": "median",
            "state_controls": "last_then_causal_ffill",
            "state_control_ffill_limit_minutes": CONTROL_FFILL_LIMIT_MINUTES,
            "Caudal_impulses": "sum_min_count_1_no_fill",
            "empty_bin_observation_mask": False,
            "coverage": {
                "nominal_raw_samples_per_minute": NOMINAL_RAW_SAMPLES_PER_MINUTE,
                "fraction": "min(source_row_count_per_bin/6,1)",
                "minimum_fraction_inclusive": MINIMUM_COVERAGE_FRACTION,
            },
            "reset_before": {
                "raw_gap_threshold_seconds": 60,
                "mapping": "minute_containing_first_raw_row_after_gap",
                "first_requested_minute_is_reset": True,
            },
        },
    }
    access_metadata = dict(access_body)
    access_metadata["access_fingerprint"] = _fingerprint(access_body)
    return _resample_selected_rows(
        timestamps,
        numeric_values,
        start=interval_start,
        stop=interval_stop,
        access_metadata=access_metadata,
    )


def load_governed_metropt_partition(
    csv_path: str | Path,
    partition_name: str,
    *,
    claim: Any,
    access_intent: Any,
    access_intent_path: str | Path,
    phase_manifest_fingerprint: str,
    timestamp_audit: MetroPTTimestampAudit,
    chunksize: int = 100_000,
    runtime_preflight_path: str | Path | None = None,
) -> MetroPTMinuteInterval:
    """Load one frozen partition after authenticating its persisted intent.

    This is the only numeric loader intended for formal staged runners.  All
    authorization and frozen-boundary checks happen before the low-level
    loader is called, so a denied request cannot cause a pandas sensor scan.
    ``partition_name`` is the complete request surface: arbitrary subintervals
    cannot be expressed through this API.
    """

    # Imported lazily so governance remains independent of the data module.
    from .governance import (
        DEVELOPMENT_PROTOCOL_SHA256,
        PARTITIONS,
        PHASE_DEVELOPMENT,
        PHASE_PARTITIONS,
        SOURCE_FILE_SPECS,
        AccessDeniedError,
        AuthenticatedDocument,
        IntegrityError,
        LedgerClaim,
        PORTABILITY_V1_1_DEVELOPMENT_ID,
        access_is_reusable_for_protocol,
        authenticate_fingerprinted_json,
        authorized_partitions_for_protocol,
        authenticate_phase_access_intent,
        canonical_fingerprint,
    )

    if not isinstance(claim, LedgerClaim):
        raise AccessDeniedError("formal numeric access requires a LedgerClaim")
    if not isinstance(access_intent, AuthenticatedDocument):
        raise AccessDeniedError(
            "formal numeric access requires an authenticated access intent"
        )

    # Re-read the durable intent from disk and authenticate it against the
    # exact claim before inspecting a source path or invoking pandas.
    persisted_intent = authenticate_phase_access_intent(
        access_intent_path,
        claim,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
    )
    if (
        persisted_intent.fingerprint != access_intent.fingerprint
        or persisted_intent.value != access_intent.value
        or persisted_intent.record != access_intent.record
    ):
        raise IntegrityError("supplied access intent differs from the persisted intent")

    ledger = claim.document.value
    phase = ledger.get("phase")
    if phase not in PHASE_PARTITIONS:
        raise IntegrityError("workspace ledger has an unsupported phase")
    protocol_id = str(ledger.get("protocol_id"))
    authorized = list(
        authorized_partitions_for_protocol(str(phase), protocol_id)
    )
    ledger_body = dict(ledger)
    stored_ledger_fingerprint = ledger_body.pop("ledger_fingerprint", None)
    if (
        claim.document.fingerprint_field != "ledger_fingerprint"
        or claim.document.fingerprint is None
        or stored_ledger_fingerprint != claim.document.fingerprint
        or canonical_fingerprint(ledger_body) != claim.document.fingerprint
        or ledger.get("schema") != "metropt_cared_bc_workspace_access_ledger"
        or ledger.get("schema_version") != 1
        or (
            phase == PHASE_DEVELOPMENT
            and protocol_id not in DEVELOPMENT_PROTOCOL_SHA256
        )
        or (
            phase == PHASE_DEVELOPMENT
            and ledger.get("charter_sha256")
            != DEVELOPMENT_PROTOCOL_SHA256.get(protocol_id)
        )
        or ledger.get("phase_manifest_fingerprint") != phase_manifest_fingerprint
        or ledger.get("authorized_partitions") != authorized
        or ledger.get("sensor_numeric_access_committed") is not True
        or ledger.get("reusable") is not access_is_reusable_for_protocol(
            str(phase), protocol_id
        )
        or access_intent.value.get("phase") != phase
        or access_intent.value.get("phase_manifest_fingerprint")
        != phase_manifest_fingerprint
        or access_intent.value.get("authorized_partitions") != authorized
        or access_intent.value.get("single_access") is not (
            not access_is_reusable_for_protocol(str(phase), protocol_id)
        )
        or access_intent.value.get("numeric_access_must_follow_this_record") is not True
    ):
        raise IntegrityError("ledger or access intent violates the frozen access contract")
    if phase == PHASE_DEVELOPMENT and any(
        ledger.get(name) is not None
        for name in (
            "final_protocol_sha256",
            "development_completion_fingerprint",
            "development_completion_decision",
            "calibration_completion_fingerprint",
            "calibration_completion_decision",
            "calibration_certificate_fingerprint",
        )
    ):
        raise IntegrityError("development ledger contains a future-phase binding")
    if not claim.may_authorize_numeric_access:
        raise AccessDeniedError("the workspace ledger cannot authorize numeric access")

    if not isinstance(partition_name, str) or not partition_name:
        raise AccessDeniedError("partition_name must name one frozen partition")
    partition_by_name = {partition.name: partition for partition in PARTITIONS}
    if partition_name not in authorized:
        raise AccessDeniedError(
            f"partition {partition_name!r} is not authorized for phase {phase!r}"
        )
    partition = partition_by_name.get(partition_name)
    if partition is None:
        raise IntegrityError("authorized partition is absent from the frozen partition table")

    if protocol_id == PORTABILITY_V1_1_DEVELOPMENT_ID:
        if runtime_preflight_path is None:
            raise AccessDeniedError(
                "v1_1 portability access requires a persisted runtime preflight"
            )
        preflight = authenticate_fingerprinted_json(
            runtime_preflight_path,
            fingerprint_field="preflight_fingerprint",
        )
        if (
            preflight.fingerprint
            != access_intent.value.get("runtime_preflight_fingerprint")
            or preflight.value.get("schema")
            != "metropt_cared_bc_v1_1_runtime_preflight"
            or preflight.value.get("schema_version") != 1
            or preflight.value.get("phase") != PHASE_DEVELOPMENT
            or preflight.value.get("protocol_id") != protocol_id
            or preflight.value.get("phase_manifest_fingerprint")
            != phase_manifest_fingerprint
            or preflight.value.get("passed") is not True
            or preflight.value.get("numeric_access_occurred_before_preflight")
            is not False
        ):
            raise IntegrityError(
                "runtime preflight does not authorize v1_1 numeric access"
            )

    sensor_spec = next(spec for spec in SOURCE_FILE_SPECS if spec.key == "sensor_csv")
    if ledger.get("sensor_csv_sha256") != sensor_spec.sha256:
        raise IntegrityError("workspace ledger is not bound to the frozen sensor CSV")
    if not isinstance(timestamp_audit, MetroPTTimestampAudit):
        raise IntegrityError("formal numeric access requires a timestamp audit")
    expected_partition_counts = {
        frozen.name: frozen.row_count for frozen in PARTITIONS
    }
    if dict(timestamp_audit.partition_row_counts) != expected_partition_counts:
        raise IntegrityError("timestamp audit partition counts differ from the frozen table")
    if timestamp_audit.row_count != sum(expected_partition_counts.values()):
        raise IntegrityError("timestamp audit total row count differs from the frozen table")
    if (
        timestamp_audit.source_sha256 != sensor_spec.sha256
        or timestamp_audit.source_size_bytes != sensor_spec.size_bytes
    ):
        raise IntegrityError("timestamp audit is not bound to the frozen sensor CSV")
    audit_body = timestamp_audit.as_dict()
    if (
        audit_body.get("sensor_numeric_values_parsed") is not False
        or audit_body.get("numeric_columns_parsed") != []
    ):
        raise IntegrityError("timestamp audit parsed a sensor numeric value")

    # Re-authenticate the actual path and bytes before crossing the lower
    # numeric-loader boundary.  This scans only header bytes and a byte hash;
    # it never parses a sensor field.
    source_path = _validated_source_path(csv_path)
    _authenticate_audit(source_path, timestamp_audit)

    interval = load_metropt_interval(
        source_path,
        partition.start,
        partition.end,
        chunksize=chunksize,
        timestamp_audit=timestamp_audit,
    )
    metadata = dict(interval.access_metadata)
    requested = metadata.get("requested_interval")
    if not isinstance(requested, Mapping):
        raise IntegrityError("numeric loader omitted its requested interval")
    try:
        observed_start = pd.Timestamp(requested.get("start_inclusive"))
        observed_stop = pd.Timestamp(requested.get("stop_exclusive"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError("numeric loader returned invalid interval boundaries") from exc
    if (
        observed_start != pd.Timestamp(partition.start)
        or observed_stop != pd.Timestamp(partition.end)
        or requested.get("notation") != "[start, stop)"
        or metadata.get("requested_source_row_count") != partition.row_count
        or metadata.get("source_sha256") != sensor_spec.sha256
        or metadata.get("unrequested_numeric_values_parsed") is not False
        or metadata.get("direct_alarm_values_parsed") is not False
    ):
        raise IntegrityError("numeric loader result differs from its frozen partition")

    governed_body = {
        "formal_governed_entry": True,
        "phase": phase,
        "partition": partition.as_dict(),
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "workspace_ledger_fingerprint": claim.document.fingerprint,
        "access_intent_fingerprint": persisted_intent.fingerprint,
    }
    metadata["governed_partition_access"] = governed_body
    metadata["access_fingerprint"] = _fingerprint(
        {key: value for key, value in metadata.items() if key != "access_fingerprint"}
    )
    return MetroPTMinuteInterval(
        timestamps=interval.timestamps,
        analog=interval.analog,
        controls=interval.controls,
        analog_observed_mask=interval.analog_observed_mask,
        control_observed_mask=interval.control_observed_mask,
        control_imputed_mask=interval.control_imputed_mask,
        bin_observed_mask=interval.bin_observed_mask,
        coverage_fraction=interval.coverage_fraction,
        eligible=interval.eligible,
        reset_before=interval.reset_before,
        source_row_count_per_bin=interval.source_row_count_per_bin,
        access_metadata=metadata,
    )


# The protocol-qualified name is intentionally the governed entry.  The
# unrestricted boundary API remains available only under its explicit
# low-level name, load_metropt_interval.
load_metropt_cared_bc_interval = load_governed_metropt_partition


__all__ = [
    "ANALOG_COLUMNS",
    "APPROVED_NUMERIC_COLUMNS",
    "CONTROL_FFILL_LIMIT_MINUTES",
    "DIRECT_ALARM_COLUMNS",
    "IMPULSE_COLUMN",
    "MetroPTDataError",
    "MetroPTMinuteInterval",
    "MetroPTTimestampAudit",
    "RESAMPLE_FREQUENCY",
    "SAFE_CONTROL_COLUMNS",
    "STATE_CONTROL_COLUMNS",
    "TIMESTAMP_COLUMN",
    "audit_metropt_timestamps",
    "load_governed_metropt_partition",
    "load_metropt_cared_bc_interval",
    "load_metropt_interval",
]
