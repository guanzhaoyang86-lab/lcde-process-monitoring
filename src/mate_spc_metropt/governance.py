"""Fail-closed governance primitives for the MetroPT CARED-BC experiment.

This module never parses sensor values.  It authenticates immutable inputs,
claims phase access before a runner opens an allowed numeric partition, and
binds output artifacts into verifiable completion records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


CHARTER_ID = "metropt_cared_bc_v0_development"
PORTABILITY_DEVELOPMENT_ID = "metropt_cared_bc_v1_portability_development"
PORTABILITY_V1_1_DEVELOPMENT_ID = (
    "metropt_cared_bc_v1_1_portability_development"
)
DEVELOPMENT_PROTOCOL_IDS = (
    CHARTER_ID,
    PORTABILITY_DEVELOPMENT_ID,
    PORTABILITY_V1_1_DEVELOPMENT_ID,
)
PORTABILITY_DEVELOPMENT_SHA256 = (
    "a557aed1a2c8f1dd7de1d34251ccb93e86316d38abff98acc97e1fe476a125fa"
)
PORTABILITY_V1_1_DEVELOPMENT_SHA256 = (
    "2b694d95748fa82153b25c22028030a842f724b7e18a34e5769dce095ff12183"
)
CHARTER_SCHEMA = "metropt_cared_bc_development_charter"
CHARTER_SCHEMA_VERSION = 1
DEVELOPMENT_CHARTER_SHA256 = (
    "11beddf51f11a42b8a41fb25954179e7612c387d0dad3bd74980202e63314382"
)
DEVELOPMENT_PROTOCOL_SHA256: Mapping[str, str] = {
    CHARTER_ID: DEVELOPMENT_CHARTER_SHA256,
    PORTABILITY_DEVELOPMENT_ID: PORTABILITY_DEVELOPMENT_SHA256,
    PORTABILITY_V1_1_DEVELOPMENT_ID: PORTABILITY_V1_1_DEVELOPMENT_SHA256,
}

PHASE_DEVELOPMENT = "development"
PHASE_CALIBRATION = "calibration"
PHASE_CONFIRMATION = "confirmation"
PHASES = (PHASE_DEVELOPMENT, PHASE_CALIBRATION, PHASE_CONFIRMATION)

DEVELOPMENT_PASS = "PASS_DEVELOPMENT_POWER_GATE"
DEVELOPMENT_ABORT = "ABORTED_BEFORE_CALIBRATION_NUMERIC_ACCESS"
CALIBRATION_PASS = "PASS_CALIBRATION_READY"
CALIBRATION_ABORT = "ABORTED_BEFORE_CONFIRMATION_NUMERIC_ACCESS"
CONFIRMATION_PASS = "PASS_LOCKED_FORWARD_CONFIRMATION"
CONFIRMATION_FAIL = "FAIL_LOCKED_FORWARD_CONFIRMATION_NO_RETUNING"

PHASE_DECISIONS: Mapping[str, frozenset[str]] = {
    PHASE_DEVELOPMENT: frozenset((DEVELOPMENT_PASS, DEVELOPMENT_ABORT)),
    PHASE_CALIBRATION: frozenset((CALIBRATION_PASS, CALIBRATION_ABORT)),
    PHASE_CONFIRMATION: frozenset((CONFIRMATION_PASS, CONFIRMATION_FAIL)),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT_FIELDS = {
    "intent_fingerprint",
    "ledger_fingerprint",
    "artifact_manifest_fingerprint",
    "completion_fingerprint",
    "certificate_fingerprint",
    "manifest_fingerprint",
    "preflight_fingerprint",
}


class GovernanceError(ValueError):
    """Base error for a failed governance or authentication check."""


class AccessDeniedError(GovernanceError):
    """Raised before numeric access when a phase is not authorized."""


class IntegrityError(GovernanceError):
    """Raised when a frozen file, record, or fingerprint has changed."""


@dataclass(frozen=True)
class SourceFileSpec:
    """Frozen source-file identity relative to the workspace root."""

    key: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("source relative_path must be nonempty")
        relative = PurePosixPath(self.relative_path)
        if (
            "\\" in self.relative_path
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValueError(f"unsafe source relative_path: {self.relative_path!r}")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError(f"{self.key} SHA-256 must be 64 lowercase hex characters")
        if not self.key or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("source key must be nonempty and size must be nonnegative")


@dataclass(frozen=True)
class PartitionSpec:
    """Frozen half-open timestamp partition."""

    name: str
    start: str
    end: str
    row_count: int
    current_numeric_access: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "row_count": self.row_count,
            "current_numeric_access": self.current_numeric_access,
        }


@dataclass(frozen=True)
class FailureIntervalSpec:
    """Frozen official failure interval used only for partition governance."""

    event_id: str
    role: str
    start: str
    end: str
    row_count: int


@dataclass(frozen=True)
class AuthenticatedDocument:
    """A JSON document authenticated from the same bytes that were parsed."""

    value: dict[str, Any]
    record: dict[str, Any]
    fingerprint: str | None = None
    fingerprint_field: str | None = None


@dataclass(frozen=True)
class DevelopmentInputAuthentication:
    """Authenticated charter and byte-level source bindings."""

    charter: AuthenticatedDocument
    sources: dict[str, dict[str, Any]]
    source_fingerprint: str


@dataclass(frozen=True)
class LedgerClaim:
    """Workspace access ledger plus whether it was created in this call.

    A freshly claimed non-reusable ledger can authorize one access-intent
    creation.  A ledger read back later is audit evidence only.
    """

    document: AuthenticatedDocument
    fresh: bool

    @property
    def may_authorize_numeric_access(self) -> bool:
        reusable = self.document.value.get("reusable") is True
        return self.fresh or reusable


SOURCE_FILE_SPECS = (
    SourceFileSpec(
        key="archive",
        relative_path="data/metropt.zip",
        size_bytes=218381995,
        sha256="aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a",
    ),
    SourceFileSpec(
        key="sensor_csv",
        relative_path="data/metropt/MetroPT3(AirCompressor).csv",
        size_bytes=218300507,
        sha256="db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24",
    ),
    SourceFileSpec(
        key="description_pdf",
        relative_path="data/metropt/Data Description_Metro.pdf",
        size_bytes=81208,
        sha256="b00fac0e8899854078309bef4adaa480d82ecf14dc81c5097c3646973e824127",
    ),
)

PARTITIONS = (
    PartitionSpec(
        "fit",
        "2020-02-01 00:00:00",
        "2020-02-21 00:00:00",
        151297,
        "ALLOWED",
    ),
    PartitionSpec(
        "dev_healthy",
        "2020-02-21 00:00:00",
        "2020-03-01 00:00:00",
        63553,
        "ALLOWED",
    ),
    PartitionSpec(
        "calibration_a",
        "2020-03-01 00:00:00",
        "2020-04-11 00:00:00",
        300769,
        "FORBIDDEN_SEALED",
    ),
    PartitionSpec(
        "embargo_a",
        "2020-04-11 00:00:00",
        "2020-04-18 00:00:00",
        46945,
        "FORBIDDEN_PERMANENT_GUARD_BAND",
    ),
    PartitionSpec(
        "dev_fault_f1",
        "2020-04-18 00:00:00",
        "2020-04-19 00:00:00",
        8663,
        "ALLOWED",
    ),
    PartitionSpec(
        "embargo_b",
        "2020-04-19 00:00:00",
        "2020-04-22 00:00:00",
        22834,
        "FORBIDDEN_PERMANENT_GUARD_BAND",
    ),
    PartitionSpec(
        "calibration_b",
        "2020-04-22 00:00:00",
        "2020-05-21 00:00:00",
        188937,
        "FORBIDDEN_SEALED",
    ),
    PartitionSpec(
        "confirmation",
        "2020-05-21 00:00:00",
        "2020-09-01 00:00:00",
        733420,
        "FORBIDDEN_SEALED",
    ),
    PartitionSpec(
        "post_study_tail",
        "2020-09-01 00:00:00",
        "2020-09-01 04:00:00",
        530,
        "FORBIDDEN_OUTSIDE_STUDY",
    ),
)

FAILURE_INTERVALS = (
    FailureIntervalSpec(
        "F1",
        "development_fault",
        "2020-04-18 00:00:00",
        "2020-04-19 00:00:00",
        8663,
    ),
    FailureIntervalSpec(
        "F2",
        "sealed_confirmation_fault",
        "2020-05-29 23:30:00",
        "2020-05-30 06:00:00",
        2360,
    ),
    FailureIntervalSpec(
        "F3",
        "sealed_confirmation_fault",
        "2020-06-05 10:00:00",
        "2020-06-07 14:30:00",
        17315,
    ),
    FailureIntervalSpec(
        "F4",
        "sealed_confirmation_fault",
        "2020-07-15 14:30:00",
        "2020-07-15 19:00:00",
        1622,
    ),
)

PHASE_PARTITIONS: Mapping[str, tuple[str, ...]] = {
    PHASE_DEVELOPMENT: ("fit", "dev_healthy", "dev_fault_f1"),
    PHASE_CALIBRATION: ("calibration_a", "calibration_b"),
    PHASE_CONFIRMATION: ("confirmation",),
}

PORTABILITY_V1_1_DEVELOPMENT_PARTITIONS = (
    "dev_healthy",
    "dev_fault_f1",
)


def authorized_partitions_for_protocol(
    phase: str, protocol_id: str
) -> tuple[str, ...]:
    """Return the exact immutable authorization set for a phase/protocol pair.

    Legacy development ledgers retain their original three-partition meaning
    so their completed chains remain authenticatable.  The governance-repair
    protocol is a replay and therefore receives no ``fit`` capability.
    Unknown development IDs retain the legacy shape only while an in-memory
    ledger is being constructed; the governed data boundary separately rejects
    every ID absent from ``DEVELOPMENT_PROTOCOL_SHA256`` before source access.
    """

    checked_phase = _validate_phase(phase)
    if (
        checked_phase == PHASE_DEVELOPMENT
        and protocol_id == PORTABILITY_V1_1_DEVELOPMENT_ID
    ):
        return PORTABILITY_V1_1_DEVELOPMENT_PARTITIONS
    return PHASE_PARTITIONS[checked_phase]


def access_is_reusable_for_protocol(phase: str, protocol_id: str) -> bool:
    """Return whether an authenticated ledger may authorize a later intent."""

    checked_phase = _validate_phase(phase)
    return bool(
        checked_phase == PHASE_DEVELOPMENT
        and protocol_id != PORTABILITY_V1_1_DEVELOPMENT_ID
    )


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise GovernanceError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _validate_fingerprint(value: object, *, name: str) -> str:
    return _validate_sha256(value, name=name)


def _validate_phase(phase: str) -> str:
    if phase not in PHASES:
        raise GovernanceError(f"unsupported phase: {phase!r}")
    return phase


def _validate_relative_path(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GovernanceError(f"{name} must be a nonempty relative path")
    if "\\" in value:
        raise GovernanceError(f"{name} must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise GovernanceError(f"{name} is unsafe: {value!r}")
    return path.as_posix()


def _clean_json(value: object, *, path: str = "$") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GovernanceError(f"non-finite JSON number at {path}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise GovernanceError(f"non-string JSON key at {path}")
            cleaned[key] = _clean_json(item, path=f"{path}.{key}")
        return cleaned
    if isinstance(value, (list, tuple)):
        return [
            _clean_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise GovernanceError(f"unsupported JSON value {type(value).__name__} at {path}")


def canonical_json_bytes(value: object) -> bytes:
    """Return strict canonical UTF-8 JSON used for semantic fingerprints."""

    return json.dumps(
        _clean_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: object) -> str:
    """SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pretty_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _clean_json(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise IntegrityError(f"symbolic-link path component is forbidden: {candidate}")


def _require_regular_file(path: Path) -> Path:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular file is missing: {path}")
    return path


def _require_regular_directory(path: Path) -> Path:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"required regular directory is missing: {path}")
    return path


def _regular_file_bytes(path: Path) -> bytes:
    path = _require_regular_file(path)
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise IntegrityError(f"file changed while it was authenticated: {path}")
    return payload


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a non-symlink regular file without interpreting its contents."""

    target = _require_regular_file(Path(path))
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = target.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise IntegrityError(f"file changed while it was hashed: {target}")
    return digest.hexdigest()


def artifact_record(path: str | os.PathLike[str]) -> dict[str, object]:
    """Return the byte identity of a regular non-symlink file."""

    target = _require_regular_file(Path(path))
    return {
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _strict_json_loads(payload: bytes, *, source: str) -> dict[str, Any]:
    def reject_constant(token: str) -> object:
        raise GovernanceError(f"{source} contains forbidden JSON token {token}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GovernanceError(f"{source} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GovernanceError(f"{source} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise GovernanceError(f"{source} must contain a JSON object")
    _clean_json(value)
    return value


def read_strict_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one finite JSON object, rejecting duplicate keys and symlinks."""

    target = Path(path)
    return _strict_json_loads(_regular_file_bytes(target), source=str(target))


def parse_sha256_sidecar(
    path: str | os.PathLike[str], *, expected_filename: str
) -> str:
    """Parse one standard ``digest  filename`` checksum line."""

    target = Path(path)
    try:
        text = _regular_file_bytes(target).decode("ascii")
    except UnicodeDecodeError as error:
        raise IntegrityError(f"checksum sidecar is not ASCII: {target}") from error
    lines = text.splitlines()
    if len(lines) != 1:
        raise IntegrityError(f"checksum sidecar must contain exactly one line: {target}")
    parts = lines[0].split()
    if len(parts) != 2 or parts[1].lstrip("*") != expected_filename:
        raise IntegrityError(f"sidecar does not authenticate {expected_filename}")
    return _validate_sha256(parts[0], name="sidecar digest")


def authenticate_json_with_sidecar(
    path: str | os.PathLike[str],
    sidecar_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> AuthenticatedDocument:
    """Authenticate and parse JSON from exactly the bytes whose hash is checked."""

    target = Path(path)
    sidecar = Path(sidecar_path)
    payload = _regular_file_bytes(target)
    observed = hashlib.sha256(payload).hexdigest()
    declared = parse_sha256_sidecar(sidecar, expected_filename=target.name)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, name="expected JSON SHA-256")
    if observed != declared or (
        expected_sha256 is not None and observed != expected_sha256
    ):
        raise IntegrityError(f"frozen SHA-256 mismatch for {target.name}")
    value = _strict_json_loads(payload, source=str(target))
    record = {
        "filename": target.name,
        "size_bytes": len(payload),
        "sha256": observed,
        "sidecar_filename": sidecar.name,
        "sidecar_sha256": sha256_file(sidecar),
    }
    return AuthenticatedDocument(value=value, record=record)


def atomic_create_new_json(
    path: str | os.PathLike[str], value: object
) -> dict[str, object]:
    """Atomically publish a new JSON file and never replace any path.

    The parent must already be a real directory.  An exclusive hard-link from
    a flushed same-directory temporary file closes the check/create race.
    """

    target = Path(path)
    _require_regular_directory(target.parent)
    if target.is_symlink() or os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing path: {target}")
    payload = _pretty_json_bytes(value)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, target)
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    if target.is_symlink():
        raise IntegrityError(f"atomic target unexpectedly became a symlink: {target}")
    return artifact_record(target)


def create_fingerprinted_json(
    path: str | os.PathLike[str],
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> AuthenticatedDocument:
    """Create-new a document whose semantic body has a canonical fingerprint."""

    if fingerprint_field not in _FINGERPRINT_FIELDS:
        raise GovernanceError(f"unsupported fingerprint field: {fingerprint_field}")
    if fingerprint_field in body:
        raise GovernanceError(f"body already contains {fingerprint_field}")
    clean_body = _clean_json(body)
    if not isinstance(clean_body, dict):
        raise GovernanceError("fingerprinted JSON body must be an object")
    fingerprint = canonical_fingerprint(clean_body)
    value = {**clean_body, fingerprint_field: fingerprint}
    record = atomic_create_new_json(path, value)
    return AuthenticatedDocument(
        value=value,
        record=record,
        fingerprint=fingerprint,
        fingerprint_field=fingerprint_field,
    )


def authenticate_fingerprinted_json(
    path: str | os.PathLike[str],
    *,
    fingerprint_field: str,
    expected_body: Mapping[str, object] | None = None,
) -> AuthenticatedDocument:
    """Authenticate a semantic fingerprint and optionally an exact body."""

    if fingerprint_field not in _FINGERPRINT_FIELDS:
        raise GovernanceError(f"unsupported fingerprint field: {fingerprint_field}")
    target = Path(path)
    value = read_strict_json(target)
    stored = value.get(fingerprint_field)
    body = dict(value)
    body.pop(fingerprint_field, None)
    expected_fingerprint = canonical_fingerprint(body)
    if stored != expected_fingerprint:
        raise IntegrityError(f"{target} has a corrupt {fingerprint_field}")
    if expected_body is not None and body != _clean_json(expected_body):
        raise IntegrityError(f"{target} differs from the expected frozen body")
    return AuthenticatedDocument(
        value=value,
        record=artifact_record(target),
        fingerprint=expected_fingerprint,
        fingerprint_field=fingerprint_field,
    )


def authenticate_source_files(
    workspace_root: str | os.PathLike[str],
    specs: Sequence[SourceFileSpec] = SOURCE_FILE_SPECS,
) -> dict[str, dict[str, object]]:
    """Authenticate source bytes by size and SHA without parsing their content."""

    root = _require_regular_directory(Path(workspace_root)).resolve(strict=True)
    records: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for spec in specs:
        if spec.key in seen:
            raise GovernanceError(f"duplicate source key: {spec.key}")
        seen.add(spec.key)
        relative = _validate_relative_path(spec.relative_path, name="source path")
        candidate = root.joinpath(*PurePosixPath(relative).parts)
        _require_regular_file(candidate)
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"source escapes workspace root: {relative}") from error
        observed_size = candidate.stat().st_size
        observed_sha = sha256_file(candidate)
        if observed_size != spec.size_bytes or observed_sha != spec.sha256:
            raise IntegrityError(f"frozen source mismatch: {spec.key}")
        records[spec.key] = {
            "relative_path": relative,
            "size_bytes": observed_size,
            "sha256": observed_sha,
        }
    return records


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GovernanceError(f"{name} must be a JSON object")
    return value


def _expect(mapping: Mapping[str, object], key: str, expected: object, *, where: str) -> None:
    if mapping.get(key) != expected:
        raise GovernanceError(f"{where}.{key} changed")


def validate_development_charter(charter: Mapping[str, object]) -> None:
    """Validate the frozen semantics of the development charter.

    Byte-level immutability is enforced separately by
    :func:`authenticate_development_inputs`; this function makes the critical
    access, split, source, and method boundaries explicit as defense in depth.
    """

    _expect(charter, "schema", CHARTER_SCHEMA, where="charter")
    _expect(charter, "schema_version", CHARTER_SCHEMA_VERSION, where="charter")
    _expect(charter, "charter_id", CHARTER_ID, where="charter")
    _expect(
        charter,
        "status",
        "DEVELOPMENT_ONLY_SEALED_CALIBRATION_AND_CONFIRMATION",
        where="charter",
    )

    method = _mapping(charter.get("method_family"), name="method_family")
    _expect(method, "name", "CARED-BC", where="method_family")
    _expect(method, "family_is_fixed", True, where="method_family")

    source = _mapping(charter.get("source_binding"), name="source_binding")
    _expect(source, "uci_id", 791, where="source_binding")
    _expect(source, "dataset_doi", "10.24432/C5VW3R", where="source_binding")
    for spec in SOURCE_FILE_SPECS:
        value = _mapping(source.get(spec.key), name=f"source_binding.{spec.key}")
        _expect(value, "path", spec.relative_path, where=f"source_binding.{spec.key}")
        _expect(value, "size_bytes", spec.size_bytes, where=f"source_binding.{spec.key}")
        _expect(value, "sha256", spec.sha256, where=f"source_binding.{spec.key}")

    audit = _mapping(charter.get("timestamp_only_audit"), name="timestamp_only_audit")
    expected_audit = {
        "sensor_values_parsed": False,
        "row_count": 1516948,
        "minimum_timestamp": "2020-02-01 00:00:00",
        "maximum_timestamp": "2020-09-01 03:59:50",
        "timestamps_monotone_non_decreasing": True,
        "duplicate_timestamp_count": 0,
        "nonpositive_delta_count": 0,
        "median_positive_delta_seconds": 10,
        "modal_positive_delta_seconds": 10,
        "gap_count_strictly_greater_than_10_seconds": 51147,
        "gap_count_strictly_greater_than_60_seconds": 331,
    }
    for key, expected in expected_audit.items():
        _expect(audit, key, expected, where="timestamp_only_audit")

    split = _mapping(charter.get("partition_convention"), name="partition_convention")
    _expect(split, "interval_semantics", "left_closed_right_open", where="partition_convention")
    _expect(split, "row_count_total", 1516948, where="partition_convention")
    _expect(split, "partitions_are_disjoint_and_exhaustive", True, where="partition_convention")
    expected_partitions = [partition.as_dict() for partition in PARTITIONS]
    if split.get("partitions") != expected_partitions:
        raise GovernanceError("frozen timestamp partitions changed")
    if sum(partition.row_count for partition in PARTITIONS) != 1516948:
        raise AssertionError("internal partition constants do not sum to the audit total")
    for left, right in zip(PARTITIONS, PARTITIONS[1:], strict=False):
        if datetime.fromisoformat(left.end) != datetime.fromisoformat(right.start):
            raise AssertionError("internal partition constants are not contiguous")

    failures = _mapping(
        charter.get("official_failure_intervals"),
        name="official_failure_intervals",
    )
    observed_events = failures.get("events")
    if not isinstance(observed_events, list) or len(observed_events) != 4:
        raise GovernanceError("official failure interval count changed")
    for observed, expected in zip(observed_events, FAILURE_INTERVALS, strict=True):
        event = _mapping(observed, name=f"failure {expected.event_id}")
        required = {
            "event_id": expected.event_id,
            "role": expected.role,
            "start": expected.start,
            "end": expected.end,
            "timestamp_only_row_count": expected.row_count,
            "fault_type": "air_leak",
            "severity": "high_stress",
        }
        for key, value in required.items():
            _expect(event, key, value, where=f"failure.{expected.event_id}")

    transform = _mapping(
        charter.get("frozen_data_transform"), name="frozen_data_transform"
    )
    resampling = _mapping(transform.get("resampling"), name="resampling")
    _expect(resampling, "period", "1 minute", where="resampling")
    _expect(resampling, "primary_tcn_input_width", 13, where="resampling")
    _expect(
        resampling,
        "coverage_and_missingness_use",
        "eligibility and reset only; coverage or mask values are not model inputs",
        where="resampling",
    )
    _expect(transform, "causal_window_minutes", 10, where="frozen_data_transform")
    _expect(
        transform,
        "forecast_horizons_minutes",
        [1, 3, 5],
        where="frozen_data_transform",
    )
    _expect(
        transform,
        "continuous_channels",
        [
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Oil_temperature",
            "Motor_current",
        ],
        where="frozen_data_transform",
    )
    _expect(
        transform,
        "control_state_channels",
        ["COMP", "DV_eletric", "Towers", "MPG", "Pressure_switch"],
        where="frozen_data_transform",
    )
    _expect(
        transform,
        "flow_channel",
        "Caudal_impulses",
        where="frozen_data_transform",
    )
    _expect(
        transform,
        "direct_alarm_channels_excluded_from_primary_model",
        ["LPS", "Oil_level"],
        where="frozen_data_transform",
    )

    contract = _mapping(
        charter.get("development_method_contract"),
        name="development_method_contract",
    )
    encoder = _mapping(contract.get("encoder"), name="encoder")
    expected_encoder = {
        "type": "causal temporal convolutional network",
        "residual_blocks": 4,
        "channels": 64,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8],
        "latent_width": 32,
        "dropout": 0.1,
        "seeds": [17, 29, 43],
        "required_training_device": "cuda",
    }
    for key, expected in expected_encoder.items():
        _expect(encoder, key, expected, where="development_method_contract.encoder")
    _expect(
        encoder,
        "primary_loss",
        "multi-horizon Huber forecast loss with deterministic control-state-balanced sampling",
        where="development_method_contract.encoder",
    )
    _expect(encoder, "auxiliary_loss", None, where="development_method_contract.encoder")
    _expect(encoder, "auxiliary_loss_weight", 0.0, where="development_method_contract.encoder")

    residual = _mapping(contract.get("residual_representation"), name="residual_representation")
    _expect(
        residual,
        "state_covariance",
        "fit-only Ledoit-Wolf shrinkage covariance",
        where="residual_representation",
    )
    _expect(residual, "rare_state_minimum_fit_minutes", 500, where="residual_representation")
    _expect(
        residual,
        "rare_state_action",
        "pool to the nearest frequent control state by Hamming distance with deterministic lexical tie break",
        where="residual_representation",
    )
    direction = _mapping(contract.get("fault_direction"), name="fault_direction")
    _expect(direction, "location_estimator", "Huber mean", where="fault_direction")
    _expect(
        direction,
        "source",
        "F1 only versus dev_healthy matched by control state",
        where="fault_direction",
    )
    cusum = _mapping(contract.get("cusum"), name="cusum")
    _expect(cusum, "page_kappa", 0.5, where="cusum")

    power = _mapping(charter.get("development_power_gate"), name="development_power_gate")
    _expect(power, "must_pass_before_calibration_unsealing", True, where="development_power_gate")
    _expect(
        power,
        "failure_action",
        "ABORT_BEFORE_CALIBRATION_NUMERIC_ACCESS",
        where="development_power_gate",
    )

    calibration = _mapping(
        charter.get("future_calibration_contract_not_yet_authorized"),
        name="future_calibration_contract_not_yet_authorized",
    )
    _expect(calibration, "data", ["calibration_a", "calibration_b"], where="calibration")
    _expect(calibration, "minimum_qualified_days", 39, where="calibration")
    _expect(calibration, "target_daily_any_alarm_probability", 0.05, where="calibration")

    confirmation = _mapping(
        charter.get("future_confirmation_contract_not_yet_authorized"),
        name="future_confirmation_contract_not_yet_authorized",
    )
    _expect(confirmation, "data", "confirmation", where="confirmation")
    _expect(confirmation, "fault_events", ["F2", "F3", "F4"], where="confirmation")
    _expect(confirmation, "single_numeric_access", True, where="confirmation")
    _expect(confirmation, "fit_selection_calibration_or_gate_change", False, where="confirmation")

    governance = _mapping(charter.get("governance_state"), name="governance_state")
    _expect(
        governance,
        "numeric_access_currently_allowed",
        list(PHASE_PARTITIONS[PHASE_DEVELOPMENT]),
        where="governance_state",
    )
    _expect(
        governance,
        "numeric_access_currently_forbidden",
        [
            "calibration_a",
            "embargo_a",
            "embargo_b",
            "calibration_b",
            "confirmation",
            "post_study_tail",
        ],
        where="governance_state",
    )
    for key in (
        "calibration_unseal_requires_final_protocol_and_power_gate",
        "confirmation_unseal_requires_frozen_calibration_certificate",
        "failed_confirmation_forbids_same_data_retuning",
        "this_file_is_not_a_final_analysis_protocol",
    ):
        _expect(governance, key, True, where="governance_state")


def authenticate_development_inputs(
    charter_path: str | os.PathLike[str],
    charter_sidecar_path: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str],
) -> DevelopmentInputAuthentication:
    """Authenticate the exact charter and all three source byte identities."""

    charter = authenticate_json_with_sidecar(
        charter_path,
        charter_sidecar_path,
        expected_sha256=DEVELOPMENT_CHARTER_SHA256,
    )
    validate_development_charter(charter.value)
    sources = authenticate_source_files(workspace_root)
    return DevelopmentInputAuthentication(
        charter=charter,
        sources=sources,
        source_fingerprint=canonical_fingerprint(sources),
    )


def _phase_permissions(phase: str, protocol_id: str) -> dict[str, bool]:
    authorized = authorized_partitions_for_protocol(phase, protocol_id)
    return {
        "fit_permitted": "fit" in authorized,
        "selection_permitted": (
            phase == PHASE_DEVELOPMENT
            and protocol_id != PORTABILITY_V1_1_DEVELOPMENT_ID
        ),
        "calibration_permitted": phase == PHASE_CALIBRATION,
        "gate_change_permitted": False,
        "same_data_retuning_permitted": False,
    }


def build_workspace_access_ledger(
    *,
    phase: str,
    workspace_relative_path: str,
    protocol_id: str,
    charter_sha256: str,
    final_protocol_sha256: str | None,
    sensor_csv_sha256: str,
    source_fingerprint: str,
    phase_manifest_fingerprint: str,
    output_directory: str | os.PathLike[str],
    development_completion_fingerprint: str | None = None,
    development_completion_decision: str | None = None,
    calibration_completion_fingerprint: str | None = None,
    calibration_completion_decision: str | None = None,
    calibration_certificate_fingerprint: str | None = None,
) -> dict[str, object]:
    """Construct the exact semantic body for a workspace phase ledger.

    Fingerprints must come from :func:`authenticate_completion_record` (and
    then :func:`require_completion_decision`) in the runner.  Requiring the
    matching decision in this immutable body prevents a valid ABORT/FAIL
    completion from being rebound as a phase-opening predecessor.
    """

    phase = _validate_phase(phase)
    relative = _validate_relative_path(
        workspace_relative_path, name="workspace ledger path"
    )
    if not protocol_id:
        raise GovernanceError("protocol_id must be nonempty")
    _validate_sha256(charter_sha256, name="charter SHA-256")
    _validate_sha256(sensor_csv_sha256, name="sensor CSV SHA-256")
    _validate_fingerprint(source_fingerprint, name="source fingerprint")
    _validate_fingerprint(
        phase_manifest_fingerprint, name="phase manifest fingerprint"
    )
    output = _require_regular_directory(Path(output_directory)).resolve(strict=True)

    if phase == PHASE_DEVELOPMENT:
        if final_protocol_sha256 is not None:
            raise AccessDeniedError("development must run under the charter, not a final protocol")
        if any(
            item is not None
            for item in (
                development_completion_fingerprint,
                development_completion_decision,
                calibration_completion_fingerprint,
                calibration_completion_decision,
                calibration_certificate_fingerprint,
            )
        ):
            raise AccessDeniedError("development cannot bind future phase artifacts")
    elif phase == PHASE_CALIBRATION:
        if final_protocol_sha256 is None:
            raise AccessDeniedError("calibration requires a frozen final protocol SHA-256")
        _validate_sha256(final_protocol_sha256, name="final protocol SHA-256")
        _validate_fingerprint(
            development_completion_fingerprint,
            name="development completion fingerprint",
        )
        if development_completion_decision != DEVELOPMENT_PASS:
            raise AccessDeniedError(
                "calibration requires an authenticated development PASS decision"
            )
        if any(
            item is not None
            for item in (
                calibration_completion_fingerprint,
                calibration_completion_decision,
                calibration_certificate_fingerprint,
            )
        ):
            raise AccessDeniedError("calibration cannot bind future calibration artifacts")
    else:
        if final_protocol_sha256 is None:
            raise AccessDeniedError("confirmation requires a frozen final protocol SHA-256")
        _validate_sha256(final_protocol_sha256, name="final protocol SHA-256")
        _validate_fingerprint(
            development_completion_fingerprint,
            name="development completion fingerprint",
        )
        if development_completion_decision != DEVELOPMENT_PASS:
            raise AccessDeniedError(
                "confirmation requires an authenticated development PASS decision"
            )
        _validate_fingerprint(
            calibration_completion_fingerprint,
            name="calibration completion fingerprint",
        )
        if calibration_completion_decision != CALIBRATION_PASS:
            raise AccessDeniedError(
                "confirmation requires an authenticated calibration PASS decision"
            )
        _validate_fingerprint(
            calibration_certificate_fingerprint,
            name="calibration certificate fingerprint",
        )

    return {
        "schema": "metropt_cared_bc_workspace_access_ledger",
        "schema_version": 1,
        "phase": phase,
        "workspace_relative_path": relative,
        "protocol_id": protocol_id,
        "charter_sha256": charter_sha256,
        "final_protocol_sha256": final_protocol_sha256,
        "sensor_csv_sha256": sensor_csv_sha256,
        "source_fingerprint": source_fingerprint,
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "output_directory": str(output),
        "authorized_partitions": list(
            authorized_partitions_for_protocol(phase, protocol_id)
        ),
        "development_completion_fingerprint": development_completion_fingerprint,
        "development_completion_decision": development_completion_decision,
        "calibration_completion_fingerprint": calibration_completion_fingerprint,
        "calibration_completion_decision": calibration_completion_decision,
        "calibration_certificate_fingerprint": calibration_certificate_fingerprint,
        "sensor_numeric_access_committed": True,
        "reusable": access_is_reusable_for_protocol(phase, protocol_id),
    }


def claim_workspace_access_ledger(
    path: str | os.PathLike[str], **ledger_arguments: object
) -> LedgerClaim:
    """Atomically claim a phase before any numeric partition loader is called."""

    body = build_workspace_access_ledger(**ledger_arguments)  # type: ignore[arg-type]
    relative = body["workspace_relative_path"]
    if Path(path).name != PurePosixPath(str(relative)).name:
        raise GovernanceError("ledger filename differs from workspace_relative_path")
    document = create_fingerprinted_json(
        path, body, fingerprint_field="ledger_fingerprint"
    )
    verified = authenticate_fingerprinted_json(
        path,
        fingerprint_field="ledger_fingerprint",
        expected_body=body,
    )
    if verified.fingerprint != document.fingerprint:
        raise IntegrityError("workspace ledger failed post-create verification")
    return LedgerClaim(document=verified, fresh=True)


def authenticate_workspace_access_ledger(
    path: str | os.PathLike[str], *, expected_body: Mapping[str, object]
) -> LedgerClaim:
    """Authenticate existing ledger evidence without re-authorizing access."""

    document = authenticate_fingerprinted_json(
        path,
        fingerprint_field="ledger_fingerprint",
        expected_body=expected_body,
    )
    phase = _validate_phase(str(document.value.get("phase")))
    protocol_id = str(document.value.get("protocol_id"))
    expected_authorized = list(
        authorized_partitions_for_protocol(phase, protocol_id)
    )
    if document.value.get("authorized_partitions") != expected_authorized:
        raise IntegrityError("ledger partition authorization changed")
    if document.value.get("sensor_numeric_access_committed") is not True:
        raise IntegrityError("ledger does not commit sensor access")
    if document.value.get("reusable") is not access_is_reusable_for_protocol(
        phase, protocol_id
    ):
        raise IntegrityError("ledger reuse policy changed")
    return LedgerClaim(document=document, fresh=False)


def _ledger_reference(claim: LedgerClaim) -> dict[str, object]:
    value = claim.document.value
    return {
        "workspace_relative_path": value["workspace_relative_path"],
        "size_bytes": claim.document.record["size_bytes"],
        "sha256": claim.document.record["sha256"],
        "ledger_fingerprint": claim.document.fingerprint,
    }


def _access_intent_body(
    claim: LedgerClaim,
    *,
    phase_manifest_fingerprint: str,
    require_authorization: bool,
    runtime_preflight_fingerprint: str | None,
) -> dict[str, object]:
    ledger = claim.document.value
    phase = _validate_phase(str(ledger.get("phase")))
    protocol_id = str(ledger.get("protocol_id"))
    authorized = list(authorized_partitions_for_protocol(phase, protocol_id))
    _validate_fingerprint(
        phase_manifest_fingerprint, name="phase manifest fingerprint"
    )
    if ledger.get("phase_manifest_fingerprint") != phase_manifest_fingerprint:
        raise IntegrityError("ledger and access intent phase manifests disagree")
    if require_authorization and not claim.may_authorize_numeric_access:
        raise AccessDeniedError(
            f"existing non-reusable {phase} ledger is audit-only; numeric access is spent"
        )
    v1_1 = protocol_id == PORTABILITY_V1_1_DEVELOPMENT_ID
    if v1_1:
        _validate_fingerprint(
            runtime_preflight_fingerprint,
            name="runtime preflight fingerprint",
        )
    elif runtime_preflight_fingerprint is not None:
        raise AccessDeniedError(
            "runtime preflight binding is reserved for the v1_1 portability protocol"
        )
    body = {
        "schema": "metropt_cared_bc_phase_access_intent",
        "schema_version": 1,
        "phase": phase,
        "protocol_id": protocol_id,
        "charter_sha256": ledger["charter_sha256"],
        "final_protocol_sha256": ledger["final_protocol_sha256"],
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "authorized_partitions": authorized,
        "single_access": not access_is_reusable_for_protocol(phase, protocol_id),
        "permissions": _phase_permissions(phase, protocol_id),
        "workspace_ledger": _ledger_reference(claim),
        "numeric_access_must_follow_this_record": True,
    }
    if v1_1:
        body["runtime_preflight_fingerprint"] = runtime_preflight_fingerprint
        body["runtime_preflight_must_precede_numeric_access"] = True
    return body


def build_phase_access_intent(
    claim: LedgerClaim,
    *,
    phase_manifest_fingerprint: str,
    runtime_preflight_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build an access intent; stale calibration/confirmation ledgers fail closed."""

    return _access_intent_body(
        claim,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        require_authorization=True,
        runtime_preflight_fingerprint=runtime_preflight_fingerprint,
    )


def create_phase_access_intent(
    path: str | os.PathLike[str],
    claim: LedgerClaim,
    *,
    phase_manifest_fingerprint: str,
    runtime_preflight_fingerprint: str | None = None,
) -> AuthenticatedDocument:
    """Create the durable intent immediately before a permitted numeric load."""

    body = build_phase_access_intent(
        claim,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        runtime_preflight_fingerprint=runtime_preflight_fingerprint,
    )
    return create_fingerprinted_json(
        path, body, fingerprint_field="intent_fingerprint"
    )


def authenticate_phase_access_intent(
    path: str | os.PathLike[str],
    claim: LedgerClaim,
    *,
    phase_manifest_fingerprint: str,
    runtime_preflight_fingerprint: str | None = None,
) -> AuthenticatedDocument:
    """Audit an existing access intent; this does not grant another access."""

    if (
        runtime_preflight_fingerprint is None
        and claim.document.value.get("protocol_id")
        == PORTABILITY_V1_1_DEVELOPMENT_ID
    ):
        persisted = read_strict_json(path)
        candidate = persisted.get("runtime_preflight_fingerprint")
        runtime_preflight_fingerprint = _validate_fingerprint(
            candidate, name="runtime preflight fingerprint"
        )
    body = _access_intent_body(
        claim,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        require_authorization=False,
        runtime_preflight_fingerprint=runtime_preflight_fingerprint,
    )
    return authenticate_fingerprinted_json(
        path,
        fingerprint_field="intent_fingerprint",
        expected_body=body,
    )


def _safe_artifact_path(output_directory: Path, relative_name: str) -> Path:
    relative = _validate_relative_path(relative_name, name="artifact name")
    root = _require_regular_directory(output_directory).resolve(strict=True)
    candidate = output_directory.joinpath(*PurePosixPath(relative).parts)
    _require_regular_file(candidate)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise IntegrityError(f"artifact escapes output directory: {relative}") from error
    return candidate


def create_artifact_manifest(
    path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    phase: str,
    protocol_id: str,
    protocol_sha256: str,
    phase_manifest_fingerprint: str,
    access_intent: AuthenticatedDocument,
    artifact_names: Iterable[str],
) -> AuthenticatedDocument:
    """Hash a fixed set of phase outputs into a create-new artifact manifest."""

    phase = _validate_phase(phase)
    _validate_sha256(protocol_sha256, name="protocol SHA-256")
    _validate_fingerprint(
        phase_manifest_fingerprint, name="phase manifest fingerprint"
    )
    if access_intent.value.get("phase") != phase:
        raise IntegrityError("access intent phase differs from artifact manifest")
    names = sorted(set(artifact_names))
    if not names:
        raise GovernanceError("artifact manifest cannot be empty")
    output = Path(output_directory)
    records: dict[str, object] = {}
    for name in names:
        candidate = _safe_artifact_path(output, name)
        record = artifact_record(candidate)
        record.pop("filename")
        records[_validate_relative_path(name, name="artifact name")] = record
    body = {
        "schema": "metropt_cared_bc_artifact_manifest",
        "schema_version": 1,
        "phase": phase,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "access_intent": {
            **access_intent.record,
            "intent_fingerprint": access_intent.fingerprint,
        },
        "artifacts": records,
    }
    return create_fingerprinted_json(
        path, body, fingerprint_field="artifact_manifest_fingerprint"
    )


def authenticate_artifact_manifest(
    path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    phase: str,
    protocol_id: str,
    protocol_sha256: str,
    phase_manifest_fingerprint: str,
    access_intent: AuthenticatedDocument,
    expected_artifact_names: Sequence[str],
) -> AuthenticatedDocument:
    """Verify manifest identity and re-hash every declared output artifact."""

    document = authenticate_fingerprinted_json(
        path, fingerprint_field="artifact_manifest_fingerprint"
    )
    value = document.value
    expected_identity = {
        "schema": "metropt_cared_bc_artifact_manifest",
        "schema_version": 1,
        "phase": _validate_phase(phase),
        "protocol_id": protocol_id,
        "protocol_sha256": _validate_sha256(
            protocol_sha256, name="protocol SHA-256"
        ),
        "phase_manifest_fingerprint": _validate_fingerprint(
            phase_manifest_fingerprint, name="phase manifest fingerprint"
        ),
        "access_intent": {
            **access_intent.record,
            "intent_fingerprint": access_intent.fingerprint,
        },
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise IntegrityError(f"artifact manifest identity changed: {key}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise IntegrityError("artifact manifest has no artifact mapping")
    names = sorted(set(expected_artifact_names))
    if set(artifacts) != set(names):
        raise IntegrityError("artifact manifest file set changed")
    output = Path(output_directory)
    for name in names:
        expected = artifacts.get(name)
        if not isinstance(expected, Mapping):
            raise IntegrityError(f"invalid artifact record: {name}")
        observed = artifact_record(_safe_artifact_path(output, name))
        observed.pop("filename")
        if dict(expected) != observed:
            raise IntegrityError(f"completed artifact changed: {name}")
    return document


def _completion_body(
    *,
    claim: LedgerClaim,
    access_intent: AuthenticatedDocument,
    artifact_manifest: AuthenticatedDocument,
    decision: str,
) -> dict[str, object]:
    ledger = claim.document.value
    phase = _validate_phase(str(ledger.get("phase")))
    if decision not in PHASE_DECISIONS[phase]:
        raise GovernanceError(f"decision {decision!r} is invalid for {phase}")
    if access_intent.value.get("phase") != phase or artifact_manifest.value.get("phase") != phase:
        raise IntegrityError("completion chain has inconsistent phases")
    if (
        access_intent.value.get("phase_manifest_fingerprint")
        != ledger.get("phase_manifest_fingerprint")
        or artifact_manifest.value.get("phase_manifest_fingerprint")
        != ledger.get("phase_manifest_fingerprint")
    ):
        raise IntegrityError("completion chain has inconsistent phase manifests")
    return {
        "schema": "metropt_cared_bc_phase_completion",
        "schema_version": 1,
        "phase": phase,
        "protocol_id": ledger["protocol_id"],
        "charter_sha256": ledger["charter_sha256"],
        "final_protocol_sha256": ledger["final_protocol_sha256"],
        "phase_manifest_fingerprint": ledger["phase_manifest_fingerprint"],
        "decision": decision,
        "workspace_ledger": _ledger_reference(claim),
        "access_intent": {
            **access_intent.record,
            "intent_fingerprint": access_intent.fingerprint,
        },
        "artifact_manifest": {
            **artifact_manifest.record,
            "artifact_manifest_fingerprint": artifact_manifest.fingerprint,
        },
    }


def create_completion_record(
    path: str | os.PathLike[str],
    *,
    claim: LedgerClaim,
    access_intent: AuthenticatedDocument,
    artifact_manifest: AuthenticatedDocument,
    decision: str,
) -> AuthenticatedDocument:
    """Create-new a phase completion bound to ledger, intent, and artifacts."""

    body = _completion_body(
        claim=claim,
        access_intent=access_intent,
        artifact_manifest=artifact_manifest,
        decision=decision,
    )
    return create_fingerprinted_json(
        path, body, fingerprint_field="completion_fingerprint"
    )


def authenticate_completion_record(
    path: str | os.PathLike[str],
    *,
    output_directory: str | os.PathLike[str],
    ledger_path: str | os.PathLike[str],
    expected_ledger_body: Mapping[str, object],
    access_intent_path: str | os.PathLike[str],
    artifact_manifest_path: str | os.PathLike[str],
    expected_artifact_names: Sequence[str],
) -> AuthenticatedDocument:
    """Authenticate a complete phase chain and every output artifact."""

    claim = authenticate_workspace_access_ledger(
        ledger_path, expected_body=expected_ledger_body
    )
    ledger = claim.document.value
    phase = _validate_phase(str(ledger.get("phase")))
    phase_fingerprint = str(ledger.get("phase_manifest_fingerprint"))
    access_intent = authenticate_phase_access_intent(
        access_intent_path,
        claim,
        phase_manifest_fingerprint=phase_fingerprint,
    )
    effective_protocol_sha = (
        ledger.get("final_protocol_sha256") or ledger.get("charter_sha256")
    )
    artifact_manifest = authenticate_artifact_manifest(
        artifact_manifest_path,
        output_directory,
        phase=phase,
        protocol_id=str(ledger.get("protocol_id")),
        protocol_sha256=str(effective_protocol_sha),
        phase_manifest_fingerprint=phase_fingerprint,
        access_intent=access_intent,
        expected_artifact_names=expected_artifact_names,
    )
    completion = authenticate_fingerprinted_json(
        path, fingerprint_field="completion_fingerprint"
    )
    decision = completion.value.get("decision")
    if decision not in PHASE_DECISIONS[phase]:
        raise IntegrityError("completion decision is invalid for its phase")
    expected_body = _completion_body(
        claim=claim,
        access_intent=access_intent,
        artifact_manifest=artifact_manifest,
        decision=str(decision),
    )
    observed_body = dict(completion.value)
    observed_body.pop("completion_fingerprint", None)
    if observed_body != expected_body:
        raise IntegrityError("completion record differs from its authenticated chain")
    return completion


def require_completion_decision(
    completion: AuthenticatedDocument,
    *,
    phase: str,
    required_decision: str,
) -> str:
    """Return a predecessor fingerprint only after an authenticated PASS gate."""

    phase = _validate_phase(phase)
    if required_decision not in PHASE_DECISIONS[phase]:
        raise GovernanceError("required decision is not valid for the phase")
    if (
        completion.fingerprint_field != "completion_fingerprint"
        or completion.value.get("phase") != phase
        or completion.value.get("decision") != required_decision
        or completion.fingerprint is None
    ):
        raise AccessDeniedError(
            f"{phase} completion did not authenticate required decision {required_decision}"
        )
    return _validate_fingerprint(
        completion.fingerprint, name=f"{phase} completion fingerprint"
    )


__all__ = [
    "AccessDeniedError",
    "AuthenticatedDocument",
    "CALIBRATION_ABORT",
    "CALIBRATION_PASS",
    "CHARTER_ID",
    "CHARTER_SCHEMA",
    "CHARTER_SCHEMA_VERSION",
    "CONFIRMATION_FAIL",
    "CONFIRMATION_PASS",
    "DEVELOPMENT_ABORT",
    "DEVELOPMENT_CHARTER_SHA256",
    "DEVELOPMENT_PROTOCOL_IDS",
    "DEVELOPMENT_PROTOCOL_SHA256",
    "DEVELOPMENT_PASS",
    "DevelopmentInputAuthentication",
    "FAILURE_INTERVALS",
    "FailureIntervalSpec",
    "GovernanceError",
    "IntegrityError",
    "LedgerClaim",
    "PARTITIONS",
    "PHASE_CALIBRATION",
    "PHASE_CONFIRMATION",
    "PHASE_DECISIONS",
    "PHASE_DEVELOPMENT",
    "PHASE_PARTITIONS",
    "PORTABILITY_DEVELOPMENT_ID",
    "PORTABILITY_DEVELOPMENT_SHA256",
    "PORTABILITY_V1_1_DEVELOPMENT_ID",
    "PORTABILITY_V1_1_DEVELOPMENT_PARTITIONS",
    "PORTABILITY_V1_1_DEVELOPMENT_SHA256",
    "PartitionSpec",
    "SOURCE_FILE_SPECS",
    "SourceFileSpec",
    "artifact_record",
    "access_is_reusable_for_protocol",
    "authorized_partitions_for_protocol",
    "atomic_create_new_json",
    "authenticate_artifact_manifest",
    "authenticate_completion_record",
    "authenticate_development_inputs",
    "authenticate_fingerprinted_json",
    "authenticate_json_with_sidecar",
    "authenticate_phase_access_intent",
    "authenticate_source_files",
    "authenticate_workspace_access_ledger",
    "build_phase_access_intent",
    "build_workspace_access_ledger",
    "canonical_fingerprint",
    "canonical_json_bytes",
    "claim_workspace_access_ledger",
    "create_artifact_manifest",
    "create_completion_record",
    "create_fingerprinted_json",
    "create_phase_access_intent",
    "parse_sha256_sidecar",
    "read_strict_json",
    "require_completion_decision",
    "sha256_file",
    "validate_development_charter",
]
