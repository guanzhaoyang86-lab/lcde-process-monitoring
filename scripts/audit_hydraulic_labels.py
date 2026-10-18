"""Perform the frozen, label-only feasibility audit for UCI Hydraulic 447.

This program deliberately has no sensor-data loader.  It may read the content of
``profile.txt`` and the frozen protocol, and it may stream the downloaded archive
solely to compute its SHA-256 digest.  Every other file below ``--dataset-dir`` is
observed through filesystem metadata only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid
from typing import Iterable, Mapping, Sequence


SCHEMA = "hydraulic_label_only_audit"
SCHEMA_VERSION = 1
COMPLETION_SCHEMA = "hydraulic_label_only_audit_completion"
EXPECTED_ROWS = 2205
EXPECTED_COLUMNS = (
    "cooler_condition_percent",
    "valve_condition_percent",
    "internal_pump_leakage",
    "hydraulic_accumulator_bar",
    "stable_flag",
)
EXPECTED_STABLE_VALUES = (0, 1)
EXPECTED_STABLE_FLAG_VALUE = 0
EXPECTED_VALVE_VALUES = (100, 90, 80, 73)
RESULT_NAME = "label_audit.json"
COMPLETION_NAME = "completion.json"
OUTPUT_FILES = frozenset({RESULT_NAME, COMPLETION_NAME})
INTEGER_TOKEN = re.compile(r"[+-]?\d+\Z")


def _canonical_json_bytes(value: object) -> bytes:
    """Return the single canonical representation used for every fingerprint."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_regular_file(path: Path, description: str) -> Path:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise ValueError(f"{description} must not be a symbolic link: {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"{description} is not a regular file: {path}")
    return path.resolve(strict=True)


class LabelOnlyContentReader:
    """Capability guard for the only input files whose bytes may be accessed."""

    _TEXT_ROLES = frozenset({"profile", "protocol", "protocol_checksum"})
    _HASH_ROLES = frozenset({"profile", "protocol", "archive"})

    def __init__(
        self,
        *,
        profile: Path,
        protocol: Path,
        protocol_checksum: Path,
        archive: Path,
    ) -> None:
        self._paths = {
            "profile": profile.resolve(strict=True),
            "protocol": protocol.resolve(strict=True),
            "protocol_checksum": protocol_checksum.resolve(strict=True),
            "archive": archive.resolve(strict=True),
        }
        self.access_log: list[dict[str, str]] = []

    def _authorized(self, path: Path, role: str, operations: frozenset[str]) -> Path:
        if role not in operations:
            raise PermissionError(f"content operation is forbidden for role {role!r}")
        resolved = path.resolve(strict=True)
        if self._paths.get(role) != resolved:
            raise PermissionError(f"path is not the registered {role} input: {path}")
        return resolved

    def read_text(self, path: Path, *, role: str) -> str:
        resolved = self._authorized(path, role, self._TEXT_ROLES)
        self.access_log.append({"role": role, "operation": "read_utf8_text"})
        return resolved.read_text(encoding="utf-8")

    def sha256(self, path: Path, *, role: str) -> str:
        resolved = self._authorized(path, role, self._HASH_ROLES)
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.access_log.append({"role": role, "operation": "sha256_stream"})
        return digest.hexdigest()


def _parse_checksum_sidecar(text: str, expected_filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("protocol SHA-256 sidecar must contain exactly one record")
    fields = lines[0].split()
    if len(fields) != 2:
        raise ValueError("protocol SHA-256 sidecar must use '<digest> <filename>'")
    digest, filename = fields
    filename = filename.removeprefix("*")
    if filename != expected_filename:
        raise ValueError("protocol SHA-256 sidecar names a different protocol file")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("protocol SHA-256 sidecar contains an invalid digest")
    return digest.lower()


def _validate_protocol(protocol: Mapping[str, object]) -> dict[str, object]:
    if protocol.get("schema") != "hydraulic_ws_pac_acquisition_protocol":
        raise ValueError("unexpected acquisition protocol schema")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported acquisition protocol schema version")
    if protocol.get("frozen_before_download") is not True:
        raise ValueError("acquisition protocol was not frozen before download")
    if protocol.get("protocol_id") != "hydraulic_ws_pac_v1":
        raise ValueError("unexpected acquisition protocol id")
    declared = protocol.get("declared_before_access")
    if not isinstance(declared, dict):
        raise ValueError("protocol has no declared_before_access object")
    expected = {
        "instances": EXPECTED_ROWS,
        "target_file": "profile.txt",
        "profile_columns": list(EXPECTED_COLUMNS),
        "stable_flag_value": EXPECTED_STABLE_FLAG_VALUE,
        "healthy_valve_value": EXPECTED_VALVE_VALUES[0],
        "faulty_valve_values": list(EXPECTED_VALVE_VALUES[1:]),
    }
    for key, value in expected.items():
        if declared.get(key) != value:
            raise ValueError(f"protocol declaration {key!r} is not the frozen value")
    rules = protocol.get("feasibility_rules")
    if not isinstance(rules, dict):
        raise ValueError("protocol has no feasibility_rules object")
    if rules.get("episode_definition") != (
        "maximal contiguous rows with identical five-column profile tuple"
    ):
        raise ValueError("protocol episode definition is not the frozen definition")
    minimum = rules.get("minimum_pac_calibration_healthy_cycles")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("protocol minimum healthy-cycle count is invalid")
    if rules.get("abort_certificate_if_minimum_not_met") is not True:
        raise ValueError("protocol must abort the certificate when the minimum is unmet")
    permissions = protocol.get("first_stage_permissions")
    if not isinstance(permissions, dict):
        raise ValueError("protocol has no first_stage_permissions object")
    allowed = permissions.get("allowed")
    forbidden = permissions.get("forbidden_until_analysis_protocol_is_frozen")
    if not isinstance(allowed, list) or "profile.txt label-only feasibility audit" not in allowed:
        raise ValueError("protocol does not authorize the label-only feasibility audit")
    if not isinstance(forbidden, list) or "reading sensor values" not in forbidden:
        raise ValueError("protocol does not explicitly forbid reading sensor values")
    return {
        "protocol_id": protocol.get("protocol_id"),
        "minimum_pac_calibration_healthy_cycles": minimum,
        "abort_certificate_if_minimum_not_met": True,
    }


def parse_profile(text: str) -> list[tuple[int, int, int, int, int]]:
    """Parse and fully validate the five integer label columns."""

    lines = text.splitlines()
    if len(lines) != EXPECTED_ROWS:
        raise ValueError(
            f"profile.txt must contain exactly {EXPECTED_ROWS} rows; found {len(lines)}"
        )
    rows: list[tuple[int, int, int, int, int]] = []
    for row_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != len(EXPECTED_COLUMNS):
            raise ValueError(
                f"profile.txt row {row_number} must have exactly five columns"
            )
        if any(INTEGER_TOKEN.fullmatch(field) is None for field in fields):
            raise ValueError(f"profile.txt row {row_number} contains a non-integer token")
        values = tuple(int(field) for field in fields)
        rows.append(values)  # type: ignore[arg-type]

    stable_values = {row[4] for row in rows}
    if stable_values != set(EXPECTED_STABLE_VALUES):
        raise ValueError(
            "stable_flag must contain exactly the integer values 0 and 1; "
            f"found {sorted(stable_values)}"
        )
    valve_values = {row[1] for row in rows}
    if valve_values != set(EXPECTED_VALVE_VALUES):
        raise ValueError(
            "valve_condition_percent must contain exactly 100, 90, 80, and 73; "
            f"found {sorted(valve_values)}"
        )
    return rows


def build_episodes(
    rows: Sequence[tuple[int, int, int, int, int]],
) -> list[dict[str, object]]:
    """Build maximal contiguous episodes using equality of the complete tuple."""

    if not rows:
        return []
    episodes: list[dict[str, object]] = []
    start = 0
    current = rows[0]

    def append_episode(stop_exclusive: int) -> None:
        episode_id = len(episodes) + 1
        episodes.append(
            {
                "episode_id": episode_id,
                "start_row_1based": start + 1,
                "end_row_1based_inclusive": stop_exclusive,
                "length_cycles": stop_exclusive - start,
                "profile_tuple": list(current),
                "profile": dict(zip(EXPECTED_COLUMNS, current)),
            }
        )

    for index in range(1, len(rows)):
        if rows[index] != current:
            append_episode(index)
            start = index
            current = rows[index]
    append_episode(len(rows))
    return episodes


def _tuple_count_records(
    tuples: Iterable[tuple[int, int, int, int, int]],
) -> list[dict[str, object]]:
    counts = Counter(tuples)
    return [
        {"profile_tuple": list(profile), "cycle_count": counts[profile]}
        for profile in sorted(counts)
    ]


def summarize_cycles(
    rows: Sequence[tuple[int, int, int, int, int]],
    *,
    stable_only: bool,
) -> dict[str, object]:
    selected = [row for row in rows if not stable_only or row[4] == 0]
    by_valve = Counter(row[1] for row in selected)
    return {
        "cycle_count": len(selected),
        "by_valve_condition_percent": {
            str(value): by_valve[value] for value in EXPECTED_VALVE_VALUES
        },
        "by_complete_profile_tuple": _tuple_count_records(selected),
    }


def summarize_episodes(
    episodes: Sequence[Mapping[str, object]],
    *,
    stable_only: bool,
) -> dict[str, object]:
    selected = [
        episode
        for episode in episodes
        if not stable_only
        or int(episode["profile"]["stable_flag"]) == EXPECTED_STABLE_FLAG_VALUE  # type: ignore[index]
    ]
    lengths = [int(episode["length_cycles"]) for episode in selected]
    by_valve: dict[str, dict[str, object]] = {}
    for value in EXPECTED_VALVE_VALUES:
        subset = [
            episode
            for episode in selected
            if int(episode["profile"]["valve_condition_percent"]) == value  # type: ignore[index]
        ]
        by_valve[str(value)] = {
            "episode_count": len(subset),
            "cycle_count": sum(int(episode["length_cycles"]) for episode in subset),
            "episode_ids": [int(episode["episode_id"]) for episode in subset],
            "lengths_cycles": [int(episode["length_cycles"]) for episode in subset],
        }
    return {
        "episode_count": len(selected),
        "cycle_count": sum(lengths),
        "minimum_length_cycles": min(lengths) if lengths else None,
        "maximum_length_cycles": max(lengths) if lengths else None,
        "by_valve_condition_percent": by_valve,
    }


def collect_file_metadata(dataset_dir: Path) -> list[dict[str, object]]:
    """Inventory regular files without opening any of them or following links."""

    root = dataset_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset directory is not a directory: {dataset_dir}")
    records: list[dict[str, object]] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValueError(f"dataset inventory refuses symbolic link: {candidate}")
        for name in file_names:
            candidate = current_path / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"dataset inventory refuses symbolic link: {candidate}")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"dataset inventory found a non-regular file: {candidate}")
            records.append(
                {
                    "relative_path": candidate.relative_to(root).as_posix(),
                    "size_bytes": info.st_size,
                }
            )
    records.sort(key=lambda record: str(record["relative_path"]))
    return records


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def build_audit(
    *,
    dataset_dir: Path,
    archive_path: Path,
    protocol_path: Path,
    protocol_checksum_path: Path,
) -> dict[str, object]:
    dataset_root = dataset_dir.resolve(strict=True)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset directory is not a directory: {dataset_dir}")
    profile_path = _require_regular_file(dataset_root / "profile.txt", "profile")
    archive = _require_regular_file(archive_path, "source archive")
    protocol_file = _require_regular_file(protocol_path, "acquisition protocol")
    checksum_file = _require_regular_file(
        protocol_checksum_path, "acquisition protocol checksum"
    )
    reader = LabelOnlyContentReader(
        profile=profile_path,
        protocol=protocol_file,
        protocol_checksum=checksum_file,
        archive=archive,
    )

    protocol_sha256 = reader.sha256(protocol_file, role="protocol")
    declared_protocol_sha256 = _parse_checksum_sidecar(
        reader.read_text(checksum_file, role="protocol_checksum"), protocol_file.name
    )
    if protocol_sha256 != declared_protocol_sha256:
        raise ValueError("acquisition protocol does not match its frozen SHA-256 sidecar")
    protocol_object = json.loads(reader.read_text(protocol_file, role="protocol"))
    if not isinstance(protocol_object, dict):
        raise ValueError("acquisition protocol JSON must be an object")
    protocol_values = _validate_protocol(protocol_object)

    profile_sha256 = reader.sha256(profile_path, role="profile")
    rows = parse_profile(reader.read_text(profile_path, role="profile"))
    archive_sha256 = reader.sha256(archive, role="archive")
    inventory = collect_file_metadata(dataset_root)
    episodes = build_episodes(rows)
    all_counts = summarize_cycles(rows, stable_only=False)
    stable_counts = summarize_cycles(rows, stable_only=True)
    all_episode_summary = summarize_episodes(episodes, stable_only=False)
    stable_episode_summary = summarize_episodes(episodes, stable_only=True)
    minimum = int(protocol_values["minimum_pac_calibration_healthy_cycles"])
    stable_healthy = int(
        stable_counts["by_valve_condition_percent"][str(EXPECTED_VALVE_VALUES[0])]  # type: ignore[index]
    )
    eligible = stable_healthy >= minimum

    body: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_values["protocol_id"],
        "audit_scope": {
            "label_only": True,
            "sensor_values_accessed": False,
            "sensor_content_hashes_computed": False,
            "permitted_content_inputs": [
                "profile.txt labels",
                "frozen acquisition protocol",
                "protocol SHA-256 sidecar",
                "source archive bytes for SHA-256 only",
            ],
            "other_extracted_files": "filesystem metadata only",
            "content_access_log": reader.access_log,
        },
        "inputs": {
            "profile": {
                "relative_path": profile_path.relative_to(dataset_root).as_posix(),
                "size_bytes": profile_path.stat().st_size,
                "sha256": profile_sha256,
            },
            "archive": {
                "filename": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": archive_sha256,
                "content_use": "sha256_only",
            },
            "protocol": {
                "filename": protocol_file.name,
                "size_bytes": protocol_file.stat().st_size,
                "sha256": protocol_sha256,
                "declared_sha256": declared_protocol_sha256,
            },
        },
        "validations": {
            "shape": [len(rows), len(EXPECTED_COLUMNS)],
            "integer_only": True,
            "profile_columns": list(EXPECTED_COLUMNS),
            "stable_flag_values": list(EXPECTED_STABLE_VALUES),
            "valve_condition_values": list(EXPECTED_VALVE_VALUES),
            "complete_profile_tuple_episode_definition": True,
        },
        "dataset_file_metadata": {
            "file_count": len(inventory),
            "files": inventory,
            "content_read": False,
        },
        "cycle_counts": {"all": all_counts, "stable": stable_counts},
        "episode_summaries": {
            "all": all_episode_summary,
            "stable": stable_episode_summary,
        },
        "episodes": episodes,
        "feasibility": {
            "stable_healthy_cycle_count": stable_healthy,
            "minimum_pac_calibration_healthy_cycles": minimum,
            "minimum_met": eligible,
            "certificate_status": "ELIGIBLE" if eligible else "ABORT_MINIMUM_NOT_MET",
            "abort_certificate_if_minimum_not_met": protocol_values[
                "abort_certificate_if_minimum_not_met"
            ],
        },
        "canonicalization": {
            "algorithm": "SHA-256(canonical UTF-8 JSON)",
            "json_rules": "sort_keys=true,separators=(',',':'),ensure_ascii=false,allow_nan=false",
            "fingerprint_excludes_field": "audit_fingerprint",
        },
    }
    return {**body, "audit_fingerprint": canonical_fingerprint(body)}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _sha256_output_artifact(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Atomically publish a new file without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _completion_body(result_path: Path, audit_fingerprint: str) -> dict[str, object]:
    return {
        "schema": COMPLETION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "audit_fingerprint": audit_fingerprint,
        "sensor_values_accessed": False,
        "artifacts": {
            RESULT_NAME: {
                "sha256": _sha256_output_artifact(result_path),
                "size_bytes": result_path.stat().st_size,
            }
        },
    }


def verify_completion(
    output_dir: Path, *, expected_audit_fingerprint: str | None = None
) -> dict[str, object] | None:
    if not output_dir.exists():
        return None
    entries = {path.name for path in output_dir.iterdir()}
    if not entries:
        return None
    if COMPLETION_NAME not in entries:
        raise RuntimeError("partial label-audit output exists; refusing to overwrite it")
    if entries != OUTPUT_FILES:
        raise RuntimeError(
            "completed label-audit directory has missing or unexpected artifacts"
        )
    completion_path = output_dir / COMPLETION_NAME
    result_path = output_dir / RESULT_NAME
    completion = _read_json(completion_path)
    stored_completion_fingerprint = completion.pop("completion_fingerprint", None)
    if stored_completion_fingerprint != canonical_fingerprint(completion):
        raise ValueError("completion fingerprint is corrupt")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError("completion schema is invalid")
    if completion.get("sensor_values_accessed") is not False:
        raise ValueError("completion does not certify label-only access")
    result = _read_json(result_path)
    stored_audit_fingerprint = result.pop("audit_fingerprint", None)
    if stored_audit_fingerprint != canonical_fingerprint(result):
        raise ValueError("label-audit fingerprint is corrupt")
    if result.get("audit_scope", {}).get("sensor_values_accessed") is not False:  # type: ignore[union-attr]
        raise ValueError("label-audit result does not certify label-only access")
    if stored_audit_fingerprint != completion.get("audit_fingerprint"):
        raise ValueError("completion authenticates a different label audit")
    if (
        expected_audit_fingerprint is not None
        and stored_audit_fingerprint != expected_audit_fingerprint
    ):
        raise ValueError("completed output belongs to different inputs or labels")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {RESULT_NAME}:
        raise ValueError("completion artifact map is invalid")
    record = artifacts[RESULT_NAME]
    if not isinstance(record, dict):
        raise ValueError("completion artifact record is invalid")
    if record.get("sha256") != _sha256_output_artifact(result_path):
        raise ValueError("label-audit artifact SHA-256 mismatch")
    if record.get("size_bytes") != result_path.stat().st_size:
        raise ValueError("label-audit artifact size mismatch")
    return {**completion, "completion_fingerprint": stored_completion_fingerprint}


def run_audit(
    *,
    dataset_dir: Path,
    archive_path: Path,
    protocol_path: Path,
    protocol_checksum_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    dataset_root = dataset_dir.resolve(strict=True)
    output = output_dir.resolve()
    if _is_within(output, dataset_root):
        raise ValueError("output directory must not be inside the extracted dataset")
    if output.exists() and not output.is_dir():
        raise ValueError("output path exists and is not a directory")
    if output.exists():
        entries = {path.name for path in output.iterdir()}
        if entries and COMPLETION_NAME not in entries:
            raise RuntimeError("partial label-audit output exists; refusing to overwrite it")

    result = build_audit(
        dataset_dir=dataset_root,
        archive_path=archive_path,
        protocol_path=protocol_path,
        protocol_checksum_path=protocol_checksum_path,
    )
    fingerprint = str(result["audit_fingerprint"])
    completed = verify_completion(output, expected_audit_fingerprint=fingerprint)
    if completed is not None:
        return completed

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / RESULT_NAME
    _atomic_write_new(result_path, _pretty_json_bytes(result))
    completion_body = _completion_body(result_path, fingerprint)
    completion = {
        **completion_body,
        "completion_fingerprint": canonical_fingerprint(completion_body),
    }
    _atomic_write_new(output / COMPLETION_NAME, _pretty_json_bytes(completion))
    verified = verify_completion(output, expected_audit_fingerprint=fingerprint)
    if verified is None:
        raise RuntimeError("label-audit completion disappeared during verification")
    return verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/hydraulic_uci_447")
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("data/hydraulic_uci_447.zip")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/hydraulic_ws_pac_v1_acquisition.json"),
    )
    parser.add_argument(
        "--protocol-sha256",
        type=Path,
        default=Path("protocols/hydraulic_ws_pac_v1_acquisition.sha256"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hydraulic_ws_pac_v1_label_audit"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    completion = run_audit(
        dataset_dir=args.dataset_dir,
        archive_path=args.archive,
        protocol_path=args.protocol,
        protocol_checksum_path=args.protocol_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir),
                "audit_fingerprint": completion["audit_fingerprint"],
                "sensor_values_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
