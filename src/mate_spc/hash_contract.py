"""Unambiguous SHA-256 identities for canonical JSON and JSON files.

Protocol handshakes have two legitimate, distinct byte domains:

``canonical_json_sha256``
    SHA-256 of the parsed JSON value serialized with the settings in
    :func:`canonical_json_bytes`, without a trailing line terminator.  A
    backend that receives only a parsed document can reproduce this identity.

``file_bytes_sha256``
    SHA-256 of the exact bytes stored on disk, including the required final
    LF.  The writer, sidecar, commit record, and file verifier use this value.

Runner/backend protocols must never compare one field with the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


JSON_FILE_TERMINATOR = b"\n"


def _json_value(value: Any, *, location: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: nonfinite JSON number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location}: JSON object keys must be strings")
            result[key] = _json_value(item, location=f"{location}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{location}: unsupported JSON value {type(value).__name__}")


def canonical_json_bytes(document: Any) -> bytes:
    """Serialize one strict JSON value to the shared canonical byte domain."""

    return json.dumps(
        _json_value(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_file_bytes(document: Any) -> bytes:
    """Return canonical JSON followed by the single required LF byte."""

    return canonical_json_bytes(document) + JSON_FILE_TERMINATOR


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Hash an explicit byte domain."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_bytes requires a bytes-like value")
    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without parsing or normalization."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(document: Any) -> str:
    """Hash the canonical parsed-document representation (no final LF)."""

    return sha256_bytes(canonical_json_bytes(document))


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant forbidden: {value}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def strict_json_from_bytes(value: bytes | bytearray | memoryview) -> Any:
    """Parse UTF-8 JSON while rejecting duplicates and nonfinite constants."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("strict_json_from_bytes requires a bytes-like value")
    try:
        text = bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("JSON file is not strict UTF-8") from error
    parsed = json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_no_duplicate_keys,
    )
    return _json_value(parsed)


@dataclass(frozen=True)
class JsonHashContract:
    """Both non-interchangeable identities for one JSON artifact."""

    canonical_json_sha256: str
    file_bytes_sha256: str
    canonical_json_size_bytes: int
    file_size_bytes: int
    file_bytes_are_canonical_json_plus_lf: bool

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": "mate_spc_json_hash_contract_v1",
            "canonical_json_sha256": self.canonical_json_sha256,
            "file_bytes_sha256": self.file_bytes_sha256,
            "canonical_json_size_bytes": self.canonical_json_size_bytes,
            "file_size_bytes": self.file_size_bytes,
            "file_bytes_are_canonical_json_plus_lf": (
                self.file_bytes_are_canonical_json_plus_lf
            ),
        }


def json_hash_contract(
    document: Any, *, file_bytes: bytes | bytearray | memoryview | None = None
) -> JsonHashContract:
    """Build both hashes; default file bytes are canonical JSON plus one LF."""

    canonical = canonical_json_bytes(document)
    stored = (
        canonical + JSON_FILE_TERMINATOR
        if file_bytes is None
        else bytes(file_bytes)
    )
    return JsonHashContract(
        canonical_json_sha256=sha256_bytes(canonical),
        file_bytes_sha256=sha256_bytes(stored),
        canonical_json_size_bytes=len(canonical),
        file_size_bytes=len(stored),
        file_bytes_are_canonical_json_plus_lf=(
            stored == canonical + JSON_FILE_TERMINATOR
        ),
    )


def inspect_json_file(path: str | Path) -> tuple[Any, JsonHashContract]:
    """Parse one file and return its parsed document plus both identities."""

    raw = Path(path).read_bytes()
    document = strict_json_from_bytes(raw)
    return document, json_hash_contract(document, file_bytes=raw)


def require_canonical_json_file(path: str | Path) -> tuple[Any, JsonHashContract]:
    """Fail closed unless the file is canonical JSON plus exactly one final LF."""

    document, contract = inspect_json_file(path)
    if not contract.file_bytes_are_canonical_json_plus_lf:
        raise ValueError("JSON file bytes must be canonical JSON plus exactly one LF")
    return document, contract


__all__ = [
    "JSON_FILE_TERMINATOR",
    "JsonHashContract",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "canonical_json_sha256",
    "inspect_json_file",
    "json_hash_contract",
    "require_canonical_json_file",
    "sha256_bytes",
    "sha256_file",
    "strict_json_from_bytes",
]
