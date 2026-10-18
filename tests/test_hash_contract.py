"""Cross-boundary contract tests for future runner/backend JSON handshakes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mate_spc.hash_contract import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_json_sha256,
    json_hash_contract,
    require_canonical_json_file,
    sha256_file,
    strict_json_from_bytes,
)


DOCUMENT = {
    "candidate": "v7-template",
    "nested": {"enabled": True, "threshold": 0.05},
    "unicode": "叶",
}


def test_canonical_and_file_byte_hashes_are_named_and_not_interchangeable(
    tmp_path: Path,
) -> None:
    canonical = canonical_json_bytes(DOCUMENT)
    stored = canonical_json_file_bytes(DOCUMENT)
    assert stored == canonical + bytes([10])
    contract = json_hash_contract(DOCUMENT, file_bytes=stored)
    assert contract.canonical_json_sha256 == hashlib.sha256(canonical).hexdigest()
    assert contract.file_bytes_sha256 == hashlib.sha256(stored).hexdigest()
    assert contract.canonical_json_sha256 != contract.file_bytes_sha256
    assert contract.file_bytes_are_canonical_json_plus_lf is True

    path = tmp_path / "protocol.json"
    path.write_bytes(stored)
    parsed, inspected = require_canonical_json_file(path)
    assert parsed == DOCUMENT
    assert inspected == contract
    assert sha256_file(path) == contract.file_bytes_sha256


def test_v7_runner_backend_minimal_handshake_template() -> None:
    # Runner/file boundary: sidecar and commit bind exact stored bytes.
    stored = canonical_json_file_bytes(DOCUMENT)
    runner_contract = json_hash_contract(DOCUMENT, file_bytes=stored)
    sidecar_file_bytes_sha256 = hashlib.sha256(stored).hexdigest()
    assert sidecar_file_bytes_sha256 == runner_contract.file_bytes_sha256

    # Backend/document boundary: a parsed document can reproduce only the
    # canonical identity, which the runner compares to the same named field.
    parsed = strict_json_from_bytes(stored)
    backend_canonical_json_sha256 = canonical_json_sha256(parsed)
    assert (
        backend_canonical_json_sha256
        == runner_contract.canonical_json_sha256
    )
    assert backend_canonical_json_sha256 != sidecar_file_bytes_sha256


def test_noncanonical_file_and_invalid_json_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "missing-final-lf.json"
    path.write_bytes(canonical_json_bytes(DOCUMENT))
    with pytest.raises(ValueError, match="exactly one LF"):
        require_canonical_json_file(path)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_json_from_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        strict_json_from_bytes(b'{"value":NaN}\n')
    with pytest.raises(ValueError, match="nonfinite"):
        strict_json_from_bytes(b'{"value":1e999}\n')


def test_json_hash_contract_rejects_ambiguous_input_types() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes({1: "ambiguous"})
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_json_bytes({"value": float("inf")})
