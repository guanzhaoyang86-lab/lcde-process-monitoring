from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mate_spc_cris_v9.monitor import (
    MonitorConfig,
    PointCalibrator,
    apply_threshold,
    calibrate_trajectory_threshold,
    scan_innovations,
)
from mate_spc_cris_v9.study_io import (
    build_freeze_manifest,
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_json_sha256,
    create_confirmation_global_claim,
    load_point_calibrator_npz,
    load_trajectory_threshold_npz,
    read_canonical_json,
    save_point_calibrator_npz,
    save_trajectory_threshold_npz,
    sha256_file,
    verify_freeze_manifest,
    write_freeze_manifest,
    write_json_atomic,
    write_json_exclusive,
)


def _producer_calibrator() -> PointCalibrator:
    rng = np.random.default_rng(20260909)
    healthy = rng.normal(size=(5, 96, 3, 6)).astype(np.float64)
    config = MonitorConfig(top_k=2, moment_weights=(0.5, 0.25, 0.25))
    return PointCalibrator.fit(healthy, config)


def _npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def test_canonical_json_atomic_hash_and_strict_reader(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "record.json"
    document = {"z": [3, "μ"], "a": {"ok": True}}
    record = write_json_atomic(path, document)

    expected = b'{"a":{"ok":true},"z":[3,"\xce\xbc"]}'
    assert canonical_json_bytes(document) == expected
    assert canonical_json_file_bytes(document) == expected + b"\n"
    assert path.read_bytes() == expected + b"\n"
    assert read_canonical_json(path) == document
    assert record["file_bytes_sha256"] == sha256_file(path)
    assert canonical_json_sha256(document) != sha256_file(path)

    replacement = {"new": 7}
    write_json_atomic(path, replacement)
    assert read_canonical_json(path) == replacement
    path.write_bytes(canonical_json_bytes(replacement))
    with pytest.raises(ValueError, match="canonical JSON plus exactly one LF"):
        read_canonical_json(path)


def test_canonical_json_rejects_nonfinite_and_exclusive_is_no_clobber(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_json_bytes({"bad": np.nan})

    path = tmp_path / "global.claim.json"
    first = {"attempt": 1}
    write_json_exclusive(path, first)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"attempt": 2})
    assert path.read_bytes() == original
    assert read_canonical_json(path) == first


def test_real_point_calibrator_producer_consumer_roundtrip(tmp_path: Path) -> None:
    producer = _producer_calibrator()
    path = save_point_calibrator_npz(producer, tmp_path / "point-calibrator.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema",
            "schema_version",
            "config_json",
            "components_json",
            "horizon_count",
            "dimension_count",
            "sample_offsets",
            "sorted_samples_flat",
        }
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        assert archive["sample_offsets"].dtype == np.dtype(np.int64)
        assert archive["sorted_samples_flat"].dtype == np.dtype(np.float64)

    consumer = load_point_calibrator_npz(path)
    assert consumer.config == producer.config
    assert consumer.components == producer.components
    assert consumer.horizon_count == producer.horizon_count
    assert consumer.dimension_count == producer.dimension_count
    for expected, observed in zip(
        producer.sorted_samples, consumer.sorted_samples, strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
        assert not observed.flags.writeable

    probe = np.random.default_rng(77).normal(size=(120, 3, 6)).astype(np.float64)
    producer_scan = scan_innovations(probe, producer)
    consumer_scan = scan_innovations(probe, consumer)
    np.testing.assert_array_equal(consumer_scan.score, producer_scan.score)
    np.testing.assert_array_equal(consumer_scan.p_values, producer_scan.p_values)
    np.testing.assert_array_equal(consumer_scan.e_values, producer_scan.e_values)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("sample_offsets", lambda value: value.astype(np.int32), "int64"),
        ("sorted_samples_flat", lambda value: value.astype(np.float32), "float64"),
        ("schema_version", lambda value: value.astype(np.int32), "int64"),
        ("schema", lambda value: np.asarray(9, dtype=np.int64), "Unicode"),
    ),
)
def test_point_loader_rejects_wrong_array_dtypes(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    valid = save_point_calibrator_npz(_producer_calibrator(), tmp_path / "valid.npz")
    payload = _npz_payload(valid)
    payload[field] = replacement(payload[field])  # type: ignore[operator]
    damaged = tmp_path / f"bad-{field}.npz"
    np.savez(damaged, **payload)
    with pytest.raises(ValueError, match=message):
        load_point_calibrator_npz(damaged)


def test_point_loader_rejects_unexpected_key_config_and_component_identity(
    tmp_path: Path,
) -> None:
    valid = save_point_calibrator_npz(_producer_calibrator(), tmp_path / "valid.npz")
    original = _npz_payload(valid)

    extra = dict(original)
    extra["surprise"] = np.asarray(1, dtype=np.int64)
    extra_path = tmp_path / "extra.npz"
    np.savez(extra_path, **extra)
    with pytest.raises(ValueError, match="unexpected=.*surprise"):
        load_point_calibrator_npz(extra_path)

    bad_config = dict(original)
    config = json.loads(str(bad_config["config_json"].item()))
    config["rolling_scales"] = [4, 16, 63]
    bad_config["config_json"] = np.asarray(
        canonical_json_bytes(config).decode("utf-8")
    )
    config_path = tmp_path / "bad-config.npz"
    np.savez(config_path, **bad_config)
    with pytest.raises(ValueError, match="rolling_scales"):
        load_point_calibrator_npz(config_path)

    bad_component = dict(original)
    components = json.loads(str(bad_component["components_json"].item()))
    components[0]["name"] = "forged/component/name"
    bad_component["components_json"] = np.asarray(
        canonical_json_bytes(components).decode("utf-8")
    )
    component_path = tmp_path / "bad-component.npz"
    np.savez(component_path, **bad_component)
    with pytest.raises(ValueError, match="name does not match"):
        load_point_calibrator_npz(component_path)


def test_point_loader_never_enables_pickle(tmp_path: Path) -> None:
    valid = save_point_calibrator_npz(_producer_calibrator(), tmp_path / "valid.npz")
    payload = _npz_payload(valid)
    payload["sorted_samples_flat"] = np.asarray([object()], dtype=object)
    damaged = tmp_path / "object.npz"
    np.savez(damaged, **payload)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_point_calibrator_npz(damaged)


def test_real_threshold_producer_consumer_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(314)
    score_paths = [rng.normal(size=80).astype(np.float64) for _ in range(20)]
    producer = calibrate_trajectory_threshold(score_paths, alpha=0.10)
    path = save_trajectory_threshold_npz(producer, tmp_path / "threshold.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema",
            "schema_version",
            "threshold_json",
            "calibration_maxima",
        }
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        assert archive["calibration_maxima"].dtype == np.dtype(np.float64)

    consumer = load_trajectory_threshold_npz(path)
    assert consumer.value == producer.value
    assert consumer.alpha == producer.alpha
    assert consumer.calibration_run_count == producer.calibration_run_count
    assert consumer.ascending_rank == producer.ascending_rank
    assert consumer.comparison == "strictly_greater_than"
    np.testing.assert_array_equal(consumer.calibration_maxima, producer.calibration_maxima)
    probe = np.asarray([producer.value, producer.value + 1.0, np.nan])
    np.testing.assert_array_equal(
        apply_threshold(probe, consumer), apply_threshold(probe, producer)
    )


def test_threshold_loader_rejects_unexpected_key_dtype_and_metadata(
    tmp_path: Path,
) -> None:
    score_paths = np.arange(20 * 8, dtype=np.float64).reshape(20, 8)
    threshold = calibrate_trajectory_threshold(score_paths, alpha=0.10)
    valid = save_trajectory_threshold_npz(threshold, tmp_path / "valid-threshold.npz")
    original = _npz_payload(valid)

    extra = dict(original)
    extra["unexpected"] = np.asarray(1)
    extra_path = tmp_path / "threshold-extra.npz"
    np.savez(extra_path, **extra)
    with pytest.raises(ValueError, match="unexpected=.*unexpected"):
        load_trajectory_threshold_npz(extra_path)

    bad_dtype = dict(original)
    bad_dtype["calibration_maxima"] = original["calibration_maxima"].astype(np.float32)
    dtype_path = tmp_path / "threshold-float32.npz"
    np.savez(dtype_path, **bad_dtype)
    with pytest.raises(ValueError, match="float64"):
        load_trajectory_threshold_npz(dtype_path)

    bad_metadata = dict(original)
    document = json.loads(str(original["threshold_json"].item()))
    document["comparison"] = "greater_than_or_equal"
    bad_metadata["threshold_json"] = np.asarray(
        canonical_json_bytes(document).decode("utf-8")
    )
    metadata_path = tmp_path / "threshold-comparison.npz"
    np.savez(metadata_path, **bad_metadata)
    with pytest.raises(ValueError, match="strictly_greater_than"):
        load_trajectory_threshold_npz(metadata_path)


def test_freeze_manifest_binds_and_rejects_any_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    artifact = tmp_path / "artifact.bin"
    source.write_bytes(b"source-v1\n")
    artifact.write_bytes(b"artifact-v1\n")
    source_hash = sha256_file(source)
    artifact_hash = sha256_file(artifact)

    manifest = build_freeze_manifest(
        root=tmp_path,
        sources={"source": source},
        artifacts={"artifact": artifact},
        expected_source_sha256={"source": source_hash},
        expected_artifact_sha256={"artifact": artifact_hash},
        metadata={"candidate_id": "cris-k2-equal"},
    )
    verified = verify_freeze_manifest(manifest, root=tmp_path)
    assert verified["source_count"] == 1
    assert verified["artifact_count"] == 1
    assert verified["manifest_canonical_json_sha256"] == canonical_json_sha256(manifest)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_freeze_manifest(
            root=tmp_path,
            sources={"source": source},
            artifacts={"artifact": artifact},
            expected_source_sha256={"source": "0" * 64},
        )

    # Preserve size so this specifically exercises the content hash check.
    artifact.write_bytes(b"artifact-v2\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_freeze_manifest(manifest, root=tmp_path)


def test_freeze_manifest_file_is_canonical_exclusive_and_verified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    artifact = tmp_path / "artifact.bin"
    source.write_bytes(b"source\n")
    artifact.write_bytes(b"artifact\n")
    manifest_path = tmp_path / "freeze-manifest.json"
    write_freeze_manifest(
        manifest_path,
        root=tmp_path,
        sources={"source": source},
        artifacts={"artifact": artifact},
    )
    document = read_canonical_json(manifest_path)
    verify_freeze_manifest(document, root=tmp_path)
    assert manifest_path.read_bytes() == canonical_json_file_bytes(document)

    with pytest.raises(FileExistsError):
        write_freeze_manifest(
            manifest_path,
            root=tmp_path,
            sources={"source": source},
            artifacts={"artifact": artifact},
        )


def test_confirmation_global_claim_is_verified_and_one_shot(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    source = tmp_path / "source.py"
    artifact = tmp_path / "artifact.bin"
    freeze = tmp_path / "freeze.json"
    claim = tmp_path / "confirmation-global.claim.json"
    write_json_atomic(protocol, {"protocol_id": "cris-v9-test"})
    source.write_bytes(b"source\n")
    artifact.write_bytes(b"artifact\n")
    write_freeze_manifest(
        freeze,
        root=tmp_path,
        sources={"source": source},
        artifacts={"artifact": artifact},
    )

    record = create_confirmation_global_claim(
        claim,
        root=tmp_path,
        candidate_id="cris-k2-equal",
        protocol_path=protocol,
        freeze_manifest_path=freeze,
        expected_protocol_sha256=sha256_file(protocol),
        expected_freeze_manifest_sha256=sha256_file(freeze),
        metadata={"seed_first": 900001, "seed_last": 950200},
    )
    document = read_canonical_json(claim)
    assert record["file_bytes_sha256"] == sha256_file(claim)
    assert document["candidate_id"] == "cris-k2-equal"
    assert document["protocol"]["file_bytes_sha256"] == sha256_file(protocol)
    assert document["freeze_manifest"]["file_bytes_sha256"] == sha256_file(freeze)

    with pytest.raises(FileExistsError):
        create_confirmation_global_claim(
            claim,
            root=tmp_path,
            candidate_id="cris-k2-equal",
            protocol_path=protocol,
            freeze_manifest_path=freeze,
        )

    never_created = tmp_path / "bad-hash.claim.json"
    with pytest.raises(ValueError, match="protocol SHA-256 mismatch"):
        create_confirmation_global_claim(
            never_created,
            root=tmp_path,
            candidate_id="cris-k2-equal",
            protocol_path=protocol,
            freeze_manifest_path=freeze,
            expected_protocol_sha256="0" * 64,
        )
    assert not never_created.exists()
