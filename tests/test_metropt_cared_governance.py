from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mate_spc_metropt.governance import (
    AccessDeniedError,
    CALIBRATION_PASS,
    CHARTER_ID,
    CHARTER_SCHEMA,
    CHARTER_SCHEMA_VERSION,
    CONFIRMATION_PASS,
    DEVELOPMENT_PASS,
    FAILURE_INTERVALS,
    GovernanceError,
    IntegrityError,
    PARTITIONS,
    PHASE_CALIBRATION,
    PHASE_CONFIRMATION,
    PHASE_DEVELOPMENT,
    PHASE_PARTITIONS,
    PORTABILITY_DEVELOPMENT_ID,
    PORTABILITY_DEVELOPMENT_SHA256,
    PORTABILITY_V1_1_DEVELOPMENT_ID,
    PORTABILITY_V1_1_DEVELOPMENT_PARTITIONS,
    PORTABILITY_V1_1_DEVELOPMENT_SHA256,
    SOURCE_FILE_SPECS,
    SourceFileSpec,
    atomic_create_new_json,
    authorized_partitions_for_protocol,
    authenticate_completion_record,
    authenticate_fingerprinted_json,
    authenticate_json_with_sidecar,
    authenticate_phase_access_intent,
    authenticate_source_files,
    authenticate_workspace_access_ledger,
    build_workspace_access_ledger,
    canonical_fingerprint,
    canonical_json_bytes,
    claim_workspace_access_ledger,
    create_artifact_manifest,
    create_completion_record,
    create_fingerprinted_json,
    create_phase_access_intent,
    read_strict_json,
    require_completion_decision,
    validate_development_charter,
)


H_CHARTER = hashlib.sha256(b"synthetic-charter").hexdigest()
H_FINAL_PROTOCOL = hashlib.sha256(b"synthetic-final-protocol").hexdigest()
H_SENSOR = hashlib.sha256(b"synthetic-sensor-source").hexdigest()
H_SOURCES = hashlib.sha256(b"synthetic-source-manifest").hexdigest()


def _phase_fingerprint(phase: str) -> str:
    return canonical_fingerprint({"phase": phase, "frozen": True})


def test_v1_1_development_ledger_and_intent_are_exactly_least_privilege(
    tmp_path: Path,
) -> None:
    output = tmp_path / "v1_1-output"
    output.mkdir()
    ledger_path = output / "development_access_ledger.json"
    arguments = {
        "phase": PHASE_DEVELOPMENT,
        "workspace_relative_path": ledger_path.name,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "charter_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": H_SENSOR,
        "source_fingerprint": H_SOURCES,
        "phase_manifest_fingerprint": _phase_fingerprint("v1_1"),
        "output_directory": output,
    }
    body = build_workspace_access_ledger(**arguments)
    assert body["authorized_partitions"] == list(
        PORTABILITY_V1_1_DEVELOPMENT_PARTITIONS
    )
    claim = claim_workspace_access_ledger(ledger_path, **arguments)
    intent = create_phase_access_intent(
        output / "development_access_intent.json",
        claim,
        phase_manifest_fingerprint=str(arguments["phase_manifest_fingerprint"]),
        runtime_preflight_fingerprint="f" * 64,
    )
    assert intent.value["authorized_partitions"] == [
        "dev_healthy",
        "dev_fault_f1",
    ]
    assert intent.value["permissions"]["fit_permitted"] is False
    assert intent.value["permissions"]["selection_permitted"] is False
    assert intent.value["single_access"] is True
    assert intent.value["runtime_preflight_fingerprint"] == "f" * 64
    assert body["reusable"] is False
    authenticated = authenticate_workspace_access_ledger(
        ledger_path, expected_body=body
    )
    assert authenticated.document.fingerprint == claim.document.fingerprint

    assert authorized_partitions_for_protocol(
        PHASE_DEVELOPMENT, PORTABILITY_DEVELOPMENT_ID
    ) == PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    legacy_output = tmp_path / "legacy-output"
    legacy_output.mkdir()
    legacy = build_workspace_access_ledger(
        **{
            **arguments,
            "workspace_relative_path": "legacy.json",
            "protocol_id": PORTABILITY_DEVELOPMENT_ID,
            "charter_sha256": PORTABILITY_DEVELOPMENT_SHA256,
            "output_directory": legacy_output,
        }
    )
    assert legacy["authorized_partitions"] == list(
        PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    )


def _valid_charter() -> dict[str, object]:
    return {
        "schema": CHARTER_SCHEMA,
        "schema_version": CHARTER_SCHEMA_VERSION,
        "charter_id": CHARTER_ID,
        "status": "DEVELOPMENT_ONLY_SEALED_CALIBRATION_AND_CONFIRMATION",
        "method_family": {"name": "CARED-BC", "family_is_fixed": True},
        "source_binding": {
            "uci_id": 791,
            "dataset_doi": "10.24432/C5VW3R",
            **{
                spec.key: {
                    "path": spec.relative_path,
                    "size_bytes": spec.size_bytes,
                    "sha256": spec.sha256,
                }
                for spec in SOURCE_FILE_SPECS
            },
        },
        "timestamp_only_audit": {
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
        },
        "partition_convention": {
            "interval_semantics": "left_closed_right_open",
            "row_count_total": 1516948,
            "partitions_are_disjoint_and_exhaustive": True,
            "partitions": [partition.as_dict() for partition in PARTITIONS],
        },
        "official_failure_intervals": {
            "events": [
                {
                    "event_id": event.event_id,
                    "role": event.role,
                    "start": event.start,
                    "end": event.end,
                    "timestamp_only_row_count": event.row_count,
                    "fault_type": "air_leak",
                    "severity": "high_stress",
                }
                for event in FAILURE_INTERVALS
            ]
        },
        "frozen_data_transform": {
            "resampling": {
                "period": "1 minute",
                "primary_tcn_input_width": 13,
                "coverage_and_missingness_use": (
                    "eligibility and reset only; coverage or mask values are not model inputs"
                ),
            },
            "causal_window_minutes": 10,
            "forecast_horizons_minutes": [1, 3, 5],
            "direct_alarm_channels_excluded_from_primary_model": [
                "LPS",
                "Oil_level",
            ],
            "continuous_channels": [
                "TP2",
                "TP3",
                "H1",
                "DV_pressure",
                "Reservoirs",
                "Oil_temperature",
                "Motor_current",
            ],
            "control_state_channels": [
                "COMP",
                "DV_eletric",
                "Towers",
                "MPG",
                "Pressure_switch",
            ],
            "flow_channel": "Caudal_impulses",
        },
        "development_method_contract": {
            "encoder": {
                "type": "causal temporal convolutional network",
                "residual_blocks": 4,
                "channels": 64,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8],
                "latent_width": 32,
                "dropout": 0.1,
                "seeds": [17, 29, 43],
                "primary_loss": (
                    "multi-horizon Huber forecast loss with deterministic "
                    "control-state-balanced sampling"
                ),
                "auxiliary_loss": None,
                "auxiliary_loss_weight": 0.0,
                "required_training_device": "cuda",
            },
            "residual_representation": {
                "state_covariance": "fit-only Ledoit-Wolf shrinkage covariance",
                "rare_state_minimum_fit_minutes": 500,
                "rare_state_action": (
                    "pool to the nearest frequent control state by Hamming "
                    "distance with deterministic lexical tie break"
                ),
            },
            "fault_direction": {
                "source": "F1 only versus dev_healthy matched by control state",
                "location_estimator": "Huber mean",
            },
            "cusum": {"page_kappa": 0.5},
        },
        "development_power_gate": {
            "must_pass_before_calibration_unsealing": True,
            "failure_action": "ABORT_BEFORE_CALIBRATION_NUMERIC_ACCESS",
        },
        "future_calibration_contract_not_yet_authorized": {
            "data": ["calibration_a", "calibration_b"],
            "minimum_qualified_days": 39,
            "target_daily_any_alarm_probability": 0.05,
        },
        "future_confirmation_contract_not_yet_authorized": {
            "data": "confirmation",
            "fault_events": ["F2", "F3", "F4"],
            "single_numeric_access": True,
            "fit_selection_calibration_or_gate_change": False,
        },
        "governance_state": {
            "numeric_access_currently_allowed": list(
                PHASE_PARTITIONS[PHASE_DEVELOPMENT]
            ),
            "numeric_access_currently_forbidden": [
                "calibration_a",
                "embargo_a",
                "embargo_b",
                "calibration_b",
                "confirmation",
                "post_study_tail",
            ],
            "calibration_unseal_requires_final_protocol_and_power_gate": True,
            "confirmation_unseal_requires_frozen_calibration_certificate": True,
            "failed_confirmation_forbids_same_data_retuning": True,
            "this_file_is_not_a_final_analysis_protocol": True,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _ledger_arguments(
    *,
    phase: str,
    output_directory: Path,
    ledger_name: str,
    development_fingerprint: str | None = None,
    development_decision: str | None = None,
    calibration_fingerprint: str | None = None,
    calibration_decision: str | None = None,
    certificate_fingerprint: str | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "workspace_relative_path": f"ledgers/{ledger_name}",
        "protocol_id": (
            CHARTER_ID if phase == PHASE_DEVELOPMENT else "metropt-final-v1"
        ),
        "charter_sha256": H_CHARTER,
        "final_protocol_sha256": (
            None if phase == PHASE_DEVELOPMENT else H_FINAL_PROTOCOL
        ),
        "sensor_csv_sha256": H_SENSOR,
        "source_fingerprint": H_SOURCES,
        "phase_manifest_fingerprint": _phase_fingerprint(phase),
        "output_directory": output_directory,
        "development_completion_fingerprint": development_fingerprint,
        "development_completion_decision": development_decision,
        "calibration_completion_fingerprint": calibration_fingerprint,
        "calibration_completion_decision": calibration_decision,
        "calibration_certificate_fingerprint": certificate_fingerprint,
    }


def _complete_phase(
    *,
    phase: str,
    root: Path,
    ledger_arguments: dict[str, object],
    decision: str,
    extra_artifacts: list[str] | None = None,
) -> tuple[object, object, object, object, dict[str, object], list[str]]:
    output = Path(ledger_arguments["output_directory"])
    output.mkdir(exist_ok=True)
    ledger_name = Path(str(ledger_arguments["workspace_relative_path"])).name
    ledger_path = root / ledger_name
    ledger_body = build_workspace_access_ledger(**ledger_arguments)
    claim = claim_workspace_access_ledger(ledger_path, **ledger_arguments)
    intent_path = output / "access_intent.json"
    intent = create_phase_access_intent(
        intent_path,
        claim,
        phase_manifest_fingerprint=str(
            ledger_arguments["phase_manifest_fingerprint"]
        ),
    )
    artifact_names = ["metrics.json", *(extra_artifacts or [])]
    _write_json(output / "metrics.json", {"phase": phase, "synthetic": True})
    manifest_path = output / "artifact_manifest.json"
    effective_protocol_hash = (
        H_CHARTER if phase == PHASE_DEVELOPMENT else H_FINAL_PROTOCOL
    )
    manifest = create_artifact_manifest(
        manifest_path,
        output,
        phase=phase,
        protocol_id=str(ledger_arguments["protocol_id"]),
        protocol_sha256=effective_protocol_hash,
        phase_manifest_fingerprint=str(
            ledger_arguments["phase_manifest_fingerprint"]
        ),
        access_intent=intent,
        artifact_names=artifact_names,
    )
    completion_path = output / "completion.json"
    completion = create_completion_record(
        completion_path,
        claim=claim,
        access_intent=intent,
        artifact_manifest=manifest,
        decision=decision,
    )
    authenticated = authenticate_completion_record(
        completion_path,
        output_directory=output,
        ledger_path=ledger_path,
        expected_ledger_body=ledger_body,
        access_intent_path=intent_path,
        artifact_manifest_path=manifest_path,
        expected_artifact_names=artifact_names,
    )
    assert authenticated.fingerprint == completion.fingerprint
    return (
        claim,
        intent,
        manifest,
        authenticated,
        ledger_body,
        artifact_names,
    )


def test_canonical_fingerprint_is_stable_and_strict() -> None:
    first = {"z": [3, 2, 1], "a": {"beta": True, "alpha": None}}
    second = {"a": {"alpha": None, "beta": True}, "z": [3, 2, 1]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_fingerprint(first) == canonical_fingerprint(second)
    with pytest.raises(GovernanceError, match="non-finite"):
        canonical_fingerprint({"invalid": float("nan")})


def test_strict_json_sidecar_duplicate_key_and_tamper_detection(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "protocol.json"
    _write_json(document_path, {"schema": "synthetic", "value": 7})
    digest = hashlib.sha256(document_path.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "protocol.json.sha256"
    sidecar_path.write_text(f"{digest}  {document_path.name}\n", encoding="ascii")

    authenticated = authenticate_json_with_sidecar(
        document_path,
        sidecar_path,
        expected_sha256=digest,
    )
    assert authenticated.record["sha256"] == digest

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"key": 1, "key": 2}\n', encoding="utf-8")
    with pytest.raises(GovernanceError, match="duplicate JSON key"):
        read_strict_json(duplicate_path)

    document_path.write_text('{"schema": "tampered"}\n', encoding="utf-8")
    with pytest.raises(IntegrityError, match="mismatch"):
        authenticate_json_with_sidecar(document_path, sidecar_path)


def test_atomic_create_new_refuses_overwrite_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    record = atomic_create_new_json(target, {"value": 1})
    assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        atomic_create_new_json(target, {"value": 2})

    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        return
    with pytest.raises((FileExistsError, IntegrityError)):
        atomic_create_new_json(link, {"value": 3})


def test_source_binding_authenticates_three_synthetic_files(
    tmp_path: Path,
) -> None:
    payloads = {
        "archive": b"archive-bytes",
        "sensor_csv": b"timestamp,synthetic\n",
        "description_pdf": b"%PDF-synthetic",
    }
    specs: list[SourceFileSpec] = []
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.bin"
        path.write_bytes(payload)
        specs.append(
            SourceFileSpec(
                key=key,
                relative_path=path.name,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    records = authenticate_source_files(tmp_path, specs=specs)
    assert set(records) == set(payloads)

    (tmp_path / "sensor_csv.bin").write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="sensor_csv"):
        authenticate_source_files(tmp_path, specs=specs)


def test_charter_validator_freezes_method_source_splits_and_access() -> None:
    charter = _valid_charter()
    validate_development_charter(charter)

    changed = copy.deepcopy(charter)
    governance = changed["governance_state"]
    assert isinstance(governance, dict)
    governance["confirmation_unseal_requires_frozen_calibration_certificate"] = False
    with pytest.raises(GovernanceError, match="changed"):
        validate_development_charter(changed)

    changed = copy.deepcopy(charter)
    source = changed["source_binding"]
    assert isinstance(source, dict)
    sensor_csv = source["sensor_csv"]
    assert isinstance(sensor_csv, dict)
    sensor_csv["size_bytes"] = 1
    with pytest.raises(GovernanceError, match="size_bytes changed"):
        validate_development_charter(changed)


def test_three_phase_chain_is_fail_closed_and_confirmation_is_one_shot(
    tmp_path: Path,
) -> None:
    development_arguments = _ledger_arguments(
        phase=PHASE_DEVELOPMENT,
        output_directory=tmp_path / "development",
        ledger_name="development.json",
    )
    (
        _development_claim,
        _development_intent,
        _development_manifest,
        development_completion,
        _development_ledger_body,
        _development_artifacts,
    ) = _complete_phase(
        phase=PHASE_DEVELOPMENT,
        root=tmp_path,
        ledger_arguments=development_arguments,
        decision=DEVELOPMENT_PASS,
    )
    development_fingerprint = require_completion_decision(
        development_completion,
        phase=PHASE_DEVELOPMENT,
        required_decision=DEVELOPMENT_PASS,
    )

    calibration_output = tmp_path / "calibration"
    calibration_output.mkdir()
    missing_pass = _ledger_arguments(
        phase=PHASE_CALIBRATION,
        output_directory=calibration_output,
        ledger_name="calibration-invalid.json",
        development_fingerprint=development_fingerprint,
    )
    with pytest.raises(AccessDeniedError, match="development PASS"):
        build_workspace_access_ledger(**missing_pass)
    missing_protocol = dict(missing_pass)
    missing_protocol["development_completion_decision"] = DEVELOPMENT_PASS
    missing_protocol["final_protocol_sha256"] = None
    with pytest.raises(AccessDeniedError, match="final protocol"):
        build_workspace_access_ledger(**missing_protocol)
    calibration_output.rmdir()

    calibration_certificate_name = "calibration_certificate.json"
    calibration_output = tmp_path / "calibration"
    calibration_output.mkdir()
    certificate = create_fingerprinted_json(
        calibration_output / calibration_certificate_name,
        {
            "schema": "synthetic_calibration_certificate",
            "thresholds_frozen": True,
        },
        fingerprint_field="certificate_fingerprint",
    )
    assert certificate.fingerprint is not None
    calibration_arguments = _ledger_arguments(
        phase=PHASE_CALIBRATION,
        output_directory=calibration_output,
        ledger_name="calibration.json",
        development_fingerprint=development_fingerprint,
        development_decision=DEVELOPMENT_PASS,
    )
    (
        calibration_claim,
        _calibration_intent,
        _calibration_manifest,
        calibration_completion,
        calibration_ledger_body,
        _calibration_artifacts,
    ) = _complete_phase(
        phase=PHASE_CALIBRATION,
        root=tmp_path,
        ledger_arguments=calibration_arguments,
        decision=CALIBRATION_PASS,
        extra_artifacts=[calibration_certificate_name],
    )
    assert calibration_claim.document.value["reusable"] is False
    calibration_fingerprint = require_completion_decision(
        calibration_completion,
        phase=PHASE_CALIBRATION,
        required_decision=CALIBRATION_PASS,
    )
    certificate = authenticate_fingerprinted_json(
        calibration_output / calibration_certificate_name,
        fingerprint_field="certificate_fingerprint",
    )
    assert certificate.fingerprint is not None

    confirmation_output = tmp_path / "confirmation"
    confirmation_output.mkdir()
    wrong_calibration_decision = _ledger_arguments(
        phase=PHASE_CONFIRMATION,
        output_directory=confirmation_output,
        ledger_name="confirmation-invalid.json",
        development_fingerprint=development_fingerprint,
        development_decision=DEVELOPMENT_PASS,
        calibration_fingerprint=calibration_fingerprint,
        calibration_decision="ABORTED_BEFORE_CONFIRMATION_NUMERIC_ACCESS",
        certificate_fingerprint=certificate.fingerprint,
    )
    with pytest.raises(AccessDeniedError, match="calibration PASS"):
        build_workspace_access_ledger(**wrong_calibration_decision)
    missing_certificate = dict(wrong_calibration_decision)
    missing_certificate["calibration_completion_decision"] = CALIBRATION_PASS
    missing_certificate["calibration_certificate_fingerprint"] = None
    with pytest.raises(GovernanceError, match="calibration certificate"):
        build_workspace_access_ledger(**missing_certificate)
    confirmation_output.rmdir()

    confirmation_arguments = _ledger_arguments(
        phase=PHASE_CONFIRMATION,
        output_directory=confirmation_output,
        ledger_name="confirmation.json",
        development_fingerprint=development_fingerprint,
        development_decision=DEVELOPMENT_PASS,
        calibration_fingerprint=calibration_fingerprint,
        calibration_decision=CALIBRATION_PASS,
        certificate_fingerprint=certificate.fingerprint,
    )
    (
        confirmation_claim,
        confirmation_intent,
        _confirmation_manifest,
        _confirmation_completion,
        confirmation_ledger_body,
        confirmation_artifacts,
    ) = _complete_phase(
        phase=PHASE_CONFIRMATION,
        root=tmp_path,
        ledger_arguments=confirmation_arguments,
        decision=CONFIRMATION_PASS,
    )
    assert confirmation_claim.document.value["reusable"] is False

    confirmation_ledger_path = tmp_path / "confirmation.json"
    stale_claim = authenticate_workspace_access_ledger(
        confirmation_ledger_path,
        expected_body=confirmation_ledger_body,
    )
    assert stale_claim.fresh is False
    with pytest.raises(AccessDeniedError, match="audit-only"):
        create_phase_access_intent(
            confirmation_output / "second_access.json",
            stale_claim,
            phase_manifest_fingerprint=_phase_fingerprint(PHASE_CONFIRMATION),
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        claim_workspace_access_ledger(
            confirmation_ledger_path,
            **confirmation_arguments,
        )
    authenticated_intent = authenticate_phase_access_intent(
        confirmation_output / "access_intent.json",
        stale_claim,
        phase_manifest_fingerprint=_phase_fingerprint(PHASE_CONFIRMATION),
    )
    assert authenticated_intent.fingerprint == confirmation_intent.fingerprint

    (confirmation_output / "metrics.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError, match="artifact changed"):
        authenticate_completion_record(
            confirmation_output / "completion.json",
            output_directory=confirmation_output,
            ledger_path=confirmation_ledger_path,
            expected_ledger_body=confirmation_ledger_body,
            access_intent_path=confirmation_output / "access_intent.json",
            artifact_manifest_path=confirmation_output / "artifact_manifest.json",
            expected_artifact_names=confirmation_artifacts,
        )

    # The calibration chain remains independently authentic after confirmation.
    authenticate_workspace_access_ledger(
        tmp_path / "calibration.json",
        expected_body=calibration_ledger_body,
    )
