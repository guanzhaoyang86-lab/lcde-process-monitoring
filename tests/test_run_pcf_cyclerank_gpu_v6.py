"""Governance-only tests for the fixed v6 CUDA runner.

These tests never import the numeric backend and never inspect an HDF5 file.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_pcf_cyclerank_gpu_v6.py"
SPEC = importlib.util.spec_from_file_location("pcf_cyclerank_v6_governance", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _protocol_fixture() -> tuple[dict[str, Any], dict[str, str]]:
    plan = runner.development_access_plan()
    protocol = {
        "candidate": runner.METHOD_CONFIG,
        "access_plan": {"canonical_sha256": runner._hash_bytes(runner._canonical_bytes(plan))},
        "bindings": {"implementation": {}},
    }
    hashes = {"document_sha256": "a" * 64, "commit_sha256": "b" * 64}
    return protocol, hashes


def _backend_result(
    protocol_sha256: str,
    *,
    decision: str = "DEVELOPMENT_NO_GO",
    discrete_equal: bool = False,
    repeat_passes: bool = False,
) -> dict[str, Any]:
    is_go = decision == "DEVELOPMENT_GO"
    aggregate = {"synthetic_complete_five_fold_aggregate": True}
    artifact = {"path": "C:/synthetic/artifact", "size_bytes": 1, "sha256": "0" * 64}
    return {
        "schema": runner.BACKEND_RESULT_SCHEMA,
        "protocol_id": runner.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "candidate_id": runner.CANDIDATE_ID,
        "candidate_count": 1,
        "source_sha256": runner.SOURCE_SHA256,
        "source_size_bytes": runner.SOURCE_SIZE_BYTES,
        "fold_map_sha256": runner.FOLD_MAP_SHA256,
        "access_plan_sha256": runner.ACCESS_PLAN_SHA256,
        "topology": {
            "selected_run_count": 400,
            "counts_by_status": {"SimulationCompleted": 400},
            "counts_by_idv": {"1": 100, "3": 100, "9": 100, "15": 100},
            "unique_seed_count": 395,
            "duplicate_seed_value_count": 5,
            "cross_fold_seed_collision_count": 0,
            "numeric_value_access": {
                "processdata_slices": 400,
                "processdata_rows_per_slice": 500,
                "processdata_columns_half_open": [0, 53],
                "idv_init_slices": 400,
                "root_dataset_value_reads": 0,
                "processdata_values_outside_authorized_windows": 0,
            },
        },
        "environment": {
            "passed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5090",
        },
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
        "mode1_access_count": 0,
        "rieth_accessed": False,
        "rieth_access_count": 0,
        "decision": decision,
        "gates": {
            "all_pass": is_go,
            "checks": {"synthetic": is_go},
            "aggregate_metrics": aggregate,
        },
        "aggregate": aggregate,
        "selected_candidate_id": runner.CANDIDATE_ID if is_go else None,
        "folds": [{"outer_fold": fold} for fold in range(5)],
        "authoritative_cuda_repeat": {
            "scope": "per_fold_inner_complete_score_scan_replay",
            "atol": 1e-6,
            "discrete_equal": discrete_equal,
            "passes": repeat_passes,
        },
        "cross_device_report_only": {"is_go_gate": False},
        "final_refit": (
            {
                "schema": "pcf_cyclerank_final_mode2_refit_v1",
                "candidate_id": runner.CANDIDATE_ID,
                "runs_per_class": 100,
                "scan_calibrator_fit_healthy_runs": 100,
                "created_only_after_development_go": True,
                "target_domain_calibration_not_performed": True,
                "feature_state": dict(artifact),
                "model": dict(artifact),
                "scan_calibrator": dict(artifact),
                "manifest": dict(artifact),
            }
            if is_go
            else None
        ),
    }


def _backend_protocol_validation(protocol_sha256: str) -> dict[str, Any]:
    return {
        "valid": True,
        "schema": runner.PROTOCOL_SCHEMA,
        "protocol_id": runner.PROTOCOL_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "candidate_count": 1,
        "canonical_document_sha256": protocol_sha256,
        "fold_map_sha256": runner.FOLD_MAP_SHA256,
    }


def test_runner_imports_only_python_standard_library() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "datetime",
        "hashlib",
        "importlib",
        "json",
        "math",
        "os",
        "pathlib",
        "subprocess",
        "sys",
        "traceback",
        "types",
        "typing",
    }


def test_fixed_identity_endpoints_features_training_scan_and_gates() -> None:
    config = runner.METHOD_CONFIG
    assert runner.CANDIDATE_ID == "pcf-cyclerank-scan-v6-fixed"
    assert config["candidate_id"] == runner.CANDIDATE_ID
    assert config["candidate_count"] == 1
    assert config["seed"] == 20260910
    assert config["data"]["healthy_window_zero_based_half_open"] == [100, 600]
    assert config["data"]["fault_window_zero_based_half_open"] == [440, 940]
    assert config["data"]["fault_local_onset_index"] == 160
    assert config["endpoints"]["counts"] == {
        "training_all": 93,
        "training_prefix": 8,
        "training_monitor": 85,
        "ignored_fault_transition": 8,
        "mature_fault": 77,
        "dense_inference_monitor": 340,
    }
    bfeat = config["paired_domains"]["Bfeat"]
    assert bfeat["blocks"] == [572, 24, 8]
    assert len(bfeat["channel_summary_names"]) == 11
    assert len(bfeat["loop_pairs_zero_based"]) == 2
    assert len(bfeat["loop_order_each_pair"]) == 12
    assert len(bfeat["health_feature_names"]) == 8
    assert bfeat["health_subspace"] == {
        "fit_population": "60 fit-healthy normalized prefixes rows0:128 only",
        "go_final_refit_population": "100 healthy normalized prefixes rows0:128 only",
        "pca_rank": 16,
        "dpca_lag": 4,
        "dpca_rank": 16,
        "sfa_tail_count": 4,
        "var_order": 8,
        "subspace_ridge": 1e-6,
    }
    assert "5100" in bfeat["normalizer_fit_pool"]
    assert "60x85" in config["stage1"]["training_scope"]
    assert config["stage1"]["epochs"] == 120
    assert config["stage1"]["batch_size_pairs"] == 128
    assert config["stage1"]["optimizer"]["betas"] == [0.9, 0.999]
    assert config["stage2"]["epochs"] == 80
    assert config["stage2"]["optimizer"]["betas"] == [0.9, 0.999]
    assert config["stage2"]["run_batch_per_class"] == 2
    assert config["stage2"]["joint_batch_total_runs"] == 8
    assert config["stage2"]["balanced_batches_per_epoch"] == 30
    assert config["stage2"]["loss_weights"] == {
        "balanced_endpoint_bce": 0.40,
        "paired_rank": 0.30,
        "supervised_contrastive": 0.20,
        "mature_fault_attribution_ce": 0.10,
    }
    assert config["go_only_final_refit"] == {
        "trigger": "only after all five frozen development folds produce DEVELOPMENT_GO",
        "source": "the same already-authorized 400 Mode2 magnitude50 records only",
        "runs_per_class": 100,
        "total_runs": 400,
        "channel_and_health_fit": "100 healthy prefixes rows0:128 only",
        "b_standardizer_rows": 8500,
        "stage1_health_pairs": 8500,
        "stage2_runs_per_class_batch": 2,
        "stage2_balanced_batches_per_epoch": 50,
        "scan_calibrator_fit": "100 healthy x 340 dense monitor logits after same-run prefix8 median subtraction",
        "scan_calibrator_saved_before_mode1": True,
        "same_candidate_seed_epochs_optimizers_and_losses": True,
        "may_change_development_decision": False,
        "target_domain_calibration_performed": False,
        "mode1_access_authorized": False,
    }
    scan = config["scan"]
    assert scan["prefix_endpoints"] == list(range(128, 157, 4))
    assert scan["dense_monitor_endpoints"] == [160, 499, 1]
    assert scan["sustained_windows"] == [4, 8, 16, 32, 64]
    assert scan["intermittent_window"] == 16
    assert scan["intermittent_top_count"] == 4
    assert scan["drift"] == 0.25
    assert scan["raw_first_endpoint"] == 163
    assert scan["first_alarm_eligible_endpoint"] == 164
    assert scan["threshold_comparison"] == ">"
    repeat_scope = config["determinism"]["authoritative_repeat"]
    assert "inner-healthy CUDA score" in repeat_scope
    assert "training and outer scoring are not replayed" in repeat_scope
    assert "two complete deterministic CUDA executions" not in repeat_scope
    assert runner.GO_GATES["healthy"]["overall_false_alarms_max"] == 10
    assert runner.GO_GATES["healthy"]["per_fold_false_alarms_max"] == 4
    assert runner.GO_GATES["each_fault_3_9_15"]["overall_detection_rate_min"] == 0.85
    assert runner.GO_GATES["each_fault_3_9_15"]["per_fold_detection_rate_min"] == 0.70


def test_required_hash_manifest_uses_actual_v6_paths() -> None:
    required = set(runner.REQUIRED_IMPLEMENTATION_PATHS)
    assert {
        "src/mate_spc_pcfcycle/backend_v6.py",
        "src/mate_spc_pcfcycle/features.py",
        "src/mate_spc_pcfcycle/model.py",
        "src/mate_spc_pcfcycle/loss.py",
        "src/mate_spc_pcfcycle/training.py",
        "src/mate_spc_pcfcycle/scan.py",
        "src/mate_spc_pcfcycle/evaluation.py",
        "tests/test_pcfcycle_backend_v6.py",
        "tests/test_pcfcycle_features.py",
        "tests/test_pcfcycle_model_v6.py",
        "tests/test_pcfcycle_loss_v6.py",
        "tests/test_pcfcycle_training_v6.py",
        "tests/test_pcfcycle_scan.py",
        "tests/test_pcfcycle_evaluation.py",
        "tests/test_run_pcf_cyclerank_gpu_v6.py",
    } <= required
    assert "tests/test_pcfcycle_model.py" not in required
    assert "tests/test_pcfcycle_loss.py" not in required


def test_collision_safe_folds_and_exact_mode2_access_plan() -> None:
    runner._validate_folds()
    assert runner._fold_map_hash() == runner.FOLD_MAP_SHA256
    fold_map = runner._canonical_fold_map()
    assert sorted(fold_map) == list(range(1, 101))
    for component in runner.SEED_CONNECTED_COMPONENTS:
        assert len({fold_map[run] for run in component}) == 1
    for outer_fold in range(5):
        roles = runner.fold_roles(outer_fold)
        assert len(roles["fit_run_ids_per_class"]) == 60
        assert len(roles["inner_healthy_run_ids"]) == 20
        assert len(roles["outer_run_ids_per_class"]) == 20
        assert roles["inner_fault_values_authorized"] is False

    plan = runner.development_access_plan()
    assert len(plan) == 400
    assert runner._hash_bytes(runner._canonical_bytes(plan)) == runner.ACCESS_PLAN_SHA256
    assert runner.ACCESS_PLAN_SHA256 == (
        "74b19927fee7d31f33c3b64d62a0c5235b58ad41e5d2d7d28b583a6ff4e43466"
    )
    assert len({row["processdata_path"] for row in plan}) == 400
    assert {row["idv"] for row in plan} == {1, 3, 9, 15}
    assert {row["simulation_run"] for row in plan} == set(range(1, 101))
    assert all(row["mode"] == 2 and row["magnitude_percent"] == 50 for row in plan)
    assert all("Mode1" not in json.dumps(row) and "Rieth" not in json.dumps(row) for row in plan)
    assert all("*" not in row["processdata_path"] for row in plan)
    assert all(row["processdata_column_zero_based_half_open"] == [0, 53] for row in plan)
    assert all(row["idv_init_row_zero_based_half_open"] == [0, 3] for row in plan)
    assert all(row["idv_init_column_zero_based_half_open"] == [0, 28] for row in plan)
    assert all(row["group_attributes_authorized"] == ["seed"] for row in plan)
    assert all(row["root_dataset_value_reads_authorized"] == [] for row in plan)
    assert all(
        row["processdata_values_outside_row_window_authorized"] is False
        for row in plan
    )
    assert all(
        row["processdata_row_zero_based_half_open"]
        == row["window_zero_based_half_open"]
        for row in plan
    )
    assert {
        tuple(row["window_zero_based_half_open"])
        for row in plan
        if row["idv"] == 1
    } == {(100, 600)}
    assert {
        tuple(row["window_zero_based_half_open"])
        for row in plan
        if row["idv"] != 1
    } == {(440, 940)}


def test_history_ledger_preserves_mode1_lockbox_and_records_prior_consumption() -> None:
    ledger = runner._validate_ledger()
    assert ledger["v6_development_authorization"]["mode1_access_authorized"] is False
    assert ledger["v6_development_authorization"]["rieth_access_authorized"] is False
    assert (
        ledger["extended_tep_mode2_roles"]["magnitude50_idv1_3_9_15_runs1_100"]["v6_role"]
        == "sole repeatable exploratory development pool"
    )
    assert (
        ledger["extended_tep_mode1_roles"]["magnitude25_idv1_runs1_100"]["access_level"]
        == "NEVER_NUMERIC_ACCESSED"
    )
    assert (
        ledger["extended_tep_mode1_roles"]["magnitude75_idv1_3_9_15_runs1_100"]["access_level"]
        == "NEVER_NUMERIC_ACCESSED"
    )
    history = {row["method_version"]: row for row in ledger["method_history"]}
    assert history["PCLR-CUSUM-v2"]["decision"] == "DEVELOPMENT_NO_GO"
    assert history["MSCD-TCN-SPC-v3"]["decision"] == "DEVELOPMENT_NO_GO"
    assert history["CF-Max-GLR15-v4"]["decision"] == "DEVELOPMENT_NO_GO"
    assert history["CP-ETC-Scan-v5"]["decision"] == "DATA_CONTRACT_NO_GO"
    assert history["CP-ETC-Scan-v5.1"]["decision"] == "DEVELOPMENT_NO_GO"


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner._strict_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="constant forbidden"):
        runner._strict_json(nonfinite)
    with pytest.raises(ValueError, match="nonfinite"):
        runner._canonical_bytes({"a": float("inf")})


@pytest.mark.parametrize(
    ("passed", "cuda_available"),
    [(False, True), (True, False), (False, False)],
)
def test_backend_preflight_requires_both_flags(
    passed: bool, cuda_available: bool
) -> None:
    report = {
        "passed": passed,
        "cuda_available": cuda_available,
        "device_name": "NVIDIA GeForce RTX 5090",
    }
    with pytest.raises(RuntimeError, match="did not pass"):
        runner._validate_backend_preflight(report)
    good = dict(report, passed=True, cuda_available=True)
    assert runner._validate_backend_preflight(good) == good


def test_no_go_may_report_failed_cuda_repeat_but_go_may_not() -> None:
    protocol_sha256 = "a" * 64
    no_go = _backend_result(protocol_sha256)
    assert runner._validate_development_result(
        no_go, protocol_sha256=protocol_sha256
    ) == no_go

    invalid_go = _backend_result(
        protocol_sha256,
        decision="DEVELOPMENT_GO",
        discrete_equal=False,
        repeat_passes=False,
    )
    with pytest.raises(ValueError, match="GO requires"):
        runner._validate_development_result(
            invalid_go, protocol_sha256=protocol_sha256
        )

    valid_go = _backend_result(
        protocol_sha256,
        decision="DEVELOPMENT_GO",
        discrete_equal=True,
        repeat_passes=True,
    )
    assert runner._validate_development_result(
        valid_go, protocol_sha256=protocol_sha256
    ) == valid_go


def test_result_validator_rejects_mode1_and_false_selected_candidate() -> None:
    protocol_sha256 = "a" * 64
    result = _backend_result(protocol_sha256)
    result["mode1_accessed"] = True
    with pytest.raises(ValueError, match="identity/result"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)
    result = _backend_result(protocol_sha256)
    result["selected_candidate_id"] = runner.CANDIDATE_ID
    with pytest.raises(ValueError, match="selected-candidate"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_size_bytes", 1),
        ("access_plan_sha256", "0" * 64),
        ("mode1_access_count", 1),
        ("rieth_access_count", 1),
    ],
)
def test_result_validator_rejects_identity_and_access_drift(
    field: str, value: Any
) -> None:
    protocol_sha256 = "a" * 64
    result = _backend_result(protocol_sha256)
    result[field] = value
    with pytest.raises(ValueError, match="identity/result"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)


def test_result_validator_rejects_numeric_access_repeat_and_aggregate_drift() -> None:
    protocol_sha256 = "a" * 64
    result = _backend_result(protocol_sha256)
    result["topology"]["numeric_value_access"]["root_dataset_value_reads"] = 1
    with pytest.raises(ValueError, match="numeric-access topology"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)

    result = _backend_result(protocol_sha256)
    result["topology"]["numeric_value_access"]["root_dataset_value_reads"] = False
    with pytest.raises(ValueError, match="numeric-access topology"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)

    result = _backend_result(protocol_sha256)
    result["authoritative_cuda_repeat"]["scope"] = "whole_training_replay"
    with pytest.raises(ValueError, match="scope or tolerance"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)

    result = _backend_result(protocol_sha256)
    result["aggregate"] = {"changed": True}
    with pytest.raises(ValueError, match="aggregate differs"):
        runner._validate_development_result(result, protocol_sha256=protocol_sha256)


def test_result_validator_enforces_go_only_final_refit() -> None:
    protocol_sha256 = "a" * 64
    no_go = _backend_result(protocol_sha256)
    no_go["final_refit"] = {"unexpected": True}
    with pytest.raises(ValueError, match="NO_GO must not"):
        runner._validate_development_result(no_go, protocol_sha256=protocol_sha256)

    go = _backend_result(
        protocol_sha256,
        decision="DEVELOPMENT_GO",
        discrete_equal=True,
        repeat_passes=True,
    )
    go["final_refit"]["scan_calibrator_fit_healthy_runs"] = 60
    with pytest.raises(ValueError, match="final-refit metadata"):
        runner._validate_development_result(go, protocol_sha256=protocol_sha256)


def test_freeze_refuses_to_publish_until_every_required_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(runner, "REQUIRED_IMPLEMENTATION_PATHS", ("missing.py",))
    with pytest.raises(FileNotFoundError, match="missing.py"):
        runner.freeze_development()
    assert not protocol_path.exists()
    assert not runner._sidecar_path(protocol_path).exists()
    assert not runner._commit_path(protocol_path).exists()


def test_freeze_atomically_binds_all_synthetic_implementation_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_path = tmp_path / "method.md"
    ledger_path = tmp_path / "ledger.json"
    implementation_path = tmp_path / "implementation.py"
    protocol_path = tmp_path / "protocol.json"
    document_path.write_text("fixed method\n", encoding="utf-8")
    ledger_path.write_text("{}\n", encoding="utf-8")
    implementation_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "DOCUMENT_PATH", document_path)
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(runner, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(runner, "REQUIRED_IMPLEMENTATION_PATHS", ("implementation.py",))
    monkeypatch.setattr(runner, "_validate_ledger", lambda: {})

    frozen = runner.freeze_development()
    assert frozen["status"] == "FROZEN"
    protocol, hashes = runner._verify_protocol_bundle()
    assert hashes["document_sha256"] == frozen["document_sha256"]
    assert protocol["source"]["mode"] == 2
    assert protocol["source"]["magnitude_percent"] == 50
    assert protocol["source"]["idvs"] == [1, 3, 9, 15]
    assert protocol["source"]["mode1_access_authorized"] is False
    assert protocol["source"]["rieth_access_authorized"] is False
    assert protocol["folds"]["run_to_fold"] == {
        str(run): fold for run, fold in sorted(runner._canonical_fold_map().items())
    }
    assert protocol["access_plan"]["record_count"] == 400
    assert protocol["access_plan"]["selector_count"] == 400
    assert protocol["access_plan"]["canonical_sha256"] == runner.ACCESS_PLAN_SHA256
    assert protocol["numeric_backend_contract"]["cuda_repeat_atol"] == 1e-6
    assert protocol["numeric_backend_contract"]["cpu_drift_is_go_gate"] is False
    binding = protocol["bindings"]["implementation"]["implementation.py"]
    assert binding == {
        "sha256": runner._hash_file(implementation_path),
        "size_bytes": implementation_path.stat().st_size,
    }
    existing = runner.freeze_development()
    assert existing["status"] == "VERIFIED_EXISTING_FROZEN_PROTOCOL"


def test_metadata_preflight_does_not_import_numeric_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, hashes = _protocol_fixture()
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, hashes))
    monkeypatch.setattr(
        runner,
        "_source_record",
        lambda: {"sha256": runner.SOURCE_SHA256, "hdf5_opened": False},
    )
    monkeypatch.setattr(
        runner,
        "_gpu_presence_preflight",
        lambda: {"rtx5090_present": True},
    )
    monkeypatch.setattr(runner, "_run_test_suite", lambda: {"returncode": 0})
    monkeypatch.setattr(
        runner,
        "_load_backend",
        lambda: pytest.fail("metadata preflight imported the numeric backend"),
    )
    report = runner.preflight()
    assert report["passed"] is True
    assert report["backend_imported"] is False
    assert report["hdf5_opened"] is False


def test_read_only_preflight_precedes_claim_and_intent_precedes_backend_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, hashes = _protocol_fixture()
    output_root = tmp_path / "outputs"
    claim_path = output_root / "development_access.claim.json"
    intent_path = output_root / runner.DEVELOPMENT_DIRECTORY_NAME / "access_intent.json"
    events: list[str] = []

    def fake_preflight() -> dict[str, Any]:
        assert not claim_path.exists()
        assert not output_root.exists()
        events.append("preflight")
        return {
            "passed": True,
            "source": {"sha256": runner.SOURCE_SHA256, "hdf5_opened": False},
        }

    class FakeBackend:
        @staticmethod
        def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
            events.append("validate_protocol")
            assert value is protocol
            return _backend_protocol_validation(hashes["document_sha256"])

        @staticmethod
        def cuda_preflight() -> dict[str, Any]:
            events.append("cuda_preflight")
            return {
                "passed": True,
                "cuda_available": True,
                "device_name": "NVIDIA GeForce RTX 5090",
            }

        @staticmethod
        def run_development(**kwargs: Any) -> dict[str, Any]:
            events.append("run_development")
            assert kwargs["device"] == "cuda"
            assert kwargs["fold_map"] == runner._canonical_fold_map()
            return _backend_result(hashes["document_sha256"])

    def fake_load_backend() -> Any:
        events.append("load_backend")
        assert claim_path.is_file()
        assert intent_path.is_file()
        assert runner._sidecar_path(intent_path).is_file()
        assert runner._commit_path(intent_path).is_file()
        return FakeBackend

    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, hashes))
    monkeypatch.setattr(runner, "preflight", fake_preflight)
    monkeypatch.setattr(runner, "_prepare_backend_environment", lambda: None)
    monkeypatch.setattr(runner, "_load_backend", fake_load_backend)

    completion = runner.run_development(output_root=output_root)
    assert completion["decision"] == "DEVELOPMENT_NO_GO"
    assert events == [
        "preflight",
        "load_backend",
        "validate_protocol",
        "cuda_preflight",
        "run_development",
    ]
    assert claim_path.is_file()
    assert intent_path.is_file()
    completed = json.loads(
        (output_root / runner.DEVELOPMENT_DIRECTORY_NAME / "completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert completed["mode1_accessed"] is False
    assert completed["decision"] == "DEVELOPMENT_NO_GO"


def test_preflight_failure_does_not_burn_global_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, hashes = _protocol_fixture()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, hashes))

    def fail_preflight() -> dict[str, Any]:
        raise RuntimeError("synthetic preflight failure")

    monkeypatch.setattr(runner, "preflight", fail_preflight)
    with pytest.raises(RuntimeError, match="synthetic preflight"):
        runner.run_development(output_root=output_root)
    assert not (output_root / "development_access.claim.json").exists()
    assert not (output_root / runner.DEVELOPMENT_DIRECTORY_NAME).exists()


def test_backend_failure_after_intent_is_immutably_recorded_as_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, hashes = _protocol_fixture()
    output_root = tmp_path / "outputs"

    class FailingBackend:
        @staticmethod
        def validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
            return _backend_protocol_validation(hashes["document_sha256"])

        @staticmethod
        def cuda_preflight() -> dict[str, Any]:
            return {
                "passed": True,
                "cuda_available": True,
                "device_name": "NVIDIA GeForce RTX 5090",
            }

        @staticmethod
        def run_development(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("synthetic failure after intent")

    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, hashes))
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda: {
            "passed": True,
            "source": {"sha256": runner.SOURCE_SHA256, "hdf5_opened": False},
        },
    )
    monkeypatch.setattr(runner, "_prepare_backend_environment", lambda: None)
    monkeypatch.setattr(runner, "_load_backend", lambda: FailingBackend)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_development(output_root=output_root)
    directory = output_root / runner.DEVELOPMENT_DIRECTORY_NAME
    assert (directory / "access_intent.json").is_file()
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    assert completion["decision"] == "ABORTED_AFTER_NUMERIC_ACCESS"
    assert completion["access_intent_committed"] is True
    with pytest.raises(FileExistsError, match="prior v6 global claim"):
        runner.run_development(output_root=output_root)
