"""Read-free governance tests for the fixed FARM-RankCUSUM v7 runner."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_farm_rankcusum_gpu_v7.py"
SPEC = importlib.util.spec_from_file_location("farm_rankcusum_v7_governance", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _protocol_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = {
        "schema": runner.PROTOCOL_SCHEMA,
        "protocol_id": runner.PROTOCOL_ID,
        "candidate": runner.METHOD_CONFIG,
        "bindings": {"implementation": {}},
    }
    identities = {
        "file_bytes_sha256": "a" * 64,
        "canonical_json_sha256": "b" * 64,
        "commit_file_bytes_sha256": "c" * 64,
    }
    return protocol, identities


def _backend_result(
    protocol_canonical_json_sha256: str,
    *,
    decision: str = "DEVELOPMENT_NO_GO",
    repeat_passes: bool = False,
    discrete_equal: bool = False,
) -> dict[str, Any]:
    is_go = decision == "DEVELOPMENT_GO"
    aggregate = {"synthetic_complete_five_fold_aggregate": True}
    access_state = {
        "schema": "farm_rankcusum_v7_backend_access_state",
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "hdf5_open_started": True,
        "hdf5_open_completed": True,
        "processdata_slices_read": 400,
        "idv_init_slices_read": 400,
        "numeric_slices_read": 800,
        "training_or_fit_started": True,
        "backend_completed": True,
        "mode1_opens": 0,
    }
    return {
        "schema": runner.BACKEND_RESULT_SCHEMA,
        "protocol_id": runner.PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": runner.CANDIDATE_ID,
        "candidate_count": 1,
        "source_file_bytes_sha256": runner.SOURCE_SHA256,
        "source_size_bytes": runner.SOURCE_SIZE_BYTES,
        "fold_map_canonical_json_sha256": runner.FOLD_MAP_SHA256,
        "access_plan_canonical_json_sha256": runner.ACCESS_PLAN_SHA256,
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
        "backend_access_state": {
            "final_state": access_state,
            "artifact": {
                "path": "C:/synthetic/backend_access_state.json",
                "file_bytes_sha256": "1" * 64,
                "size_bytes": 1,
            },
        },
        "environment": {
            "passed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
        },
        "mode1_accessed": False,
        "mode1_opens": 0,
        "rieth_accessed": False,
        "rieth_opens": 0,
        "fault_fit_run_count": 0,
        "fault_threshold_run_count": 0,
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
            "scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
            "atol": 1e-6,
            "maximum_absolute_error": 0.0 if repeat_passes else None,
            "discrete_equal": discrete_equal,
            "passes": repeat_passes,
        },
        "cross_device_report_only": {"is_go_gate": False},
        "development_decision_before_final_refit": {
            "path": "C:/synthetic/development_decision_before_final_refit.json",
            "file_bytes_sha256": "3" * 64,
            "size_bytes": 1,
        },
        "final_refit": (
            {
                "schema": "farm_rankcusum_v7_final_mode2_refit",
                "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
                "candidate_id": runner.CANDIDATE_ID,
                "fit_healthy_run_count": 100,
                "fit_fault_run_count": 0,
                "reference_shape": [46, 50000],
                "created_only_after_development_go": True,
                "target_domain_threshold_not_fitted": True,
                "reference": {
                    "path": "C:/synthetic/healthy_reference.npz",
                    "file_bytes_sha256": "0" * 64,
                    "size_bytes": 1,
                },
                "manifest": {
                    "path": "C:/synthetic/manifest.json",
                    "file_bytes_sha256": "2" * 64,
                    "size_bytes": 1,
                },
            }
            if is_go
            else None
        ),
    }


def test_runner_source_has_no_eager_numeric_or_hdf5_import() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    assert not ({"h5py", "numpy", "torch", "pandas"} & imported)


def test_hash_contract_loader_avoids_package_and_numeric_imports() -> None:
    code = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER_PATH)!r})
spec = importlib.util.spec_from_file_location('isolated_farm_v7_runner', path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._canonical_bytes({{'x': 1}}) == b'{{\"x\":1}}'
for forbidden in ('numpy', 'torch', 'h5py'):
    assert forbidden not in sys.modules, sorted(
        name for name in sys.modules if name == forbidden or name.startswith(forbidden + '.')
    )
assert 'mate_spc' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_transitive_backend_imports_are_hash_bound() -> None:
    required = {
        "docs/gpu_minirocket_logodds_v8_contingency.md",
        "docs/gpu_minirocket_logodds_v8_contingency.json",
        "src/mate_spc/__init__.py",
        "src/mate_spc/model.py",
        "src/mate_spc/datasets.py",
        "src/mate_spc/missingness.py",
        "src/mate_spc/monitor.py",
        "src/mate_spc/state_space.py",
        "src/mate_spc/utils.py",
        "src/mate_spc_cpetc/__init__.py",
        "src/mate_spc_cpetc/data_v5.py",
        "src/mate_spc_cpetc/monitor.py",
        "src/mate_spc_pclr/__init__.py",
        "src/mate_spc_pclr/data.py",
    }
    assert required <= set(runner.BASE_IMPLEMENTATION_PATHS)


def test_v8_contingency_is_pre_result_non_authorizing_and_fold_identical() -> None:
    charter = runner._validate_contingency_charter()
    assert charter["created_before_formal_v7_data_access"] is True
    assert charter["activation"]["allowed_trigger"] == {
        "candidate_id": runner.CANDIDATE_ID,
        "result": "DEVELOPMENT_NO_GO",
    }
    assert charter["activation"]["requires_separate_pre_access_protocol"] is True
    assert charter["authorization"]["allows_h5_access"] is False
    assert charter["authorization"]["allows_real_experiment"] is False
    assert charter["fold_assignment"]["run_to_fold_sha256"] == runner.FOLD_MAP_SHA256


def test_fixed_method_identity_channels_scan_and_gates() -> None:
    config = runner.METHOD_CONFIG
    assert runner.CANDIDATE_ID == "farm-rankcusum-v7-fixed"
    assert runner.PROTOCOL_SCHEMA == "farm_rankcusum_v7_preregistered_development_protocol"
    assert config["channels"]["zero_based_indices"] == [*range(41), 46, 47, 48, 50, 51]
    assert config["channels"]["count"] == 46
    assert config["reference"]["allowed_healthy_run_counts"] == [60, 100]
    assert config["reference"]["fault_values_used"] is False
    assert config["monitor"]["k"] == 1.3
    assert config["monitor"]["top_r"] == 4
    assert config["monitor"]["dense_endpoints_inclusive"] == [160, 499]
    assert config["monitor"]["alarm_comparison"] == ">"
    assert config["inner_threshold"]["healthy_runs"] == 20
    assert config["inner_threshold"]["minimum_exact_distinct_maxima"] == 18
    assert runner.GO_GATES["healthy"]["overall_false_alarms_max"] == 10
    assert runner.GO_GATES["each_fault_3_9_15"]["overall_detection_rate_min"] == 0.85


def test_fold_map_and_exact_access_plan_are_unchanged() -> None:
    runner._validate_folds()
    plan = runner.development_access_plan()
    assert len(plan) == 400
    assert runner._hash_bytes(runner._canonical_bytes(plan)) == runner.ACCESS_PLAN_SHA256
    assert runner._fold_map_hash() == runner.FOLD_MAP_SHA256
    assert sum(item["role"] == "healthy_control" for item in plan) == 100
    assert all(item["window_zero_based_half_open"] == [100, 600] for item in plan if item["idv"] == 1)
    assert all(item["window_zero_based_half_open"] == [440, 940] for item in plan if item["idv"] != 1)
    for fold in range(5):
        roles = runner.fold_roles(fold)
        assert len(roles["fit_healthy_run_ids"]) == 60
        assert len(roles["inner_healthy_run_ids"]) == 20
        assert len(roles["outer_run_ids_per_class"]) == 20
        assert roles["inner_fault_values_authorized"] is False


def test_commit_keeps_file_and_canonical_hash_domains_separate(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    identities = runner._commit_json(path, {"schema": "synthetic", "叶": True}, commit_schema="synthetic_commit")
    assert identities["file_bytes_sha256"] != identities["canonical_json_sha256"]
    document, verified = runner._verify_committed_json(path, commit_schema="synthetic_commit")
    assert document == {"schema": "synthetic", "叶": True}
    assert verified["file_bytes_sha256"] == identities["file_bytes_sha256"]
    assert verified["canonical_json_sha256"] == identities["canonical_json_sha256"]
    assert path.read_bytes().endswith(b"\n")


def test_protocol_document_names_hash_domains_and_preclaim_controls_exactly() -> None:
    document = runner._protocol_document({})
    assert document["hash_contract"] == {
        "canonicalization": "UTF-8 sorted-key compact JSON, no NaN, no terminal LF",
        "persisted_file": "canonical JSON plus exactly one terminal LF",
        "sidecar_and_commit_identity": "file_bytes_sha256",
        "runner_backend_identity": "canonical_json_sha256",
        "cross_domain_comparison_forbidden": True,
    }
    governance = document["governance"]
    assert governance["preclaim_handshake_validates_persisted_protocol"] is True
    assert governance["preclaim_h5py_import_blocker_required"] is True
    assert governance["handshaken_backend_instance_reused_after_intent"] is True
    assert governance["mode1_access_authorized"] is False
    numeric = document["numeric_backend_contract"]
    assert numeric["validation_hash_domain"] == "canonical_json_without_file_terminator"
    assert numeric["validation_must_not_compare_file_bytes_sha256"] is True


def test_production_handshake_uses_only_canonical_identity_and_stays_h5_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_hash = "d" * 64
    events: list[str] = []

    class Backend:
        @staticmethod
        def validate_protocol(protocol: Any, *, expected_canonical_json_sha256: str) -> dict[str, Any]:
            events.append("validate")
            assert expected_canonical_json_sha256 == canonical_hash
            return {
                "valid": True,
                "schema": runner.PROTOCOL_SCHEMA,
                "protocol_id": runner.PROTOCOL_ID,
                "candidate_id": runner.CANDIDATE_ID,
                "candidate_count": 1,
                "protocol_canonical_json_sha256": canonical_hash,
                "fold_map_canonical_json_sha256": runner.FOLD_MAP_SHA256,
                "access_plan_canonical_json_sha256": runner.ACCESS_PLAN_SHA256,
                "hash_domain": "canonical_json_without_file_terminator",
                "file_bytes_sha256_compared": False,
            }

        @staticmethod
        def cuda_preflight() -> dict[str, Any]:
            events.append("cuda")
            return {
                "passed": True,
                "cuda_available": True,
                "device_name": "NVIDIA RTX 5090",
            }

    monkeypatch.delitem(sys.modules, "h5py", raising=False)
    monkeypatch.setattr(runner, "_load_backend", lambda: Backend)
    _backend, report = runner._production_handshake({}, canonical_json_sha256=canonical_hash)
    assert events == ["validate", "cuda"]
    assert report["h5py_module_loaded"] is False
    assert report["h5py_import_attempted"] is False
    assert "h5py" not in sys.modules


def test_production_handshake_blocks_even_a_caught_h5py_import_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_hash = "3" * 64

    class Backend:
        @staticmethod
        def validate_protocol(
            protocol: Any, *, expected_canonical_json_sha256: str
        ) -> dict[str, Any]:
            del protocol, expected_canonical_json_sha256
            try:
                __import__("h5py")
            except RuntimeError:
                pass
            return {
                "valid": True,
                "schema": runner.PROTOCOL_SCHEMA,
                "protocol_id": runner.PROTOCOL_ID,
                "candidate_id": runner.CANDIDATE_ID,
                "candidate_count": 1,
                "protocol_canonical_json_sha256": canonical_hash,
                "fold_map_canonical_json_sha256": runner.FOLD_MAP_SHA256,
                "access_plan_canonical_json_sha256": runner.ACCESS_PLAN_SHA256,
                "hash_domain": "canonical_json_without_file_terminator",
                "file_bytes_sha256_compared": False,
            }

        @staticmethod
        def cuda_preflight() -> dict[str, Any]:
            return {
                "passed": True,
                "cuda_available": True,
                "device_name": "NVIDIA RTX 5090",
            }

    for name in list(sys.modules):
        if name == "h5py" or name.startswith("h5py."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(runner, "_load_backend", lambda: Backend)
    with pytest.raises(RuntimeError, match="attempted to load h5py"):
        runner._production_handshake({}, canonical_json_sha256=canonical_hash)
    assert not any(isinstance(item, runner._H5pyImportBlocker) for item in sys.meta_path)
    assert not runner._loaded_h5py_modules()


def test_result_validator_enforces_no_go_and_go_only_refit() -> None:
    digest = "e" * 64
    no_go = _backend_result(digest)
    assert runner._validate_development_result(
        no_go, protocol_canonical_json_sha256=digest
    ) == no_go
    go = _backend_result(
        digest,
        decision="DEVELOPMENT_GO",
        repeat_passes=True,
        discrete_equal=True,
    )
    assert runner._validate_development_result(
        go, protocol_canonical_json_sha256=digest
    ) == go
    no_go["final_refit"] = go["final_refit"]
    with pytest.raises(ValueError, match="NO_GO"):
        runner._validate_development_result(
            no_go, protocol_canonical_json_sha256=digest
        )


def test_result_validator_rejects_hash_domain_drift() -> None:
    digest = "f" * 64
    result = _backend_result(digest)
    result["protocol_canonical_json_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        runner._validate_development_result(
            result, protocol_canonical_json_sha256=digest
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("rieth_accessed",), True),
        (("fault_fit_run_count",), 1),
        (("candidate_count",), True),
        (("topology", "unique_seed_count"), 394),
        (("topology", "numeric_value_access", "idv_init_slices"), 399),
        (("backend_access_state", "final_state", "numeric_slices_read"), 799),
        (("development_decision_before_final_refit", "path"), "wrong.json"),
    ],
)
def test_result_validator_rejects_access_contract_corruption(
    path: tuple[str, ...], value: Any
) -> None:
    digest = "4" * 64
    result = _backend_result(digest)
    cursor: dict[str, Any] = result
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        runner._validate_development_result(
            result, protocol_canonical_json_sha256=digest
        )


def _access_state(**updates: Any) -> dict[str, Any]:
    state = {
        "schema": "farm_rankcusum_v7_backend_access_state",
        "protocol_canonical_json_sha256": "5" * 64,
        "hdf5_open_started": False,
        "hdf5_open_completed": False,
        "processdata_slices_read": 0,
        "idv_init_slices_read": 0,
        "numeric_slices_read": 0,
        "training_or_fit_started": False,
        "backend_completed": False,
        "mode1_opens": 0,
    }
    state.update(updates)
    return state


@pytest.mark.parametrize(
    ("claim", "intent", "state", "expected"),
    [
        (False, False, None, "INFRASTRUCTURE_FAILURE_BEFORE_CLAIM"),
        (True, False, None, "ABORTED_AFTER_CLAIM_BEFORE_ACCESS_INTENT"),
        (True, True, _access_state(), "ABORTED_AFTER_INTENT_BEFORE_HDF5_OPEN"),
        (
            True,
            True,
            _access_state(hdf5_open_started=True),
            "ABORTED_DURING_HDF5_OPEN_BEFORE_COMPLETION",
        ),
        (
            True,
            True,
            _access_state(hdf5_open_started=True, hdf5_open_completed=True),
            "ABORTED_AFTER_HDF5_OPEN_BEFORE_NUMERIC_SLICE",
        ),
        (
            True,
            True,
            _access_state(
                hdf5_open_started=True,
                hdf5_open_completed=True,
                processdata_slices_read=1,
                numeric_slices_read=1,
            ),
            "ABORTED_AFTER_NUMERIC_READ",
        ),
        (
            True,
            True,
            _access_state(numeric_slices_read=1),
            "ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN",
        ),
    ],
)
def test_failure_states_use_positive_backend_phase_evidence(
    claim: bool,
    intent: bool,
    state: dict[str, Any] | None,
    expected: str,
) -> None:
    assert runner._failure_state(
        claim_written=claim,
        intent_written=intent,
        backend_state=state,
    ) == expected


def test_preflight_failure_does_not_burn_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, identities = _protocol_fixture()
    claim_path = tmp_path / "global.claim.json"
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, identities))
    monkeypatch.setattr(runner, "GLOBAL_CLAIM_PATH", claim_path)
    monkeypatch.setattr(
        runner,
        "_preclaim_audit",
        lambda protocol, identities: (_ for _ in ()).throw(
            RuntimeError("synthetic preclaim failure")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic preclaim"):
        runner.run_development(output_root=tmp_path)
    assert not claim_path.exists()
    assert not (tmp_path / runner.DEVELOPMENT_DIRECTORY_NAME).exists()


def test_successful_runner_orders_preflight_claim_intent_then_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, identities = _protocol_fixture()
    output_root = tmp_path / "alternate-output-root"
    claim_path = tmp_path / "global.claim.json"
    events: list[str] = []
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, identities))
    monkeypatch.setattr(runner, "GLOBAL_CLAIM_PATH", claim_path)

    class Backend:
        @staticmethod
        def run_development(**kwargs: Any) -> dict[str, Any]:
            events.append("backend")
            directory = output_root / runner.DEVELOPMENT_DIRECTORY_NAME
            assert claim_path.is_file()
            assert (directory / "access_intent.json").is_file()
            assert kwargs["protocol_canonical_json_sha256"] == identities["canonical_json_sha256"]
            return _backend_result(identities["canonical_json_sha256"])

    def fake_preclaim_audit(
        persisted_protocol: Any, persisted_identities: Any
    ) -> tuple[Any, dict[str, Any]]:
        events.append("preflight")
        assert persisted_protocol is protocol
        assert persisted_identities is identities
        assert not claim_path.exists()
        return Backend, {
            "source": {"sha256": runner.SOURCE_SHA256},
            "production_backend_handshake": {"hdf5_opened": False},
        }

    monkeypatch.setattr(runner, "_preclaim_audit", fake_preclaim_audit)
    completion = runner.run_development(output_root=output_root)
    assert events == ["preflight", "backend"]
    assert completion["decision"] == "DEVELOPMENT_NO_GO"
    assert claim_path.is_file()
    assert not (output_root / "development_access.claim.json").exists()
    assert (
        output_root / runner.DEVELOPMENT_DIRECTORY_NAME / "development_result.json"
    ).is_file()


def test_ledger_keeps_mode1_sealed_and_mode2_exploratory() -> None:
    ledger = runner._validate_ledger()
    assert ledger["extended_tep_mode1_roles"]["magnitude25_idv1_runs1_100"]["state"] == "SEALED"
    assert ledger["extended_tep_mode1_roles"]["magnitude75_idv1_3_9_15_runs1_100"]["state"] == "SEALED"
    assert "exploratory" in ledger["v7_candidate"]["development_status"]
