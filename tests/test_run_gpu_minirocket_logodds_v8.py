"""Read-free governance tests for the fixed v8 runner."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gpu_minirocket_logodds_v8.py"
SPEC = importlib.util.spec_from_file_location("gpu_minirocket_v8_governance", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _protocol_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = runner._protocol_document(runner._implementation_records())
    identities = {
        "file_bytes_sha256": "a" * 64,
        "canonical_json_sha256": runner._hash_bytes(runner._canonical_bytes(protocol)),
        "commit_file_bytes_sha256": "c" * 64,
    }
    return protocol, identities


def _record(path: Path) -> dict[str, Any]:
    path.write_text("{}\n", encoding="utf-8", newline="\n")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "file_bytes_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json_record(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_bytes(runner._json_file_bytes(value))
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "file_bytes_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _backend_result(tmp_path: Path, protocol_hash: str, *,
                    decision: str = "DEVELOPMENT_NO_GO") -> dict[str, Any]:
    is_go = decision == "GO"
    repeat = {
        "scope": "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay",
        "atol": 1e-6, "maximum_absolute_error": 0.0,
        "discrete_equal": True, "passes": True, "folds": [],
    }
    gates = {"all_pass": is_go, "checks": {"synthetic": is_go},
             "aggregate_metrics": {"synthetic": True}}
    state = {
        "schema": "gpu_minirocket_logodds_v8_backend_access_state",
        "protocol_canonical_json_sha256": protocol_hash,
        "hdf5_open_started": True, "hdf5_open_completed": True,
        "processdata_slices_read": 400, "idv_init_slices_read": 400,
        "numeric_slices_read": 800, "training_or_fit_started": True,
        "processdata_slices_completed": 400, "idv_init_slices_completed": 400,
        "folds_completed": 5, "current_fold": None, "decision_persisted": True,
        "final_refit_started": is_go,
        "backend_completed": True, "mode1_opens": 0, "rieth_opens": 0,
    }
    state_record = _json_record(tmp_path / "backend_access_state.json", state)
    result: dict[str, Any] = {
        "schema": runner.BACKEND_RESULT_SCHEMA, "protocol_id": runner.PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": runner.CANDIDATE_ID, "candidate_count": 1,
        "source_file_bytes_sha256": runner.SOURCE_SHA256,
        "source_size_bytes": runner.SOURCE_SIZE_BYTES,
        "fold_map_canonical_json_sha256": runner.FOLD_MAP_SHA256,
        "access_plan_canonical_json_sha256": runner.ACCESS_PLAN_SHA256,
        "activation_result_file_bytes_sha256": runner.V7_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": runner.V7_DECISION_SHA256,
        "topology": {
            "selected_run_count": 400, "counts_by_status": {"SimulationCompleted": 400},
            "counts_by_idv": {"1": 100, "3": 100, "9": 100, "15": 100},
            "unique_seed_count": 395, "duplicate_seed_value_count": 5,
            "cross_fold_seed_collision_count": 0,
            "numeric_value_access": {
                "processdata_slices": 400, "idv_init_slices": 400,
                "processdata_columns_half_open": [0, 53],
                "root_dataset_value_reads": 0,
                "processdata_values_outside_authorized_windows": 0,
            },
        },
        "environment": {"passed": True, "cuda_available": True,
                        "device_name": runner.ENVIRONMENT_LOCK["gpu_name"],
                        "device_capability": runner.ENVIRONMENT_LOCK["gpu_compute_capability"],
                        "device_total_memory": runner.ENVIRONMENT_LOCK["gpu_total_memory_bytes"],
                        "python_version": runner.ENVIRONMENT_LOCK["python_version"],
                        "numpy_version": runner.ENVIRONMENT_LOCK["numpy_version"],
                        "h5py_distribution_version":
                            runner.ENVIRONMENT_LOCK["h5py_distribution_version"],
                        "torch_version": runner.ENVIRONMENT_LOCK["torch_version"],
                        "torch_cuda_version": runner.ENVIRONMENT_LOCK["torch_cuda_build"],
                        "cudnn_version": runner.ENVIRONMENT_LOCK["cudnn_version"],
                        "golden_handshake_passed": True},
        "backend_access_state": {
            "final_state": state,
            "artifact": state_record,
        },
        "run_authorization": {},
        "mode1_accessed": False, "mode1_opens": 0,
        "rieth_accessed": False, "rieth_opens": 0,
        "inner_threshold_fault_run_count": 0,
        "decision": decision,
        "selected_candidate_id": runner.CANDIDATE_ID if is_go else None,
        "folds": [{"outer_fold": fold} for fold in range(5)],
        "authoritative_cuda_repeat": repeat,
        "cross_device_report_only": {"is_go_gate": False},
        "gates": gates, "aggregate": gates["aggregate_metrics"],
        "development_decision_before_final_refit": None,
        "final_refit": None,
    }
    if is_go:
        result["final_refit"] = {
            "schema": "gpu_minirocket_logodds_v8_final_mode2_refit",
            "protocol_canonical_json_sha256": protocol_hash,
            "candidate_id": runner.CANDIDATE_ID, "runs_per_class": 100,
            "created_only_after_development_go": True, "mode1_accessed": False,
            "state": _record(tmp_path / "final_model_state.npz"),
            "manifest": _record(tmp_path / "manifest.json"),
        }
    evidence = {
        "schema": "gpu_minirocket_logodds_v8_development_decision_before_final_refit",
        "protocol_id": runner.PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": runner.CANDIDATE_ID,
        "activation_result_file_bytes_sha256": runner.V7_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": runner.V7_DECISION_SHA256,
        "development_decision_determined": True, "decision": decision,
        "selected_candidate_id": result["selected_candidate_id"],
        "folds": result["folds"], "gates": result["gates"],
        "aggregate": result["aggregate"],
        "authoritative_cuda_repeat": result["authoritative_cuda_repeat"],
        "cross_device_report_only": result["cross_device_report_only"],
        "final_refit_permitted": is_go, "final_refit_started": False,
        "final_model_usable": False, "mode1_accessed": False,
        "mode1_opens": 0, "rieth_accessed": False, "rieth_opens": 0,
        "no_go_terminates_current_development_pool_method_search": not is_go,
    }
    result["development_decision_before_final_refit"] = _json_record(
        tmp_path / "development_decision_before_final_refit.json", evidence)
    return result


def test_static_runner_has_no_h5py_import_and_inserts_project_root() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imports = {alias.name for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(name == "h5py" or name.startswith("h5py.") for name in imports)
    assert str(PROJECT_ROOT) in sys.path


def test_fold_and_access_hashes_are_frozen() -> None:
    runner._validate_folds()
    assert runner._fold_map_hash() == runner.FOLD_MAP_SHA256
    assert len(runner.development_access_plan()) == 400
    assert runner._hash_bytes(runner._canonical_bytes(
        runner.development_access_plan())) == runner.ACCESS_PLAN_SHA256


def test_exact_v7_no_go_activation_is_hash_bound() -> None:
    evidence = runner._validate_activation()
    assert evidence["trigger"]["result"] == "DEVELOPMENT_NO_GO"
    assert evidence["development_result"]["file_bytes_sha256"] == runner.V7_RESULT_SHA256
    assert evidence["decision_before_refit"]["file_bytes_sha256"] == runner.V7_DECISION_SHA256
    assert evidence["numeric_metrics_used_to_modify_v8"] is False
    assert evidence["v7_bound_contingent_charters"][
        "docs/gpu_minirocket_logodds_v8_contingency.json"
    ]["file_bytes_sha256"] == runner.CHARTER_JSON_V7_BOUND_SHA256


def test_activation_rejects_mutated_v7_protocol_or_charter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_json = runner._strict_json
    mutated = deepcopy(original_json(runner.V7_PROTOCOL_PATH))
    mutated["bindings"]["implementation"][
        "docs/gpu_minirocket_logodds_v8_contingency.json"
    ]["size_bytes"] += 1
    monkeypatch.setattr(
        runner, "_strict_json",
        lambda path: mutated if path == runner.V7_PROTOCOL_PATH else original_json(path))
    with pytest.raises(ValueError, match="v7 frozen protocol"):
        runner._validate_activation()
    monkeypatch.setattr(runner, "_strict_json", original_json)
    original_hash = runner._hash_file
    monkeypatch.setattr(
        runner, "_hash_file",
        lambda path: ("0" * 64 if path == runner.CHARTER_JSON_PATH else original_hash(path)))
    with pytest.raises(ValueError, match="pre-result v8 charter"):
        runner._validate_activation()


def test_charter_and_ledger_machine_validation() -> None:
    assert runner._validate_charter()["minirocket"]["effective_feature_count"] == 9996
    assert runner._validate_ledger()["upstream_identity"]["peeled_commit_sha"] == runner.UPSTREAM_PEELED_COMMIT_SHA
    charter = json.loads(runner.CHARTER_JSON_PATH.read_text(encoding="utf-8"))
    charter["monitor"]["cusum"]["drift"] = 0.3
    original = runner._strict_json
    runner._strict_json = lambda path: charter if path == runner.CHARTER_JSON_PATH else original(path)
    try:
        with pytest.raises(ValueError, match="charter"):
            runner._validate_charter()
    finally:
        runner._strict_json = original


def test_implementation_manifest_includes_vendor_license_and_golden() -> None:
    paths = set(runner._implementation_paths())
    required = {
        "vendor/tsai_v1_0_1_minirocket/minirocket_plus.py",
        "vendor/tsai_v1_0_1_minirocket/LICENSE",
        "vendor/tsai_v1_0_1_minirocket/PROVENANCE.json",
        "vendor/tsai_v1_0_1_minirocket/golden/manifest.json",
        "vendor/tsai_v1_0_1_minirocket/golden/upstream_v1_0_1_synthetic.npz",
    }
    assert required <= paths
    records = runner._implementation_records()
    assert required <= set(records)


def test_atomic_commit_separates_file_and_canonical_hashes(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    identities = runner._commit_json(path, {"z": 1},
                                     commit_schema="synthetic_commit")
    assert identities["file_bytes_sha256"] != identities["canonical_json_sha256"]
    document, verified = runner._verify_committed_json(
        path, commit_schema="synthetic_commit")
    assert document == {"z": 1}
    assert verified == identities
    with pytest.raises(FileExistsError):
        runner._commit_json(path, {"z": 1}, commit_schema="synthetic_commit")


def test_backend_accepts_actual_protocol_document() -> None:
    import torch  # preserve the Windows import order before backend/NumPy
    del torch
    from mate_spc_minirocket_v8 import backend
    protocol = runner._protocol_document(runner._implementation_records())
    digest = runner._hash_bytes(runner._canonical_bytes(protocol))
    report = backend.validate_protocol(
        protocol, expected_canonical_json_sha256=digest)
    assert report["valid"] is True
    changed = deepcopy(protocol)
    changed["candidate"]["minirocket"]["effective_feature_count"] = 9995
    with pytest.raises(ValueError, match="MiniROCKET"):
        backend.validate_protocol(changed)


def test_h5py_import_blocker_detects_caught_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    blocker = runner._H5pyImportBlocker()
    with pytest.raises(RuntimeError, match="h5py import attempted"):
        blocker.find_spec("h5py")
    assert blocker.attempted is True


def test_runner_rejects_detached_and_forged_backend_objects() -> None:
    implementation = runner._implementation_records()
    with pytest.raises(RuntimeError, match="untrusted|detached"):
        runner._attest_backend_module(SimpleNamespace(), implementation)

    import importlib.machinery
    import types
    package = sys.modules.get("mate_spc_minirocket_v8")
    assert package is not None
    path = PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "backend.py"
    fake = types.ModuleType(runner.BACKEND_MODULE)
    loader = importlib.machinery.SourceFileLoader(runner.BACKEND_MODULE, str(path))
    fake.__file__ = str(path)
    fake.__loader__ = loader
    fake.__package__ = "mate_spc_minirocket_v8"
    fake.__spec__ = importlib.util.spec_from_loader(
        runner.BACKEND_MODULE, loader, origin=str(path))
    for name in runner.BACKEND_REQUIRED_FUNCTIONS:
        setattr(fake, name, lambda *args, **kwargs: {})
    original = sys.modules.get(runner.BACKEND_MODULE)
    original_attribute = getattr(package, "backend", None)
    sys.modules[runner.BACKEND_MODULE] = fake
    package.backend = fake
    try:
        with pytest.raises(RuntimeError, match="callable origin"):
            runner._attest_backend_module(fake, implementation)
    finally:
        if original is None:
            sys.modules.pop(runner.BACKEND_MODULE, None)
        else:
            sys.modules[runner.BACKEND_MODULE] = original
        package.backend = original_attribute


@pytest.mark.parametrize("decision", ["DEVELOPMENT_NO_GO", "GO"])
def test_result_validator_rejects_hollow_fold_and_fake_refit(
    tmp_path: Path, decision: str,
) -> None:
    result = _backend_result(tmp_path, "d" * 64, decision=decision)
    with pytest.raises(ValueError):
        runner._validate_development_result(
            result, protocol_canonical_json_sha256="d" * 64,
            expected_run_authorization={})


def test_failure_state_distinguishes_access_phases() -> None:
    assert runner._failure_state(claim_written=False, intent_written=False,
                                 backend_state=None) == "INFRASTRUCTURE_FAILURE_BEFORE_CLAIM"
    assert runner._failure_state(claim_written=True, intent_written=False,
                                 backend_state=None) == "ABORTED_AFTER_CLAIM_BEFORE_ACCESS_INTENT"
    assert runner._failure_state(
        claim_written=True, intent_written=False, backend_state=None,
        intent_bundle_observed=True) == "ABORTED_DURING_ACCESS_INTENT_COMMIT"
    state = {
        "schema": "gpu_minirocket_logodds_v8_backend_access_state",
        "protocol_canonical_json_sha256": "a" * 64,
        "hdf5_open_started": False, "hdf5_open_completed": False,
        "processdata_slices_read": 0, "idv_init_slices_read": 0,
        "numeric_slices_read": 0, "training_or_fit_started": False,
        "processdata_slices_completed": 0, "idv_init_slices_completed": 0,
        "backend_completed": False, "mode1_opens": 0, "rieth_opens": 0,
    }
    assert runner._failure_state(claim_written=True, intent_written=True,
                                 backend_state=state) == "ABORTED_AFTER_INTENT_BEFORE_HDF5_OPEN"


def _patch_claim_only_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "output"
    claim = root / "development_access.claim.json"
    failure = root / "development_access.claim.failure.json"
    protocol, identities = _protocol_fixture()
    monkeypatch.setattr(runner, "GLOBAL_CLAIM_PATH", claim)
    monkeypatch.setattr(runner, "GLOBAL_CLAIM_FAILURE_PATH", failure)
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT_ROOT", root)
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda: (protocol, identities))
    monkeypatch.setattr(runner, "_validate_activation", lambda: {})
    monkeypatch.setattr(runner, "_require_formal_development_invocation", lambda: {
        "synthetic_test_only": True})
    monkeypatch.setattr(runner, "_preclaim_audit", lambda *args, **kwargs: (SimpleNamespace(), {}))
    return claim, failure


def test_claim_commit_partial_failure_is_recorded_and_blocks_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    claim, failure = _patch_claim_only_prefix(monkeypatch, tmp_path)
    def partial_commit(value: Any, *, ownership: dict[str, bool]) -> dict[str, Any]:
        ownership["lease_acquired"] = True
        runner._write_json(claim, value)
        raise RuntimeError("synthetic partial claim commit")

    monkeypatch.setattr(runner, "_commit_global_claim", partial_commit)
    with pytest.raises(RuntimeError, match="partial claim"):
        runner.run_development(output_root=tmp_path / "output")
    assert claim.is_file() and failure.is_file()
    evidence = runner._strict_json(failure)
    assert evidence["decision"] == "FAIL_CLOSED_NO_RERUN"
    assert evidence["failure_stage"] == "global_claim_commit"
    assert evidence["observed_claim_bundle"]["document"]["exists"] is True
    with pytest.raises(FileExistsError, match="prior v8 global claim"):
        runner.run_development(output_root=tmp_path / "output")


def test_global_claim_fsync_failure_retains_lease_and_blocks_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    claim, failure = _patch_claim_only_prefix(monkeypatch, tmp_path)
    original_fsync = runner.os.fsync
    calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic claim fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="claim fsync"):
        runner.run_development(output_root=tmp_path / "output")
    assert claim.is_file() and failure.is_file()
    assert not runner._sidecar_path(claim).exists()
    evidence = runner._strict_json(failure)
    assert evidence["decision"] == "FAIL_CLOSED_NO_RERUN"
    assert evidence["observed_claim_bundle"]["document"]["exists"] is True
    with pytest.raises(FileExistsError, match="prior v8 global claim"):
        runner.run_development(output_root=tmp_path / "output")


def test_directory_failure_after_committed_claim_records_exact_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    claim, failure = _patch_claim_only_prefix(monkeypatch, tmp_path)
    directory = tmp_path / "output" / runner.DEVELOPMENT_DIRECTORY_NAME
    original_mkdir = Path.mkdir

    def guarded_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == directory:
            raise PermissionError("synthetic directory failure")
        original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    with pytest.raises(PermissionError, match="directory failure"):
        runner.run_development(output_root=tmp_path / "output")
    assert claim.is_file() and runner._commit_path(claim).is_file() and failure.is_file()
    evidence = runner._strict_json(failure)
    assert evidence["failure_stage"] == "fixed_artifact_directory_creation_after_claim"
    identity = evidence["claim_identity_if_committed"]
    assert identity["file_bytes_sha256"] == hashlib.sha256(claim.read_bytes()).hexdigest()
    assert identity["commit_file_bytes_sha256"] == hashlib.sha256(
        runner._commit_path(claim).read_bytes()).hexdigest()


def test_partial_access_intent_bundle_is_conservatively_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _, _ = _patch_claim_only_prefix(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_preclaim_audit", lambda *args, **kwargs: (
        SimpleNamespace(), {"source": {
            "sha256": runner.SOURCE_SHA256, "size_bytes": runner.SOURCE_SIZE_BYTES,
            "hdf5_opened": False}}))
    original_commit = runner._commit_json

    def partial_intent(path: Path, value: Any, *, commit_schema: str) -> dict[str, Any]:
        if path.name == "access_intent.json":
            runner._write_json(path, value)
            raise RuntimeError("synthetic partial intent commit")
        return original_commit(path, value, commit_schema=commit_schema)

    monkeypatch.setattr(runner, "_commit_json", partial_intent)
    with pytest.raises(RuntimeError, match="partial intent"):
        runner.run_development(output_root=tmp_path / "output")
    run_dir = tmp_path / "output" / runner.DEVELOPMENT_DIRECTORY_NAME
    failure = runner._strict_json(run_dir / "failure.json")
    completion = runner._strict_json(run_dir / "completion.json")
    assert failure["decision"] == "ABORTED_DURING_ACCESS_INTENT_COMMIT"
    assert failure["observed_access_intent_bundle"]["document"]["exists"] is True
    assert failure["access_intent_identity_if_committed"] is None
    assert completion["access_intent_commit_file_bytes_sha256"] is None


def test_formal_development_rejects_output_override_before_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner, "_verify_protocol_bundle",
        lambda: (_ for _ in ()).throw(AssertionError("preflight reached")))
    with pytest.raises(ValueError, match="frozen default"):
        runner.run_development(output_root=tmp_path)


def test_formal_invocation_is_rejected_before_protocol_or_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "_verify_protocol_bundle",
        lambda: (_ for _ in ()).throw(AssertionError("protocol reached")))
    with pytest.raises(RuntimeError, match="invocation contract"):
        runner.run_development(output_root=runner.DEFAULT_OUTPUT_ROOT)


def test_access_state_must_correlate_with_decision_and_completed_folds(
    tmp_path: Path,
) -> None:
    result = _backend_result(tmp_path, "a" * 64, decision="DEVELOPMENT_NO_GO")
    report = result["backend_access_state"]
    runner._validate_backend_access_state_report(
        report, protocol_canonical_json_sha256="a" * 64,
        decision="DEVELOPMENT_NO_GO", numeric_folds_completed=5)
    for field, changed in (
        ("folds_completed", 0), ("current_fold", 4),
        ("decision_persisted", False), ("final_refit_started", True),
    ):
        forged = deepcopy(report)
        forged["final_state"][field] = changed
        with pytest.raises(ValueError, match="access-state final state"):
            runner._validate_backend_access_state_report(
                forged, protocol_canonical_json_sha256="a" * 64,
                decision="DEVELOPMENT_NO_GO", numeric_folds_completed=5)
