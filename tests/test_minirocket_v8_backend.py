"""Synthetic-only tests for the v8 production backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import torch  # required Windows DLL order
import numpy as np
import pytest

from mate_spc.hash_contract import canonical_json_file_bytes, canonical_json_sha256
from mate_spc_minirocket_v8 import backend
from mate_spc_minirocket_v8 import method
from mate_spc_minirocket_v8.transform import _fit_transform_impl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_gpu_minirocket_logodds_v8.py"
SPEC = importlib.util.spec_from_file_location("v8_runner_for_backend_tests", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _protocol() -> dict[str, Any]:
    return runner._protocol_document(runner._implementation_records())


def _fold_map() -> dict[int, int]:
    return {run: fold for fold, runs in backend.FOLD_RUN_IDS.items() for run in runs}


def _topology() -> dict[str, Any]:
    return {
        "selected_run_count": 400, "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {"1": 100, "3": 100, "9": 100, "15": 100},
        "unique_seed_count": 395, "duplicate_seed_value_count": 5,
        "cross_fold_seed_collision_count": 0,
        "numeric_value_access": {
            "processdata_slices": 400, "processdata_rows_per_slice": 500,
            "processdata_columns_half_open": [0, 53], "idv_init_slices": 400,
            "root_dataset_value_reads": 0,
            "processdata_values_outside_authorized_windows": 0,
        },
    }


def _identity_blocks() -> tuple[backend.NumericBlock, ...]:
    shared = np.zeros((500, 52), dtype=np.float64)
    return tuple(
        backend.NumericBlock(
            simulation_run=run_id,
            conditions=tuple(
                backend.NumericRun(
                    name=f"synthetic-{idv}-{run_id}", x=shared,
                    source_idv=idv, simulation_run=run_id,
                    rng_seed=10_000_000 + 100 * run_id + idv)
                for idv in (1, 3, 9, 15)))
        for run_id in range(1, 101))


def test_clean_subprocess_imports_backend_from_nonproject_cwd(tmp_path: Path) -> None:
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(PROJECT_ROOT)!r});"
        f"sys.path.insert(0,{str(PROJECT_ROOT / 'src')!r});"
        "import mate_spc_minirocket_v8.backend as b;"
        "assert 'torch' in sys.modules and 'numpy' in sys.modules;"
        "assert 'mate_spc' not in sys.modules and 'mate_spc.model' not in sys.modules;"
        "print(b.CANDIDATE_ID)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert backend.CANDIDATE_ID in completed.stdout


@pytest.mark.skipif(
    not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0).upper(),
    reason="formal golden preflight requires the RTX 5090",
)
def test_real_cuda_preflight_uses_exact_golden_status_contract() -> None:
    report = backend.cuda_preflight()
    assert report["passed"] is True
    assert report["golden_handshake_passed"] is True
    assert report["golden_handshake"]["status"] == "PASS"


def test_protocol_uses_canonical_hash_and_rejects_mutations() -> None:
    protocol = _protocol()
    canonical = canonical_json_sha256(protocol)
    file_hash = hashlib.sha256(canonical_json_file_bytes(protocol)).hexdigest()
    assert canonical != file_hash
    report = backend.validate_protocol(
        protocol, expected_canonical_json_sha256=canonical)
    assert report["protocol_canonical_json_sha256"] == canonical
    assert report["file_bytes_sha256_compared"] is False
    with pytest.raises(ValueError, match="canonical_json_sha256"):
        backend.validate_protocol(protocol, expected_canonical_json_sha256=file_hash)
    changed = deepcopy(protocol)
    changed["candidate"]["activation"]["required_exact_result"] = "GO"
    with pytest.raises(ValueError, match="activation"):
        backend.validate_protocol(changed)
    changed = deepcopy(protocol)
    changed["candidate"]["linear_head"]["max_iter"] = 101
    with pytest.raises(ValueError, match="linear-head"):
        backend.validate_protocol(changed)


def test_exact_access_plan_and_fold_map_fail_closed() -> None:
    plan = backend.development_access_plan()
    assert len(plan) == 400
    assert backend._validate_access_plan(plan) == plan
    changed = deepcopy(plan)
    changed[0]["processdata_row_zero_based_half_open"] = [0, 500]
    with pytest.raises(ValueError, match="exact 400"):
        backend._validate_access_plan(changed)
    folds = _fold_map()
    assert backend._validate_fold_map(folds) == folds
    folds[1] = 0
    with pytest.raises(ValueError, match="collision-safe"):
        backend._validate_fold_map(folds)


def test_physical_claim_and_committed_intent_gate_precedes_h5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_dir = tmp_path / backend.DEVELOPMENT_DIRECTORY_NAME
    run_dir.mkdir()
    artifact_root = run_dir / "artifacts"
    protocol_path = tmp_path / "gpu_minirocket_logodds_v8_mode2_development.json"
    monkeypatch.setattr(backend, "EXPECTED_PROTOCOL_PATH", protocol_path.resolve())
    monkeypatch.setattr(backend, "EXPECTED_FIXED_DIRECTORY", run_dir.resolve())
    monkeypatch.setattr(backend, "EXPECTED_ARTIFACT_ROOT", artifact_root.resolve())
    monkeypatch.setattr(backend, "preclaim_bind_method_graph", lambda implementation: {
        "bound": True, "hdf5_loaded": False})
    protocol_identity = runner._commit_json(
        protocol_path, runner._protocol_document(runner._implementation_records()),
        commit_schema="gpu_minirocket_logodds_v8_protocol_atomic_commit")
    protocol_hash = protocol_identity["canonical_json_sha256"]
    preflight_identity = runner._write_json(run_dir / "preflight.json", {"passed": True})
    common = {
        "protocol_id": backend.PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": backend.CANDIDATE_ID,
        "candidate_count": 1,
        "activation_result_file_bytes_sha256": backend.ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": backend.ACTIVATION_DECISION_SHA256,
        "access_plan_canonical_json_sha256": backend.ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": backend.FOLD_MAP_CANONICAL_JSON_SHA256,
    }
    claim_path = tmp_path / "development_access.claim.json"
    monkeypatch.setattr(backend, "EXPECTED_GLOBAL_CLAIM_PATH", claim_path.resolve())
    claim_identity = runner._commit_json(claim_path, {
        "schema": "gpu_minirocket_logodds_v8_exclusive_global_development_claim",
        "claim_attempt_id": "1" * 32,
        **common, "artifact_root": str(run_dir.resolve()),
        "backend_artifact_root": str(artifact_root.resolve()),
        "numeric_execution_claim_path":
            str((run_dir / "numeric_execution.claim.json").resolve()),
        "formal_invocation": backend.EXPECTED_FORMAL_INVOCATION,
        "fixed_directory": backend.DEVELOPMENT_DIRECTORY_NAME,
        "rescue_iteration_ids_allowed": False,
    }, commit_schema="gpu_minirocket_logodds_v8_global_claim_atomic_commit")
    intent_path = run_dir / "access_intent.json"
    intent_identity = runner._commit_json(intent_path, {
        "schema": "gpu_minirocket_logodds_v8_mode2_development_access_intent",
        **common, "committed_after_read_free_backend_handshake": True,
        "committed_before_hdf5_import_or_open": True,
        "access_plan": backend.development_access_plan(),
        "fold_run_ids": {str(fold): list(runs)
                         for fold, runs in backend.FOLD_RUN_IDS.items()},
        "authorized_reads_after_commit": "only exact 400 Mode2 magnitude50 selectors",
        "wildcards_authorized": False,
        "inner_fault_threshold_values_authorized": False,
        "mode1_access_authorized": False, "mode1_hdf5_opens_or_value_reads": 0,
        "rieth_access_authorized": False,
        "backend_module": "mate_spc_minirocket_v8.backend",
        "backend_imported_before_commit": True,
        "production_backend_instance_reused_after_commit": True,
        "h5py_module_loaded_before_commit": False,
        "h5py_import_attempted_before_commit": False,
        "artifact_root": str(artifact_root.resolve()),
        "numeric_execution_claim_path":
            str((run_dir / "numeric_execution.claim.json").resolve()),
        "formal_invocation": backend.EXPECTED_FORMAL_INVOCATION,
        "preflight_file_bytes_sha256": preflight_identity["file_bytes_sha256"],
        "source": {"sha256": backend.MODE2_SHA256,
                   "size_bytes": backend.SOURCE_SIZE_BYTES, "hdf5_opened": False},
        "implementation": runner._implementation_records(),
        "global_claim_path": str(claim_path.resolve()),
        "global_claim_file_bytes_sha256": claim_identity["file_bytes_sha256"],
        "global_claim_canonical_json_sha256": claim_identity["canonical_json_sha256"],
        "global_claim_file_size_bytes": claim_identity["file_size_bytes"],
        "global_claim_commit_file_bytes_sha256": claim_identity["commit_file_bytes_sha256"],
    }, commit_schema="gpu_minirocket_logodds_v8_access_intent_atomic_commit")
    report = backend._validate_run_authorization(
        artifact_root=artifact_root, protocol_canonical_json_sha256=protocol_hash,
        protocol_path=protocol_path,
        protocol_file_bytes_sha256=protocol_identity["file_bytes_sha256"],
        protocol_commit_file_bytes_sha256=protocol_identity["commit_file_bytes_sha256"],
        claim_path=claim_path,
        claim_file_bytes_sha256=claim_identity["file_bytes_sha256"],
        claim_commit_file_bytes_sha256=claim_identity["commit_file_bytes_sha256"],
        access_intent_path=intent_path,
        access_intent_file_bytes_sha256=intent_identity["file_bytes_sha256"],
        access_intent_commit_file_bytes_sha256=intent_identity["commit_file_bytes_sha256"])
    assert report["validated_before_hdf5_import_or_open"] is True
    with pytest.raises(ValueError, match="exact fixed artifacts"):
        backend._validate_run_authorization(
            artifact_root=run_dir / "artifacts2",
            protocol_canonical_json_sha256=protocol_hash,
            protocol_path=protocol_path,
            protocol_file_bytes_sha256=protocol_identity["file_bytes_sha256"],
            protocol_commit_file_bytes_sha256=protocol_identity["commit_file_bytes_sha256"],
            claim_path=claim_path,
            claim_file_bytes_sha256=claim_identity["file_bytes_sha256"],
            claim_commit_file_bytes_sha256=claim_identity["commit_file_bytes_sha256"],
            access_intent_path=intent_path,
            access_intent_file_bytes_sha256=intent_identity["file_bytes_sha256"],
            access_intent_commit_file_bytes_sha256=
                intent_identity["commit_file_bytes_sha256"])
    with pytest.raises(ValueError, match="identity mismatch"):
        backend._validate_run_authorization(
            artifact_root=artifact_root, protocol_canonical_json_sha256=protocol_hash,
            protocol_path=protocol_path,
            protocol_file_bytes_sha256=protocol_identity["file_bytes_sha256"],
            protocol_commit_file_bytes_sha256=protocol_identity["commit_file_bytes_sha256"],
            claim_path=claim_path, claim_file_bytes_sha256="0" * 64,
            claim_commit_file_bytes_sha256=claim_identity["commit_file_bytes_sha256"],
            access_intent_path=intent_path,
            access_intent_file_bytes_sha256=intent_identity["file_bytes_sha256"],
            access_intent_commit_file_bytes_sha256=intent_identity["commit_file_bytes_sha256"])


def test_public_run_rejects_missing_physical_authorization_before_h5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        backend, "_load_numeric_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("H5 path reached")))
    with pytest.raises(ValueError, match="exact frozen v8 protocol"):
        backend.run_development(
            access_plan=backend.development_access_plan(), fold_map=_fold_map(),
            artifact_root=tmp_path / backend.DEVELOPMENT_DIRECTORY_NAME / "artifacts",
            protocol_canonical_json_sha256="a" * 64,
            protocol_path=tmp_path / "not-the-frozen-protocol.json",
            protocol_file_bytes_sha256="1" * 64,
            protocol_commit_file_bytes_sha256="2" * 64,
            claim_path=tmp_path / "not-the-global-claim.json",
            claim_file_bytes_sha256="b" * 64,
            claim_commit_file_bytes_sha256="3" * 64,
            access_intent_path=tmp_path / "access_intent.json",
            access_intent_file_bytes_sha256="c" * 64,
            access_intent_commit_file_bytes_sha256="d" * 64)


def test_access_state_is_atomic_and_mode1_rieth_are_fixed_zero(tmp_path: Path) -> None:
    state = backend._AccessState(
        tmp_path / "backend_access_state.json",
        protocol_canonical_json_sha256="a" * 64)
    state.update(hdf5_open_started=True, hdf5_open_completed=True)
    state.record_processdata_slice()
    state.record_idv_init_slice()
    state.update(training_or_fit_started=True, folds_completed=1, current_fold=1)
    document = json.loads(state.path.read_text(encoding="utf-8"))
    assert document["numeric_slices_read"] == 2
    assert document["mode1_opens"] == document["rieth_opens"] == 0
    with pytest.raises(RuntimeError, match="Mode1"):
        state.update(mode1_opens=1)


class _FakeGroup:
    def __init__(self, seed: int) -> None:
        self.attrs = {"seed": np.asarray([seed], dtype=np.int64)}


class _FakeDataset:
    def __init__(self, kind: str, idv: int) -> None:
        self.kind, self.idv = kind, idv
        self.shape = (2000, 54) if kind == "processdata" else (3, 28)
        self.dtype = np.dtype(np.float64)
        self.calls: list[Any] = []

    def __getitem__(self, key: Any) -> np.ndarray:
        self.calls.append(key)
        if self.kind == "processdata":
            rows, columns = key
            assert columns == slice(0, 53)
            start, stop = rows.start, rows.stop
            result = np.zeros((stop - start, 53), dtype=np.float64)
            result[:, 0] = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
            return result
        assert key == (slice(0, 3), slice(0, 28))
        result = np.zeros((3, 28), dtype=np.float64)
        result[:, self.idv - 1] = (0.0, 0.5, 30.0)
        return result


class _FakeHandle:
    def __init__(self, plan: list[dict[str, Any]]) -> None:
        self.objects: dict[str, Any] = {}
        duplicate = {owner: seed for seed, owners in backend.ALLOWED_DUPLICATE_SEED_OWNERS.items()
                     for owner in owners}
        self.accessed: list[str] = []
        for selector in plan:
            owner = (selector["simulation_run"], selector["idv"])
            seed = duplicate.get(owner, 10_000_000 + 100 * owner[0] + owner[1])
            self.objects[selector["group_path"]] = _FakeGroup(seed)
            self.objects[selector["processdata_path"]] = _FakeDataset("processdata", owner[1])
            self.objects[selector["idv_init_path"]] = _FakeDataset("idv_init", owner[1])

    def __contains__(self, path: str) -> bool:
        return path in self.objects

    def __getitem__(self, path: str) -> Any:
        self.accessed.append(path)
        return self.objects[path]


class _Counter:
    def __init__(self) -> None:
        self.processdata = self.idv_init = 0

    def record_processdata_slice(self) -> None:
        self.processdata += 1

    def record_idv_init_slice(self) -> None:
        self.idv_init += 1


def test_exact_reader_reads_only_400_authorized_slices() -> None:
    plan = backend.development_access_plan()
    handle, counter = _FakeHandle(plan), _Counter()
    topology, blocks = backend._read_exact_blocks(
        handle, plan, access_state=counter)  # type: ignore[arg-type]
    assert topology == _topology()
    assert len(blocks) == 100
    assert (counter.processdata, counter.idv_init) == (400, 400)
    assert "Processdata_Labels" not in handle.accessed
    for selector in plan:
        start, stop = selector["window_zero_based_half_open"]
        assert handle.objects[selector["processdata_path"]].calls == [
            (slice(start, stop), slice(0, 53))]
        assert handle.objects[selector["idv_init_path"]].calls == [
            (slice(0, 3), slice(0, 28))]


def test_slice_attempt_is_persisted_before_decode_failure(tmp_path: Path) -> None:
    plan = backend.development_access_plan()
    handle = _FakeHandle(plan)

    class FailingDataset(_FakeDataset):
        def __getitem__(self, key: Any) -> np.ndarray:
            self.calls.append(key)
            raise RuntimeError("synthetic failure after slice request")

    first_path = plan[0]["processdata_path"]
    handle.objects[first_path] = FailingDataset("processdata", 1)
    state = backend._AccessState(
        tmp_path / "backend_access_state.json",
        protocol_canonical_json_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        backend._read_exact_blocks(handle, plan, access_state=state)
    persisted = json.loads(state.path.read_text(encoding="utf-8"))
    assert persisted["processdata_slices_read"] == 1
    assert persisted["processdata_slices_completed"] == 0
    assert persisted["numeric_slices_read"] == 1


def test_scoring_rng_guard_restores_python_numpy_and_torch() -> None:
    backend._configure_determinism()
    with backend._PreserveRng():
        random.random()
        np.random.random()
        torch.rand(3)
    observed = (random.random(), float(np.random.random()), torch.rand(3))
    backend._configure_determinism()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_readonly_training_and_scoring_arrays_never_alias_into_torch() -> None:
    from mate_spc_minirocket_v8.model import _torch_linear_fit

    features = np.asarray([
        [-1.0, -1.0], [-2.0, -1.0], [1.0, -1.0], [2.0, -1.0],
        [-1.0, 1.0], [-2.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ], dtype=np.float32)
    labels = np.repeat(np.arange(4, dtype=np.int64), 2)
    features.setflags(write=False)
    labels.setflags(write=False)
    result = _torch_linear_fit(
        features, labels, device="cpu", enforce_frozen_shape=False)
    assert result[0].shape == (4, 2)


def test_preclaim_graph_rejects_fake_preloaded_method_in_clean_subprocess() -> None:
    source = r'''
import importlib.util
from pathlib import Path
import sys, types
root = Path.cwd()
sys.path[:0] = [str(root), str(root / "src")]
import torch
from mate_spc_minirocket_v8 import backend
spec = importlib.util.spec_from_file_location("v8_graph_attack_runner", root / "scripts" / "run_gpu_minirocket_logodds_v8.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
fake = types.ModuleType("mate_spc_minirocket_v8.method")
fake.__file__ = str(root / "src" / "mate_spc_minirocket_v8" / "method.py")
sys.modules[fake.__name__] = fake
sys.modules["mate_spc_minirocket_v8"].method = fake
try:
    backend.preclaim_bind_method_graph(runner._implementation_records())
except RuntimeError as error:
    assert "preloaded" in str(error)
    print("FAKE_METHOD_REJECTED")
else:
    raise AssertionError("fake method was accepted")
'''
    environment = dict(os.environ)
    environment["MATE_SPC_V8_FORBID_REAL_H5"] = "1"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, "-c", source], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, check=False, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FAKE_METHOD_REJECTED" in completed.stdout


def test_numeric_execution_claim_is_exclusive_under_concurrency(tmp_path: Path) -> None:
    path = tmp_path / "numeric_execution.claim.json"

    def attempt(index: int) -> str:
        try:
            backend._exclusive_json(path, {"winner": index})
            return "won"
        except FileExistsError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, (1, 2)))
    assert sorted(outcomes) == ["lost", "won"]
    assert json.loads(path.read_text(encoding="utf-8"))["winner"] in {1, 2}


def test_existing_numeric_execution_claim_rejects_before_h5(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_run_inputs(monkeypatch, go=False)
    token = tmp_path / "numeric_execution.claim.json"
    token.write_text("partial-token", encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        backend, "_load_numeric_blocks",
        lambda *args, **kwargs: calls.append(True))
    with pytest.raises(FileExistsError, match="forbids replay"):
        backend.run_development(
            access_plan=backend.development_access_plan(), fold_map=_fold_map(),
            artifact_root=tmp_path / "artifacts",
            protocol_canonical_json_sha256="a" * 64,
            protocol_path=tmp_path / "protocol.json",
            protocol_file_bytes_sha256="1" * 64,
            protocol_commit_file_bytes_sha256="2" * 64,
            claim_path=tmp_path / "claim.json", claim_file_bytes_sha256="d" * 64,
            claim_commit_file_bytes_sha256="3" * 64,
            access_intent_path=tmp_path / "intent.json",
            access_intent_file_bytes_sha256="e" * 64,
            access_intent_commit_file_bytes_sha256="f" * 64)
    assert calls == []


def test_same_authorization_cannot_execute_backend_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_run_inputs(monkeypatch, go=False)
    arguments = dict(
        access_plan=backend.development_access_plan(), fold_map=_fold_map(),
        artifact_root=tmp_path / "artifacts", protocol_canonical_json_sha256="a" * 64,
        protocol_path=tmp_path / "protocol.json",
        protocol_file_bytes_sha256="1" * 64,
        protocol_commit_file_bytes_sha256="2" * 64,
        claim_path=tmp_path / "claim.json", claim_file_bytes_sha256="d" * 64,
        claim_commit_file_bytes_sha256="3" * 64,
        access_intent_path=tmp_path / "intent.json",
        access_intent_file_bytes_sha256="e" * 64,
        access_intent_commit_file_bytes_sha256="f" * 64)
    backend.run_development(**arguments)
    with pytest.raises(FileExistsError, match="forbids replay"):
        backend.run_development(**arguments)


def test_inner_replay_covers_fit_calibration_scan_threshold_and_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maxima = np.arange(1.0, 21.0, dtype=np.float64)
    paths = np.zeros((20, 340), dtype=np.float64)
    paths[:, -1] = maxima
    replay = {
        "fit_q": np.zeros((60, 348), dtype=np.float64),
        "calibration": method.MonitorCalibration(center=0.0, scale=1.0),
        "inner_q": np.zeros((20, 348), dtype=np.float64),
        "scores": method.SequentialScoreBatch(paths=paths),
    }
    monkeypatch.setattr(backend, "_replay_inner", lambda *args, **kwargs: replay)
    threshold, report = None, None
    calibration, threshold, report = backend._audit_inner(
        SimpleNamespace(), tuple(range(60)), tuple(range(20)), device="cuda")
    assert calibration.center == 0.0
    assert threshold.threshold == 20.0
    assert report["passes"] is True
    assert report["scope"] == (
        "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay")


def test_complete_fold_and_final_pipeline_no_pickle_roundtrip(tmp_path: Path) -> None:
    windows = np.random.default_rng(812).normal(size=(8, 52, 128)).astype(np.float32)
    method.set_determinism(backend.GLOBAL_SEED)
    transform = replace(
        _fit_transform_impl(windows, device="cpu", enforce_frozen_count=False),
        source_device="cuda:0")
    normalizer = method.RobustNormalizer(
        np.linspace(-0.25, 0.25, 52, dtype=np.float64),
        np.linspace(0.5, 1.5, 52, dtype=np.float64))
    standardizer = method.FeatureStandardizer(
        np.linspace(-0.2, 0.2, 9996, dtype=np.float64),
        np.linspace(0.5, 1.5, 9996, dtype=np.float64))
    head_rng = np.random.default_rng(9182)
    head = method.LinearHeadState(
        weight=head_rng.normal(0.0, 1e-3, size=(4, 9996)).astype(np.float32),
        bias=np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32), initial_objective=1.0,
        final_objective=0.5, final_max_abs_gradient=0.1, source_device="cuda")
    pipeline = backend._FoldPipeline(normalizer, transform, standardizer, head)
    calibration = method.MonitorCalibration(center=0.0, scale=1.0)
    threshold = method.InnerThreshold(
        threshold=1.0, rank19=0.0, distinct_count=20, repeat_error=0.0)

    fold_dir = tmp_path / "fold_0"
    frozen = backend._save_fold_artifacts(
        pipeline, calibration, threshold, fold_dir, outer_fold=0,
        roles=method.fold_roles(0).to_document(), protocol_hash="a" * 64,
        rng_hashes={"synthetic": "b" * 64})
    restored, restored_calibration, restored_threshold = backend.load_frozen_fold_pipeline(
        frozen["freeze_manifest"]["path"])
    first_q = backend._pipeline_window_log_odds(pipeline, windows[:2], device="cpu")
    second_q = backend._pipeline_window_log_odds(restored, windows[:2], device="cpu")
    np.testing.assert_array_equal(first_q, second_q)
    assert float(np.ptp(first_q)) > 1e-8

    # Exercise the complete path, including normalization and all 348 frozen
    # prefix/monitor endpoints.  Nonzero head parameters make this sensitive to
    # any incorrect transform or standardizer restoration.
    raw_run = np.random.default_rng(991).normal(size=(500, 52)).astype(np.float64)
    numeric_run = backend.NumericRun(
        name="synthetic/H/Run1", x=raw_run, source_idv=1,
        simulation_run=1, rng_seed=991)
    original_complete_q = backend._score_complete(
        pipeline, (numeric_run,), device="cpu")
    restored_complete_q = backend._score_complete(
        restored, (numeric_run,), device="cpu")
    np.testing.assert_array_equal(original_complete_q, restored_complete_q)
    assert float(np.ptp(original_complete_q)) > 1e-8
    dense = np.linspace(-1.0, 2.0, 340, dtype=np.float64)[None, :]
    prefix = np.zeros((1, 8), dtype=np.float64)
    first_scan = method.scan_batch(
        original_complete_q[:, 8:], original_complete_q[:, :8], calibration).paths
    second_scan = method.scan_batch(
        restored_complete_q[:, 8:], restored_complete_q[:, :8],
        restored_calibration).paths
    np.testing.assert_array_equal(first_scan, second_scan)
    assert restored_threshold.to_document() == threshold.to_document()
    with np.load(fold_dir / "fold_model_state.npz", allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)

    final_dir = tmp_path / "final_refit"
    final_dir.mkdir()
    method.save_minirocket_state_npz(final_dir / "minirocket_transform.npz", transform)
    before, after = {"synthetic": "c" * 64}, {"synthetic": "d" * 64}
    backend._save_complete_pipeline(
        final_dir / "final_model_state.npz", pipeline, training_example_count=34_000,
        rng_before=before, rng_after=after)
    manifest = {
        "schema": "gpu_minirocket_logodds_v8_final_mode2_refit",
        "protocol_canonical_json_sha256": "a" * 64,
        "candidate_id": backend.CANDIDATE_ID, "runs_per_class": 100,
        "total_fit_run_count": 400, "training_example_count": 34_000,
        "created_only_after_development_go": True,
        "continued_rng_stream_without_reseed": True,
        "rng_state_before": before, "rng_state_after": after,
        "target_domain_threshold_not_fitted": True,
        "mode1_accessed": False, "mode1_opens": 0,
        "transform": backend._file_record(final_dir / "minirocket_transform.npz"),
        "state": backend._file_record(final_dir / "final_model_state.npz"),
    }
    backend._atomic_json(final_dir / "manifest.json", manifest)
    restored_final = backend.load_final_refit_pipeline(final_dir / "manifest.json")
    final_q = backend._pipeline_window_log_odds(restored_final, windows[:2], device="cpu")
    np.testing.assert_array_equal(first_q, final_q)

    with np.load(final_dir / "final_model_state.npz", allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["normalizer_location"] = arrays["normalizer_location"].astype(np.float32)
    bad = tmp_path / "wrong_dtype.npz"
    np.savez_compressed(bad, **arrays)
    with pytest.raises(ValueError, match="wrong persisted shape or dtype"):
        backend._load_core_arrays(
            bad, expected_schema="gpu_minirocket_logodds_v8_complete_pipeline",
            expected_training_examples=34_000, include_calibration=False)


def _patch_run_inputs(monkeypatch: pytest.MonkeyPatch, *, go: bool) -> list[int]:
    monkeypatch.setattr(backend, "_validate_run_authorization", lambda **kwargs: {
        "validated_before_hdf5_import_or_open": True,
        "claim_file_bytes_sha256": "d" * 64,
        "access_intent_file_bytes_sha256": "e" * 64,
        "access_intent_commit_file_bytes_sha256": "f" * 64,
    })
    monkeypatch.setattr(backend, "_require_cuda", lambda device: {
        "passed": True, "cuda_available": True,
        "device_name": "NVIDIA RTX 5090", "golden_handshake_passed": True})
    blocks = _identity_blocks()

    def fake_load(plan: Any, *, access_state: Any) -> Any:
        access_state.update(hdf5_open_started=True, hdf5_open_completed=True,
                            processdata_slices_read=400, idv_init_slices_read=400,
                            numeric_slices_read=800, processdata_slices_completed=400,
                            idv_init_slices_completed=400)
        return _topology(), blocks

    monkeypatch.setattr(backend, "_load_numeric_blocks", fake_load)
    monkeypatch.setattr(backend, "_validate_blocks", lambda value, folds: tuple(value))
    seed_calls: list[int] = []

    class Gate:
        all_pass = go

        def to_document(self) -> dict[str, Any]:
            return {"all_pass": go, "checks": {"synthetic": go},
                    "aggregate_metrics": {"synthetic": True}}

    fake_api = SimpleNamespace(
        set_determinism=lambda seed: seed_calls.append(seed),
        evaluate_gates=lambda rows, flags: Gate())
    monkeypatch.setattr(backend, "_method_api", lambda: fake_api)

    def fake_fold(*args: Any, outer_fold: int, **kwargs: Any) -> Any:
        rows = tuple(SimpleNamespace(
            source_idv=idv, simulation_run=run, alarm=False, delay=(None if idv == 1 else 340),
            to_document=lambda: {}) for run in backend.FOLD_RUN_IDS[outer_fold]
                     for idv in backend.CONDITION_IDVS)
        return {
            "outer_fold": outer_fold, "numeric_valid": True,
            "artifacts_frozen_before_outer": {
                "transform_structure_sha256": f"{outer_fold + 1:064x}"},
            "authoritative_cuda_repeat": {
                "maximum_absolute_error": 0.0, "discrete_equal": True, "passes": True},
            "cross_device_report_only": {
                "is_go_gate": False, "completed": True,
                "maximum_absolute_error": 0.0, "discrete_equal": True},
        }, rows

    monkeypatch.setattr(backend, "_execute_fold", fake_fold)
    return seed_calls


@pytest.mark.parametrize(("go", "expected"), [(False, "DEVELOPMENT_NO_GO"), (True, "GO")])
def test_run_development_seeds_once_persists_decision_before_refit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, go: bool, expected: str,
) -> None:
    seed_calls = _patch_run_inputs(monkeypatch, go=go)
    refit_calls: list[bool] = []

    def fake_refit(*args: Any, artifact_root: Path, **kwargs: Any) -> dict[str, Any]:
        refit_calls.append(True)
        assert (artifact_root / "development_decision_before_final_refit.json").is_file()
        return {
            "schema": "gpu_minirocket_logodds_v8_final_mode2_refit",
            "candidate_id": backend.CANDIDATE_ID, "runs_per_class": 100,
            "created_only_after_development_go": True, "mode1_accessed": False}

    monkeypatch.setattr(backend, "_run_final_refit", fake_refit)
    result = backend.run_development(
        access_plan=backend.development_access_plan(), fold_map=_fold_map(),
        artifact_root=tmp_path / expected, protocol_canonical_json_sha256="b" * 64,
        protocol_path=tmp_path / "protocol.json",
        protocol_file_bytes_sha256="1" * 64,
        protocol_commit_file_bytes_sha256="2" * 64,
        claim_path=tmp_path / "claim.json", claim_file_bytes_sha256="d" * 64,
        claim_commit_file_bytes_sha256="3" * 64,
        access_intent_path=tmp_path / "intent.json",
        access_intent_file_bytes_sha256="e" * 64,
        access_intent_commit_file_bytes_sha256="f" * 64)
    assert result["decision"] == expected
    assert seed_calls == [backend.GLOBAL_SEED]
    assert refit_calls == ([True] if go else [])
    assert result["method_search_on_current_pool_terminated"] is (not go)
    state = result["backend_access_state"]["final_state"]
    assert state["folds_completed"] == 5 and state["backend_completed"] is True
    decision = json.loads(
        (tmp_path / expected / "development_decision_before_final_refit.json").read_text(
            encoding="utf-8"))
    assert decision["decision"] == expected
    assert decision["final_refit_started"] is False
    assert decision["final_model_usable"] is False


def test_numeric_no_go_writes_five_fold_decision_and_no_refit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_run_inputs(monkeypatch, go=False)
    monkeypatch.setattr(
        backend, "_execute_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            backend._FoldNumericNoGo("synthetic threshold degeneracy")))
    refit: list[bool] = []
    monkeypatch.setattr(backend, "_run_final_refit",
                        lambda *args, **kwargs: refit.append(True))
    root = tmp_path / "numeric"
    result = backend.run_development(
        access_plan=backend.development_access_plan(), fold_map=_fold_map(),
        artifact_root=root, protocol_canonical_json_sha256="c" * 64,
        protocol_path=tmp_path / "protocol.json",
        protocol_file_bytes_sha256="1" * 64,
        protocol_commit_file_bytes_sha256="2" * 64,
        claim_path=tmp_path / "claim.json", claim_file_bytes_sha256="d" * 64,
        claim_commit_file_bytes_sha256="3" * 64,
        access_intent_path=tmp_path / "intent.json",
        access_intent_file_bytes_sha256="e" * 64,
        access_intent_commit_file_bytes_sha256="f" * 64)
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert len(result["folds"]) == 5 and result["final_refit"] is None
    assert refit == []
    persisted = json.loads(
        (root / "development_decision_before_final_refit.json").read_text(encoding="utf-8"))
    assert persisted["decision"] == "DEVELOPMENT_NO_GO"
    assert persisted["no_go_terminates_current_development_pool_method_search"] is True
