from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_cp_etc_scan_gpu_v5_1.py"
)


@pytest.fixture
def runner():
    name = "run_cp_etc_scan_gpu_v5_1_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_sets_cublas_and_imports_torch_before_backend():
    source = RUNNER_PATH.read_text("utf-8")
    assert source.index('os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG"') < source.index("import torch")
    assert source.index("import torch") < source.index("from mate_spc_cpetc import backend_v5")


def test_live_protocol_base_amendment_and_fold_binding_verify_without_h5(
    runner, monkeypatch: pytest.MonkeyPatch,
):
    backend = runner._load_backend()
    h5py_before = sys.modules.get("h5py")
    if h5py_before is not None:
        monkeypatch.setattr(
            h5py_before,
            "File",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("protocol verification must not open HDF5")
            ),
        )
    bundle = runner._verify_protocol_bundle(backend)
    assert bundle["base"]["sha256"] == runner.BASE_PROTOCOL_SHA256
    assert bundle["amendment"]["sha256"] == runner.AMENDMENT_SHA256
    assert sys.modules.get("h5py") is h5py_before


class FakeBackend:
    BACKEND_SCHEMA = "synthetic_backend"
    PROTOCOL_ID = "cp-etc-scan-v5.1-mode2-development"
    CANDIDATE_ID = "cp-etc-scan-v5-mode2-fixed"
    EXPECTED_FOLD_MAP_SHA256 = "f" * 64
    EXPECTED_ACCESS_PLAN_SHA256 = "unset"

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.output_root: Path | None = None
        self.observed_intent = False
        self.calls = 0

    def cuda_preflight(self):
        return {"cuda_available": True, "device_type": "cuda"}

    def development_access_plan(self):
        return [{"synthetic": True}]

    def validate_development_access_plan(self, plan):
        assert plan == [{"synthetic": True}]

    def _canonical_fold_map(self):
        return {run: (run - 1) % 5 for run in range(1, 101)}

    def development_iteration(self, *, access_plan, fold_map, artifact_root, device):
        self.calls += 1
        assert access_plan == [{"synthetic": True}]
        assert len(fold_map) == 100
        assert device == "cuda"
        self.observed_intent = all(
            (artifact_root.parent / name).is_file()
            for name in (
                "access_intent.json", "access_intent.json.sha256",
                "access_intent.json.commit.json",
            )
        )
        assert self.observed_intent
        if self.fail:
            raise RuntimeError("synthetic failure")
        artifact_root.mkdir(parents=True)
        return {
            "schema": self.BACKEND_SCHEMA,
            "protocol_id": self.PROTOCOL_ID,
            "protocol_sha256": "5" * 64,
            "source_sha256": "8" * 64,
            "candidate_id": self.CANDIDATE_ID,
            "candidate_count": 1,
            "fold_map_sha256": self.EXPECTED_FOLD_MAP_SHA256,
            "mode1_accessed": False,
            "decision": "DEVELOPMENT_NO_GO",
            "gates": {"all_pass": False},
            "selected_candidate_id": None,
            "formal_mode2_refit": None,
            "folds": [],
        }


def _synthetic_preflight(runner, backend):
    return {
        "schema": "cp_etc_scan_v5_1_preflight_v1",
        "created_utc": "synthetic",
        "protocols": {},
        "prior_governance": {},
        "source": {
            "path": "synthetic.h5", "filename": "TEP_Mode2.h5",
            "size_bytes": 1, "sha256": runner.SOURCE_SHA256,
            "hdf5_opened": False,
        },
        "runtime": backend.cuda_preflight(),
        "gpu": {"synthetic": True},
        "pre_numeric_tests": {
            "returncode": 0, "passed": 1, "skipped": 0,
            "failed": 0, "errors": 0,
        },
        "environment": {},
        "implementation": {},
        "git": {"commit": None, "dirty": None},
        "hdf5_opened": False,
    }


def _bind_synthetic_plan(runner, backend, monkeypatch):
    digest = runner._hash_bytes(
        runner._canonical_bytes(backend.development_access_plan())
    )
    backend.EXPECTED_ACCESS_PLAN_SHA256 = digest
    monkeypatch.setattr(runner, "ACCESS_PLAN_SHA256", digest)


def test_preflight_can_be_mocked_without_constructing_archive(runner, monkeypatch):
    backend = FakeBackend()
    _bind_synthetic_plan(runner, backend, monkeypatch)
    monkeypatch.setattr(runner, "_verify_protocol_bundle", lambda _backend: {})
    monkeypatch.setattr(
        runner, "_source_record",
        lambda: {"path": "synthetic", "sha256": runner.SOURCE_SHA256, "hdf5_opened": False},
    )
    monkeypatch.setattr(runner, "_implementation_records", lambda: {})
    monkeypatch.setattr(runner, "_governance_records", lambda: {})
    monkeypatch.setattr(runner, "_gpu_manifest", lambda: {"synthetic": True})
    monkeypatch.setattr(
        runner, "_hdf5_library_manifest",
        lambda: {"h5py": "synthetic", "hdf5_library": "synthetic", "file_opened": False},
    )
    monkeypatch.setattr(
        runner, "_pre_numeric_test_evidence",
        lambda: {
            "returncode": 0, "passed": 1, "skipped": 0,
            "failed": 0, "errors": 0, "warnings": 0,
        },
    )
    monkeypatch.setattr(runner, "_git_revision", lambda: {"commit": None, "dirty": None})
    result = runner.preflight(backend=backend)
    assert result["hdf5_opened"] is False
    assert backend.calls == 0


def test_development_commits_intent_before_backend_and_writes_hashes(
    runner, monkeypatch, tmp_path: Path,
):
    backend = FakeBackend()
    _bind_synthetic_plan(runner, backend, monkeypatch)
    monkeypatch.setattr(
        runner, "preflight", lambda *, backend: _synthetic_preflight(runner, backend)
    )
    monkeypatch.setattr(
        runner, "_validate_development_result", lambda raw, _backend: dict(raw)
    )
    summary = runner.run_development(
        output_root=tmp_path, iteration_id="synthetic_no_go", backend=backend,
    )
    directory = Path(summary["directory"])
    assert backend.observed_intent is True
    assert summary["decision"] == "DEVELOPMENT_NO_GO"
    assert (directory / "access_intent.json").is_file()
    assert (directory / "access_intent.json.sha256").is_file()
    assert (directory / "access_intent.json.commit.json").is_file()
    assert (directory / "development_result.json").is_file()
    assert (directory / "development_result.json.sha256").is_file()
    assert (tmp_path / "development_access.claim.json").is_file()
    completion = json.loads((directory / "completion.json").read_text("utf-8"))
    assert completion["mode1_accessed"] is False
    with pytest.raises(FileExistsError):
        runner.run_development(
            output_root=tmp_path, iteration_id="synthetic_no_go", backend=backend,
        )
    with pytest.raises(FileExistsError):
        runner.run_development(
            output_root=tmp_path, iteration_id="different_id_forbidden", backend=backend,
        )


def test_failure_preserves_access_intent_and_emits_failure_record(
    runner, monkeypatch, tmp_path: Path,
):
    backend = FakeBackend(fail=True)
    _bind_synthetic_plan(runner, backend, monkeypatch)
    monkeypatch.setattr(
        runner, "preflight", lambda *, backend: _synthetic_preflight(runner, backend)
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_development(
            output_root=tmp_path, iteration_id="synthetic_failure", backend=backend,
        )
    directory = tmp_path / "development_iterations" / "synthetic_failure"
    assert (directory / "access_intent.json").is_file()
    failure = json.loads((directory / "failure.json").read_text("utf-8"))
    assert failure["error_type"] == "RuntimeError"
    assert failure["metrics_revealed"] is False
    assert failure["automatic_rerun_allowed"] is False
    assert not (directory / "development_result.json").exists()


def test_result_validator_rejects_missing_full_replay(runner):
    backend = FakeBackend()
    raw = {
        "schema": backend.BACKEND_SCHEMA,
        "protocol_id": backend.PROTOCOL_ID,
        "protocol_sha256": runner.AMENDMENT_SHA256,
        "source_sha256": runner.SOURCE_SHA256,
        "candidate_id": backend.CANDIDATE_ID,
        "candidate_count": 1,
        "fold_map_sha256": backend.EXPECTED_FOLD_MAP_SHA256,
        "mode1_accessed": False,
        "decision": "DEVELOPMENT_NO_GO",
        "gates": {"all_pass": False},
        "selected_candidate_id": None,
        "formal_mode2_refit": None,
        "folds": [],
    }
    with pytest.raises(ValueError, match="five folds"):
        runner._validate_development_result(raw, backend)
