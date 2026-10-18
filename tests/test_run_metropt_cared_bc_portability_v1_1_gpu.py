from __future__ import annotations

import torch  # noqa: F401 -- torch-first Windows import order

import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest

from mate_spc_metropt.governance import (
    authenticate_workspace_access_ledger,
    create_artifact_manifest,
    create_completion_record,
)
from mate_spc_metropt.model import (
    AlignedResidualBatch,
    CaredBCScorer,
    DirectionalGLRModel,
    InferenceWindowBatch,
    RankNormalTransform,
    StateConditionalReference,
)
from mate_spc_metropt.portability_replay import EvidenceReplayError


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_metropt_cared_bc_portability_v1_1_gpu.py"
SPEC = importlib.util.spec_from_file_location("metropt_portability_v11_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@dataclass
class Case:
    config: Any
    boundary: Any
    preflight: Any
    calls: list[str]


def _scorer() -> CaredBCScorer:
    n = 31
    identity = np.eye(n, dtype=np.float64)
    reference = StateConditionalReference(
        np.array([0], dtype=np.int64), np.array([100], dtype=np.int64),
        np.zeros((1, n)), identity[None], identity[None],
        np.zeros(n), identity, identity,
    )
    vector = np.ones(n, dtype=np.float64) / np.sqrt(n)
    direction = DirectionalGLRModel(
        np.array([0], dtype=np.int64), np.array([10], dtype=np.int64),
        vector[None], vector,
    )
    rank = RankNormalTransform(np.column_stack((np.linspace(0, 100, 64), np.linspace(0, 20, 64))))
    return CaredBCScorer(reference, direction, rank, threshold=None)


def _metadata(*, scorer: bool = False) -> dict[str, Any]:
    value = {
        "charter_sha256": runner.legacy_v1.DEVELOPMENT_CHARTER_SHA256,
        "development_only": True,
        "phase": runner.PHASE_DEVELOPMENT,
        "protocol_id": runner.legacy_v1.CHARTER_ID,
        "source_fingerprint": "340805938a72af9c1f9273a3c75b90a246b11b6aea7bc254c6d0e7103602d705",
        "phase_manifest_fingerprint": "d26b87f46f7125fdb90ea8a07622e1438833171aadd3a7f4f0cd2a14f3970943",
    }
    if scorer:
        value.update({"threshold_status": "uncalibrated", "provisional_threshold_saved_in_result_only": True})
    return value


def _write_protocol(workspace: Path) -> str:
    value = deepcopy(json.loads((ROOT / runner.DEFAULT_PROTOCOL_PATH).read_text(encoding="utf-8")))
    value["numeric_access"]["expected_counts"] = {
        "dev_healthy": {"inference_windows": 4, "aligned_scores": 3},
        "dev_fault_f1": {"inference_windows": 4, "aligned_scores": 3},
    }
    value["output_contract"]["directory"] = "new"
    protocol = workspace / "protocol.json"
    protocol.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    (workspace / "protocol.sha256").write_text(f"{digest}  protocol.json\n", encoding="ascii")
    return digest


def _interval(name: str, sensor_sha: str) -> Any:
    start = np.datetime64("2020-04-17" if name == "dev_healthy" else "2020-04-18", "ns")
    timestamps = start + np.arange(8) * np.timedelta64(1, "m")
    reset = np.zeros(8, dtype=bool)
    reset[0] = True
    marker = 0.05 if name == "dev_healthy" else 2.0
    return SimpleNamespace(
        timestamps=timestamps,
        analog=np.full((8, 7), marker, dtype=np.float64),
        controls=np.zeros((8, 6), dtype=np.float64),
        eligible=np.ones(8, dtype=bool), reset_before=reset,
        access_metadata={
            "source_sha256": sensor_sha,
            "unrequested_numeric_values_parsed": False,
            "direct_alarm_values_parsed": False,
            "numeric_columns_parsed": list(runner.APPROVED_NUMERIC_COLUMNS),
        },
    )


def _boundary(workspace: Path, sensor_sha: str, calls: list[str]) -> Any:
    def load(csv_path: Path, name: str, **kwargs: Any) -> Any:
        assert csv_path == workspace / "missing.csv"
        assert not csv_path.exists()
        assert (workspace / "new" / runner.PREFLIGHT_NAME).is_file()
        assert (workspace / "new" / runner.INTENT_NAME).is_file()
        assert kwargs["runtime_preflight_path"].is_file()
        calls.append(f"load:{name}")
        return _interval(name, sensor_sha)

    def windows(analog: np.ndarray, controls: np.ndarray, timestamps: np.ndarray, **kwargs: Any) -> InferenceWindowBatch:
        del controls, kwargs
        return InferenceWindowBatch(
            np.repeat(np.concatenate((analog[:1], np.zeros((1, 6))), axis=1)[None], 40, axis=0).reshape(4, 10, 13),
            np.arange(4, dtype=np.int64), timestamps[:4],
        )

    def predict(ensemble: Any, inputs: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
        del ensemble, batch_size
        calls.append(f"predict:{device}")
        offset = 0.0 if device == "cpu" else 1.0e-12
        return np.full((inputs.shape[0], 3, 7), float(inputs[0, 0, 0]) + offset)

    def align(window: Any, prediction: np.ndarray, analog: np.ndarray, controls: np.ndarray, timestamps: np.ndarray, **kwargs: Any) -> AlignedResidualBatch:
        del window, controls, kwargs
        residual = np.zeros((3, 31), dtype=np.float64)
        residual[:, :21] = prediction[0, 0, 0]
        residual[:, 21:] = analog[0, 0]
        return AlignedResidualBatch(
            residual, residual[:, :21], residual[:, 21:],
            np.arange(3, dtype=np.int64), timestamps[:3],
            np.zeros(3, dtype=np.int64), np.arange(9, dtype=np.int64).reshape(3, 3),
            np.array([True, False, False], dtype=bool),
        )
    return SimpleNamespace(
        load_partition=load, build_inference_windows=windows,
        predict=predict, align=align,
        passed=True, score=lambda *a, **k: {"passed": True},
    )


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    digest = _write_protocol(tmp_path)
    monkeypatch.setattr(runner, "PORTABILITY_V1_1_DEVELOPMENT_SHA256", digest)
    v0, v1 = tmp_path / "v0", tmp_path / "v1"
    v0.mkdir()
    v1.mkdir()
    (v0 / "marker.bin").write_bytes(b"immutable-v0")
    ensemble_bytes, scorer_bytes = b"synthetic-ensemble", b"synthetic-scorer"
    (v1 / runner.ENSEMBLE_NAME).write_bytes(ensemble_bytes)
    (v1 / runner.SCORER_NAME).write_bytes(scorer_bytes)
    bindings = dict(runner.V1_FILE_BINDINGS)
    for name in (runner.ENSEMBLE_NAME, runner.SCORER_NAME):
        path = v1 / name
        bindings[name] = (path.stat().st_size, runner.sha256_file(path))
    monkeypatch.setattr(runner, "V1_FILE_BINDINGS", bindings)

    sensor_sha = "a" * 64
    sources = {"sensor_csv": {"relative_path": "missing.csv", "size_bytes": 1, "sha256": sensor_sha}}
    audit = runner.MetroPTTimestampAudit(
        source_path=str(tmp_path / "missing.csv"), source_size_bytes=1,
        source_mtime_ns=1, source_sha256=sensor_sha, header=("timestamp",),
        row_count=2, first_timestamp="2020-04-17T00:00:00",
        last_timestamp="2020-04-18T00:00:00", minimum_interval_seconds=60.0,
        modal_interval_seconds=60.0, modal_interval_count=1,
        maximum_interval_seconds=60.0, chunks_scanned=1,
        partition_row_counts={"dev_healthy": 1, "dev_fault_f1": 1},
    )

    def parents(config: Any) -> Any:
        return runner._ParentEvidence(
            timestamp_audit=audit, sources=sources,
            source_fingerprint=runner.canonical_fingerprint(sources),
            sensor_csv_relative_path="missing.csv",
            v0_1_snapshot=runner._snapshot_directory(v0),
            v0_1_completion_fingerprint="b" * 64,
            v1_snapshot=runner._snapshot_directory(v1),
            v1_completion_fingerprint="c" * 64,
        )

    monkeypatch.setattr(runner, "_authenticate_parents", parents)
    synthetic_scorer = _scorer()
    monkeypatch.setattr(
        runner.CaredEnsemble,
        "load_npz",
        classmethod(lambda cls, path: (SimpleNamespace(dtype="float64"), _metadata())),
    )
    monkeypatch.setattr(
        runner.CaredBCScorer,
        "load_npz",
        classmethod(lambda cls, path: (synthetic_scorer, _metadata(scorer=True))),
    )
    calls: list[str] = []

    def preflight(config: Any) -> dict[str, Any]:
        del config
        assert (tmp_path / "new" / runner.LEDGER_NAME).is_file()
        assert not (tmp_path / "new" / runner.INTENT_NAME).exists()
        calls.append("preflight")
        return {
            "python_version": "3.11.14", "numpy_version": "2.4.0",
            "pandas_version": "2.3.3", "torch_version": "2.10.0.dev20251124+cu128",
            "cuda_runtime_version": "12.8", "device_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "compute_capability": [12, 0], "primary_device": "cuda",
            "comparison_device": "cpu", "dtype": "float64", "batch_size": 1024,
            "deterministic_algorithms": True, "cuda_matmul_tf32": False,
            "cudnn_tf32": False, "cuda_float64_smoke_value": 14.0,
            "torch_cuda_available": True,
        }

    config = runner.PortabilityV11RunConfig(
        csv_path=tmp_path / "missing.csv", output_directory=tmp_path / "new",
        workspace_root=tmp_path, protocol_path=tmp_path / "protocol.json",
        protocol_sidecar_path=tmp_path / "protocol.sha256",
        expected_protocol_sha256=digest, v0_1_output_directory=v0,
        v1_output_directory=v1,
        v1_protocol_path=tmp_path / "unused-v1.json",
        v1_protocol_sidecar_path=tmp_path / "unused-v1.sha256",
    )
    return Case(config, _boundary(tmp_path, sensor_sha, calls), preflight, calls)


def test_preflight_precedes_intent_and_each_partition_loads_once(case: Case) -> None:
    outcome = runner._run_v1_1_core(
        case.config, boundary=case.boundary, preflight_probe=case.preflight
    )
    assert outcome.decision == runner.DEVELOPMENT_ABORT
    assert case.calls[:3] == ["preflight", "load:dev_healthy", "load:dev_fault_f1"]
    assert case.calls.count("load:dev_healthy") == 1
    assert case.calls.count("load:dev_fault_f1") == 1
    intent = runner.read_strict_json(case.config.output_directory / runner.INTENT_NAME)
    preflight = runner.authenticate_fingerprinted_json(
        case.config.output_directory / runner.PREFLIGHT_NAME,
        fingerprint_field="preflight_fingerprint",
    )
    assert intent["runtime_preflight_fingerprint"] == preflight.fingerprint
    assert intent["permissions"]["fit_permitted"] is False
    assert intent["permissions"]["selection_permitted"] is False
    assert intent["single_access"] is True
    assert not (case.config.output_directory / runner.ABORT_NAME).exists()


def test_malicious_passed_and_score_attributes_cannot_forge_pass(case: Case) -> None:
    assert case.boundary.passed is True
    assert case.boundary.score()["passed"] is True
    outcome = runner._run_v1_1_core(
        case.config, boundary=case.boundary, preflight_probe=case.preflight
    )
    assert outcome.decision == runner.DEVELOPMENT_ABORT
    result = runner.read_strict_json(case.config.output_directory / runner.RESULT_NAME)
    assert result["development_gate_passed"] is False
    assert result["next_method_if_failed"] == "MB-CycleMIL-BC"


def test_failed_preflight_creates_authenticated_abort_before_loader(case: Case) -> None:
    def failed(config: Any) -> dict[str, Any]:
        del config
        case.calls.append("failed-preflight")
        raise RuntimeError("synthetic CUDA failure")

    outcome = runner._run_v1_1_core(
        case.config, boundary=case.boundary, preflight_probe=failed
    )
    assert outcome.decision == runner.DEVELOPMENT_ABORT
    assert case.calls == ["failed-preflight"]
    assert (case.config.output_directory / runner.COMPLETION_NAME).is_file()
    assert (case.config.output_directory / runner.INTENT_NAME).is_file()
    abort = runner.read_strict_json(case.config.output_directory / runner.ABORT_NAME)
    assert abort["reason"] == "RUNTIME_PREFLIGHT_FAILED_BEFORE_NUMERIC_ACCESS"

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("completed ABORT reentry crossed a numerical seam")

    reused = runner._run_v1_1_core(
        case.config,
        boundary=SimpleNamespace(
            load_partition=forbidden, build_inference_windows=forbidden,
            predict=forbidden, align=forbidden,
        ),
        preflight_probe=forbidden,
    )
    assert reused.reused_completed_bundle is True
    assert reused.decision == runner.DEVELOPMENT_ABORT


def test_self_rehashed_stage_tamper_is_rejected_by_semantic_reentry(case: Case) -> None:
    first = runner._run_v1_1_core(
        case.config, boundary=case.boundary, preflight_probe=case.preflight
    )
    assert first.decision == runner.DEVELOPMENT_ABORT
    output = case.config.output_directory
    evidence_path = output / runner.EVIDENCE_NAME
    with np.load(evidence_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    key = "dev_healthy__cpu__energy"
    payload[key][0] = np.nextafter(payload[key][0], np.inf)
    evidence_path.unlink()
    np.savez_compressed(evidence_path, **payload)

    # Rebuild a byte-authentic chain around the malicious evidence. Semantic
    # replay must still reject it.
    (output / runner.COMPLETION_NAME).unlink()
    (output / runner.ARTIFACT_MANIFEST_NAME).unlink()
    ledger_doc = runner.authenticate_fingerprinted_json(
        output / runner.LEDGER_NAME, fingerprint_field="ledger_fingerprint"
    )
    claim = runner.LedgerClaim(document=ledger_doc, fresh=False)
    intent = runner.authenticate_fingerprinted_json(
        output / runner.INTENT_NAME, fingerprint_field="intent_fingerprint"
    )
    phase = runner.authenticate_fingerprinted_json(
        output / runner.PHASE_MANIFEST_NAME, fingerprint_field="manifest_fingerprint"
    )
    manifest = create_artifact_manifest(
        output / runner.ARTIFACT_MANIFEST_NAME,
        output,
        phase=runner.PHASE_DEVELOPMENT,
        protocol_id=runner.PORTABILITY_V1_1_DEVELOPMENT_ID,
        protocol_sha256=runner.PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        phase_manifest_fingerprint=str(phase.fingerprint),
        access_intent=intent,
        artifact_names=runner.NORMAL_ARTIFACT_NAMES,
    )
    create_completion_record(
        output / runner.COMPLETION_NAME,
        claim=claim,
        access_intent=intent,
        artifact_manifest=manifest,
        decision=runner.DEVELOPMENT_ABORT,
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("completed reentry crossed a CSV or inference seam")

    with pytest.raises(EvidenceReplayError, match="scorer recomputation"):
        runner._run_v1_1_core(
            case.config,
            boundary=SimpleNamespace(
                load_partition=forbidden, build_inference_windows=forbidden,
                predict=forbidden, align=forbidden,
            ),
            preflight_probe=forbidden,
        )


@pytest.mark.parametrize(
    "tamper",
    ["schema", "runtime", "execution_order", "metadata", "negative_timing"],
)
def test_self_rehashed_execution_record_tamper_is_rejected_on_reentry(
    case: Case, tamper: str
) -> None:
    first = runner._run_v1_1_core(
        case.config, boundary=case.boundary, preflight_probe=case.preflight
    )
    assert first.decision == runner.DEVELOPMENT_ABORT
    output = case.config.output_directory
    execution_path = output / runner.GPU_LOG_NAME
    execution = runner.read_strict_json(execution_path)
    if tamper == "schema":
        execution["schema"] = "forged_execution"
    elif tamper == "runtime":
        execution["runtime"] = {**execution["runtime"], "dtype": "float32"}
    elif tamper == "execution_order":
        execution["full_sequence_executions"][0], execution["full_sequence_executions"][1] = (
            execution["full_sequence_executions"][1],
            execution["full_sequence_executions"][0],
        )
    elif tamper == "metadata":
        execution["scorer_metadata"] = {
            **execution["scorer_metadata"], "threshold_status": "forged"
        }
    elif tamper == "negative_timing":
        execution["full_sequence_executions"][0]["prediction_seconds"] = -1.0
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(tamper)
    execution_path.unlink()
    runner.atomic_create_new_json(execution_path, execution)

    # Rebuild the byte-hash chain to prove that authentication alone cannot
    # make semantically forged execution evidence acceptable.
    (output / runner.COMPLETION_NAME).unlink()
    (output / runner.ARTIFACT_MANIFEST_NAME).unlink()
    ledger_doc = runner.authenticate_fingerprinted_json(
        output / runner.LEDGER_NAME, fingerprint_field="ledger_fingerprint"
    )
    claim = runner.LedgerClaim(document=ledger_doc, fresh=False)
    intent = runner.authenticate_fingerprinted_json(
        output / runner.INTENT_NAME, fingerprint_field="intent_fingerprint"
    )
    phase = runner.authenticate_fingerprinted_json(
        output / runner.PHASE_MANIFEST_NAME, fingerprint_field="manifest_fingerprint"
    )
    manifest = create_artifact_manifest(
        output / runner.ARTIFACT_MANIFEST_NAME,
        output,
        phase=runner.PHASE_DEVELOPMENT,
        protocol_id=runner.PORTABILITY_V1_1_DEVELOPMENT_ID,
        protocol_sha256=runner.PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        phase_manifest_fingerprint=str(phase.fingerprint),
        access_intent=intent,
        artifact_names=runner.NORMAL_ARTIFACT_NAMES,
    )
    create_completion_record(
        output / runner.COMPLETION_NAME,
        claim=claim,
        access_intent=intent,
        artifact_manifest=manifest,
        decision=first.decision,
    )

    with pytest.raises(runner.IntegrityError, match="execution"):
        runner._run_v1_1_core(
            case.config, boundary=case.boundary, preflight_probe=case.preflight
        )


def test_manifest_finalization_failure_becomes_authenticated_abort(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner._create_manifest
    calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic manifest commit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_create_manifest", fail_once)
    with pytest.raises(RuntimeError, match="manifest commit failure"):
        runner._run_v1_1_core(
            case.config, boundary=case.boundary, preflight_probe=case.preflight
        )
    output = case.config.output_directory
    completion = runner.authenticate_fingerprinted_json(
        output / runner.COMPLETION_NAME, fingerprint_field="completion_fingerprint"
    )
    abort = runner.read_strict_json(output / runner.ABORT_NAME)
    assert completion.value["decision"] == runner.DEVELOPMENT_ABORT
    assert abort["reason"] == "FINALIZATION_FAILED"


def test_completion_atomic_failure_leaves_nonreusable_fail_closed_partial(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_completion(*args: Any, **kwargs: Any) -> Any:
        raise OSError("synthetic completion atomic failure")

    monkeypatch.setattr(runner, "create_completion_record", fail_completion)
    with pytest.raises(OSError, match="completion atomic failure"):
        runner._run_v1_1_core(
            case.config, boundary=case.boundary, preflight_probe=case.preflight
        )
    output = case.config.output_directory
    assert (output / runner.ARTIFACT_MANIFEST_NAME).is_file()
    assert not (output / runner.COMPLETION_NAME).exists()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("partial reentry crossed a numerical seam")

    with pytest.raises(runner.PortabilityV11RunnerError, match="partial"):
        runner._run_v1_1_core(
            case.config,
            boundary=SimpleNamespace(
                load_partition=forbidden, build_inference_windows=forbidden,
                predict=forbidden, align=forbidden,
            ),
            preflight_probe=forbidden,
        )


def test_public_runner_has_no_injectable_gate_score_or_fit_surface() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    public = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_portability_v1_1_development"
    )
    assert [argument.arg for argument in public.args.args] == ["config"]
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "train_cared_ensemble" not in imported
    assert "fit_cared_scorer" not in imported
    assert "fit" not in called_attributes
    assert "backward" not in called_attributes
    assert not hasattr(runner._NumericalBoundary, "portability_gate")
