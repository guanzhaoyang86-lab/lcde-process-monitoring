from __future__ import annotations

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

import torch  # noqa: F401 -- import before numpy on the Windows CUDA environment
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_metropt_cared_bc_portability_gpu.py"
SPEC = importlib.util.spec_from_file_location("metropt_portability_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@dataclass
class SyntheticCase:
    config: Any
    hooks: Any
    calls: list[str]
    prior_snapshot: dict[str, dict[str, Any]]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_prior(workspace: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    output = workspace / "prior"
    output.mkdir()
    sources = {
        "sensor_csv": {
            "relative_path": "missing.csv",
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
    }
    source_fingerprint = runner.canonical_fingerprint(sources)
    phase_body = {
        "schema": "synthetic_prior_phase",
        "schema_version": 1,
        "phase": runner.PHASE_DEVELOPMENT,
        "protocol_id": runner.CHARTER_ID,
        "charter_sha256": runner.DEVELOPMENT_CHARTER_SHA256,
        "output_directory_relative_path": "prior",
        "ledger_workspace_relative_path": "prior/development_access_ledger.json",
        "sensor_csv_relative_path": "missing.csv",
        "source_fingerprint": source_fingerprint,
        "sources": sources,
    }
    phase = runner.create_fingerprinted_json(
        output / runner.PHASE_MANIFEST_NAME,
        phase_body,
        fingerprint_field="manifest_fingerprint",
    )
    ledger_arguments = {
        "phase": runner.PHASE_DEVELOPMENT,
        "workspace_relative_path": "prior/development_access_ledger.json",
        "protocol_id": runner.CHARTER_ID,
        "charter_sha256": runner.DEVELOPMENT_CHARTER_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": "a" * 64,
        "source_fingerprint": source_fingerprint,
        "phase_manifest_fingerprint": phase.fingerprint,
        "output_directory": output,
    }
    claim = runner.claim_workspace_access_ledger(
        output / runner.LEDGER_NAME, **ledger_arguments
    )
    intent = runner.create_phase_access_intent(
        output / runner.INTENT_NAME,
        claim,
        phase_manifest_fingerprint=phase.fingerprint,
    )

    audit = runner.MetroPTTimestampAudit(
        source_path=str(workspace / "missing.csv"),
        source_size_bytes=1,
        source_mtime_ns=1,
        source_sha256="a" * 64,
        header=("timestamp",),
        row_count=2,
        first_timestamp="2020-02-21T00:00:00",
        last_timestamp="2020-04-18T00:00:00",
        minimum_interval_seconds=60.0,
        modal_interval_seconds=60.0,
        modal_interval_count=1,
        maximum_interval_seconds=60.0,
        chunks_scanned=1,
        partition_row_counts={"dev_healthy": 1, "dev_fault_f1": 1},
    )
    runner.atomic_create_new_json(output / "timestamp_audit.json", audit.as_dict())
    (output / runner.ENSEMBLE_NAME).write_bytes(b"synthetic frozen ensemble")
    (output / runner.SCORER_NAME).write_bytes(b"synthetic frozen scorer")
    gates = {
        name: {"passed": True}
        for name in (
            "tcn_relative_improvement",
            "f1_alarm",
            "f1_first_alarm_delay_minutes",
            "f1_maximum_to_threshold_ratio",
            "direct_alarm_channels_excluded",
            "feature_and_score_causality",
        )
    }
    prior_result = {
        "schema": "synthetic_prior_result",
        "phase": runner.PHASE_DEVELOPMENT,
        "decision": runner.DEVELOPMENT_ABORT,
        "phase_manifest_fingerprint": phase.fingerprint,
        "provisional_threshold": {"value": 36.02052930938596},
        "training": {
            "relative_improvement_over_persistence": 0.5025773655903889
        },
        "gates": gates,
    }
    runner.atomic_create_new_json(output / runner.RESULT_NAME, prior_result)
    runner.atomic_create_new_json(
        output / runner.GPU_LOG_NAME,
        {"schema": "synthetic_prior_gpu", "parity_failed": True},
    )
    manifest = runner.create_artifact_manifest(
        output / runner.ARTIFACT_MANIFEST_NAME,
        output,
        phase=runner.PHASE_DEVELOPMENT,
        protocol_id=runner.CHARTER_ID,
        protocol_sha256=runner.DEVELOPMENT_CHARTER_SHA256,
        phase_manifest_fingerprint=phase.fingerprint,
        access_intent=intent,
        artifact_names=runner.PRIOR_ARTIFACT_NAMES,
    )
    completion = runner.create_completion_record(
        output / runner.COMPLETION_NAME,
        claim=claim,
        access_intent=intent,
        artifact_manifest=manifest,
        decision=runner.DEVELOPMENT_ABORT,
    )
    expectation = runner.PriorBundleExpectation(
        output_relative_path="prior",
        phase_manifest_sha256=str(phase.record["sha256"]),
        phase_manifest_fingerprint=str(phase.fingerprint),
        artifact_manifest_sha256=str(manifest.record["sha256"]),
        artifact_manifest_fingerprint=str(manifest.fingerprint),
        completion_sha256=str(completion.record["sha256"]),
        completion_fingerprint=str(completion.fingerprint),
        result_sha256=runner.sha256_file(output / runner.RESULT_NAME),
        gpu_execution_sha256=runner.sha256_file(output / runner.GPU_LOG_NAME),
        timestamp_audit_sha256=runner.sha256_file(output / "timestamp_audit.json"),
        ensemble=runner.FrozenFile(
            runner.ENSEMBLE_NAME,
            (output / runner.ENSEMBLE_NAME).stat().st_size,
            runner.sha256_file(output / runner.ENSEMBLE_NAME),
        ),
        scorer=runner.FrozenFile(
            runner.SCORER_NAME,
            (output / runner.SCORER_NAME).stat().st_size,
            runner.sha256_file(output / runner.SCORER_NAME),
        ),
    )
    return expectation, runner._snapshot_directory(output)


def _make_protocol(
    workspace: Path,
    expectation: Any,
    *,
    output_relative: str = "new",
) -> str:
    value = deepcopy(
        json.loads(
            (ROOT / runner.DEFAULT_PROTOCOL_PATH).read_text(encoding="utf-8")
        )
    )
    value["parent_bindings"]["v0_1_parity_abort"] = {
        "output_directory": expectation.output_relative_path,
        "decision": runner.DEVELOPMENT_ABORT,
        "phase_manifest_sha256": expectation.phase_manifest_sha256,
        "phase_manifest_fingerprint": expectation.phase_manifest_fingerprint,
        "artifact_manifest_sha256": expectation.artifact_manifest_sha256,
        "artifact_manifest_fingerprint": expectation.artifact_manifest_fingerprint,
        "completion_sha256": expectation.completion_sha256,
        "completion_fingerprint": expectation.completion_fingerprint,
        "result_sha256": expectation.result_sha256,
        "gpu_execution_sha256": expectation.gpu_execution_sha256,
    }
    value["fixed_artifacts"]["ensemble"] = {
        "path": f"{expectation.output_relative_path}/{runner.ENSEMBLE_NAME}",
        "size_bytes": expectation.ensemble.size_bytes,
        "sha256": expectation.ensemble.sha256,
    }
    value["fixed_artifacts"]["uncalibrated_scorer"] = {
        "path": f"{expectation.output_relative_path}/{runner.SCORER_NAME}",
        "size_bytes": expectation.scorer.size_bytes,
        "sha256": expectation.scorer.sha256,
        "threshold_must_be_null": True,
    }
    value["numeric_access"]["expected_counts"] = {
        "dev_healthy": {"inference_windows": 4, "aligned_scores": 3},
        "dev_fault_f1": {"inference_windows": 4, "aligned_scores": 3},
    }
    value["output_contract"]["directory"] = output_relative
    protocol = workspace / "protocol.json"
    _write_json(protocol, value)
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    (workspace / "protocol.sha256").write_text(
        f"{digest}  protocol.json\n", encoding="ascii"
    )
    return digest


def _interval(name: str) -> Any:
    start = np.datetime64(
        "2020-02-21T00:00:00" if name == "dev_healthy" else "2020-04-18T00:00:00",
        "ns",
    )
    timestamps = start + np.arange(8) * np.timedelta64(1, "m")
    analog = np.zeros((8, 7), dtype=np.float64)
    analog[:, 0] = 1.0 if name == "dev_healthy" else 2.0
    reset = np.zeros(8, dtype=bool)
    reset[0] = True
    return SimpleNamespace(
        timestamps=timestamps,
        analog=analog,
        controls=np.zeros((8, 6), dtype=np.float64),
        eligible=np.ones(8, dtype=bool),
        reset_before=reset,
        access_metadata={
            "source_sha256": "a" * 64,
            "unrequested_numeric_values_parsed": False,
            "direct_alarm_values_parsed": False,
            "numeric_columns_parsed": list(runner.APPROVED_NUMERIC_COLUMNS),
        },
    )


def _hooks(workspace: Path, calls: list[str]) -> Any:
    class Ensemble:
        dtype = "float64"

    class Scorer:
        threshold = None

    def load_partition(csv_path: Path, name: str, **kwargs: Any) -> Any:
        assert csv_path == workspace / "missing.csv"
        assert not csv_path.exists()
        assert (workspace / "new" / runner.PHASE_MANIFEST_NAME).is_file()
        assert (workspace / "new" / runner.LEDGER_NAME).is_file()
        assert (workspace / "new" / runner.INTENT_NAME).is_file()
        assert kwargs["claim"].may_authorize_numeric_access
        calls.append(f"load:{name}")
        return _interval(name)

    def build_windows(
        analog: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        del controls, timestamps, kwargs
        marker = int(analog[0, 0])
        calls.append(f"windows:{marker}")
        return SimpleNamespace(inputs=np.full((4, 10, 13), marker, dtype=np.float64))

    def predict(
        ensemble: Any,
        inputs: np.ndarray,
        *,
        device: str,
        batch_size: int,
    ) -> np.ndarray:
        del ensemble
        assert batch_size == 1024
        calls.append(f"predict:{int(inputs[0, 0, 0])}:{device}")
        offset = 0.0 if device == "cpu" else 5.0e-10
        return np.full((inputs.shape[0], 3, 7), offset, dtype=np.float64)

    def align(
        windows: Any,
        prediction: np.ndarray,
        analog: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        del windows, controls, kwargs
        marker = int(analog[0, 0])
        residual = np.zeros((3, 31), dtype=np.float64)
        residual[:, :21] = float(prediction[0, 0, 0])
        reset = np.array([True, False, False], dtype=bool)
        return SimpleNamespace(
            residuals=residual,
            prediction_residuals=residual[:, :21],
            physical_features=residual[:, 21:],
            score_indices=np.array([2, 3, 4], dtype=np.int64),
            score_timestamps=timestamps[2:5],
            state_codes=np.full(3, marker, dtype=np.int64),
            source_window_rows=np.arange(9, dtype=np.int64).reshape(3, 3),
            reset_before=reset,
        )

    def score(
        scorer: Any,
        residuals: np.ndarray,
        state_codes: np.ndarray,
        timestamps: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        del scorer, residuals, kwargs
        is_fault = int(state_codes[0]) == 2
        envelope = (
            np.array([0.0, 50.0, 80.0])
            if is_fault
            else np.array([0.0, 1.0, 2.0])
        )
        return SimpleNamespace(envelope=envelope, timestamps=timestamps)

    return runner.PortabilityHooks(
        load_partition=load_partition,
        load_ensemble=lambda path: (Ensemble(), {"loaded": path.name}),
        load_scorer=lambda path: (Scorer(), {"loaded": path.name}),
        build_inference_windows=build_windows,
        predict=predict,
        align=align,
        score=score,
        runtime_metadata=lambda config: {
            "primary_device": config.device,
            "comparison_device": config.comparison_device,
            "device_name": "synthetic RTX 5090",
            "dtype": config.dtype,
            "batch_size": config.batch_size,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
        },
        portability_gate=runner.audit_development_portability,
    )


@pytest.fixture
def synthetic_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticCase:
    expectation, prior_snapshot = _make_prior(tmp_path)
    digest = _make_protocol(tmp_path, expectation)
    monkeypatch.setattr(runner, "PORTABILITY_DEVELOPMENT_SHA256", digest)
    calls: list[str] = []
    config = runner.PortabilityRunConfig(
        csv_path=tmp_path / "missing.csv",
        output_directory=tmp_path / "new",
        workspace_root=tmp_path,
        protocol_path=tmp_path / "protocol.json",
        protocol_sidecar_path=tmp_path / "protocol.sha256",
        expected_protocol_sha256=digest,
        prior_output_directory=tmp_path / "prior",
        prior_expectation=expectation,
    )
    return SyntheticCase(config, _hooks(tmp_path, calls), calls, prior_snapshot)


def test_pass_runs_complete_both_backends_and_creates_authenticated_chain(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case
    outcome = runner.run_portability_development(case.config, hooks=case.hooks)

    assert outcome.decision == runner.DEVELOPMENT_PASS
    assert outcome.reused_completed_bundle is False
    assert case.calls[:2] == ["load:dev_healthy", "load:dev_fault_f1"]
    assert [item for item in case.calls if item.startswith("load:")] == [
        "load:dev_healthy",
        "load:dev_fault_f1",
    ]
    assert [item for item in case.calls if item.startswith("predict:")] == [
        "predict:1:cpu",
        "predict:1:cuda",
        "predict:2:cpu",
        "predict:2:cuda",
    ]
    result = runner.read_strict_json(case.config.output_directory / runner.RESULT_NAME)
    assert result["fit_partition_loaded"] is False
    assert result["development_gate_passed"] is True
    assert result["calibration_numeric_access_authorized_by_this_result"] is False
    assert result["portability_audit"]["exact_backend_checks"]["passed"] is True
    assert result["portability_audit"]["alarm_array_hashes"]["dev_fault_f1"]["cpu"]["count"] == 2
    assert runner._snapshot_directory(case.config.prior_output_directory) == case.prior_snapshot
    assert runner.sha256_file(case.config.output_directory / runner.ENSEMBLE_NAME) == case.config.prior_expectation.ensemble.sha256
    assert runner.sha256_file(case.config.output_directory / runner.SCORER_NAME) == case.config.prior_expectation.scorer.sha256
    with np.load(case.config.output_directory / runner.EVIDENCE_NAME, allow_pickle=False) as evidence:
        assert evidence["dev_fault_f1__cpu__alarm"].tolist() == [False, True, True]
        assert float(evidence["h_eff"]) > 36.02052930938596


def test_completed_reentry_performs_zero_numeric_or_model_calls(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case
    first = runner.run_portability_development(case.config, hooks=case.hooks)
    assert first.decision == runner.DEVELOPMENT_PASS

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("completed reentry crossed a numerical hook")

    forbidden_hooks = runner.PortabilityHooks(
        load_partition=forbidden,
        load_ensemble=forbidden,
        load_scorer=forbidden,
        build_inference_windows=forbidden,
        predict=forbidden,
        align=forbidden,
        score=forbidden,
        runtime_metadata=forbidden,
        portability_gate=forbidden,
    )
    second = runner.run_portability_development(case.config, hooks=forbidden_hooks)
    assert second.reused_completed_bundle is True
    assert second.decision == runner.DEVELOPMENT_PASS
    assert not case.config.csv_path.exists()


def test_scientific_gate_failure_is_a_real_authenticated_abort(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case

    def fail_gate(**kwargs: Any) -> dict[str, Any]:
        value = dict(runner.audit_development_portability(**kwargs))
        value["passed"] = False
        value["decision"] = "FAIL_DEVELOPMENT_PORTABILITY_AUDIT"
        return value

    hooks = runner.PortabilityHooks(
        **{
            **case.hooks.__dict__,
            "portability_gate": fail_gate,
        }
    )
    outcome = runner.run_portability_development(case.config, hooks=hooks)
    assert outcome.decision == runner.DEVELOPMENT_ABORT
    assert outcome.result["next_method_if_failed"] == "MB-CycleMIL-BC"
    assert outcome.result["next_phase_status"] == "CALIBRATION_AND_CONFIRMATION_REMAIN_SEALED"
    assert (case.config.output_directory / runner.COMPLETION_NAME).is_file()
    assert not (case.config.output_directory / runner.ABORT_NAME).exists()


def test_numeric_exception_leaves_auditable_abort_and_reentry_uses_no_hooks(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case

    def broken_score(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic scorer failure")

    broken = runner.PortabilityHooks(
        **{**case.hooks.__dict__, "score": broken_score}
    )
    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        runner.run_portability_development(case.config, hooks=broken)
    output = case.config.output_directory
    assert (output / runner.INTENT_NAME).is_file()
    assert (output / runner.ABORT_NAME).is_file()
    assert (output / runner.COMPLETION_NAME).is_file()
    abort = runner.read_strict_json(output / runner.ABORT_NAME)
    assert abort["decision"] == runner.DEVELOPMENT_ABORT
    assert abort["calibration_numeric_access_authorized"] is False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ABORT reentry crossed a numerical hook")

    no_numeric = runner.PortabilityHooks(*([forbidden] * 9))
    reused = runner.run_portability_development(case.config, hooks=no_numeric)
    assert reused.reused_completed_bundle is True
    assert reused.decision == runner.DEVELOPMENT_ABORT


def test_protocol_authentication_failure_precedes_output_and_numeric_access(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case
    case.config.protocol_sidecar_path.write_text(
        f"{'0' * 64}  protocol.json\n", encoding="ascii"
    )
    with pytest.raises(runner.IntegrityError):
        runner.run_portability_development(case.config, hooks=case.hooks)
    assert not case.config.output_directory.exists()
    assert case.calls == []
    assert runner._snapshot_directory(case.config.prior_output_directory) == case.prior_snapshot


def test_prior_npz_tamper_fails_before_output_and_is_never_repaired(
    synthetic_case: SyntheticCase,
) -> None:
    case = synthetic_case
    target = case.config.prior_output_directory / runner.ENSEMBLE_NAME
    target.write_bytes(target.read_bytes() + b"tamper")
    tampered = target.read_bytes()
    with pytest.raises(runner.IntegrityError):
        runner.run_portability_development(case.config, hooks=case.hooks)
    assert target.read_bytes() == tampered
    assert not case.config.output_directory.exists()
    assert case.calls == []


def test_runner_has_no_estimation_or_backward_api_surface() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "train_cared_ensemble" not in imported | called_names
    assert "fit_cared_scorer" not in imported | called_names
    assert "build_causal_windows" not in imported | called_names
    assert "fit" not in called_attributes
    assert "backward" not in called_attributes
