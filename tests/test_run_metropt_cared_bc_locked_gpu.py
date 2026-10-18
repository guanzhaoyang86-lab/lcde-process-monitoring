from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch  # noqa: F401 -- Windows requires torch before NumPy
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_metropt_cared_bc_locked_gpu.py"
SPEC = importlib.util.spec_from_file_location("metropt_locked_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@dataclass
class SyntheticCase:
    config: Any
    hooks: Any
    protocol: Any
    development: Any
    calls: list[str]
    calibration_qualified_days: int = 41
    cuda_prediction_delta: float = 0.0
    cuda_physical_delta: float = 0.0
    omit_event: str | None = None
    fail_partition: str | None = None


class SyntheticEnsemble:
    dtype = "float64"

    @classmethod
    def load_npz(cls, path: Path) -> tuple["SyntheticEnsemble", dict[str, Any]]:
        assert path.read_bytes() == b"synthetic locked ensemble"
        return cls(), {"role": "synthetic_fixed_ensemble"}


class SyntheticScorer:
    threshold = None

    @classmethod
    def load_npz(cls, path: Path) -> tuple["SyntheticScorer", dict[str, Any]]:
        assert path.read_bytes() == b"synthetic locked scorer"
        return cls(), {"role": "synthetic_uncalibrated_scorer"}

    def score(
        self,
        residuals: Any,
        state_codes: Any,
        timestamps: Any,
        *,
        reset_before: Any,
    ) -> Any:
        residual = np.asarray(residuals, dtype=np.float64)
        target = np.asarray(residual[:, 0], dtype=np.float64)
        evidence = np.column_stack((target + 0.5, np.full(target.size, 0.5)))
        page = runner.dual_page_cusum(
            evidence,
            np.asarray(timestamps, dtype="datetime64[ns]"),
            reset_before=np.asarray(reset_before, dtype=np.bool_),
        )
        return SimpleNamespace(
            energy=np.abs(target),
            directional_glr=np.zeros(target.size, dtype=np.float64),
            standardized_evidence=evidence,
            cusum=page.cusum,
            envelope=page.envelope,
            reset=page.reset,
            alarm=page.alarm,
            threshold=None,
        )


def _timestamp_audit(workspace: Path) -> Any:
    sensor = next(item for item in runner.SOURCE_FILE_SPECS if item.key == "sensor_csv")
    return runner.MetroPTTimestampAudit(
        source_path=str(workspace / sensor.relative_path),
        source_size_bytes=sensor.size_bytes,
        source_mtime_ns=1,
        source_sha256=sensor.sha256,
        header=("Unnamed: 0", "timestamp", *runner.APPROVED_NUMERIC_COLUMNS, "LPS", "Oil_level"),
        row_count=sum(item.row_count for item in runner.PARTITIONS),
        first_timestamp=runner.PARTITIONS[0].start,
        last_timestamp="2020-08-31 23:59:59",
        minimum_interval_seconds=1.0,
        modal_interval_seconds=10.0,
        modal_interval_count=1,
        maximum_interval_seconds=60.0,
        chunks_scanned=1,
        partition_row_counts={item.name: item.row_count for item in runner.PARTITIONS},
    )


def _protocol_value(workspace: Path, output_root: Path, development: Any) -> dict[str, Any]:
    sensor = next(item for item in runner.SOURCE_FILE_SPECS if item.key == "sensor_csv")
    return {
        "protocol_id": "synthetic_metropt_locked_v1",
        "source_binding": {
            "source_fingerprint": development.source_fingerprint,
            "sensor_csv": {
                "path": sensor.relative_path,
                "size_bytes": sensor.size_bytes,
                "sha256": sensor.sha256,
            },
            "timestamp_audit": {
                "path": "synthetic_dev/timestamp_audit.json",
                "size_bytes": 1,
                "sha256": "1" * 64,
            },
        },
        "fixed_artifacts": {
            "ensemble": {
                "path": "synthetic_dev/cared_ensemble.npz",
                "size_bytes": development.ensemble.size_bytes,
                "sha256": development.ensemble.sha256,
            },
            "uncalibrated_scorer": {
                "path": "synthetic_dev/cared_scorer.npz",
                "size_bytes": development.scorer.size_bytes,
                "sha256": development.scorer.sha256,
                "threshold_must_be_null": True,
            },
        },
        "implementation_snapshot": {
            key: {"path": value, "size_bytes": 1, "sha256": "2" * 64}
            for key, value in {
                "runner": "scripts/run_metropt_cared_bc_locked_gpu.py",
                "data": "src/mate_spc_metropt/data.py",
                "governance": "src/mate_spc_metropt/governance.py",
                "model": "src/mate_spc_metropt/model.py",
                "locked_evaluation": "src/mate_spc_metropt/locked_evaluation.py",
                "portability": "src/mate_spc_metropt/portability.py",
            }.items()
        },
        "runtime": {
            "primary_device": "cuda", "comparison_device": "cpu",
            "dtype": "float64", "batch_size": 1024,
            "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "deterministic_algorithms": True,
        },
    }


def _minute_interval(
    case: SyntheticCase,
    partition_name: str,
    *,
    claim: Any,
    access_intent: Any,
    phase_manifest_fingerprint: str,
) -> Any:
    partition = runner._partition(partition_name)
    start = np.datetime64(partition.start, "m")
    end = np.datetime64(partition.end, "m")
    timestamps = np.arange(start, end, np.timedelta64(1, "m")).astype("datetime64[ns]")
    count = timestamps.size
    analog = np.zeros((count, 7), dtype=np.float64)
    controls = np.zeros((count, 6), dtype=np.float64)
    analog_mask = np.ones_like(analog, dtype=np.bool_)
    control_mask = np.ones_like(controls, dtype=np.bool_)
    reset = np.zeros(count, dtype=np.bool_)
    reset[0] = True
    rows = np.full(count, 6, dtype=np.int64)
    return runner.MetroPTMinuteInterval(
        timestamps=timestamps,
        analog=analog,
        controls=controls,
        analog_observed_mask=analog_mask,
        control_observed_mask=control_mask,
        control_imputed_mask=np.zeros_like(control_mask),
        bin_observed_mask=np.ones(count, dtype=np.bool_),
        coverage_fraction=np.ones(count, dtype=np.float64),
        eligible=np.ones(count, dtype=np.bool_),
        reset_before=reset,
        source_row_count_per_bin=rows,
        access_metadata={
            "numeric_columns_parsed": list(runner.APPROVED_NUMERIC_COLUMNS),
            "unrequested_numeric_values_parsed": False,
            "direct_alarm_values_parsed": False,
            "governed_partition_access": {
                "formal_governed_entry": True,
                "phase": claim.document.value["phase"],
                "partition": partition.as_dict(),
                "phase_manifest_fingerprint": phase_manifest_fingerprint,
                "workspace_ledger_fingerprint": claim.document.fingerprint,
                "access_intent_fingerprint": access_intent.fingerprint,
            },
        },
    )


def _selected_rows(case: SyntheticCase, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = timestamps[0].astype("datetime64[m]")
    end = timestamps[-1].astype("datetime64[m]") + np.timedelta64(1, "m")
    selected: list[int] = []
    targets: list[float] = []
    days = np.arange(start.astype("datetime64[D]"), end.astype("datetime64[D]"), dtype="datetime64[D]")
    is_a = start == np.datetime64("2020-03-01T00:00", "m")
    is_b = start == np.datetime64("2020-04-22T00:00", "m")
    if is_a or is_b:
        for day_index, day in enumerate(days):
            per_day = 1
            if is_a and day_index < case.calibration_qualified_days:
                per_day = 2
            for offset in range(per_day):
                timestamp = day.astype("datetime64[m]") + np.timedelta64(10 + offset, "m")
                selected.append(int((timestamp - start) / np.timedelta64(1, "m")))
                targets.append(float(day_index + 1 if is_a else 1.0))
    else:
        values: dict[int, float] = {}
        for day in days:
            for offset in (10, 11):
                timestamp = day.astype("datetime64[m]") + np.timedelta64(offset, "m")
                values[int((timestamp - start) / np.timedelta64(1, "m"))] = 1.0
        for event in runner.locked.CONFIRMATION_EVENTS:
            if case.omit_event == event.event_id:
                continue
            timestamp = np.datetime64(event.start, "m")
            values[int((timestamp - start) / np.timedelta64(1, "m"))] = 100.0
        selected = sorted(values)
        targets = [values[index] for index in selected]
    return np.asarray(selected, dtype=np.int64), np.asarray(targets, dtype=np.float64)


def _make_hooks(case: SyntheticCase) -> Any:
    def runtime_preflight(config: Any) -> dict[str, Any]:
        case.calls.append("preflight")
        return {
            "backend": "torch", "torch_version": "synthetic",
            "cuda_runtime_version": "synthetic", "primary_device": "cuda",
            "comparison_device": "cpu", "device_index": 0,
            "device_name": "synthetic GPU", "dtype": "float64",
            "batch_size": config.batch_size, "deterministic_algorithms": True,
            "cuda_matmul_tf32": False, "cudnn_tf32": False,
            "cuda_float64_smoke_test": True,
        }

    def load_partition(csv_path: Path, partition_name: str, **kwargs: Any) -> Any:
        case.calls.append(partition_name)
        if case.fail_partition == partition_name:
            raise RuntimeError(f"synthetic failure in {partition_name}")
        return _minute_interval(
            case, partition_name, claim=kwargs["claim"],
            access_intent=kwargs["access_intent"],
            phase_manifest_fingerprint=kwargs["phase_manifest_fingerprint"],
        )

    def build_windows(
        analog: np.ndarray, controls: np.ndarray, timestamps: np.ndarray, **kwargs: Any
    ) -> Any:
        selected, targets = _selected_rows(case, np.asarray(timestamps))
        return SimpleNamespace(
            inputs=np.zeros((selected.size, 10, 13), dtype=np.float64),
            selected=selected,
            targets=targets,
        )

    def predict(ensemble: Any, inputs: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
        result = np.zeros((inputs.shape[0], 3, 7), dtype=np.float64)
        if device == "cuda":
            result += case.cuda_prediction_delta
        return result

    def align(
        windows: Any, predictions: np.ndarray,
        analog: np.ndarray, controls: np.ndarray, timestamps: np.ndarray, **kwargs: Any,
    ) -> Any:
        count = windows.selected.size
        residual = np.zeros((count, 31), dtype=np.float64)
        residual[:, 0] = windows.targets
        residual[:, 1] = predictions[:, 0, 0]
        residual[:, -1] = (
            case.cuda_physical_delta if np.any(predictions != 0.0) else 0.0
        )
        reset = np.ones(count, dtype=np.bool_)
        return SimpleNamespace(
            residuals=residual,
            score_timestamps=np.asarray(timestamps)[windows.selected],
            score_indices=windows.selected,
            state_codes=np.zeros(count, dtype=np.int64),
            source_window_rows=np.zeros((count, 3), dtype=np.int64),
            reset_before=reset,
        )

    return runner.LockedHooks(
        load_partition=load_partition,
        build_inference_windows=build_windows,
        predict=predict,
        align=align,
        runtime_preflight=runtime_preflight,
    )


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    development_dir = workspace / "synthetic_dev"
    development_dir.mkdir()
    ensemble_path = development_dir / runner.ENSEMBLE_NAME
    scorer_path = development_dir / runner.SCORER_NAME
    ensemble_path.write_bytes(b"synthetic locked ensemble")
    scorer_path.write_bytes(b"synthetic locked scorer")
    marker = development_dir / "immutable_marker.json"
    marker.write_text("{}\n", encoding="utf-8")
    ensemble_record = runner.artifact_record(ensemble_path)
    scorer_record = runner.artifact_record(scorer_path)
    completion = runner.AuthenticatedDocument(
        value={"decision": runner.DEVELOPMENT_PASS},
        record={"filename": "development_completion.json", "size_bytes": 1, "sha256": "3" * 64},
        fingerprint="4" * 64,
        fingerprint_field="completion_fingerprint",
    )
    audit = _timestamp_audit(workspace)
    development = runner.DevelopmentPass(
        output_directory=development_dir,
        completion=completion,
        phase_manifest=runner.AuthenticatedDocument({}, {}, "5" * 64, "manifest_fingerprint"),
        result={"decision": runner.DEVELOPMENT_PASS},
        portability_audit={"passed": True},
        portability_delta=0.25,
        source_fingerprint="6" * 64,
        timestamp_audit=audit,
        ensemble=runner.FrozenArtifact(
            ensemble_path, ensemble_record["size_bytes"], ensemble_record["sha256"]
        ),
        scorer=runner.FrozenArtifact(
            scorer_path, scorer_record["size_bytes"], scorer_record["sha256"]
        ),
        directory_snapshot=runner._snapshot_directory(development_dir),
    )
    output_root = workspace / "outputs" / "locked"
    protocol_value = _protocol_value(workspace, output_root, development)
    protocol = runner.AuthenticatedDocument(
        value=protocol_value,
        record={"filename": "final.json", "size_bytes": 1, "sha256": "f" * 64},
    )
    config = runner.LockedRunConfig(
        csv_path=workspace / "missing_metropt.csv",
        final_protocol_path=workspace / "protocols" / "final.json",
        final_protocol_sidecar_path=workspace / "protocols" / "final.sha256",
        expected_final_protocol_sha256="f" * 64,
        output_root=output_root,
        workspace_root=workspace,
        phase="all",
    )
    case = SyntheticCase(config, None, protocol, development, [])
    case.hooks = _make_hooks(case)
    monkeypatch.setattr(runner, "CaredEnsemble", SyntheticEnsemble)
    monkeypatch.setattr(runner, "CaredBCScorer", SyntheticScorer)
    monkeypatch.setattr(runner, "_authenticate_final_protocol", lambda config: protocol)
    monkeypatch.setattr(
        runner, "_authenticate_development_pass", lambda config, final_protocol: development
    )
    monkeypatch.setattr(runner.locked, "MINIMUM_SCORES_PER_DAY", 2)
    return case


def _strict_protocol_document(case: SyntheticCase) -> Any:
    workspace = case.config.workspace_root
    implementation_paths = {
        "runner": "scripts/run_metropt_cared_bc_locked_gpu.py",
        "data": "src/mate_spc_metropt/data.py",
        "governance": "src/mate_spc_metropt/governance.py",
        "model": "src/mate_spc_metropt/model.py",
        "locked_evaluation": "src/mate_spc_metropt/locked_evaluation.py",
        "portability": "src/mate_spc_metropt/portability.py",
    }
    implementation: dict[str, Any] = {}
    for name, relative in implementation_paths.items():
        source = ROOT / relative
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        record = runner.artifact_record(target)
        implementation[name] = {
            "path": relative,
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
    file_record = {"path": "synthetic_dev/x.json", "size_bytes": 1, "sha256": "1" * 64}
    fingerprinted = {**file_record, "manifest_fingerprint": "2" * 64}
    artifact_manifest = {**file_record, "artifact_manifest_fingerprint": "3" * 64}
    completion = {
        **file_record, "completion_fingerprint": "4" * 64,
        "decision": runner.DEVELOPMENT_PASS,
    }
    value = {
        "schema": runner.FINAL_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": "synthetic_metropt_locked_v1",
        "protocol_version": "1.0.0-test",
        "status": runner.FINAL_PROTOCOL_STATUS,
        "disclosure": {
            "created_after_observing_development": True,
            "development_values_observed": True,
            "calibration_numeric_values_observed": False,
            "confirmation_numeric_values_observed": False,
            "model_and_scorer_frozen": True,
            "same_data_retuning_forbidden": True,
        },
        "development_pass_binding": {
            "output_directory": "synthetic_dev",
            "protocol": {
                "path": "protocols/dev.json", "sidecar_path": "protocols/dev.sha256",
                "sha256": runner.PORTABILITY_V1_1_DEVELOPMENT_SHA256,
            },
            "phase_manifest": fingerprinted,
            "artifact_manifest": artifact_manifest,
            "completion": completion,
            "result": file_record,
            "portability_audit": file_record,
            "portability_evidence": {"path": "synthetic_dev/x.npz", "size_bytes": 1, "sha256": "1" * 64},
            "gpu_execution": file_record,
            "artifact_names": [
                runner.ENSEMBLE_NAME, runner.SCORER_NAME, "portability_evidence.npz",
                "portability_audit.json", "development_result.json",
            ],
            "global_envelope_delta": 0.25,
            "evidence_semantic_sha256": "5" * 64,
            "v1_projection_semantic_sha256": "6" * 64,
        },
        "source_binding": case.protocol.value["source_binding"],
        "fixed_artifacts": case.protocol.value["fixed_artifacts"],
        "implementation_snapshot": implementation,
        "runtime": case.protocol.value["runtime"],
        "cpu_cuda_primitive_parity": {
            "complete_sequences_required": True,
            "prediction_shape_tail": [3, 7],
            "residual_dimension": 31,
            "forecast_residual_dimensions": 21,
            "physical_residual_dimensions": 10,
            "prediction_max_abs_error": 1.0e-9,
            "forecast_residual_max_abs_error": 1.0e-9,
            "physical_residual_exact": True,
            "timestamps_exact": True,
            "indices_exact": True,
            "state_codes_exact": True,
            "reset_before_exact": True,
            "source_window_rows_exact": True,
            "score_reset_exact": True,
        },
        "calibration": {
            "partitions": list(runner.CALIBRATION_PARTITIONS),
            "candidate_complete_days": 70,
            "minimum_qualified_days": 39,
            "minimum_scores_per_qualified_day": 1141,
            "alpha": 0.05,
            "primary_boundary_source": "cuda",
            "rank_formula": "ceil((n+1)*(1-0.05))",
            "effective_boundary_formula": "nextafter(raw_boundary + development_global_envelope_delta + 1e-9, +infinity)",
            "complete_cpu_cuda_inference_required": True,
            "cpu_cuda_alarm_arrays_exact": True,
            "minimum_boundary_distance_strictly_greater_than": 1.0e-9,
        },
        "confirmation": {
            "partition": "confirmation",
            "start_inclusive": runner.locked.CONFIRMATION_START,
            "end_exclusive": runner.locked.CONFIRMATION_END,
            "fault_event_ids": ["F2", "F3", "F4"],
            "fault_event_intervals": [
                {"event_id": event.event_id, "start": event.start, "end": event.end}
                for event in runner.locked.CONFIRMATION_EVENTS
            ],
            "recovery_exclusion_hours": 72,
            "availability_lag_minutes": 1,
            "alarm_event_semantics": "false_to_true_onset_only",
            "persistent_pre_event_alarm_does_not_count": True,
            "event_detection_requirement": "3_of_3",
            "maximum_each_delay_minutes": 60.0,
            "maximum_median_delay_minutes": 30.0,
            "minimum_qualified_healthy_days": 39,
            "minimum_scores_per_qualified_day": 1141,
            "maximum_healthy_day_alarm_rate": 0.05,
            "maximum_healthy_day_cp95_upper": 0.10,
            "maximum_healthy_time_alarm_fraction": 0.01,
            "cpu_cuda_alarm_arrays_exact": True,
            "threshold_adjustment_forbidden": True,
        },
        "output_contract": {
            "calibration_directory": runner._workspace_relative(
                case.config.phase_directory(runner.PHASE_CALIBRATION), workspace
            ),
            "confirmation_directory": runner._workspace_relative(
                case.config.phase_directory(runner.PHASE_CONFIRMATION), workspace
            ),
            "create_new_only": True,
            "completed_reentry_reads_csv": False,
            "calibration_and_confirmation_separate_single_access": True,
            "runtime_preflight_intent_loader_binding": (
                "locked_runner_authenticated_numeric_authorization"
            ),
        },
    }
    return runner.AuthenticatedDocument(
        value=value,
        record={"filename": "final.json", "size_bytes": 1, "sha256": "f" * 64},
    )


def _bomb(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("completed reentry invoked a runtime or CSV hook")


def test_full_two_phase_pass_and_completed_reentry_reads_no_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    outcome = runner._run_locked_experiment_for_test(case.config, hooks=case.hooks)
    assert outcome.calibration is not None
    assert outcome.confirmation is not None
    assert outcome.calibration.decision == runner.CALIBRATION_PASS
    assert outcome.confirmation.decision == runner.CONFIRMATION_PASS
    assert case.calls == ["preflight", "calibration_a", "calibration_b", "preflight", "confirmation"]
    assert not case.config.csv_path.exists()

    forbidden = runner.LockedHooks(
        load_partition=_bomb,
        build_inference_windows=_bomb,
        predict=_bomb,
        align=_bomb,
        runtime_preflight=_bomb,
    )
    reused = runner._run_locked_experiment_for_test(case.config, hooks=forbidden)
    assert reused.calibration is not None and reused.calibration.reused_completed_bundle
    assert reused.confirmation is not None and reused.confirmation.reused_completed_bundle
    assert case.calls == ["preflight", "calibration_a", "calibration_b", "preflight", "confirmation"]


def test_preflight_failure_is_before_intent_and_numeric_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.hooks = replace(case.hooks, runtime_preflight=_bomb)
    with pytest.raises(AssertionError, match="runtime or CSV"):
        runner._run_calibration_for_test(case.config, hooks=case.hooks)
    output = case.config.output_root / runner.PHASE_CALIBRATION
    assert (output / runner.PREACCESS_ABORT_NAME).is_file()
    abort = runner.authenticate_fingerprinted_json(
        output / runner.PREACCESS_ABORT_NAME,
        fingerprint_field="completion_fingerprint",
    )
    assert abort.value["numeric_access_occurred"] is False
    assert abort.value["decision"] == runner.CALIBRATION_ABORT
    assert not (output / runner.INTENT_NAME[runner.PHASE_CALIBRATION]).exists()
    assert not (output / runner.COMPLETION_NAME[runner.PHASE_CALIBRATION]).exists()
    assert case.calls == []


def test_insufficient_calibration_days_aborts_and_keeps_confirmation_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.calibration_qualified_days = 38
    outcome = runner._run_locked_experiment_for_test(case.config, hooks=case.hooks)
    assert outcome.calibration is not None
    assert outcome.calibration.decision == runner.CALIBRATION_ABORT
    assert outcome.confirmation is None
    assert not (case.config.output_root / runner.PHASE_CONFIRMATION).exists()
    assert case.calls == ["preflight", "calibration_a", "calibration_b"]
    with pytest.raises(runner.AccessDeniedError, match="did not PASS"):
        runner._run_confirmation_for_test(case.config, hooks=case.hooks)
    assert case.calls == ["preflight", "calibration_a", "calibration_b"]


def test_cpu_cuda_primitive_difference_aborts_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.cuda_prediction_delta = 2.0e-9
    outcome = runner._run_calibration_for_test(case.config, hooks=case.hooks)
    assert outcome.decision == runner.CALIBRATION_ABORT
    parity = outcome.result["analysis"]["cpu_cuda_parity"]
    assert not parity["calibration_a"]["primitive_parity_passed"]


def test_physical_residual_requires_bitwise_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.cuda_prediction_delta = np.nextafter(0.0, 1.0)
    case.cuda_physical_delta = 1.0e-15
    outcome = runner._run_calibration_for_test(case.config, hooks=case.hooks)
    assert outcome.decision == runner.CALIBRATION_ABORT
    exact = outcome.result["analysis"]["cpu_cuda_parity"]["calibration_a"]["exact_checks"]
    assert exact["physical_residual_exact"] is False


def test_confirmation_failure_is_final_and_does_not_change_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.omit_event = "F4"
    outcome = runner._run_locked_experiment_for_test(case.config, hooks=case.hooks)
    assert outcome.calibration is not None and outcome.calibration.decision == runner.CALIBRATION_PASS
    assert outcome.confirmation is not None and outcome.confirmation.decision == runner.CONFIRMATION_FAIL
    result = outcome.confirmation.result
    assert result["analysis"]["locked_evaluation"]["events"]["detected_event_count"] == 2
    assert result["analysis"]["threshold_adjusted"] is False
    certificate = runner.read_strict_json(
        case.config.output_root / runner.PHASE_CALIBRATION / runner.CERTIFICATE_NAME
    )
    assert result["analysis"]["effective_boundary"] == certificate["effective_boundary"]


def test_numeric_exception_creates_authentic_abort_and_never_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case.fail_partition = "calibration_b"
    with pytest.raises(RuntimeError, match="calibration_b"):
        runner._run_calibration_for_test(case.config, hooks=case.hooks)
    assert case.calls == ["preflight", "calibration_a", "calibration_b"]
    forbidden = runner.LockedHooks(
        load_partition=_bomb,
        build_inference_windows=_bomb,
        predict=_bomb,
        align=_bomb,
        runtime_preflight=_bomb,
    )
    reused = runner._run_calibration_for_test(case.config, hooks=forbidden)
    assert reused.decision == runner.CALIBRATION_ABORT
    assert reused.reused_completed_bundle
    assert case.calls == ["preflight", "calibration_a", "calibration_b"]


def test_pickle_free_evidence_semantic_tamper_is_rejected_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    passed = runner._run_calibration_for_test(case.config, hooks=case.hooks)
    assert passed.decision == runner.CALIBRATION_PASS
    output = case.config.output_root / runner.PHASE_CALIBRATION
    evidence = output / runner.EVIDENCE_NAME[runner.PHASE_CALIBRATION]
    with np.load(evidence, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["calibration_a__cuda__energy"][0] += 1.0
    np.savez_compressed(evidence, **payload)

    manifest_path = output / runner.ARTIFACT_MANIFEST_NAME[runner.PHASE_CALIBRATION]
    completion_path = output / runner.COMPLETION_NAME[runner.PHASE_CALIBRATION]
    manifest_path.unlink()
    completion_path.unlink()
    ledger = runner.authenticate_fingerprinted_json(
        output / runner.LEDGER_NAME[runner.PHASE_CALIBRATION],
        fingerprint_field="ledger_fingerprint",
    )
    claim = runner.LedgerClaim(ledger, fresh=False)
    intent = runner.authenticate_fingerprinted_json(
        output / runner.INTENT_NAME[runner.PHASE_CALIBRATION],
        fingerprint_field="intent_fingerprint",
    )
    manifest = runner.create_artifact_manifest(
        manifest_path, output, phase=runner.PHASE_CALIBRATION,
        protocol_id=case.protocol.value["protocol_id"],
        protocol_sha256=case.protocol.record["sha256"],
        phase_manifest_fingerprint=ledger.value["phase_manifest_fingerprint"],
        access_intent=intent,
        artifact_names=runner._normal_artifact_names(runner.PHASE_CALIBRATION),
    )
    runner.create_completion_record(
        completion_path, claim=claim, access_intent=intent,
        artifact_manifest=manifest, decision=runner.CALIBRATION_PASS,
    )
    forbidden = replace(case.hooks, load_partition=_bomb, runtime_preflight=_bomb)
    with pytest.raises(runner.IntegrityError, match="scorer-reproducible"):
        runner._run_calibration_for_test(case.config, hooks=forbidden)
    assert case.calls == ["preflight", "calibration_a", "calibration_b"]


def test_runner_ast_contains_no_training_or_refit_calls() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    forbidden = {"fit", "train", "backward", "optimizer", "build_causal_windows"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called.add(function.id)
            elif isinstance(function, ast.Attribute):
                called.add(function.attr)
    assert not (called & forbidden)
    assert "build_causal_input_windows" in RUNNER_PATH.read_text(encoding="utf-8")


def test_unfrozen_protocol_sha_fails_before_any_output_or_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    invalid = replace(case.config, expected_final_protocol_sha256="0" * 64)
    with pytest.raises(runner.AccessDeniedError, match="real frozen"):
        runner._run_locked_experiment_for_test(invalid, hooks=case.hooks)
    assert case.calls == []
    assert not invalid.output_root.exists()


def test_final_protocol_schema_is_strict_and_binds_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    document = _strict_protocol_document(case)
    runner._validate_final_protocol(document, case.config)
    altered = dict(document.value)
    altered["unexpected_after_freeze"] = True
    with pytest.raises(runner.IntegrityError, match="fields changed"):
        runner._validate_final_protocol(replace(document, value=altered), case.config)


def test_formal_entry_has_no_injectable_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    assert "hooks" not in inspect.signature(runner.run_locked_experiment).parameters
    assert "hooks" not in inspect.signature(runner.run_calibration).parameters
    assert "hooks" not in inspect.signature(runner.run_confirmation).parameters
    with pytest.raises(TypeError, match="hooks"):
        runner.run_calibration(case.config, hooks=case.hooks)
    assert case.calls == []
    assert not case.config.output_root.exists()


def test_output_overlap_fails_before_hook_or_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    before = runner._snapshot_directory(case.development.output_directory)
    overlapping = replace(
        case.config,
        output_root=case.development.output_directory / "locked_child",
    )
    with pytest.raises(runner.AccessDeniedError, match="overlap"):
        runner._run_locked_experiment_for_test(overlapping, hooks=case.hooks)
    assert case.calls == []
    assert not overlapping.output_root.exists()
    assert runner._snapshot_directory(case.development.output_directory) == before


def test_post_create_authentication_error_has_superseding_abort_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    authenticate = runner.authenticate_completion_record
    attempts = 0

    def fail_first_authentication(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise runner.IntegrityError("synthetic post-create authentication failure")
        return authenticate(*args, **kwargs)

    monkeypatch.setattr(
        runner, "authenticate_completion_record", fail_first_authentication
    )
    with pytest.raises(runner.IntegrityError, match="post-create"):
        runner._run_calibration_for_test(case.config, hooks=case.hooks)
    output = case.config.phase_directory(runner.PHASE_CALIBRATION)
    assert (output / runner.COMPLETION_NAME[runner.PHASE_CALIBRATION]).is_file()
    assert (output / runner.ABORT_COMPLETION_NAME[runner.PHASE_CALIBRATION]).is_file()
    forbidden = replace(case.hooks, load_partition=_bomb, runtime_preflight=_bomb)
    reused = runner._run_calibration_for_test(case.config, hooks=forbidden)
    assert reused.decision == runner.CALIBRATION_ABORT
    assert reused.reused_completed_bundle
