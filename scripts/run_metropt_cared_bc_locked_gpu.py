#!/usr/bin/env python
"""Sealed calibration and locked confirmation for MetroPT CARED-BC.

The final analysis protocol is intentionally not embedded here: callers must
provide its JSON, sidecar, and expected SHA-256 explicitly.  The protocol must
bind an authenticated v1_1 development PASS, its fixed uncalibrated ensemble
and scorer, the timestamp-only source audit, and this implementation snapshot.

No training or parameter-selection API is imported.  Calibration and
confirmation use separate, non-reusable governed access ledgers.  Completed
phase reentry reconstructs the scientific result from saved numeric evidence
without opening the MetroPT CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

# Import torch before NumPy on the Windows CUDA runtime (shm.dll ordering).
import torch
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mate_spc_metropt import locked_evaluation as locked  # noqa: E402
from mate_spc_metropt.data import (  # noqa: E402
    ANALOG_COLUMNS,
    APPROVED_NUMERIC_COLUMNS,
    SAFE_CONTROL_COLUMNS,
    MetroPTMinuteInterval,
    MetroPTTimestampAudit,
    load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (  # noqa: E402
    CALIBRATION_ABORT,
    CALIBRATION_PASS,
    CONFIRMATION_FAIL,
    CONFIRMATION_PASS,
    DEVELOPMENT_CHARTER_SHA256,
    DEVELOPMENT_PASS,
    PARTITIONS,
    PHASE_CALIBRATION,
    PHASE_CONFIRMATION,
    PHASE_DEVELOPMENT,
    PORTABILITY_V1_1_DEVELOPMENT_ID,
    PORTABILITY_V1_1_DEVELOPMENT_SHA256,
    SOURCE_FILE_SPECS,
    AccessDeniedError,
    AuthenticatedDocument,
    GovernanceError,
    IntegrityError,
    LedgerClaim,
    artifact_record,
    atomic_create_new_json,
    authenticate_completion_record,
    authenticate_fingerprinted_json,
    authenticate_json_with_sidecar,
    authenticate_phase_access_intent,
    build_workspace_access_ledger,
    canonical_fingerprint,
    claim_workspace_access_ledger,
    create_artifact_manifest,
    create_completion_record,
    create_fingerprinted_json,
    create_phase_access_intent,
    read_strict_json,
    sha256_file,
)
from mate_spc_metropt.model import (  # noqa: E402
    CaredBCScorer,
    CaredEnsemble,
    align_inference_forecasts_to_score_times,
    build_causal_input_windows,
    dual_page_cusum,
)


SCHEMA_VERSION = 1
FINAL_PROTOCOL_SCHEMA = "metropt_cared_bc_locked_analysis_protocol"
FINAL_PROTOCOL_STATUS = "FROZEN_BEFORE_CALIBRATION_NUMERIC_ACCESS"
UNFROZEN_SHA256 = "UNFROZEN_FINAL_PROTOCOL_SHA256_REQUIRED"

CALIBRATION_PARTITIONS = ("calibration_a", "calibration_b")
CONFIRMATION_PARTITIONS = ("confirmation",)
DEVICES = ("cpu", "cuda")
PREDICTION_TOLERANCE = 1.0e-9
PHYSICAL_RESIDUAL_START = 21
BOUNDARY_DISTANCE = 1.0e-9

PHASE_MANIFEST_NAME = {
    PHASE_CALIBRATION: "calibration_phase_manifest.json",
    PHASE_CONFIRMATION: "confirmation_phase_manifest.json",
}
LEDGER_NAME = {
    PHASE_CALIBRATION: "calibration_access_ledger.json",
    PHASE_CONFIRMATION: "confirmation_access_ledger.json",
}
INTENT_NAME = {
    PHASE_CALIBRATION: "calibration_access_intent.json",
    PHASE_CONFIRMATION: "confirmation_access_intent.json",
}
EVIDENCE_NAME = {
    PHASE_CALIBRATION: "calibration_evidence.npz",
    PHASE_CONFIRMATION: "confirmation_evidence.npz",
}
ANALYSIS_NAME = {
    PHASE_CALIBRATION: "calibration_analysis.json",
    PHASE_CONFIRMATION: "confirmation_analysis.json",
}
RESULT_NAME = {
    PHASE_CALIBRATION: "calibration_result.json",
    PHASE_CONFIRMATION: "confirmation_result.json",
}
GPU_LOG_NAME = {
    PHASE_CALIBRATION: "gpu_execution_calibration.json",
    PHASE_CONFIRMATION: "gpu_execution_confirmation.json",
}
PREFLIGHT_NAME = {
    PHASE_CALIBRATION: "runtime_preflight_calibration.json",
    PHASE_CONFIRMATION: "runtime_preflight_confirmation.json",
}
NUMERIC_AUTHORIZATION_NAME = {
    PHASE_CALIBRATION: "calibration_numeric_authorization.json",
    PHASE_CONFIRMATION: "confirmation_numeric_authorization.json",
}
ARTIFACT_MANIFEST_NAME = {
    PHASE_CALIBRATION: "calibration_artifact_manifest.json",
    PHASE_CONFIRMATION: "confirmation_artifact_manifest.json",
}
COMPLETION_NAME = {
    PHASE_CALIBRATION: "calibration_completion.json",
    PHASE_CONFIRMATION: "confirmation_completion.json",
}
ABORT_ARTIFACT_MANIFEST_NAME = {
    PHASE_CALIBRATION: "calibration_abort_artifact_manifest.json",
    PHASE_CONFIRMATION: "confirmation_abort_artifact_manifest.json",
}
ABORT_COMPLETION_NAME = {
    PHASE_CALIBRATION: "calibration_abort_completion.json",
    PHASE_CONFIRMATION: "confirmation_abort_completion.json",
}
CERTIFICATE_NAME = "calibration_certificate.json"
ENSEMBLE_NAME = "cared_ensemble.npz"
SCORER_NAME = "cared_scorer.npz"
ABORT_NAME = "ABORTED.json"
PREACCESS_ABORT_NAME = "PREACCESS_ABORTED.json"


class LockedRunnerError(RuntimeError):
    """Raised when the frozen two-stage run cannot proceed safely."""


@dataclass(frozen=True)
class LockedRunConfig:
    csv_path: Path
    final_protocol_path: Path
    final_protocol_sidecar_path: Path
    expected_final_protocol_sha256: str
    output_root: Path = WORKSPACE_ROOT / "outputs/metropt_cared_bc_locked_gpu"
    workspace_root: Path = WORKSPACE_ROOT
    phase: str = "all"
    device: str = "cuda"
    comparison_device: str = "cpu"
    dtype: str = "float64"
    batch_size: int = 1024
    csv_chunksize: int = 100_000
    _execution_mode: str = "production"

    def __post_init__(self) -> None:
        for name in (
            "csv_path",
            "final_protocol_path",
            "final_protocol_sidecar_path",
            "output_root",
            "workspace_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))

    def phase_directory(self, phase: str) -> Path:
        if phase not in (PHASE_CALIBRATION, PHASE_CONFIRMATION):
            raise LockedRunnerError(f"unsupported locked phase: {phase}")
        return self.output_root / phase


@dataclass(frozen=True)
class LockedHooks:
    """Injectable numeric boundary for tests; no estimator hook exists."""

    load_partition: Callable[..., MetroPTMinuteInterval]
    build_inference_windows: Callable[..., Any]
    predict: Callable[..., np.ndarray]
    align: Callable[..., Any]
    runtime_preflight: Callable[[LockedRunConfig], Mapping[str, Any]]


@dataclass(frozen=True)
class PhaseOutcome:
    phase: str
    decision: str
    completion: AuthenticatedDocument
    result: Mapping[str, Any]
    reused_completed_bundle: bool


@dataclass(frozen=True)
class LockedRunOutcome:
    calibration: PhaseOutcome | None
    confirmation: PhaseOutcome | None


@dataclass(frozen=True)
class FrozenArtifact:
    source_path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DevelopmentPass:
    output_directory: Path
    completion: AuthenticatedDocument
    phase_manifest: AuthenticatedDocument
    result: Mapping[str, Any]
    portability_audit: Mapping[str, Any]
    portability_delta: float
    source_fingerprint: str
    timestamp_audit: MetroPTTimestampAudit
    ensemble: FrozenArtifact
    scorer: FrozenArtifact
    directory_snapshot: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class CalibrationBundle:
    outcome: PhaseOutcome
    certificate: AuthenticatedDocument
    effective_boundary: float


@dataclass(frozen=True)
class LockedPath:
    prediction: np.ndarray
    residual: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    state_codes: np.ndarray
    reset_before: np.ndarray
    source_window_rows: np.ndarray
    energy: np.ndarray
    directional_glr: np.ndarray
    standardized_evidence: np.ndarray
    cusum: np.ndarray
    envelope: np.ndarray
    score_reset: np.ndarray
    alarm: np.ndarray | None = None


def _default_predict(
    ensemble: Any,
    inputs: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    return ensemble.predict(inputs, device=device, batch_size=batch_size)


def _default_runtime_preflight(config: LockedRunConfig) -> dict[str, Any]:
    requested = torch.device(config.device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise LockedRunnerError("the locked run requires an available CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = torch.cuda.current_device() if requested.index is None else requested.index
    probe = torch.tensor([1.0, 2.0], dtype=torch.float64, device=requested)
    observed = float(torch.sum(probe * probe).item())
    torch.cuda.synchronize(requested)
    if observed != 5.0:
        raise LockedRunnerError("CUDA float64 smoke computation failed")
    return {
        "backend": "torch",
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "primary_device": str(requested),
        "comparison_device": config.comparison_device,
        "device_index": int(index),
        "device_name": str(torch.cuda.get_device_name(index)),
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "deterministic_algorithms": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cuda_float64_smoke_test": True,
    }


def default_locked_hooks() -> LockedHooks:
    return LockedHooks(
        load_partition=load_governed_metropt_partition,
        build_inference_windows=build_causal_input_windows,
        predict=_default_predict,
        align=align_inference_forecasts_to_score_times,
        runtime_preflight=_default_runtime_preflight,
    )


def _absolute(path: Path, root: Path) -> Path:
    return Path(path if path.is_absolute() else root / path).absolute()


def _workspace_relative(path: Path, workspace: Path) -> str:
    root = workspace.resolve(strict=True)
    candidate = path.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise LockedRunnerError("path must remain inside workspace_root") from exc
    if not relative.parts:
        raise LockedRunnerError("path must not equal workspace_root")
    return PurePosixPath(*relative.parts).as_posix()


def _bound_path(value: Any, workspace: Path, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{name} must be a workspace-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise IntegrityError(f"{name} must be a normalized workspace-relative path")
    result = workspace.joinpath(*pure.parts)
    if _workspace_relative(result, workspace) != pure.as_posix():
        raise IntegrityError(f"{name} escapes workspace_root")
    return result


def _nested(value: Mapping[str, Any], name: str, *, where: str = "record") -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise IntegrityError(f"{where}.{name} must be an object")
    return result


def _require_keys(value: Mapping[str, Any], keys: set[str], *, where: str) -> None:
    observed = set(value)
    if observed != keys:
        missing = sorted(keys - observed)
        extra = sorted(observed - keys)
        raise IntegrityError(f"{where} fields changed; missing={missing}, extra={extra}")


def _sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise IntegrityError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _fingerprint(value: Any, *, name: str) -> str:
    return _sha(value, name=name)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrityError(f"{name} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise IntegrityError(f"{name} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise IntegrityError(f"{name} must be finite and non-negative")
    return result


def _assert_artifact(
    path: Path,
    *,
    sha256: str,
    size_bytes: int | None = None,
) -> None:
    record = artifact_record(path)
    if record.get("sha256") != sha256 or (
        size_bytes is not None and record.get("size_bytes") != size_bytes
    ):
        raise IntegrityError(f"frozen artifact changed: {path}")


def _validate_config(config: LockedRunConfig) -> None:
    if config.phase not in (PHASE_CALIBRATION, PHASE_CONFIRMATION, "all"):
        raise AccessDeniedError("phase must be calibration, confirmation, or all")
    if config.device != "cuda" or config.comparison_device != "cpu":
        raise LockedRunnerError("frozen devices are CUDA primary and CPU comparison")
    if config.dtype != "float64":
        raise LockedRunnerError("frozen dtype is float64")
    if config._execution_mode not in ("production", "synthetic_test"):
        raise AccessDeniedError("unrecognized locked-runner execution mode")
    for name, value in (("batch_size", config.batch_size), ("csv_chunksize", config.csv_chunksize)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise LockedRunnerError(f"{name} must be a positive integer")
    digest = _sha(config.expected_final_protocol_sha256, name="expected final protocol SHA")
    if config.expected_final_protocol_sha256 == UNFROZEN_SHA256 or digest == "0" * 64:
        raise AccessDeniedError("a real frozen final protocol SHA is required")


def _strict_json_record(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{name} is not strict JSON") from exc
    return result


def _file_binding(
    value: Mapping[str, Any],
    workspace: Path,
    *,
    where: str,
    fingerprint_field: str | None = None,
    decision: bool = False,
) -> tuple[Path, int, str, str | None]:
    keys = {"path", "size_bytes", "sha256"}
    if fingerprint_field is not None:
        keys.add(fingerprint_field)
    if decision:
        keys.add("decision")
    _require_keys(value, keys, where=where)
    path = _bound_path(value["path"], workspace, name=f"{where}.path")
    size = _positive_int(value["size_bytes"], name=f"{where}.size_bytes")
    digest = _sha(value["sha256"], name=f"{where}.sha256")
    fingerprint = None
    if fingerprint_field is not None:
        fingerprint = _fingerprint(
            value[fingerprint_field], name=f"{where}.{fingerprint_field}"
        )
    return path, size, digest, fingerprint


def _validate_implementation_snapshot(
    value: Mapping[str, Any], workspace: Path
) -> None:
    expected_paths = {
        "runner": "scripts/run_metropt_cared_bc_locked_gpu.py",
        "data": "src/mate_spc_metropt/data.py",
        "governance": "src/mate_spc_metropt/governance.py",
        "model": "src/mate_spc_metropt/model.py",
        "locked_evaluation": "src/mate_spc_metropt/locked_evaluation.py",
        "portability": "src/mate_spc_metropt/portability.py",
    }
    _require_keys(value, set(expected_paths), where="implementation_snapshot")
    for name, relative in expected_paths.items():
        record = _nested(value, name, where="implementation_snapshot")
        _require_keys(record, {"path", "size_bytes", "sha256"}, where=f"snapshot.{name}")
        if record.get("path") != relative:
            raise IntegrityError(f"snapshot path changed: {name}")
        path = _bound_path(relative, workspace, name=f"snapshot.{name}.path")
        _assert_artifact(
            path,
            sha256=_sha(record["sha256"], name=f"snapshot.{name}.sha256"),
            size_bytes=_positive_int(record["size_bytes"], name=f"snapshot.{name}.size"),
        )


def _validate_final_protocol(
    document: AuthenticatedDocument,
    config: LockedRunConfig,
) -> None:
    value = document.value
    _require_keys(
        value,
        {
            "schema",
            "schema_version",
            "protocol_id",
            "protocol_version",
            "status",
            "disclosure",
            "development_pass_binding",
            "source_binding",
            "fixed_artifacts",
            "implementation_snapshot",
            "runtime",
            "cpu_cuda_primitive_parity",
            "calibration",
            "confirmation",
            "output_contract",
        },
        where="final_protocol",
    )
    if (
        value.get("schema") != FINAL_PROTOCOL_SCHEMA
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("protocol_id"), str)
        or not value.get("protocol_id")
        or not isinstance(value.get("protocol_version"), str)
        or not value.get("protocol_version")
        or value.get("status") != FINAL_PROTOCOL_STATUS
        or document.record.get("sha256") != config.expected_final_protocol_sha256
    ):
        raise IntegrityError("final protocol identity changed")

    disclosure = _nested(value, "disclosure")
    _require_keys(
        disclosure,
        {
            "created_after_observing_development",
            "development_values_observed",
            "calibration_numeric_values_observed",
            "confirmation_numeric_values_observed",
            "model_and_scorer_frozen",
            "same_data_retuning_forbidden",
        },
        where="disclosure",
    )
    expected_disclosure = {
        "created_after_observing_development": True,
        "development_values_observed": True,
        "calibration_numeric_values_observed": False,
        "confirmation_numeric_values_observed": False,
        "model_and_scorer_frozen": True,
        "same_data_retuning_forbidden": True,
    }
    if dict(disclosure) != expected_disclosure:
        raise IntegrityError("final protocol disclosure changed")

    development = _nested(value, "development_pass_binding")
    _require_keys(
        development,
        {
            "output_directory",
            "protocol",
            "phase_manifest",
            "artifact_manifest",
            "completion",
            "result",
            "portability_audit",
            "portability_evidence",
            "gpu_execution",
            "artifact_names",
            "global_envelope_delta",
            "evidence_semantic_sha256",
            "v1_projection_semantic_sha256",
        },
        where="development_pass_binding",
    )
    _bound_path(
        development["output_directory"],
        config.workspace_root,
        name="development output",
    )
    dev_protocol = _nested(development, "protocol")
    _require_keys(dev_protocol, {"path", "sidecar_path", "sha256"}, where="dev.protocol")
    _bound_path(dev_protocol["path"], config.workspace_root, name="dev.protocol.path")
    _bound_path(
        dev_protocol["sidecar_path"],
        config.workspace_root,
        name="dev.protocol.sidecar_path",
    )
    _sha(dev_protocol["sha256"], name="dev.protocol.sha256")
    for name, fingerprint_field, decision in (
        ("phase_manifest", "manifest_fingerprint", False),
        ("artifact_manifest", "artifact_manifest_fingerprint", False),
        ("completion", "completion_fingerprint", True),
    ):
        record = _nested(development, name)
        _file_binding(
            record,
            config.workspace_root,
            where=f"development.{name}",
            fingerprint_field=fingerprint_field,
            decision=decision,
        )
    if _nested(development, "completion").get("decision") != DEVELOPMENT_PASS:
        raise AccessDeniedError("final protocol does not bind a development PASS")
    for name in (
        "result", "portability_audit", "portability_evidence", "gpu_execution"
    ):
        _file_binding(
            _nested(development, name),
            config.workspace_root,
            where=f"development.{name}",
        )
    names = development.get("artifact_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise IntegrityError("development artifact_names are invalid")
    _finite_nonnegative(
        development["global_envelope_delta"], name="development global delta"
    )
    _sha(
        development["evidence_semantic_sha256"],
        name="development evidence semantic SHA",
    )
    _sha(
        development["v1_projection_semantic_sha256"],
        name="development v1 projection semantic SHA",
    )

    source = _nested(value, "source_binding")
    _require_keys(
        source,
        {"source_fingerprint", "sensor_csv", "timestamp_audit"},
        where="source_binding",
    )
    _fingerprint(source["source_fingerprint"], name="source fingerprint")
    sensor = _nested(source, "sensor_csv")
    _require_keys(sensor, {"path", "size_bytes", "sha256"}, where="source.sensor_csv")
    sensor_spec = next(item for item in SOURCE_FILE_SPECS if item.key == "sensor_csv")
    if (
        sensor.get("path") != sensor_spec.relative_path
        or sensor.get("size_bytes") != sensor_spec.size_bytes
        or sensor.get("sha256") != sensor_spec.sha256
    ):
        raise IntegrityError("final protocol sensor binding changed")
    _file_binding(
        _nested(source, "timestamp_audit"),
        config.workspace_root,
        where="source.timestamp_audit",
    )

    fixed = _nested(value, "fixed_artifacts")
    _require_keys(fixed, {"ensemble", "uncalibrated_scorer"}, where="fixed_artifacts")
    _file_binding(
        _nested(fixed, "ensemble"),
        config.workspace_root,
        where="fixed.ensemble",
    )
    scorer_binding = _nested(fixed, "uncalibrated_scorer")
    _require_keys(
        scorer_binding,
        {"path", "size_bytes", "sha256", "threshold_must_be_null"},
        where="fixed.uncalibrated_scorer",
    )
    if scorer_binding.get("threshold_must_be_null") is not True:
        raise IntegrityError("final protocol scorer is not frozen as uncalibrated")
    _bound_path(scorer_binding["path"], config.workspace_root, name="fixed.scorer.path")
    _positive_int(scorer_binding["size_bytes"], name="fixed.scorer.size")
    _sha(scorer_binding["sha256"], name="fixed.scorer.sha256")

    runtime = _nested(value, "runtime")
    if runtime != {
        "primary_device": "cuda",
        "comparison_device": "cpu",
        "dtype": "float64",
        "batch_size": 1024,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
    }:
        raise IntegrityError("final protocol runtime changed")
    if config.batch_size != 1024:
        raise IntegrityError("configured batch size differs from final protocol")

    primitive_parity = _nested(value, "cpu_cuda_primitive_parity")
    if primitive_parity != {
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
    }:
        raise IntegrityError("final protocol CPU/CUDA primitive parity contract changed")

    calibration = _nested(value, "calibration")
    if calibration != {
        "partitions": list(CALIBRATION_PARTITIONS),
        "candidate_complete_days": 70,
        "minimum_qualified_days": 39,
        "minimum_scores_per_qualified_day": 1141,
        "alpha": 0.05,
        "primary_boundary_source": "cuda",
        "rank_formula": "ceil((n+1)*(1-0.05))",
        "effective_boundary_formula": (
            "nextafter(raw_boundary + development_global_envelope_delta + 1e-9, +infinity)"
        ),
        "complete_cpu_cuda_inference_required": True,
        "cpu_cuda_alarm_arrays_exact": True,
        "minimum_boundary_distance_strictly_greater_than": 1.0e-9,
    }:
        raise IntegrityError("final protocol calibration contract changed")

    confirmation = _nested(value, "confirmation")
    if confirmation != {
        "partition": "confirmation",
        "start_inclusive": locked.CONFIRMATION_START,
        "end_exclusive": locked.CONFIRMATION_END,
        "fault_event_ids": ["F2", "F3", "F4"],
        "fault_event_intervals": [
            {"event_id": event.event_id, "start": event.start, "end": event.end}
            for event in locked.CONFIRMATION_EVENTS
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
    }:
        raise IntegrityError("final protocol confirmation contract changed")

    output = _nested(value, "output_contract")
    if output != {
        "calibration_directory": _workspace_relative(
            config.phase_directory(PHASE_CALIBRATION), config.workspace_root
        ),
        "confirmation_directory": _workspace_relative(
            config.phase_directory(PHASE_CONFIRMATION), config.workspace_root
        ),
        "create_new_only": True,
        "completed_reentry_reads_csv": False,
        "calibration_and_confirmation_separate_single_access": True,
        "runtime_preflight_intent_loader_binding": (
            "locked_runner_authenticated_numeric_authorization"
        ),
    }:
        raise IntegrityError("final protocol output contract changed")
    _validate_implementation_snapshot(
        _nested(value, "implementation_snapshot"), config.workspace_root
    )


def _authenticate_final_protocol(config: LockedRunConfig) -> AuthenticatedDocument:
    document = authenticate_json_with_sidecar(
        config.final_protocol_path,
        config.final_protocol_sidecar_path,
        expected_sha256=config.expected_final_protocol_sha256,
    )
    _validate_final_protocol(document, config)
    return document


def _timestamp_audit_from_json(value: Mapping[str, Any]) -> MetroPTTimestampAudit:
    required = {
        "schema": "metropt_cared_bc_timestamp_audit",
        "schema_version": 1,
        "strictly_increasing": True,
        "duplicate_timestamp_count": 0,
        "timestamp_values_parsed": True,
        "sensor_numeric_values_parsed": False,
        "numeric_columns_parsed": [],
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise IntegrityError(f"timestamp audit changed: {key}")
    audit = MetroPTTimestampAudit(
        source_path=str(value["source_path"]),
        source_size_bytes=int(value["source_size_bytes"]),
        source_mtime_ns=int(value["source_mtime_ns"]),
        source_sha256=str(value["source_sha256"]),
        header=tuple(str(item) for item in value["header"]),
        row_count=int(value["row_count"]),
        first_timestamp=str(value["first_timestamp"]),
        last_timestamp=str(value["last_timestamp"]),
        minimum_interval_seconds=value.get("minimum_interval_seconds"),
        modal_interval_seconds=value.get("modal_interval_seconds"),
        modal_interval_count=int(value["modal_interval_count"]),
        maximum_interval_seconds=value.get("maximum_interval_seconds"),
        chunks_scanned=int(value["chunks_scanned"]),
        partition_row_counts={
            str(key): int(item)
            for key, item in _nested(value, "partition_row_counts").items()
        },
    )
    if audit.as_dict() != dict(value):
        raise IntegrityError("timestamp audit is not canonical")
    return audit


def _snapshot_directory(directory: Path) -> dict[str, dict[str, Any]]:
    root = directory.resolve(strict=True)
    if directory.is_symlink() or not root.is_dir():
        raise IntegrityError("bound output must be a real directory")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError(f"bound output contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = dict(artifact_record(path))
    if not result:
        raise IntegrityError("bound development output is empty")
    return result


def _authenticate_development_pass(
    config: LockedRunConfig,
    final_protocol: AuthenticatedDocument,
) -> DevelopmentPass:
    value = final_protocol.value
    binding = _nested(value, "development_pass_binding")
    output = _bound_path(
        binding["output_directory"], config.workspace_root, name="development output"
    )
    snapshot = _snapshot_directory(output)

    dev_protocol_binding = _nested(binding, "protocol")
    dev_protocol_path = _bound_path(
        dev_protocol_binding["path"], config.workspace_root, name="dev protocol path"
    )
    dev_sidecar_path = _bound_path(
        dev_protocol_binding["sidecar_path"],
        config.workspace_root,
        name="dev protocol sidecar path",
    )
    dev_protocol_sha = _sha(
        dev_protocol_binding["sha256"], name="dev protocol SHA"
    )
    dev_protocol = authenticate_json_with_sidecar(
        dev_protocol_path,
        dev_sidecar_path,
        expected_sha256=dev_protocol_sha,
    )
    dev_protocol_id = dev_protocol.value.get("protocol_id")
    if (
        dev_protocol_id != PORTABILITY_V1_1_DEVELOPMENT_ID
        or dev_protocol_sha != PORTABILITY_V1_1_DEVELOPMENT_SHA256
    ):
        raise IntegrityError("final protocol is not bound to the frozen v1_1 development protocol")

    phase_binding = _nested(binding, "phase_manifest")
    phase_path, phase_size, phase_sha, phase_fingerprint = _file_binding(
        phase_binding,
        config.workspace_root,
        where="development.phase_manifest",
        fingerprint_field="manifest_fingerprint",
    )
    phase = authenticate_fingerprinted_json(
        phase_path, fingerprint_field="manifest_fingerprint"
    )
    if (
        phase_path.parent != output
        or phase.record.get("size_bytes") != phase_size
        or phase.record.get("sha256") != phase_sha
        or phase.fingerprint != phase_fingerprint
        or phase.value.get("phase") != PHASE_DEVELOPMENT
        or phase.value.get("protocol_id") != dev_protocol_id
        or phase.value.get("protocol_sha256") != dev_protocol_sha
        or phase.value.get("protocol_record") != dev_protocol.record
        or phase.value.get("output_directory_relative_path")
        != _workspace_relative(output, config.workspace_root)
    ):
        raise IntegrityError("development phase manifest differs from final binding")
    phase_numeric = _nested(phase.value, "numeric_access", where="development.phase_manifest")
    phase_runtime = _nested(phase.value, "runtime", where="development.phase_manifest")
    if (
        phase.value.get("schema") != "metropt_cared_bc_v1_1_portability_phase_manifest"
        or phase.value.get("schema_version") != SCHEMA_VERSION
        or phase_numeric.get("exact_authorized_partitions")
        != ["dev_healthy", "dev_fault_f1"]
        or phase_numeric.get("fit_permitted") is not False
        or phase_numeric.get("selection_permitted") is not False
        or phase_numeric.get("single_access") is not True
        or phase_numeric.get("reusable") is not False
        or phase_runtime.get("preflight_before_intent_and_numeric_loader") is not True
        or phase.value.get("training_refitting_selection_or_gate_change_performed")
        is not False
        or phase.value.get("calibration_numeric_access_authorized") is not False
        or phase.value.get("confirmation_numeric_access_authorized") is not False
    ):
        raise IntegrityError("development phase did not preserve v1_1 least privilege")

    source_binding = _nested(value, "source_binding")
    source_fingerprint = _fingerprint(
        source_binding["source_fingerprint"], name="source fingerprint"
    )
    if phase.value.get("source_fingerprint") != source_fingerprint:
        raise IntegrityError("development and final source fingerprints disagree")
    sources = _nested(phase.value, "sources")
    if canonical_fingerprint(sources) != source_fingerprint:
        raise IntegrityError("development source map fingerprint changed")
    sensor = _nested(source_binding, "sensor_csv")
    phase_sensor = _nested(sources, "sensor_csv")
    if (
        phase_sensor.get("relative_path") != sensor.get("path")
        or phase_sensor.get("size_bytes") != sensor.get("size_bytes")
        or phase_sensor.get("sha256") != sensor.get("sha256")
        or phase.value.get("sensor_csv_relative_path") != sensor.get("path")
    ):
        raise IntegrityError("development sensor source binding changed")
    expected_csv = _bound_path(sensor["path"], config.workspace_root, name="sensor CSV")
    if config.csv_path != expected_csv:
        raise IntegrityError("configured CSV path differs from the final protocol")

    artifact_binding = _nested(binding, "artifact_manifest")
    manifest_path, manifest_size, manifest_sha, manifest_fingerprint = _file_binding(
        artifact_binding,
        config.workspace_root,
        where="development.artifact_manifest",
        fingerprint_field="artifact_manifest_fingerprint",
    )
    artifact_manifest = authenticate_fingerprinted_json(
        manifest_path,
        fingerprint_field="artifact_manifest_fingerprint",
    )
    artifact_names = tuple(str(name) for name in binding["artifact_names"])
    required_development_artifacts = {
        ENSEMBLE_NAME,
        SCORER_NAME,
        "portability_evidence.npz",
        "portability_audit.json",
        "development_result.json",
    }
    if (
        manifest_path.parent != output
        or artifact_manifest.record.get("size_bytes") != manifest_size
        or artifact_manifest.record.get("sha256") != manifest_sha
        or artifact_manifest.fingerprint != manifest_fingerprint
        or set(_nested(artifact_manifest.value, "artifacts")) != set(artifact_names)
        or not required_development_artifacts.issubset(artifact_names)
        or phase.value.get("output_artifacts") != list(artifact_names)
    ):
        raise IntegrityError("development artifact manifest changed")

    ledger = authenticate_fingerprinted_json(
        output / "development_access_ledger.json",
        fingerprint_field="ledger_fingerprint",
    )
    ledger_body = dict(ledger.value)
    ledger_body.pop("ledger_fingerprint", None)
    if (
        ledger.value.get("phase") != PHASE_DEVELOPMENT
        or ledger.value.get("protocol_id") != dev_protocol_id
        or ledger.value.get("charter_sha256") != dev_protocol_sha
        or ledger.value.get("final_protocol_sha256") is not None
        or ledger.value.get("source_fingerprint") != source_fingerprint
        or ledger.value.get("phase_manifest_fingerprint") != phase.fingerprint
        or ledger.value.get("output_directory") != str(output.resolve(strict=True))
        or ledger.value.get("authorized_partitions")
        != ["dev_healthy", "dev_fault_f1"]
        or ledger.value.get("sensor_numeric_access_committed") is not True
        or ledger.value.get("reusable") is not False
    ):
        raise IntegrityError("development access ledger changed")
    dev_claim = LedgerClaim(document=ledger, fresh=False)
    preflight = authenticate_fingerprinted_json(
        output / "runtime_preflight.json",
        fingerprint_field="preflight_fingerprint",
    )
    _require_keys(
        preflight.value,
        {
            "schema", "schema_version", "phase", "protocol_id", "protocol_sha256",
            "phase_manifest_fingerprint", "numeric_access_occurred_before_preflight",
            "access_intent_existed_before_preflight", "runtime", "checks", "passed",
            "error", "preflight_fingerprint",
        },
        where="v1_1 development runtime preflight",
    )
    preflight_runtime = _nested(preflight.value, "runtime", where="development.preflight")
    preflight_checks = _nested(preflight.value, "checks", where="development.preflight")
    expected_runtime = _nested(
        _nested(dev_protocol.value, "runtime_preflight", where="development.protocol"),
        "expected_runtime",
        where="development.protocol.runtime_preflight",
    )
    if (
        preflight.value.get("schema") != "metropt_cared_bc_v1_1_runtime_preflight"
        or preflight.value.get("schema_version") != SCHEMA_VERSION
        or preflight.value.get("phase") != PHASE_DEVELOPMENT
        or preflight.value.get("protocol_id") != dev_protocol_id
        or preflight.value.get("protocol_sha256") != dev_protocol_sha
        or preflight.value.get("phase_manifest_fingerprint") != phase.fingerprint
        or preflight.value.get("numeric_access_occurred_before_preflight") is not False
        or preflight.value.get("access_intent_existed_before_preflight") is not False
        or preflight.value.get("passed") is not True
        or preflight.value.get("error") is not None
        or not preflight_checks
        or any(check is not True for check in preflight_checks.values())
        or any(preflight_runtime.get(key) != expected for key, expected in expected_runtime.items())
    ):
        raise IntegrityError("development runtime preflight is not an authenticated PASS")
    intent = authenticate_phase_access_intent(
        output / "development_access_intent.json",
        dev_claim,
        phase_manifest_fingerprint=str(phase.fingerprint),
        runtime_preflight_fingerprint=str(preflight.fingerprint),
    )
    if (
        intent.value.get("authorized_partitions") != ["dev_healthy", "dev_fault_f1"]
        or intent.value.get("single_access") is not True
        or intent.value.get("runtime_preflight_must_precede_numeric_access") is not True
        or intent.value.get("runtime_preflight_fingerprint") != preflight.fingerprint
    ):
        raise IntegrityError("development access intent changed")

    completion_binding = _nested(binding, "completion")
    completion_path, completion_size, completion_sha, completion_fingerprint = _file_binding(
        completion_binding,
        config.workspace_root,
        where="development.completion",
        fingerprint_field="completion_fingerprint",
        decision=True,
    )
    completion = authenticate_completion_record(
        completion_path,
        output_directory=output,
        ledger_path=output / "development_access_ledger.json",
        expected_ledger_body=ledger_body,
        access_intent_path=output / "development_access_intent.json",
        artifact_manifest_path=manifest_path,
        expected_artifact_names=artifact_names,
    )
    if (
        completion_path.parent != output
        or completion.record.get("size_bytes") != completion_size
        or completion.record.get("sha256") != completion_sha
        or completion.fingerprint != completion_fingerprint
        or completion.value.get("decision") != DEVELOPMENT_PASS
    ):
        raise AccessDeniedError("bound development completion is not an authenticated PASS")

    bound_documents: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for name in ("result", "portability_audit"):
        path, size, digest, _ = _file_binding(
            _nested(binding, name),
            config.workspace_root,
            where=f"development.{name}",
        )
        if path.parent != output:
            raise IntegrityError(f"development {name} is outside its output")
        _assert_artifact(path, sha256=digest, size_bytes=size)
        bound_documents[name] = (path, read_strict_json(path))
    evidence_path, evidence_size, evidence_sha, _ = _file_binding(
        _nested(binding, "portability_evidence"),
        config.workspace_root,
        where="development.portability_evidence",
    )
    if evidence_path.parent != output:
        raise IntegrityError("development portability evidence is outside its output")
    _assert_artifact(evidence_path, sha256=evidence_sha, size_bytes=evidence_size)
    gpu_path, gpu_size, gpu_sha, _ = _file_binding(
        _nested(binding, "gpu_execution"),
        config.workspace_root,
        where="development.gpu_execution",
    )
    if gpu_path.parent != output:
        raise IntegrityError("development GPU execution is outside its output")
    _assert_artifact(gpu_path, sha256=gpu_sha, size_bytes=gpu_size)

    result = bound_documents["result"][1]
    portability_audit = bound_documents["portability_audit"][1]
    _require_keys(
        result,
        {
            "schema", "schema_version", "phase", "protocol_id", "protocol_sha256",
            "phase_manifest_fingerprint", "source_fingerprint", "decision",
            "development_gate_passed", "numeric_partitions_loaded_exactly_once",
            "fit_partition_loaded", "training_refitting_selection_or_gate_change_performed",
            "fixed_artifacts_byte_identical", "portability_audit_sha256",
            "evidence_semantic_sha256", "v1_projection_semantic_sha256",
            "next_method_if_failed", "further_cared_bc_repair_permitted_if_failed",
            "calibration_numeric_access_authorized_by_this_result",
            "confirmation_numeric_access_authorized_by_this_result", "next_phase_status",
        },
        where="v1_1 development result",
    )
    if (
        result.get("schema") != "metropt_cared_bc_v1_1_portability_development_result"
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("phase") != PHASE_DEVELOPMENT
        or result.get("protocol_id") != dev_protocol_id
        or result.get("protocol_sha256") != dev_protocol_sha
        or result.get("phase_manifest_fingerprint") != phase.fingerprint
        or result.get("source_fingerprint") != source_fingerprint
        or result.get("decision") != DEVELOPMENT_PASS
        or result.get("development_gate_passed") is not True
        or result.get("numeric_partitions_loaded_exactly_once")
        != ["dev_healthy", "dev_fault_f1"]
        or result.get("fit_partition_loaded") is not False
        or result.get("training_refitting_selection_or_gate_change_performed") is not False
        or result.get("next_method_if_failed") is not None
        or result.get("further_cared_bc_repair_permitted_if_failed") is not False
        or result.get("calibration_numeric_access_authorized_by_this_result") is not False
        or result.get("confirmation_numeric_access_authorized_by_this_result") is not False
        or result.get("next_phase_status")
        != "FINAL_ANALYSIS_PROTOCOL_REQUIRED_BEFORE_CALIBRATION_ACCESS"
    ):
        raise AccessDeniedError("development result is not a valid portability PASS")
    _require_keys(
        portability_audit,
        {
            "schema", "schema_version", "protocol_id", "protocol_sha256",
            "phase_manifest_fingerprint", "trusted_certificate_type",
            "core_portability_audit", "exact_backend_checks", "alarm_array_hashes",
            "full_sequence_max_abs_differences", "evidence_semantic_sha256",
            "v1_projection_semantic_sha256", "target_checks", "passed", "decision",
        },
        where="v1_1 portability audit",
    )
    target_checks = _nested(portability_audit, "target_checks")
    backend_checks = _nested(portability_audit, "exact_backend_checks")
    if (
        portability_audit.get("schema")
        != "metropt_cared_bc_v1_1_full_portability_adjudication"
        or portability_audit.get("schema_version") != SCHEMA_VERSION
        or portability_audit.get("protocol_id") != dev_protocol_id
        or portability_audit.get("protocol_sha256") != dev_protocol_sha
        or portability_audit.get("phase_manifest_fingerprint") != phase.fingerprint
        or portability_audit.get("trusted_certificate_type")
        != "CertifiedDevelopmentPortabilityAudit"
        or portability_audit.get("passed") is not True
        or portability_audit.get("decision") != DEVELOPMENT_PASS
        or not target_checks
        or any(value is not True for value in target_checks.values())
        or backend_checks.get("passed") is not True
        or result.get("portability_audit_sha256")
        != sha256_file(bound_documents["portability_audit"][0])
        or result.get("evidence_semantic_sha256")
        != portability_audit.get("evidence_semantic_sha256")
        or result.get("v1_projection_semantic_sha256")
        != portability_audit.get("v1_projection_semantic_sha256")
        or binding.get("evidence_semantic_sha256")
        != result.get("evidence_semantic_sha256")
        or binding.get("v1_projection_semantic_sha256")
        != result.get("v1_projection_semantic_sha256")
    ):
        raise AccessDeniedError("development portability audit is not a valid PASS")
    _sha(result["evidence_semantic_sha256"], name="development evidence semantic SHA")
    _sha(result["v1_projection_semantic_sha256"], name="v1 projection semantic SHA")
    fixed_identity = _nested(result, "fixed_artifacts_byte_identical")
    if (
        fixed_identity.get(ENSEMBLE_NAME) != artifact_record(output / ENSEMBLE_NAME)
        or fixed_identity.get(SCORER_NAME) != artifact_record(output / SCORER_NAME)
        or set(fixed_identity) != {ENSEMBLE_NAME, SCORER_NAME}
    ):
        raise IntegrityError("v1_1 development fixed-artifact identity changed")
    core = _nested(portability_audit, "core_portability_audit")
    threshold = _nested(core, "threshold")
    delta = _finite_nonnegative(
        threshold.get("global_envelope_delta"), name="development portability delta"
    )
    declared_delta = _finite_nonnegative(
        binding["global_envelope_delta"], name="declared development delta"
    )
    if delta != declared_delta or core.get("passed") is not True:
        raise IntegrityError("development portability delta or core gate changed")

    timestamp_path, timestamp_size, timestamp_sha, _ = _file_binding(
        _nested(source_binding, "timestamp_audit"),
        config.workspace_root,
        where="source.timestamp_audit",
    )
    _assert_artifact(timestamp_path, sha256=timestamp_sha, size_bytes=timestamp_size)
    timestamp_audit = _timestamp_audit_from_json(read_strict_json(timestamp_path))
    if (
        timestamp_audit.source_sha256 != sensor["sha256"]
        or timestamp_audit.source_size_bytes != sensor["size_bytes"]
    ):
        raise IntegrityError("timestamp audit and sensor binding disagree")

    fixed = _nested(value, "fixed_artifacts")
    frozen: dict[str, FrozenArtifact] = {}
    for key, filename in (("ensemble", ENSEMBLE_NAME), ("uncalibrated_scorer", SCORER_NAME)):
        item = _nested(fixed, key)
        source_path = _bound_path(item["path"], config.workspace_root, name=f"fixed.{key}.path")
        size = _positive_int(item["size_bytes"], name=f"fixed.{key}.size")
        digest = _sha(item["sha256"], name=f"fixed.{key}.sha256")
        if source_path != output / filename:
            raise IntegrityError(f"fixed {key} path is not the development artifact")
        _assert_artifact(source_path, sha256=digest, size_bytes=size)
        frozen[key] = FrozenArtifact(source_path, size, digest)

    if _snapshot_directory(output) != snapshot:
        raise IntegrityError("development PASS changed during authentication")
    return DevelopmentPass(
        output_directory=output,
        completion=completion,
        phase_manifest=phase,
        result=result,
        portability_audit=portability_audit,
        portability_delta=delta,
        source_fingerprint=source_fingerprint,
        timestamp_audit=timestamp_audit,
        ensemble=frozen["ensemble"],
        scorer=frozen["uncalibrated_scorer"],
        directory_snapshot=snapshot,
    )


def _validate_runtime_record(
    value: Mapping[str, Any], config: LockedRunConfig
) -> dict[str, Any]:
    record = _strict_json_record(value, name="runtime preflight")
    _require_keys(
        record,
        {
            "backend", "torch_version", "cuda_runtime_version", "primary_device",
            "comparison_device", "device_index", "device_name", "dtype",
            "batch_size", "deterministic_algorithms", "cuda_matmul_tf32", "cudnn_tf32",
            "cuda_float64_smoke_test",
        },
        where="runtime preflight",
    )
    if (
        record["backend"] != "torch"
        or not isinstance(record["torch_version"], str) or not record["torch_version"]
        or not isinstance(record["cuda_runtime_version"], str) or not record["cuda_runtime_version"]
        or record["primary_device"] != config.device
        or record["comparison_device"] != config.comparison_device
        or isinstance(record["device_index"], bool)
        or not isinstance(record["device_index"], int) or record["device_index"] < 0
        or not isinstance(record["device_name"], str) or not record["device_name"]
        or record["dtype"] != "float64"
        or record["batch_size"] != config.batch_size
        or record["deterministic_algorithms"] is not True
        or record["cuda_matmul_tf32"] is not False
        or record["cudnn_tf32"] is not False
        or record["cuda_float64_smoke_test"] is not True
    ):
        raise LockedRunnerError("runtime preflight violates the frozen CUDA contract")
    return record


def _ensure_fresh_directory(path: Path) -> None:
    if path.exists():
        raise AccessDeniedError(f"partial or unrecognized phase directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _copy_frozen(source: FrozenArtifact, destination: Path) -> FrozenArtifact:
    if destination.exists():
        raise IntegrityError(f"refusing to overwrite frozen artifact: {destination}")
    shutil.copyfile(source.source_path, destination)
    _assert_artifact(destination, sha256=source.sha256, size_bytes=source.size_bytes)
    return FrozenArtifact(destination, source.size_bytes, source.sha256)


def _partition(name: str) -> Any:
    matches = [partition for partition in PARTITIONS if partition.name == name]
    if len(matches) != 1:
        raise IntegrityError(f"frozen partition table lacks exactly one {name!r}")
    return matches[0]


def _phase_partitions(phase: str) -> tuple[str, ...]:
    if phase == PHASE_CALIBRATION:
        return CALIBRATION_PARTITIONS
    if phase == PHASE_CONFIRMATION:
        return CONFIRMATION_PARTITIONS
    raise LockedRunnerError(f"unsupported locked phase: {phase}")


def _phase_manifest_body(
    *, config: LockedRunConfig, protocol: AuthenticatedDocument,
    development: DevelopmentPass, phase: str,
    calibration_completion: AuthenticatedDocument | None = None,
    calibration_certificate: AuthenticatedDocument | None = None,
) -> dict[str, Any]:
    value = protocol.value
    names = _phase_partitions(phase)
    output = config.phase_directory(phase)
    source_binding = _nested(value, "source_binding", where="protocol")
    body: dict[str, Any] = {
        "schema": "metropt_cared_bc_locked_phase_manifest", "schema_version": SCHEMA_VERSION,
        "phase": phase, "protocol_id": value["protocol_id"],
        "protocol_sha256": config.expected_final_protocol_sha256,
        "protocol": {
            "path": _workspace_relative(config.final_protocol_path, config.workspace_root),
            "sidecar_path": _workspace_relative(config.final_protocol_sidecar_path, config.workspace_root),
            "size_bytes": protocol.record["size_bytes"], "sha256": protocol.record["sha256"],
        },
        "development_completion": {
            **development.completion.record,
            "completion_fingerprint": development.completion.fingerprint,
            "decision": DEVELOPMENT_PASS,
        },
        "source_fingerprint": development.source_fingerprint,
        "sensor_csv": dict(_nested(source_binding, "sensor_csv")),
        "timestamp_audit": dict(_nested(source_binding, "timestamp_audit")),
        "fixed_artifacts": dict(_nested(value, "fixed_artifacts", where="protocol")),
        "implementation_snapshot": dict(_nested(value, "implementation_snapshot", where="protocol")),
        "authorized_partitions": [_partition(name).as_dict() for name in names],
        "runtime": dict(_nested(value, "runtime", where="protocol")),
        "execution_boundary": (
            {
                "mode": "production",
                "injectable_hooks": False,
                "governed_loader": "mate_spc_metropt.data.load_governed_metropt_partition",
                "window_builder": "mate_spc_metropt.model.build_causal_input_windows",
                "predictor": "mate_spc_metropt.model.CaredEnsemble.predict",
                "alignment": "mate_spc_metropt.model.align_inference_forecasts_to_score_times",
                "scorer": "mate_spc_metropt.model.CaredBCScorer.score",
            }
            if config._execution_mode == "production"
            else {
                "mode": "synthetic_test",
                "injectable_hooks": True,
                "artifacts_valid_for_formal_analysis": False,
            }
        ),
        "numeric_access": {
            "single_access": True, "reusable": False,
            "only_after_runtime_preflight": True,
            "runtime_preflight_bound_to_numeric_authorization": True,
            "model_training_permitted": False, "scorer_fitting_permitted": False,
            "threshold_selection_permitted": phase == PHASE_CALIBRATION,
            "threshold_change_permitted": False,
        },
        "output_directory": str(output.resolve(strict=False)),
        "workspace_ledger": _workspace_relative(output / LEDGER_NAME[phase], config.workspace_root),
    }
    if phase == PHASE_CONFIRMATION:
        if calibration_completion is None or calibration_certificate is None:
            raise AccessDeniedError("confirmation manifest requires calibration PASS bindings")
        body["calibration_completion"] = {
            **calibration_completion.record,
            "completion_fingerprint": calibration_completion.fingerprint,
            "decision": CALIBRATION_PASS,
        }
        body["calibration_certificate"] = {
            **calibration_certificate.record,
            "certificate_fingerprint": calibration_certificate.fingerprint,
        }
    return _strict_json_record(body, name=f"{phase} phase manifest")


def _ledger_arguments(
    *, config: LockedRunConfig, protocol: AuthenticatedDocument,
    development: DevelopmentPass, phase: str, phase_manifest_fingerprint: str,
    calibration_completion: AuthenticatedDocument | None = None,
    calibration_certificate: AuthenticatedDocument | None = None,
) -> dict[str, Any]:
    sensor_spec = next(item for item in SOURCE_FILE_SPECS if item.key == "sensor_csv")
    arguments: dict[str, Any] = {
        "phase": phase,
        "workspace_relative_path": _workspace_relative(
            config.phase_directory(phase) / LEDGER_NAME[phase], config.workspace_root
        ),
        "protocol_id": protocol.value["protocol_id"],
        "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
        "final_protocol_sha256": config.expected_final_protocol_sha256,
        "sensor_csv_sha256": sensor_spec.sha256,
        "source_fingerprint": development.source_fingerprint,
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "output_directory": config.phase_directory(phase),
        "development_completion_fingerprint": development.completion.fingerprint,
        "development_completion_decision": DEVELOPMENT_PASS,
    }
    if phase == PHASE_CONFIRMATION:
        if calibration_completion is None or calibration_certificate is None:
            raise AccessDeniedError("confirmation ledger requires calibration PASS bindings")
        arguments.update(
            calibration_completion_fingerprint=calibration_completion.fingerprint,
            calibration_completion_decision=CALIBRATION_PASS,
            calibration_certificate_fingerprint=calibration_certificate.fingerprint,
        )
    return arguments


def _validate_interval(
    interval: MetroPTMinuteInterval, *, partition_name: str, phase: str,
    phase_manifest_fingerprint: str, claim: LedgerClaim,
    intent: AuthenticatedDocument,
) -> None:
    if not isinstance(interval, MetroPTMinuteInterval):
        raise IntegrityError("governed loader returned an unexpected interval type")
    count = int(interval.timestamps.size)
    if (
        interval.timestamps.dtype != np.dtype("datetime64[ns]")
        or interval.timestamps.shape != (count,)
        or np.any(np.isnat(interval.timestamps))
        or (count > 1 and np.any(np.diff(interval.timestamps) <= np.timedelta64(0, "ns")))
        or interval.analog.dtype != np.float64
        or interval.analog.shape != (count, len(ANALOG_COLUMNS))
        or interval.controls.dtype != np.float64
        or interval.controls.shape != (count, len(SAFE_CONTROL_COLUMNS))
        or interval.eligible.dtype != np.bool_
        or interval.eligible.shape != (count,)
        or interval.reset_before.dtype != np.bool_
        or interval.reset_before.shape != (count,)
    ):
        raise IntegrityError("governed interval arrays violate the frozen schema")
    if count and (
        np.any(~np.isfinite(interval.analog[interval.eligible]))
        or np.any(~np.isfinite(interval.controls[interval.eligible]))
    ):
        raise IntegrityError("eligible governed interval values are non-finite")
    partition = _partition(partition_name)
    left = np.datetime64(partition.start, "ns")
    right = np.datetime64(partition.end, "ns")
    if count and (np.any(interval.timestamps < left) or np.any(interval.timestamps >= right)):
        raise IntegrityError("governed interval escaped its frozen half-open boundary")
    metadata = interval.access_metadata
    governed = metadata.get("governed_partition_access") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("unrequested_numeric_values_parsed") is not False
        or metadata.get("direct_alarm_values_parsed") is not False
        or metadata.get("numeric_columns_parsed") != list(APPROVED_NUMERIC_COLUMNS)
        or not isinstance(governed, Mapping)
        or governed.get("formal_governed_entry") is not True
        or governed.get("phase") != phase
        or governed.get("partition") != partition.as_dict()
        or governed.get("phase_manifest_fingerprint") != phase_manifest_fingerprint
        or governed.get("workspace_ledger_fingerprint") != claim.document.fingerprint
        or governed.get("access_intent_fingerprint") != intent.fingerprint
    ):
        raise IntegrityError("governed interval access metadata is not authenticated")


def _finite_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape or not np.all(np.isfinite(array)):
        raise IntegrityError(f"{name} must be finite float64 with shape {shape}")
    return np.array(array, copy=True)


def _integer_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.shape != shape:
        raise IntegrityError(f"{name} must be int64 with shape {shape}")
    return np.array(array, copy=True)


def _bool_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise IntegrityError(f"{name} must be bool with shape {shape}")
    return np.array(array, copy=True)


def _validate_locked_path(path: LockedPath, *, name: str) -> LockedPath:
    prediction_raw = np.asarray(path.prediction)
    if prediction_raw.ndim != 3 or prediction_raw.shape[1:] != (3, 7):
        raise IntegrityError(f"{name}.prediction must have shape (N, 3, 7)")
    prediction = _finite_array(prediction_raw, tuple(prediction_raw.shape), name=f"{name}.prediction")
    residual_raw = np.asarray(path.residual)
    if residual_raw.ndim != 2 or residual_raw.shape[1:] != (31,):
        raise IntegrityError(f"{name}.residual must have shape (M, 31)")
    residual = _finite_array(residual_raw, tuple(residual_raw.shape), name=f"{name}.residual")
    count = residual.shape[0]
    timestamps = np.asarray(path.timestamps)
    if (
        timestamps.dtype != np.dtype("datetime64[ns]") or timestamps.shape != (count,)
        or np.any(np.isnat(timestamps))
        or (count > 1 and np.any(np.diff(timestamps) <= np.timedelta64(0, "ns")))
    ):
        raise IntegrityError(f"{name}.timestamps are invalid")
    indices = _integer_array(path.indices, (count,), name=f"{name}.indices")
    state = _integer_array(path.state_codes, (count,), name=f"{name}.state_codes")
    reset = _bool_array(path.reset_before, (count,), name=f"{name}.reset_before")
    rows_raw = np.asarray(path.source_window_rows)
    rows = _integer_array(
        rows_raw, (count, 3), name=f"{name}.source_window_rows"
    )
    if (
        np.any(indices < 0)
        or (count > 1 and np.any(np.diff(indices) <= 0))
        or np.any(rows < 0)
        or (rows.size and np.any(rows >= prediction.shape[0]))
    ):
        raise IntegrityError(f"{name} alignment provenance is out of bounds")
    energy = _finite_array(path.energy, (count,), name=f"{name}.energy")
    directional = _finite_array(path.directional_glr, (count,), name=f"{name}.directional_glr")
    evidence = _finite_array(path.standardized_evidence, (count, 2), name=f"{name}.standardized_evidence")
    cusum = _finite_array(path.cusum, (count, 2), name=f"{name}.cusum")
    envelope = _finite_array(path.envelope, (count,), name=f"{name}.envelope")
    score_reset = _bool_array(path.score_reset, (count,), name=f"{name}.score_reset")
    if np.any(energy < 0.0) or np.any(directional < 0.0) or np.any(cusum < 0.0) or np.any(envelope < 0.0):
        raise IntegrityError(f"{name} contains a negative monitoring statistic")
    page = dual_page_cusum(evidence, timestamps, reset_before=reset)
    if (
        not np.array_equal(page.cusum, cusum)
        or not np.array_equal(page.envelope, envelope)
        or not np.array_equal(page.reset, score_reset)
    ):
        raise IntegrityError(f"{name} saved Page-CUSUM path is not reproducible")
    alarm = None if path.alarm is None else _bool_array(path.alarm, (count,), name=f"{name}.alarm")
    return LockedPath(
        prediction=prediction, residual=residual,
        timestamps=np.array(timestamps, dtype="datetime64[ns]", copy=True),
        indices=indices, state_codes=state, reset_before=reset,
        source_window_rows=rows, energy=energy, directional_glr=directional,
        standardized_evidence=evidence, cusum=cusum, envelope=envelope,
        score_reset=score_reset, alarm=alarm,
    )


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _atomic_npz_new(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        Path(temporary).unlink()
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def _load_fixed_models(
    output: Path, development: DevelopmentPass,
) -> tuple[CaredEnsemble, CaredBCScorer, dict[str, Any]]:
    ensemble_file = _copy_frozen(development.ensemble, output / ENSEMBLE_NAME)
    scorer_file = _copy_frozen(development.scorer, output / SCORER_NAME)
    ensemble, ensemble_metadata = CaredEnsemble.load_npz(ensemble_file.source_path)
    scorer, scorer_metadata = CaredBCScorer.load_npz(scorer_file.source_path)
    if type(ensemble) is not CaredEnsemble or ensemble.dtype != "float64":
        raise IntegrityError("fixed artifact is not a concrete float64 CARED ensemble")
    if type(scorer) is not CaredBCScorer or scorer.threshold is not None:
        raise IntegrityError("fixed artifact is not a concrete uncalibrated CARED scorer")
    metadata = {
        "ensemble": _strict_json_record(ensemble_metadata, name="ensemble metadata"),
        "scorer": _strict_json_record(scorer_metadata, name="scorer metadata"),
        "ensemble_sha256": ensemble_file.sha256,
        "scorer_sha256": scorer_file.sha256,
    }
    return ensemble, scorer, metadata


def _locked_path_from_outputs(
    prediction: Any, aligned: Any, score: Any, *, name: str,
) -> LockedPath:
    if getattr(score, "threshold", object()) is not None:
        raise IntegrityError("uncalibrated scorer unexpectedly emitted a threshold")
    score_alarm = np.asarray(getattr(score, "alarm"))
    if score_alarm.dtype != np.bool_ or np.any(score_alarm):
        raise IntegrityError("uncalibrated scorer unexpectedly emitted an alarm")
    path = LockedPath(
        prediction=np.asarray(prediction),
        residual=np.asarray(aligned.residuals),
        timestamps=np.asarray(aligned.score_timestamps),
        indices=np.asarray(aligned.score_indices),
        state_codes=np.asarray(aligned.state_codes),
        reset_before=np.asarray(aligned.reset_before),
        source_window_rows=np.asarray(aligned.source_window_rows),
        energy=np.asarray(score.energy),
        directional_glr=np.asarray(score.directional_glr),
        standardized_evidence=np.asarray(score.standardized_evidence),
        cusum=np.asarray(score.cusum),
        envelope=np.asarray(score.envelope),
        score_reset=np.asarray(score.reset),
    )
    return _validate_locked_path(path, name=name)


def _score_partition(
    *, name: str, interval: MetroPTMinuteInterval, ensemble: CaredEnsemble,
    scorer: CaredBCScorer, config: LockedRunConfig, hooks: LockedHooks,
) -> tuple[dict[str, LockedPath], dict[str, Any]]:
    windows = hooks.build_inference_windows(
        interval.analog, interval.controls, interval.timestamps,
        eligible=interval.eligible, reset_before=interval.reset_before,
    )
    inputs = np.asarray(windows.inputs)
    if inputs.dtype != np.float64 or inputs.ndim != 3 or inputs.shape[1:] != (10, 13):
        raise IntegrityError(f"{name} inference windows violate the frozen input schema")
    if not np.all(np.isfinite(inputs)):
        raise IntegrityError(f"{name} inference windows contain a non-finite value")
    paths: dict[str, LockedPath] = {}
    execution: dict[str, Any] = {
        "input_window_count": int(inputs.shape[0]),
        "minute_count": int(interval.timestamps.size),
        "devices": {},
    }
    for label, device in (("cuda", config.device), ("cpu", config.comparison_device)):
        started = time.perf_counter()
        prediction = hooks.predict(
            ensemble, inputs, device=device, batch_size=config.batch_size
        )
        aligned = hooks.align(
            windows, prediction,
            interval.analog, interval.controls, interval.timestamps,
            eligible=interval.eligible, reset_before=interval.reset_before,
        )
        score = scorer.score(
            aligned.residuals, aligned.state_codes, aligned.score_timestamps,
            reset_before=aligned.reset_before,
        )
        elapsed = time.perf_counter() - started
        paths[label] = _locked_path_from_outputs(
            prediction, aligned, score, name=f"{name}.{label}"
        )
        if paths[label].indices.size and np.any(
            paths[label].indices >= interval.timestamps.size
        ):
            raise IntegrityError(f"{name}.{label} score index escaped the minute interval")
        execution["devices"][label] = {
            "requested_device": device,
            "prediction_count": int(paths[label].prediction.shape[0]),
            "score_count": int(paths[label].residual.shape[0]),
            "elapsed_seconds": float(elapsed),
            "complete_sequence": True,
        }
    return paths, _strict_json_record(execution, name=f"{name} execution")


def _maximum_absolute_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise IntegrityError("cannot compare arrays with different shapes")
    if first.size == 0:
        return 0.0
    result = float(np.max(np.abs(first - second)))
    if not math.isfinite(result):
        raise IntegrityError("parity error is non-finite")
    return result


def _parity_report(paths: Mapping[str, Mapping[str, LockedPath]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    all_passed = True
    for split, devices in paths.items():
        if set(devices) != set(DEVICES):
            raise IntegrityError(f"{split} lacks complete CPU/CUDA paths")
        cpu, cuda = devices["cpu"], devices["cuda"]
        errors = {
            "prediction_max_abs_error": _maximum_absolute_error(cpu.prediction, cuda.prediction),
            "forecast_residual_max_abs_error": _maximum_absolute_error(
                cpu.residual[:, :PHYSICAL_RESIDUAL_START],
                cuda.residual[:, :PHYSICAL_RESIDUAL_START],
            ),
            "energy_max_abs_error": _maximum_absolute_error(cpu.energy, cuda.energy),
            "directional_glr_max_abs_error": _maximum_absolute_error(
                cpu.directional_glr, cuda.directional_glr
            ),
            "standardized_evidence_max_abs_error": _maximum_absolute_error(
                cpu.standardized_evidence, cuda.standardized_evidence
            ),
            "cusum_max_abs_error": _maximum_absolute_error(cpu.cusum, cuda.cusum),
            "envelope_max_abs_error": _maximum_absolute_error(cpu.envelope, cuda.envelope),
        }
        exact = {
            "physical_residual_exact": bool(np.array_equal(
                cpu.residual[:, PHYSICAL_RESIDUAL_START:],
                cuda.residual[:, PHYSICAL_RESIDUAL_START:],
            )),
            "timestamps_exact": bool(np.array_equal(cpu.timestamps, cuda.timestamps)),
            "indices_exact": bool(np.array_equal(cpu.indices, cuda.indices)),
            "state_codes_exact": bool(np.array_equal(cpu.state_codes, cuda.state_codes)),
            "reset_before_exact": bool(np.array_equal(cpu.reset_before, cuda.reset_before)),
            "source_window_rows_exact": bool(np.array_equal(
                cpu.source_window_rows, cuda.source_window_rows
            )),
            "score_reset_exact": bool(np.array_equal(cpu.score_reset, cuda.score_reset)),
        }
        primitive_ok = bool(
            errors["prediction_max_abs_error"] <= PREDICTION_TOLERANCE
            and errors["forecast_residual_max_abs_error"] <= PREDICTION_TOLERANCE
            and all(exact.values())
        )
        report[split] = {
            "continuous_errors": errors, "exact_checks": exact,
            "primitive_parity_passed": primitive_ok,
        }
        all_passed &= primitive_ok
    report["passed"] = bool(all_passed)
    return _strict_json_record(report, name="CPU/CUDA parity report")


def _with_boundary(path: LockedPath, boundary: float) -> LockedPath:
    alarm = locked.strict_alarm_mask(path.envelope, boundary)
    return replace(path, alarm=alarm)


def _boundary_report(
    paths: Mapping[str, Mapping[str, LockedPath]], boundary: float
) -> tuple[dict[str, Any], dict[str, dict[str, LockedPath]]]:
    result: dict[str, Any] = {}
    thresholded: dict[str, dict[str, LockedPath]] = {}
    passed = True
    for split, devices in paths.items():
        thresholded[split] = {
            device: _with_boundary(path, boundary) for device, path in devices.items()
        }
        cpu, cuda = thresholded[split]["cpu"], thresholded[split]["cuda"]
        exact = bool(np.array_equal(cpu.alarm, cuda.alarm))
        distances: dict[str, float | None] = {}
        separated = True
        for device, path in thresholded[split].items():
            distance = (
                float(np.min(np.abs(path.envelope - boundary)))
                if path.envelope.size else None
            )
            if distance is not None and (not math.isfinite(distance) or distance <= BOUNDARY_DISTANCE):
                separated = False
            distances[device] = distance
        split_passed = bool(exact and separated)
        result[split] = {
            "cpu_cuda_alarm_exact": exact,
            "minimum_absolute_boundary_distance": distances,
            "minimum_required_exclusive": BOUNDARY_DISTANCE,
            "all_distances_strictly_greater_than_minimum": separated,
            "passed": split_passed,
        }
        passed &= split_passed
    result["passed"] = bool(passed)
    return _strict_json_record(result, name="boundary report"), thresholded


def _path_array_records(
    paths: Mapping[str, Mapping[str, LockedPath]]
) -> dict[str, Any]:
    fields = (
        "prediction", "residual", "timestamps", "indices", "state_codes",
        "reset_before", "source_window_rows", "energy", "directional_glr",
        "standardized_evidence", "cusum", "envelope", "score_reset", "alarm",
    )
    result: dict[str, Any] = {}
    for split, devices in paths.items():
        result[split] = {}
        for device, path in devices.items():
            device_record: dict[str, Any] = {}
            for field in fields:
                value = getattr(path, field)
                if value is None:
                    device_record[field] = None
                    continue
                array = np.asarray(value)
                if field == "timestamps":
                    array = array.astype("datetime64[ns]").astype(np.int64)
                device_record[field] = {
                    "shape": list(array.shape), "dtype": str(array.dtype),
                    "sha256": _array_sha256(array),
                }
            result[split][device] = device_record
    return _strict_json_record(result, name="numeric evidence records")


def _save_evidence(
    path: Path, *, phase: str, protocol_id: str, protocol_sha256: str,
    phase_manifest_fingerprint: str,
    paths: Mapping[str, Mapping[str, LockedPath]], boundary: float | None,
) -> dict[str, Any]:
    expected_splits = set(_phase_partitions(phase))
    if set(paths) != expected_splits:
        raise IntegrityError("numeric evidence has the wrong partition set")
    payload: dict[str, np.ndarray] = {}
    for split in _phase_partitions(phase):
        if set(paths[split]) != set(DEVICES):
            raise IntegrityError(f"{split} evidence lacks a complete device pair")
        for device in DEVICES:
            item = _validate_locked_path(paths[split][device], name=f"{split}.{device}")
            prefix = f"{split}__{device}"
            payload[f"{prefix}__prediction"] = item.prediction
            payload[f"{prefix}__residual"] = item.residual
            payload[f"{prefix}__timestamps_ns"] = item.timestamps.astype(np.int64)
            payload[f"{prefix}__indices"] = item.indices
            payload[f"{prefix}__state_codes"] = item.state_codes
            payload[f"{prefix}__reset_before"] = item.reset_before
            payload[f"{prefix}__source_window_rows"] = item.source_window_rows
            payload[f"{prefix}__energy"] = item.energy
            payload[f"{prefix}__directional_glr"] = item.directional_glr
            payload[f"{prefix}__standardized_evidence"] = item.standardized_evidence
            payload[f"{prefix}__cusum"] = item.cusum
            payload[f"{prefix}__envelope"] = item.envelope
            payload[f"{prefix}__score_reset"] = item.score_reset
            if boundary is not None:
                if item.alarm is None:
                    raise IntegrityError("thresholded evidence lacks alarm decisions")
                payload[f"{prefix}__alarm"] = item.alarm
            elif item.alarm is not None:
                raise IntegrityError("unthresholded evidence unexpectedly contains alarms")
    metadata = {
        "schema": "metropt_cared_bc_locked_numeric_evidence",
        "schema_version": SCHEMA_VERSION,
        "phase": phase, "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "partitions": list(_phase_partitions(phase)),
        "devices": list(DEVICES),
        "effective_boundary": boundary,
        "pickle_free": True,
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )
    _atomic_npz_new(path, payload)
    return _path_array_records(paths)


def _expected_evidence_keys(
    phase: str, *, thresholded: bool
) -> set[str]:
    result = {"metadata_json"}
    fields = {
        "prediction", "residual", "timestamps_ns", "indices", "state_codes",
        "reset_before", "source_window_rows", "energy", "directional_glr",
        "standardized_evidence", "cusum", "envelope", "score_reset",
    }
    if thresholded:
        fields.add("alarm")
    for split in _phase_partitions(phase):
        for device in DEVICES:
            result.update(f"{split}__{device}__{field}" for field in fields)
    return result


def _load_evidence(
    path: Path, *, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, scorer: CaredBCScorer,
) -> tuple[dict[str, dict[str, LockedPath]], float | None]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise IntegrityError("numeric evidence lacks metadata")
        raw_metadata = archive["metadata_json"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in "SU":
            raise IntegrityError("numeric evidence metadata is not a scalar string")
        try:
            metadata = json.loads(
                str(raw_metadata.item()),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IntegrityError("numeric evidence metadata is invalid JSON") from exc
        expected_metadata = {
            "schema": "metropt_cared_bc_locked_numeric_evidence",
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "partitions": list(_phase_partitions(phase)), "devices": list(DEVICES),
            "effective_boundary": metadata.get("effective_boundary"), "pickle_free": True,
        }
        if metadata != expected_metadata:
            raise IntegrityError("numeric evidence metadata changed")
        boundary_raw = metadata["effective_boundary"]
        boundary = None if boundary_raw is None else _finite_nonnegative(
            boundary_raw, name="saved effective boundary"
        )
        expected_keys = _expected_evidence_keys(phase, thresholded=boundary is not None)
        if set(archive.files) != expected_keys:
            raise IntegrityError("numeric evidence key set changed")
        paths: dict[str, dict[str, LockedPath]] = {}
        for split in _phase_partitions(phase):
            paths[split] = {}
            for device in DEVICES:
                prefix = f"{split}__{device}"
                timestamps_ns = np.asarray(archive[f"{prefix}__timestamps_ns"])
                if timestamps_ns.dtype != np.int64:
                    raise IntegrityError("saved timestamps must be int64 nanoseconds")
                candidate = LockedPath(
                    prediction=np.array(archive[f"{prefix}__prediction"], copy=True),
                    residual=np.array(archive[f"{prefix}__residual"], copy=True),
                    timestamps=np.array(timestamps_ns, dtype="datetime64[ns]", copy=True),
                    indices=np.array(archive[f"{prefix}__indices"], copy=True),
                    state_codes=np.array(archive[f"{prefix}__state_codes"], copy=True),
                    reset_before=np.array(archive[f"{prefix}__reset_before"], copy=True),
                    source_window_rows=np.array(archive[f"{prefix}__source_window_rows"], copy=True),
                    energy=np.array(archive[f"{prefix}__energy"], copy=True),
                    directional_glr=np.array(archive[f"{prefix}__directional_glr"], copy=True),
                    standardized_evidence=np.array(
                        archive[f"{prefix}__standardized_evidence"], copy=True
                    ),
                    cusum=np.array(archive[f"{prefix}__cusum"], copy=True),
                    envelope=np.array(archive[f"{prefix}__envelope"], copy=True),
                    score_reset=np.array(archive[f"{prefix}__score_reset"], copy=True),
                    alarm=(
                        np.array(archive[f"{prefix}__alarm"], copy=True)
                        if boundary is not None else None
                    ),
                )
                item = _validate_locked_path(candidate, name=f"saved.{split}.{device}")
                rescored = scorer.score(
                    item.residual, item.state_codes, item.timestamps,
                    reset_before=item.reset_before,
                )
                for field in (
                    "energy", "directional_glr", "standardized_evidence",
                    "cusum", "envelope",
                ):
                    if not np.array_equal(np.asarray(getattr(rescored, field)), getattr(item, field)):
                        raise IntegrityError(f"saved {split}.{device}.{field} is not scorer-reproducible")
                if not np.array_equal(np.asarray(rescored.reset), item.score_reset):
                    raise IntegrityError(f"saved {split}.{device} score resets changed")
                if boundary is not None and not np.array_equal(
                    locked.strict_alarm_mask(item.envelope, boundary), item.alarm
                ):
                    raise IntegrityError(f"saved {split}.{device} alarms changed")
                paths[split][device] = item
    return paths, boundary


def _calibration_analysis(
    *, paths: Mapping[str, Mapping[str, LockedPath]],
    development: DevelopmentPass, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, dict[str, LockedPath]], float | None, str]:
    if set(paths) != set(CALIBRATION_PARTITIONS):
        raise IntegrityError("calibration analysis has the wrong partition set")
    day_reports: dict[str, Any] = {}
    for name in CALIBRATION_PARTITIONS:
        partition = _partition(name)
        cuda_path = paths[name]["cuda"]
        day_reports[name] = locked.enumerate_complete_days(
            cuda_path.timestamps, cuda_path.envelope,
            start=partition.start, end=partition.end, partition_name=name,
        )
    candidate_count = int(sum(report["candidate_day_count"] for report in day_reports.values()))
    qualified_count = int(sum(report["qualified_day_count"] for report in day_reports.values()))
    if candidate_count != 70:
        raise IntegrityError("frozen calibration intervals must enumerate exactly 70 complete days")
    parity = _parity_report(paths)
    qualification_passed = qualified_count >= locked.MINIMUM_CALIBRATION_DAYS
    calibration_report: dict[str, Any] | None = None
    boundary_report: dict[str, Any] | None = None
    boundary: float | None = None
    thresholded: dict[str, dict[str, LockedPath]] = {
        split: dict(devices) for split, devices in paths.items()
    }
    if qualification_passed:
        calibration_report = locked.calibrate_locked_threshold(
            day_reports["calibration_a"], day_reports["calibration_b"],
            portability_delta=development.portability_delta,
        )
        boundary = _finite_nonnegative(
            calibration_report["effective_boundary"], name="effective boundary"
        )
        raw = float(calibration_report["raw_boundary"])
        expected = float(np.nextafter(
            raw + development.portability_delta + locked.BOUNDARY_EPSILON, np.inf
        ))
        if boundary != expected:
            raise IntegrityError("calibration boundary differs from the frozen guarded formula")
        boundary_report, thresholded = _boundary_report(paths, boundary)
    gates: dict[str, Any] = {
        "exactly_70_candidate_days": {
            "observed": candidate_count, "requirement": "== 70",
            "passed": candidate_count == 70,
        },
        "minimum_39_qualified_days": {
            "observed": qualified_count, "requirement": ">= 39",
            "passed": qualification_passed,
        },
        "complete_cpu_cuda_primitive_parity": {
            "observed": bool(parity["passed"]), "requirement": "true",
            "passed": bool(parity["passed"]),
        },
        "cpu_cuda_alarm_and_boundary_separation": {
            "observed": None if boundary_report is None else bool(boundary_report["passed"]),
            "requirement": "true after threshold construction",
            "passed": bool(boundary_report is not None and boundary_report["passed"]),
        },
    }
    passed = bool(all(item["passed"] for item in gates.values()))
    decision = CALIBRATION_PASS if passed else CALIBRATION_ABORT
    analysis = {
        "schema": "metropt_cared_bc_locked_calibration_analysis",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_CALIBRATION,
        "protocol_id": protocol.value["protocol_id"],
        "protocol_sha256": protocol.record["sha256"],
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "decision": decision, "passed": passed,
        "primary_boundary_source": "cuda",
        "day_enumerations": day_reports,
        "calibration": calibration_report,
        "cpu_cuda_parity": parity,
        "boundary_validation": boundary_report,
        "gates": gates,
        "model_training_performed": False,
        "scorer_fitting_performed": False,
        "confirmation_numeric_values_observed": False,
    }
    return (
        _strict_json_record(analysis, name="calibration analysis"),
        thresholded, boundary, decision,
    )


def _confirmation_analysis(
    *, paths: Mapping[str, Mapping[str, LockedPath]], boundary: float,
    protocol: AuthenticatedDocument, phase_manifest_fingerprint: str,
    calibration_certificate: AuthenticatedDocument,
) -> tuple[dict[str, Any], dict[str, dict[str, LockedPath]], str]:
    if set(paths) != set(CONFIRMATION_PARTITIONS):
        raise IntegrityError("confirmation analysis has the wrong partition set")
    parity = _parity_report(paths)
    thresholded = {
        split: {device: _with_boundary(item, boundary) for device, item in devices.items()}
        for split, devices in paths.items()
    }
    cpu = thresholded["confirmation"]["cpu"]
    cuda = thresholded["confirmation"]["cuda"]
    alarm_exact = bool(np.array_equal(cpu.alarm, cuda.alarm))
    alarm_report = {
        "cpu_cuda_alarm_exact": alarm_exact,
        "cpu_alarm_count": int(np.count_nonzero(cpu.alarm)),
        "cuda_alarm_count": int(np.count_nonzero(cuda.alarm)),
        "cpu_alarm_sha256": _array_sha256(cpu.alarm),
        "cuda_alarm_sha256": _array_sha256(cuda.alarm),
        "effective_boundary": boundary,
    }
    locked_report = locked.evaluate_locked_confirmation(
        cuda.timestamps, cuda.alarm, parity_exact=alarm_exact,
        start=locked.CONFIRMATION_START, end=locked.CONFIRMATION_END,
    )
    runner_checks = {
        "complete_cpu_cuda_primitive_parity": {
            "observed": bool(parity["passed"]), "requirement": "true",
            "passed": bool(parity["passed"]),
        },
        "cpu_cuda_alarm_decisions_exact": {
            "observed": alarm_exact, "requirement": "true", "passed": alarm_exact,
        },
        "locked_confirmation_gates": {
            "observed": bool(locked_report["passed"]), "requirement": "true",
            "passed": bool(locked_report["passed"]),
        },
    }
    passed = bool(all(item["passed"] for item in runner_checks.values()))
    decision = CONFIRMATION_PASS if passed else CONFIRMATION_FAIL
    analysis = {
        "schema": "metropt_cared_bc_locked_confirmation_analysis",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_CONFIRMATION,
        "protocol_id": protocol.value["protocol_id"],
        "protocol_sha256": protocol.record["sha256"],
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "calibration_certificate_fingerprint": calibration_certificate.fingerprint,
        "decision": decision, "passed": passed,
        "effective_boundary": boundary,
        "threshold_adjusted": False,
        "cpu_cuda_parity": parity,
        "alarm_validation": alarm_report,
        "locked_evaluation": locked_report,
        "runner_checks": runner_checks,
        "model_training_performed": False,
        "scorer_fitting_performed": False,
    }
    return _strict_json_record(analysis, name="confirmation analysis"), thresholded, decision


def _calibration_certificate_body(
    *, protocol: AuthenticatedDocument, development: DevelopmentPass,
    phase_manifest_fingerprint: str, analysis: Mapping[str, Any],
    evidence_path: Path, analysis_path: Path,
    numeric_records: Mapping[str, Any], decision: str,
) -> dict[str, Any]:
    passed = decision == CALIBRATION_PASS
    calibration = analysis.get("calibration")
    if passed and not isinstance(calibration, Mapping):
        raise IntegrityError("calibration PASS lacks a threshold report")
    return _strict_json_record(
        {
            "schema": "metropt_cared_bc_locked_calibration_certificate",
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE_CALIBRATION,
            "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "development_completion_fingerprint": development.completion.fingerprint,
            "development_portability_delta": development.portability_delta,
            "decision": decision, "passed": passed,
            "effective_boundary": (
                float(calibration["effective_boundary"])
                if isinstance(calibration, Mapping) else None
            ),
            "calibration_analysis_fingerprint": canonical_fingerprint(analysis),
            "calibration_analysis": dict(analysis),
            "numeric_evidence": dict(artifact_record(evidence_path)),
            "analysis_artifact": dict(artifact_record(analysis_path)),
            "numeric_array_records": dict(numeric_records),
            "fixed_artifacts": {
                ENSEMBLE_NAME: {
                    "size_bytes": development.ensemble.size_bytes,
                    "sha256": development.ensemble.sha256,
                },
                SCORER_NAME: {
                    "size_bytes": development.scorer.size_bytes,
                    "sha256": development.scorer.sha256,
                    "threshold": None,
                },
            },
            "confirmation_numeric_access_authorized_by_certificate_alone": False,
            "confirmation_requires_authenticated_pass_completion": True,
            "same_data_retuning_permitted": False,
        },
        name="calibration certificate",
    )


def _result_body(
    *, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, decision: str,
    analysis: Mapping[str, Any], evidence_path: Path,
    certificate: AuthenticatedDocument | None = None,
) -> dict[str, Any]:
    if phase == PHASE_CALIBRATION:
        passed = decision == CALIBRATION_PASS
        schema = "metropt_cared_bc_locked_calibration_result"
    elif phase == PHASE_CONFIRMATION:
        passed = decision == CONFIRMATION_PASS
        schema = "metropt_cared_bc_locked_confirmation_result"
    else:
        raise LockedRunnerError("invalid result phase")
    body: dict[str, Any] = {
        "schema": schema, "schema_version": SCHEMA_VERSION,
        "phase": phase, "protocol_id": protocol.value["protocol_id"],
        "protocol_sha256": protocol.record["sha256"],
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "decision": decision, "passed": passed,
        "analysis_fingerprint": canonical_fingerprint(analysis),
        "analysis": dict(analysis),
        "numeric_evidence": dict(artifact_record(evidence_path)),
        "model_training_performed": False,
        "scorer_fitting_performed": False,
        "threshold_adjusted_after_calibration": False,
    }
    if phase == PHASE_CALIBRATION:
        if certificate is None:
            raise IntegrityError("calibration result requires its certificate")
        body["calibration_certificate"] = {
            **certificate.record,
            "certificate_fingerprint": certificate.fingerprint,
        }
        body["confirmation_numeric_access_authorized_by_result_alone"] = False
        body["confirmation_requires_authenticated_pass_completion_and_certificate"] = True
    else:
        if certificate is None:
            raise IntegrityError("confirmation result requires the frozen certificate")
        body["calibration_certificate_fingerprint"] = certificate.fingerprint
        body["no_retuning_after_confirmation"] = True
    return _strict_json_record(body, name=f"{phase} result")


def _runtime_log_body(
    *, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, preflight: AuthenticatedDocument,
    execution: Mapping[str, Any], fixed_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return _strict_json_record(
        {
            "schema": "metropt_cared_bc_locked_gpu_execution",
            "schema_version": SCHEMA_VERSION,
            "phase": phase, "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "runtime_preflight": {
                **preflight.record, "preflight_fingerprint": preflight.fingerprint
            },
            "partitions": dict(execution),
            "fixed_artifact_metadata": dict(fixed_metadata),
            "complete_cpu_cuda_inference": True,
            "training_or_fitting_api_called": False,
        },
        name=f"{phase} GPU execution log",
    )


def _normal_artifact_names(phase: str) -> tuple[str, ...]:
    names = [
        PHASE_MANIFEST_NAME[phase], PREFLIGHT_NAME[phase],
        NUMERIC_AUTHORIZATION_NAME[phase], ENSEMBLE_NAME,
        SCORER_NAME, EVIDENCE_NAME[phase], ANALYSIS_NAME[phase],
        RESULT_NAME[phase], GPU_LOG_NAME[phase],
    ]
    if phase == PHASE_CALIBRATION:
        names.append(CERTIFICATE_NAME)
    return tuple(names)


def _create_phase_completion(
    *, output: Path, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, claim: LedgerClaim,
    intent: AuthenticatedDocument, artifact_names: tuple[str, ...], decision: str,
    artifact_manifest_name: str | None = None,
    completion_name: str | None = None,
) -> AuthenticatedDocument:
    manifest_name = artifact_manifest_name or ARTIFACT_MANIFEST_NAME[phase]
    completed_name = completion_name or COMPLETION_NAME[phase]
    manifest = create_artifact_manifest(
        output / manifest_name, output,
        phase=phase, protocol_id=protocol.value["protocol_id"],
        protocol_sha256=protocol.record["sha256"],
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        access_intent=intent, artifact_names=artifact_names,
    )
    created = create_completion_record(
        output / completed_name, claim=claim, access_intent=intent,
        artifact_manifest=manifest, decision=decision,
    )
    ledger_body = dict(claim.document.value)
    ledger_body.pop("ledger_fingerprint", None)
    verified = authenticate_completion_record(
        output / completed_name, output_directory=output,
        ledger_path=output / LEDGER_NAME[phase], expected_ledger_body=ledger_body,
        access_intent_path=output / INTENT_NAME[phase],
        artifact_manifest_path=output / manifest_name,
        expected_artifact_names=artifact_names,
    )
    if created.fingerprint != verified.fingerprint:
        raise IntegrityError("phase completion failed post-create authentication")
    return verified


def _finalize_runtime_abort(
    *, output: Path, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, claim: LedgerClaim,
    intent: AuthenticatedDocument, error: Exception,
) -> AuthenticatedDocument:
    decision = CALIBRATION_ABORT if phase == PHASE_CALIBRATION else CONFIRMATION_FAIL
    atomic_create_new_json(
        output / ABORT_NAME,
        {
            "schema": "metropt_cared_bc_locked_runtime_abort",
            "schema_version": SCHEMA_VERSION, "phase": phase,
            "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "decision": decision, "error_type": type(error).__name__,
            "error_message": str(error),
            "numeric_access_cannot_be_repeated": True,
            "confirmation_remains_sealed": phase == PHASE_CALIBRATION,
        },
    )
    candidates = (*_normal_artifact_names(phase), ABORT_NAME)
    names = tuple(
        name for name in candidates
        if (output / name).is_file() and not (output / name).is_symlink()
    )
    if PHASE_MANIFEST_NAME[phase] not in names or ABORT_NAME not in names:
        raise IntegrityError("runtime ABORT lacks its immutable phase evidence")
    manifest_name = ARTIFACT_MANIFEST_NAME[phase]
    completion_name = COMPLETION_NAME[phase]
    if (output / manifest_name).is_file():
        names = (*names, manifest_name)
        if (output / completion_name).is_file():
            names = (*names, completion_name)
        manifest_name = ABORT_ARTIFACT_MANIFEST_NAME[phase]
        completion_name = ABORT_COMPLETION_NAME[phase]
    return _create_phase_completion(
        output=output, phase=phase, protocol=protocol,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        claim=claim, intent=intent, artifact_names=names, decision=decision,
        artifact_manifest_name=manifest_name,
        completion_name=completion_name,
    )


def _runtime_preflight_body(
    *, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return _strict_json_record(
        {
            "schema": "metropt_cared_bc_locked_runtime_preflight",
            "schema_version": SCHEMA_VERSION,
            "phase": phase, "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "passed": True, "numeric_access_occurred_before_preflight": False,
            "runtime": dict(runtime),
        },
        name=f"{phase} runtime preflight",
    )


def _numeric_authorization_body(
    *, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, preflight: AuthenticatedDocument,
    intent: AuthenticatedDocument,
) -> dict[str, Any]:
    return _strict_json_record(
        {
            "schema": "metropt_cared_bc_locked_numeric_authorization",
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "runtime_preflight": {
                **preflight.record,
                "preflight_fingerprint": preflight.fingerprint,
            },
            "access_intent": {
                **intent.record,
                "intent_fingerprint": intent.fingerprint,
            },
            "authorized_partitions": list(_phase_partitions(phase)),
            "runtime_preflight_passed_before_authorization": True,
            "numeric_loader_must_authenticate_this_record": True,
        },
        name=f"{phase} numeric authorization",
    )


def _authenticate_numeric_authorization(
    *, path: Path, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, preflight: AuthenticatedDocument,
    intent: AuthenticatedDocument,
) -> AuthenticatedDocument:
    return authenticate_fingerprinted_json(
        path,
        fingerprint_field="manifest_fingerprint",
        expected_body=_numeric_authorization_body(
            phase=phase,
            protocol=protocol,
            phase_manifest_fingerprint=phase_manifest_fingerprint,
            preflight=preflight,
            intent=intent,
        ),
    )


def _load_and_score_phase(
    *, config: LockedRunConfig, hooks: LockedHooks,
    protocol: AuthenticatedDocument, development: DevelopmentPass, phase: str,
    phase_manifest_fingerprint: str, claim: LedgerClaim,
    intent: AuthenticatedDocument, preflight_path: Path,
    numeric_authorization: AuthenticatedDocument,
    numeric_authorization_path: Path,
    ensemble: CaredEnsemble, scorer: CaredBCScorer,
) -> tuple[dict[str, dict[str, LockedPath]], dict[str, Any]]:
    paths: dict[str, dict[str, LockedPath]] = {}
    execution: dict[str, Any] = {}
    for name in _phase_partitions(phase):
        verified_authorization = _authenticate_numeric_authorization(
            path=numeric_authorization_path,
            phase=phase,
            protocol=protocol,
            phase_manifest_fingerprint=phase_manifest_fingerprint,
            preflight=authenticate_fingerprinted_json(
                preflight_path, fingerprint_field="preflight_fingerprint"
            ),
            intent=intent,
        )
        if (
            verified_authorization.fingerprint != numeric_authorization.fingerprint
            or verified_authorization.record != numeric_authorization.record
            or verified_authorization.value != numeric_authorization.value
        ):
            raise IntegrityError("numeric authorization changed before partition load")
        interval = hooks.load_partition(
            config.csv_path, name, claim=claim, access_intent=intent,
            access_intent_path=config.phase_directory(phase) / INTENT_NAME[phase],
            phase_manifest_fingerprint=phase_manifest_fingerprint,
            timestamp_audit=development.timestamp_audit,
            chunksize=config.csv_chunksize,
            runtime_preflight_path=preflight_path,
        )
        _validate_interval(
            interval, partition_name=name, phase=phase,
            phase_manifest_fingerprint=phase_manifest_fingerprint,
            claim=claim, intent=intent,
        )
        paths[name], execution[name] = _score_partition(
            name=name, interval=interval, ensemble=ensemble, scorer=scorer,
            config=config, hooks=hooks,
        )
    if tuple(paths) != _phase_partitions(phase):
        raise AccessDeniedError("numeric loader crossed the frozen phase scope")
    return paths, execution


def _write_preaccess_abort(
    *, output: Path, phase: str, protocol: AuthenticatedDocument,
    phase_manifest_fingerprint: str, claim: LedgerClaim, error: Exception,
) -> None:
    create_fingerprinted_json(
        output / PREACCESS_ABORT_NAME,
        {
            "schema": "metropt_cared_bc_locked_preaccess_abort",
            "schema_version": SCHEMA_VERSION,
            "phase": phase, "protocol_id": protocol.value["protocol_id"],
            "protocol_sha256": protocol.record["sha256"],
            "phase_manifest_fingerprint": phase_manifest_fingerprint,
            "workspace_ledger_fingerprint": claim.document.fingerprint,
            "decision": CALIBRATION_ABORT if phase == PHASE_CALIBRATION else CONFIRMATION_FAIL,
            "error_type": type(error).__name__, "error_message": str(error),
            "numeric_access_occurred": False,
            "access_intent_created": False,
            "ledger_is_spent_and_may_not_be_reused": True,
            "confirmation_remains_sealed": phase == PHASE_CALIBRATION,
        },
        fingerprint_field="completion_fingerprint",
    )


def _run_fresh_phase(
    *, config: LockedRunConfig, hooks: LockedHooks,
    protocol: AuthenticatedDocument, development: DevelopmentPass,
    phase: str, calibration_bundle: CalibrationBundle | None = None,
) -> PhaseOutcome:
    output = config.phase_directory(phase)
    calibration_completion = (
        calibration_bundle.outcome.completion if calibration_bundle is not None else None
    )
    calibration_certificate = (
        calibration_bundle.certificate if calibration_bundle is not None else None
    )
    phase_body = _phase_manifest_body(
        config=config, protocol=protocol, development=development, phase=phase,
        calibration_completion=calibration_completion,
        calibration_certificate=calibration_certificate,
    )
    phase_fingerprint = canonical_fingerprint(phase_body)
    _ensure_fresh_directory(output)
    phase_document = create_fingerprinted_json(
        output / PHASE_MANIFEST_NAME[phase], phase_body,
        fingerprint_field="manifest_fingerprint",
    )
    if phase_document.fingerprint != phase_fingerprint:
        raise IntegrityError("created phase manifest changed")
    ledger_arguments = _ledger_arguments(
        config=config, protocol=protocol, development=development, phase=phase,
        phase_manifest_fingerprint=phase_fingerprint,
        calibration_completion=calibration_completion,
        calibration_certificate=calibration_certificate,
    )
    claim = claim_workspace_access_ledger(output / LEDGER_NAME[phase], **ledger_arguments)
    try:
        runtime = _validate_runtime_record(hooks.runtime_preflight(config), config)
        preflight = create_fingerprinted_json(
            output / PREFLIGHT_NAME[phase],
            _runtime_preflight_body(
                phase=phase, protocol=protocol,
                phase_manifest_fingerprint=phase_fingerprint, runtime=runtime,
            ),
            fingerprint_field="preflight_fingerprint",
        )
    except Exception as error:
        _write_preaccess_abort(
            output=output, phase=phase, protocol=protocol,
            phase_manifest_fingerprint=phase_fingerprint, claim=claim, error=error,
        )
        raise
    intent = create_phase_access_intent(
        output / INTENT_NAME[phase], claim,
        phase_manifest_fingerprint=phase_fingerprint,
    )
    numeric_authorization = create_fingerprinted_json(
        output / NUMERIC_AUTHORIZATION_NAME[phase],
        _numeric_authorization_body(
            phase=phase, protocol=protocol,
            phase_manifest_fingerprint=phase_fingerprint,
            preflight=preflight, intent=intent,
        ),
        fingerprint_field="manifest_fingerprint",
    )
    try:
        ensemble, scorer, fixed_metadata = _load_fixed_models(output, development)
        paths, execution = _load_and_score_phase(
            config=config, hooks=hooks, protocol=protocol,
            development=development, phase=phase,
            phase_manifest_fingerprint=phase_fingerprint, claim=claim,
            intent=intent, preflight_path=output / PREFLIGHT_NAME[phase],
            numeric_authorization=numeric_authorization,
            numeric_authorization_path=output / NUMERIC_AUTHORIZATION_NAME[phase],
            ensemble=ensemble, scorer=scorer,
        )
        if _snapshot_directory(development.output_directory) != development.directory_snapshot:
            raise IntegrityError("development PASS changed during locked execution")

        if phase == PHASE_CALIBRATION:
            analysis, persisted_paths, boundary, decision = _calibration_analysis(
                paths=paths, development=development, protocol=protocol,
                phase_manifest_fingerprint=phase_fingerprint,
            )
        else:
            if calibration_bundle is None:
                raise AccessDeniedError("confirmation requires an authenticated calibration PASS")
            boundary = calibration_bundle.effective_boundary
            analysis, persisted_paths, decision = _confirmation_analysis(
                paths=paths, boundary=boundary, protocol=protocol,
                phase_manifest_fingerprint=phase_fingerprint,
                calibration_certificate=calibration_bundle.certificate,
            )

        numeric_records = _save_evidence(
            output / EVIDENCE_NAME[phase], phase=phase,
            protocol_id=protocol.value["protocol_id"],
            protocol_sha256=protocol.record["sha256"],
            phase_manifest_fingerprint=phase_fingerprint,
            paths=persisted_paths, boundary=boundary,
        )
        atomic_create_new_json(output / ANALYSIS_NAME[phase], analysis)
        if phase == PHASE_CALIBRATION:
            certificate = create_fingerprinted_json(
                output / CERTIFICATE_NAME,
                _calibration_certificate_body(
                    protocol=protocol, development=development,
                    phase_manifest_fingerprint=phase_fingerprint,
                    analysis=analysis, evidence_path=output / EVIDENCE_NAME[phase],
                    analysis_path=output / ANALYSIS_NAME[phase],
                    numeric_records=numeric_records, decision=decision,
                ),
                fingerprint_field="certificate_fingerprint",
            )
        else:
            certificate = calibration_bundle.certificate
        result = _result_body(
            phase=phase, protocol=protocol,
            phase_manifest_fingerprint=phase_fingerprint, decision=decision,
            analysis=analysis, evidence_path=output / EVIDENCE_NAME[phase],
            certificate=certificate,
        )
        atomic_create_new_json(output / RESULT_NAME[phase], result)
        atomic_create_new_json(
            output / GPU_LOG_NAME[phase],
            _runtime_log_body(
                phase=phase, protocol=protocol,
                phase_manifest_fingerprint=phase_fingerprint,
                preflight=preflight, execution=execution,
                fixed_metadata=fixed_metadata,
            ),
        )
        completion = _create_phase_completion(
            output=output, phase=phase, protocol=protocol,
            phase_manifest_fingerprint=phase_fingerprint, claim=claim,
            intent=intent, artifact_names=_normal_artifact_names(phase),
            decision=decision,
        )
        return PhaseOutcome(phase, decision, completion, result, False)
    except Exception as error:
        if not (output / ABORT_COMPLETION_NAME[phase]).exists():
            _finalize_runtime_abort(
                output=output, phase=phase, protocol=protocol,
                phase_manifest_fingerprint=phase_fingerprint,
                claim=claim, intent=intent, error=error,
            )
        raise


def _artifact_names_from_manifest(
    output: Path, phase: str, *, manifest_name: str | None = None
) -> tuple[str, ...]:
    selected_manifest = manifest_name or ARTIFACT_MANIFEST_NAME[phase]
    document = authenticate_fingerprinted_json(
        output / selected_manifest,
        fingerprint_field="artifact_manifest_fingerprint",
    )
    artifacts = _nested(document.value, "artifacts", where="artifact manifest")
    names = tuple(sorted(str(name) for name in artifacts))
    normal = set(_normal_artifact_names(phase))
    observed = set(names)
    if observed != normal:
        allowed = normal | {ABORT_NAME}
        if selected_manifest == ABORT_ARTIFACT_MANIFEST_NAME[phase]:
            allowed.update(
                {ARTIFACT_MANIFEST_NAME[phase], COMPLETION_NAME[phase]}
            )
        if (
            ABORT_NAME not in observed
            or not observed.issubset(allowed)
            or PHASE_MANIFEST_NAME[phase] not in observed
            or PREFLIGHT_NAME[phase] not in observed
        ):
            raise IntegrityError("completed phase artifact set is invalid")
    return names


def _read_fixed_models(
    output: Path, development: DevelopmentPass,
) -> tuple[CaredEnsemble, CaredBCScorer, dict[str, Any]]:
    ensemble_path = output / ENSEMBLE_NAME
    scorer_path = output / SCORER_NAME
    _assert_artifact(
        ensemble_path, sha256=development.ensemble.sha256,
        size_bytes=development.ensemble.size_bytes,
    )
    _assert_artifact(
        scorer_path, sha256=development.scorer.sha256,
        size_bytes=development.scorer.size_bytes,
    )
    ensemble, ensemble_metadata = CaredEnsemble.load_npz(ensemble_path)
    scorer, scorer_metadata = CaredBCScorer.load_npz(scorer_path)
    if type(ensemble) is not CaredEnsemble or ensemble.dtype != "float64":
        raise IntegrityError("completed bundle ensemble is not frozen float64 CARED")
    if type(scorer) is not CaredBCScorer or scorer.threshold is not None:
        raise IntegrityError("completed bundle scorer is not frozen uncalibrated CARED")
    metadata = {
        "ensemble": _strict_json_record(ensemble_metadata, name="ensemble metadata"),
        "scorer": _strict_json_record(scorer_metadata, name="scorer metadata"),
        "ensemble_sha256": development.ensemble.sha256,
        "scorer_sha256": development.scorer.sha256,
    }
    return ensemble, scorer, metadata


def _validate_execution_log(
    value: Mapping[str, Any], *, phase: str,
    protocol: AuthenticatedDocument, phase_manifest_fingerprint: str,
    preflight: AuthenticatedDocument, paths: Mapping[str, Mapping[str, LockedPath]],
    fixed_metadata: Mapping[str, Any], config: LockedRunConfig,
) -> None:
    _require_keys(
        value,
        {
            "schema", "schema_version", "phase", "protocol_id", "protocol_sha256",
            "phase_manifest_fingerprint", "runtime_preflight", "partitions",
            "fixed_artifact_metadata", "complete_cpu_cuda_inference",
            "training_or_fitting_api_called",
        },
        where="GPU execution log",
    )
    if (
        value.get("schema") != "metropt_cared_bc_locked_gpu_execution"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("phase") != phase
        or value.get("protocol_id") != protocol.value["protocol_id"]
        or value.get("protocol_sha256") != protocol.record["sha256"]
        or value.get("phase_manifest_fingerprint") != phase_manifest_fingerprint
        or value.get("runtime_preflight")
        != {**preflight.record, "preflight_fingerprint": preflight.fingerprint}
        or value.get("fixed_artifact_metadata") != fixed_metadata
        or value.get("complete_cpu_cuda_inference") is not True
        or value.get("training_or_fitting_api_called") is not False
    ):
        raise IntegrityError("GPU execution log identity changed")
    partitions = _nested(value, "partitions", where="GPU execution log")
    if set(partitions) != set(_phase_partitions(phase)):
        raise IntegrityError("GPU execution log partition set changed")
    for split in _phase_partitions(phase):
        record = _nested(partitions, split, where="GPU execution log partitions")
        _require_keys(
            record, {"input_window_count", "minute_count", "devices"},
            where=f"execution.{split}",
        )
        if (
            record.get("input_window_count") != paths[split]["cuda"].prediction.shape[0]
            or isinstance(record.get("minute_count"), bool)
            or not isinstance(record.get("minute_count"), int)
            or int(record["minute_count"]) < paths[split]["cuda"].residual.shape[0]
        ):
            raise IntegrityError(f"execution counts changed for {split}")
        devices = _nested(record, "devices", where=f"execution.{split}")
        if set(devices) != set(DEVICES):
            raise IntegrityError(f"execution device set changed for {split}")
        for device in DEVICES:
            item = _nested(devices, device, where=f"execution.{split}.devices")
            _require_keys(
                item,
                {"requested_device", "prediction_count", "score_count", "elapsed_seconds", "complete_sequence"},
                where=f"execution.{split}.{device}",
            )
            elapsed = _finite_nonnegative(
                item["elapsed_seconds"], name=f"execution {split}.{device} elapsed"
            )
            expected_device = config.device if device == "cuda" else config.comparison_device
            if (
                item.get("requested_device") != expected_device
                or item.get("prediction_count") != paths[split][device].prediction.shape[0]
                or item.get("score_count") != paths[split][device].residual.shape[0]
                or item.get("complete_sequence") is not True
                or elapsed != item["elapsed_seconds"]
                or (
                    paths[split][device].indices.size
                    and np.any(paths[split][device].indices >= int(record["minute_count"]))
                )
            ):
                raise IntegrityError(f"execution metadata changed for {split}.{device}")


def _authenticate_completed_phase(
    *, config: LockedRunConfig, protocol: AuthenticatedDocument,
    development: DevelopmentPass, phase: str,
    calibration_bundle: CalibrationBundle | None = None,
) -> tuple[PhaseOutcome, AuthenticatedDocument | None, float | None]:
    output = config.phase_directory(phase)
    calibration_completion = (
        calibration_bundle.outcome.completion if calibration_bundle is not None else None
    )
    calibration_certificate = (
        calibration_bundle.certificate if calibration_bundle is not None else None
    )
    phase_body = _phase_manifest_body(
        config=config, protocol=protocol, development=development, phase=phase,
        calibration_completion=calibration_completion,
        calibration_certificate=calibration_certificate,
    )
    phase_document = authenticate_fingerprinted_json(
        output / PHASE_MANIFEST_NAME[phase], fingerprint_field="manifest_fingerprint",
        expected_body=phase_body,
    )
    phase_fingerprint = str(phase_document.fingerprint)
    ledger_arguments = _ledger_arguments(
        config=config, protocol=protocol, development=development, phase=phase,
        phase_manifest_fingerprint=phase_fingerprint,
        calibration_completion=calibration_completion,
        calibration_certificate=calibration_certificate,
    )
    normal_completion = output / COMPLETION_NAME[phase]
    abort_completion = output / ABORT_COMPLETION_NAME[phase]
    if abort_completion.is_file():
        completion_path = abort_completion
        manifest_name = ABORT_ARTIFACT_MANIFEST_NAME[phase]
    else:
        completion_path = normal_completion
        manifest_name = ARTIFACT_MANIFEST_NAME[phase]
    names = _artifact_names_from_manifest(
        output, phase, manifest_name=manifest_name
    )
    if abort_completion.is_file() and normal_completion.is_file() and not {
        ARTIFACT_MANIFEST_NAME[phase], COMPLETION_NAME[phase]
    }.issubset(names):
        raise IntegrityError("superseding ABORT does not bind the failed normal completion")
    completion = authenticate_completion_record(
        completion_path, output_directory=output,
        ledger_path=output / LEDGER_NAME[phase],
        expected_ledger_body=build_workspace_access_ledger(**ledger_arguments),
        access_intent_path=output / INTENT_NAME[phase],
        artifact_manifest_path=output / manifest_name,
        expected_artifact_names=names,
    )
    decision = str(completion.value["decision"])
    preflight = authenticate_fingerprinted_json(
        output / PREFLIGHT_NAME[phase], fingerprint_field="preflight_fingerprint"
    )
    runtime_value = _nested(preflight.value, "runtime", where="runtime preflight")
    runtime = _validate_runtime_record(runtime_value, config)
    expected_preflight = _runtime_preflight_body(
        phase=phase, protocol=protocol,
        phase_manifest_fingerprint=phase_fingerprint, runtime=runtime,
    )
    observed_preflight = dict(preflight.value)
    observed_preflight.pop("preflight_fingerprint", None)
    if observed_preflight != expected_preflight:
        raise IntegrityError("saved runtime preflight changed")
    persisted_intent = authenticate_fingerprinted_json(
        output / INTENT_NAME[phase], fingerprint_field="intent_fingerprint"
    )
    _authenticate_numeric_authorization(
        path=output / NUMERIC_AUTHORIZATION_NAME[phase],
        phase=phase,
        protocol=protocol,
        phase_manifest_fingerprint=phase_fingerprint,
        preflight=preflight,
        intent=persisted_intent,
    )
    if ABORT_NAME in names:
        abort = read_strict_json(output / ABORT_NAME)
        expected_decision = CALIBRATION_ABORT if phase == PHASE_CALIBRATION else CONFIRMATION_FAIL
        _require_keys(
            abort,
            {
                "schema", "schema_version", "phase", "protocol_id", "protocol_sha256",
                "phase_manifest_fingerprint", "decision", "error_type", "error_message",
                "numeric_access_cannot_be_repeated", "confirmation_remains_sealed",
            },
            where="runtime ABORT",
        )
        if (
            abort.get("schema") != "metropt_cared_bc_locked_runtime_abort"
            or abort.get("schema_version") != SCHEMA_VERSION
            or abort.get("phase") != phase
            or abort.get("protocol_id") != protocol.value["protocol_id"]
            or abort.get("protocol_sha256") != protocol.record["sha256"]
            or abort.get("phase_manifest_fingerprint") != phase_fingerprint
            or abort.get("decision") != expected_decision
            or decision != expected_decision
            or abort.get("numeric_access_cannot_be_repeated") is not True
            or abort.get("confirmation_remains_sealed") is not (
                phase == PHASE_CALIBRATION
            )
            or not isinstance(abort.get("error_type"), str)
            or not abort.get("error_type")
            or not isinstance(abort.get("error_message"), str)
        ):
            raise IntegrityError("runtime ABORT completion changed")
        for name, frozen in (
            (ENSEMBLE_NAME, development.ensemble),
            (SCORER_NAME, development.scorer),
        ):
            if name in names:
                _assert_artifact(
                    output / name, sha256=frozen.sha256, size_bytes=frozen.size_bytes
                )
        return PhaseOutcome(phase, decision, completion, abort, True), None, None

    if set(names) != set(_normal_artifact_names(phase)):
        raise IntegrityError("normal completed phase artifact set changed")
    _, scorer, fixed_metadata = _read_fixed_models(output, development)
    paths, saved_boundary = _load_evidence(
        output / EVIDENCE_NAME[phase], phase=phase, protocol=protocol,
        phase_manifest_fingerprint=phase_fingerprint, scorer=scorer,
    )
    saved_analysis = read_strict_json(output / ANALYSIS_NAME[phase])
    certificate: AuthenticatedDocument | None
    if phase == PHASE_CALIBRATION:
        recomputed, recomputed_paths, boundary, recomputed_decision = _calibration_analysis(
            paths={
                split: {device: replace(path, alarm=None) for device, path in devices.items()}
                for split, devices in paths.items()
            },
            development=development, protocol=protocol,
            phase_manifest_fingerprint=phase_fingerprint,
        )
        if recomputed != saved_analysis or boundary != saved_boundary:
            raise IntegrityError("saved calibration analysis is not reproducible")
        if boundary is not None:
            for split in CALIBRATION_PARTITIONS:
                for device in DEVICES:
                    if not np.array_equal(
                        recomputed_paths[split][device].alarm, paths[split][device].alarm
                    ):
                        raise IntegrityError("saved calibration alarm path changed")
        numeric_records = _path_array_records(paths)
        certificate_body = _calibration_certificate_body(
            protocol=protocol, development=development,
            phase_manifest_fingerprint=phase_fingerprint,
            analysis=recomputed, evidence_path=output / EVIDENCE_NAME[phase],
            analysis_path=output / ANALYSIS_NAME[phase],
            numeric_records=numeric_records, decision=recomputed_decision,
        )
        certificate = authenticate_fingerprinted_json(
            output / CERTIFICATE_NAME, fingerprint_field="certificate_fingerprint",
            expected_body=certificate_body,
        )
    else:
        if calibration_bundle is None or saved_boundary != calibration_bundle.effective_boundary:
            raise IntegrityError("confirmation evidence changed the calibration boundary")
        recomputed, recomputed_paths, recomputed_decision = _confirmation_analysis(
            paths={
                split: {device: replace(path, alarm=None) for device, path in devices.items()}
                for split, devices in paths.items()
            },
            boundary=calibration_bundle.effective_boundary,
            protocol=protocol, phase_manifest_fingerprint=phase_fingerprint,
            calibration_certificate=calibration_bundle.certificate,
        )
        if recomputed != saved_analysis:
            raise IntegrityError("saved confirmation analysis is not reproducible")
        for device in DEVICES:
            if not np.array_equal(
                recomputed_paths["confirmation"][device].alarm,
                paths["confirmation"][device].alarm,
            ):
                raise IntegrityError("saved confirmation alarm path changed")
        certificate = calibration_bundle.certificate
        boundary = saved_boundary
    if recomputed_decision != decision:
        raise IntegrityError("completion decision differs from recomputed locked gates")
    expected_result = _result_body(
        phase=phase, protocol=protocol,
        phase_manifest_fingerprint=phase_fingerprint,
        decision=recomputed_decision, analysis=recomputed,
        evidence_path=output / EVIDENCE_NAME[phase], certificate=certificate,
    )
    saved_result = read_strict_json(output / RESULT_NAME[phase])
    if saved_result != expected_result:
        raise IntegrityError("saved phase result is not reproducible")
    execution_log = read_strict_json(output / GPU_LOG_NAME[phase])
    _validate_execution_log(
        execution_log, phase=phase, protocol=protocol,
        phase_manifest_fingerprint=phase_fingerprint, preflight=preflight,
        paths=paths, fixed_metadata=fixed_metadata, config=config,
    )
    if _snapshot_directory(development.output_directory) != development.directory_snapshot:
        raise IntegrityError("development PASS changed during completed reentry")
    return (
        PhaseOutcome(phase, decision, completion, saved_result, True),
        certificate, boundary,
    )


def _normalize_config(config: LockedRunConfig) -> LockedRunConfig:
    _validate_config(config)
    workspace = config.workspace_root.resolve(strict=True)
    normalized = replace(
        config,
        workspace_root=workspace,
        csv_path=_absolute(config.csv_path, workspace),
        final_protocol_path=_absolute(config.final_protocol_path, workspace),
        final_protocol_sidecar_path=_absolute(
            config.final_protocol_sidecar_path, workspace
        ),
        output_root=_absolute(config.output_root, workspace),
    )
    _workspace_relative(normalized.csv_path, workspace)
    _workspace_relative(normalized.final_protocol_path, workspace)
    _workspace_relative(normalized.final_protocol_sidecar_path, workspace)
    _workspace_relative(normalized.output_root, workspace)
    return normalized


def _has_completed_phase(config: LockedRunConfig, phase: str) -> bool:
    output = config.phase_directory(phase)
    return (
        (output / COMPLETION_NAME[phase]).is_file()
        or (output / ABORT_COMPLETION_NAME[phase]).is_file()
    )


def _assert_locked_paths_disjoint(
    config: LockedRunConfig, development: DevelopmentPass
) -> None:
    """Reject every output alias before a phase directory or ledger can exist."""

    paths = {
        "development": development.output_directory.resolve(strict=True),
        "calibration": config.phase_directory(PHASE_CALIBRATION).resolve(strict=False),
        "confirmation": config.phase_directory(PHASE_CONFIRMATION).resolve(strict=False),
    }
    for left_name, left in paths.items():
        for right_name, right in paths.items():
            if left_name >= right_name:
                continue
            if left == right or left in right.parents or right in left.parents:
                raise AccessDeniedError(
                    f"locked output paths overlap: {left_name} and {right_name}"
                )


def _obtain_calibration(
    *, config: LockedRunConfig, hooks: LockedHooks,
    protocol: AuthenticatedDocument, development: DevelopmentPass,
) -> tuple[PhaseOutcome, CalibrationBundle | None]:
    output = config.phase_directory(PHASE_CALIBRATION)
    fresh_outcome: PhaseOutcome | None = None
    if _has_completed_phase(config, PHASE_CALIBRATION):
        outcome, certificate, boundary = _authenticate_completed_phase(
            config=config, protocol=protocol, development=development,
            phase=PHASE_CALIBRATION,
        )
    else:
        if output.exists():
            raise AccessDeniedError(
                "partial calibration output exists; sealed numeric access cannot be repeated"
            )
        fresh_outcome = _run_fresh_phase(
            config=config, hooks=hooks, protocol=protocol,
            development=development, phase=PHASE_CALIBRATION,
        )
        outcome, certificate, boundary = _authenticate_completed_phase(
            config=config, protocol=protocol, development=development,
            phase=PHASE_CALIBRATION,
        )
        if outcome.completion.fingerprint != fresh_outcome.completion.fingerprint:
            raise IntegrityError("fresh calibration completion changed during authentication")
        outcome = fresh_outcome
    if outcome.decision != CALIBRATION_PASS:
        return outcome, None
    if certificate is None or boundary is None:
        raise IntegrityError("calibration PASS lacks its authenticated certificate boundary")
    if (
        certificate.value.get("schema")
        != "metropt_cared_bc_locked_calibration_certificate"
        or certificate.value.get("decision") != CALIBRATION_PASS
        or certificate.value.get("passed") is not True
        or certificate.value.get("effective_boundary") != boundary
    ):
        raise AccessDeniedError("calibration certificate cannot unlock confirmation")
    return outcome, CalibrationBundle(outcome, certificate, boundary)


def _obtain_confirmation(
    *, config: LockedRunConfig, hooks: LockedHooks,
    protocol: AuthenticatedDocument, development: DevelopmentPass,
    calibration: CalibrationBundle,
) -> PhaseOutcome:
    output = config.phase_directory(PHASE_CONFIRMATION)
    if _has_completed_phase(config, PHASE_CONFIRMATION):
        outcome, _, boundary = _authenticate_completed_phase(
            config=config, protocol=protocol, development=development,
            phase=PHASE_CONFIRMATION, calibration_bundle=calibration,
        )
        if boundary != calibration.effective_boundary:
            raise IntegrityError("completed confirmation changed its locked boundary")
        return outcome
    if output.exists():
        raise AccessDeniedError(
            "partial confirmation output exists; sealed numeric access cannot be repeated"
        )
    fresh = _run_fresh_phase(
        config=config, hooks=hooks, protocol=protocol,
        development=development, phase=PHASE_CONFIRMATION,
        calibration_bundle=calibration,
    )
    verified, _, boundary = _authenticate_completed_phase(
        config=config, protocol=protocol, development=development,
        phase=PHASE_CONFIRMATION, calibration_bundle=calibration,
    )
    if (
        verified.completion.fingerprint != fresh.completion.fingerprint
        or boundary != calibration.effective_boundary
    ):
        raise IntegrityError("fresh confirmation completion changed during authentication")
    return fresh


def _run_locked_experiment_core(
    config: LockedRunConfig, *, hooks: LockedHooks,
) -> LockedRunOutcome:
    """Run or authenticate the sealed calibration/confirmation experiment.

    Authentication reads only protocol and saved artifact bytes.  A completed
    phase is reconstructed without invoking runtime hooks or opening the CSV.
    """

    normalized = _normalize_config(config)
    protocol = _authenticate_final_protocol(normalized)
    if (
        normalized._execution_mode == "synthetic_test"
        and not str(protocol.value.get("protocol_id", "")).startswith("synthetic_")
    ):
        raise AccessDeniedError("test boundary cannot issue artifacts for a formal protocol")
    development = _authenticate_development_pass(normalized, protocol)
    _assert_locked_paths_disjoint(normalized, development)
    calibration_outcome: PhaseOutcome | None = None
    confirmation_outcome: PhaseOutcome | None = None

    if normalized.phase in (PHASE_CALIBRATION, "all"):
        calibration_outcome, calibration_bundle = _obtain_calibration(
            config=normalized, hooks=hooks,
            protocol=protocol, development=development,
        )
        if normalized.phase == "all" and calibration_bundle is not None:
            confirmation_outcome = _obtain_confirmation(
                config=normalized, hooks=hooks,
                protocol=protocol, development=development,
                calibration=calibration_bundle,
            )
    else:
        if not _has_completed_phase(normalized, PHASE_CALIBRATION):
            raise AccessDeniedError(
                "confirmation remains sealed until calibration has an authenticated completion"
            )
        calibration_outcome, calibration_bundle = _obtain_calibration(
            config=normalized, hooks=hooks,
            protocol=protocol, development=development,
        )
        if calibration_bundle is None:
            raise AccessDeniedError(
                "confirmation remains sealed because calibration did not PASS"
            )
        confirmation_outcome = _obtain_confirmation(
            config=normalized, hooks=hooks,
            protocol=protocol, development=development,
            calibration=calibration_bundle,
        )
    return LockedRunOutcome(calibration_outcome, confirmation_outcome)


def run_locked_experiment(config: LockedRunConfig) -> LockedRunOutcome:
    """Formal entry point; its numerical boundary is not injectable."""

    if config._execution_mode != "production":
        raise AccessDeniedError("formal entry point requires production execution mode")
    return _run_locked_experiment_core(config, hooks=default_locked_hooks())


def run_calibration(config: LockedRunConfig) -> PhaseOutcome:
    outcome = run_locked_experiment(replace(config, phase=PHASE_CALIBRATION))
    if outcome.calibration is None:
        raise AssertionError("calibration outcome is missing")
    return outcome.calibration


def run_confirmation(config: LockedRunConfig) -> PhaseOutcome:
    outcome = run_locked_experiment(replace(config, phase=PHASE_CONFIRMATION))
    if outcome.confirmation is None:
        raise AssertionError("confirmation outcome is missing")
    return outcome.confirmation


def _run_locked_experiment_for_test(
    config: LockedRunConfig, *, hooks: LockedHooks,
) -> LockedRunOutcome:
    """Synthetic-only seam; its phase manifest is permanently non-formal."""

    if "PYTEST_CURRENT_TEST" not in os.environ:
        raise AccessDeniedError("synthetic hook boundary is available only under pytest")
    return _run_locked_experiment_core(
        replace(config, _execution_mode="synthetic_test"), hooks=hooks
    )


def _run_calibration_for_test(
    config: LockedRunConfig, *, hooks: LockedHooks,
) -> PhaseOutcome:
    outcome = _run_locked_experiment_for_test(
        replace(config, phase=PHASE_CALIBRATION), hooks=hooks
    )
    if outcome.calibration is None:
        raise AssertionError("calibration outcome is missing")
    return outcome.calibration


def _run_confirmation_for_test(
    config: LockedRunConfig, *, hooks: LockedHooks,
) -> PhaseOutcome:
    outcome = _run_locked_experiment_for_test(
        replace(config, phase=PHASE_CONFIRMATION), hooks=hooks
    )
    if outcome.confirmation is None:
        raise AssertionError("confirmation outcome is missing")
    return outcome.confirmation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--final-protocol", type=Path, required=True)
    parser.add_argument("--final-protocol-sidecar", type=Path, required=True)
    parser.add_argument("--final-protocol-sha256", required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/metropt_cared_bc_locked_gpu"),
    )
    parser.add_argument(
        "--phase", choices=(PHASE_CALIBRATION, PHASE_CONFIRMATION, "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--csv-chunksize", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outcome = run_locked_experiment(
        LockedRunConfig(
            csv_path=args.csv_path,
            final_protocol_path=args.final_protocol,
            final_protocol_sidecar_path=args.final_protocol_sidecar,
            expected_final_protocol_sha256=args.final_protocol_sha256,
            output_root=args.output_root,
            phase=args.phase, device=args.device, dtype=args.dtype,
            batch_size=args.batch_size, csv_chunksize=args.csv_chunksize,
        )
    )
    summary = {
        "calibration": (
            None if outcome.calibration is None else {
                "decision": outcome.calibration.decision,
                "reused_completed_bundle": outcome.calibration.reused_completed_bundle,
                "completion_fingerprint": outcome.calibration.completion.fingerprint,
            }
        ),
        "confirmation": (
            None if outcome.confirmation is None else {
                "decision": outcome.confirmation.decision,
                "reused_completed_bundle": outcome.confirmation.reused_completed_bundle,
                "completion_fingerprint": outcome.confirmation.completion.fingerprint,
            }
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
