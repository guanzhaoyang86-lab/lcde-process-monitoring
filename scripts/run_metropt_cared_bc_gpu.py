#!/usr/bin/env python
"""Development-only GPU runner for the frozen MetroPT CARED-BC charter.

This entry point intentionally has no calibration or confirmation execution
path.  It authenticates source bytes, durably claims development access, writes
an access intent, and only then asks the phase-isolated data layer for the three
development partitions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import sys
import time
from typing import Any, Callable, Mapping

import torch
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mate_spc_metropt.data import (  # noqa: E402
    ANALOG_COLUMNS,
    APPROVED_NUMERIC_COLUMNS,
    DIRECT_ALARM_COLUMNS,
    SAFE_CONTROL_COLUMNS,
    MetroPTMinuteInterval,
    MetroPTTimestampAudit,
    audit_metropt_timestamps,
    load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (  # noqa: E402
    CHARTER_ID,
    DEVELOPMENT_ABORT,
    DEVELOPMENT_CHARTER_SHA256,
    DEVELOPMENT_PASS,
    PARTITIONS,
    PHASE_CALIBRATION,
    PHASE_CONFIRMATION,
    PHASE_DEVELOPMENT,
    PHASE_PARTITIONS,
    SOURCE_FILE_SPECS,
    AccessDeniedError,
    AuthenticatedDocument,
    DevelopmentInputAuthentication,
    GovernanceError,
    IntegrityError,
    LedgerClaim,
    artifact_record,
    atomic_create_new_json,
    authenticate_completion_record,
    authenticate_development_inputs,
    authenticate_fingerprinted_json,
    authenticate_json_with_sidecar,
    build_workspace_access_ledger,
    canonical_fingerprint,
    claim_workspace_access_ledger,
    create_artifact_manifest,
    create_completion_record,
    create_fingerprinted_json,
    create_phase_access_intent,
    read_strict_json,
)
from mate_spc_metropt.model import (  # noqa: E402
    ANALOG_NAMES,
    SAFE_CONTROL_NAMES,
    AlignedResidualBatch,
    CaredBCScorer,
    CaredScorePath,
    CaredTrainingResult,
    InferenceWindowBatch,
    WindowBatch,
    align_inference_forecasts_to_score_times,
    build_causal_input_windows,
    build_causal_windows,
    cared_architecture_spec,
    fit_cared_scorer,
    train_cared_ensemble,
)


RUNNER_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = "outputs/metropt_cared_bc_gpu_v0_1"
DEFAULT_CHARTER_PATH = "protocols/metropt_cared_bc_v0_development_charter.json"
DEFAULT_CHARTER_SIDECAR_PATH = (
    "protocols/metropt_cared_bc_v0_development_charter.sha256"
)
DEFAULT_PARENT_AMENDMENT_PATH = (
    "protocols/metropt_cared_bc_v0_preaccess_amendment_001.json"
)
DEFAULT_PARENT_AMENDMENT_SIDECAR_PATH = (
    "protocols/metropt_cared_bc_v0_preaccess_amendment_001.sha256"
)
PARENT_AMENDMENT_ID = "metropt_cared_bc_v0_preaccess_amendment_001"
PARENT_AMENDMENT_SHA256 = (
    "9b998cd4866097443971f0f1fe6a18b6f9aba26964c164ed6a47d7684143bc83"
)
DEFAULT_AMENDMENT_PATH = (
    "protocols/metropt_cared_bc_v0_preaccess_amendment_002.json"
)
DEFAULT_AMENDMENT_SIDECAR_PATH = (
    "protocols/metropt_cared_bc_v0_preaccess_amendment_002.sha256"
)
AMENDMENT_ID = "metropt_cared_bc_v0_preaccess_amendment_002"
AMENDMENT_SHA256 = "cf68a99a74ac1505975987785222d3095c8243394d13f60d5134ca8b4fe7f9bc"
FAILED_DEVELOPMENT_OUTPUT = "outputs/metropt_cared_bc_gpu_v0"
RECOVERY_OUTPUT = "outputs/metropt_cared_bc_gpu_v0_1"
FAILED_ABORT_SHA256 = "08d570c0a044a56cf5b91237ceb8a3c501aacfb84e05e2c4ddf9cb13a3ccaf14"
FAILED_COMPLETION_SHA256 = (
    "c57d421901ea1f38ea14d1412740439f5a2276cca15b51df27f83a74a3f4ebf6"
)
FAILED_COMPLETION_FINGERPRINT = (
    "305fe648c3a09e51fab2af0524902ed06c869d6098b660f5896630dbf10d30ee"
)
TIMESTAMP_DAILY_AUDIT_PATH = (
    "protocols/metropt_dev_healthy_timestamp_daily_audit.json"
)
TIMESTAMP_DAILY_AUDIT_SHA256 = (
    "f3ece710913539ce8961c9eac981b835d97c6bda70aa3c32c44e5c87def0a9cd"
)
COVERAGE_DIAGNOSTIC_PATH = (
    "outputs/metropt_cared_bc_v0_postabort_diagnostics/"
    "development_coverage_diagnostic.json"
)
COVERAGE_DIAGNOSTIC_SHA256 = (
    "764f321b36f038d85d62de08c71347e068ac15cfc8f8970ba44260d3bd9817cc"
)
LEDGER_NAME = "development_access_ledger.json"
PHASE_MANIFEST_NAME = "development_phase_manifest.json"
ACCESS_INTENT_NAME = "development_access_intent.json"
TIMESTAMP_AUDIT_NAME = "timestamp_audit.json"
ENSEMBLE_NAME = "cared_ensemble.npz"
SCORER_NAME = "cared_scorer.npz"
RESULT_NAME = "development_result.json"
GPU_LOG_NAME = "gpu_execution_development.json"
ARTIFACT_MANIFEST_NAME = "development_artifact_manifest.json"
COMPLETION_NAME = "development_completion.json"
ABORT_NAME = "ABORTED.json"
PARITY_SAMPLE_WINDOWS = 2048
PARITY_ATOL = 1.0e-9
REQUIRED_HEALTHY_DAYS = 9
MINIMUM_QUALIFIED_HEALTHY_DAYS = 6
MINIMUM_VALID_SCORES_PER_HEALTHY_DAY = 1141

DEVELOPMENT_ARTIFACT_NAMES: tuple[str, ...] = (
    PHASE_MANIFEST_NAME,
    TIMESTAMP_AUDIT_NAME,
    ENSEMBLE_NAME,
    SCORER_NAME,
    RESULT_NAME,
    GPU_LOG_NAME,
)

_PARTITION_BY_NAME = {partition.name: partition for partition in PARTITIONS}
_SENSOR_SPEC = next(spec for spec in SOURCE_FILE_SPECS if spec.key == "sensor_csv")


class DevelopmentRunnerError(RuntimeError):
    """Raised when the development-only runner cannot complete safely."""


class InsufficientQualifiedDaysError(DevelopmentRunnerError):
    """Raised after all nine candidate days have been explicitly audited."""

    def __init__(self, day_records: list[dict[str, Any]]) -> None:
        self.day_records = tuple(dict(record) for record in day_records)
        self.qualified_day_count = sum(
            record.get("qualified") is True for record in self.day_records
        )
        super().__init__(
            "dev_healthy produced "
            f"{self.qualified_day_count} qualified calendar days; at least "
            f"{MINIMUM_QUALIFIED_HEALTHY_DAYS} are required"
        )


@dataclass(frozen=True)
class DevelopmentRunConfig:
    csv_path: Path
    output_directory: Path
    workspace_root: Path = WORKSPACE_ROOT
    charter_path: Path = WORKSPACE_ROOT / DEFAULT_CHARTER_PATH
    charter_sidecar_path: Path = WORKSPACE_ROOT / DEFAULT_CHARTER_SIDECAR_PATH
    parent_amendment_path: Path = WORKSPACE_ROOT / DEFAULT_PARENT_AMENDMENT_PATH
    parent_amendment_sidecar_path: Path = (
        WORKSPACE_ROOT / DEFAULT_PARENT_AMENDMENT_SIDECAR_PATH
    )
    amendment_path: Path = WORKSPACE_ROOT / DEFAULT_AMENDMENT_PATH
    amendment_sidecar_path: Path = WORKSPACE_ROOT / DEFAULT_AMENDMENT_SIDECAR_PATH
    phase: str = PHASE_DEVELOPMENT
    device: str = "cuda"
    dtype: str = "float64"
    batch_size: int = 1024
    csv_chunksize: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "csv_path",
            "output_directory",
            "workspace_root",
            "charter_path",
            "charter_sidecar_path",
            "parent_amendment_path",
            "parent_amendment_sidecar_path",
            "amendment_path",
            "amendment_sidecar_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.phase not in (
            PHASE_DEVELOPMENT,
            PHASE_CALIBRATION,
            PHASE_CONFIRMATION,
        ):
            raise DevelopmentRunnerError(f"unsupported phase: {self.phase!r}")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise DevelopmentRunnerError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise DevelopmentRunnerError("batch_size must be a positive integer")
        if not isinstance(self.csv_chunksize, int) or isinstance(self.csv_chunksize, bool):
            raise DevelopmentRunnerError("csv_chunksize must be a positive integer")
        if self.csv_chunksize <= 0:
            raise DevelopmentRunnerError("csv_chunksize must be a positive integer")


@dataclass(frozen=True)
class DevelopmentHooks:
    """Injectable numerical boundary used by synthetic runner tests."""

    authenticate_inputs: Callable[..., DevelopmentInputAuthentication]
    authenticate_amendment: Callable[..., AuthenticatedDocument]
    authenticate_recovery_bindings: Callable[..., Mapping[str, Any]]
    timestamp_audit: Callable[..., MetroPTTimestampAudit]
    load_partition: Callable[..., MetroPTMinuteInterval]
    build_training_windows: Callable[..., WindowBatch]
    build_inference_windows: Callable[..., InferenceWindowBatch]
    train_ensemble: Callable[..., CaredTrainingResult]
    align_inference_forecasts: Callable[..., AlignedResidualBatch]
    fit_scorer: Callable[..., CaredBCScorer]
    runtime_metadata: Callable[[DevelopmentRunConfig], Mapping[str, Any]]
    parity_check: Callable[..., Mapping[str, Any]]
    causality_check: Callable[..., bool]


@dataclass(frozen=True)
class DevelopmentRunOutcome:
    decision: str
    completion: AuthenticatedDocument
    result: Mapping[str, Any]
    reused_completed_bundle: bool


def _default_runtime_metadata(config: DevelopmentRunConfig) -> dict[str, Any]:
    requested = torch.device(config.device)
    if requested.type != "cuda":
        raise DevelopmentRunnerError("CARED-BC development requires --device cuda")
    if not torch.cuda.is_available():
        raise DevelopmentRunnerError("CUDA was requested but torch.cuda.is_available() is false")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    return {
        "backend": "torch",
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "device": str(requested),
        "device_index": int(index),
        "device_name": str(torch.cuda.get_device_name(index)),
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "deterministic_algorithms_requested": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
    }


def default_development_hooks() -> DevelopmentHooks:
    return DevelopmentHooks(
        authenticate_inputs=authenticate_development_inputs,
        authenticate_amendment=authenticate_json_with_sidecar,
        authenticate_recovery_bindings=authenticate_recovery_bindings,
        timestamp_audit=audit_metropt_timestamps,
        load_partition=load_governed_metropt_partition,
        build_training_windows=build_causal_windows,
        build_inference_windows=build_causal_input_windows,
        train_ensemble=train_cared_ensemble,
        align_inference_forecasts=align_inference_forecasts_to_score_times,
        fit_scorer=fit_cared_scorer,
        runtime_metadata=_default_runtime_metadata,
        parity_check=check_cpu_cuda_sample_parity,
        causality_check=check_development_causality,
    )


def _absolute_without_access(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return Path(candidate.absolute())


def _validate_development_config(config: DevelopmentRunConfig) -> None:
    if config.phase != PHASE_DEVELOPMENT:
        raise AccessDeniedError(
            "this runner is development-only; calibration and confirmation numeric "
            "access remain sealed"
        )
    if config.dtype != "float64":
        raise DevelopmentRunnerError("the frozen development dtype is float64")
    if not config.device.lower().startswith("cuda"):
        raise DevelopmentRunnerError("the frozen development device is CUDA")


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    root = workspace_root.resolve(strict=True)
    candidate = path.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise DevelopmentRunnerError(
            "output_directory must remain inside workspace_root for ledger identity"
        ) from exc
    if not relative.parts:
        raise DevelopmentRunnerError("output_directory must not equal workspace_root")
    return PurePosixPath(*relative.parts).as_posix()


def _source_records(authentication: DevelopmentInputAuthentication) -> dict[str, Any]:
    records = authentication.sources
    if set(records) != {spec.key for spec in SOURCE_FILE_SPECS}:
        raise IntegrityError("authenticated source record set is incomplete")
    return {key: dict(value) for key, value in sorted(records.items())}


def validate_preaccess_amendment(amendment: AuthenticatedDocument) -> None:
    """Validate the original amendment that remains the recovery parent."""

    value = amendment.value
    if (
        value.get("schema") != "metropt_cared_bc_preaccess_amendment"
        or value.get("schema_version") != 1
        or value.get("amendment_id") != PARENT_AMENDMENT_ID
        or value.get("status") != "FROZEN_BEFORE_ANY_SENSOR_NUMERIC_ACCESS"
    ):
        raise IntegrityError("pre-access amendment identity or status changed")
    parent = value.get("parent_charter")
    if not isinstance(parent, Mapping) or parent.get("sha256") != DEVELOPMENT_CHARTER_SHA256:
        raise IntegrityError("pre-access amendment parent charter changed")
    scope = value.get("scope")
    governed = value.get("governed_numeric_entry")
    day_rule = value.get("complete_and_qualified_days")
    if (
        not isinstance(scope, Mapping)
        or scope.get("sensor_numeric_values_accessed_before_freeze") is not False
        or scope.get("sealed_partitions_remain_sealed") is not True
        or not isinstance(governed, Mapping)
        or governed.get("formal_runner_must_use_authorized_partition_loader") is not True
        or not isinstance(day_rule, Mapping)
        or day_rule.get("minimum_valid_score_count_inclusive")
        != MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
    ):
        raise IntegrityError("pre-access amendment safety contract changed")
    if amendment.record.get("sha256") != PARENT_AMENDMENT_SHA256:
        raise IntegrityError("parent pre-access amendment SHA-256 changed")


def validate_recovery_amendment(
    amendment: AuthenticatedDocument,
    parent_amendment: AuthenticatedDocument,
) -> None:
    """Validate amendment 002 and its exact parent/evidence declarations."""

    value = amendment.value
    if (
        value.get("schema") != "metropt_cared_bc_development_recovery_amendment"
        or value.get("schema_version") != 1
        or value.get("amendment_id") != AMENDMENT_ID
        or value.get("status")
        != (
            "FROZEN_BEFORE_REPEATED_DEVELOPMENT_RUN_AND_BEFORE_ANY_"
            "CALIBRATION_OR_CONFIRMATION_ACCESS"
        )
        or amendment.record.get("sha256") != AMENDMENT_SHA256
    ):
        raise IntegrityError("recovery amendment identity or status changed")
    parent = value.get("parent_bindings")
    if (
        not isinstance(parent, Mapping)
        or parent.get("development_charter_sha256") != DEVELOPMENT_CHARTER_SHA256
        or parent.get("preaccess_amendment_001_sha256")
        != PARENT_AMENDMENT_SHA256
        or parent_amendment.record.get("sha256") != PARENT_AMENDMENT_SHA256
    ):
        raise IntegrityError("recovery amendment parent binding changed")

    evidence = value.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("timestamp_only_daily_audit_path")
        != TIMESTAMP_DAILY_AUDIT_PATH
        or evidence.get("timestamp_only_daily_audit_sha256")
        != TIMESTAMP_DAILY_AUDIT_SHA256
        or evidence.get("development_coverage_diagnostic_path")
        != COVERAGE_DIAGNOSTIC_PATH
        or evidence.get("development_coverage_diagnostic_sha256")
        != COVERAGE_DIAGNOSTIC_SHA256
    ):
        raise IntegrityError("recovery amendment evidence binding changed")

    failed = value.get("failed_run_binding")
    if (
        not isinstance(failed, Mapping)
        or failed.get("output_directory") != FAILED_DEVELOPMENT_OUTPUT
        or failed.get("decision") != DEVELOPMENT_ABORT
        or failed.get("reason")
        != "dev_healthy did not produce exactly nine complete calendar-day score blocks"
        or failed.get("abort_record_sha256") != FAILED_ABORT_SHA256
        or failed.get("completion_sha256") != FAILED_COMPLETION_SHA256
        or failed.get("completion_fingerprint") != FAILED_COMPLETION_FINGERPRINT
        or failed.get("model_artifact_was_saved") is not False
        or failed.get("scorer_or_power_gate_was_evaluated") is not False
        or failed.get("calibration_numeric_access_occurred") is not False
        or failed.get("confirmation_numeric_access_occurred") is not False
    ):
        raise IntegrityError("recovery amendment failed-run binding changed")

    revision = value.get("development_threshold_revision")
    unchanged = value.get("unchanged_requirements")
    recovery = value.get("recovery_run")
    sealed = [
        partition.name
        for partition in PARTITIONS
        if partition.name not in PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    ]
    if (
        not isinstance(revision, Mapping)
        or revision.get("minimum_qualified_development_days")
        != MINIMUM_QUALIFIED_HEALTHY_DAYS
        or revision.get("qualification_rule_unchanged")
        != "At least 1141 causal scores out of the theoretical 1426."
        or revision.get("unqualified_dates")
        != "Excluded from the provisional maximum and reported with their score counts and reason."
        or revision.get("insufficient_qualified_days_action") != DEVELOPMENT_ABORT
        or not isinstance(unchanged, Mapping)
        or unchanged.get("method_family") != "CARED-BC"
        or unchanged.get("minimum_score_count_per_qualified_day")
        != MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
        or unchanged.get("tcn_improvement_gate") != 0.05
        or unchanged.get("f1_latency_gate_minutes") != 60
        or unchanged.get("f1_peak_to_boundary_ratio_gate") != 1.25
        or unchanged.get("sealed_partitions") != sealed
        or not isinstance(recovery, Mapping)
        or recovery.get("new_output_directory_required") != RECOVERY_OUTPUT
        or recovery.get("old_failed_bundle_must_remain_unchanged") is not True
        or recovery.get("development_numeric_partitions_may_be_reused")
        != list(PHASE_PARTITIONS[PHASE_DEVELOPMENT])
        or recovery.get("calibration_or_confirmation_access_authorized") is not False
        or recovery.get("switch_method_family_if_recovery_power_gate_fails") is not True
    ):
        raise IntegrityError("recovery amendment experiment contract changed")


def _workspace_bound_recovery_file(workspace: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if (
        not relative_name
        or "\\" in relative_name
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise IntegrityError("recovery binding contains an unsafe path")
    root = workspace.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise IntegrityError("recovery binding escapes the workspace") from error
    return candidate


def _authenticate_plain_recovery_json(
    path: Path, expected_sha256: str
) -> AuthenticatedDocument:
    before = artifact_record(path)
    if before.get("sha256") != expected_sha256:
        raise IntegrityError(f"recovery evidence hash changed: {path.name}")
    value = read_strict_json(path)
    after = artifact_record(path)
    if after != before:
        raise IntegrityError(f"recovery evidence changed while authenticating: {path.name}")
    return AuthenticatedDocument(value=value, record=before)


def authenticate_recovery_bindings(
    workspace_root: Path,
    recovery_amendment: AuthenticatedDocument,
) -> dict[str, Any]:
    """Authenticate the evidence and complete failed-run chain pinned by 002."""

    workspace = workspace_root.resolve(strict=True)
    evidence = recovery_amendment.value["evidence"]
    failed = recovery_amendment.value["failed_run_binding"]
    timestamp_path = _workspace_bound_recovery_file(
        workspace, str(evidence["timestamp_only_daily_audit_path"])
    )
    coverage_path = _workspace_bound_recovery_file(
        workspace, str(evidence["development_coverage_diagnostic_path"])
    )
    timestamp_document = _authenticate_plain_recovery_json(
        timestamp_path, TIMESTAMP_DAILY_AUDIT_SHA256
    )
    coverage_document = _authenticate_plain_recovery_json(
        coverage_path, COVERAGE_DIAGNOSTIC_SHA256
    )
    expected_dates = [
        str(date)
        for date in np.arange(
            np.datetime64(_PARTITION_BY_NAME["dev_healthy"].start, "D"),
            np.datetime64(_PARTITION_BY_NAME["dev_healthy"].end, "D"),
            dtype="datetime64[D]",
        )
    ]
    daily_counts = timestamp_document.value.get("daily_timestamp_counts")
    if (
        timestamp_document.value.get("schema")
        != "metropt_timestamp_only_daily_partition_audit"
        or timestamp_document.value.get("source_csv_sha256") != _SENSOR_SPEC.sha256
        or timestamp_document.value.get("partition") != "dev_healthy"
        or timestamp_document.value.get("timestamp_column_only") is not True
        or timestamp_document.value.get("sensor_numeric_values_parsed") is not False
        or timestamp_document.value.get("total_timestamp_count")
        != _PARTITION_BY_NAME["dev_healthy"].row_count
        or not isinstance(daily_counts, Mapping)
        or list(daily_counts) != expected_dates
        or daily_counts.get("2020-02-29") != 0
    ):
        raise IntegrityError("timestamp-only recovery evidence changed semantically")
    days = coverage_document.value.get("days")
    if (
        coverage_document.value.get("schema")
        != "metropt_cared_bc_development_coverage_diagnostic"
        or coverage_document.value.get("source_csv_sha256") != _SENSOR_SPEC.sha256
        or coverage_document.value.get("partition") != "dev_healthy"
        or coverage_document.value.get("sensor_numeric_scope") != "development_only"
        or coverage_document.value.get("sealed_partition_access") is not False
        or not isinstance(days, list)
        or [item.get("date") for item in days if isinstance(item, Mapping)]
        != expected_dates
        or sum(
            item.get("qualified_at_1141") is True
            for item in days
            if isinstance(item, Mapping)
        )
        != MINIMUM_QUALIFIED_HEALTHY_DAYS
    ):
        raise IntegrityError("development coverage recovery evidence changed semantically")

    failed_output = _workspace_bound_recovery_file(
        workspace, str(failed["output_directory"])
    )
    if not failed_output.is_dir() or failed_output.is_symlink():
        raise IntegrityError("frozen failed-run bundle is not a real directory")
    abort_path = failed_output / ABORT_NAME
    abort_document = _authenticate_plain_recovery_json(
        abort_path, FAILED_ABORT_SHA256
    )
    if (
        abort_document.value.get("decision") != DEVELOPMENT_ABORT
        or abort_document.value.get("phase") != PHASE_DEVELOPMENT
        or abort_document.value.get("error_message") != failed.get("reason")
        or abort_document.value.get("calibration_numeric_access_authorized") is not False
        or abort_document.value.get("confirmation_numeric_access_authorized") is not False
    ):
        raise IntegrityError("frozen failed-run abort record changed semantically")

    ledger_path = failed_output / LEDGER_NAME
    ledger_value = read_strict_json(ledger_path)
    expected_ledger_body = dict(ledger_value)
    expected_ledger_body.pop("ledger_fingerprint", None)
    artifact_manifest_path = failed_output / ARTIFACT_MANIFEST_NAME
    artifact_names = _artifact_names_from_manifest(artifact_manifest_path)
    completion = authenticate_completion_record(
        failed_output / COMPLETION_NAME,
        output_directory=failed_output,
        ledger_path=ledger_path,
        expected_ledger_body=expected_ledger_body,
        access_intent_path=failed_output / ACCESS_INTENT_NAME,
        artifact_manifest_path=artifact_manifest_path,
        expected_artifact_names=artifact_names,
    )
    if (
        completion.record.get("sha256") != FAILED_COMPLETION_SHA256
        or completion.fingerprint != FAILED_COMPLETION_FINGERPRINT
        or completion.value.get("decision") != DEVELOPMENT_ABORT
    ):
        raise IntegrityError("frozen failed-run completion binding changed")
    for forbidden_artifact in (ENSEMBLE_NAME, SCORER_NAME, RESULT_NAME, GPU_LOG_NAME):
        if (failed_output / forbidden_artifact).exists():
            raise IntegrityError("failed bundle unexpectedly contains a fitted artifact")

    return {
        "schema": "metropt_cared_bc_recovery_binding_authentication",
        "schema_version": 1,
        "timestamp_only_daily_audit": dict(timestamp_document.record),
        "development_coverage_diagnostic": dict(coverage_document.record),
        "failed_run": {
            "output_directory": FAILED_DEVELOPMENT_OUTPUT,
            "abort_record": dict(abort_document.record),
            "completion_record": dict(completion.record),
            "completion_fingerprint": completion.fingerprint,
            "decision": DEVELOPMENT_ABORT,
            "fitted_artifacts_absent": True,
        },
    }


def validate_recovery_bindings(bindings: Mapping[str, Any]) -> None:
    timestamp = bindings.get("timestamp_only_daily_audit")
    coverage = bindings.get("development_coverage_diagnostic")
    failed = bindings.get("failed_run")
    if (
        bindings.get("schema")
        != "metropt_cared_bc_recovery_binding_authentication"
        or bindings.get("schema_version") != 1
        or not isinstance(timestamp, Mapping)
        or timestamp.get("sha256") != TIMESTAMP_DAILY_AUDIT_SHA256
        or not isinstance(coverage, Mapping)
        or coverage.get("sha256") != COVERAGE_DIAGNOSTIC_SHA256
        or not isinstance(failed, Mapping)
        or failed.get("output_directory") != FAILED_DEVELOPMENT_OUTPUT
        or not isinstance(failed.get("abort_record"), Mapping)
        or failed["abort_record"].get("sha256") != FAILED_ABORT_SHA256
        or not isinstance(failed.get("completion_record"), Mapping)
        or failed["completion_record"].get("sha256") != FAILED_COMPLETION_SHA256
        or failed.get("completion_fingerprint") != FAILED_COMPLETION_FINGERPRINT
        or failed.get("decision") != DEVELOPMENT_ABORT
        or failed.get("fitted_artifacts_absent") is not True
    ):
        raise IntegrityError("authenticated recovery bindings are incomplete or changed")


def build_development_phase_manifest(
    config: DevelopmentRunConfig,
    authentication: DevelopmentInputAuthentication,
    parent_amendment: AuthenticatedDocument,
    amendment: AuthenticatedDocument,
    recovery_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable semantic manifest before any numeric phase access."""

    workspace = config.workspace_root.resolve(strict=True)
    output = _absolute_without_access(config.output_directory, workspace)
    output_relative = _workspace_relative(output, workspace)
    if (
        output_relative == FAILED_DEVELOPMENT_OUTPUT
        or output_relative.startswith(f"{FAILED_DEVELOPMENT_OUTPUT}/")
    ):
        raise AccessDeniedError("recovery output must not modify the old failed bundle")
    sources = _source_records(authentication)
    sensor_record = sources["sensor_csv"]
    if sensor_record.get("relative_path") != _SENSOR_SPEC.relative_path:
        raise IntegrityError("authenticated sensor CSV relative path changed")
    if sensor_record.get("sha256") != _SENSOR_SPEC.sha256:
        raise IntegrityError("authenticated sensor CSV SHA-256 changed")

    expected_csv = workspace.joinpath(*PurePosixPath(_SENSOR_SPEC.relative_path).parts)
    supplied_csv = _absolute_without_access(config.csv_path, workspace)
    if supplied_csv.resolve(strict=True) != expected_csv.resolve(strict=True):
        raise IntegrityError("csv_path is not the charter-bound MetroPT sensor CSV")

    permitted = [
        _PARTITION_BY_NAME[name].as_dict()
        for name in PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    ]
    forbidden = [
        partition.as_dict()
        for partition in PARTITIONS
        if partition.name not in PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    ]
    ledger_relative = (
        PurePosixPath(output_relative) / LEDGER_NAME
    ).as_posix()
    return {
        "schema": "metropt_cared_bc_development_phase_manifest",
        "schema_version": RUNNER_SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": CHARTER_ID,
        "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
        "charter_record": dict(authentication.charter.record),
        "preaccess_amendment": {
            "amendment_id": AMENDMENT_ID,
            "sha256": AMENDMENT_SHA256,
            "record": dict(amendment.record),
        },
        "parent_preaccess_amendment": {
            "amendment_id": PARENT_AMENDMENT_ID,
            "sha256": PARENT_AMENDMENT_SHA256,
            "record": dict(parent_amendment.record),
        },
        "recovery_bindings": dict(recovery_bindings),
        "source_fingerprint": authentication.source_fingerprint,
        "sources": sources,
        "sensor_csv_relative_path": _SENSOR_SPEC.relative_path,
        "output_directory_relative_path": output_relative,
        "ledger_workspace_relative_path": ledger_relative,
        "numeric_partitions": permitted,
        "sealed_partitions": forbidden,
        "interval_semantics": "left_closed_right_open",
        "architecture": cared_architecture_spec(),
        "runtime": {
            "device": config.device,
            "dtype": config.dtype,
            "batch_size": config.batch_size,
            "csv_chunksize": config.csv_chunksize,
        },
        "approved_input_columns": list(APPROVED_NUMERIC_COLUMNS),
        "direct_alarm_columns_excluded": list(DIRECT_ALARM_COLUMNS),
        "development_threshold": {
            "rule": "maximum_qualified_dev_healthy_day_cusum_maximum",
            "candidate_days": REQUIRED_HEALTHY_DAYS,
            "minimum_qualified_days": MINIMUM_QUALIFIED_HEALTHY_DAYS,
            "minimum_valid_scores_per_day": MINIMUM_VALID_SCORES_PER_HEALTHY_DAY,
            "unqualified_days_excluded_from_maximum": True,
            "zero_score_days_explicitly_reported": True,
            "comparison": "strictly_greater_than",
            "provisional_only": True,
        },
        "development_gates": {
            "minimum_relative_improvement_over_persistence": 0.05,
            "f1_alarm_required": True,
            "maximum_f1_first_alarm_delay_minutes": 60.0,
            "minimum_f1_maximum_to_threshold_ratio": 1.25,
            "cpu_cuda_alarm_exact": True,
            "parity_atol": PARITY_ATOL,
        },
        "artifact_names": list(DEVELOPMENT_ARTIFACT_NAMES),
        "future_numeric_access_authorized": False,
        "recovery_run": True,
        "old_failed_bundle_must_remain_unchanged": True,
    }


def _ledger_arguments(
    config: DevelopmentRunConfig,
    phase_manifest: Mapping[str, Any],
    phase_manifest_fingerprint: str,
) -> dict[str, Any]:
    sources = phase_manifest.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("sensor_csv"), Mapping
    ):
        raise IntegrityError("phase manifest has no authenticated sensor source")
    return {
        "phase": PHASE_DEVELOPMENT,
        "workspace_relative_path": str(
            phase_manifest["ledger_workspace_relative_path"]
        ),
        "protocol_id": CHARTER_ID,
        "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": str(sources["sensor_csv"]["sha256"]),
        "source_fingerprint": str(phase_manifest["source_fingerprint"]),
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "output_directory": config.output_directory,
    }


def _validate_timestamp_audit(
    audit: MetroPTTimestampAudit,
    authentication: DevelopmentInputAuthentication,
) -> None:
    sensor = authentication.sources["sensor_csv"]
    if audit.source_sha256 != sensor["sha256"]:
        raise IntegrityError("timestamp audit source hash differs from authenticated CSV")
    if audit.source_size_bytes != sensor["size_bytes"]:
        raise IntegrityError("timestamp audit source size differs from authenticated CSV")
    if audit.row_count != sum(partition.row_count for partition in PARTITIONS):
        raise IntegrityError("timestamp audit row count differs from the charter")
    expected_partition_counts = {
        partition.name: partition.row_count for partition in PARTITIONS
    }
    if dict(audit.partition_row_counts) != expected_partition_counts:
        raise IntegrityError("timestamp audit partition counts differ from the charter")
    if np.datetime64(audit.first_timestamp) != np.datetime64(PARTITIONS[0].start):
        raise IntegrityError("timestamp audit first timestamp differs from the charter")
    if np.datetime64(audit.last_timestamp) != np.datetime64("2020-09-01T03:59:50"):
        raise IntegrityError("timestamp audit last timestamp differs from the charter")
    if audit.modal_interval_seconds != 10.0:
        raise IntegrityError("timestamp audit modal cadence differs from the charter")
    document = audit.as_dict()
    if (
        document.get("sensor_numeric_values_parsed") is not False
        or document.get("numeric_columns_parsed") != []
    ):
        raise IntegrityError("timestamp audit parsed a forbidden sensor value")


def _validate_loaded_interval(
    name: str,
    interval: MetroPTMinuteInterval,
    authentication: DevelopmentInputAuthentication,
) -> None:
    partition = _PARTITION_BY_NAME[name]
    metadata = interval.access_metadata
    if metadata.get("requested_source_row_count") != partition.row_count:
        raise IntegrityError(f"{name} source-row count differs from the charter")
    if metadata.get("source_sha256") != authentication.sources["sensor_csv"]["sha256"]:
        raise IntegrityError(f"{name} source hash differs from authentication")
    if metadata.get("unrequested_numeric_values_parsed") is not False:
        raise AccessDeniedError(f"{name} loader parsed numeric values outside its interval")
    if metadata.get("direct_alarm_values_parsed") is not False:
        raise AccessDeniedError(f"{name} loader parsed direct alarm values")
    if metadata.get("numeric_columns_parsed") != list(APPROVED_NUMERIC_COLUMNS):
        raise IntegrityError(f"{name} loader numeric column order changed")
    if tuple(interval.feature_names) != APPROVED_NUMERIC_COLUMNS:
        raise IntegrityError(f"{name} returned an altered feature order")
    expected_start = np.datetime64(partition.start, "ns")
    expected_stop = np.datetime64(partition.end, "ns")
    if interval.timestamps[0] != expected_start:
        raise IntegrityError(f"{name} minute grid does not start at the partition boundary")
    if interval.timestamps[-1] != expected_stop - np.timedelta64(1, "m"):
        raise IntegrityError(f"{name} minute grid does not end at the right-open boundary")
    if interval.analog.shape != (interval.timestamps.size, len(ANALOG_COLUMNS)):
        raise IntegrityError(f"{name} analog matrix shape changed")
    if interval.controls.shape != (
        interval.timestamps.size,
        len(SAFE_CONTROL_COLUMNS),
    ):
        raise IntegrityError(f"{name} control matrix shape changed")
    reset_before = np.asarray(interval.reset_before)
    if (
        reset_before.shape != (interval.timestamps.size,)
        or reset_before.dtype.kind != "b"
        or not bool(reset_before[0])
    ):
        raise IntegrityError(f"{name} raw-gap reset mask is invalid")


def _load_development_partitions(
    config: DevelopmentRunConfig,
    hooks: DevelopmentHooks,
    audit: MetroPTTimestampAudit,
    authentication: DevelopmentInputAuthentication,
    claim: LedgerClaim,
    access_intent: AuthenticatedDocument,
    phase_manifest_fingerprint: str,
) -> dict[str, MetroPTMinuteInterval]:
    loaded: dict[str, MetroPTMinuteInterval] = {}
    for name in PHASE_PARTITIONS[PHASE_DEVELOPMENT]:
        interval = hooks.load_partition(
            config.csv_path,
            name,
            claim=claim,
            access_intent=access_intent,
            access_intent_path=config.output_directory / ACCESS_INTENT_NAME,
            phase_manifest_fingerprint=phase_manifest_fingerprint,
            chunksize=config.csv_chunksize,
            timestamp_audit=audit,
        )
        _validate_loaded_interval(name, interval, authentication)
        loaded[name] = interval
    if tuple(loaded) != PHASE_PARTITIONS[PHASE_DEVELOPMENT]:
        raise AccessDeniedError("development loader crossed a sealed partition boundary")
    return loaded


def _predict_and_align(
    *,
    name: str,
    interval: MetroPTMinuteInterval,
    windows: InferenceWindowBatch,
    ensemble: Any,
    hooks: DevelopmentHooks,
    config: DevelopmentRunConfig,
) -> tuple[np.ndarray, AlignedResidualBatch, dict[str, Any]]:
    started = time.perf_counter()
    prediction = ensemble.predict(
        windows.inputs,
        device=config.device,
        batch_size=config.batch_size,
    )
    elapsed = time.perf_counter() - started
    aligned = hooks.align_inference_forecasts(
        windows,
        prediction,
        interval.analog,
        interval.controls,
        interval.timestamps,
        eligible=interval.eligible,
        reset_before=interval.reset_before,
    )
    return prediction, aligned, {
        "partition": name,
        "window_count": int(windows.inputs.shape[0]),
        "aligned_score_count": int(aligned.residuals.shape[0]),
        "prediction_seconds": float(elapsed),
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
    }


def _complete_day_threshold(
    score_path: CaredScorePath,
    score_timestamps: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    timestamps = np.asarray(score_timestamps, dtype="datetime64[ns]")
    envelope = np.asarray(score_path.envelope, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or timestamps.shape != envelope.shape
        or np.isnat(timestamps).any()
        or not np.isfinite(envelope).all()
        or (
            timestamps.size > 1
            and (np.diff(timestamps.astype(np.int64)) <= 0).any()
        )
    ):
        raise DevelopmentRunnerError("dev_healthy CUSUM path is invalid")
    partition = _PARTITION_BY_NAME["dev_healthy"]
    expected_dates = np.arange(
        np.datetime64(partition.start, "D"),
        np.datetime64(partition.end, "D"),
        dtype="datetime64[D]",
    )
    if expected_dates.size != REQUIRED_HEALTHY_DAYS:
        raise DevelopmentRunnerError("frozen dev_healthy interval is not nine days")
    records: list[dict[str, Any]] = []
    maxima: list[float] = []
    dates = timestamps.astype("datetime64[D]")
    if timestamps.size and not np.isin(dates, expected_dates).all():
        raise DevelopmentRunnerError("dev_healthy score falls outside its frozen dates")
    for date in expected_dates:
        selected = envelope[dates == date]
        score_count = int(selected.size)
        qualified = score_count >= MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
        maximum = None if score_count == 0 else float(np.max(selected))
        if qualified:
            assert maximum is not None
            maxima.append(maximum)
            reason: str | None = None
        elif score_count == 0:
            reason = "zero_scores"
        else:
            reason = "score_count_below_minimum"
        records.append(
            {
                "date": str(date),
                "score_count": score_count,
                "valid_score_count": score_count,
                "minimum_required_score_count": (
                    MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
                ),
                "qualified": qualified,
                "included_in_provisional_maximum": qualified,
                "unqualified_reason": reason,
                "cusum_maximum": maximum,
            }
        )
    if len(maxima) < MINIMUM_QUALIFIED_HEALTHY_DAYS:
        raise InsufficientQualifiedDaysError(records)
    threshold = float(max(maxima))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise DevelopmentRunnerError("provisional development threshold must be positive")
    return threshold, records


def _maximum_absolute_error(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or not (
        np.isfinite(first).all() and np.isfinite(second).all()
    ):
        raise DevelopmentRunnerError("CPU/CUDA parity arrays are incompatible")
    return float(np.max(np.abs(first - second))) if first.size else 0.0


def check_cpu_cuda_sample_parity(
    *,
    ensemble: Any,
    thresholded_scorer: Any,
    intervals: Mapping[str, MetroPTMinuteInterval],
    windows: Mapping[str, InferenceWindowBatch],
    cuda_predictions: Mapping[str, np.ndarray],
    batch_size: int,
    atol: float = PARITY_ATOL,
) -> dict[str, Any]:
    """Compare sampled CPU and CUDA predictions through exact alarm decisions."""

    split_records: dict[str, Any] = {}
    all_passed = True
    for name in ("dev_healthy", "dev_fault_f1"):
        batch = windows[name]
        count = min(PARITY_SAMPLE_WINDOWS, int(batch.inputs.shape[0]))
        if count < 16:
            raise DevelopmentRunnerError(f"{name} has too few windows for parity")
        selected = np.arange(count, dtype=np.int64)
        subset = batch.subset(selected)
        cpu_prediction = ensemble.predict(
            subset.inputs, device="cpu", batch_size=batch_size
        )
        cuda_prediction = np.asarray(cuda_predictions[name])[selected]
        interval = intervals[name]
        cpu_aligned = align_inference_forecasts_to_score_times(
            subset,
            cpu_prediction,
            interval.analog,
            interval.controls,
            interval.timestamps,
            eligible=interval.eligible,
            reset_before=interval.reset_before,
        )
        cuda_aligned = align_inference_forecasts_to_score_times(
            subset,
            cuda_prediction,
            interval.analog,
            interval.controls,
            interval.timestamps,
            eligible=interval.eligible,
            reset_before=interval.reset_before,
        )
        if not np.array_equal(cpu_aligned.score_timestamps, cuda_aligned.score_timestamps):
            raise DevelopmentRunnerError("CPU/CUDA aligned timestamps differ")
        cpu_path = thresholded_scorer.score(
            cpu_aligned.residuals,
            cpu_aligned.state_codes,
            cpu_aligned.score_timestamps,
            reset_before=cpu_aligned.reset_before,
        )
        cuda_path = thresholded_scorer.score(
            cuda_aligned.residuals,
            cuda_aligned.state_codes,
            cuda_aligned.score_timestamps,
            reset_before=cuda_aligned.reset_before,
        )
        errors = {
            "prediction_max_abs_error": _maximum_absolute_error(
                cpu_prediction, cuda_prediction
            ),
            "residual_max_abs_error": _maximum_absolute_error(
                cpu_aligned.residuals, cuda_aligned.residuals
            ),
            "standardized_evidence_max_abs_error": _maximum_absolute_error(
                cpu_path.standardized_evidence, cuda_path.standardized_evidence
            ),
            "cusum_max_abs_error": _maximum_absolute_error(
                cpu_path.cusum, cuda_path.cusum
            ),
            "envelope_max_abs_error": _maximum_absolute_error(
                cpu_path.envelope, cuda_path.envelope
            ),
        }
        alarm_exact = bool(np.array_equal(cpu_path.alarm, cuda_path.alarm))
        passed = alarm_exact and all(value <= atol for value in errors.values())
        split_records[name] = {
            "sample_window_count": count,
            "atol": float(atol),
            **errors,
            "alarm_decisions_exact": alarm_exact,
            "passed": passed,
        }
        all_passed &= passed
    return {
        "schema": "metropt_cared_bc_cpu_cuda_sample_parity",
        "schema_version": RUNNER_SCHEMA_VERSION,
        "device_pair": ["cpu", "cuda"],
        "dtype": "float64",
        "atol": float(atol),
        "splits": split_records,
        "alarm_decisions_exact": all(
            bool(record["alarm_decisions_exact"])
            for record in split_records.values()
        ),
        "passed": bool(all_passed),
    }


def check_development_causality(
    *,
    intervals: Mapping[str, MetroPTMinuteInterval],
    windows: Mapping[str, InferenceWindowBatch],
    aligned: Mapping[str, AlignedResidualBatch],
) -> bool:
    """Recheck timestamp directionality at the runner/model boundary."""

    expected_horizons = np.asarray([1, 3, 5], dtype="timedelta64[m]")
    for name in PHASE_PARTITIONS[PHASE_DEVELOPMENT]:
        interval = intervals[name]
        batch = windows[name]
        residual = aligned[name]
        anchor_indices = np.asarray(batch.anchor_indices, dtype=np.int64)
        if (
            anchor_indices.ndim != 1
            or (anchor_indices < 0).any()
            or (anchor_indices >= interval.timestamps.size).any()
            or not np.array_equal(
                interval.timestamps[anchor_indices], batch.anchor_timestamps
            )
        ):
            return False
        source_rows = np.asarray(residual.source_window_rows, dtype=np.int64)
        if (
            source_rows.shape != (residual.score_timestamps.size, 3)
            or (source_rows < 0).any()
            or (source_rows >= batch.anchor_timestamps.size).any()
        ):
            return False
        deltas = (
            np.asarray(residual.score_timestamps, dtype="datetime64[ns]")[:, None]
            - np.asarray(batch.anchor_timestamps, dtype="datetime64[ns]")[
                source_rows
            ]
        ).astype("timedelta64[m]")
        if not np.array_equal(
            deltas,
            np.broadcast_to(expected_horizons, deltas.shape),
        ):
            return False
        score_indices = np.asarray(residual.score_indices, dtype=np.int64)
        if (
            score_indices.ndim != 1
            or (score_indices < 0).any()
            or (score_indices >= interval.timestamps.size).any()
            or not np.array_equal(
                interval.timestamps[score_indices], residual.score_timestamps
            )
        ):
            return False
        projected_reset = np.asarray(residual.reset_before)
        if (
            projected_reset.shape != score_indices.shape
            or projected_reset.dtype.kind != "b"
            or not bool(projected_reset[0])
        ):
            return False
        partition = _PARTITION_BY_NAME[name]
        score_ts = np.asarray(residual.score_timestamps, dtype="datetime64[ns]")
        if np.any(score_ts < np.datetime64(partition.start, "ns")) or np.any(
            score_ts >= np.datetime64(partition.end, "ns")
        ):
            return False
    return True


def _training_summary(training: Any) -> dict[str, Any]:
    histories: list[dict[str, Any]] = []
    for history in training.histories:
        histories.append(
            {
                "seed": int(history.seed),
                "training_loss": [float(value) for value in history.training_loss],
                "validation_loss": [float(value) for value in history.validation_loss],
                "best_epoch_one_based": int(history.best_epoch_one_based),
                "best_validation_loss": float(history.best_validation_loss),
                "stopped_early": bool(history.stopped_early),
            }
        )
    return {
        "ensemble_validation_huber_loss": float(
            training.ensemble_validation_huber_loss
        ),
        "persistence_validation_huber_loss": float(
            training.persistence_validation_huber_loss
        ),
        "relative_improvement_over_persistence": float(
            training.relative_improvement_over_persistence
        ),
        "members": histories,
    }


def evaluate_development_power_gate(
    *,
    training: Any,
    threshold: float,
    healthy_path: Any,
    f1_path: Any,
    f1_timestamps: np.ndarray,
    direct_alarm_excluded: bool,
    causality_passed: bool,
    parity: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Evaluate only the seven frozen development power requirements."""

    improvement = float(training.relative_improvement_over_persistence)
    f1_alarm = np.asarray(f1_path.alarm, dtype=bool)
    f1_envelope = np.asarray(f1_path.envelope, dtype=np.float64)
    timestamps = np.asarray(f1_timestamps, dtype="datetime64[ns]")
    if f1_alarm.shape != timestamps.shape or f1_envelope.shape != timestamps.shape:
        raise DevelopmentRunnerError("F1 path arrays are inconsistent")
    alarm_indices = np.flatnonzero(f1_alarm)
    first_alarm = None
    first_alarm_availability = None
    delay_minutes: float | str = "+infinity"
    if alarm_indices.size:
        first_alarm_timestamp = timestamps[int(alarm_indices[0])]
        first_alarm = str(first_alarm_timestamp)
        availability_timestamp = first_alarm_timestamp + np.timedelta64(1, "m")
        first_alarm_availability = str(availability_timestamp)
        event_start = np.datetime64(_PARTITION_BY_NAME["dev_fault_f1"].start, "ns")
        delay_minutes = float(
            (availability_timestamp - event_start) / np.timedelta64(1, "m")
        )
    maximum = float(np.max(f1_envelope))
    ratio = float(maximum / threshold)
    healthy_alarm_count = int(np.count_nonzero(healthy_path.alarm))

    gates = {
        "tcn_relative_improvement": {
            "observed": improvement,
            "requirement": ">= 0.05",
            "passed": improvement >= 0.05,
        },
        "f1_alarm": {
            "observed": bool(alarm_indices.size),
            "requirement": "true",
            "passed": bool(alarm_indices.size),
        },
        "f1_first_alarm_delay_minutes": {
            "observed": delay_minutes,
            "requirement": "<= 60",
            "passed": isinstance(delay_minutes, float)
            and 0.0 <= delay_minutes <= 60.0,
        },
        "f1_maximum_to_threshold_ratio": {
            "observed": ratio,
            "requirement": ">= 1.25",
            "passed": ratio >= 1.25,
        },
        "direct_alarm_channels_excluded": {
            "observed": bool(direct_alarm_excluded),
            "requirement": "true",
            "passed": bool(direct_alarm_excluded),
        },
        "feature_and_score_causality": {
            "observed": bool(causality_passed),
            "requirement": "true",
            "passed": bool(causality_passed),
        },
        "cpu_cuda_sample_parity": {
            "observed": bool(parity.get("passed")),
            "alarm_decisions_exact": bool(parity.get("alarm_decisions_exact")),
            "requirement": "passed with exact alarm decisions",
            "passed": bool(parity.get("passed"))
            and bool(parity.get("alarm_decisions_exact")),
        },
    }
    passed = all(bool(record["passed"]) for record in gates.values())
    decision = DEVELOPMENT_PASS if passed else DEVELOPMENT_ABORT
    f1_metrics = {
        "alarm_count": int(alarm_indices.size),
        "first_alarm_score_minute": first_alarm,
        "first_alarm_availability_timestamp": first_alarm_availability,
        "first_alarm_delay_minutes": delay_minutes,
        "maximum_cusum": maximum,
        "provisional_threshold": float(threshold),
        "maximum_to_threshold_ratio": ratio,
    }
    return decision, gates, f1_metrics


def _direct_alarm_exclusion_is_proven(
    intervals: Mapping[str, MetroPTMinuteInterval],
) -> bool:
    names_match = (
        tuple(ANALOG_NAMES) == tuple(ANALOG_COLUMNS)
        and tuple(SAFE_CONTROL_NAMES) == tuple(SAFE_CONTROL_COLUMNS)
    )
    excluded = set(DIRECT_ALARM_COLUMNS).isdisjoint(
        set(ANALOG_COLUMNS).union(SAFE_CONTROL_COLUMNS)
    )
    metadata_safe = all(
        interval.access_metadata.get("direct_alarm_values_parsed") is False
        and interval.access_metadata.get("excluded_direct_alarm_columns")
        == list(DIRECT_ALARM_COLUMNS)
        for interval in intervals.values()
    )
    return bool(names_match and excluded and metadata_safe)


def _normal_artifact_manifest(
    output: Path,
    *,
    claim: LedgerClaim,
    intent: AuthenticatedDocument,
    phase_manifest_fingerprint: str,
) -> AuthenticatedDocument:
    del claim
    return create_artifact_manifest(
        output / ARTIFACT_MANIFEST_NAME,
        output,
        phase=PHASE_DEVELOPMENT,
        protocol_id=CHARTER_ID,
        protocol_sha256=DEVELOPMENT_CHARTER_SHA256,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        access_intent=intent,
        artifact_names=DEVELOPMENT_ARTIFACT_NAMES,
    )


def _create_completion(
    output: Path,
    *,
    claim: LedgerClaim,
    intent: AuthenticatedDocument,
    artifact_manifest: AuthenticatedDocument,
    decision: str,
) -> AuthenticatedDocument:
    return create_completion_record(
        output / COMPLETION_NAME,
        claim=claim,
        access_intent=intent,
        artifact_manifest=artifact_manifest,
        decision=decision,
    )


def _finalize_runtime_abort(
    output: Path,
    *,
    claim: LedgerClaim,
    intent: AuthenticatedDocument,
    phase_manifest_fingerprint: str,
    error: Exception,
) -> AuthenticatedDocument:
    abort_body: dict[str, Any] = {
        "schema": "metropt_cared_bc_development_runtime_abort",
        "schema_version": RUNNER_SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "decision": DEVELOPMENT_ABORT,
        "phase_manifest_fingerprint": phase_manifest_fingerprint,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "calibration_numeric_access_authorized": False,
        "confirmation_numeric_access_authorized": False,
    }
    if isinstance(error, InsufficientQualifiedDaysError):
        abort_body["development_day_qualification"] = {
            "candidate_day_count": REQUIRED_HEALTHY_DAYS,
            "minimum_qualified_day_count": MINIMUM_QUALIFIED_HEALTHY_DAYS,
            "qualified_day_count": error.qualified_day_count,
            "days": [dict(record) for record in error.day_records],
        }
    atomic_create_new_json(
        output / ABORT_NAME,
        abort_body,
    )
    candidates = [
        *DEVELOPMENT_ARTIFACT_NAMES,
        ABORT_NAME,
    ]
    artifact_names = tuple(
        name
        for name in candidates
        if (output / name).is_file() and not (output / name).is_symlink()
    )
    manifest = create_artifact_manifest(
        output / ARTIFACT_MANIFEST_NAME,
        output,
        phase=PHASE_DEVELOPMENT,
        protocol_id=CHARTER_ID,
        protocol_sha256=DEVELOPMENT_CHARTER_SHA256,
        phase_manifest_fingerprint=phase_manifest_fingerprint,
        access_intent=intent,
        artifact_names=artifact_names,
    )
    return _create_completion(
        output,
        claim=claim,
        intent=intent,
        artifact_manifest=manifest,
        decision=DEVELOPMENT_ABORT,
    )


def _artifact_names_from_manifest(path: Path) -> tuple[str, ...]:
    document = read_strict_json(path)
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise IntegrityError("saved artifact manifest has no artifact mapping")
    names = tuple(sorted(str(name) for name in artifacts))
    normal = set(DEVELOPMENT_ARTIFACT_NAMES)
    observed = set(names)
    if observed != normal:
        if ABORT_NAME not in observed or not observed.issubset(normal | {ABORT_NAME}):
            raise IntegrityError("saved development artifact set is not permitted")
        if PHASE_MANIFEST_NAME not in observed:
            raise IntegrityError("runtime abort is not bound to its phase manifest")
    return names


def _validate_saved_phase_manifest(
    config: DevelopmentRunConfig,
    document: AuthenticatedDocument,
) -> None:
    value = document.value
    expected_identity = {
        "schema": "metropt_cared_bc_development_phase_manifest",
        "schema_version": RUNNER_SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": CHARTER_ID,
        "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
        "sensor_csv_relative_path": _SENSOR_SPEC.relative_path,
        "approved_input_columns": list(APPROVED_NUMERIC_COLUMNS),
        "direct_alarm_columns_excluded": list(DIRECT_ALARM_COLUMNS),
        "future_numeric_access_authorized": False,
        "recovery_run": True,
        "old_failed_bundle_must_remain_unchanged": True,
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise IntegrityError(f"completed phase manifest identity changed: {key}")
    amendment = value.get("preaccess_amendment")
    if (
        not isinstance(amendment, Mapping)
        or amendment.get("amendment_id") != AMENDMENT_ID
        or amendment.get("sha256") != AMENDMENT_SHA256
        or not isinstance(amendment.get("record"), Mapping)
        or amendment["record"].get("sha256") != AMENDMENT_SHA256
    ):
        raise IntegrityError("completed phase manifest amendment binding changed")
    parent_amendment = value.get("parent_preaccess_amendment")
    if (
        not isinstance(parent_amendment, Mapping)
        or parent_amendment.get("amendment_id") != PARENT_AMENDMENT_ID
        or parent_amendment.get("sha256") != PARENT_AMENDMENT_SHA256
        or not isinstance(parent_amendment.get("record"), Mapping)
        or parent_amendment["record"].get("sha256") != PARENT_AMENDMENT_SHA256
    ):
        raise IntegrityError("completed phase manifest parent amendment binding changed")
    recovery_bindings = value.get("recovery_bindings")
    if not isinstance(recovery_bindings, Mapping):
        raise IntegrityError("completed phase manifest has no recovery bindings")
    validate_recovery_bindings(recovery_bindings)
    threshold = value.get("development_threshold")
    if (
        not isinstance(threshold, Mapping)
        or threshold.get("rule")
        != "maximum_qualified_dev_healthy_day_cusum_maximum"
        or threshold.get("candidate_days") != REQUIRED_HEALTHY_DAYS
        or threshold.get("minimum_qualified_days")
        != MINIMUM_QUALIFIED_HEALTHY_DAYS
        or threshold.get("minimum_valid_scores_per_day")
        != MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
        or threshold.get("unqualified_days_excluded_from_maximum") is not True
        or threshold.get("zero_score_days_explicitly_reported") is not True
    ):
        raise IntegrityError("completed phase manifest threshold rule changed")
    runtime = value.get("runtime")
    expected_runtime = {
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "csv_chunksize": config.csv_chunksize,
    }
    if runtime != expected_runtime:
        raise IntegrityError("completed phase manifest runtime differs from this command")
    expected_output = _workspace_relative(
        _absolute_without_access(config.output_directory, config.workspace_root),
        config.workspace_root,
    )
    if value.get("output_directory_relative_path") != expected_output:
        raise IntegrityError("completed phase manifest output directory changed")
    expected_csv = (
        config.workspace_root.resolve(strict=True)
        / PurePosixPath(_SENSOR_SPEC.relative_path)
    )
    supplied_csv = _absolute_without_access(config.csv_path, config.workspace_root)
    # Lexical equality intentionally avoids opening or hashing the CSV on a
    # completed re-entry.
    if supplied_csv != expected_csv:
        raise IntegrityError("completed invocation uses a different CSV path")


def authenticate_completed_development(
    config: DevelopmentRunConfig,
) -> DevelopmentRunOutcome:
    """Verify and reuse a completed bundle without touching the source CSV."""

    output = _absolute_without_access(config.output_directory, config.workspace_root)
    phase_document = authenticate_fingerprinted_json(
        output / PHASE_MANIFEST_NAME,
        fingerprint_field="manifest_fingerprint",
    )
    _validate_saved_phase_manifest(config, phase_document)
    if phase_document.fingerprint is None:
        raise IntegrityError("completed phase manifest has no fingerprint")
    ledger_arguments = _ledger_arguments(
        config, phase_document.value, phase_document.fingerprint
    )
    expected_ledger_body = build_workspace_access_ledger(**ledger_arguments)
    artifact_names = _artifact_names_from_manifest(
        output / ARTIFACT_MANIFEST_NAME
    )
    completion = authenticate_completion_record(
        output / COMPLETION_NAME,
        output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=expected_ledger_body,
        access_intent_path=output / ACCESS_INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=artifact_names,
    )
    decision = str(completion.value["decision"])
    if RESULT_NAME in artifact_names:
        result = read_strict_json(output / RESULT_NAME)
        if (
            result.get("phase") != PHASE_DEVELOPMENT
            or result.get("decision") != decision
            or result.get("phase_manifest_fingerprint")
            != phase_document.fingerprint
            or result.get("parent_preaccess_amendment_sha256")
            != PARENT_AMENDMENT_SHA256
            or result.get("preaccess_amendment_sha256") != AMENDMENT_SHA256
        ):
            raise IntegrityError("saved development result differs from completion")
        threshold = result.get("provisional_threshold")
        days = threshold.get("days") if isinstance(threshold, Mapping) else None
        if not isinstance(days, list) or len(days) != REQUIRED_HEALTHY_DAYS:
            raise IntegrityError("saved development result day records changed")
        qualified = [
            record
            for record in days
            if isinstance(record, Mapping) and record.get("qualified") is True
        ]
        if (
            threshold.get("candidate_day_count") != REQUIRED_HEALTHY_DAYS
            or threshold.get("minimum_qualified_day_count")
            != MINIMUM_QUALIFIED_HEALTHY_DAYS
            or threshold.get("qualified_day_count") != len(qualified)
            or len(qualified) < MINIMUM_QUALIFIED_HEALTHY_DAYS
            or any(
                record.get("score_count", -1)
                < MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
                or record.get("included_in_provisional_maximum") is not True
                or not isinstance(record.get("cusum_maximum"), (int, float))
                for record in qualified
            )
            or float(threshold.get("value"))
            != max(float(record["cusum_maximum"]) for record in qualified)
        ):
            raise IntegrityError("saved development result threshold evidence changed")
    else:
        result = read_strict_json(output / ABORT_NAME)
        if result.get("decision") != DEVELOPMENT_ABORT:
            raise IntegrityError("saved runtime abort has an invalid decision")
    return DevelopmentRunOutcome(
        decision=decision,
        completion=completion,
        result=result,
        reused_completed_bundle=True,
    )


def _ensure_fresh_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise IntegrityError("output path must be a real directory")
        if any(output.iterdir()):
            raise DevelopmentRunnerError(
                "incomplete output directory is not reusable; choose a fresh output directory"
            )
        return
    output.mkdir(parents=True, exist_ok=False)


def run_development(
    config: DevelopmentRunConfig,
    *,
    hooks: DevelopmentHooks | None = None,
) -> DevelopmentRunOutcome:
    """Run or authenticate one governed development experiment."""

    _validate_development_config(config)
    workspace = config.workspace_root.resolve(strict=True)
    output = _absolute_without_access(config.output_directory, workspace)
    object.__setattr__(config, "workspace_root", workspace)
    object.__setattr__(config, "output_directory", output)
    object.__setattr__(
        config, "csv_path", _absolute_without_access(config.csv_path, workspace)
    )
    object.__setattr__(
        config, "charter_path", _absolute_without_access(config.charter_path, workspace)
    )
    object.__setattr__(
        config,
        "charter_sidecar_path",
        _absolute_without_access(config.charter_sidecar_path, workspace),
    )
    object.__setattr__(
        config,
        "parent_amendment_path",
        _absolute_without_access(config.parent_amendment_path, workspace),
    )
    object.__setattr__(
        config,
        "parent_amendment_sidecar_path",
        _absolute_without_access(config.parent_amendment_sidecar_path, workspace),
    )
    object.__setattr__(
        config,
        "amendment_path",
        _absolute_without_access(config.amendment_path, workspace),
    )
    object.__setattr__(
        config,
        "amendment_sidecar_path",
        _absolute_without_access(config.amendment_sidecar_path, workspace),
    )

    output_relative = _workspace_relative(output, workspace)
    if (
        output_relative == FAILED_DEVELOPMENT_OUTPUT
        or output_relative.startswith(f"{FAILED_DEVELOPMENT_OUTPUT}/")
    ):
        raise AccessDeniedError("recovery run cannot write into the old failed bundle")

    if (output / COMPLETION_NAME).is_file():
        return authenticate_completed_development(config)
    if output.exists() and any(output.iterdir()):
        raise DevelopmentRunnerError(
            "partial development bundle exists without a completion record; "
            "numeric access will not be repeated"
        )

    active_hooks = hooks or default_development_hooks()
    authentication = active_hooks.authenticate_inputs(
        config.charter_path,
        config.charter_sidecar_path,
        workspace,
    )
    parent_amendment = active_hooks.authenticate_amendment(
        config.parent_amendment_path,
        config.parent_amendment_sidecar_path,
        expected_sha256=PARENT_AMENDMENT_SHA256,
    )
    validate_preaccess_amendment(parent_amendment)
    amendment = active_hooks.authenticate_amendment(
        config.amendment_path,
        config.amendment_sidecar_path,
        expected_sha256=AMENDMENT_SHA256,
    )
    validate_recovery_amendment(amendment, parent_amendment)
    recovery_bindings = dict(
        active_hooks.authenticate_recovery_bindings(workspace, amendment)
    )
    validate_recovery_bindings(recovery_bindings)
    phase_body = build_development_phase_manifest(
        config,
        authentication,
        parent_amendment,
        amendment,
        recovery_bindings,
    )
    phase_fingerprint = canonical_fingerprint(phase_body)

    _ensure_fresh_output(output)
    phase_document = create_fingerprinted_json(
        output / PHASE_MANIFEST_NAME,
        phase_body,
        fingerprint_field="manifest_fingerprint",
    )
    if phase_document.fingerprint != phase_fingerprint:
        raise IntegrityError("created phase manifest fingerprint changed")

    ledger_arguments = _ledger_arguments(config, phase_body, phase_fingerprint)
    claim = claim_workspace_access_ledger(
        output / LEDGER_NAME,
        **ledger_arguments,
    )
    intent = create_phase_access_intent(
        output / ACCESS_INTENT_NAME,
        claim,
        phase_manifest_fingerprint=phase_fingerprint,
    )

    try:
        # This is the first timestamp/content interpretation after the durable
        # access intent.  It still parses timestamps only across the full file.
        audit = active_hooks.timestamp_audit(
            config.csv_path, chunksize=config.csv_chunksize
        )
        _validate_timestamp_audit(audit, authentication)
        atomic_create_new_json(output / TIMESTAMP_AUDIT_NAME, audit.as_dict())

        intervals = _load_development_partitions(
            config,
            active_hooks,
            audit,
            authentication,
            claim,
            intent,
            phase_fingerprint,
        )
        training_windows = {
            name: active_hooks.build_training_windows(
                intervals[name].analog,
                intervals[name].controls,
                intervals[name].timestamps,
                eligible=intervals[name].eligible,
                reset_before=intervals[name].reset_before,
            )
            for name in ("fit", "dev_healthy")
        }
        inference_windows = {
            name: active_hooks.build_inference_windows(
                interval.analog,
                interval.controls,
                interval.timestamps,
                eligible=interval.eligible,
                reset_before=interval.reset_before,
            )
            for name, interval in intervals.items()
        }
        runtime = dict(active_hooks.runtime_metadata(config))
        training_started = time.perf_counter()
        training = active_hooks.train_ensemble(
            training_windows["fit"],
            training_windows["dev_healthy"],
            device=config.device,
            dtype=config.dtype,
            batch_size=config.batch_size,
        )
        training_seconds = time.perf_counter() - training_started
        ensemble = training.ensemble

        predictions: dict[str, np.ndarray] = {}
        aligned: dict[str, AlignedResidualBatch] = {}
        prediction_logs: list[dict[str, Any]] = []
        for name in PHASE_PARTITIONS[PHASE_DEVELOPMENT]:
            prediction, residual, execution = _predict_and_align(
                name=name,
                interval=intervals[name],
                windows=inference_windows[name],
                ensemble=ensemble,
                hooks=active_hooks,
                config=config,
            )
            predictions[name] = prediction
            aligned[name] = residual
            prediction_logs.append(execution)

        scorer = active_hooks.fit_scorer(
            aligned["fit"].residuals,
            aligned["fit"].state_codes,
            aligned["dev_healthy"].residuals,
            aligned["dev_healthy"].state_codes,
            aligned["dev_fault_f1"].residuals,
            aligned["dev_fault_f1"].state_codes,
        )
        if getattr(scorer, "threshold", None) is not None:
            raise DevelopmentRunnerError(
                "development scorer must remain uncalibrated and threshold-free"
            )
        healthy_unthresholded = scorer.score(
            aligned["dev_healthy"].residuals,
            aligned["dev_healthy"].state_codes,
            aligned["dev_healthy"].score_timestamps,
            reset_before=aligned["dev_healthy"].reset_before,
        )
        provisional_threshold, daily_records = _complete_day_threshold(
            healthy_unthresholded,
            aligned["dev_healthy"].score_timestamps,
        )
        thresholded_scorer = scorer.with_threshold(provisional_threshold)
        healthy_path = thresholded_scorer.score(
            aligned["dev_healthy"].residuals,
            aligned["dev_healthy"].state_codes,
            aligned["dev_healthy"].score_timestamps,
            reset_before=aligned["dev_healthy"].reset_before,
        )
        f1_path = thresholded_scorer.score(
            aligned["dev_fault_f1"].residuals,
            aligned["dev_fault_f1"].state_codes,
            aligned["dev_fault_f1"].score_timestamps,
            reset_before=aligned["dev_fault_f1"].reset_before,
        )

        parity = dict(
            active_hooks.parity_check(
                ensemble=ensemble,
                thresholded_scorer=thresholded_scorer,
                intervals=intervals,
                windows=inference_windows,
                cuda_predictions=predictions,
                batch_size=config.batch_size,
                atol=PARITY_ATOL,
            )
        )
        direct_alarm_excluded = _direct_alarm_exclusion_is_proven(intervals)
        causality_passed = bool(
            active_hooks.causality_check(
                intervals=intervals,
                windows=inference_windows,
                aligned=aligned,
            )
        )
        if int(np.count_nonzero(healthy_path.alarm)) != 0:
            raise DevelopmentRunnerError(
                "strict provisional threshold unexpectedly alarms on dev_healthy"
            )
        decision, gates, f1_metrics = evaluate_development_power_gate(
            training=training,
            threshold=provisional_threshold,
            healthy_path=healthy_path,
            f1_path=f1_path,
            f1_timestamps=aligned["dev_fault_f1"].score_timestamps,
            direct_alarm_excluded=direct_alarm_excluded,
            causality_passed=causality_passed,
            parity=parity,
        )

        model_metadata = {
            "protocol_id": CHARTER_ID,
            "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
            "parent_preaccess_amendment_sha256": PARENT_AMENDMENT_SHA256,
            "preaccess_amendment_sha256": AMENDMENT_SHA256,
            "phase": PHASE_DEVELOPMENT,
            "phase_manifest_fingerprint": phase_fingerprint,
            "source_fingerprint": authentication.source_fingerprint,
            "development_only": True,
        }
        ensemble.save_npz(output / ENSEMBLE_NAME, metadata=model_metadata)
        scorer.save_npz(
            output / SCORER_NAME,
            metadata={
                **model_metadata,
                "threshold_status": "uncalibrated",
                "provisional_threshold_saved_in_result_only": True,
            },
        )

        execution_document = {
            "schema": "metropt_cared_bc_gpu_execution_development",
            "schema_version": RUNNER_SCHEMA_VERSION,
            "phase": PHASE_DEVELOPMENT,
            "phase_manifest_fingerprint": phase_fingerprint,
            "runtime": runtime,
            "training_seconds": float(training_seconds),
            "predictions": prediction_logs,
            "cpu_cuda_sample_parity": parity,
        }
        atomic_create_new_json(output / GPU_LOG_NAME, execution_document)

        result = {
            "schema": "metropt_cared_bc_development_result",
            "schema_version": RUNNER_SCHEMA_VERSION,
            "protocol_id": CHARTER_ID,
            "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
            "parent_preaccess_amendment_sha256": PARENT_AMENDMENT_SHA256,
            "preaccess_amendment_sha256": AMENDMENT_SHA256,
            "phase": PHASE_DEVELOPMENT,
            "phase_manifest_fingerprint": phase_fingerprint,
            "source_fingerprint": authentication.source_fingerprint,
            "decision": decision,
            "development_gate_passed": decision == DEVELOPMENT_PASS,
            "calibration_numeric_access_authorized_by_this_result": False,
            "confirmation_numeric_access_authorized_by_this_result": False,
            "partition_access": {
                name: {
                    "start": _PARTITION_BY_NAME[name].start,
                    "end": _PARTITION_BY_NAME[name].end,
                    "source_row_count": interval.access_metadata[
                        "requested_source_row_count"
                    ],
                    "minute_count": int(interval.timestamps.size),
                    "eligible_minute_count": int(np.count_nonzero(interval.eligible)),
                    "training_window_count": (
                        int(training_windows[name].inputs.shape[0])
                        if name in training_windows
                        else None
                    ),
                    "inference_window_count": int(
                        inference_windows[name].inputs.shape[0]
                    ),
                    "aligned_score_count": int(aligned[name].residuals.shape[0]),
                    "access_fingerprint": interval.access_metadata.get(
                        "access_fingerprint"
                    ),
                }
                for name, interval in intervals.items()
            },
            "training": _training_summary(training),
            "provisional_threshold": {
                "value": provisional_threshold,
                "rule": (
                    "maximum over qualified dev_healthy calendar-day CUSUM maxima"
                ),
                "comparison": "strictly_greater_than",
                "candidate_day_count": REQUIRED_HEALTHY_DAYS,
                "minimum_qualified_day_count": MINIMUM_QUALIFIED_HEALTHY_DAYS,
                "qualified_day_count": sum(
                    record["qualified"] is True for record in daily_records
                ),
                "days": daily_records,
                "not_a_calibration_certificate": True,
            },
            "f1": f1_metrics,
            "direct_alarm_channels_excluded": list(DIRECT_ALARM_COLUMNS),
            "cpu_cuda_sample_parity": parity,
            "gates": gates,
            "next_phase_status": (
                "FINAL_PROTOCOL_REQUIRED_BEFORE_CALIBRATION_ACCESS"
                if decision == DEVELOPMENT_PASS
                else "CALIBRATION_REMAINS_SEALED"
            ),
        }
        atomic_create_new_json(output / RESULT_NAME, result)
    except Exception as error:
        _finalize_runtime_abort(
            output,
            claim=claim,
            intent=intent,
            phase_manifest_fingerprint=phase_fingerprint,
            error=error,
        )
        raise

    artifact_manifest = _normal_artifact_manifest(
        output,
        claim=claim,
        intent=intent,
        phase_manifest_fingerprint=phase_fingerprint,
    )
    completion = _create_completion(
        output,
        claim=claim,
        intent=intent,
        artifact_manifest=artifact_manifest,
        decision=decision,
    )
    expected_ledger_body = build_workspace_access_ledger(**ledger_arguments)
    verified = authenticate_completion_record(
        output / COMPLETION_NAME,
        output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=expected_ledger_body,
        access_intent_path=output / ACCESS_INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=DEVELOPMENT_ARTIFACT_NAMES,
    )
    if verified.fingerprint != completion.fingerprint:
        raise IntegrityError("development completion failed post-create verification")
    return DevelopmentRunOutcome(
        decision=decision,
        completion=verified,
        result=result,
        reused_completed_bundle=False,
    )


def _resolve_cli_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the charter-authorized MetroPT CARED-BC development phase. "
            "Calibration and confirmation remain sealed."
        )
    )
    parser.add_argument("csv_path")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"create-new experiment bundle (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--phase",
        default=PHASE_DEVELOPMENT,
        choices=(PHASE_DEVELOPMENT, PHASE_CALIBRATION, PHASE_CONFIRMATION),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--csv-chunksize", type=int, default=100_000)
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--charter", default=DEFAULT_CHARTER_PATH)
    parser.add_argument("--charter-sidecar", default=DEFAULT_CHARTER_SIDECAR_PATH)
    parser.add_argument("--parent-amendment", default=DEFAULT_PARENT_AMENDMENT_PATH)
    parser.add_argument(
        "--parent-amendment-sidecar",
        default=DEFAULT_PARENT_AMENDMENT_SIDECAR_PATH,
    )
    parser.add_argument("--amendment", default=DEFAULT_AMENDMENT_PATH)
    parser.add_argument(
        "--amendment-sidecar", default=DEFAULT_AMENDMENT_SIDECAR_PATH
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    workspace = Path(arguments.workspace_root).resolve()
    config = DevelopmentRunConfig(
        csv_path=_resolve_cli_path(arguments.csv_path, workspace),
        output_directory=_resolve_cli_path(arguments.output_dir, workspace),
        workspace_root=workspace,
        charter_path=_resolve_cli_path(arguments.charter, workspace),
        charter_sidecar_path=_resolve_cli_path(
            arguments.charter_sidecar, workspace
        ),
        parent_amendment_path=_resolve_cli_path(
            arguments.parent_amendment, workspace
        ),
        parent_amendment_sidecar_path=_resolve_cli_path(
            arguments.parent_amendment_sidecar, workspace
        ),
        amendment_path=_resolve_cli_path(arguments.amendment, workspace),
        amendment_sidecar_path=_resolve_cli_path(
            arguments.amendment_sidecar, workspace
        ),
        phase=arguments.phase,
        device=arguments.device,
        dtype=arguments.dtype,
        batch_size=arguments.batch_size,
        csv_chunksize=arguments.csv_chunksize,
    )
    try:
        outcome = run_development(config)
    except (GovernanceError, DevelopmentRunnerError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "ERROR",
                    "phase": arguments.phase,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "calibration_numeric_access_authorized": False,
                    "confirmation_numeric_access_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": outcome.decision,
                "completion_fingerprint": outcome.completion.fingerprint,
                "reused_completed_bundle": outcome.reused_completed_bundle,
                "output_directory": str(config.output_directory),
            },
            sort_keys=True,
        )
    )
    return 0 if outcome.decision == DEVELOPMENT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DevelopmentHooks",
    "DevelopmentRunConfig",
    "DevelopmentRunOutcome",
    "DevelopmentRunnerError",
    "InsufficientQualifiedDaysError",
    "authenticate_completed_development",
    "authenticate_recovery_bindings",
    "build_argument_parser",
    "build_development_phase_manifest",
    "check_cpu_cuda_sample_parity",
    "check_development_causality",
    "default_development_hooks",
    "evaluate_development_power_gate",
    "main",
    "run_development",
    "validate_preaccess_amendment",
    "validate_recovery_amendment",
    "validate_recovery_bindings",
]
