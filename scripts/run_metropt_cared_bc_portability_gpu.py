#!/usr/bin/env python
"""Fixed-model, full-sequence CPU/CUDA portability adjudication for MetroPT.

The runner is deliberately development-only.  It authenticates the frozen
v1 protocol and the complete v0_1 ABORT chain, creates a new governed access
claim, and only then reloads ``dev_healthy`` and ``dev_fault_f1``.  The frozen
ensemble and uncalibrated scorer are copied byte-for-byte; no training or
parameter estimation API is imported by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
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
    SAFE_CONTROL_COLUMNS,
    MetroPTMinuteInterval,
    MetroPTTimestampAudit,
    load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (  # noqa: E402
    CHARTER_ID,
    DEVELOPMENT_ABORT,
    DEVELOPMENT_CHARTER_SHA256,
    DEVELOPMENT_PASS,
    PHASE_DEVELOPMENT,
    PORTABILITY_DEVELOPMENT_ID,
    PORTABILITY_DEVELOPMENT_SHA256,
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
    AlignedResidualBatch,
    CaredBCScorer,
    CaredEnsemble,
    CaredScorePath,
    InferenceWindowBatch,
    align_inference_forecasts_to_score_times,
    build_causal_input_windows,
)
from mate_spc_metropt.portability import (  # noqa: E402
    PortabilityPath,
    audit_development_portability,
)


SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_PATH = "protocols/metropt_cared_bc_v1_portability_development.json"
DEFAULT_PROTOCOL_SIDECAR_PATH = (
    "protocols/metropt_cared_bc_v1_portability_development.sha256"
)
DEFAULT_PRIOR_OUTPUT = "outputs/metropt_cared_bc_gpu_v0_1"
DEFAULT_OUTPUT = "outputs/metropt_cared_bc_gpu_v1_portability"

LEDGER_NAME = "development_access_ledger.json"
INTENT_NAME = "development_access_intent.json"
PHASE_MANIFEST_NAME = "development_phase_manifest.json"
ENSEMBLE_NAME = "cared_ensemble.npz"
SCORER_NAME = "cared_scorer.npz"
EVIDENCE_NAME = "portability_evidence.npz"
AUDIT_NAME = "portability_audit.json"
RESULT_NAME = "development_result.json"
GPU_LOG_NAME = "gpu_execution_development.json"
ARTIFACT_MANIFEST_NAME = "development_artifact_manifest.json"
COMPLETION_NAME = "development_completion.json"
ABORT_NAME = "ABORTED.json"

NORMAL_ARTIFACT_NAMES = (
    PHASE_MANIFEST_NAME,
    ENSEMBLE_NAME,
    SCORER_NAME,
    EVIDENCE_NAME,
    AUDIT_NAME,
    RESULT_NAME,
    GPU_LOG_NAME,
)
PRIOR_ARTIFACT_NAMES = (
    "development_phase_manifest.json",
    "timestamp_audit.json",
    "cared_ensemble.npz",
    "cared_scorer.npz",
    "development_result.json",
    "gpu_execution_development.json",
)
LOAD_PARTITIONS = ("dev_healthy", "dev_fault_f1")
F1_ORIGIN = np.datetime64("2020-04-18T00:00:00", "ns")

AMENDMENT_001_SHA256 = (
    "9b998cd4866097443971f0f1fe6a18b6f9aba26964c164ed6a47d7684143bc83"
)
AMENDMENT_002_SHA256 = (
    "cf68a99a74ac1505975987785222d3095c8243394d13f60d5134ca8b4fe7f9bc"
)
V0_ABORT_SHA256 = "08d570c0a044a56cf5b91237ceb8a3c501aacfb84e05e2c4ddf9cb13a3ccaf14"
V0_COMPLETION_SHA256 = (
    "c57d421901ea1f38ea14d1412740439f5a2276cca15b51df27f83a74a3f4ebf6"
)
V0_COMPLETION_FINGERPRINT = (
    "305fe648c3a09e51fab2af0524902ed06c869d6098b660f5896630dbf10d30ee"
)


class PortabilityRunnerError(RuntimeError):
    """Raised when the fixed portability experiment cannot proceed safely."""


@dataclass(frozen=True)
class FrozenFile:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PriorBundleExpectation:
    output_relative_path: str = DEFAULT_PRIOR_OUTPUT
    phase_manifest_sha256: str = (
        "8c6275c800b99cb8842ad7ad3915ae996c36a184b86bfd9bcd7897924add5016"
    )
    phase_manifest_fingerprint: str = (
        "d26b87f46f7125fdb90ea8a07622e1438833171aadd3a7f4f0cd2a14f3970943"
    )
    artifact_manifest_sha256: str = (
        "d2e0543d80cd0e3526e3b5239e020f174d1030d0c3f7da75313090eba8d2e64f"
    )
    artifact_manifest_fingerprint: str = (
        "6218409d21de6cf5af82e94bb095a139d91a67e99b287b18cc8902402cfcefca"
    )
    completion_sha256: str = (
        "e68fe843083473088725d43f1a5d7639524263a62d6f55f8981d088fc8e504d2"
    )
    completion_fingerprint: str = (
        "54a40a2ae70bedc8c3275d28af27ca29d4f6a801cce0040abd72cff5fc639db0"
    )
    result_sha256: str = (
        "6a195f2f90383a9dcfc4c339cc629e2612e15e5d8cefa70bfb2faca4f731e116"
    )
    gpu_execution_sha256: str = (
        "d62340eb1667292365b3edb572ae39e5ecb59dd5fb58594f42549080552ed494"
    )
    timestamp_audit_sha256: str = (
        "c62d9c9dce50ea63507c301b8f7d6344d5426176c07dc098ac8219831944cbcc"
    )
    ensemble: FrozenFile = FrozenFile(
        ENSEMBLE_NAME,
        2_376_101,
        "4e75c7718482e66f0fd535a5bd0bf4c0dd78ec3335c3f420797d728c8e9d970f",
    )
    scorer: FrozenFile = FrozenFile(
        SCORER_NAME,
        181_175,
        "a7da9f5980778f0f65d51e7ab64e1cee3a9604cb186025d1240c45623d7d1887",
    )


@dataclass(frozen=True)
class PortabilityRunConfig:
    csv_path: Path
    output_directory: Path = WORKSPACE_ROOT / DEFAULT_OUTPUT
    workspace_root: Path = WORKSPACE_ROOT
    protocol_path: Path = WORKSPACE_ROOT / DEFAULT_PROTOCOL_PATH
    protocol_sidecar_path: Path = WORKSPACE_ROOT / DEFAULT_PROTOCOL_SIDECAR_PATH
    expected_protocol_sha256: str = PORTABILITY_DEVELOPMENT_SHA256
    prior_output_directory: Path = WORKSPACE_ROOT / DEFAULT_PRIOR_OUTPUT
    prior_expectation: PriorBundleExpectation = field(
        default_factory=PriorBundleExpectation
    )
    phase: str = PHASE_DEVELOPMENT
    device: str = "cuda"
    comparison_device: str = "cpu"
    dtype: str = "float64"
    batch_size: int = 1024
    csv_chunksize: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "csv_path",
            "output_directory",
            "workspace_root",
            "protocol_path",
            "protocol_sidecar_path",
            "prior_output_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True)
class PortabilityHooks:
    """Injectable numerical boundary for tests; no estimation hook exists."""

    load_partition: Callable[..., MetroPTMinuteInterval]
    load_ensemble: Callable[[Path], tuple[Any, Mapping[str, Any]]]
    load_scorer: Callable[[Path], tuple[Any, Mapping[str, Any]]]
    build_inference_windows: Callable[..., InferenceWindowBatch]
    predict: Callable[..., np.ndarray]
    align: Callable[..., AlignedResidualBatch]
    score: Callable[..., CaredScorePath]
    runtime_metadata: Callable[[PortabilityRunConfig], Mapping[str, Any]]
    portability_gate: Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class PortabilityRunOutcome:
    decision: str
    completion: AuthenticatedDocument
    result: Mapping[str, Any]
    reused_completed_bundle: bool


@dataclass(frozen=True)
class _PriorBundle:
    phase_manifest: AuthenticatedDocument
    completion: AuthenticatedDocument
    result: Mapping[str, Any]
    timestamp_audit: MetroPTTimestampAudit
    snapshot: Mapping[str, Mapping[str, Any]]


def _default_predict(
    ensemble: Any, inputs: np.ndarray, *, device: str, batch_size: int
) -> np.ndarray:
    return ensemble.predict(inputs, device=device, batch_size=batch_size)


def _default_score(
    scorer: Any,
    residuals: np.ndarray,
    state_codes: np.ndarray,
    timestamps: np.ndarray,
    *,
    reset_before: np.ndarray,
) -> CaredScorePath:
    return scorer.score(
        residuals, state_codes, timestamps, reset_before=reset_before
    )


def _default_runtime_metadata(config: PortabilityRunConfig) -> dict[str, Any]:
    requested = torch.device(config.device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise PortabilityRunnerError("the frozen portability run requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = torch.cuda.current_device() if requested.index is None else requested.index
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
    }


def default_portability_hooks() -> PortabilityHooks:
    return PortabilityHooks(
        load_partition=load_governed_metropt_partition,
        load_ensemble=CaredEnsemble.load_npz,
        load_scorer=CaredBCScorer.load_npz,
        build_inference_windows=build_causal_input_windows,
        predict=_default_predict,
        align=align_inference_forecasts_to_score_times,
        score=_default_score,
        runtime_metadata=_default_runtime_metadata,
        portability_gate=audit_development_portability,
    )


def _absolute(path: Path, root: Path) -> Path:
    return Path(path if path.is_absolute() else root / path).absolute()


def _workspace_relative(path: Path, workspace: Path) -> str:
    root = workspace.resolve(strict=True)
    candidate = path.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PortabilityRunnerError("path must remain inside workspace_root") from exc
    if not relative.parts:
        raise PortabilityRunnerError("path must not equal workspace_root")
    return PurePosixPath(*relative.parts).as_posix()


def _validate_config(config: PortabilityRunConfig) -> None:
    if config.phase != PHASE_DEVELOPMENT:
        raise AccessDeniedError("portability runner is development-only")
    if config.device.lower() != "cuda" or config.comparison_device.lower() != "cpu":
        raise PortabilityRunnerError("frozen devices are CUDA primary and CPU comparison")
    if config.dtype != "float64":
        raise PortabilityRunnerError("frozen portability dtype is float64")
    for name, value in (("batch_size", config.batch_size), ("csv_chunksize", config.csv_chunksize)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PortabilityRunnerError(f"{name} must be a positive integer")
    if config.expected_protocol_sha256 != PORTABILITY_DEVELOPMENT_SHA256:
        raise IntegrityError("expected protocol SHA differs from the compiled frozen SHA")


def _nested(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise IntegrityError(f"frozen protocol has invalid {name}")
    return result


def _validate_protocol(
    document: AuthenticatedDocument,
    config: PortabilityRunConfig,
) -> None:
    value = document.value
    expected_identity = {
        "schema": "metropt_cared_bc_portability_development_protocol",
        "schema_version": 1,
        "protocol_id": PORTABILITY_DEVELOPMENT_ID,
        "status": "FROZEN_BEFORE_PORTABILITY_NUMERIC_ACCESS",
    }
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise IntegrityError(f"frozen portability protocol changed: {key}")
    if document.record.get("sha256") != PORTABILITY_DEVELOPMENT_SHA256:
        raise IntegrityError("portability protocol byte identity changed")

    disclosure = _nested(value, "disclosure")
    for key, expected in {
        "created_after_observing_v0_1_abort": True,
        "not_an_original_v0_or_v0_1_preregistered_pass": True,
        "development_sensor_values_already_observed": True,
        "calibration_numeric_access_occurred": False,
        "confirmation_numeric_access_occurred": False,
        "sealed_partitions_remain_closed": True,
        "old_completed_bundles_must_remain_byte_identical": True,
    }.items():
        if disclosure.get(key) is not expected:
            raise IntegrityError(f"protocol disclosure changed: {key}")

    expectation = config.prior_expectation
    parents = _nested(value, "parent_bindings")
    fixed_parent_hashes = {
        "development_charter": DEVELOPMENT_CHARTER_SHA256,
        "preaccess_amendment_001": AMENDMENT_001_SHA256,
        "preaccess_amendment_002": AMENDMENT_002_SHA256,
    }
    for name, digest in fixed_parent_hashes.items():
        if _nested(parents, name).get("sha256") != digest:
            raise IntegrityError(f"protocol parent binding changed: {name}")
    v0 = _nested(parents, "v0_structural_abort")
    if (
        v0.get("decision") != DEVELOPMENT_ABORT
        or v0.get("abort_record_sha256") != V0_ABORT_SHA256
        or v0.get("completion_sha256") != V0_COMPLETION_SHA256
        or v0.get("completion_fingerprint") != V0_COMPLETION_FINGERPRINT
    ):
        raise IntegrityError("protocol v0 structural ABORT binding changed")
    prior = _nested(parents, "v0_1_parity_abort")
    expected_prior = {
        "output_directory": expectation.output_relative_path,
        "decision": DEVELOPMENT_ABORT,
        "phase_manifest_sha256": expectation.phase_manifest_sha256,
        "phase_manifest_fingerprint": expectation.phase_manifest_fingerprint,
        "artifact_manifest_sha256": expectation.artifact_manifest_sha256,
        "artifact_manifest_fingerprint": expectation.artifact_manifest_fingerprint,
        "completion_sha256": expectation.completion_sha256,
        "completion_fingerprint": expectation.completion_fingerprint,
        "result_sha256": expectation.result_sha256,
        "gpu_execution_sha256": expectation.gpu_execution_sha256,
    }
    for key, expected in expected_prior.items():
        if prior.get(key) != expected:
            raise IntegrityError(f"protocol v0_1 binding changed: {key}")

    fixed = _nested(value, "fixed_artifacts")
    ensemble = _nested(fixed, "ensemble")
    scorer = _nested(fixed, "uncalibrated_scorer")
    for observed, expected_file in ((ensemble, expectation.ensemble), (scorer, expectation.scorer)):
        expected_path = (
            PurePosixPath(expectation.output_relative_path) / expected_file.name
        ).as_posix()
        if (
            observed.get("path") != expected_path
            or observed.get("size_bytes") != expected_file.size_bytes
            or observed.get("sha256") != expected_file.sha256
        ):
            raise IntegrityError(f"protocol fixed artifact changed: {expected_file.name}")
    if scorer.get("threshold_must_be_null") is not True:
        raise IntegrityError("protocol no longer requires an uncalibrated scorer")
    for key in (
        "copy_exact_bytes_into_new_output",
        "ensemble_seeds_weights_scalers_and_dtype_frozen",
        "scorer_covariance_direction_rank_reference_and_cusum_frozen",
        "retraining_forbidden",
        "scorer_refitting_forbidden",
        "optimizer_or_backward_forbidden",
    ):
        if fixed.get(key) is not True:
            raise IntegrityError(f"protocol fixed-artifact rule changed: {key}")

    numeric = _nested(value, "numeric_access")
    if (
        numeric.get("runner_may_load_only") != list(LOAD_PARTITIONS)
        or numeric.get("fit_partition_must_not_be_loaded") is not True
        or numeric.get("sealed_or_excluded_partitions")
        != [
            "fit",
            "calibration_a",
            "embargo_a",
            "embargo_b",
            "calibration_b",
            "confirmation",
            "post_study_tail",
        ]
        or numeric.get("complete_sequence_required") is not True
        or numeric.get("sampling_or_prefix_parity_forbidden") is not True
        or numeric.get("governed_partition_loader_required") is not True
    ):
        raise IntegrityError("protocol numeric access contract changed")
    counts = _nested(numeric, "expected_counts")
    for split in LOAD_PARTITIONS:
        record = _nested(counts, split)
        for key in ("inference_windows", "aligned_scores"):
            item = record.get(key)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise IntegrityError(f"protocol expected count is invalid: {split}.{key}")

    runtime = _nested(value, "runtime")
    if runtime != {
        "primary_device": "cuda",
        "comparison_device": "cpu",
        "dtype": "float64",
        "batch_size": 1024,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
    }:
        raise IntegrityError("protocol runtime changed")
    if config.batch_size != int(runtime["batch_size"]):
        raise IntegrityError("runner batch size differs from the frozen protocol")

    gate = _nested(value, "portability_gate")
    required = _nested(gate, "required_checks")
    if (
        gate.get("primitive_max_absolute_tolerance") != 1.0e-9
        or gate.get("boundary_distance_tolerance") != 1.0e-9
        or gate.get("prediction_shape") != ["N", 3, 7]
        or gate.get("residual_shape") != ["M", 31]
        or gate.get("forecast_residual_width") != 21
        or gate.get("physical_feature_width") != 10
        or gate.get("physical_features_cpu_cuda_exact") is not True
        or gate.get("timestamps_score_indices_state_codes_and_reset_exact") is not True
        or gate.get("effective_boundary_can_only_increase") is not True
        or gate.get("effective_boundary_formula")
        != "nextafter(h_raw + global_envelope_delta + 1e-9, +infinity)"
        or gate.get("alarm_comparison") != "strictly_greater_than"
        or required.get("full_cpu_cuda_alarm_arrays_exact") is not True
        or required.get("dev_healthy_alarm_count_each_device") != 0
        or required.get("f1_detected_each_device") is not True
        or required.get("f1_first_alarm_availability_exact") is not True
        or required.get("f1_availability_lag_minutes") != 1
        or required.get("f1_maximum_delay_minutes_inclusive") != 60
        or required.get("f1_minimum_peak_to_effective_boundary_ratio_inclusive") != 1.25
        or required.get("minimum_absolute_distance_of_every_envelope_to_boundary_strictly_greater_than")
        != 1.0e-9
    ):
        raise IntegrityError("protocol portability gate changed")

    output_contract = _nested(value, "output_contract")
    if (
        output_contract.get("directory")
        != _workspace_relative(config.output_directory, config.workspace_root)
        or output_contract.get("copy_fixed_npz_artifacts") is not True
        or output_contract.get("save_alarm_array_hashes") is not True
        or output_contract.get("create_new_only") is not True
        or output_contract.get("completed_reentry_must_not_read_csv") is not True
        or output_contract.get("implementation_snapshot_required_before_numeric_access") is not True
        or output_contract.get("calibration_access_authorized_directly_by_this_protocol") is not False
        or output_contract.get("final_analysis_protocol_still_required_after_pass") is not True
    ):
        raise IntegrityError("protocol output contract changed")
    outcome = _nested(value, "outcome_policy")
    if (
        outcome.get("pass_decision") != DEVELOPMENT_PASS
        or outcome.get("failure_decision") != DEVELOPMENT_ABORT
        or outcome.get("failure_next_method_family") != "MB-CycleMIL-BC"
        or outcome.get("further_cared_bc_development_repair_after_failure_forbidden") is not True
        or outcome.get("calibration_and_confirmation_remain_sealed_on_failure") is not True
    ):
        raise IntegrityError("protocol outcome policy changed")


def _snapshot_directory(directory: Path) -> dict[str, dict[str, Any]]:
    root = directory.resolve(strict=True)
    if directory.is_symlink() or not root.is_dir():
        raise IntegrityError("prior output must be a real directory")
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise IntegrityError(f"prior output contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    if not result:
        raise IntegrityError("prior output is empty")
    return result


def _assert_record(path: Path, expected_sha: str, expected_size: int | None = None) -> None:
    record = artifact_record(path)
    if record.get("sha256") != expected_sha or (
        expected_size is not None and record.get("size_bytes") != expected_size
    ):
        raise IntegrityError(f"frozen artifact changed: {path.name}")


def _audit_from_document(value: Mapping[str, Any]) -> MetroPTTimestampAudit:
    required_identity = {
        "schema": "metropt_cared_bc_timestamp_audit",
        "schema_version": 1,
        "strictly_increasing": True,
        "duplicate_timestamp_count": 0,
        "timestamp_values_parsed": True,
        "sensor_numeric_values_parsed": False,
        "numeric_columns_parsed": [],
    }
    for key, expected in required_identity.items():
        if value.get(key) != expected:
            raise IntegrityError(f"prior timestamp audit changed: {key}")
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
        raise IntegrityError("prior timestamp audit is not canonical")
    return audit


def _authenticate_prior_bundle(
    config: PortabilityRunConfig,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> _PriorBundle:
    output = config.prior_output_directory
    expectation = config.prior_expectation
    expected_relative = _workspace_relative(output, config.workspace_root)
    if expected_relative != expectation.output_relative_path:
        raise IntegrityError("prior output path differs from the frozen binding")

    phase = authenticate_fingerprinted_json(
        output / PHASE_MANIFEST_NAME, fingerprint_field="manifest_fingerprint"
    )
    if (
        phase.record.get("sha256") != expectation.phase_manifest_sha256
        or phase.fingerprint != expectation.phase_manifest_fingerprint
        or phase.value.get("protocol_id") != CHARTER_ID
        or phase.value.get("charter_sha256") != DEVELOPMENT_CHARTER_SHA256
        or phase.value.get("phase") != PHASE_DEVELOPMENT
        or phase.value.get("output_directory_relative_path")
        != expectation.output_relative_path
    ):
        raise IntegrityError("prior phase manifest differs from its frozen binding")
    sources = _nested(phase.value, "sources")
    sensor = _nested(sources, "sensor_csv")
    if phase.value.get("source_fingerprint") != canonical_fingerprint(sources):
        raise IntegrityError("prior source fingerprint changed")

    ledger_value = read_strict_json(output / LEDGER_NAME)
    ledger_body = dict(ledger_value)
    ledger_body.pop("ledger_fingerprint", None)
    expected_ledger_fields = {
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": CHARTER_ID,
        "charter_sha256": DEVELOPMENT_CHARTER_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": sensor.get("sha256"),
        "source_fingerprint": phase.value.get("source_fingerprint"),
        "phase_manifest_fingerprint": phase.fingerprint,
        "output_directory": str(output.resolve(strict=True)),
    }
    for key, expected in expected_ledger_fields.items():
        if ledger_body.get(key) != expected:
            raise IntegrityError(f"prior ledger changed: {key}")
    expected_ledger = build_workspace_access_ledger(
        phase=PHASE_DEVELOPMENT,
        workspace_relative_path=(
            PurePosixPath(expectation.output_relative_path) / LEDGER_NAME
        ).as_posix(),
        protocol_id=CHARTER_ID,
        charter_sha256=DEVELOPMENT_CHARTER_SHA256,
        final_protocol_sha256=None,
        sensor_csv_sha256=str(sensor["sha256"]),
        source_fingerprint=str(phase.value["source_fingerprint"]),
        phase_manifest_fingerprint=str(phase.fingerprint),
        output_directory=output,
    )
    if ledger_body != expected_ledger:
        raise IntegrityError("prior ledger semantic body changed")

    manifest = authenticate_fingerprinted_json(
        output / ARTIFACT_MANIFEST_NAME,
        fingerprint_field="artifact_manifest_fingerprint",
    )
    if (
        manifest.record.get("sha256") != expectation.artifact_manifest_sha256
        or manifest.fingerprint != expectation.artifact_manifest_fingerprint
        or set(_nested(manifest.value, "artifacts")) != set(PRIOR_ARTIFACT_NAMES)
    ):
        raise IntegrityError("prior artifact manifest differs from its frozen binding")
    completion = authenticate_completion_record(
        output / COMPLETION_NAME,
        output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=expected_ledger,
        access_intent_path=output / INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=PRIOR_ARTIFACT_NAMES,
    )
    if (
        completion.record.get("sha256") != expectation.completion_sha256
        or completion.fingerprint != expectation.completion_fingerprint
        or completion.value.get("decision") != DEVELOPMENT_ABORT
    ):
        raise IntegrityError("prior completion differs from its frozen ABORT")

    _assert_record(output / RESULT_NAME, expectation.result_sha256)
    _assert_record(output / GPU_LOG_NAME, expectation.gpu_execution_sha256)
    _assert_record(
        output / ENSEMBLE_NAME,
        expectation.ensemble.sha256,
        expectation.ensemble.size_bytes,
    )
    _assert_record(
        output / SCORER_NAME,
        expectation.scorer.sha256,
        expectation.scorer.size_bytes,
    )
    _assert_record(output / "timestamp_audit.json", expectation.timestamp_audit_sha256)
    result = read_strict_json(output / RESULT_NAME)
    evidence = _nested(result, "fixed_development_evidence") if "fixed_development_evidence" in result else None
    threshold = _nested(result, "provisional_threshold") if evidence is None else evidence
    h_raw = threshold.get("value", threshold.get("raw_provisional_threshold"))
    training = _nested(result, "training") if "training" in result else evidence
    improvement = training.get(
        "relative_improvement_over_persistence",
        training.get("tcn_relative_improvement_over_persistence"),
    )
    if (
        result.get("decision") != DEVELOPMENT_ABORT
        or float(h_raw) != 36.02052930938596
        or float(improvement) != 0.5025773655903889
    ):
        raise IntegrityError("prior fixed scientific evidence changed")
    if "gates" in result:
        gates = _nested(result, "gates")
        required_gates = (
            "tcn_relative_improvement",
            "f1_alarm",
            "f1_first_alarm_delay_minutes",
            "f1_maximum_to_threshold_ratio",
            "direct_alarm_channels_excluded",
            "feature_and_score_causality",
        )
        if any(_nested(gates, name).get("passed") is not True for name in required_gates):
            raise IntegrityError("a prior non-parity development gate no longer passes")

    audit_value = read_strict_json(output / "timestamp_audit.json")
    return _PriorBundle(
        phase_manifest=phase,
        completion=completion,
        result=result,
        timestamp_audit=_audit_from_document(audit_value),
        snapshot=dict(snapshot),
    )


def _implementation_snapshot() -> dict[str, Mapping[str, Any]]:
    paths = {
        "runner": Path(__file__).resolve(strict=True),
        "portability": SOURCE_ROOT / "mate_spc_metropt" / "portability.py",
        "model": SOURCE_ROOT / "mate_spc_metropt" / "model.py",
        "data": SOURCE_ROOT / "mate_spc_metropt" / "data.py",
        "governance": SOURCE_ROOT / "mate_spc_metropt" / "governance.py",
    }
    return {name: artifact_record(path) for name, path in sorted(paths.items())}


def _build_phase_manifest(
    config: PortabilityRunConfig,
    protocol: AuthenticatedDocument,
    prior: _PriorBundle,
) -> dict[str, Any]:
    sources = _nested(prior.phase_manifest.value, "sources")
    output_relative = _workspace_relative(config.output_directory, config.workspace_root)
    counts = _nested(_nested(protocol.value, "numeric_access"), "expected_counts")
    return {
        "schema": "metropt_cared_bc_portability_development_phase_manifest",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": PORTABILITY_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_DEVELOPMENT_SHA256,
        "protocol_record": dict(protocol.record),
        "prior_v0_1_abort": {
            "output_directory": config.prior_expectation.output_relative_path,
            "completion_sha256": config.prior_expectation.completion_sha256,
            "completion_fingerprint": config.prior_expectation.completion_fingerprint,
            "decision": DEVELOPMENT_ABORT,
            "directory_snapshot_fingerprint": canonical_fingerprint(prior.snapshot),
        },
        "fixed_artifacts": {
            ENSEMBLE_NAME: {
                "size_bytes": config.prior_expectation.ensemble.size_bytes,
                "sha256": config.prior_expectation.ensemble.sha256,
            },
            SCORER_NAME: {
                "size_bytes": config.prior_expectation.scorer.size_bytes,
                "sha256": config.prior_expectation.scorer.sha256,
                "threshold_must_be_null": True,
            },
            "exact_byte_copy_required": True,
            "estimation_forbidden": True,
        },
        "numeric_access": {
            "loaded_partitions": list(LOAD_PARTITIONS),
            "fit_loaded": False,
            "expected_counts": {name: dict(counts[name]) for name in LOAD_PARTITIONS},
            "complete_sequence_both_devices": True,
            "sealed_partitions_remain_closed": True,
        },
        "source_fingerprint": prior.phase_manifest.value["source_fingerprint"],
        "sources": {key: dict(item) for key, item in sources.items()},
        "sensor_csv_relative_path": prior.phase_manifest.value["sensor_csv_relative_path"],
        "output_directory_relative_path": output_relative,
        "ledger_workspace_relative_path": (
            PurePosixPath(output_relative) / LEDGER_NAME
        ).as_posix(),
        "runtime": {
            "primary_device": config.device,
            "comparison_device": config.comparison_device,
            "dtype": config.dtype,
            "batch_size": config.batch_size,
            "csv_chunksize": config.csv_chunksize,
        },
        "implementation_snapshot": _implementation_snapshot(),
        "portability_gate": dict(protocol.value["portability_gate"]),
        "output_artifacts": list(NORMAL_ARTIFACT_NAMES),
        "calibration_numeric_access_authorized": False,
        "confirmation_numeric_access_authorized": False,
    }


def _ledger_arguments(
    config: PortabilityRunConfig,
    phase: Mapping[str, Any],
    phase_fingerprint: str,
) -> dict[str, Any]:
    sources = _nested(phase, "sources")
    sensor = _nested(sources, "sensor_csv")
    return {
        "phase": PHASE_DEVELOPMENT,
        "workspace_relative_path": str(phase["ledger_workspace_relative_path"]),
        "protocol_id": PORTABILITY_DEVELOPMENT_ID,
        "charter_sha256": PORTABILITY_DEVELOPMENT_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": str(sensor["sha256"]),
        "source_fingerprint": str(phase["source_fingerprint"]),
        "phase_manifest_fingerprint": phase_fingerprint,
        "output_directory": config.output_directory,
    }


def _ensure_fresh_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise IntegrityError("output must be a real directory")
        if any(output.iterdir()):
            raise PortabilityRunnerError(
                "partial output exists without a reusable completion record"
            )
        return
    output.mkdir(parents=True, exist_ok=False)


def _copy_frozen(source: Path, destination: Path, expected: FrozenFile) -> None:
    _assert_record(source, expected.sha256, expected.size_bytes)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    shutil.copyfile(source, destination)
    _assert_record(destination, expected.sha256, expected.size_bytes)


def _validate_interval(name: str, interval: Any, sensor_sha: str) -> None:
    timestamps = np.asarray(interval.timestamps, dtype="datetime64[ns]")
    analog = np.asarray(interval.analog, dtype=np.float64)
    controls = np.asarray(interval.controls, dtype=np.float64)
    eligible = np.asarray(interval.eligible)
    reset = np.asarray(interval.reset_before)
    if (
        timestamps.ndim != 1
        or timestamps.size == 0
        or analog.shape != (timestamps.size, len(ANALOG_COLUMNS))
        or controls.shape != (timestamps.size, len(SAFE_CONTROL_COLUMNS))
        or eligible.shape != (timestamps.size,)
        or eligible.dtype.kind != "b"
        or reset.shape != (timestamps.size,)
        or reset.dtype.kind != "b"
        or not bool(reset[0])
    ):
        raise IntegrityError(f"{name} interval has invalid arrays")
    metadata = interval.access_metadata
    if (
        metadata.get("source_sha256") != sensor_sha
        or metadata.get("unrequested_numeric_values_parsed") is not False
        or metadata.get("direct_alarm_values_parsed") is not False
        or metadata.get("numeric_columns_parsed") != list(APPROVED_NUMERIC_COLUMNS)
    ):
        raise IntegrityError(f"{name} governed-access evidence changed")


def _load_intervals(
    config: PortabilityRunConfig,
    hooks: PortabilityHooks,
    prior: _PriorBundle,
    claim: LedgerClaim,
    intent: AuthenticatedDocument,
    phase_fingerprint: str,
) -> dict[str, Any]:
    sensor = _nested(_nested(prior.phase_manifest.value, "sources"), "sensor_csv")
    intervals: dict[str, Any] = {}
    for name in LOAD_PARTITIONS:
        interval = hooks.load_partition(
            config.csv_path,
            name,
            claim=claim,
            access_intent=intent,
            access_intent_path=config.output_directory / INTENT_NAME,
            phase_manifest_fingerprint=phase_fingerprint,
            chunksize=config.csv_chunksize,
            timestamp_audit=prior.timestamp_audit,
        )
        _validate_interval(name, interval, str(sensor["sha256"]))
        intervals[name] = interval
    if tuple(intervals) != LOAD_PARTITIONS:
        raise AccessDeniedError("numeric loader crossed the frozen portability scope")
    return intervals


def _validate_alignment(name: str, aligned: Any, expected_count: int) -> None:
    residual = np.asarray(aligned.residuals, dtype=np.float64)
    timestamp = np.asarray(aligned.score_timestamps, dtype="datetime64[ns]")
    indices = np.asarray(aligned.score_indices)
    states = np.asarray(aligned.state_codes)
    reset = np.asarray(aligned.reset_before)
    if (
        residual.shape != (expected_count, 31)
        or timestamp.shape != (expected_count,)
        or indices.shape != (expected_count,)
        or states.shape != (expected_count,)
        or reset.shape != (expected_count,)
        or not np.isfinite(residual).all()
    ):
        raise IntegrityError(f"{name} aligned path violates frozen count/shape")


def _path(
    prediction: np.ndarray,
    aligned: Any,
    score: Any,
) -> PortabilityPath:
    return PortabilityPath(
        prediction=np.asarray(prediction),
        residual=np.asarray(aligned.residuals),
        timestamps=np.asarray(aligned.score_timestamps),
        indices=np.asarray(aligned.score_indices),
        state_codes=np.asarray(aligned.state_codes),
        reset_before=np.asarray(aligned.reset_before),
        envelope=np.asarray(score.envelope),
    )


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _alarm_evidence(
    paths: Mapping[str, Mapping[str, PortabilityPath]], h_eff: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for split, devices in paths.items():
        records[split] = {}
        for device, path in devices.items():
            alarm = np.asarray(path.envelope > h_eff, dtype=np.bool_)
            key = f"{split}__{device}__alarm"
            arrays[key] = alarm
            records[split][device] = {
                "count": int(np.count_nonzero(alarm)),
                "shape": list(alarm.shape),
                "dtype": str(alarm.dtype),
                "sha256": _array_sha256(alarm),
            }
    return records, arrays


def _save_evidence(
    path: Path,
    paths: Mapping[str, Mapping[str, PortabilityPath]],
    alarms: Mapping[str, np.ndarray],
    audit: Mapping[str, Any],
) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload: dict[str, np.ndarray] = {}
    for split, devices in paths.items():
        for device, item in devices.items():
            prefix = f"{split}__{device}"
            payload[f"{prefix}__prediction"] = np.asarray(item.prediction)
            payload[f"{prefix}__residual"] = np.asarray(item.residual)
            payload[f"{prefix}__timestamps_ns"] = np.asarray(
                item.timestamps, dtype="datetime64[ns]"
            ).astype(np.int64)
            payload[f"{prefix}__indices"] = np.asarray(item.indices)
            payload[f"{prefix}__state_codes"] = np.asarray(item.state_codes)
            payload[f"{prefix}__reset_before"] = np.asarray(item.reset_before)
            payload[f"{prefix}__envelope"] = np.asarray(item.envelope)
    payload.update({key: np.asarray(value) for key, value in alarms.items()})
    payload["h_raw"] = np.asarray(float(audit["threshold"]["h_raw"]))
    payload["global_envelope_delta"] = np.asarray(
        float(audit["threshold"]["global_envelope_delta"])
    )
    payload["h_eff"] = np.asarray(float(audit["threshold"]["h_eff"]))
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
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


def _exact_backend_checks(aligned: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = (
        "score_timestamps",
        "score_indices",
        "state_codes",
        "reset_before",
        "source_window_rows",
        "physical_features",
    )
    for split, devices in aligned.items():
        cpu = devices["cpu"]
        cuda = devices["cuda"]
        checks = {
            f"{field}_exact": bool(
                np.array_equal(np.asarray(getattr(cpu, field)), np.asarray(getattr(cuda, field)))
            )
            for field in fields
        }
        result[split] = {"checks": checks, "passed": bool(all(checks.values()))}
    result["passed"] = bool(all(result[name]["passed"] for name in LOAD_PARTITIONS))
    return result


def _legacy_evidence(
    protocol: Mapping[str, Any],
    prior_result: Mapping[str, Any],
    prior_completion_fingerprint: str,
) -> dict[str, Any]:
    declared = _nested(protocol, "fixed_development_evidence")
    threshold = _nested(prior_result, "provisional_threshold")
    training = _nested(prior_result, "training")
    if (
        float(threshold.get("value")) != float(declared["raw_provisional_threshold"])
        or float(training.get("relative_improvement_over_persistence"))
        != float(declared["tcn_relative_improvement_over_persistence"])
    ):
        raise IntegrityError("protocol and prior scientific evidence disagree")
    names = declared.get("legacy_nonparity_gates_must_all_remain_passed")
    gates = _nested(prior_result, "gates")
    if not isinstance(names, list) or any(_nested(gates, str(name)).get("passed") is not True for name in names):
        raise IntegrityError("legacy non-parity gate evidence is incomplete")
    return {
        "h_raw": float(threshold["value"]),
        "tcn_relative_improvement_over_persistence": float(
            training["relative_improvement_over_persistence"]
        ),
        "gate_names": list(names),
        "all_passed": True,
        "source_completion_fingerprint": prior_completion_fingerprint,
    }


def _artifact_names_from_manifest(path: Path) -> tuple[str, ...]:
    value = read_strict_json(path)
    artifacts = _nested(value, "artifacts")
    names = tuple(sorted(str(name) for name in artifacts))
    normal = set(NORMAL_ARTIFACT_NAMES)
    observed = set(names)
    if observed != normal:
        if ABORT_NAME not in observed or not observed.issubset(normal | {ABORT_NAME}):
            raise IntegrityError("saved portability artifact set is invalid")
        if PHASE_MANIFEST_NAME not in observed:
            raise IntegrityError("runtime ABORT is not bound to its phase manifest")
    return names


def _create_artifact_manifest(
    output: Path,
    intent: AuthenticatedDocument,
    phase_fingerprint: str,
    artifact_names: tuple[str, ...],
) -> AuthenticatedDocument:
    return create_artifact_manifest(
        output / ARTIFACT_MANIFEST_NAME,
        output,
        phase=PHASE_DEVELOPMENT,
        protocol_id=PORTABILITY_DEVELOPMENT_ID,
        protocol_sha256=PORTABILITY_DEVELOPMENT_SHA256,
        phase_manifest_fingerprint=phase_fingerprint,
        access_intent=intent,
        artifact_names=artifact_names,
    )


def _finalize_abort(
    output: Path,
    claim: LedgerClaim,
    intent: AuthenticatedDocument,
    phase_fingerprint: str,
    error: Exception,
) -> AuthenticatedDocument:
    atomic_create_new_json(
        output / ABORT_NAME,
        {
            "schema": "metropt_cared_bc_portability_runtime_abort",
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE_DEVELOPMENT,
            "protocol_id": PORTABILITY_DEVELOPMENT_ID,
            "protocol_sha256": PORTABILITY_DEVELOPMENT_SHA256,
            "phase_manifest_fingerprint": phase_fingerprint,
            "decision": DEVELOPMENT_ABORT,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "calibration_numeric_access_authorized": False,
            "confirmation_numeric_access_authorized": False,
        },
    )
    candidates = (*NORMAL_ARTIFACT_NAMES, ABORT_NAME)
    names = tuple(
        name
        for name in candidates
        if (output / name).is_file() and not (output / name).is_symlink()
    )
    manifest = _create_artifact_manifest(output, intent, phase_fingerprint, names)
    created = create_completion_record(
        output / COMPLETION_NAME,
        claim=claim,
        access_intent=intent,
        artifact_manifest=manifest,
        decision=DEVELOPMENT_ABORT,
    )
    ledger_body = dict(claim.document.value)
    ledger_body.pop("ledger_fingerprint", None)
    verified = authenticate_completion_record(
        output / COMPLETION_NAME,
        output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=ledger_body,
        access_intent_path=output / INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=names,
    )
    if verified.fingerprint != created.fingerprint:
        raise IntegrityError("runtime ABORT failed post-create authentication")
    return verified


def _authenticate_completed(
    config: PortabilityRunConfig,
    protocol: AuthenticatedDocument,
    prior: _PriorBundle,
) -> PortabilityRunOutcome:
    output = config.output_directory
    phase = authenticate_fingerprinted_json(
        output / PHASE_MANIFEST_NAME, fingerprint_field="manifest_fingerprint"
    )
    if (
        phase.value.get("schema")
        != "metropt_cared_bc_portability_development_phase_manifest"
        or phase.value.get("protocol_id") != PORTABILITY_DEVELOPMENT_ID
        or phase.value.get("protocol_sha256") != PORTABILITY_DEVELOPMENT_SHA256
        or phase.value.get("protocol_record") != protocol.record
        or phase.value.get("output_directory_relative_path")
        != _workspace_relative(output, config.workspace_root)
        or _nested(phase.value, "prior_v0_1_abort").get("completion_fingerprint")
        != prior.completion.fingerprint
    ):
        raise IntegrityError("completed portability phase manifest changed")
    expected_csv = config.workspace_root / PurePosixPath(
        str(phase.value["sensor_csv_relative_path"])
    )
    if config.csv_path != expected_csv:
        raise IntegrityError("completed reentry names a different CSV path")
    ledger_arguments = _ledger_arguments(config, phase.value, str(phase.fingerprint))
    names = _artifact_names_from_manifest(output / ARTIFACT_MANIFEST_NAME)
    completion = authenticate_completion_record(
        output / COMPLETION_NAME,
        output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=build_workspace_access_ledger(**ledger_arguments),
        access_intent_path=output / INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=names,
    )
    decision = str(completion.value["decision"])
    if RESULT_NAME in names:
        result = read_strict_json(output / RESULT_NAME)
        if (
            result.get("decision") != decision
            or result.get("protocol_sha256") != PORTABILITY_DEVELOPMENT_SHA256
            or result.get("phase_manifest_fingerprint") != phase.fingerprint
        ):
            raise IntegrityError("saved portability result differs from completion")
    else:
        result = read_strict_json(output / ABORT_NAME)
        if decision != DEVELOPMENT_ABORT or result.get("decision") != DEVELOPMENT_ABORT:
            raise IntegrityError("saved runtime ABORT differs from completion")
    return PortabilityRunOutcome(decision, completion, result, True)


def run_portability_development(
    config: PortabilityRunConfig,
    *,
    hooks: PortabilityHooks | None = None,
) -> PortabilityRunOutcome:
    """Run or authenticate the frozen fixed-model portability experiment."""

    _validate_config(config)
    workspace = config.workspace_root.resolve(strict=True)
    object.__setattr__(config, "workspace_root", workspace)
    for name in (
        "csv_path",
        "output_directory",
        "protocol_path",
        "protocol_sidecar_path",
        "prior_output_directory",
    ):
        object.__setattr__(config, name, _absolute(getattr(config, name), workspace))
    output = config.output_directory
    prior_output = config.prior_output_directory
    if output == prior_output or prior_output in output.parents or output in prior_output.parents:
        raise AccessDeniedError("new output and immutable prior output must be disjoint")

    prior_snapshot = _snapshot_directory(prior_output)
    try:
        protocol = authenticate_json_with_sidecar(
            config.protocol_path,
            config.protocol_sidecar_path,
            expected_sha256=PORTABILITY_DEVELOPMENT_SHA256,
        )
        _validate_protocol(protocol, config)
        prior = _authenticate_prior_bundle(config, prior_snapshot)

        if (output / COMPLETION_NAME).is_file():
            return _authenticate_completed(config, protocol, prior)
        if output.exists() and any(output.iterdir()):
            raise PortabilityRunnerError(
                "partial portability output exists; numeric access will not be repeated"
            )

        expected_csv = workspace / PurePosixPath(
            str(prior.phase_manifest.value["sensor_csv_relative_path"])
        )
        if config.csv_path != expected_csv:
            raise IntegrityError("csv_path differs from the source bound by v0_1")

        phase_body = _build_phase_manifest(config, protocol, prior)
        phase_fingerprint = canonical_fingerprint(phase_body)
        _ensure_fresh_output(output)
        phase_document = create_fingerprinted_json(
            output / PHASE_MANIFEST_NAME,
            phase_body,
            fingerprint_field="manifest_fingerprint",
        )
        if phase_document.fingerprint != phase_fingerprint:
            raise IntegrityError("created portability phase manifest changed")
        ledger_arguments = _ledger_arguments(config, phase_body, phase_fingerprint)
        claim = claim_workspace_access_ledger(output / LEDGER_NAME, **ledger_arguments)
        intent = create_phase_access_intent(
            output / INTENT_NAME,
            claim,
            phase_manifest_fingerprint=phase_fingerprint,
        )

        active = hooks or default_portability_hooks()
        try:
            _copy_frozen(
                prior_output / ENSEMBLE_NAME,
                output / ENSEMBLE_NAME,
                config.prior_expectation.ensemble,
            )
            _copy_frozen(
                prior_output / SCORER_NAME,
                output / SCORER_NAME,
                config.prior_expectation.scorer,
            )
            ensemble, ensemble_metadata = active.load_ensemble(output / ENSEMBLE_NAME)
            scorer, scorer_metadata = active.load_scorer(output / SCORER_NAME)
            if getattr(scorer, "threshold", None) is not None:
                raise IntegrityError("the copied scorer is not uncalibrated")
            if getattr(ensemble, "dtype", config.dtype) != config.dtype:
                raise IntegrityError("the copied ensemble dtype changed")

            intervals = _load_intervals(
                config, active, prior, claim, intent, phase_fingerprint
            )
            windows = {
                name: active.build_inference_windows(
                    interval.analog,
                    interval.controls,
                    interval.timestamps,
                    eligible=interval.eligible,
                    reset_before=interval.reset_before,
                )
                for name, interval in intervals.items()
            }
            expected_counts = _nested(
                _nested(protocol.value, "numeric_access"), "expected_counts"
            )
            for name in LOAD_PARTITIONS:
                expected = int(_nested(expected_counts, name)["inference_windows"])
                if int(windows[name].inputs.shape[0]) != expected:
                    raise IntegrityError(f"{name} inference-window count changed")

            runtime = dict(active.runtime_metadata(config))
            predictions: dict[str, dict[str, np.ndarray]] = {
                name: {} for name in LOAD_PARTITIONS
            }
            aligned: dict[str, dict[str, Any]] = {name: {} for name in LOAD_PARTITIONS}
            scores: dict[str, dict[str, Any]] = {name: {} for name in LOAD_PARTITIONS}
            timings: list[dict[str, Any]] = []
            for name in LOAD_PARTITIONS:
                interval = intervals[name]
                for device_name, device in (
                    ("cpu", config.comparison_device),
                    ("cuda", config.device),
                ):
                    started = time.perf_counter()
                    prediction = np.asarray(
                        active.predict(
                            ensemble,
                            windows[name].inputs,
                            device=device,
                            batch_size=config.batch_size,
                        ),
                        dtype=np.float64,
                    )
                    elapsed = time.perf_counter() - started
                    expected_windows = int(
                        _nested(expected_counts, name)["inference_windows"]
                    )
                    if prediction.shape != (expected_windows, 3, 7) or not np.isfinite(prediction).all():
                        raise IntegrityError(f"{name}.{device_name} prediction shape changed")
                    aligned_path = active.align(
                        windows[name],
                        prediction,
                        interval.analog,
                        interval.controls,
                        interval.timestamps,
                        eligible=interval.eligible,
                        reset_before=interval.reset_before,
                    )
                    expected_scores = int(_nested(expected_counts, name)["aligned_scores"])
                    _validate_alignment(f"{name}.{device_name}", aligned_path, expected_scores)
                    score_path = active.score(
                        scorer,
                        aligned_path.residuals,
                        aligned_path.state_codes,
                        aligned_path.score_timestamps,
                        reset_before=aligned_path.reset_before,
                    )
                    if np.asarray(score_path.envelope).shape != (expected_scores,):
                        raise IntegrityError(f"{name}.{device_name} score count changed")
                    predictions[name][device_name] = prediction
                    aligned[name][device_name] = aligned_path
                    scores[name][device_name] = score_path
                    timings.append(
                        {
                            "partition": name,
                            "device": device,
                            "prediction_seconds": float(elapsed),
                            "inference_window_count": expected_windows,
                            "aligned_score_count": expected_scores,
                        }
                    )

            paths = {
                name: {
                    device: _path(
                        predictions[name][device],
                        aligned[name][device],
                        scores[name][device],
                    )
                    for device in ("cpu", "cuda")
                }
                for name in LOAD_PARTITIONS
            }
            legacy = _legacy_evidence(
                protocol.value,
                prior.result,
                str(prior.completion.fingerprint),
            )
            core_audit = dict(
                active.portability_gate(
                    healthy_cpu=paths["dev_healthy"]["cpu"],
                    healthy_cuda=paths["dev_healthy"]["cuda"],
                    f1_cpu=paths["dev_fault_f1"]["cpu"],
                    f1_cuda=paths["dev_fault_f1"]["cuda"],
                    h_raw=legacy["h_raw"],
                    f1_origin=F1_ORIGIN,
                )
            )
            if not isinstance(core_audit.get("passed"), bool):
                raise IntegrityError("portability gate returned an invalid decision")
            exact = _exact_backend_checks(aligned)
            overall_passed = bool(core_audit["passed"] and exact["passed"] and legacy["all_passed"])
            decision = DEVELOPMENT_PASS if overall_passed else DEVELOPMENT_ABORT
            h_eff = float(_nested(core_audit, "threshold")["h_eff"])
            alarm_records, alarm_arrays = _alarm_evidence(paths, h_eff)
            audit_document = {
                "schema": "metropt_cared_bc_full_development_portability_adjudication",
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PORTABILITY_DEVELOPMENT_ID,
                "protocol_sha256": PORTABILITY_DEVELOPMENT_SHA256,
                "phase_manifest_fingerprint": phase_fingerprint,
                "core_portability_audit": core_audit,
                "exact_backend_checks": exact,
                "legacy_nonparity_gate_evidence": legacy,
                "alarm_array_hashes": alarm_records,
                "passed": overall_passed,
                "decision": decision,
            }
            atomic_create_new_json(output / AUDIT_NAME, audit_document)
            _save_evidence(output / EVIDENCE_NAME, paths, alarm_arrays, core_audit)
            execution = {
                "schema": "metropt_cared_bc_portability_gpu_execution",
                "schema_version": SCHEMA_VERSION,
                "phase": PHASE_DEVELOPMENT,
                "protocol_id": PORTABILITY_DEVELOPMENT_ID,
                "protocol_sha256": PORTABILITY_DEVELOPMENT_SHA256,
                "phase_manifest_fingerprint": phase_fingerprint,
                "runtime": runtime,
                "full_sequence_executions": timings,
                "ensemble_metadata": dict(ensemble_metadata),
                "scorer_metadata": dict(scorer_metadata),
                "training_or_refitting_performed": False,
            }
            atomic_create_new_json(output / GPU_LOG_NAME, execution)

            if _snapshot_directory(prior_output) != prior_snapshot:
                raise IntegrityError("immutable v0_1 output changed during portability run")
            result = {
                "schema": "metropt_cared_bc_portability_development_result",
                "schema_version": SCHEMA_VERSION,
                "phase": PHASE_DEVELOPMENT,
                "protocol_id": PORTABILITY_DEVELOPMENT_ID,
                "protocol_sha256": PORTABILITY_DEVELOPMENT_SHA256,
                "phase_manifest_fingerprint": phase_fingerprint,
                "source_fingerprint": phase_body["source_fingerprint"],
                "decision": decision,
                "development_gate_passed": overall_passed,
                "fixed_artifacts_byte_identical": {
                    ENSEMBLE_NAME: artifact_record(output / ENSEMBLE_NAME),
                    SCORER_NAME: artifact_record(output / SCORER_NAME),
                },
                "prior_v0_1_directory_unchanged": True,
                "numeric_partitions_loaded": list(LOAD_PARTITIONS),
                "fit_partition_loaded": False,
                "portability_audit": audit_document,
                "next_method_if_failed": "MB-CycleMIL-BC" if not overall_passed else None,
                "further_cared_bc_repair_permitted_if_failed": False,
                "calibration_numeric_access_authorized_by_this_result": False,
                "confirmation_numeric_access_authorized_by_this_result": False,
                "next_phase_status": (
                    "FINAL_ANALYSIS_PROTOCOL_REQUIRED_BEFORE_CALIBRATION_ACCESS"
                    if overall_passed
                    else "CALIBRATION_AND_CONFIRMATION_REMAIN_SEALED"
                ),
            }
            atomic_create_new_json(output / RESULT_NAME, result)
        except Exception as error:
            _finalize_abort(output, claim, intent, phase_fingerprint, error)
            raise

        manifest = _create_artifact_manifest(
            output, intent, phase_fingerprint, NORMAL_ARTIFACT_NAMES
        )
        created = create_completion_record(
            output / COMPLETION_NAME,
            claim=claim,
            access_intent=intent,
            artifact_manifest=manifest,
            decision=decision,
        )
        verified = authenticate_completion_record(
            output / COMPLETION_NAME,
            output_directory=output,
            ledger_path=output / LEDGER_NAME,
            expected_ledger_body=build_workspace_access_ledger(**ledger_arguments),
            access_intent_path=output / INTENT_NAME,
            artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
            expected_artifact_names=NORMAL_ARTIFACT_NAMES,
        )
        if verified.fingerprint != created.fingerprint:
            raise IntegrityError("portability completion failed post-create authentication")
        return PortabilityRunOutcome(decision, verified, result, False)
    finally:
        if _snapshot_directory(prior_output) != prior_snapshot:
            raise IntegrityError("immutable v0_1 output changed during this invocation")


def _resolve(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen full-sequence MetroPT CARED-BC portability audit."
    )
    parser.add_argument("csv_path")
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--prior-output-dir", default=DEFAULT_PRIOR_OUTPUT)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--protocol-sidecar", default=DEFAULT_PROTOCOL_SIDECAR_PATH)
    parser.add_argument("--phase", default=PHASE_DEVELOPMENT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--comparison-device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--csv-chunksize", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    workspace = Path(args.workspace_root).resolve()
    config = PortabilityRunConfig(
        csv_path=_resolve(args.csv_path, workspace),
        output_directory=_resolve(args.output_dir, workspace),
        workspace_root=workspace,
        protocol_path=_resolve(args.protocol, workspace),
        protocol_sidecar_path=_resolve(args.protocol_sidecar, workspace),
        prior_output_directory=_resolve(args.prior_output_dir, workspace),
        phase=args.phase,
        device=args.device,
        comparison_device=args.comparison_device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        csv_chunksize=args.csv_chunksize,
    )
    try:
        outcome = run_portability_development(config)
    except (GovernanceError, PortabilityRunnerError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "ERROR",
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
    "FrozenFile",
    "PortabilityHooks",
    "PortabilityRunConfig",
    "PortabilityRunOutcome",
    "PriorBundleExpectation",
    "default_portability_hooks",
    "run_portability_development",
]
