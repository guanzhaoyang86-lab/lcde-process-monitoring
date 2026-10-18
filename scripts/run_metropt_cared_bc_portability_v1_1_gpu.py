#!/usr/bin/env python
"""Governance-repaired, parameter-identical CARED-BC v1.1 portability replay.

The public entry point has no injectable gate, scorer, or model loader. It
records a CUDA preflight before the access intent and numeric loading, then
makes its decision only after pickle-free evidence is reloaded and recomputed.
"""

from __future__ import annotations

# Torch must precede NumPy on the supported Windows CUDA environment.
import torch

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src"
for _import_root in (WORKSPACE_ROOT, SOURCE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import scripts.run_metropt_cared_bc_portability_gpu as legacy_v1  # noqa: E402
from mate_spc_metropt.data import (  # noqa: E402
    ANALOG_COLUMNS, APPROVED_NUMERIC_COLUMNS, SAFE_CONTROL_COLUMNS,
    MetroPTTimestampAudit, load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (  # noqa: E402
    DEVELOPMENT_ABORT, DEVELOPMENT_PASS, PHASE_DEVELOPMENT,
    PORTABILITY_DEVELOPMENT_SHA256, PORTABILITY_V1_1_DEVELOPMENT_ID,
    PORTABILITY_V1_1_DEVELOPMENT_SHA256, AccessDeniedError,
    AuthenticatedDocument, IntegrityError, LedgerClaim, artifact_record,
    atomic_create_new_json, authenticate_completion_record,
    authenticate_fingerprinted_json, authenticate_json_with_sidecar,
    build_workspace_access_ledger, canonical_fingerprint,
    claim_workspace_access_ledger, create_artifact_manifest,
    create_completion_record, create_fingerprinted_json,
    create_phase_access_intent, read_strict_json, sha256_file,
)
from mate_spc_metropt.model import (  # noqa: E402
    AlignedResidualBatch, CaredBCScorer, CaredEnsemble, InferenceWindowBatch,
    align_inference_forecasts_to_score_times, build_causal_input_windows,
)
from mate_spc_metropt.portability import (  # noqa: E402
    CertifiedDevelopmentPortabilityAudit, certify_development_portability,
)
from mate_spc_metropt.portability_replay import (  # noqa: E402
    DEVICES, F1_ORIGIN, SPLITS, ReplayedPortabilityEvidence, alarm_evidence,
    build_evidence_payload, portability_path, replay_portability_evidence,
    save_evidence_npz,
)

SCHEMA_VERSION = 1
DEFAULT_PROTOCOL_PATH = "protocols/metropt_cared_bc_v1_1_portability_development.json"
DEFAULT_PROTOCOL_SIDECAR_PATH = "protocols/metropt_cared_bc_v1_1_portability_development.sha256"
DEFAULT_V1_PROTOCOL_PATH = "protocols/metropt_cared_bc_v1_portability_development.json"
DEFAULT_V1_PROTOCOL_SIDECAR_PATH = "protocols/metropt_cared_bc_v1_portability_development.sha256"
DEFAULT_V0_1_OUTPUT = "outputs/metropt_cared_bc_gpu_v0_1"
DEFAULT_V1_OUTPUT = "outputs/metropt_cared_bc_gpu_v1_portability"
DEFAULT_OUTPUT = "outputs/metropt_cared_bc_gpu_v1_1_portability"

PHASE_MANIFEST_NAME = "development_phase_manifest.json"
LEDGER_NAME = "development_access_ledger.json"
PREFLIGHT_NAME = "runtime_preflight.json"
INTENT_NAME = "development_access_intent.json"
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
    PHASE_MANIFEST_NAME, PREFLIGHT_NAME, ENSEMBLE_NAME, SCORER_NAME,
    EVIDENCE_NAME, AUDIT_NAME, RESULT_NAME, GPU_LOG_NAME,
)
LOAD_PARTITIONS = SPLITS

EXPECTED_V1_DIRECTORY_FINGERPRINT = "2f6b7e0244e330fb97eccec469c9916dc7c17f32a0624b051fd9ff64fb293839"
EXPECTED_V1_PROJECTION_SEMANTIC_SHA256 = "59597b715973599186767ca1a68e7a47728fb1ddd4216be8d6d570379297289c"
EXPECTED_CORE_CANONICAL_SHA256 = "7ff32dc008fcdaffd4317fe2988082f10e62de2b5281bfb0020f95132d192b52"
EXPECTED_EXACT_CANONICAL_SHA256 = "a8dd75575b8fc4e8830a780fa34eb836e065209d75cc575e1bebd0d81c5563db"
EXPECTED_ALARM_CANONICAL_SHA256 = "e35bcae4b78cefe2e5c1c51669466e424afad40e484fd77ba701ce4e6129b9d8"
EXPECTED_H_RAW = 36.02052930938596
EXPECTED_GLOBAL_DELTA = 0.21800327523659302
EXPECTED_H_EFF = 36.23853258562255
EXPECTED_F1_AVAILABILITY = "2020-04-18T00:49:00.000000000"
EXPECTED_F1_DELAY_MINUTES = 49.0

V1_FILE_BINDINGS: Mapping[str, tuple[int, str]] = {
    PHASE_MANIFEST_NAME: (5805, "aa651028d69e0189816822c977d118eb6e3d1d84051a7359619a27a51e427896"),
    LEDGER_NAME: (1258, "eeb793ef8b2a7e2783f4d121dbea97929a17e6f5fe89a030cffb94393eca3c99"),
    INTENT_NAME: (1175, "223bd93465af30666ddbccb8c3c5a2160411e484a708adcded64a324445e1f8e"),
    ARTIFACT_MANIFEST_NAME: (1808, "6b98662a4fbbf0949c7a87cf28b081c3bea9778326385a55f1519afcc4bffb1d"),
    COMPLETION_NAME: (1433, "6e1f01f6965b99fc3be956971cc7e63acbbe506ade156e84e8b947c552f04c35"),
    RESULT_NAME: (8028, "934ef9d83e3819aada2c0c63776d0da6bb945cd410ed6e920d4ee2c82f65fd48"),
    AUDIT_NAME: (6221, "d8e36e32db09db306ae098c5b80c03ed6b78cfb392c2bf7302bd2f481f328416"),
    GPU_LOG_NAME: (2994, "c7504489bdfa850eea2eea66846988087479f4ed390cd90070a54c9cb682d7fe"),
    EVIDENCE_NAME: (4461790, "2d99f2c370b6f97c1c841f0d345f292ca85d74a399247846f984d98e5c70b2b5"),
    ENSEMBLE_NAME: (2376101, "4e75c7718482e66f0fd535a5bd0bf4c0dd78ec3335c3f420797d728c8e9d970f"),
    SCORER_NAME: (181175, "a7da9f5980778f0f65d51e7ab64e1cee3a9604cb186025d1240c45623d7d1887"),
}
V1_FINGERPRINT_BINDINGS = {
    PHASE_MANIFEST_NAME: ("manifest_fingerprint", "ee5299de7b290d71a6cba85b812fa1b6bc1e07630aad27cb000357469c9db7a5"),
    LEDGER_NAME: ("ledger_fingerprint", "6137cc0bb6618fa896387f348de5ee250f92065754c1e715d89b551aa98e7927"),
    INTENT_NAME: ("intent_fingerprint", "f69d65e6cd73b8c8a4e53760a17a275713769bf9603de06aa5b8cd036d18d8f4"),
    ARTIFACT_MANIFEST_NAME: ("artifact_manifest_fingerprint", "e724b89cfc0e9d5a2eddd339f7877312616d4417b00843604cf73c62984b2c5f"),
    COMPLETION_NAME: ("completion_fingerprint", "c3a25fd985861df50206b86df2c0c5c772e9dc5a04126b7fe6928fbde1323396"),
}


class PortabilityV11RunnerError(RuntimeError):
    """Fail-closed v1.1 runner error."""


@dataclass(frozen=True)
class PortabilityV11RunConfig:
    csv_path: Path
    output_directory: Path = WORKSPACE_ROOT / DEFAULT_OUTPUT
    workspace_root: Path = WORKSPACE_ROOT
    protocol_path: Path = WORKSPACE_ROOT / DEFAULT_PROTOCOL_PATH
    protocol_sidecar_path: Path = WORKSPACE_ROOT / DEFAULT_PROTOCOL_SIDECAR_PATH
    expected_protocol_sha256: str = PORTABILITY_V1_1_DEVELOPMENT_SHA256
    v0_1_output_directory: Path = WORKSPACE_ROOT / DEFAULT_V0_1_OUTPUT
    v1_output_directory: Path = WORKSPACE_ROOT / DEFAULT_V1_OUTPUT
    v1_protocol_path: Path = WORKSPACE_ROOT / DEFAULT_V1_PROTOCOL_PATH
    v1_protocol_sidecar_path: Path = WORKSPACE_ROOT / DEFAULT_V1_PROTOCOL_SIDECAR_PATH
    phase: str = PHASE_DEVELOPMENT
    device: str = "cuda"
    comparison_device: str = "cpu"
    dtype: str = "float64"
    batch_size: int = 1024
    csv_chunksize: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "csv_path", "output_directory", "workspace_root", "protocol_path",
            "protocol_sidecar_path", "v0_1_output_directory", "v1_output_directory",
            "v1_protocol_path", "v1_protocol_sidecar_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True)
class PortabilityV11Outcome:
    decision: str
    completion: AuthenticatedDocument
    result: Mapping[str, Any]
    reused_completed_bundle: bool


@dataclass(frozen=True)
class _ParentEvidence:
    timestamp_audit: MetroPTTimestampAudit
    sources: Mapping[str, Mapping[str, Any]]
    source_fingerprint: str
    sensor_csv_relative_path: str
    v0_1_snapshot: Mapping[str, Mapping[str, Any]]
    v0_1_completion_fingerprint: str
    v1_snapshot: Mapping[str, Mapping[str, Any]]
    v1_completion_fingerprint: str


@dataclass(frozen=True)
class _NumericalBoundary:
    """Private test seam; no score, model-loader, or gate callback exists."""
    load_partition: Callable[..., Any]
    build_inference_windows: Callable[..., Any]
    predict: Callable[..., Any]
    align: Callable[..., Any]


def _default_predict(ensemble: CaredEnsemble, inputs: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    return ensemble.predict(inputs, device=device, batch_size=batch_size)


def _default_boundary() -> _NumericalBoundary:
    return _NumericalBoundary(
        load_governed_metropt_partition, build_causal_input_windows,
        _default_predict, align_inference_forecasts_to_score_times,
    )


def _absolute(path: Path, root: Path) -> Path:
    return Path(path if path.is_absolute() else root / path).absolute()


def _workspace_relative(path: Path, workspace: Path) -> str:
    root = workspace.resolve(strict=True)
    candidate = path.resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PortabilityV11RunnerError("path must remain inside workspace_root") from exc
    if not relative.parts:
        raise PortabilityV11RunnerError("path must not equal workspace_root")
    return PurePosixPath(*relative.parts).as_posix()


def _nested(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise IntegrityError(f"invalid or missing mapping: {name}")
    return result


def _validate_config(config: PortabilityV11RunConfig) -> None:
    if config.phase != PHASE_DEVELOPMENT:
        raise AccessDeniedError("v1.1 portability is development-only")
    if config.device.lower() != "cuda" or config.comparison_device.lower() != "cpu":
        raise PortabilityV11RunnerError("frozen devices are CUDA and CPU")
    if config.dtype != "float64" or config.batch_size != 1024:
        raise PortabilityV11RunnerError("frozen dtype/batch are float64/1024")
    if isinstance(config.csv_chunksize, bool) or config.csv_chunksize <= 0:
        raise PortabilityV11RunnerError("csv_chunksize must be positive")
    if config.expected_protocol_sha256 != PORTABILITY_V1_1_DEVELOPMENT_SHA256:
        raise IntegrityError("configured protocol SHA differs from compiled v1.1 SHA")


def _implementation_snapshot() -> dict[str, Mapping[str, Any]]:
    paths = {
        "runner": Path(__file__).resolve(strict=True),
        "model": SOURCE_ROOT / "mate_spc_metropt" / "model.py",
        "portability": SOURCE_ROOT / "mate_spc_metropt" / "portability.py",
        "portability_replay": SOURCE_ROOT / "mate_spc_metropt" / "portability_replay.py",
        "data": SOURCE_ROOT / "mate_spc_metropt" / "data.py",
        "governance": SOURCE_ROOT / "mate_spc_metropt" / "governance.py",
    }
    return {name: artifact_record(path) for name, path in sorted(paths.items())}


def _validate_protocol(protocol: AuthenticatedDocument, config: PortabilityV11RunConfig) -> None:
    value = protocol.value
    identity = {
        "schema": "metropt_cared_bc_portability_governance_repair_protocol",
        "schema_version": 1,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_version": "1.1.0",
        "status": "FROZEN_BEFORE_V1_1_NUMERIC_REPLAY",
    }
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise IntegrityError("v1.1 protocol identity changed")
    if protocol.record.get("sha256") != PORTABILITY_V1_1_DEVELOPMENT_SHA256:
        raise IntegrityError("v1.1 protocol byte identity changed")
    disclosure = _nested(value, "disclosure")
    for key in (
        "created_after_observing_v1_pass",
        "not_an_original_v0_v0_1_or_v1_preregistered_pass",
        "v1_completion_is_governance_invalid",
        "sealed_partitions_remain_closed",
        "v0_v0_1_and_v1_outputs_must_remain_byte_identical",
    ):
        if disclosure.get(key) is not True:
            raise IntegrityError(f"v1.1 disclosure changed: {key}")
    numeric = _nested(value, "numeric_access")
    if (
        numeric.get("exact_authorized_partitions") != list(LOAD_PARTITIONS)
        or numeric.get("fit_permitted") is not False
        or numeric.get("selection_permitted") is not False
        or numeric.get("single_access") is not True
        or numeric.get("reusable") is not False
        or numeric.get("maximum_loads_per_authorized_partition")
        != {name: 1 for name in LOAD_PARTITIONS}
    ):
        raise IntegrityError("v1.1 exact numeric authorization changed")
    runtime = _nested(value, "runtime_preflight")
    if (
        runtime.get("must_complete_after_ledger_before_intent_and_numeric_loader") is not True
        or runtime.get("primary_device") != "cuda"
        or runtime.get("comparison_device") != "cpu"
        or runtime.get("dtype") != "float64"
        or runtime.get("batch_size") != 1024
    ):
        raise IntegrityError("v1.1 runtime preflight contract changed")
    trusted = _nested(value, "trusted_audit")
    if (
        trusted.get("function") != "mate_spc_metropt.portability.certify_development_portability"
        or trusted.get("concrete_type") != "CertifiedDevelopmentPortabilityAudit"
        or trusted.get("injectable_gate_hook_forbidden") is not True
        or trusted.get("completed_reentry_must_recompute_scorer_stages_from_fixed_scorer") is not True
    ):
        raise IntegrityError("trusted audit contract changed")
    output = _nested(value, "output_contract")
    if (
        output.get("directory") != _workspace_relative(config.output_directory, config.workspace_root)
        or output.get("create_new_only") is not True
        or output.get("partial_bundle_fails_closed_without_numeric_reentry") is not True
    ):
        raise IntegrityError("v1.1 output contract changed")
    targets = _nested(value, "reproduction_targets_from_v1")
    if _nested(targets, "canonical_sha256") != {
        "v1_35_array_projection": EXPECTED_V1_PROJECTION_SEMANTIC_SHA256,
        "core_portability_audit": EXPECTED_CORE_CANONICAL_SHA256,
        "exact_backend_checks": EXPECTED_EXACT_CANONICAL_SHA256,
        "alarm_array_hashes": EXPECTED_ALARM_CANONICAL_SHA256,
    }:
        raise IntegrityError("v1 reproduction canonical targets changed")
    if _nested(targets, "threshold") != {
        "h_raw": EXPECTED_H_RAW,
        "global_envelope_delta": EXPECTED_GLOBAL_DELTA,
        "h_eff": EXPECTED_H_EFF,
    }:
        raise IntegrityError("v1 reproduction threshold targets changed")
    bindings = _nested(value, "scientific_implementation_bindings")
    snapshot = _implementation_snapshot()
    for name in ("model", "portability", "portability_replay"):
        expected = _nested(bindings, name)
        if expected.get("size_bytes") != snapshot[name]["size_bytes"] or expected.get("sha256") != snapshot[name]["sha256"]:
            raise IntegrityError(f"scientific implementation changed: {name}")


def _snapshot_directory(directory: Path) -> dict[str, dict[str, Any]]:
    return legacy_v1._snapshot_directory(directory)


def _authenticate_parents(config: PortabilityV11RunConfig) -> _ParentEvidence:
    v0_snapshot = _snapshot_directory(config.v0_1_output_directory)
    legacy_config = legacy_v1.PortabilityRunConfig(
        csv_path=config.csv_path,
        output_directory=config.v1_output_directory,
        workspace_root=config.workspace_root,
        protocol_path=config.v1_protocol_path,
        protocol_sidecar_path=config.v1_protocol_sidecar_path,
        expected_protocol_sha256=PORTABILITY_DEVELOPMENT_SHA256,
        prior_output_directory=config.v0_1_output_directory,
    )
    v1_protocol = authenticate_json_with_sidecar(
        config.v1_protocol_path, config.v1_protocol_sidecar_path,
        expected_sha256=PORTABILITY_DEVELOPMENT_SHA256,
    )
    legacy_v1._validate_protocol(v1_protocol, legacy_config)
    prior = legacy_v1._authenticate_prior_bundle(legacy_config, v0_snapshot)
    v1_snapshot = _snapshot_directory(config.v1_output_directory)
    outcome = legacy_v1._authenticate_completed(legacy_config, v1_protocol, prior)
    if outcome.decision != DEVELOPMENT_PASS:
        raise IntegrityError("frozen v1 parent is not the observed invalid PASS")
    for name, (size, digest) in V1_FILE_BINDINGS.items():
        record = artifact_record(config.v1_output_directory / name)
        if record.get("size_bytes") != size or record.get("sha256") != digest:
            raise IntegrityError(f"frozen v1 file changed: {name}")
    for name, (field, fingerprint) in V1_FINGERPRINT_BINDINGS.items():
        document = authenticate_fingerprinted_json(
            config.v1_output_directory / name, fingerprint_field=field
        )
        if document.fingerprint != fingerprint:
            raise IntegrityError(f"frozen v1 fingerprint changed: {name}")
    if canonical_fingerprint(v1_snapshot) != EXPECTED_V1_DIRECTORY_FINGERPRINT:
        raise IntegrityError("frozen v1 directory snapshot changed")
    sources = _nested(prior.phase_manifest.value, "sources")
    return _ParentEvidence(
        timestamp_audit=prior.timestamp_audit,
        sources={key: dict(item) for key, item in sources.items()},
        source_fingerprint=str(prior.phase_manifest.value["source_fingerprint"]),
        sensor_csv_relative_path=str(prior.phase_manifest.value["sensor_csv_relative_path"]),
        v0_1_snapshot=v0_snapshot,
        v0_1_completion_fingerprint=str(prior.completion.fingerprint),
        v1_snapshot=v1_snapshot,
        v1_completion_fingerprint=str(outcome.completion.fingerprint),
    )


def _build_phase_manifest(config: PortabilityV11RunConfig, protocol: AuthenticatedDocument, parents: _ParentEvidence) -> dict[str, Any]:
    output_relative = _workspace_relative(config.output_directory, config.workspace_root)
    counts = _nested(_nested(protocol.value, "numeric_access"), "expected_counts")
    return {
        "schema": "metropt_cared_bc_v1_1_portability_phase_manifest",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "protocol_record": dict(protocol.record),
        "invalid_v1_parent": {
            "output_directory": _workspace_relative(config.v1_output_directory, config.workspace_root),
            "completion_fingerprint": parents.v1_completion_fingerprint,
            "directory_snapshot_fingerprint": canonical_fingerprint(parents.v1_snapshot),
            "governance_valid": False,
            "numeric_targets_only": True,
        },
        "v0_1_abort_parent": {
            "output_directory": _workspace_relative(config.v0_1_output_directory, config.workspace_root),
            "completion_fingerprint": parents.v0_1_completion_fingerprint,
            "directory_snapshot_fingerprint": canonical_fingerprint(parents.v0_1_snapshot),
        },
        "source_fingerprint": parents.source_fingerprint,
        "sources": {key: dict(item) for key, item in parents.sources.items()},
        "sensor_csv_relative_path": parents.sensor_csv_relative_path,
        "output_directory_relative_path": output_relative,
        "ledger_workspace_relative_path": (PurePosixPath(output_relative) / LEDGER_NAME).as_posix(),
        "numeric_access": {
            "exact_authorized_partitions": list(LOAD_PARTITIONS),
            "expected_counts": {name: dict(counts[name]) for name in LOAD_PARTITIONS},
            "fit_permitted": False, "selection_permitted": False,
            "single_access": True, "reusable": False,
        },
        "runtime": {
            "primary_device": config.device, "comparison_device": config.comparison_device,
            "dtype": config.dtype, "batch_size": config.batch_size,
            "csv_chunksize": config.csv_chunksize,
            "preflight_before_intent_and_numeric_loader": True,
        },
        "implementation_snapshot": _implementation_snapshot(),
        "output_artifacts": list(NORMAL_ARTIFACT_NAMES),
        "training_refitting_selection_or_gate_change_performed": False,
        "calibration_numeric_access_authorized": False,
        "confirmation_numeric_access_authorized": False,
    }


def _ledger_arguments(config: PortabilityV11RunConfig, phase: Mapping[str, Any], phase_fingerprint: str) -> dict[str, Any]:
    sensor = _nested(_nested(phase, "sources"), "sensor_csv")
    return {
        "phase": PHASE_DEVELOPMENT,
        "workspace_relative_path": str(phase["ledger_workspace_relative_path"]),
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "charter_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "final_protocol_sha256": None,
        "sensor_csv_sha256": str(sensor["sha256"]),
        "source_fingerprint": str(phase["source_fingerprint"]),
        "phase_manifest_fingerprint": phase_fingerprint,
        "output_directory": config.output_directory,
    }


def _ensure_fresh_output(output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise IntegrityError("output must be a regular directory")
        if any(output.iterdir()):
            raise PortabilityV11RunnerError("partial v1.1 output exists; fail closed without numeric reentry")
        return
    output.mkdir(parents=True, exist_ok=False)


def _runtime_probe(config: PortabilityV11RunConfig) -> dict[str, Any]:
    requested = torch.device(config.device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise PortabilityV11RunnerError("v1.1 requires an available CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = torch.cuda.current_device() if requested.index is None else requested.index
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=requested)
    smoke = float(torch.sum(tensor * tensor).item())
    torch.cuda.synchronize(index)
    if smoke != 14.0:
        raise PortabilityV11RunnerError("CUDA float64 smoke kernel returned a wrong value")
    capability = torch.cuda.get_device_capability(index)
    return {
        "backend": "torch",
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "numpy_version": str(np.__version__), "pandas_version": str(pd.__version__),
        "torch_version": str(torch.__version__), "cuda_runtime_version": str(torch.version.cuda),
        "device_index": int(index), "device_name": str(torch.cuda.get_device_name(index)),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "primary_device": str(requested), "comparison_device": config.comparison_device,
        "dtype": config.dtype, "batch_size": config.batch_size,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_float64_smoke_value": smoke,
    }


def _preflight_body(config: PortabilityV11RunConfig, phase_fingerprint: str, protocol: Mapping[str, Any], probe: Callable[[PortabilityV11RunConfig], Mapping[str, Any]]) -> dict[str, Any]:
    try:
        runtime = dict(probe(config))
        expected = _nested(_nested(protocol, "runtime_preflight"), "expected_runtime")
        checks = {
            "torch_cuda_available": bool(torch.cuda.is_available()) if probe is _runtime_probe else bool(runtime.get("torch_cuda_available", True)),
            "primary_device": runtime.get("primary_device") == "cuda",
            "comparison_device": runtime.get("comparison_device") == "cpu",
            "dtype": runtime.get("dtype") == "float64",
            "batch_size": runtime.get("batch_size") == 1024,
            "deterministic_algorithms": runtime.get("deterministic_algorithms") is True,
            "cuda_matmul_tf32_disabled": runtime.get("cuda_matmul_tf32") is False,
            "cudnn_tf32_disabled": runtime.get("cudnn_tf32") is False,
            "cuda_float64_smoke": runtime.get("cuda_float64_smoke_value") == 14.0,
            "runtime_versions_and_device_exact": all(runtime.get(key) == expected.get(key) for key in expected),
        }
        passed = bool(all(checks.values()))
        error = None
    except Exception as exc:
        runtime, checks, passed = {}, {"probe_completed": False}, False
        error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "schema": "metropt_cared_bc_v1_1_runtime_preflight", "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT, "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "numeric_access_occurred_before_preflight": False,
        "access_intent_existed_before_preflight": False,
        "runtime": runtime, "checks": checks, "passed": passed, "error": error,
    }


def _copy_frozen(source: Path, destination: Path, name: str) -> None:
    size, digest = V1_FILE_BINDINGS[name]
    record = artifact_record(source)
    if record.get("size_bytes") != size or record.get("sha256") != digest:
        raise IntegrityError(f"fixed parent artifact changed: {name}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    shutil.copyfile(source, destination)
    copied = artifact_record(destination)
    if copied.get("size_bytes") != size or copied.get("sha256") != digest:
        raise IntegrityError(f"fixed artifact copy changed: {name}")


def _validate_fixed_models(ensemble: CaredEnsemble, ensemble_metadata: Mapping[str, Any], scorer: CaredBCScorer, scorer_metadata: Mapping[str, Any]) -> None:
    common = {
        "charter_sha256": legacy_v1.DEVELOPMENT_CHARTER_SHA256,
        "development_only": True, "phase": PHASE_DEVELOPMENT,
        "protocol_id": legacy_v1.CHARTER_ID,
        "source_fingerprint": "340805938a72af9c1f9273a3c75b90a246b11b6aea7bc254c6d0e7103602d705",
        "phase_manifest_fingerprint": "d26b87f46f7125fdb90ea8a07622e1438833171aadd3a7f4f0cd2a14f3970943",
    }
    if ensemble.dtype != "float64" or any(ensemble_metadata.get(k) != v for k, v in common.items()):
        raise IntegrityError("fixed ensemble semantic metadata changed")
    if scorer.threshold is not None or any(scorer_metadata.get(k) != v for k, v in common.items()):
        raise IntegrityError("fixed scorer semantic metadata changed")
    if scorer_metadata.get("threshold_status") != "uncalibrated" or scorer_metadata.get("provisional_threshold_saved_in_result_only") is not True:
        raise IntegrityError("fixed scorer is no longer uncalibrated")


def _validate_execution_record(
    execution: Mapping[str, Any],
    *,
    phase_fingerprint: str,
    preflight: AuthenticatedDocument,
    counts: Mapping[str, Any],
    ensemble_metadata: Mapping[str, Any],
    scorer_metadata: Mapping[str, Any],
) -> None:
    """Reconstruct every stable execution-record field on completed reentry."""
    expected_keys = {
        "schema", "schema_version", "phase", "protocol_id", "protocol_sha256",
        "phase_manifest_fingerprint", "runtime_preflight_fingerprint", "runtime",
        "full_sequence_executions", "ensemble_metadata", "scorer_metadata",
        "training_refitting_selection_or_gate_change_performed",
    }
    if set(execution) != expected_keys:
        raise IntegrityError("completed v1.1 execution record schema changed")
    expected_identity = {
        "schema": "metropt_cared_bc_v1_1_gpu_execution",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "runtime_preflight_fingerprint": preflight.fingerprint,
        "runtime": preflight.value.get("runtime"),
        "ensemble_metadata": dict(ensemble_metadata),
        "scorer_metadata": dict(scorer_metadata),
        "training_refitting_selection_or_gate_change_performed": False,
    }
    if any(execution.get(key) != value for key, value in expected_identity.items()):
        raise IntegrityError("completed v1.1 execution record identity/runtime changed")

    rows = execution.get("full_sequence_executions")
    expected_rows = [
        (split, device_name, device)
        for split in SPLITS
        for device_name, device in (("cpu", "cpu"), ("cuda", "cuda"))
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise IntegrityError("completed v1.1 execution sequence changed")
    row_keys = {
        "partition", "device", "prediction_seconds",
        "inference_window_count", "aligned_score_count",
    }
    for row, (split, _device_name, device) in zip(rows, expected_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise IntegrityError("completed v1.1 execution row schema changed")
        duration = row.get("prediction_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
        ):
            raise IntegrityError("completed v1.1 execution timing is invalid")
        split_counts = _nested(counts, split)
        if (
            row.get("partition") != split
            or row.get("device") != device
            or row.get("inference_window_count") != split_counts.get("inference_windows")
            or row.get("aligned_score_count") != split_counts.get("aligned_scores")
        ):
            raise IntegrityError("completed v1.1 execution row semantics changed")


def _validate_interval(name: str, interval: Any, sensor_sha: str) -> None:
    timestamps = np.asarray(interval.timestamps, dtype="datetime64[ns]")
    analog = np.asarray(interval.analog, dtype=np.float64)
    controls = np.asarray(interval.controls, dtype=np.float64)
    eligible, reset = np.asarray(interval.eligible), np.asarray(interval.reset_before)
    if (
        timestamps.ndim != 1 or timestamps.size == 0
        or analog.shape != (timestamps.size, len(ANALOG_COLUMNS))
        or controls.shape != (timestamps.size, len(SAFE_CONTROL_COLUMNS))
        or eligible.shape != (timestamps.size,) or eligible.dtype.kind != "b"
        or reset.shape != (timestamps.size,) or reset.dtype.kind != "b"
        or not bool(reset[0])
    ):
        raise IntegrityError(f"{name} interval arrays changed")
    metadata = interval.access_metadata
    if (
        metadata.get("source_sha256") != sensor_sha
        or metadata.get("unrequested_numeric_values_parsed") is not False
        or metadata.get("direct_alarm_values_parsed") is not False
        or metadata.get("numeric_columns_parsed") != list(APPROVED_NUMERIC_COLUMNS)
    ):
        raise IntegrityError(f"{name} governed access metadata changed")


def _load_intervals(config: PortabilityV11RunConfig, boundary: _NumericalBoundary, parents: _ParentEvidence, claim: LedgerClaim, intent: AuthenticatedDocument, phase_fingerprint: str) -> dict[str, Any]:
    sensor = _nested(parents.sources, "sensor_csv")
    intervals: dict[str, Any] = {}
    for name in LOAD_PARTITIONS:
        if name in intervals:
            raise AccessDeniedError(f"duplicate v1.1 partition request: {name}")
        interval = boundary.load_partition(
            config.csv_path, name, claim=claim, access_intent=intent,
            access_intent_path=config.output_directory / INTENT_NAME,
            phase_manifest_fingerprint=phase_fingerprint,
            chunksize=config.csv_chunksize, timestamp_audit=parents.timestamp_audit,
            runtime_preflight_path=config.output_directory / PREFLIGHT_NAME,
        )
        _validate_interval(name, interval, str(sensor["sha256"]))
        intervals[name] = interval
    if tuple(intervals) != tuple(LOAD_PARTITIONS):
        raise AccessDeniedError("v1.1 loader crossed exact partition scope")
    return intervals


def _validate_aligned(name: str, batch: Any, count: int) -> AlignedResidualBatch:
    if not isinstance(batch, AlignedResidualBatch) or batch.residuals.shape != (count, 31):
        raise IntegrityError(f"{name} aligned count/type changed")
    return batch


def _target_checks(replay: ReplayedPortabilityEvidence) -> dict[str, bool]:
    core = replay.certificate.as_dict()
    threshold = _nested(core, "threshold")
    f1_devices = _nested(_nested(_nested(core, "splits"), "dev_fault_f1"), "devices")
    reported = {
        "dev_healthy": {
            "energy": 2.3163693185779266e-11,
            "directional_glr": 3.6193270602780103e-13,
            "standardized_evidence": 0.1712539623305891,
            "cusum": 0.2184869636321487,
            "envelope": 0.21800327523659302,
        },
        "dev_fault_f1": {
            "energy": 1.724401954561472e-09,
            "directional_glr": 6.252776074688882e-12,
            "standardized_evidence": 0.0, "cusum": 0.0, "envelope": 0.0,
        },
    }
    return {
        "trusted_certificate_passed": replay.certificate.passed,
        "exact_backend_checks_passed": bool(replay.exact_backend_checks["passed"]),
        "v1_projection_semantic_sha256_exact": replay.v1_projection_semantic_sha256 == EXPECTED_V1_PROJECTION_SEMANTIC_SHA256,
        "core_portability_audit_canonical_sha256_exact": canonical_fingerprint(core) == EXPECTED_CORE_CANONICAL_SHA256,
        "exact_backend_checks_canonical_sha256_exact": canonical_fingerprint(replay.exact_backend_checks) == EXPECTED_EXACT_CANONICAL_SHA256,
        "alarm_array_hashes_canonical_sha256_exact": canonical_fingerprint(replay.alarm_array_hashes) == EXPECTED_ALARM_CANONICAL_SHA256,
        "h_raw_exact": float(threshold["h_raw"]) == EXPECTED_H_RAW,
        "global_envelope_delta_exact": float(threshold["global_envelope_delta"]) == EXPECTED_GLOBAL_DELTA,
        "h_eff_exact": float(threshold["h_eff"]) == EXPECTED_H_EFF,
        "f1_first_alarm_availability_exact": all(
            _nested(f1_devices, device).get("first_alarm_availability") == EXPECTED_F1_AVAILABILITY
            for device in DEVICES
        ),
        "f1_first_alarm_delay_exact": all(
            _nested(f1_devices, device).get("first_alarm_delay_minutes") == EXPECTED_F1_DELAY_MINUTES
            for device in DEVICES
        ),
        "reported_scorer_stage_differences_ieee_exact": all(
            replay.full_sequence_max_abs_differences[split][field] == expected
            for split, fields in reported.items() for field, expected in fields.items()
        ),
    }


def _audit_body(phase_fingerprint: str, replay: ReplayedPortabilityEvidence) -> dict[str, Any]:
    checks = _target_checks(replay)
    passed = bool(all(checks.values()))
    return {
        "schema": "metropt_cared_bc_v1_1_full_portability_adjudication",
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "trusted_certificate_type": type(replay.certificate).__name__,
        "core_portability_audit": replay.certificate.as_dict(),
        "exact_backend_checks": dict(replay.exact_backend_checks),
        "alarm_array_hashes": dict(replay.alarm_array_hashes),
        "full_sequence_max_abs_differences": dict(replay.full_sequence_max_abs_differences),
        "evidence_semantic_sha256": replay.semantic_sha256,
        "v1_projection_semantic_sha256": replay.v1_projection_semantic_sha256,
        "target_checks": checks,
        "passed": passed,
        "decision": DEVELOPMENT_PASS if passed else DEVELOPMENT_ABORT,
    }


def _result_body(output: Path, phase_fingerprint: str, source_fingerprint: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    passed, decision = bool(audit["passed"]), str(audit["decision"])
    return {
        "schema": "metropt_cared_bc_v1_1_portability_development_result",
        "schema_version": SCHEMA_VERSION, "phase": PHASE_DEVELOPMENT,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "source_fingerprint": source_fingerprint,
        "decision": decision, "development_gate_passed": passed,
        "numeric_partitions_loaded_exactly_once": list(LOAD_PARTITIONS),
        "fit_partition_loaded": False,
        "training_refitting_selection_or_gate_change_performed": False,
        "fixed_artifacts_byte_identical": {
            ENSEMBLE_NAME: artifact_record(output / ENSEMBLE_NAME),
            SCORER_NAME: artifact_record(output / SCORER_NAME),
        },
        "portability_audit_sha256": sha256_file(output / AUDIT_NAME),
        "evidence_semantic_sha256": audit["evidence_semantic_sha256"],
        "v1_projection_semantic_sha256": audit["v1_projection_semantic_sha256"],
        "next_method_if_failed": "MB-CycleMIL-BC" if not passed else None,
        "further_cared_bc_repair_permitted_if_failed": False,
        "calibration_numeric_access_authorized_by_this_result": False,
        "confirmation_numeric_access_authorized_by_this_result": False,
        "next_phase_status": "FINAL_ANALYSIS_PROTOCOL_REQUIRED_BEFORE_CALIBRATION_ACCESS" if passed else "CALIBRATION_AND_CONFIRMATION_REMAIN_SEALED",
    }


def _create_manifest(output: Path, intent: AuthenticatedDocument, phase_fingerprint: str, names: tuple[str, ...]) -> AuthenticatedDocument:
    return create_artifact_manifest(
        output / ARTIFACT_MANIFEST_NAME, output,
        phase=PHASE_DEVELOPMENT, protocol_id=PORTABILITY_V1_1_DEVELOPMENT_ID,
        protocol_sha256=PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        phase_manifest_fingerprint=phase_fingerprint,
        access_intent=intent, artifact_names=names,
    )


def _finalize_abort(output: Path, claim: LedgerClaim, intent: AuthenticatedDocument, phase_fingerprint: str, error: Exception | None, *, reason: str) -> PortabilityV11Outcome:
    body = {
        "schema": "metropt_cared_bc_v1_1_runtime_abort", "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT, "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "decision": DEVELOPMENT_ABORT, "reason": reason,
        "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        "calibration_numeric_access_authorized": False,
        "confirmation_numeric_access_authorized": False,
        "next_method": "MB-CycleMIL-BC",
    }
    atomic_create_new_json(output / ABORT_NAME, body)
    candidates = tuple(
        name for name in (*NORMAL_ARTIFACT_NAMES, ABORT_NAME)
        if (output / name).is_file() and not (output / name).is_symlink()
    )
    manifest = _create_manifest(output, intent, phase_fingerprint, candidates)
    created = create_completion_record(
        output / COMPLETION_NAME, claim=claim, access_intent=intent,
        artifact_manifest=manifest, decision=DEVELOPMENT_ABORT,
    )
    ledger_body = dict(claim.document.value)
    ledger_body.pop("ledger_fingerprint", None)
    verified = authenticate_completion_record(
        output / COMPLETION_NAME, output_directory=output,
        ledger_path=output / LEDGER_NAME, expected_ledger_body=ledger_body,
        access_intent_path=output / INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=candidates,
    )
    if verified.fingerprint != created.fingerprint:
        raise IntegrityError("v1.1 ABORT completion failed authentication")
    return PortabilityV11Outcome(DEVELOPMENT_ABORT, verified, body, False)


def _artifact_names_from_manifest(output: Path) -> tuple[str, ...]:
    names = tuple(sorted(str(name) for name in _nested(read_strict_json(output / ARTIFACT_MANIFEST_NAME), "artifacts")))
    normal, observed = set(NORMAL_ARTIFACT_NAMES), set(names)
    if observed != normal:
        if ABORT_NAME not in observed or not observed.issubset(normal | {ABORT_NAME}):
            raise IntegrityError("v1.1 completed artifact set changed")
        if PHASE_MANIFEST_NAME not in observed or PREFLIGHT_NAME not in observed:
            raise IntegrityError("v1.1 ABORT lacks preaccess evidence")
    return names


def _rebuild_abort_body(value: Mapping[str, Any], phase_fingerprint: str) -> dict[str, Any]:
    reason = value.get("reason")
    if reason not in {
        "RUNTIME_PREFLIGHT_FAILED_BEFORE_NUMERIC_ACCESS",
        "POST_INTENT_EXECUTION_FAILED",
        "FINALIZATION_FAILED",
    }:
        raise IntegrityError("completed v1.1 ABORT reason is invalid")
    error = value.get("error")
    if reason == "RUNTIME_PREFLIGHT_FAILED_BEFORE_NUMERIC_ACCESS":
        if error is not None:
            raise IntegrityError("preflight ABORT must not carry a post-intent error")
    elif (
        not isinstance(error, Mapping)
        or set(error) != {"type", "message"}
        or not all(isinstance(error[key], str) for key in error)
    ):
        raise IntegrityError("completed v1.1 ABORT error is malformed")
    return {
        "schema": "metropt_cared_bc_v1_1_runtime_abort",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE_DEVELOPMENT,
        "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
        "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        "phase_manifest_fingerprint": phase_fingerprint,
        "decision": DEVELOPMENT_ABORT,
        "reason": reason,
        "error": None if error is None else dict(error),
        "calibration_numeric_access_authorized": False,
        "confirmation_numeric_access_authorized": False,
        "next_method": "MB-CycleMIL-BC",
    }


def _authenticate_completed(config: PortabilityV11RunConfig, protocol: AuthenticatedDocument, parents: _ParentEvidence) -> PortabilityV11Outcome:
    output = config.output_directory
    phase = authenticate_fingerprinted_json(
        output / PHASE_MANIFEST_NAME, fingerprint_field="manifest_fingerprint"
    )
    expected_phase = _build_phase_manifest(config, protocol, parents)
    observed_phase = dict(phase.value)
    observed_phase.pop("manifest_fingerprint", None)
    if observed_phase != expected_phase:
        raise IntegrityError("completed v1.1 phase manifest changed")
    phase_fingerprint = str(phase.fingerprint)
    preflight = authenticate_fingerprinted_json(
        output / PREFLIGHT_NAME, fingerprint_field="preflight_fingerprint"
    )
    if (
        preflight.value.get("phase_manifest_fingerprint") != phase_fingerprint
        or preflight.value.get("protocol_id") != PORTABILITY_V1_1_DEVELOPMENT_ID
        or preflight.value.get("numeric_access_occurred_before_preflight") is not False
    ):
        raise IntegrityError("completed v1.1 preflight changed")
    ledger_arguments = _ledger_arguments(config, phase.value, phase_fingerprint)
    names = _artifact_names_from_manifest(output)
    completion = authenticate_completion_record(
        output / COMPLETION_NAME, output_directory=output,
        ledger_path=output / LEDGER_NAME,
        expected_ledger_body=build_workspace_access_ledger(**ledger_arguments),
        access_intent_path=output / INTENT_NAME,
        artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
        expected_artifact_names=names,
    )
    decision = str(completion.value["decision"])
    if ABORT_NAME in names:
        abort = read_strict_json(output / ABORT_NAME)
        if decision != DEVELOPMENT_ABORT or abort != _rebuild_abort_body(abort, phase_fingerprint):
            raise IntegrityError("completed v1.1 ABORT semantics changed")
        if abort["reason"] == "RUNTIME_PREFLIGHT_FAILED_BEFORE_NUMERIC_ACCESS":
            if preflight.value.get("passed") is not False or any(
                name in names for name in (ENSEMBLE_NAME, SCORER_NAME, EVIDENCE_NAME)
            ):
                raise IntegrityError("preflight ABORT artifact semantics changed")
        elif preflight.value.get("passed") is not True:
            raise IntegrityError("post-intent ABORT has no passing preflight")
        return PortabilityV11Outcome(decision, completion, abort, True)
    if set(names) != set(NORMAL_ARTIFACT_NAMES) or preflight.value.get("passed") is not True:
        raise IntegrityError("completed v1.1 normal bundle is malformed")
    for fixed_name in (ENSEMBLE_NAME, SCORER_NAME):
        size, digest = V1_FILE_BINDINGS[fixed_name]
        fixed_record = artifact_record(output / fixed_name)
        if fixed_record.get("size_bytes") != size or fixed_record.get("sha256") != digest:
            raise IntegrityError(f"completed fixed artifact bytes changed: {fixed_name}")
    ensemble, ensemble_metadata = CaredEnsemble.load_npz(output / ENSEMBLE_NAME)
    scorer, scorer_metadata = CaredBCScorer.load_npz(output / SCORER_NAME)
    _validate_fixed_models(ensemble, ensemble_metadata, scorer, scorer_metadata)
    counts = _nested(_nested(protocol.value, "numeric_access"), "expected_counts")
    replay = replay_portability_evidence(
        output / EVIDENCE_NAME, scorer=scorer, expected_counts=counts,
        expected_h_raw=EXPECTED_H_RAW, f1_origin=F1_ORIGIN,
    )
    expected_audit = _audit_body(phase_fingerprint, replay)
    if read_strict_json(output / AUDIT_NAME) != expected_audit:
        raise IntegrityError("completed v1.1 audit fails semantic reconstruction")
    expected_result = _result_body(output, phase_fingerprint, parents.source_fingerprint, expected_audit)
    result = read_strict_json(output / RESULT_NAME)
    if result != expected_result or decision != expected_audit["decision"]:
        raise IntegrityError("completed v1.1 result/decision fails semantic reconstruction")
    execution = read_strict_json(output / GPU_LOG_NAME)
    _validate_execution_record(
        execution,
        phase_fingerprint=phase_fingerprint,
        preflight=preflight,
        counts=counts,
        ensemble_metadata=ensemble_metadata,
        scorer_metadata=scorer_metadata,
    )
    return PortabilityV11Outcome(decision, completion, result, True)


def _normalize_paths(config: PortabilityV11RunConfig) -> tuple[Path, tuple[Path, Path]]:
    workspace = config.workspace_root.resolve(strict=True)
    object.__setattr__(config, "workspace_root", workspace)
    for name in (
        "csv_path", "output_directory", "protocol_path", "protocol_sidecar_path",
        "v0_1_output_directory", "v1_output_directory", "v1_protocol_path",
        "v1_protocol_sidecar_path",
    ):
        object.__setattr__(config, name, _absolute(getattr(config, name), workspace))
    output = config.output_directory
    old_directories = (config.v0_1_output_directory, config.v1_output_directory)
    if any(output == old or old in output.parents or output in old.parents for old in old_directories):
        raise AccessDeniedError("v1.1 output must be disjoint from immutable parents")
    return output, old_directories


def _run_v1_1_core(
    config: PortabilityV11RunConfig,
    *,
    boundary: _NumericalBoundary,
    preflight_probe: Callable[[PortabilityV11RunConfig], Mapping[str, Any]],
) -> PortabilityV11Outcome:
    _validate_config(config)
    output, old_directories = _normalize_paths(config)
    old_snapshots = {str(path): _snapshot_directory(path) for path in old_directories}
    try:
        protocol = authenticate_json_with_sidecar(
            config.protocol_path, config.protocol_sidecar_path,
            expected_sha256=PORTABILITY_V1_1_DEVELOPMENT_SHA256,
        )
        _validate_protocol(protocol, config)
        parents = _authenticate_parents(config)
        expected_csv = config.workspace_root / PurePosixPath(parents.sensor_csv_relative_path)
        if config.csv_path != expected_csv:
            raise IntegrityError("csv_path differs from authenticated parent source")
        if (output / COMPLETION_NAME).is_file():
            return _authenticate_completed(config, protocol, parents)
        if output.exists() and any(output.iterdir()):
            raise PortabilityV11RunnerError("partial v1.1 output exists; fail closed without numeric reentry")
        _ensure_fresh_output(output)
        phase_body = _build_phase_manifest(config, protocol, parents)
        phase = create_fingerprinted_json(
            output / PHASE_MANIFEST_NAME, phase_body,
            fingerprint_field="manifest_fingerprint",
        )
        phase_fingerprint = str(phase.fingerprint)
        ledger_arguments = _ledger_arguments(config, phase_body, phase_fingerprint)
        claim = claim_workspace_access_ledger(output / LEDGER_NAME, **ledger_arguments)

        # Deliberately before both the intent and any numeric loader.
        preflight_body = _preflight_body(config, phase_fingerprint, protocol.value, preflight_probe)
        preflight = create_fingerprinted_json(
            output / PREFLIGHT_NAME, preflight_body,
            fingerprint_field="preflight_fingerprint",
        )
        intent = create_phase_access_intent(
            output / INTENT_NAME, claim,
            phase_manifest_fingerprint=phase_fingerprint,
            runtime_preflight_fingerprint=str(preflight.fingerprint),
        )
        if preflight.value.get("passed") is not True:
            return _finalize_abort(
                output, claim, intent, phase_fingerprint, None,
                reason="RUNTIME_PREFLIGHT_FAILED_BEFORE_NUMERIC_ACCESS",
            )

        try:
            _copy_frozen(config.v1_output_directory / ENSEMBLE_NAME, output / ENSEMBLE_NAME, ENSEMBLE_NAME)
            _copy_frozen(config.v1_output_directory / SCORER_NAME, output / SCORER_NAME, SCORER_NAME)
            ensemble, ensemble_metadata = CaredEnsemble.load_npz(output / ENSEMBLE_NAME)
            scorer, scorer_metadata = CaredBCScorer.load_npz(output / SCORER_NAME)
            _validate_fixed_models(ensemble, ensemble_metadata, scorer, scorer_metadata)
            intervals = _load_intervals(config, boundary, parents, claim, intent, phase_fingerprint)
            counts = _nested(_nested(protocol.value, "numeric_access"), "expected_counts")
            windows: dict[str, InferenceWindowBatch] = {}
            for split in LOAD_PARTITIONS:
                interval = intervals[split]
                window = boundary.build_inference_windows(
                    interval.analog, interval.controls, interval.timestamps,
                    eligible=interval.eligible, reset_before=interval.reset_before,
                )
                if not isinstance(window, InferenceWindowBatch):
                    raise IntegrityError(f"{split} did not produce InferenceWindowBatch")
                if window.inputs.shape[0] != int(counts[split]["inference_windows"]):
                    raise IntegrityError(f"{split} inference-window count changed")
                windows[split] = window

            predictions: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
            aligned: dict[str, dict[str, AlignedResidualBatch]] = {split: {} for split in SPLITS}
            scores: dict[str, dict[str, Any]] = {split: {} for split in SPLITS}
            timings: list[dict[str, Any]] = []
            for split in SPLITS:
                interval = intervals[split]
                for device_name, device in (("cpu", config.comparison_device), ("cuda", config.device)):
                    started = time.perf_counter()
                    prediction = np.asarray(
                        boundary.predict(
                            ensemble, windows[split].inputs,
                            device=device, batch_size=config.batch_size,
                        ), dtype=np.float64,
                    )
                    elapsed = float(time.perf_counter() - started)
                    expected_windows = int(counts[split]["inference_windows"])
                    if prediction.shape != (expected_windows, 3, 7) or not np.isfinite(prediction).all():
                        raise IntegrityError(f"{split}.{device_name} prediction changed")
                    batch = boundary.align(
                        windows[split], prediction, interval.analog, interval.controls,
                        interval.timestamps, eligible=interval.eligible,
                        reset_before=interval.reset_before,
                    )
                    batch = _validate_aligned(
                        f"{split}.{device_name}", batch,
                        int(counts[split]["aligned_scores"]),
                    )
                    score = scorer.score(
                        batch.residuals, batch.state_codes, batch.score_timestamps,
                        reset_before=batch.reset_before,
                    )
                    predictions[split][device_name] = prediction
                    aligned[split][device_name] = batch
                    scores[split][device_name] = score
                    timings.append({
                        "partition": split, "device": device,
                        "prediction_seconds": elapsed,
                        "inference_window_count": expected_windows,
                        "aligned_score_count": int(counts[split]["aligned_scores"]),
                    })
            paths = {
                split: {
                    device: portability_path(
                        predictions[split][device], aligned[split][device], scores[split][device]
                    ) for device in DEVICES
                } for split in SPLITS
            }
            certificate = certify_development_portability(
                healthy_cpu=paths["dev_healthy"]["cpu"],
                healthy_cuda=paths["dev_healthy"]["cuda"],
                f1_cpu=paths["dev_fault_f1"]["cpu"],
                f1_cuda=paths["dev_fault_f1"]["cuda"],
                h_raw=EXPECTED_H_RAW, f1_origin=F1_ORIGIN,
            )
            if type(certificate) is not CertifiedDevelopmentPortabilityAudit:
                raise IntegrityError("trusted certifier returned an invalid type")
            core = certificate.as_dict()
            _, alarm_arrays = alarm_evidence(paths, float(_nested(core, "threshold")["h_eff"]))
            payload = build_evidence_payload(
                paths=paths, aligned=aligned, scores=scores,
                alarms=alarm_arrays, core_audit=core,
            )
            save_evidence_npz(output / EVIDENCE_NAME, payload)

            # Only this persisted replay can authorize PASS.
            replay = replay_portability_evidence(
                output / EVIDENCE_NAME, scorer=scorer, expected_counts=counts,
                expected_h_raw=EXPECTED_H_RAW, f1_origin=F1_ORIGIN,
            )
            audit = _audit_body(phase_fingerprint, replay)
            atomic_create_new_json(output / AUDIT_NAME, audit)
            execution = {
                "schema": "metropt_cared_bc_v1_1_gpu_execution",
                "schema_version": SCHEMA_VERSION, "phase": PHASE_DEVELOPMENT,
                "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
                "protocol_sha256": PORTABILITY_V1_1_DEVELOPMENT_SHA256,
                "phase_manifest_fingerprint": phase_fingerprint,
                "runtime_preflight_fingerprint": preflight.fingerprint,
                "runtime": preflight.value["runtime"],
                "full_sequence_executions": timings,
                "ensemble_metadata": dict(ensemble_metadata),
                "scorer_metadata": dict(scorer_metadata),
                "training_refitting_selection_or_gate_change_performed": False,
            }
            atomic_create_new_json(output / GPU_LOG_NAME, execution)
            result = _result_body(output, phase_fingerprint, parents.source_fingerprint, audit)
            atomic_create_new_json(output / RESULT_NAME, result)
        except Exception as exc:
            _finalize_abort(
                output, claim, intent, phase_fingerprint, exc,
                reason="POST_INTENT_EXECUTION_FAILED",
            )
            raise

        decision = str(result["decision"])
        try:
            manifest = _create_manifest(output, intent, phase_fingerprint, NORMAL_ARTIFACT_NAMES)
            created = create_completion_record(
                output / COMPLETION_NAME, claim=claim, access_intent=intent,
                artifact_manifest=manifest, decision=decision,
            )
        except Exception as exc:
            if not (output / ARTIFACT_MANIFEST_NAME).exists():
                _finalize_abort(
                    output, claim, intent, phase_fingerprint, exc,
                    reason="FINALIZATION_FAILED",
                )
            raise
        verified = authenticate_completion_record(
            output / COMPLETION_NAME, output_directory=output,
            ledger_path=output / LEDGER_NAME,
            expected_ledger_body=build_workspace_access_ledger(**ledger_arguments),
            access_intent_path=output / INTENT_NAME,
            artifact_manifest_path=output / ARTIFACT_MANIFEST_NAME,
            expected_artifact_names=NORMAL_ARTIFACT_NAMES,
        )
        if verified.fingerprint != created.fingerprint:
            raise IntegrityError("v1.1 completion failed post-create authentication")
        return PortabilityV11Outcome(decision, verified, result, False)
    finally:
        for path in old_directories:
            if _snapshot_directory(path) != old_snapshots[str(path)]:
                raise IntegrityError(f"immutable parent changed during v1.1: {path}")


def run_portability_v1_1_development(config: PortabilityV11RunConfig) -> PortabilityV11Outcome:
    """Run the non-injectable production experiment or semantically verify reentry."""
    return _run_v1_1_core(config, boundary=_default_boundary(), preflight_probe=_runtime_probe)


def _resolve(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run governed MetroPT CARED-BC v1.1 portability.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--workspace", default=str(WORKSPACE_ROOT))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--protocol-sidecar", default=DEFAULT_PROTOCOL_SIDECAR_PATH)
    parser.add_argument("--v0-1-output", default=DEFAULT_V0_1_OUTPUT)
    parser.add_argument("--v1-output", default=DEFAULT_V1_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve(strict=True)
    config = PortabilityV11RunConfig(
        csv_path=_resolve(args.csv, workspace),
        output_directory=_resolve(args.output, workspace), workspace_root=workspace,
        protocol_path=_resolve(args.protocol, workspace),
        protocol_sidecar_path=_resolve(args.protocol_sidecar, workspace),
        v0_1_output_directory=_resolve(args.v0_1_output, workspace),
        v1_output_directory=_resolve(args.v1_output, workspace),
    )
    try:
        outcome = run_portability_v1_1_development(config)
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "COMPLETE", "decision": outcome.decision,
        "completion_fingerprint": outcome.completion.fingerprint,
        "reused_completed_bundle": outcome.reused_completed_bundle,
    }, sort_keys=True))
    return 0 if outcome.decision == DEVELOPMENT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PortabilityV11Outcome", "PortabilityV11RunConfig",
    "PortabilityV11RunnerError", "run_portability_v1_1_development",
]
