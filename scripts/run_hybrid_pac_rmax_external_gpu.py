"""Two-stage GPU audit for Hybrid Directional–Energy PAC-RMax GLR-CUSUM.

This is a post-failure sequential external-partition audit, not a pristine or
independent confirmation. The q99 rule was selected after the earlier q96 final
audit failed. Its operational threshold is calibrated once from fault-free
training runs 301--400. Only a completed, hash-bound calibration may authorize
training runs 401--500 for the external audit.

Because pyreadr decodes a whole RData object before row selection, the separation
is administrative rather than physical. The fault-free training file may also
have been decoded wholesale by earlier work; this limitation is disclosed in
every protocol, intent, manifest, and report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc_directional.glr import (
    FAULT_IDS,
    PERSISTENCE_WINDOW,
    Z_CLIP,
    DirectionalGLRPath,
    compare_directional_paths,
    load_directional_glr_model,
    score_directional_glr_runs_numpy,
)
from mate_spc_directional.gpu import score_directional_glr_cuda
from mate_spc_external.rieth_training import (
    DATASET_DOI,
    EXTERNAL_RIETH_FILES,
    FULL_DECODE_DISCLOSURE,
    INTENT_SCHEMA,
    INTENT_SCHEMA_VERSION,
    ExternalRiethRun,
    SafeRiethTrainingArchive,
    load_access_grant,
)
from mate_spc_hybrid.features import load_hybrid_feature_transformer
from mate_spc_pac import pac_tolerance_rank
from mate_spc_pac.certificate import build_calibration_certificate


SCHEMA = "hybrid_directional_energy_pac_rmax_external_partition_audit"
SCHEMA_VERSION = 2
ANALYSIS_STATUS = "post_failure_sequential_external_partition_audit"
METHOD = "hybrid_directional_energy_pac_rmax_glr_cusum"
CALIBRATION_DECISION = "PAC_Q99_CALIBRATION_FROZEN_FOR_EXTERNAL_AUDIT"
PASS_DECISION = "PASS_POST_FAILURE_EXTERNAL_PARTITION_AUDIT"
FAIL_DECISION = "FAIL_POST_FAILURE_EXTERNAL_PARTITION_AUDIT_NO_RETUNING"
OLD_FINAL_FAILURE_DECISION = "FAIL_FINAL_CONFIRMATION_NO_RETUNING"
OLD_HYBRID_AUDIT_DECISION = "PASS_ALL_FROZEN_AUDIT_GATES"

CALIBRATION_RUNS = tuple(range(301, 401))
AUDIT_RUNS = tuple(range(401, 501))
FAULT_START = 20
SEQUENCE_LENGTH = 500
VALID_START = 63
FIRST_ALARM_ELIGIBLE = VALID_START + PERSISTENCE_WINDOW - 1
THEORETICAL_MINIMUM_DELAY = FIRST_ALARM_ELIGIBLE - FAULT_START
PAC_EPSILON = 0.05
PAC_DELTA = 0.05
PAC_RANK = 99
POSTHOC_DESIGN_Q99 = 254.4559644524996
POSTHOC_DESIGN_Q99_HEX = "0x1.fce9742c373fap+7"
OLD_Q96_THRESHOLD = 165.05745028203916
CPU_GPU_TOLERANCE = 1e-10
HEALTHY_ALARM_RUNS_MAXIMUM = 4
PER_FAULT_DETECTIONS_MINIMUM = 90
TOTAL_FAULT_DETECTIONS_MINIMUM = 270
MEDIAN_RESTRICTED_DELAY_MAXIMUM = 120.0
PER_FAULT_RESTRICTED_P90_MAXIMUM = 180.0
MISS_DELAY = SEQUENCE_LENGTH - FAULT_START

HYBRID_DIRECTORY = Path("outputs/hybrid_energy_glr_gpu_v1")
OLD_FINAL_DIRECTORY = Path("outputs/hybrid_energy_glr_gpu_v1_final_confirmation")
TRANSFORMER_NAME = "hybrid_transformer.npz"
MODEL_NAME = "hybrid_glr_uncalibrated.npz"
DESIGN_SCORES_NAME = "calibration_scores.npz"
USED_HYBRID_ARTIFACTS = (TRANSFORMER_NAME, MODEL_NAME, DESIGN_SCORES_NAME)
IMPLEMENTATION_FILES = (
    "scripts/run_hybrid_pac_rmax_external_gpu.py",
    "tests/test_run_hybrid_pac_rmax_external_gpu.py",
    "src/mate_spc_external/__init__.py",
    "src/mate_spc_external/rieth_training.py",
    "src/mate_spc_hybrid/features.py",
    "src/mate_spc_directional/features.py",
    "src/mate_spc_directional/glr.py",
    "src/mate_spc_directional/gpu.py",
    "src/mate_spc_pac/__init__.py",
    "src/mate_spc_pac/certificate.py",
    "src/mate_spc_pac/common.py",
    "src/mate_spc_pac/probability.py",
)
CALIBRATION_ARTIFACTS = (
    "REPORT.md",
    "access_intent.json",
    "artifact_manifest.json",
    "calibration_evidence.npz",
    "calibration_result.json",
    "data_access_audit.json",
    "gpu_execution.json",
    "pac_certificate.json",
    "parity.json",
    "phase_manifest.json",
    "protocol_snapshot.json",
    "runtime_preflight.json",
)
AUDIT_ARTIFACTS = (
    "REPORT.md",
    "access_intent.json",
    "artifact_manifest.json",
    "audit_evidence.npz",
    "data_access_audit.json",
    "gpu_execution.json",
    "parity.json",
    "phase_manifest.json",
    "protocol_snapshot.json",
    "result.json",
    "runtime_preflight.json",
)
FORBIDDEN_ACTIONS = [
    "training",
    "refitting",
    "hyperparameter_selection",
    "threshold_selection_from_audit_rows",
    "same_data_retuning_after_result",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="freeze protocol without decoding either external RData file"
    )
    freeze.add_argument("--protocol", type=Path, required=True)
    run = subparsers.add_parser("run", help="execute calibration and/or audit")
    run.add_argument("data_dir", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--phase", choices=("calibration", "audit", "all"), default="all")
    run.add_argument("--device", choices=("cuda", "cuda:0"), required=True)
    run.add_argument("--batch-size", type=int, default=32)
    return parser


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _clean_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite values cannot be serialized to JSON")
        return result
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        _clean_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _hash_record(path: Path) -> dict[str, int | str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_clean_json(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_fingerprinted_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    body = _strict_json(path)
    fingerprint = body.get(field)
    unsigned = dict(body)
    unsigned.pop(field, None)
    if not isinstance(fingerprint, str) or fingerprint != _canonical_sha256(unsigned):
        raise ValueError(f"{path.name} has an invalid {field}")
    return body, fingerprint


def _verify_completion_directory(directory: Path, expected_decision: str) -> dict[str, Any]:
    completion_path = directory / "completion.json"
    completion, fingerprint = _verify_fingerprinted_json(
        completion_path, "completion_fingerprint"
    )
    if completion.get("decision") != expected_decision:
        raise ValueError(f"{directory.name} has the wrong frozen decision")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{directory.name} completion has no artifacts")
    for relative, expected in artifacts.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise ValueError("completion artifact map is malformed")
        if _hash_record(directory / relative) != expected:
            raise ValueError(f"completed artifact changed: {directory.name}/{relative}")
    return {
        "directory": directory.as_posix(),
        "decision": expected_decision,
        "completion_sha256": _sha256_file(completion_path),
        "completion_fingerprint": fingerprint,
        "phase_fingerprint": completion.get("phase_fingerprint"),
        "artifact_records": artifacts,
    }


def _implementation_snapshot(project_root: Path) -> dict[str, dict[str, int | str]]:
    return {relative: _hash_record(project_root / relative) for relative in IMPLEMENTATION_FILES}


def _monitor_binding(artifact_records: Mapping[str, Any]) -> dict[str, Any]:
    deployment = {
        "method": METHOD,
        "transformer_sha256": artifact_records[TRANSFORMER_NAME]["sha256"],
        "uncalibrated_model_sha256": artifact_records[MODEL_NAME]["sha256"],
        "sequence_length": SEQUENCE_LENGTH,
        "valid_start_zero_based": VALID_START,
        "first_alarm_eligible_zero_based": FIRST_ALARM_ELIGIBLE,
        "fault_start_zero_based": FAULT_START,
        "fault_head_order": list(FAULT_IDS),
        "run_state_reset": True,
    }
    monitor = {
        "density_ratio_heads": list(FAULT_IDS),
        "kappa_per_head": [0.75, 0.75, 0.75],
        "z_clip": [-Z_CLIP, Z_CLIP],
        "page_cusum": "one_sided_reset_at_zero",
        "persistence": "3_of_3_per_head_then_family_maximum",
        "comparison": "strictly_greater_than",
        "threshold_rule": "PAC_RMax_second_largest_q99_of_100_run_maxima",
    }
    return {
        "deployment_spec": deployment,
        "deployment_spec_sha256": _canonical_sha256(deployment),
        "monitor_configuration": monitor,
        "monitor_configuration_sha256": _canonical_sha256(monitor),
    }


def _validate_frozen_method(
    project_root: Path, artifact_records: Mapping[str, Any]
) -> dict[str, Any]:
    hybrid_dir = project_root / HYBRID_DIRECTORY
    for name in USED_HYBRID_ARTIFACTS:
        if _hash_record(hybrid_dir / name) != artifact_records.get(name):
            raise ValueError(f"hybrid completion does not authenticate {name}")
    transformer, transformer_metadata = load_hybrid_feature_transformer(
        hybrid_dir / TRANSFORMER_NAME
    )
    model, model_metadata = load_directional_glr_model(hybrid_dir / MODEL_NAME)
    if transformer.valid_start_ != VALID_START:
        raise ValueError("frozen hybrid transformer valid_start changed")
    if model.threshold is not None:
        raise ValueError("runner requires the uncalibrated frozen GLR model")
    if model.feature_count != transformer.output_feature_count_:
        raise ValueError("hybrid transformer/model feature counts disagree")
    if tuple(model.fault_ids) != tuple(FAULT_IDS):
        raise ValueError("frozen model fault-head order changed")
    if not np.array_equal(model.kappa, np.full(3, 0.75, dtype=np.float64)):
        raise ValueError("frozen GLR kappa is not exactly 0.75 per head")
    return {
        "feature_count": int(model.feature_count),
        "hybrid_output_feature_count": int(transformer.output_feature_count_),
        "transformer_metadata_sha256": _canonical_sha256(transformer_metadata),
        "model_metadata_sha256": _canonical_sha256(model_metadata),
    }


def _posthoc_design_evidence(
    project_root: Path, monitor_binding: Mapping[str, Any]
) -> dict[str, Any]:
    path = project_root / HYBRID_DIRECTORY / DESIGN_SCORES_NAME
    required = {
        "run_names", "fault_ids", "simulation_runs", "logits",
        "standardized_logits", "cusum", "component_persistence", "envelope",
        "eligible", "alarm", "threshold",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("design calibration_scores.npz key set changed")
        if any(archive[name].dtype.kind == "O" for name in archive.files):
            raise ValueError("object arrays are forbidden in design evidence")
        names = np.array(archive["run_names"], copy=True)
        fault_ids = np.array(archive["fault_ids"], copy=True)
        simulation_runs = np.array(archive["simulation_runs"], copy=True)
        envelope = np.array(archive["envelope"], copy=True)
        eligible = np.array(archive["eligible"], copy=True)
        old_threshold = np.array(archive["threshold"], copy=True)
    expected_names = np.asarray(
        [f"Rieth-fault_free_testing-fault-00-run-{run:03d}" for run in range(301, 401)]
    )
    if not np.array_equal(names, expected_names):
        raise ValueError("post-hoc design run identity/order changed")
    if not np.array_equal(fault_ids, np.zeros(100, dtype=np.int64)):
        raise ValueError("post-hoc design fault IDs changed")
    if not np.array_equal(simulation_runs, np.arange(301, 401, dtype=np.int64)):
        raise ValueError("post-hoc design simulation runs changed")
    if envelope.shape != (100, SEQUENCE_LENGTH) or envelope.dtype != np.float64:
        raise ValueError("post-hoc design envelope shape/dtype changed")
    expected_eligible = np.zeros_like(eligible, dtype=bool)
    expected_eligible[:, FIRST_ALARM_ELIGIBLE:] = True
    if eligible.dtype != np.bool_ or not np.array_equal(eligible, expected_eligible):
        raise ValueError("post-hoc design eligibility changed")
    if not np.isneginf(envelope[:, :FIRST_ALARM_ELIGIBLE]).all():
        raise ValueError("post-hoc design leading envelope must be -inf")
    if not np.isfinite(envelope[:, FIRST_ALARM_ELIGIBLE:]).all():
        raise ValueError("post-hoc design eligible envelope is non-finite")
    if old_threshold.shape != (1,) or float(old_threshold[0]) != OLD_Q96_THRESHOLD:
        raise ValueError("observed q96 threshold changed")
    maxima = [
        (str(name), float(np.max(values[FIRST_ALARM_ELIGIBLE:])))
        for name, values in zip(names.tolist(), envelope)
    ]
    certificate = build_calibration_certificate(
        deployment_spec_sha256=str(monitor_binding["deployment_spec_sha256"]),
        monitor_configuration_sha256=str(monitor_binding["monitor_configuration_sha256"]),
        run_maxima=maxima,
        deferred_runs=(),
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )
    if certificate.rank.rank_one_based != PAC_RANK:
        raise ValueError("post-hoc design PAC rank is not q99")
    if certificate.threshold != POSTHOC_DESIGN_Q99:
        raise ValueError("post-hoc q99 design statistic changed")
    if certificate.threshold.hex() != POSTHOC_DESIGN_Q99_HEX:
        raise ValueError("post-hoc q99 binary64 encoding changed")
    return {
        "source_scores": _hash_record(path),
        "source_partition": "fault_free_testing",
        "source_runs": [301, 400],
        "source_values_already_consumed": True,
        "observed_original_q96_threshold": OLD_Q96_THRESHOLD,
        "posthoc_q99_second_largest_threshold": POSTHOC_DESIGN_Q99,
        "posthoc_q99_binary64_hex": POSTHOC_DESIGN_Q99_HEX,
        "posthoc_q99_rank_driver_run": certificate.rank_driver_run,
        "rank": certificate.rank.as_dict(),
        "calibration_maxima_sha256": certificate.calibration_maxima_sha256,
        "operational_threshold_reuse_forbidden": True,
        "purpose": "design evidence for freezing the q99 rule only",
    }


def build_protocol_template(project_root: Path) -> dict[str, Any]:
    """Build the sole accepted protocol; no external RData is decoded here."""

    project_root = project_root.resolve()
    hybrid = _verify_completion_directory(
        project_root / HYBRID_DIRECTORY, OLD_HYBRID_AUDIT_DECISION
    )
    old_final = _verify_completion_directory(
        project_root / OLD_FINAL_DIRECTORY, OLD_FINAL_FAILURE_DECISION
    )
    used_records = {
        name: hybrid["artifact_records"][name] for name in USED_HYBRID_ARTIFACTS
    }
    method_shape = _validate_frozen_method(project_root, used_records)
    monitor_binding = _monitor_binding(used_records)
    design = _posthoc_design_evidence(project_root, monitor_binding)
    rank = pac_tolerance_rank(100, epsilon=PAC_EPSILON, delta=PAC_DELTA)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "method": METHOD,
        "governance_disclosure": {
            "created_after_original_q96_final_failure_was_observed": True,
            "q99_rule_selected_after_original_q96_final_failure": True,
            "post_failure_method_revision": True,
            "not_a_pristine_independent_final_confirmation": True,
            "interpretation": "sequential exploratory external-partition validation",
            "old_testing_runs_401_500_consumed": True,
            "fault_free_training_partition_may_have_been_whole_file_decoded_before": True,
            "fault_free_training_rows_301_500_not_previously_selected_or_scored": True,
            "faulty_training_rows_401_500_numeric_values_unread_before_protocol_freeze": True,
            "user_authorized_new_method_after_failure": True,
            "same_external_audit_rows_retuning_after_result_forbidden": True,
        },
        "predecessor_hybrid_audit": {
            key: hybrid[key]
            for key in (
                "directory", "decision", "completion_sha256",
                "completion_fingerprint", "phase_fingerprint",
            )
        },
        "observed_failed_final_confirmation": {
            key: old_final[key]
            for key in (
                "directory", "decision", "completion_sha256",
                "completion_fingerprint", "phase_fingerprint",
            )
        },
        "frozen_inputs": {
            "directory": HYBRID_DIRECTORY.as_posix(),
            "artifacts": used_records,
            "safe_serialization": "numeric NPZ loaded with allow_pickle=False",
            **method_shape,
        },
        "monitor_binding": monitor_binding,
        "posthoc_design_evidence": design,
        "operational_pac_calibration": {
            "partition": "fault_free_training",
            "fault_ids": [0],
            "simulation_runs": list(CALIBRATION_RUNS),
            "run_count": 100,
            "rank_rule": "second_largest_order_statistic_no_interpolation",
            "threshold_source": "CPU_NumPy_reference_after_full_CPU_CUDA_parity",
            "rank": rank.as_dict(),
            "threshold_numeric_value_unknown_at_protocol_freeze": True,
            "posthoc_design_numeric_threshold_reuse_forbidden": True,
            "completion_required_before_audit_open": True,
        },
        "external_partition_audit": {
            "dataset": "Rieth et al. multi-run Tennessee Eastman process",
            "doi": DATASET_DOI,
            "healthy": {
                "partition": "fault_free_training",
                "fault_ids": [0],
                "simulation_runs": list(AUDIT_RUNS),
            },
            "faults": {
                "partition": "faulty_training",
                "fault_ids": list(FAULT_IDS),
                "simulation_runs": list(AUDIT_RUNS),
            },
            "files": {key: dict(value) for key, value in EXTERNAL_RIETH_FILES.items()},
            "sequence_length": SEQUENCE_LENGTH,
            "fault_onset": "sample 21, represented as zero-based index 20",
            "fault_start_zero_based": FAULT_START,
            "valid_start_zero_based": VALID_START,
            "first_alarm_eligible_zero_based": FIRST_ALARM_ELIGIBLE,
            "theoretical_minimum_detection_delay_samples": THEORETICAL_MINIMUM_DELAY,
            "pre_fault_interval_not_used_as_performance_evidence": [0, FAULT_START],
            "miss_delay_samples": MISS_DELAY,
            "total_sequences": 400,
            "decode_disclosure": FULL_DECODE_DISCLOSURE,
            "fault_free_file_whole_decoded_once_per_phase": True,
            "faulty_file_not_decoded_until_audit_open": True,
        },
        "runtime": {
            "required_device": "cuda",
            "score_dtype": "float64",
            "tf32": False,
            "deterministic_algorithms": True,
            "cpu_reference": "independent NumPy directional GLR-CUSUM",
            "cpu_cuda_max_abs_tolerance": CPU_GPU_TOLERANCE,
            "calibration_cpu_cuda_q99_threshold_max_abs_tolerance": CPU_GPU_TOLERANCE,
            "full_run_alarm_arrays_must_match_exactly": True,
        },
        "success_gates": {
            "all_required": True,
            "healthy_alarm_runs_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
            "healthy_gate_interpretation": (
                "at most 4/100 makes the one-sided 95% Clopper-Pearson upper "
                "bound less than 0.10; whether the observed count also makes that "
                "upper bound less than 0.05 is reported dynamically"
            ),
            "pac_primary_claim": (
                "under exchangeable run-level maxima, q99 provides epsilon=0.05 "
                "coverage with the certificate confidence"
            ),
            "per_fault_detected_runs_minimum": PER_FAULT_DETECTIONS_MINIMUM,
            "total_fault_detected_runs_minimum": TOTAL_FAULT_DETECTIONS_MINIMUM,
            "total_gate_is_redundant_with_three_per_fault_gates": True,
            "per_fault_median_restricted_delay_samples_maximum": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "overall_median_restricted_delay_report_only": True,
            "per_fault_restricted_p90_delay_samples_maximum": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "restricted_delay_miss_imputation_samples": MISS_DELAY,
            "cpu_cuda_alarm_arrays_exact": True,
            "cpu_cuda_all_numeric_paths_within_tolerance": True,
        },
        "three_logical_numeric_reads": [
            "calibration: fault_free_training runs 301-400",
            "audit: fault_free_training runs 401-500",
            "audit: faulty_training faults 3/9/15 runs 401-500",
        ],
        "access_order": [
            "freeze protocol and sidecar",
            "write calibration access intent and phase manifest",
            "decode/select/score calibration rows and atomically complete calibration",
            "write threshold-bound audit-open intent and phase manifest",
            "decode/select/score audit rows and atomically complete audit",
        ],
        "forbidden_actions": [
            "training_on_calibration_or_audit_rows",
            "refitting_on_calibration_or_audit_rows",
            "hyperparameter_selection_on_calibration_or_audit_rows",
            "threshold_selection_on_audit_rows",
            "same_audit_rows_retuning_after_result",
            "relabeling_this_audit_as_independent_final_confirmation",
        ],
        "output_contract": {
            "atomic_new_phase_directories": ["calibration", "audit"],
            "completed_phase_reentry_reads_no_RData": True,
            "calibration_artifacts": list(CALIBRATION_ARTIFACTS) + ["completion.json"],
            "audit_artifacts": list(AUDIT_ARTIFACTS) + ["completion.json"],
        },
        "implementation_snapshot": _implementation_snapshot(project_root),
    }


def freeze_protocol(project_root: Path, protocol_path: Path) -> str:
    protocol_path = protocol_path.resolve()
    sidecar = protocol_path.with_suffix(".sha256")
    if protocol_path.exists() or sidecar.exists():
        raise FileExistsError("protocol or SHA-256 sidecar already exists")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(protocol_path, build_protocol_template(project_root))
    digest = _sha256_file(protocol_path)
    try:
        _write_text(sidecar, f"{digest}  {protocol_path.name}\n")
    except Exception:
        protocol_path.unlink(missing_ok=True)
        raise
    return digest


def _load_frozen_protocol(
    project_root: Path,
    protocol_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    protocol_path = protocol_path.resolve()
    expected = _require_sha256(expected_sha256, label="protocol_sha256")
    if _sha256_file(protocol_path) != expected:
        raise ValueError("protocol file SHA-256 differs from --protocol-sha256")
    sidecar = protocol_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != (
        f"{expected}  {protocol_path.name}\n"
    ):
        raise ValueError("protocol SHA-256 sidecar is missing or differs")
    protocol = _strict_json(protocol_path)
    if protocol != build_protocol_template(project_root):
        raise ValueError("frozen protocol differs from authenticated code/input template")
    return protocol, expected


def _runtime_preflight(device: str) -> dict[str, Any]:
    import torch

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audit requires a CUDA device")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(target)
    probe = torch.tensor([1.0, 2.0], dtype=torch.float64, device=target)
    probe_value = float(torch.sum(probe * probe).cpu().item())
    if probe_value != 5.0:
        raise RuntimeError("CUDA float64 preflight failed")
    return {
        "device": str(target),
        "device_name": str(torch.cuda.get_device_name(target)),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float64",
        "tf32": False,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "probe_sum_of_squares": probe_value,
    }


def _authorization_record(
    partition: str, fault_ids: Sequence[int], runs: Sequence[int]
) -> dict[str, Any]:
    source = EXTERNAL_RIETH_FILES[partition]
    return {
        "partition": partition,
        "file_name": source["file_name"],
        "expected_md5": source["md5"],
        "expected_sha256": source["sha256"],
        "expected_size_bytes": source["size_bytes"],
        "fault_ids": list(fault_ids),
        "simulation_runs_by_fault": {str(fault): list(runs) for fault in fault_ids},
        "expected_run_length": SEQUENCE_LENGTH,
        "fault_start_zero_based": FAULT_START,
        "decode_scope": "complete_partition_then_exact_row_selection",
    }


def _calibration_access_intent(
    protocol_sha256: str, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": INTENT_SCHEMA,
        "schema_version": INTENT_SCHEMA_VERSION,
        "phase": "calibration",
        "protocol_sha256": protocol_sha256,
        "created_before_external_numeric_decode": True,
        "whole_partition_decode_acknowledged": FULL_DECODE_DISCLOSURE,
        "authorized_reads": [
            _authorization_record("fault_free_training", (0,), CALIBRATION_RUNS)
        ],
        "forbidden_actions": FORBIDDEN_ACTIONS,
        "phase_binding": {
            "deployment_spec_sha256": protocol["monitor_binding"]["deployment_spec_sha256"],
            "monitor_configuration_sha256": protocol["monitor_binding"][
                "monitor_configuration_sha256"
            ],
            "threshold_rule": "PAC q99 second-largest of 100 CPU run maxima",
            "numeric_threshold_unknown_before_this_access": True,
        },
    }


def _audit_access_intent(
    protocol_sha256: str, calibration_binding: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": INTENT_SCHEMA,
        "schema_version": INTENT_SCHEMA_VERSION,
        "phase": "audit",
        "protocol_sha256": protocol_sha256,
        "created_before_external_numeric_decode": True,
        "whole_partition_decode_acknowledged": FULL_DECODE_DISCLOSURE,
        "authorized_reads": [
            _authorization_record("fault_free_training", (0,), AUDIT_RUNS),
            _authorization_record("faulty_training", FAULT_IDS, AUDIT_RUNS),
        ],
        "forbidden_actions": FORBIDDEN_ACTIONS,
        "phase_binding": dict(calibration_binding),
    }


def _phase_manifest(
    *,
    phase: str,
    protocol_sha256: str,
    intent_sha256: str,
    runtime_sha256: str,
    source_records: Mapping[str, Any],
    calibration_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = {
        "schema": "hybrid_pac_rmax_external_phase_manifest",
        "schema_version": 1,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "access_intent_sha256": intent_sha256,
        "runtime_preflight_sha256": runtime_sha256,
        "source_records": source_records,
        "calibration_binding": calibration_binding,
        "written_before_any_phase_pyreadr_decode": True,
        "whole_partition_decode_acknowledged": FULL_DECODE_DISCLOSURE,
    }
    return {**body, "phase_fingerprint": _canonical_sha256(body)}


def _phase_staging(output_root: Path, phase: str) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / phase
    prefix = f".{phase}.inprogress-"
    leftovers = [path for path in output_root.iterdir() if path.name.startswith(prefix)]
    if leftovers:
        raise RuntimeError(
            "an earlier phase staging directory exists; do not reopen RData: "
            + ", ".join(str(path) for path in leftovers)
        )
    return target, Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))


def _copy_protocol_snapshot(protocol_path: Path, staging: Path) -> None:
    with (staging / "protocol_snapshot.json").open("xb") as stream:
        stream.write(protocol_path.resolve().read_bytes())
        stream.flush()
        os.fsync(stream.fileno())


def _feature_mapping(
    transformer: Any, runs: Iterable[ExternalRiethRun]
) -> dict[str, np.ndarray]:
    return {run.name: transformer.transform(run.x) for run in runs}


def _score_with_full_parity(
    transformer: Any,
    model: Any,
    runs: tuple[ExternalRiethRun, ...],
    *,
    device: str,
    batch_size: int,
) -> tuple[
    dict[str, DirectionalGLRPath],
    dict[str, DirectionalGLRPath],
    dict[str, Any],
    dict[str, Any],
]:
    gpu_paths: dict[str, DirectionalGLRPath] = {}
    cpu_paths_all: dict[str, DirectionalGLRPath] = {}
    parity_records: list[dict[str, Any]] = []
    gpu_calls: list[dict[str, Any]] = []
    transform_seconds = 0.0
    cpu_score_seconds = 0.0
    wall_started = time.perf_counter()
    error_keys = (
        "logits_max_abs_error", "standardized_logits_max_abs_error",
        "clipped_evidence_max_abs_error", "cusum_max_abs_error",
        "component_persistence_max_abs_error", "envelope_max_abs_error",
    )
    maxima = {key: 0.0 for key in error_keys}
    for start in range(0, len(runs), batch_size):
        chunk = runs[start : start + batch_size]
        transform_started = time.perf_counter()
        features = _feature_mapping(transformer, chunk)
        transform_seconds += time.perf_counter() - transform_started
        gpu = score_directional_glr_cuda(
            model, features, valid_start=VALID_START, device=device, batch_size=batch_size
        )
        cpu_started = time.perf_counter()
        cpu = score_directional_glr_runs_numpy(model, features, valid_start=VALID_START)
        cpu_score_seconds += time.perf_counter() - cpu_started
        gpu_calls.append(gpu.execution)
        for run in chunk:
            if run.name in gpu_paths:
                raise RuntimeError("a run was scored twice")
            comparison = compare_directional_paths(
                cpu[run.name], gpu.by_run[run.name], atol=CPU_GPU_TOLERANCE
            )
            for key in error_keys:
                maxima[key] = max(maxima[key], float(comparison[key]))
            parity_records.append({"run": run.name, **comparison})
            gpu_paths[run.name] = gpu.by_run[run.name]
            cpu_paths_all[run.name] = cpu[run.name]
        del features, cpu, gpu
        gc.collect()
    expected_names = {run.name for run in runs}
    if set(gpu_paths) != expected_names or set(cpu_paths_all) != expected_names:
        raise RuntimeError("scoring did not return every run")
    parity = {
        "atol": CPU_GPU_TOLERANCE,
        "run_count": len(runs),
        "all_numeric_paths_within_tolerance": all(
            value <= CPU_GPU_TOLERANCE for value in maxima.values()
        ),
        "all_eligibility_arrays_exact": all(
            bool(record["eligible_exact"]) for record in parity_records
        ),
        "all_alarm_arrays_exact": all(bool(record["alarm_exact"]) for record in parity_records),
        "maximum_errors": maxima,
        "records": parity_records,
    }
    execution = {
        "backend": "pytorch_cuda_float64_with_independent_numpy_reference",
        "device": device,
        "dtype": "float64",
        "tf32": False,
        "deterministic_algorithms": True,
        "sequence_count": len(runs),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": int(model.feature_count),
        "batch_size": batch_size,
        "feature_transform_wall_seconds": transform_seconds,
        "cpu_reference_score_wall_seconds": cpu_score_seconds,
        "total_wall_seconds": time.perf_counter() - wall_started,
        "cuda_event_elapsed_seconds": sum(
            float(call["cuda_event_elapsed_seconds"]) for call in gpu_calls
        ),
        "peak_cuda_memory_bytes": max(int(call["peak_cuda_memory_bytes"]) for call in gpu_calls),
        "cuda_calls": gpu_calls,
    }
    return gpu_paths, cpu_paths_all, parity, execution


def _certificate_from_paths(
    paths: Mapping[str, DirectionalGLRPath], protocol: Mapping[str, Any]
) -> Any:
    maxima: list[tuple[str, float]] = []
    for name in sorted(paths):
        values = np.asarray(paths[name].envelope, dtype=np.float64)
        if not np.isneginf(values[:FIRST_ALARM_ELIGIBLE]).all():
            raise ValueError(f"{name}: pre-eligible envelope is not -inf")
        if not np.isfinite(values[FIRST_ALARM_ELIGIBLE:]).all():
            raise ValueError(f"{name}: eligible envelope is not finite")
        maxima.append((name, float(np.max(values[FIRST_ALARM_ELIGIBLE:]))))
    if len(maxima) != 100:
        raise ValueError("PAC calibration requires exactly 100 run maxima")
    binding = protocol["monitor_binding"]
    return build_calibration_certificate(
        deployment_spec_sha256=str(binding["deployment_spec_sha256"]),
        monitor_configuration_sha256=str(binding["monitor_configuration_sha256"]),
        run_maxima=maxima,
        deferred_runs=(),
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )


def _calibration_parity(
    cpu_paths: Mapping[str, DirectionalGLRPath],
    gpu_paths: Mapping[str, DirectionalGLRPath],
    protocol: Mapping[str, Any],
    base_parity: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    cpu = _certificate_from_paths(cpu_paths, protocol)
    gpu = _certificate_from_paths(gpu_paths, protocol)
    threshold_error = abs(float(cpu.threshold) - float(gpu.threshold))
    driver_exact = cpu.rank_driver_run == gpu.rank_driver_run
    rank_exact = cpu.rank.rank_one_based == gpu.rank.rank_one_based == PAC_RANK
    maxima_errors = []
    cpu_maxima = dict(cpu.run_maxima)
    gpu_maxima = dict(gpu.run_maxima)
    if set(cpu_maxima) != set(gpu_maxima):
        raise AssertionError("CPU/CUDA calibration run identities differ")
    for name in sorted(cpu_maxima):
        maxima_errors.append(abs(cpu_maxima[name] - gpu_maxima[name]))
    maximum_run_maximum_error = max(maxima_errors)
    passed = (
        bool(base_parity["all_numeric_paths_within_tolerance"])
        and bool(base_parity["all_alarm_arrays_exact"])
        and driver_exact
        and rank_exact
        and threshold_error <= CPU_GPU_TOLERANCE
        and maximum_run_maximum_error <= CPU_GPU_TOLERANCE
    )
    if not passed:
        raise AssertionError("CPU/CUDA PAC q99 calibration parity failed")
    details = {
        **dict(base_parity),
        "cpu_q99_threshold": float(cpu.threshold),
        "cuda_q99_threshold": float(gpu.threshold),
        "q99_threshold_max_abs_error": threshold_error,
        "cpu_q99_driver_run": cpu.rank_driver_run,
        "cuda_q99_driver_run": gpu.rank_driver_run,
        "q99_driver_run_exact": driver_exact,
        "q99_rank_exact": rank_exact,
        "maximum_per_run_maximum_abs_error": maximum_run_maximum_error,
        "operational_threshold_source": "CPU_NumPy_reference",
        "passed": passed,
    }
    return cpu, details


def _alarm_episode_count(alarm: np.ndarray) -> int:
    values = np.asarray(alarm, dtype=bool)
    return int(np.sum(values & ~np.r_[False, values[:-1]]))


def _healthy_records(
    runs: Iterable[ExternalRiethRun], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        path = paths[run.name]
        indices = np.flatnonzero(path.alarm)
        records.append(
            {
                "run": run.name,
                "simulation_run": int(run.metadata["simulation_run"]),
                "any_alarm": bool(indices.size),
                "alarm_episode_count": _alarm_episode_count(path.alarm),
                "first_alarm_index": int(indices[0]) if indices.size else None,
                "maximum_envelope": float(np.max(path.envelope[path.eligible])),
            }
        )
    return records


def _fault_records(
    runs: Iterable[ExternalRiethRun], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    timeline = np.arange(SEQUENCE_LENGTH)
    for run in runs:
        path = paths[run.name]
        post = np.flatnonzero(path.alarm & (timeline >= FAULT_START))
        first = int(post[0]) if post.size else None
        fault_id = int(run.metadata["fault_id"])
        attributed = None
        if first is not None:
            attributed = int(FAULT_IDS[int(np.argmax(path.component_persistence[first]))])
        delay = first - FAULT_START if first is not None else None
        records.append(
            {
                "run": run.name,
                "fault_id": fault_id,
                "simulation_run": int(run.metadata["simulation_run"]),
                "detected": first is not None,
                "first_fault_alarm_index": first,
                "delay_samples": delay,
                "restricted_delay_samples_miss_at_run_end": delay if delay is not None else MISS_DELAY,
                "pre_fault_alarm_episode_count": _alarm_episode_count(path.alarm[:FAULT_START]),
                "pre_fault_interval_used_as_performance_evidence": False,
                "attributed_fault_head_at_first_alarm": attributed,
                "attribution_matches_fault": attributed == fault_id if attributed else False,
                "maximum_post_fault_envelope": float(
                    np.max(path.envelope[max(FAULT_START, FIRST_ALARM_ELIGIBLE) :])
                ),
            }
        )
    return records


def _empirical_upper_order_statistic(values: Sequence[int], probability: float) -> float:
    if not values or not 0.0 < probability <= 1.0:
        raise ValueError("order-statistic inputs are invalid")
    ordered = sorted(int(value) for value in values)
    rank = math.ceil(probability * len(ordered))
    return float(ordered[rank - 1])


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return sum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def _clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if not 0 <= k <= n or not 0.0 < alpha < 1.0:
        raise ValueError("Clopper-Pearson inputs are invalid")
    if k == n:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(k, n, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return math.nextafter(upper, math.inf)


def _summarize_audit(
    healthy_records: list[dict[str, Any]],
    fault_records: list[dict[str, Any]],
    parity: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    healthy_alarms = sum(bool(record["any_alarm"]) for record in healthy_records)
    healthy_cp_upper = _clopper_pearson_upper(healthy_alarms, len(healthy_records))
    healthy_pass = healthy_alarms <= HEALTHY_ALARM_RUNS_MAXIMUM
    by_fault: dict[str, Any] = {}
    detected_counts: list[int] = []
    all_restricted: list[int] = []
    median_passes: list[bool] = []
    p90_passes: list[bool] = []
    for fault_id in FAULT_IDS:
        records = [record for record in fault_records if record["fault_id"] == fault_id]
        detected = sum(bool(record["detected"]) for record in records)
        delays = [int(record["restricted_delay_samples_miss_at_run_end"]) for record in records]
        median = float(statistics.median(delays))
        p90 = _empirical_upper_order_statistic(delays, 0.90)
        detected_counts.append(detected)
        all_restricted.extend(delays)
        median_passes.append(median <= MEDIAN_RESTRICTED_DELAY_MAXIMUM)
        p90_passes.append(p90 <= PER_FAULT_RESTRICTED_P90_MAXIMUM)
        by_fault[f"{fault_id:02d}"] = {
            "run_count": len(records),
            "detected_runs": detected,
            "detection_rate": detected / len(records),
            "median_restricted_delay_samples": median,
            "p90_restricted_delay_samples_higher_order_statistic": p90,
            "passed_detection_gate": detected >= PER_FAULT_DETECTIONS_MINIMUM,
            "passed_median_gate": median <= MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "passed_p90_gate": p90 <= PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "records": records,
        }
    total_detected = sum(detected_counts)
    overall_median = float(statistics.median(all_restricted))
    parity_pass = bool(parity["all_numeric_paths_within_tolerance"]) and bool(
        parity["all_alarm_arrays_exact"]
    )
    passed = (
        healthy_pass
        and min(detected_counts) >= PER_FAULT_DETECTIONS_MINIMUM
        and total_detected >= TOTAL_FAULT_DETECTIONS_MINIMUM
        and all(median_passes)
        and all(p90_passes)
        and parity_pass
    )
    return {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "phase": "post_failure_sequential_external_partition_audit",
        "method": METHOD,
        "passed": passed,
        "decision": PASS_DECISION if passed else FAIL_DECISION,
        "threshold": threshold,
        "threshold_binary64_hex": float(threshold).hex(),
        "comparison": "strictly_greater_than",
        "healthy": {
            "run_count": len(healthy_records),
            "runs_with_any_alarm": healthy_alarms,
            "alarm_run_rate": healthy_alarms / len(healthy_records),
            "one_sided_clopper_pearson_95_upper": healthy_cp_upper,
            "empirical_cp95_upper_below_0_05": healthy_cp_upper < 0.05,
            "empirical_cp95_upper_below_0_10": healthy_cp_upper < 0.10,
            "gate_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
            "passed": healthy_pass,
            "healthy_gate_by_itself_targets_ten_percent_upper": True,
            "records": healthy_records,
        },
        "faults": {
            "per_fault": by_fault,
            "detected_runs": total_detected,
            "run_count": len(fault_records),
            "detection_rate": total_detected / len(fault_records),
            "per_fault_minimum_gate": PER_FAULT_DETECTIONS_MINIMUM,
            "total_minimum_gate": TOTAL_FAULT_DETECTIONS_MINIMUM,
            "total_gate_redundant_with_per_fault_gates": True,
            "overall_median_restricted_delay_samples": overall_median,
            "overall_median_is_report_only": True,
            "per_fault_median_delay_maximum_gate": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "per_fault_p90_maximum_gate": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "miss_delay_imputation_samples": MISS_DELAY,
            "theoretical_minimum_delay_samples": THEORETICAL_MINIMUM_DELAY,
        },
        "parity_gate_passed": parity_pass,
        "governance_interpretation": (
            "sequential exploratory evidence after failed q96 final confirmation; "
            "not an independent confirmatory result"
        ),
    }


def _pack_paths(
    runs: tuple[ExternalRiethRun, ...],
    paths: Mapping[str, DirectionalGLRPath],
    *,
    operational_threshold: float,
) -> dict[str, np.ndarray]:
    ordered = [paths[run.name] for run in runs]
    return {
        "run_names": np.asarray([run.name for run in runs]),
        "partitions": np.asarray([str(run.metadata["partition"]) for run in runs]),
        "fault_ids": np.asarray([int(run.metadata["fault_id"]) for run in runs], dtype=np.int64),
        "simulation_runs": np.asarray(
            [int(run.metadata["simulation_run"]) for run in runs], dtype=np.int64
        ),
        "logits": np.stack([path.logits for path in ordered]),
        "standardized_logits": np.stack([path.standardized_logits for path in ordered]),
        "clipped_evidence": np.stack([path.clipped_evidence for path in ordered]),
        "cusum": np.stack([path.cusum for path in ordered]),
        "component_persistence": np.stack([path.component_persistence for path in ordered]),
        "envelope": np.stack([path.envelope for path in ordered]),
        "eligible": np.stack([path.eligible for path in ordered]),
        "alarm": np.stack([path.alarm for path in ordered]),
        "operational_threshold": np.asarray([operational_threshold], dtype=np.float64),
        "fault_start_zero_based": np.asarray([FAULT_START], dtype=np.int64),
    }


def _validate_evidence(
    path: Path, *, phase: str, operational_threshold: float
) -> dict[str, np.ndarray]:
    required = {
        "run_names", "partitions", "fault_ids", "simulation_runs", "logits",
        "standardized_logits", "clipped_evidence", "cusum",
        "component_persistence", "envelope", "eligible", "alarm",
        "operational_threshold", "fault_start_zero_based",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"{phase} evidence key set changed")
        if any(archive[name].dtype.kind == "O" for name in archive.files):
            raise ValueError("object arrays are forbidden in phase evidence")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if phase not in {"calibration", "audit"}:
        raise ValueError("unsupported evidence phase")
    count = 100 if phase == "calibration" else 400
    vector_fields = ("envelope", "eligible", "alarm")
    component_fields = (
        "logits", "standardized_logits", "clipped_evidence", "cusum",
        "component_persistence",
    )
    for name in vector_fields:
        if arrays[name].shape != (count, SEQUENCE_LENGTH):
            raise ValueError(f"{phase} {name} shape changed")
    for name in component_fields:
        if arrays[name].shape != (count, SEQUENCE_LENGTH, len(FAULT_IDS)):
            raise ValueError(f"{phase} {name} shape changed")
    for name in ("logits", "standardized_logits", "clipped_evidence", "cusum",
                 "component_persistence", "envelope", "operational_threshold"):
        if arrays[name].dtype != np.float64:
            raise ValueError(f"{phase} {name} must be binary64")
    if arrays["alarm"].dtype != np.bool_ or arrays["eligible"].dtype != np.bool_:
        raise ValueError("phase decision arrays must be boolean")
    if arrays["fault_ids"].dtype != np.int64 or arrays["simulation_runs"].dtype != np.int64:
        raise ValueError("phase identities must be int64")
    if arrays["fault_start_zero_based"].dtype != np.int64 or not np.array_equal(
        arrays["fault_start_zero_based"], np.asarray([FAULT_START], dtype=np.int64)
    ):
        raise ValueError("phase evidence fault onset changed")
    if arrays["operational_threshold"].shape != (1,) or (
        float(arrays["operational_threshold"][0]) != float(operational_threshold)
    ):
        raise ValueError("phase evidence operational threshold changed")
    if not math.isfinite(float(operational_threshold)) or operational_threshold < 0.0:
        raise ValueError("phase evidence operational threshold is invalid")
    expected_eligible = np.zeros((count, SEQUENCE_LENGTH), dtype=bool)
    expected_eligible[:, FIRST_ALARM_ELIGIBLE:] = True
    if not np.array_equal(arrays["eligible"], expected_eligible):
        raise ValueError("phase evidence eligibility changed")
    for name in ("logits", "standardized_logits", "clipped_evidence", "cusum"):
        if not np.isnan(arrays[name][:, :VALID_START]).all() or not np.isfinite(
            arrays[name][:, VALID_START:]
        ).all():
            raise ValueError(f"{phase} {name} leading/eligible numeric structure changed")
    if not np.isneginf(arrays["component_persistence"][:, :FIRST_ALARM_ELIGIBLE]).all():
        raise ValueError(f"{phase} component-persistence prefix changed")
    if not np.isfinite(
        arrays["component_persistence"][:, FIRST_ALARM_ELIGIBLE:]
    ).all():
        raise ValueError(f"{phase} component-persistence eligible values changed")
    if not np.isneginf(arrays["envelope"][:, :FIRST_ALARM_ELIGIBLE]).all() or not np.isfinite(
        arrays["envelope"][:, FIRST_ALARM_ELIGIBLE:]
    ).all():
        raise ValueError(f"{phase} envelope numeric structure changed")
    if phase == "audit" and not np.array_equal(
        arrays["alarm"], arrays["eligible"] & (arrays["envelope"] > operational_threshold)
    ):
        raise ValueError("audit alarm is not exact strict-threshold replay")
    if phase == "calibration":
        expected_faults = np.zeros(100, dtype=np.int64)
        expected_runs = np.asarray(CALIBRATION_RUNS, dtype=np.int64)
        expected_partitions = np.asarray(["fault_free_training"] * 100)
        expected_names = np.asarray(
            [f"Rieth-fault_free_training-fault-00-run-{run:03d}" for run in CALIBRATION_RUNS]
        )
        if np.any(arrays["alarm"]):
            raise ValueError("uncalibrated evidence must not contain alarm decisions")
    elif phase == "audit":
        expected_faults = np.asarray(
            [0] * 100 + [fault for fault in FAULT_IDS for _ in AUDIT_RUNS], dtype=np.int64
        )
        expected_runs = np.asarray(list(AUDIT_RUNS) * 4, dtype=np.int64)
        expected_partitions = np.asarray(
            ["fault_free_training"] * 100 + ["faulty_training"] * 300
        )
        expected_names = np.asarray(
            [f"Rieth-fault_free_training-fault-00-run-{run:03d}" for run in AUDIT_RUNS]
            + [
                f"Rieth-faulty_training-fault-{fault:02d}-run-{run:03d}"
                for fault in FAULT_IDS for run in AUDIT_RUNS
            ]
        )
    for name in ("run_names", "partitions"):
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in {"U", "S"}:
            raise ValueError(f"{phase} {name} string-vector schema changed")
    if not np.array_equal(arrays["run_names"], expected_names):
        raise ValueError(f"{phase} run-name identity/order changed")
    if not np.array_equal(arrays["fault_ids"], expected_faults):
        raise ValueError(f"{phase} fault identity/order changed")
    if not np.array_equal(arrays["simulation_runs"], expected_runs):
        raise ValueError(f"{phase} simulation-run identity/order changed")
    if not np.array_equal(arrays["partitions"], expected_partitions):
        raise ValueError(f"{phase} partition identity/order changed")
    return arrays


def _audit_records_from_evidence(
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct every run-level decision used by the audit result."""

    healthy: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    timeline = np.arange(SEQUENCE_LENGTH)
    for index, raw_name in enumerate(arrays["run_names"].tolist()):
        name = str(raw_name)
        fault_id = int(arrays["fault_ids"][index])
        simulation_run = int(arrays["simulation_runs"][index])
        alarm = np.asarray(arrays["alarm"][index], dtype=bool)
        envelope = np.asarray(arrays["envelope"][index], dtype=np.float64)
        eligible = np.asarray(arrays["eligible"][index], dtype=bool)
        indices = np.flatnonzero(alarm)
        if fault_id == 0:
            healthy.append(
                {
                    "run": name,
                    "simulation_run": simulation_run,
                    "any_alarm": bool(indices.size),
                    "alarm_episode_count": _alarm_episode_count(alarm),
                    "first_alarm_index": int(indices[0]) if indices.size else None,
                    "maximum_envelope": float(np.max(envelope[eligible])),
                }
            )
            continue
        post = np.flatnonzero(alarm & (timeline >= FAULT_START))
        first = int(post[0]) if post.size else None
        attributed = None
        if first is not None:
            attributed = int(
                FAULT_IDS[int(np.argmax(arrays["component_persistence"][index, first]))]
            )
        delay = first - FAULT_START if first is not None else None
        faults.append(
            {
                "run": name,
                "fault_id": fault_id,
                "simulation_run": simulation_run,
                "detected": first is not None,
                "first_fault_alarm_index": first,
                "delay_samples": delay,
                "restricted_delay_samples_miss_at_run_end": (
                    delay if delay is not None else MISS_DELAY
                ),
                "pre_fault_alarm_episode_count": _alarm_episode_count(alarm[:FAULT_START]),
                "pre_fault_interval_used_as_performance_evidence": False,
                "attributed_fault_head_at_first_alarm": attributed,
                "attribution_matches_fault": attributed == fault_id if attributed else False,
                "maximum_post_fault_envelope": float(
                    np.max(envelope[max(FAULT_START, FIRST_ALARM_ELIGIBLE):])
                ),
            }
        )
    if len(healthy) != 100 or len(faults) != 300:
        raise ValueError("audit evidence did not reconstruct exactly 100 healthy/300 fault runs")
    return healthy, faults


def _artifact_manifest(
    root: Path, phase: str, protocol_sha256: str, artifacts: Sequence[str]
) -> dict[str, Any]:
    names = tuple(name for name in artifacts if name != "artifact_manifest.json")
    body = {
        "schema": "hybrid_pac_rmax_external_artifact_manifest",
        "schema_version": 1,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "artifacts": {name: _hash_record(root / name) for name in names},
    }
    return {**body, "manifest_fingerprint": _canonical_sha256(body)}


def _completion(
    root: Path,
    phase: str,
    protocol_sha256: str,
    decision: str,
    artifacts: Sequence[str],
) -> dict[str, Any]:
    phase_manifest, phase_fingerprint = _verify_fingerprinted_json(
        root / "phase_manifest.json", "phase_fingerprint"
    )
    if phase_manifest.get("phase") != phase:
        raise ValueError("phase manifest phase differs during completion")
    body = {
        "schema": "hybrid_pac_rmax_external_completion",
        "schema_version": 1,
        "phase": phase,
        "phase_fingerprint": phase_fingerprint,
        "protocol_sha256": protocol_sha256,
        "decision": decision,
        "artifacts": {name: _hash_record(root / name) for name in artifacts},
    }
    return {**body, "completion_fingerprint": _canonical_sha256(body)}


def _verify_completed_phase(
    target: Path,
    *,
    phase: str,
    protocol_sha256: str,
    artifacts: Sequence[str],
    decisions: set[str],
) -> dict[str, Any] | None:
    if not target.exists():
        return None
    if not target.is_dir() or not (target / "completion.json").is_file():
        raise RuntimeError(f"partial {phase} output exists; RData must not be reopened")
    completion, fingerprint = _verify_fingerprinted_json(
        target / "completion.json", "completion_fingerprint"
    )
    if completion.get("schema") != "hybrid_pac_rmax_external_completion":
        raise ValueError("phase completion schema changed")
    if completion.get("phase") != phase or completion.get("protocol_sha256") != protocol_sha256:
        raise ValueError("phase completion binding changed")
    if completion.get("decision") not in decisions:
        raise ValueError("phase completion decision changed")
    records = completion.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(artifacts):
        raise ValueError("phase completion artifact set changed")
    expected_files = set(artifacts) | {"completion.json"}
    if {path.name for path in target.iterdir()} != expected_files:
        raise ValueError("phase directory contains missing or extra files")
    for name, expected in records.items():
        if _hash_record(target / name) != expected:
            raise ValueError(f"completed {phase} artifact changed: {name}")
    manifest, _ = _verify_fingerprinted_json(
        target / "artifact_manifest.json", "manifest_fingerprint"
    )
    if manifest.get("phase") != phase or manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("artifact manifest binding changed")
    if manifest != _artifact_manifest(target, phase, protocol_sha256, artifacts):
        raise ValueError("artifact manifest no longer exactly describes the completed phase")
    phase_manifest, phase_fingerprint = _verify_fingerprinted_json(
        target / "phase_manifest.json", "phase_fingerprint"
    )
    if phase_manifest.get("phase") != phase or phase_manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("phase manifest binding changed")
    if completion.get("phase_fingerprint") != phase_fingerprint:
        raise ValueError("completion/phase-manifest fingerprints differ")
    if _sha256_file(target / "protocol_snapshot.json") != protocol_sha256:
        raise ValueError("phase protocol snapshot changed")
    return {**completion, "completion_fingerprint": fingerprint}


def _verify_calibration_bundle(
    target: Path, protocol: Mapping[str, Any], protocol_sha256: str
) -> dict[str, Any] | None:
    completion = _verify_completed_phase(
        target,
        phase="calibration",
        protocol_sha256=protocol_sha256,
        artifacts=CALIBRATION_ARTIFACTS,
        decisions={CALIBRATION_DECISION},
    )
    if completion is None:
        return None
    result = _strict_json(target / "calibration_result.json")
    certificate = _strict_json(target / "pac_certificate.json")
    threshold = float(result.get("threshold"))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("calibration threshold is invalid")
    if result.get("decision") != CALIBRATION_DECISION:
        raise ValueError("calibration result decision changed")
    arrays = _validate_evidence(
        target / "calibration_evidence.npz",
        phase="calibration",
        operational_threshold=threshold,
    )
    maxima = [
        (str(name), float(np.max(values[FIRST_ALARM_ELIGIBLE:])))
        for name, values in zip(arrays["run_names"].tolist(), arrays["envelope"])
    ]
    binding = protocol["monitor_binding"]
    recomputed = build_calibration_certificate(
        deployment_spec_sha256=str(binding["deployment_spec_sha256"]),
        monitor_configuration_sha256=str(binding["monitor_configuration_sha256"]),
        run_maxima=maxima,
        deferred_runs=(),
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )
    if recomputed.as_dict() != certificate or recomputed.threshold != threshold:
        raise ValueError("calibration evidence/certificate/result disagree")
    expected_result_fields = {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "phase": "operational_pac_calibration",
        "decision": CALIBRATION_DECISION,
        "method": METHOD,
        "protocol_sha256": protocol_sha256,
        "threshold": threshold,
        "threshold_binary64_hex": threshold.hex(),
        "threshold_source": "CPU_NumPy_reference",
        "comparison": "strictly_greater_than",
        "rank_one_based": recomputed.rank.rank_one_based,
        "rank_rule": "second_largest_order_statistic_no_interpolation",
        "rank_driver_run": recomputed.rank_driver_run,
        "achieved_confidence": recomputed.rank.achieved_confidence,
        "upper_tail_probability_bound": recomputed.rank.upper_tail_probability_bound,
        "minimum_sample_count": recomputed.rank.minimum_sample_count,
        "calibration_run_count": 100,
        "calibration_exceedance_runs_under_strict_comparison": sum(
            maximum > threshold for _, maximum in recomputed.run_maxima
        ),
        "posthoc_design_threshold_reused": False,
        "threshold_derived_exclusively_from_operational_calibration_evidence": True,
        "audit_opened": False,
    }
    if any(result.get(key) != value for key, value in expected_result_fields.items()):
        raise ValueError("calibration result schema/provenance fields changed")
    parity = _strict_json(target / "parity.json")
    if (
        parity.get("run_count") != 100
        or parity.get("atol") != CPU_GPU_TOLERANCE
        or parity.get("all_numeric_paths_within_tolerance") is not True
        or parity.get("all_eligibility_arrays_exact") is not True
        or parity.get("all_alarm_arrays_exact") is not True
        or parity.get("passed") is not True
        or parity.get("q99_driver_run_exact") is not True
        or parity.get("q99_rank_exact") is not True
        or float(parity.get("cpu_q99_threshold")) != threshold
        or parity.get("cpu_q99_driver_run") != recomputed.rank_driver_run
        or float(parity.get("q99_threshold_max_abs_error")) > CPU_GPU_TOLERANCE
        or float(parity.get("maximum_per_run_maximum_abs_error")) > CPU_GPU_TOLERANCE
    ):
        raise ValueError("calibration CPU/CUDA parity artifact changed")
    return {
        "completion": completion,
        "completion_sha256": _sha256_file(target / "completion.json"),
        "certificate": certificate,
        "certificate_sha256": _sha256_file(target / "pac_certificate.json"),
        "evidence_sha256": _sha256_file(target / "calibration_evidence.npz"),
        "threshold": threshold,
        "threshold_binary64_hex": threshold.hex(),
    }


def _calibration_binding(calibration: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(calibration["threshold"])
    return {
        "calibration_completion_sha256": calibration["completion_sha256"],
        "calibration_completion_fingerprint": calibration["completion"][
            "completion_fingerprint"
        ],
        "pac_certificate_sha256": calibration["certificate_sha256"],
        "calibration_evidence_sha256": calibration["evidence_sha256"],
        "operational_threshold": threshold,
        "operational_threshold_binary64_hex": threshold.hex(),
        "threshold_source": "CPU_NumPy_reference",
        "calibration_decision": CALIBRATION_DECISION,
    }


def _verify_audit_bundle(
    target: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    calibration_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    threshold = float(calibration_binding["operational_threshold"])
    completion = _verify_completed_phase(
        target,
        phase="audit",
        protocol_sha256=protocol_sha256,
        artifacts=AUDIT_ARTIFACTS,
        decisions={PASS_DECISION, FAIL_DECISION},
    )
    if completion is None:
        return None
    result = _strict_json(target / "result.json")
    if result.get("decision") != completion.get("decision"):
        raise ValueError("audit result/completion decisions differ")
    if float(result.get("threshold")) != threshold:
        raise ValueError("audit result threshold differs from calibration")
    arrays = _validate_evidence(
        target / "audit_evidence.npz", phase="audit", operational_threshold=threshold
    )
    parity = _strict_json(target / "parity.json")
    maximum_errors = parity.get("maximum_errors")
    if (
        parity.get("run_count") != 400
        or parity.get("atol") != CPU_GPU_TOLERANCE
        or parity.get("all_numeric_paths_within_tolerance") is not True
        or parity.get("all_eligibility_arrays_exact") is not True
        or parity.get("all_alarm_arrays_exact") is not True
        or not isinstance(maximum_errors, dict)
        or set(maximum_errors) != {
            "logits_max_abs_error", "standardized_logits_max_abs_error",
            "clipped_evidence_max_abs_error", "cusum_max_abs_error",
            "component_persistence_max_abs_error", "envelope_max_abs_error",
        }
        or any(
            not math.isfinite(float(value)) or float(value) > CPU_GPU_TOLERANCE
            for value in maximum_errors.values()
        )
    ):
        raise ValueError("audit CPU/CUDA parity artifact changed")
    healthy_records, fault_records = _audit_records_from_evidence(arrays)
    recomputed = _summarize_audit(healthy_records, fault_records, parity, threshold)
    recomputed.update(
        {
            "protocol_sha256": protocol_sha256,
            "calibration_binding": dict(calibration_binding),
            "pac_primary_claim": protocol["success_gates"]["pac_primary_claim"],
        }
    )
    if result != recomputed:
        raise ValueError("audit result is not an exact replay of saved evidence and gates")
    return completion


def _calibration_report(result: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PAC-RMax operational calibration",
            "",
            f"Decision: **{result['decision']}**",
            "",
            "This is the first operational threshold calculation for this revised method.",
            "The q99 rule was selected after the earlier q96 final failure, but the",
            "old post-hoc value 254.4559644524996 was not reused.",
            "",
            f"- CPU NumPy q99 threshold: {result['threshold']:.16g}.",
            f"- CUDA q99 threshold: {result['cuda_threshold']:.16g}.",
            f"- Rank: {result['rank_one_based']}/100 (second largest, no interpolation).",
            f"- Driver: {result['rank_driver_run']}.",
            f"- Certificate confidence: {result['achieved_confidence']:.12g}.",
            f"- CPU/CUDA threshold error: {result['cpu_cuda_threshold_abs_error']:.3e}.",
            f"- CUDA device: {execution['runtime']['device_name']}.",
            "- pyreadr decoded the whole fault-free training file before selecting runs 301--400.",
            "",
        ]
    )


def _audit_report(result: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    lines = [
        "# Hybrid Directional–Energy PAC-RMax GLR-CUSUM",
        "",
        "## Governance",
        "",
        "This is a post-failure sequential external-partition audit, not a pristine",
        "or independent final confirmation. The q99 rule was chosen after observing",
        "the earlier q96 failure; its operational value came only from completed",
        "fault-free training runs 301--400 before this audit was opened.",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Quantity | Observed | Frozen gate |",
        "|---|---:|---:|",
        (
            f"| Healthy alarm runs | {result['healthy']['runs_with_any_alarm']}/100 "
            f"| <= {HEALTHY_ALARM_RUNS_MAXIMUM} |"
        ),
        (
            f"| Healthy one-sided CP95 upper | "
            f"{result['healthy']['one_sided_clopper_pearson_95_upper']:.6f} | < 0.10 via count gate |"
        ),
        (
            f"| Total detected faults | {result['faults']['detected_runs']}/300 "
            f"| >= {TOTAL_FAULT_DETECTIONS_MINIMUM} |"
        ),
        (
            "| Overall median restricted delay | "
            f"{result['faults']['overall_median_restricted_delay_samples']:.3f} "
            "| report only |"
        ),
    ]
    for fault_id in FAULT_IDS:
        item = result["faults"]["per_fault"][f"{fault_id:02d}"]
        lines.extend(
            [
                f"| Fault {fault_id} detected | {item['detected_runs']}/100 | >= 90 |",
                (
                    f"| Fault {fault_id} restricted median delay | "
                    f"{item['median_restricted_delay_samples']:.3f} "
                    f"| <= {MEDIAN_RESTRICTED_DELAY_MAXIMUM:.0f} |"
                ),
                (
                    f"| Fault {fault_id} restricted p90 delay | "
                    f"{item['p90_restricted_delay_samples_higher_order_statistic']:.3f} "
                    f"| <= {PER_FAULT_RESTRICTED_P90_MAXIMUM:.0f} |"
                ),
            ]
        )
    lines.extend(
        [
            f"| Full CPU/CUDA alarm arrays exact | {result['parity_gate_passed']} | True |",
            "",
            f"- Operational CPU-reference threshold: {result['threshold']:.16g} (strict >).",
            f"- CUDA device: {execution['runtime']['device_name']}.",
            f"- Score wall time: {execution['score']['total_wall_seconds']:.3f} s.",
            f"- The earliest eligible detection delay is {THEORETICAL_MINIMUM_DELAY} samples.",
            "- Pre-fault samples 0--19 are structural context, not performance evidence.",
            "- Misses receive the predeclared restricted delay of 480 samples.",
            "- Both authorized RData files were decoded wholesale by pyreadr.",
            "- No audit-row training, refitting, selection, or retuning is authorized.",
            "",
        ]
    )
    return "\n".join(lines)


def _finalize_phase(
    *,
    staging: Path,
    target: Path,
    phase: str,
    protocol_sha256: str,
    decision: str,
    artifacts: Sequence[str],
) -> None:
    _write_json(
        staging / "artifact_manifest.json",
        _artifact_manifest(staging, phase, protocol_sha256, artifacts),
    )
    _write_json(
        staging / "completion.json",
        _completion(staging, phase, protocol_sha256, decision, artifacts),
    )
    if target.exists():
        raise FileExistsError(f"{phase} target appeared during atomic finalization")
    os.replace(staging, target)


def _load_method(project_root: Path) -> tuple[Any, Any]:
    hybrid_dir = project_root / HYBRID_DIRECTORY
    transformer, _ = load_hybrid_feature_transformer(hybrid_dir / TRANSFORMER_NAME)
    model, _ = load_directional_glr_model(hybrid_dir / MODEL_NAME)
    if model.threshold is not None:
        raise ValueError("only the frozen uncalibrated model may enter this runner")
    return transformer, model


def _run_calibration(
    *,
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    target = output_root / "calibration"
    verified = _verify_calibration_bundle(target, protocol, protocol_sha256)
    if verified is not None:
        print("verified completed calibration; no RData source was accessed", flush=True)
        return verified
    target, staging = _phase_staging(output_root, "calibration")
    try:
        _copy_protocol_snapshot(protocol_path, staging)
        transformer, uncalibrated = _load_method(project_root)
        runtime = _runtime_preflight(device)
        _write_json(staging / "runtime_preflight.json", runtime)
        archive = SafeRiethTrainingArchive(data_dir)
        source_records = archive.verify_sources()
        intent = _calibration_access_intent(protocol_sha256, protocol)
        _write_json(staging / "access_intent.json", intent)
        intent_sha256 = _sha256_file(staging / "access_intent.json")
        phase_manifest = _phase_manifest(
            phase="calibration",
            protocol_sha256=protocol_sha256,
            intent_sha256=intent_sha256,
            runtime_sha256=_sha256_file(staging / "runtime_preflight.json"),
            source_records=source_records,
            calibration_binding=None,
        )
        _write_json(staging / "phase_manifest.json", phase_manifest)
        grant = load_access_grant(
            staging / "access_intent.json",
            expected_protocol_sha256=protocol_sha256,
            expected_intent_sha256=intent_sha256,
        )
        print("calibration: whole-file decode FF-training, selecting runs 301..400", flush=True)
        runs = archive.load_runs(
            grant,
            "fault_free_training",
            {0: CALIBRATION_RUNS},
            expected_length=SEQUENCE_LENGTH,
            fault_start=FAULT_START,
        )
        grant.assert_fully_consumed()
        if len(runs) != 100:
            raise RuntimeError("calibration loader did not return exactly 100 runs")
        data_access = {
            "analysis_status": ANALYSIS_STATUS,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": _sha256_file(staging / "phase_manifest.json"),
            "source_records": source_records,
            "numeric_decode_count": 1,
            "decoded_partitions": ["fault_free_training"],
            "selected_runs": [301, 400],
            "decode_disclosure": FULL_DECODE_DISCLOSURE,
            "audit_rows_not_selected_transformed_or_scored": True,
            "faulty_training_not_decoded": True,
        }
        _write_json(staging / "data_access_audit.json", data_access)
        print("calibration: CUDA float64 scoring plus full CPU reference parity", flush=True)
        gpu_paths, cpu_paths, base_parity, score_execution = _score_with_full_parity(
            transformer, uncalibrated, runs, device=device, batch_size=batch_size
        )
        certificate, parity = _calibration_parity(
            cpu_paths, gpu_paths, protocol, base_parity
        )
        threshold = float(certificate.threshold)
        _write_json(staging / "parity.json", parity)
        _write_json(staging / "pac_certificate.json", certificate.as_dict())
        result = {
            "schema_version": 1,
            "analysis_status": ANALYSIS_STATUS,
            "phase": "operational_pac_calibration",
            "decision": CALIBRATION_DECISION,
            "method": METHOD,
            "protocol_sha256": protocol_sha256,
            "threshold": threshold,
            "threshold_binary64_hex": threshold.hex(),
            "threshold_source": "CPU_NumPy_reference",
            "cuda_threshold": float(parity["cuda_q99_threshold"]),
            "cpu_cuda_threshold_abs_error": float(parity["q99_threshold_max_abs_error"]),
            "comparison": "strictly_greater_than",
            "rank_one_based": certificate.rank.rank_one_based,
            "rank_rule": "second_largest_order_statistic_no_interpolation",
            "rank_driver_run": certificate.rank_driver_run,
            "achieved_confidence": certificate.rank.achieved_confidence,
            "upper_tail_probability_bound": certificate.rank.upper_tail_probability_bound,
            "minimum_sample_count": certificate.rank.minimum_sample_count,
            "calibration_run_count": 100,
            "calibration_exceedance_runs_under_strict_comparison": sum(
                maximum > threshold for _, maximum in certificate.run_maxima
            ),
            "posthoc_design_threshold_reused": False,
            "threshold_derived_exclusively_from_operational_calibration_evidence": True,
            "audit_opened": False,
        }
        execution = {
            "analysis_status": ANALYSIS_STATUS,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "runtime": runtime,
            "score": score_execution,
        }
        _write_json(staging / "calibration_result.json", result)
        _write_json(staging / "gpu_execution.json", execution)
        _write_npz(
            staging / "calibration_evidence.npz",
            _pack_paths(runs, cpu_paths, operational_threshold=threshold),
        )
        _validate_evidence(
            staging / "calibration_evidence.npz",
            phase="calibration",
            operational_threshold=threshold,
        )
        _write_text(staging / "REPORT.md", _calibration_report(result, execution))
        _finalize_phase(
            staging=staging,
            target=target,
            phase="calibration",
            protocol_sha256=protocol_sha256,
            decision=CALIBRATION_DECISION,
            artifacts=CALIBRATION_ARTIFACTS,
        )
        verified = _verify_calibration_bundle(target, protocol, protocol_sha256)
        if verified is None:
            raise RuntimeError("calibration completion disappeared")
        print(f"calibration frozen: CPU q99={threshold:.17g}", flush=True)
        return verified
    except Exception:
        print(f"calibration not finalized; staging retained at {staging}", file=sys.stderr, flush=True)
        raise


def _run_audit(
    *,
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    calibration = _verify_calibration_bundle(
        output_root / "calibration", protocol, protocol_sha256
    )
    if calibration is None:
        raise RuntimeError("audit cannot open before a completed PAC calibration")
    threshold = float(calibration["threshold"])
    calibration_binding = _calibration_binding(calibration)
    target = output_root / "audit"
    verified = _verify_audit_bundle(
        target, protocol, protocol_sha256, calibration_binding
    )
    if verified is not None:
        print("verified completed audit; no RData source was accessed", flush=True)
        return verified
    target, staging = _phase_staging(output_root, "audit")
    try:
        _copy_protocol_snapshot(protocol_path, staging)
        transformer, uncalibrated = _load_method(project_root)
        model = uncalibrated.with_threshold(threshold)
        runtime = _runtime_preflight(device)
        _write_json(staging / "runtime_preflight.json", runtime)
        archive = SafeRiethTrainingArchive(data_dir)
        source_records = archive.verify_sources()
        intent = _audit_access_intent(protocol_sha256, calibration_binding)
        _write_json(staging / "access_intent.json", intent)
        intent_sha256 = _sha256_file(staging / "access_intent.json")
        phase_manifest = _phase_manifest(
            phase="audit",
            protocol_sha256=protocol_sha256,
            intent_sha256=intent_sha256,
            runtime_sha256=_sha256_file(staging / "runtime_preflight.json"),
            source_records=source_records,
            calibration_binding=calibration_binding,
        )
        _write_json(staging / "phase_manifest.json", phase_manifest)
        grant = load_access_grant(
            staging / "access_intent.json",
            expected_protocol_sha256=protocol_sha256,
            expected_intent_sha256=intent_sha256,
        )
        print("audit: whole-file decode FF-training, selecting healthy runs 401..500", flush=True)
        healthy = archive.load_runs(
            grant,
            "fault_free_training",
            {0: AUDIT_RUNS},
            expected_length=SEQUENCE_LENGTH,
            fault_start=FAULT_START,
        )
        gc.collect()
        print("audit: whole-file decode Faulty-training, selecting faults 3/9/15 runs 401..500", flush=True)
        faults = archive.load_runs(
            grant,
            "faulty_training",
            {fault_id: AUDIT_RUNS for fault_id in FAULT_IDS},
            expected_length=SEQUENCE_LENGTH,
            fault_start=FAULT_START,
        )
        grant.assert_fully_consumed()
        if len(healthy) != 100 or len(faults) != 300:
            raise RuntimeError("audit loader returned the wrong run count")
        data_access = {
            "analysis_status": ANALYSIS_STATUS,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": _sha256_file(staging / "phase_manifest.json"),
            "calibration_binding": calibration_binding,
            "source_records": source_records,
            "numeric_decode_count": 2,
            "decoded_partition_order": ["fault_free_training", "faulty_training"],
            "selected_healthy_runs": [401, 500],
            "selected_fault_ids": list(FAULT_IDS),
            "selected_fault_runs": [401, 500],
            "decode_disclosure": FULL_DECODE_DISCLOSURE,
            "audit_opened_only_after_calibration_completion": True,
            "threshold_selected_from_audit_rows": False,
        }
        _write_json(staging / "data_access_audit.json", data_access)
        all_runs = healthy + faults
        print("audit: CUDA float64 scoring 400 runs plus full CPU parity", flush=True)
        gpu_paths, _cpu_paths, parity, score_execution = _score_with_full_parity(
            transformer, model, all_runs, device=device, batch_size=batch_size
        )
        if not parity["all_alarm_arrays_exact"]:
            raise AssertionError("full 400-run CPU/CUDA alarm-array parity failed")
        _write_json(staging / "parity.json", parity)
        healthy_records = _healthy_records(healthy, gpu_paths)
        fault_records = _fault_records(faults, gpu_paths)
        result = _summarize_audit(healthy_records, fault_records, parity, threshold)
        result.update(
            {
                "protocol_sha256": protocol_sha256,
                "calibration_binding": calibration_binding,
                "pac_primary_claim": protocol["success_gates"]["pac_primary_claim"],
            }
        )
        execution = {
            "analysis_status": ANALYSIS_STATUS,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "runtime": runtime,
            "score": score_execution,
        }
        _write_json(staging / "result.json", result)
        _write_json(staging / "gpu_execution.json", execution)
        _write_npz(
            staging / "audit_evidence.npz",
            _pack_paths(all_runs, gpu_paths, operational_threshold=threshold),
        )
        _validate_evidence(
            staging / "audit_evidence.npz",
            phase="audit",
            operational_threshold=threshold,
        )
        _write_text(staging / "REPORT.md", _audit_report(result, execution))
        _finalize_phase(
            staging=staging,
            target=target,
            phase="audit",
            protocol_sha256=protocol_sha256,
            decision=str(result["decision"]),
            artifacts=AUDIT_ARTIFACTS,
        )
        verified = _verify_audit_bundle(
            target, protocol, protocol_sha256, calibration_binding
        )
        if verified is None:
            raise RuntimeError("audit completion disappeared")
        print(f"external audit decision: {result['decision']}", flush=True)
        return verified
    except Exception:
        print(f"audit not finalized; staging retained at {staging}", file=sys.stderr, flush=True)
        raise


def run_two_stage_audit(
    *,
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    phase: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    protocol, protocol_sha256 = _load_frozen_protocol(
        project_root, protocol_path, protocol_sha256
    )
    latest: dict[str, Any] = {}
    if phase in {"calibration", "all"}:
        latest = _run_calibration(
            project_root=project_root,
            data_dir=data_dir,
            output_root=output_root,
            protocol_path=protocol_path,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            device=device,
            batch_size=batch_size,
        )
    if phase in {"audit", "all"}:
        latest = _run_audit(
            project_root=project_root,
            data_dir=data_dir,
            output_root=output_root,
            protocol_path=protocol_path,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            device=device,
            batch_size=batch_size,
        )
    return latest


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "freeze":
        digest = freeze_protocol(project_root, args.protocol)
        print(f"frozen protocol SHA-256: {digest}", flush=True)
        return 0
    run_two_stage_audit(
        project_root=project_root,
        data_dir=args.data_dir,
        output_root=args.output_root,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        phase=args.phase,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
