"""Run the frozen q100 PAC hybrid GLR-CUSUM row-level holdout audit on CUDA.

This runner is a post-failure method revision.  It does not reinterpret the
failed q99 audit.  The revised rule (epsilon=.05, delta=.01, rank 100/100) and
its exact threshold are derived only from the already-consumed calibration
evidence, then frozen before this runner may decode the still-unselected
holdout rows.  This is a sequential within-Rieth row-level audit; partition
decoding and cross-partition seed coupling prevent an independence claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


def _requested_device(tokens: Sequence[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# This Windows environment requires CUDA/PyTorch to load before NumPy/MKL.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _torch_preload  # noqa: F401

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_hybrid_pac_rmax_external_gpu as ext  # noqa: E402
import run_directional_glr_gpu as base  # noqa: E402
from mate_spc.datasets import RiethTEPArchive, SensorSequence  # noqa: E402
from mate_spc_directional.glr import FAULT_IDS, DirectionalGLRPath  # noqa: E402
from mate_spc_pac.certificate import build_calibration_certificate  # noqa: E402


SCHEMA = "hybrid_directional_energy_pac_q100_within_rieth_row_level_holdout_audit"
SCHEMA_VERSION = 2
ANALYSIS_STATUS = "post_failure_sequential_within_Rieth_row_level_holdout_audit"
METHOD = "hybrid_directional_energy_pac_q100_glr_cusum"
PASS_DECISION = "PASS_POST_FAILURE_SEQUENTIAL_WITHIN_RIETH_ROW_LEVEL_HOLDOUT_AUDIT"
FAIL_DECISION = "FAIL_POST_FAILURE_SEQUENTIAL_WITHIN_RIETH_ROW_LEVEL_HOLDOUT_AUDIT_NO_RETUNING"
PHASE = ANALYSIS_STATUS
PHASE_DIRECTORY = "audit"

PAC_EPSILON = 0.05
PAC_DELTA = 0.01
PAC_RANK = 100
Q100_THRESHOLD = 236.3968052224969
Q100_THRESHOLD_HEX = "0x1.d8cb2a0ddb033p+7"
CALIBRATION_RUNS = tuple(range(301, 401))
HEALTHY_RUNS = tuple(range(121, 201))
FAULT_RUNS = tuple(range(301, 401))
FAULT_START = 160
SEQUENCE_LENGTH = 500
EXPECTED_HEALTHY_SOURCE_LENGTH = 500
EXPECTED_FAULT_SOURCE_LENGTH = 960
VALID_START = 63
FIRST_ALARM_ELIGIBLE = 65
MISS_DELAY = SEQUENCE_LENGTH - FAULT_START

HEALTHY_ALARM_RUNS_MAXIMUM = 3
PER_FAULT_DETECTIONS_MINIMUM = 90
TOTAL_FAULT_DETECTIONS_MINIMUM = 270
MEDIAN_RESTRICTED_DELAY_MAXIMUM = 120.0
PER_FAULT_RESTRICTED_P90_MAXIMUM = 180.0
CPU_GPU_TOLERANCE = 1e-10

HYBRID_DIRECTORY = Path("outputs/hybrid_energy_glr_gpu_v1")
EXTERNAL_DIRECTORY = Path("outputs/hybrid_pac_rmax_external_gpu_v1")
TRANSFORMER_NAME = "hybrid_transformer.npz"
MODEL_NAME = "hybrid_glr_uncalibrated.npz"
DESIGN_SCORES_NAME = "calibration_scores.npz"
CALIBRATION_EVIDENCE = Path("calibration/calibration_evidence.npz")
CALIBRATION_COMPLETION = Path("calibration/completion.json")
FAILED_AUDIT_COMPLETION = Path("audit/completion.json")
OLD_HYBRID_AUDIT_SCORES = Path("audit_scores.npz")
OLD_HYBRID_AUDIT_SCORES_SHA256 = "c4d66bbded5fe854dd3330bfe7a99891006377337c7c2d5a1e3a62b1c8527454"
EXTERNAL_AUDIT_EVIDENCE = Path("audit/audit_evidence.npz")
EXTERNAL_AUDIT_EVIDENCE_SHA256 = "5a91591b81b15269aae2f1b52047bedcf72cc5df67750052c65ce73a9ef14804"
Q100_DRIVER_RUN = "Rieth-fault_free_training-fault-00-run-338"
Q100_CALIBRATION_MAXIMA_SHA256 = "bf8f78cb77c781fc58a035dbbf08b71cfd926cd671fd35da9a1b30ac2a8b6c03"
ABORTED_V1_1_PROTOCOL = Path("protocols/hybrid_pac_q100_holdout_v1_1.json")
ABORTED_V1_1_PROTOCOL_SHA256 = "74e7370f4ea83cb6eaf5924afb22f8296a0da7f9b5e0d464b9c2b4fe01141521"
ABORTED_V1_1_FAILURE_RECORD = Path(
    "outputs/hybrid_pac_q100_holdout_gpu_v1_1/FAILED_BEFORE_RDATA_DECODE.json"
)
ABORTED_V1_1_FAILURE_RECORD_SHA256 = "c81560209fc0c3ead208c47de09da6e86035ea4b231ca85b4874632ef938f99b"

EXPECTED_ARTIFACT_SHA256 = {
    TRANSFORMER_NAME: "09cca42496360e3132239150b3bf5f5dcd761df61eb35ae7082465c187fb28ab",
    MODEL_NAME: "0e2bfb1e6cfad1d6016f4fb3413c8a517c24fbeb6648bb8aad0533f23c17858d",
    DESIGN_SCORES_NAME: "2139c9f05be6c7b1acfbf520b2a1c14411a9b515b4399352c30b62a599ad4151",
    CALIBRATION_EVIDENCE.as_posix(): "9494122e2332a655a90916b8fe8544b7b51029a29ef95392162f7875921786ab",
}

SOURCE_FILES = {
    "fault_free_training": {
        "file_name": "TEP_FaultFree_Training.RData",
        "md5": "ec126484534331f85001d8c4ebce6d17",
        "sha256": "12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940",
        "size_bytes": 24678017,
    },
    "faulty_testing": {
        "file_name": "TEP_Faulty_Testing.RData",
        "md5": "556bdb64c83021bc0c5f92e427753565",
        "sha256": "c2f1aabc77498efda4d5f9ad81ecc829e2fcd98752b84cf5b2a06558cabacc36",
        "size_bytes": 836882037,
    },
}

IMPLEMENTATION_FILES = (
    "scripts/run_hybrid_pac_q100_holdout_gpu.py",
    "tests/test_run_hybrid_pac_q100_holdout_gpu.py",
    "scripts/run_hybrid_pac_rmax_external_gpu.py",
    "scripts/run_directional_glr_gpu.py",
    "src/mate_spc/datasets.py",
    "src/mate_spc_hybrid/features.py",
    "src/mate_spc_directional/features.py",
    "src/mate_spc_directional/glr.py",
    "src/mate_spc_directional/gpu.py",
    "src/mate_spc_pac/certificate.py",
    "src/mate_spc_pac/common.py",
    "src/mate_spc_pac/probability.py",
)

ARTIFACTS = (
    "REPORT.md",
    "access_intent.json",
    "artifact_manifest.json",
    "audit_evidence.npz",
    "data_access_audit.json",
    "gpu_execution.json",
    "parity.json",
    "phase_manifest.json",
    "protocol_snapshot.json",
    "q100_certificate.json",
    "result.json",
    "runtime_preflight.json",
)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - authenticates the published file only
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _monitor_binding(artifact_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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
        "z_clip": [-8.0, 8.0],
        "page_cusum": "one_sided_reset_at_zero",
        "persistence": "3_of_3_per_head_then_family_maximum",
        "comparison": "strictly_greater_than",
        "threshold_rule": "PAC_RMax_maximum_q100_of_100_run_maxima",
        "epsilon": PAC_EPSILON,
        "delta": PAC_DELTA,
    }
    return {
        "deployment_spec": deployment,
        "deployment_spec_sha256": ext._canonical_sha256(deployment),
        "monitor_configuration": monitor,
        "monitor_configuration_sha256": ext._canonical_sha256(monitor),
    }


def _calibration_maxima(project_root: Path) -> tuple[tuple[str, float], ...]:
    path = project_root / EXTERNAL_DIRECTORY / CALIBRATION_EVIDENCE
    if ext._sha256_file(path) != EXPECTED_ARTIFACT_SHA256[CALIBRATION_EVIDENCE.as_posix()]:
        raise ValueError("the consumed calibration evidence changed")
    required = {"run_names", "partitions", "fault_ids", "simulation_runs", "envelope", "eligible"}
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("calibration evidence lacks q100 derivation fields")
        if any(archive[name].dtype.kind == "O" for name in archive.files):
            raise ValueError("object arrays are forbidden in calibration evidence")
        names = np.array(archive["run_names"], copy=True)
        partitions = np.array(archive["partitions"], copy=True)
        fault_ids = np.array(archive["fault_ids"], copy=True)
        simulation_runs = np.array(archive["simulation_runs"], copy=True)
        envelope = np.array(archive["envelope"], dtype=np.float64, copy=True)
        eligible = np.array(archive["eligible"], dtype=bool, copy=True)
    if names.shape != (100,) or envelope.shape != (100, SEQUENCE_LENGTH):
        raise ValueError("calibration evidence dimensions changed")
    if not np.array_equal(simulation_runs, np.asarray(CALIBRATION_RUNS, dtype=np.int64)):
        raise ValueError("calibration run identities changed")
    if not np.array_equal(fault_ids, np.zeros(100, dtype=np.int64)):
        raise ValueError("calibration is no longer all healthy")
    if not np.array_equal(partitions, np.asarray(["fault_free_training"] * 100)):
        raise ValueError("calibration partition changed")
    maxima: list[tuple[str, float]] = []
    for index, name in enumerate(names.tolist()):
        values = envelope[index, eligible[index]]
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("calibration eligible maxima are invalid")
        maxima.append((str(name), float(np.max(values))))
    return tuple(maxima)


def _q100_certificate(project_root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    certificate = build_calibration_certificate(
        deployment_spec_sha256=str(binding["deployment_spec_sha256"]),
        monitor_configuration_sha256=str(binding["monitor_configuration_sha256"]),
        run_maxima=_calibration_maxima(project_root),
        deferred_runs=(),
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )
    if certificate.rank.rank_one_based != PAC_RANK:
        raise ValueError("delta=.01 PAC certificate is not q100")
    if certificate.threshold != Q100_THRESHOLD or certificate.threshold.hex() != Q100_THRESHOLD_HEX:
        raise ValueError("q100 threshold differs from the frozen binary64 value")
    if certificate.rank_driver_run != Q100_DRIVER_RUN:
        raise ValueError("q100 rank-driver run changed")
    if certificate.calibration_maxima_sha256 != Q100_CALIBRATION_MAXIMA_SHA256:
        raise ValueError("q100 canonical calibration-maxima fingerprint changed")
    return certificate.as_dict()


def build_protocol_template(project_root: Path) -> dict[str, Any]:
    hybrid = project_root / HYBRID_DIRECTORY
    artifacts = {
        name: ext._hash_record(hybrid / name)
        for name in (TRANSFORMER_NAME, MODEL_NAME, DESIGN_SCORES_NAME)
    }
    for name, expected in EXPECTED_ARTIFACT_SHA256.items():
        if name in artifacts and artifacts[name]["sha256"] != expected:
            raise ValueError(f"frozen hybrid artifact changed: {name}")
    method_validation = ext._validate_frozen_method(project_root, artifacts)
    binding = _monitor_binding(artifacts)
    certificate = _q100_certificate(project_root, binding)
    implementation = {
        relative: ext._hash_record(project_root / relative) for relative in IMPLEMENTATION_FILES
    }
    calibration_completion = ext._verify_completion_directory(
        project_root / EXTERNAL_DIRECTORY / "calibration",
        ext.CALIBRATION_DECISION,
    )
    failed_audit_completion = ext._verify_completion_directory(
        project_root / EXTERNAL_DIRECTORY / "audit",
        ext.FAIL_DECISION,
    )
    old_hybrid_scores = ext._hash_record(project_root / HYBRID_DIRECTORY / OLD_HYBRID_AUDIT_SCORES)
    if old_hybrid_scores["sha256"] != OLD_HYBRID_AUDIT_SCORES_SHA256:
        raise ValueError("old hybrid audit scores changed")
    external_audit_evidence = ext._hash_record(
        project_root / EXTERNAL_DIRECTORY / EXTERNAL_AUDIT_EVIDENCE
    )
    if external_audit_evidence["sha256"] != EXTERNAL_AUDIT_EVIDENCE_SHA256:
        raise ValueError("external-training audit evidence changed")
    aborted_v1_1_protocol = ext._hash_record(project_root / ABORTED_V1_1_PROTOCOL)
    if aborted_v1_1_protocol["sha256"] != ABORTED_V1_1_PROTOCOL_SHA256:
        raise ValueError("aborted v1.1 protocol changed")
    aborted_v1_1_failure = ext._hash_record(project_root / ABORTED_V1_1_FAILURE_RECORD)
    if aborted_v1_1_failure["sha256"] != ABORTED_V1_1_FAILURE_RECORD_SHA256:
        raise ValueError("aborted v1.1 failure record changed")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "method": METHOD,
        "governance": {
            "created_after_q99_within_Rieth_audit_failed": True,
            "q100_rule_selected_after_q99_result_was_observed": True,
            "q99_failed_healthy_gate": "5/100 versus at most 4/100",
            "old_401_500_rows_are_not_reused": True,
            "selected_holdout_rows_have_never_been_selected_transformed_or_scored": True,
            "source_RData_partitions_were_previously_whole_file_decoded": True,
            "partition_level_unseen_claim": False,
            "row_level_unscored_claim": True,
            "statistical_independence_not_established": True,
            "common_random_number_coupling_across_partitions_not_established": True,
            "same_301_400_identifiers_in_calibration_and_fault_audit_do_not_establish_independence": True,
            "not_an_independent_external_or_final_confirmation": True,
            "interpretation": ANALYSIS_STATUS,
            "failure_action": "report failure and do not retune on these rows",
            "v1_1_infrastructure_failure": {
                "superseded_protocol": {
                    "path": ABORTED_V1_1_PROTOCOL.as_posix(),
                    **aborted_v1_1_protocol,
                },
                "failure_record": {
                    "path": ABORTED_V1_1_FAILURE_RECORD.as_posix(),
                    **aborted_v1_1_failure,
                },
                "exception": (
                    "AttributeError: module 'run_hybrid_pac_rmax_external_gpu' "
                    "has no attribute 'base'"
                ),
                "failed_expression": "ext.base._run_ids(archive.load_runs(...), ...)",
                "python_call_target_resolution_preceded_argument_evaluation": True,
                "archive_load_runs_call_count": 0,
                "source_raw_byte_hashing_and_MD5_verification_occurred": True,
                "pyreadr_import_or_read_r_occurred": False,
                "row_selection_feature_transformation_or_scoring_occurred": False,
                "holdout_numeric_results_observed": False,
                "v1_1_protocol_and_staging_preserved_not_reinterpreted_as_complete": True,
                "v1_2_change_scope": "implementation namespace wiring only",
                "algorithm_unchanged": True,
                "model_and_feature_transformer_unchanged": True,
                "q100_threshold_and_certificate_unchanged": True,
                "row_level_split_unchanged": True,
                "success_gates_unchanged": True,
            },
        },
        "frozen_method": {
            "directory": HYBRID_DIRECTORY.as_posix(),
            "artifacts": artifacts,
            **method_validation,
        },
        "revised_pac_calibration": {
            "source_partition": "fault_free_training",
            "source_runs": [301, 400],
            "source_values_already_consumed": True,
            "calibration_evidence": {
                "path": (EXTERNAL_DIRECTORY / CALIBRATION_EVIDENCE).as_posix(),
                **ext._hash_record(project_root / EXTERNAL_DIRECTORY / CALIBRATION_EVIDENCE),
            },
            "calibration_completion": calibration_completion,
            "failed_q99_audit_completion": failed_audit_completion,
            "epsilon": PAC_EPSILON,
            "delta": PAC_DELTA,
            "rank_one_based": PAC_RANK,
            "rank_rule": "maximum_order_statistic_no_interpolation",
            "threshold": Q100_THRESHOLD,
            "threshold_binary64_hex": Q100_THRESHOLD_HEX,
            "rank_driver_run": Q100_DRIVER_RUN,
            "canonical_calibration_maxima_sha256": Q100_CALIBRATION_MAXIMA_SHA256,
            "certificate": {k: v for k, v in certificate.items() if k != "run_maxima"},
            "threshold_frozen_before_new_holdout_numeric_access": True,
        },
        "posthoc_design_evidence": {
            "observed_before_this_revision_was_frozen": True,
            "materially_influenced_most_likely_to_succeed_selection": True,
            "old_hybrid_testing_audit_q100_replay": {
                "evidence_path": (HYBRID_DIRECTORY / OLD_HYBRID_AUDIT_SCORES).as_posix(),
                "evidence": old_hybrid_scores,
                "partition_and_runs": "fault_free_testing and faulty_testing runs 201-300",
                "healthy_alarm_runs": 1,
                "healthy_run_count": 100,
                "faults": {
                    "03": {"detected": 100, "run_count": 100, "median_delay": 59.0, "p90_delay": 70.0},
                    "09": {"detected": 100, "run_count": 100, "median_delay": 100.0, "p90_delay": 153.0},
                    "15": {"detected": 100, "run_count": 100, "median_delay": 84.0, "p90_delay": 127.0},
                },
                "pre_fault_alarm_episodes": 0,
            },
            "external_training_partition_audit_q100_replay": {
                "evidence_path": (EXTERNAL_DIRECTORY / EXTERNAL_AUDIT_EVIDENCE).as_posix(),
                "evidence": external_audit_evidence,
                "partition_and_runs": "fault_free_training and faulty_training runs 401-500",
                "healthy_alarm_runs": 4,
                "healthy_run_count": 100,
                "faults": {
                    "03": {"detected": 100, "run_count": 100, "median_delay": 78.0, "p90_delay": 83.0},
                    "09": {"detected": 100, "run_count": 100, "median_delay": 104.0, "p90_delay": 153.0},
                    "15": {"detected": 100, "run_count": 100, "median_delay": 98.0, "p90_delay": 139.0},
                },
            },
            "old_replays_are_not_current_audit_results": True,
        },
        "monitor_binding": binding,
        "row_level_holdout_audit_data": {
            "dataset": "Rieth et al. multi-run Tennessee Eastman process",
            "doi": "10.7910/DVN/6C3JR1",
            "healthy": {
                "partition": "fault_free_training",
                "simulation_runs": list(HEALTHY_RUNS),
                "expected_source_length": EXPECTED_HEALTHY_SOURCE_LENGTH,
            },
            "faults": {
                "partition": "faulty_testing",
                "fault_ids": list(FAULT_IDS),
                "simulation_runs": list(FAULT_RUNS),
                "expected_source_length": EXPECTED_FAULT_SOURCE_LENGTH,
                "fault_start_zero_based": FAULT_START,
            },
            "scored_interval_half_open": [0, SEQUENCE_LENGTH],
            "source_files": SOURCE_FILES,
            "total_sequences": len(HEALTHY_RUNS) + len(FAULT_IDS) * len(FAULT_RUNS),
            "decode_disclosure": "pyreadr decodes a complete RData object before exact row selection",
        },
        "success_gates": {
            "all_required": True,
            "healthy_alarm_runs_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
            "healthy_gate_note": (
                "3/80 has one-sided CP95 upper 0.09407454652521366, which supports "
                "only the broad p<0.10 audit ceiling and does not establish p<0.05"
            ),
            "per_fault_detected_runs_minimum": PER_FAULT_DETECTIONS_MINIMUM,
            "total_fault_detected_runs_minimum": TOTAL_FAULT_DETECTIONS_MINIMUM,
            "per_fault_median_restricted_delay_samples_maximum": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "per_fault_restricted_p90_delay_samples_maximum": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "restricted_delay_miss_imputation_samples": MISS_DELAY,
            "fault_detection_requires_new_alarm_episode_start_at_or_after_fault_onset": True,
            "cpu_cuda_all_numeric_paths_within_tolerance": CPU_GPU_TOLERANCE,
            "cpu_cuda_alarm_arrays_exact": True,
        },
        "runtime": {
            "required_device": "cuda",
            "score_dtype": "float64",
            "tf32": False,
            "deterministic_algorithms": True,
            "cpu_reference": "separate NumPy implementation of directional GLR-CUSUM",
        },
        "forbidden_actions": [
            "training_on_row_level_audit_rows",
            "refitting_on_row_level_audit_rows",
            "threshold_or_gate_selection_on_row_level_audit_rows",
            "same_rows_retuning_after_result",
            "claiming_independent_external_or_final_confirmation",
        ],
        "output_contract": {
            "atomic_row_level_audit_directory": True,
            "completed_reentry_reads_no_RData": True,
            "artifacts": list(ARTIFACTS) + ["completion.json"],
        },
        "implementation_snapshot": implementation,
    }


def freeze_protocol(project_root: Path, protocol_path: Path) -> str:
    protocol_path = protocol_path.resolve()
    sidecar = protocol_path.with_suffix(".sha256")
    if protocol_path.exists() or sidecar.exists():
        raise FileExistsError("protocol or SHA-256 sidecar already exists")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    ext._write_json(protocol_path, build_protocol_template(project_root))
    digest = ext._sha256_file(protocol_path)
    try:
        ext._write_text(sidecar, f"{digest}  {protocol_path.name}\n")
    except Exception:
        protocol_path.unlink(missing_ok=True)
        raise
    return digest


def _load_protocol(project_root: Path, path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = ext._require_sha256(expected_sha256, label="protocol_sha256")
    path = path.resolve()
    if ext._sha256_file(path) != expected:
        raise ValueError("protocol SHA-256 differs from the command-line binding")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != f"{expected}  {path.name}\n":
        raise ValueError("protocol sidecar is missing or differs")
    protocol = ext._strict_json(path)
    if protocol != build_protocol_template(project_root):
        raise ValueError("protocol differs from the authenticated implementation template")
    return protocol, expected


def _source_records(data_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for partition, expected in SOURCE_FILES.items():
        path = data_dir / str(expected["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = {
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "md5": _md5_file(path),
            "sha256": ext._sha256_file(path),
        }
        if observed != expected:
            raise ValueError(f"source file changed: {partition}")
        records[partition] = observed
    return records


def _episode_starts(alarm: np.ndarray) -> np.ndarray:
    values = np.asarray(alarm, dtype=bool)
    if values.shape != (SEQUENCE_LENGTH,):
        raise ValueError("alarm path has the wrong length")
    return np.flatnonzero(values & ~np.r_[False, values[:-1]])


def _healthy_records(
    runs: Iterable[SensorSequence], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        path = paths[run.name]
        starts = _episode_starts(path.alarm)
        rows.append({
            "run": run.name,
            "simulation_run": int(run.metadata["simulation_run"]),
            "any_alarm": bool(starts.size),
            "alarm_episode_count": int(starts.size),
            "first_alarm_index": int(starts[0]) if starts.size else None,
            "maximum_envelope": float(np.max(path.envelope[path.eligible])),
        })
    return rows


def _fault_records(
    runs: Iterable[SensorSequence], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        path = paths[run.name]
        starts = _episode_starts(path.alarm)
        post = starts[starts >= FAULT_START]
        first = int(post[0]) if post.size else None
        fault_id = int(run.metadata["fault_id"])
        attributed = None
        if first is not None:
            attributed = int(FAULT_IDS[int(np.argmax(path.component_persistence[first]))])
        delay = first - FAULT_START if first is not None else None
        rows.append({
            "run": run.name,
            "fault_id": fault_id,
            "simulation_run": int(run.metadata["simulation_run"]),
            "detected": first is not None,
            "first_fault_alarm_episode_index": first,
            "delay_samples": delay,
            "restricted_delay_samples_miss_at_run_end": delay if delay is not None else MISS_DELAY,
            "pre_fault_alarm_episode_count": int(np.sum(starts < FAULT_START)),
            "pre_fault_interval_used_as_performance_evidence": False,
            "attributed_fault_head_at_first_alarm": attributed,
            "attribution_matches_fault": attributed == fault_id if attributed is not None else False,
            "maximum_post_fault_envelope": float(np.max(path.envelope[FAULT_START:])),
        })
    return rows


def _summary(
    healthy_records: list[dict[str, Any]],
    fault_records: list[dict[str, Any]],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    if len(healthy_records) != len(HEALTHY_RUNS) or len(fault_records) != len(FAULT_IDS) * len(FAULT_RUNS):
        raise ValueError("row-level audit record counts changed")
    healthy_alarms = sum(bool(row["any_alarm"]) for row in healthy_records)
    cp_upper = ext._clopper_pearson_upper(healthy_alarms, len(healthy_records))
    per_fault: dict[str, Any] = {}
    detection_counts: list[int] = []
    median_passes: list[bool] = []
    p90_passes: list[bool] = []
    all_restricted: list[int] = []
    for fault_id in FAULT_IDS:
        rows = [row for row in fault_records if int(row["fault_id"]) == fault_id]
        if len(rows) != len(FAULT_RUNS):
            raise ValueError(f"fault {fault_id}: row-level audit run count changed")
        detected = sum(bool(row["detected"]) for row in rows)
        restricted = [int(row["restricted_delay_samples_miss_at_run_end"]) for row in rows]
        median = float(statistics.median(restricted))
        p90 = ext._empirical_upper_order_statistic(restricted, 0.90)
        detection_counts.append(detected)
        median_passes.append(median <= MEDIAN_RESTRICTED_DELAY_MAXIMUM)
        p90_passes.append(p90 <= PER_FAULT_RESTRICTED_P90_MAXIMUM)
        all_restricted.extend(restricted)
        attribution_matches = sum(bool(row["attribution_matches_fault"]) for row in rows)
        per_fault[f"{fault_id:02d}"] = {
            "run_count": len(rows),
            "detected_runs": detected,
            "detection_rate": detected / len(rows),
            "median_restricted_delay_samples": median,
            "p90_restricted_delay_samples_higher_order_statistic": p90,
            "passed_detection_gate": detected >= PER_FAULT_DETECTIONS_MINIMUM,
            "passed_median_gate": median <= MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "passed_p90_gate": p90 <= PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "attribution_matches_fault_runs": attribution_matches,
            "attribution_match_rate": attribution_matches / len(rows),
            "attribution_is_report_only": True,
            "records": rows,
        }
    total_detected = sum(detection_counts)
    parity_pass = bool(parity["all_numeric_paths_within_tolerance"]) and bool(
        parity["all_alarm_arrays_exact"]
    )
    passed = (
        healthy_alarms <= HEALTHY_ALARM_RUNS_MAXIMUM
        and min(detection_counts) >= PER_FAULT_DETECTIONS_MINIMUM
        and total_detected >= TOTAL_FAULT_DETECTIONS_MINIMUM
        and all(median_passes)
        and all(p90_passes)
        and parity_pass
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "phase": PHASE,
        "method": METHOD,
        "passed": passed,
        "decision": PASS_DECISION if passed else FAIL_DECISION,
        "threshold": Q100_THRESHOLD,
        "threshold_binary64_hex": Q100_THRESHOLD_HEX,
        "comparison": "strictly_greater_than",
        "healthy": {
            "run_count": len(healthy_records),
            "runs_with_any_alarm": healthy_alarms,
            "alarm_run_rate": healthy_alarms / len(healthy_records),
            "one_sided_clopper_pearson_95_upper": cp_upper,
            "gate_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
            "passed": healthy_alarms <= HEALTHY_ALARM_RUNS_MAXIMUM,
            "records": healthy_records,
        },
        "faults": {
            "per_fault": per_fault,
            "detected_runs": total_detected,
            "run_count": len(fault_records),
            "detection_rate": total_detected / len(fault_records),
            "per_fault_minimum_gate": PER_FAULT_DETECTIONS_MINIMUM,
            "total_minimum_gate": TOTAL_FAULT_DETECTIONS_MINIMUM,
            "overall_median_restricted_delay_samples": float(statistics.median(all_restricted)),
            "per_fault_median_delay_maximum_gate": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "per_fault_p90_maximum_gate": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "miss_delay_imputation_samples": MISS_DELAY,
            "attribution_is_report_only": True,
        },
        "parity_gate_passed": parity_pass,
        "healthy_CP95_interpretation": (
            "an upper bound below 0.10 supports only the broad p<0.10 audit ceiling; "
            "it does not establish p<0.05"
        ),
        "governance_interpretation": ANALYSIS_STATUS,
    }


def _pack_paths(
    runs: tuple[SensorSequence, ...], paths: Mapping[str, DirectionalGLRPath]
) -> dict[str, np.ndarray]:
    arrays = ext._pack_paths(runs, paths, operational_threshold=Q100_THRESHOLD)
    arrays["fault_start_zero_based"] = np.asarray([FAULT_START], dtype=np.int64)
    return arrays


def _validate_evidence(path: Path) -> dict[str, np.ndarray]:
    required = {
        "run_names", "partitions", "fault_ids", "simulation_runs", "logits",
        "standardized_logits", "clipped_evidence", "cusum", "component_persistence",
        "envelope", "eligible", "alarm", "operational_threshold", "fault_start_zero_based",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("row-level audit evidence field set changed")
        try:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        except ValueError as error:
            raise ValueError("row-level audit evidence is unsafe or unreadable") from error
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("row-level audit evidence contains an object array")
    count = len(HEALTHY_RUNS) + len(FAULT_IDS) * len(FAULT_RUNS)
    shapes = {
        "run_names": (count,),
        "partitions": (count,),
        "fault_ids": (count,),
        "simulation_runs": (count,),
        "logits": (count, SEQUENCE_LENGTH, len(FAULT_IDS)),
        "standardized_logits": (count, SEQUENCE_LENGTH, len(FAULT_IDS)),
        "clipped_evidence": (count, SEQUENCE_LENGTH, len(FAULT_IDS)),
        "cusum": (count, SEQUENCE_LENGTH, len(FAULT_IDS)),
        "component_persistence": (count, SEQUENCE_LENGTH, len(FAULT_IDS)),
        "envelope": (count, SEQUENCE_LENGTH),
        "eligible": (count, SEQUENCE_LENGTH),
        "alarm": (count, SEQUENCE_LENGTH),
        "operational_threshold": (1,),
        "fault_start_zero_based": (1,),
    }
    for name, expected_shape in shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"row-level audit evidence {name} shape changed")
    float_fields = (
        "logits", "standardized_logits", "clipped_evidence", "cusum",
        "component_persistence", "envelope", "operational_threshold",
    )
    if any(arrays[name].dtype != np.dtype(np.float64) for name in float_fields):
        raise ValueError("row-level audit floating evidence must be float64")
    if arrays["fault_ids"].dtype != np.dtype(np.int64) or arrays["simulation_runs"].dtype != np.dtype(np.int64):
        raise ValueError("row-level audit identities must be int64")
    if arrays["fault_start_zero_based"].dtype != np.dtype(np.int64):
        raise ValueError("row-level audit fault onset must be int64")
    if arrays["eligible"].dtype != np.dtype(np.bool_) or arrays["alarm"].dtype != np.dtype(np.bool_):
        raise ValueError("row-level audit eligible/alarm fields must be bool")
    if arrays["run_names"].dtype.kind != "U" or arrays["partitions"].dtype.kind != "U":
        raise ValueError("row-level audit names and partitions must be Unicode arrays")
    if float(arrays["operational_threshold"][0]) != Q100_THRESHOLD:
        raise ValueError("row-level audit evidence threshold changed")
    if int(arrays["fault_start_zero_based"][0]) != FAULT_START:
        raise ValueError("row-level audit evidence fault onset changed")
    expected_faults = np.asarray(
        [0] * len(HEALTHY_RUNS) + [fault for fault in FAULT_IDS for _ in FAULT_RUNS],
        dtype=np.int64,
    )
    expected_runs = np.asarray(list(HEALTHY_RUNS) + list(FAULT_RUNS) * len(FAULT_IDS), dtype=np.int64)
    expected_partitions = np.asarray(
        ["fault_free_training"] * len(HEALTHY_RUNS)
        + ["faulty_testing"] * (len(FAULT_IDS) * len(FAULT_RUNS))
    )
    expected_names = np.asarray(
        [
            f"Rieth-fault_free_training-fault-00-run-{run:03d}"
            for run in HEALTHY_RUNS
        ]
        + [
            f"Rieth-faulty_testing-fault-{fault:02d}-run-{run:03d}"
            for fault in FAULT_IDS
            for run in FAULT_RUNS
        ]
    )
    if not np.array_equal(arrays["run_names"], expected_names):
        raise ValueError("row-level audit exact run names changed")
    if not np.array_equal(arrays["fault_ids"], expected_faults):
        raise ValueError("row-level audit fault identities changed")
    if not np.array_equal(arrays["simulation_runs"], expected_runs):
        raise ValueError("row-level audit run identities changed")
    if not np.array_equal(arrays["partitions"], expected_partitions):
        raise ValueError("row-level audit partitions changed")
    expected_eligible = np.zeros((count, SEQUENCE_LENGTH), dtype=bool)
    expected_eligible[:, FIRST_ALARM_ELIGIBLE:] = True
    if not np.array_equal(arrays["eligible"], expected_eligible):
        raise ValueError("row-level audit eligibility mask changed")
    if not np.isneginf(arrays["envelope"][:, :FIRST_ALARM_ELIGIBLE]).all():
        raise ValueError("row-level audit pre-eligible envelope must be -inf")
    if not np.isfinite(arrays["envelope"][:, FIRST_ALARM_ELIGIBLE:]).all():
        raise ValueError("row-level audit eligible envelope must be finite")
    replay = arrays["eligible"] & (arrays["envelope"] > Q100_THRESHOLD)
    if not np.array_equal(arrays["alarm"], replay):
        raise ValueError("row-level audit alarm is not a strict q100 replay")
    return arrays


def _records_from_evidence(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    healthy: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    for index, raw_name in enumerate(arrays["run_names"]):
        name = str(raw_name)
        fault_id = int(arrays["fault_ids"][index])
        run_id = int(arrays["simulation_runs"][index])
        alarm = arrays["alarm"][index].astype(bool)
        envelope = arrays["envelope"][index]
        eligible = arrays["eligible"][index].astype(bool)
        starts = _episode_starts(alarm)
        if fault_id == 0:
            healthy.append({
                "run": name, "simulation_run": run_id, "any_alarm": bool(starts.size),
                "alarm_episode_count": int(starts.size),
                "first_alarm_index": int(starts[0]) if starts.size else None,
                "maximum_envelope": float(np.max(envelope[eligible])),
            })
            continue
        post = starts[starts >= FAULT_START]
        first = int(post[0]) if post.size else None
        attributed = None
        if first is not None:
            attributed = int(FAULT_IDS[int(np.argmax(arrays["component_persistence"][index, first]))])
        delay = first - FAULT_START if first is not None else None
        faults.append({
            "run": name, "fault_id": fault_id, "simulation_run": run_id,
            "detected": first is not None, "first_fault_alarm_episode_index": first,
            "delay_samples": delay,
            "restricted_delay_samples_miss_at_run_end": delay if delay is not None else MISS_DELAY,
            "pre_fault_alarm_episode_count": int(np.sum(starts < FAULT_START)),
            "pre_fault_interval_used_as_performance_evidence": False,
            "attributed_fault_head_at_first_alarm": attributed,
            "attribution_matches_fault": attributed == fault_id if attributed is not None else False,
            "maximum_post_fault_envelope": float(np.max(envelope[FAULT_START:])),
        })
    return healthy, faults


def _report(result: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen q100 PAC hybrid within-Rieth row-level holdout audit", "",
        f"Analysis status: `{ANALYSIS_STATUS}`", "",
        f"Decision: **{result['decision']}**", "",
        "This post-q99-failure sequential audit uses rows that were not previously ",
        "selected, transformed, or scored. Their RData partitions were previously ",
        "decoded wholesale, and statistical independence/common-random-number ",
        "coupling across partitions is not established. It is not an independent, ",
        "external, or final confirmation.", "",
        "| Quantity | Observed | Frozen gate |", "|---|---:|---:|",
        f"| Healthy alarm runs | {result['healthy']['runs_with_any_alarm']}/80 | <= 3 |",
        f"| Healthy one-sided CP95 upper | {result['healthy']['one_sided_clopper_pearson_95_upper']:.6f} | < 0.10 through count gate |",
        f"| Total detected faults | {result['faults']['detected_runs']}/300 | >= 270 |",
    ]
    for fault_id in FAULT_IDS:
        item = result["faults"]["per_fault"][f"{fault_id:02d}"]
        lines.extend([
            f"| Fault {fault_id} detected | {item['detected_runs']}/100 | >= 90 |",
            f"| Fault {fault_id} median restricted delay | {item['median_restricted_delay_samples']:.3f} | <= 120 |",
            f"| Fault {fault_id} p90 restricted delay | {item['p90_restricted_delay_samples_higher_order_statistic']:.3f} | <= 180 |",
            f"| Fault {fault_id} attribution matches | {item['attribution_matches_fault_runs']}/100 | report only |",
        ])
    lines.extend([
        "", f"- Frozen strict q100 threshold: {Q100_THRESHOLD!r} ({Q100_THRESHOLD_HEX}).",
        f"- CUDA device: {execution['runtime']['device_name']}.",
        f"- Total scoring wall time including CPU parity: {execution['score']['total_wall_seconds']:.3f} s.",
        f"- Full CPU/CUDA alarm arrays exact: {result['parity_gate_passed']}.",
        "- A healthy CP95 upper bound below 0.10 supports only the broad p<0.10 audit ceiling; it does not establish p<0.05.",
        "- Fault-head attribution is descriptive and is not a success gate.",
        "- No row-level audit training, calibration, selection, or retuning is authorized.", "",
    ])
    return "\n".join(lines)


def _verify_completed(target: Path, protocol_sha256: str) -> dict[str, Any] | None:
    completion = ext._verify_completed_phase(
        target, phase=PHASE, protocol_sha256=protocol_sha256,
        artifacts=ARTIFACTS, decisions={PASS_DECISION, FAIL_DECISION},
    )
    if completion is None:
        return None
    arrays = _validate_evidence(target / "audit_evidence.npz")
    parity = ext._strict_json(target / "parity.json")
    healthy, faults = _records_from_evidence(arrays)
    replay = _summary(healthy, faults, parity)
    result = ext._strict_json(target / "result.json")
    if result != replay or result["decision"] != completion["decision"]:
        raise ValueError("completed result is not an exact evidence replay")
    return completion


def _run(
    *, project_root: Path, data_dir: Path, output_root: Path, protocol_path: Path,
    protocol: Mapping[str, Any], protocol_sha256: str, device: str, batch_size: int,
) -> dict[str, Any]:
    target = output_root / PHASE_DIRECTORY
    completed = _verify_completed(target, protocol_sha256)
    if completed is not None:
        print("q100 row-level audit is already complete; authenticated without RData access", flush=True)
        return completed

    source_records = _source_records(data_dir)
    runtime = ext._runtime_preflight(device)
    target, staging = ext._phase_staging(output_root, PHASE_DIRECTORY)
    ext._write_json(staging / "runtime_preflight.json", runtime)
    access_intent = {
        "schema": "hybrid_pac_q100_holdout_access_intent", "schema_version": 1,
        "phase": PHASE, "protocol_sha256": protocol_sha256,
        "created_before_any_row_level_holdout_pyreadr_decode": True,
        "authorized_reads": {
            "healthy": ["fault_free_training", list(HEALTHY_RUNS)],
            "faults": ["faulty_testing", list(FAULT_IDS), list(FAULT_RUNS)],
        },
        "threshold": Q100_THRESHOLD, "threshold_binary64_hex": Q100_THRESHOLD_HEX,
        "forbidden": protocol["forbidden_actions"],
        "whole_partition_decode_acknowledged": True,
        "statistical_independence_or_common_random_number_coupling_established": False,
    }
    ext._write_json(staging / "access_intent.json", access_intent)
    phase_body = {
        "schema": "hybrid_pac_q100_holdout_phase_manifest", "schema_version": 1,
        "phase": PHASE, "protocol_sha256": protocol_sha256,
        "access_intent_sha256": ext._sha256_file(staging / "access_intent.json"),
        "runtime_preflight_sha256": ext._sha256_file(staging / "runtime_preflight.json"),
        "source_records": source_records,
        "written_before_any_row_level_holdout_pyreadr_decode": True,
    }
    ext._write_json(
        staging / "phase_manifest.json",
        {**phase_body, "phase_fingerprint": ext._canonical_sha256(phase_body)},
    )
    ext._copy_protocol_snapshot(protocol_path, staging)
    ext._write_json(staging / "q100_certificate.json", protocol["revised_pac_calibration"]["certificate"])

    transformer, uncalibrated = ext._load_method(project_root)
    model = uncalibrated.with_threshold(Q100_THRESHOLD)
    archive = RiethTEPArchive(data_dir, verify_checksums=True)
    if archive.verify(("fault_free_training", "faulty_testing")) != {
        key: str(value["md5"]) for key, value in SOURCE_FILES.items()
    }:
        raise ValueError("Rieth MD5 verification differs from the frozen source binding")

    print("opening only FF-training healthy runs 121..200", flush=True)
    healthy_full = base._run_ids(
        archive.load_runs(
            "fault_free_training", {0: HEALTHY_RUNS}, fault_start=FAULT_START,
            split_manipulated=False,
        ),
        HEALTHY_RUNS, partition="fault_free_training",
        expected_length=EXPECTED_HEALTHY_SOURCE_LENGTH, label="q100 healthy holdout",
    )
    base._validate_full_values(healthy_full)
    print("opening only faulty-testing faults 3/9/15 runs 301..400", flush=True)
    faults_full = base._fault_run_ids(
        archive.load_runs(
            "faulty_testing", {fault: FAULT_RUNS for fault in FAULT_IDS},
            fault_start=FAULT_START, split_manipulated=False,
        ),
        FAULT_RUNS, expected_length=EXPECTED_FAULT_SOURCE_LENGTH,
        label="q100 fault holdout",
    )
    base._validate_full_values(faults_full)
    healthy = tuple(base._truncate_run(run, SEQUENCE_LENGTH) for run in healthy_full)
    faults = tuple(base._truncate_run(run, SEQUENCE_LENGTH) for run in faults_full)
    runs = healthy + faults
    if len(runs) != 380:
        raise RuntimeError("q100 holdout must contain exactly 380 sequences")

    gpu_paths, cpu_paths, parity, score_execution = ext._score_with_full_parity(
        transformer, model, runs, device=device, batch_size=batch_size
    )
    if set(gpu_paths) != set(cpu_paths):
        raise RuntimeError("CPU/CUDA path identities differ")
    result = _summary(_healthy_records(healthy, gpu_paths), _fault_records(faults, gpu_paths), parity)
    result["protocol_sha256"] = protocol_sha256
    # Remove the binding before replay comparison; it is restored below during verification.
    result_without_binding = dict(result)
    result_without_binding.pop("protocol_sha256")
    execution = {
        "schema": SCHEMA, "schema_version": 1, "phase": PHASE,
        "protocol_sha256": protocol_sha256, "runtime": runtime,
        "score": score_execution, "frozen_threshold": Q100_THRESHOLD,
    }
    access_audit = {
        "schema": "hybrid_pac_q100_holdout_data_access_audit", "schema_version": 1,
        "selected_healthy_partition": "fault_free_training",
        "selected_healthy_runs": [121, 200],
        "selected_fault_partition": "faulty_testing",
        "selected_fault_ids": list(FAULT_IDS),
        "selected_fault_runs": [301, 400],
        "selected_sequence_count": len(runs),
        "fault_sequences_truncated_to_half_open": [0, SEQUENCE_LENGTH],
        "RData_whole_file_decode_occurred": True,
        "selected_rows_previously_scored": False,
        "statistical_independence_or_common_random_number_coupling_established": False,
        "training_refitting_selection_or_threshold_change": False,
    }
    ext._write_json(staging / "result.json", result_without_binding)
    ext._write_json(staging / "parity.json", parity)
    ext._write_json(staging / "gpu_execution.json", execution)
    ext._write_json(staging / "data_access_audit.json", access_audit)
    ext._write_npz(staging / "audit_evidence.npz", _pack_paths(runs, gpu_paths))
    ext._write_text(staging / "REPORT.md", _report(result_without_binding, execution))
    ext._finalize_phase(
        staging=staging, target=target, phase=PHASE, protocol_sha256=protocol_sha256,
        decision=str(result_without_binding["decision"]), artifacts=ARTIFACTS,
    )
    verified = _verify_completed(target, protocol_sha256)
    if verified is None:
        raise RuntimeError("q100 row-level audit completion disappeared")
    return verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="write the immutable protocol and sidecar")
    freeze.add_argument("--protocol", type=Path, required=True)
    run = sub.add_parser("run", help="run or authenticate the sequential row-level audit")
    run.add_argument("data_dir", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--device", choices=("cuda",), required=True)
    run.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = _SCRIPT_DIR.parent
    if args.command == "freeze":
        digest = freeze_protocol(project_root, args.protocol)
        print(json.dumps({"protocol": str(args.protocol.resolve()), "sha256": digest}), flush=True)
        return 0
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    protocol, digest = _load_protocol(project_root, args.protocol, args.protocol_sha256)
    completion = _run(
        project_root=project_root, data_dir=args.data_dir.resolve(),
        output_root=args.output_root.resolve(), protocol_path=args.protocol,
        protocol=protocol, protocol_sha256=digest, device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "decision": completion["decision"],
        "completion_fingerprint": completion["completion_fingerprint"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
