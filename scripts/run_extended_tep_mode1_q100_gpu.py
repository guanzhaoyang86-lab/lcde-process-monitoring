"""Two-stage CUDA confirmation of the frozen hybrid monitor on Extended TEP.

The monitor and every hyperparameter come from the completed Rieth study.  This
script performs only a domain-specific healthy calibration (100 pre-fault
prefixes) followed by one sealed audit (100 healthy prefixes and 100 runs for
each of IDV 3, 9, and 15).  Audit rows may never be used to retune the method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def _requested_device(tokens: Sequence[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# On this Windows host PyTorch must load before NumPy/MKL for CUDA runs.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _torch_preload  # noqa: F401

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_hybrid_pac_rmax_external_gpu as ext  # noqa: E402
from mate_spc_directional.glr import FAULT_IDS, DirectionalGLRPath  # noqa: E402
from mate_spc_external.extended_tep_mode1 import (  # noqa: E402
    AUDIT_FAULT_IDS,
    AUDIT_RUNS,
    CALIBRATION_RUNS,
    CALIBRATION_WINDOW,
    DATASET_DOI,
    ExtendedTEPMode1Archive,
    FAULT_WINDOW,
    LOCAL_FAULT_START,
    MODE1_FILE_ID,
    MODE1_FILE_NAME,
    MODE1_MD5,
    MODE1_SIZE_BYTES,
    README_MD5,
    SEQUENCE_LENGTH,
    locate_unique_run,
    validate_processdata_labels,
)
from mate_spc_pac.certificate import build_calibration_certificate  # noqa: E402


SCHEMA = "hybrid_q100_extended_tep_mode1_external_confirmation"
SCHEMA_VERSION = 1
ANALYSIS_STATUS = (
    "locally_frozen_before_numeric_decode_cross_simulator_"
    "domain_calibrated_external_confirmation"
)
METHOD = "hybrid_directional_energy_pac_q100_glr_cusum"
CALIBRATION_DECISION = "EXTENDED_TEP_Q100_CALIBRATION_FROZEN_FOR_AUDIT"
PASS_DECISION = "PASS_EXTENDED_TEP_MODE1_EXTERNAL_CONFIRMATION"
FAIL_DECISION = "FAIL_EXTENDED_TEP_MODE1_EXTERNAL_CONFIRMATION_NO_RETUNING"

PAC_EPSILON = 0.05
PAC_DELTA = 0.01
PAC_RANK = 100
VALID_START = 63
FIRST_ALARM_ELIGIBLE = 65
FAULT_START = LOCAL_FAULT_START
MISS_DELAY = SEQUENCE_LENGTH - FAULT_START
CPU_GPU_TOLERANCE = 1e-10

HEALTHY_ALARM_RUNS_MAXIMUM = 4
PER_FAULT_DETECTIONS_MINIMUM = 90
TOTAL_FAULT_DETECTIONS_MINIMUM = 270
MEDIAN_RESTRICTED_DELAY_MAXIMUM = 120.0
PER_FAULT_RESTRICTED_P90_MAXIMUM = 180.0

DATA_FILE = MODE1_FILE_NAME
README_FILE = "Readme.html"
ARTICLE_METADATA_FILE = "figshare_article_13385936.json"
HYBRID_DIRECTORY = Path("outputs/hybrid_energy_glr_gpu_v1")
TRANSFORMER_NAME = "hybrid_transformer.npz"
MODEL_NAME = "hybrid_glr_uncalibrated.npz"
DESIGN_SCORES_NAME = "calibration_scores.npz"
EXPECTED_METHOD_SHA256 = {
    TRANSFORMER_NAME: "09cca42496360e3132239150b3bf5f5dcd761df61eb35ae7082465c187fb28ab",
    MODEL_NAME: "0e2bfb1e6cfad1d6016f4fb3413c8a517c24fbeb6648bb8aad0533f23c17858d",
    DESIGN_SCORES_NAME: "2139c9f05be6c7b1acfbf520b2a1c14411a9b515b4399352c30b62a599ad4151",
}
PRIOR_PROTOCOL = Path("protocols/hybrid_pac_q100_holdout_v1_2.json")
PRIOR_AUDIT = Path("outputs/hybrid_pac_q100_holdout_gpu_v1_2/audit")
EXPECTED_PRIOR_SHA256 = {
    "protocol": "9a06e32b07ff95f7955876e6c573de363891c0bad76b1fa12b0b3c6adb0629e7",
    "completion": "947bf7f1ead3055eea4987f42d29cbb1d675d0a96d335177e2abccd3163e9f19",
    "result": "570bd94fe34d3faf253e5606cf1b40a6d0cf5f406ece2f447b861f8fc5d8a369",
}

IMPLEMENTATION_FILES = (
    "scripts/run_extended_tep_mode1_q100_gpu.py",
    "tests/test_run_extended_tep_mode1_q100_gpu.py",
    "src/mate_spc_external/extended_tep_mode1.py",
    "scripts/run_hybrid_pac_rmax_external_gpu.py",
    "src/mate_spc_external/rieth_training.py",
    "src/mate_spc_hybrid/features.py",
    "src/mate_spc_directional/features.py",
    "src/mate_spc_directional/glr.py",
    "src/mate_spc_directional/gpu.py",
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
FORBIDDEN_ACTIONS = (
    "training_on_external_calibration_or_audit_rows",
    "refitting_on_external_calibration_or_audit_rows",
    "hyperparameter_selection_on_external_rows",
    "threshold_selection_from_audit_rows",
    "same_audit_rows_retuning_after_result",
    "claiming_zero_shot_or_fully_independent_confirmation",
)


def _hash_file_pair(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - authenticates the publisher checksum
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _source_records(data_dir: Path) -> dict[str, dict[str, int | str]]:
    data_dir = data_dir.resolve()
    records: dict[str, dict[str, int | str]] = {}
    for name in (DATA_FILE, README_FILE, ARTICLE_METADATA_FILE):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        md5, sha256 = _hash_file_pair(path)
        records[name] = {
            "file_name": name,
            "size_bytes": path.stat().st_size,
            "md5": md5,
            "sha256": sha256,
        }
    mode1 = records[DATA_FILE]
    if mode1["size_bytes"] != MODE1_SIZE_BYTES or mode1["md5"] != MODE1_MD5:
        raise ValueError("TEP_Mode1.h5 differs from the official Figshare size/MD5")
    if records[README_FILE]["md5"] != README_MD5:
        raise ValueError("Readme.html differs from the official Figshare MD5")
    article = ext._strict_json(data_dir / ARTICLE_METADATA_FILE)
    if (
        article.get("id") != 13385936
        or article.get("doi") != DATASET_DOI
        or article.get("version") != 1
        or article.get("status") != "public"
        or article.get("defined_type_name") != "dataset"
    ):
        raise ValueError("Figshare article identity/DOI/version/type changed")
    license_record = article.get("license")
    if not isinstance(license_record, dict) or license_record.get("name") != "CC0":
        raise ValueError("Figshare article license is not CC0")
    files = article.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare article has no file list")
    matches = [
        item
        for item in files
        if isinstance(item, dict)
        and item.get("id") == MODE1_FILE_ID
        and item.get("name") == MODE1_FILE_NAME
    ]
    if len(matches) != 1:
        raise ValueError("Figshare article has no unique official Mode 1 file record")
    published = matches[0]
    if (
        published.get("size") != MODE1_SIZE_BYTES
        or published.get("supplied_md5") != MODE1_MD5
        or published.get("computed_md5") != MODE1_MD5
        or published.get("download_url")
        != f"https://ndownloader.figshare.com/files/{MODE1_FILE_ID}"
    ):
        raise ValueError("Figshare Mode 1 file metadata changed")
    return records


def _label_and_topology_record(data_dir: Path) -> dict[str, Any]:
    """Read HDF metadata only; no processdata or idv_init numeric values."""

    import h5py

    path = data_dir.resolve() / DATA_FILE
    selections = [(1, run) for run in CALIBRATION_RUNS]
    selections += [(1, run) for run in AUDIT_RUNS]
    selections += [(fault, run) for fault in AUDIT_FAULT_IDS for run in AUDIT_RUNS]
    locations: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        labels = validate_processdata_labels(handle)
        for idv, run in selections:
            located = locate_unique_run(handle, idv=idv, simulation_run=run)
            processdata = handle[located.processdata_path]
            idv_init_path = f"{located.group_path}/idv_init"
            if idv_init_path not in handle or not isinstance(handle[idv_init_path], h5py.Dataset):
                raise ValueError(f"{idv_init_path}: missing exact idv_init dataset")
            idv_init = handle[idv_init_path]
            if processdata.shape != (2001, 54) or np.dtype(processdata.dtype) != np.dtype(np.float64):
                raise ValueError(f"{located.processdata_path}: metadata shape/dtype changed")
            if idv_init.shape != (3, 28) or np.dtype(idv_init.dtype) != np.dtype(np.float64):
                raise ValueError(f"{idv_init_path}: metadata shape/dtype changed")
            locations.append(
                {
                    "idv": idv,
                    "run": run,
                    "status": located.status,
                    "processdata_path": located.processdata_path,
                }
            )
    return {
        "processdata_labels": list(labels),
        "processdata_labels_sha256": ext._canonical_sha256(list(labels)),
        "selection_count": len(locations),
        "selection_topology_sha256": ext._canonical_sha256(locations),
        "completion_status_counts": {
            status: sum(row["status"] == status for row in locations)
            for status in ("SimulationCompleted", "SimulationStopped")
        },
        "numeric_processdata_or_idv_init_values_read": False,
    }


def _method_records(project_root: Path) -> dict[str, dict[str, int | str]]:
    records = {
        name: ext._hash_record(project_root / HYBRID_DIRECTORY / name)
        for name in EXPECTED_METHOD_SHA256
    }
    for name, expected in EXPECTED_METHOD_SHA256.items():
        if records[name]["sha256"] != expected:
            raise ValueError(f"frozen method artifact changed: {name}")
    ext._validate_frozen_method(project_root, records)
    return records


def _prior_provenance(project_root: Path) -> dict[str, Any]:
    paths = {
        "protocol": project_root / PRIOR_PROTOCOL,
        "completion": project_root / PRIOR_AUDIT / "completion.json",
        "result": project_root / PRIOR_AUDIT / "result.json",
    }
    records = {name: ext._hash_record(path) for name, path in paths.items()}
    for name, expected in EXPECTED_PRIOR_SHA256.items():
        if records[name]["sha256"] != expected:
            raise ValueError(f"prior q100 provenance changed: {name}")
    return {
        "analysis": "within_Rieth_row_level_holdout_used_only_as_design_provenance",
        "records": records,
        "numeric_threshold_not_reused": True,
    }


def _monitor_binding(method_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    deployment = {
        "method": METHOD,
        "transformer_sha256": method_records[TRANSFORMER_NAME]["sha256"],
        "uncalibrated_model_sha256": method_records[MODEL_NAME]["sha256"],
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
        "threshold_rule": "PAC_RMax_maximum_q100_of_100_external_run_maxima",
        "epsilon": PAC_EPSILON,
        "delta": PAC_DELTA,
    }
    return {
        "deployment_spec": deployment,
        "deployment_spec_sha256": ext._canonical_sha256(deployment),
        "monitor_configuration": monitor,
        "monitor_configuration_sha256": ext._canonical_sha256(monitor),
    }


def _implementation_snapshot(project_root: Path) -> dict[str, dict[str, int | str]]:
    return {name: ext._hash_record(project_root / name) for name in IMPLEMENTATION_FILES}


def build_protocol_template(project_root: Path, data_dir: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_dir = data_dir.resolve()
    if tuple(FAULT_IDS) != tuple(AUDIT_FAULT_IDS) or tuple(FAULT_IDS) != (3, 9, 15):
        raise ValueError("fault-head identity/order changed before protocol freeze")
    if FIRST_ALARM_ELIGIBLE != VALID_START + ext.PERSISTENCE_WINDOW - 1:
        raise ValueError("eligibility geometry changed before protocol freeze")
    method_records = _method_records(project_root)
    binding = _monitor_binding(method_records)
    source_records = _source_records(data_dir)
    metadata = _label_and_topology_record(data_dir)
    rank = ext.pac_tolerance_rank(100, epsilon=PAC_EPSILON, delta=PAC_DELTA)
    if rank.rank_one_based != PAC_RANK:
        raise AssertionError("epsilon=.05, delta=.01, n=100 is not rank 100")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "method": METHOD,
        "governance": {
            "protocol_created_before_any_selected_processdata_or_idv_init_numeric_value": True,
            "source_bytes_hashed_and_HDF_metadata_read_before_freeze": True,
            "prior_Rieth_result_used_as_design_provenance_only": True,
            "cross_simulator_but_not_zero_shot": True,
            "threshold_domain_calibrated_on_external_prefault_prefixes": True,
            "common_random_number_coupling_across_IDVs_unknown": True,
            "same_external_audit_rows_retuning_after_result_forbidden": True,
        },
        "dataset": {
            "name": "Extended Tennessee Eastman Process Simulation Dataset Mode 1",
            "doi": DATASET_DOI,
            "figshare_article_id": 13385936,
            "license": "CC0",
            "source_records": source_records,
            "metadata_record": metadata,
            "canonical_projection": "processdata[1:, 1:53]",
            "excluded_column": "XMV12 agitator speed",
            "sample_period_hours": 0.05,
            "global_fault_first_affected_index_after_projection": 600,
        },
        "frozen_inputs": {
            "directory": HYBRID_DIRECTORY.as_posix(),
            "artifacts": method_records,
            "safe_serialization": "numeric NPZ loaded with allow_pickle=False",
        },
        "monitor_binding": binding,
        "prior_q100_design_provenance": _prior_provenance(project_root),
        "external_calibration": {
            "source_idv": 1,
            "magnitude_percent": 100,
            "simulation_runs": list(CALIBRATION_RUNS),
            "normalized_window_half_open": list(CALIBRATION_WINDOW),
            "run_count": 100,
            "rank_rule": "maximum_order_statistic_no_interpolation",
            "rank": rank.as_dict(),
            "numeric_threshold_unknown_at_protocol_freeze": True,
            "completion_required_before_audit_open": True,
        },
        "external_audit": {
            "healthy": {
                "source_idv": 1,
                "simulation_runs": list(AUDIT_RUNS),
                "normalized_window_half_open": list(CALIBRATION_WINDOW),
            },
            "faults": {
                "fault_ids": list(AUDIT_FAULT_IDS),
                "simulation_runs": list(AUDIT_RUNS),
                "normalized_window_half_open": list(FAULT_WINDOW),
                "local_fault_start_zero_based": FAULT_START,
            },
            "healthy_run_count": 100,
            "fault_run_count": 300,
            "total_run_count": 400,
            "fault_detection_requires_new_alarm_episode_start_at_or_after": FAULT_START,
            "pre_fault_alarm_continuing_across_onset_is_not_a_detection": True,
        },
        "runtime": {
            "required_device": "cuda",
            "score_dtype": "float64",
            "tf32": False,
            "deterministic_algorithms": True,
            "cpu_reference": "independent NumPy directional GLR-CUSUM",
            "cpu_cuda_max_abs_tolerance": CPU_GPU_TOLERANCE,
            "full_run_alarm_and_eligibility_arrays_must_match_exactly": True,
        },
        "success_gates": {
            "all_required": True,
            "healthy_alarm_runs_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
            "healthy_gate_interpretation": "at most 4/100 gives one-sided CP95 upper below 0.10",
            "strong_descriptive_FPR_below_0_05_reported_only_if_CP95_upper_below_0_05": True,
            "pac_primary_claim": "under exchangeable external healthy run maxima, q100 gives epsilon=.05 coverage at its exact certificate confidence",
            "per_fault_detected_runs_minimum": PER_FAULT_DETECTIONS_MINIMUM,
            "total_fault_detected_runs_minimum": TOTAL_FAULT_DETECTIONS_MINIMUM,
            "total_gate_redundant_with_three_per_fault_gates": True,
            "per_fault_median_restricted_delay_samples_maximum": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "per_fault_restricted_p90_delay_samples_maximum": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "restricted_delay_miss_imputation_samples": MISS_DELAY,
            "cpu_cuda_alarm_arrays_exact": True,
            "cpu_cuda_eligibility_arrays_exact": True,
            "cpu_cuda_all_numeric_paths_within_tolerance": True,
            "fault_head_attribution_report_only": True,
        },
        "access_order": [
            "freeze protocol and SHA-256 sidecar",
            "write calibration intent and phase manifest",
            "read/score only calibration windows and atomically seal q100 threshold",
            "authenticate calibration and write threshold-bound audit intent/manifest",
            "read/score audit windows once and atomically seal result",
        ],
        "forbidden_actions": list(FORBIDDEN_ACTIONS),
        "implementation_snapshot": _implementation_snapshot(project_root),
    }


def _new_freeze_temp(destination: Path) -> tuple[int, Path]:
    """Create an exclusively owned temporary file beside its destination."""

    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.freeze-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    return descriptor, Path(raw_path)


def _write_fsynced_json_descriptor(descriptor: int, value: object) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            ext._clean_json(value),
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_fsynced_text_descriptor(descriptor: int, value: str) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _freeze_file_identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return int(status.st_dev), int(status.st_ino)


def _rollback_exact_published_file(
    path: Path,
    expected_sha256: str,
    expected_identity: tuple[int, int],
) -> None:
    """Remove only a file published by this call and still byte-identical."""

    if not path.exists():
        return
    if (
        not path.is_file()
        or _freeze_file_identity(path) != expected_identity
        or ext._sha256_file(path) != expected_sha256
    ):
        raise RuntimeError(
            f"freeze rollback refused to remove an unowned, changed, or non-file target: {path}"
        )
    path.unlink()


def freeze_protocol(project_root: Path, data_dir: Path, protocol_path: Path) -> str:
    """Publish protocol plus sidecar with the sidecar acting as commit marker.

    A filesystem cannot atomically rename two independent files as one
    transaction. Both files are therefore prepared, fsynced, and authenticated
    in the destination directory first. The protocol is published first and the
    sidecar last; every reader requires both, so a process kill between the two
    atomic renames leaves a fail-closed, visibly incomplete freeze. Catchable
    failures roll back only byte-identical files published by this invocation.
    """

    protocol_path = protocol_path.resolve()
    sidecar = protocol_path.with_suffix(".sha256")
    if protocol_path.exists() or sidecar.exists():
        raise FileExistsError("protocol or SHA-256 sidecar already exists")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)

    protocol_temp: Path | None = None
    sidecar_temp: Path | None = None
    protocol_published = False
    sidecar_published = False
    digest: str | None = None
    sidecar_digest: str | None = None
    protocol_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    try:
        template = build_protocol_template(project_root, data_dir)
        protocol_descriptor, protocol_temp = _new_freeze_temp(protocol_path)
        _write_fsynced_json_descriptor(protocol_descriptor, template)
        digest = ext._sha256_file(protocol_temp)
        digest = ext._require_sha256(digest, label="protocol_sha256")

        sidecar_text = f"{digest}  {protocol_path.name}\n"
        sidecar_descriptor, sidecar_temp = _new_freeze_temp(sidecar)
        _write_fsynced_text_descriptor(sidecar_descriptor, sidecar_text)
        sidecar_digest = ext._sha256_file(sidecar_temp)

        if ext._strict_json(protocol_temp) != ext._clean_json(template):
            raise RuntimeError("temporary protocol failed exact JSON verification")
        if ext._sha256_file(protocol_temp) != digest:
            raise RuntimeError("temporary protocol changed before publication")
        if sidecar_temp.read_text(encoding="utf-8") != sidecar_text:
            raise RuntimeError("temporary protocol sidecar failed exact verification")
        protocol_identity = _freeze_file_identity(protocol_temp)
        sidecar_identity = _freeze_file_identity(sidecar_temp)
        if protocol_path.exists() or sidecar.exists():
            raise FileExistsError("protocol or SHA-256 sidecar appeared during freeze")

        protocol_published = True
        os.replace(protocol_temp, protocol_path)
        protocol_temp = None
        sidecar_published = True
        os.replace(sidecar_temp, sidecar)
        sidecar_temp = None

        if (
            _freeze_file_identity(protocol_path) != protocol_identity
            or ext._sha256_file(protocol_path) != digest
        ):
            raise RuntimeError("published protocol failed SHA-256 verification")
        if (
            _freeze_file_identity(sidecar) != sidecar_identity
            or sidecar.read_text(encoding="utf-8") != sidecar_text
        ):
            raise RuntimeError("published protocol sidecar failed exact verification")
        return digest
    except BaseException as error:
        cleanup_errors: list[Exception] = []
        for temporary in (protocol_temp, sidecar_temp):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except Exception as cleanup_error:  # pragma: no cover - OS failure
                    cleanup_errors.append(cleanup_error)
        for published, path, expected, identity in (
            (sidecar_published, sidecar, sidecar_digest, sidecar_identity),
            (protocol_published, protocol_path, digest, protocol_identity),
        ):
            if published and expected is not None and identity is not None:
                try:
                    _rollback_exact_published_file(path, expected, identity)
                except Exception as cleanup_error:  # pragma: no cover - OS failure
                    cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            details = "; ".join(str(item) for item in cleanup_errors)
            raise RuntimeError(
                "protocol freeze failed and rollback was incomplete; targets remain "
                f"fail-closed: {details}"
            ) from error
        raise


def _load_protocol(protocol_path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    protocol_path = protocol_path.resolve()
    expected = ext._require_sha256(expected_sha256, label="protocol_sha256")
    if ext._sha256_file(protocol_path) != expected:
        raise ValueError("protocol file SHA-256 differs from --protocol-sha256")
    sidecar = protocol_path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != f"{expected}  {protocol_path.name}\n":
        raise ValueError("protocol SHA-256 sidecar is missing or differs")
    protocol = ext._strict_json(protocol_path)
    if protocol.get("schema") != SCHEMA or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen protocol schema")
    return protocol, expected


def _verify_incomplete_phase_inputs(
    project_root: Path, data_dir: Path, protocol: Mapping[str, Any]
) -> dict[str, dict[str, int | str]]:
    if _implementation_snapshot(project_root) != protocol.get("implementation_snapshot"):
        raise ValueError("implementation differs from the frozen protocol")
    if _method_records(project_root) != protocol["frozen_inputs"]["artifacts"]:
        raise ValueError("method artifacts differ from the frozen protocol")
    source_records = _source_records(data_dir)
    if source_records != protocol["dataset"]["source_records"]:
        raise ValueError("Extended TEP sources differ from the frozen protocol")
    metadata = _label_and_topology_record(data_dir)
    if metadata != protocol["dataset"]["metadata_record"]:
        raise ValueError("Extended TEP HDF metadata differs from the frozen protocol")
    return source_records


def _expected_identity_arrays(phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if phase == "calibration":
        names = [
            f"ExtendedTEP-Mode1-external_calibration_IDV1_prefault-fault-00-"
            f"source-IDV-01-run-{run:03d}"
            for run in CALIBRATION_RUNS
        ]
        partitions = ["external_calibration_IDV1_prefault"] * len(CALIBRATION_RUNS)
        faults = [0] * len(CALIBRATION_RUNS)
        runs = list(CALIBRATION_RUNS)
    elif phase == "audit":
        names = [
            f"ExtendedTEP-Mode1-external_audit_healthy_IDV1_prefault-fault-00-"
            f"source-IDV-01-run-{run:03d}"
            for run in AUDIT_RUNS
        ]
        names += [
            f"ExtendedTEP-Mode1-external_audit_fault_window-fault-{fault:02d}-"
            f"source-IDV-{fault:02d}-run-{run:03d}"
            for fault in AUDIT_FAULT_IDS
            for run in AUDIT_RUNS
        ]
        partitions = ["external_audit_healthy_IDV1_prefault"] * len(AUDIT_RUNS)
        partitions += ["external_audit_fault_window"] * (
            len(AUDIT_FAULT_IDS) * len(AUDIT_RUNS)
        )
        faults = [0] * len(AUDIT_RUNS)
        faults += [fault for fault in AUDIT_FAULT_IDS for _ in AUDIT_RUNS]
        runs = list(AUDIT_RUNS) + list(AUDIT_RUNS) * len(AUDIT_FAULT_IDS)
    else:
        raise ValueError("phase must be calibration or audit")
    return (
        np.asarray(names),
        np.asarray(partitions),
        np.asarray(faults, dtype=np.int64),
        np.asarray(runs, dtype=np.int64),
    )


def _pack_paths(
    runs: Sequence[Any],
    paths: Mapping[str, DirectionalGLRPath],
    *,
    operational_threshold: float,
) -> dict[str, np.ndarray]:
    ordered = [paths[run.name] for run in runs]
    if len(ordered) != len(runs) or len(paths) != len(runs):
        raise ValueError("path/run identity count changed")
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
            raise ValueError(f"{phase} evidence field set changed")
        try:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        except ValueError as error:
            raise ValueError(f"{phase} evidence is unsafe or unreadable") from error
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("object arrays are forbidden in evidence")
    count = len(CALIBRATION_RUNS) if phase == "calibration" else (
        len(AUDIT_RUNS) * (1 + len(AUDIT_FAULT_IDS))
    )
    component_shape = (count, SEQUENCE_LENGTH, len(FAULT_IDS))
    vector_shape = (count, SEQUENCE_LENGTH)
    for name in (
        "logits", "standardized_logits", "clipped_evidence", "cusum",
        "component_persistence",
    ):
        if arrays[name].shape != component_shape or arrays[name].dtype != np.dtype(np.float64):
            raise ValueError(f"{phase} {name} shape/dtype changed")
    for name in ("envelope", "eligible", "alarm"):
        if arrays[name].shape != vector_shape:
            raise ValueError(f"{phase} {name} shape changed")
    if arrays["envelope"].dtype != np.dtype(np.float64):
        raise ValueError(f"{phase} envelope must be float64")
    if arrays["eligible"].dtype != np.dtype(np.bool_) or arrays["alarm"].dtype != np.dtype(np.bool_):
        raise ValueError(f"{phase} decision arrays must be bool")
    if arrays["run_names"].shape != (count,) or arrays["run_names"].dtype.kind != "U":
        raise ValueError(f"{phase} run names changed")
    if arrays["partitions"].shape != (count,) or arrays["partitions"].dtype.kind != "U":
        raise ValueError(f"{phase} partitions changed")
    if arrays["fault_ids"].shape != (count,) or arrays["fault_ids"].dtype != np.dtype(np.int64):
        raise ValueError(f"{phase} fault identities changed")
    if arrays["simulation_runs"].shape != (count,) or arrays["simulation_runs"].dtype != np.dtype(np.int64):
        raise ValueError(f"{phase} simulation identities changed")
    if arrays["operational_threshold"].shape != (1,) or arrays["operational_threshold"].dtype != np.dtype(np.float64):
        raise ValueError(f"{phase} threshold shape/dtype changed")
    if not math.isfinite(float(operational_threshold)) or operational_threshold < 0.0:
        raise ValueError(f"{phase} operational threshold must be finite and nonnegative")
    if float(arrays["operational_threshold"][0]) != float(operational_threshold):
        raise ValueError(f"{phase} operational threshold changed")
    if arrays["fault_start_zero_based"].dtype != np.dtype(np.int64) or not np.array_equal(
        arrays["fault_start_zero_based"], np.asarray([FAULT_START], dtype=np.int64)
    ):
        raise ValueError(f"{phase} fault onset changed")
    expected_names, expected_partitions, expected_faults, expected_runs = _expected_identity_arrays(phase)
    for actual, expected, label in (
        (arrays["run_names"], expected_names, "run names"),
        (arrays["partitions"], expected_partitions, "partitions"),
        (arrays["fault_ids"], expected_faults, "fault IDs"),
        (arrays["simulation_runs"], expected_runs, "simulation runs"),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"{phase} exact {label} changed")
    expected_eligible = np.zeros(vector_shape, dtype=bool)
    expected_eligible[:, FIRST_ALARM_ELIGIBLE:] = True
    if not np.array_equal(arrays["eligible"], expected_eligible):
        raise ValueError(f"{phase} eligibility mask changed")
    for name in ("logits", "standardized_logits", "clipped_evidence", "cusum"):
        if not np.isnan(arrays[name][:, :VALID_START]).all():
            raise ValueError(f"{phase} {name} leading prefix must be NaN")
        if not np.isfinite(arrays[name][:, VALID_START:]).all():
            raise ValueError(f"{phase} {name} scored values must be finite")
    if not np.isneginf(arrays["component_persistence"][:, :FIRST_ALARM_ELIGIBLE]).all():
        raise ValueError(f"{phase} component persistence prefix changed")
    if not np.isfinite(arrays["component_persistence"][:, FIRST_ALARM_ELIGIBLE:]).all():
        raise ValueError(f"{phase} component persistence scored values must be finite")
    if not np.isneginf(arrays["envelope"][:, :FIRST_ALARM_ELIGIBLE]).all():
        raise ValueError(f"{phase} envelope prefix changed")
    if not np.isfinite(arrays["envelope"][:, FIRST_ALARM_ELIGIBLE:]).all():
        raise ValueError(f"{phase} envelope scored values must be finite")
    replay = arrays["eligible"] & (arrays["envelope"] > operational_threshold)
    if phase == "calibration":
        if arrays["alarm"].any():
            raise ValueError("uncalibrated calibration evidence must contain no alarms")
    elif not np.array_equal(arrays["alarm"], replay):
        raise ValueError("audit alarm path is not an exact strict-threshold replay")
    return arrays


def _validate_base_parity(
    parity: Mapping[str, Any],
    *,
    expected_count: int,
    expected_names: Sequence[str],
) -> None:
    if parity.get("atol") != CPU_GPU_TOLERANCE or parity.get("run_count") != expected_count:
        raise ValueError("CPU/CUDA parity tolerance or run count changed")
    for key in (
        "all_numeric_paths_within_tolerance",
        "all_eligibility_arrays_exact",
        "all_alarm_arrays_exact",
    ):
        if parity.get(key) is not True:
            raise ValueError(f"CPU/CUDA parity gate failed: {key}")
    error_keys = {
        "logits_max_abs_error",
        "standardized_logits_max_abs_error",
        "clipped_evidence_max_abs_error",
        "cusum_max_abs_error",
        "component_persistence_max_abs_error",
        "envelope_max_abs_error",
    }
    maximum_errors = parity.get("maximum_errors")
    if not isinstance(maximum_errors, dict) or set(maximum_errors) != error_keys:
        raise ValueError("CPU/CUDA maximum-error field set changed")
    for key, raw_value in maximum_errors.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0 or value > CPU_GPU_TOLERANCE:
            raise ValueError(f"CPU/CUDA maximum error failed: {key}")
    records = parity.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("CPU/CUDA parity record count changed")
    if [record.get("run") for record in records if isinstance(record, dict)] != list(expected_names):
        raise ValueError("CPU/CUDA parity run order changed")
    record_keys = {"run", *error_keys, "eligible_exact", "alarm_exact"}
    for record in records:
        if not isinstance(record, dict) or set(record) != record_keys:
            raise ValueError("CPU/CUDA parity record schema changed")
        if record["eligible_exact"] is not True or record["alarm_exact"] is not True:
            raise ValueError("CPU/CUDA decision-array parity failed")
        for key in error_keys:
            value = float(record[key])
            if not math.isfinite(value) or value < 0.0 or value > CPU_GPU_TOLERANCE:
                raise ValueError(f"CPU/CUDA per-run error failed: {key}")


def _validate_calibration_parity(
    parity: Mapping[str, Any], certificate: Any, threshold: float
) -> None:
    names = [str(name) for name in _expected_identity_arrays("calibration")[0].tolist()]
    _validate_base_parity(parity, expected_count=100, expected_names=names)
    if any(
        parity.get(key) is not True
        for key in ("passed", "q100_driver_run_exact", "q100_rank_exact")
    ):
        raise ValueError("q100 calibration parity gate failed")
    for key in ("q100_threshold_max_abs_error", "maximum_per_run_maximum_abs_error"):
        value = float(parity.get(key, math.inf))
        if not math.isfinite(value) or value < 0.0 or value > CPU_GPU_TOLERANCE:
            raise ValueError(f"q100 calibration parity error failed: {key}")
    cpu_threshold = float(parity.get("cpu_q100_threshold", math.nan))
    cuda_threshold = float(parity.get("cuda_q100_threshold", math.nan))
    if cpu_threshold != threshold or abs(cuda_threshold - threshold) > CPU_GPU_TOLERANCE:
        raise ValueError("q100 CPU/CUDA threshold binding changed")
    if abs(cpu_threshold - cuda_threshold) != float(parity["q100_threshold_max_abs_error"]):
        raise ValueError("q100 threshold error is not an exact replay")
    driver = certificate.rank_driver_run
    if (
        parity.get("cpu_q100_driver_run") != driver
        or parity.get("cuda_q100_driver_run") != driver
    ):
        raise ValueError("q100 rank-driver binding changed")


def _certificate_from_paths(
    paths: Mapping[str, DirectionalGLRPath], protocol: Mapping[str, Any]
) -> Any:
    maxima: list[tuple[str, float]] = []
    for name in sorted(paths):
        envelope = np.asarray(paths[name].envelope, dtype=np.float64)
        if envelope.shape != (SEQUENCE_LENGTH,):
            raise ValueError(f"{name}: envelope shape changed")
        if not np.isneginf(envelope[:FIRST_ALARM_ELIGIBLE]).all():
            raise ValueError(f"{name}: pre-eligible envelope changed")
        if not np.isfinite(envelope[FIRST_ALARM_ELIGIBLE:]).all():
            raise ValueError(f"{name}: eligible envelope is non-finite")
        maxima.append((name, float(np.max(envelope[FIRST_ALARM_ELIGIBLE:]))))
    if len(maxima) != 100:
        raise ValueError("q100 calibration requires exactly 100 run maxima")
    binding = protocol["monitor_binding"]
    certificate = build_calibration_certificate(
        deployment_spec_sha256=str(binding["deployment_spec_sha256"]),
        monitor_configuration_sha256=str(binding["monitor_configuration_sha256"]),
        run_maxima=maxima,
        deferred_runs=(),
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )
    if certificate.rank.rank_one_based != PAC_RANK:
        raise AssertionError("external calibration did not produce rank 100")
    return certificate


def _calibration_parity(
    cpu_paths: Mapping[str, DirectionalGLRPath],
    gpu_paths: Mapping[str, DirectionalGLRPath],
    protocol: Mapping[str, Any],
    base_parity: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    cpu = _certificate_from_paths(cpu_paths, protocol)
    gpu = _certificate_from_paths(gpu_paths, protocol)
    cpu_maxima = dict(cpu.run_maxima)
    gpu_maxima = dict(gpu.run_maxima)
    if set(cpu_maxima) != set(gpu_maxima):
        raise AssertionError("CPU/CUDA calibration identities differ")
    maximum_error = max(abs(cpu_maxima[name] - gpu_maxima[name]) for name in cpu_maxima)
    threshold_error = abs(float(cpu.threshold) - float(gpu.threshold))
    driver_exact = cpu.rank_driver_run == gpu.rank_driver_run
    passed = (
        bool(base_parity["all_numeric_paths_within_tolerance"])
        and bool(base_parity["all_eligibility_arrays_exact"])
        and bool(base_parity["all_alarm_arrays_exact"])
        and driver_exact
        and cpu.rank.rank_one_based == gpu.rank.rank_one_based == PAC_RANK
        and threshold_error <= CPU_GPU_TOLERANCE
        and maximum_error <= CPU_GPU_TOLERANCE
    )
    if not passed:
        raise AssertionError("CPU/CUDA q100 calibration parity failed")
    details = {
        **dict(base_parity),
        "cpu_q100_threshold": float(cpu.threshold),
        "cuda_q100_threshold": float(gpu.threshold),
        "q100_threshold_max_abs_error": threshold_error,
        "cpu_q100_driver_run": cpu.rank_driver_run,
        "cuda_q100_driver_run": gpu.rank_driver_run,
        "q100_driver_run_exact": driver_exact,
        "q100_rank_exact": cpu.rank.rank_one_based == gpu.rank.rank_one_based == PAC_RANK,
        "maximum_per_run_maximum_abs_error": maximum_error,
        "operational_threshold_source": "CPU_NumPy_reference",
        "passed": passed,
    }
    return cpu, details


def _episode_starts(alarm: np.ndarray) -> np.ndarray:
    values = np.asarray(alarm, dtype=bool)
    if values.shape != (SEQUENCE_LENGTH,):
        raise ValueError("alarm path has the wrong length")
    return np.flatnonzero(values & ~np.r_[False, values[:-1]])


def _healthy_records(
    runs: Iterable[Any], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        path = paths[run.name]
        starts = _episode_starts(path.alarm)
        records.append(
            {
                "run": run.name,
                "simulation_run": int(run.metadata["simulation_run"]),
                "any_alarm": bool(starts.size),
                "alarm_episode_count": int(starts.size),
                "first_alarm_index": int(starts[0]) if starts.size else None,
                "maximum_envelope": float(np.max(path.envelope[path.eligible])),
            }
        )
    return records


def _fault_records(
    runs: Iterable[Any], paths: Mapping[str, DirectionalGLRPath]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
        records.append(
            {
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
            }
        )
    return records


def _summarize_audit(
    healthy_records: list[dict[str, Any]],
    fault_records: list[dict[str, Any]],
    parity: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    if len(healthy_records) != 100 or len(fault_records) != 300:
        raise ValueError("external audit record counts changed")
    healthy_alarms = sum(bool(row["any_alarm"]) for row in healthy_records)
    cp_upper = ext._clopper_pearson_upper(healthy_alarms, len(healthy_records))
    per_fault: dict[str, Any] = {}
    detection_counts: list[int] = []
    median_passes: list[bool] = []
    p90_passes: list[bool] = []
    all_restricted: list[int] = []
    for fault_id in AUDIT_FAULT_IDS:
        rows = [row for row in fault_records if int(row["fault_id"]) == fault_id]
        if len(rows) != 100:
            raise ValueError(f"fault {fault_id}: external audit run count changed")
        detected = sum(bool(row["detected"]) for row in rows)
        restricted = [int(row["restricted_delay_samples_miss_at_run_end"]) for row in rows]
        median = float(statistics.median(restricted))
        p90 = ext._empirical_upper_order_statistic(restricted, 0.90)
        attribution_matches = sum(bool(row["attribution_matches_fault"]) for row in rows)
        detection_counts.append(detected)
        median_passes.append(median <= MEDIAN_RESTRICTED_DELAY_MAXIMUM)
        p90_passes.append(p90 <= PER_FAULT_RESTRICTED_P90_MAXIMUM)
        all_restricted.extend(restricted)
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
    parity_pass = (
        bool(parity["all_numeric_paths_within_tolerance"])
        and bool(parity["all_eligibility_arrays_exact"])
        and bool(parity["all_alarm_arrays_exact"])
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
        "phase": "audit",
        "method": METHOD,
        "passed": passed,
        "decision": PASS_DECISION if passed else FAIL_DECISION,
        "threshold": threshold,
        "threshold_binary64_hex": threshold.hex(),
        "comparison": "strictly_greater_than",
        "healthy": {
            "run_count": len(healthy_records),
            "runs_with_any_alarm": healthy_alarms,
            "alarm_run_rate": healthy_alarms / len(healthy_records),
            "one_sided_clopper_pearson_95_upper": cp_upper,
            "empirical_cp95_upper_below_0_05": cp_upper < 0.05,
            "empirical_cp95_upper_below_0_10": cp_upper < 0.10,
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
            "total_gate_redundant_with_per_fault_gates": True,
            "overall_median_restricted_delay_samples": float(statistics.median(all_restricted)),
            "overall_median_is_report_only": True,
            "per_fault_median_delay_maximum_gate": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
            "per_fault_p90_maximum_gate": PER_FAULT_RESTRICTED_P90_MAXIMUM,
            "miss_delay_imputation_samples": MISS_DELAY,
            "delay_sample_minutes": 3,
        },
        "parity_gate_passed": parity_pass,
        "governance_interpretation": (
            "cross-simulator, external-domain-calibrated confirmation; not zero-shot; "
            "common-random-number coupling across IDVs is unknown"
        ),
    }


def _records_from_evidence(
    arrays: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    healthy: list[dict[str, Any]] = []
    faults: list[dict[str, Any]] = []
    for index, raw_name in enumerate(arrays["run_names"]):
        name = str(raw_name)
        fault_id = int(arrays["fault_ids"][index])
        simulation_run = int(arrays["simulation_runs"][index])
        alarm = arrays["alarm"][index].astype(bool)
        envelope = arrays["envelope"][index]
        eligible = arrays["eligible"][index].astype(bool)
        starts = _episode_starts(alarm)
        if fault_id == 0:
            healthy.append(
                {
                    "run": name,
                    "simulation_run": simulation_run,
                    "any_alarm": bool(starts.size),
                    "alarm_episode_count": int(starts.size),
                    "first_alarm_index": int(starts[0]) if starts.size else None,
                    "maximum_envelope": float(np.max(envelope[eligible])),
                }
            )
            continue
        post = starts[starts >= FAULT_START]
        first = int(post[0]) if post.size else None
        attributed = None
        if first is not None:
            attributed = int(
                FAULT_IDS[
                    int(np.argmax(arrays["component_persistence"][index, first]))
                ]
            )
        delay = first - FAULT_START if first is not None else None
        faults.append(
            {
                "run": name,
                "fault_id": fault_id,
                "simulation_run": simulation_run,
                "detected": first is not None,
                "first_fault_alarm_episode_index": first,
                "delay_samples": delay,
                "restricted_delay_samples_miss_at_run_end": delay if delay is not None else MISS_DELAY,
                "pre_fault_alarm_episode_count": int(np.sum(starts < FAULT_START)),
                "pre_fault_interval_used_as_performance_evidence": False,
                "attributed_fault_head_at_first_alarm": attributed,
                "attribution_matches_fault": attributed == fault_id if attributed is not None else False,
                "maximum_post_fault_envelope": float(np.max(envelope[FAULT_START:])),
            }
        )
    if len(healthy) != 100 or len(faults) != 300:
        raise ValueError("audit evidence did not reconstruct 100 healthy and 300 fault records")
    return healthy, faults


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
        "schema": "extended_tep_q100_phase_manifest",
        "schema_version": 1,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "access_intent_sha256": intent_sha256,
        "runtime_preflight_sha256": runtime_sha256,
        "source_records": dict(source_records),
        "calibration_binding": dict(calibration_binding) if calibration_binding is not None else None,
        "written_before_any_selected_processdata_or_idv_init_numeric_read": True,
    }
    return {**body, "phase_fingerprint": ext._canonical_sha256(body)}


def _expected_access_intent(
    *,
    phase: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    calibration_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if phase == "calibration":
        if calibration_binding is not None:
            raise ValueError("calibration intent cannot have a calibration binding")
        calibration = protocol["external_calibration"]
        return {
            "schema": "extended_tep_q100_access_intent",
            "schema_version": 1,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "created_before_any_selected_processdata_or_idv_init_numeric_read": True,
            "authorized_reads": {
                "source_idv": calibration["source_idv"],
                "simulation_runs": list(calibration["simulation_runs"]),
                "normalized_window_half_open": list(
                    calibration["normalized_window_half_open"]
                ),
            },
            "numeric_threshold_unknown": True,
            "forbidden_actions": list(protocol["forbidden_actions"]),
        }
    if phase == "audit":
        if calibration_binding is None:
            raise ValueError("audit intent requires the authenticated calibration binding")
        audit = protocol["external_audit"]
        return {
            "schema": "extended_tep_q100_access_intent",
            "schema_version": 1,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "created_before_any_selected_processdata_or_idv_init_numeric_read": True,
            "calibration_binding": dict(calibration_binding),
            "authorized_reads": {
                "healthy": {
                    "source_idv": audit["healthy"]["source_idv"],
                    "simulation_runs": list(audit["healthy"]["simulation_runs"]),
                    "normalized_window_half_open": list(
                        audit["healthy"]["normalized_window_half_open"]
                    ),
                },
                "faults": {
                    "fault_ids": list(audit["faults"]["fault_ids"]),
                    "simulation_runs": list(audit["faults"]["simulation_runs"]),
                    "normalized_window_half_open": list(
                        audit["faults"]["normalized_window_half_open"]
                    ),
                },
            },
            "forbidden_actions": list(protocol["forbidden_actions"]),
        }
    raise ValueError("phase must be calibration or audit")


def _expected_data_access_audit(
    *,
    phase: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    intent_sha256: str,
    phase_manifest_sha256: str,
    calibration_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if phase == "calibration":
        if calibration_binding is not None:
            raise ValueError("calibration data access cannot have a calibration binding")
        calibration = protocol["external_calibration"]
        runs = list(calibration["simulation_runs"])
        return {
            "schema": "extended_tep_q100_data_access_audit",
            "schema_version": 1,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": phase_manifest_sha256,
            "selected_source_idv": calibration["source_idv"],
            "selected_runs": [runs[0], runs[-1]],
            "selected_normalized_window_half_open": list(
                calibration["normalized_window_half_open"]
            ),
            "selected_sequence_count": len(runs),
            "audit_processdata_and_idv_init_values_read": False,
        }
    if phase == "audit":
        if calibration_binding is None:
            raise ValueError("audit data access requires the calibration binding")
        audit = protocol["external_audit"]
        healthy_runs = list(audit["healthy"]["simulation_runs"])
        fault_runs = list(audit["faults"]["simulation_runs"])
        return {
            "schema": "extended_tep_q100_data_access_audit",
            "schema_version": 1,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": phase_manifest_sha256,
            "calibration_binding": dict(calibration_binding),
            "selected_healthy_source_idv": audit["healthy"]["source_idv"],
            "selected_healthy_runs": [healthy_runs[0], healthy_runs[-1]],
            "selected_healthy_window_half_open": list(
                audit["healthy"]["normalized_window_half_open"]
            ),
            "selected_fault_ids": list(audit["faults"]["fault_ids"]),
            "selected_fault_runs": [fault_runs[0], fault_runs[-1]],
            "selected_fault_window_half_open": list(
                audit["faults"]["normalized_window_half_open"]
            ),
            "selected_sequence_count": len(healthy_runs)
            + len(audit["faults"]["fault_ids"]) * len(fault_runs),
            "threshold_selected_from_audit_rows": False,
            "audit_opened_only_after_calibration_authentication": True,
        }
    raise ValueError("phase must be calibration or audit")


def _validate_runtime_preflight(runtime: Mapping[str, Any]) -> None:
    expected_keys = {
        "device",
        "device_name",
        "compute_capability",
        "total_memory_bytes",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "dtype",
        "tf32",
        "deterministic_algorithms",
        "probe_sum_of_squares",
    }
    if set(runtime) != expected_keys:
        raise ValueError("runtime preflight schema changed")
    capability = runtime["compute_capability"]
    valid_capability = (
        isinstance(capability, list)
        and len(capability) == 2
        and all(type(value) is int and value >= 0 for value in capability)
    )
    if (
        runtime["device"] not in {"cuda", "cuda:0"}
        or not isinstance(runtime["device_name"], str)
        or not runtime["device_name"]
        or not valid_capability
        or type(runtime["total_memory_bytes"]) is not int
        or runtime["total_memory_bytes"] <= 0
        or not isinstance(runtime["torch_version"], str)
        or not runtime["torch_version"]
        or not isinstance(runtime["torch_cuda_version"], str)
        or not runtime["torch_cuda_version"]
        or runtime["cuda_available"] is not True
        or runtime["dtype"] != "float64"
        or runtime["tf32"] is not False
        or runtime["deterministic_algorithms"] is not True
        or runtime["probe_sum_of_squares"] != 5.0
    ):
        raise ValueError("runtime preflight does not prove the required CUDA float64 mode")


def _verify_phase_provenance(
    target: Path,
    *,
    phase: str,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    calibration_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    intent = ext._strict_json(target / "access_intent.json")
    expected_intent = _expected_access_intent(
        phase=phase,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        calibration_binding=calibration_binding,
    )
    if intent != expected_intent:
        raise ValueError(f"{phase} access intent is not an exact protocol replay")
    intent_sha256 = ext._sha256_file(target / "access_intent.json")

    runtime = ext._strict_json(target / "runtime_preflight.json")
    _validate_runtime_preflight(runtime)
    runtime_sha256 = ext._sha256_file(target / "runtime_preflight.json")

    phase_manifest, _ = ext._verify_fingerprinted_json(
        target / "phase_manifest.json", "phase_fingerprint"
    )
    expected_manifest = _phase_manifest(
        phase=phase,
        protocol_sha256=protocol_sha256,
        intent_sha256=intent_sha256,
        runtime_sha256=runtime_sha256,
        source_records=protocol["dataset"]["source_records"],
        calibration_binding=calibration_binding,
    )
    if phase_manifest != expected_manifest:
        raise ValueError(
            f"{phase} phase manifest is not an exact intent/runtime/source replay"
        )
    phase_manifest_sha256 = ext._sha256_file(target / "phase_manifest.json")

    data_access = ext._strict_json(target / "data_access_audit.json")
    expected_data_access = _expected_data_access_audit(
        phase=phase,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        intent_sha256=intent_sha256,
        phase_manifest_sha256=phase_manifest_sha256,
        calibration_binding=calibration_binding,
    )
    if data_access != expected_data_access:
        raise ValueError(f"{phase} data-access audit is not an exact manifest replay")

    execution = ext._strict_json(target / "gpu_execution.json")
    if (
        set(execution)
        != {"schema", "schema_version", "phase", "protocol_sha256", "runtime", "score"}
        or execution["schema"] != SCHEMA
        or execution["schema_version"] != SCHEMA_VERSION
        or execution["phase"] != phase
        or execution["protocol_sha256"] != protocol_sha256
        or execution["runtime"] != runtime
        or not isinstance(execution["score"], dict)
    ):
        raise ValueError(f"{phase} GPU execution/runtime binding changed")
    score = execution["score"]
    expected_count = len(CALIBRATION_RUNS) if phase == "calibration" else 400
    if (
        score.get("backend")
        != "pytorch_cuda_float64_with_independent_numpy_reference"
        or score.get("device") != runtime["device"]
        or score.get("dtype") != "float64"
        or score.get("tf32") is not False
        or score.get("deterministic_algorithms") is not True
        or score.get("sequence_count") != expected_count
        or score.get("sequence_length") != SEQUENCE_LENGTH
        or type(score.get("batch_size")) is not int
        or score["batch_size"] < 1
    ):
        raise ValueError(f"{phase} score execution does not match the frozen GPU mode")
    cuda_calls = score.get("cuda_calls")
    if not isinstance(cuda_calls, list) or not cuda_calls:
        raise ValueError(f"{phase} score execution has no CUDA call evidence")
    if any(
        not isinstance(call, dict)
        or call.get("device") != runtime["device"]
        or call.get("device_name") != runtime["device_name"]
        or call.get("dtype") != "float64"
        or call.get("tf32") is not False
        for call in cuda_calls
    ):
        raise ValueError(f"{phase} CUDA call/runtime binding changed")
    return execution


def _phase_staging(output_root: Path, phase: str) -> tuple[Path, Path]:
    import tempfile

    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / phase
    prefix = f".{phase}.inprogress-"
    leftovers = [path for path in output_root.iterdir() if path.name.startswith(prefix)]
    if leftovers:
        raise RuntimeError(
            "an earlier phase staging directory exists; HDF5 must not be reopened: "
            + ", ".join(str(path) for path in leftovers)
        )
    return target, Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))


def _artifact_manifest(
    root: Path, phase: str, protocol_sha256: str, artifacts: Sequence[str]
) -> dict[str, Any]:
    names = tuple(name for name in artifacts if name != "artifact_manifest.json")
    body = {
        "schema": "extended_tep_q100_artifact_manifest",
        "schema_version": 1,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "artifacts": {name: ext._hash_record(root / name) for name in names},
    }
    return {**body, "manifest_fingerprint": ext._canonical_sha256(body)}


def _completion(
    root: Path,
    phase: str,
    protocol_sha256: str,
    decision: str,
    artifacts: Sequence[str],
) -> dict[str, Any]:
    phase_manifest, phase_fingerprint = ext._verify_fingerprinted_json(
        root / "phase_manifest.json", "phase_fingerprint"
    )
    if phase_manifest.get("phase") != phase:
        raise ValueError("phase manifest differs during completion")
    body = {
        "schema": "extended_tep_q100_completion",
        "schema_version": 1,
        "phase": phase,
        "phase_fingerprint": phase_fingerprint,
        "protocol_sha256": protocol_sha256,
        "decision": decision,
        "artifacts": {name: ext._hash_record(root / name) for name in artifacts},
    }
    return {**body, "completion_fingerprint": ext._canonical_sha256(body)}


def _finalize_phase(
    *,
    staging: Path,
    target: Path,
    phase: str,
    protocol_sha256: str,
    decision: str,
    artifacts: Sequence[str],
) -> None:
    ext._write_json(
        staging / "artifact_manifest.json",
        _artifact_manifest(staging, phase, protocol_sha256, artifacts),
    )
    ext._write_json(
        staging / "completion.json",
        _completion(staging, phase, protocol_sha256, decision, artifacts),
    )
    if target.exists():
        raise FileExistsError(f"{phase} target appeared during atomic finalization")
    os.replace(staging, target)


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
        raise RuntimeError(f"partial {phase} output exists; HDF5 must not be reopened")
    completion, fingerprint = ext._verify_fingerprinted_json(
        target / "completion.json", "completion_fingerprint"
    )
    if completion.get("schema") != "extended_tep_q100_completion":
        raise ValueError(f"{phase} completion schema changed")
    if completion.get("phase") != phase or completion.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"{phase} completion binding changed")
    if completion.get("decision") not in decisions:
        raise ValueError(f"{phase} completion decision changed")
    records = completion.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(artifacts):
        raise ValueError(f"{phase} completion artifact set changed")
    expected_files = set(artifacts) | {"completion.json"}
    if {path.name for path in target.iterdir()} != expected_files:
        raise ValueError(f"{phase} directory contains missing or extra files")
    for name, expected in records.items():
        if ext._hash_record(target / name) != expected:
            raise ValueError(f"completed {phase} artifact changed: {name}")
    manifest, _ = ext._verify_fingerprinted_json(
        target / "artifact_manifest.json", "manifest_fingerprint"
    )
    if manifest != _artifact_manifest(target, phase, protocol_sha256, artifacts):
        raise ValueError(f"{phase} artifact manifest is not an exact replay")
    phase_manifest, phase_fingerprint = ext._verify_fingerprinted_json(
        target / "phase_manifest.json", "phase_fingerprint"
    )
    if phase_manifest.get("phase") != phase or phase_manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"{phase} phase manifest binding changed")
    if completion.get("phase_fingerprint") != phase_fingerprint:
        raise ValueError(f"{phase} completion/manifest fingerprints differ")
    if ext._sha256_file(target / "protocol_snapshot.json") != protocol_sha256:
        raise ValueError(f"{phase} protocol snapshot changed")
    return {**completion, "completion_fingerprint": fingerprint}


def _calibration_result(
    certificate: Any, parity: Mapping[str, Any], protocol_sha256: str
) -> dict[str, Any]:
    threshold = float(certificate.threshold)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "analysis_status": ANALYSIS_STATUS,
        "phase": "calibration",
        "decision": CALIBRATION_DECISION,
        "method": METHOD,
        "protocol_sha256": protocol_sha256,
        "threshold": threshold,
        "threshold_binary64_hex": threshold.hex(),
        "threshold_source": "CPU_NumPy_reference",
        "cuda_threshold": float(parity["cuda_q100_threshold"]),
        "cpu_cuda_threshold_abs_error": float(parity["q100_threshold_max_abs_error"]),
        "comparison": "strictly_greater_than",
        "rank_one_based": certificate.rank.rank_one_based,
        "rank_rule": "maximum_order_statistic_no_interpolation",
        "rank_driver_run": certificate.rank_driver_run,
        "achieved_confidence": certificate.rank.achieved_confidence,
        "upper_tail_probability_bound": certificate.rank.upper_tail_probability_bound,
        "minimum_sample_count": certificate.rank.minimum_sample_count,
        "epsilon": PAC_EPSILON,
        "delta": PAC_DELTA,
        "calibration_run_count": 100,
        "calibration_exceedance_runs_under_strict_comparison": sum(
            maximum > threshold for _, maximum in certificate.run_maxima
        ),
        "prior_Rieth_numeric_threshold_reused": False,
        "threshold_derived_only_from_external_calibration_evidence": True,
        "audit_opened": False,
    }


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
    execution = _verify_phase_provenance(
        target,
        phase="calibration",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        calibration_binding=None,
    )
    result = ext._strict_json(target / "calibration_result.json")
    certificate_json = ext._strict_json(target / "pac_certificate.json")
    parity = ext._strict_json(target / "parity.json")
    threshold = float(result.get("threshold"))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("calibration threshold is invalid")
    arrays = _validate_evidence(
        target / "calibration_evidence.npz",
        phase="calibration",
        operational_threshold=threshold,
    )
    maxima = [
        (str(name), float(np.max(envelope[FIRST_ALARM_ELIGIBLE:])))
        for name, envelope in zip(arrays["run_names"], arrays["envelope"])
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
    if recomputed.as_dict() != certificate_json or float(recomputed.threshold) != threshold:
        raise ValueError("calibration evidence/certificate/threshold disagree")
    _validate_calibration_parity(parity, recomputed, threshold)
    if result != _calibration_result(recomputed, parity, protocol_sha256):
        raise ValueError("calibration result is not an exact evidence replay")
    if not (
        bool(parity.get("passed"))
        and bool(parity.get("all_numeric_paths_within_tolerance"))
        and bool(parity.get("all_eligibility_arrays_exact"))
        and bool(parity.get("all_alarm_arrays_exact"))
    ):
        raise ValueError("calibration parity artifact failed")
    if (target / "REPORT.md").read_text(encoding="utf-8") != _calibration_report(
        result, execution
    ):
        raise ValueError("calibration report is not an exact result/runtime replay")
    return {
        "completion": completion,
        "completion_sha256": ext._sha256_file(target / "completion.json"),
        "certificate_sha256": ext._sha256_file(target / "pac_certificate.json"),
        "evidence_sha256": ext._sha256_file(target / "calibration_evidence.npz"),
        "threshold": threshold,
        "threshold_binary64_hex": threshold.hex(),
    }


def _calibration_binding(calibration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "calibration_completion_sha256": calibration["completion_sha256"],
        "calibration_completion_fingerprint": calibration["completion"]["completion_fingerprint"],
        "pac_certificate_sha256": calibration["certificate_sha256"],
        "calibration_evidence_sha256": calibration["evidence_sha256"],
        "operational_threshold": calibration["threshold"],
        "operational_threshold_binary64_hex": calibration["threshold_binary64_hex"],
        "threshold_source": "CPU_NumPy_reference",
        "calibration_decision": CALIBRATION_DECISION,
    }


def _verify_audit_bundle(
    target: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    calibration_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    completion = _verify_completed_phase(
        target,
        phase="audit",
        protocol_sha256=protocol_sha256,
        artifacts=AUDIT_ARTIFACTS,
        decisions={PASS_DECISION, FAIL_DECISION},
    )
    if completion is None:
        return None
    execution = _verify_phase_provenance(
        target,
        phase="audit",
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        calibration_binding=calibration_binding,
    )
    threshold = float(calibration_binding["operational_threshold"])
    arrays = _validate_evidence(
        target / "audit_evidence.npz", phase="audit", operational_threshold=threshold
    )
    parity = ext._strict_json(target / "parity.json")
    audit_names = [str(name) for name in _expected_identity_arrays("audit")[0].tolist()]
    _validate_base_parity(parity, expected_count=400, expected_names=audit_names)
    healthy, faults = _records_from_evidence(arrays)
    expected = _summarize_audit(healthy, faults, parity, threshold)
    expected.update(
        {
            "protocol_sha256": protocol_sha256,
            "calibration_binding": dict(calibration_binding),
            "pac_primary_claim": protocol["success_gates"]["pac_primary_claim"],
        }
    )
    result = ext._strict_json(target / "result.json")
    if result != expected or result["decision"] != completion["decision"]:
        raise ValueError("audit result is not an exact evidence/gate replay")
    if (target / "REPORT.md").read_text(encoding="utf-8") != _audit_report(
        result, execution
    ):
        raise ValueError("audit report is not an exact result/runtime replay")
    return {**completion, "completion_fingerprint": completion["completion_fingerprint"]}


def _assert_protocol_contract(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != SCHEMA or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("frozen protocol schema changed")
    if protocol.get("analysis_status") != ANALYSIS_STATUS or protocol.get("method") != METHOD:
        raise ValueError("frozen protocol analysis binding changed")
    if tuple(FAULT_IDS) != tuple(AUDIT_FAULT_IDS) or tuple(FAULT_IDS) != (3, 9, 15):
        raise ValueError("fault-head identity/order changed")
    if FIRST_ALARM_ELIGIBLE != VALID_START + ext.PERSISTENCE_WINDOW - 1:
        raise ValueError("eligibility/persistence geometry changed")
    source = protocol["dataset"]["source_records"][DATA_FILE]
    if source["size_bytes"] != MODE1_SIZE_BYTES or source["md5"] != MODE1_MD5:
        raise ValueError("protocol does not bind the official Mode 1 source")
    calibration = protocol["external_calibration"]
    if (
        calibration["source_idv"] != 1
        or calibration["magnitude_percent"] != 100
        or calibration["simulation_runs"] != list(CALIBRATION_RUNS)
        or calibration["normalized_window_half_open"] != list(CALIBRATION_WINDOW)
        or calibration["rank"]["rank_one_based"] != PAC_RANK
    ):
        raise ValueError("external calibration contract changed")
    audit = protocol["external_audit"]
    if (
        audit["healthy"]["source_idv"] != 1
        or audit["healthy"]["simulation_runs"] != list(AUDIT_RUNS)
        or audit["healthy"]["normalized_window_half_open"] != list(CALIBRATION_WINDOW)
        or audit["faults"]["fault_ids"] != list(AUDIT_FAULT_IDS)
        or audit["faults"]["simulation_runs"] != list(AUDIT_RUNS)
        or audit["faults"]["normalized_window_half_open"] != list(FAULT_WINDOW)
        or audit["faults"]["local_fault_start_zero_based"] != FAULT_START
        or audit["fault_detection_requires_new_alarm_episode_start_at_or_after"]
        != FAULT_START
        or audit["pre_fault_alarm_continuing_across_onset_is_not_a_detection"]
        is not True
    ):
        raise ValueError("external audit contract changed")
    monitor = protocol["monitor_binding"]["monitor_configuration"]
    if (
        monitor["threshold_rule"] != "PAC_RMax_maximum_q100_of_100_external_run_maxima"
        or monitor["epsilon"] != PAC_EPSILON
        or monitor["delta"] != PAC_DELTA
        or monitor["comparison"] != "strictly_greater_than"
    ):
        raise ValueError("monitor threshold contract changed")
    expected_gates = {
        "healthy_alarm_runs_maximum": HEALTHY_ALARM_RUNS_MAXIMUM,
        "per_fault_detected_runs_minimum": PER_FAULT_DETECTIONS_MINIMUM,
        "total_fault_detected_runs_minimum": TOTAL_FAULT_DETECTIONS_MINIMUM,
        "per_fault_median_restricted_delay_samples_maximum": MEDIAN_RESTRICTED_DELAY_MAXIMUM,
        "per_fault_restricted_p90_delay_samples_maximum": PER_FAULT_RESTRICTED_P90_MAXIMUM,
        "restricted_delay_miss_imputation_samples": MISS_DELAY,
    }
    for key, expected in expected_gates.items():
        if protocol["success_gates"].get(key) != expected:
            raise ValueError(f"success gate changed: {key}")
    if protocol.get("forbidden_actions") != list(FORBIDDEN_ACTIONS):
        raise ValueError("forbidden action contract changed")


def _assert_no_staging(output_root: Path, phase: str) -> None:
    if not output_root.exists():
        return
    prefix = f".{phase}.inprogress-"
    leftovers = [path for path in output_root.iterdir() if path.name.startswith(prefix)]
    if leftovers:
        raise RuntimeError(
            "an earlier phase staging directory exists; HDF5 must not be reopened: "
            + ", ".join(str(path) for path in leftovers)
        )


def _calibration_report(result: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Extended TEP Mode 1 q100 calibration",
            "",
            f"- Analysis status: {ANALYSIS_STATUS}.",
            "- Source: IDV1 100% runs 1-100, pre-fault normalized window [0,500).",
            f"- CPU NumPy q100 threshold: {result['threshold']!r}.",
            f"- CUDA q100 threshold: {result['cuda_threshold']!r}.",
            f"- Rank: {result['rank_one_based']}/100; strict comparison: >.",
            f"- CPU/CUDA threshold absolute error: {result['cpu_cuda_threshold_abs_error']:.3e}.",
            f"- CUDA device: {execution['runtime']['device_name']}.",
            "",
            "No audit window was opened. This is domain calibration, not zero-shot validation.",
        ]
    )


def _audit_report(result: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    lines = [
        "# Extended TEP Mode 1 external confirmation",
        "",
        f"- Decision: {result['decision']}.",
        f"- Frozen external q100 threshold: {result['threshold']!r} (strict >).",
        (
            f"- Healthy any-alarm runs: {result['healthy']['runs_with_any_alarm']}/100; "
            f"one-sided CP95 upper: {result['healthy']['one_sided_clopper_pearson_95_upper']:.6f}."
        ),
        f"- CUDA device: {execution['runtime']['device_name']}.",
        "",
        "| Fault | Detected | Median restricted delay | P90 restricted delay | Attribution |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fault in AUDIT_FAULT_IDS:
        row = result["faults"]["per_fault"][f"{fault:02d}"]
        lines.append(
            f"| {fault} | {row['detected_runs']}/100 | "
            f"{row['median_restricted_delay_samples']:.1f} | "
            f"{row['p90_restricted_delay_samples_higher_order_statistic']:.1f} | "
            f"{row['attribution_matches_fault_runs']}/100 |"
        )
    lines.extend(
        [
            "",
            "Delay is measured in three-minute samples; misses are imputed as 340.",
            "Only a new alarm episode whose start is at or after local index 160 counts as detection.",
            "Attribution is report-only. Common-random-number coupling across IDVs is unknown.",
            "This is cross-simulator, external-domain-calibrated evidence, not zero-shot evidence.",
            "No training, selection, threshold change, or retuning on audit rows is authorized.",
        ]
    )
    return "\n".join(lines)


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
        print("verified completed calibration; no HDF5 source was accessed", flush=True)
        return verified
    _assert_no_staging(output_root, "calibration")
    source_records = _verify_incomplete_phase_inputs(project_root, data_dir, protocol)
    runtime = ext._runtime_preflight(device)
    transformer, uncalibrated = ext._load_method(project_root)
    if uncalibrated.threshold is not None:
        raise ValueError("calibration requires the frozen uncalibrated model")
    target, staging = _phase_staging(output_root, "calibration")
    try:
        ext._copy_protocol_snapshot(protocol_path, staging)
        ext._write_json(staging / "runtime_preflight.json", runtime)
        access_intent = {
            "schema": "extended_tep_q100_access_intent",
            "schema_version": 1,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "created_before_any_selected_processdata_or_idv_init_numeric_read": True,
            "authorized_reads": {
                "source_idv": 1,
                "simulation_runs": list(CALIBRATION_RUNS),
                "normalized_window_half_open": list(CALIBRATION_WINDOW),
            },
            "numeric_threshold_unknown": True,
            "forbidden_actions": list(FORBIDDEN_ACTIONS),
        }
        ext._write_json(staging / "access_intent.json", access_intent)
        intent_sha256 = ext._sha256_file(staging / "access_intent.json")
        ext._write_json(
            staging / "phase_manifest.json",
            _phase_manifest(
                phase="calibration",
                protocol_sha256=protocol_sha256,
                intent_sha256=intent_sha256,
                runtime_sha256=ext._sha256_file(staging / "runtime_preflight.json"),
                source_records=source_records,
                calibration_binding=None,
            ),
        )
        print("calibration: opening only IDV1 runs 1..100 window [0,500)", flush=True)
        runs = ExtendedTEPMode1Archive(data_dir / DATA_FILE).load_calibration_runs()
        if len(runs) != 100:
            raise RuntimeError("calibration loader did not return exactly 100 runs")
        data_access = {
            "schema": "extended_tep_q100_data_access_audit",
            "schema_version": 1,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": ext._sha256_file(staging / "phase_manifest.json"),
            "selected_source_idv": 1,
            "selected_runs": [1, 100],
            "selected_normalized_window_half_open": list(CALIBRATION_WINDOW),
            "selected_sequence_count": len(runs),
            "audit_processdata_and_idv_init_values_read": False,
        }
        ext._write_json(staging / "data_access_audit.json", data_access)
        print("calibration: CUDA float64 scoring plus full NumPy parity", flush=True)
        gpu_paths, cpu_paths, base_parity, score_execution = ext._score_with_full_parity(
            transformer,
            uncalibrated,
            runs,
            device=device,
            batch_size=batch_size,
        )
        certificate, parity = _calibration_parity(
            cpu_paths, gpu_paths, protocol, base_parity
        )
        threshold = float(certificate.threshold)
        _validate_calibration_parity(parity, certificate, threshold)
        result = _calibration_result(certificate, parity, protocol_sha256)
        execution = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "phase": "calibration",
            "protocol_sha256": protocol_sha256,
            "runtime": runtime,
            "score": score_execution,
        }
        ext._write_json(staging / "parity.json", parity)
        ext._write_json(staging / "pac_certificate.json", certificate.as_dict())
        ext._write_json(staging / "calibration_result.json", result)
        ext._write_json(staging / "gpu_execution.json", execution)
        ext._write_npz(
            staging / "calibration_evidence.npz",
            _pack_paths(runs, cpu_paths, operational_threshold=threshold),
        )
        _validate_evidence(
            staging / "calibration_evidence.npz",
            phase="calibration",
            operational_threshold=threshold,
        )
        ext._write_text(staging / "REPORT.md", _calibration_report(result, execution))
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
        print(f"calibration frozen: CPU q100={threshold:.17g}", flush=True)
        return verified
    except Exception:
        print(
            f"calibration not finalized; staging retained at {staging}",
            file=sys.stderr,
            flush=True,
        )
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
        raise RuntimeError("audit cannot open before a completed q100 calibration")
    calibration_binding = _calibration_binding(calibration)
    threshold = float(calibration_binding["operational_threshold"])
    target = output_root / "audit"
    verified = _verify_audit_bundle(
        target, protocol, protocol_sha256, calibration_binding
    )
    if verified is not None:
        print("verified completed audit; no HDF5 source was accessed", flush=True)
        return verified
    _assert_no_staging(output_root, "audit")
    source_records = _verify_incomplete_phase_inputs(project_root, data_dir, protocol)
    runtime = ext._runtime_preflight(device)
    transformer, uncalibrated = ext._load_method(project_root)
    model = uncalibrated.with_threshold(threshold)
    target, staging = _phase_staging(output_root, "audit")
    try:
        ext._copy_protocol_snapshot(protocol_path, staging)
        ext._write_json(staging / "runtime_preflight.json", runtime)
        access_intent = {
            "schema": "extended_tep_q100_access_intent",
            "schema_version": 1,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "created_before_any_selected_processdata_or_idv_init_numeric_read": True,
            "calibration_binding": calibration_binding,
            "authorized_reads": {
                "healthy": {
                    "source_idv": 1,
                    "simulation_runs": list(AUDIT_RUNS),
                    "normalized_window_half_open": list(CALIBRATION_WINDOW),
                },
                "faults": {
                    "fault_ids": list(AUDIT_FAULT_IDS),
                    "simulation_runs": list(AUDIT_RUNS),
                    "normalized_window_half_open": list(FAULT_WINDOW),
                },
            },
            "forbidden_actions": list(FORBIDDEN_ACTIONS),
        }
        ext._write_json(staging / "access_intent.json", access_intent)
        intent_sha256 = ext._sha256_file(staging / "access_intent.json")
        ext._write_json(
            staging / "phase_manifest.json",
            _phase_manifest(
                phase="audit",
                protocol_sha256=protocol_sha256,
                intent_sha256=intent_sha256,
                runtime_sha256=ext._sha256_file(staging / "runtime_preflight.json"),
                source_records=source_records,
                calibration_binding=calibration_binding,
            ),
        )
        print(
            "audit: opening IDV1/3/9/15 runs 101..200 only in frozen windows",
            flush=True,
        )
        healthy, faults = ExtendedTEPMode1Archive(data_dir / DATA_FILE).load_audit_runs()
        if len(healthy) != 100 or len(faults) != 300:
            raise RuntimeError("audit loader returned the wrong run count")
        all_runs = healthy + faults
        data_access = {
            "schema": "extended_tep_q100_data_access_audit",
            "schema_version": 1,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "access_intent_sha256": intent_sha256,
            "phase_manifest_sha256": ext._sha256_file(staging / "phase_manifest.json"),
            "calibration_binding": calibration_binding,
            "selected_healthy_source_idv": 1,
            "selected_healthy_runs": [101, 200],
            "selected_healthy_window_half_open": list(CALIBRATION_WINDOW),
            "selected_fault_ids": list(AUDIT_FAULT_IDS),
            "selected_fault_runs": [101, 200],
            "selected_fault_window_half_open": list(FAULT_WINDOW),
            "selected_sequence_count": len(all_runs),
            "threshold_selected_from_audit_rows": False,
            "audit_opened_only_after_calibration_authentication": True,
        }
        ext._write_json(staging / "data_access_audit.json", data_access)
        print("audit: CUDA float64 scoring of 400 runs plus full NumPy parity", flush=True)
        gpu_paths, cpu_paths, parity, score_execution = ext._score_with_full_parity(
            transformer,
            model,
            all_runs,
            device=device,
            batch_size=batch_size,
        )
        if set(gpu_paths) != set(cpu_paths):
            raise RuntimeError("CPU/CUDA audit path identities differ")
        _validate_base_parity(
            parity, expected_count=400, expected_names=[run.name for run in all_runs]
        )
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
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "phase": "audit",
            "protocol_sha256": protocol_sha256,
            "runtime": runtime,
            "score": score_execution,
        }
        ext._write_json(staging / "result.json", result)
        ext._write_json(staging / "parity.json", parity)
        ext._write_json(staging / "gpu_execution.json", execution)
        ext._write_npz(
            staging / "audit_evidence.npz",
            _pack_paths(all_runs, gpu_paths, operational_threshold=threshold),
        )
        arrays = _validate_evidence(
            staging / "audit_evidence.npz",
            phase="audit",
            operational_threshold=threshold,
        )
        replay_healthy, replay_faults = _records_from_evidence(arrays)
        replay = _summarize_audit(replay_healthy, replay_faults, parity, threshold)
        replay.update(
            {
                "protocol_sha256": protocol_sha256,
                "calibration_binding": calibration_binding,
                "pac_primary_claim": protocol["success_gates"]["pac_primary_claim"],
            }
        )
        if replay != result:
            raise AssertionError("in-memory audit result differs from saved evidence replay")
        ext._write_text(staging / "REPORT.md", _audit_report(result, execution))
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
        print(
            f"audit not finalized; staging retained at {staging}",
            file=sys.stderr,
            flush=True,
        )
        raise


def run_two_stage_confirmation(
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
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be a positive integer")
    protocol, protocol_sha256 = _load_protocol(protocol_path, protocol_sha256)
    _assert_protocol_contract(protocol)
    latest: dict[str, Any] = {}
    if phase in {"calibration", "all"}:
        latest = _run_calibration(
            project_root=project_root.resolve(),
            data_dir=data_dir.resolve(),
            output_root=output_root.resolve(),
            protocol_path=protocol_path.resolve(),
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            device=device,
            batch_size=batch_size,
        )
    if phase in {"audit", "all"}:
        latest = _run_audit(
            project_root=project_root.resolve(),
            data_dir=data_dir.resolve(),
            output_root=output_root.resolve(),
            protocol_path=protocol_path.resolve(),
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            device=device,
            batch_size=batch_size,
        )
    return latest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="freeze protocol without reading selected numeric HDF values"
    )
    freeze.add_argument("data_dir", type=Path)
    freeze.add_argument("--protocol", type=Path, required=True)
    run = subparsers.add_parser("run", help="execute or authenticate the two sealed phases")
    run.add_argument("data_dir", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--protocol-sha256", required=True)
    run.add_argument("--phase", choices=("calibration", "audit", "all"), default="all")
    run.add_argument("--device", choices=("cuda", "cuda:0"), required=True)
    run.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.command == "freeze":
        digest = freeze_protocol(project_root, args.data_dir, args.protocol)
        print(digest, flush=True)
        return 0
    run_two_stage_confirmation(
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
