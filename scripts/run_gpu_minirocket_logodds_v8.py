#!/usr/bin/env python
"""Fail-closed one-shot runner for GPU MiniROCKET Log-Odds CUSUM v8.

This module is intentionally HDF5-free.  It validates and retains one
production backend module before the exclusive claim; only that object may
execute after the access intent has been committed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
from importlib.machinery import SourceFileLoader
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Any
import uuid

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ENVIRONMENT_SITE_PACKAGES = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
# The pinned third-party distribution must win over any project-root shadow.
# Local trusted modules are subsequently loaded from hash-checked source bytes.
for _trusted_path in (ENVIRONMENT_SITE_PACKAGES, SRC_ROOT, PROJECT_ROOT):
    while str(_trusted_path) in sys.path:
        sys.path.remove(str(_trusted_path))
sys.path[:0] = [str(ENVIRONMENT_SITE_PACKAGES), str(SRC_ROOT), str(PROJECT_ROOT)]

DOCUMENT_PATH = PROJECT_ROOT / "docs" / "gpu_minirocket_logodds_v8.md"
CHARTER_MD_PATH = PROJECT_ROOT / "docs" / "gpu_minirocket_logodds_v8_contingency.md"
CHARTER_JSON_PATH = PROJECT_ROOT / "docs" / "gpu_minirocket_logodds_v8_contingency.json"
LEDGER_PATH = PROJECT_ROOT / "protocols" / "gpu_minirocket_logodds_v8_history_and_access_ledger.json"
PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "gpu_minirocket_logodds_v8_mode2_development.json"
SOURCE_PATH = PROJECT_ROOT / "data" / "extended_tep_dtu_mode2" / "TEP_Mode2.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "gpu_minirocket_logodds_v8"
DEVELOPMENT_DIRECTORY_NAME = "v8_fixed_formal"
GLOBAL_CLAIM_PATH = DEFAULT_OUTPUT_ROOT / "development_access.claim.json"
GLOBAL_CLAIM_FAILURE_PATH = DEFAULT_OUTPUT_ROOT / "development_access.claim.failure.json"
FORMAL_PYTHON_EXE = Path(r"D:\py\anaocnda\envs\torch5090\python.exe")
FORMAL_ENV_PREFIX = FORMAL_PYTHON_EXE.parent

V7_RESULT_PATH = PROJECT_ROOT / "outputs" / "farm_rankcusum_v7" / "v7_fixed_formal" / "development_result.json"
V7_DECISION_PATH = PROJECT_ROOT / "outputs" / "farm_rankcusum_v7" / "v7_fixed_formal" / "artifacts" / "development_decision_before_final_refit.json"
V7_COMPLETION_PATH = PROJECT_ROOT / "outputs" / "farm_rankcusum_v7" / "v7_fixed_formal" / "completion.json"
V7_PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "farm_rankcusum_v7_mode2_development.json"

PROTOCOL_SCHEMA = "gpu_minirocket_logodds_v8_preregistered_development_protocol"
PROTOCOL_ID = "gpu-minirocket-logodds-v8-mode2-development"
CANDIDATE_ID = "gpu-minirocket-logodds-cusum-v8-fixed"
BACKEND_MODULE = "mate_spc_minirocket_v8.backend"
BACKEND_RESULT_SCHEMA = "gpu_minirocket_logodds_v8_development_result"
BACKEND_REQUIRED_FUNCTIONS = (
    "cuda_preflight", "validate_protocol", "preclaim_bind_method_graph",
    "run_development", "load_frozen_fold_pipeline", "load_final_refit_pipeline",
)
EXPECTED_PRECLAIM_TEST_COUNT = 66
HASH_CONTRACT_SOURCE_SHA256 = "c3dcae32194e892f758c62c99853c0eb15691f13df0ac9ef0b094ecb92a02503"
HASH_CONTRACT_SOURCE_SIZE = 6_680

SOURCE_SHA256 = "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
SOURCE_SIZE_BYTES = 23_897_125_720
FOLD_MAP_SHA256 = "64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13"
ACCESS_PLAN_SHA256 = "74b19927fee7d31f33c3b64d62a0c5235b58ad41e5d2d7d28b583a6ff4e43466"
UPSTREAM_ANNOTATED_TAG_OBJECT_SHA = "2bca845b2905aa2cfcb03a2e1615bdb196b821e5"
UPSTREAM_PEELED_COMMIT_SHA = "f9ecc6b6df798946fa5fb290ac2ade9a00815289"

ENVIRONMENT_LOCK: dict[str, Any] = {
    "python_version": "3.11.14",
    "numpy_version": "2.4.0",
    "h5py_distribution_version": "3.15.1",
    "torch_version": "2.10.0.dev20251124+cu128",
    "torch_cuda_build": "12.8",
    "cudnn_version": 91002,
    "nvidia_driver_version": "595.97",
    "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "gpu_compute_capability": [12, 0],
    "gpu_total_memory_bytes": 25_650_855_936,
    "nvidia_smi_memory_total_mib_rounded": 24_463,
}

V7_PROTOCOL_CANONICAL_JSON_SHA256 = "c27242c66a650230390d68502245035f397bb61443c16af8152d8a091dcce977"
V7_RESULT_SHA256 = "62411efc8e5d454297de9c66d05d5b96b199e82d95ace19fa588359fd8574d4a"
V7_DECISION_SHA256 = "7e1a1c3f131e0d87a9069cb07641b1a4a9744006359cfe45add8f26935e1042d"
V7_COMPLETION_SHA256 = "2026202b18fc1488d52119237b7b0b39a1735194f986c3bd53e8a45b9681dc0a"
CHARTER_JSON_V7_BOUND_SHA256 = "c1815e4c15785729c5beb50f5f7e114fa65c8ec11a08a6003241f2a8812fc09a"
CHARTER_JSON_V7_BOUND_SIZE = 10_976
CHARTER_MD_V7_BOUND_SHA256 = "745809badff6f3cdcf337eda81276d81eb10f043b4dde0b92d2cc6845029c8db"
CHARTER_MD_V7_BOUND_SIZE = 13_491

FOLD_RUN_IDS: dict[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}
SEED_CONNECTED_COMPONENTS = ((14, 43, 98), (6, 76), (9, 88), (24, 33))
CONDITION_IDVS = (1, 3, 9, 15)

BASE_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "docs/gpu_minirocket_logodds_v8.md",
    "docs/gpu_minirocket_logodds_v8_contingency.md",
    "docs/gpu_minirocket_logodds_v8_contingency.json",
    "protocols/gpu_minirocket_logodds_v8_history_and_access_ledger.json",
    "scripts/run_gpu_minirocket_logodds_v8.py",
    "src/mate_spc/hash_contract.py",
)

METHOD_CONFIG: dict[str, Any] = {
    "candidate_id": CANDIDATE_ID,
    "candidate_count": 1,
    "activation": {
        "predecessor_candidate_id": "farm-rankcusum-v7-fixed",
        "required_exact_result": "DEVELOPMENT_NO_GO",
        "v7_protocol_canonical_json_sha256": V7_PROTOCOL_CANONICAL_JSON_SHA256,
        "v7_result_file_bytes_sha256": V7_RESULT_SHA256,
        "v7_decision_evidence_file_bytes_sha256": V7_DECISION_SHA256,
        "v7_completion_file_bytes_sha256": V7_COMPLETION_SHA256,
        "v7_numeric_metrics_used_to_modify_candidate": False,
    },
    "upstream": {
        "repository": "https://github.com/timeseriesAI/tsai",
        "tag": "v1.0.1",
        "tag_kind": "annotated",
        "annotated_tag_object_sha": UPSTREAM_ANNOTATED_TAG_OBJECT_SHA,
        "peeled_commit_sha": UPSTREAM_PEELED_COMMIT_SHA,
        "source_path": "tsai/models/MINIROCKETPlus_Pytorch.py",
        "license": "Apache-2.0",
    },
    "data": {
        "mode": 2, "fault_regime": "SingleFault", "magnitude_percent": 50,
        "idvs": [1, 3, 9, 15], "runs_per_idv": 100,
        "rows_per_run": 500, "process_channel_count": 52,
        "healthy_window_zero_based_half_open": [100, 600],
        "fault_window_zero_based_half_open": [440, 940],
        "fault_local_onset_index": 160, "miss_delay_samples": 340,
    },
    "normalization": {
        "fit_healthy_runs_per_fold": 60, "channel_count": 52,
        "location": "float64_median", "scale": "max(1.4826*MAD,1e-6)",
        "clip": [-10.0, 10.0], "output_dtype": "float32",
    },
    "minirocket": {
        "seed": 20260912, "window_shape": [52, 128], "kernel_length": 9,
        "kernel_count": 84, "requested_feature_count": 10000,
        "effective_feature_count": 9996, "max_dilations_per_kernel": 32,
        "dilation_integer_range": [1, 15],
        "features_per_dilation_per_kernel": [30, 19, 12, 12, 8, 4, 8, 4, 4, 3, 3, 3, 3, 3, 3],
        "max_num_channels": 9, "add_lsaz": False,
        "bias_fit_chunk_size": 1260, "ppv_comparison": ">",
        "feature_batch_size": 128, "drop_last": False,
        "training_window_count": 20400,
    },
    "linear_head": {
        "shape": [9996, 4], "parameter_count": 39988,
        "class_order": ["H", "F3", "F9", "F15"],
        "initialization": "all_zero", "class_weights": ["85/109", "85/77", "85/77", "85/77"],
        "l2": 1e-4, "optimizer": "torch.optim.LBFGS",
        "lr": 1.0, "max_iter": 100, "max_eval": 125,
        "history_size": 20, "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9, "line_search_fn": "strong_wolfe",
    },
    "training_examples": {
        "stable_order": ["class_order_H_F3_F9_F15", "source_or_run_id_ascending", "endpoint_ascending"],
        "total": 20400, "H_total": 6540, "H_healthy_count": 5100,
        "H_fault_prefix_count": 1440, "fault_class_count_each": 4620,
        "healthy_endpoints": [160, 496, 4],
        "fault_prefix_endpoints": [128, 132, 136, 140, 144, 148, 152, 156],
        "mature_fault_endpoints": [192, 496, 4],
        "fault_transition_endpoints_ignored": [160, 164, 168, 172, 176, 180, 184, 188],
    },
    "monitor": {
        "prefix_endpoints": [128, 132, 136, 140, 144, 148, 152, 156],
        "dense_monitor_endpoints_inclusive": [160, 499],
        "calibration_fit_healthy_value_count": 20400,
        "calibration_scale_floor": 1e-3, "cusum_drift": 0.25,
        "intermittent_first_endpoint": 175, "intermittent_window": 16,
        "intermittent_top_r": 4, "intermittent_coefficient": 0.5,
        "dtype": "cpu_float64",
    },
    "inner_threshold": {
        "healthy_runs": 20, "fault_values_used": False,
        "ascending_rank": 20, "comparison": ">",
        "minimum_exact_distinct_maxima": 18,
        "tail_gap_multiplier": 100.0, "repeat_error_floor": 1e-6,
    },
    "determinism": {
        "seed": 20260912, "authoritative_device": "CUDA RTX 5090",
        "same_cuda_complete_replays": 2, "same_cuda_max_abs_error": 1e-6,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "cpu_reference_is_go_gate": False,
    },
    "go_only_final_refit": {
        "trigger": "all five development folds pass",
        "runs_per_class": 100, "mode1_access_authorized": False,
    },
}

GO_GATES: dict[str, Any] = {
    "logic": "all required", "numeric_folds_valid_required": 5,
    "healthy_false_alarms_overall_max": 10,
    "healthy_false_alarms_per_fold_max": 4,
    "fault_detection_rate_each_overall_min": 0.85,
    "fault_detection_rate_each_per_fold_min": 0.70,
    "pooled_fault_median_delay_max": 140,
    "pooled_fault_nearest_rank_p90_delay_max": 220,
    "pooled_fault_run_count": 300, "miss_delay": 340,
    "authoritative_cuda_repeat_gate": True,
    "cross_device_cpu_cuda_gate": False,
}

_HASH_CONTRACT_MODULE_NAME = "_gpu_minirocket_v8_hash_contract"
_HASH_CONTRACT_MODULE: ModuleType | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_contract_module() -> ModuleType:
    global _HASH_CONTRACT_MODULE
    if _HASH_CONTRACT_MODULE is None:
        path = SRC_ROOT / "mate_spc" / "hash_contract.py"
        spec = importlib.util.spec_from_file_location(_HASH_CONTRACT_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load hash contract from {path}")
        payload = path.read_bytes()
        if (len(payload) != HASH_CONTRACT_SOURCE_SIZE or
                hashlib.sha256(payload).hexdigest() != HASH_CONTRACT_SOURCE_SHA256):
            raise RuntimeError("hash-contract source bytes differ from audited identity")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_HASH_CONTRACT_MODULE_NAME] = module
        exec(compile(payload, str(path), "exec", dont_inherit=True), module.__dict__)
        _HASH_CONTRACT_MODULE = module
    return _HASH_CONTRACT_MODULE


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number forbidden")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_json(value.tolist())
    if hasattr(value, "item"):
        return _plain_json(value.item())
    if hasattr(value, "to_document"):
        return _plain_json(value.to_document())
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return _hash_contract_module().canonical_json_bytes(_plain_json(value))


def _json_file_bytes(value: Any) -> bytes:
    return _hash_contract_module().canonical_json_file_bytes(_plain_json(value))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = _hash_contract_module().strict_json_from_bytes(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    payload = _json_file_bytes(value)
    _write_exclusive(path, payload)
    return {
        "file_bytes_sha256": _hash_bytes(payload),
        "canonical_json_sha256": _hash_bytes(payload[:-1]),
        "file_size_bytes": len(payload),
        "canonical_json_size_bytes": len(payload) - 1,
    }


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _commit_path(path: Path) -> Path:
    return path.with_name(path.name + ".commit.json")


def _commit_json(path: Path, value: Any, *, commit_schema: str) -> dict[str, Any]:
    identities = _write_json(path, value)
    sidecar = _sidecar_path(path)
    sidecar_payload = f"{identities['file_bytes_sha256']}  {path.name}\n".encode("ascii")
    _write_exclusive(sidecar, sidecar_payload)
    commit = {
        "schema": commit_schema, "committed_utc": _now(), "document_name": path.name,
        **identities, "file_bytes_are_canonical_json_plus_lf": True,
        "sidecar_name": sidecar.name,
        "sidecar_file_bytes_sha256": _hash_bytes(sidecar_payload),
    }
    commit_identity = _write_json(_commit_path(path), commit)
    if (_hash_file(path) != identities["file_bytes_sha256"] or
            _hash_file(sidecar) != commit["sidecar_file_bytes_sha256"] or
            _hash_file(_commit_path(path)) != commit_identity["file_bytes_sha256"]):
        raise RuntimeError("committed JSON verification failed")
    return {**identities,
            "sidecar_file_bytes_sha256": commit["sidecar_file_bytes_sha256"],
            "commit_file_bytes_sha256": commit_identity["file_bytes_sha256"]}


def _commit_global_claim(value: Any, *, ownership: dict[str, bool]) -> dict[str, Any]:
    """Acquire the permanent global claim; never unlink after O_EXCL succeeds."""

    path = GLOBAL_CLAIM_PATH
    payload = _json_file_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    ownership["lease_acquired"] = True
    # If write/flush/fsync fails or the process dies, the residue is retained
    # permanently.  The same candidate is never authorized to retry it.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    identities = {
        "file_bytes_sha256": _hash_bytes(payload),
        "canonical_json_sha256": _hash_bytes(payload[:-1]),
        "file_size_bytes": len(payload),
        "canonical_json_size_bytes": len(payload) - 1,
    }
    sidecar = _sidecar_path(path)
    sidecar_payload = f"{identities['file_bytes_sha256']}  {path.name}\n".encode("ascii")
    _write_exclusive(sidecar, sidecar_payload)
    commit = {
        "schema": "gpu_minirocket_logodds_v8_global_claim_atomic_commit",
        "committed_utc": _now(), "document_name": path.name, **identities,
        "file_bytes_are_canonical_json_plus_lf": True,
        "sidecar_name": sidecar.name,
        "sidecar_file_bytes_sha256": _hash_bytes(sidecar_payload),
    }
    commit_identity = _write_json(_commit_path(path), commit)
    if (_hash_file(path) != identities["file_bytes_sha256"] or
            _hash_file(sidecar) != commit["sidecar_file_bytes_sha256"] or
            _hash_file(_commit_path(path)) != commit_identity["file_bytes_sha256"]):
        raise RuntimeError("global claim committed JSON verification failed")
    return {**identities,
            "sidecar_file_bytes_sha256": commit["sidecar_file_bytes_sha256"],
            "commit_file_bytes_sha256": commit_identity["file_bytes_sha256"]}


def _verify_committed_json(path: Path, *, commit_schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = _hash_contract_module()
    sidecar, commit_path = _sidecar_path(path), _commit_path(path)
    if not path.is_file() or not sidecar.is_file() or not commit_path.is_file():
        raise FileNotFoundError(f"incomplete committed JSON bundle for {path.name}")
    document, contract = helper.require_canonical_json_file(path)
    commit, commit_contract = helper.require_canonical_json_file(commit_path)
    expected_sidecar = f"{contract.file_bytes_sha256}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"sidecar mismatch for {path.name}")
    if not isinstance(document, dict) or not isinstance(commit, dict):
        raise ValueError("committed bundle JSON objects required")
    if (commit.get("schema") != commit_schema or commit.get("document_name") != path.name or
            commit.get("file_bytes_sha256") != contract.file_bytes_sha256 or
            commit.get("canonical_json_sha256") != contract.canonical_json_sha256 or
            commit.get("file_size_bytes") != contract.file_size_bytes or
            commit.get("canonical_json_size_bytes") != contract.canonical_json_size_bytes or
            commit.get("file_bytes_are_canonical_json_plus_lf") is not True or
            commit.get("sidecar_name") != sidecar.name or
            commit.get("sidecar_file_bytes_sha256") != _hash_file(sidecar)):
        raise ValueError(f"commit identity mismatch for {path.name}")
    return document, {
        "file_bytes_sha256": contract.file_bytes_sha256,
        "canonical_json_sha256": contract.canonical_json_sha256,
        "file_size_bytes": contract.file_size_bytes,
        "canonical_json_size_bytes": contract.canonical_json_size_bytes,
        "sidecar_file_bytes_sha256": _hash_file(sidecar),
        "commit_file_bytes_sha256": commit_contract.file_bytes_sha256,
    }


def _canonical_fold_map() -> dict[int, int]:
    return {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}


def _fold_map_hash(value: Mapping[int, int] | None = None) -> str:
    fold_map = _canonical_fold_map() if value is None else dict(value)
    return _hash_bytes(_canonical_bytes({str(run): int(fold_map[run]) for run in sorted(fold_map)}))


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)) or any(len(runs) != 20 for runs in FOLD_RUN_IDS.values()):
        raise ValueError("five folds of 20 runs are required")
    flat = [run for fold in range(5) for run in FOLD_RUN_IDS[fold]]
    if sorted(flat) != list(range(1, 101)) or len(set(flat)) != 100:
        raise ValueError("folds must partition runs 1..100")
    fold_map = _canonical_fold_map()
    if any(len({fold_map[run] for run in component}) != 1 for component in SEED_CONNECTED_COMPONENTS):
        raise ValueError("duplicate-seed component crosses folds")
    if _fold_map_hash() != FOLD_MAP_SHA256:
        raise ValueError("canonical fold-map SHA-256 changed")


def fold_roles(outer_fold: int) -> dict[str, Any]:
    if isinstance(outer_fold, bool) or outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    inner_fold = (outer_fold + 1) % 5
    fit = sorted(run for fold, runs in FOLD_RUN_IDS.items()
                 if fold not in (outer_fold, inner_fold) for run in runs)
    return {
        "outer_fold": outer_fold, "inner_fold": inner_fold,
        "fit_run_ids_per_class": fit,
        "inner_healthy_run_ids": list(FOLD_RUN_IDS[inner_fold]),
        "inner_fault_values_authorized": False,
        "outer_run_ids_per_class": list(FOLD_RUN_IDS[outer_fold]),
    }


def development_access_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for run_id in range(1, 101):
        for idv in CONDITION_IDVS:
            group = f"Mode2/SingleFault/SimulationCompleted/IDV{idv}/Mode2_IDVInfo_{idv}_50/Run{run_id}"
            window = [100, 600] if idv == 1 else [440, 940]
            plan.append({
                "mode": 2, "fault_regime": "SingleFault", "status": "SimulationCompleted",
                "magnitude_percent": 50, "idv": idv, "simulation_run": run_id,
                "group_path": group, "processdata_path": f"{group}/processdata",
                "idv_init_path": f"{group}/idv_init", "window_zero_based_half_open": window,
                "processdata_row_zero_based_half_open": window,
                "processdata_column_zero_based_half_open": [0, 53],
                "idv_init_row_zero_based_half_open": [0, 3],
                "idv_init_column_zero_based_half_open": [0, 28],
                "group_attributes_authorized": ["seed"],
                "dataset_metadata_authorized": ["object_type", "shape", "dtype"],
                "root_dataset_value_reads_authorized": [],
                "processdata_values_outside_row_window_authorized": False,
                "role": "healthy_control" if idv == 1 else f"fault_{idv}",
            })
    return plan


def _file_identity(path: Path) -> dict[str, Any]:
    return {"relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "file_bytes_sha256": _hash_file(path), "size_bytes": path.stat().st_size}


def _validate_activation() -> dict[str, Any]:
    expected = ((V7_RESULT_PATH, V7_RESULT_SHA256), (V7_DECISION_PATH, V7_DECISION_SHA256),
                (V7_COMPLETION_PATH, V7_COMPLETION_SHA256))
    for path, digest in expected:
        if not path.is_file() or _hash_file(path) != digest:
            raise ValueError(f"v8 activation evidence mismatch: {path.name}")
    if not V7_PROTOCOL_PATH.is_file():
        raise ValueError("v7 frozen protocol is missing")
    v7_protocol = _strict_json(V7_PROTOCOL_PATH)
    if _hash_bytes(_canonical_bytes(v7_protocol)) != V7_PROTOCOL_CANONICAL_JSON_SHA256:
        raise ValueError("v7 frozen protocol canonical identity changed")
    implementation = v7_protocol.get("bindings", {}).get("implementation", {})
    expected_charters = {
        "docs/gpu_minirocket_logodds_v8_contingency.json": {
            "file_bytes_sha256": CHARTER_JSON_V7_BOUND_SHA256,
            "size_bytes": CHARTER_JSON_V7_BOUND_SIZE,
        },
        "docs/gpu_minirocket_logodds_v8_contingency.md": {
            "file_bytes_sha256": CHARTER_MD_V7_BOUND_SHA256,
            "size_bytes": CHARTER_MD_V7_BOUND_SIZE,
        },
    }
    if (not isinstance(implementation, Mapping) or
            any(implementation.get(path) != identity
                for path, identity in expected_charters.items()) or
            _hash_file(CHARTER_JSON_PATH) != CHARTER_JSON_V7_BOUND_SHA256 or
            CHARTER_JSON_PATH.stat().st_size != CHARTER_JSON_V7_BOUND_SIZE or
            _hash_file(CHARTER_MD_PATH) != CHARTER_MD_V7_BOUND_SHA256 or
            CHARTER_MD_PATH.stat().st_size != CHARTER_MD_V7_BOUND_SIZE):
        raise ValueError("v7 protocol no longer proves the pre-result v8 charter bytes")
    result, decision, completion = (_strict_json(V7_RESULT_PATH), _strict_json(V7_DECISION_PATH),
                                    _strict_json(V7_COMPLETION_PATH))
    if (result.get("schema") != "farm_rankcusum_v7_development_result" or
            result.get("candidate_id") != "farm-rankcusum-v7-fixed" or
            result.get("protocol_canonical_json_sha256") != V7_PROTOCOL_CANONICAL_JSON_SHA256 or
            result.get("decision") != "DEVELOPMENT_NO_GO" or
            result.get("selected_candidate_id") is not None or result.get("final_refit") is not None or
            result.get("development_decision_before_final_refit", {}).get("file_bytes_sha256") != V7_DECISION_SHA256):
        raise ValueError("v7 result is not the exact activating DEVELOPMENT_NO_GO")
    if (decision.get("schema") != "farm_rankcusum_v7_development_decision_before_final_refit" or
            decision.get("candidate_id") != "farm-rankcusum-v7-fixed" or
            decision.get("protocol_canonical_json_sha256") != V7_PROTOCOL_CANONICAL_JSON_SHA256 or
            decision.get("decision") != "DEVELOPMENT_NO_GO" or
            decision.get("development_decision_determined") is not True or
            decision.get("final_refit_permitted") is not False or
            decision.get("final_refit_started") is not False or
            decision.get("final_model_usable") is not False):
        raise ValueError("v7 append-only decision evidence is not an activating NO_GO")
    if (completion.get("schema") != "farm_rankcusum_v7_development_completion" or
            completion.get("decision") != "DEVELOPMENT_NO_GO" or
            completion.get("protocol_canonical_json_sha256") != V7_PROTOCOL_CANONICAL_JSON_SHA256 or
            completion.get("development_result_file_bytes_sha256") != V7_RESULT_SHA256 or
            completion.get("mode1_accessed") is not False):
        raise ValueError("v7 completion does not bind the activating result")
    return {
        "trigger": {"candidate_id": "farm-rankcusum-v7-fixed", "result": "DEVELOPMENT_NO_GO"},
        "protocol_canonical_json_sha256": V7_PROTOCOL_CANONICAL_JSON_SHA256,
        "v7_frozen_protocol": _file_identity(V7_PROTOCOL_PATH),
        "v7_bound_contingent_charters": expected_charters,
        "development_result": _file_identity(V7_RESULT_PATH),
        "decision_before_refit": _file_identity(V7_DECISION_PATH),
        "completion": _file_identity(V7_COMPLETION_PATH),
        "numeric_metrics_used_to_modify_v8": False,
    }


def _validate_charter() -> dict[str, Any]:
    charter = _strict_json(CHARTER_JSON_PATH)
    activation, authorization, upstream = (charter.get("activation", {}),
                                            charter.get("authorization", {}), charter.get("upstream", {}))
    folds = charter.get("fold_assignment", {})
    input_config = charter.get("input", {})
    normalization = input_config.get("normalization", {})
    mini = charter.get("minirocket", {})
    classifier = charter.get("classifier", {})
    optimizer = classifier.get("optimizer", {})
    loss = classifier.get("loss", {})
    monitor = charter.get("monitor", {})
    calibration = monitor.get("calibration", {})
    cusum = monitor.get("cusum", {})
    intermittent = monitor.get("intermittent", {})
    numeric = charter.get("numeric_validity", {})
    decision = charter.get("development_decision", {})
    determinism = charter.get("determinism", {})
    training = charter.get("training_examples_per_fold", {})
    feature_standardization = charter.get("feature_standardization", {})
    post_go = charter.get("post_go", {})
    data_scope = charter.get("data_scope", {})
    golden = charter.get("upstream_golden_handshake", {})
    zero_rescue = charter.get("zero_rescue_policy", {})
    if (charter.get("candidate_id") != CANDIDATE_ID or
            activation.get("allowed_trigger") != {"candidate_id": "farm-rankcusum-v7-fixed", "result": "DEVELOPMENT_NO_GO"} or
            activation.get("requires_separate_pre_access_protocol") is not True or
            authorization.get("document_type") != "contingent_method_charter" or
            any(authorization.get(key) is not False for key in (
                "allows_checkpoint_creation", "allows_code_implementation", "allows_h5_access",
                "allows_numeric_claims", "allows_real_experiment")) or
            upstream.get("tag") != "v1.0.1" or
            upstream.get("repository") != "https://github.com/timeseriesAI/tsai" or
            upstream.get("source_path") != "tsai/models/MINIROCKETPlus_Pytorch.py" or
            upstream.get("license") != "Apache-2.0" or
            data_scope.get("authorized_conditions_for_development") != ["H", "F3", "F9", "F15"] or
            data_scope.get("independent_split_unit") != "run" or
            data_scope.get("forbidden_sources") != [
                "Mode1", "Rieth", "sealed_calibration", "sealed_final_confirmation",
                "all_other_sources"] or
            folds.get("run_to_fold_sha256") != FOLD_MAP_SHA256 or
            folds.get("fold_run_ids") != {str(f): list(r) for f, r in FOLD_RUN_IDS.items()} or
            folds.get("run_to_fold") != {str(r): f for r, f in sorted(_canonical_fold_map().items())} or
            folds.get("role_counts_per_class") != {"fit": 60, "inner_healthy_only": 20, "outer": 20} or
            folds.get("inner_fault_values_authorized") is not False or
            input_config.get("channel_count") != 52 or input_config.get("window_length") != 128 or
            input_config.get("window_shape") != [52, 128] or
            input_config.get("channel_order") != "stored_order_0_through_51" or
            input_config.get("network_dtype") != "float32" or
            input_config.get("window_definition") != "x[t-127:t+1] inclusive causal" or
            normalization.get("clip") != [-10.0, 10.0] or
            normalization.get("fit_values") != "60 fit-healthy runs times all 500 rows" or
            normalization.get("location") != "per_channel_float64_median" or
            normalization.get("scale") != "max(1.4826*per_channel_MAD,1e-6)" or
            (mini.get("kernel_count"), mini.get("kernel_length"), mini.get("requested_feature_count"),
             mini.get("effective_feature_count"), mini.get("max_dilations_per_kernel"),
             mini.get("dilation_integer_range_for_length_128"), mini.get("max_num_channels"),
             mini.get("add_lsaz"), mini.get("bias_fit_chunk_size"), mini.get("extraction_batch_size"),
             mini.get("extraction_drop_last"), mini.get("ppv_comparator")) !=
            (84, 9, 10000, 9996, 32, [1, 15], 9, False, 1260, 128, False, "strict_greater_than") or
            mini.get("kernel_definition") != "all C(9,3) kernels with three +2 and six -1" or
            mini.get("bias_fit_pool") != "all 20400 ordered fit-fold training windows only" or
            mini.get("bias_quantiles") != "upstream_golden_ratio_low_discrepancy_sequence" or
            mini.get("upstream_padded_unpadded_allocation_preserved") is not True or
            (classifier.get("input_features"), classifier.get("output_classes"),
             classifier.get("parameter_count"), classifier.get("initialization"),
             classifier.get("class_order"), classifier.get("class_weights")) !=
            (9996, 4, 39988, "all_weights_and_biases_zero", ["H", "F3", "F9", "F15"],
             {"H": "85/109", "F3": "85/77", "F9": "85/77", "F15": "85/77"}) or
            loss.get("l2_coefficient_expression") != "(1e-4/2)*sum(W^2)" or
            loss.get("bias_penalized") is not False or
            loss.get("cross_entropy_reduction") != "pytorch_weighted_mean" or
            (optimizer.get("lr"), optimizer.get("max_iter"), optimizer.get("max_eval"),
             optimizer.get("history_size"), optimizer.get("tolerance_grad"),
             optimizer.get("tolerance_change"), optimizer.get("line_search_fn"),
              optimizer.get("full_batch")) != (1.0, 100, 125, 20, 1e-7, 1e-9, "strong_wolfe", True) or
            classifier.get("training_acceptance") != {
                "all_gradients_finite": True, "all_parameters_finite": True,
                "final_objective_le_initial_tolerance": 1e-10, "losses_finite": True} or
            feature_standardization != {
                "fit_examples": 20400, "input_dtype": "float32", "output_dtype": "float32",
                "population_std": True, "statistics_dtype": "float64", "std_floor": 1e-6} or
            monitor.get("prefix_endpoints") != [128, 132, 136, 140, 144, 148, 152, 156] or
            monitor.get("dense_monitor_endpoints") != {"start": 160, "step": 1, "stop_inclusive": 499} or
            calibration.get("pool_size") != 20400 or calibration.get("scale") != "max(1.4826*median(abs(v-center)),1e-3)" or
            calibration.get("center") != "median of pooled fit-healthy run-relative v values" or
            monitor.get("log_odds") != "logsumexp(l_F3,l_F9,l_F15)-l_H" or
            monitor.get("prefix_center") != "per-run median over eight prefix q values" or
            monitor.get("sequential_dtype") != "cpu_float64" or
            monitor.get("earliest_alarm_endpoint") != 160 or
            cusum.get("drift") != 0.25 or intermittent.get("first_endpoint") != 175 or
            intermittent.get("window_inclusive") != 16 or intermittent.get("top_r") != 4 or
            intermittent.get("coefficient") != 0.5 or
            numeric.get("formal_path") != "deterministic_cuda" or
            numeric.get("cpu_role") != "reference_report_only" or
            numeric.get("nonfinite_policy") != "fold_invalid_and_overall_DEVELOPMENT_NO_GO" or
            numeric.get("replay_scope") != "complete inner CUDA score through calibration and scan" or
            numeric.get("same_cuda_replays") != 2 or
            numeric.get("inner_distinct_maxima_min") != 18 or
            numeric.get("inner_rank_gap_rule") != "rank20-rank19 > 100*max(full_same_cuda_replay_error,1e-6)" or
            determinism.get("seed") != 20260912 or
            determinism.get("seed_targets") != ["python", "numpy", "torch_cpu", "torch_all_cuda"] or
            determinism.get("torch_deterministic_algorithms") is not True or
            determinism.get("cudnn_deterministic") is not True or
            determinism.get("cudnn_benchmark") is not False or
            determinism.get("same_cuda_max_abs_error") != 1e-6 or
            determinism.get("cublas_workspace_config") != ":4096:8" or
            (decision.get("healthy_alarms_overall_max"), decision.get("healthy_alarms_per_fold_max"),
             decision.get("fault_detection_rate_each_overall_min"),
             decision.get("fault_detection_rate_each_per_fold_min"),
             decision.get("median_delay_max"), decision.get("shared_fault_delay_p90_max"),
              decision.get("miss_delay"), decision.get("shared_fault_delay_pool_size")) !=
            (10, 4, 0.85, 0.7, 140, 220, 340, 300) or
            decision.get("all_gates_required") is not True or
            decision.get("numeric_folds_valid_required") != 5 or
            decision.get("result_if_any_gate_fails") != "DEVELOPMENT_NO_GO" or
            decision.get("result_if_all_gates_pass") != "GO" or
            training.get("total") != 20400 or training.get("H", {}).get("total_count") != 6540 or
            training.get("F3", {}).get("count") != 4620 or training.get("F9", {}).get("count") != 4620 or
            training.get("F15", {}).get("count") != 4620 or
            training.get("stable_order") != [
                "class_order_H_F3_F9_F15", "source_or_run_id_ascending", "endpoint_ascending"] or
            training.get("H", {}).get("healthy_count") != 5100 or
            training.get("H", {}).get("fault_prefix_count") != 1440 or
            training.get("H", {}).get("healthy_endpoints") != "160..496 step 4" or
            training.get("H", {}).get("fault_prefix_endpoints") != [128, 132, 136, 140, 144, 148, 152, 156] or
            any(training.get(label, {}).get("runs") != 60 or
                training.get(label, {}).get("endpoints") != "192..496 step 4"
                for label in ("F3", "F9", "F15")) or
            training.get("fault_transition_endpoints_ignored") != [160, 164, 168, 172, 176, 180, 184, 188] or
            post_go != {"authorized_by_this_charter": False, "refit_runs_per_class": 100,
                        "requires_all_five_development_folds_go": True,
                        "same_method_and_hyperparameters": True} or
            golden != {"before_any_h5_access": True, "cpu_float_atol_rtol": 1e-7,
                       "cuda_float_atol_rtol": 1e-6, "cuda_repeat_max_abs_error": 1e-6,
                       "golden_format": "npz_allow_pickle_false",
                       "integer_and_boolean_buffers_exact": True,
                       "required_scalar_cases": ["padding", "unpadding", "quantiles",
                                                 "strict_greater_than_ties"],
                       "source_of_truth": "unmodified official tagged upstream class on fixed synthetic tensor"} or
            zero_rescue != {"candidate_count": 1, "hyperparameter_grid_allowed": False,
                            "outcome_after_v8_development_no_go":
                                "terminate_current_development_pool_method_search",
                            "post_result_fault_specific_changes_allowed": False}):
        raise ValueError("v8 contingent charter semantic mismatch")
    return charter


def _validate_ledger() -> dict[str, Any]:
    ledger = _strict_json(LEDGER_PATH)
    activation = ledger.get("activation_evidence", {})
    upstream = ledger.get("upstream_identity", {})
    mode1 = ledger.get("extended_tep_mode1_roles", {})
    if (ledger.get("schema") != "gpu_minirocket_logodds_v8_history_and_access_ledger" or
            ledger.get("ledger_state") != "PREACCESS_IMPLEMENTATION_FOR_V8_MODE2_EXPLORATORY_DEVELOPMENT" or
            ledger.get("audit_scope", {}).get("hdf5_files_opened_by_this_audit") != 0 or
            activation.get("v7_development_result_file_bytes_sha256") != V7_RESULT_SHA256 or
            activation.get("v7_decision_evidence_file_bytes_sha256") != V7_DECISION_SHA256 or
            upstream.get("annotated_tag_object_sha") != UPSTREAM_ANNOTATED_TAG_OBJECT_SHA or
            upstream.get("peeled_commit_sha") != UPSTREAM_PEELED_COMMIT_SHA or
            mode1.get("magnitude25_idv1_runs1_100", {}).get("state") != "SEALED" or
            mode1.get("magnitude75_idv1_3_9_15_runs1_100", {}).get("state") != "SEALED" or
            ledger.get("v8_development_authorization", {}).get("mode1_access_authorized") is not False):
        raise ValueError("v8 history/access ledger semantic mismatch")
    return ledger


def _test_paths() -> tuple[str, ...]:
    paths = {"tests/test_hash_contract.py"}
    paths.update(path.relative_to(PROJECT_ROOT).as_posix()
                 for path in (PROJECT_ROOT / "tests").glob("test_*v8*.py"))
    return tuple(sorted(paths))


def _implementation_paths() -> tuple[str, ...]:
    paths = set(BASE_IMPLEMENTATION_PATHS) | set(_test_paths())
    shared_package = PROJECT_ROOT / "src" / "mate_spc"
    if shared_package.is_dir():
        paths.update(path.relative_to(PROJECT_ROOT).as_posix()
                     for path in shared_package.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    package = PROJECT_ROOT / "src" / "mate_spc_minirocket_v8"
    if package.is_dir():
        paths.update(path.relative_to(PROJECT_ROOT).as_posix() for path in package.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    vendor = PROJECT_ROOT / "vendor" / "tsai_v1_0_1_minirocket"
    if vendor.is_dir():
        paths.update(path.relative_to(PROJECT_ROOT).as_posix() for path in vendor.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    vendor_init = PROJECT_ROOT / "vendor" / "__init__.py"
    if vendor_init.is_file():
        paths.add("vendor/__init__.py")
    return tuple(sorted(paths))


def _implementation_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in _implementation_paths():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required v8 artifact missing: {relative}")
        records[relative] = {"file_bytes_sha256": _hash_file(path), "size_bytes": path.stat().st_size}
    required = {"src/mate_spc/__init__.py", "src/mate_spc/hash_contract.py",
                "src/mate_spc_minirocket_v8/backend.py", "src/mate_spc_minirocket_v8/model.py",
                "src/mate_spc_minirocket_v8/monitor.py", "src/mate_spc_minirocket_v8/evaluation.py",
                "vendor/tsai_v1_0_1_minirocket/minirocket_plus.py",
                "vendor/tsai_v1_0_1_minirocket/LICENSE",
                "vendor/tsai_v1_0_1_minirocket/PROVENANCE.json",
                "vendor/tsai_v1_0_1_minirocket/golden_handshake.py",
                "vendor/tsai_v1_0_1_minirocket/golden/manifest.json",
                "vendor/tsai_v1_0_1_minirocket/golden/upstream_v1_0_1_synthetic.npz"}
    if not required.issubset(records):
        raise FileNotFoundError("v8 production package is incomplete")
    return records


def _protocol_document(implementation: Mapping[str, Any]) -> dict[str, Any]:
    _validate_folds()
    activation = _validate_activation()
    _validate_charter()
    _validate_ledger()
    if _hash_bytes(_canonical_bytes(development_access_plan())) != ACCESS_PLAN_SHA256:
        raise ValueError("exact-selector access-plan SHA-256 changed")
    return {
        "schema": PROTOCOL_SCHEMA, "schema_version": 1, "protocol_id": PROTOCOL_ID,
        "protocol_state": "FROZEN_AFTER_IMPLEMENTATION_BEFORE_V8_HDF5_OPEN",
        "created_utc": _now(), "candidate": METHOD_CONFIG, "candidate_count": 1,
        "environment_lock": ENVIRONMENT_LOCK,
        "activation_evidence": activation,
        "source": {
            "relative_path": "data/extended_tep_dtu_mode2/TEP_Mode2.h5",
            "filename": "TEP_Mode2.h5", "mode": 2, "fault_regime": "SingleFault",
            "magnitude_percent": 50, "idvs": [1, 3, 9, 15],
            "sha256": SOURCE_SHA256, "size_bytes": SOURCE_SIZE_BYTES,
            "hdf5_opened_during_freeze": False, "mode1_access_authorized": False,
            "rieth_access_authorized": False,
        },
        "folds": {
            "fold_run_ids": {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()},
            "run_to_fold": {str(run): fold for run, fold in sorted(_canonical_fold_map().items())},
            "run_to_fold_canonical_json_sha256": FOLD_MAP_SHA256,
            "cross_fold_seed_collision_count": 0,
            "outer_roles": [fold_roles(fold) for fold in range(5)],
        },
        "access_plan": {
            "record_count": 400, "selector_count": 400,
            "canonical_json_sha256": ACCESS_PLAN_SHA256,
            "exact_selector_fields": list(development_access_plan()[0]),
            "wildcards_allowed": False,
        },
        "development_go_gates": GO_GATES,
        "governance": {
            "candidate_count": 1, "global_development_claim_count": 1,
            "same_candidate_rerun_after_claim": False, "rescue_iteration_ids_allowed": False,
            "preclaim_production_backend_handshake_required": True,
            "preclaim_h5py_import_blocker_required": True,
            "handshaken_backend_instance_reused_after_intent": True,
            "claim_precedes_access_intent": True,
            "access_intent_precedes_hdf5_import_or_open": True,
            "activation_evidence_revalidated_before_claim": True,
            "decision_persisted_before_go_refit": True,
            "mode1_access_authorized": False, "rieth_access_authorized": False,
            "inner_fault_numeric_access_authorized": False,
            "development_no_go_terminates_current_pool_search": True,
        },
        "hash_contract": {
            "canonicalization": "UTF-8 sorted-key compact JSON, no NaN, no terminal LF",
            "persisted_file": "canonical JSON plus exactly one terminal LF",
            "sidecar_and_commit_identity": "file_bytes_sha256",
            "runner_backend_identity": "canonical_json_sha256",
            "cross_domain_comparison_forbidden": True,
        },
        "bindings": {
            "method_document": _file_identity(DOCUMENT_PATH),
            "contingent_charter_markdown": _file_identity(CHARTER_MD_PATH),
            "contingent_charter_json": _file_identity(CHARTER_JSON_PATH),
            "history_and_access_ledger": _file_identity(LEDGER_PATH),
            "implementation": _plain_json(implementation),
            "implementation_tree_canonical_json_sha256":
                _hash_bytes(_canonical_bytes(implementation)),
        },
        "execution": {
            "working_directory": ".",
            "environment_name": "torch5090",
            "output_directory": "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal",
            "completion_path":
                "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/completion.json",
            "numeric_execution_claim_path":
                "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/numeric_execution.claim.json",
            "command": [
                "conda", "run", "--no-capture-output", "-n", "torch5090",
                "python", "-I", "-S", "-B", "-u", "scripts/run_gpu_minirocket_logodds_v8.py",
                "development",
            ],
            "direct_environment_python_invocation_authorized": False,
            "observable_invocation_contract": {
                "sys_executable": str(FORMAL_PYTHON_EXE),
                "sys_prefix": str(FORMAL_ENV_PREFIX),
                "isolated": 1, "no_site": 1, "dont_write_bytecode": 1,
                "working_directory": ".",
                "script_path": "scripts/run_gpu_minirocket_logodds_v8.py",
                "argv_tail": ["development"],
                "stdout_write_through": True, "stderr_write_through": True,
                "conda_prefix": str(FORMAL_ENV_PREFIX),
                "conda_default_env": "torch5090",
            },
            "failure_behavior": {
                "before_claim":
                    "abort_without_hdf5_access; no claim is consumed",
                "partial_or_complete_claim_before_artifact_directory":
                    "persist global claim failure evidence; fail closed; no rerun",
                "after_claim_before_numeric_access":
                    "persist failure and completion evidence; fail closed; no rerun",
                "during_or_after_numeric_access":
                    "persist conservative access state plus failure and completion; no rerun",
                "partial_or_complete_numeric_execution_claim":
                    "permanent execution claim residue; fail closed; no replay",
                "during_go_only_final_refit":
                    "retain decision-before-refit evidence; persist failure and completion; no rerun",
                "same_candidate_rerun_after_any_claim_residue": False,
            },
        },
        "numeric_backend_contract": {
            "module": BACKEND_MODULE,
            "required_functions": list(BACKEND_REQUIRED_FUNCTIONS),
            "backend_import_and_validation_before_claim": True,
            "same_backend_module_object_used_after_intent": True,
            "validation_hash_domain": "canonical_json_without_file_terminator",
            "validation_must_not_compare_file_bytes_sha256": True,
            "hdf5_import_or_open_before_intent": False,
            "golden_handshake_before_claim": True,
            "result_schema": BACKEND_RESULT_SCHEMA,
            "preclaim_test_count": EXPECTED_PRECLAIM_TEST_COUNT,
        },
    }


def freeze_development() -> dict[str, Any]:
    implementation = _implementation_records()
    bundle = (PROTOCOL_PATH, _sidecar_path(PROTOCOL_PATH), _commit_path(PROTOCOL_PATH))
    if any(path.exists() for path in bundle):
        if not all(path.is_file() for path in bundle):
            raise FileExistsError("partial v8 protocol bundle exists; refuse repair or overwrite")
        document, identities = _verify_protocol_bundle()
        return {"status": "VERIFIED_EXISTING_FROZEN_PROTOCOL",
                "protocol_id": document["protocol_id"], **identities}
    if GLOBAL_CLAIM_PATH.exists():
        raise RuntimeError("v8 global development claim exists; refuse late freeze")
    identities = _commit_json(
        PROTOCOL_PATH, _protocol_document(implementation),
        commit_schema="gpu_minirocket_logodds_v8_protocol_atomic_commit",
    )
    return {"status": "FROZEN", "protocol_id": PROTOCOL_ID, **identities}


def _verify_protocol_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    document, identities = _verify_committed_json(
        PROTOCOL_PATH, commit_schema="gpu_minirocket_logodds_v8_protocol_atomic_commit")
    if (document.get("schema") != PROTOCOL_SCHEMA or document.get("protocol_id") != PROTOCOL_ID or
            document.get("candidate") != METHOD_CONFIG or document.get("candidate_count") != 1 or
            document.get("environment_lock") != ENVIRONMENT_LOCK or
            document.get("development_go_gates") != GO_GATES or
            document.get("source", {}).get("sha256") != SOURCE_SHA256 or
            document.get("source", {}).get("size_bytes") != SOURCE_SIZE_BYTES or
            document.get("source", {}).get("mode1_access_authorized") is not False or
            document.get("access_plan", {}).get("record_count") != 400 or
            document.get("access_plan", {}).get("canonical_json_sha256") != ACCESS_PLAN_SHA256 or
            document.get("hash_contract", {}).get("runner_backend_identity") != "canonical_json_sha256"):
        raise ValueError("v8 protocol semantic binding failed")
    _validate_folds()
    folds = document.get("folds", {})
    if (folds.get("fold_run_ids") != {str(f): list(r) for f, r in FOLD_RUN_IDS.items()} or
            folds.get("run_to_fold") != {str(r): f for r, f in sorted(_canonical_fold_map().items())} or
            folds.get("run_to_fold_canonical_json_sha256") != FOLD_MAP_SHA256 or
            folds.get("cross_fold_seed_collision_count") != 0):
        raise ValueError("v8 protocol fold binding failed")
    activation = _validate_activation()
    if document.get("activation_evidence") != activation:
        raise ValueError("v8 frozen activation evidence changed")
    _validate_charter()
    _validate_ledger()
    bindings = document.get("bindings", {})
    implementation = _implementation_records()
    if (bindings.get("method_document") != _file_identity(DOCUMENT_PATH) or
            bindings.get("contingent_charter_markdown") != _file_identity(CHARTER_MD_PATH) or
            bindings.get("contingent_charter_json") != _file_identity(CHARTER_JSON_PATH) or
            bindings.get("history_and_access_ledger") != _file_identity(LEDGER_PATH) or
            bindings.get("implementation") != implementation or
            bindings.get("implementation_tree_canonical_json_sha256") !=
                _hash_bytes(_canonical_bytes(implementation))):
        raise ValueError("v8 frozen implementation binding failed")
    return document, identities


def _source_record() -> dict[str, Any]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    if SOURCE_PATH.name != "TEP_Mode2.h5" or SOURCE_PATH.stat().st_size != SOURCE_SIZE_BYTES:
        raise ValueError("Mode2 source path, name, or size changed")
    observed = _hash_file(SOURCE_PATH)
    if observed != SOURCE_SHA256:
        raise ValueError("Mode2 source SHA-256 mismatch")
    return {
        "relative_path": "data/extended_tep_dtu_mode2/TEP_Mode2.h5",
        "filename": SOURCE_PATH.name, "size_bytes": SOURCE_SIZE_BYTES,
        "sha256": observed, "hash_method": "raw streaming SHA256 in 16MiB chunks",
        "hdf5_library_imported": False, "hdf5_opened": False,
        "numeric_values_decoded": False,
    }


def _gpu_presence_preflight() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                               timeout=30, check=False)
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    expected_row = (
        f"{ENVIRONMENT_LOCK['gpu_name']}, "
        f"{ENVIRONMENT_LOCK['nvidia_smi_memory_total_mib_rounded']}, "
        f"{ENVIRONMENT_LOCK['nvidia_driver_version']}, 12.0")
    if completed.returncode != 0 or rows != [expected_row]:
        raise RuntimeError("exact RTX 5090/driver environment-lock preflight failed")
    return {"command": command, "returncode": completed.returncode,
            "rows": rows, "rtx5090_present": True}


def _run_test_suite() -> dict[str, Any]:
    test_paths = _test_paths()
    if len(test_paths) < 3:
        raise RuntimeError("v8 test suite was not fully discovered")
    # On the pinned Windows CUDA environment NumPy/pytest imported before
    # torch can make torch DLL loading fail with WinError 127.  Keep this
    # bootstrap literal and auditable: torch must be the first heavy import.
    absolute_tests = [str((PROJECT_ROOT / path).resolve()) for path in test_paths]
    environment = dict(os.environ)
    observed_cublas = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if observed_cublas not in (None, ":4096:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from frozen :4096:8")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["MATE_SPC_V8_FORBID_REAL_H5"] = "1"
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTEST_ADDOPTS",
                 "PYTEST_PLUGINS"):
        environment.pop(name, None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="mate_spc_v8_pycache_") as pycache:
        environment["PYTHONPYCACHEPREFIX"] = pycache
        # -I deliberately ignores PYTHONPYCACHEPREFIX.  Assign the fresh
        # absolute cache prefix before *any* third-party or test import so an
        # existing project __pycache__ can never supply executable bytes.
        bootstrap = (
            "import sys;"
            f"sys.pycache_prefix={str(Path(pycache).resolve())!r};"
            "sys.dont_write_bytecode=True;"
            f"assert sys.pycache_prefix=={str(Path(pycache).resolve())!r};"
            f"sys.path[:0]=[{str(ENVIRONMENT_SITE_PACKAGES)!r},{str(SRC_ROOT)!r},"
            f"{str(PROJECT_ROOT)!r}];"
            "import torch,pytest;"
            "sys.exit(pytest.main(sys.argv[1:]))")
        command = [
            sys.executable, "-I", "-S", "-c", bootstrap, *absolute_tests,
            "-q", "-W", "error", "--noconftest", "--strict-config",
            "-c", str((PROJECT_ROOT / "pyproject.toml").resolve()),
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                   timeout=1800, check=False, env=environment)
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"v8 synthetic/CUDA/governance tests failed:\n{combined[-4000:]}")
    summaries = re.findall(r"(?m)^\s*(\d+) passed in [0-9.]+s\s*$", combined)
    if summaries != [str(EXPECTED_PRECLAIM_TEST_COUNT)] or any(
            word in combined.lower() for word in
            (" skipped", " deselected", " xfailed", " xpassed", " failed", " error")):
        raise RuntimeError("v8 test suite count/status differs from the frozen no-omission suite")
    return {"command": command, "returncode": completed.returncode,
            "output_sha256": _hash_bytes(combined.encode("utf-8")),
            "output_tail": combined[-2000:], "passed_count": int(summaries[0]),
            "isolated_python": True, "site_initialization_disabled": True,
            "fresh_pycache_prefix_assigned_before_imports": True,
            "pytest_plugin_autoload_disabled": True, "conftest_disabled": True,
            "inherited_python_and_pytest_variables_cleared": True,
            "hdf5_numeric_access_authorized": False}


def _prepare_backend_environment() -> None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif observed != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from frozen :4096:8")


def _attest_exact_source_module(module: Any, *, module_name: str,
                                expected_path: Path,
                                implementation: Mapping[str, Any],
                                required_callables: Sequence[str] = ()) -> ModuleType:
    expected = expected_path.resolve()
    relative = expected.relative_to(PROJECT_ROOT).as_posix()
    record = implementation.get(relative)
    if (not isinstance(module, ModuleType) or module.__name__ != module_name or
            sys.modules.get(module_name) is not module):
        raise RuntimeError(f"untrusted or detached module object: {module_name}")
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    try:
        loaded_path = Path(loader.get_filename(module_name)).resolve()
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        raise RuntimeError(f"module has no trusted source loader: {module_name}") from error
    if (not isinstance(loader, SourceFileLoader) or spec.name != module_name or
            Path(str(spec.origin)).resolve() != expected or
            Path(str(getattr(module, "__file__", ""))).resolve() != expected or
            getattr(module, "__loader__", None) is not loader or loaded_path != expected):
        raise RuntimeError(f"module source/spec mismatch: {module_name}")
    if (not isinstance(record, Mapping) or not expected.is_file() or
            record.get("size_bytes") != expected.stat().st_size or
            record.get("file_bytes_sha256") != _hash_file(expected)):
        raise RuntimeError(f"module differs from frozen implementation: {relative}")
    for name in required_callables:
        function = getattr(module, name, None)
        source = inspect.getsourcefile(function) if callable(function) else None
        code = getattr(function, "__code__", None)
        if (not callable(function) or getattr(function, "__module__", None) != module_name or
                source is None or Path(source).resolve() != expected or code is None or
                Path(code.co_filename).resolve() != expected):
            raise RuntimeError(f"untrusted callable origin: {module_name}.{name}")
    return module


def _attest_backend_module(module: Any, implementation: Mapping[str, Any]) -> ModuleType:
    package_name = "mate_spc_minirocket_v8"
    package = sys.modules.get(package_name)
    _attest_exact_source_module(
        package, module_name=package_name,
        expected_path=SRC_ROOT / package_name / "__init__.py",
        implementation=implementation)
    backend = _attest_exact_source_module(
        module, module_name=BACKEND_MODULE,
        expected_path=SRC_ROOT / package_name / "backend.py",
        implementation=implementation, required_callables=BACKEND_REQUIRED_FUNCTIONS)
    if getattr(package, "backend", None) is not backend:
        raise RuntimeError("package backend attribute is not the attested module")
    return backend


def _load_frozen_source_bytes(module_name: str, path: Path,
                              implementation: Mapping[str, Any], *,
                              is_package: bool) -> ModuleType:
    """Hash and execute one immutable in-memory source byte string."""

    expected = path.resolve()
    relative = expected.relative_to(PROJECT_ROOT).as_posix()
    record = implementation.get(relative)
    payload = expected.read_bytes()
    if (not isinstance(record, Mapping) or record.get("size_bytes") != len(payload) or
            record.get("file_bytes_sha256") != _hash_bytes(payload)):
        raise RuntimeError(f"source bytes differ from frozen implementation: {relative}")
    locations = [str(expected.parent)] if is_package else None
    spec = importlib.util.spec_from_file_location(
        module_name, expected, submodule_search_locations=locations)
    if spec is None or not isinstance(spec.loader, SourceFileLoader):
        raise RuntimeError(f"cannot construct trusted source spec: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        code = compile(payload, str(expected), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_backend(implementation: Mapping[str, Any]) -> ModuleType:
    if not isinstance(implementation, Mapping):
        raise ValueError("frozen implementation mapping required")
    preloaded = sorted(name for name in sys.modules
                       if name == "mate_spc_minirocket_v8" or
                       name.startswith("mate_spc_minirocket_v8.") or
                       name == "vendor" or name.startswith("vendor."))
    if preloaded:
        raise RuntimeError(f"v8 production backend graph was preloaded: {preloaded}")
    package_name = "mate_spc_minirocket_v8"
    package = _load_frozen_source_bytes(
        package_name, SRC_ROOT / package_name / "__init__.py", implementation,
        is_package=True)
    try:
        module = _load_frozen_source_bytes(
            BACKEND_MODULE, SRC_ROOT / package_name / "backend.py", implementation,
            is_package=False)
    except BaseException:
        sys.modules.pop(package_name, None)
        raise
    setattr(package, "backend", module)
    return _attest_backend_module(module, implementation)


def _validate_backend_cuda(value: Any) -> dict[str, Any]:
    report = _plain_json(value)
    if (not isinstance(report, dict) or report.get("passed") is not True or
            report.get("cuda_available") is not True or
            report.get("golden_handshake_passed") is not True or
            report.get("device_name") != ENVIRONMENT_LOCK["gpu_name"] or
            report.get("device_capability") != ENVIRONMENT_LOCK["gpu_compute_capability"] or
            report.get("device_total_memory") != ENVIRONMENT_LOCK["gpu_total_memory_bytes"] or
            report.get("python_version") != ENVIRONMENT_LOCK["python_version"] or
            report.get("numpy_version") != ENVIRONMENT_LOCK["numpy_version"] or
            report.get("h5py_distribution_version") !=
                ENVIRONMENT_LOCK["h5py_distribution_version"] or
            report.get("torch_version") != ENVIRONMENT_LOCK["torch_version"] or
            report.get("torch_cuda_version") != ENVIRONMENT_LOCK["torch_cuda_build"] or
            report.get("cudnn_version") != ENVIRONMENT_LOCK["cudnn_version"]):
        raise RuntimeError("backend CUDA/golden preflight did not pass")
    return report


def _loaded_h5py_modules() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name == "h5py" or name.startswith("h5py.")))


class _H5pyImportBlocker:
    def __init__(self) -> None:
        self.attempted = False

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        del path, target
        if fullname == "h5py" or fullname.startswith("h5py."):
            self.attempted = True
            raise RuntimeError("h5py import attempted during pre-claim handshake")
        return None


def _production_handshake(protocol: Mapping[str, Any], *,
                          canonical_json_sha256: str) -> tuple[ModuleType, dict[str, Any]]:
    if _loaded_h5py_modules():
        raise RuntimeError("h5py was loaded before the pre-claim handshake")
    blocker = _H5pyImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        _prepare_backend_environment()
        implementation = protocol.get("bindings", {}).get("implementation")
        if not isinstance(implementation, Mapping):
            raise ValueError("protocol lacks frozen implementation mapping")
        preloaded = sorted(name for name in sys.modules
                           if name == "mate_spc_minirocket_v8" or
                           name.startswith("mate_spc_minirocket_v8.") or
                           name == "vendor" or name.startswith("vendor."))
        if preloaded:
            raise RuntimeError(f"v8 production graph must start clean: {preloaded}")
        backend = _load_backend(implementation)
        # Independently repeat attestation here so monkeypatching/replacing the
        # loader cannot substitute an object that merely self-reports success.
        backend = _attest_backend_module(backend, implementation)
        validation = _plain_json(backend.validate_protocol(
            protocol, expected_canonical_json_sha256=canonical_json_sha256))
        if blocker.attempted or _loaded_h5py_modules():
            raise RuntimeError("backend protocol validation attempted to load h5py")
        if (not isinstance(validation, dict) or validation.get("valid") is not True or
                validation.get("schema") != PROTOCOL_SCHEMA or
                validation.get("protocol_id") != PROTOCOL_ID or
                validation.get("candidate_id") != CANDIDATE_ID or
                validation.get("candidate_count") != 1 or
                validation.get("protocol_canonical_json_sha256") != canonical_json_sha256 or
                validation.get("fold_map_canonical_json_sha256") != FOLD_MAP_SHA256 or
                validation.get("access_plan_canonical_json_sha256") != ACCESS_PLAN_SHA256 or
                validation.get("activation_result_file_bytes_sha256") != V7_RESULT_SHA256 or
                validation.get("activation_decision_file_bytes_sha256") != V7_DECISION_SHA256 or
                validation.get("hash_domain") != "canonical_json_without_file_terminator" or
                validation.get("file_bytes_sha256_compared") is not False):
            raise ValueError("backend rejected persisted v8 protocol")
        method_graph = _plain_json(backend.preclaim_bind_method_graph(
            protocol["bindings"]["implementation"]))
        if (not isinstance(method_graph, dict) or method_graph.get("bound") is not True or
                method_graph.get("hdf5_loaded") is not False):
            raise RuntimeError("complete v8 method graph was not bound before claim")
        cuda = _validate_backend_cuda(backend.cuda_preflight())
        if blocker.attempted or _loaded_h5py_modules():
            raise RuntimeError("backend handshake attempted to load h5py")
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
    return backend, {
        "validation": validation, "cuda_and_golden": cuda,
        "method_graph": method_graph,
        "h5py_import_blocker_installed": True,
        "h5py_import_attempted": False, "h5py_module_loaded": False,
        "hdf5_opened": False, "numeric_values_decoded": False,
    }


def _preclaim_audit(protocol: Mapping[str, Any], identities: Mapping[str, Any]
                    ) -> tuple[ModuleType, dict[str, Any]]:
    if os.environ.get("MATE_SPC_V8_FORBID_REAL_H5") == "1":
        raise RuntimeError("synthetic-test guard forbids a formal v8 pre-claim audit")
    activation = _validate_activation()
    tests = _run_test_suite()
    source = _source_record()
    gpu = _gpu_presence_preflight()
    backend, handshake = _production_handshake(
        protocol, canonical_json_sha256=identities["canonical_json_sha256"])
    return backend, {
        "schema": "gpu_minirocket_logodds_v8_preclaim_production_preflight",
        "schema_version": 1, "completed_utc": _now(), "passed": True,
        "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
        "protocol_commit_file_bytes_sha256": identities["commit_file_bytes_sha256"],
        "candidate_id": CANDIDATE_ID, "activation_evidence": activation,
        "source": source, "gpu_presence": gpu, "tests": tests,
        "production_backend_handshake": handshake,
        "claim_written": False, "access_intent_written": False,
        "hdf5_opened": False, "mode1_access_authorized": False,
    }


def preflight() -> dict[str, Any]:
    protocol, identities = _verify_protocol_bundle()
    _backend, audit = _preclaim_audit(protocol, identities)
    return audit


def _is_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_file_record(value: Any, *, filename: str) -> None:
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str) or
            not Path(value["path"]).is_absolute() or Path(value["path"]).name != filename or
            not _is_lower_sha256(value.get("file_bytes_sha256")) or
            not isinstance(value.get("size_bytes"), int) or isinstance(value.get("size_bytes"), bool) or
            value["size_bytes"] <= 0):
        raise ValueError(f"invalid backend artifact record for {filename}")
    path = Path(value["path"])
    if (not path.is_file() or path.stat().st_size != value["size_bytes"] or
            _hash_file(path) != value["file_bytes_sha256"]):
        raise ValueError(f"backend artifact bytes disagree for {filename}")


def _validate_backend_access_state_report(value: Any, *,
                                          protocol_canonical_json_sha256: str,
                                          decision: str,
                                          numeric_folds_completed: int) -> None:
    if not isinstance(value, dict):
        raise ValueError("backend access-state report missing")
    state = value.get("final_state")
    if (not isinstance(state, dict) or
            state.get("schema") != "gpu_minirocket_logodds_v8_backend_access_state" or
            state.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256 or
            state.get("hdf5_open_started") is not True or
            state.get("hdf5_open_completed") is not True or
            not _is_exact_int(state.get("processdata_slices_read"), 400) or
            not _is_exact_int(state.get("idv_init_slices_read"), 400) or
            not _is_exact_int(state.get("numeric_slices_read"), 800) or
            not _is_exact_int(state.get("processdata_slices_completed"), 400) or
            not _is_exact_int(state.get("idv_init_slices_completed"), 400) or
            state.get("training_or_fit_started") is not True or
            not _is_exact_int(state.get("folds_completed"), numeric_folds_completed) or
            state.get("current_fold") is not None or
            state.get("decision_persisted") is not True or
            state.get("final_refit_started") is not (decision == "GO") or
            state.get("backend_completed") is not True or
            not _is_exact_int(state.get("mode1_opens"), 0) or
            not _is_exact_int(state.get("rieth_opens"), 0)):
        raise ValueError("backend access-state final state invalid")
    _validate_file_record(value.get("artifact"), filename="backend_access_state.json")
    artifact_document = _strict_json(Path(value["artifact"]["path"]))
    if artifact_document != state:
        raise ValueError("backend access-state artifact disagrees with final state")


def _validate_decision_evidence(value: Any, *, result: Mapping[str, Any],
                                protocol_canonical_json_sha256: str) -> None:
    _validate_file_record(value, filename="development_decision_before_final_refit.json")
    evidence = _strict_json(Path(value["path"]))
    decision = result["decision"]
    if (evidence.get("schema") !=
            "gpu_minirocket_logodds_v8_development_decision_before_final_refit" or
            evidence.get("protocol_id") != PROTOCOL_ID or
            evidence.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256 or
            evidence.get("candidate_id") != CANDIDATE_ID or
            evidence.get("activation_result_file_bytes_sha256") != V7_RESULT_SHA256 or
            evidence.get("activation_decision_file_bytes_sha256") != V7_DECISION_SHA256 or
            evidence.get("development_decision_determined") is not True or
            evidence.get("decision") != decision or
            evidence.get("selected_candidate_id") != result.get("selected_candidate_id") or
            evidence.get("folds") != result.get("folds") or
            evidence.get("gates") != result.get("gates") or
            evidence.get("aggregate") != result.get("aggregate") or
            evidence.get("authoritative_cuda_repeat") != result.get("authoritative_cuda_repeat") or
            evidence.get("cross_device_report_only") != result.get("cross_device_report_only") or
            evidence.get("final_refit_permitted") is not (decision == "GO") or
            evidence.get("final_refit_started") is not False or
            evidence.get("final_model_usable") is not False or
            evidence.get("mode1_accessed") is not False or evidence.get("mode1_opens") != 0 or
            evidence.get("rieth_accessed") is not False or evidence.get("rieth_opens") != 0 or
            evidence.get("no_go_terminates_current_development_pool_method_search")
            is not (decision == "DEVELOPMENT_NO_GO")):
        raise ValueError("append-only pre-refit decision evidence disagrees with result")


def _decision_from_document(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError("outer decision must be an object")
    from mate_spc_minirocket_v8.evaluation import RunDecision
    row = RunDecision(
        source_idv=value.get("source_idv"), simulation_run=value.get("simulation_run"),
        run_maximum=value.get("run_maximum"), alarm_endpoint=value.get("alarm_endpoint"))
    if value != row.to_document():
        raise ValueError("outer decision derived fields changed")
    return row


def _expected_fold_metrics(rows: Sequence[Any]) -> dict[str, Any]:
    import numpy as np
    healthy = tuple(row for row in rows if row.source_idv == 1)
    faults: dict[str, Any] = {}
    for fault in (3, 9, 15):
        selected = tuple(row for row in rows if row.source_idv == fault)
        delays = np.asarray([row.delay for row in selected], dtype=np.float64)
        detections = sum(row.alarm for row in selected)
        faults[str(fault)] = {
            "runs": 20, "detections": detections, "detection_rate": detections / 20,
            "median_delay_samples": float(np.median(delays)),
        }
    return {"healthy": {"runs": 20, "false_alarms": sum(row.alarm for row in healthy)},
            "faults": faults}


def _validate_complete_fold(value: Mapping[str, Any], *, outer_fold: int,
                            protocol_canonical_json_sha256: str) -> tuple[Any, ...]:
    import numpy as np
    from mate_spc_minirocket_v8 import backend as backend_module
    from mate_spc_minirocket_v8.monitor import (
        InnerThreshold, MonitorCalibration, inner_threshold)
    expected_roles = fold_roles(outer_fold)
    if (value.get("outer_fold") != outer_fold or value.get("numeric_valid") is not True or
            value.get("roles") != expected_roles or
            value.get("fit_runs_per_class") != 60 or value.get("fit_total_run_count") != 240 or
            value.get("inner_threshold_healthy_run_count") != 20 or
            value.get("inner_threshold_fault_run_count") != 0 or
            value.get("outer_run_count_per_class") != 20 or
            value.get("training_example_count") != 20400):
        raise ValueError(f"fold {outer_fold} fixed structure invalid")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 80:
        raise ValueError(f"fold {outer_fold} must contain 80 decisions")
    rows = tuple(_decision_from_document(item) for item in decisions)
    expected_identities = tuple((idv, run) for run in FOLD_RUN_IDS[outer_fold]
                                for idv in CONDITION_IDVS)
    if tuple((row.source_idv, row.simulation_run) for row in rows) != expected_identities:
        raise ValueError(f"fold {outer_fold} decision identities/order invalid")
    if value.get("metrics") != _expected_fold_metrics(rows):
        raise ValueError(f"fold {outer_fold} metrics were not derived from decisions")

    calibration_doc, threshold_doc = value.get("monitor_calibration"), value.get("threshold")
    if not isinstance(calibration_doc, dict) or not isinstance(threshold_doc, dict):
        raise ValueError(f"fold {outer_fold} calibration/threshold missing")
    calibration = MonitorCalibration(
        center=calibration_doc.get("center"), scale=calibration_doc.get("scale"),
        pool_size=calibration_doc.get("pool_size"))
    threshold = InnerThreshold(
        threshold=threshold_doc.get("threshold"), rank19=threshold_doc.get("rank19"),
        distinct_count=threshold_doc.get("distinct_count"),
        repeat_error=threshold_doc.get("repeat_error"))
    if calibration_doc != calibration.to_document() or threshold_doc != threshold.to_document():
        raise ValueError(f"fold {outer_fold} calibration/threshold document changed")

    repeat = value.get("authoritative_cuda_repeat")
    component_names = {"fit_healthy_log_odds", "calibration_center", "calibration_scale",
                       "inner_log_odds", "scan_paths", "run_maxima"}
    if (not isinstance(repeat, dict) or
            repeat.get("schema") != "gpu_minirocket_logodds_v8_authoritative_cuda_repeat" or
            repeat.get("scope") !=
                "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay" or
            repeat.get("device") != "cuda" or repeat.get("atol") != 1e-6 or
            repeat.get("passes") is not True or repeat.get("discrete_equal") is not True or
            set(repeat.get("component_maximum_absolute_errors", {})) != component_names or
            not isinstance(repeat.get("first_run_maxima"), list) or
            not isinstance(repeat.get("second_run_maxima"), list) or
            len(repeat["first_run_maxima"]) != 20 or len(repeat["second_run_maxima"]) != 20):
        raise ValueError(f"fold {outer_fold} CUDA replay structure invalid")
    components = repeat["component_maximum_absolute_errors"]
    numbers = [repeat.get("maximum_absolute_error"), *components.values(),
               *repeat["first_run_maxima"], *repeat["second_run_maxima"]]
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) or
           not math.isfinite(float(item)) or float(item) < 0 for item in numbers):
        raise ValueError(f"fold {outer_fold} CUDA replay numeric evidence invalid")
    if (float(repeat["maximum_absolute_error"]) != max(float(item) for item in components.values()) or
            float(repeat["maximum_absolute_error"]) > 1e-6 or
            threshold.repeat_error != float(repeat["maximum_absolute_error"])):
        raise ValueError(f"fold {outer_fold} CUDA replay/threshold mismatch")
    first_maxima = np.asarray(repeat["first_run_maxima"], dtype=np.float64)
    second_maxima = np.asarray(repeat["second_run_maxima"], dtype=np.float64)
    maxima_error = float(np.max(np.abs(first_maxima - second_maxima), initial=0.0))
    if float(components["run_maxima"]) != maxima_error:
        raise ValueError(f"fold {outer_fold} replay maxima error was not recomputed")
    try:
        first_derived = inner_threshold(first_maxima, float(repeat["maximum_absolute_error"]))
        second_derived = inner_threshold(second_maxima, float(repeat["maximum_absolute_error"]))
    except (ValueError, FloatingPointError) as error:
        raise ValueError(f"fold {outer_fold} replay threshold is not derivable") from error
    if (first_derived.to_document() != threshold_doc or
            second_derived.to_document() != threshold_doc):
        raise ValueError(f"fold {outer_fold} threshold does not match both replay maxima")

    frozen = value.get("artifacts_frozen_before_outer")
    if not isinstance(frozen, dict):
        raise ValueError(f"fold {outer_fold} frozen artifacts missing")
    for key, filename in (("transform", "minirocket_transform.npz"),
                          ("fold_model_state", "fold_model_state.npz"),
                          ("threshold", "inner_threshold.json"),
                          ("freeze_manifest", "freeze_manifest.json")):
        _validate_file_record(frozen.get(key), filename=filename)
    manifest_path = Path(frozen["freeze_manifest"]["path"])
    manifest = _strict_json(manifest_path)
    if (manifest.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256 or
            manifest.get("outer_fold") != outer_fold or manifest.get("roles") != expected_roles or
            manifest.get("transform") != frozen["transform"] or
            manifest.get("fold_model_state") != frozen["fold_model_state"] or
            manifest.get("threshold") != frozen["threshold"] or
            manifest.get("transform_structure_sha256") !=
                frozen.get("transform_structure_sha256")):
        raise ValueError(f"fold {outer_fold} freeze manifest/result mismatch")
    _, persisted_calibration, persisted_threshold = (
        backend_module.load_frozen_fold_pipeline(manifest_path))
    if (persisted_calibration.to_document() != calibration_doc or
            persisted_threshold.to_document() != threshold_doc):
        raise ValueError(f"fold {outer_fold} result differs from frozen calibration/threshold")

    outer_record = value.get("outer_result_append_only")
    _validate_file_record(outer_record, filename="outer_result.json")
    outer = _strict_json(Path(outer_record["path"]))
    replay = outer.get("replay")
    if (outer.get("schema") != "gpu_minirocket_logodds_v8_outer_fold_result" or
            outer.get("outer_fold") != outer_fold or outer.get("authoritative_device") != "cuda" or
            outer.get("scored_once_after_freeze") is not True or
            outer.get("metrics") != value.get("metrics") or outer.get("decisions") != decisions or
            not isinstance(replay, list) or len(replay) != 80):
        raise ValueError(f"fold {outer_fold} append-only outer artifact mismatch")
    for item, row in zip(replay, rows, strict=True):
        path = item.get("score_path_endpoints_160_499") if isinstance(item, dict) else None
        if (item.get("identity") != {"source_idv": row.source_idv,
                                     "simulation_run": row.simulation_run} or
                not isinstance(path, list) or len(path) != 340 or
                any(not isinstance(score, (int, float)) or isinstance(score, bool) or
                    not math.isfinite(float(score)) or float(score) < 0 for score in path)):
            raise ValueError(f"fold {outer_fold} outer replay path invalid")
        maximum = max(float(score) for score in path)
        alarm_indices = [index for index, score in enumerate(path)
                         if float(score) > threshold.threshold]
        endpoint = 160 + alarm_indices[0] if alarm_indices else None
        if maximum != row.run_maximum or endpoint != row.alarm_endpoint:
            raise ValueError(f"fold {outer_fold} decision not reproduced by score path")
    cpu = value.get("cross_device_report_only")
    if (not isinstance(cpu, dict) or cpu.get("is_go_gate") is not False or
            cpu.get("authoritative_device") != "cuda" or cpu.get("comparison_device") != "cpu" or
            not isinstance(cpu.get("completed"), bool)):
        raise ValueError(f"fold {outer_fold} CPU report invalid")
    return rows


def _validate_development_result(value: Any, *,
                                 protocol_canonical_json_sha256: str,
                                 expected_run_authorization: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain_json(value)
    if not isinstance(result, dict):
        raise ValueError("backend result must be a mapping")
    decision = result.get("decision")
    if (result.get("schema") != BACKEND_RESULT_SCHEMA or result.get("protocol_id") != PROTOCOL_ID or
            result.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256 or
            result.get("candidate_id") != CANDIDATE_ID or
            not _is_exact_int(result.get("candidate_count"), 1) or
            result.get("source_file_bytes_sha256") != SOURCE_SHA256 or
            not _is_exact_int(result.get("source_size_bytes"), SOURCE_SIZE_BYTES) or
            result.get("fold_map_canonical_json_sha256") != FOLD_MAP_SHA256 or
            result.get("access_plan_canonical_json_sha256") != ACCESS_PLAN_SHA256 or
            result.get("activation_result_file_bytes_sha256") != V7_RESULT_SHA256 or
            result.get("activation_decision_file_bytes_sha256") != V7_DECISION_SHA256 or
            result.get("mode1_accessed") is not False or not _is_exact_int(result.get("mode1_opens"), 0) or
            result.get("rieth_accessed") is not False or not _is_exact_int(result.get("rieth_opens"), 0) or
            not _is_exact_int(result.get("inner_threshold_fault_run_count"), 0) or
            result.get("method_search_on_current_pool_terminated") is not
                (decision == "DEVELOPMENT_NO_GO") or
            decision not in {"GO", "DEVELOPMENT_NO_GO"}):
        raise ValueError("backend result identity binding invalid")
    topology = result.get("topology")
    numeric = topology.get("numeric_value_access") if isinstance(topology, dict) else None
    if (not isinstance(topology, dict) or not _is_exact_int(topology.get("selected_run_count"), 400) or
            topology.get("counts_by_status") != {"SimulationCompleted": 400} or
            topology.get("counts_by_idv") != {"1": 100, "3": 100, "9": 100, "15": 100} or
            not _is_exact_int(topology.get("unique_seed_count"), 395) or
            not _is_exact_int(topology.get("duplicate_seed_value_count"), 5) or
            not _is_exact_int(topology.get("cross_fold_seed_collision_count"), 0) or
            not isinstance(numeric, dict) or not _is_exact_int(numeric.get("processdata_slices"), 400) or
            not _is_exact_int(numeric.get("idv_init_slices"), 400) or
            numeric.get("processdata_columns_half_open") != [0, 53] or
            not _is_exact_int(numeric.get("root_dataset_value_reads"), 0) or
            not _is_exact_int(numeric.get("processdata_values_outside_authorized_windows"), 0)):
        raise ValueError("backend exact-selector topology invalid")
    expected_authorization = _plain_json(expected_run_authorization)
    if result.get("run_authorization") != expected_authorization:
        raise ValueError("backend result does not bind the exact protocol/claim/intent authorization")
    environment = result.get("environment")
    try:
        _validate_backend_cuda(environment)
    except RuntimeError as error:
        raise ValueError("backend result is not bound to the exact environment lock") from error
    folds = result.get("folds")
    if (not isinstance(folds, list) or len(folds) != 5 or
            any(not isinstance(fold, dict) for fold in folds) or
            [fold.get("outer_fold") for fold in folds] != list(range(5))):
        raise ValueError("backend must return exactly five identified folds")
    numeric_complete = all(fold.get("numeric_valid") is True for fold in folds)
    rows: list[Any] = []
    if numeric_complete:
        for outer_fold, fold in enumerate(folds):
            rows.extend(_validate_complete_fold(
                fold, outer_fold=outer_fold,
                protocol_canonical_json_sha256=protocol_canonical_json_sha256))
    else:
        failed = next((index for index, fold in enumerate(folds)
                       if fold.get("numeric_valid") is not True), None)
        if failed is None or decision != "DEVELOPMENT_NO_GO":
            raise ValueError("numeric-invalid fold may only produce DEVELOPMENT_NO_GO")
        for outer_fold in range(failed):
            rows.extend(_validate_complete_fold(
                folds[outer_fold], outer_fold=outer_fold,
                protocol_canonical_json_sha256=protocol_canonical_json_sha256))
        failed_doc = folds[failed]
        if (failed_doc.get("numeric_valid") is not False or
                failed_doc.get("status") != "FOLD_NUMERIC_NO_GO" or
                not isinstance(failed_doc.get("error_type"), str) or
                not isinstance(failed_doc.get("error"), str)):
            raise ValueError("failed numeric fold evidence invalid")
        for outer_fold in range(failed + 1, 5):
            if folds[outer_fold] != {"outer_fold": outer_fold, "numeric_valid": False,
                                     "status": "NOT_RUN_AFTER_NUMERIC_NO_GO"}:
                raise ValueError("post-failure folds must be exact NOT_RUN placeholders")
    _validate_backend_access_state_report(
        result.get("backend_access_state"),
        protocol_canonical_json_sha256=protocol_canonical_json_sha256,
        decision=decision,
        numeric_folds_completed=(5 if numeric_complete else int(failed)))
    repeat = result.get("authoritative_cuda_repeat")
    if (not isinstance(repeat, dict) or not isinstance(repeat.get("passes"), bool) or
            not isinstance(repeat.get("discrete_equal"), bool) or repeat.get("atol") != 1e-6 or
            repeat.get("scope") != "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay"):
        raise ValueError("authoritative CUDA repeat report invalid")
    error = repeat.get("maximum_absolute_error")
    if repeat["passes"] and (not repeat.get("discrete_equal") or not isinstance(error, (int, float)) or
                             isinstance(error, bool) or not math.isfinite(float(error)) or float(error) > 1e-6):
        raise ValueError("authoritative CUDA repeat gate invalid")
    if error is not None and (not isinstance(error, (int, float)) or isinstance(error, bool) or
                              not math.isfinite(float(error)) or float(error) < 0):
        raise ValueError("authoritative repeat error invalid")
    cpu = result.get("cross_device_report_only")
    if not isinstance(cpu, dict) or cpu.get("is_go_gate") is not False:
        raise ValueError("CPU comparison must be report-only")
    gates = result.get("gates")
    if numeric_complete:
        from mate_spc_minirocket_v8.evaluation import evaluate_gates
        recomputed = evaluate_gates(rows, [True] * 5).to_document()
        if (gates != recomputed or result.get("aggregate") !=
                recomputed["aggregate_metrics"] or recomputed["decision"] != decision):
            raise ValueError("gates/aggregate/decision were not independently reproduced")
        fold_repeats = [fold["authoritative_cuda_repeat"] for fold in folds]
        if (repeat.get("folds") != fold_repeats or
                repeat.get("passes") is not all(item["passes"] for item in fold_repeats) or
                repeat.get("discrete_equal") is not
                    all(item["discrete_equal"] for item in fold_repeats) or
                float(repeat["maximum_absolute_error"]) !=
                    max(float(item["maximum_absolute_error"]) for item in fold_repeats)):
            raise ValueError("aggregate CUDA replay was not derived from five folds")
        cpu_folds = [fold["cross_device_report_only"] for fold in folds]
        if cpu.get("folds") != cpu_folds:
            raise ValueError("aggregate CPU report was not derived from five folds")
        rng = result.get("formal_rng")
        structures = [fold["artifacts_frozen_before_outer"]["transform_structure_sha256"]
                      for fold in folds]
        if (not isinstance(rng, dict) or
                rng.get("seeded_exactly_once_after_environment_preflight") is not True or
                rng.get("seed") != 20260912 or rng.get("fold_order") != [0, 1, 2, 3, 4] or
                rng.get("transform_structure_sha256_by_fold") != structures or
                rng.get("all_fold_structures_distinct") is not True or
                len(set(structures)) != 5 or
                rng.get("go_refit_continues_without_reseed") is not (decision == "GO") or
                not isinstance(rng.get("start_state"), dict)):
            raise ValueError("single continuous formal RNG evidence invalid")
    else:
        if (gates != {"all_pass": False, "checks": {"numeric_all_folds": False},
                      "aggregate_metrics": None} or result.get("aggregate") is not None or
                repeat != {"scope":
                               "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay",
                           "atol": 1e-6, "maximum_absolute_error": None,
                           "discrete_equal": False, "passes": False, "folds": []} or
                cpu != {"is_go_gate": False, "folds": []}):
            raise ValueError("numeric-NO_GO summary must be the exact fail-closed document")
    if decision == "GO" and not (repeat["passes"] and repeat["discrete_equal"]):
        raise ValueError("GO requires CUDA repeat pass")
    if result.get("selected_candidate_id") != (CANDIDATE_ID if decision == "GO" else None):
        raise ValueError("selected candidate disagrees with decision")
    _validate_decision_evidence(
        result.get("development_decision_before_final_refit"), result=result,
        protocol_canonical_json_sha256=protocol_canonical_json_sha256)
    final_refit = result.get("final_refit")
    if decision == "DEVELOPMENT_NO_GO" and final_refit is not None:
        raise ValueError("NO_GO must not create final refit")
    if decision == "GO":
        if (not isinstance(final_refit, dict) or
                final_refit.get("schema") != "gpu_minirocket_logodds_v8_final_mode2_refit" or
                final_refit.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256 or
                final_refit.get("candidate_id") != CANDIDATE_ID or
                not _is_exact_int(final_refit.get("runs_per_class"), 100) or
                not _is_exact_int(final_refit.get("total_fit_run_count"), 400) or
                not _is_exact_int(final_refit.get("training_example_count"), 34_000) or
                final_refit.get("created_only_after_development_go") is not True or
                final_refit.get("continued_rng_stream_without_reseed") is not True or
                final_refit.get("target_domain_threshold_not_fitted") is not True or
                final_refit.get("mode1_accessed") is not False or
                not _is_exact_int(final_refit.get("mode1_opens"), 0) or
                final_refit.get("strict_no_pickle_reload_verified") is not True or
                final_refit.get("reload_log_odds_max_abs_error") != 0.0):
            raise ValueError("GO-only final-refit metadata invalid")
        _validate_file_record(final_refit.get("transform"), filename="minirocket_transform.npz")
        _validate_file_record(final_refit.get("state"), filename="final_model_state.npz")
        _validate_file_record(final_refit.get("manifest"), filename="manifest.json")
        manifest = _strict_json(Path(final_refit["manifest"]["path"]))
        expected_manifest = {key: item for key, item in final_refit.items()
                             if key not in {"manifest", "strict_no_pickle_reload_verified",
                                            "reload_log_odds_max_abs_error"}}
        if manifest != expected_manifest:
            raise ValueError("GO final-refit manifest disagrees with result")
        from mate_spc_minirocket_v8 import backend as backend_module
        backend_module.load_final_refit_pipeline(final_refit["manifest"]["path"])
    return result


def _read_backend_access_state(directory: Path) -> dict[str, Any] | None:
    path = directory / "artifacts" / "backend_access_state.json"
    if not path.is_file():
        return None
    try:
        return _strict_json(path)
    except Exception as error:
        return {"state_file_invalid": True, "error_type": type(error).__name__}


def _failure_state(*, claim_written: bool, intent_written: bool,
                   backend_state: Mapping[str, Any] | None,
                   intent_bundle_observed: bool = False) -> str:
    if not claim_written:
        return "INFRASTRUCTURE_FAILURE_BEFORE_CLAIM"
    if not intent_written:
        if intent_bundle_observed:
            return "ABORTED_DURING_ACCESS_INTENT_COMMIT"
        return "ABORTED_AFTER_CLAIM_BEFORE_ACCESS_INTENT"
    if backend_state is None or backend_state.get("state_file_invalid") is True:
        return "ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN"
    booleans = ("hdf5_open_started", "hdf5_open_completed", "training_or_fit_started", "backend_completed")
    counters = ("processdata_slices_read", "idv_init_slices_read", "numeric_slices_read",
                "processdata_slices_completed", "idv_init_slices_completed",
                "mode1_opens", "rieth_opens")
    if (backend_state.get("schema") != "gpu_minirocket_logodds_v8_backend_access_state" or
            any(not isinstance(backend_state.get(k), bool) for k in booleans) or
            any(not isinstance(backend_state.get(k), int) or isinstance(backend_state.get(k), bool) or
                backend_state[k] < 0 for k in counters) or
            backend_state["processdata_slices_read"] > 400 or backend_state["idv_init_slices_read"] > 400 or
            backend_state["numeric_slices_read"] != backend_state["processdata_slices_read"] + backend_state["idv_init_slices_read"] or
            backend_state["processdata_slices_completed"] > backend_state["processdata_slices_read"] or
            backend_state["idv_init_slices_completed"] > backend_state["idv_init_slices_read"] or
            backend_state["mode1_opens"] != 0 or backend_state["rieth_opens"] != 0 or
            (backend_state["hdf5_open_completed"] and not backend_state["hdf5_open_started"]) or
            (backend_state["numeric_slices_read"] > 0 and not backend_state["hdf5_open_completed"]) or
            (backend_state["training_or_fit_started"] and backend_state["numeric_slices_read"] != 800)):
        return "ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN"
    if backend_state["numeric_slices_read"] > 0:
        return "ABORTED_AFTER_NUMERIC_READ"
    if backend_state["hdf5_open_completed"]:
        return "ABORTED_AFTER_HDF5_OPEN_BEFORE_NUMERIC_SLICE"
    if backend_state["hdf5_open_started"]:
        return "ABORTED_DURING_HDF5_OPEN_BEFORE_COMPLETION"
    return "ABORTED_AFTER_INTENT_BEFORE_HDF5_OPEN"


def _bundle_observation(path: Path) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for label, candidate in (("document", path), ("sidecar", _sidecar_path(path)),
                             ("commit", _commit_path(path))):
        record: dict[str, Any] = {"path": str(candidate.resolve()),
                                  "exists": candidate.is_file()}
        if candidate.is_file():
            record.update({"file_bytes_sha256": _hash_file(candidate),
                           "size_bytes": candidate.stat().st_size})
        observed[label] = record
    return observed


def _absolute_file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "file_bytes_sha256": _hash_file(path)}


def _write_failure_and_completion(*, directory: Path, protocol_identities: Mapping[str, Any],
                                  stage: str, error: BaseException, claim_written: bool,
                                  global_claim_file_bytes_sha256: str,
                                  global_claim_commit_file_bytes_sha256: str,
                                  intent_written: bool, preflight_file_bytes_sha256: str | None,
                                  access_intent_identity: Mapping[str, Any] | None) -> dict[str, Any]:
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    backend_state = _read_backend_access_state(directory)
    intent_bundle = _bundle_observation(directory / "access_intent.json")
    intent_observed = any(record["exists"] for record in intent_bundle.values())
    decision = _failure_state(claim_written=claim_written, intent_written=intent_written,
                              backend_state=backend_state,
                              intent_bundle_observed=intent_observed)
    execution_claim_path = directory / "numeric_execution.claim.json"
    execution_claim = (_absolute_file_record(execution_claim_path)
                       if execution_claim_path.is_file() else None)
    intent_identity = _plain_json(access_intent_identity)
    failure = {
        "schema": "gpu_minirocket_logodds_v8_development_failure",
        "failed_utc": _now(), "decision": decision, "failure_stage": stage,
        "exception_type": type(error).__name__, "exception_message": str(error),
        "traceback_sha256": _hash_bytes(trace.encode("utf-8")),
        "claim_written": claim_written, "access_intent_written": intent_written,
        "global_claim_file_bytes_sha256": global_claim_file_bytes_sha256,
        "global_claim_commit_file_bytes_sha256": global_claim_commit_file_bytes_sha256,
        "access_intent_identity_if_committed": intent_identity,
        "observed_access_intent_bundle": intent_bundle,
        "numeric_execution_claim": execution_claim,
        "backend_access_state": backend_state, "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0, "rieth_accessed": False,
    }
    failure_identity = _write_json(directory / "failure.json", failure)
    completion = {
        "schema": "gpu_minirocket_logodds_v8_development_completion",
        "completed_utc": _now(), "decision": decision, "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": protocol_identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": protocol_identities["canonical_json_sha256"],
        "failure_file_bytes_sha256": failure_identity["file_bytes_sha256"],
        "global_claim_file_bytes_sha256": global_claim_file_bytes_sha256,
        "global_claim_commit_file_bytes_sha256": global_claim_commit_file_bytes_sha256,
        "preflight_file_bytes_sha256": preflight_file_bytes_sha256,
        "access_intent_file_bytes_sha256": (
            intent_identity.get("file_bytes_sha256") if intent_identity else None),
        "access_intent_canonical_json_sha256": (
            intent_identity.get("canonical_json_sha256") if intent_identity else None),
        "access_intent_file_size_bytes": (
            intent_identity.get("file_size_bytes") if intent_identity else None),
        "access_intent_sidecar_file_bytes_sha256": (
            intent_identity.get("sidecar_file_bytes_sha256") if intent_identity else None),
        "access_intent_commit_file_bytes_sha256": (
            intent_identity.get("commit_file_bytes_sha256") if intent_identity else None),
        "observed_access_intent_bundle": intent_bundle,
        "numeric_execution_claim": execution_claim,
        "mode1_accessed": False, "mode1_hdf5_opens_or_value_reads": 0,
        "rieth_accessed": False,
    }
    identity = _write_json(directory / "completion.json", completion)
    return {**completion, "completion_file_bytes_sha256": identity["file_bytes_sha256"]}


def _write_claim_creation_failure(*, stage: str, error: BaseException,
                                  protocol_identities: Mapping[str, Any],
                                  claim_identity: Mapping[str, Any] | None,
                                  claim_attempt_id: str,
                                  claim_lease_acquired: bool) -> None:
    """Persist fail-closed evidence when the claim bundle/directory cannot finish."""

    observed = _bundle_observation(GLOBAL_CLAIM_PATH)
    # A concurrent loser never acquired the kernel lease and must not poison
    # the winner.  An owner writes failure evidence even when its partial claim
    # bytes cannot be parsed.
    if not claim_lease_acquired:
        return
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    payload = {
        "schema": "gpu_minirocket_logodds_v8_global_claim_creation_failure",
        "failed_utc": _now(), "failure_stage": stage,
        "decision": "FAIL_CLOSED_NO_RERUN",
        "exception_type": type(error).__name__, "exception_message": str(error),
        "traceback_sha256": _hash_bytes(trace.encode("utf-8")),
        "protocol_file_bytes_sha256": protocol_identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": protocol_identities["canonical_json_sha256"],
        "claim_identity_if_committed": _plain_json(claim_identity),
        "claim_attempt_id": claim_attempt_id,
        "observed_claim_bundle": observed,
        "hdf5_import_or_open_started": False,
        "mode1_accessed": False, "mode1_hdf5_opens_or_value_reads": 0,
        "rieth_accessed": False,
    }
    try:
        _write_json(GLOBAL_CLAIM_FAILURE_PATH, payload)
    except BaseException:
        # Never mask the primary claim/creation error.  Any partial claim bundle
        # still blocks future runs via the preclaim existence check below.
        pass


def _require_formal_development_invocation() -> dict[str, Any]:
    """Reject any launch that is not the frozen isolated/unbuffered command."""

    observation = {
        "sys_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "script_path": str(Path(sys.argv[0]).resolve()),
        "argv_tail": list(sys.argv[1:]),
        "working_directory": str(Path.cwd().resolve()),
        "isolated": int(sys.flags.isolated),
        "no_site": int(sys.flags.no_site),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
        "stdout_write_through": getattr(sys.stdout, "write_through", None),
        "stderr_write_through": getattr(sys.stderr, "write_through", None),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "pythonpath_ignored_by_isolated_mode": bool(sys.flags.isolated),
    }
    if (
        Path(sys.executable).resolve() != FORMAL_PYTHON_EXE.resolve()
        or Path(sys.prefix).resolve() != FORMAL_ENV_PREFIX.resolve()
        or Path(sys.argv[0]).resolve() != Path(__file__).resolve()
        or sys.argv[1:] != ["development"]
        or Path.cwd().resolve() != PROJECT_ROOT.resolve()
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or getattr(sys.stdout, "write_through", None) is not True
        or getattr(sys.stderr, "write_through", None) is not True
        or Path(os.environ.get("CONDA_PREFIX", "")).resolve() != FORMAL_ENV_PREFIX.resolve()
        or os.environ.get("CONDA_DEFAULT_ENV") != "torch5090"
    ):
        raise RuntimeError("formal v8 development invocation contract was not satisfied")
    return observation


def run_development(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    if Path(output_root).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("formal v8 output_root must be the frozen default directory")
    invocation = _require_formal_development_invocation()
    protocol, identities = _verify_protocol_bundle()
    plan = development_access_plan()
    plan_sha256 = _hash_bytes(_canonical_bytes(plan))
    if plan_sha256 != ACCESS_PLAN_SHA256:
        raise ValueError("live v8 access plan differs from frozen plan")
    # This validates activation immediately before the claim, in addition to
    # its protocol-bound identity.
    activation = _validate_activation()
    backend, audit = _preclaim_audit(protocol, identities)

    root = Path(output_root)
    directory = root / DEVELOPMENT_DIRECTORY_NAME
    claim_paths = (GLOBAL_CLAIM_PATH, _sidecar_path(GLOBAL_CLAIM_PATH),
                   _commit_path(GLOBAL_CLAIM_PATH), GLOBAL_CLAIM_FAILURE_PATH)
    if any(path.exists() for path in claim_paths) or directory.exists():
        raise FileExistsError("prior v8 global claim or fixed directory exists")
    root.mkdir(parents=True, exist_ok=True)
    claim_attempt_id = uuid.uuid4().hex
    claim = {
        "schema": "gpu_minirocket_logodds_v8_exclusive_global_development_claim",
        "claimed_utc": _now(), "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
        "candidate_id": CANDIDATE_ID, "candidate_count": 1,
        "activation_result_file_bytes_sha256": V7_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": V7_DECISION_SHA256,
        "access_plan_canonical_json_sha256": plan_sha256,
        "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
        "claim_attempt_id": claim_attempt_id,
        "formal_invocation": invocation,
        "artifact_root": str(directory.resolve()), "fixed_directory": DEVELOPMENT_DIRECTORY_NAME,
        "backend_artifact_root": str((directory / "artifacts").resolve()),
        "numeric_execution_claim_path":
            str((directory / "numeric_execution.claim.json").resolve()),
        "rescue_iteration_ids_allowed": False,
        "policy": "Any prior claim blocks every rerun, including alternate output roots.",
    }
    claim_identity: Mapping[str, Any] | None = None
    claim_ownership = {"lease_acquired": False}
    try:
        claim_identity = _commit_global_claim(claim, ownership=claim_ownership)
        directory.mkdir(parents=False, exist_ok=False)
    except BaseException as error:
        _write_claim_creation_failure(
            stage=("global_claim_commit" if claim_identity is None
                   else "fixed_artifact_directory_creation_after_claim"),
            error=error, protocol_identities=identities,
            claim_identity=claim_identity, claim_attempt_id=claim_attempt_id,
            claim_lease_acquired=claim_ownership["lease_acquired"])
        raise
    assert claim_identity is not None

    stage = "persist_preclaim_preflight"
    claim_written, intent_written = True, False
    preflight_sha256: str | None = None
    intent_identities: Mapping[str, Any] | None = None
    try:
        preflight_identity = _write_json(directory / "preflight.json", audit)
        preflight_sha256 = preflight_identity["file_bytes_sha256"]
        stage = "access_intent_commit"
        intent = {
            "schema": "gpu_minirocket_logodds_v8_mode2_development_access_intent",
            "schema_version": 1, "committed_utc": _now(),
            "committed_after_read_free_backend_handshake": True,
            "committed_before_hdf5_import_or_open": True,
            "protocol_id": PROTOCOL_ID,
            "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
            "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
            "candidate_id": CANDIDATE_ID, "candidate_count": 1,
            "activation_evidence": activation, "source": audit["source"],
            "access_plan": plan, "access_plan_canonical_json_sha256": plan_sha256,
            "fold_run_ids": {str(f): list(r) for f, r in FOLD_RUN_IDS.items()},
            "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
            "authorized_reads_after_commit": "only exact 400 Mode2 magnitude50 selectors",
            "wildcards_authorized": False, "inner_fault_threshold_values_authorized": False,
            "mode1_access_authorized": False, "mode1_hdf5_opens_or_value_reads": 0,
            "mode1_magnitude25_state": "SEALED", "mode1_magnitude75_state": "SEALED",
            "rieth_access_authorized": False, "backend_module": BACKEND_MODULE,
            "backend_imported_before_commit": True,
            "production_backend_instance_reused_after_commit": True,
            "h5py_module_loaded_before_commit": False,
            "h5py_import_attempted_before_commit": False,
            "artifact_root": str((directory / "artifacts").resolve()),
            "numeric_execution_claim_path":
                str((directory / "numeric_execution.claim.json").resolve()),
            "formal_invocation": invocation,
            "preflight_file_bytes_sha256": preflight_sha256,
            "global_claim_path": str(GLOBAL_CLAIM_PATH.resolve()),
            "global_claim_file_bytes_sha256": claim_identity["file_bytes_sha256"],
            "global_claim_canonical_json_sha256": claim_identity["canonical_json_sha256"],
            "global_claim_file_size_bytes": claim_identity["file_size_bytes"],
            "global_claim_commit_file_bytes_sha256":
                claim_identity["commit_file_bytes_sha256"],
            "implementation": protocol["bindings"]["implementation"],
        }
        intent_identities = _commit_json(
            directory / "access_intent.json", intent,
            commit_schema="gpu_minirocket_logodds_v8_access_intent_atomic_commit")
        intent_written = True
        if GLOBAL_CLAIM_FAILURE_PATH.exists():
            raise RuntimeError("competing or partial global-claim failure evidence exists")
        stage = "numeric_backend_development"
        raw = backend.run_development(
            access_plan=plan, fold_map=_canonical_fold_map(),
            artifact_root=directory / "artifacts",
            protocol_canonical_json_sha256=identities["canonical_json_sha256"],
            protocol_path=PROTOCOL_PATH,
            protocol_file_bytes_sha256=identities["file_bytes_sha256"],
            protocol_commit_file_bytes_sha256=identities["commit_file_bytes_sha256"],
            claim_path=GLOBAL_CLAIM_PATH,
            claim_file_bytes_sha256=claim_identity["file_bytes_sha256"],
            claim_commit_file_bytes_sha256=claim_identity["commit_file_bytes_sha256"],
            access_intent_path=directory / "access_intent.json",
            access_intent_file_bytes_sha256=intent_identities["file_bytes_sha256"],
            access_intent_commit_file_bytes_sha256=
                intent_identities["commit_file_bytes_sha256"],
            device="cuda")
        execution_claim_path = directory / "numeric_execution.claim.json"
        if not execution_claim_path.is_file():
            raise ValueError("backend did not persist the numeric execution claim")
        execution_claim_record = _absolute_file_record(execution_claim_path)
        if raw.get("run_authorization", {}).get("numeric_execution_claim") != execution_claim_record:
            raise ValueError("backend result does not bind the physical numeric execution claim")
        expected_run_authorization = {
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
            "protocol_commit_file_bytes_sha256": identities["commit_file_bytes_sha256"],
            "claim_path": str(GLOBAL_CLAIM_PATH.resolve()),
            "claim_file_bytes_sha256": claim_identity["file_bytes_sha256"],
            "claim_commit_file_bytes_sha256": claim_identity["commit_file_bytes_sha256"],
            "access_intent_path": str((directory / "access_intent.json").resolve()),
            "access_intent_file_bytes_sha256": intent_identities["file_bytes_sha256"],
            "access_intent_commit_file_bytes_sha256":
                intent_identities["commit_file_bytes_sha256"],
            "validated_before_hdf5_import_or_open": True,
            "method_graph_bound_before_hdf5": True,
            "numeric_execution_claim": execution_claim_record,
        }
        result = _validate_development_result(
            raw, protocol_canonical_json_sha256=identities["canonical_json_sha256"],
            expected_run_authorization=expected_run_authorization)
        result_identity = _write_json(directory / "development_result.json", result)
        completion = {
            "schema": "gpu_minirocket_logodds_v8_development_completion",
            "completed_utc": _now(), "decision": result["decision"],
            "protocol_id": PROTOCOL_ID,
            "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
            "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
            "preflight_file_bytes_sha256": preflight_sha256,
            "global_claim_file_bytes_sha256": claim_identity["file_bytes_sha256"],
            "global_claim_commit_file_bytes_sha256":
                claim_identity["commit_file_bytes_sha256"],
            "access_intent_file_bytes_sha256": intent_identities["file_bytes_sha256"],
            "access_intent_commit_file_bytes_sha256": intent_identities["commit_file_bytes_sha256"],
            "access_intent_canonical_json_sha256": intent_identities["canonical_json_sha256"],
            "access_intent_file_size_bytes": intent_identities["file_size_bytes"],
            "access_intent_sidecar_file_bytes_sha256":
                intent_identities["sidecar_file_bytes_sha256"],
            "numeric_execution_claim": execution_claim_record,
            "development_result_file_bytes_sha256": result_identity["file_bytes_sha256"],
            "source_file_bytes_sha256": SOURCE_SHA256,
            "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
            "access_plan_canonical_json_sha256": ACCESS_PLAN_SHA256,
            "activation_result_file_bytes_sha256": V7_RESULT_SHA256,
            "activation_decision_file_bytes_sha256": V7_DECISION_SHA256,
            "candidate_count": 1, "mode1_accessed": False,
            "mode1_hdf5_opens_or_value_reads": 0, "rieth_accessed": False,
        }
        completion_identity = _write_json(directory / "completion.json", completion)
        return {**completion,
                "completion_file_bytes_sha256": completion_identity["file_bytes_sha256"]}
    except BaseException as error:
        _write_failure_and_completion(
            directory=directory, protocol_identities=identities, stage=stage, error=error,
            claim_written=claim_written, intent_written=intent_written,
            global_claim_file_bytes_sha256=claim_identity["file_bytes_sha256"],
            global_claim_commit_file_bytes_sha256=
                claim_identity["commit_file_bytes_sha256"],
            preflight_file_bytes_sha256=preflight_sha256,
            access_intent_identity=intent_identities)
        raise


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    directory = Path(output_root) / DEVELOPMENT_DIRECTORY_NAME
    completion, result = directory / "completion.json", directory / "development_result.json"
    activation_valid = False
    try:
        _validate_activation()
        activation_valid = True
    except Exception:
        pass
    return {
        "protocol_bundle_exists": all(path.is_file() for path in
                                      (PROTOCOL_PATH, _sidecar_path(PROTOCOL_PATH), _commit_path(PROTOCOL_PATH))),
        "activation_exact_v7_no_go": activation_valid,
        "claim_exists": GLOBAL_CLAIM_PATH.is_file(),
        "claim_creation_failure_exists": GLOBAL_CLAIM_FAILURE_PATH.is_file(),
        "completion": _strict_json(completion) if completion.is_file() else None,
        "development_result": _strict_json(result) if result.is_file() else None,
        "mode1_state": "SEALED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-development", "preflight", "development", "status"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-development":
        report = freeze_development()
    elif args.command == "preflight":
        report = preflight()
    elif args.command == "development":
        report = run_development(output_root=args.output_root)
    else:
        report = status(output_root=args.output_root)
    print(json.dumps(_plain_json(report), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
