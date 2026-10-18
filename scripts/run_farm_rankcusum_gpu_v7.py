#!/usr/bin/env python
"""Fail-closed governance runner for FARM-RankCUSUM v7 on RTX 5090.

The production backend is deliberately imported during a read-free,
pre-claim handshake.  The actual HDF5 source may be opened only after the
exclusive claim and exact access intent have both been committed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DOCUMENT_PATH = PROJECT_ROOT / "docs" / "farm_rankcusum_v7.md"
LEDGER_PATH = (
    PROJECT_ROOT / "protocols" / "farm_rankcusum_v7_history_and_access_ledger.json"
)
V8_CHARTER_JSON_PATH = PROJECT_ROOT / "docs" / "gpu_minirocket_logodds_v8_contingency.json"
PROTOCOL_PATH = (
    PROJECT_ROOT / "protocols" / "farm_rankcusum_v7_mode2_development.json"
)
SOURCE_PATH = PROJECT_ROOT / "data" / "extended_tep_dtu_mode2" / "TEP_Mode2.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "farm_rankcusum_v7"
DEVELOPMENT_DIRECTORY_NAME = "v7_fixed_formal"
GLOBAL_CLAIM_PATH = DEFAULT_OUTPUT_ROOT / "development_access.claim.json"

PROTOCOL_SCHEMA = "farm_rankcusum_v7_preregistered_development_protocol"
PROTOCOL_ID = "farm-rankcusum-v7-mode2-development"
CANDIDATE_ID = "farm-rankcusum-v7-fixed"
BACKEND_MODULE = "mate_spc_farm_v7.backend"
BACKEND_RESULT_SCHEMA = "farm_rankcusum_v7_development_result"

SOURCE_SHA256 = "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
SOURCE_SIZE_BYTES = 23_897_125_720
FOLD_MAP_SHA256 = "64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13"
ACCESS_PLAN_SHA256 = "74b19927fee7d31f33c3b64d62a0c5235b58ad41e5d2d7d28b583a6ff4e43466"

FOLD_RUN_IDS: dict[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}
SEED_CONNECTED_COMPONENTS = ((14, 43, 98), (6, 76), (9, 88), (24, 33))
CONDITION_IDVS = (1, 3, 9, 15)
CHANNEL_INDICES = tuple(range(41)) + (46, 47, 48, 50, 51)

BASE_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "docs/farm_rankcusum_v7.md",
    "docs/gpu_minirocket_logodds_v8_contingency.md",
    "docs/gpu_minirocket_logodds_v8_contingency.json",
    "protocols/farm_rankcusum_v7_history_and_access_ledger.json",
    "scripts/run_farm_rankcusum_gpu_v7.py",
    "src/mate_spc/hash_contract.py",
    "src/mate_spc/__init__.py",
    "src/mate_spc/model.py",
    "src/mate_spc/datasets.py",
    "src/mate_spc/missingness.py",
    "src/mate_spc/monitor.py",
    "src/mate_spc/state_space.py",
    "src/mate_spc/utils.py",
    "src/mate_spc_cpetc/__init__.py",
    "src/mate_spc_cpetc/data_v5.py",
    "src/mate_spc_cpetc/monitor.py",
    "src/mate_spc_pclr/__init__.py",
    "src/mate_spc_pclr/data.py",
    "src/mate_spc_farm_v7/__init__.py",
    "src/mate_spc_farm_v7/monitor.py",
    "src/mate_spc_farm_v7/evaluation.py",
    "src/mate_spc_farm_v7/backend.py",
)
TEST_NAME_PATTERNS = (
    "test_farm_rankcusum_v7*.py",
    "test_run_farm_rankcusum_gpu_v7.py",
)

METHOD_CONFIG: dict[str, Any] = {
    "candidate_id": CANDIDATE_ID,
    "candidate_count": 1,
    "data": {
        "mode": 2,
        "fault_regime": "SingleFault",
        "magnitude_percent": 50,
        "idvs": [1, 3, 9, 15],
        "runs_per_idv": 100,
        "run_length": 500,
        "feature_count": 52,
        "rows_per_run": 500,
        "process_channel_count": 52,
        "healthy_window_zero_based_half_open": [100, 600],
        "fault_window_zero_based_half_open": [440, 940],
        "fault_local_onset_index": 160,
        "monitor_endpoints_inclusive": [160, 499],
        "miss_delay_samples": 340,
    },
    "channels": {
        "zero_based_indices": list(CHANNEL_INDICES),
        "count": 46,
        "names": [
            *[f"XMEAS{i}" for i in range(1, 42)],
            "XMV6",
            "XMV7",
            "XMV8",
            "XMV10",
            "XMV11",
        ],
    },
    "reference": {
        "fit_population": "outer-fold fit-role healthy runs only",
        "selected_channel_count": 46,
        "fit_healthy_runs_per_fold": 60,
        "fit_reference_values_per_channel": 30000,
        "fault_runs_used": 0,
        "allowed_healthy_run_counts": [60, 100],
        "development_healthy_run_count": 60,
        "go_only_refit_healthy_run_count": 100,
        "rows_per_run": 500,
        "development_reference_shape": [46, 30000],
        "go_only_refit_reference_shape": [46, 50000],
        "sort": "stable ascending per channel",
        "dtype": "float64",
        "normalization": False,
        "fault_values_used": False,
    },
    "monitor": {
        "searchsorted_side": "left",
        "posterior_ecdf": "(searchsorted_left+1)/(N+2)",
        "two_sided": True,
        "k": 1.3,
        "top_r": 4,
        "reset_endpoint": 160,
        "dense_endpoints_inclusive": [160, 499],
        "run_score": "maximum of top4 two-sided CUSUM sum",
        "alarm_comparison": ">",
        "persistence": 1,
        "dtype": "float64",
    },
    "inner_threshold": {
        "healthy_runs": 20,
        "fault_inner_values_used": False,
        "ascending_rank": 20,
        "comparison": ">",
        "minimum_exact_distinct_maxima": 18,
        "tail_gap_multiplier": 100.0,
        "repeat_error_floor": 1e-10,
        "reachable_upper_bound": "340*4*(log(N+2)-1.3)",
    },
    "determinism": {
        "authoritative_device": "CUDA RTX 5090",
        "authoritative_repeat_scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        "authoritative_repeat_atol": 1e-6,
        "authoritative_discrete_exact": True,
        "cross_device_cpu_cuda": "report_only_not_a_go_gate",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    },
    "go_only_final_refit": {
        "trigger": "only after all five outer folds satisfy every development GO gate",
        "source": "the same already-consumed Mode2 magnitude50 pool",
        "healthy_runs": 100,
        "fault_runs_used": 0,
        "target_domain_calibration_performed": False,
        "mode1_access_authorized": False,
    },
}

_HASH_CONTRACT_MODULE_NAME = "_farm_rankcusum_v7_hash_contract"
_HASH_CONTRACT_MODULE: ModuleType | None = None

GO_GATES: dict[str, Any] = {
    "logic": "all required",
    "healthy": {
        "overall_false_alarms_max": 10,
        "overall_denominator": 100,
        "per_fold_false_alarms_max": 4,
        "per_fold_denominator": 20,
    },
    "each_fault_3_9_15": {
        "overall_detection_rate_min": 0.85,
        "overall_denominator": 100,
        "per_fold_detection_rate_min": 0.70,
        "per_fold_denominator": 20,
        "median_delay_samples_max": 140,
        "nearest_rank_p90_delay_samples_max": 220,
        "miss_delay_samples": 340,
    },
    "threshold_nondegeneracy_all_folds": True,
    "authoritative_cuda_repeat_gate": True,
    "cross_device_cpu_cuda_gate": False,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_contract_module() -> ModuleType:
    """Load the stdlib-only helper without executing ``mate_spc.__init__``.

    On Windows, importing the package would import NumPy before the production
    backend has deliberately imported torch.  Direct file loading keeps both
    the governance path and the torch-first DLL ordering explicit.
    """

    global _HASH_CONTRACT_MODULE
    if _HASH_CONTRACT_MODULE is not None:
        return _HASH_CONTRACT_MODULE
    path = SRC_ROOT / "mate_spc" / "hash_contract.py"
    spec = importlib.util.spec_from_file_location(_HASH_CONTRACT_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load hash contract from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HASH_CONTRACT_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(_HASH_CONTRACT_MODULE_NAME) is module:
            del sys.modules[_HASH_CONTRACT_MODULE_NAME]
        raise
    _HASH_CONTRACT_MODULE = module
    return module


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
    raise TypeError(f"cannot serialize {type(value).__name__} to strict JSON")


def _canonical_bytes(value: Any) -> bytes:
    helper = _hash_contract_module()
    return helper.canonical_json_bytes(_plain_json(value))


def _json_file_bytes(value: Any) -> bytes:
    helper = _hash_contract_module()
    return helper.canonical_json_file_bytes(_plain_json(value))


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    helper = _hash_contract_module()
    value = helper.strict_json_from_bytes(path.read_bytes())
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
    sidecar_payload = (
        f"{identities['file_bytes_sha256']}  {path.name}\n".encode("ascii")
    )
    _write_exclusive(sidecar, sidecar_payload)
    commit = {
        "schema": commit_schema,
        "committed_utc": _now(),
        "document_name": path.name,
        **identities,
        "file_bytes_are_canonical_json_plus_lf": True,
        "sidecar_name": sidecar.name,
        "sidecar_file_bytes_sha256": _hash_bytes(sidecar_payload),
    }
    commit_identity = _write_json(_commit_path(path), commit)
    if (
        _hash_file(path) != identities["file_bytes_sha256"]
        or _hash_file(sidecar) != commit["sidecar_file_bytes_sha256"]
        or _hash_file(_commit_path(path)) != commit_identity["file_bytes_sha256"]
    ):
        raise RuntimeError("atomic committed-JSON verification failed")
    return {
        **identities,
        "sidecar_file_bytes_sha256": commit["sidecar_file_bytes_sha256"],
        "commit_file_bytes_sha256": commit_identity["file_bytes_sha256"],
    }


def _verify_committed_json(
    path: Path, *, commit_schema: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = _hash_contract_module()
    sidecar = _sidecar_path(path)
    commit_path = _commit_path(path)
    if not path.is_file() or not sidecar.is_file() or not commit_path.is_file():
        raise FileNotFoundError(f"incomplete committed JSON bundle for {path.name}")
    document, contract = helper.require_canonical_json_file(path)
    if not isinstance(document, dict):
        raise ValueError("committed protocol must be a JSON object")
    expected_sidecar = f"{contract.file_bytes_sha256}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"SHA-256 sidecar mismatch for {path.name}")
    commit, commit_contract = helper.require_canonical_json_file(commit_path)
    if not isinstance(commit, dict):
        raise ValueError("commit record must be a JSON object")
    if (
        commit.get("schema") != commit_schema
        or commit.get("document_name") != path.name
        or commit.get("file_bytes_sha256") != contract.file_bytes_sha256
        or commit.get("canonical_json_sha256") != contract.canonical_json_sha256
        or commit.get("file_size_bytes") != contract.file_size_bytes
        or commit.get("canonical_json_size_bytes") != contract.canonical_json_size_bytes
        or commit.get("file_bytes_are_canonical_json_plus_lf") is not True
        or commit.get("sidecar_name") != sidecar.name
        or commit.get("sidecar_file_bytes_sha256") != _hash_file(sidecar)
    ):
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
    return {run_id: fold for fold, runs in FOLD_RUN_IDS.items() for run_id in runs}


def _fold_map_hash(value: Mapping[int, int] | None = None) -> str:
    fold_map = _canonical_fold_map() if value is None else dict(value)
    payload = {str(run): int(fold_map[run]) for run in sorted(fold_map)}
    return _hash_bytes(_canonical_bytes(payload))


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)):
        raise ValueError("fold IDs must be exactly 0..4")
    if any(len(runs) != 20 for runs in FOLD_RUN_IDS.values()):
        raise ValueError("each fold must contain 20 run IDs")
    flat = [run for fold in range(5) for run in FOLD_RUN_IDS[fold]]
    if sorted(flat) != list(range(1, 101)) or len(set(flat)) != 100:
        raise ValueError("folds must partition run IDs 1..100 exactly")
    fold_map = _canonical_fold_map()
    if any(len({fold_map[run] for run in component}) != 1 for component in SEED_CONNECTED_COMPONENTS):
        raise ValueError("a duplicate-seed connected component crosses folds")
    if _fold_map_hash(fold_map) != FOLD_MAP_SHA256:
        raise ValueError("canonical fold-map SHA-256 changed")


def fold_roles(outer_fold: int) -> dict[str, Any]:
    if isinstance(outer_fold, bool) or outer_fold not in range(5):
        raise ValueError("outer_fold must be an integer in 0..4")
    inner_fold = (outer_fold + 1) % 5
    fit = sorted(
        run
        for fold, runs in FOLD_RUN_IDS.items()
        if fold not in (outer_fold, inner_fold)
        for run in runs
    )
    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "fit_healthy_run_ids": fit,
        "inner_healthy_run_ids": list(FOLD_RUN_IDS[inner_fold]),
        "inner_fault_values_authorized": False,
        "outer_run_ids_per_class": list(FOLD_RUN_IDS[outer_fold]),
    }


def development_access_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for run_id in range(1, 101):
        for idv in CONDITION_IDVS:
            group = (
                f"Mode2/SingleFault/SimulationCompleted/IDV{idv}/"
                f"Mode2_IDVInfo_{idv}_50/Run{run_id}"
            )
            window = [100, 600] if idv == 1 else [440, 940]
            plan.append(
                {
                    "mode": 2,
                    "fault_regime": "SingleFault",
                    "status": "SimulationCompleted",
                    "magnitude_percent": 50,
                    "idv": idv,
                    "simulation_run": run_id,
                    "group_path": group,
                    "processdata_path": f"{group}/processdata",
                    "idv_init_path": f"{group}/idv_init",
                    "window_zero_based_half_open": window,
                    "processdata_row_zero_based_half_open": window,
                    "processdata_column_zero_based_half_open": [0, 53],
                    "idv_init_row_zero_based_half_open": [0, 3],
                    "idv_init_column_zero_based_half_open": [0, 28],
                    "group_attributes_authorized": ["seed"],
                    "dataset_metadata_authorized": ["object_type", "shape", "dtype"],
                    "root_dataset_value_reads_authorized": [],
                    "processdata_values_outside_row_window_authorized": False,
                    "role": "healthy_control" if idv == 1 else f"fault_{idv}",
                }
            )
    return plan


def _test_paths() -> tuple[str, ...]:
    found: set[str] = {"tests/test_hash_contract.py"}
    test_root = PROJECT_ROOT / "tests"
    for pattern in TEST_NAME_PATTERNS:
        for path in test_root.glob(pattern):
            found.add(path.relative_to(PROJECT_ROOT).as_posix())
    return tuple(sorted(found))


def _implementation_paths() -> tuple[str, ...]:
    return tuple(BASE_IMPLEMENTATION_PATHS) + _test_paths()


def _implementation_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in _implementation_paths():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required v7 artifact is missing: {relative}")
        records[relative] = {
            "file_bytes_sha256": _hash_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _validate_ledger() -> dict[str, Any]:
    ledger = _strict_json(LEDGER_PATH)
    mode1 = ledger.get("extended_tep_mode1_roles", {})
    mode2 = ledger.get("extended_tep_mode2_roles", {})
    source = ledger.get("source_identities", {}).get("extended_tep_mode2", {})
    if (
        ledger.get("schema") != "farm_rankcusum_v7_history_and_access_ledger"
        or ledger.get("ledger_state") != "PREACCESS_TEXT_AUDIT_FOR_V7_MODE2_EXPLORATORY_DEVELOPMENT"
        or ledger.get("audit_scope", {}).get("hdf5_files_opened_by_this_audit") != 0
        or ledger.get("audit_scope", {}).get("numeric_values_decoded_by_this_audit") != 0
        or ledger.get("v7_candidate", {}).get("candidate_id") != CANDIDATE_ID
        or ledger.get("v7_candidate", {}).get("candidate_count") != 1
        or source.get("sha256") != SOURCE_SHA256
        or source.get("size_bytes") != SOURCE_SIZE_BYTES
        or mode2.get("magnitude50_idv1_3_9_15_runs1_100", {}).get("access_level_before_v7") != "SELECTED_OR_SCORED"
        or mode1.get("magnitude25_idv1_runs1_100", {}).get("access_level") != "NEVER_NUMERIC_ACCESSED"
        or mode1.get("magnitude75_idv1_3_9_15_runs1_100", {}).get("access_level") != "NEVER_NUMERIC_ACCESSED"
        or ledger.get("v7_development_authorization", {}).get("mode1_access_authorized") is not False
    ):
        raise ValueError("v7 history/access ledger semantic binding failed")
    return ledger


def _validate_contingency_charter() -> dict[str, Any]:
    charter = _strict_json(V8_CHARTER_JSON_PATH)
    activation = charter.get("activation", {})
    authorization = charter.get("authorization", {})
    folds = charter.get("fold_assignment", {})
    expected_fold_ids = {
        str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()
    }
    expected_run_to_fold = {
        str(run): fold for run, fold in sorted(_canonical_fold_map().items())
    }
    expected_outer_roles = [
        {
            "outer_fold": outer,
            "inner_fold": (outer + 1) % 5,
            "fit_folds": [
                fold for fold in range(5) if fold not in (outer, (outer + 1) % 5)
            ],
        }
        for outer in range(5)
    ]
    if (
        charter.get("candidate_id") != "gpu-minirocket-logodds-cusum-v8-fixed"
        or charter.get("created_before_formal_v7_data_access") is not True
        or activation.get("allowed_trigger")
        != {"candidate_id": CANDIDATE_ID, "result": "DEVELOPMENT_NO_GO"}
        or activation.get("requires_separate_pre_access_protocol") is not True
        or authorization.get("document_type") != "contingent_method_charter"
        or any(
            authorization.get(field) is not False
            for field in (
                "allows_checkpoint_creation",
                "allows_code_implementation",
                "allows_h5_access",
                "allows_numeric_claims",
                "allows_real_experiment",
            )
        )
        or folds.get("fold_run_ids") != expected_fold_ids
        or folds.get("run_to_fold") != expected_run_to_fold
        or folds.get("run_to_fold_sha256") != FOLD_MAP_SHA256
        or folds.get("outer_roles") != expected_outer_roles
        or folds.get("inner_fault_values_authorized") is not False
    ):
        raise ValueError("pre-result v8 contingency charter semantic binding failed")
    return charter


def _protocol_document(implementation: Mapping[str, Any]) -> dict[str, Any]:
    _validate_folds()
    _validate_ledger()
    _validate_contingency_charter()
    plan_hash = _hash_bytes(_canonical_bytes(development_access_plan()))
    if plan_hash != ACCESS_PLAN_SHA256:
        raise ValueError("exact-selector access-plan SHA-256 changed")
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_state": "FROZEN_AFTER_IMPLEMENTATION_BEFORE_V7_HDF5_OPEN",
        "created_utc": _now(),
        "candidate": METHOD_CONFIG,
        "candidate_count": 1,
        "source": {
            "relative_path": "data/extended_tep_dtu_mode2/TEP_Mode2.h5",
            "filename": "TEP_Mode2.h5",
            "mode": 2,
            "fault_regime": "SingleFault",
            "magnitude_percent": 50,
            "idvs": [1, 3, 9, 15],
            "sha256": SOURCE_SHA256,
            "size_bytes": SOURCE_SIZE_BYTES,
            "hdf5_opened_during_freeze": False,
            "mode1_access_authorized": False,
            "rieth_access_authorized": False,
        },
        "folds": {
            "source": "collision-safe v5.1 assignment retained unchanged",
            "fold_run_ids": {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()},
            "run_to_fold": {
                str(run): fold for run, fold in sorted(_canonical_fold_map().items())
            },
            "run_to_fold_canonical_json_sha256": FOLD_MAP_SHA256,
            "cross_fold_seed_collision_count": 0,
            "outer_roles": [fold_roles(fold) for fold in range(5)],
        },
        "access_plan": {
            "record_count": 400,
            "selector_count": 400,
            "canonical_json_sha256": ACCESS_PLAN_SHA256,
            "exact_selector_fields": list(development_access_plan()[0]),
            "wildcards_allowed": False,
        },
        "development_go_gates": GO_GATES,
        "governance": {
            "candidate_count": 1,
            "global_development_claim_count": 1,
            "rescue_iteration_ids_allowed": False,
            "same_candidate_rerun_after_claim": False,
            "preclaim_production_backend_handshake_required": True,
            "preclaim_handshake_validates_persisted_protocol": True,
            "preclaim_h5py_import_blocker_required": True,
            "handshaken_backend_instance_reused_after_intent": True,
            "preclaim_hdf5_import_or_open_allowed": False,
            "mode1_access_authorized": False,
            "rieth_access_authorized": False,
            "mode1_magnitude25_state": "SEALED",
            "mode1_magnitude75_state": "SEALED",
            "inner_fault_numeric_access_authorized": False,
            "fault_fit_or_threshold_authorized": False,
            "development_decisions": ["DEVELOPMENT_GO", "DEVELOPMENT_NO_GO"],
        },
        "hash_contract": {
            "canonicalization": "UTF-8 sorted-key compact JSON, no NaN, no terminal LF",
            "persisted_file": "canonical JSON plus exactly one terminal LF",
            "sidecar_and_commit_identity": "file_bytes_sha256",
            "runner_backend_identity": "canonical_json_sha256",
            "cross_domain_comparison_forbidden": True,
        },
        "reporting": {
            "development_status": "adaptive exploratory feasibility only",
            "mode1_future_status": "prospectively held-out target-domain severity cell, not wholly unseen dataset-level zero-shot external validation",
            "pooled_fault_binomial_n300_forbidden": True,
        },
        "bindings": {
            "method_document": {
                "relative_path": "docs/farm_rankcusum_v7.md",
                "file_bytes_sha256": _hash_file(DOCUMENT_PATH),
            },
            "history_and_access_ledger": {
                "relative_path": "protocols/farm_rankcusum_v7_history_and_access_ledger.json",
                "file_bytes_sha256": _hash_file(LEDGER_PATH),
            },
            "implementation": _plain_json(implementation),
        },
        "numeric_backend_contract": {
            "module": BACKEND_MODULE,
            "required_functions": ["cuda_preflight", "validate_protocol", "run_development"],
            "backend_import_and_validation_before_claim": True,
            "same_backend_module_object_used_after_intent": True,
            "validation_hash_domain": "canonical_json_without_file_terminator",
            "validation_must_not_compare_file_bytes_sha256": True,
            "hdf5_import_or_open_before_intent": False,
            "result_schema": BACKEND_RESULT_SCHEMA,
            "protocol_identity_field": "protocol_canonical_json_sha256",
            "cuda_repeat_atol": 1e-6,
            "cpu_drift_is_go_gate": False,
        },
    }


def freeze_development() -> dict[str, Any]:
    implementation = _implementation_records()
    existing = (PROTOCOL_PATH, _sidecar_path(PROTOCOL_PATH), _commit_path(PROTOCOL_PATH))
    if any(path.exists() for path in existing):
        if not all(path.is_file() for path in existing):
            raise FileExistsError("partial v7 protocol bundle exists; refuse repair or overwrite")
        document, identities = _verify_protocol_bundle()
        return {
            "status": "VERIFIED_EXISTING_FROZEN_PROTOCOL",
            "protocol_id": document["protocol_id"],
            **identities,
        }
    if GLOBAL_CLAIM_PATH.exists():
        raise RuntimeError("v7 global development claim already exists; refuse late freeze")
    document = _protocol_document(implementation)
    identities = _commit_json(
        PROTOCOL_PATH,
        document,
        commit_schema="farm_rankcusum_v7_protocol_atomic_commit",
    )
    return {"status": "FROZEN", "protocol_id": PROTOCOL_ID, **identities}


def _verify_protocol_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    document, identities = _verify_committed_json(
        PROTOCOL_PATH,
        commit_schema="farm_rankcusum_v7_protocol_atomic_commit",
    )
    expected_hash_contract = {
        "canonicalization": "UTF-8 sorted-key compact JSON, no NaN, no terminal LF",
        "persisted_file": "canonical JSON plus exactly one terminal LF",
        "sidecar_and_commit_identity": "file_bytes_sha256",
        "runner_backend_identity": "canonical_json_sha256",
        "cross_domain_comparison_forbidden": True,
    }
    if (
        document.get("schema") != PROTOCOL_SCHEMA
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("candidate") != METHOD_CONFIG
        or document.get("candidate_count") != 1
        or document.get("development_go_gates") != GO_GATES
        or document.get("source", {}).get("sha256") != SOURCE_SHA256
        or document.get("source", {}).get("size_bytes") != SOURCE_SIZE_BYTES
        or document.get("source", {}).get("mode1_access_authorized") is not False
        or document.get("access_plan", {}).get("record_count") != 400
        or document.get("access_plan", {}).get("canonical_json_sha256") != ACCESS_PLAN_SHA256
        or _hash_bytes(_canonical_bytes(development_access_plan())) != ACCESS_PLAN_SHA256
        or document.get("hash_contract") != expected_hash_contract
    ):
        raise ValueError("v7 protocol semantic binding failed")
    _validate_folds()
    folds = document.get("folds", {})
    if (
        folds.get("fold_run_ids") != {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()}
        or folds.get("run_to_fold") != {
            str(run): fold for run, fold in sorted(_canonical_fold_map().items())
        }
        or folds.get("run_to_fold_canonical_json_sha256") != FOLD_MAP_SHA256
        or folds.get("cross_fold_seed_collision_count") != 0
    ):
        raise ValueError("v7 protocol fold binding failed")
    _validate_ledger()
    bindings = document.get("bindings", {})
    if (
        bindings.get("method_document", {}).get("file_bytes_sha256") != _hash_file(DOCUMENT_PATH)
        or bindings.get("history_and_access_ledger", {}).get("file_bytes_sha256") != _hash_file(LEDGER_PATH)
        or bindings.get("implementation") != _implementation_records()
    ):
        raise ValueError("v7 frozen implementation binding failed")
    return document, identities


def _source_record() -> dict[str, Any]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    if SOURCE_PATH.name != "TEP_Mode2.h5" or SOURCE_PATH.stat().st_size != SOURCE_SIZE_BYTES:
        raise ValueError("official Mode2 source path, name, or size changed")
    observed = _hash_file(SOURCE_PATH)
    if observed != SOURCE_SHA256:
        raise ValueError("official Mode2 source SHA-256 mismatch")
    return {
        "relative_path": "data/extended_tep_dtu_mode2/TEP_Mode2.h5",
        "filename": SOURCE_PATH.name,
        "size_bytes": SOURCE_SIZE_BYTES,
        "sha256": observed,
        "hash_method": "raw streaming SHA256 in 16MiB chunks",
        "hdf5_library_imported": False,
        "hdf5_opened": False,
        "numeric_values_decoded": False,
    }


def _gpu_presence_preflight() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not any("5090" in row for row in rows):
        raise RuntimeError("RTX 5090 GPU presence preflight failed")
    return {"command": command, "returncode": completed.returncode, "rows": rows, "rtx5090_present": True}


def _run_test_suite() -> dict[str, Any]:
    test_paths = _test_paths()
    if len(test_paths) < 2:
        raise RuntimeError("v7 test suite was not discovered")
    command = [sys.executable, "-m", "pytest", *test_paths, "-q", "-W", "error"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"v7 synthetic/CUDA/governance tests failed:\n{combined[-4000:]}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_sha256": _hash_bytes(combined.encode("utf-8")),
        "output_tail": combined[-2000:],
        "hdf5_numeric_access_authorized": False,
    }


def _prepare_backend_environment() -> None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif observed != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from frozen :4096:8")


def _load_backend() -> ModuleType:
    module = importlib.import_module(BACKEND_MODULE)
    for name in ("cuda_preflight", "validate_protocol", "run_development"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"{BACKEND_MODULE}.{name} is required")
    return module


def _validate_backend_cuda(value: Any) -> dict[str, Any]:
    report = _plain_json(value)
    if not isinstance(report, dict):
        raise ValueError("backend CUDA preflight must return a mapping")
    if report.get("passed") is not True or report.get("cuda_available") is not True:
        raise RuntimeError("backend CUDA preflight did not pass")
    if "5090" not in json.dumps(report, sort_keys=True).lower():
        raise RuntimeError("backend CUDA preflight did not bind RTX 5090")
    return report


def _loaded_h5py_modules() -> tuple[str, ...]:
    return tuple(
        sorted(name for name in sys.modules if name == "h5py" or name.startswith("h5py."))
    )


class _H5pyImportBlocker:
    """Fail closed on any attempted h5py import during the read-free handshake."""

    def __init__(self) -> None:
        self.attempted = False

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> None:
        del path, target
        if fullname == "h5py" or fullname.startswith("h5py."):
            self.attempted = True
            raise RuntimeError(
                "h5py import attempted during the pre-claim backend handshake"
            )
        return None


def _production_handshake(
    protocol: Mapping[str, Any], *, canonical_json_sha256: str
) -> tuple[ModuleType, dict[str, Any]]:
    loaded_before = _loaded_h5py_modules()
    if loaded_before:
        raise RuntimeError(
            "h5py was imported before the pre-claim backend handshake: "
            + ", ".join(loaded_before)
        )
    blocker = _H5pyImportBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        _prepare_backend_environment()
        backend = _load_backend()
        validation = _plain_json(
            backend.validate_protocol(
                protocol,
                expected_canonical_json_sha256=canonical_json_sha256,
            )
        )
        if (
            not isinstance(validation, dict)
            or validation.get("valid") is not True
            or validation.get("schema") != PROTOCOL_SCHEMA
            or validation.get("protocol_id") != PROTOCOL_ID
            or validation.get("candidate_id") != CANDIDATE_ID
            or validation.get("candidate_count") != 1
            or validation.get("protocol_canonical_json_sha256")
            != canonical_json_sha256
            or validation.get("fold_map_canonical_json_sha256") != FOLD_MAP_SHA256
            or validation.get("access_plan_canonical_json_sha256")
            != ACCESS_PLAN_SHA256
            or validation.get("hash_domain")
            != "canonical_json_without_file_terminator"
            or validation.get("file_bytes_sha256_compared") is not False
        ):
            raise ValueError("backend rejected the persisted v7 protocol")
        cuda = _validate_backend_cuda(backend.cuda_preflight())
        loaded_after = _loaded_h5py_modules()
        if blocker.attempted or loaded_after:
            raise RuntimeError(
                "backend validation/CUDA preflight attempted to load h5py before intent"
            )
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
    return backend, {
        "validation": validation,
        "cuda": cuda,
        "h5py_import_blocker_installed": True,
        "h5py_import_attempted": False,
        "h5py_module_loaded": False,
        "hdf5_opened": False,
        "numeric_values_decoded": False,
    }


def _preclaim_audit(
    protocol: Mapping[str, Any], identities: Mapping[str, Any]
) -> tuple[ModuleType, dict[str, Any]]:
    tests = _run_test_suite()
    source = _source_record()
    gpu = _gpu_presence_preflight()
    backend, handshake = _production_handshake(
        protocol,
        canonical_json_sha256=identities["canonical_json_sha256"],
    )
    return backend, {
        "schema": "farm_rankcusum_v7_preclaim_production_preflight",
        "schema_version": 1,
        "completed_utc": _now(),
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
        "protocol_commit_file_bytes_sha256": identities["commit_file_bytes_sha256"],
        "candidate_id": CANDIDATE_ID,
        "source": source,
        "gpu_presence": gpu,
        "tests": tests,
        "production_backend_handshake": handshake,
        "claim_written": False,
        "access_intent_written": False,
        "hdf5_opened": False,
        "mode1_access_authorized": False,
    }


def preflight() -> dict[str, Any]:
    protocol, identities = _verify_protocol_bundle()
    _backend, audit = _preclaim_audit(protocol, identities)
    return audit


def _is_exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_file_record(value: Any, *, filename: str) -> None:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not Path(value["path"]).is_absolute()
        or Path(value["path"]).name != filename
        or not _is_lower_sha256(value.get("file_bytes_sha256"))
        or not isinstance(value.get("size_bytes"), int)
        or isinstance(value.get("size_bytes"), bool)
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"invalid backend artifact record for {filename}")


def _validate_backend_access_state_report(
    value: Any, *, protocol_canonical_json_sha256: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError("backend access-state report is missing")
    state = value.get("final_state")
    if (
        not isinstance(state, dict)
        or state.get("schema") != "farm_rankcusum_v7_backend_access_state"
        or state.get("protocol_canonical_json_sha256")
        != protocol_canonical_json_sha256
        or state.get("hdf5_open_started") is not True
        or state.get("hdf5_open_completed") is not True
        or not _is_exact_int(state.get("processdata_slices_read"), 400)
        or not _is_exact_int(state.get("idv_init_slices_read"), 400)
        or not _is_exact_int(state.get("numeric_slices_read"), 800)
        or state.get("training_or_fit_started") is not True
        or state.get("backend_completed") is not True
        or not _is_exact_int(state.get("mode1_opens"), 0)
    ):
        raise ValueError("backend access-state final state is invalid")
    _validate_file_record(value.get("artifact"), filename="backend_access_state.json")


def _validate_development_result(
    value: Any, *, protocol_canonical_json_sha256: str
) -> dict[str, Any]:
    result = _plain_json(value)
    if not isinstance(result, dict):
        raise ValueError("backend development result must be a mapping")
    decision = result.get("decision")
    if (
        result.get("schema") != BACKEND_RESULT_SCHEMA
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("protocol_canonical_json_sha256") != protocol_canonical_json_sha256
        or result.get("candidate_id") != CANDIDATE_ID
        or not _is_exact_int(result.get("candidate_count"), 1)
        or result.get("source_file_bytes_sha256") != SOURCE_SHA256
        or not _is_exact_int(result.get("source_size_bytes"), SOURCE_SIZE_BYTES)
        or result.get("fold_map_canonical_json_sha256") != FOLD_MAP_SHA256
        or result.get("access_plan_canonical_json_sha256") != ACCESS_PLAN_SHA256
        or result.get("mode1_accessed") is not False
        or not _is_exact_int(result.get("mode1_opens"), 0)
        or result.get("rieth_accessed") is not False
        or not _is_exact_int(result.get("rieth_opens"), 0)
        or not _is_exact_int(result.get("fault_fit_run_count"), 0)
        or not _is_exact_int(result.get("fault_threshold_run_count"), 0)
        or decision not in {"DEVELOPMENT_GO", "DEVELOPMENT_NO_GO"}
    ):
        raise ValueError("backend development identity/result binding is invalid")
    topology = result.get("topology")
    numeric_access = topology.get("numeric_value_access") if isinstance(topology, dict) else None
    if (
        not isinstance(topology, dict)
        or not _is_exact_int(topology.get("selected_run_count"), 400)
        or topology.get("counts_by_status") != {"SimulationCompleted": 400}
        or topology.get("counts_by_idv") != {"1": 100, "3": 100, "9": 100, "15": 100}
        or not _is_exact_int(topology.get("unique_seed_count"), 395)
        or not _is_exact_int(topology.get("duplicate_seed_value_count"), 5)
        or not _is_exact_int(topology.get("cross_fold_seed_collision_count"), 0)
        or not isinstance(numeric_access, dict)
        or not _is_exact_int(numeric_access.get("processdata_slices"), 400)
        or not _is_exact_int(numeric_access.get("processdata_rows_per_slice"), 500)
        or numeric_access.get("processdata_columns_half_open") != [0, 53]
        or not _is_exact_int(numeric_access.get("idv_init_slices"), 400)
        or not _is_exact_int(
            numeric_access.get("processdata_values_outside_authorized_windows"), 0
        )
        or not _is_exact_int(numeric_access.get("root_dataset_value_reads"), 0)
    ):
        raise ValueError("backend exact-selector topology/access report is invalid")
    _validate_backend_access_state_report(
        result.get("backend_access_state"),
        protocol_canonical_json_sha256=protocol_canonical_json_sha256,
    )
    environment = result.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("passed") is not True
        or environment.get("cuda_available") is not True
        or "5090" not in json.dumps(environment, sort_keys=True).lower()
    ):
        raise ValueError("backend result is not bound to RTX 5090 CUDA")
    folds = result.get("folds")
    if (
        not isinstance(folds, list)
        or len(folds) != 5
        or any(not isinstance(fold, dict) for fold in folds)
        or any(
            not isinstance(fold.get("outer_fold"), int)
            or isinstance(fold.get("outer_fold"), bool)
            for fold in folds
        )
        or {fold["outer_fold"] for fold in folds} != set(range(5))
    ):
        raise ValueError("backend must return exactly five identified folds")
    repeat = result.get("authoritative_cuda_repeat")
    if (
        not isinstance(repeat, dict)
        or not isinstance(repeat.get("passes"), bool)
        or not isinstance(repeat.get("discrete_equal"), bool)
        or not isinstance(repeat.get("atol"), (int, float))
        or isinstance(repeat.get("atol"), bool)
        or repeat.get("atol") != 1e-6
        or repeat.get("scope")
        != "per_fold_inner_healthy_complete_cuda_score_and_decision_replay"
        or (repeat["passes"] and not repeat["discrete_equal"])
    ):
        raise ValueError("authoritative CUDA repeat report is invalid")
    repeat_error = repeat.get("maximum_absolute_error")
    if repeat["passes"] and (
        not isinstance(repeat_error, (int, float))
        or isinstance(repeat_error, bool)
        or not math.isfinite(float(repeat_error))
        or float(repeat_error) > 1e-6
    ):
        raise ValueError("authoritative CUDA repeat error exceeds its gate")
    if repeat_error is not None and (
        not isinstance(repeat_error, (int, float))
        or isinstance(repeat_error, bool)
        or not math.isfinite(float(repeat_error))
        or float(repeat_error) < 0.0
    ):
        raise ValueError("authoritative CUDA repeat error is invalid")
    cross_device = result.get("cross_device_report_only")
    if not isinstance(cross_device, dict) or cross_device.get("is_go_gate") is not False:
        raise ValueError("CPU/CUDA comparison must be present and report-only")
    gates = result.get("gates")
    if (
        not isinstance(gates, dict)
        or not isinstance(gates.get("all_pass"), bool)
        or not isinstance(gates.get("checks"), dict)
        or gates.get("all_pass") != (decision == "DEVELOPMENT_GO")
        or result.get("aggregate") != gates.get("aggregate_metrics")
    ):
        raise ValueError("development gates and decision disagree")
    if decision == "DEVELOPMENT_GO" and not (
        repeat["passes"] and repeat["discrete_equal"]
    ):
        raise ValueError("DEVELOPMENT_GO requires authoritative repeat pass")
    selected = result.get("selected_candidate_id")
    if selected != (CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None):
        raise ValueError("selected candidate disagrees with decision")
    _validate_file_record(
        result.get("development_decision_before_final_refit"),
        filename="development_decision_before_final_refit.json",
    )
    final_refit = result.get("final_refit")
    if decision == "DEVELOPMENT_NO_GO" and final_refit is not None:
        raise ValueError("NO_GO must not create a final refit")
    if decision == "DEVELOPMENT_GO":
        if (
            not isinstance(final_refit, dict)
            or final_refit.get("schema") != "farm_rankcusum_v7_final_mode2_refit"
            or final_refit.get("protocol_canonical_json_sha256")
            != protocol_canonical_json_sha256
            or final_refit.get("candidate_id") != CANDIDATE_ID
            or not _is_exact_int(final_refit.get("fit_healthy_run_count"), 100)
            or not _is_exact_int(final_refit.get("fit_fault_run_count"), 0)
            or final_refit.get("reference_shape") != [46, 50000]
            or final_refit.get("created_only_after_development_go") is not True
            or final_refit.get("target_domain_threshold_not_fitted") is not True
        ):
            raise ValueError("GO-only final reference metadata is invalid")
        _validate_file_record(
            final_refit.get("reference"), filename="healthy_reference.npz"
        )
        _validate_file_record(final_refit.get("manifest"), filename="manifest.json")
    return result


def _read_backend_access_state(directory: Path) -> dict[str, Any] | None:
    path = directory / "artifacts" / "backend_access_state.json"
    if not path.is_file():
        return None
    try:
        state = _strict_json(path)
    except Exception as error:
        return {"state_file_invalid": True, "error_type": type(error).__name__}
    return state


def _failure_state(
    *, claim_written: bool, intent_written: bool, backend_state: Mapping[str, Any] | None
) -> str:
    if not claim_written:
        return "INFRASTRUCTURE_FAILURE_BEFORE_CLAIM"
    if not intent_written:
        return "ABORTED_AFTER_CLAIM_BEFORE_ACCESS_INTENT"
    if backend_state is None or backend_state.get("state_file_invalid") is True:
        return "ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN"
    boolean_fields = (
        "hdf5_open_started",
        "hdf5_open_completed",
        "training_or_fit_started",
        "backend_completed",
    )
    counter_fields = (
        "processdata_slices_read",
        "idv_init_slices_read",
        "numeric_slices_read",
        "mode1_opens",
    )
    if (
        backend_state.get("schema") != "farm_rankcusum_v7_backend_access_state"
        or not _is_lower_sha256(
            backend_state.get("protocol_canonical_json_sha256")
        )
        or any(not isinstance(backend_state.get(key), bool) for key in boolean_fields)
        or any(
            not isinstance(backend_state.get(key), int)
            or isinstance(backend_state.get(key), bool)
            or backend_state[key] < 0
            for key in counter_fields
        )
        or backend_state["processdata_slices_read"] > 400
        or backend_state["idv_init_slices_read"] > 400
        or backend_state["numeric_slices_read"]
        != backend_state["processdata_slices_read"]
        + backend_state["idv_init_slices_read"]
        or backend_state["mode1_opens"] != 0
        or (
            backend_state["hdf5_open_completed"]
            and not backend_state["hdf5_open_started"]
        )
        or (
            backend_state["numeric_slices_read"] > 0
            and not backend_state["hdf5_open_completed"]
        )
        or (
            backend_state["training_or_fit_started"]
            and backend_state["numeric_slices_read"] != 800
        )
        or (
            backend_state["backend_completed"]
            and not backend_state["training_or_fit_started"]
        )
    ):
        return "ABORTED_DURING_NUMERIC_BACKEND_ACCESS_STATE_UNCERTAIN"
    if backend_state["numeric_slices_read"] > 0:
        return "ABORTED_AFTER_NUMERIC_READ"
    if backend_state["hdf5_open_completed"]:
        return "ABORTED_AFTER_HDF5_OPEN_BEFORE_NUMERIC_SLICE"
    if backend_state["hdf5_open_started"]:
        return "ABORTED_DURING_HDF5_OPEN_BEFORE_COMPLETION"
    return "ABORTED_AFTER_INTENT_BEFORE_HDF5_OPEN"


def _write_failure_and_completion(
    *,
    directory: Path,
    protocol_identities: Mapping[str, Any],
    stage: str,
    error: BaseException,
    claim_written: bool,
    intent_written: bool,
    preflight_file_bytes_sha256: str | None,
    access_intent_file_bytes_sha256: str | None,
) -> dict[str, Any]:
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    backend_state = _read_backend_access_state(directory)
    decision = _failure_state(
        claim_written=claim_written,
        intent_written=intent_written,
        backend_state=backend_state,
    )
    failure = {
        "schema": "farm_rankcusum_v7_development_failure",
        "failed_utc": _now(),
        "decision": decision,
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback_sha256": _hash_bytes(trace.encode("utf-8")),
        "claim_written": claim_written,
        "access_intent_written": intent_written,
        "backend_access_state": backend_state,
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
    }
    failure_identity = _write_json(directory / "failure.json", failure)
    completion = {
        "schema": "farm_rankcusum_v7_development_completion",
        "completed_utc": _now(),
        "decision": decision,
        "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": protocol_identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": protocol_identities["canonical_json_sha256"],
        "failure_file_bytes_sha256": failure_identity["file_bytes_sha256"],
        "preflight_file_bytes_sha256": preflight_file_bytes_sha256,
        "access_intent_file_bytes_sha256": access_intent_file_bytes_sha256,
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
    }
    completion_identity = _write_json(directory / "completion.json", completion)
    return {**completion, "completion_file_bytes_sha256": completion_identity["file_bytes_sha256"]}


def run_development(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol, identities = _verify_protocol_bundle()
    plan = development_access_plan()
    plan_sha256 = _hash_bytes(_canonical_bytes(plan))
    if plan_sha256 != ACCESS_PLAN_SHA256:
        raise ValueError("live v7 access plan differs from the frozen plan")

    # Retain the exact production backend object that validates the persisted
    # protocol.  The same object crosses the claim/intent boundary; no second
    # import or substitute implementation is accepted for numeric execution.
    backend, audit = _preclaim_audit(protocol, identities)

    root = Path(output_root)
    directory = root / DEVELOPMENT_DIRECTORY_NAME
    claim_path = GLOBAL_CLAIM_PATH
    if claim_path.exists() or directory.exists():
        raise FileExistsError("a prior v7 global claim or fixed directory already exists")
    root.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "farm_rankcusum_v7_exclusive_global_development_claim",
        "claimed_utc": _now(),
        "protocol_id": PROTOCOL_ID,
        "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
        "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "access_plan_canonical_json_sha256": plan_sha256,
        "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
        "artifact_root": str(directory.resolve()),
        "fixed_directory": DEVELOPMENT_DIRECTORY_NAME,
        "rescue_iteration_ids_allowed": False,
        "policy": "Any prior claim blocks every rerun, including renamed rescue iterations.",
    }
    _write_json(claim_path, claim)
    directory.mkdir(parents=False, exist_ok=False)

    stage = "persist_preclaim_preflight"
    claim_written = True
    intent_written = False
    preflight_sha256: str | None = None
    intent_sha256: str | None = None
    try:
        preflight_identity = _write_json(directory / "preflight.json", audit)
        preflight_sha256 = preflight_identity["file_bytes_sha256"]
        stage = "access_intent_commit"
        intent = {
            "schema": "farm_rankcusum_v7_mode2_development_access_intent",
            "schema_version": 1,
            "committed_utc": _now(),
            "committed_after_read_free_backend_handshake": True,
            "committed_before_hdf5_import_or_open": True,
            "protocol_id": PROTOCOL_ID,
            "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
            "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
            "candidate_id": CANDIDATE_ID,
            "candidate_count": 1,
            "source": audit["source"],
            "access_plan": plan,
            "access_plan_canonical_json_sha256": plan_sha256,
            "fold_run_ids": {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()},
            "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
            "authorized_reads_after_commit": "only the exact 400 Mode2 magnitude50 selectors",
            "wildcards_authorized": False,
            "inner_fault_values_authorized": False,
            "mode1_access_authorized": False,
            "mode1_hdf5_opens_or_value_reads": 0,
            "mode1_magnitude25_state": "SEALED",
            "mode1_magnitude75_state": "SEALED",
            "rieth_access_authorized": False,
            "backend_module": BACKEND_MODULE,
            "backend_imported_before_commit": True,
            "production_backend_instance_reused_after_commit": True,
            "h5py_module_loaded_before_commit": False,
            "h5py_import_attempted_before_commit": False,
            "preflight_file_bytes_sha256": preflight_sha256,
            "implementation": protocol["bindings"]["implementation"],
        }
        intent_identities = _commit_json(
            directory / "access_intent.json",
            intent,
            commit_schema="farm_rankcusum_v7_access_intent_atomic_commit",
        )
        intent_sha256 = intent_identities["file_bytes_sha256"]
        intent_written = True

        stage = "numeric_backend_development"
        raw_result = backend.run_development(
            access_plan=plan,
            fold_map=_canonical_fold_map(),
            artifact_root=directory / "artifacts",
            protocol_canonical_json_sha256=identities["canonical_json_sha256"],
            device="cuda",
        )
        result = _validate_development_result(
            raw_result,
            protocol_canonical_json_sha256=identities["canonical_json_sha256"],
        )
        result_identity = _write_json(directory / "development_result.json", result)
        completion = {
            "schema": "farm_rankcusum_v7_development_completion",
            "completed_utc": _now(),
            "decision": result["decision"],
            "protocol_id": PROTOCOL_ID,
            "protocol_file_bytes_sha256": identities["file_bytes_sha256"],
            "protocol_canonical_json_sha256": identities["canonical_json_sha256"],
            "preflight_file_bytes_sha256": preflight_sha256,
            "access_intent_file_bytes_sha256": intent_sha256,
            "access_intent_commit_file_bytes_sha256": intent_identities["commit_file_bytes_sha256"],
            "development_result_file_bytes_sha256": result_identity["file_bytes_sha256"],
            "source_file_bytes_sha256": SOURCE_SHA256,
            "fold_map_canonical_json_sha256": FOLD_MAP_SHA256,
            "access_plan_canonical_json_sha256": ACCESS_PLAN_SHA256,
            "candidate_count": 1,
            "mode1_accessed": False,
            "mode1_hdf5_opens_or_value_reads": 0,
            "rieth_accessed": False,
        }
        completion_identity = _write_json(directory / "completion.json", completion)
        return {**completion, "completion_file_bytes_sha256": completion_identity["file_bytes_sha256"]}
    except BaseException as error:
        _write_failure_and_completion(
            directory=directory,
            protocol_identities=identities,
            stage=stage,
            error=error,
            claim_written=claim_written,
            intent_written=intent_written,
            preflight_file_bytes_sha256=preflight_sha256,
            access_intent_file_bytes_sha256=intent_sha256,
        )
        raise


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root)
    directory = root / DEVELOPMENT_DIRECTORY_NAME
    completion = directory / "completion.json"
    result = directory / "development_result.json"
    return {
        "protocol_bundle_exists": all(
            path.is_file()
            for path in (PROTOCOL_PATH, _sidecar_path(PROTOCOL_PATH), _commit_path(PROTOCOL_PATH))
        ),
        "claim_exists": GLOBAL_CLAIM_PATH.is_file(),
        "completion": _strict_json(completion) if completion.is_file() else None,
        "development_result": _strict_json(result) if result.is_file() else None,
        "mode1_state": "SEALED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze-development", "preflight", "development", "status"),
    )
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
