#!/usr/bin/env python
"""Fail-closed governance shell for PCF-CycleRank-Scan v6.

This module intentionally imports only the Python standard library.  It does
not import h5py, NumPy, PyTorch, or the numeric backend until the development
access intent has been durably committed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
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
DOCUMENT_PATH = PROJECT_ROOT / "docs" / "pcf_cyclerank_v6.md"
LEDGER_PATH = PROJECT_ROOT / "protocols" / "v6_history_and_access_ledger.json"
PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "pcf_cyclerank_v6_mode2_development.json"
SOURCE_PATH = PROJECT_ROOT / "data" / "extended_tep_dtu_mode2" / "TEP_Mode2.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pcf_cyclerank_v6"
DEVELOPMENT_DIRECTORY_NAME = "v6_fixed_formal"

PROTOCOL_ID = "pcf-cyclerank-scan-v6-mode2-development"
PROTOCOL_SCHEMA = "pcf_cyclerank_v6_preregistered_development_protocol"
CANDIDATE_ID = "pcf-cyclerank-scan-v6-fixed"
BACKEND_MODULE = "mate_spc_pcfcycle.backend_v6"
BACKEND_RESULT_SCHEMA = "pcf_cyclerank_v6_development_result"

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

REQUIRED_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "docs/pcf_cyclerank_v6.md",
    "protocols/v6_history_and_access_ledger.json",
    "scripts/run_pcf_cyclerank_gpu_v6.py",
    "src/mate_spc_pcfcycle/__init__.py",
    "src/mate_spc_pcfcycle/backend_v6.py",
    "src/mate_spc_pcfcycle/features.py",
    "src/mate_spc_pcfcycle/model.py",
    "src/mate_spc_pcfcycle/loss.py",
    "src/mate_spc_pcfcycle/training.py",
    "src/mate_spc_pcfcycle/scan.py",
    "src/mate_spc_pcfcycle/evaluation.py",
    "tests/test_pcfcycle_backend_v6.py",
    "tests/test_pcfcycle_features.py",
    "tests/test_pcfcycle_model_v6.py",
    "tests/test_pcfcycle_loss_v6.py",
    "tests/test_pcfcycle_training_v6.py",
    "tests/test_pcfcycle_scan.py",
    "tests/test_pcfcycle_evaluation.py",
    "tests/test_run_pcf_cyclerank_gpu_v6.py",
)

TEST_PATHS = tuple(path for path in REQUIRED_IMPLEMENTATION_PATHS if path.startswith("tests/"))

METHOD_CONFIG: dict[str, Any] = {
    "candidate_id": CANDIDATE_ID,
    "candidate_count": 1,
    "seed": 20260910,
    "network_dtype": "float32",
    "tail_and_scan_dtype": "float64_cpu_after_authoritative_cuda_logits",
    "data": {
        "mode": 2,
        "fault_regime": "SingleFault",
        "magnitude_percent": 50,
        "idvs": [1, 3, 9, 15],
        "runs_per_idv": 100,
        "healthy_window_zero_based_half_open": [100, 600],
        "fault_window_zero_based_half_open": [440, 940],
        "fault_local_onset_index": 160,
        "miss_delay_samples": 340,
    },
    "endpoints": {
        "window_length": 128,
        "inclusive_window": "x[t-127:t+1]",
        "training_all": [128, 496, 4],
        "training_prefix": [128, 156, 4],
        "training_monitor": [160, 496, 4],
        "ignored_fault_transition": [160, 188, 4],
        "mature_fault": [192, 496, 4],
        "dense_inference_monitor": [160, 499, 1],
        "counts": {
            "training_all": 93,
            "training_prefix": 8,
            "training_monitor": 85,
            "ignored_fault_transition": 8,
            "mature_fault": 77,
            "dense_inference_monitor": 340,
        },
    },
    "paired_domains": {
        "A": {
            "shape": [52, 128],
            "definition": "fit-healthy globally normalized current128 minus same-run fixed prefix0:128 positionwise",
            "normalizer": "per-channel median and max(1.4826*MAD,1e-6) fitted from 60 fit-healthy prefixes rows0:128 only, clip[-10,10]",
            "go_final_refit_normalizer": "same formula fitted from 100 healthy prefixes rows0:128 only",
            "secondary_normalization": False,
        },
        "Bfeat": {
            "dimension": 604,
            "blocks": [572, 24, 8],
            "raw_order": "52 channel-major x 11 CHANNEL_SUMMARY_NAMES, then 2 LOOP_PAIRS x 12 _LOOP_ORDER, then 8 HEALTH_FEATURE_NAMES; exact names and formulas are bound by features.py and its tests",
            "channel_summary_names": [
                "delta_mean",
                "delta_log_variance",
                "delta_log_diff_energy",
                "delta_log_rfft_share[1:2]",
                "delta_log_rfft_share[2:4]",
                "delta_log_rfft_share[4:8]",
                "delta_log_rfft_share[8:16]",
                "delta_log_rfft_share[16:32]",
                "delta_log_rfft_share[32:65]",
                "delta_normalized_spectral_entropy",
                "delta_normalized_spectral_centroid",
            ],
            "loop_pairs_zero_based": [
                [51, 10, "xmv11_xmeas11"],
                [51, 21, "xmv11_xmeas22"],
            ],
            "loop_order_each_pair": [
                "current_dwell",
                "delta_dwell",
                "current_stick_slip",
                "delta_stick_slip",
                "current_reversal",
                "delta_reversal",
                "current_oriented_area",
                "delta_oriented_area",
                "delta_lag_asymmetry",
                "log_arx_residual_ratio",
                "delta_welch_coherence",
                "delta_actuator_psd_concentration",
            ],
            "health_feature_names": [
                "pca_t2",
                "pca_q",
                "lag4_dpca_t2",
                "lag4_dpca_q",
                "sfa_slow",
                "sfa_fast",
                "var8_state_mahalanobis",
                "var8_residual_mahalanobis",
            ],
            "health_subspace": {
                "fit_population": "60 fit-healthy normalized prefixes rows0:128 only",
                "go_final_refit_population": "100 healthy normalized prefixes rows0:128 only",
                "pca_rank": 16,
                "dpca_lag": 4,
                "dpca_rank": 16,
                "sfa_tail_count": 4,
                "var_order": 8,
                "subspace_ridge": 1e-6,
            },
            "normalizer_fit_pool": "60 fit-healthy runs x 85 sparse monitor endpoints = 5100 windows",
            "go_final_refit_normalizer_fit_pool": "100 healthy runs x 85 sparse monitor endpoints = 8500 windows",
            "normalizer": "per-feature median and max(1.4826*MAD,1e-6), clip[-10,10]",
            "same_frozen_normalizer_for_all_roles_and_checkpoint": True,
            "full_definition": "docs/pcf_cyclerank_v6.md#frozen-604-dimensional-bfeat-order",
        },
    },
    "stage1": {
        "training_scope": "outer-fold fit only: health only, 60x85 sparse monitor endpoint pairs 160..496 step4; prefix endpoints excluded",
        "G_AB": [
            "Conv1d52-128-k7-s2-p3,GN8,GELU",
            "Conv1d128-256-k5-s2-p2,GN16,GELU",
            "Conv1d256-256-k3-s2-p1,GN16,GELU",
            "time-mean-max-concat512,Linear512-256,GELU,Linear256-604",
        ],
        "G_BA": [
            "Linear604-512,GELU,Linear512-4096,GELU,reshape256x16",
            "ConvT256-256-k4-s2-p1,GN16,GELU",
            "ConvT256-128-k4-s2-p1,GN8,GELU",
            "ConvT128-52-k4-s2-p1,no-final-activation",
        ],
        "D_A": "PatchGAN Conv52-64-128-256 strides2 then Conv256-1 k3; GN8/GN16 and LeakyReLU0.2",
        "D_B": "MLP604-256-64-1 with LeakyReLU0.2",
        "loss": {
            "discriminator_each_domain": "0.5*(MSE(real,1)+MSE(fake_detach,0))",
            "generator_adversarial_each_domain": "0.5*MSE(discriminator(fake),1)",
            "cycle_A_weight": 10.0,
            "cycle_B_weight": 10.0,
            "paired_AB_weight": 5.0,
            "paired_BA_weight": 5.0,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
        },
        "epochs": 120,
        "batch_size_pairs": 128,
        "drop_last": False,
        "checkpoint": "last_epoch",
        "early_stopping": False,
    },
    "residual_signature": {
        "dimension": 656,
        "first52": "log1p(mean_time((A-A_cycle)^2))",
        "last604": "abs(Bfeat-Bfeat_cycle) in the frozen standardized B space",
    },
    "stage2": {
        "encoder": "Linear656-128,GELU,Dropout0.1,Linear128-32,GELU",
        "detection_head": "Linear32-1",
        "attribution_head": "Linear32-3 with IDV3,9,15 mapped to 0,1,2",
        "loss_weights": {
            "balanced_endpoint_bce": 0.40,
            "paired_rank": 0.30,
            "supervised_contrastive": 0.20,
            "mature_fault_attribution_ce": 0.10,
        },
        "balanced_bce": "negative population is healthy monitor85 plus fault prefix8, positive population is mature-fault77; 0.5 weight per binary population and three faults equal",
        "paired_rank": "per fault run mean softplus(1-(mature_logit-median(prefix8_logits))), three faults equal",
        "supcon": "L2-normalized 32D, labels0/3/9/15, self excluded, temperature0.1, at least2 per class",
        "attribution": "cross entropy on mature fault endpoints only",
        "fault_transition_160_188_used": False,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
        },
        "epochs": 80,
        "run_batch_per_class": 2,
        "joint_batch_total_runs": 8,
        "balanced_batches_per_epoch": 30,
        "checkpoint": "last_epoch",
        "early_stopping": False,
    },
    "go_only_final_refit": {
        "trigger": "only after all five frozen development folds produce DEVELOPMENT_GO",
        "source": "the same already-authorized 400 Mode2 magnitude50 records only",
        "runs_per_class": 100,
        "total_runs": 400,
        "channel_and_health_fit": "100 healthy prefixes rows0:128 only",
        "b_standardizer_rows": 8500,
        "stage1_health_pairs": 8500,
        "stage2_runs_per_class_batch": 2,
        "stage2_balanced_batches_per_epoch": 50,
        "scan_calibrator_fit": "100 healthy x 340 dense monitor logits after same-run prefix8 median subtraction",
        "scan_calibrator_saved_before_mode1": True,
        "same_candidate_seed_epochs_optimizers_and_losses": True,
        "may_change_development_decision": False,
        "target_domain_calibration_performed": False,
        "mode1_access_authorized": False,
    },
    "scan": {
        "prefix_endpoints": [128, 132, 136, 140, 144, 148, 152, 156],
        "dense_monitor_endpoints": [160, 499, 1],
        "fit_health_center": "m=median of all 60x340 v values after same-run prefix8 median subtraction",
        "fit_health_scale": "s=max(1.4826*median(abs(v-m)),1e-3)",
        "standardized_logit": "u=(v-m)/s",
        "drift": 0.25,
        "sustained_windows": [4, 8, 16, 32, 64],
        "sustained": "sum(last w of (u-drift))/sqrt(w), full windows only",
        "intermittent_window": 16,
        "intermittent_top_count": 4,
        "intermittent": "0.5*sum(top4(max(u-drift,0)) in latest16)",
        "raw_score": "max(sustained_available,intermittent_if_available)",
        "raw_first_endpoint": 163,
        "persistence_count": 2,
        "persistent_score": "min(raw_score[t-1],raw_score[t])",
        "first_alarm_eligible_endpoint": 164,
        "run_maximum": "max persistent score",
        "threshold_comparison": ">",
        "clipping": False,
        "ecdf": False,
    },
    "inner_threshold": {
        "runs": 20,
        "fault_inner_runs_used": 0,
        "ascending_rank": 20,
        "threshold": "maximum inner healthy run maximum",
        "comparison": ">",
        "minimum_exact_distinct_maxima": 18,
        "top_gap": "rank20-rank19 > 100*max(authoritative CUDA repeat error,1e-7)",
    },
    "determinism": {
        "authoritative_device": "CUDA RTX 5090",
        "authoritative_repeat": "within each fold, replay the frozen model's complete inner-healthy CUDA score -> frozen-calibrator transform -> scan -> decision path twice; training and outer scoring are not replayed",
        "numeric_absolute_tolerance": 1e-6,
        "authoritative_repeat_is_gate": True,
        "cross_device_cpu_cuda": "report_only_not_a_go_gate",
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    },
}

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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant forbidden: {value}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_file_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
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


def _write_json(path: Path, value: Any) -> str:
    payload = _json_file_bytes(value)
    _write_exclusive(path, payload)
    return _hash_bytes(payload)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _commit_path(path: Path) -> Path:
    return path.with_name(path.name + ".commit.json")


def _commit_json(path: Path, value: Any, *, commit_schema: str) -> dict[str, Any]:
    document_sha256 = _write_json(path, value)
    sidecar = _sidecar_path(path)
    sidecar_payload = f"{document_sha256}  {path.name}\n".encode("ascii")
    _write_exclusive(sidecar, sidecar_payload)
    commit = {
        "schema": commit_schema,
        "committed_utc": _now(),
        "document_name": path.name,
        "document_sha256": document_sha256,
        "document_size_bytes": path.stat().st_size,
        "sidecar_name": sidecar.name,
        "sidecar_sha256": _hash_bytes(sidecar_payload),
    }
    commit_sha256 = _write_json(_commit_path(path), commit)
    if (
        _hash_file(path) != document_sha256
        or _hash_file(sidecar) != commit["sidecar_sha256"]
        or _hash_file(_commit_path(path)) != commit_sha256
    ):
        raise RuntimeError("atomic JSON commit verification failed")
    return {
        "document_sha256": document_sha256,
        "sidecar_sha256": commit["sidecar_sha256"],
        "commit_sha256": commit_sha256,
    }


def _verify_committed_json(path: Path, *, commit_schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = _sidecar_path(path)
    commit_path = _commit_path(path)
    if not path.is_file() or not sidecar.is_file() or not commit_path.is_file():
        raise FileNotFoundError(f"incomplete committed JSON bundle for {path.name}")
    document_sha256 = _hash_file(path)
    expected_sidecar = f"{document_sha256}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"SHA-256 sidecar mismatch for {path.name}")
    commit = _strict_json(commit_path)
    if (
        commit.get("schema") != commit_schema
        or commit.get("document_name") != path.name
        or commit.get("document_sha256") != document_sha256
        or commit.get("document_size_bytes") != path.stat().st_size
        or commit.get("sidecar_name") != sidecar.name
        or commit.get("sidecar_sha256") != _hash_file(sidecar)
    ):
        raise ValueError(f"commit record mismatch for {path.name}")
    return _strict_json(path), {
        "document_sha256": document_sha256,
        "sidecar_sha256": _hash_file(sidecar),
        "commit_sha256": _hash_file(commit_path),
    }


def _canonical_fold_map() -> dict[int, int]:
    return {run_id: fold for fold, runs in FOLD_RUN_IDS.items() for run_id in runs}


def _fold_map_hash(value: Mapping[int, int] | None = None) -> str:
    fold_map = _canonical_fold_map() if value is None else dict(value)
    payload = {str(run): int(fold_map[run]) for run in sorted(fold_map)}
    return _hash_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)):
        raise ValueError("fold IDs must be exactly 0..4")
    if any(len(runs) != 20 for runs in FOLD_RUN_IDS.values()):
        raise ValueError("every fold must contain exactly 20 run IDs")
    flat = [run for fold in range(5) for run in FOLD_RUN_IDS[fold]]
    if len(flat) != 100 or sorted(flat) != list(range(1, 101)) or len(set(flat)) != 100:
        raise ValueError("folds must partition run IDs 1..100 exactly once")
    fold_map = _canonical_fold_map()
    for component in SEED_CONNECTED_COMPONENTS:
        if len({fold_map[run] for run in component}) != 1:
            raise ValueError("seed-connected run component crosses folds")
    if _fold_map_hash(fold_map) != FOLD_MAP_SHA256:
        raise ValueError("v5.1 canonical fold-map SHA-256 mismatch")


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
        "fit_run_ids_per_class": fit,
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
                    "window_zero_based_half_open": [100, 600] if idv == 1 else [440, 940],
                    "processdata_row_zero_based_half_open": [100, 600] if idv == 1 else [440, 940],
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


def _validate_ledger() -> dict[str, Any]:
    ledger = _strict_json(LEDGER_PATH)
    if (
        ledger.get("schema") != "pcf_cyclerank_v6_history_and_access_ledger"
        or ledger.get("ledger_state") != "FROZEN_TEXT_ARTIFACT_AUDIT_BEFORE_ANY_V6_HDF5_OPEN"
        or ledger.get("audit_scope", {}).get("hdf5_files_opened_by_this_audit") != 0
        or ledger.get("v6_lockbox_budget", {}).get("candidate_count") != 1
        or ledger.get("v6_development_authorization", {}).get("mode1_access_authorized") is not False
        or ledger.get("v6_development_authorization", {}).get("rieth_access_authorized") is not False
        or ledger.get("extended_tep_mode1_roles", {})
        .get("magnitude25_idv1_runs1_100", {})
        .get("access_level")
        != "NEVER_NUMERIC_ACCESSED"
        or ledger.get("extended_tep_mode1_roles", {})
        .get("magnitude75_idv1_3_9_15_runs1_100", {})
        .get("access_level")
        != "NEVER_NUMERIC_ACCESSED"
        or ledger.get("extended_tep_mode2_roles", {})
        .get("magnitude50_idv1_3_9_15_runs1_100", {})
        .get("access_level")
        != "SELECTED_OR_SCORED"
    ):
        raise ValueError("v6 history/access ledger semantic binding failed")
    source = ledger.get("source_identities", {}).get("extended_tep_mode2", {})
    if source.get("sha256") != SOURCE_SHA256 or source.get("size_bytes") != SOURCE_SIZE_BYTES:
        raise ValueError("ledger Mode2 source identity mismatch")
    return ledger


def _implementation_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required v6 implementation artifact is missing: {relative}")
        records[relative] = {
            "sha256": _hash_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _protocol_document(implementation: Mapping[str, Any]) -> dict[str, Any]:
    _validate_folds()
    _validate_ledger()
    plan = development_access_plan()
    if _hash_bytes(_canonical_bytes(plan)) != ACCESS_PLAN_SHA256:
        raise ValueError("v6 exact-selector access-plan SHA-256 changed")
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_state": "FROZEN_AFTER_ALL_IMPLEMENTATION_FILES_AND_BEFORE_ANY_V6_HDF5_OPEN",
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
            "source": "exact v5.1 collision-safe seed-connected assignment",
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
            "canonical_sha256": ACCESS_PLAN_SHA256,
            "exact_selector_fields": list(plan[0]),
            "wildcards_allowed": False,
        },
        "development_go_gates": GO_GATES,
        "governance": {
            "candidate_count": 1,
            "global_development_claim_count": 1,
            "rescue_iteration_ids_allowed": False,
            "mode1_access_authorized": False,
            "rieth_access_authorized": False,
            "mode1_magnitude25_state": "SEALED",
            "mode1_magnitude75_state": "SEALED",
            "inner_fault_numeric_access_authorized": False,
            "development_decisions": ["DEVELOPMENT_GO", "DEVELOPMENT_NO_GO"],
            "failure_before_intent": "INFRASTRUCTURE_FAILURE_BEFORE_NUMERIC_ACCESS",
            "failure_after_intent": "ABORTED_AFTER_NUMERIC_ACCESS",
            "same_candidate_rerun_after_claim": False,
        },
        "reporting": {
            "development_status": "adaptive exploratory feasibility only",
            "mode1_future_status": "prospectively held-out target-domain severity cell, not wholly unseen dataset-level zero-shot external validation",
            "pooled_fault_binomial_n300_forbidden": True,
            "healthy_final_claim": "exact one-sided 95% Clopper-Pearson; <=4/100 supports below0.10, <=1/100 supports below0.05",
        },
        "bindings": {
            "method_document": {
                "relative_path": "docs/pcf_cyclerank_v6.md",
                "sha256": _hash_file(DOCUMENT_PATH),
            },
            "history_and_access_ledger": {
                "relative_path": "protocols/v6_history_and_access_ledger.json",
                "sha256": _hash_file(LEDGER_PATH),
            },
            "implementation": _plain_json(implementation),
        },
        "numeric_backend_contract": {
            "module": BACKEND_MODULE,
            "required_functions": ["cuda_preflight", "validate_protocol", "run_development"],
            "backend_import_before_access_intent": False,
            "result_schema": BACKEND_RESULT_SCHEMA,
            "cuda_repeat_atol": 1e-6,
            "cpu_drift_is_go_gate": False,
            "authoritative_repeat_scope": "per-fold inner-healthy frozen-model score -> frozen-calibrator transform -> scan -> decision double replay",
        },
    }


def freeze_development() -> dict[str, Any]:
    implementation = _implementation_records()
    existing = (PROTOCOL_PATH, _sidecar_path(PROTOCOL_PATH), _commit_path(PROTOCOL_PATH))
    if any(path.exists() for path in existing):
        if not all(path.is_file() for path in existing):
            raise FileExistsError("partial v6 protocol bundle exists; refuse repair or overwrite")
        document, hashes = _verify_protocol_bundle()
        return {"status": "VERIFIED_EXISTING_FROZEN_PROTOCOL", **hashes, "protocol_id": document["protocol_id"]}
    document = _protocol_document(implementation)
    hashes = _commit_json(
        PROTOCOL_PATH,
        document,
        commit_schema="pcf_cyclerank_v6_protocol_atomic_commit",
    )
    return {"status": "FROZEN", **hashes, "protocol_id": PROTOCOL_ID}


def _verify_protocol_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    document, hashes = _verify_committed_json(
        PROTOCOL_PATH,
        commit_schema="pcf_cyclerank_v6_protocol_atomic_commit",
    )
    if (
        document.get("schema") != PROTOCOL_SCHEMA
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("candidate") != METHOD_CONFIG
        or document.get("candidate_count") != 1
        or document.get("development_go_gates") != GO_GATES
        or document.get("source", {}).get("sha256") != SOURCE_SHA256
        or document.get("source", {}).get("size_bytes") != SOURCE_SIZE_BYTES
        or document.get("source", {}).get("mode") != 2
        or document.get("source", {}).get("magnitude_percent") != 50
        or document.get("source", {}).get("idvs") != [1, 3, 9, 15]
        or document.get("source", {}).get("mode1_access_authorized") is not False
        or document.get("source", {}).get("rieth_access_authorized") is not False
        or document.get("governance", {}).get("mode1_access_authorized") is not False
        or document.get("governance", {}).get("rieth_access_authorized") is not False
        or document.get("access_plan", {}).get("record_count") != 400
        or document.get("access_plan", {}).get("selector_count") != 400
        or document.get("access_plan", {}).get("canonical_sha256")
        != ACCESS_PLAN_SHA256
        or _hash_bytes(_canonical_bytes(development_access_plan()))
        != ACCESS_PLAN_SHA256
    ):
        raise ValueError("v6 protocol semantic binding failed")
    _validate_folds()
    fold_document = document.get("folds", {})
    if (
        fold_document.get("fold_run_ids")
        != {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()}
        or fold_document.get("run_to_fold")
        != {str(run): fold for run, fold in sorted(_canonical_fold_map().items())}
        or fold_document.get("run_to_fold_canonical_json_sha256") != FOLD_MAP_SHA256
        or fold_document.get("cross_fold_seed_collision_count") != 0
    ):
        raise ValueError("v6 protocol fold binding failed")
    numeric_contract = document.get("numeric_backend_contract", {})
    if (
        numeric_contract.get("cuda_repeat_atol") != 1e-6
        or numeric_contract.get("cpu_drift_is_go_gate") is not False
    ):
        raise ValueError("v6 protocol numeric replay binding failed")
    ledger = _validate_ledger()
    del ledger
    bindings = document.get("bindings", {})
    if (
        bindings.get("method_document", {}).get("sha256") != _hash_file(DOCUMENT_PATH)
        or bindings.get("history_and_access_ledger", {}).get("sha256") != _hash_file(LEDGER_PATH)
        or bindings.get("implementation") != _implementation_records()
    ):
        raise ValueError("v6 protocol implementation binding failed")
    return document, hashes


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
        "--query-gpu=name,memory.total,driver_version",
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
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi GPU presence preflight failed")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not any("5090" in row for row in rows):
        raise RuntimeError("frozen RTX 5090 device was not found")
    return {
        "command": command,
        "returncode": completed.returncode,
        "rows": rows,
        "rtx5090_present": True,
        "note": "PyTorch/CUDA runtime validation is called from the frozen backend only after the access intent is committed.",
    }


def _run_test_suite() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *TEST_PATHS, "-q", "-W", "error"]
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
        raise RuntimeError("frozen v6 synthetic/governance test suite failed")
    return {
        "command": command,
        "returncode": completed.returncode,
        "output_sha256": _hash_bytes(combined.encode("utf-8")),
        "output_tail": combined[-2000:],
        "hdf5_numeric_access_authorized": False,
    }


def preflight() -> dict[str, Any]:
    protocol, protocol_hashes = _verify_protocol_bundle()
    source = _source_record()
    gpu = _gpu_presence_preflight()
    tests = _run_test_suite()
    return {
        "schema": "pcf_cyclerank_v6_governance_preflight",
        "schema_version": 1,
        "completed_utc": _now(),
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hashes["document_sha256"],
        "protocol_commit_sha256": protocol_hashes["commit_sha256"],
        "candidate_id": protocol["candidate"]["candidate_id"],
        "source": source,
        "gpu_presence": gpu,
        "tests": tests,
        "backend_imported": False,
        "hdf5_opened": False,
        "mode1_access_authorized": False,
    }


def _load_backend() -> ModuleType:
    module = importlib.import_module(BACKEND_MODULE)
    for name in ("cuda_preflight", "validate_protocol", "run_development"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"{BACKEND_MODULE}.{name} is required")
    return module


def _prepare_backend_environment() -> None:
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif observed != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from frozen :4096:8")


def _validate_backend_preflight(value: Any) -> dict[str, Any]:
    report = _plain_json(value)
    if not isinstance(report, dict):
        raise ValueError("backend cuda_preflight must return a mapping")
    passed = report.get("passed")
    cuda_available = report.get("cuda_available")
    if passed is not True or cuda_available is not True:
        raise RuntimeError("numeric backend CUDA preflight did not pass")
    encoded = json.dumps(report, sort_keys=True).lower()
    if "5090" not in encoded:
        raise RuntimeError("numeric backend did not bind the RTX 5090")
    return report


def _validate_development_result(value: Any, *, protocol_sha256: str) -> dict[str, Any]:
    result = _plain_json(value)
    if not isinstance(result, dict):
        raise ValueError("backend development result must be a mapping")
    decision = result.get("decision")
    if (
        result.get("schema") != BACKEND_RESULT_SCHEMA
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("protocol_sha256") != protocol_sha256
        or result.get("candidate_id") != CANDIDATE_ID
        or result.get("candidate_count") != 1
        or result.get("source_sha256") != SOURCE_SHA256
        or result.get("source_size_bytes") != SOURCE_SIZE_BYTES
        or result.get("fold_map_sha256") != FOLD_MAP_SHA256
        or result.get("access_plan_sha256") != ACCESS_PLAN_SHA256
        or result.get("mode1_accessed") is not False
        or result.get("mode1_hdf5_opens_or_value_reads") != 0
        or result.get("mode1_access_count") != 0
        or result.get("rieth_accessed") is not False
        or result.get("rieth_access_count") != 0
        or decision not in {"DEVELOPMENT_GO", "DEVELOPMENT_NO_GO"}
    ):
        raise ValueError("backend development identity/result binding is invalid")
    for name in (
        "candidate_count",
        "source_size_bytes",
        "mode1_hdf5_opens_or_value_reads",
        "mode1_access_count",
        "rieth_access_count",
    ):
        if type(result.get(name)) is not int:
            raise ValueError(f"backend result {name} must be an exact integer")

    topology = result.get("topology")
    numeric_access = topology.get("numeric_value_access") if isinstance(topology, dict) else None
    counts_by_status = topology.get("counts_by_status") if isinstance(topology, dict) else None
    counts_by_idv = topology.get("counts_by_idv") if isinstance(topology, dict) else None
    topology_integer_fields = (
        "selected_run_count",
        "unique_seed_count",
        "duplicate_seed_value_count",
        "cross_fold_seed_collision_count",
    )
    numeric_access_integer_fields = (
        "processdata_slices",
        "processdata_rows_per_slice",
        "idv_init_slices",
        "root_dataset_value_reads",
        "processdata_values_outside_authorized_windows",
    )
    if (
        not isinstance(topology, dict)
        or any(type(topology.get(name)) is not int for name in topology_integer_fields)
        or not isinstance(numeric_access, dict)
        or any(
            type(numeric_access.get(name)) is not int
            for name in numeric_access_integer_fields
        )
        or not isinstance(counts_by_status, dict)
        or any(type(value) is not int for value in counts_by_status.values())
        or not isinstance(counts_by_idv, dict)
        or any(type(value) is not int for value in counts_by_idv.values())
        or topology.get("selected_run_count") != 400
        or counts_by_status != {"SimulationCompleted": 400}
        or counts_by_idv != {"1": 100, "3": 100, "9": 100, "15": 100}
        or topology.get("unique_seed_count") != 395
        or topology.get("duplicate_seed_value_count") != 5
        or topology.get("cross_fold_seed_collision_count") != 0
        or numeric_access
        != {
            "processdata_slices": 400,
            "processdata_rows_per_slice": 500,
            "processdata_columns_half_open": [0, 53],
            "idv_init_slices": 400,
            "root_dataset_value_reads": 0,
            "processdata_values_outside_authorized_windows": 0,
        }
    ):
        raise ValueError("backend exact-selector numeric-access topology is invalid")
    environment = result.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("passed") is not True
        or environment.get("cuda_available") is not True
        or "5090" not in json.dumps(environment, sort_keys=True).lower()
    ):
        raise ValueError("backend result is not bound to the authoritative RTX 5090 CUDA run")
    gates = result.get("gates")
    all_pass = gates.get("all_pass") if isinstance(gates, dict) else None
    if not isinstance(all_pass, bool) or all_pass != (decision == "DEVELOPMENT_GO"):
        raise ValueError("backend decision and all-pass gate disagree")
    selected = result.get("selected_candidate_id")
    expected_selected = CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None
    if selected != expected_selected:
        raise ValueError("backend selected-candidate binding disagrees with decision")
    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("backend must return all five outer folds")
    if any(not isinstance(fold, dict) for fold in folds) or {
        fold.get("outer_fold") for fold in folds
    } != set(range(5)) or any(type(fold.get("outer_fold")) is not int for fold in folds):
        raise ValueError("backend outer-fold identities must be exactly 0 through 4")
    repeat = result.get("authoritative_cuda_repeat")
    if not isinstance(repeat, dict):
        raise ValueError("authoritative CUDA repeat report is missing")
    discrete_equal = repeat.get("discrete_equal")
    repeat_passes = repeat.get("passes")
    if not isinstance(discrete_equal, bool) or not isinstance(repeat_passes, bool):
        raise ValueError("authoritative CUDA repeat gates must be booleans")
    if (
        repeat.get("scope") != "per_fold_inner_complete_score_scan_replay"
        or repeat.get("atol") != 1e-6
        or isinstance(repeat.get("atol"), bool)
    ):
        raise ValueError("authoritative CUDA repeat scope or tolerance changed")
    if decision == "DEVELOPMENT_GO" and not (discrete_equal and repeat_passes):
        raise ValueError("a development GO requires both CUDA repeat gates")
    cross_device = result.get("cross_device_report_only")
    if not isinstance(cross_device, dict) or cross_device.get("is_go_gate") is not False:
        raise ValueError("cross-device comparison must be present and report-only")
    if not isinstance(gates.get("checks"), dict):
        raise ValueError("backend GO-gate checks are missing")
    if result.get("aggregate") != gates.get("aggregate_metrics"):
        raise ValueError("backend aggregate differs from the GO-gate aggregate")

    final_refit = result.get("final_refit")
    if decision == "DEVELOPMENT_NO_GO":
        if final_refit is not None:
            raise ValueError("a development NO_GO must not create a final refit")
    else:
        if (
            not isinstance(final_refit, dict)
            or final_refit.get("schema") != "pcf_cyclerank_final_mode2_refit_v1"
            or final_refit.get("candidate_id") != CANDIDATE_ID
            or final_refit.get("runs_per_class") != 100
            or final_refit.get("scan_calibrator_fit_healthy_runs") != 100
            or final_refit.get("created_only_after_development_go") is not True
            or final_refit.get("target_domain_calibration_not_performed") is not True
        ):
            raise ValueError("development GO final-refit metadata is invalid")
        for name in ("feature_state", "model", "scan_calibrator", "manifest"):
            record = final_refit.get(name)
            digest = record.get("sha256") if isinstance(record, dict) else None
            size = record.get("size_bytes") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not record["path"]
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(size) is not int
                or size <= 0
            ):
                raise ValueError(f"development GO final-refit {name} record is invalid")
    return result


def _write_failure_and_completion(
    *,
    directory: Path,
    protocol_sha256: str,
    stage: str,
    error: BaseException,
    intent_committed: bool,
    preflight_sha256: str | None,
    access_intent_sha256: str | None,
) -> dict[str, Any]:
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    decision = (
        "ABORTED_AFTER_NUMERIC_ACCESS"
        if intent_committed
        else "INFRASTRUCTURE_FAILURE_BEFORE_NUMERIC_ACCESS"
    )
    failure = {
        "schema": "pcf_cyclerank_v6_development_failure",
        "failed_utc": _now(),
        "decision": decision,
        "failure_stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback_sha256": _hash_bytes(trace.encode("utf-8")),
        "access_intent_committed": intent_committed,
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
    }
    failure_sha256 = _write_json(directory / "failure.json", failure)
    completion = {
        "schema": "pcf_cyclerank_v6_development_completion",
        "completed_utc": _now(),
        "decision": decision,
        "protocol_sha256": protocol_sha256,
        "failure_sha256": failure_sha256,
        "preflight_sha256": preflight_sha256,
        "access_intent_sha256": access_intent_sha256,
        "access_intent_committed": intent_committed,
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
    }
    completion_sha256 = _write_json(directory / "completion.json", completion)
    return {**completion, "completion_sha256": completion_sha256}


def run_development(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol, protocol_hashes = _verify_protocol_bundle()
    protocol_sha256 = protocol_hashes["document_sha256"]
    plan = development_access_plan()
    plan_sha256 = _hash_bytes(_canonical_bytes(plan))
    if (
        plan_sha256 != ACCESS_PLAN_SHA256
        or plan_sha256 != protocol["access_plan"]["canonical_sha256"]
    ):
        raise ValueError("live v6 access plan differs from frozen protocol")

    # This preflight is read-only: protocol/source-byte verification, GPU
    # presence, and synthetic tests.  A transient failure here must not burn
    # the one global numeric-access claim.
    audit = preflight()

    root = Path(output_root)
    directory = root / DEVELOPMENT_DIRECTORY_NAME
    claim_path = root / "development_access.claim.json"
    if claim_path.exists() or directory.exists():
        raise FileExistsError("a prior v6 global claim or fixed development directory exists")
    root.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "pcf_cyclerank_v6_exclusive_global_development_claim",
        "claimed_utc": _now(),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "access_plan_sha256": plan_sha256,
        "fold_map_sha256": FOLD_MAP_SHA256,
        "fixed_directory": DEVELOPMENT_DIRECTORY_NAME,
        "rescue_iteration_ids_allowed": False,
        "policy": "Any prior claim blocks every automatic rerun, including a renamed iteration or infrastructure rescue.",
    }
    _write_json(claim_path, claim)
    directory.mkdir(parents=False, exist_ok=False)

    stage = "governance_preflight"
    intent_committed = False
    preflight_sha256: str | None = None
    intent_sha256: str | None = None
    try:
        preflight_sha256 = _write_json(directory / "preflight.json", audit)

        stage = "access_intent_commit"
        intent = {
            "schema": "pcf_cyclerank_v6_mode2_development_access_intent",
            "schema_version": 1,
            "committed_utc": _now(),
            "committed_before_backend_import_or_hdf5_open": True,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha256,
            "protocol_commit_sha256": protocol_hashes["commit_sha256"],
            "candidate_id": CANDIDATE_ID,
            "candidate_count": 1,
            "source": audit["source"],
            "access_plan": plan,
            "access_plan_sha256": plan_sha256,
            "fold_run_ids": {str(fold): list(runs) for fold, runs in FOLD_RUN_IDS.items()},
            "fold_map_sha256": FOLD_MAP_SHA256,
            "authorized_reads_after_this_commit": "only the exact 400 Mode2 magnitude50 selectors above",
            "wildcards_authorized": False,
            "inner_fault_values_authorized": False,
            "mode1_access_authorized": False,
            "mode1_hdf5_opens_or_value_reads": 0,
            "mode1_magnitude25_state": "SEALED",
            "mode1_magnitude75_state": "SEALED",
            "rieth_access_authorized": False,
            "backend_module": BACKEND_MODULE,
            "backend_imported_before_this_commit": False,
            "preflight_sha256": preflight_sha256,
            "implementation": protocol["bindings"]["implementation"],
        }
        intent_hashes = _commit_json(
            directory / "access_intent.json",
            intent,
            commit_schema="pcf_cyclerank_v6_access_intent_atomic_commit",
        )
        intent_sha256 = intent_hashes["document_sha256"]
        intent_committed = True

        stage = "backend_import_and_cuda_preflight"
        _prepare_backend_environment()
        backend = _load_backend()
        backend_validation = _plain_json(backend.validate_protocol(protocol))
        if (
            not isinstance(backend_validation, dict)
            or backend_validation.get("valid") is not True
            or backend_validation.get("schema") != PROTOCOL_SCHEMA
            or backend_validation.get("protocol_id") != PROTOCOL_ID
            or backend_validation.get("candidate_id") != CANDIDATE_ID
            or backend_validation.get("candidate_count") != 1
            or backend_validation.get("canonical_document_sha256") != protocol_sha256
            or backend_validation.get("fold_map_sha256") != FOLD_MAP_SHA256
        ):
            raise ValueError("numeric backend rejected the frozen v6 protocol")
        backend_preflight = _validate_backend_preflight(backend.cuda_preflight())
        backend_preflight_sha256 = _write_json(
            directory / "backend_cuda_preflight.json", backend_preflight
        )

        stage = "numeric_backend_development"
        raw_result = backend.run_development(
            access_plan=plan,
            fold_map=_canonical_fold_map(),
            artifact_root=directory / "artifacts",
            protocol_sha256=protocol_sha256,
            device="cuda",
        )
        result = _validate_development_result(raw_result, protocol_sha256=protocol_sha256)
        result_sha256 = _write_json(directory / "development_result.json", result)
        completion = {
            "schema": "pcf_cyclerank_v6_development_completion",
            "completed_utc": _now(),
            "decision": result["decision"],
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": protocol_sha256,
            "preflight_sha256": preflight_sha256,
            "backend_cuda_preflight_sha256": backend_preflight_sha256,
            "access_intent_sha256": intent_sha256,
            "access_intent_commit_sha256": intent_hashes["commit_sha256"],
            "development_result_sha256": result_sha256,
            "source_sha256": SOURCE_SHA256,
            "fold_map_sha256": FOLD_MAP_SHA256,
            "candidate_count": 1,
            "mode1_accessed": False,
            "mode1_hdf5_opens_or_value_reads": 0,
            "rieth_accessed": False,
        }
        completion_sha256 = _write_json(directory / "completion.json", completion)
        return {**completion, "completion_sha256": completion_sha256}
    except BaseException as error:
        _write_failure_and_completion(
            directory=directory,
            protocol_sha256=protocol_sha256,
            stage=stage,
            error=error,
            intent_committed=intent_committed,
            preflight_sha256=preflight_sha256,
            access_intent_sha256=intent_sha256,
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze-development", "preflight", "development"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-development":
        report = freeze_development()
    elif args.command == "preflight":
        report = preflight()
    else:
        report = run_development(output_root=args.output_root)
    print(json.dumps(_plain_json(report), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
