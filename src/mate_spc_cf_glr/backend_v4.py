"""Torch-first numeric backend for the governed CF-Max-GLR15 v4 study."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

# Windows CUDA runtime contract: torch must be the first native numeric import.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # pragma: no cover - import-order regression
    _torch = None
else:
    try:  # pragma: no cover - environment dependent
        import torch as _torch
    except (ImportError, OSError):  # pragma: no cover - environment dependent
        _torch = None

import numpy as np

from mate_spc_pclr.data_v2 import PCLRMode1Archive, PCLRRun
from .features import EXPERT_FEATURE_COUNTS, FEATURE_SCHEMA, extract_expert_feature_paths
from .model import (
    CFGLRConfig,
    CFGLRModel,
    CFGLRTrainingConfig,
    fit_cf_glr_cuda,
    load_cf_glr_model,
)
from .monitor import (
    FittedExpertStandardizer,
    apply_strict_threshold,
    fit_rank20_threshold,
    fused_monitor_path,
)


ALL_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
RUNS = tuple(range(1, 101))
FOLD_COUNT = 5
FOLD_SALT = "cf-max-glr15-v4-mode1-mag50-three-way-2026-09-08"
UNIQUE_CANDIDATE_ID = "cf-max-glr15-fixed"
PARITY_ATOL = 1.0e-9
FLOAT_TOLERANCE = 1.0e-12
MONITOR_START = 160
RUN_LENGTH = 500
MISS_DELAY = 340
BACKEND_SCHEMA = "cf_max_glr15_backend_v4"
BOUND_MODEL_SCHEMA = "cf_max_glr15_bound_model_v1"
BOUND_MONITOR_SCHEMA = "cf_max_glr15_bound_monitor_v1"

DEVELOPMENT_GATES = {
    "healthy_alarm_runs_max": 10,
    "healthy_alarm_runs_per_fold_max": 4,
    "each_fault_detection_rate_min": 0.85,
    "each_fold_each_fault_detection_rate_min": 0.70,
    "each_fault_median_delay_max": 140,
    "each_fault_nearest_rank_p90_delay_max": 220,
}
CONFIRMATION_GATES = {
    "healthy_alarm_runs_max": 4,
    "each_fault_detected_runs_min": 90,
    "each_fault_median_delay_max": 140,
    "each_fault_nearest_rank_p90_delay_max": 220,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _plain_json(value: object) -> Any:
    return json.loads(_canonical_json(value))


def _config_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _expected_config() -> dict[str, Any]:
    return {
        "schema": "cf_max_glr15_complete_config",
        "schema_version": 1,
        "candidate": {"candidate_id": UNIQUE_CANDIDATE_ID, "candidate_count": 1},
        "features": {
            "schema": "cf_glr_mechanism_features_v2",
            "dimensions_by_fault": {"3": 20, "9": 20, "15": 28},
            "canonical_zero_half_open": [0, 128],
            "prefix_feature_half_open": [128, 160],
            "monitor_half_open": [160, 500],
            "causal_no_future_rows": True,
        },
        "experts": {
            "3": {"architecture": "linear", "hidden_width": 0, "seeds": [20260909]},
            "9": {"architecture": "linear", "hidden_width": 0, "seeds": [20260909]},
            "15": {
                "architecture": "gelu_mlp_28x16x1",
                "hidden_width": 16,
                "seeds": [20260909, 20260910, 20260911],
                "ensemble": "arithmetic_mean_endpoint_logits",
            },
        },
        "training": {
            "optimizer": "AdamW",
            "epochs": 80,
            "learning_rate": 0.001,
            "batching": "full_batch_per_expert_all_fit_healthy_and_matching_fault_bags",
            "weight_decay": 0.0,
            "gradient_clip": None,
            "dtype": "float64",
            "device": "cuda",
            "mil_endpoints_half_open_step": [160, 500, 4],
            "mil_top_fraction": 0.10,
            "mil_top_count": 9,
            "prefix_negative_endpoints_half_open_step": [128, 160, 4],
            "prefix_negative_endpoint_count": 8,
            "healthy_bag_bce_weight": 0.5,
            "fault_bag_bce_weight": 0.5,
            "prefix_negative_weight": 0.1,
            "l2_weight": 0.0001,
            "validation_or_early_stopping": False,
        },
        "monitor": {
            "fit": "per_expert_global_median_MAD_on_fit_healthy_monitored_points",
            "fusion": "U_t=max_k_standardized_logit",
            "persistence": "P_t=min(U_t_minus_1,U_t)",
            "run_maximum": "max_t_P_t",
            "inner_calibration_rank": 20,
            "inner_calibration_size": 20,
            "formal_calibration_rank": 98,
            "formal_calibration_size": 100,
            "comparison": "strictly_greater_than",
            "attribution": "argmax_expert_at_first_alarm_lowest_fault_id_tie",
        },
        "nested_development": {
            "fold_count": 5,
            "fold_salt": FOLD_SALT,
            "outer_test": "F_j",
            "inner_calibration": "F_(j+1)_mod_5_healthy_only",
            "fit": "remaining_three_folds",
            "fit_runs_per_class": 60,
            "calibration_healthy_runs": 20,
            "outer_test_runs_per_class": 20,
        },
        "development_gates": dict(DEVELOPMENT_GATES),
        "confirmation_gates": dict(CONFIRMATION_GATES),
    }


def _validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("config must be a mapping")
    try:
        normalized = _plain_json(value)
    except (TypeError, ValueError) as error:
        raise ValueError("config must be finite JSON") from error
    if _canonical_json(normalized) != _canonical_json(_expected_config()):
        raise ValueError("config differs from the exact audited v4 specification")
    if FEATURE_SCHEMA != normalized["features"]["schema"]:
        raise RuntimeError("feature module schema differs from the v4 protocol")
    if dict(EXPERT_FEATURE_COUNTS) != {3: 20, 9: 20, 15: 28}:
        raise RuntimeError("expert feature geometry differs from the v4 protocol")
    return normalized


def _require_cuda_float64(*, device: str, dtype: str) -> None:
    if device != "cuda" or dtype != "float64":
        raise ValueError("v4 numeric phases require CUDA float64")
    if _torch is None:
        raise RuntimeError("PyTorch must be imported before NumPy in a fresh process")
    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def cuda_preflight() -> Mapping[str, Any]:
    _require_cuda_float64(device="cuda", dtype="float64")
    return {
        "cuda_available": True,
        "device_type": "cuda",
        "dtype": "float64",
        "torch_version": str(_torch.__version__),
        "cuda_version": str(_torch.version.cuda),
        "device_name": str(_torch.cuda.get_device_properties(0).name),
    }


def _canonical_fold_map() -> dict[int, int]:
    ranked = sorted(
        RUNS,
        key=lambda run: hashlib.sha256(f"{FOLD_SALT}|{run}".encode("ascii")).hexdigest(),
    )
    return dict(sorted((run, index % FOLD_COUNT) for index, run in enumerate(ranked)))


def _validate_fold_map(value: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(value, Mapping) or any(
        type(key) is not int or type(item) is not int for key, item in value.items()
    ):
        raise ValueError("fold_map must contain exact integer identities")
    normalized = dict(value)
    if normalized != _canonical_fold_map():
        raise ValueError("fold_map differs from the canonical salted mapping")
    return normalized


def _plans() -> dict[str, tuple[dict[str, Any], ...]]:
    development: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for run in RUNS:
        for idv in ALL_IDVS:
            development.append({
                "idv": idv,
                "magnitude_percent": 50,
                "simulation_run": run,
                "window_half_open": [100, 600] if idv == 1 else [440, 940],
                "role": "healthy_development" if idv == 1 else f"fault-{idv}_development",
            })
            confirmation.append({
                "idv": idv,
                "magnitude_percent": 75,
                "simulation_run": run,
                "window_half_open": [100, 600] if idv == 1 else [440, 940],
                "role": "healthy_confirmation" if idv == 1 else f"fault-{idv}_confirmation",
            })
    calibration = tuple({
        "idv": 1,
        "magnitude_percent": 25,
        "simulation_run": run,
        "window_half_open": [100, 600],
        "role": "healthy_calibration_prefault_only",
    } for run in RUNS)
    return {
        "development": tuple(development),
        "calibrate": calibration,
        "confirm": tuple(confirmation),
    }


_PLANS = _plans()


def _validate_access_plan(
    value: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("access_plan must be a sequence")
    required = {"idv", "magnitude_percent", "simulation_run", "window_half_open", "role"}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"access_plan[{index}] fields changed")
        window = raw["window_half_open"]
        if isinstance(window, (str, bytes)) or not isinstance(window, Sequence) or len(window) != 2:
            raise ValueError("access plan window is malformed")
        numeric = (raw["idv"], raw["magnitude_percent"], raw["simulation_run"], *window)
        if any(type(item) is not int for item in numeric):
            raise ValueError("access plan identities must be exact integers")
        if not isinstance(raw["role"], str):
            raise ValueError("access plan role must be a string")
        normalized.append({
            "idv": raw["idv"],
            "magnitude_percent": raw["magnitude_percent"],
            "simulation_run": raw["simulation_run"],
            "window_half_open": [window[0], window[1]],
            "role": raw["role"],
        })
    candidate = tuple(normalized)
    matches = [phase for phase, expected in _PLANS.items() if candidate == expected]
    if len(matches) != 1:
        raise ValueError("access plan is not one exact canonical v4 phase")
    return matches[0], candidate


def load_numeric_runs(
    source_path: Path, access_plan: Sequence[Mapping[str, Any]]
) -> tuple[PCLRRun, ...]:
    """Materialize exactly one already-authorized public data-v2 partition."""

    phase, _ = _validate_access_plan(access_plan)
    archive = PCLRMode1Archive(source_path)
    if phase == "development":
        runs = tuple(
            condition
            for group in archive.load_development_groups()
            for condition in group.conditions
        )
    elif phase == "calibrate":
        runs = archive.load_calibration_runs()
    else:
        healthy, faults = archive.load_final_runs()
        runs = tuple(healthy) + tuple(faults)
    expected = {
        (row["idv"], row["simulation_run"], row["magnitude_percent"])
        for row in access_plan
    }
    observed = {
        (run.source_idv, run.simulation_run, run.fault_magnitude_percent)
        for run in runs
    }
    if len(runs) != len(expected) or observed != expected:
        raise RuntimeError("materialized run identities differ from the authorized plan")
    return tuple(sorted(runs, key=lambda run: (run.simulation_run, run.source_idv)))


def _run_key(run: PCLRRun) -> tuple[int, int, int]:
    return (run.source_idv, run.simulation_run, run.fault_magnitude_percent)


def _build_feature_cache(runs: Sequence[PCLRRun]) -> dict[tuple[int, int, int], Any]:
    cache = {
        _run_key(run): extract_expert_feature_paths(run.x)
        for run in runs
    }
    if len(cache) != len(runs):
        raise RuntimeError("feature cache run identities are duplicated")
    return cache


def _feature_batches(
    runs: Sequence[PCLRRun],
    cache: Mapping[tuple[int, int, int], Any] | None = None,
) -> dict[int, np.ndarray]:
    if not runs:
        raise ValueError("feature batch cannot be empty")
    available = _build_feature_cache(runs) if cache is None else cache
    try:
        paths = [available[_run_key(run)] for run in runs]
    except KeyError as error:
        raise RuntimeError("feature cache lacks a requested run") from error
    return {
        fault: np.stack([path.for_fault(fault) for path in paths], axis=0)
        .astype(np.float64, copy=False)
        for fault in FAULT_IDVS
    }


def _subset(
    runs: Sequence[PCLRRun], *, idvs: Sequence[int], simulation_runs: Sequence[int]
) -> tuple[PCLRRun, ...]:
    allowed_ids = set(idvs)
    allowed_runs = set(simulation_runs)
    selected = tuple(
        run for run in runs
        if run.source_idv in allowed_ids and run.simulation_run in allowed_runs
    )
    expected = len(allowed_ids) * len(allowed_runs)
    if len(selected) != expected:
        raise RuntimeError("run subset differs from its exact Cartesian role")
    return tuple(sorted(selected, key=lambda run: (run.simulation_run, run.source_idv)))


def _fit_model(
    runs: Sequence[PCLRRun],
    cache: Mapping[tuple[int, int, int], Any] | None = None,
) -> Any:
    healthy = _subset(
        runs, idvs=(1,), simulation_runs=sorted({run.simulation_run for run in runs})
    )
    fault_runs = {
        fault: tuple(run for run in runs if run.source_idv == fault)
        for fault in FAULT_IDVS
    }
    if any(len(value) != len(healthy) for value in fault_runs.values()):
        raise RuntimeError("expert fitting requires balanced healthy/target-fault runs")
    healthy_all = _feature_batches(healthy, cache)
    target = {
        fault: _feature_batches(fault_runs[fault], cache)[fault]
        for fault in FAULT_IDVS
    }
    return fit_cf_glr_cuda(
        healthy_all,
        target,
        config=CFGLRConfig(),
        training=CFGLRTrainingConfig(),
    )


def _score_model(
    model: CFGLRModel,
    runs: Sequence[PCLRRun],
    *,
    device: str,
    cache: Mapping[tuple[int, int, int], Any] | None = None,
) -> np.ndarray:
    if device not in ("cpu", "cuda"):
        raise ValueError("score device must be cpu or cuda")
    logits = model.score_runs(_feature_batches(runs, cache), device=device).logits
    if logits.shape != (len(runs), RUN_LENGTH, len(FAULT_IDVS)):
        raise RuntimeError("expert score geometry changed")
    return logits


def _score_rows(
    runs: Sequence[PCLRRun],
    gpu_logits: np.ndarray,
    cpu_logits: np.ndarray,
    standardizer: FittedExpertStandardizer,
    threshold: float,
) -> tuple[list[dict[str, Any]], float, bool]:
    if gpu_logits.shape != cpu_logits.shape or gpu_logits.shape[0] != len(runs):
        raise ValueError("CPU/CUDA score batches differ")
    rows: list[dict[str, Any]] = []
    discrete = True
    maximum_error = 0.0
    for run, gpu, cpu in zip(runs, gpu_logits, cpu_logits, strict=True):
        error = float(np.max(np.abs(gpu - cpu)))
        if not math.isfinite(error) or error > PARITY_ATOL:
            raise RuntimeError(f"CPU/CUDA logits differ by {error:.3e}")
        maximum_error = max(maximum_error, error)
        gpu_decision = apply_strict_threshold(
            fused_monitor_path(gpu, standardizer), threshold
        )
        cpu_decision = apply_strict_threshold(
            fused_monitor_path(cpu, standardizer), threshold
        )
        gpu_discrete = (
            gpu_decision.alarm,
            gpu_decision.first_alarm_index,
            gpu_decision.attributed_fault,
        )
        cpu_discrete = (
            cpu_decision.alarm,
            cpu_decision.first_alarm_index,
            cpu_decision.attributed_fault,
        )
        discrete = discrete and gpu_discrete == cpu_discrete
        rows.append({
            "idv": run.source_idv,
            "simulation_run": run.simulation_run,
            "run_maximum": gpu_decision.run_maximum,
            "parity_max_abs_error": error,
            "alarm": gpu_decision.alarm,
            "first_alarm_index": gpu_decision.first_alarm_index,
            "attribution": gpu_decision.attributed_fault,
        })
    return rows, maximum_error, bool(discrete)


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def _development_pass(folds: Sequence[Mapping[str, Any]]) -> bool:
    if not all(fold["calibration_nondegenerate"] for fold in folds):
        return False
    rows = [row for fold in folds for row in fold["rows"]]
    healthy = [row for row in rows if row["idv"] == 1]
    if sum(row["alarm"] for row in healthy) > 10:
        return False
    for fold in folds:
        if sum(row["alarm"] for row in fold["rows"] if row["idv"] == 1) > 4:
            return False
    for fault in FAULT_IDVS:
        chosen = [row for row in rows if row["idv"] == fault]
        delays = [
            row["first_alarm_index"] - MONITOR_START if row["alarm"] else MISS_DELAY
            for row in chosen
        ]
        if (
            sum(row["alarm"] for row in chosen) / len(chosen) < 0.85
            or statistics.median(delays) > 140
            or _nearest_rank(delays, 0.90) > 220
        ):
            return False
        for fold in folds:
            fold_rows = [row for row in fold["rows"] if row["idv"] == fault]
            if sum(row["alarm"] for row in fold_rows) / len(fold_rows) < 0.70:
                return False
    return True


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = _canonical_bytes(value)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"nonfinite JSON constant is forbidden: {token}")

    value = json.loads(path.read_text("utf-8"), parse_constant=reject)
    if not isinstance(value, dict) or path.read_bytes() != _canonical_bytes(value):
        raise ValueError("monitor JSON must be a canonical object")
    return value


def _monitor_payload(
    standardizer: FittedExpertStandardizer, config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": BOUND_MONITOR_SCHEMA,
        "backend_schema": BACKEND_SCHEMA,
        "complete_config_sha256": _config_sha256(config),
        "candidate_id": UNIQUE_CANDIDATE_ID,
        "standardizer": standardizer.to_document(),
        "fit_scope": "all_100_magnitude50_development_healthy_logits_only",
        "held_out_25_or_75_percent_accessed": False,
    }


def _model_metadata(
    training_result: Any,
    *,
    config: Mapping[str, Any],
    monitor_payload: Mapping[str, Any],
    fit_runs: Sequence[PCLRRun],
) -> dict[str, Any]:
    counts = {
        str(idv): sum(run.source_idv == idv for run in fit_runs)
        for idv in ALL_IDVS
    }
    return _plain_json({
        "schema": BOUND_MODEL_SCHEMA,
        "backend_schema": BACKEND_SCHEMA,
        "complete_config_sha256": _config_sha256(config),
        "candidate_id": UNIQUE_CANDIDATE_ID,
        "model_config": asdict(training_result.model.config),
        "training_config": asdict(training_result.training),
        "final_losses": dict(training_result.final_losses),
        "fit_run_count_by_class": counts,
        "fit_run_names": sorted(run.name for run in fit_runs),
        "feature_standardizer_fit_scope": (
            "balanced_development_fit_healthy_and_matching_fault_"
            "prefix_plus_monitor_endpoints_only"
        ),
        "monitor_document_sha256": hashlib.sha256(
            _canonical_bytes(monitor_payload)
        ).hexdigest(),
        "held_out_25_or_75_percent_accessed": False,
    })


def _save_bound_artifacts(
    training_result: Any,
    standardizer: FittedExpertStandardizer,
    *,
    config: Mapping[str, Any],
    fit_runs: Sequence[PCLRRun],
    model_path: Path,
    monitor_path: Path,
) -> dict[str, Any]:
    if model_path.exists() or monitor_path.exists():
        raise FileExistsError("bound model/monitor destination already exists")
    monitor_payload = _monitor_payload(standardizer, config)
    metadata = _model_metadata(
        training_result,
        config=config,
        monitor_payload=monitor_payload,
        fit_runs=fit_runs,
    )
    _atomic_json(monitor_path, monitor_payload)
    training_result.model.save_npz(model_path, metadata=metadata)
    return metadata


def verify_bound_artifacts(
    *, model_path: Path, monitor_path: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    complete = _validate_config(config)
    model, raw_metadata = load_cf_glr_model(model_path)
    metadata = dict(raw_metadata)
    expected_runtime_model_config = asdict(CFGLRConfig())
    expected_model_config = _plain_json(expected_runtime_model_config)
    if asdict(model.config) != expected_runtime_model_config:
        raise ValueError("bound model config differs from the fixed v4 model")
    monitor_payload = _strict_json(monitor_path)
    expected_monitor_keys = {
        "schema", "backend_schema", "complete_config_sha256", "candidate_id",
        "standardizer", "fit_scope", "held_out_25_or_75_percent_accessed",
    }
    if set(monitor_payload) != expected_monitor_keys:
        raise ValueError("bound monitor fields changed")
    if (
        monitor_payload["schema"] != BOUND_MONITOR_SCHEMA
        or monitor_payload["backend_schema"] != BACKEND_SCHEMA
        or monitor_payload["complete_config_sha256"] != _config_sha256(complete)
        or monitor_payload["candidate_id"] != UNIQUE_CANDIDATE_ID
        or monitor_payload["fit_scope"]
        != "all_100_magnitude50_development_healthy_logits_only"
        or monitor_payload["held_out_25_or_75_percent_accessed"] is not False
    ):
        raise ValueError("bound monitor identity/scope changed")
    standardizer = FittedExpertStandardizer.from_document(
        monitor_payload["standardizer"]
    )
    expected_metadata_keys = {
        "schema", "backend_schema", "complete_config_sha256", "candidate_id",
        "model_config", "training_config", "final_losses",
        "fit_run_count_by_class", "fit_run_names",
        "feature_standardizer_fit_scope", "monitor_document_sha256",
        "held_out_25_or_75_percent_accessed",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError("bound model metadata fields changed")
    expected_training = asdict(CFGLRTrainingConfig())
    expected_names = sorted(
        "PCLR-v2-ExtendedTEP-Mode1-development-magnitude-50-"
        f"fault-{(0 if idv == 1 else idv):02d}-IDV-{idv:02d}-run-{run:03d}"
        for idv in ALL_IDVS for run in RUNS
    )
    names = metadata["fit_run_names"]
    if (
        metadata["schema"] != BOUND_MODEL_SCHEMA
        or metadata["backend_schema"] != BACKEND_SCHEMA
    ):
        raise ValueError("bound model metadata schema/config changed")
    if (
        metadata["complete_config_sha256"] != _config_sha256(complete)
        or metadata["candidate_id"] != UNIQUE_CANDIDATE_ID
        or metadata["model_config"] != expected_model_config
        or metadata["training_config"] != expected_training
        or metadata["fit_run_count_by_class"] != {"1": 100, "3": 100, "9": 100, "15": 100}
        or names != expected_names
        or metadata["feature_standardizer_fit_scope"]
        != "balanced_development_fit_healthy_and_matching_fault_prefix_plus_monitor_endpoints_only"
        or metadata["monitor_document_sha256"]
        != hashlib.sha256(_canonical_bytes(monitor_payload)).hexdigest()
        or metadata["held_out_25_or_75_percent_accessed"] is not False
    ):
        raise ValueError("bound model metadata identity/scope changed")
    losses = metadata["final_losses"]
    if (
        not isinstance(losses, Mapping)
        or set(losses) != {
            "fault_3_member_0", "fault_9_member_0",
            "fault_15_member_0", "fault_15_member_1", "fault_15_member_2",
        }
        or any(
            type(value) not in (int, float) or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in losses.values()
        )
    ):
        raise ValueError("bound final losses changed or are nonfinite")
    return {
        "model": model,
        "standardizer": standardizer,
        "model_metadata": metadata,
        "monitor_document": monitor_payload,
    }


def development_iteration(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    fold_map: Mapping[int, int],
    config: Mapping[str, Any],
    model_path: Path,
    monitor_path: Path,
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete = _validate_config(config)
    folds_by_run = _validate_fold_map(fold_map)
    phase, _ = _validate_access_plan(access_plan)
    if phase != "development":
        raise ValueError("development received a non-development plan")
    if model_path.exists() or monitor_path.exists():
        raise FileExistsError("development artifact destination already exists")
    runs = load_numeric_runs(source_path, access_plan)
    feature_cache = _build_feature_cache(runs)
    folds: list[dict[str, Any]] = []
    for outer in range(FOLD_COUNT):
        calibration_fold = (outer + 1) % FOLD_COUNT
        fit_ids = sorted(
            run for run, fold in folds_by_run.items()
            if fold not in (outer, calibration_fold)
        )
        calibration_ids = sorted(
            run for run, fold in folds_by_run.items() if fold == calibration_fold
        )
        test_ids = sorted(run for run, fold in folds_by_run.items() if fold == outer)
        fit_runs = _subset(runs, idvs=ALL_IDVS, simulation_runs=fit_ids)
        calibration_runs = _subset(
            runs, idvs=(1,), simulation_runs=calibration_ids
        )
        test_runs = _subset(runs, idvs=ALL_IDVS, simulation_runs=test_ids)
        print(
            f"[v4] outer {outer + 1}/5: fit 60/class; "
            f"calibrate 20 healthy; test 20/class",
            flush=True,
        )
        result = _fit_model(fit_runs, feature_cache)
        print(f"[v4] outer {outer + 1}/5: training complete; scoring", flush=True)
        fit_healthy = tuple(run for run in fit_runs if run.source_idv == 1)
        fit_logits = _score_model(
            result.model, fit_healthy, device="cuda", cache=feature_cache
        )
        standardizer = FittedExpertStandardizer.fit(fit_logits)
        calibration_gpu = _score_model(
            result.model, calibration_runs, device="cuda", cache=feature_cache
        )
        calibration_cpu = _score_model(
            result.model, calibration_runs, device="cpu", cache=feature_cache
        )
        calibration_error = float(np.max(np.abs(calibration_gpu - calibration_cpu)))
        if not math.isfinite(calibration_error) or calibration_error > PARITY_ATOL:
            raise RuntimeError("inner calibration CPU/CUDA parity failed")
        calibration = fit_rank20_threshold(calibration_gpu, standardizer)
        test_gpu = _score_model(
            result.model, test_runs, device="cuda", cache=feature_cache
        )
        test_cpu = _score_model(
            result.model, test_runs, device="cpu", cache=feature_cache
        )
        rows, test_error, discrete = _score_rows(
            test_runs, test_gpu, test_cpu, standardizer, calibration.threshold
        )
        parity = max(calibration_error, test_error)
        tolerance = max(FLOAT_TOLERANCE, parity)
        ordered = np.sort(calibration.run_maxima)
        distinct = int(1 + sum(
            current - previous > tolerance
            for previous, current in zip(ordered, ordered[1:])
        ))
        gap = calibration.threshold - calibration.rank19
        nondegenerate = bool(
            distinct >= 19
            and calibration.threshold > calibration.rank19
            and gap > 100.0 * tolerance
        )
        folds.append({
            "outer_fold": outer,
            "calibration_fold": calibration_fold,
            "fit_run_ids_by_class": {str(idv): fit_ids for idv in ALL_IDVS},
            "calibration_healthy_run_ids": calibration_ids,
            "excluded_calibration_fault_run_ids_by_class": {
                str(idv): calibration_ids for idv in FAULT_IDVS
            },
            "test_run_ids_by_class": {str(idv): test_ids for idv in ALL_IDVS},
            "calibration_healthy_maxima": calibration.run_maxima.tolist(),
            "threshold": calibration.threshold,
            "rank19": calibration.rank19,
            "distinct_maxima": distinct,
            "adjacent_tail_gap": gap,
            "calibration_nondegenerate": nondegenerate,
            "rows": rows,
            "score_parity_max_abs_error": parity,
            "discrete_parity_equal": discrete,
        })
        print(
            f"[v4] outer {outer + 1}/5: complete "
            f"(threshold={calibration.threshold:.6g}, "
            f"nondegenerate={nondegenerate}, parity={parity:.3e})",
            flush=True,
        )
        del result
        _torch.cuda.empty_cache()

    selected = UNIQUE_CANDIDATE_ID if _development_pass(folds) else None
    metadata: Mapping[str, Any] | None = None
    final_parity: float | None = None
    if selected is not None:
        print("[v4] all development gates passed; final 100/class refit", flush=True)
        final = _fit_model(runs, feature_cache)
        final_gpu = _score_model(
            final.model, runs, device="cuda", cache=feature_cache
        )
        final_cpu = _score_model(
            final.model, runs, device="cpu", cache=feature_cache
        )
        final_parity = float(np.max(np.abs(final_gpu - final_cpu)))
        if not math.isfinite(final_parity) or final_parity > PARITY_ATOL:
            raise RuntimeError("final refit CPU/CUDA parity failed")
        healthy_indices = [
            index for index, run in enumerate(runs) if run.source_idv == 1
        ]
        standardizer = FittedExpertStandardizer.fit(final_gpu[healthy_indices])
        metadata = _save_bound_artifacts(
            final,
            standardizer,
            config=complete,
            fit_runs=runs,
            model_path=model_path,
            monitor_path=monitor_path,
        )
        print(
            f"[v4] final refit/model-monitor save complete "
            f"(parity={final_parity:.3e})",
            flush=True,
        )
    else:
        print("[v4] development gates failed; no model/monitor saved", flush=True)
    return {
        "schema": "cf_max_glr15_development_backend_v1",
        "config": complete,
        "candidate_id": UNIQUE_CANDIDATE_ID,
        "folds": folds,
        "runtime": dict(cuda_preflight()),
        "selected_candidate_id": selected,
        "model_metadata": None if metadata is None else dict(metadata),
        "model_parity_max_abs_error": final_parity,
    }


def _load_bound(
    model_path: Path, monitor_path: Path, config: Mapping[str, Any]
) -> tuple[CFGLRModel, FittedExpertStandardizer]:
    verified = verify_bound_artifacts(
        model_path=model_path, monitor_path=monitor_path, config=config
    )
    model = verified["model"]
    standardizer = verified["standardizer"]
    if not isinstance(model, CFGLRModel) or not isinstance(
        standardizer, FittedExpertStandardizer
    ):
        raise RuntimeError("bound artifact verifier returned invalid numeric objects")
    return model, standardizer


def calibrate(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    model_path: Path,
    monitor_path: Path,
    config: Mapping[str, Any],
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete = _validate_config(config)
    phase, _ = _validate_access_plan(access_plan)
    if phase != "calibrate":
        raise ValueError("calibration received a non-calibration plan")
    model, standardizer = _load_bound(model_path, monitor_path, complete)
    runs = load_numeric_runs(source_path, access_plan)
    feature_cache = _build_feature_cache(runs)
    gpu = _score_model(model, runs, device="cuda", cache=feature_cache)
    cpu = _score_model(model, runs, device="cpu", cache=feature_cache)
    rows: list[dict[str, Any]] = []
    gpu_maxima: list[float] = []
    cpu_maxima: list[float] = []
    maximum_error = 0.0
    for run, gpu_run, cpu_run in zip(runs, gpu, cpu, strict=True):
        error = float(np.max(np.abs(gpu_run - cpu_run)))
        if not math.isfinite(error) or error > PARITY_ATOL:
            raise RuntimeError("formal calibration CPU/CUDA parity failed")
        maximum_error = max(maximum_error, error)
        gpu_maximum = fused_monitor_path(gpu_run, standardizer).run_maximum
        cpu_maximum = fused_monitor_path(cpu_run, standardizer).run_maximum
        gpu_maxima.append(gpu_maximum)
        cpu_maxima.append(cpu_maximum)
        rows.append({
            "idv": run.source_idv,
            "simulation_run": run.simulation_run,
            "run_maximum": gpu_maximum,
            "parity_max_abs_error": error,
        })
    gpu_order = np.argsort(np.asarray(gpu_maxima), kind="stable")
    cpu_order = np.argsort(np.asarray(cpu_maxima), kind="stable")
    if not np.array_equal(gpu_order, cpu_order):
        raise RuntimeError("formal calibration CPU/CUDA run ordering differs")
    return {
        "schema": "cf_max_glr15_calibration_backend_v1",
        "config": complete,
        "rows": rows,
        "model_parity_max_abs_error": maximum_error,
        "discrete_parity_equal": True,
    }


def confirm(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    model_path: Path,
    monitor_path: Path,
    config: Mapping[str, Any],
    threshold: float,
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete = _validate_config(config)
    if isinstance(threshold, bool) or type(threshold) not in (int, float):
        raise ValueError("confirmation threshold must be a JSON number")
    boundary = float(threshold)
    if not math.isfinite(boundary):
        raise ValueError("confirmation threshold must be finite")
    phase, _ = _validate_access_plan(access_plan)
    if phase != "confirm":
        raise ValueError("confirmation received a non-confirmation plan")
    model, standardizer = _load_bound(model_path, monitor_path, complete)
    runs = load_numeric_runs(source_path, access_plan)
    feature_cache = _build_feature_cache(runs)
    gpu = _score_model(model, runs, device="cuda", cache=feature_cache)
    cpu = _score_model(model, runs, device="cpu", cache=feature_cache)
    rows, parity, discrete = _score_rows(
        runs, gpu, cpu, standardizer, boundary
    )
    if not discrete:
        raise RuntimeError("confirmation CPU/CUDA alarm/attribution outputs differ")
    return {
        "schema": "cf_max_glr15_confirmation_backend_v1",
        "config": complete,
        "threshold": boundary,
        "rows": rows,
        "model_parity_max_abs_error": parity,
        "discrete_parity_equal": True,
    }


__all__ = [
    "calibrate",
    "confirm",
    "cuda_preflight",
    "development_iteration",
    "load_numeric_runs",
    "verify_bound_artifacts",
]
