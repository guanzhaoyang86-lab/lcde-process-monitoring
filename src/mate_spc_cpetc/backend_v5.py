"""Governed five-fold backend for CP-ETC-Scan v5.1.

PyTorch is imported before NumPy for the project's Windows CUDA environment.
The HDF5 dependency remains lazy in ``data_v5`` and is reached only after the
runner has authenticated the protocol bundle and committed an access intent.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .data_v5 import (
    CONDITION_IDVS,
    FAULT_IDVS,
    MODE2_DEVELOPMENT_ALLOCATION,
    MODE2_SHA256,
    PROTOCOL_SHA256,
    DevelopmentBlock,
    Mode2DevelopmentArchive,
)
from .monitor import (
    EXPERT_COUNT,
    FAULT_IDS,
    MONITOR_COUNT,
    MONITOR_ENDPOINTS,
    ExpertTailCalibrator,
    apply_strict_threshold,
    fused_monitor_path,
)

BACKEND_SCHEMA = "cp_etc_scan_v5_1_development_backend_v1"
PROTOCOL_ID = "cp-etc-scan-v5.1-mode2-development"
BASE_PROTOCOL_ID = "cp-etc-scan-v5-mode2-development"
CANDIDATE_ID = "cp-etc-scan-v5-mode2-fixed"
RANDOM_SEED = 20_260_912
FOLD_COUNT = 5
ALL_ENDPOINT_COUNT = 93
PREFIX_ENDPOINT_COUNT = 8
PARITY_ATOL = 1.0e-5

V51_FOLD_RUN_IDS: Mapping[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}
EXPECTED_FOLD_MAP_SHA256 = "64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13"
EXPECTED_ACCESS_PLAN_SHA256 = "387ece90e94e7161767374be34ed6d032c10f45a60353c9de74be7b493ca173f"


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON values are forbidden")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not strict JSON")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        _plain_json(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
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


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_bytes(value))


def _canonical_fold_map() -> dict[int, int]:
    result = {
        run_id: fold_id for fold_id, run_ids in V51_FOLD_RUN_IDS.items()
        for run_id in run_ids
    }
    if tuple(sorted(result)) != tuple(range(1, 101)) or any(
        len(V51_FOLD_RUN_IDS[index]) != 20 for index in range(FOLD_COUNT)
    ):
        raise RuntimeError("embedded v5.1 fold assignment is malformed")
    return result


def _fold_map_sha256_unchecked(fold_map: Mapping[int, int]) -> str:
    payload = {str(run_id): int(fold_map[run_id]) for run_id in sorted(fold_map)}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fold_map_sha256(fold_map: Mapping[int, int] | None = None) -> str:
    canonical = _canonical_fold_map() if fold_map is None else _validate_fold_map(fold_map)
    return _fold_map_sha256_unchecked(canonical)


def _validate_fold_map(value: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("fold map must be a mapping")
    if any(isinstance(key, bool) or isinstance(fold, bool) for key, fold in value.items()):
        raise ValueError("boolean fold entries are forbidden")
    try:
        candidate = {int(key): int(fold) for key, fold in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("fold map contains non-integer entries") from error
    if candidate != _canonical_fold_map():
        raise ValueError("fold map differs from the exact v5.1 seed-component assignment")
    if _fold_map_sha256_unchecked(candidate) != EXPECTED_FOLD_MAP_SHA256:
        raise ValueError("v5.1 canonical run-to-fold hash mismatch")
    return candidate


def fold_roles(outer_fold: int) -> dict[str, list[int]]:
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int):
        raise ValueError("outer fold must be an integer")
    if outer_fold not in range(FOLD_COUNT):
        raise ValueError("outer fold must be in 0..4")
    inner_fold = (outer_fold + 1) % FOLD_COUNT
    fit = sorted(
        run_id for fold_id, run_ids in V51_FOLD_RUN_IDS.items()
        if fold_id not in (outer_fold, inner_fold) for run_id in run_ids
    )
    return {
        "fit": fit,
        "inner_healthy": list(V51_FOLD_RUN_IDS[inner_fold]),
        "outer": list(V51_FOLD_RUN_IDS[outer_fold]),
    }


def development_access_plan() -> list[dict[str, Any]]:
    return [{
        "mode": 2,
        "fault_regime": "SingleFault",
        "magnitude_percent": 50,
        "idv": idv,
        "simulation_run": run_id,
        "window_half_open": [100, 600] if idv == 1 else [440, 940],
        "role": "cross_operating_mode_development_only",
    } for run_id in range(1, 101) for idv in CONDITION_IDVS]


_DEVELOPMENT_PLAN = tuple(development_access_plan())


def validate_development_access_plan(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("development access plan must be a sequence")
    try:
        candidate = [_plain_json(dict(row)) for row in value]
    except (TypeError, ValueError) as error:
        raise ValueError("development access plan is malformed") from error
    expected = [dict(row) for row in _DEVELOPMENT_PLAN]
    if candidate != expected:
        raise ValueError("development access plan differs from the frozen Mode2 allocation")
    if _sha256_document(candidate) != EXPECTED_ACCESS_PLAN_SHA256:
        raise ValueError("development access-plan hash mismatch")
    return candidate


def _configure_determinism() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _require_cuda(device: str) -> None:
    if device != "cuda":
        raise ValueError("formal v5.1 development training requires device='cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal v5.1 development")


def cuda_preflight() -> Mapping[str, Any]:
    available = bool(torch.cuda.is_available())
    return {
        "cuda_available": available,
        "device_type": "cuda" if available else None,
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if available else None,
        "network_dtype": "float32",
        "tail_dtype": "float64",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def load_numeric_blocks(
    access_plan: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[DevelopmentBlock, ...]]:
    """Validate the plan before constructing or opening the HDF5 archive."""
    validate_development_access_plan(access_plan)
    archive = Mode2DevelopmentArchive()
    topology = archive.validate_topology(MODE2_DEVELOPMENT_ALLOCATION)
    blocks = archive.load_development_blocks(MODE2_DEVELOPMENT_ALLOCATION)
    return _plain_json(vars(topology)), blocks


def _run_id(run: Any) -> int:
    return int(run.simulation_run)


def _run_idv(run: Any) -> int:
    return int(run.source_idv)


def _run_fault(run: Any) -> int:
    return int(run.fault_id) if hasattr(run, "fault_id") else (
        0 if _run_idv(run) == 1 else _run_idv(run)
    )


def _validate_blocks(blocks: Sequence[Any]) -> tuple[Any, ...]:
    if len(blocks) != 100:
        raise ValueError("v5.1 development requires exactly 100 balanced blocks")
    ordered = tuple(sorted(blocks, key=lambda item: int(item.simulation_run)))
    if tuple(int(block.simulation_run) for block in ordered) != tuple(range(1, 101)):
        raise ValueError("development blocks must be exactly Run1..100")
    seen_names: set[str] = set()
    for block in ordered:
        conditions = tuple(block.conditions)
        if tuple(_run_idv(run) for run in conditions) != tuple(CONDITION_IDVS):
            raise ValueError("each block must contain IDV1/3/9/15 exactly once")
        for run in conditions:
            if _run_id(run) != int(block.simulation_run):
                raise ValueError("condition run ID differs from its balanced block")
            name = str(run.name)
            if name in seen_names:
                raise ValueError("duplicate development run identity")
            seen_names.add(name)
            if hasattr(run, "x"):
                values = np.asarray(run.x)
                if values.shape != (500, 52) or not np.isfinite(values).all():
                    raise ValueError("development sequence must be finite 500x52")
    if len(seen_names) != 400:
        raise ValueError("development requires exactly 400 unique run objects")
    return ordered


def _select_runs(
    blocks: Sequence[Any], run_ids: Sequence[int], idvs: Sequence[int],
) -> tuple[Any, ...]:
    selected_ids = set(int(value) for value in run_ids)
    selected_idvs = tuple(int(value) for value in idvs)
    result = tuple(
        run for block in blocks if int(block.simulation_run) in selected_ids
        for run in block.conditions if _run_idv(run) in selected_idvs
    )
    if len(result) != len(selected_ids) * len(selected_idvs):
        raise RuntimeError("fold selection count changed")
    return result


def _fit_model(runs: Sequence[Any], *, device: str) -> Any:
    from .model import fit_cpetc_model
    return fit_cpetc_model(tuple(runs), device=device)


def _score_model(fitted: Any, runs: Sequence[Any], *, device: str) -> np.ndarray:
    from .model import score_cpetc_logits
    result = score_cpetc_logits(fitted, tuple(runs), device=device)
    if isinstance(result, torch.Tensor):
        result = result.detach().cpu().numpy()
    values = np.asarray(result, dtype=np.float64)
    expected = (len(runs), ALL_ENDPOINT_COUNT, EXPERT_COUNT)
    if values.shape != expected or not np.isfinite(values).all():
        raise RuntimeError(f"model logits must be finite with shape {expected}")
    return values


def _save_checkpoint(fitted: Any, path: Path) -> Mapping[str, Any]:
    from .model import save_cpetc_checkpoint
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = save_cpetc_checkpoint(fitted, path)
    if not path.is_file():
        raise RuntimeError("checkpoint saver did not create its destination")
    return {
        "path": str(path.resolve()), "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "metadata": {} if raw is None else _plain_json(dict(raw)),
    }


def _component_audit_hashes(fitted: Any) -> Mapping[str, str]:
    from .model import component_audit_hashes
    raw = dict(component_audit_hashes(fitted))
    if set(raw) != {"normalizer", "autoencoder", "network", "loop_reference"}:
        raise RuntimeError("model component audit hashes are incomplete")
    model = fitted.model if hasattr(fitted, "model") else fitted
    loop = model.require_fitted().to_document()
    dwell = {
        "dwell_threshold": loop["dwell_threshold"],
        "dwell_quantile": loop["dwell_quantile"],
    }
    arx = {
        "arx_coefficients": loop["arx_coefficients"],
        "residual_denominators": loop["residual_denominators"],
        "ridge_diagonal": loop["ridge_diagonal"],
    }
    value = {
        "normalization": raw["normalizer"],
        "dwell": _sha256_document(dwell),
        "arx": _sha256_document(arx),
        "autoencoder": raw["autoencoder"],
        "classifier": raw["network"],
    }
    if any(not isinstance(item, str) or len(item) != 64 for item in value.values()):
        raise RuntimeError("model component hash format changed")
    return value


def _require_training_contract(fitted: Any, *, runs: int, healthy: int) -> None:
    required = {
        "training_run_count": runs,
        "healthy_run_count": healthy,
        "requested_device": "cuda",
        "dtype": "float32",
    }
    if any(getattr(fitted, key, None) != value for key, value in required.items()):
        raise RuntimeError("training result identity/count/device differs from protocol")
    ae = tuple(float(value) for value in getattr(fitted, "ae_loss_history", ()))
    mil = tuple(float(value) for value in getattr(fitted, "mil_loss_history", ()))
    if len(ae) != 40 or len(mil) != 60 or not all(
        math.isfinite(value) for value in (*ae, *mil)
    ):
        raise RuntimeError("training must retain exactly 40 AE and 60 MIL epochs")


def _parameter_counts(fitted: Any) -> dict[str, int]:
    model = fitted.model if hasattr(fitted, "model") else fitted
    classifier = sum(parameter.numel() for parameter in model.network.parameters())
    autoencoder = sum(parameter.numel() for parameter in model.autoencoder.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    counts = {
        "classifier_exact": int(classifier),
        "autoencoder_exact": int(autoencoder),
        "total_exact": int(total),
        "trainable_at_checkpoint": int(trainable),
    }
    if counts != {
        "classifier_exact": 9_207,
        "autoencoder_exact": 6_604,
        "total_exact": 15_811,
        "trainable_at_checkpoint": 9_207,
    }:
        raise RuntimeError(f"exact v5 model parameter counts changed: {counts}")
    return counts


def _monitor_logits(all_logits: np.ndarray) -> np.ndarray:
    values = np.asarray(all_logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (ALL_ENDPOINT_COUNT, EXPERT_COUNT):
        raise ValueError("all endpoint logits have invalid geometry")
    result = values[:, PREFIX_ENDPOINT_COUNT:, :]
    if result.shape[1:] != (MONITOR_COUNT, EXPERT_COUNT):
        raise RuntimeError("monitor endpoint slicing changed")
    return result


def _operational_threshold(
    inner_healthy_monitor_logits: np.ndarray,
    calibrator: ExpertTailCalibrator,
) -> dict[str, Any]:
    """Rank-20 threshold using one frozen fit-healthy ECDF (never LOO)."""
    values = np.asarray(inner_healthy_monitor_logits, dtype=np.float64)
    if values.shape != (20, MONITOR_COUNT, EXPERT_COUNT):
        raise ValueError("inner threshold requires 20 healthy monitor-logit paths")
    maxima = np.asarray([
        fused_monitor_path(value, calibrator).run_maximum for value in values
    ], dtype=np.float64)
    if not np.isfinite(maxima).all():
        raise FloatingPointError("inner threshold maxima are nonfinite")
    ordered = np.sort(maxima)
    threshold = float(ordered[-1])
    rank19 = float(ordered[-2])
    tolerance = 100.0 * np.finfo(np.float64).eps * max(
        1.0, abs(threshold), abs(rank19)
    )
    distinct = int(np.unique(maxima).size)
    gap = threshold - rank19
    return {
        "procedure": "fixed_fit_healthy_ecdf_then_inner_rank20",
        "operational_loo": False,
        "run_maxima": maxima.tolist(),
        "sorted_run_maxima": ordered.tolist(),
        "threshold": threshold,
        "rank19": rank19,
        "distinct_maxima": distinct,
        "adjacent_tail_gap": gap,
        "numerical_tolerance": tolerance,
        "nondegenerate": bool(distinct >= 19 and gap > tolerance),
    }


def _path_difference(left: Any, right: Any) -> tuple[dict[str, float], bool]:
    errors = {
        "pvalues": float(np.max(np.abs(left.pvalues - right.pvalues))),
        "fused_pvalues": float(np.max(np.abs(
            left.fused_pvalues - right.fused_pvalues
        ))),
        "evidence": float(np.max(np.abs(left.evidence - right.evidence))),
        "scan": float(np.max(np.abs(left.scan_scores - right.scan_scores))),
        "run_maximum": abs(float(left.run_maximum) - float(right.run_maximum)),
    }
    discrete = bool(
        np.array_equal(left.winning_expert_indices, right.winning_expert_indices)
        and np.array_equal(left.eligible, right.eligible)
    )
    return errors, discrete


def _score_outer_rows(
    runs: Sequence[Any], gpu_logits: np.ndarray, cpu_logits: np.ndarray,
    calibrator: ExpertTailCalibrator, threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = (len(runs), ALL_ENDPOINT_COUNT, EXPERT_COUNT)
    if gpu_logits.shape != expected or cpu_logits.shape != expected:
        raise ValueError("CPU/CUDA outer logits geometry differs")
    rows: list[dict[str, Any]] = []
    maximums = {
        "logits": 0.0, "pvalues": 0.0, "fused_pvalues": 0.0,
        "evidence": 0.0, "scan": 0.0, "run_maximum": 0.0,
    }
    alarm_equal = True
    winner_equal = True
    for run, gpu_all, cpu_all in zip(runs, gpu_logits, cpu_logits, strict=True):
        logit_error = float(np.max(np.abs(gpu_all - cpu_all)))
        maximums["logits"] = max(maximums["logits"], logit_error)
        gpu_path = fused_monitor_path(gpu_all[PREFIX_ENDPOINT_COUNT:], calibrator)
        cpu_path = fused_monitor_path(cpu_all[PREFIX_ENDPOINT_COUNT:], calibrator)
        errors, path_discrete = _path_difference(gpu_path, cpu_path)
        for key, value in errors.items():
            maximums[key] = max(maximums[key], value)
        gpu_decision = apply_strict_threshold(gpu_path, threshold)
        cpu_decision = apply_strict_threshold(cpu_path, threshold)
        same_alarm = (
            gpu_decision.alarm == cpu_decision.alarm
            and gpu_decision.first_alarm_endpoint == cpu_decision.first_alarm_endpoint
            and gpu_decision.attributed_fault == cpu_decision.attributed_fault
        )
        alarm_equal = alarm_equal and same_alarm
        winner_equal = winner_equal and path_discrete
        winners = [FAULT_IDS[int(index)] for index in gpu_path.winning_expert_indices]
        rows.append({
            "name": str(run.name),
            "idv": _run_idv(run),
            "actual_fault": _run_fault(run),
            "simulation_run": _run_id(run),
            "rng_seed": int(run.rng_seed) if hasattr(run, "rng_seed") else None,
            "all_endpoints": list(range(128, 497, 4)),
            "all_logits": gpu_all.tolist(),
            "monitor_endpoints": list(MONITOR_ENDPOINTS),
            "pvalues": gpu_path.pvalues.tolist(),
            "fused_pvalues": gpu_path.fused_pvalues.tolist(),
            "evidence": gpu_path.evidence.tolist(),
            "scan_scores": gpu_path.scan_scores.tolist(),
            "winning_experts": winners,
            "decision": {
                "alarm": gpu_decision.alarm,
                "first_alarm_position": gpu_decision.first_alarm_position,
                "first_alarm_endpoint": gpu_decision.first_alarm_endpoint,
                "attributed_fault": gpu_decision.attributed_fault,
                "delay_samples": gpu_decision.delay,
                "delay_hours": 0.05 * gpu_decision.delay,
                "run_maximum": gpu_decision.run_maximum,
                "threshold": gpu_decision.threshold,
            },
            "cpu_cuda_max_abs": {"logits": logit_error, **errors},
            "cpu_cuda_alarm_equal": same_alarm,
            "cpu_cuda_winners_equal": path_discrete,
        })
    all_finite = all(math.isfinite(value) for value in maximums.values())
    overall = max(maximums.values())
    parity = {
        "max_absolute_difference": maximums,
        "overall_max_absolute_difference": overall,
        "tolerance": PARITY_ATOL,
        "alarm_arrays_exact": bool(alarm_equal),
        "winner_arrays_exact": bool(winner_equal),
        "all_outputs_finite": all_finite,
        "passes": bool(
            all_finite and overall <= PARITY_ATOL and alarm_equal and winner_equal
        ),
    }
    return rows, parity


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("nearest rank requires at least one value")
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(float(probability) * len(ordered))) - 1]


def _winner_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {str(fault): 0 for fault in FAULT_IDS}
    for row in rows:
        if int(row["idv"]) == 1:
            for winner in row["winning_experts"]:
                counts[str(int(winner))] += 1
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("healthy winner population is empty")
    shares = {key: value / total for key, value in counts.items()}
    return {
        "population_endpoints": total,
        "counts": counts,
        "shares": shares,
        "maximum_share": max(shares.values()),
    }


def _fold_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    healthy = [row for row in rows if int(row["idv"]) == 1]
    false_alarms = sum(bool(row["decision"]["alarm"]) for row in healthy)
    faults: dict[str, Any] = {}
    for fault in FAULT_IDS:
        selected = [row for row in rows if int(row["idv"]) == fault]
        detections = sum(bool(row["decision"]["alarm"]) for row in selected)
        delays = [int(row["decision"]["delay_samples"]) for row in selected]
        faults[str(fault)] = {
            "runs": len(selected),
            "detections": detections,
            "detection_rate": detections / len(selected),
            "median_delay_samples": float(statistics.median(delays)),
            "nearest_rank_p90_delay_samples": _nearest_rank(delays, 0.90),
        }
    return {
        "healthy": {
            "runs": len(healthy), "false_alarms": false_alarms,
            "false_alarm_rate": false_alarms / len(healthy),
        },
        "faults": faults,
        "healthy_winner_dominance": _winner_summary(rows),
    }


def _aggregate_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for fold in folds for row in fold["outer_rows"]]
    result = _fold_metrics(rows)
    confusion = {
        str(actual): {"3": 0, "9": 0, "15": 0, "none": 0}
        for actual in FAULT_IDS
    }
    for row in rows:
        actual = int(row["idv"])
        if actual in FAULT_IDS:
            attributed = row["decision"]["attributed_fault"]
            key = "none" if attributed is None else str(int(attributed))
            confusion[str(actual)][key] += 1
    result["attribution_confusion_report_only"] = confusion
    return result


def _gate_report(
    folds: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    fold_false_alarm = [
        int(fold["metrics"]["healthy"]["false_alarms"]) for fold in folds
    ]
    healthy_total = int(aggregate["healthy"]["false_alarms"])
    healthy_gate = healthy_total <= 10 and all(value <= 4 for value in fold_false_alarm)
    fault_gates: dict[str, Any] = {}
    for fault in FAULT_IDS:
        key = str(fault)
        metrics = aggregate["faults"][key]
        fold_rates = [
            float(fold["metrics"]["faults"][key]["detection_rate"])
            for fold in folds
        ]
        checks = {
            "aggregate_detection_rate": float(metrics["detection_rate"]) >= 0.85,
            "per_fold_detection_rate": all(value >= 0.70 for value in fold_rates),
            "aggregate_median_delay": float(metrics["median_delay_samples"]) <= 140.0,
            "aggregate_nearest_rank_p90_delay": float(
                metrics["nearest_rank_p90_delay_samples"]
            ) <= 220.0,
        }
        fault_gates[key] = {**checks, "passes": all(checks.values())}
    aggregate_dominance = float(
        aggregate["healthy_winner_dominance"]["maximum_share"]
    )
    fold_dominance = [
        float(fold["metrics"]["healthy_winner_dominance"]["maximum_share"])
        for fold in folds
    ]
    winner_gate = aggregate_dominance <= 0.60 and all(
        value <= 0.70 for value in fold_dominance
    )
    threshold_gate = all(bool(fold["inner_threshold"]["nondegenerate"]) for fold in folds)
    parity_gate = all(bool(fold["parity"]["passes"]) for fold in folds)
    checks = {
        "healthy_false_alarm": bool(healthy_gate),
        "fault_3": bool(fault_gates["3"]["passes"]),
        "fault_9": bool(fault_gates["9"]["passes"]),
        "fault_15": bool(fault_gates["15"]["passes"]),
        "healthy_winner_dominance": bool(winner_gate),
        "threshold_nondegeneracy": bool(threshold_gate),
        "cpu_cuda_parity": bool(parity_gate),
    }
    return {
        "healthy": {
            "overall_false_alarms": healthy_total,
            "per_fold_false_alarms": fold_false_alarm,
            "overall_max": 10, "per_fold_max": 4,
            "passes": bool(healthy_gate),
        },
        "faults": fault_gates,
        "winner_dominance": {
            "aggregate_maximum_share": aggregate_dominance,
            "per_fold_maximum_shares": fold_dominance,
            "aggregate_limit": 0.60, "per_fold_limit": 0.70,
            "passes": bool(winner_gate),
        },
        "threshold_nondegeneracy": bool(threshold_gate),
        "cpu_cuda_parity": bool(parity_gate),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def _save_fold_artifacts(
    *, fitted: Any, calibrator: ExpertTailCalibrator,
    threshold: Mapping[str, Any], fold_dir: Path,
) -> dict[str, Any]:
    checkpoint = _save_checkpoint(fitted, fold_dir / "checkpoint.pt")
    components = dict(_component_audit_hashes(fitted))
    ecdf_document = {
        "schema": "cp_etc_scan_v5_1_fit_healthy_ecdf_v1",
        "fit_healthy_runs": calibrator.calibration_run_count,
        "endpoint_count": calibrator.endpoint_count,
        "samples_per_expert": calibrator.samples_per_expert,
        "fault_ids": list(calibrator.fault_ids),
        "sorted_logits": calibrator.sorted_logits.tolist(),
    }
    threshold_document = {
        "schema": "cp_etc_scan_v5_1_inner_threshold_v1", **dict(threshold),
    }
    ecdf_path = fold_dir / "fit_healthy_ecdf.json"
    threshold_path = fold_dir / "inner_threshold.json"
    components_path = fold_dir / "component_hashes.json"
    _atomic_json(ecdf_path, ecdf_document)
    _atomic_json(threshold_path, threshold_document)
    _atomic_json(components_path, components)
    return {
        "checkpoint": checkpoint,
        "component_hashes": components,
        "component_hash_document_sha256": _sha256_file(components_path),
        "ecdf_sha256": _sha256_file(ecdf_path),
        "threshold_sha256": _sha256_file(threshold_path),
    }


def _save_final_artifacts(
    *, fitted: Any, calibrator: ExpertTailCalibrator, artifact_root: Path,
) -> dict[str, Any]:
    final_dir = artifact_root / "formal_mode2_refit"
    checkpoint = _save_checkpoint(fitted, final_dir / "model.pt")
    components = dict(_component_audit_hashes(fitted))
    ecdf_document = {
        "schema": "cp_etc_scan_v5_1_final_mode2_ecdf_v1",
        "fit_healthy_runs": calibrator.calibration_run_count,
        "endpoint_count": calibrator.endpoint_count,
        "samples_per_expert": calibrator.samples_per_expert,
        "fault_ids": list(calibrator.fault_ids),
        "sorted_logits": calibrator.sorted_logits.tolist(),
    }
    ecdf_path = final_dir / "mode2_healthy_ecdf.json"
    component_path = final_dir / "component_hashes.json"
    _atomic_json(ecdf_path, ecdf_document)
    _atomic_json(component_path, components)
    binding = {
        "schema": "cp_etc_scan_v5_1_final_mode2_refit_binding_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": MODE2_SHA256,
        "candidate_id": CANDIDATE_ID,
        "checkpoint_sha256": checkpoint["sha256"],
        "ecdf_sha256": _sha256_file(ecdf_path),
        "component_hashes": components,
        "mode1_accessed": False,
    }
    binding_path = final_dir / "binding.json"
    _atomic_json(binding_path, binding)
    return {
        **binding,
        "binding_sha256": _sha256_file(binding_path),
        "checkpoint": checkpoint,
        "paths": {
            "checkpoint": str((final_dir / "model.pt").resolve()),
            "ecdf": str(ecdf_path.resolve()),
            "binding": str(binding_path.resolve()),
        },
    }


def _is_fold_numeric_no_go(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "arx ridge system",
        "arx fit produced nonfinite",
        "linalg.solve",
        "linear solve",
    ))


def _fold_numeric_no_go_result(
    *, topology: Mapping[str, Any], completed_folds: Sequence[Mapping[str, Any]],
    outer_fold: int, inner_fold: int, roles: Mapping[str, Sequence[int]],
    error: BaseException, artifact_root: Path,
) -> Mapping[str, Any]:
    document = {
        "schema": "cp_etc_scan_v5_1_fold_numeric_no_go_v1",
        "decision": "DEVELOPMENT_NO_GO",
        "outer_fold": outer_fold,
        "inner_calibration_fold": inner_fold,
        "fit_run_ids_by_class": {
            str(idv): list(roles["fit"]) for idv in CONDITION_IDVS
        },
        "error_type": type(error).__name__,
        "error_message": str(error),
        "outer_data_scored": False,
        "rescue_allowed": False,
    }
    path = artifact_root / f"fold_{outer_fold}" / "fold_numeric_no_go.json"
    _atomic_json(path, document)
    return _plain_json({
        "schema": BACKEND_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "base_protocol_id": BASE_PROTOCOL_ID,
        "source_sha256": MODE2_SHA256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "fold_map_sha256": EXPECTED_FOLD_MAP_SHA256,
        "topology": topology,
        "runtime": dict(cuda_preflight()),
        "folds": list(completed_folds),
        "aggregate_metrics": None,
        "gates": {"fold_numeric_guard": False, "all_pass": False},
        "decision": "DEVELOPMENT_NO_GO",
        "selected_candidate_id": None,
        "formal_mode2_refit": None,
        "termination": {
            "kind": "FOLD_NUMERIC_NO_GO",
            "outer_fold": outer_fold,
            "record_path": str(path.resolve()),
            "record_sha256": _sha256_file(path),
            "outer_data_scored": False,
        },
        "mode1_accessed": False,
    })


def development_iteration(
    *, access_plan: Sequence[Mapping[str, Any]], fold_map: Mapping[int, int],
    artifact_root: Path, device: str = "cuda",
) -> Mapping[str, Any]:
    """Run the one authorized OOF evaluation, then refit only after a GO."""
    _require_cuda(device)
    validate_development_access_plan(access_plan)
    canonical_folds = _validate_fold_map(fold_map)
    if _fold_map_sha256_unchecked(canonical_folds) != EXPECTED_FOLD_MAP_SHA256:
        raise ValueError("fold binding mismatch")
    if artifact_root.exists() and any(artifact_root.iterdir()):
        raise FileExistsError("backend artifact root must be absent or empty")
    artifact_root.mkdir(parents=True, exist_ok=True)
    _configure_determinism()
    topology, raw_blocks = load_numeric_blocks(access_plan)
    blocks = _validate_blocks(raw_blocks)
    folds: list[dict[str, Any]] = []

    for outer_fold in range(FOLD_COUNT):
        roles = fold_roles(outer_fold)
        inner_fold = (outer_fold + 1) % FOLD_COUNT
        fit_runs = _select_runs(blocks, roles["fit"], CONDITION_IDVS)
        fit_healthy = tuple(run for run in fit_runs if _run_idv(run) == 1)
        inner_healthy = _select_runs(blocks, roles["inner_healthy"], (1,))
        outer_runs = _select_runs(blocks, roles["outer"], CONDITION_IDVS)
        print(
            f"[CP-ETC v5.1] fold {outer_fold + 1}/5: "
            "fit 60/class, threshold 20 healthy, outer 20/class",
            flush=True,
        )
        _configure_determinism()
        try:
            fitted = _fit_model(fit_runs, device=device)
        except (ValueError, FloatingPointError, RuntimeError) as error:
            if not _is_fold_numeric_no_go(error):
                raise
            return _fold_numeric_no_go_result(
                topology=topology, completed_folds=folds,
                outer_fold=outer_fold, inner_fold=inner_fold, roles=roles,
                error=error, artifact_root=artifact_root,
            )
        _require_training_contract(fitted, runs=240, healthy=60)
        parameter_counts = _parameter_counts(fitted)
        fit_logits = _score_model(fitted, fit_healthy, device="cuda")
        calibrator = ExpertTailCalibrator.fit(_monitor_logits(fit_logits))
        if calibrator.samples_per_expert != 5_100:
            raise RuntimeError("fold ECDF must contain exactly 5100 scores per expert")
        inner_gpu = _score_model(fitted, inner_healthy, device="cuda")
        inner_cpu = _score_model(fitted, inner_healthy, device="cpu")
        threshold = _operational_threshold(_monitor_logits(inner_gpu), calibrator)
        inner_cpu_threshold = _operational_threshold(
            _monitor_logits(inner_cpu), calibrator
        )
        inner_error = float(np.max(np.abs(inner_gpu - inner_cpu)))
        inner_max_error = float(np.max(np.abs(
            np.asarray(threshold["run_maxima"], dtype=np.float64)
            - np.asarray(inner_cpu_threshold["run_maxima"], dtype=np.float64)
        )))
        threshold_error = abs(
            float(threshold["threshold"])
            - float(inner_cpu_threshold["threshold"])
        )
        # Freeze and hash every fitted object before the one-shot outer access.
        fold_dir = artifact_root / f"fold_{outer_fold}"
        artifacts = _save_fold_artifacts(
            fitted=fitted, calibrator=calibrator, threshold=threshold,
            fold_dir=fold_dir,
        )
        outer_gpu = _score_model(fitted, outer_runs, device="cuda")
        outer_cpu = _score_model(fitted, outer_runs, device="cpu")
        rows, parity = _score_outer_rows(
            outer_runs, outer_gpu, outer_cpu, calibrator,
            float(threshold["threshold"]),
        )
        parity["inner_report_only"] = {
            "not_a_go_gate": True,
            "logits_max_absolute_difference": inner_error,
            "run_max_max_absolute_difference": inner_max_error,
            "threshold_absolute_difference": threshold_error,
        }
        fold_result: dict[str, Any] = {
            "outer_fold": outer_fold,
            "inner_calibration_fold": inner_fold,
            "fit_run_ids_by_class": {
                str(idv): list(roles["fit"]) for idv in CONDITION_IDVS
            },
            "inner_healthy_run_ids": list(roles["inner_healthy"]),
            "unused_inner_fault_run_ids_by_class": {
                str(idv): list(roles["inner_healthy"]) for idv in FAULT_IDVS
            },
            "outer_run_ids_by_class": {
                str(idv): list(roles["outer"]) for idv in CONDITION_IDVS
            },
            "fit_healthy_ecdf": {
                "run_count": calibrator.calibration_run_count,
                "endpoints_per_run": calibrator.endpoint_count,
                "reference_count_per_expert": calibrator.samples_per_expert,
                "minimum_pvalue": calibrator.minimum_pvalue,
                "sha256": artifacts["ecdf_sha256"],
            },
            "inner_threshold": threshold,
            "outer_rows": rows,
            "parity": parity,
            "parameter_counts": parameter_counts,
            "artifacts": artifacts,
        }
        fold_result["metrics"] = _fold_metrics(rows)
        outer_result_path = fold_dir / "outer_result.json"
        outer_result_sha256 = _sha256_document(fold_result)
        _atomic_json(outer_result_path, fold_result)
        _atomic_bytes(
            fold_dir / "outer_result.json.sha256",
            f"{outer_result_sha256}  {outer_result_path.name}\n".encode("ascii"),
        )
        fold_result["outer_result_artifact"] = {
            "path": str(outer_result_path.resolve()),
            "sha256": outer_result_sha256,
        }
        folds.append(fold_result)
        del fitted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = _aggregate_metrics(folds)
    gates = _gate_report(folds, aggregate)
    decision = "DEVELOPMENT_GO" if gates["all_pass"] else "DEVELOPMENT_NO_GO"
    decision_document = {
        "schema": "cp_etc_scan_v5_1_development_decision_pre_refit_v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_sha256": MODE2_SHA256,
        "candidate_id": CANDIDATE_ID,
        "fold_outer_result_sha256": [
            fold["outer_result_artifact"]["sha256"] for fold in folds
        ],
        "aggregate_metrics": aggregate,
        "gates": gates,
        "decision": decision,
        "mode1_accessed": False,
    }
    decision_path = artifact_root / "development_decision_pre_refit.json"
    _atomic_json(decision_path, decision_document)
    decision_artifact = {
        "path": str(decision_path.resolve()),
        "sha256": _sha256_file(decision_path),
    }
    final_refit: Mapping[str, Any] | None = None
    if decision == "DEVELOPMENT_GO":
        print("[CP-ETC v5.1] all gates pass; final Mode2 100/class refit", flush=True)
        all_runs = _select_runs(blocks, range(1, 101), CONDITION_IDVS)
        healthy_runs = tuple(run for run in all_runs if _run_idv(run) == 1)
        _configure_determinism()
        fitted = _fit_model(all_runs, device=device)
        _require_training_contract(fitted, runs=400, healthy=100)
        parameter_counts = _parameter_counts(fitted)
        healthy_gpu = _score_model(fitted, healthy_runs, device="cuda")
        healthy_cpu = _score_model(fitted, healthy_runs, device="cpu")
        final_parity = float(np.max(np.abs(healthy_gpu - healthy_cpu)))
        if not math.isfinite(final_parity) or final_parity > PARITY_ATOL:
            raise RuntimeError("final refit CPU/CUDA parity failed")
        final_calibrator = ExpertTailCalibrator.fit(_monitor_logits(healthy_gpu))
        if final_calibrator.samples_per_expert != 8_500:
            raise RuntimeError("final Mode2 ECDF must contain 8500 scores per expert")
        final_refit = {
            **_save_final_artifacts(
                fitted=fitted, calibrator=final_calibrator,
                artifact_root=artifact_root,
            ),
            "healthy_cpu_cuda_logits_max_absolute_difference": final_parity,
            "fit_run_count_by_class": {str(idv): 100 for idv in CONDITION_IDVS},
            "parameter_counts": parameter_counts,
            "mode1_accessed": False,
        }
    else:
        print("[CP-ETC v5.1] DEVELOPMENT_NO_GO; no formal model produced", flush=True)

    return _plain_json({
        "schema": BACKEND_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "base_protocol_id": BASE_PROTOCOL_ID,
        "source_sha256": MODE2_SHA256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "fold_map_sha256": EXPECTED_FOLD_MAP_SHA256,
        "topology": topology,
        "runtime": dict(cuda_preflight()),
        "folds": folds,
        "aggregate_metrics": aggregate,
        "gates": gates,
        "development_decision_pre_refit": decision_artifact,
        "decision": decision,
        "selected_candidate_id": CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None,
        "formal_mode2_refit": final_refit,
        "mode1_accessed": False,
    })


__all__ = [
    "BACKEND_SCHEMA", "BASE_PROTOCOL_ID", "CANDIDATE_ID",
    "EXPECTED_ACCESS_PLAN_SHA256",
    "EXPECTED_FOLD_MAP_SHA256", "PROTOCOL_ID", "V51_FOLD_RUN_IDS",
    "_canonical_fold_map", "_fold_map_sha256", "_nearest_rank",
    "_operational_threshold", "cuda_preflight", "development_access_plan",
    "development_iteration", "fold_roles", "load_numeric_blocks",
    "validate_development_access_plan",
]
