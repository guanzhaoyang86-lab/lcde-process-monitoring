"""Governed numeric adapter for the PCLR-CUSUM experiment runner.

The public runner deliberately delays importing this module until after it has
written an access intent and phase manifest.  Consequently this module must
not inspect the Extended TEP archive at import time.  All HDF5 reads happen in
``load_numeric_runs`` and are restricted to the caller's exact, canonical
access plan.

The adapter keeps the three fitted stages explicit:

* a fold-local baseline MAD/causal VARX feature model;
* fold-local density-ratio heads trained with true causal prefix features;
* Page-CUSUM scoring for a selected drift, with a threshold fitted from the
  training-fold healthy run maxima only.

NumPy and PyTorch paths both use the canonical implementations in
``mate_spc_pclr.model``.  This is intentionally less clever than maintaining a
second copy of Page/persistence logic: the runner independently replays the
recurrence, and this module independently executes it on CPU and CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# On Windows the runner loads PyTorch's OpenMP runtime before importing this
# numeric backend. Keeping the same local order avoids a second inversion once
# execution of this submodule begins (the package entry point itself must also
# be reached through the runner's preflight order).
try:  # pragma: no cover - unavailable runtime is checked at call time
    import torch as _torch
except (ImportError, OSError):  # pragma: no cover - environment dependent
    _torch = None

import numpy as np

from . import data as pclr_data
from .data import (
    CALIBRATION,
    CALIBRATION_RUNS,
    CONDITION_IDVS,
    DEVELOPMENT,
    DEVELOPMENT_RUNS,
    FAULT_IDVS,
    FINAL,
    FINAL_RUNS,
    HEALTHY_WINDOW,
    FAULT_WINDOW,
    LOCAL_FAULT_START,
    DevelopmentGroup,
    PCLRMode1Archive,
    PCLRRun,
)
from .model import (
    FAULT_IDS,
    MONITOR_START,
    MONITOR_STOP,
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    PCLRConfig,
    PCLRFeatureModel,
    PCLRHeads,
    PCLRModel,
    PCLRMonitoringPath,
    first_episode_detection,
    fit_pclr_heads_cuda,
    load_pclr_model,
    monitor_batch_torch_float64,
    page_cusum_numpy,
)


SEQUENCE_LENGTH = 500
MISS_DELAY = SEQUENCE_LENGTH - LOCAL_FAULT_START
PARITY_ATOL = 1.0e-10
CV_HEAD_EPOCHS = 120


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _require_cuda_float64(*, device: str, dtype: str) -> None:
    if device != "cuda" or dtype != "float64":
        raise ValueError("the governed PCLR backend requires device='cuda', dtype='float64'")
    if _torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("a CUDA-enabled PyTorch installation is required")
    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class NumericRunBatch:
    """Ordered calibration/final runs exactly matching one access plan."""

    partition: str
    runs: tuple[PCLRRun, ...]
    access_keys: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.partition not in (CALIBRATION, FINAL):
            raise ValueError("numeric run batches are only calibration or final")
        if not self.runs or len(self.runs) != len(self.access_keys):
            raise ValueError("numeric run batch and access keys must be nonempty and aligned")
        observed = tuple((run.source_idv, run.simulation_run) for run in self.runs)
        if observed != self.access_keys:
            raise ValueError("numeric runs do not preserve access-plan order")


@dataclass(frozen=True)
class FoldFeatureFit:
    """Feature model fitted solely from the named training CRN groups."""

    model: PCLRFeatureModel
    fold: int
    varx_ridge: float
    training_group_keys: tuple[int, ...]


@dataclass(frozen=True)
class FoldFeatures:
    """Transformed run paths with immutable fold/provenance information."""

    feature_fit: FoldFeatureFit
    fold: int
    role: str
    group_keys: tuple[int, ...]
    monitoring: Mapping[str, np.ndarray]
    prefixes: Mapping[str, np.ndarray]
    labels: Mapping[str, int]
    simulation_runs: Mapping[str, int]
    batch_size: int


@dataclass(frozen=True)
class FoldHeads:
    """Heads plus training-only features needed for fold-local thresholds."""

    heads: PCLRHeads
    feature_fit: FoldFeatureFit
    fold: int
    classifier_ridge: float
    training_features: FoldFeatures
    fit_metadata: Mapping[str, Any]


def _canonical_plan() -> dict[str, tuple[dict[str, Any], ...]]:
    development: list[dict[str, Any]] = []
    for run in DEVELOPMENT_RUNS:
        development.append(
            {
                "idv": 1,
                "simulation_run": run,
                "window_half_open": list(HEALTHY_WINDOW),
                "role": "healthy",
            }
        )
        development.extend(
            {
                "idv": fault,
                "simulation_run": run,
                "window_half_open": list(FAULT_WINDOW),
                "role": f"fault-{fault}",
            }
            for fault in FAULT_IDVS
        )
    calibration = tuple(
        {
            "idv": 1,
            "simulation_run": run,
            "window_half_open": list(HEALTHY_WINDOW),
            "role": "healthy_calibration",
        }
        for run in CALIBRATION_RUNS
    )
    final: list[dict[str, Any]] = [
        {
            "idv": 1,
            "simulation_run": run,
            "window_half_open": list(HEALTHY_WINDOW),
            "role": "healthy_confirmation",
        }
        for run in FINAL_RUNS
    ]
    final.extend(
        {
            "idv": fault,
            "simulation_run": run,
            "window_half_open": list(FAULT_WINDOW),
            "role": f"fault-{fault}",
        }
        for fault in FAULT_IDVS
        for run in FINAL_RUNS
    )
    return {
        DEVELOPMENT: tuple(development),
        CALIBRATION: calibration,
        FINAL: tuple(final),
    }


_PLANS = _canonical_plan()


def _validate_access_plan(access_plan: Sequence[Mapping[str, Any]]) -> tuple[str, tuple[dict[str, Any], ...]]:
    if isinstance(access_plan, (str, bytes)) or not isinstance(access_plan, Sequence):
        raise TypeError("access_plan must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    required = {"idv", "simulation_run", "window_half_open", "role"}
    for index, raw in enumerate(access_plan):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"access_plan[{index}] fields differ from the frozen contract")
        window_raw = raw["window_half_open"]
        if (
            isinstance(window_raw, (str, bytes))
            or not isinstance(window_raw, Sequence)
            or len(window_raw) != 2
        ):
            raise ValueError(f"access_plan[{index}] has an invalid window")
        normalized.append(
            {
                "idv": _plain_int(raw["idv"], name=f"access_plan[{index}].idv"),
                "simulation_run": _plain_int(
                    raw["simulation_run"], name=f"access_plan[{index}].simulation_run"
                ),
                "window_half_open": [
                    _plain_int(window_raw[0], name="window start"),
                    _plain_int(window_raw[1], name="window stop"),
                ],
                "role": str(raw["role"]),
            }
        )
    value = tuple(normalized)
    # Rebuild the immutable-by-construction contract instead of trusting a
    # module-level collection whose nested dictionaries a caller could mutate.
    matches = [
        partition
        for partition, expected in _canonical_plan().items()
        if value == expected
    ]
    if len(matches) != 1:
        raise ValueError("access_plan is not one exact canonical PCLR phase allocation")
    return matches[0], value


def load_numeric_runs(
    source_path: str | Path,
    access_plan: Sequence[Mapping[str, Any]],
) -> Mapping[int, DevelopmentGroup] | NumericRunBatch:
    """Read exactly one canonical access plan, once, in declared order.

    Access-plan validation occurs before the archive object is constructed, so
    malformed, partial, reordered, embargoed, or cross-partition requests
    cannot even open the HDF5 file.
    """

    partition, plan = _validate_access_plan(access_plan)
    archive = PCLRMode1Archive(Path(source_path))
    runs: list[PCLRRun] = []
    # One read-only handle avoids hundreds of independent HDF5 opens while
    # preserving the exact plan order.  The data module re-authorizes every
    # materialization before touching a dataset.
    with archive._open() as handle:
        pclr_data.validate_processdata_labels(handle)
        for row in plan:
            runs.append(
                pclr_data._materialize_run(
                    handle,
                    partition=partition,
                    idv=int(row["idv"]),
                    simulation_run=int(row["simulation_run"]),
                )
            )

    if partition == DEVELOPMENT:
        by_key: dict[int, list[PCLRRun]] = {run: [] for run in DEVELOPMENT_RUNS}
        for item in runs:
            by_key[item.simulation_run].append(item)
        grouped: dict[int, DevelopmentGroup] = {}
        for run in DEVELOPMENT_RUNS:
            conditions = tuple(by_key[run])
            grouped[run] = DevelopmentGroup(run, conditions)
        return MappingProxyType(grouped)
    keys = tuple((int(row["idv"]), int(row["simulation_run"])) for row in plan)
    return NumericRunBatch(partition, tuple(runs), keys)


def _validated_groups(groups: Mapping[int, DevelopmentGroup], *, name: str) -> tuple[tuple[int, DevelopmentGroup], ...]:
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"{name} must be a nonempty run-keyed mapping")
    result: list[tuple[int, DevelopmentGroup]] = []
    for raw_key in sorted(groups):
        key = _plain_int(raw_key, name=f"{name} key")
        group = groups[raw_key]
        if not isinstance(group, DevelopmentGroup) or group.simulation_run != key:
            raise ValueError(f"{name} must map each simulation run to its DevelopmentGroup")
        result.append((key, group))
    return tuple(result)


def _flatten_groups(
    groups: Mapping[int, DevelopmentGroup], *, name: str
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int], tuple[int, ...]]:
    raw_runs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    simulation_runs: dict[str, int] = {}
    keys: list[int] = []
    for key, group in _validated_groups(groups, name=name):
        keys.append(key)
        for run in group.conditions:
            if run.name in raw_runs:
                raise ValueError(f"duplicate run name {run.name!r}")
            label = 0 if run.source_idv == 1 else run.source_idv
            if label not in (0, *FAULT_IDS):
                raise ValueError("development group contains an unsupported condition")
            raw_runs[run.name] = run.x
            labels[run.name] = label
            simulation_runs[run.name] = key
    return raw_runs, labels, simulation_runs, tuple(keys)


def fit_fold_feature_model(
    *,
    train_groups: Mapping[int, DevelopmentGroup],
    varx_ridge: float,
    fold: int,
    device: str,
    dtype: str,
) -> FoldFeatureFit:
    """Fit MAD floors and causal VARX from training groups only."""

    _require_cuda_float64(device=device, dtype=dtype)
    fold_int = _plain_int(fold, name="fold")
    ridge = _finite_positive(varx_ridge, name="varx_ridge")
    raw_runs, _, _, keys = _flatten_groups(train_groups, name="train_groups")
    feature_model = PCLRFeatureModel.fit(
        raw_runs, config=PCLRConfig(var_ridge_alpha=ridge)
    )
    return FoldFeatureFit(feature_model, fold_int, ridge, keys)


def transform_fold_groups(
    *,
    feature_model: FoldFeatureFit,
    groups: Mapping[int, DevelopmentGroup],
    role: str,
    fold: int,
    device: str,
    dtype: str,
    batch_size: int,
) -> FoldFeatures:
    """Transform groups only after a fold-local feature fit is fixed."""

    _require_cuda_float64(device=device, dtype=dtype)
    size = _positive_int(batch_size, name="batch_size")
    if not isinstance(feature_model, FoldFeatureFit):
        raise TypeError("feature_model must be a FoldFeatureFit")
    fold_int = _plain_int(fold, name="fold")
    if fold_int != feature_model.fold:
        raise ValueError("transform fold differs from the feature fit")
    if role not in ("training", "validation"):
        raise ValueError("role must be 'training' or 'validation'")
    raw_runs, labels, simulation_runs, keys = _flatten_groups(groups, name=f"{role}_groups")
    if role == "training":
        if keys != feature_model.training_group_keys:
            raise ValueError("training transform groups differ from feature-fit groups")
    elif set(keys) & set(feature_model.training_group_keys):
        raise ValueError("validation groups overlap feature-fit training groups")
    monitoring, prefixes = feature_model.model.transform_training_runs(raw_runs)
    return FoldFeatures(
        feature_fit=feature_model,
        fold=fold_int,
        role=role,
        group_keys=keys,
        monitoring=MappingProxyType(monitoring),
        prefixes=MappingProxyType(prefixes),
        labels=MappingProxyType(labels),
        simulation_runs=MappingProxyType(simulation_runs),
        batch_size=size,
    )


def fit_fold_heads(
    *,
    train_features: FoldFeatures,
    classifier_ridge: float,
    fold: int,
    device: str,
    dtype: str,
) -> FoldHeads:
    """Fit three heads and their healthy normalization from training only."""

    _require_cuda_float64(device=device, dtype=dtype)
    if not isinstance(train_features, FoldFeatures) or train_features.role != "training":
        raise TypeError("train_features must be a training FoldFeatures object")
    fold_int = _plain_int(fold, name="fold")
    if fold_int != train_features.fold:
        raise ValueError("head-fit fold differs from transformed training features")
    ridge = _finite_positive(classifier_ridge, name="classifier_ridge")
    heads, metadata = fit_pclr_heads_cuda(
        train_features.monitoring,
        train_features.labels,
        prefix_features_by_run=train_features.prefixes,
        ridge_lambda=ridge,
        kappa=0.0,
        device=device,
        epochs=CV_HEAD_EPOCHS,
    )
    if (
        float(metadata.get("prefix_rank_weight", 0.0)) <= 0.0
        or metadata.get("prefix_rank_contract")
        != "causal_local_128_160_mean_vs_post_160_500_top_quartile"
    ):
        raise RuntimeError("head fit did not use the frozen true-prefix ranking loss")
    return FoldHeads(
        heads=heads,
        feature_fit=train_features.feature_fit,
        fold=fold_int,
        classifier_ridge=ridge,
        training_features=train_features,
        fit_metadata=MappingProxyType(dict(metadata)),
    )


def _score_feature_batch(
    features_by_name: Mapping[str, np.ndarray],
    heads: PCLRHeads,
    *,
    names: Sequence[str],
    threshold: float | None,
    device: str,
    batch_size: int,
) -> tuple[list[PCLRMonitoringPath], Any, dict[str, Any]]:
    """Score runs with per-run NumPy and run-batched canonical CUDA paths."""

    ordered = tuple(names)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("score batch names must be nonempty and unique")
    if any(name not in features_by_name for name in ordered):
        raise ValueError("score batch names are absent from the feature mapping")
    matrix = np.stack(
        [np.asarray(features_by_name[name], dtype=np.float64) for name in ordered],
        axis=0,
    )
    indices = np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64)
    cuda = monitor_batch_torch_float64(
        matrix,
        heads,
        local_indices=indices,
        threshold=threshold,
        device=device,
        batch_size=batch_size,
    )
    cpu_paths = [
        page_cusum_numpy(
            heads.logits_numpy(matrix[index]),
            heads,
            local_indices=indices,
            threshold=threshold,
        )
        for index in range(len(ordered))
    ]
    numeric_errors = {
        field: float(
            np.max(
                np.abs(
                    np.stack(
                        [np.asarray(getattr(path, field)) for path in cpu_paths],
                        axis=0,
                    )
                    - np.asarray(getattr(cuda, field))
                ),
                initial=0.0,
            )
        )
        for field in (
            "raw_logits",
            "standardized_logits",
            "cusum",
            "component_persistence",
            "envelope",
        )
    }
    cpu_eligible = np.stack([path.eligible for path in cpu_paths], axis=0)
    cpu_alarm = np.stack([path.alarm for path in cpu_paths], axis=0)
    cpu_episode = np.stack([path.episode_start for path in cpu_paths], axis=0)
    cpu_attribution = np.stack(
        [np.argmax(path.component_persistence, axis=1) for path in cpu_paths],
        axis=0,
    )
    eligibility_exact = bool(np.array_equal(cpu_eligible, cuda.eligible))
    alarm_exact = bool(
        np.array_equal(cpu_alarm, cuda.alarm)
        and np.array_equal(cpu_episode, cuda.episode_start)
    )
    attribution_exact = bool(
        np.array_equal(
            cpu_attribution,
            np.argmax(cuda.component_persistence, axis=2),
        )
    )
    return cpu_paths, cuda, {
        "max_abs_score_error": max(numeric_errors.values()),
        "numeric_stage_max_abs_errors": numeric_errors,
        "eligibility_exact": eligibility_exact,
        "alarm_exact": alarm_exact,
        "attribution_exact": attribution_exact,
    }


def _threshold_from_training_healthy(
    heads: FoldHeads,
    *,
    page_drift: float,
    threshold_rank: int,
    device: str,
) -> tuple[float, dict[str, Any]]:
    actual = heads.heads.with_kappa(page_drift)
    healthy_names = sorted(
        (name for name, label in heads.training_features.labels.items() if int(label) == 0),
        key=lambda name: heads.training_features.simulation_runs[name],
    )
    rank = _positive_int(threshold_rank, name="threshold_rank")
    if rank > len(healthy_names):
        raise ValueError("threshold_rank exceeds the training healthy run count")
    cpu_paths, cuda, batch_parity = _score_feature_batch(
        heads.training_features.monitoring,
        actual,
        names=healthy_names,
        threshold=None,
        device=device,
        batch_size=heads.training_features.batch_size,
    )
    if any(path.alarm.any() for path in cpu_paths) or cuda.alarm.any():
        raise RuntimeError("unthresholded training path emitted an alarm")
    cpu_maxima = np.asarray([path.run_max for path in cpu_paths], dtype=np.float64)
    cuda_maxima = np.asarray(cuda.run_max, dtype=np.float64)
    threshold = float(np.sort(np.asarray(cpu_maxima, dtype=np.float64))[rank - 1])
    if not math.isfinite(threshold) or threshold < 0.0:
        raise FloatingPointError("fold-local training threshold is invalid")
    return threshold, {
        "training_healthy_run_count": len(healthy_names),
        "training_healthy_rank": rank,
        "training_threshold": threshold,
        "training_healthy_cpu_run_maxima": cpu_maxima,
        "training_healthy_cuda_run_maxima": cuda_maxima,
        "training_cuda_threshold_at_same_rank": float(
            np.sort(np.asarray(cuda_maxima, dtype=np.float64))[rank - 1]
        ),
        **batch_parity,
    }


def score_fold_candidate(
    *,
    heads: FoldHeads,
    validation_features: FoldFeatures,
    page_drift: float,
    fold: int,
    device: str,
    dtype: str,
    development_threshold_rank: int = 79,
    threshold_source: str = "80_training_healthy_run_maxima_only",
    threshold_rule: str = "noninterpolated_ascending_order_statistic",
    threshold_comparison: str = "strictly_greater_than",
) -> dict[str, Any]:
    """Fit a training-only fold threshold and score held-out CRN groups.

    The default rank/count encode the frozen five-fold 80/20 design.  The
    runner passes them explicitly; defaults retain a convenient direct API for
    synthetic tests without weakening the runtime checks.
    """

    _require_cuda_float64(device=device, dtype=dtype)
    if not isinstance(heads, FoldHeads):
        raise TypeError("heads must be a FoldHeads object")
    if not isinstance(validation_features, FoldFeatures) or validation_features.role != "validation":
        raise TypeError("validation_features must be a validation FoldFeatures object")
    fold_int = _plain_int(fold, name="fold")
    if fold_int != heads.fold or fold_int != validation_features.fold:
        raise ValueError("score fold differs from fitted/transformed objects")
    if heads.feature_fit is not validation_features.feature_fit:
        raise ValueError("validation features do not come from the fitted fold feature model")
    if set(heads.training_features.group_keys) & set(validation_features.group_keys):
        raise ValueError("validation groups overlap head-fit training groups")
    drift = _finite_nonnegative(page_drift, name="page_drift")
    if _positive_int(
        development_threshold_rank, name="development_threshold_rank"
    ) != 79:
        raise ValueError("development threshold rank must be frozen at 79 of 80")
    if threshold_source != "80_training_healthy_run_maxima_only":
        raise ValueError("development threshold source differs from the frozen contract")
    if threshold_rule != "noninterpolated_ascending_order_statistic":
        raise ValueError("development threshold rule differs from the frozen contract")
    if threshold_comparison != "strictly_greater_than":
        raise ValueError("development threshold comparison must be strict >")
    expected_healthy = 80
    actual_healthy = sum(label == 0 for label in heads.training_features.labels.values())
    if actual_healthy != expected_healthy:
        raise ValueError(
            "head-fit training healthy count differs from the frozen threshold contract"
        )
    threshold, parity = _threshold_from_training_healthy(
        heads,
        page_drift=drift,
        threshold_rank=development_threshold_rank,
        device=device,
    )
    actual_heads = heads.heads.with_kappa(drift)
    validation_names = sorted(
        validation_features.monitoring,
        key=lambda name: (
            validation_features.simulation_runs[name],
            validation_features.labels[name],
        ),
    )
    cpu_paths, _, validation_parity = _score_feature_batch(
        validation_features.monitoring,
        actual_heads,
        names=validation_names,
        threshold=threshold,
        device=device,
        batch_size=validation_features.batch_size,
    )
    for field in ("eligibility_exact", "alarm_exact", "attribution_exact"):
        parity[field] = bool(parity[field] and validation_parity[field])
    parity["max_abs_score_error"] = max(
        float(parity["max_abs_score_error"]),
        float(validation_parity["max_abs_score_error"]),
    )
    fault_records: list[dict[str, Any]] = []
    healthy_records: list[dict[str, Any]] = []
    for name, cpu in zip(validation_names, cpu_paths, strict=True):
        label = int(validation_features.labels[name])
        simulation_run = int(validation_features.simulation_runs[name])
        detection_index = first_episode_detection(cpu, onset=LOCAL_FAULT_START)
        detected = detection_index is not None
        common = {
            "simulation_run": simulation_run,
            "fold": fold_int,
        }
        if label == 0:
            healthy_records.append({**common, "alarmed": detected})
        else:
            fault_records.append(
                {
                    "fault_id": label,
                    **common,
                    "detected": detected,
                    "restricted_delay_samples": (
                        int(detection_index - LOCAL_FAULT_START)
                        if detection_index is not None
                        else MISS_DELAY
                    ),
                }
            )
    fault_records.sort(key=lambda row: (int(row["fault_id"]), int(row["simulation_run"])))
    healthy_records.sort(key=lambda row: int(row["simulation_run"]))
    return {
        "records": fault_records,
        "healthy_records": healthy_records,
        "training_healthy_cpu_run_maxima": parity[
            "training_healthy_cpu_run_maxima"
        ],
        "training_healthy_cuda_run_maxima": parity[
            "training_healthy_cuda_run_maxima"
        ],
        "operational_threshold_q79": float(parity["training_threshold"]),
        "parity": {
            "eligibility_exact": bool(parity["eligibility_exact"]),
            "alarm_exact": bool(parity["alarm_exact"]),
            "max_abs_score_error": float(parity["max_abs_score_error"]),
            "attribution_exact": bool(parity["attribution_exact"]),
        },
    }


def _model_arrays(model: PCLRModel, *, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the exact safe-NPZ payload understood by ``load_pclr_model``."""

    config = asdict(model.feature_model.config)
    threshold_present = model.heads.threshold is not None
    return {
        "schema": np.array(MODEL_SCHEMA, dtype=np.str_),
        "schema_version": np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
        "config_json": np.array(_canonical_json(config), dtype=np.str_),
        "metadata_json": np.array(_canonical_json(dict(metadata)), dtype=np.str_),
        "mad_floor": np.asarray(model.feature_model.mad_floor, dtype=np.float64),
        "var_coefficients": np.asarray(
            model.feature_model.var_coefficients, dtype=np.float64
        ),
        "weights": np.asarray(model.heads.weights, dtype=np.float64),
        "bias": np.asarray(model.heads.bias, dtype=np.float64),
        "healthy_logit_center": np.asarray(
            model.heads.healthy_logit_center, dtype=np.float64
        ),
        "healthy_logit_scale": np.asarray(
            model.heads.healthy_logit_scale, dtype=np.float64
        ),
        "kappa": np.asarray(model.heads.kappa, dtype=np.float64),
        "fault_ids": np.asarray(model.heads.fault_ids, dtype=np.int64),
        "threshold_present": np.array(int(threshold_present), dtype=np.int64),
        "threshold": np.array(
            0.0 if model.heads.threshold is None else model.heads.threshold,
            dtype=np.float64,
        ),
    }


def refit_selected_model(
    *,
    raw_development_groups: Mapping[int, DevelopmentGroup],
    selected_params: Mapping[str, Any],
    steps: int,
    fit_all_components_from_scratch: bool,
    device: str,
    dtype: str,
) -> dict[str, np.ndarray]:
    """Refit every learned component from raw development groups."""

    _require_cuda_float64(device=device, dtype=dtype)
    if fit_all_components_from_scratch is not True:
        raise ValueError("final refit must fit all components from scratch")
    if not isinstance(selected_params, Mapping) or set(selected_params) != {
        "varx_ridge",
        "classifier_ridge",
        "page_drift",
    }:
        raise ValueError("selected_params fields differ from the frozen grid")
    epochs = _positive_int(steps, name="steps")
    varx_ridge = _finite_positive(selected_params["varx_ridge"], name="varx_ridge")
    classifier_ridge = _finite_positive(
        selected_params["classifier_ridge"], name="classifier_ridge"
    )
    page_drift = _finite_nonnegative(selected_params["page_drift"], name="page_drift")
    raw_runs, labels, _, group_keys = _flatten_groups(
        raw_development_groups, name="raw_development_groups"
    )
    feature_model = PCLRFeatureModel.fit(
        raw_runs, config=PCLRConfig(var_ridge_alpha=varx_ridge)
    )
    monitoring, prefixes = feature_model.transform_training_runs(raw_runs)
    heads, fit_metadata = fit_pclr_heads_cuda(
        monitoring,
        labels,
        prefix_features_by_run=prefixes,
        ridge_lambda=classifier_ridge,
        kappa=page_drift,
        device=device,
        epochs=epochs,
    )
    model = PCLRModel(feature_model, heads.with_threshold(None))
    return _model_arrays(
        model,
        metadata={
            "backend": "mate_spc_pclr.backend",
            "fit_all_components_from_scratch": True,
            "development_group_count": len(group_keys),
            "development_run_count": len(raw_runs),
            "development_runs": [min(group_keys), max(group_keys)],
            "selected_params": {
                "varx_ridge": varx_ridge,
                "classifier_ridge": classifier_ridge,
                "page_drift": page_drift,
            },
            "final_refit_steps": epochs,
            "steps": epochs,
            "head_fit": dict(fit_metadata),
        },
    )


def _pad_path(path: PCLRMonitoringPath) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(
        path.local_indices,
        np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64),
    ):
        raise ValueError("core monitor path local indices changed")
    standardized = np.zeros((SEQUENCE_LENGTH, len(FAULT_IDS)), dtype=np.float64)
    cusum = np.zeros_like(standardized)
    persistent = np.zeros_like(standardized)
    scores = np.zeros(SEQUENCE_LENGTH, dtype=np.float64)
    eligible = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    standardized[MONITOR_START:MONITOR_STOP] = path.standardized_logits
    cusum[MONITOR_START:MONITOR_STOP] = path.cusum
    persistent[MONITOR_START:MONITOR_STOP] = path.component_persistence
    scores[MONITOR_START:MONITOR_STOP] = path.envelope
    eligible[MONITOR_START:MONITOR_STOP] = path.eligible
    return standardized, cusum, persistent, scores, eligible


def score_paths(
    *,
    model_path: str | Path,
    data: NumericRunBatch,
    device: str,
    dtype: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Score a calibration/final batch on canonical NumPy and CUDA paths."""

    _require_cuda_float64(device=device, dtype=dtype)
    size = _positive_int(batch_size, name="batch_size")
    if not isinstance(data, NumericRunBatch):
        raise TypeError("data must be a NumericRunBatch returned by load_numeric_runs")
    model, _ = load_pclr_model(Path(model_path))
    if model.heads.threshold is not None:
        raise ValueError("the development model must remain uncalibrated")
    count = len(data.runs)
    component_shape = (count, SEQUENCE_LENGTH, len(FAULT_IDS))
    result = {
        "cpu_standardized_logits": np.zeros(component_shape, dtype=np.float64),
        "cpu_cusum": np.zeros(component_shape, dtype=np.float64),
        "cpu_component_persistence": np.zeros(component_shape, dtype=np.float64),
        "cpu_scores": np.zeros((count, SEQUENCE_LENGTH), dtype=np.float64),
        "cpu_eligible": np.zeros((count, SEQUENCE_LENGTH), dtype=np.bool_),
        "cuda_standardized_logits": np.zeros(component_shape, dtype=np.float64),
        "cuda_cusum": np.zeros(component_shape, dtype=np.float64),
        "cuda_component_persistence": np.zeros(component_shape, dtype=np.float64),
        "cuda_scores": np.zeros((count, SEQUENCE_LENGTH), dtype=np.float64),
        "cuda_eligible": np.zeros((count, SEQUENCE_LENGTH), dtype=np.bool_),
    }
    raw_logit_max_error = 0.0
    for offset in range(0, count, size):
        stop = min(count, offset + size)
        feature_paths = [
            model.feature_model.transform_run_detailed(data.runs[index].x)
            for index in range(offset, stop)
        ]
        feature_matrix = np.stack([path.values for path in feature_paths], axis=0)
        cuda_batch = monitor_batch_torch_float64(
            feature_matrix,
            model.heads,
            local_indices=feature_paths[0].local_indices,
            threshold=None,
            device=device,
            batch_size=size,
        )
        for relative, feature_path in enumerate(feature_paths):
            index = offset + relative
            raw_cpu = model.heads.logits_numpy(feature_path.values)
            cpu = page_cusum_numpy(
                raw_cpu,
                model.heads,
                local_indices=feature_path.local_indices,
                threshold=None,
            )
            cuda = cuda_batch.run(relative)
            raw_logit_max_error = max(
                raw_logit_max_error,
                float(np.max(np.abs(raw_cpu - cuda.raw_logits), initial=0.0)),
            )
            if not (
                np.array_equal(cpu.eligible, cuda.eligible)
                and np.array_equal(cpu.alarm, cuda.alarm)
                and np.array_equal(cpu.episode_start, cuda.episode_start)
                and np.array_equal(
                    np.argmax(cpu.component_persistence, axis=1),
                    np.argmax(cuda.component_persistence, axis=1),
                )
            ):
                raise AssertionError(
                    "NumPy/CUDA eligibility, alarm, episode, or attribution parity failed"
                )
            for prefix, path in (("cpu", cpu), ("cuda", cuda)):
                standardized, cusum, persistent, scores, eligible = _pad_path(path)
                result[f"{prefix}_standardized_logits"][index] = standardized
                result[f"{prefix}_cusum"][index] = cusum
                result[f"{prefix}_component_persistence"][index] = persistent
                result[f"{prefix}_scores"][index] = scores
                result[f"{prefix}_eligible"][index] = eligible
    max_error = max(
        float(
            np.max(
                np.abs(result[f"cpu_{field}"] - result[f"cuda_{field}"]),
                initial=0.0,
            )
        )
        for field in ("standardized_logits", "cusum", "component_persistence", "scores")
    )
    max_error = max(max_error, raw_logit_max_error)
    if max_error > PARITY_ATOL:
        raise AssertionError(
            f"NumPy/CUDA full score-path parity failed: {max_error:.3e} > {PARITY_ATOL:.3e}"
        )
    if not np.array_equal(result["cpu_eligible"], result["cuda_eligible"]):
        raise AssertionError("NumPy/CUDA eligibility parity failed")
    return result


__all__ = [
    "FoldFeatureFit",
    "FoldFeatures",
    "FoldHeads",
    "NumericRunBatch",
    "fit_fold_feature_model",
    "fit_fold_heads",
    "load_numeric_runs",
    "refit_selected_model",
    "score_fold_candidate",
    "score_paths",
    "transform_fold_groups",
]
