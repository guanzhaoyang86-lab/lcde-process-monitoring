"""Governed numeric adapter for the magnitude-separated PCLR v2 protocol.

This module is deliberately separate from :mod:`mate_spc_pclr.backend`.  The
v1 files and artifacts remain immutable.  V2 changes only the data-governance
contract: development uses the unconsumed 50 percent branch, calibration uses
the IDV1 25 percent branch, and final confirmation uses the 75 percent branch.

The common ``simulation_run`` number across the four development conditions is
only a balanced stratification block.  It does not imply common random
numbers; the data loader validates every condition's own RNG seed.

No archive is opened at import time.  Numeric HDF5 access is possible only
through :func:`load_numeric_runs`, after an exact canonical plan has passed
validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

# Preserve the Windows CUDA/OpenMP import order used by the governed runner.
# Metadata-only callers may already have imported NumPy.  In that case, do
# not attempt a late Torch DLL load on Windows: numeric entry points remain
# fail-closed, while the governed CLI's preflight imports Torch first.
if "torch" in sys.modules:  # pragma: no branch - deterministic import state
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # pragma: no cover - import-order regression test
    _torch = None
else:
    try:  # pragma: no cover - runtime availability is checked at call time
        import torch as _torch
    except (ImportError, OSError):  # pragma: no cover - environment dependent
        _torch = None

import numpy as np

from . import data_v2 as pclr_data
from .data_v2 import (
    CALIBRATION,
    CALIBRATION_RUNS,
    CONDITION_IDVS,
    DEVELOPMENT,
    DEVELOPMENT_RUNS,
    FAULT_IDVS,
    FINAL,
    FINAL_RUNS,
    FAULT_WINDOW,
    HEALTHY_WINDOW,
    LOCAL_FAULT_START,
    DevelopmentGroup,
    PCLRMode1Archive,
    PCLRRun,
)
from .model import (
    FAULT_IDS,
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    PCLRConfig,
    PCLRFeatureModel,
    PCLRHeads,
    PCLRModel,
    fit_pclr_heads_cuda,
)


SEQUENCE_LENGTH = 500
MISS_DELAY = SEQUENCE_LENGTH - LOCAL_FAULT_START
PARITY_ATOL = 1.0e-10
CV_HEAD_EPOCHS = 120

DEVELOPMENT_MAGNITUDE_PERCENT = 50
CALIBRATION_MAGNITUDE_PERCENT = 25
FINAL_MAGNITUDE_PERCENT = 75
GROUPING_SEMANTICS = "balanced_stratification_block_not_crn"

_MAGNITUDE_BY_PARTITION = MappingProxyType(
    {
        DEVELOPMENT: DEVELOPMENT_MAGNITUDE_PERCENT,
        CALIBRATION: CALIBRATION_MAGNITUDE_PERCENT,
        FINAL: FINAL_MAGNITUDE_PERCENT,
    }
)


def _load_numeric_core() -> Any:
    """Import the unchanged numeric recurrence only after CUDA preflight."""

    from . import backend

    return backend


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _require_cuda_float64(*, device: str, dtype: str) -> None:
    if device != "cuda" or dtype != "float64":
        raise ValueError("the governed PCLR v2 backend requires device='cuda', dtype='float64'")
    if _torch is None:  # pragma: no cover - environment dependent
        raise RuntimeError("a CUDA-enabled PyTorch installation is required")
    if not _torch.cuda.is_available():  # pragma: no cover - environment dependent
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


def _run_magnitude(run: PCLRRun) -> int:
    try:
        value = run.metadata["fault_magnitude_percent"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("numeric run lacks fault_magnitude_percent identity") from exc
    return _plain_int(value, name="fault_magnitude_percent")


def _run_rng_seed(run: PCLRRun) -> int:
    try:
        value = run.metadata["rng_seed"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("numeric run lacks its loader-validated rng_seed identity") from exc
    return _plain_int(value, name="rng_seed")


@dataclass(frozen=True)
class NumericRunBatch:
    """Ordered calibration/final runs matching one magnitude-aware plan."""

    partition: str
    runs: tuple[PCLRRun, ...]
    access_keys: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if self.partition not in (CALIBRATION, FINAL):
            raise ValueError("numeric run batches are only calibration or final")
        if not self.runs or len(self.runs) != len(self.access_keys):
            raise ValueError("numeric run batch and access keys must be nonempty and aligned")
        observed = tuple(
            (_run_magnitude(run), int(run.source_idv), int(run.simulation_run))
            for run in self.runs
        )
        if observed != self.access_keys:
            raise ValueError("numeric runs do not preserve the magnitude-aware access-plan order")
        if len(set(self.access_keys)) != len(self.access_keys):
            raise ValueError("numeric access keys must be unique")
        identities = tuple(
            (
                str(run.metadata.get("partition")),
                _run_magnitude(run),
                int(run.source_idv),
                int(run.simulation_run),
                _run_rng_seed(run),
            )
            for run in self.runs
        )
        if any(
            identity[0] != self.partition
            or identity[1] != _MAGNITUDE_BY_PARTITION[self.partition]
            or identity[4] <= 0
            for identity in identities
        ):
            raise ValueError("numeric run partition, magnitude, or RNG seed is invalid")
        if len(set(identities)) != len(identities):
            raise ValueError("numeric run identities must be unique")
        if len({identity[4] for identity in identities}) != len(identities):
            raise ValueError("numeric RNG seeds must be distinct")


@dataclass(frozen=True)
class FoldFeatureFit:
    """Feature model fitted solely from named training stratification blocks."""

    model: PCLRFeatureModel
    fold: int
    varx_ridge: float
    training_group_keys: tuple[int, ...]


@dataclass(frozen=True)
class FoldFeatures:
    """Transformed paths with immutable fold and block provenance."""

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
    """Heads plus training-only features for fold-local thresholds."""

    heads: PCLRHeads
    feature_fit: FoldFeatureFit
    fold: int
    classifier_ridge: float
    training_features: FoldFeatures
    fit_metadata: Mapping[str, Any]


def _canonical_plan() -> dict[str, tuple[dict[str, Any], ...]]:
    development: list[dict[str, Any]] = []
    for simulation_run in DEVELOPMENT_RUNS:
        development.append(
            {
                "idv": 1,
                "simulation_run": simulation_run,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE_PERCENT,
                "window_half_open": list(HEALTHY_WINDOW),
                "role": "healthy",
            }
        )
        development.extend(
            {
                "idv": fault,
                "simulation_run": simulation_run,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE_PERCENT,
                "window_half_open": list(FAULT_WINDOW),
                "role": f"fault-{fault}",
            }
            for fault in FAULT_IDVS
        )
    calibration = tuple(
        {
            "idv": 1,
            "simulation_run": simulation_run,
            "magnitude_percent": CALIBRATION_MAGNITUDE_PERCENT,
            "window_half_open": list(HEALTHY_WINDOW),
            "role": "healthy_calibration",
        }
        for simulation_run in CALIBRATION_RUNS
    )
    final: list[dict[str, Any]] = [
        {
            "idv": 1,
            "simulation_run": simulation_run,
            "magnitude_percent": FINAL_MAGNITUDE_PERCENT,
            "window_half_open": list(HEALTHY_WINDOW),
            "role": "healthy_confirmation",
        }
        for simulation_run in FINAL_RUNS
    ]
    final.extend(
        {
            "idv": fault,
            "simulation_run": simulation_run,
            "magnitude_percent": FINAL_MAGNITUDE_PERCENT,
            "window_half_open": list(FAULT_WINDOW),
            "role": f"fault-{fault}",
        }
        for fault in FAULT_IDVS
        for simulation_run in FINAL_RUNS
    )
    return {
        DEVELOPMENT: tuple(development),
        CALIBRATION: calibration,
        FINAL: tuple(final),
    }


_PLANS = _canonical_plan()


def _validate_access_plan(
    access_plan: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if isinstance(access_plan, (str, bytes)) or not isinstance(access_plan, Sequence):
        raise TypeError("access_plan must be a sequence of mappings")
    required = {
        "idv",
        "simulation_run",
        "magnitude_percent",
        "window_half_open",
        "role",
    }
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(access_plan):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"access_plan[{index}] fields differ from the frozen v2 contract")
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
                "magnitude_percent": _plain_int(
                    raw["magnitude_percent"], name=f"access_plan[{index}].magnitude_percent"
                ),
                "window_half_open": [
                    _plain_int(window_raw[0], name="window start"),
                    _plain_int(window_raw[1], name="window stop"),
                ],
                "role": str(raw["role"]),
            }
        )
    value = tuple(normalized)
    matches = [
        partition
        for partition, expected in _canonical_plan().items()
        if value == expected
    ]
    if len(matches) != 1:
        raise ValueError("access_plan is not one exact canonical PCLR v2 phase allocation")
    return matches[0], value


def _validate_loaded_identity(
    run: PCLRRun, *, partition: str, row: Mapping[str, Any]
) -> tuple[str, int, int, int, int]:
    magnitude = int(row["magnitude_percent"])
    idv = int(row["idv"])
    simulation_run = int(row["simulation_run"])
    if (
        str(run.metadata.get("partition")) != partition
        or int(run.source_idv) != idv
        or int(run.simulation_run) != simulation_run
        or _run_magnitude(run) != magnitude
    ):
        raise ValueError("materialized run identity differs from the authorized access row")
    return partition, magnitude, idv, simulation_run, _run_rng_seed(run)


def load_numeric_runs(
    source_path: str | Path,
    access_plan: Sequence[Mapping[str, Any]],
) -> Mapping[int, DevelopmentGroup] | NumericRunBatch:
    """Read exactly one canonical v2 plan in its declared order."""

    partition, plan = _validate_access_plan(access_plan)
    archive = PCLRMode1Archive(Path(source_path))
    runs: list[PCLRRun] = []
    identities: list[tuple[str, int, int, int, int]] = []
    with archive._open() as handle:
        pclr_data.validate_processdata_labels(handle)
        for row in plan:
            run = pclr_data._materialize_run(
                handle,
                partition=partition,
                idv=int(row["idv"]),
                simulation_run=int(row["simulation_run"]),
            )
            identities.append(_validate_loaded_identity(run, partition=partition, row=row))
            runs.append(run)
    if len(set(identities)) != len(identities):
        raise ValueError("materialized v2 run identities are not unique")
    if any(identity[4] <= 0 for identity in identities):
        raise ValueError("materialized v2 RNG seeds must be positive")
    if len({identity[4] for identity in identities}) != len(identities):
        raise ValueError("materialized v2 RNG seeds must be globally distinct")

    if partition == DEVELOPMENT:
        by_key: dict[int, list[PCLRRun]] = {
            simulation_run: [] for simulation_run in DEVELOPMENT_RUNS
        }
        for item in runs:
            by_key[int(item.simulation_run)].append(item)
        grouped = {
            simulation_run: DevelopmentGroup(
                simulation_run, tuple(by_key[simulation_run])
            )
            for simulation_run in DEVELOPMENT_RUNS
        }
        return MappingProxyType(grouped)
    keys = tuple(
        (
            int(row["magnitude_percent"]),
            int(row["idv"]),
            int(row["simulation_run"]),
        )
        for row in plan
    )
    return NumericRunBatch(partition, tuple(runs), keys)


def _validated_groups(
    groups: Mapping[int, DevelopmentGroup], *, name: str
) -> tuple[tuple[int, DevelopmentGroup], ...]:
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError(f"{name} must be a nonempty run-keyed mapping")
    result: list[tuple[int, DevelopmentGroup]] = []
    identities: set[tuple[str, int, int, int, int]] = set()
    rng_seeds: set[int] = set()
    for raw_key in sorted(groups):
        key = _plain_int(raw_key, name=f"{name} key")
        group = groups[raw_key]
        if not isinstance(group, DevelopmentGroup) or int(group.simulation_run) != key:
            raise ValueError(f"{name} must map each simulation run to its DevelopmentGroup")
        for run in group.conditions:
            identity = (
                str(run.metadata.get("partition")),
                _run_magnitude(run),
                int(run.source_idv),
                int(run.simulation_run),
                _run_rng_seed(run),
            )
            if (
                identity[0] != DEVELOPMENT
                or identity[1] != DEVELOPMENT_MAGNITUDE_PERCENT
                or identity[3] != key
            ):
                raise ValueError("development group violates the frozen v2 identity contract")
            if identity in identities:
                raise ValueError("development run identities must be unique")
            if identity[4] <= 0 or identity[4] in rng_seeds:
                raise ValueError("development RNG seeds must be positive and globally distinct")
            identities.add(identity)
            rng_seeds.add(identity[4])
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
            label = 0 if int(run.source_idv) == 1 else int(run.source_idv)
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
    _require_cuda_float64(device=device, dtype=dtype)
    size = _positive_int(batch_size, name="batch_size")
    if not isinstance(feature_model, FoldFeatureFit):
        raise TypeError("feature_model must be a FoldFeatureFit")
    fold_int = _plain_int(fold, name="fold")
    if fold_int != feature_model.fold:
        raise ValueError("transform fold differs from the feature fit")
    if role not in ("training", "validation"):
        raise ValueError("role must be 'training' or 'validation'")
    raw_runs, labels, simulation_runs, keys = _flatten_groups(
        groups, name=f"{role}_groups"
    )
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


def _threshold_from_training_healthy(
    heads: FoldHeads,
    *,
    page_drift: float,
    threshold_rank: int,
    device: str,
) -> tuple[float, dict[str, Any]]:
    # The numerical recurrence is unchanged from v1.  The v2 governance and
    # grouping semantics are enforced before reaching this pure numeric core.
    return _load_numeric_core()._threshold_from_training_healthy(
        heads,
        page_drift=page_drift,
        threshold_rank=threshold_rank,
        device=device,
    )


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
    if _positive_int(development_threshold_rank, name="development_threshold_rank") != 79:
        raise ValueError("development threshold rank must be frozen at 79 of 80")
    if threshold_source != "80_training_healthy_run_maxima_only":
        raise ValueError("development threshold source differs from the frozen contract")
    if threshold_rule != "noninterpolated_ascending_order_statistic":
        raise ValueError("development threshold rule differs from the frozen contract")
    if threshold_comparison != "strictly_greater_than":
        raise ValueError("development threshold comparison must be strict >")
    actual_healthy = sum(
        int(label) == 0 for label in heads.training_features.labels.values()
    )
    if actual_healthy != 80:
        raise ValueError("head-fit training healthy count differs from the frozen threshold contract")
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
    numeric_core = _load_numeric_core()
    cpu_paths, _, validation_parity = numeric_core._score_feature_batch(
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
        detection_index = numeric_core.first_episode_detection(
            cpu, onset=LOCAL_FAULT_START
        )
        detected = detection_index is not None
        common = {"simulation_run": simulation_run, "fold": fold_int}
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
        "training_healthy_cpu_run_maxima": parity["training_healthy_cpu_run_maxima"],
        "training_healthy_cuda_run_maxima": parity["training_healthy_cuda_run_maxima"],
        "operational_threshold_q79": float(parity["training_threshold"]),
        "parity": {
            "eligibility_exact": bool(parity["eligibility_exact"]),
            "alarm_exact": bool(parity["alarm_exact"]),
            "max_abs_score_error": float(parity["max_abs_score_error"]),
            "attribution_exact": bool(parity["attribution_exact"]),
        },
    }


def _model_arrays(model: PCLRModel, *, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    config = asdict(model.feature_model.config)
    threshold_present = model.heads.threshold is not None
    return {
        "schema": np.array(MODEL_SCHEMA, dtype=np.str_),
        "schema_version": np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
        "config_json": np.array(_canonical_json(config), dtype=np.str_),
        "metadata_json": np.array(_canonical_json(dict(metadata)), dtype=np.str_),
        "mad_floor": np.asarray(model.feature_model.mad_floor, dtype=np.float64),
        "var_coefficients": np.asarray(model.feature_model.var_coefficients, dtype=np.float64),
        "weights": np.asarray(model.heads.weights, dtype=np.float64),
        "bias": np.asarray(model.heads.bias, dtype=np.float64),
        "healthy_logit_center": np.asarray(model.heads.healthy_logit_center, dtype=np.float64),
        "healthy_logit_scale": np.asarray(model.heads.healthy_logit_scale, dtype=np.float64),
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
            "backend": "mate_spc_pclr.backend_v2",
            "fit_all_components_from_scratch": True,
            "development_group_count": len(group_keys),
            "development_run_count": len(raw_runs),
            "development_runs": [min(group_keys), max(group_keys)],
            "development_magnitude_percent": DEVELOPMENT_MAGNITUDE_PERCENT,
            "grouping_semantics": GROUPING_SEMANTICS,
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


def score_paths(
    *,
    model_path: str | Path,
    data: NumericRunBatch,
    device: str,
    dtype: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Score a v2 batch with the unchanged canonical NumPy/CUDA recurrence."""

    _require_cuda_float64(device=device, dtype=dtype)
    if not isinstance(data, NumericRunBatch):
        raise TypeError("data must be a v2 NumericRunBatch returned by load_numeric_runs")
    numeric_core = _load_numeric_core()
    legacy_batch = numeric_core.NumericRunBatch(
        data.partition,
        data.runs,
        tuple((idv, simulation_run) for _, idv, simulation_run in data.access_keys),
    )
    result = numeric_core.score_paths(
        model_path=model_path,
        data=legacy_batch,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    # Identity arrays are deliberately emitted in the exact data.runs order.
    # The runner archives and revalidates them before any phase transition.
    result["source_idvs"] = np.asarray(
        [int(run.source_idv) for run in data.runs], dtype=np.int64
    )
    result["simulation_runs"] = np.asarray(
        [int(run.simulation_run) for run in data.runs], dtype=np.int64
    )
    result["magnitude_percent"] = np.asarray(
        [_run_magnitude(run) for run in data.runs], dtype=np.int64
    )
    result["rng_seeds"] = np.asarray(
        [_run_rng_seed(run) for run in data.runs], dtype=np.int64
    )
    return result


__all__ = [
    "CALIBRATION_MAGNITUDE_PERCENT",
    "DEVELOPMENT_MAGNITUDE_PERCENT",
    "FINAL_MAGNITUDE_PERCENT",
    "FoldFeatureFit",
    "FoldFeatures",
    "FoldHeads",
    "GROUPING_SEMANTICS",
    "NumericRunBatch",
    "fit_fold_feature_model",
    "fit_fold_heads",
    "load_numeric_runs",
    "refit_selected_model",
    "score_fold_candidate",
    "score_paths",
    "transform_fold_groups",
]
