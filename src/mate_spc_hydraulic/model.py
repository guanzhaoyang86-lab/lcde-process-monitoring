"""Phase-directional monitoring for the UCI hydraulic test-rig cycles.

The module deliberately separates the three data roles used by the method:

* ``fit_hydraulic_directional_cuda`` consumes development cycles only;
* ``fit_hydraulic_grouped_cv_cuda`` selects a ridge value with development
  folds and then refits on all development cycles; and
* ``calibrate_hydraulic_pac`` consumes independent healthy calibration cycles
  only after every model parameter has been frozen.

PyTorch is imported lazily and is used only by the CUDA fitting/scoring entry
points.  The NumPy scorer is an independent reference implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from mate_spc_pac import PACRank, pac_tolerance_rank


PHASE_COUNT = 60
FEATURE_COUNT = 88
HEALTHY_LABEL = 100
POSITIVE_LABELS = (90, 80, 73)
VALVE_LABELS = (HEALTHY_LABEL, *POSITIVE_LABELS)
HEAD_COUNT = len(POSITIVE_LABELS)
ROBUST_SCALE_FACTOR = 1.4826
SCALE_FLOOR = 1.0e-6
GLOBAL_SCALE_FLOOR_FRACTION = 0.1
Z_CLIP = 8.0
DEFAULT_KAPPA = 0.75
PERSISTENCE_WINDOW = 3
RIDGE_LAMBDA_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
CV_FOLD_COUNT = 3
CV_FOLD_SALT = "HYDRAULIC-PADR-PAC-v1-CV"
MODEL_SCHEMA = "mate_spc_hydraulic_phase_directional"
MODEL_SCHEMA_VERSION = 1
STRICT_COMPARISON = "strictly_greater_than"


def hydraulic_scorer_spec() -> dict[str, object]:
    """Return a fresh copy of the immutable, protocol-frozen scorer contract."""
    return {
        "schema": "hydraulic_padr_pac_scorer_spec",
        "schema_version": 1,
        "cycle_horizon_seconds": PHASE_COUNT,
        "complete_horizon_required": True,
        "positive_heads_in_order": list(POSITIVE_LABELS),
        "healthy_label": HEALTHY_LABEL,
        "phase_normalization": (
            "development_healthy_per_phase_per_head_median_and_1.4826_MAD_"
            "floor_max_1e-6_0.1_times_head_global_healthy_robust_scale"
        ),
        "standardized_logit_clip": [-Z_CLIP, Z_CLIP],
        "page_kappa": [DEFAULT_KAPPA] * HEAD_COUNT,
        "page_reset": "zero_at_start_of_each_official_cycle",
        "persistence": {
            "consecutive_seconds": PERSISTENCE_WINDOW,
            "rule": "minimum_within_each_head",
        },
        "envelope": "maximum_persistent_page_statistic_across_heads",
        "run_loss": "maximum_envelope_over_complete_60_second_cycle",
        "alarm_comparison": STRICT_COMPARISON,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


HYDRAULIC_SCORER_SPEC_SHA256 = hashlib.sha256(
    _canonical_json_bytes(hydraulic_scorer_spec())
).hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _readonly_float_array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array


def normalize_kappa(
    value: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return a read-only, nonnegative allowance for the three heads."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), HEAD_COUNT)
    if array.shape != (HEAD_COUNT,) or not np.isfinite(array).all():
        raise ValueError(f"kappa must be a finite scalar or length-{HEAD_COUNT} vector")
    if np.any(array < 0.0):
        raise ValueError("kappa must be nonnegative")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def validate_residual_features(features: Any, *, name: str = "features") -> np.ndarray:
    """Validate and copy a ``(cycles, 60, 88)`` residual tensor."""
    values = np.asarray(features, dtype=np.float64)
    expected_tail = (PHASE_COUNT, FEATURE_COUNT)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != expected_tail:
        raise ValueError(
            f"{name} must have shape (positive_cycles, {PHASE_COUNT}, {FEATURE_COUNT})"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(values, dtype=np.float64, copy=True, order="C")


def validate_valve_labels(
    labels: Any,
    *,
    cycle_count: int,
    require_all: bool,
) -> np.ndarray:
    """Validate the fixed UCI valve labels without accepting float coercion."""
    if isinstance(labels, (str, bytes)):
        raise ValueError("valve_labels must be a one-dimensional integer sequence")
    values = np.asarray(labels)
    if values.ndim != 1 or values.shape[0] != cycle_count:
        raise ValueError(f"valve_labels must have shape ({cycle_count},)")
    normalized: list[int] = []
    for index, value in enumerate(values.tolist()):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"valve label at cycle {index} must be an integer")
        label = int(value)
        if label not in VALVE_LABELS:
            raise ValueError(
                f"unsupported valve label {label} at cycle {index}; "
                f"expected one of {VALVE_LABELS}"
            )
        normalized.append(label)
    result = np.asarray(normalized, dtype=np.int64)
    if require_all:
        missing = [label for label in VALVE_LABELS if not np.any(result == label)]
        if missing:
            raise ValueError(f"development data are missing valve labels {missing}")
    result.setflags(write=False)
    return result


def _normalize_nuisance_groups(
    nuisance_groups: Any,
    *,
    cycle_count: int,
) -> tuple[str, ...]:
    if isinstance(nuisance_groups, (str, bytes)):
        raise ValueError("nuisance_groups must be a one-dimensional identifier sequence")
    values = np.asarray(nuisance_groups)
    if values.ndim != 1 or values.shape[0] != cycle_count:
        raise ValueError(f"nuisance_groups must have shape ({cycle_count},)")
    result: list[str] = []
    for index, raw in enumerate(values.tolist()):
        if isinstance(raw, bool):
            raise ValueError(f"nuisance group at cycle {index} must not be boolean")
        if isinstance(raw, int):
            result.append(f"integer:{raw}")
        elif isinstance(raw, str) and raw:
            result.append(f"string:{raw}")
        else:
            raise ValueError(
                f"nuisance group at cycle {index} must be a nonempty string or integer"
            )
    return tuple(result)


def _validate_development_data(
    features: Any,
    valve_labels: Any,
    nuisance_groups: Any,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    values = validate_residual_features(features, name="development features")
    labels = validate_valve_labels(
        valve_labels,
        cycle_count=values.shape[0],
        require_all=True,
    )
    groups = _normalize_nuisance_groups(
        nuisance_groups,
        cycle_count=values.shape[0],
    )
    return values, labels, groups


def cell_balanced_cycle_weights(
    valve_labels: Any,
    nuisance_groups: Any,
    *,
    positive_label: int,
) -> np.ndarray:
    """Weight every observed ``(nuisance group, class)`` cell equally.

    The returned vector is zero for the two positive classes not used by the
    requested one-vs-100 head.  Its nonzero values sum to one.  The fitting
    objective takes a phase mean inside each cycle, so phases are equal within
    a cycle, cycles are equal inside a cell, and cells have equal total mass.
    """
    if isinstance(positive_label, bool) or positive_label not in POSITIVE_LABELS:
        raise ValueError(f"positive_label must be one of {POSITIVE_LABELS}")
    labels_array = np.asarray(valve_labels)
    if labels_array.ndim != 1:
        raise ValueError("valve_labels must be one-dimensional")
    labels = validate_valve_labels(
        valve_labels,
        cycle_count=int(labels_array.shape[0]),
        require_all=False,
    )
    groups = _normalize_nuisance_groups(
        nuisance_groups,
        cycle_count=labels.shape[0],
    )
    selected = (labels == HEALTHY_LABEL) | (labels == positive_label)
    if not np.any(labels == HEALTHY_LABEL) or not np.any(labels == positive_label):
        raise ValueError(
            f"one-vs-{HEALTHY_LABEL} head {positive_label} requires both classes"
        )
    cells = sorted({(groups[i], int(labels[i])) for i in np.flatnonzero(selected)})
    weights = np.zeros(labels.shape[0], dtype=np.float64)
    cell_mass = 1.0 / float(len(cells))
    for group, label in cells:
        indices = np.asarray(
            [
                index
                for index in range(labels.shape[0])
                if groups[index] == group and int(labels[index]) == label
            ],
            dtype=np.int64,
        )
        weights[indices] = cell_mass / float(indices.size)
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-14):
        raise AssertionError("cell-balanced weights do not sum to one")
    weights.setflags(write=False)
    return weights


def _training_targets_and_weights(
    labels: np.ndarray,
    groups: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((labels.shape[0], HEAD_COUNT), dtype=np.float64)
    weights = np.zeros_like(targets)
    for head, positive_label in enumerate(POSITIVE_LABELS):
        targets[:, head] = labels == positive_label
        weights[:, head] = cell_balanced_cycle_weights(
            labels,
            groups,
            positive_label=positive_label,
        )
    return targets, weights


def fit_healthy_phase_normalization(
    logits: Any,
    valve_labels: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit phase/head median and ``1.4826*MAD`` on healthy dev cycles."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1:] != (
        PHASE_COUNT,
        HEAD_COUNT,
    ):
        raise ValueError(
            f"logits must have shape (positive_cycles, {PHASE_COUNT}, {HEAD_COUNT})"
        )
    if not np.isfinite(values).all():
        raise ValueError("logits must contain only finite values")
    labels = validate_valve_labels(
        valve_labels,
        cycle_count=values.shape[0],
        require_all=False,
    )
    healthy = values[labels == HEALTHY_LABEL]
    if healthy.shape[0] < 1:
        raise ValueError("at least one healthy development cycle is required")
    center = np.median(healthy, axis=0)
    raw_scale = ROBUST_SCALE_FACTOR * np.median(
        np.abs(healthy - center[None, :, :]),
        axis=0,
    )
    global_center = np.median(healthy, axis=(0, 1))
    global_scale = ROBUST_SCALE_FACTOR * np.median(
        np.abs(healthy - global_center[None, None, :]),
        axis=(0, 1),
    )
    floor_by_head = np.maximum(
        SCALE_FLOOR,
        GLOBAL_SCALE_FLOOR_FRACTION * global_scale,
    )
    floored = raw_scale < floor_by_head[None, :]
    scale = np.maximum(raw_scale, floor_by_head[None, :])
    for array in (center, scale, floored):
        array.setflags(write=False)
    return center, scale, floored


@dataclass(frozen=True)
class HydraulicDirectionalModel:
    """Frozen three-head classifier, phase normalizer, and monitor settings."""

    positive_labels: tuple[int, int, int]
    weights: np.ndarray
    bias: np.ndarray
    phase_logit_center: np.ndarray
    phase_logit_scale: np.ndarray
    kappa: np.ndarray
    threshold: float | None = None

    def __post_init__(self) -> None:
        labels = tuple(int(value) for value in self.positive_labels)
        if labels != POSITIVE_LABELS:
            raise ValueError(f"positive_labels must be exactly {POSITIVE_LABELS}")
        weights = _readonly_float_array(self.weights, ndim=2, name="weights")
        if weights.shape != (HEAD_COUNT, FEATURE_COUNT):
            raise ValueError(f"weights must have shape ({HEAD_COUNT}, {FEATURE_COUNT})")
        bias = _readonly_float_array(self.bias, ndim=1, name="bias")
        center = _readonly_float_array(
            self.phase_logit_center,
            ndim=2,
            name="phase_logit_center",
        )
        scale = _readonly_float_array(
            self.phase_logit_scale,
            ndim=2,
            name="phase_logit_scale",
        )
        if bias.shape != (HEAD_COUNT,):
            raise ValueError(f"bias must have shape ({HEAD_COUNT},)")
        expected_phase_shape = (PHASE_COUNT, HEAD_COUNT)
        if center.shape != expected_phase_shape or scale.shape != expected_phase_shape:
            raise ValueError(
                "phase_logit_center and phase_logit_scale must have shape "
                f"{expected_phase_shape}"
            )
        if np.any(scale < SCALE_FLOOR):
            raise ValueError(f"phase_logit_scale must be at least {SCALE_FLOOR}")
        kappa = normalize_kappa(self.kappa)
        threshold = self.threshold
        if threshold is not None:
            threshold = _nonnegative_float(threshold, name="threshold")
        object.__setattr__(self, "positive_labels", labels)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "phase_logit_center", center)
        object.__setattr__(self, "phase_logit_scale", scale)
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "threshold", threshold)

    @property
    def head_count(self) -> int:
        return HEAD_COUNT

    @property
    def feature_count(self) -> int:
        return FEATURE_COUNT

    def with_threshold(self, threshold: float | None) -> "HydraulicDirectionalModel":
        return replace(self, threshold=threshold)

    def with_kappa(
        self,
        kappa: float | Sequence[float] | np.ndarray,
    ) -> "HydraulicDirectionalModel":
        return replace(self, kappa=normalize_kappa(kappa))

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_hydraulic_model(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "HydraulicDirectionalModel":
        model, _ = load_hydraulic_model(path)
        return model


@dataclass(frozen=True)
class HydraulicDirectionalPath:
    """Batched, per-cycle causal monitoring paths."""

    positive_labels: tuple[int, int, int]
    logits: np.ndarray
    standardized_logits: np.ndarray
    clipped_evidence: np.ndarray
    cusum: np.ndarray
    component_persistence: np.ndarray
    envelope: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    threshold: float | None

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "logits": self.logits,
            "standardized_logits": self.standardized_logits,
            "clipped_evidence": self.clipped_evidence,
            "cusum": self.cusum,
            "component_persistence": self.component_persistence,
            "envelope": self.envelope,
            "eligible": self.eligible,
            "alarm": self.alarm,
        }


@dataclass(frozen=True)
class HydraulicCVFoldResult:
    ridge_lambda: float
    held_out_fold: int
    training_cycle_count: int
    held_out_cycle_count: int
    held_out_nuisance_groups: tuple[str, ...]
    validation_log_loss: float
    convergence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "ridge_lambda": self.ridge_lambda,
            "held_out_fold": self.held_out_fold,
            "training_cycle_count": self.training_cycle_count,
            "held_out_cycle_count": self.held_out_cycle_count,
            "held_out_nuisance_groups": list(self.held_out_nuisance_groups),
            "validation_log_loss": self.validation_log_loss,
            "convergence": dict(self.convergence),
        }


@dataclass(frozen=True)
class HydraulicCVResult:
    selected_lambda: float
    mean_validation_loss_by_lambda: tuple[tuple[float, float], ...]
    folds: tuple[HydraulicCVFoldResult, ...]
    final_fit: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_lambda": self.selected_lambda,
            "mean_validation_loss_by_lambda": [
                {"ridge_lambda": ridge, "mean_validation_log_loss": loss}
                for ridge, loss in self.mean_validation_loss_by_lambda
            ],
            "folds": [fold.as_dict() for fold in self.folds],
            "final_fit": dict(self.final_fit),
            "selection_scope": "development_grouped_cv_only",
        }


def hydraulic_directional_model_sha256(model: HydraulicDirectionalModel) -> str:
    """Fingerprint every pre-threshold parameter used by hydraulic scoring."""
    if not isinstance(model, HydraulicDirectionalModel):
        raise TypeError("model must be a HydraulicDirectionalModel")
    digest = hashlib.sha256()
    digest.update(MODEL_SCHEMA.encode("utf-8"))
    digest.update(np.asarray(POSITIVE_LABELS, dtype="<i8").tobytes(order="C"))
    for name, value in (
        ("weights", model.weights),
        ("bias", model.bias),
        ("phase_logit_center", model.phase_logit_center),
        ("phase_logit_scale", model.phase_logit_scale),
        ("kappa", model.kappa),
    ):
        array = np.asarray(value, dtype="<f8", order="C")
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def hydraulic_calibration_sha256(
    *,
    model_sha256: str,
    run_maxima: Sequence[tuple[str, float]],
    rank: PACRank,
) -> str:
    """Bind exact maxima and the PAC request to one model and scorer spec."""
    model_fingerprint = _validate_sha256(model_sha256, name="model_sha256")
    if not isinstance(rank, PACRank):
        raise TypeError("rank must be a PACRank")
    maxima = sorted((str(name), float(value)) for name, value in run_maxima)
    payload = {
        "model_sha256": model_fingerprint,
        "rank": rank.as_dict(),
        "run_maxima": [
            {"cycle": name, "maximum_hex": value.hex()} for name, value in maxima
        ],
        "scorer_spec_sha256": HYDRAULIC_SCORER_SPEC_SHA256,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class HydraulicPACCalibration:
    """PAC threshold certificate specialized to complete hydraulic cycles."""

    status: str
    threshold: float
    rank: PACRank
    rank_driver_cycle: str
    run_maxima: tuple[tuple[str, float], ...]
    scorer_spec_sha256: str
    model_sha256: str
    calibration_sha256: str
    comparison: str = STRICT_COMPARISON

    def __post_init__(self) -> None:
        if not isinstance(self.rank, PACRank):
            raise TypeError("rank must be a PACRank")
        if self.status != self.rank.status:
            raise ValueError("calibration status must equal the PAC rank status")
        if self.comparison != STRICT_COMPARISON:
            raise ValueError(f"comparison must be {STRICT_COMPARISON}")
        scorer_fingerprint = _validate_sha256(
            self.scorer_spec_sha256,
            name="scorer_spec_sha256",
        )
        if scorer_fingerprint != HYDRAULIC_SCORER_SPEC_SHA256:
            raise ValueError("scorer specification fingerprint mismatch")
        model_fingerprint = _validate_sha256(self.model_sha256, name="model_sha256")
        calibration_fingerprint = _validate_sha256(
            self.calibration_sha256,
            name="calibration_sha256",
        )
        maxima = tuple(sorted((str(name), float(value)) for name, value in self.run_maxima))
        names = [name for name, _ in maxima]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("calibration cycle names must be nonempty and unique")
        if any(not np.isfinite(value) or value < 0.0 for _, value in maxima):
            raise ValueError("calibration cycle maxima must be finite and nonnegative")
        if len(maxima) != self.rank.sample_count:
            raise ValueError("PAC rank sample count must equal the calibration cycle count")
        ordered = sorted(maxima, key=lambda item: (item[1], item[0]))
        driver, threshold = ordered[self.rank.rank_one_based - 1]
        if self.rank_driver_cycle != driver or float(self.threshold) != threshold:
            raise ValueError("threshold or rank driver does not match calibration maxima")
        expected_fingerprint = hydraulic_calibration_sha256(
            model_sha256=model_fingerprint,
            run_maxima=maxima,
            rank=self.rank,
        )
        if calibration_fingerprint != expected_fingerprint:
            raise ValueError("hydraulic calibration fingerprint mismatch")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "run_maxima", maxima)
        object.__setattr__(self, "scorer_spec_sha256", scorer_fingerprint)
        object.__setattr__(self, "model_sha256", model_fingerprint)
        object.__setattr__(self, "calibration_sha256", calibration_fingerprint)

    @property
    def sample_count(self) -> int:
        return self.rank.sample_count

    @property
    def epsilon(self) -> float:
        return self.rank.epsilon

    @property
    def delta(self) -> float:
        return self.rank.delta

    @property
    def rank_one_based(self) -> int:
        return self.rank.rank_one_based

    @property
    def achieved_confidence(self) -> float:
        return self.rank.achieved_confidence

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "rank_driver_cycle": self.rank_driver_cycle,
            "sample_count": self.sample_count,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rank_one_based": self.rank_one_based,
            "achieved_confidence": self.achieved_confidence,
            "rank": self.rank.as_dict(),
            "run_maxima": [
                {"cycle": name, "maximum": maximum}
                for name, maximum in self.run_maxima
            ],
            "scorer_spec": hydraulic_scorer_spec(),
            "scorer_spec_sha256": self.scorer_spec_sha256,
            "model_sha256": self.model_sha256,
            "calibration_sha256": self.calibration_sha256,
        }


def predict_hydraulic_logits_numpy(
    model: HydraulicDirectionalModel,
    features: Any,
) -> np.ndarray:
    """Return raw one-vs-healthy logits with shape ``(cycles, 60, 3)``."""
    if not isinstance(model, HydraulicDirectionalModel):
        raise TypeError("model must be a HydraulicDirectionalModel")
    values = validate_residual_features(features, name="scoring features")
    logits = values @ model.weights.T + model.bias
    if not np.isfinite(logits).all():
        raise FloatingPointError("hydraulic logits are non-finite")
    return np.asarray(logits, dtype=np.float64)


def _page_cusum_numpy_batch(evidence: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    increments = evidence - kappa[None, None, :]
    partial = np.cumsum(increments, axis=1, dtype=np.float64)
    augmented = np.concatenate(
        (np.zeros((evidence.shape[0], 1, HEAD_COUNT)), partial),
        axis=1,
    )
    running_minimum = np.minimum.accumulate(augmented, axis=1)[:, 1:]
    return np.maximum(partial - running_minimum, 0.0)


def _path_from_numpy_logits(
    model: HydraulicDirectionalModel,
    logits: np.ndarray,
) -> HydraulicDirectionalPath:
    standardized = (
        logits - model.phase_logit_center[None, :, :]
    ) / model.phase_logit_scale[None, :, :]
    clipped = np.clip(standardized, -Z_CLIP, Z_CLIP)
    cusum = _page_cusum_numpy_batch(clipped, model.kappa)
    component = np.full(cusum.shape, -np.inf, dtype=np.float64)
    for phase in range(PERSISTENCE_WINDOW - 1, PHASE_COUNT):
        component[:, phase] = np.min(
            cusum[:, phase - PERSISTENCE_WINDOW + 1 : phase + 1],
            axis=1,
        )
    envelope = np.full((logits.shape[0], PHASE_COUNT), -np.inf, dtype=np.float64)
    first = PERSISTENCE_WINDOW - 1
    envelope[:, first:] = np.max(component[:, first:], axis=2)
    eligible = np.zeros(envelope.shape, dtype=bool)
    eligible[:, first:] = True
    alarm = np.zeros(envelope.shape, dtype=bool)
    if model.threshold is not None:
        alarm = eligible & (envelope > model.threshold)
    return HydraulicDirectionalPath(
        positive_labels=POSITIVE_LABELS,
        logits=logits,
        standardized_logits=standardized,
        clipped_evidence=clipped,
        cusum=cusum,
        component_persistence=component,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm,
        threshold=model.threshold,
    )


def score_hydraulic_numpy(
    model: HydraulicDirectionalModel,
    features: Any,
) -> HydraulicDirectionalPath:
    """Score independent cycles; Page state is reset on every cycle."""
    logits = predict_hydraulic_logits_numpy(model, features)
    return _path_from_numpy_logits(model, logits)


def balanced_validation_log_loss_numpy(
    model: HydraulicDirectionalModel,
    features: Any,
    valve_labels: Any,
    nuisance_groups: Any,
) -> float:
    """Cell-balanced three-head log loss, excluding the ridge penalty."""
    values = validate_residual_features(features, name="validation features")
    labels = validate_valve_labels(
        valve_labels,
        cycle_count=values.shape[0],
        require_all=True,
    )
    groups = _normalize_nuisance_groups(
        nuisance_groups,
        cycle_count=values.shape[0],
    )
    targets, cycle_weights = _training_targets_and_weights(labels, groups)
    logits = predict_hydraulic_logits_numpy(model, values)
    losses: list[float] = []
    for head in range(HEAD_COUNT):
        phase_loss = np.logaddexp(0.0, logits[:, :, head])
        phase_loss -= targets[:, head, None] * logits[:, :, head]
        cycle_loss = np.mean(phase_loss, axis=1)
        losses.append(float(np.sum(cycle_weights[:, head] * cycle_loss)))
    return float(np.mean(losses))


def _torch_and_cuda_device(device: str) -> tuple[Any, Any, dict[str, object]]:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("a CUDA-enabled PyTorch installation is required") from error
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is not available")
    properties = torch.cuda.get_device_properties(target)
    runtime = {
        "backend": "pytorch_cuda_hydraulic_phase_directional",
        "requested_device": str(device),
        "resolved_device": str(target),
        "device_name": str(torch.cuda.get_device_name(target)),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "dtype": "float64",
        "tf32": False,
    }
    return torch, target, runtime


@contextmanager
def _tf32_disabled(torch: Any) -> Iterator[None]:
    previous_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn = bool(torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def fit_hydraulic_directional_cuda(
    development_features: Any,
    valve_labels: Any,
    nuisance_groups: Any,
    *,
    ridge_lambda: float,
    kappa: float | Sequence[float] | np.ndarray = DEFAULT_KAPPA,
    device: str = "cuda",
    max_iter: int = 500,
    max_eval: int | None = 625,
    tolerance_grad: float = 1.0e-7,
    tolerance_change: float = 1.0e-10,
    history_size: int = 100,
    require_convergence: bool = False,
) -> tuple[HydraulicDirectionalModel, dict[str, object]]:
    """Fit the three cell-balanced ridge-logistic heads on CUDA float64."""
    features, labels, groups = _validate_development_data(
        development_features,
        valve_labels,
        nuisance_groups,
    )
    ridge_lambda = _positive_float(ridge_lambda, name="ridge_lambda")
    normalized_kappa = normalize_kappa(kappa)
    max_iter = _positive_integer(max_iter, name="max_iter")
    history_size = _positive_integer(history_size, name="history_size")
    if max_eval is None:
        max_eval = max(1, int(max_iter * 5 // 4))
    max_eval = _positive_integer(max_eval, name="max_eval")
    tolerance_grad = _positive_float(tolerance_grad, name="tolerance_grad")
    tolerance_change = _positive_float(tolerance_change, name="tolerance_change")
    if not isinstance(require_convergence, (bool, np.bool_)):
        raise ValueError("require_convergence must be boolean")
    targets, cycle_weights = _training_targets_and_weights(labels, groups)

    torch, target, runtime = _torch_and_cuda_device(device)
    feature_tensor = torch.as_tensor(features, dtype=torch.float64, device=target)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64, device=target)
    cycle_weight_tensor = torch.as_tensor(
        cycle_weights,
        dtype=torch.float64,
        device=target,
    )
    coefficients = torch.zeros(
        (HEAD_COUNT, FEATURE_COUNT),
        dtype=torch.float64,
        device=target,
        requires_grad=True,
    )
    intercept = torch.zeros(
        HEAD_COUNT,
        dtype=torch.float64,
        device=target,
        requires_grad=True,
    )

    def head_losses() -> list[Any]:
        logits = feature_tensor @ coefficients.T + intercept
        element_losses = torch.nn.functional.softplus(logits) - target_tensor[:, None, :] * logits
        cycle_losses = torch.mean(element_losses, dim=1)
        losses: list[Any] = []
        for head in range(HEAD_COUNT):
            data_loss = torch.sum(cycle_weight_tensor[:, head] * cycle_losses[:, head])
            penalty = 0.5 * ridge_lambda * torch.sum(coefficients[head] ** 2)
            losses.append(data_loss + penalty)
        return losses

    def objective() -> Any:
        return torch.mean(torch.stack(head_losses()))

    optimizer = torch.optim.LBFGS(
        [coefficients, intercept],
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    closure_evaluations = 0

    def closure() -> Any:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("ridge-logistic objective became non-finite")
        loss.backward()
        return loss

    torch.cuda.reset_peak_memory_stats(target)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with _tf32_disabled(torch):
        with torch.no_grad():
            initial_objective = float(objective().detach().cpu())
        begin.record()
        optimizer.step(closure)
        end.record()
        end.synchronize()
        elapsed_ms = float(begin.elapsed_time(end))
        final_loss = objective()
        gradients = torch.autograd.grad(final_loss, (coefficients, intercept))
        gradient_inf_norm = max(
            float(torch.max(torch.abs(gradient)).detach().cpu())
            for gradient in gradients
        )
        final_objective = float(final_loss.detach().cpu())
        final_head_objectives = [float(loss.detach().cpu()) for loss in head_losses()]
        fitted_weights = coefficients.detach().cpu().numpy()
        fitted_bias = intercept.detach().cpu().numpy()

    state = optimizer.state.get(coefficients, {})
    optimizer_iterations = int(state.get("n_iter", 0))
    optimizer_evaluations = int(state.get("func_evals", closure_evaluations))
    if gradient_inf_norm <= tolerance_grad:
        termination_reason = "gradient_tolerance"
        converged = True
    elif optimizer_iterations >= max_iter:
        termination_reason = "maximum_iterations"
        converged = False
    elif optimizer_evaluations >= max_eval:
        termination_reason = "maximum_function_evaluations"
        converged = False
    else:
        termination_reason = "optimizer_change_tolerance"
        converged = True
    if bool(require_convergence) and not converged:
        raise RuntimeError(
            "CUDA LBFGS did not converge: "
            f"reason={termination_reason}, gradient_inf_norm={gradient_inf_norm:.3e}"
        )

    dev_logits = features @ fitted_weights.T + fitted_bias
    center, scale, floored = fit_healthy_phase_normalization(dev_logits, labels)
    model = HydraulicDirectionalModel(
        positive_labels=POSITIVE_LABELS,
        weights=fitted_weights,
        bias=fitted_bias,
        phase_logit_center=center,
        phase_logit_scale=scale,
        kappa=normalized_kappa,
        threshold=None,
    )
    label_counts = {
        str(label): int(np.count_nonzero(labels == label)) for label in VALVE_LABELS
    }
    cell_counts = {
        str(positive): int(
            np.count_nonzero(
                cell_balanced_cycle_weights(labels, groups, positive_label=positive)
            )
        )
        for positive in POSITIVE_LABELS
    }
    metadata: dict[str, object] = {
        "backend": "pytorch_cuda_joint_three_head_lbfgs",
        "device": str(target),
        "device_name": runtime["device_name"],
        "dtype": "float64",
        "tf32": False,
        "positive_labels": list(POSITIVE_LABELS),
        "healthy_label": HEALTHY_LABEL,
        "feature_count": FEATURE_COUNT,
        "phase_count": PHASE_COUNT,
        "development_cycle_count": int(features.shape[0]),
        "development_cycle_count_by_label": label_counts,
        "nuisance_group_count": len(set(groups)),
        "included_cycle_count_by_head": cell_counts,
        "loss_weighting": (
            "equal_heads_equal_observed_nuisance_class_cells_"
            "equal_cycles_within_cell_equal_phases_within_cycle"
        ),
        "ridge_lambda": ridge_lambda,
        "ridge_penalty": "lambda_over_2_times_squared_weight_norm_per_head",
        "bias_penalized": False,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "final_head_objectives": final_head_objectives,
        "optimizer": "torch.optim.LBFGS_strong_wolfe",
        "initialization": "all_zero_no_randomness",
        "determinism_scope": "fixed_device_fixed_input_order_no_random_operations",
        "max_iter": max_iter,
        "max_eval": max_eval,
        "history_size": history_size,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "optimizer_iterations": optimizer_iterations,
        "closure_evaluations": closure_evaluations,
        "optimizer_function_evaluations": optimizer_evaluations,
        "gradient_inf_norm": gradient_inf_norm,
        "termination_reason": termination_reason,
        "converged": converged,
        "cuda_event_elapsed_seconds": elapsed_ms / 1000.0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
        "normalization_source": "development_healthy_cycles_only",
        "normalization_location": "per_phase_per_head_median_across_cycles",
        "normalization_scale": (
            "per_phase_per_head_1.4826_MAD_floor_max_1e-6_"
            "0.1_times_head_global_healthy_robust_scale"
        ),
        "normalization_floored_phase_head_count": int(np.count_nonzero(floored)),
        "kappa": normalized_kappa.tolist(),
        "runtime_environment": runtime,
        "data_scope": "development_only_no_calibration_or_final_input",
    }
    del feature_tensor, target_tensor, cycle_weight_tensor, coefficients, intercept
    return model, metadata


def validate_grouped_folds(
    valve_labels: Any,
    nuisance_groups: Any,
    fold_by_cycle: Any,
) -> np.ndarray:
    """Validate class-complete folds with no nuisance group split across folds."""
    label_values = np.asarray(valve_labels)
    if label_values.ndim != 1:
        raise ValueError("valve_labels must be one-dimensional")
    labels = validate_valve_labels(
        valve_labels,
        cycle_count=label_values.shape[0],
        require_all=True,
    )
    groups = _normalize_nuisance_groups(
        nuisance_groups,
        cycle_count=labels.shape[0],
    )
    raw_folds = np.asarray(fold_by_cycle)
    if raw_folds.ndim != 1 or raw_folds.shape[0] != labels.shape[0]:
        raise ValueError(f"fold_by_cycle must have shape ({labels.shape[0]},)")
    folds: list[int] = []
    for index, value in enumerate(raw_folds.tolist()):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"fold identifier at cycle {index} must be a nonnegative integer"
            )
        folds.append(int(value))
    result = np.asarray(folds, dtype=np.int64)
    fold_values = sorted(set(folds))
    if fold_values != list(range(CV_FOLD_COUNT)):
        raise ValueError(
            f"protocol grouped CV folds must be exactly {list(range(CV_FOLD_COUNT))}"
        )
    group_fold: dict[str, int] = {}
    for group, fold in zip(groups, folds, strict=True):
        previous = group_fold.setdefault(group, fold)
        if previous != fold:
            raise ValueError(f"nuisance group {group!r} is split across folds")
    required = set(VALVE_LABELS)
    for held_out_fold in fold_values:
        held_labels = set(int(value) for value in labels[result == held_out_fold])
        train_labels = set(int(value) for value in labels[result != held_out_fold])
        if held_labels != required:
            raise ValueError(
                f"held-out fold {held_out_fold} lacks valve labels "
                f"{sorted(required - held_labels)}"
            )
        if train_labels != required:
            raise ValueError(
                f"training complement for fold {held_out_fold} lacks valve labels "
                f"{sorted(required - train_labels)}"
            )
    result.setflags(write=False)
    return result


def protocol_grouped_fold_ids(nuisance_groups: Any) -> np.ndarray:
    """Apply the frozen SHA256 ordering and round-robin three-fold assignment."""
    if isinstance(nuisance_groups, (str, bytes)):
        raise ValueError("nuisance_groups must be a one-dimensional identifier sequence")
    raw = np.asarray(nuisance_groups)
    if raw.ndim != 1 or raw.shape[0] < 1:
        raise ValueError("nuisance_groups must be a nonempty one-dimensional sequence")
    group_text: list[str] = []
    for index, value in enumerate(raw.tolist()):
        if isinstance(value, bool):
            raise ValueError(f"nuisance group at cycle {index} must not be boolean")
        if isinstance(value, int):
            group_text.append(str(value))
        elif isinstance(value, str) and value:
            group_text.append(value)
        else:
            raise ValueError(
                f"nuisance group at cycle {index} must be a nonempty string or integer"
            )
    unique = sorted(
        set(group_text),
        key=lambda group: (
            hashlib.sha256(f"{CV_FOLD_SALT}|{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    if len(unique) < CV_FOLD_COUNT:
        raise ValueError(
            f"protocol grouped CV requires at least {CV_FOLD_COUNT} nuisance groups"
        )
    group_to_fold = {
        group: index % CV_FOLD_COUNT for index, group in enumerate(unique)
    }
    result = np.asarray([group_to_fold[group] for group in group_text], dtype=np.int64)
    result.setflags(write=False)
    return result


def _ridge_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("ridge_lambdas must be a nonempty sequence")
    normalized = tuple(_positive_float(value, name="ridge lambda") for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("ridge_lambdas must be nonempty and unique")
    return normalized


def fit_hydraulic_grouped_cv_cuda(
    development_features: Any,
    valve_labels: Any,
    nuisance_groups: Any,
    *,
    ridge_lambdas: Sequence[float] = RIDGE_LAMBDA_GRID,
    fold_by_cycle: Any,
    kappa: float | Sequence[float] | np.ndarray = DEFAULT_KAPPA,
    device: str = "cuda",
    max_iter: int = 500,
    max_eval: int | None = 625,
    tolerance_grad: float = 1.0e-7,
    tolerance_change: float = 1.0e-10,
    history_size: int = 100,
    require_convergence: bool = False,
) -> tuple[HydraulicDirectionalModel, HydraulicCVResult]:
    """Select ridge lambda by nuisance-grouped development-only CV."""
    protocol_folds = protocol_grouped_fold_ids(nuisance_groups)
    features, labels, groups = _validate_development_data(
        development_features,
        valve_labels,
        nuisance_groups,
    )
    folds = validate_grouped_folds(labels, groups, fold_by_cycle)
    if not np.array_equal(folds, protocol_folds):
        raise ValueError(
            "fold_by_cycle does not match the frozen SHA256 protocol assignment"
        )
    lambdas = _ridge_grid(ridge_lambdas)
    if lambdas != RIDGE_LAMBDA_GRID:
        raise ValueError(
            f"protocol ridge_lambdas must be exactly {RIDGE_LAMBDA_GRID}"
        )
    common = {
        "kappa": kappa,
        "device": device,
        "max_iter": max_iter,
        "max_eval": max_eval,
        "tolerance_grad": tolerance_grad,
        "tolerance_change": tolerance_change,
        "history_size": history_size,
        "require_convergence": require_convergence,
    }
    fold_results: list[HydraulicCVFoldResult] = []
    mean_losses: list[tuple[float, float]] = []
    for ridge_lambda in lambdas:
        candidate_losses: list[float] = []
        for held_out_fold in sorted(set(folds.tolist())):
            training_mask = folds != held_out_fold
            held_mask = ~training_mask
            train_groups = np.asarray(groups, dtype=str)[training_mask]
            held_groups = np.asarray(groups, dtype=str)[held_mask]
            fold_model, convergence = fit_hydraulic_directional_cuda(
                features[training_mask],
                labels[training_mask],
                train_groups,
                ridge_lambda=ridge_lambda,
                **common,
            )
            validation_loss = balanced_validation_log_loss_numpy(
                fold_model,
                features[held_mask],
                labels[held_mask],
                held_groups,
            )
            candidate_losses.append(validation_loss)
            fold_results.append(
                HydraulicCVFoldResult(
                    ridge_lambda=ridge_lambda,
                    held_out_fold=held_out_fold,
                    training_cycle_count=int(np.count_nonzero(training_mask)),
                    held_out_cycle_count=int(np.count_nonzero(held_mask)),
                    held_out_nuisance_groups=tuple(sorted(set(held_groups.tolist()))),
                    validation_log_loss=validation_loss,
                    convergence=convergence,
                )
            )
        mean_losses.append((ridge_lambda, float(np.mean(candidate_losses))))
    selected_lambda = min(mean_losses, key=lambda item: (item[1], -item[0]))[0]
    final_model, final_fit = fit_hydraulic_directional_cuda(
        features,
        labels,
        groups,
        ridge_lambda=selected_lambda,
        **common,
    )
    return final_model, HydraulicCVResult(
        selected_lambda=selected_lambda,
        mean_validation_loss_by_lambda=tuple(mean_losses),
        folds=tuple(fold_results),
        final_fit=final_fit,
    )


def score_hydraulic_cuda(
    model: HydraulicDirectionalModel,
    features: Any,
    *,
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[HydraulicDirectionalPath, dict[str, object]]:
    """Independently score batches on CUDA and return CPU NumPy diagnostics."""
    if not isinstance(model, HydraulicDirectionalModel):
        raise TypeError("model must be a HydraulicDirectionalModel")
    values = validate_residual_features(features, name="CUDA scoring features")
    batch_size = _positive_integer(batch_size, name="batch_size")
    torch, target, runtime = _torch_and_cuda_device(device)
    tensors = {
        "weights": torch.as_tensor(
            np.array(model.weights, copy=True),
            dtype=torch.float64,
            device=target,
        ),
        "bias": torch.as_tensor(
            np.array(model.bias, copy=True),
            dtype=torch.float64,
            device=target,
        ),
        "center": torch.as_tensor(
            np.array(model.phase_logit_center, copy=True),
            dtype=torch.float64,
            device=target,
        ),
        "scale": torch.as_tensor(
            np.array(model.phase_logit_scale, copy=True),
            dtype=torch.float64,
            device=target,
        ),
        "kappa": torch.as_tensor(
            np.array(model.kappa, copy=True),
            dtype=torch.float64,
            device=target,
        ),
    }
    names = (
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
        "envelope",
        "eligible",
        "alarm",
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(target)
    with _tf32_disabled(torch), torch.no_grad():
        begin.record()
        for start in range(0, values.shape[0], batch_size):
            stop = min(values.shape[0], start + batch_size)
            batch = torch.as_tensor(
                values[start:stop],
                dtype=torch.float64,
                device=target,
            )
            logits = batch @ tensors["weights"].T + tensors["bias"]
            standardized = (logits - tensors["center"]) / tensors["scale"]
            clipped = torch.clamp(standardized, min=-Z_CLIP, max=Z_CLIP)
            increments = clipped - tensors["kappa"]
            partial = torch.cumsum(increments, dim=1)
            augmented = torch.cat(
                (
                    torch.zeros(
                        (batch.shape[0], 1, HEAD_COUNT),
                        dtype=torch.float64,
                        device=target,
                    ),
                    partial,
                ),
                dim=1,
            )
            running_minimum = torch.cummin(augmented, dim=1).values[:, 1:]
            cusum = torch.clamp_min(partial - running_minimum, 0.0)
            component = torch.full_like(cusum, -torch.inf)
            windows = cusum.unfold(1, PERSISTENCE_WINDOW, 1)
            component[:, PERSISTENCE_WINDOW - 1 :] = torch.min(
                windows,
                dim=-1,
            ).values
            envelope = torch.max(component, dim=2).values
            eligible = torch.zeros(
                envelope.shape,
                dtype=torch.bool,
                device=target,
            )
            eligible[:, PERSISTENCE_WINDOW - 1 :] = True
            if model.threshold is None:
                alarm = torch.zeros_like(eligible)
            else:
                alarm = eligible & (envelope > model.threshold)
            arrays = {
                "logits": logits,
                "standardized_logits": standardized,
                "clipped_evidence": clipped,
                "cusum": cusum,
                "component_persistence": component,
                "envelope": envelope,
                "eligible": eligible,
                "alarm": alarm,
            }
            for name, tensor in arrays.items():
                collected[name].append(tensor.cpu().numpy())
        end.record()
        end.synchronize()
        elapsed_ms = float(begin.elapsed_time(end))
    joined = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    path = HydraulicDirectionalPath(
        positive_labels=POSITIVE_LABELS,
        logits=joined["logits"],
        standardized_logits=joined["standardized_logits"],
        clipped_evidence=joined["clipped_evidence"],
        cusum=joined["cusum"],
        component_persistence=joined["component_persistence"],
        envelope=joined["envelope"],
        eligible=joined["eligible"],
        alarm=joined["alarm"],
        threshold=model.threshold,
    )
    execution = {
        "backend": "pytorch_cuda_batched_hydraulic_scoring",
        "device": str(target),
        "device_name": runtime["device_name"],
        "dtype": "float64",
        "tf32": False,
        "cycle_count": int(values.shape[0]),
        "phase_count": PHASE_COUNT,
        "feature_count": FEATURE_COUNT,
        "head_count": HEAD_COUNT,
        "batch_size": batch_size,
        "page_state_scope": "reset_per_cycle",
        "cuda_event_elapsed_seconds": elapsed_ms / 1000.0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(target)),
        "runtime_environment": runtime,
    }
    return path, execution


def _finite_maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise AssertionError(f"array shape mismatch: {left.shape} != {right.shape}")
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        raise AssertionError("NaN locations differ")
    if not np.array_equal(np.isposinf(left), np.isposinf(right)):
        raise AssertionError("positive-infinity locations differ")
    if not np.array_equal(np.isneginf(left), np.isneginf(right)):
        raise AssertionError("negative-infinity locations differ")
    finite = np.isfinite(left)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def compare_hydraulic_paths(
    reference: HydraulicDirectionalPath,
    candidate: HydraulicDirectionalPath,
    *,
    atol: float = 1.0e-10,
) -> dict[str, float | bool]:
    """Assert NumPy/CUDA numeric agreement and exact binary decisions."""
    atol = _nonnegative_float(atol, name="atol")
    if reference.positive_labels != candidate.positive_labels:
        raise AssertionError("positive head order differs")
    errors: dict[str, float] = {}
    for name in (
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
        "envelope",
    ):
        errors[name] = _finite_maximum_error(
            np.asarray(getattr(reference, name)),
            np.asarray(getattr(candidate, name)),
        )
        if errors[name] > atol:
            raise AssertionError(
                f"{name} maximum absolute error {errors[name]:.3e} exceeds {atol:.3e}"
            )
    eligible_exact = bool(np.array_equal(reference.eligible, candidate.eligible))
    alarm_exact = bool(np.array_equal(reference.alarm, candidate.alarm))
    if not eligible_exact or not alarm_exact:
        raise AssertionError("eligibility or alarm decisions differ")
    return {
        **{f"{name}_max_abs_error": error for name, error in errors.items()},
        "eligible_exact": eligible_exact,
        "alarm_exact": alarm_exact,
    }


def _calibration_cycle_names(
    cycle_names: Sequence[str] | None,
    *,
    cycle_count: int,
) -> tuple[str, ...]:
    if cycle_names is None:
        return tuple(f"healthy-calibration-{index:06d}" for index in range(cycle_count))
    if isinstance(cycle_names, (str, bytes)) or len(cycle_names) != cycle_count:
        raise ValueError(f"cycle_names must contain exactly {cycle_count} names")
    names = tuple(cycle_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("cycle_names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("cycle_names must be unique")
    return names


def _complete_cycle_maximum(name: str, envelope: np.ndarray) -> float:
    values = np.asarray(envelope, dtype=np.float64)
    if values.shape != (PHASE_COUNT,):
        raise ValueError(
            f"envelope for {name!r} must cover exactly {PHASE_COUNT} seconds"
        )
    first = PERSISTENCE_WINDOW - 1
    if not np.isneginf(values[:first]).all() or not np.isfinite(values[first:]).all():
        raise ValueError(
            f"envelope for {name!r} must have {first} leading -inf values "
            "followed by a finite complete horizon"
        )
    maximum = float(np.max(values[first:]))
    if maximum < 0.0:
        raise ValueError(f"envelope maximum for {name!r} must be nonnegative")
    return maximum


def calibrate_hydraulic_pac(
    frozen_model: HydraulicDirectionalModel,
    healthy_calibration_features: Any,
    *,
    cycle_names: Sequence[str] | None = None,
    epsilon: float = 0.05,
    delta: float = 0.05,
    required_count: int | None = 90,
) -> tuple[HydraulicDirectionalModel, HydraulicPACCalibration]:
    """Attach a PAC run-maximum threshold from independent healthy cycles."""
    if not isinstance(frozen_model, HydraulicDirectionalModel):
        raise TypeError("frozen_model must be a HydraulicDirectionalModel")
    if not np.array_equal(
        frozen_model.kappa,
        np.full(HEAD_COUNT, DEFAULT_KAPPA, dtype=np.float64),
    ):
        raise ValueError(
            f"PAC-PADR-CUSUM calibration requires protocol kappa={DEFAULT_KAPPA}"
        )
    features = validate_residual_features(
        healthy_calibration_features,
        name="healthy calibration features",
    )
    if required_count is not None:
        required_count = _positive_integer(required_count, name="required_count")
        if features.shape[0] != required_count:
            raise ValueError(
                f"expected exactly {required_count} healthy calibration cycles; "
                f"received {features.shape[0]}"
            )
    names = _calibration_cycle_names(cycle_names, cycle_count=features.shape[0])
    unthresholded = frozen_model.with_threshold(None)
    path = score_hydraulic_numpy(unthresholded, features)
    maxima = tuple(
        (name, _complete_cycle_maximum(name, path.envelope[index]))
        for index, name in enumerate(names)
    )
    rank = pac_tolerance_rank(
        len(maxima),
        epsilon=epsilon,
        delta=delta,
    )
    ordered = sorted(maxima, key=lambda item: (item[1], item[0]))
    driver, threshold = ordered[rank.rank_one_based - 1]
    model_fingerprint = hydraulic_directional_model_sha256(unthresholded)
    calibration_fingerprint = hydraulic_calibration_sha256(
        model_sha256=model_fingerprint,
        run_maxima=maxima,
        rank=rank,
    )
    calibration = HydraulicPACCalibration(
        status=rank.status,
        threshold=threshold,
        rank=rank,
        rank_driver_cycle=driver,
        run_maxima=maxima,
        scorer_spec_sha256=HYDRAULIC_SCORER_SPEC_SHA256,
        model_sha256=model_fingerprint,
        calibration_sha256=calibration_fingerprint,
    )
    return frozen_model.with_threshold(calibration.threshold), calibration


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON safe")


def save_hydraulic_model(
    model: HydraulicDirectionalModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the strict numeric/JSON model schema without pickle."""
    if not isinstance(model, HydraulicDirectionalModel):
        raise TypeError("model must be a HydraulicDirectionalModel")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("hydraulic model path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(
        {} if metadata is None else dict(metadata),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    threshold = (
        np.empty(0, dtype=np.float64)
        if model.threshold is None
        else np.asarray([model.threshold], dtype=np.float64)
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(
                handle,
                schema=np.asarray(MODEL_SCHEMA),
                schema_version=np.asarray([MODEL_SCHEMA_VERSION], dtype=np.int64),
                positive_labels=np.asarray(POSITIVE_LABELS, dtype=np.int64),
                weights=model.weights,
                bias=model.bias,
                phase_logit_center=model.phase_logit_center,
                phase_logit_scale=model.phase_logit_scale,
                kappa=model.kappa,
                threshold=threshold,
                metadata_json=np.asarray(metadata_text),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def _load_numeric_archive_array(archive: Any, key: str) -> np.ndarray:
    """Load one real numeric field without permitting coercion from text/pickle."""
    try:
        array = np.asarray(archive[key])
    except ValueError as error:
        raise ValueError(f"{key} must be a non-pickled real numeric array") from error
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{key} must have a real numeric dtype")
    return array


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"metadata_json contains forbidden non-finite constant {value}")


def load_hydraulic_model(
    path: str | os.PathLike[str],
) -> tuple[HydraulicDirectionalModel, dict[str, Any]]:
    """Load and validate the exact archive schema with ``allow_pickle=False``."""
    required = {
        "schema",
        "schema_version",
        "positive_labels",
        "weights",
        "bias",
        "phase_logit_center",
        "phase_logit_scale",
        "kappa",
        "threshold",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != required:
            raise ValueError(
                "invalid hydraulic model archive keys; "
                f"missing={sorted(required - keys)}, extra={sorted(keys - required)}"
            )
        schema = np.asarray(archive["schema"])
        version = np.asarray(archive["schema_version"])
        positive_labels = np.asarray(archive["positive_labels"])
        metadata_json = np.asarray(archive["metadata_json"])
        if schema.shape != () or schema.dtype.kind not in "US":
            raise ValueError("schema must be a scalar string")
        if str(schema.item()) != MODEL_SCHEMA:
            raise ValueError("unsupported hydraulic model schema")
        if version.shape != (1,) or version.dtype.kind not in "iu":
            raise ValueError("schema_version must be an integer vector with shape (1,)")
        if int(version[0]) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported hydraulic model schema version")
        if (
            positive_labels.shape != (HEAD_COUNT,)
            or positive_labels.dtype.kind not in "iu"
        ):
            raise ValueError(
                f"positive_labels must be an integer vector with shape ({HEAD_COUNT},)"
            )
        if tuple(int(value) for value in positive_labels.tolist()) != POSITIVE_LABELS:
            raise ValueError(f"positive_labels must be exactly {POSITIVE_LABELS}")
        if metadata_json.shape != () or metadata_json.dtype.kind not in "US":
            raise ValueError("metadata_json must be a scalar string")
        numeric = {
            key: _load_numeric_archive_array(archive, key)
            for key in (
                "weights",
                "bias",
                "phase_logit_center",
                "phase_logit_scale",
                "kappa",
                "threshold",
            )
        }
        threshold = numeric["threshold"]
        if threshold.shape not in ((0,), (1,)):
            raise ValueError("threshold archive field must contain zero or one value")
        if threshold.size == 1 and not np.isfinite(threshold[0]):
            raise ValueError("threshold archive value must be finite")
        model = HydraulicDirectionalModel(
            positive_labels=tuple(int(value) for value in positive_labels.tolist()),
            weights=numeric["weights"],
            bias=numeric["bias"],
            phase_logit_center=numeric["phase_logit_center"],
            phase_logit_scale=numeric["phase_logit_scale"],
            kappa=numeric["kappa"],
            threshold=None if threshold.size == 0 else float(threshold[0]),
        )
        try:
            metadata = json.loads(
                str(metadata_json.item()),
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError("metadata_json must contain strict finite JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("model metadata must decode to a JSON object")
    return model, metadata


__all__ = [
    "CV_FOLD_COUNT",
    "CV_FOLD_SALT",
    "DEFAULT_KAPPA",
    "FEATURE_COUNT",
    "GLOBAL_SCALE_FLOOR_FRACTION",
    "HEAD_COUNT",
    "HEALTHY_LABEL",
    "HYDRAULIC_SCORER_SPEC_SHA256",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "PERSISTENCE_WINDOW",
    "PHASE_COUNT",
    "POSITIVE_LABELS",
    "ROBUST_SCALE_FACTOR",
    "RIDGE_LAMBDA_GRID",
    "SCALE_FLOOR",
    "STRICT_COMPARISON",
    "VALVE_LABELS",
    "Z_CLIP",
    "HydraulicCVFoldResult",
    "HydraulicCVResult",
    "HydraulicDirectionalModel",
    "HydraulicDirectionalPath",
    "HydraulicPACCalibration",
    "balanced_validation_log_loss_numpy",
    "calibrate_hydraulic_pac",
    "cell_balanced_cycle_weights",
    "compare_hydraulic_paths",
    "fit_healthy_phase_normalization",
    "fit_hydraulic_directional_cuda",
    "fit_hydraulic_grouped_cv_cuda",
    "hydraulic_calibration_sha256",
    "hydraulic_directional_model_sha256",
    "hydraulic_scorer_spec",
    "load_hydraulic_model",
    "normalize_kappa",
    "predict_hydraulic_logits_numpy",
    "protocol_grouped_fold_ids",
    "save_hydraulic_model",
    "score_hydraulic_cuda",
    "score_hydraulic_numpy",
    "validate_grouped_folds",
    "validate_residual_features",
    "validate_valve_labels",
]
