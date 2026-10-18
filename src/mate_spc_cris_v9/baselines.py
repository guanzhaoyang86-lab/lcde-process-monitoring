"""Frozen healthy-only baselines for the CRIS-v9 synthetic study.

The public workflow is deliberately identical for all three baselines::

    baseline = BaselineClass.fit(healthy_fit_runs)
    threshold = baseline.calibrate_trajectory_threshold(healthy_calibration_runs)
    score = baseline.score(one_run)
    decision = baseline.apply_trajectory_threshold(one_run, threshold)

Only numeric trajectories are accepted; labels, fault identities, and onset
times are absent from every fitting and calibration API.  Calibration always
reduces each complete, independent healthy trajectory to exactly one maximum.
Every alarm comparison is strict ``score > threshold``.

The predictive-residual baseline accepts the aligned innovations produced by
``CRISModel.innovations``.  It intentionally discards their directional and
cross-coordinate structure and applies only ordinary squared energy plus a
one-sided CUSUM; it is not the CRIS-v9 candidate monitor.
An all-NaN innovation timestamp contributes zero increment and holds the
existing CUSUM state; partially observed timestamps average only finite terms.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Protocol, Sequence

import numpy as np


PCA_VARIANCE_FRACTION = 0.90
CUSUM_ALLOWANCE = 0.50
SCALE_FLOOR = 1.0e-8
STRICT_COMPARISON = "strictly_greater_than"


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Integral, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _readonly(value: Any, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _numeric_run(value: Any, *, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric array")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if result.ndim != ndim or result.shape[0] < 1 or any(size < 1 for size in result.shape[1:]):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional trajectory")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _innovation_run(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric array")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    if result.ndim != 3 or min(result.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape (T,H,D)")
    if np.isinf(result).any():
        raise ValueError(f"{name} may contain leading NaN, but not infinity")
    return result


def _run_collection(values: Any, *, ndim: int, name: str) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim == ndim:
            supplied: Sequence[Any] = (values,)
        elif values.ndim == ndim + 1:
            supplied = tuple(values[index] for index in range(values.shape[0]))
        else:
            raise ValueError(f"{name} has the wrong number of dimensions")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        supplied = values
    else:
        raise ValueError(f"{name} must be an ndarray or a sequence of trajectories")
    if len(supplied) < 1:
        raise ValueError(f"{name} must contain at least one complete trajectory")
    return tuple(
        _numeric_run(value, ndim=ndim, name=f"{name}[{index}]")
        for index, value in enumerate(supplied)
    )


def _innovation_collection(values: Any, *, name: str) -> tuple[np.ndarray, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            supplied: Sequence[Any] = (values,)
        elif values.ndim == 4:
            supplied = tuple(values[index] for index in range(values.shape[0]))
        else:
            raise ValueError(f"{name} has the wrong number of dimensions")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        supplied = values
    else:
        raise ValueError(f"{name} must be an ndarray or a sequence of trajectories")
    if len(supplied) < 1:
        raise ValueError(f"{name} must contain at least one complete trajectory")
    return tuple(
        _innovation_run(value, name=f"{name}[{index}]")
        for index, value in enumerate(supplied)
    )


@dataclass(frozen=True, slots=True)
class BaselineScore:
    """One finite causal point-evidence and monitoring path."""

    baseline_name: str
    point_score: np.ndarray
    path: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_name, str) or not self.baseline_name:
            raise ValueError("baseline_name must be nonempty text")
        point = np.asarray(self.point_score)
        path = np.asarray(self.path)
        if point.dtype != np.dtype(np.float64) or path.dtype != np.dtype(np.float64):
            raise ValueError("point_score and path must be float64")
        if point.ndim != 1 or path.shape != point.shape or point.size < 1:
            raise ValueError("point_score and path must have the same nonempty shape (T,)")
        if not np.isfinite(point).all() or not np.isfinite(path).all():
            raise ValueError("baseline scores must be finite")
        object.__setattr__(self, "point_score", _readonly(point, dtype=np.float64))
        object.__setattr__(self, "path", _readonly(path, dtype=np.float64))

    @property
    def trajectory_maximum(self) -> float:
        return float(np.max(self.path))


@dataclass(frozen=True, slots=True)
class BaselineThreshold:
    """One split-conformal order statistic over whole-trajectory maxima."""

    baseline_name: str
    alpha: float
    rank: int
    threshold: float
    calibration_maxima: np.ndarray
    comparison: str = STRICT_COMPARISON

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_name, str) or not self.baseline_name:
            raise ValueError("baseline_name must be nonempty text")
        alpha = _finite_real(self.alpha, name="alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        rank = _positive_integer(self.rank, name="rank")
        maxima = np.asarray(self.calibration_maxima)
        if maxima.dtype != np.dtype(np.float64) or maxima.ndim != 1 or maxima.size < 1:
            raise ValueError("calibration_maxima must be a nonempty float64 vector")
        if not np.isfinite(maxima).all() or rank > maxima.size:
            raise ValueError("calibration maxima and rank are invalid")
        threshold = _finite_real(self.threshold, name="threshold")
        expected = float(np.sort(maxima, kind="stable")[rank - 1])
        if threshold != expected:
            raise ValueError("threshold is not the declared order statistic")
        if self.comparison != STRICT_COMPARISON:
            raise ValueError(f"comparison must be {STRICT_COMPARISON!r}")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "calibration_maxima", _readonly(maxima, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class TrajectoryDecision:
    """Strict-threshold alarm trace for one complete trajectory."""

    baseline_name: str
    threshold: float
    maximum: float
    alarm: np.ndarray
    first_alarm_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_name, str) or not self.baseline_name:
            raise ValueError("baseline_name must be nonempty text")
        threshold = _finite_real(self.threshold, name="threshold")
        maximum = _finite_real(self.maximum, name="maximum")
        raw_alarm = np.asarray(self.alarm)
        if raw_alarm.dtype != np.dtype(np.bool_) or raw_alarm.ndim != 1 or raw_alarm.size < 1:
            raise ValueError("alarm must be a nonempty boolean vector")
        alarm = _readonly(raw_alarm, dtype=np.bool_)
        expected_first = int(np.flatnonzero(alarm)[0]) if np.any(alarm) else None
        if self.first_alarm_index != expected_first:
            raise ValueError("first_alarm_index does not match alarm")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "alarm", alarm)

    @property
    def detected(self) -> bool:
        return self.first_alarm_index is not None


class _ScoringBaseline(Protocol):
    name: str

    def score(self, values: Any) -> BaselineScore: ...


class _BaselineInterface:
    """Shared calibration and strict-decision methods."""

    name: str

    def score(self, values: Any) -> BaselineScore:  # pragma: no cover - abstract protocol
        raise NotImplementedError

    def calibrate_trajectory_threshold(
        self,
        healthy_calibration_runs: Any,
        *,
        alpha: float = 0.10,
    ) -> BaselineThreshold:
        return calibrate_trajectory_threshold(self, healthy_calibration_runs, alpha=alpha)

    def apply_trajectory_threshold(
        self, values: Any, threshold: BaselineThreshold
    ) -> TrajectoryDecision:
        return apply_trajectory_threshold(self.score(values), threshold)


def _matrix_dimensions(runs: Sequence[np.ndarray], *, name: str) -> int:
    feature_dim = int(runs[0].shape[1])
    if any(run.shape[1] != feature_dim for run in runs):
        raise ValueError(f"all {name} trajectories must have the same feature dimension")
    return feature_dim


def _standardization(runs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate(runs, axis=0)
    center = np.mean(joined, axis=0, dtype=np.float64)
    scale = np.maximum(np.std(joined, axis=0, dtype=np.float64), SCALE_FLOOR)
    return center, scale


def _squared_energy(standardized: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.mean(np.square(standardized), axis=1, dtype=np.float64)
    if not np.isfinite(energy).all():
        raise ValueError("squared energy overflowed")
    return energy


def _cusum(increments: np.ndarray) -> np.ndarray:
    values = _numeric_run(increments[:, None], ndim=2, name="CUSUM increments")[:, 0]
    result = np.empty(values.shape, dtype=np.float64)
    state = 0.0
    for index, increment in enumerate(values):
        state = max(0.0, state + float(increment))
        if not math.isfinite(state):
            raise ValueError("CUSUM path overflowed")
        result[index] = state
    return result


@dataclass(frozen=True, slots=True)
class PCAT2QBaseline(_BaselineInterface):
    """Healthy-fit standardized PCA with a combined T2 and Q point score."""

    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    t2_reference_mean: float
    q_reference_mean: float
    variance_fraction: float = PCA_VARIANCE_FRACTION
    name: str = "pca_t2_plus_q"

    def __post_init__(self) -> None:
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        components = np.asarray(self.components)
        eigenvalues = np.asarray(self.eigenvalues)
        if center.dtype != np.dtype(np.float64) or center.ndim != 1 or center.size < 1:
            raise ValueError("PCA center must be a nonempty float64 vector")
        if scale.dtype != np.dtype(np.float64) or scale.shape != center.shape:
            raise ValueError("PCA scale must match center")
        if components.dtype != np.dtype(np.float64) or components.ndim != 2:
            raise ValueError("PCA components must be a float64 matrix")
        if components.shape[1] != center.size or not 1 <= components.shape[0] <= center.size:
            raise ValueError("PCA component geometry is invalid")
        if eigenvalues.dtype != np.dtype(np.float64) or eigenvalues.shape != (
            components.shape[0],
        ):
            raise ValueError("PCA eigenvalues do not match components")
        if (
            not np.isfinite(center).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(components).all()
            or not np.isfinite(eigenvalues).all()
            or np.any(scale <= 0.0)
            or np.any(eigenvalues <= 0.0)
        ):
            raise ValueError("PCA state must be finite with positive scales")
        t2_mean = _finite_real(self.t2_reference_mean, name="t2_reference_mean")
        q_mean = _finite_real(self.q_reference_mean, name="q_reference_mean")
        variance_fraction = _finite_real(self.variance_fraction, name="variance_fraction")
        if t2_mean <= 0.0 or q_mean <= 0.0:
            raise ValueError("PCA reference means must be positive")
        if variance_fraction != PCA_VARIANCE_FRACTION:
            raise ValueError(f"variance_fraction is frozen at {PCA_VARIANCE_FRACTION}")
        if self.name != "pca_t2_plus_q":
            raise ValueError("PCA baseline name is frozen")
        object.__setattr__(self, "center", _readonly(center, dtype=np.float64))
        object.__setattr__(self, "scale", _readonly(scale, dtype=np.float64))
        object.__setattr__(self, "components", _readonly(components, dtype=np.float64))
        object.__setattr__(self, "eigenvalues", _readonly(eigenvalues, dtype=np.float64))
        object.__setattr__(self, "t2_reference_mean", t2_mean)
        object.__setattr__(self, "q_reference_mean", q_mean)

    @classmethod
    def fit(cls, healthy_fit_runs: Any) -> "PCAT2QBaseline":
        runs = _run_collection(healthy_fit_runs, ndim=2, name="healthy_fit_runs")
        feature_dim = _matrix_dimensions(runs, name="healthy fit")
        center, scale = _standardization(runs)
        standardized = (np.concatenate(runs, axis=0) - center[None, :]) / scale[None, :]
        if standardized.shape[0] < 2:
            raise ValueError("PCA fitting requires at least two healthy observations")
        _left, singular_values, right = np.linalg.svd(standardized, full_matrices=False)
        variances = np.square(singular_values) / (standardized.shape[0] - 1)
        positive = variances > SCALE_FLOOR
        if not np.any(positive):
            raise ValueError("healthy PCA reference has no positive-variance direction")
        positive_variances = variances[positive]
        cumulative = np.cumsum(positive_variances) / np.sum(positive_variances)
        component_count = int(np.searchsorted(cumulative, PCA_VARIANCE_FRACTION) + 1)
        component_count = min(component_count, positive_variances.size)
        components = right[:component_count]
        eigenvalues = np.maximum(variances[:component_count], SCALE_FLOOR)
        projected = standardized @ components.T
        with np.errstate(over="ignore", invalid="ignore"):
            t2 = np.sum(np.square(projected) / eigenvalues[None, :], axis=1)
            residual = standardized - projected @ components
            q = np.sum(np.square(residual), axis=1)
        if not np.isfinite(t2).all() or not np.isfinite(q).all():
            raise ValueError("healthy PCA score overflowed")
        return cls(
            center=np.asarray(center, dtype=np.float64),
            scale=np.asarray(scale, dtype=np.float64),
            components=np.asarray(components, dtype=np.float64),
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            t2_reference_mean=max(float(np.mean(t2)), SCALE_FLOOR),
            q_reference_mean=max(float(np.mean(q)), SCALE_FLOOR),
        )

    def score(self, values: Any) -> BaselineScore:
        run = _numeric_run(values, ndim=2, name="values")
        if run.shape[1] != self.center.size:
            raise ValueError("values feature dimension does not match fitted PCA")
        standardized = (run - self.center[None, :]) / self.scale[None, :]
        projected = standardized @ self.components.T
        with np.errstate(over="ignore", invalid="ignore"):
            t2 = np.sum(np.square(projected) / self.eigenvalues[None, :], axis=1)
            residual = standardized - projected @ self.components
            q = np.sum(np.square(residual), axis=1)
            combined = t2 / self.t2_reference_mean + q / self.q_reference_mean
        if not np.isfinite(combined).all():
            raise ValueError("PCA T2+Q score overflowed")
        point = np.asarray(combined, dtype=np.float64)
        return BaselineScore(self.name, point, point)


@dataclass(frozen=True, slots=True)
class EnergyCUSUMBaseline(_BaselineInterface):
    """Ordinary standardized observation-energy CUSUM."""

    center: np.ndarray
    scale: np.ndarray
    energy_center: float
    energy_scale: float
    allowance: float = CUSUM_ALLOWANCE
    name: str = "energy_cusum"

    def __post_init__(self) -> None:
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        if center.dtype != np.dtype(np.float64) or center.ndim != 1 or center.size < 1:
            raise ValueError("energy center must be a nonempty float64 vector")
        if scale.dtype != np.dtype(np.float64) or scale.shape != center.shape:
            raise ValueError("energy scale vector must match center")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise ValueError("energy standardization must be finite and positive")
        energy_center = _finite_real(self.energy_center, name="energy_center")
        energy_scale = _finite_real(self.energy_scale, name="energy_scale")
        allowance = _finite_real(self.allowance, name="allowance")
        if energy_scale <= 0.0 or allowance != CUSUM_ALLOWANCE:
            raise ValueError("energy CUSUM scale/allowance is invalid")
        if self.name != "energy_cusum":
            raise ValueError("energy baseline name is frozen")
        object.__setattr__(self, "center", _readonly(center, dtype=np.float64))
        object.__setattr__(self, "scale", _readonly(scale, dtype=np.float64))
        object.__setattr__(self, "energy_center", energy_center)
        object.__setattr__(self, "energy_scale", energy_scale)
        object.__setattr__(self, "allowance", allowance)

    @classmethod
    def fit(cls, healthy_fit_runs: Any) -> "EnergyCUSUMBaseline":
        runs = _run_collection(healthy_fit_runs, ndim=2, name="healthy_fit_runs")
        _matrix_dimensions(runs, name="healthy fit")
        center, scale = _standardization(runs)
        energies = np.concatenate(
            [_squared_energy((run - center[None, :]) / scale[None, :]) for run in runs]
        )
        if energies.size < 2:
            raise ValueError("energy fitting requires at least two healthy observations")
        return cls(
            center=np.asarray(center, dtype=np.float64),
            scale=np.asarray(scale, dtype=np.float64),
            energy_center=float(np.mean(energies)),
            energy_scale=max(float(np.std(energies)), SCALE_FLOOR),
        )

    def score(self, values: Any) -> BaselineScore:
        run = _numeric_run(values, ndim=2, name="values")
        if run.shape[1] != self.center.size:
            raise ValueError("values feature dimension does not match fitted energy baseline")
        energy = _squared_energy((run - self.center[None, :]) / self.scale[None, :])
        increments = (energy - self.energy_center) / self.energy_scale - self.allowance
        if not np.isfinite(increments).all():
            raise ValueError("energy CUSUM evidence overflowed")
        return BaselineScore(
            self.name,
            np.asarray(increments, dtype=np.float64),
            _cusum(np.asarray(increments, dtype=np.float64)),
        )


def _innovation_energy(run: np.ndarray) -> np.ndarray:
    finite = np.isfinite(run)
    count = np.sum(finite, axis=(1, 2), dtype=np.int64)
    filled = np.where(finite, run, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_sum = np.sum(np.square(filled), axis=(1, 2), dtype=np.float64)
    energy = np.full(run.shape[0], np.nan, dtype=np.float64)
    available = count > 0
    energy[available] = squared_sum[available] / count[available]
    if np.isinf(energy).any():
        raise ValueError("predictive residual energy overflowed")
    return energy


@dataclass(frozen=True, slots=True)
class PredictiveResidualCUSUMBaseline(_BaselineInterface):
    """Ordinary squared CRIS-innovation energy followed by a CUSUM."""

    innovation_shape: tuple[int, int]
    energy_center: float
    energy_scale: float
    allowance: float = CUSUM_ALLOWANCE
    name: str = "predictive_residual_cusum"

    def __post_init__(self) -> None:
        if (
            type(self.innovation_shape) is not tuple
            or len(self.innovation_shape) != 2
        ):
            raise ValueError("innovation_shape must be a length-two tuple")
        shape = tuple(
            _positive_integer(value, name=f"innovation_shape[{index}]")
            for index, value in enumerate(self.innovation_shape)
        )
        energy_center = _finite_real(self.energy_center, name="energy_center")
        energy_scale = _finite_real(self.energy_scale, name="energy_scale")
        allowance = _finite_real(self.allowance, name="allowance")
        if energy_scale <= 0.0 or allowance != CUSUM_ALLOWANCE:
            raise ValueError("predictive residual CUSUM scale/allowance is invalid")
        if self.name != "predictive_residual_cusum":
            raise ValueError("predictive residual baseline name is frozen")
        object.__setattr__(self, "innovation_shape", shape)
        object.__setattr__(self, "energy_center", energy_center)
        object.__setattr__(self, "energy_scale", energy_scale)
        object.__setattr__(self, "allowance", allowance)

    @classmethod
    def fit(cls, healthy_fit_runs: Any) -> "PredictiveResidualCUSUMBaseline":
        runs = _innovation_collection(healthy_fit_runs, name="healthy_fit_runs")
        shape = tuple(int(value) for value in runs[0].shape[1:])
        if any(run.shape[1:] != shape for run in runs):
            raise ValueError("all healthy innovation trajectories must share (H,D)")
        energies = np.concatenate([_innovation_energy(run) for run in runs])
        finite = energies[np.isfinite(energies)]
        if finite.size < 2:
            raise ValueError("predictive baseline needs two finite healthy energy points")
        return cls(
            innovation_shape=shape,
            energy_center=float(np.mean(finite)),
            energy_scale=max(float(np.std(finite)), SCALE_FLOOR),
        )

    def score(self, values: Any) -> BaselineScore:
        run = _innovation_run(values, name="values")
        if run.shape[1:] != self.innovation_shape:
            raise ValueError("values innovation shape does not match fitted baseline")
        energy = _innovation_energy(run)
        available = np.isfinite(energy)
        increments = np.zeros(energy.shape, dtype=np.float64)
        increments[available] = (
            (energy[available] - self.energy_center) / self.energy_scale - self.allowance
        )
        if not np.isfinite(increments).all():
            raise ValueError("predictive residual CUSUM evidence overflowed")
        return BaselineScore(self.name, increments, _cusum(increments))


def _calibration_runs_for_baseline(
    baseline: _ScoringBaseline, values: Any
) -> tuple[np.ndarray, ...]:
    if isinstance(baseline, PredictiveResidualCUSUMBaseline):
        return _innovation_collection(values, name="healthy_calibration_runs")
    return _run_collection(values, ndim=2, name="healthy_calibration_runs")


def calibrate_trajectory_threshold(
    baseline: _ScoringBaseline,
    healthy_calibration_runs: Any,
    *,
    alpha: float = 0.10,
) -> BaselineThreshold:
    """Calibrate from exactly one maximum per complete independent healthy run."""

    if not hasattr(baseline, "score") or not isinstance(getattr(baseline, "name", None), str):
        raise ValueError("baseline must expose name and score")
    alpha_value = _finite_real(alpha, name="alpha")
    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    runs = _calibration_runs_for_baseline(baseline, healthy_calibration_runs)
    maxima = np.asarray(
        [baseline.score(run).trajectory_maximum for run in runs], dtype=np.float64
    )
    rank = int(math.ceil((maxima.size + 1) * (1.0 - alpha_value)))
    if rank > maxima.size:
        raise ValueError(
            "requested trajectory threshold is unreachable for this calibration count"
        )
    threshold = float(np.sort(maxima, kind="stable")[rank - 1])
    return BaselineThreshold(
        baseline_name=baseline.name,
        alpha=alpha_value,
        rank=rank,
        threshold=threshold,
        calibration_maxima=maxima,
    )


def apply_trajectory_threshold(
    score: BaselineScore, threshold: BaselineThreshold
) -> TrajectoryDecision:
    """Apply the frozen strict ``>`` trajectory threshold."""

    if not isinstance(score, BaselineScore) or not isinstance(threshold, BaselineThreshold):
        raise ValueError("score and threshold have wrong types")
    if score.baseline_name != threshold.baseline_name:
        raise ValueError("score and threshold belong to different baselines")
    alarm = np.asarray(score.path > threshold.threshold, dtype=np.bool_)
    first = int(np.flatnonzero(alarm)[0]) if np.any(alarm) else None
    return TrajectoryDecision(
        baseline_name=score.baseline_name,
        threshold=threshold.threshold,
        maximum=score.trajectory_maximum,
        alarm=alarm,
        first_alarm_index=first,
    )


__all__ = [
    "BaselineScore",
    "BaselineThreshold",
    "CUSUM_ALLOWANCE",
    "EnergyCUSUMBaseline",
    "PCAT2QBaseline",
    "PCA_VARIANCE_FRACTION",
    "PredictiveResidualCUSUMBaseline",
    "STRICT_COMPARISON",
    "TrajectoryDecision",
    "apply_trajectory_threshold",
    "calibrate_trajectory_threshold",
]
