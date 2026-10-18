"""Causal innovation scan for the CRIS-v9 candidate.

This module is intentionally independent of the forecasting model.  Its only
numeric input is a ``float64``-compatible innovation cube with shape
``(time, horizon, dimension)``.  Non-finite observations are treated as
UNKNOWN; infinities are rejected while NaNs pause the affected sequential
states.

The implementation makes the statistical contract explicit:

* location, scale, and lag-one dependence moments;
* causal trailing scales 4/16/64 and separate positive/negative top-k tails;
* upper-tail split-conformal p-values with the conservative ``>=`` tie rule;
* the power e-calibrator ``(1-eta) * p**(-eta)`` with ``eta=0.5``;
* two log-domain, two-state HMM scans; and
* a fixed-weight log-mean-exp merge (missing components are held, never
  opportunistically reweighted).

Trajectory thresholds are calibrated from one maximum per independent healthy
run.  The conformal order statistic is refused when the requested rank does
not exist, and alarms always use a strict ``>`` comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np


MOMENTS: tuple[str, ...] = ("location", "scale", "dependence")
SIDES: tuple[str, ...] = ("positive", "negative")
ROLLING_SCALES: tuple[int, ...] = (4, 16, 64)
HMM_MODES: tuple[tuple[float, float], ...] = ((0.02, 0.20), (0.10, 0.50))
DEFAULT_ETA = 0.5
DEFAULT_DEPENDENCE_LAG = 1
DEFAULT_RESET_GAP = 64


class UnreachableThresholdError(ValueError):
    """Raised when split-conformal calibration asks for rank ``k > n``."""


def _integer(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    owned = np.array(array, dtype=dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _validate_innovations(value: object, *, name: str = "innovations") -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a real numeric array")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or min(result.shape) < 1:
        raise ValueError(f"{name} must have shape (T, H, D) with nonzero axes")
    if np.isinf(result).any():
        raise ValueError(f"{name} may contain NaN for UNKNOWN, but not infinity")
    return result


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Frozen scan configuration.

    ``top_k`` and ``moment_weights`` are the only intended development-candidate
    knobs.  The rolling scales, power calibrator, HMM modes, and lag are checked
    against the CRIS-v9 contract rather than silently accepted as alternatives.
    A zero moment weight is allowed for a pre-registered ablation, but at least
    one moment must have positive weight.
    """

    top_k: int = 4
    moment_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rolling_scales: tuple[int, int, int] = ROLLING_SCALES
    eta: float = DEFAULT_ETA
    hmm_modes: tuple[tuple[float, float], tuple[float, float]] = HMM_MODES
    dependence_lag: int = DEFAULT_DEPENDENCE_LAG
    reset_gap: int = DEFAULT_RESET_GAP

    def __post_init__(self) -> None:
        top_k = _integer(self.top_k, name="top_k", minimum=1)
        if type(self.moment_weights) is not tuple or len(self.moment_weights) != 3:
            raise ValueError("moment_weights must be a length-3 tuple")
        weights = tuple(
            _finite_real(value, name=f"moment_weights[{index}]")
            for index, value in enumerate(self.moment_weights)
        )
        if any(value < 0.0 for value in weights) or not any(
            value > 0.0 for value in weights
        ):
            raise ValueError("moment_weights must be nonnegative with positive sum")
        if tuple(self.rolling_scales) != ROLLING_SCALES:
            raise ValueError("rolling_scales are frozen at (4, 16, 64)")
        eta = _finite_real(self.eta, name="eta")
        if eta != DEFAULT_ETA:
            raise ValueError("eta is frozen at 0.5")
        if tuple(self.hmm_modes) != HMM_MODES:
            raise ValueError("hmm_modes are frozen at ((.02,.2),(.1,.5))")
        lag = _integer(self.dependence_lag, name="dependence_lag", minimum=1)
        if lag != DEFAULT_DEPENDENCE_LAG:
            raise ValueError("dependence_lag is frozen at 1")
        reset_gap = _integer(self.reset_gap, name="reset_gap", minimum=1)
        if reset_gap != DEFAULT_RESET_GAP:
            raise ValueError(f"reset_gap is frozen at {DEFAULT_RESET_GAP}")
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "moment_weights", weights)
        object.__setattr__(self, "rolling_scales", ROLLING_SCALES)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "hmm_modes", HMM_MODES)
        object.__setattr__(self, "dependence_lag", lag)
        object.__setattr__(self, "reset_gap", reset_gap)


@dataclass(frozen=True, slots=True, order=True)
class ComponentKey:
    """Identity of one calibrated point-statistic tail."""

    moment: str
    horizon: int
    scale: int
    side: str

    def __post_init__(self) -> None:
        if self.moment not in MOMENTS:
            raise ValueError(f"moment must be one of {MOMENTS}")
        horizon = _integer(self.horizon, name="horizon", minimum=0)
        scale = _integer(self.scale, name="scale", minimum=1)
        if scale not in ROLLING_SCALES:
            raise ValueError(f"scale must be one of {ROLLING_SCALES}")
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "scale", scale)

    @property
    def name(self) -> str:
        return f"{self.moment}/h{self.horizon}/w{self.scale}/{self.side}"


@dataclass(frozen=True, slots=True)
class PointStatistics:
    """Aligned causal point statistics; NaN entries are UNKNOWN."""

    values: np.ndarray
    components: tuple[ComponentKey, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.dtype != np.dtype(np.float64) or values.ndim != 2:
            raise ValueError("values must be a float64 array with shape (T, C)")
        if values.shape[0] < 1 or values.shape[1] != len(self.components):
            raise ValueError("values and components are not aligned")
        if type(self.components) is not tuple or not self.components:
            raise ValueError("components must be a nonempty tuple")
        if any(not isinstance(item, ComponentKey) for item in self.components):
            raise ValueError("components must contain ComponentKey values")
        if len(set(self.components)) != len(self.components):
            raise ValueError("component identities must be unique")
        if np.isinf(values).any():
            raise ValueError("point statistics may be NaN, but not infinite")
        object.__setattr__(self, "values", _readonly(values, dtype=np.float64))

    @property
    def valid(self) -> np.ndarray:
        return _readonly(np.isfinite(self.values), dtype=np.bool_)


def _raw_moments(innovations: np.ndarray, lag: int) -> dict[str, np.ndarray]:
    location = np.array(innovations, dtype=np.float64, order="C", copy=True)
    scale = (np.square(location) - 1.0) / math.sqrt(2.0)
    scale[np.isnan(location)] = np.nan
    dependence = np.full_like(location, np.nan)
    dependence[lag:] = location[lag:] * location[:-lag]
    return {
        "location": location,
        "scale": scale,
        "dependence": dependence,
    }


def _causal_standardized_roll(raw: np.ndarray, window: int) -> np.ndarray:
    """Return trailing sums divided by sqrt(available count), causally.

    A coordinate is reported only when its *current* raw moment is available.
    Thus an all-NaN timestamp becomes UNKNOWN immediately rather than leaking a
    stale moving average.  Earlier finite samples in the trailing window still
    provide history after a short intermittent gap.
    """

    finite = np.isfinite(raw)
    filled = np.where(finite, raw, 0.0)
    cumulative_sum = np.cumsum(filled, axis=0, dtype=np.float64)
    cumulative_count = np.cumsum(finite.astype(np.int64), axis=0, dtype=np.int64)
    result = np.empty_like(raw, dtype=np.float64)
    for time in range(raw.shape[0]):
        start = time + 1 - window
        sums = cumulative_sum[time].copy()
        counts = cumulative_count[time].copy()
        if start > 0:
            sums -= cumulative_sum[start - 1]
            counts -= cumulative_count[start - 1]
        current = finite[time]
        valid = current & (counts > 0)
        row = np.full(raw.shape[1:], np.nan, dtype=np.float64)
        row[valid] = sums[valid] / np.sqrt(counts[valid].astype(np.float64))
        result[time] = row
    return result


def _topk_two_sided(values: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Select once by absolute magnitude, then retain both directional tails."""

    time_count, horizon_count, dimension_count = values.shape
    if top_k > dimension_count:
        raise ValueError("top_k cannot exceed the innovation dimension count")
    finite = np.isfinite(values)
    enough = np.sum(finite, axis=2) >= top_k
    magnitudes = np.where(finite, np.abs(values), -math.inf)
    # Stable descending order supplies a deterministic dimension-index tie
    # policy.  The same selected coordinates feed both one-sided tails.
    selected_indices = np.argsort(-magnitudes, axis=2, kind="stable")[:, :, :top_k]
    selected = np.take_along_axis(values, selected_indices, axis=2)
    positive = np.mean(np.maximum(selected, 0.0), axis=2)
    negative = np.mean(np.maximum(-selected, 0.0), axis=2)
    positive[~enough] = np.nan
    negative[~enough] = np.nan
    return positive, negative


def build_point_statistics(
    innovations: object, config: MonitorConfig | None = None
) -> PointStatistics:
    """Construct every causal CRIS-v9 point statistic from one complete run."""

    cfg = MonitorConfig() if config is None else config
    if not isinstance(cfg, MonitorConfig):
        raise TypeError("config must be MonitorConfig")
    cube = _validate_innovations(innovations)
    if cfg.top_k > cube.shape[2]:
        raise ValueError("top_k cannot exceed innovations.shape[2]")
    raw_by_moment = _raw_moments(cube, cfg.dependence_lag)
    columns: list[np.ndarray] = []
    components: list[ComponentKey] = []
    # This loop order is part of the stable component identity contract.
    for moment in MOMENTS:
        aggregated = {
            scale: _topk_two_sided(
                _causal_standardized_roll(raw_by_moment[moment], scale), cfg.top_k
            )
            for scale in ROLLING_SCALES
        }
        for horizon in range(cube.shape[1]):
            for scale in ROLLING_SCALES:
                upper, lower = aggregated[scale]
                columns.extend((upper[:, horizon], lower[:, horizon]))
                components.extend(
                    (
                        ComponentKey(moment, horizon, scale, "positive"),
                        ComponentKey(moment, horizon, scale, "negative"),
                    )
                )
    values = np.column_stack(columns).astype(np.float64, copy=False)
    return PointStatistics(values=values, components=tuple(components))


@dataclass(frozen=True, slots=True)
class PointCalibrator:
    """Sorted healthy point-statistic references for empirical tail ranks.

    Serially dependent time points are pooled inside each component.  These
    ranks are monitoring features, not conformal p-values; finite-sample
    false-alarm control comes only from the outer independent-run maxima.
    """

    components: tuple[ComponentKey, ...]
    sorted_samples: tuple[np.ndarray, ...]
    horizon_count: int
    dimension_count: int
    config: MonitorConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, MonitorConfig):
            raise ValueError("config must be MonitorConfig")
        horizons = _integer(self.horizon_count, name="horizon_count", minimum=1)
        dimensions = _integer(self.dimension_count, name="dimension_count", minimum=1)
        if self.config.top_k > dimensions:
            raise ValueError("config.top_k cannot exceed dimension_count")
        if type(self.components) is not tuple or not self.components:
            raise ValueError("components must be a nonempty tuple")
        if type(self.sorted_samples) is not tuple or len(self.sorted_samples) != len(
            self.components
        ):
            raise ValueError("sorted_samples must align with components")
        expected = _component_keys(horizons)
        if self.components != expected:
            raise ValueError("components do not match the frozen CRIS-v9 ordering")
        owned_samples: list[np.ndarray] = []
        for index, sample in enumerate(self.sorted_samples):
            values = np.asarray(sample)
            if (
                values.dtype != np.dtype(np.float64)
                or values.ndim != 1
                or values.size < 1
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"sorted_samples[{index}] must be nonempty finite float64")
            if np.any(values[1:] < values[:-1]):
                raise ValueError(f"sorted_samples[{index}] must be nondecreasing")
            owned_samples.append(_readonly(values, dtype=np.float64))
        object.__setattr__(self, "sorted_samples", tuple(owned_samples))
        object.__setattr__(self, "horizon_count", horizons)
        object.__setattr__(self, "dimension_count", dimensions)

    @classmethod
    def fit(
        cls,
        healthy_innovations: Iterable[object] | np.ndarray,
        config: MonitorConfig | None = None,
    ) -> "PointCalibrator":
        """Fit point references from healthy runs without pooling run maxima."""

        cfg = MonitorConfig() if config is None else config
        if not isinstance(cfg, MonitorConfig):
            raise TypeError("config must be MonitorConfig")
        if isinstance(healthy_innovations, np.ndarray) and healthy_innovations.ndim == 4:
            raw_runs = tuple(healthy_innovations[index] for index in range(healthy_innovations.shape[0]))
        else:
            try:
                raw_runs = tuple(healthy_innovations)
            except TypeError as exc:
                raise ValueError("healthy_innovations must be an iterable of runs") from exc
        if not raw_runs:
            raise ValueError("healthy_innovations must contain at least one run")
        cubes = tuple(
            _validate_innovations(run, name=f"healthy_innovations[{index}]")
            for index, run in enumerate(raw_runs)
        )
        horizon_count, dimension_count = cubes[0].shape[1:]
        if cfg.top_k > dimension_count:
            raise ValueError("config.top_k cannot exceed the innovation dimension count")
        if any(cube.shape[1:] != (horizon_count, dimension_count) for cube in cubes):
            raise ValueError("all healthy runs must share horizon and dimension counts")
        statistics = tuple(build_point_statistics(cube, cfg) for cube in cubes)
        components = statistics[0].components
        references: list[np.ndarray] = []
        for component in range(len(components)):
            finite_parts = tuple(
                item.values[np.isfinite(item.values[:, component]), component]
                for item in statistics
            )
            joined = np.concatenate(finite_parts) if finite_parts else np.empty(0)
            if joined.size == 0:
                raise ValueError(
                    f"healthy calibration has no available values for {components[component].name}"
                )
            references.append(np.sort(joined.astype(np.float64), kind="stable"))
        return cls(
            components=components,
            sorted_samples=tuple(references),
            horizon_count=horizon_count,
            dimension_count=dimension_count,
            config=cfg,
        )

    @classmethod
    def from_sorted_samples(
        cls,
        sorted_samples: Sequence[np.ndarray],
        *,
        horizon_count: int,
        dimension_count: int,
        config: MonitorConfig | None = None,
    ) -> "PointCalibrator":
        """Construct an auditable calibrator from already sorted references."""

        cfg = MonitorConfig() if config is None else config
        horizons = _integer(horizon_count, name="horizon_count", minimum=1)
        samples = tuple(
            np.asarray(value, dtype=np.float64) for value in tuple(sorted_samples)
        )
        return cls(
            components=_component_keys(horizons),
            sorted_samples=samples,
            horizon_count=horizons,
            dimension_count=dimension_count,
            config=cfg,
        )

    def point_statistics(self, innovations: object) -> PointStatistics:
        cube = _validate_innovations(innovations)
        if cube.shape[1:] != (self.horizon_count, self.dimension_count):
            raise ValueError("innovation horizon/dimension shape differs from calibration")
        return build_point_statistics(cube, self.config)

    def p_values(self, value: PointStatistics | object) -> np.ndarray:
        """Return tie-conservative empirical upper-tail ranks."""

        statistics = value if isinstance(value, PointStatistics) else self.point_statistics(value)
        if statistics.components != self.components:
            raise ValueError("point-statistic component identities differ from calibration")
        p_values = np.full(statistics.values.shape, np.nan, dtype=np.float64)
        for component, reference in enumerate(self.sorted_samples):
            observed = statistics.values[:, component]
            valid = np.isfinite(observed)
            if not np.any(valid):
                continue
            # side='left' makes n-left equal the number of samples >= value.
            left = np.searchsorted(reference, observed[valid], side="left")
            greater_equal = reference.size - left
            p_values[valid, component] = (1.0 + greater_equal) / float(
                reference.size + 1
            )
        return _readonly(p_values, dtype=np.float64)

    def e_values(self, value: PointStatistics | object) -> np.ndarray:
        """Return the fixed power transform; no e-value validity is claimed."""
        p_values = self.p_values(value)
        result = np.full(p_values.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(p_values)
        result[valid] = (1.0 - self.config.eta) * np.power(
            p_values[valid], -self.config.eta
        )
        return _readonly(result, dtype=np.float64)


def _component_keys(horizon_count: int) -> tuple[ComponentKey, ...]:
    return tuple(
        ComponentKey(moment, horizon, scale, side)
        for moment in MOMENTS
        for horizon in range(horizon_count)
        for scale in ROLLING_SCALES
        for side in SIDES
    )


@dataclass(frozen=True, slots=True)
class HMMTrace:
    """Log-domain forward trace for one e-value stream and one HMM mode."""

    score: np.ndarray
    held_score: np.ndarray
    log_state0: np.ndarray
    log_state1: np.ndarray
    valid: np.ndarray
    reset: np.ndarray
    rho: float
    recovery: float

    def __post_init__(self) -> None:
        length = np.asarray(self.score).size
        for name, value, dtype in (
            ("score", self.score, np.float64),
            ("held_score", self.held_score, np.float64),
            ("log_state0", self.log_state0, np.float64),
            ("log_state1", self.log_state1, np.float64),
            ("valid", self.valid, np.bool_),
            ("reset", self.reset, np.bool_),
        ):
            array = np.asarray(value)
            if array.dtype != np.dtype(dtype) or array.shape != (length,):
                raise ValueError(f"{name} must be an aligned one-dimensional array")
            object.__setattr__(self, name, _readonly(array, dtype=dtype))


def hmm_log_forward(
    log_e_values: object,
    *,
    rho: float,
    recovery: float,
    valid: object | None = None,
    reset_gap: int = DEFAULT_RESET_GAP,
) -> HMMTrace:
    """Run the two-state HMM likelihood-ratio recursion in log space.

    State 0 has emission likelihood ratio one and state 1 has likelihood ratio
    ``e_t``.  ``rho`` is the 0->1 onset probability and ``recovery`` is the
    1->0 probability.  Before the first valid point (and after a long-gap
    reset), the known state is 0.  UNKNOWN entries cause neither a transition
    nor an emission update.
    """

    raw = np.asarray(log_e_values)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise ValueError("log_e_values must be a real numeric vector")
    log_e = np.asarray(log_e_values, dtype=np.float64)
    if log_e.ndim != 1 or log_e.size < 1 or np.isinf(log_e).any():
        raise ValueError("log_e_values must be a nonempty vector without infinity")
    if valid is None:
        valid_mask = np.isfinite(log_e)
    else:
        valid_mask = np.asarray(valid)
        if valid_mask.dtype != np.dtype(np.bool_) or valid_mask.shape != log_e.shape:
            raise ValueError("valid must be a boolean vector aligned to log_e_values")
        if np.any(valid_mask & ~np.isfinite(log_e)):
            raise ValueError("valid log_e_values must be finite")
    onset = _finite_real(rho, name="rho")
    exit_probability = _finite_real(recovery, name="recovery")
    if not 0.0 < onset < 1.0 or not 0.0 < exit_probability < 1.0:
        raise ValueError("rho and recovery must lie strictly in (0,1)")
    gap_limit = _integer(reset_gap, name="reset_gap", minimum=1)

    score = np.full(log_e.shape, np.nan, dtype=np.float64)
    held = np.empty_like(log_e)
    state0_path = np.empty_like(log_e)
    state1_path = np.empty_like(log_e)
    reset = np.zeros(log_e.shape, dtype=np.bool_)
    log_state0 = 0.0
    log_state1 = -math.inf
    gap = 0
    log_p00 = math.log1p(-onset)
    log_p01 = math.log(onset)
    log_p10 = math.log(exit_probability)
    log_p11 = math.log1p(-exit_probability)
    for time in range(log_e.size):
        if valid_mask[time]:
            gap = 0
            predicted0 = float(
                np.logaddexp(log_state0 + log_p00, log_state1 + log_p10)
            )
            predicted1 = float(
                np.logaddexp(log_state0 + log_p01, log_state1 + log_p11)
            )
            log_state0 = predicted0
            log_state1 = predicted1 + float(log_e[time])
            current = float(np.logaddexp(log_state0, log_state1))
            if not math.isfinite(current):
                raise FloatingPointError("HMM recursion produced a non-finite score")
            score[time] = current
        else:
            gap += 1
            if gap == gap_limit:
                log_state0 = 0.0
                log_state1 = -math.inf
                reset[time] = True
        held[time] = float(np.logaddexp(log_state0, log_state1))
        state0_path[time] = log_state0
        state1_path[time] = log_state1
    return HMMTrace(
        score=score,
        held_score=held,
        log_state0=state0_path,
        log_state1=state1_path,
        valid=np.asarray(valid_mask, dtype=np.bool_),
        reset=reset,
        rho=onset,
        recovery=exit_probability,
    )


def _weighted_logmeanexp(
    values: np.ndarray, log_weights: np.ndarray
) -> np.ndarray:
    shifted = values + log_weights[None, :]
    pivot = np.max(shifted, axis=1)
    return pivot + np.log(
        np.sum(np.exp(shifted - pivot[:, None]), axis=1, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class ScanResult:
    """CRIS path and its auditable point/component decomposition."""

    score: np.ndarray
    valid: np.ndarray
    component_scores: np.ndarray
    component_valid: np.ndarray
    p_values: np.ndarray
    e_values: np.ndarray
    components: tuple[ComponentKey, ...]
    modes: tuple[tuple[float, float], ...]
    alarm: np.ndarray | None = None

    def __post_init__(self) -> None:
        score = np.asarray(self.score)
        valid = np.asarray(self.valid)
        component_scores = np.asarray(self.component_scores)
        component_valid = np.asarray(self.component_valid)
        p_values = np.asarray(self.p_values)
        e_values = np.asarray(self.e_values)
        time_count = score.size
        component_count = len(self.components)
        mode_count = len(self.modes)
        if score.dtype != np.dtype(np.float64) or score.shape != (time_count,):
            raise ValueError("score must be a one-dimensional float64 array")
        if valid.dtype != np.dtype(np.bool_) or valid.shape != score.shape:
            raise ValueError("valid must be an aligned boolean array")
        if np.any(valid & ~np.isfinite(score)) or np.any(~valid & ~np.isnan(score)):
            raise ValueError("score must be finite exactly where valid is true")
        if component_scores.dtype != np.dtype(np.float64) or component_scores.shape != (
            time_count,
            component_count,
            mode_count,
        ):
            raise ValueError("component_scores must have shape (T, C, M)")
        if not np.isfinite(component_scores).all():
            raise ValueError("held component_scores must be finite")
        if component_valid.dtype != np.dtype(np.bool_) or component_valid.shape != (
            time_count,
            component_count,
        ):
            raise ValueError("component_valid must have shape (T, C)")
        for name, value in (("p_values", p_values), ("e_values", e_values)):
            if value.dtype != np.dtype(np.float64) or value.shape != component_valid.shape:
                raise ValueError(f"{name} must have shape (T, C)")
            if np.isinf(value).any():
                raise ValueError(f"{name} may contain NaN, but not infinity")
        if self.alarm is not None:
            alarm = np.asarray(self.alarm)
            if alarm.dtype != np.dtype(np.bool_) or alarm.shape != score.shape:
                raise ValueError("alarm must be None or an aligned boolean array")
            if np.any(alarm & ~valid):
                raise ValueError("UNKNOWN points cannot alarm")
            object.__setattr__(self, "alarm", _readonly(alarm, dtype=np.bool_))
        for name, value, dtype in (
            ("score", score, np.float64),
            ("valid", valid, np.bool_),
            ("component_scores", component_scores, np.float64),
            ("component_valid", component_valid, np.bool_),
            ("p_values", p_values, np.float64),
            ("e_values", e_values, np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype=dtype))

    @property
    def score_path(self) -> np.ndarray:
        return self.score

    @property
    def first_alarm_index(self) -> int | None:
        if self.alarm is None or not np.any(self.alarm):
            return None
        return int(np.flatnonzero(self.alarm)[0])


def scan_innovations(
    innovations: object,
    calibrator: PointCalibrator,
    config: MonitorConfig | None = None,
    *,
    threshold: "TrajectoryThreshold | float | None" = None,
) -> ScanResult:
    """Score one innovation run, optionally applying a strict trajectory threshold."""

    if not isinstance(calibrator, PointCalibrator):
        raise TypeError("calibrator must be PointCalibrator")
    cfg = calibrator.config if config is None else config
    if not isinstance(cfg, MonitorConfig):
        raise TypeError("config must be MonitorConfig")
    if cfg != calibrator.config:
        raise ValueError("scan config differs from point-calibration config")
    statistics = calibrator.point_statistics(innovations)
    p_values = calibrator.p_values(statistics)
    e_values = np.full(p_values.shape, np.nan, dtype=np.float64)
    component_valid = np.isfinite(p_values)
    e_values[component_valid] = (1.0 - cfg.eta) * np.power(
        p_values[component_valid], -cfg.eta
    )
    log_e = np.full(e_values.shape, np.nan, dtype=np.float64)
    log_e[component_valid] = np.log(e_values[component_valid])

    time_count, component_count = p_values.shape
    mode_count = len(cfg.hmm_modes)
    held_scores = np.empty((time_count, component_count, mode_count), dtype=np.float64)
    for component in range(component_count):
        for mode, (rho, recovery) in enumerate(cfg.hmm_modes):
            trace = hmm_log_forward(
                log_e[:, component],
                rho=rho,
                recovery=recovery,
                valid=component_valid[:, component],
                reset_gap=cfg.reset_gap,
            )
            held_scores[:, component, mode] = trace.held_score

    moment_weights = dict(zip(MOMENTS, cfg.moment_weights, strict=True))
    combination_weights = np.asarray(
        [
            moment_weights[component.moment]
            for component in calibrator.components
            for _mode in cfg.hmm_modes
        ],
        dtype=np.float64,
    )
    positive = combination_weights > 0.0
    normalized = combination_weights[positive] / np.sum(combination_weights[positive])
    log_weights = np.log(normalized)
    flattened = held_scores.reshape(time_count, -1)
    combined = _weighted_logmeanexp(flattened[:, positive], log_weights)
    component_enabled = np.asarray(
        [moment_weights[item.moment] > 0.0 for item in calibrator.components],
        dtype=np.bool_,
    )
    valid = np.any(component_valid[:, component_enabled], axis=1)
    score = np.where(valid, combined, np.nan).astype(np.float64)
    alarm = None if threshold is None else apply_threshold(score, threshold)
    return ScanResult(
        score=score,
        valid=valid.astype(np.bool_),
        component_scores=held_scores,
        component_valid=component_valid.astype(np.bool_),
        p_values=p_values,
        e_values=e_values,
        components=calibrator.components,
        modes=cfg.hmm_modes,
        alarm=alarm,
    )


@dataclass(frozen=True, slots=True)
class TrajectoryThreshold:
    """Reachable split-conformal threshold based on independent run maxima."""

    value: float
    alpha: float
    calibration_run_count: int
    ascending_rank: int
    calibration_maxima: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        value = _finite_real(self.value, name="value")
        alpha = _finite_real(self.alpha, name="alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        count = _integer(
            self.calibration_run_count, name="calibration_run_count", minimum=1
        )
        rank = _integer(self.ascending_rank, name="ascending_rank", minimum=1)
        maxima = np.asarray(self.calibration_maxima)
        if (
            maxima.dtype != np.dtype(np.float64)
            or maxima.shape != (count,)
            or not np.isfinite(maxima).all()
        ):
            raise ValueError("calibration_maxima must be an aligned finite float64 vector")
        expected_rank = int(math.ceil((count + 1) * (1.0 - alpha)))
        if expected_rank > count:
            raise UnreachableThresholdError(
                f"requested conformal rank {expected_rank} exceeds n={count}"
            )
        if rank != expected_rank:
            raise ValueError("ascending_rank does not match ceil((n+1)(1-alpha))")
        ordered = np.sort(maxima, kind="stable")
        if value != float(ordered[rank - 1]):
            raise ValueError("threshold value does not match its conformal order statistic")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "calibration_run_count", count)
        object.__setattr__(self, "ascending_rank", rank)
        object.__setattr__(self, "calibration_maxima", _readonly(maxima, dtype=np.float64))

    @property
    def threshold(self) -> float:
        return self.value

    @property
    def comparison(self) -> str:
        return "strictly_greater_than"


def _score_paths(value: Iterable[object] | np.ndarray) -> tuple[np.ndarray, ...]:
    if isinstance(value, np.ndarray):
        raw = np.asarray(value)
        if raw.ndim == 1:
            candidates = (raw,)
        elif raw.ndim == 2:
            candidates = tuple(raw[index] for index in range(raw.shape[0]))
        else:
            raise ValueError("score paths must be one- or two-dimensional")
    else:
        try:
            candidates = tuple(value)
        except TypeError as exc:
            raise ValueError("score_paths must be an iterable of paths") from exc
    if not candidates:
        raise ValueError("score_paths must contain at least one run")
    paths: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        raw = np.asarray(candidate)
        if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
            raise ValueError(f"score_paths[{index}] must be real numeric")
        path = np.asarray(candidate, dtype=np.float64)
        if path.ndim != 1 or path.size < 1 or np.isinf(path).any():
            raise ValueError(f"score_paths[{index}] must be a nonempty vector without infinity")
        if not np.isfinite(path).any():
            raise ValueError(f"score_paths[{index}] has no valid score")
        paths.append(path)
    return tuple(paths)


def calibrate_trajectory_threshold(
    healthy_score_paths: Iterable[object] | np.ndarray, *, alpha: float
) -> TrajectoryThreshold:
    """Calibrate from one maximum per independent healthy run."""

    paths = _score_paths(healthy_score_paths)
    level = _finite_real(alpha, name="alpha")
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    maxima = np.asarray(
        [float(np.max(path[np.isfinite(path)])) for path in paths], dtype=np.float64
    )
    count = maxima.size
    rank = int(math.ceil((count + 1) * (1.0 - level)))
    if rank > count:
        raise UnreachableThresholdError(
            "trajectory threshold is unreachable: "
            f"ceil((n+1)(1-alpha))={rank} exceeds n={count}; "
            "use more independent healthy runs or a larger alpha"
        )
    ordered = np.sort(maxima, kind="stable")
    return TrajectoryThreshold(
        value=float(ordered[rank - 1]),
        alpha=level,
        calibration_run_count=count,
        ascending_rank=rank,
        calibration_maxima=maxima,
    )


def apply_threshold(
    score_path: object, threshold: TrajectoryThreshold | float
) -> np.ndarray:
    """Return point alarms using strict ``>``; NaN/UNKNOWN never alarms."""

    raw = np.asarray(score_path)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise ValueError("score_path must be a real numeric array")
    scores = np.asarray(score_path, dtype=np.float64)
    if scores.ndim not in (1, 2) or scores.size < 1 or np.isinf(scores).any():
        raise ValueError("score_path must be a nonempty 1D/2D array without infinity")
    cutoff = threshold.value if isinstance(threshold, TrajectoryThreshold) else _finite_real(
        threshold, name="threshold"
    )
    alarms = np.isfinite(scores) & (scores > cutoff)
    return _readonly(alarms.astype(np.bool_), dtype=np.bool_)


__all__ = [
    "ComponentKey",
    "DEFAULT_DEPENDENCE_LAG",
    "DEFAULT_ETA",
    "DEFAULT_RESET_GAP",
    "HMM_MODES",
    "HMMTrace",
    "MOMENTS",
    "MonitorConfig",
    "PointCalibrator",
    "PointStatistics",
    "ROLLING_SCALES",
    "SIDES",
    "ScanResult",
    "TrajectoryThreshold",
    "UnreachableThresholdError",
    "apply_threshold",
    "build_point_statistics",
    "calibrate_trajectory_threshold",
    "hmm_log_forward",
    "scan_innovations",
]
