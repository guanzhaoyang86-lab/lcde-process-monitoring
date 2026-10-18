"""Run-level evaluation for CRIS-v9.

Every observation passed to this module represents one complete, independent
run.  Fault detection is event based: only an alarm in the half-open interval
``[onset, event_stop)`` receives detection credit.  A pre-onset alarm is
reported separately and never substitutes for an event-matched alarm.  Misses
receive the full event-window length as their delay before median and
nearest-rank p90 summaries are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Hashable, Iterable, Sequence

import numpy as np


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


def _validate_identity(value: object, *, name: str) -> Hashable:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} cannot be boolean")
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be hashable") from exc
    if isinstance(value, str) and not value:
        raise ValueError(f"{name} cannot be empty")
    return value  # type: ignore[return-value]


def nearest_rank(values: Sequence[float], probability: float) -> float:
    """Return the non-interpolated nearest-rank sample quantile."""

    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise ValueError("values must be an iterable") from exc
    if not supplied:
        raise ValueError("values must be nonempty")
    numeric = np.asarray(
        [
            _finite_real(value, name=f"values[{index}]")
            for index, value in enumerate(supplied)
        ],
        dtype=np.float64,
    )
    probability_value = _finite_real(probability, name="probability")
    if not 0.0 < probability_value <= 1.0:
        raise ValueError("probability must lie in (0,1]")
    rank = int(math.ceil(probability_value * numeric.size))
    return float(np.sort(numeric, kind="stable")[rank - 1])


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Stable P[X <= successes] for X~Binomial(trials, probability)."""

    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = tuple(
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_p
        + (trials - index) * log_q
        for index in range(successes + 1)
    )
    pivot = max(terms)
    return math.exp(pivot) * math.fsum(math.exp(term - pivot) for term in terms)


def _solve_binomial_cdf(successes: int, trials: int, target: float) -> float:
    """Solve the decreasing binomial CDF in p by deterministic bisection."""

    lower = 0.0
    upper = 1.0
    for _ in range(160):
        midpoint = 0.5 * (lower + upper)
        if midpoint == lower or midpoint == upper:
            break
        if _binomial_cdf(successes, trials, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return float(0.5 * (lower + upper))


def clopper_pearson_upper(
    successes: int, trials: int, *, confidence: float = 0.95
) -> float:
    """Exact one-sided Clopper--Pearson upper confidence bound."""

    observed = _integer(successes, name="successes", minimum=0)
    count = _integer(trials, name="trials", minimum=1)
    if observed > count:
        raise ValueError("successes cannot exceed trials")
    level = _finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if observed == count:
        return 1.0
    return _solve_binomial_cdf(observed, count, 1.0 - level)


def clopper_pearson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact equal-tailed two-sided Clopper--Pearson confidence interval."""

    observed = _integer(successes, name="successes", minimum=0)
    count = _integer(trials, name="trials", minimum=1)
    if observed > count:
        raise ValueError("successes cannot exceed trials")
    level = _finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    tail = 0.5 * (1.0 - level)
    lower = (
        0.0
        if observed == 0
        else _solve_binomial_cdf(observed - 1, count, 1.0 - tail)
    )
    upper = (
        1.0
        if observed == count
        else _solve_binomial_cdf(observed, count, tail)
    )
    return (lower, upper)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Thresholded alarms for one independent complete run.

    Healthy runs use ``onset=None`` and ``event_stop=None``.  Fault runs use a
    left-closed/right-open event window.  ``alarm`` is a full-length boolean
    path; UNKNOWN score locations should already be represented as ``False``.
    """

    run_id: Hashable
    condition: Hashable
    alarm: np.ndarray
    onset: int | None = None
    event_stop: int | None = None

    def __post_init__(self) -> None:
        run_id = _validate_identity(self.run_id, name="run_id")
        condition = _validate_identity(self.condition, name="condition")
        alarm = np.asarray(self.alarm)
        if alarm.dtype != np.dtype(np.bool_) or alarm.ndim != 1 or alarm.size < 1:
            raise ValueError("alarm must be a nonempty one-dimensional boolean array")
        if (self.onset is None) != (self.event_stop is None):
            raise ValueError("onset and event_stop must either both be set or both be None")
        onset: int | None = None
        event_stop: int | None = None
        if self.onset is not None and self.event_stop is not None:
            onset = _integer(self.onset, name="onset", minimum=0)
            event_stop = _integer(self.event_stop, name="event_stop", minimum=1)
            if not onset < event_stop <= alarm.size:
                raise ValueError("fault window must satisfy 0 <= onset < event_stop <= run length")
        owned = np.array(alarm, dtype=np.bool_, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "alarm", owned)
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "event_stop", event_stop)

    @classmethod
    def from_score_path(
        cls,
        *,
        run_id: Hashable,
        condition: Hashable,
        score_path: object,
        threshold: float | object,
        onset: int | None = None,
        event_stop: int | None = None,
    ) -> "RunOutcome":
        """Build an outcome using finite ``score > threshold`` decisions."""

        raw = np.asarray(score_path)
        if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
            raise ValueError("score_path must be a real numeric vector")
        scores = np.asarray(score_path, dtype=np.float64)
        if scores.ndim != 1 or scores.size < 1 or np.isinf(scores).any():
            raise ValueError("score_path must be a nonempty vector without infinity")
        cutoff_source = getattr(threshold, "value", threshold)
        cutoff = _finite_real(cutoff_source, name="threshold")
        alarm = np.isfinite(scores) & (scores > cutoff)
        return cls(
            run_id=run_id,
            condition=condition,
            alarm=alarm.astype(np.bool_),
            onset=onset,
            event_stop=event_stop,
        )

    @property
    def is_fault(self) -> bool:
        return self.onset is not None

    @property
    def any_alarm(self) -> bool:
        return bool(np.any(self.alarm))

    @property
    def pre_onset_false_positive(self) -> bool:
        return bool(self.is_fault and np.any(self.alarm[: self.onset]))

    @property
    def detection_index(self) -> int | None:
        if not self.is_fault:
            return None
        matched = np.flatnonzero(self.alarm[self.onset : self.event_stop])
        if matched.size == 0:
            return None
        return int(self.onset + matched[0])

    @property
    def detected(self) -> bool:
        return self.detection_index is not None

    @property
    def miss_delay(self) -> int | None:
        if not self.is_fault:
            return None
        return int(self.event_stop - self.onset)

    @property
    def delay(self) -> int | None:
        if not self.is_fault:
            return None
        index = self.detection_index
        return self.miss_delay if index is None else int(index - self.onset)


def right_grace_mask(fault_mask: object, grace_samples: int) -> np.ndarray:
    """Expand active samples to a fixed right-sided matching grace window."""

    raw = np.asarray(fault_mask)
    if raw.dtype != np.dtype(np.bool_) or raw.ndim != 1 or raw.size < 1:
        raise ValueError("fault_mask must be a nonempty boolean vector")
    grace = _integer(grace_samples, name="grace_samples", minimum=0)
    eligible = np.array(raw, dtype=np.bool_, order="C", copy=True)
    active = np.flatnonzero(raw)
    for offset in range(1, grace + 1):
        shifted = active + offset
        shifted = shifted[shifted < raw.size]
        eligible[shifted] = True
    eligible.setflags(write=False)
    return eligible


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """Run-level episode matching based on *alarm onsets*, not alarm occupancy."""

    run_id: Hashable
    condition: Hashable
    alarm: np.ndarray
    fault_mask: np.ndarray
    grace_samples: int = 0

    def __post_init__(self) -> None:
        run_id = _validate_identity(self.run_id, name="run_id")
        condition = _validate_identity(self.condition, name="condition")
        alarm = np.asarray(self.alarm)
        mask = np.asarray(self.fault_mask)
        if alarm.dtype != np.dtype(np.bool_) or alarm.ndim != 1 or alarm.size < 1:
            raise ValueError("alarm must be a nonempty boolean vector")
        if mask.dtype != np.dtype(np.bool_) or mask.shape != alarm.shape:
            raise ValueError("fault_mask must be a boolean vector aligned with alarm")
        if not np.any(mask):
            raise ValueError("episode evaluation requires at least one affected observation")
        grace = _integer(self.grace_samples, name="grace_samples", minimum=0)
        owned_alarm = np.array(alarm, dtype=np.bool_, order="C", copy=True)
        owned_mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
        owned_alarm.setflags(write=False)
        owned_mask.setflags(write=False)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "alarm", owned_alarm)
        object.__setattr__(self, "fault_mask", owned_mask)
        object.__setattr__(self, "grace_samples", grace)

    @property
    def first_affected_index(self) -> int:
        return int(np.flatnonzero(self.fault_mask)[0])

    @property
    def alarm_onsets(self) -> np.ndarray:
        previous = np.concatenate((np.asarray([False]), self.alarm[:-1]))
        result = self.alarm & ~previous
        result.setflags(write=False)
        return result

    @property
    def eligible_mask(self) -> np.ndarray:
        return right_grace_mask(self.fault_mask, self.grace_samples)

    @property
    def detection_index(self) -> int | None:
        matched = np.flatnonzero(self.alarm_onsets & self.eligible_mask)
        return None if matched.size == 0 else int(matched[0])

    @property
    def detected(self) -> bool:
        return self.detection_index is not None

    @property
    def delay(self) -> int:
        index = self.detection_index
        if index is None:
            return int(self.alarm.size - self.first_affected_index)
        return int(index - self.first_affected_index)

    @property
    def pre_effect_false_alarm(self) -> bool:
        return bool(np.any(self.alarm_onsets[: self.first_affected_index]))

    @property
    def inactive_gap_alarm_onsets(self) -> int:
        after_effect = np.arange(self.alarm.size) >= self.first_affected_index
        invalid = self.alarm_onsets & after_effect & ~self.eligible_mask
        return int(np.count_nonzero(invalid))


@dataclass(frozen=True, slots=True)
class HealthyMetrics:
    runs: int
    alarm_runs: int
    alarm_rate: float
    one_sided_cp_upper: float

    def __post_init__(self) -> None:
        runs = _integer(self.runs, name="runs", minimum=1)
        alarms = _integer(self.alarm_runs, name="alarm_runs", minimum=0)
        if alarms > runs:
            raise ValueError("alarm_runs cannot exceed runs")
        expected_rate = alarms / runs
        if _finite_real(self.alarm_rate, name="alarm_rate") != expected_rate:
            raise ValueError("alarm_rate is inconsistent with counts")
        bound = _finite_real(self.one_sided_cp_upper, name="one_sided_cp_upper")
        if not expected_rate <= bound <= 1.0:
            raise ValueError("one_sided_cp_upper is outside [alarm_rate,1]")

    def to_document(self) -> dict[str, int | float]:
        return {
            "runs": self.runs,
            "alarm_runs": self.alarm_runs,
            "alarm_rate": self.alarm_rate,
            "one_sided_cp_upper": self.one_sided_cp_upper,
        }


@dataclass(frozen=True, slots=True)
class FaultMetrics:
    condition: Hashable | str
    runs: int
    detections: int
    misses: int
    recall: float
    delays: tuple[int, ...]
    median_delay: float
    nearest_rank_p90_delay: float
    pre_onset_false_positive_runs: int
    pre_onset_false_positive_rate: float

    def __post_init__(self) -> None:
        runs = _integer(self.runs, name="runs", minimum=1)
        detections = _integer(self.detections, name="detections", minimum=0)
        misses = _integer(self.misses, name="misses", minimum=0)
        if detections + misses != runs:
            raise ValueError("detections + misses must equal runs")
        if type(self.delays) is not tuple or len(self.delays) != runs:
            raise ValueError("delays must contain one value per fault run")
        parsed_delays = tuple(
            _integer(value, name=f"delays[{index}]", minimum=0)
            for index, value in enumerate(self.delays)
        )
        recall = _finite_real(self.recall, name="recall")
        if recall != detections / runs:
            raise ValueError("recall is inconsistent with counts")
        median = _finite_real(self.median_delay, name="median_delay")
        p90 = _finite_real(self.nearest_rank_p90_delay, name="nearest_rank_p90_delay")
        if median != float(np.median(np.asarray(parsed_delays, dtype=np.float64))):
            raise ValueError("median_delay is inconsistent with delays")
        if p90 != nearest_rank(parsed_delays, 0.90):
            raise ValueError("nearest_rank_p90_delay is inconsistent with delays")
        pre = _integer(
            self.pre_onset_false_positive_runs,
            name="pre_onset_false_positive_runs",
            minimum=0,
        )
        if pre > runs:
            raise ValueError("pre_onset_false_positive_runs cannot exceed runs")
        pre_rate = _finite_real(
            self.pre_onset_false_positive_rate,
            name="pre_onset_false_positive_rate",
        )
        if pre_rate != pre / runs:
            raise ValueError("pre_onset_false_positive_rate is inconsistent with counts")
        object.__setattr__(self, "delays", parsed_delays)

    def to_document(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "runs": self.runs,
            "detections": self.detections,
            "misses": self.misses,
            "recall": self.recall,
            "delays": list(self.delays),
            "median_delay": self.median_delay,
            "nearest_rank_p90_delay": self.nearest_rank_p90_delay,
            "pre_onset_false_positive_runs": self.pre_onset_false_positive_runs,
            "pre_onset_false_positive_rate": self.pre_onset_false_positive_rate,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    healthy: HealthyMetrics
    faults_overall: FaultMetrics
    faults_by_condition: tuple[FaultMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.healthy, HealthyMetrics):
            raise ValueError("healthy must be HealthyMetrics")
        if not isinstance(self.faults_overall, FaultMetrics):
            raise ValueError("faults_overall must be FaultMetrics")
        if type(self.faults_by_condition) is not tuple or not self.faults_by_condition:
            raise ValueError("faults_by_condition must be a nonempty tuple")
        if any(not isinstance(item, FaultMetrics) for item in self.faults_by_condition):
            raise ValueError("faults_by_condition must contain FaultMetrics")
        if sum(item.runs for item in self.faults_by_condition) != self.faults_overall.runs:
            raise ValueError("per-condition fault counts do not match overall counts")

    def fault(self, condition: Hashable) -> FaultMetrics:
        for item in self.faults_by_condition:
            if item.condition == condition:
                return item
        raise KeyError(condition)

    def to_document(self) -> dict[str, object]:
        return {
            "healthy": self.healthy.to_document(),
            "faults_overall": self.faults_overall.to_document(),
            "faults_by_condition": [
                item.to_document() for item in self.faults_by_condition
            ],
            "event_matching": "first alarm in [onset,event_stop); pre-onset alarms receive no detection credit",
            "miss_delay": "event_stop - onset",
            "p90_definition": "nearest rank: sorted[ceil(0.90*n)-1]",
        }


def _fault_metrics(condition: Hashable | str, rows: Sequence[RunOutcome]) -> FaultMetrics:
    if not rows or any(not row.is_fault for row in rows):
        raise ValueError("fault metric rows must be nonempty fault outcomes")
    delays = tuple(int(row.delay) for row in rows if row.delay is not None)
    detections = sum(row.detected for row in rows)
    pre_onset = sum(row.pre_onset_false_positive for row in rows)
    return FaultMetrics(
        condition=condition,
        runs=len(rows),
        detections=detections,
        misses=len(rows) - detections,
        recall=detections / len(rows),
        delays=delays,
        median_delay=float(np.median(np.asarray(delays, dtype=np.float64))),
        nearest_rank_p90_delay=nearest_rank(delays, 0.90),
        pre_onset_false_positive_runs=pre_onset,
        pre_onset_false_positive_rate=pre_onset / len(rows),
    )


def evaluate_runs(
    outcomes: Iterable[RunOutcome], *, confidence: float = 0.95
) -> EvaluationReport:
    """Evaluate complete runs as independent units, never as individual rows."""

    try:
        rows = tuple(outcomes)
    except TypeError as exc:
        raise ValueError("outcomes must be an iterable") from exc
    if not rows or any(not isinstance(row, RunOutcome) for row in rows):
        raise ValueError("outcomes must be a nonempty iterable of RunOutcome values")
    identities = tuple((row.condition, row.run_id) for row in rows)
    if len(set(identities)) != len(identities):
        raise ValueError("each condition/run identity must appear exactly once")
    healthy_rows = tuple(row for row in rows if not row.is_fault)
    fault_rows = tuple(row for row in rows if row.is_fault)
    if not healthy_rows or not fault_rows:
        raise ValueError("evaluation requires at least one healthy and one fault run")
    healthy_alarm_runs = sum(row.any_alarm for row in healthy_rows)
    healthy = HealthyMetrics(
        runs=len(healthy_rows),
        alarm_runs=healthy_alarm_runs,
        alarm_rate=healthy_alarm_runs / len(healthy_rows),
        one_sided_cp_upper=clopper_pearson_upper(
            healthy_alarm_runs, len(healthy_rows), confidence=confidence
        ),
    )

    conditions = sorted(
        {row.condition for row in fault_rows},
        key=lambda value: (type(value).__name__, repr(value)),
    )
    by_condition = tuple(
        _fault_metrics(
            condition,
            tuple(row for row in fault_rows if row.condition == condition),
        )
        for condition in conditions
    )
    overall = _fault_metrics("__all_faults__", fault_rows)
    return EvaluationReport(
        healthy=healthy,
        faults_overall=overall,
        faults_by_condition=by_condition,
    )


__all__ = [
    "EpisodeOutcome",
    "EvaluationReport",
    "FaultMetrics",
    "HealthyMetrics",
    "RunOutcome",
    "clopper_pearson_interval",
    "clopper_pearson_upper",
    "evaluate_runs",
    "nearest_rank",
    "right_grace_mask",
]
