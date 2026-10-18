"""Pure locked-calibration and confirmation logic for MetroPT CARED-BC.

This module deliberately performs no file I/O and imports no MetroPT data
loader.  Its inputs are already-computed score timestamps, CUSUM envelopes,
and alarm decisions.  Every public report is validated as strict JSON before
it is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MINIMUM_SCORES_PER_DAY = 1141
MINIMUM_CALIBRATION_DAYS = 39
MINIMUM_HEALTHY_CONFIRMATION_DAYS = 39
CALIBRATION_ALPHA = 0.05
BOUNDARY_EPSILON = 1.0e-9
AVAILABILITY_LAG_MINUTES = 1
RECOVERY_HOURS = 72
MAXIMUM_EVENT_DELAY_MINUTES = 60.0
MAXIMUM_MEDIAN_DELAY_MINUTES = 30.0
MAXIMUM_HEALTHY_DAY_ALARM_RATE = 0.05
MAXIMUM_HEALTHY_DAY_ALARM_UCB = 0.10
MAXIMUM_HEALTHY_TIME_ALARM_FRACTION = 0.01

CONFIRMATION_START = "2020-05-21 00:00:00"
CONFIRMATION_END = "2020-09-01 00:00:00"


class LockedEvaluationValidationError(ValueError):
    """Raised when locked-evaluation evidence is malformed or non-finite."""


@dataclass(frozen=True)
class FaultInterval:
    """One left-closed, right-open maintenance-report fault interval."""

    event_id: str
    start: str
    end: str


CONFIRMATION_EVENTS: tuple[FaultInterval, ...] = (
    FaultInterval("F2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    FaultInterval("F3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    FaultInterval("F4", "2020-07-15 14:30:00", "2020-07-15 19:00:00"),
)


def _strict_json(record: dict[str, Any], *, name: str) -> dict[str, Any]:
    try:
        json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise LockedEvaluationValidationError(f"{name} is not strict JSON") from exc
    return record


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise LockedEvaluationValidationError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise LockedEvaluationValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise LockedEvaluationValidationError(f"{name} must be at least {minimum}")
    return result


def _finite_scalar(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise LockedEvaluationValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LockedEvaluationValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise LockedEvaluationValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise LockedEvaluationValidationError(f"{name} must be at least {minimum}")
    return result


def _timestamp_scalar(value: Any, *, name: str) -> np.datetime64:
    try:
        array = np.asarray(value, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise LockedEvaluationValidationError(f"{name} must be a timestamp") from exc
    if array.shape != () or bool(np.isnat(array)):
        raise LockedEvaluationValidationError(f"{name} must be a valid scalar timestamp")
    return np.datetime64(array, "ns")


def _timestamp_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise LockedEvaluationValidationError(
            f"{name} must contain valid timestamps"
        ) from exc
    if result.ndim != 1 or np.any(np.isnat(result)):
        raise LockedEvaluationValidationError(
            f"{name} must be a one-dimensional vector without NaT"
        )
    if result.size > 1 and np.any(np.diff(result) <= np.timedelta64(0, "ns")):
        raise LockedEvaluationValidationError(f"{name} must be strictly increasing")
    return np.array(result, dtype="datetime64[ns]", copy=True)


def _finite_vector(
    value: Any,
    *,
    name: str,
    length: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LockedEvaluationValidationError(f"{name} must be numeric") from exc
    if result.ndim != 1 or (length is not None and result.shape != (length,)):
        expected = "one-dimensional" if length is None else f"shape ({length},)"
        raise LockedEvaluationValidationError(f"{name} must have {expected}")
    if not np.all(np.isfinite(result)):
        raise LockedEvaluationValidationError(f"{name} contains a non-finite value")
    if nonnegative and np.any(result < 0.0):
        raise LockedEvaluationValidationError(f"{name} must be non-negative")
    return np.array(result, dtype=np.float64, copy=True)


def _boolean_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.shape != (length,):
        raise LockedEvaluationValidationError(
            f"{name} must be a boolean vector with shape ({length},)"
        )
    return np.array(result, dtype=np.bool_, copy=True)


def _timestamp_text(value: np.datetime64) -> str:
    return str(np.datetime_as_string(value.astype("datetime64[ns]"), unit="ns"))


def _day_text(value: np.datetime64) -> str:
    return str(np.datetime_as_string(value.astype("datetime64[D]"), unit="D"))


def _complete_days(start: np.datetime64, end: np.datetime64) -> np.ndarray:
    if end <= start:
        raise LockedEvaluationValidationError("interval end must be after start")
    first = start.astype("datetime64[D]")
    if start != first.astype("datetime64[ns]"):
        first += np.timedelta64(1, "D")
    stop = end.astype("datetime64[D]")
    return np.arange(first, stop, dtype="datetime64[D]")


def enumerate_complete_days(
    timestamps: Any,
    envelope: Any,
    *,
    start: Any,
    end: Any,
    partition_name: str,
) -> dict[str, Any]:
    """Enumerate every full calendar day wholly inside ``[start, end)``.

    Missing days and days below 1,141 scores are retained explicitly.  Only
    timestamps within the supplied half-open interval are accepted.
    """

    if not isinstance(partition_name, str) or not partition_name:
        raise LockedEvaluationValidationError("partition_name must be non-empty")
    ts = _timestamp_vector(timestamps, name="timestamps")
    values = _finite_vector(
        envelope,
        name="envelope",
        length=int(ts.size),
        nonnegative=True,
    )
    left = _timestamp_scalar(start, name="start")
    right = _timestamp_scalar(end, name="end")
    days = _complete_days(left, right)
    if ts.size and (np.any(ts < left) or np.any(ts >= right)):
        raise LockedEvaluationValidationError(
            "timestamps must lie inside the supplied left-closed, right-open interval"
        )

    records: list[dict[str, Any]] = []
    for day in days:
        selected = ts.astype("datetime64[D]") == day
        count = int(np.count_nonzero(selected))
        maximum = float(np.max(values[selected])) if count else None
        qualified = count >= MINIMUM_SCORES_PER_DAY
        if qualified:
            reason = "qualified"
        elif count == 0:
            reason = "zero_scores"
        else:
            reason = "insufficient_scores"
        records.append(
            {
                "date": _day_text(day),
                "score_count": count,
                "maximum": maximum,
                "qualified": qualified,
                "qualification_reason": reason,
                "included_in_calibration": qualified,
            }
        )

    report: dict[str, Any] = {
        "schema": "metropt_cared_complete_day_enumeration",
        "schema_version": 1,
        "partition": partition_name,
        "interval": {
            "start_inclusive": _timestamp_text(left),
            "end_exclusive": _timestamp_text(right),
            "notation": "[start, end)",
        },
        "minimum_scores_per_qualified_day": MINIMUM_SCORES_PER_DAY,
        "candidate_day_count": int(days.size),
        "qualified_day_count": int(sum(record["qualified"] for record in records)),
        "days": records,
    }
    return _strict_json(report, name="complete-day report")


def _validated_day_records(
    report: Mapping[str, Any], *, name: str
) -> list[dict[str, Any]]:
    if not isinstance(report, Mapping):
        raise LockedEvaluationValidationError(f"{name} must be a day report")
    if (
        report.get("schema") != "metropt_cared_complete_day_enumeration"
        or report.get("schema_version") != 1
        or report.get("minimum_scores_per_qualified_day")
        != MINIMUM_SCORES_PER_DAY
    ):
        raise LockedEvaluationValidationError(f"{name} has an incompatible schema")
    days = report.get("days")
    if not isinstance(days, list):
        raise LockedEvaluationValidationError(f"{name}.days must be a list")
    if report.get("candidate_day_count") != len(days):
        raise LockedEvaluationValidationError(f"{name} candidate-day count changed")

    validated: list[dict[str, Any]] = []
    previous: str | None = None
    qualified_count = 0
    for index, raw in enumerate(days):
        if not isinstance(raw, Mapping):
            raise LockedEvaluationValidationError(f"{name}.days[{index}] is invalid")
        date = raw.get("date")
        if not isinstance(date, str):
            raise LockedEvaluationValidationError(f"{name}.days[{index}].date is invalid")
        parsed_date = _timestamp_scalar(date, name=f"{name}.days[{index}].date")
        if parsed_date != parsed_date.astype("datetime64[D]").astype("datetime64[ns]"):
            raise LockedEvaluationValidationError(f"{name}.days[{index}].date is not a day")
        if previous is not None and date <= previous:
            raise LockedEvaluationValidationError(f"{name} day dates must be unique and sorted")
        previous = date
        count = _integer(raw.get("score_count"), name=f"{name}.score_count")
        qualified = raw.get("qualified")
        if not isinstance(qualified, (bool, np.bool_)):
            raise LockedEvaluationValidationError(f"{name}.qualified must be boolean")
        expected_qualified = count >= MINIMUM_SCORES_PER_DAY
        if bool(qualified) is not expected_qualified:
            raise LockedEvaluationValidationError(f"{name} qualification changed")
        maximum_raw = raw.get("maximum")
        if count == 0:
            if maximum_raw is not None:
                raise LockedEvaluationValidationError(f"{name} zero-score maximum must be null")
            maximum: float | None = None
        else:
            maximum = _finite_scalar(
                maximum_raw,
                name=f"{name}.maximum",
                minimum=0.0,
            )
        qualified_count += int(expected_qualified)
        validated.append(
            {
                "date": date,
                "score_count": count,
                "maximum": maximum,
                "qualified": expected_qualified,
            }
        )
    if report.get("qualified_day_count") != qualified_count:
        raise LockedEvaluationValidationError(f"{name} qualified-day count changed")
    return validated


def calibrate_locked_threshold(
    calibration_a: Mapping[str, Any],
    calibration_b: Mapping[str, Any],
    *,
    portability_delta: Any,
) -> dict[str, Any]:
    """Merge qualified A/B days and compute the locked guarded boundary."""

    first = _validated_day_records(calibration_a, name="calibration_a")
    second = _validated_day_records(calibration_b, name="calibration_b")
    combined = first + second
    dates = [record["date"] for record in combined]
    if len(set(dates)) != len(dates):
        raise LockedEvaluationValidationError("calibration A/B day dates overlap")

    qualified = [record for record in combined if record["qualified"]]
    count = len(qualified)
    if count < MINIMUM_CALIBRATION_DAYS:
        raise LockedEvaluationValidationError(
            f"at least {MINIMUM_CALIBRATION_DAYS} qualified calibration days are required"
        )
    maxima = np.asarray([record["maximum"] for record in qualified], dtype=np.float64)
    if maxima.shape != (count,) or not np.all(np.isfinite(maxima)):
        raise LockedEvaluationValidationError("qualified calibration maxima are invalid")
    rank = int(math.ceil((count + 1) * (1.0 - CALIBRATION_ALPHA)))
    if rank < 1 or rank > count:
        raise LockedEvaluationValidationError("calibration rank is not finite-sample valid")
    raw = float(np.sort(maxima, kind="mergesort")[rank - 1])
    delta = _finite_scalar(
        portability_delta,
        name="portability_delta",
        minimum=0.0,
    )
    guard = delta + BOUNDARY_EPSILON
    # Evaluation order is part of the frozen cross-platform boundary formula.
    # Keep this as (raw + delta) + epsilon; raw + (delta + epsilon) can differ
    # by one ULP and would make the sealed runner reject a valid certificate.
    guarded = raw + delta + BOUNDARY_EPSILON
    effective = float(np.nextafter(guarded, np.inf))
    if not all(math.isfinite(value) for value in (guard, guarded, effective)):
        raise LockedEvaluationValidationError("guarded calibration boundary is non-finite")
    if not effective > guarded >= raw:
        raise LockedEvaluationValidationError("effective boundary is not strictly guarded")

    report: dict[str, Any] = {
        "schema": "metropt_cared_locked_calibration",
        "schema_version": 1,
        "alpha": CALIBRATION_ALPHA,
        "minimum_qualified_days": MINIMUM_CALIBRATION_DAYS,
        "candidate_day_count": len(combined),
        "qualified_day_count": count,
        "qualified_dates": [record["date"] for record in qualified],
        "qualified_daily_maxima": [float(value) for value in maxima.tolist()],
        "rank_one_based": rank,
        "rank_formula": "ceil((n+1)*(1-0.05))",
        "raw_boundary": raw,
        "portability_delta": delta,
        "boundary_epsilon": BOUNDARY_EPSILON,
        "numerical_guard": guard,
        "effective_boundary": effective,
        "effective_formula": (
            "nextafter(raw_boundary + portability_delta + 1e-9, +infinity)"
        ),
        "comparison": "strictly_greater_than",
        "exchangeable_day_alarm_probability_bound": float(
            (count + 1 - rank) / float(count + 1)
        ),
    }
    return _strict_json(report, name="calibration report")


def strict_alarm_mask(envelope: Any, effective_boundary: Any) -> np.ndarray:
    """Apply the only permitted alarm comparison: ``envelope > boundary``."""

    values = _finite_vector(
        envelope,
        name="envelope",
        nonnegative=True,
    )
    boundary = _finite_scalar(
        effective_boundary,
        name="effective_boundary",
        minimum=0.0,
    )
    return np.asarray(values > boundary, dtype=np.bool_)


def _validated_events(events: Iterable[FaultInterval | Mapping[str, Any]]) -> list[dict[str, Any]]:
    try:
        supplied = list(events)
    except TypeError as exc:
        raise LockedEvaluationValidationError("events must be iterable") from exc
    if not supplied:
        raise LockedEvaluationValidationError("at least one fault event is required")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, event in enumerate(supplied):
        if isinstance(event, FaultInterval):
            event_id, start, end = event.event_id, event.start, event.end
        elif isinstance(event, Mapping):
            event_id, start, end = event.get("event_id"), event.get("start"), event.get("end")
        else:
            raise LockedEvaluationValidationError(f"events[{index}] is invalid")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise LockedEvaluationValidationError("event IDs must be unique and non-empty")
        left = _timestamp_scalar(start, name=f"events[{index}].start")
        right = _timestamp_scalar(end, name=f"events[{index}].end")
        if right <= left:
            raise LockedEvaluationValidationError("event end must be after start")
        seen.add(event_id)
        records.append(
            {
                "event_id": event_id,
                "start": left,
                "end": right,
            }
        )
    return records


def _alarm_onsets(alarm: np.ndarray) -> np.ndarray:
    onsets = np.array(alarm, dtype=np.bool_, copy=True)
    if onsets.size > 1:
        onsets[1:] &= ~alarm[:-1]
    return onsets


def evaluate_confirmation_events(
    timestamps: Any,
    alarm: Any,
    *,
    events: Sequence[FaultInterval | Mapping[str, Any]] = CONFIRMATION_EVENTS,
) -> dict[str, Any]:
    """Measure F2/F3/F4 using new alarm onsets and t+1 availability."""

    ts = _timestamp_vector(timestamps, name="timestamps")
    decisions = _boolean_vector(alarm, name="alarm", length=int(ts.size))
    event_records = _validated_events(events)
    onsets = _alarm_onsets(decisions)
    availability = ts + np.timedelta64(AVAILABILITY_LAG_MINUTES, "m")

    results: list[dict[str, Any]] = []
    for event in event_records:
        left = event["start"]
        right = event["end"]
        score_in_event = (ts >= left) & (ts < right)
        available_in_event = (availability >= left) & (availability < right)
        credited = onsets & score_in_event & available_in_event
        locations = np.flatnonzero(credited)
        detected = bool(locations.size)
        if detected:
            first = int(locations[0])
            score_time = ts[first]
            available_time = availability[first]
            delay_ns = int(
                (available_time - left).astype("timedelta64[ns]").astype(np.int64)
            )
            delay = float(delay_ns / (60.0 * 1.0e9))
            if not math.isfinite(delay) or delay < 0.0:
                raise LockedEvaluationValidationError("event delay is invalid")
            score_text: str | None = _timestamp_text(score_time)
            availability_text: str | None = _timestamp_text(available_time)
        else:
            delay = None
            score_text = None
            availability_text = None
        results.append(
            {
                "event_id": event["event_id"],
                "start_inclusive": _timestamp_text(left),
                "end_exclusive": _timestamp_text(right),
                "detected": detected,
                "credited_alarm_onset_count": int(locations.size),
                "alarm_score_count_in_interval": int(
                    np.count_nonzero(decisions & score_in_event & available_in_event)
                ),
                "first_alarm_score_timestamp": score_text,
                "first_alarm_availability_timestamp": availability_text,
                "first_alarm_delay_minutes": delay,
            }
        )

    report: dict[str, Any] = {
        "schema": "metropt_cared_confirmation_events",
        "schema_version": 1,
        "interval_semantics": "left_closed_right_open",
        "alarm_semantics": "false_to_true_onset; first partition point true is an onset",
        "pre_event_persistent_alarm_credit": False,
        "availability_lag_minutes": AVAILABILITY_LAG_MINUTES,
        "event_count": len(results),
        "detected_event_count": int(sum(record["detected"] for record in results)),
        "events": results,
    }
    return _strict_json(report, name="confirmation event report")


def clopper_pearson_upper_bound(
    alarm_days: Any,
    healthy_days: Any,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided binomial upper confidence bound."""

    successes = _integer(alarm_days, name="alarm_days")
    trials = _integer(healthy_days, name="healthy_days", minimum=1)
    if successes > trials:
        raise LockedEvaluationValidationError("alarm_days cannot exceed healthy_days")
    level = _finite_scalar(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise LockedEvaluationValidationError("confidence must lie in (0, 1)")
    if successes == trials:
        return 1.0

    # The one-sided Clopper--Pearson upper endpoint is the unique p for
    # which P[Binomial(trials, p) <= successes] == 1-confidence.  Solving
    # this monotone equation avoids adding a runtime SciPy dependency and is
    # numerically stable for the frozen confirmation horizon (at most 103
    # calendar days).  Log-sum-exp keeps the binomial CDF finite at the
    # endpoints and in the small-tail regime.
    target_cdf = 1.0 - level

    def binomial_cdf(probability: float) -> float:
        if probability <= 0.0:
            return 1.0
        if probability >= 1.0:
            return 0.0
        log_p = math.log(probability)
        log_q = math.log1p(-probability)
        terms = [
            math.lgamma(trials + 1)
            - math.lgamma(index + 1)
            - math.lgamma(trials - index + 1)
            + index * log_p
            + (trials - index) * log_q
            for index in range(successes + 1)
        ]
        pivot = max(terms)
        return math.exp(pivot) * math.fsum(
            math.exp(term - pivot) for term in terms
        )

    lower = 0.0
    upper = 1.0
    for _ in range(128):
        midpoint = 0.5 * (lower + upper)
        if midpoint == lower or midpoint == upper:
            break
        if binomial_cdf(midpoint) > target_cdf:
            lower = midpoint
        else:
            upper = midpoint
    result = float(0.5 * (lower + upper))
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LockedEvaluationValidationError(
            "Clopper-Pearson upper bound is non-finite"
        )
    return result


def evaluate_healthy_confirmation_days(
    timestamps: Any,
    alarm: Any,
    *,
    start: Any = CONFIRMATION_START,
    end: Any = CONFIRMATION_END,
    events: Sequence[FaultInterval | Mapping[str, Any]] = CONFIRMATION_EVENTS,
) -> dict[str, Any]:
    """Compute locked healthy-day and healthy-score-time false-alarm metrics."""

    ts = _timestamp_vector(timestamps, name="timestamps")
    decisions = _boolean_vector(alarm, name="alarm", length=int(ts.size))
    left = _timestamp_scalar(start, name="start")
    right = _timestamp_scalar(end, name="end")
    days = _complete_days(left, right)
    if ts.size and (np.any(ts < left) or np.any(ts >= right)):
        raise LockedEvaluationValidationError(
            "timestamps must lie inside the confirmation interval"
        )
    fault_events = _validated_events(events)
    recovery = np.timedelta64(RECOVERY_HOURS, "h")
    date_values = ts.astype("datetime64[D]")
    records: list[dict[str, Any]] = []
    for day in days:
        day_start = day.astype("datetime64[ns]")
        day_end = day_start + np.timedelta64(1, "D")
        overlap_ids = [
            str(event["event_id"])
            for event in fault_events
            if day_start < event["end"] + recovery and day_end > event["start"]
        ]
        selected = date_values == day
        score_count = int(np.count_nonzero(selected))
        alarm_count = int(np.count_nonzero(decisions[selected]))
        coverage_qualified = score_count >= MINIMUM_SCORES_PER_DAY
        excluded = bool(overlap_ids)
        healthy_qualified = bool(coverage_qualified and not excluded)
        if excluded:
            reason = "fault_or_72h_recovery_overlap"
        elif score_count == 0:
            reason = "zero_scores"
        elif not coverage_qualified:
            reason = "insufficient_scores"
        else:
            reason = "qualified_healthy_day"
        records.append(
            {
                "date": _day_text(day),
                "score_count": score_count,
                "alarm_score_count": alarm_count,
                "coverage_qualified": coverage_qualified,
                "fault_or_recovery_excluded": excluded,
                "overlapping_event_ids": overlap_ids,
                "healthy_qualified": healthy_qualified,
                "any_alarm": bool(alarm_count > 0) if healthy_qualified else None,
                "qualification_reason": reason,
            }
        )

    healthy_candidates = [record for record in records if not record["fault_or_recovery_excluded"]]
    qualified = [record for record in records if record["healthy_qualified"]]
    healthy_count = len(qualified)
    if healthy_count == 0:
        raise LockedEvaluationValidationError(
            "at least one qualified healthy confirmation day is required"
        )
    alarm_day_count = int(sum(bool(record["any_alarm"]) for record in qualified))
    total_scores = int(sum(record["score_count"] for record in qualified))
    total_alarm_scores = int(sum(record["alarm_score_count"] for record in qualified))
    if total_scores <= 0:
        raise LockedEvaluationValidationError("healthy scored-time denominator is zero")
    day_rate = float(alarm_day_count / healthy_count)
    time_fraction = float(total_alarm_scores / total_scores)
    upper = clopper_pearson_upper_bound(alarm_day_count, healthy_count)
    if not all(math.isfinite(value) for value in (day_rate, time_fraction, upper)):
        raise LockedEvaluationValidationError("healthy false-alarm metric is non-finite")

    report: dict[str, Any] = {
        "schema": "metropt_cared_confirmation_healthy_days",
        "schema_version": 1,
        "interval": {
            "start_inclusive": _timestamp_text(left),
            "end_exclusive": _timestamp_text(right),
            "notation": "[start, end)",
        },
        "minimum_scores_per_qualified_day": MINIMUM_SCORES_PER_DAY,
        "recovery_exclusion_hours": RECOVERY_HOURS,
        "candidate_complete_day_count": int(days.size),
        "fault_or_recovery_excluded_day_count": int(
            sum(record["fault_or_recovery_excluded"] for record in records)
        ),
        "healthy_candidate_day_count": len(healthy_candidates),
        "qualified_healthy_day_count": healthy_count,
        "healthy_alarm_day_count": alarm_day_count,
        "healthy_day_any_alarm_rate": day_rate,
        "healthy_day_alarm_rate_cp95_upper": upper,
        "healthy_score_count": total_scores,
        "healthy_alarm_score_count": total_alarm_scores,
        "healthy_scored_time_alarm_fraction": time_fraction,
        "days": records,
    }
    return _strict_json(report, name="healthy confirmation report")


def evaluate_locked_confirmation(
    timestamps: Any,
    alarm: Any,
    *,
    parity_exact: Any,
    start: Any = CONFIRMATION_START,
    end: Any = CONFIRMATION_END,
    events: Sequence[FaultInterval | Mapping[str, Any]] = CONFIRMATION_EVENTS,
) -> dict[str, Any]:
    """Evaluate all locked F2/F3/F4 and healthy false-alarm gates."""

    if not isinstance(parity_exact, (bool, np.bool_)):
        raise LockedEvaluationValidationError("parity_exact must be boolean")
    event_report = evaluate_confirmation_events(timestamps, alarm, events=events)
    healthy_report = evaluate_healthy_confirmation_days(
        timestamps,
        alarm,
        start=start,
        end=end,
        events=events,
    )
    records = event_report["events"]
    expected_ids = [event.event_id for event in CONFIRMATION_EVENTS]
    observed_ids = [record["event_id"] for record in records]
    if observed_ids != expected_ids:
        raise LockedEvaluationValidationError(
            "locked confirmation requires F2, F3, and F4 in frozen order"
        )
    detected = all(bool(record["detected"]) for record in records)
    delays = [record["first_alarm_delay_minutes"] for record in records]
    finite_delays = [float(value) for value in delays if value is not None]
    every_delay = bool(
        len(finite_delays) == 3
        and all(0.0 <= value <= MAXIMUM_EVENT_DELAY_MINUTES for value in finite_delays)
    )
    median_delay = float(np.median(finite_delays)) if len(finite_delays) == 3 else None
    median_ok = bool(
        median_delay is not None
        and median_delay <= MAXIMUM_MEDIAN_DELAY_MINUTES
    )
    healthy_days = int(healthy_report["qualified_healthy_day_count"])
    checks: dict[str, dict[str, Any]] = {
        "event_detection_3_of_3": {
            "observed": int(event_report["detected_event_count"]),
            "requirement": "== 3",
            "passed": bool(detected and event_report["detected_event_count"] == 3),
        },
        "each_event_delay_minutes": {
            "observed": delays,
            "requirement": "each <= 60",
            "passed": every_delay,
        },
        "median_event_delay_minutes": {
            "observed": median_delay,
            "requirement": "<= 30",
            "passed": median_ok,
        },
        "minimum_qualified_healthy_days": {
            "observed": healthy_days,
            "requirement": ">= 39",
            "passed": healthy_days >= MINIMUM_HEALTHY_CONFIRMATION_DAYS,
        },
        "healthy_day_any_alarm_rate": {
            "observed": float(healthy_report["healthy_day_any_alarm_rate"]),
            "requirement": "<= 0.05",
            "passed": healthy_report["healthy_day_any_alarm_rate"]
            <= MAXIMUM_HEALTHY_DAY_ALARM_RATE,
        },
        "healthy_day_alarm_rate_cp95_upper": {
            "observed": float(healthy_report["healthy_day_alarm_rate_cp95_upper"]),
            "requirement": "<= 0.10",
            "passed": healthy_report["healthy_day_alarm_rate_cp95_upper"]
            <= MAXIMUM_HEALTHY_DAY_ALARM_UCB,
        },
        "healthy_scored_time_alarm_fraction": {
            "observed": float(
                healthy_report["healthy_scored_time_alarm_fraction"]
            ),
            "requirement": "<= 0.01",
            "passed": healthy_report["healthy_scored_time_alarm_fraction"]
            <= MAXIMUM_HEALTHY_TIME_ALARM_FRACTION,
        },
        "cpu_cuda_alarm_decisions_exact": {
            "observed": bool(parity_exact),
            "requirement": "true",
            "passed": bool(parity_exact),
        },
    }
    passed = bool(all(record["passed"] for record in checks.values()))
    report: dict[str, Any] = {
        "schema": "metropt_cared_locked_confirmation_evaluation",
        "schema_version": 1,
        "decision": (
            "PASS_LOCKED_FORWARD_CONFIRMATION"
            if passed
            else "FAIL_LOCKED_FORWARD_CONFIRMATION_NO_RETUNING"
        ),
        "passed": passed,
        "events": event_report,
        "healthy": healthy_report,
        "checks": checks,
    }
    return _strict_json(report, name="locked confirmation report")


__all__ = [
    "AVAILABILITY_LAG_MINUTES",
    "BOUNDARY_EPSILON",
    "CALIBRATION_ALPHA",
    "CONFIRMATION_END",
    "CONFIRMATION_EVENTS",
    "CONFIRMATION_START",
    "FaultInterval",
    "LockedEvaluationValidationError",
    "MINIMUM_CALIBRATION_DAYS",
    "MINIMUM_HEALTHY_CONFIRMATION_DAYS",
    "MINIMUM_SCORES_PER_DAY",
    "RECOVERY_HOURS",
    "calibrate_locked_threshold",
    "clopper_pearson_upper_bound",
    "enumerate_complete_days",
    "evaluate_confirmation_events",
    "evaluate_healthy_confirmation_days",
    "evaluate_locked_confirmation",
    "strict_alarm_mask",
]
