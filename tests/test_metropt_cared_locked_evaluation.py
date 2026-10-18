from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mate_spc_metropt.locked_evaluation import (
    BOUNDARY_EPSILON,
    CONFIRMATION_END,
    CONFIRMATION_EVENTS,
    CONFIRMATION_START,
    MINIMUM_SCORES_PER_DAY,
    FaultInterval,
    LockedEvaluationValidationError,
    calibrate_locked_threshold,
    clopper_pearson_upper_bound,
    enumerate_complete_days,
    evaluate_confirmation_events,
    evaluate_healthy_confirmation_days,
    evaluate_locked_confirmation,
    strict_alarm_mask,
)


def _scores_for_days(
    start: str,
    maxima: list[float] | np.ndarray,
    *,
    count_per_day: int = MINIMUM_SCORES_PER_DAY,
) -> tuple[np.ndarray, np.ndarray, str]:
    day0 = np.datetime64(start, "D")
    timestamp_rows: list[np.ndarray] = []
    envelope_rows: list[np.ndarray] = []
    for offset, maximum in enumerate(np.asarray(maxima, dtype=np.float64)):
        day = day0 + np.timedelta64(offset, "D")
        timestamps = day.astype("datetime64[m]") + np.arange(count_per_day).astype(
            "timedelta64[m]"
        )
        timestamp_rows.append(timestamps.astype("datetime64[ns]"))
        envelope_rows.append(np.full(count_per_day, float(maximum), dtype=np.float64))
    stop = day0 + np.timedelta64(len(maxima), "D")
    return (
        np.concatenate(timestamp_rows) if timestamp_rows else np.array([], dtype="datetime64[ns]"),
        np.concatenate(envelope_rows) if envelope_rows else np.array([], dtype=np.float64),
        str(stop),
    )


def _day_report(
    partition: str,
    start: str,
    maxima: list[float] | np.ndarray,
) -> dict[str, object]:
    timestamps, envelope, stop = _scores_for_days(start, maxima)
    return enumerate_complete_days(
        timestamps,
        envelope,
        start=start,
        end=stop,
        partition_name=partition,
    )


def _full_confirmation_minutes() -> np.ndarray:
    start = np.datetime64(CONFIRMATION_START, "m")
    end = np.datetime64(CONFIRMATION_END, "m")
    count = int((end - start) / np.timedelta64(1, "m"))
    return (start + np.arange(count).astype("timedelta64[m]")).astype(
        "datetime64[ns]"
    )


def test_complete_day_enumeration_keeps_zero_insufficient_and_exact_boundary() -> None:
    second = np.datetime64("2020-01-02", "m") + np.arange(1140).astype(
        "timedelta64[m]"
    )
    third = np.datetime64("2020-01-03", "m") + np.arange(1141).astype(
        "timedelta64[m]"
    )
    timestamps = np.concatenate((second, third)).astype("datetime64[ns]")
    envelope = np.arange(timestamps.size, dtype=np.float64)

    report = enumerate_complete_days(
        timestamps,
        envelope,
        start="2020-01-01 00:00:00",
        end="2020-01-04 00:00:00",
        partition_name="synthetic",
    )

    assert report["candidate_day_count"] == 3
    assert report["qualified_day_count"] == 1
    assert report["days"][0] == {
        "date": "2020-01-01",
        "score_count": 0,
        "maximum": None,
        "qualified": False,
        "qualification_reason": "zero_scores",
        "included_in_calibration": False,
    }
    assert report["days"][1]["score_count"] == 1140
    assert report["days"][1]["qualification_reason"] == "insufficient_scores"
    assert report["days"][2]["score_count"] == 1141
    assert report["days"][2]["qualified"] is True
    json.dumps(report, allow_nan=False)


def test_complete_days_exclude_partial_boundary_days() -> None:
    timestamps, envelope, _ = _scores_for_days("2020-01-02", [1.0, 2.0])
    report = enumerate_complete_days(
        timestamps,
        envelope,
        start="2020-01-01 12:00:00",
        end="2020-01-04 12:00:00",
        partition_name="partial_edges",
    )
    assert [record["date"] for record in report["days"]] == [
        "2020-01-02",
        "2020-01-03",
    ]


def test_calibration_rank_for_39_days_is_38_and_ties_are_deterministic() -> None:
    first = _day_report("calibration_a", "2020-01-01", [5.0] * 20)
    second = _day_report("calibration_b", "2020-02-01", [5.0] * 19)

    result = calibrate_locked_threshold(first, second, portability_delta=2.0e-8)

    assert result["qualified_day_count"] == 39
    assert result["rank_one_based"] == 38
    assert result["raw_boundary"] == 5.0
    assert result["numerical_guard"] == 2.0e-8 + BOUNDARY_EPSILON
    assert result["effective_boundary"] == np.nextafter(
        5.0 + 2.0e-8 + BOUNDARY_EPSILON, np.inf
    )
    assert result["exchangeable_day_alarm_probability_bound"] == 0.05
    json.dumps(result, allow_nan=False)


def test_calibration_boundary_evaluation_order_is_frozen() -> None:
    raw = 36.02052930938596
    delta = 0.21800327523659302
    first = _day_report("calibration_a", "2020-01-01", [raw] * 20)
    second = _day_report("calibration_b", "2020-02-01", [raw] * 19)

    result = calibrate_locked_threshold(first, second, portability_delta=delta)

    expected = float(np.nextafter(raw + delta + BOUNDARY_EPSILON, np.inf))
    assert result["effective_boundary"] == expected
    assert raw + delta + BOUNDARY_EPSILON != raw + (delta + BOUNDARY_EPSILON)


def test_calibration_rank_for_70_days_is_68() -> None:
    maxima = np.arange(1.0, 71.0)
    first = _day_report("calibration_a", "2020-03-01", maxima[:41])
    second = _day_report("calibration_b", "2020-04-22", maxima[41:])

    result = calibrate_locked_threshold(first, second, portability_delta=0.0)

    assert result["candidate_day_count"] == 70
    assert result["qualified_day_count"] == 70
    assert result["rank_one_based"] == 68
    assert result["raw_boundary"] == 68.0
    assert result["exchangeable_day_alarm_probability_bound"] == pytest.approx(3 / 71)


def test_calibration_rejects_fewer_than_39_qualified_days_and_overlapping_dates() -> None:
    first = _day_report("calibration_a", "2020-01-01", [1.0] * 20)
    short = _day_report("calibration_b", "2020-02-01", [2.0] * 18)
    with pytest.raises(LockedEvaluationValidationError, match="at least 39"):
        calibrate_locked_threshold(first, short, portability_delta=0.0)

    overlap = _day_report("calibration_b", "2020-01-20", [2.0] * 20)
    with pytest.raises(LockedEvaluationValidationError, match="overlap"):
        calibrate_locked_threshold(first, overlap, portability_delta=0.0)


def test_strict_threshold_does_not_alarm_on_tie() -> None:
    boundary = 3.0
    values = [np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)]
    assert strict_alarm_mask(values, boundary).tolist() == [False, False, True]


def test_pre_event_persistent_alarm_is_not_credited_but_new_onset_is() -> None:
    event = FaultInterval("F2", "2020-05-29 23:30:00", "2020-05-30 00:00:00")
    timestamps = np.array(
        [
            "2020-05-29T23:28",
            "2020-05-29T23:29",
            "2020-05-29T23:30",
            "2020-05-29T23:31",
            "2020-05-29T23:32",
        ],
        dtype="datetime64[ns]",
    )
    persistent = evaluate_confirmation_events(
        timestamps,
        np.array([True, True, True, True, True]),
        events=(event,),
    )
    assert persistent["events"][0]["detected"] is False

    new_onset = evaluate_confirmation_events(
        timestamps,
        np.array([True, True, True, False, True]),
        events=(event,),
    )
    record = new_onset["events"][0]
    assert record["detected"] is True
    assert record["first_alarm_score_timestamp"] == "2020-05-29T23:32:00.000000000"
    assert record["first_alarm_availability_timestamp"] == "2020-05-29T23:33:00.000000000"
    assert record["first_alarm_delay_minutes"] == 3.0


def test_event_availability_is_left_closed_right_open_and_first_true_is_onset() -> None:
    event = FaultInterval("F2", "2020-01-01 00:00:00", "2020-01-01 00:10:00")
    at_start = evaluate_confirmation_events(
        np.array(["2020-01-01T00:00"], dtype="datetime64[ns]"),
        np.array([True]),
        events=(event,),
    )
    assert at_start["events"][0]["detected"] is True
    assert at_start["events"][0]["first_alarm_delay_minutes"] == 1.0

    at_last_score = evaluate_confirmation_events(
        np.array(["2020-01-01T00:09"], dtype="datetime64[ns]"),
        np.array([True]),
        events=(event,),
    )
    assert at_last_score["events"][0]["detected"] is False


def test_confirmation_calendar_has_15_excluded_and_88_healthy_candidate_days() -> None:
    timestamps = _full_confirmation_minutes()
    alarm = np.zeros(timestamps.size, dtype=bool)

    report = evaluate_healthy_confirmation_days(timestamps, alarm)

    excluded = [
        record["date"]
        for record in report["days"]
        if record["fault_or_recovery_excluded"]
    ]
    assert report["candidate_complete_day_count"] == 103
    assert report["fault_or_recovery_excluded_day_count"] == 15
    assert report["healthy_candidate_day_count"] == 88
    assert report["qualified_healthy_day_count"] == 88
    assert excluded == [
        "2020-05-29",
        "2020-05-30",
        "2020-05-31",
        "2020-06-01",
        "2020-06-02",
        "2020-06-05",
        "2020-06-06",
        "2020-06-07",
        "2020-06-08",
        "2020-06-09",
        "2020-06-10",
        "2020-07-15",
        "2020-07-16",
        "2020-07-17",
        "2020-07-18",
    ]
    assert report["healthy_day_any_alarm_rate"] == 0.0
    assert report["healthy_scored_time_alarm_fraction"] == 0.0
    json.dumps(report, allow_nan=False)


def test_clopper_pearson_zero_and_all_alarm_boundaries() -> None:
    zero = clopper_pearson_upper_bound(0, 88)
    assert zero == pytest.approx(1.0 - 0.05 ** (1.0 / 88.0))
    assert math.isfinite(zero)
    assert clopper_pearson_upper_bound(88, 88) == 1.0


def test_clopper_pearson_matches_optional_scipy_reference() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for successes, trials in ((1, 5), (3, 20), (17, 103), (51, 103), (102, 103)):
        expected = scipy_stats.beta.ppf(0.95, successes + 1, trials - successes)
        assert clopper_pearson_upper_bound(successes, trials) == pytest.approx(
            expected,
            abs=1.0e-12,
        )


@pytest.mark.parametrize(
    "successes,trials",
    [(1, 0), (2, 1), (-1, 10), (True, 10), (1.5, 10)],
)
def test_clopper_pearson_rejects_invalid_counts(successes: object, trials: object) -> None:
    with pytest.raises(LockedEvaluationValidationError):
        clopper_pearson_upper_bound(successes, trials)


def test_full_locked_confirmation_passes_all_gates_and_parity_is_binding() -> None:
    timestamps = _full_confirmation_minutes()
    alarm = np.zeros(timestamps.size, dtype=bool)
    for event in CONFIRMATION_EVENTS:
        location = int(
            (np.datetime64(event.start, "m") - timestamps[0].astype("datetime64[m]"))
            / np.timedelta64(1, "m")
        )
        alarm[location] = True

    passed = evaluate_locked_confirmation(timestamps, alarm, parity_exact=True)
    assert passed["passed"] is True
    assert passed["decision"] == "PASS_LOCKED_FORWARD_CONFIRMATION"
    assert passed["events"]["detected_event_count"] == 3
    assert passed["checks"]["median_event_delay_minutes"]["observed"] == 1.0
    assert passed["healthy"]["qualified_healthy_day_count"] == 88
    json.dumps(passed, allow_nan=False)

    failed = evaluate_locked_confirmation(timestamps, alarm, parity_exact=False)
    assert failed["passed"] is False
    assert failed["decision"] == "FAIL_LOCKED_FORWARD_CONFIRMATION_NO_RETUNING"
    assert failed["checks"]["cpu_cuda_alarm_decisions_exact"]["passed"] is False


@pytest.mark.parametrize(
    "timestamps,envelope,match",
    [
        (
            np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            np.array([1.0]),
            "shape",
        ),
        (
            np.array(["2020-01-01"], dtype="datetime64[ns]"),
            np.array([np.nan]),
            "non-finite",
        ),
        (
            np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[ns]"),
            np.array([1.0, 2.0]),
            "strictly increasing",
        ),
    ],
)
def test_day_enumeration_rejects_shape_nan_and_duplicate_timestamps(
    timestamps: np.ndarray,
    envelope: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(LockedEvaluationValidationError, match=match):
        enumerate_complete_days(
            timestamps,
            envelope,
            start="2020-01-01",
            end="2020-01-03",
            partition_name="bad",
        )


@pytest.mark.parametrize("delta", [-1.0, np.nan, np.inf, True])
def test_calibration_rejects_invalid_portability_delta(delta: object) -> None:
    first = _day_report("calibration_a", "2020-01-01", [1.0] * 20)
    second = _day_report("calibration_b", "2020-02-01", [2.0] * 19)
    with pytest.raises(LockedEvaluationValidationError):
        calibrate_locked_threshold(first, second, portability_delta=delta)


def test_alarm_and_event_inputs_reject_wrong_dtype_shape_and_nat() -> None:
    with pytest.raises(LockedEvaluationValidationError, match="boolean"):
        evaluate_confirmation_events(
            np.array(["2020-01-01"], dtype="datetime64[ns]"),
            np.array([1], dtype=np.int64),
            events=(FaultInterval("F2", "2020-01-01", "2020-01-02"),),
        )
    with pytest.raises(LockedEvaluationValidationError, match="NaT"):
        evaluate_confirmation_events(
            np.array(["NaT"], dtype="datetime64[ns]"),
            np.array([False]),
            events=(FaultInterval("F2", "2020-01-01", "2020-01-02"),),
        )
    with pytest.raises(LockedEvaluationValidationError, match="non-finite"):
        strict_alarm_mask([1.0, np.inf], 2.0)


def test_healthy_metrics_reject_zero_qualified_denominator() -> None:
    timestamps = np.array(["2020-05-21T00:00"], dtype="datetime64[ns]")
    with pytest.raises(LockedEvaluationValidationError, match="at least one"):
        evaluate_healthy_confirmation_days(timestamps, np.array([False]))
