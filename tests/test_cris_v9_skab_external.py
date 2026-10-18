from __future__ import annotations

import numpy as np
import pytest

from mate_spc_cris_v9.skab_external import (
    binary_average_precision,
    evaluate_skab_path,
    prepend_warmup,
    warmup_standardize,
)


def _seconds(values: list[int] | np.ndarray) -> np.ndarray:
    origin = np.datetime64("2026-01-01T00:00:00", "s")
    return origin + np.asarray(values, dtype="timedelta64[s]")


def test_warmup_standardize_fits_only_the_prefix() -> None:
    prefix = np.column_stack(
        (np.arange(300, dtype=np.float64), np.full(300, 7.0))
    )
    first = np.vstack((prefix, [[300.0, 8.0], [301.0, 9.0]]))
    second = first.copy()
    second[300:] = [[30_000.0, -50.0], [40_000.0, -60.0]]

    standardized = warmup_standardize(first)
    changed_future = warmup_standardize(second)

    np.testing.assert_array_equal(standardized[:300], changed_future[:300])
    np.testing.assert_allclose(standardized[:300, 0].mean(), 0.0, atol=1e-14)
    np.testing.assert_allclose(standardized[:300, 0].std(), 1.0, atol=1e-14)
    np.testing.assert_array_equal(standardized[:300, 1], np.zeros(300))
    assert standardized[300, 1] == pytest.approx(1.0e8)
    assert standardized.dtype == np.float64
    assert standardized.flags.c_contiguous
    assert standardized.flags.writeable is False


@pytest.mark.parametrize(
    ("x", "warmup", "error"),
    [
        (np.ones(4), 2, ValueError),
        (np.empty((0, 2)), 1, ValueError),
        (np.ones((4, 0)), 1, ValueError),
        (np.ones((4, 2), dtype=bool), 2, TypeError),
        (np.array([[1.0], [np.nan]]), 1, ValueError),
        (np.ones((4, 2)), 0, ValueError),
        (np.ones((4, 2)), 5, ValueError),
        (np.ones((4, 2)), 2.0, TypeError),
    ],
)
def test_warmup_standardize_rejects_invalid_input(
    x: np.ndarray, warmup: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        warmup_standardize(x, warmup=warmup)  # type: ignore[arg-type]


def test_prepend_warmup_has_exact_length_and_readonly_prefix() -> None:
    result = prepend_warmup([1.5, 2.5, 3.5], total=5, warmup=2)
    np.testing.assert_array_equal(result, [0.0, 0.0, 1.5, 2.5, 3.5])
    assert result.dtype == np.float64
    assert result.flags.writeable is False


@pytest.mark.parametrize(
    ("path", "total", "warmup", "error"),
    [
        ([1.0, 2.0], 5, 2, ValueError),
        ([1.0], 1, 1, ValueError),
        ([1.0], 0, 0, ValueError),
        ([np.inf], 1, 0, ValueError),
        ([1.0], 1.0, 0, TypeError),
        ([1.0], 1, True, TypeError),
    ],
)
def test_prepend_warmup_rejects_invalid_contracts(
    path: list[float], total: object, warmup: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        prepend_warmup(  # type: ignore[arg-type]
            path,
            total=total,
            warmup=warmup,
        )


def test_binary_average_precision_matches_noninterpolated_definition() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert binary_average_precision(labels, scores) == pytest.approx(5.0 / 6.0)
    assert binary_average_precision([1, 0], [0.4, 0.4]) == pytest.approx(0.5)
    assert binary_average_precision([0, 1], [0.4, 0.4]) == pytest.approx(0.5)
    assert binary_average_precision([1, 1], [0.2, 0.1]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("labels", "scores", "error"),
    [
        ([1, 0], [0.5], ValueError),
        ([1.0, 0.0], [0.5, 0.4], ValueError),
        ([1, 2], [0.5, 0.4], ValueError),
        ([0, 0], [0.5, 0.4], ValueError),
        ([1, 0], [np.nan, 0.4], ValueError),
        ([1, 0], [[0.5, 0.4]], ValueError),
    ],
)
def test_binary_average_precision_rejects_invalid_input(
    labels: object, scores: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        binary_average_precision(labels, scores)


def test_evaluate_skab_path_uses_strict_threshold_and_new_alarm_edges() -> None:
    scores = np.array([0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 3.0, 0.0])
    anomaly = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    result = evaluate_skab_path(scores, 1.0, anomaly, _seconds(np.arange(10)))

    assert set(result) == {
        "detected",
        "delay_seconds",
        "within_60",
        "pre_event_false_alarm",
        "occupancy",
        "episodes",
        "episodes_per_hour",
        "unadjusted_f1",
        "average_precision",
    }
    assert result["detected"] is True
    assert result["delay_seconds"] == pytest.approx(1.0)
    assert result["within_60"] is True
    assert result["pre_event_false_alarm"] is True
    assert result["occupancy"] == pytest.approx(0.5)
    assert result["episodes"] == 3
    assert result["episodes_per_hour"] == pytest.approx(1200.0)
    assert result["unadjusted_f1"] == pytest.approx(6.0 / 11.0)
    assert result["average_precision"] == pytest.approx(
        binary_average_precision(anomaly, scores)
    )


def test_alarm_active_at_event_onset_is_not_a_new_detection() -> None:
    scores = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    anomaly = np.array([0, 0, 0, 0, 1, 1])

    result = evaluate_skab_path(scores, 1.0, anomaly, _seconds(np.arange(6)))

    assert result["detected"] is False
    assert result["delay_seconds"] is None
    assert result["within_60"] is False
    assert result["pre_event_false_alarm"] is True
    assert result["episodes"] == 1


def test_detection_delay_uses_timestamps_and_sixty_seconds_is_inclusive() -> None:
    anomaly = [0, 1, 1]
    score = [0.0, 0.0, 2.0]
    at_boundary = evaluate_skab_path(score, 1.0, anomaly, _seconds([0, 5, 65]))
    after_boundary = evaluate_skab_path(score, 1.0, anomaly, _seconds([0, 5, 66]))

    assert at_boundary["delay_seconds"] == pytest.approx(60.0)
    assert at_boundary["within_60"] is True
    assert after_boundary["delay_seconds"] == pytest.approx(61.0)
    assert after_boundary["within_60"] is False


def test_timestamp_duration_does_not_overflow_int64_subtraction() -> None:
    timestamps = np.array(["1700-01-01", "2200-01-01"], dtype="datetime64[D]")
    result = evaluate_skab_path([0.0, 2.0], 1.0, [0, 1], timestamps)

    assert result["detected"] is True
    assert result["delay_seconds"] == pytest.approx(0.0)
    assert 0.0 < result["episodes_per_hour"] < 1.0


@pytest.mark.parametrize(
    ("score", "threshold", "anomaly", "timestamps", "error"),
    [
        ([0.0], 1.0, [1], _seconds([0]), ValueError),
        ([0.0, 1.0], 1.0, [1], _seconds([0, 1]), ValueError),
        ([0.0, np.nan], 1.0, [0, 1], _seconds([0, 1]), ValueError),
        ([0.0, 1.0], np.nan, [0, 1], _seconds([0, 1]), ValueError),
        ([0.0, 1.0], True, [0, 1], _seconds([0, 1]), TypeError),
        ([0.0, 1.0], 1.0, [0, 0], _seconds([0, 1]), ValueError),
        ([0.0, 1.0], 1.0, [0.0, 1.0], _seconds([0, 1]), ValueError),
        ([0.0, 1.0], 1.0, [0, 1], np.array([0, 1]), ValueError),
        ([0.0, 1.0], 1.0, [0, 1], _seconds([0, 0]), ValueError),
        (
            [0.0, 1.0],
            1.0,
            [0, 1],
            np.array(["1000-01-01", "1000-01-02"], dtype="datetime64[D]"),
            ValueError,
        ),
        (
            [0.0, 1.0],
            1.0,
            [0, 1],
            np.array([np.datetime64("NaT"), np.datetime64("2026-01-01")]),
            ValueError,
        ),
    ],
)
def test_evaluate_skab_path_rejects_invalid_input(
    score: object,
    threshold: object,
    anomaly: object,
    timestamps: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        evaluate_skab_path(score, threshold, anomaly, timestamps)
