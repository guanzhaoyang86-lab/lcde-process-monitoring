from __future__ import annotations

import math

import numpy as np
import pytest

from mate_spc_cris_v9.evaluation import (
    EpisodeOutcome,
    RunOutcome,
    clopper_pearson_interval,
    clopper_pearson_upper,
    evaluate_runs,
    nearest_rank,
    right_grace_mask,
)


def _alarms(length: int, *indices: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.bool_)
    result[list(indices)] = True
    return result


def test_nearest_rank_is_noninterpolated() -> None:
    assert nearest_rank(tuple(range(10)), 0.90) == 8.0
    assert nearest_rank((2.0, 100.0), 0.50) == 2.0
    with pytest.raises(ValueError):
        nearest_rank((), 0.90)


def test_event_matching_excludes_pre_onset_and_post_window_alarms() -> None:
    miss = RunOutcome(
        run_id=1,
        condition="fault-3",
        alarm=_alarms(10, 1, 7),
        onset=3,
        event_stop=7,
    )
    assert miss.pre_onset_false_positive
    assert not miss.detected
    assert miss.detection_index is None
    assert miss.miss_delay == 4
    assert miss.delay == 4

    detected = RunOutcome(
        run_id=2,
        condition="fault-3",
        alarm=_alarms(10, 1, 4, 5),
        onset=3,
        event_stop=7,
    )
    assert detected.pre_onset_false_positive
    assert detected.detected
    assert detected.detection_index == 4
    assert detected.delay == 1


def test_episode_matching_uses_alarm_onsets_and_fixed_right_grace() -> None:
    fault = np.zeros(12, dtype=np.bool_)
    fault[5:7] = True
    assert np.flatnonzero(right_grace_mask(fault, 2)).tolist() == [5, 6, 7, 8]

    alarm = np.zeros(12, dtype=np.bool_)
    alarm[3:6] = True
    continuous = EpisodeOutcome(
        run_id="continuous",
        condition="intermittent_pump",
        alarm=alarm,
        fault_mask=fault,
        grace_samples=2,
    )
    assert continuous.pre_effect_false_alarm
    assert not continuous.detected

    matched = EpisodeOutcome(
        run_id="grace",
        condition="intermittent_pump",
        alarm=_alarms(12, 8),
        fault_mask=fault,
        grace_samples=2,
    )
    assert matched.detected
    assert matched.detection_index == 8
    assert matched.delay == 3

    inactive = EpisodeOutcome(
        run_id="inactive",
        condition="intermittent_pump",
        alarm=_alarms(12, 9),
        fault_mask=fault,
        grace_samples=2,
    )
    assert not inactive.detected
    assert inactive.inactive_gap_alarm_onsets == 1
    assert inactive.delay == 7


def test_run_outcome_from_scores_uses_strict_threshold_and_unknown_false() -> None:
    row = RunOutcome.from_score_path(
        run_id="h1",
        condition="healthy",
        score_path=np.asarray([np.nan, 2.0, 2.0001]),
        threshold=2.0,
    )
    assert row.alarm.tolist() == [False, False, True]
    assert row.any_alarm
    assert not row.is_fault


def test_evaluate_runs_counts_independent_runs_misses_delays_and_pre_fp() -> None:
    rows = (
        RunOutcome("h1", "healthy", _alarms(10)),
        RunOutcome("h2", "healthy", _alarms(10, 2, 3, 4)),
        RunOutcome("a", "fault-3", _alarms(10, 1, 4), onset=3, event_stop=7),
        RunOutcome("b", "fault-3", _alarms(10, 1, 7), onset=3, event_stop=7),
        RunOutcome("c", "fault-9", _alarms(10, 3), onset=3, event_stop=9),
    )
    report = evaluate_runs(rows)

    assert report.healthy.runs == 2
    assert report.healthy.alarm_runs == 1
    assert report.healthy.alarm_rate == 0.5
    fault3 = report.fault("fault-3")
    assert fault3.runs == 2
    assert fault3.detections == 1
    assert fault3.misses == 1
    assert fault3.recall == 0.5
    assert fault3.delays == (1, 4)
    assert fault3.median_delay == 2.5
    assert fault3.nearest_rank_p90_delay == 4.0
    assert fault3.pre_onset_false_positive_runs == 2
    assert fault3.pre_onset_false_positive_rate == 1.0

    assert report.faults_overall.runs == 3
    assert report.faults_overall.detections == 2
    assert report.faults_overall.recall == pytest.approx(2.0 / 3.0)
    assert report.faults_overall.delays == (1, 4, 0)
    assert report.faults_overall.median_delay == 1.0
    assert report.faults_overall.nearest_rank_p90_delay == 4.0


def test_duplicate_run_identity_is_rejected() -> None:
    rows = (
        RunOutcome(1, "healthy", _alarms(5)),
        RunOutcome(1, "healthy", _alarms(5)),
        RunOutcome(1, "fault", _alarms(5), onset=1, event_stop=5),
    )
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_runs(rows)


def test_clopper_pearson_boundaries_and_closed_form_zero_events() -> None:
    confidence = 0.95
    expected = 1.0 - (1.0 - confidence) ** (1.0 / 100.0)
    assert clopper_pearson_upper(0, 100, confidence=confidence) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )
    assert clopper_pearson_upper(100, 100) == 1.0
    lower, upper = clopper_pearson_interval(0, 100, confidence=confidence)
    assert lower == 0.0
    assert upper == pytest.approx(1.0 - 0.025 ** (1.0 / 100.0), rel=1e-12)
    assert math.isclose(clopper_pearson_interval(100, 100)[1], 1.0)


def test_clopper_pearson_matches_scipy_when_available() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    for successes, trials in ((1, 20), (4, 100), (50, 100), (99, 100)):
        expected_upper = scipy_stats.beta.ppf(0.95, successes + 1, trials - successes)
        assert clopper_pearson_upper(successes, trials) == pytest.approx(
            expected_upper, rel=1e-11, abs=1e-13
        )
        expected_interval = scipy_stats.beta.interval(
            0.95, successes, trials - successes + 1
        )[0], scipy_stats.beta.interval(
            0.95, successes + 1, trials - successes
        )[1]
        observed = clopper_pearson_interval(successes, trials)
        np.testing.assert_allclose(observed, expected_interval, rtol=1e-11, atol=1e-13)
