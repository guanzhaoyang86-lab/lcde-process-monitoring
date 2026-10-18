from __future__ import annotations

import unittest

import numpy as np

from mate_spc.metrics import event_metrics


class EventMetricTests(unittest.TestCase):
    def test_timing_arguments_must_be_finite_and_integral(self) -> None:
        alarm = np.zeros(5, dtype=bool)
        fault = np.zeros(5, dtype=bool)
        for kwargs in (
            {"sample_period_seconds": float("nan")},
            {"sample_period_seconds": 1.0, "warning_horizon": 1.5},
            {"sample_period_seconds": 1.0, "refractory": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "invalid timing argument"
            ):
                event_metrics(alarm, fault, **kwargs)

    def test_one_alarm_episode_cannot_detect_two_fault_events(self) -> None:
        alarm = np.array([False, True, False, False, False])
        fault = np.array([True, True, False, True, True])
        result = event_metrics(alarm, fault, 1.0, warning_horizon=3)
        self.assertEqual(result.event_count, 2)
        self.assertEqual(result.events_detected, 1)
        self.assertEqual(result.event_recall, 0.5)

    def test_unknown_normal_time_is_excluded_from_alarm_exposure(self) -> None:
        alarm = np.array([False, False, False, True])
        fault = np.zeros(4, dtype=bool)
        decision = np.array([False, False, True, True])
        result = event_metrics(alarm, fault, 3600.0, decision_mask=decision)
        self.assertEqual(result.monitored_normal_hours, 2.0)
        self.assertEqual(result.decision_coverage, 0.5)
        self.assertEqual(result.normal_alarm_occupancy, 0.5)

    def test_longstanding_nuisance_alarm_is_not_a_fault_detection(self) -> None:
        alarm = np.ones(12, dtype=bool)
        fault = np.zeros(12, dtype=bool)
        fault[10] = True
        result = event_metrics(alarm, fault, 1.0, warning_horizon=0)
        self.assertEqual(result.events_detected, 0)
        self.assertEqual(result.false_alarm_episodes, 2)

    def test_refractory_merge_cannot_pull_old_alarm_into_warning_window(self) -> None:
        alarm = np.array([False, True, False, True, False])
        fault = np.array([False, False, False, True, True])
        result = event_metrics(
            alarm, fault, 1.0, warning_horizon=1, refractory=2
        )
        self.assertEqual(result.events_detected, 0)


if __name__ == "__main__":
    unittest.main()
