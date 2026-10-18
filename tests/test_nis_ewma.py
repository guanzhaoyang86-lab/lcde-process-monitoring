from __future__ import annotations

import unittest

import numpy as np

from mate_spc.nis_ewma import NISEWMAMonitor


def _nis_from_transformed(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(values, dtype=float))


def _grid_design(t0: float = 0.0) -> list[tuple[str, np.ndarray]]:
    tail = _nis_from_transformed(np.array([0.0, 1.0, 2.0, 3.0]))
    return [
        (f"design-{index:02d}", np.concatenate([[t0], tail]))
        for index in range(1, 21)
    ]


class NISEWMATests(unittest.TestCase):
    def test_t0_is_excluded_and_reflected_ewma_and_3of3_are_exact(self) -> None:
        monitor = NISEWMAMonitor().fit_design(_grid_design(t0=1e100))
        self.assertAlmostEqual(monitor.center_, 1.5)
        self.assertAlmostEqual(monitor.scale_, 1.4826)

        desired_z = np.array([-1.0, 2.0, -1.0, 3.0])
        transformed = monitor.center_ + monitor.scale_ * desired_z
        nis = np.concatenate([[1e200], _nis_from_transformed(transformed)])
        result = monitor.decision_path(nis, threshold=0.1)

        np.testing.assert_allclose(result.standardized[1:], desired_z, atol=1e-14)
        np.testing.assert_allclose(
            result.ewma[1:],
            np.array([0.0, 0.4, 0.12, 0.696]),
            atol=1e-14,
        )
        self.assertTrue(np.isneginf(result.ewma[0]))
        self.assertTrue(np.isneginf(result.persistence[:3]).all())
        np.testing.assert_allclose(result.persistence[3:], np.array([0.0, 0.12]))
        np.testing.assert_array_equal(
            result.eligible,
            np.array([False, False, False, True, True]),
        )
        np.testing.assert_array_equal(
            result.alarm,
            np.array([False, False, False, False, True]),
        )

        equal_threshold = monitor.decision_path(
            nis,
            threshold=float(result.persistence[-1]),
        )
        self.assertFalse(equal_threshold.alarm.any())

        reference = NISEWMAMonitor().fit_design(_grid_design(t0=0.0))
        self.assertEqual(monitor.center_, reference.center_)
        self.assertEqual(monitor.scale_, reference.scale_)

    def test_roundoff_negative_is_clamped_and_material_negative_is_rejected(self) -> None:
        monitor = NISEWMAMonitor().fit_design(_grid_design())
        result = monitor.decision_path(
            np.array([99.0, -1e-13, 0.0, 1.0, 2.0])
        )
        np.testing.assert_array_equal(result.transformed[1:3], np.zeros(2))
        with self.assertRaisesRegex(ValueError, "below -1e-12"):
            monitor.decision_path(np.array([0.0, -1e-6, 0.0, 1.0]))

    def test_mad_falls_back_to_sample_sd_and_total_degeneracy_fails(self) -> None:
        tail = _nis_from_transformed(np.array([0.0, 0.0, 0.0, 1.0]))
        design = [
            (f"fallback-{index:02d}", np.concatenate([[0.0], tail]))
            for index in range(1, 21)
        ]
        monitor = NISEWMAMonitor().fit_design(design)
        pooled = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), 20)
        self.assertEqual(monitor.scale_source_, "sample_sd")
        self.assertAlmostEqual(monitor.scale_, float(np.std(pooled, ddof=1)))

        degenerate = [
            (f"zero-{index:02d}", np.zeros(5))
            for index in range(1, 21)
        ]
        with self.assertRaisesRegex(ValueError, "degenerate"):
            NISEWMAMonitor().fit_design(degenerate)

    def test_100_run_conformal_rank_driver_summary_and_strict_alarm(self) -> None:
        monitor = NISEWMAMonitor().fit_design(_grid_design())
        calibration: list[tuple[str, np.ndarray]] = []
        expected_maxima: list[float] = []
        for index in range(1, 101):
            level = 0.02 * index
            transformed = np.full(5, monitor.center_ + monitor.scale_ * level)
            nis = np.concatenate([[1e50], _nis_from_transformed(transformed)])
            name = f"cal-{index:03d}"
            calibration.append((name, nis))
            expected_maxima.append(float(monitor.decision_path(nis).persistence.max()))

        monitor.calibrate(calibration)
        expected_threshold = sorted(expected_maxima)[95]
        self.assertAlmostEqual(monitor.threshold_, expected_threshold)
        self.assertEqual(monitor.calibrator_.base_run_count_, 100)
        self.assertEqual(monitor.calibrator_.calibration_unit_, "healthy_run_maximum")
        self.assertEqual(
            monitor.calibrator_.mechanism_thresholds_["base_observed"],
            monitor.threshold_,
        )

        summary = monitor.summary_
        self.assertEqual(summary.conformal_rank_one_based, 96)
        self.assertEqual(summary.rank_driver_run, "cal-096")
        self.assertAlmostEqual(summary.rank_driver_score, monitor.threshold_)
        self.assertEqual(dict(summary.order_statistics)[96], monitor.threshold_)
        self.assertEqual(len(summary.run_maxima), 100)
        self.assertEqual(
            monitor.calibration_summary_["alarm_comparison"],
            "strictly_greater_than",
        )

        self.assertFalse(monitor.predict(calibration[95][1]).alarm.any())
        self.assertTrue(monitor.predict(calibration[96][1]).alarm.any())

    def test_counts_names_shapes_and_finite_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            NISEWMAMonitor().fit_design(_grid_design()[:-1])

        duplicates = _grid_design()
        duplicates[-1] = (duplicates[0][0], duplicates[-1][1])
        with self.assertRaisesRegex(ValueError, "unique"):
            NISEWMAMonitor().fit_design(duplicates)

        nonfinite = _grid_design()
        nonfinite[0] = (nonfinite[0][0], np.array([0.0, 1.0, np.nan, 2.0]))
        with self.assertRaisesRegex(ValueError, "finite"):
            NISEWMAMonitor().fit_design(nonfinite)

        with self.assertRaisesRegex(RuntimeError, "not fitted"):
            NISEWMAMonitor().calibrate([])

        monitor = NISEWMAMonitor().fit_design(_grid_design())
        calibration = [
            (f"cal-{index:03d}", np.arange(5, dtype=float))
            for index in range(1, 100)
        ]
        with self.assertRaisesRegex(ValueError, "exactly 100"):
            monitor.calibrate(calibration)
        with self.assertRaisesRegex(ValueError, "length at least 4"):
            monitor.decision_path(np.ones(3))
        with self.assertRaisesRegex(ValueError, "finite"):
            monitor.decision_path(np.array([0.0, 1.0, 2.0, np.inf]))


if __name__ == "__main__":
    unittest.main()
