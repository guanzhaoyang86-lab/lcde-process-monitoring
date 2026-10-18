from __future__ import annotations

import unittest

import numpy as np

from mate_spc.cva_mewma import CausalCVAMEWMAMonitor, _past_blocks
from mate_spc.dynamic_common import (
    calibrate_run_envelopes,
    family_persistence_path,
)
from mate_spc.multiscale_sparse import CausalMultiScaleSparseMonitor


def _design(length: int = 80) -> list[tuple[str, np.ndarray]]:
    records: list[tuple[str, np.ndarray]] = []
    for run_id in range(1, 21):
        rng = np.random.default_rng(1000 + run_id)
        innovations = rng.normal(size=(length, 52))
        values = np.empty_like(innovations)
        values[0] = innovations[0]
        for index in range(1, length):
            values[index] = 0.55 * values[index - 1] + innovations[index]
        values += 0.05 * np.sin(np.arange(length)[:, None] / 7.0)
        records.append((f"design-{run_id:02d}", values))
    return records


class DynamicCommonTests(unittest.TestCase):
    def test_persistence_is_applied_before_family_maximum(self) -> None:
        component = np.array(
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 0.0],
                [10.0, 0.0],
                [10.0, 0.0],
            ]
        )
        result = family_persistence_path(
            component, valid_start=0, window=3, threshold=5.0
        )
        self.assertEqual(result.persistence[2], 0.0)
        self.assertFalse(result.alarm[2])
        self.assertTrue(result.alarm[4])
        equal = family_persistence_path(
            component, valid_start=0, window=3, threshold=10.0
        )
        self.assertFalse(equal.alarm.any())

    def test_run_conformal_uses_q96_and_run_names_break_ties(self) -> None:
        runs = [
            (f"run-{index:03d}", np.array([-np.inf, float(index)]))
            for index in range(1, 101)
        ]
        summary = calibrate_run_envelopes(runs)
        self.assertEqual(summary.rank_one_based, 96)
        self.assertEqual(summary.threshold, 96.0)
        self.assertEqual(summary.rank_driver_run, "run-096")

    def test_run_conformal_rejects_nonfinite_values_after_prefix(self) -> None:
        valid = [
            (f"run-{index:03d}", np.array([-np.inf, float(index)]))
            for index in range(1, 101)
        ]
        for corrupt in (
            np.array([-np.inf, np.nan, 1.0]),
            np.array([-np.inf, np.inf, 1.0]),
            np.array([-np.inf, 1.0, -np.inf]),
        ):
            records = list(valid)
            records[0] = ("run-001", corrupt)
            with self.assertRaisesRegex(ValueError, "leading -inf"):
                calibrate_run_envelopes(records)


class CVAMEWMATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = _design()
        cls.monitor = CausalCVAMEWMAMonitor().fit_design(cls.design)

    def test_past_block_order_alignment_and_design_shape(self) -> None:
        values = np.arange(8 * 2, dtype=float).reshape(8, 2)
        blocks = _past_blocks(values, 3)
        np.testing.assert_array_equal(
            blocks[0], np.concatenate((values[2], values[1], values[0]))
        )
        self.assertEqual(self.monitor.design_pair_count_, 20 * (80 - 5 - 5 + 1))
        self.assertEqual(self.monitor.design_online_score_count_, 20 * (80 - 5 + 1))
        self.assertGreaterEqual(self.monitor.rank_, 1)
        self.assertLess(self.monitor.rank_, 260)

    def test_online_path_is_causal_and_has_exact_eligibility(self) -> None:
        values = self.design[0][1]
        original = self.monitor.decision_path(values)
        changed = values.copy()
        changed[51:] += 1e6
        counterfactual = self.monitor.decision_path(changed)
        np.testing.assert_allclose(
            original.persistence[:51],
            counterfactual.persistence[:51],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(np.isneginf(original.persistence[:6]).all())
        self.assertFalse(original.eligible[:6].any())
        self.assertTrue(original.eligible[6:].all())

    def test_calibration_and_strict_alarm(self) -> None:
        calibration = [
            (f"cal-{index:03d}", np.array([-np.inf, float(index)]))
            for index in range(1, 101)
        ]
        monitor = self.monitor
        monitor.calibrate_envelopes(calibration)
        self.assertEqual(monitor.threshold_, 96.0)
        values = self.design[1][1]
        path = monitor.predict(values)
        np.testing.assert_array_equal(
            path.alarm, path.eligible & (path.persistence > 96.0)
        )


class MultiScaleSparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = _design()
        cls.monitor = CausalMultiScaleSparseMonitor().fit_design(cls.design)

    def test_fixed_family_and_eligibility(self) -> None:
        self.assertEqual(len(self.monitor.component_names_), 9)
        result = self.monitor.decision_path(self.design[0][1])
        self.assertEqual(result.component_score.shape, (80, 9))
        self.assertTrue(np.isneginf(result.persistence[:61]).all())
        self.assertFalse(result.eligible[:61].any())
        self.assertTrue(result.eligible[61:].all())

    def test_online_path_is_causal(self) -> None:
        values = self.design[2][1]
        original = self.monitor.decision_path(values)
        changed = values.copy()
        changed[71:] *= -1000.0
        counterfactual = self.monitor.decision_path(changed)
        np.testing.assert_allclose(
            original.persistence[:71],
            counterfactual.persistence[:71],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
