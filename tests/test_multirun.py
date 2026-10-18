from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mate_spc.datasets import (
    RIETH_FEATURE_COLUMNS,
    SensorSequence,
    rieth_frame_to_runs,
)
from mate_spc.missingness import MissingnessConfig
from mate_spc.model import MAPICConfig
from mate_spc.monitor import BlockConformalAlarmCalibrator
from mate_spc.multirun import (
    aggregate_multirun_reports,
    evaluate_multirun_stress,
    fit_multirun_suite,
    write_multirun_markdown,
)


def _normal_run(name: str, seed: int, rows: int = 50) -> SensorSequence:
    rng = np.random.default_rng(seed)
    x = np.zeros((rows, 3), dtype=float)
    noise = rng.normal(scale=0.2, size=x.shape)
    for index in range(1, rows):
        x[index] = 0.6 * x[index - 1] + noise[index]
    return SensorSequence(
        name=name,
        x=x,
        fault=np.zeros(rows, dtype=bool),
        observed_mask=np.ones_like(x, dtype=bool),
        feature_names=("a", "b", "c"),
        sample_period_seconds=180.0,
    )


class RiethAdapterTests(unittest.TestCase):
    def test_frame_is_split_into_strict_run_boundaries(self) -> None:
        rows = []
        for fault_id, run_id in ((1, 1), (3, 2)):
            for sample in range(1, 5):
                row = {
                    "faultNumber": fault_id,
                    "simulationRun": run_id,
                    "sample": sample,
                }
                row.update(
                    {
                        column: float(1000 * fault_id + 10 * run_id + sample)
                        for column in RIETH_FEATURE_COLUMNS
                    }
                )
                rows.append(row)
        frame = pd.DataFrame(rows[::-1])
        runs = rieth_frame_to_runs(
            frame,
            "faulty_testing",
            {1: (1,), 3: (2,)},
            source_name="fixture.RData",
            source_md5="fixture",
            fault_start=2,
        )
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].metadata["simulation_run"], 1)
        self.assertEqual(runs[1].metadata["fault_id"], 3)
        np.testing.assert_array_equal(
            runs[0].fault, np.array([False, False, True, True])
        )
        self.assertEqual(runs[0].x.shape, (4, 52))
        self.assertTrue(np.all(np.diff(runs[0].x[:, 0]) == 1.0))


class RunCalibratorTests(unittest.TestCase):
    def test_masks_and_time_are_collapsed_within_run(self) -> None:
        records = []
        for run_id in range(1, 5):
            for mechanism, offset in (("base", 0.0), ("mcar", 0.2)):
                for repeat in range(2):
                    values = np.array(
                        [0.0, float(run_id) + offset + 0.01 * repeat]
                    )
                    records.append(
                        (mechanism, f"run-{run_id}", values)
                    )
        calibrator = BlockConformalAlarmCalibrator(
            alpha=0.25, block_size=2, min_blocks=2
        ).fit_run_clusters(records)
        self.assertEqual(calibrator.base_run_count_, 4)
        self.assertEqual(calibrator.calibration_unit_, "healthy_run_maximum")
        self.assertAlmostEqual(calibrator.threshold_, 4.21)
        self.assertEqual(calibrator.augmentation_counts_["mcar"], 2)


class MultiRunEndToEndTests(unittest.TestCase):
    def test_suite_fit_evaluate_and_run_bootstrap(self) -> None:
        train = tuple(
            _normal_run(f"train-{index}", 100 + index)
            for index in range(3)
        )
        calibration = tuple(
            _normal_run(f"cal-{index}", 200 + index)
            for index in range(4)
        )
        config = MAPICConfig(
            alpha=0.25,
            block_size=5,
            min_calibration_blocks=2,
            calibration_repeats=1,
            persistence_k=2,
            persistence_window=2,
            seed=7,
        )
        suite = fit_multirun_suite(
            train,
            calibration,
            config,
            baseline_methods=("kalman_mean",),
        )
        self.assertEqual(
            suite.proposed.monitor.filter_model.training_transition_count_,
            3 * 49,
        )
        self.assertEqual(
            suite.proposed.calibration_summary_["base_run_count"], 4
        )
        scenario = MissingnessConfig("mcar", rate=0.1)
        reports = []
        for run_id in (1, 2):
            base = _normal_run(f"fault-1-run-{run_id}", 300 + run_id)
            x = base.x.copy()
            x[25:, 0] += 2.0
            test = SensorSequence(
                name=base.name,
                x=x,
                fault=np.arange(x.shape[0]) >= 25,
                observed_mask=np.ones_like(x, dtype=bool),
                feature_names=base.feature_names,
                sample_period_seconds=180.0,
                metadata={"fault_id": 1, "simulation_run": run_id},
            )
            reports.append(
                evaluate_multirun_stress(
                    suite, test, (scenario,), seeds=(11,)
                )
            )
        aggregate = aggregate_multirun_reports(
            reports, bootstrap_replicates=20, bootstrap_seed=4
        )
        self.assertEqual(aggregate["run_count"], 2)
        self.assertEqual(aggregate["nested_stress_views_per_run"], 1)
        self.assertEqual(aggregate["run_counts_by_fault"], {"01": 2})
        self.assertIn("UD-MAPIC", aggregate["overall"])
        self.assertEqual(
            aggregate["run_clustered_bootstrap"]["replicates"], 20
        )
        with TemporaryDirectory() as folder:
            path = Path(folder) / "report.md"
            write_multirun_markdown(aggregate, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("Run-clustered 95% bootstrap intervals", text)
        self.assertIn("Paired UD-MAPIC differences", text)


if __name__ == "__main__":
    unittest.main()
