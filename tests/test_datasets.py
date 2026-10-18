from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mate_spc.datasets import BatchDistillationArchive, load_metropt, load_tep, phase_basis


class DatasetAdapterTests(unittest.TestCase):
    def test_batch_archives_align_by_time_and_restore_natural_mask(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name:
            folder = Path(folder_name)
            key = "Operation/system/operating_point_031/train_normal_experiment_001.csv"
            contents = {
                "labels": "Time,Label (anomaly)\n00:00:00,0\n00:00:01,0\n00:00:02,2\n",
                "sensors": "Time,S1,S2\n00:00:00,1,2\n00:00:01,3,4\n00:00:02,5,6\n",
                "actuators": "Time,A1\n00:00:00,0\n00:00:01,1\n00:00:02,1\n",
                "uncertainties": (
                    "Time,Data_interpolated,+-dS1\n"
                    "00:00:00,0,0.1\n00:00:01,1,0.1\n00:00:02,0,0.1\n"
                ),
            }
            paths = {}
            for name, content in contents.items():
                path = folder / f"{name}.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr(f"root/{key}", content)
                paths[name] = path
            with BatchDistillationArchive(
                paths["labels"], paths["sensors"], paths["actuators"], paths["uncertainties"]
            ) as dataset:
                run = dataset.load_run(key)
                joint = dataset.load_run(key, actuator_mode="features")
            self.assertEqual(run.x.shape, (3, 2))
            self.assertTrue(np.isnan(run.x[1]).all())
            self.assertTrue(run.observed_mask[0].all())
            self.assertFalse(run.observed_mask[1].any())
            np.testing.assert_array_equal(run.fault, [False, False, True])
            self.assertEqual(run.covariates.shape, (3, 4))
            self.assertEqual(run.sample_period_seconds, 1.0)
            self.assertEqual(joint.x.shape, (3, 3))
            self.assertEqual(joint.covariates.shape, (3, 3))
            self.assertTrue(np.isnan(joint.x[1]).all())

            # Elapsed-time covariates are prefix invariant and therefore do not
            # reveal the as-yet-unknown final batch length.
            np.testing.assert_allclose(phase_basis(3)[:2], phase_basis(8)[:2])

    def test_metropt_official_intervals_and_covariate_split(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name:
            path = Path(folder_name) / "metro.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": [
                        "2020-03-01 00:00:00",
                        "2020-03-01 00:00:10",
                        "2020-04-18 12:00:00",
                    ],
                    "TP2": [1, 2, 3],
                    "TP3": [1, 2, 3],
                    "H1": [1, 2, 3],
                    "DV_pressure": [1, 2, 3],
                    "Reservoirs": [1, 2, 3],
                    "Oil_temperature": [20, 21, 22],
                    "Motor_current": [4, 5, 6],
                    "COMP": [0, 1, 1],
                }
            )
            frame.to_csv(path, index=False)
            run = load_metropt(path, resample=None, analog_only=True)
            self.assertEqual(run.x.shape, (3, 7))
            self.assertEqual(run.covariates.shape, (3, 1))
            np.testing.assert_array_equal(run.fault, [False, False, True])

    def test_metropt_resampling_preserves_empty_bins_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name:
            path = Path(folder_name) / "metro.csv"
            frame = pd.DataFrame(
                {
                    "timestamp": [
                        "2020-03-01 00:00:05",
                        "2020-03-01 00:02:05",
                    ],
                    "TP2": [1.0, 3.0],
                    "TP3": [2.0, 4.0],
                    "H1": [1.0, 1.0],
                    "DV_pressure": [2.0, 2.0],
                    "Reservoirs": [3.0, 3.0],
                    "Oil_temperature": [20.0, 22.0],
                    "Motor_current": [4.0, 6.0],
                    "COMP": [0.0, 1.0],
                }
            )
            frame.to_csv(path, index=False)
            run = load_metropt(path, resample="1min", analog_only=True)
            self.assertEqual(run.x.shape, (3, 7))
            self.assertTrue(run.observed_mask[0].all())
            self.assertFalse(run.observed_mask[1].any())
            self.assertTrue(run.observed_mask[2].all())
            self.assertTrue(np.isfinite(run.x).all())
            np.testing.assert_array_equal(run.covariates[:, 0], [0.0, 0.0, 1.0])
            self.assertEqual(run.metadata["covariate_gap_policy"], "causal_forward_fill")
            self.assertEqual(run.sample_period_seconds, 60.0)

    def test_tep_transposed_training_format(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name:
            path = Path(folder_name) / "d00.dat"
            matrix = np.arange(52 * 30, dtype=float).reshape(52, 30)
            np.savetxt(path, matrix)
            run = load_tep(path.parent, fault_id=0, test=False)
            self.assertEqual(run.x.shape, (30, 41))
            self.assertEqual(run.covariates.shape, (30, 11))
            self.assertFalse(run.fault.any())


if __name__ == "__main__":
    unittest.main()
