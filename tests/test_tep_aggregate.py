from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from mate_spc.tep_aggregate import (
    aggregate_tep_fault_files,
    write_tep_aggregate,
    write_tep_markdown,
)


def _report(fault_id: int, detected: tuple[int, int]) -> dict[str, object]:
    records = []
    for index, seed in enumerate((11, 29)):
        methods = {}
        for method, offset in (("UD-MAPIC", 0), ("median", 1)):
            found = min(1, detected[index] + offset)
            methods[method] = {
                "monitoring": {
                    "events_detected": found,
                    "event_count": 1,
                    "median_delay_samples": 4.0 + index if found else None,
                    "false_alarm_episodes": index,
                    "monitored_normal_hours": 2.0,
                    "point_f1": 0.5 + 0.1 * index,
                    "decision_coverage": 0.9,
                },
                "unknown_rate": 0.1,
                "imputation": {
                    "rmse": 2.0 + index,
                    "n_hidden": 10.0,
                },
            }
        records.append(
            {
                "scenario_id": "mcar|rate=0.1",
                "scenario": {"mechanism": "mcar", "rate": 0.1},
                "seed": seed,
                "methods": methods,
            }
        )
    return {
        "method": "UD-MAPIC",
        "configuration": {"alpha": 0.05},
        "evaluation_protocol": {"warning_horizon_samples": 0},
        "train": {
            "name": "TEP-fault-00-train",
            "rows": 10,
            "features": 3,
            "fault_samples": 0,
            "sample_period_seconds": 180.0,
        },
        "calibration": {
            "name": "TEP-fault-00-test",
            "rows": 10,
            "features": 3,
            "fault_samples": 0,
            "sample_period_seconds": 180.0,
        },
        "test": {
            "name": f"TEP-fault-{fault_id:02d}-test",
            "rows": 10,
            "features": 3,
            "fault_samples": 5,
            "sample_period_seconds": 180.0,
        },
        "calibration_summary": {"threshold": 3.0, "counts": [4, 5]},
        "ablation_calibration_summaries": {
            "MAPIC_integrated_expectation": {"threshold": 3.5}
        },
        "records": records,
    }


class TEPAggregateTests(unittest.TestCase):
    def test_aggregate_validates_and_pools_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for fault_id, detected in ((1, (1, 0)), (2, (1, 1))):
                path = root / f"fault_{fault_id:02d}.json"
                path.write_text(json.dumps(_report(fault_id, detected)), encoding="utf-8")
                paths.append(path)
            report = aggregate_tep_fault_files(paths, expected_faults=(1, 2))
            self.assertEqual(report["fault_count"], 2)
            self.assertEqual(report["total_cases"], 4)
            ud = report["overall"]["UD-MAPIC"]
            self.assertEqual(ud["detected_event_evaluations"], 3)
            self.assertEqual(ud["event_evaluation_count"], 4)
            self.assertAlmostEqual(ud["event_evaluation_recall"], 0.75)
            self.assertAlmostEqual(ud["false_alarm_episodes_per_100h"], 25.0)
            self.assertAlmostEqual(
                ud["pooled_scale_normalized_imputation_rmse"],
                math.sqrt((4.0 + 9.0) / 2.0),
            )
            self.assertEqual(report["files"][0]["path"], "fault_01.json")
            json_path = root / "summary.json"
            csv_path = root / "summary.csv"
            markdown_path = root / "summary.md"
            write_tep_aggregate(report, json_path, csv_path)
            write_tep_markdown(report, markdown_path)
            self.assertEqual(json.loads(json_path.read_text())["fault_count"], 2)
            self.assertEqual(len(csv_path.read_text().splitlines()), 5)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("faults 1–2 missingness stress report", markdown)
            self.assertIn("UD-MAPIC by fault", markdown)
            self.assertIn("| 01 | 1/2 |", markdown)
            self.assertIn(r"mcar\|rate=0.1", markdown)
            self.assertIn("Median delay among detections", markdown)

    def test_material_calibration_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _report(1, (1, 1))
            second = _report(2, (1, 1))
            second_summary = second["calibration_summary"]
            assert isinstance(second_summary, dict)
            second_summary["threshold"] = 3.01
            paths = []
            for fault_id, value in ((1, first), (2, second)):
                path = root / f"fault_{fault_id:02d}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(
                ValueError, "fitted calibration mismatch"
            ):
                aggregate_tep_fault_files(paths, expected_faults=(1, 2))

    def test_roundoff_level_calibration_difference_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _report(1, (1, 1))
            second = _report(2, (1, 1))
            second_summary = second["calibration_summary"]
            assert isinstance(second_summary, dict)
            second_summary["threshold"] = 3.0 + 1e-11
            paths = []
            for fault_id, value in ((1, first), (2, second)):
                path = root / f"fault_{fault_id:02d}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            aggregate_tep_fault_files(paths, expected_faults=(1, 2))

    def test_missing_fault_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fault_01.json"
            path.write_text(json.dumps(_report(1, (1, 1))), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fault coverage mismatch"):
                aggregate_tep_fault_files([path], expected_faults=(1, 2))


if __name__ == "__main__":
    unittest.main()
