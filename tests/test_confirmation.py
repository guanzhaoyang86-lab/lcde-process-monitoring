from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mate_spc.baselines import _mask_only_context_features
from mate_spc.cli import (
    BATCH_NAMES,
    BATCH_V112_MD5,
    _validate_batch_op028_v1,
    _verify_batch_archives,
    _write_json,
    build_parser,
)
from mate_spc.confirmation import (
    _aggregate_visible_results,
    _validate_aligned,
    run_batch_confirmation,
    temporal_confirmation_split,
)
from mate_spc.datasets import SensorSequence
from mate_spc.missingness import MissingnessConfig
from mate_spc.model import MAPICConfig
from mate_spc.monitor import PosteriorIntegratedMonitor
from mate_spc.synthetic import make_demo_data


class ConfirmationProtocolTests(unittest.TestCase):
    def test_temporal_split_preserves_original_time_and_aligned_metadata(self) -> None:
        n_rows = 100
        released = np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2)
        observed = np.ones_like(released, dtype=bool)
        observed[[11, 70]] = False
        x = released.copy()
        x[~observed] = np.nan
        sequence = SensorSequence(
            name="normal",
            x=x,
            fault=np.zeros(n_rows, dtype=bool),
            timestamps=np.arange(n_rows).astype("timedelta64[s]"),
            observed_mask=observed,
            covariates=np.arange(n_rows, dtype=float)[:, None],
            feature_names=("x1", "x2"),
            covariate_names=("elapsed",),
            metadata={
                "released_values": released,
                "visible_fault": np.zeros(n_rows, dtype=bool),
                "scalar_note": "preserved",
            },
        )
        train, calibration, protocol = temporal_confirmation_split(
            sequence, train_fraction=0.6, purge_samples=10
        )
        self.assertEqual(train.x.shape[0], 54)
        self.assertEqual(calibration.x.shape[0], 36)
        self.assertEqual(protocol["purge_start"], 54)
        self.assertEqual(protocol["calibration_start"], 64)
        self.assertEqual(calibration.covariates[0, 0], 64.0)
        self.assertEqual(calibration.timestamps[0], np.timedelta64(64, "s"))
        np.testing.assert_array_equal(
            calibration.metadata["released_values"], released[64:]
        )
        self.assertEqual(calibration.metadata["scalar_note"], "preserved")

    def test_confirmation_parser_freezes_recommended_defaults(self) -> None:
        args = build_parser().parse_args(["batch-confirm", "data"])
        self.assertEqual(args.design_fraction, 0.65)
        self.assertEqual(args.purge_minutes, 10.0)
        self.assertEqual(args.block_minutes, 5.0)
        self.assertEqual(args.alpha, 0.05)
        _validate_batch_op028_v1(args)

        changed = build_parser().parse_args(
            ["batch-confirm", "data", "--alpha", "0.1"]
        )
        with self.assertRaisesRegex(ValueError, "is frozen"):
            _validate_batch_op028_v1(changed)

    def test_batch_archive_checksums_are_computed_not_asserted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            expected: dict[str, str] = {}
            for name, filename in BATCH_NAMES.items():
                payload = f"archive-{name}".encode()
                (folder / filename).write_bytes(payload)
                expected[name] = hashlib.md5(payload).hexdigest()
            with patch.dict(BATCH_V112_MD5, expected, clear=True):
                self.assertEqual(_verify_batch_archives(folder), expected)
                (folder / BATCH_NAMES["labels"]).write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    _verify_batch_archives(folder)

    def test_elapsed_quality_context_accepts_original_slice_origin(self) -> None:
        train, _, _ = make_demo_data(90, 80, 70, dimension=3)
        monitor = PosteriorIntegratedMonitor().fit(
            train.x,
            train.covariates,
        )
        elapsed_start = 37
        scored = monitor.score(
            train.x[:12],
            exogenous=train.covariates[:12],
            elapsed_start=elapsed_start,
        )
        np.testing.assert_allclose(
            scored.context_features[:, 4],
            np.log1p(np.arange(elapsed_start, elapsed_start + 12)),
        )
        mask_context = _mask_only_context_features(
            np.ones((12, 3), dtype=bool), elapsed_start
        )
        np.testing.assert_allclose(
            mask_context[:, 2],
            np.log1p(np.arange(elapsed_start, elapsed_start + 12)),
        )

    def test_confirmation_runner_fits_once_and_returns_json_ready_structure(self) -> None:
        normal, _, test = make_demo_data(120, 80, 70, dimension=3)
        normal = replace(
            normal,
            metadata={**normal.metadata, "released_values": normal.x.copy()},
        )
        config = MAPICConfig(
            alpha=0.5,
            block_size=8,
            min_calibration_blocks=2,
            calibration_repeats=1,
            scenarios=(MissingnessConfig("mcar", 0.1),),
            regime_max_depth=1,
            seed=4,
        )
        report = run_batch_confirmation(
            normal,
            (test,),
            config,
            train_fraction=0.6,
            purge_samples=4,
            baseline_methods=("kalman_mean", "linear_offline"),
        )
        self.assertIsInstance(report["configuration"], dict)
        run = next(iter(report["test_runs"].values()))
        self.assertEqual(
            set(run["methods"]),
            {
                "UD-MAPIC",
                "MAPIC_integrated_expectation",
                "UD-MAPIC_mask_only_context",
                "kalman_mean",
                "linear_offline",
            },
        )
        self.assertIn("UD-MAPIC", report["aggregate_visible_effect"])
        self.assertEqual(
            report["elapsed_context"]["calibration_start_sample"],
            report["split"]["calibration_start"],
        )
        for method_report in run["methods"].values():
            self.assertIn("non_normal_episode_monitoring", method_report)
            self.assertNotIn("cause_present_monitoring", method_report)
        self.assertTrue(run["methods"]["UD-MAPIC"]["causal"])
        self.assertFalse(run["methods"]["linear_offline"]["causal"])
        self.assertFalse(report["method_metadata"]["linear_offline"]["causal"])

    def test_confirmation_rejects_duplicate_or_normal_test_names(self) -> None:
        normal, _, test = make_demo_data(80, 80, 70, dimension=3)
        duplicate = replace(test)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            _validate_aligned(normal, (test, duplicate))
        collision = replace(test, name=normal.name)
        with self.assertRaisesRegex(ValueError, "normal run"):
            _validate_aligned(normal, (collision,))

    def test_zero_denominator_aggregate_serializes_as_null(self) -> None:
        evaluations = {
            "all_unknown": {
                "methods": {
                    "UD-MAPIC": {
                        "visible_effect_monitoring": {
                            "event_count": 0,
                            "events_detected": 0,
                            "false_alarm_episodes": 0,
                            "monitored_normal_hours": 0.0,
                        },
                        "unknown_rate": 1.0,
                    }
                }
            }
        }
        aggregate = _aggregate_visible_results(evaluations, {"all_unknown": 5})
        result = aggregate["UD-MAPIC"]
        self.assertTrue(np.isnan(result["event_recall"]))
        self.assertTrue(np.isnan(result["false_alarm_episodes_per_100h"]))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            _write_json({"aggregate": aggregate}, output)
            serialized = json.loads(output.read_text(encoding="utf-8"))
        self.assertIsNone(serialized["aggregate"]["UD-MAPIC"]["event_recall"])
        self.assertIsNone(
            serialized["aggregate"]["UD-MAPIC"]["false_alarm_episodes_per_100h"]
        )


if __name__ == "__main__":
    unittest.main()
