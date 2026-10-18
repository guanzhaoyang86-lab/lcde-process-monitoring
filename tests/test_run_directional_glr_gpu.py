from __future__ import annotations

from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mate_spc.datasets import RIETH_FEATURE_COLUMNS, SensorSequence
from mate_spc_directional.glr import FAULT_IDS, DirectionalGLRPath


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_directional_glr_gpu.py"
)
GLOBALS = runpy.run_path(str(SCRIPT))


def _run(*, fault_id: int, run_id: int, length: int = 960) -> SensorSequence:
    fault = np.zeros(length, dtype=bool)
    if fault_id:
        fault[160:] = True
    partition = "faulty_testing" if fault_id else "fault_free_testing"
    return SensorSequence(
        name=f"synthetic-fault-{fault_id:02d}-run-{run_id:03d}",
        x=np.zeros((length, 52), dtype=float),
        fault=fault,
        observed_mask=np.ones((length, 52), dtype=bool),
        feature_names=tuple(RIETH_FEATURE_COLUMNS),
        metadata={
            "fault_id": fault_id,
            "simulation_run": run_id,
            "partition": partition,
        },
    )


def _path(envelope: np.ndarray) -> DirectionalGLRPath:
    length = envelope.size
    component = np.repeat(envelope[:, None], 3, axis=1)
    return DirectionalGLRPath(
        fault_ids=FAULT_IDS,
        logits=np.zeros((length, 3)),
        standardized_logits=np.zeros((length, 3)),
        clipped_evidence=np.zeros((length, 3)),
        cusum=np.zeros((length, 3)),
        component_persistence=component,
        envelope=envelope,
        eligible=np.ones(length, dtype=bool),
        alarm=envelope > 1.0,
        valid_start=0,
        threshold=1.0,
    )


class DirectionalRunnerTests(unittest.TestCase):
    def test_same_simulation_id_is_always_in_the_same_grouped_fold(self) -> None:
        fold_for_run = GLOBALS["_fold_for_run"]
        for run_id in range(1, 101):
            runs = tuple(_run(fault_id=fault_id, run_id=run_id, length=500) for fault_id in (
                0,
                3,
                9,
                15,
            ))
            self.assertEqual(
                {fold_for_run(run) for run in runs},
                {(run_id - 1) % 5},
            )

    def test_protocol_freezes_splits_features_optimizer_and_500_sample_horizon(
        self,
    ) -> None:
        protocol = GLOBALS["_development_protocol"](
            checksums={"partition": "digest"},
            source_hashes={"source.py": "digest"},
            cva_reference={"model_sha256": "digest"},
            batch_size=16,
        )
        transformer = protocol["method"]["feature_transformer"]
        classifier = protocol["method"]["classifier"]
        self.assertEqual(transformer["signed_residual_dimension"], 204)
        self.assertEqual(transformer["residual_windows_samples"], [1, 5, 20, 60])
        self.assertEqual(len(transformer["control_pairs_zero_based_xmv_xmeas"]), 12)
        self.assertEqual(transformer["raw_feature_count"], 1080)
        self.assertEqual(transformer["valid_start_zero_based"], 63)
        self.assertTrue(classifier["optimizer"]["require_convergence"])
        self.assertEqual(
            protocol["development_splits"]["development_truncation_half_open"],
            [0, 500],
        )
        self.assertEqual(
            protocol["predeclared_audit"]["all_sequence_truncation_half_open"],
            [0, 500],
        )
        self.assertEqual(protocol["development_splits"]["fold_ids"], list(range(5)))
        self.assertEqual(protocol["reserved"]["final_healthy_401_500"], (
            "do_not_select_score_or_inspect"
        ))

    def test_truncation_makes_a_miss_exactly_340_samples(self) -> None:
        run = GLOBALS["_truncate_run"](_run(fault_id=15, run_id=1), 500)
        self.assertEqual(run.x.shape, (500, 52))
        envelope = np.zeros(500, dtype=float)
        records = GLOBALS["_fault_records"](
            (run,),
            {run.name: _path(envelope)},
            threshold=1.0,
        )
        self.assertFalse(records[0]["detected"])
        self.assertEqual(records[0]["restricted_delay_samples_miss_at_run_end"], 340)

    def test_prefault_episode_continuing_after_onset_is_not_credited(self) -> None:
        run = GLOBALS["_truncate_run"](_run(fault_id=3, run_id=1), 500)
        envelope = np.zeros(500, dtype=float)
        envelope[150:180] = 2.0
        records = GLOBALS["_fault_records"](
            (run,),
            {run.name: _path(envelope)},
            threshold=1.0,
        )
        self.assertFalse(records[0]["detected"])
        self.assertEqual(records[0]["pre_fault_alarm_episode_count"], 1)

        envelope[220:225] = 2.0
        records = GLOBALS["_fault_records"](
            (run,),
            {run.name: _path(envelope)},
            threshold=1.0,
        )
        self.assertTrue(records[0]["detected"])
        self.assertEqual(records[0]["first_fault_alarm_episode_index"], 220)
        self.assertEqual(records[0]["delay_samples"], 60)

    def test_selection_key_uses_the_frozen_lexicographic_order(self) -> None:
        key = GLOBALS["_selection_key"]
        baseline = {"selection_key": [70, 220, -80.0, 0.001, 0.25]}
        more_balanced = {"selection_key": [71, 210, -100.0, 0.0001, 0.1]}
        faster = {"selection_key": [70, 220, -70.0, 0.001, 0.25]}
        larger_lambda = {"selection_key": [70, 220, -80.0, 0.01, 0.25]}
        larger_kappa = {"selection_key": [70, 220, -80.0, 0.001, 0.5]}
        self.assertGreater(key(more_balanced), key(baseline))
        self.assertGreater(key(faster), key(baseline))
        self.assertGreater(key(larger_lambda), key(baseline))
        self.assertGreater(key(larger_kappa), key(baseline))

    def test_all_three_audit_gates_are_required(self) -> None:
        health = [
            {"any_alarm": index < 9, "alarm_episode_count": int(index < 9)}
            for index in range(100)
        ]
        faults = []
        for fault_id in FAULT_IDS:
            for index in range(100):
                detected = index < 70
                faults.append(
                    {
                        "fault_id": fault_id,
                        "detected": detected,
                        "delay_samples": 20 if detected else None,
                        "restricted_delay_samples_miss_at_run_end": (
                            20 if detected else 340
                        ),
                        "pre_fault_alarm_episode_count": 0,
                        "attribution_matches_fault": detected,
                    }
                )
        passed = GLOBALS["_audit_summary"](health, faults)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["faults"]["overall"]["detected_runs"], 210)

        health[9]["any_alarm"] = True
        failed_health = GLOBALS["_audit_summary"](health, faults)
        self.assertFalse(failed_health["passed"])

        health[9]["any_alarm"] = False
        for index in range(59, 70):
            faults[index]["detected"] = False
            faults[index]["delay_samples"] = None
            faults[index]["restricted_delay_samples_miss_at_run_end"] = 340
        failed_fault = GLOBALS["_audit_summary"](health, faults)
        self.assertFalse(failed_fault["passed"])
        self.assertFalse(failed_fault["faults"]["per_fault_passed"])
        self.assertFalse(failed_fault["faults"]["total_passed"])

    def test_manifest_and_completion_detect_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / "development_manifest.json"
            protocol = {"schema_version": 1, "frozen": True}
            fingerprint = GLOBALS["_load_or_create_manifest"](
                manifest,
                protocol,
                require_empty_parent=True,
            )
            self.assertEqual(len(fingerprint), 64)
            self.assertEqual(
                GLOBALS["_load_or_create_manifest"](
                    manifest,
                    protocol,
                    require_empty_parent=True,
                ),
                fingerprint,
            )
            artifact = folder / "artifact.txt"
            artifact.write_text("frozen", encoding="utf-8")
            completion = folder / "completion.json"
            GLOBALS["_write_completion"](
                completion,
                folder,
                fingerprint,
                "COMPLETE",
                ("artifact.txt",),
            )
            self.assertIsNotNone(
                GLOBALS["_verify_completion"](completion, folder, fingerprint)
            )
            artifact.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                GLOBALS["_verify_completion"](completion, folder, fingerprint)

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                GLOBALS["_load_or_create_manifest"](
                    folder / "development_manifest.json",
                    {"schema_version": 1},
                    require_empty_parent=True,
                )


if __name__ == "__main__":
    unittest.main()
