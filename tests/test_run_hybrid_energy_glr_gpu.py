from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mate_spc.datasets import RIETH_FEATURE_COLUMNS, SensorSequence
from mate_spc_directional.glr import (
    FAULT_IDS,
    DirectionalGLRPath,
    conformal_run_threshold,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_hybrid_energy_glr_gpu.py"
)
GLOBALS = runpy.run_path(str(SCRIPT))
BASE = GLOBALS["base"]


def _run(*, fault_id: int, run_id: int, length: int = 960) -> SensorSequence:
    fault = np.zeros(length, dtype=bool)
    if fault_id:
        fault[160:] = True
    partition = "faulty_testing" if fault_id else "fault_free_testing"
    return SensorSequence(
        name=f"hybrid-fault-{fault_id:02d}-run-{run_id:03d}",
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


def _health_records(alarm_count: int) -> list[dict[str, object]]:
    return [
        {
            "any_alarm": index < alarm_count,
            "alarm_episode_count": int(index < alarm_count),
        }
        for index in range(100)
    ]


def _fault_records(counts: tuple[int, int, int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fault_id, count in zip(FAULT_IDS, counts):
        for index in range(100):
            detected = index < count
            records.append(
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
    return records


class HybridEnergyGLRRunnerTests(unittest.TestCase):
    def test_same_simulation_id_puts_all_four_classes_in_one_fold(self) -> None:
        for run_id in range(1, 201):
            runs = tuple(
                _run(fault_id=fault_id, run_id=run_id, length=500)
                for fault_id in (0, *FAULT_IDS)
            )
            self.assertEqual(
                {BASE._fold_for_run(run) for run in runs},
                {(run_id - 1) % 5},
            )

    def test_development_uses_runs_1_to_200_and_truncates_at_500(self) -> None:
        protocol = GLOBALS["_development_protocol"](
            checksums={"partition": "digest"},
            source_hashes={"source.py": "digest"},
            directional_reference={"transformer_sha256": "digest"},
            batch_size=16,
        )
        splits = protocol["development_splits"]
        self.assertEqual(GLOBALS["DEVELOPMENT_RUNS"], tuple(range(1, 201)))
        self.assertEqual(splits["classifier_normal"], [
            "fault_free_testing",
            list(range(1, 201)),
        ])
        self.assertEqual(splits["classifier_faults"][2], list(range(1, 201)))
        self.assertEqual(splits["all_test_sequences_truncated_half_open"], [0, 500])
        self.assertEqual(splits["per_fold_per_class_runs"], 40)

        truncated = BASE._truncate_run(_run(fault_id=15, run_id=200), 500)
        self.assertEqual(truncated.x.shape, (500, 52))
        self.assertEqual(truncated.fault.shape, (500,))

    def test_two_hundred_oof_healthy_runs_use_finite_sample_q191(self) -> None:
        envelopes = {
            f"oof-normal-{index:03d}": np.array([-np.inf, float(index)])
            for index in range(1, 201)
        }
        threshold, metadata = conformal_run_threshold(
            envelopes,
            alpha=0.05,
            required_count=200,
        )
        self.assertEqual(threshold, 191.0)
        self.assertEqual(metadata["rank_one_based"], 191)
        self.assertEqual(metadata["rank_driver_run"], "oof-normal-191")
        self.assertEqual(GLOBALS["DEVELOPMENT_RANK"], 191)

    def test_selection_order_is_min_class_fault9_total_delay_lambda_kappa(self) -> None:
        key = GLOBALS["_selection_key"]
        baseline = {"selection_key": [70, 100, 220, -80.0, 0.01, 0.75]}
        alternatives = (
            {"selection_key": [71, 0, 0, -340.0, 0.001, 0.5]},
            {"selection_key": [70, 101, 0, -340.0, 0.001, 0.5]},
            {"selection_key": [70, 100, 221, -340.0, 0.001, 0.5]},
            {"selection_key": [70, 100, 220, -79.0, 0.001, 0.5]},
            {"selection_key": [70, 100, 220, -80.0, 0.1, 0.5]},
            {"selection_key": [70, 100, 220, -80.0, 0.01, 1.0]},
        )
        for alternative in alternatives:
            self.assertGreater(key(alternative), key(baseline))
        self.assertLess(
            key({"selection_key": [69, 999, 999, 0.0, 999.0, 999.0]}),
            key(baseline),
        )

        protocol = GLOBALS["_development_protocol"](
            checksums={},
            source_hashes={},
            directional_reference={},
            batch_size=8,
        )
        self.assertEqual(
            protocol["development_selection"][
                "selection_key_maximized_lexicographically"
            ],
            [
                "minimum_detected_runs_across_fault_classes",
                "fault_9_detected_runs",
                "total_detected_fault_runs",
                "negative_overall_restricted_mean_delay_miss_as_340",
                "larger_ridge_lambda",
                "larger_kappa",
            ],
        )

    def test_miss_is_340_and_prefault_episode_is_not_detection(self) -> None:
        run = BASE._truncate_run(_run(fault_id=9, run_id=1), 500)
        no_alarm = np.zeros(500)
        records = BASE._fault_records(
            (run,),
            {run.name: _path(no_alarm)},
            threshold=1.0,
        )
        self.assertFalse(records[0]["detected"])
        self.assertEqual(records[0]["restricted_delay_samples_miss_at_run_end"], 340)

        prefault = np.zeros(500)
        prefault[150:180] = 2.0
        records = BASE._fault_records(
            (run,),
            {run.name: _path(prefault)},
            threshold=1.0,
        )
        self.assertFalse(records[0]["detected"])
        self.assertEqual(records[0]["pre_fault_alarm_episode_count"], 1)

        prefault[220:225] = 2.0
        records = BASE._fault_records(
            (run,),
            {run.name: _path(prefault)},
            threshold=1.0,
        )
        self.assertTrue(records[0]["detected"])
        self.assertEqual(records[0]["first_fault_alarm_episode_index"], 220)
        self.assertEqual(records[0]["delay_samples"], 60)

    def test_audit_requires_health_per_class_and_total_gates_together(self) -> None:
        passing = BASE._audit_summary(_health_records(9), _fault_records((70, 70, 70)))
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["faults"]["overall"]["detected_runs"], 210)

        health_failure = BASE._audit_summary(
            _health_records(10),
            _fault_records((70, 70, 70)),
        )
        self.assertFalse(health_failure["passed"])
        self.assertFalse(health_failure["healthy"]["passed"])
        self.assertTrue(health_failure["faults"]["per_fault_passed"])
        self.assertTrue(health_failure["faults"]["total_passed"])

        class_failure = BASE._audit_summary(
            _health_records(9),
            _fault_records((59, 80, 80)),
        )
        self.assertFalse(class_failure["passed"])
        self.assertFalse(class_failure["faults"]["per_fault_passed"])
        self.assertTrue(class_failure["faults"]["total_passed"])

        total_failure = BASE._audit_summary(
            _health_records(9),
            _fault_records((69, 70, 70)),
        )
        self.assertFalse(total_failure["passed"])
        self.assertTrue(total_failure["faults"]["per_fault_passed"])
        self.assertFalse(total_failure["faults"]["total_passed"])

    def test_manifest_and_completion_reject_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            manifest = folder / "development_manifest.json"
            protocol = {"schema_version": 1, "frozen": True}
            fingerprint = BASE._load_or_create_manifest(
                manifest,
                protocol,
                require_empty_parent=True,
            )
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            stored["frozen"] = False
            manifest.write_text(json.dumps(stored), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint is corrupt"):
                BASE._load_or_create_manifest(
                    manifest,
                    protocol,
                    require_empty_parent=True,
                )

        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            artifact = folder / "artifact.txt"
            artifact.write_text("frozen", encoding="utf-8")
            completion = folder / "completion.json"
            BASE._write_completion(
                completion,
                folder,
                "a" * 64,
                "COMPLETE",
                ("artifact.txt",),
            )
            self.assertIsNotNone(
                BASE._verify_completion(completion, folder, "a" * 64)
            )
            artifact.write_text("tamper", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                BASE._verify_completion(completion, folder, "a" * 64)

    def test_final_healthy_and_fault_runs_401_to_500_are_never_selected(self) -> None:
        protocol = GLOBALS["_development_protocol"](
            checksums={},
            source_hashes={},
            directional_reference={},
            batch_size=8,
        )
        self.assertEqual(GLOBALS["FINAL_RUNS"], tuple(range(401, 501)))
        self.assertEqual(
            protocol["reserved"]["final_healthy"],
            [list(range(401, 501)), "do_not_select_score_or_inspect"],
        )
        self.assertEqual(
            protocol["reserved"]["final_all_faults"],
            [list(range(401, 501)), "do_not_select_score_or_inspect"],
        )

        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        load_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_runs"
        ]
        self.assertEqual(len(load_calls), 6)
        selected_split_names: set[str] = set()
        for call in load_calls:
            names = {
                node.id for node in ast.walk(call) if isinstance(node, ast.Name)
            }
            self.assertNotIn("FINAL_RUNS", names)
            self.assertNotIn("FAULT_RUNS_FORBIDDEN_AFTER_FAILURE", names)
            selected_split_names.update(
                names
                & {
                    "REFERENCE_HEALTH_RUNS",
                    "DEVELOPMENT_RUNS",
                    "CALIBRATION_RUNS",
                    "AUDIT_RUNS",
                }
            )
        self.assertEqual(
            selected_split_names,
            {
                "REFERENCE_HEALTH_RUNS",
                "DEVELOPMENT_RUNS",
                "CALIBRATION_RUNS",
                "AUDIT_RUNS",
            },
        )


if __name__ == "__main__":
    unittest.main()
