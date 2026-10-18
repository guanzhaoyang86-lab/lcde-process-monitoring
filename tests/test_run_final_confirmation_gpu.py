from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from mate_spc.datasets import RIETH_FEATURE_COLUMNS, SensorSequence
from mate_spc_directional.glr import FAULT_IDS, DirectionalGLRPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_final_confirmation_gpu.py"
PARENT_ROOT = PROJECT_ROOT / "outputs" / "hybrid_energy_glr_gpu_v1"
GLOBALS = runpy.run_path(str(SCRIPT))
BASE = GLOBALS["base"]
RUNTIME_GLOBALS = GLOBALS["_run_final"].__globals__


def _run(*, fault_id: int, run_id: int, length: int = 960) -> SensorSequence:
    fault = np.zeros(length, dtype=bool)
    if fault_id:
        fault[160:] = True
    partition = "faulty_testing" if fault_id else "fault_free_testing"
    return SensorSequence(
        name=f"final-fault-{fault_id:02d}-run-{run_id:03d}",
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
    component = np.repeat(envelope[:, None], len(FAULT_IDS), axis=1)
    return DirectionalGLRPath(
        fault_ids=FAULT_IDS,
        logits=np.zeros((length, len(FAULT_IDS))),
        standardized_logits=np.zeros((length, len(FAULT_IDS))),
        clipped_evidence=np.zeros((length, len(FAULT_IDS))),
        cusum=np.zeros((length, len(FAULT_IDS))),
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


def _minimal_parent() -> dict[str, object]:
    return {
        "completion_path": "parent/completion.json",
        "completion_sha256": "a" * 64,
        "completion_fingerprint": "b" * 64,
        "phase_fingerprint": "c" * 64,
        "artifacts": {},
    }


def _parent_for_run(dataset_md5: dict[str, str]) -> dict[str, object]:
    return {
        **_minimal_parent(),
        "audit_base": {"dataset_md5": dataset_md5},
    }


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _clone_parent_with_links(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(source / "completion.json", destination / "completion.json")
    copied_name = "hybrid_transformer.npz"
    for name in GLOBALS["PARENT_ARTIFACTS"]:
        source_path = source / name
        destination_path = destination / name
        if name == copied_name:
            shutil.copy2(source_path, destination_path)
            continue
        try:
            os.link(source_path, destination_path)
        except OSError:
            shutil.copy2(source_path, destination_path)


class FinalConfirmationRunnerTests(unittest.TestCase):
    def test_protocol_freezes_only_runs_401_to_500_and_parent_parameters(self) -> None:
        protocol = GLOBALS["_final_protocol"](
            parent=_minimal_parent(),
            checksums={"fault_free_testing": "digest"},
            source_hashes={"source.py": "digest"},
            batch_size=16,
        )
        self.assertEqual(GLOBALS["FINAL_RUNS"], tuple(range(401, 501)))
        self.assertEqual(GLOBALS["SEQUENCE_STOP"], 500)
        self.assertEqual(
            protocol["final_split"]["healthy"],
            ["fault_free_testing", list(range(401, 501))],
        )
        self.assertEqual(
            protocol["final_split"]["faults"],
            ["faulty_testing", list(FAULT_IDS), list(range(401, 501))],
        )
        self.assertEqual(
            protocol["final_split"]["sequence_truncation_half_open"],
            [0, 500],
        )
        frozen = protocol["frozen_method"]
        self.assertEqual(frozen["ridge_lambda"], 0.001)
        self.assertEqual(frozen["kappa_per_head"], [0.75, 0.75, 0.75])
        self.assertEqual(frozen["threshold"], 165.05745028203916)
        self.assertEqual(frozen["comparison"], "strictly_greater_than")
        self.assertEqual(
            protocol["dataset"]["dataset_md5"],
            {"fault_free_testing": "digest"},
        )
        self.assertTrue(all(protocol["prohibitions"].values()))
        self.assertNotIn("development_split", protocol)
        self.assertNotIn("training_split", protocol)
        self.assertNotIn("calibration_split", protocol)

    def test_ast_loads_only_final_runs_after_manifest_and_never_fits(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        run_final = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_run_final"
        )
        calls = [node for node in ast.walk(run_final) if isinstance(node, ast.Call)]
        load_calls = [node for node in calls if _call_name(node) == "load_runs"]
        self.assertEqual(len(load_calls), 2)
        forbidden_split_names = {
            "REFERENCE_HEALTH_RUNS",
            "DEVELOPMENT_RUNS",
            "CALIBRATION_RUNS",
            "AUDIT_RUNS",
            "FAULT_RUNS_FORBIDDEN_AFTER_FAILURE",
        }
        for call in load_calls:
            names = {
                item.id for item in ast.walk(call) if isinstance(item, ast.Name)
            }
            self.assertIn("FINAL_RUNS", names)
            self.assertTrue(forbidden_split_names.isdisjoint(names))

        manifest_calls = [
            node for node in calls if _call_name(node) == "_load_or_create_manifest"
        ]
        self.assertEqual(len(manifest_calls), 1)
        self.assertLess(manifest_calls[0].lineno, min(node.lineno for node in load_calls))
        completed_calls = [
            node for node in calls if _call_name(node) == "_verify_final_completion"
        ]
        self.assertGreaterEqual(len(completed_calls), 2)
        self.assertLess(completed_calls[0].lineno, min(node.lineno for node in load_calls))
        archive_calls = [node for node in calls if _call_name(node) == "RiethTEPArchive"]
        verify_calls = [node for node in calls if _call_name(node) == "verify"]
        self.assertEqual(len(archive_calls), 1)
        self.assertEqual(len(verify_calls), 1)
        self.assertLess(manifest_calls[0].lineno, archive_calls[0].lineno)
        self.assertLess(manifest_calls[0].lineno, verify_calls[0].lineno)
        self.assertLess(verify_calls[0].lineno, min(node.lineno for node in load_calls))

        forbidden_calls = {
            "fit",
            "fit_directional_glr_cuda",
            "fit_directional_glr_grid_cuda",
            "conformal_run_threshold",
            "calibrate",
        }
        self.assertTrue(forbidden_calls.isdisjoint(filter(None, map(_call_name, calls))))

    def test_explicit_parent_identity_anchors_match_frozen_files(self) -> None:
        completion_path = PARENT_ROOT / "completion.json"
        transformer_path = PARENT_ROOT / "hybrid_transformer.npz"
        model_path = PARENT_ROOT / "hybrid_glr_calibrated.npz"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        self.assertEqual(
            BASE._sha256_file(completion_path),
            GLOBALS["EXPECTED_PARENT_COMPLETION_SHA256"],
        )
        self.assertEqual(
            completion["completion_fingerprint"],
            GLOBALS["EXPECTED_PARENT_COMPLETION_FINGERPRINT"],
        )
        self.assertEqual(
            completion["phase_fingerprint"],
            GLOBALS["EXPECTED_PARENT_PHASE_FINGERPRINT"],
        )
        self.assertEqual(
            BASE._sha256_file(transformer_path),
            GLOBALS["EXPECTED_TRANSFORMER_SHA256"],
        )
        self.assertEqual(
            transformer_path.stat().st_size,
            GLOBALS["EXPECTED_TRANSFORMER_SIZE"],
        )
        self.assertEqual(
            BASE._sha256_file(model_path),
            GLOBALS["EXPECTED_CALIBRATED_MODEL_SHA256"],
        )
        self.assertEqual(
            model_path.stat().st_size,
            GLOBALS["EXPECTED_CALIBRATED_MODEL_SIZE"],
        )

    def test_completed_reentry_performs_no_archive_construction_or_access(self) -> None:
        dataset_md5 = {
            "fault_free_testing": "healthy-md5",
            "faulty_testing": "fault-md5",
        }
        completed = {"decision": GLOBALS["PASS_DECISION"]}
        archive = mock.Mock(side_effect=AssertionError("archive must stay unopened"))
        load_method = mock.Mock(side_effect=AssertionError("model must stay unloaded"))
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            args = SimpleNamespace(
                output_dir=output_dir,
                data_dir=Path("unused-data"),
                batch_size=8,
                device="cuda",
            )
            with mock.patch.dict(
                RUNTIME_GLOBALS,
                {
                    "_verify_parent_completion": mock.Mock(
                        return_value=_parent_for_run(dataset_md5)
                    ),
                    "_source_hashes": mock.Mock(return_value={}),
                    "_verify_final_completion": mock.Mock(return_value=completed),
                    "_load_and_validate_frozen_method": load_method,
                    "RiethTEPArchive": archive,
                },
            ):
                observed = GLOBALS["_run_final"](
                    args,
                    project_root=PROJECT_ROOT,
                    hybrid_dir=PARENT_ROOT,
                )
            self.assertIs(observed, completed)
            archive.assert_not_called()
            load_method.assert_not_called()
            manifest = json.loads(
                (output_dir / "final_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset"]["dataset_md5"], dataset_md5)

    def test_new_run_freezes_manifest_before_verify_and_load(self) -> None:
        dataset_md5 = {
            "fault_free_testing": "healthy-md5",
            "faulty_testing": "fault-md5",
        }
        events: list[str] = []
        original_manifest = BASE._load_or_create_manifest

        def record_manifest(*args, **kwargs):
            events.append("manifest")
            return original_manifest(*args, **kwargs)

        class StopAtLoad(RuntimeError):
            pass

        class FakeArchive:
            def __init__(self, *_args, **_kwargs) -> None:
                events.append("archive_constructed")

            def verify(self, _partitions):
                events.append("archive_verified")
                return dict(dataset_md5)

            def load_runs(self, *_args, **_kwargs):
                events.append("load_runs")
                raise StopAtLoad

        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            args = SimpleNamespace(
                output_dir=output_dir,
                data_dir=Path("fake-data"),
                batch_size=8,
                device="cuda",
            )
            with mock.patch.object(
                BASE,
                "_load_or_create_manifest",
                side_effect=record_manifest,
            ), mock.patch.dict(
                RUNTIME_GLOBALS,
                {
                    "_verify_parent_completion": mock.Mock(
                        return_value=_parent_for_run(dataset_md5)
                    ),
                    "_source_hashes": mock.Mock(return_value={}),
                    "_verify_final_completion": mock.Mock(return_value=None),
                    "_load_and_validate_frozen_method": mock.Mock(
                        return_value=(None, None, None, None)
                    ),
                    "RiethTEPArchive": FakeArchive,
                },
            ):
                with self.assertRaises(StopAtLoad):
                    GLOBALS["_run_final"](
                        args,
                        project_root=PROJECT_ROOT,
                        hybrid_dir=PARENT_ROOT,
                    )
            self.assertEqual(
                events,
                ["manifest", "archive_constructed", "archive_verified", "load_runs"],
            )
            self.assertTrue((output_dir / "final_manifest.json").is_file())

    def test_dataset_checksum_mismatch_fails_before_any_load_runs(self) -> None:
        dataset_md5 = {
            "fault_free_testing": "healthy-md5",
            "faulty_testing": "fault-md5",
        }
        load_runs = mock.Mock(side_effect=AssertionError("must not load mismatched data"))

        class MismatchedArchive:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def verify(self, _partitions):
                return {
                    "fault_free_testing": "wrong",
                    "faulty_testing": "wrong",
                }

            def load_runs(self, *args, **kwargs):
                return load_runs(*args, **kwargs)

        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            args = SimpleNamespace(
                output_dir=output_dir,
                data_dir=Path("fake-data"),
                batch_size=8,
                device="cuda",
            )
            with mock.patch.dict(
                RUNTIME_GLOBALS,
                {
                    "_verify_parent_completion": mock.Mock(
                        return_value=_parent_for_run(dataset_md5)
                    ),
                    "_source_hashes": mock.Mock(return_value={}),
                    "_verify_final_completion": mock.Mock(return_value=None),
                    "_load_and_validate_frozen_method": mock.Mock(
                        return_value=(None, None, None, None)
                    ),
                    "RiethTEPArchive": MismatchedArchive,
                },
            ):
                with self.assertRaisesRegex(ValueError, "MD5"):
                    GLOBALS["_run_final"](
                        args,
                        project_root=PROJECT_ROOT,
                        hybrid_dir=PARENT_ROOT,
                    )
            load_runs.assert_not_called()
            manifest = json.loads(
                (output_dir / "final_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset"]["dataset_md5"], dataset_md5)

    def test_all_three_frozen_gates_are_required_and_decisions_are_final(self) -> None:
        decision = GLOBALS["_decision"]
        self.assertEqual(
            decision(9, (60, 60, 90), 210),
            "PASS_FINAL_CONFIRMATION",
        )
        self.assertEqual(
            decision(10, (100, 100, 100), 300),
            "FAIL_FINAL_CONFIRMATION_NO_RETUNING",
        )
        self.assertEqual(
            decision(9, (59, 100, 100), 259),
            "FAIL_FINAL_CONFIRMATION_NO_RETUNING",
        )
        self.assertEqual(
            decision(9, (70, 70, 69), 209),
            "FAIL_FINAL_CONFIRMATION_NO_RETUNING",
        )
        with self.assertRaisesRegex(ValueError, "every fault head"):
            decision(9, (60, 60), 210)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            decision(-1, (60, 60, 90), 210)

        passing = GLOBALS["_final_summary"](
            _health_records(9),
            _fault_records((60, 60, 90)),
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["decision"], "PASS_FINAL_CONFIRMATION")
        failing = GLOBALS["_final_summary"](
            _health_records(10),
            _fault_records((100, 100, 100)),
        )
        self.assertFalse(failing["passed"])
        self.assertEqual(
            failing["decision"],
            "FAIL_FINAL_CONFIRMATION_NO_RETUNING",
        )

    def test_miss_is_340_and_prefault_episode_is_not_detection(self) -> None:
        run = BASE._truncate_run(_run(fault_id=9, run_id=401), 500)
        no_alarm = np.zeros(500)
        record = BASE._fault_records(
            (run,),
            {run.name: _path(no_alarm)},
            threshold=1.0,
        )[0]
        self.assertFalse(record["detected"])
        self.assertEqual(record["restricted_delay_samples_miss_at_run_end"], 340)

        prefault = np.zeros(500)
        prefault[150:180] = 2.0
        record = BASE._fault_records(
            (run,),
            {run.name: _path(prefault)},
            threshold=1.0,
        )[0]
        self.assertFalse(record["detected"])
        self.assertEqual(record["pre_fault_alarm_episode_count"], 1)

        prefault[220:225] = 2.0
        record = BASE._fault_records(
            (run,),
            {run.name: _path(prefault)},
            threshold=1.0,
        )[0]
        self.assertTrue(record["detected"])
        self.assertEqual(record["first_fault_alarm_episode_index"], 220)
        self.assertEqual(record["delay_samples"], 60)

    def test_report_accepts_summary_fields_and_records_frozen_statistics(self) -> None:
        result = GLOBALS["_final_summary"](
            _health_records(9),
            _fault_records((100, 100, 100)),
        )
        execution = {
            "cuda_validation": {"cpu_cuda_alarm_exact": True},
            "scoring": {"device_name": "test CUDA device"},
        }
        report = GLOBALS["_report"](result, execution)
        self.assertIn("PASS_FINAL_CONFIRMATION", report)
        self.assertIn("9/100", report)
        self.assertIn("300/300", report)
        self.assertIn("165.057450282039", report)
        self.assertIn("No fit, calibration, threshold selection", report)

    def test_parent_completion_and_artifact_hash_tampering_are_rejected(self) -> None:
        self.assertTrue((PARENT_ROOT / "completion.json").is_file())
        parent = GLOBALS["_verify_parent_completion"](PARENT_ROOT)
        self.assertEqual(parent["completion"]["decision"], GLOBALS["PARENT_DECISION"])
        self.assertEqual(
            parent["aggregate"]["selected_hyperparameters"]["threshold"],
            GLOBALS["EXPECTED_THRESHOLD"],
        )

        with TemporaryDirectory() as temporary:
            clone = Path(temporary) / "parent"
            _clone_parent_with_links(PARENT_ROOT, clone)
            completion_path = clone / "completion.json"
            body = json.loads(completion_path.read_text(encoding="utf-8"))
            body["decision"] = "tampered"
            completion_path.write_text(json.dumps(body), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint|corrupt"):
                GLOBALS["_verify_parent_completion"](clone)

            shutil.copy2(PARENT_ROOT / "completion.json", completion_path)
            with (clone / "hybrid_transformer.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash changed|SHA-256|size changed"):
                GLOBALS["_verify_parent_completion"](clone)

    def test_final_completion_tampering_and_partial_reentry_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "final_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            GLOBALS["_refuse_unsafe_partial_reentry"](root)
            extra = root / "partial.txt"
            extra.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "incomplete|do not silently"):
                GLOBALS["_refuse_unsafe_partial_reentry"](root)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in GLOBALS["FINAL_ARTIFACTS"]:
                (root / name).write_text("frozen", encoding="utf-8")
            fingerprint = "d" * 64
            BASE._write_completion(
                root / "completion.json",
                root,
                fingerprint,
                GLOBALS["PASS_DECISION"],
                GLOBALS["FINAL_ARTIFACTS"],
            )
            (root / "REPORT.md").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                GLOBALS["_verify_final_completion"](root, fingerprint)


if __name__ == "__main__":
    unittest.main()
