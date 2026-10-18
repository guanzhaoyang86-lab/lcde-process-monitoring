from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "audit_hydraulic_labels.py"
GLOBALS = runpy.run_path(str(SCRIPT))


def _protocol() -> dict[str, object]:
    return {
        "schema": "hydraulic_ws_pac_acquisition_protocol",
        "schema_version": 1,
        "protocol_id": "hydraulic_ws_pac_v1",
        "frozen_before_download": True,
        "declared_before_access": {
            "instances": 2205,
            "target_file": "profile.txt",
            "profile_columns": [
                "cooler_condition_percent",
                "valve_condition_percent",
                "internal_pump_leakage",
                "hydraulic_accumulator_bar",
                "stable_flag",
            ],
            "stable_flag_value": 0,
            "healthy_valve_value": 100,
            "faulty_valve_values": [90, 80, 73],
        },
        "feasibility_rules": {
            "episode_definition": (
                "maximal contiguous rows with identical five-column profile tuple"
            ),
            "minimum_pac_calibration_healthy_cycles": 59,
            "abort_certificate_if_minimum_not_met": True,
        },
        "first_stage_permissions": {
            "allowed": ["profile.txt label-only feasibility audit"],
            "forbidden_until_analysis_protocol_is_frozen": ["reading sensor values"],
        },
    }


def _rows() -> list[tuple[int, int, int, int, int]]:
    return (
        [(3, 100, 0, 130, 0)] * 600
        + [(3, 90, 0, 130, 0)] * 500
        + [(20, 80, 1, 115, 0)] * 400
        + [(20, 73, 1, 100, 0)] * 400
        + [(3, 100, 0, 130, 1)] * 305
    )


def _profile_text(rows: list[tuple[int, int, int, int, int]] | None = None) -> str:
    values = _rows() if rows is None else rows
    return "\n".join("\t".join(str(value) for value in row) for row in values) + "\n"


class Fixture:
    def __init__(self, root: Path) -> None:
        self.dataset = root / "hydraulic"
        self.dataset.mkdir()
        (self.dataset / "profile.txt").write_text(_profile_text(), encoding="utf-8")
        for index in range(17):
            (self.dataset / f"sensor_{index:02d}.txt").write_text(
                "FORBIDDEN SENSOR CONTENT", encoding="utf-8"
            )
        self.archive = root / "hydraulic.zip"
        self.archive.write_bytes(b"synthetic archive bytes")
        self.protocol = root / "protocol.json"
        self.protocol.write_text(
            json.dumps(_protocol(), indent=2) + "\n", encoding="utf-8"
        )
        digest = hashlib.sha256(self.protocol.read_bytes()).hexdigest()
        self.sidecar = root / "protocol.sha256"
        self.sidecar.write_text(f"{digest}  {self.protocol.name}\n", encoding="utf-8")
        self.output = root / "output"

    def kwargs(self) -> dict[str, Path]:
        return {
            "dataset_dir": self.dataset,
            "archive_path": self.archive,
            "protocol_path": self.protocol,
            "protocol_checksum_path": self.sidecar,
            "output_dir": self.output,
        }


class HydraulicLabelAuditTests(unittest.TestCase):
    def test_profile_validation_is_exact_and_integer_only(self) -> None:
        parsed = GLOBALS["parse_profile"](_profile_text())
        self.assertEqual(len(parsed), 2205)
        self.assertEqual(len(parsed[0]), 5)

        with self.assertRaisesRegex(ValueError, "exactly 2205"):
            GLOBALS["parse_profile"](_profile_text(_rows()[:-1]))
        bad_width = _rows()
        bad_width[4] = (3, 100, 0, 130)  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "exactly five columns"):
            GLOBALS["parse_profile"](_profile_text(bad_width))
        non_integer = _profile_text().replace("3\t100", "3.0\t100", 1)
        with self.assertRaisesRegex(ValueError, "non-integer"):
            GLOBALS["parse_profile"](non_integer)

    def test_profile_requires_exact_stability_and_valve_categories(self) -> None:
        only_stable = [(*row[:4], 0) for row in _rows()]
        with self.assertRaisesRegex(ValueError, "exactly the integer values 0 and 1"):
            GLOBALS["parse_profile"](_profile_text(only_stable))
        wrong_valve = list(_rows())
        wrong_valve[1100] = (3, 91, 0, 130, 0)
        with self.assertRaisesRegex(ValueError, "exactly 100, 90, 80, and 73"):
            GLOBALS["parse_profile"](_profile_text(wrong_valve))

    def test_episodes_use_full_tuple_and_are_maximal(self) -> None:
        rows = [
            (3, 100, 0, 130, 0),
            (3, 100, 0, 130, 0),
            (3, 100, 1, 130, 0),
            (3, 100, 1, 130, 0),
            (3, 100, 0, 130, 0),
        ]
        episodes = GLOBALS["build_episodes"](rows)
        self.assertEqual([item["length_cycles"] for item in episodes], [2, 2, 1])
        self.assertEqual(
            [(item["start_row_1based"], item["end_row_1based_inclusive"]) for item in episodes],
            [(1, 2), (3, 4), (5, 5)],
        )
        self.assertEqual(episodes[1]["profile_tuple"], [3, 100, 1, 130, 0])

    def test_end_to_end_never_opens_sensor_files_and_reports_counts(self) -> None:
        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            original_open = Path.open
            opened_dataset_files: list[str] = []

            def guarded_open(path: Path, *args: object, **kwargs: object):
                candidate = path.resolve()
                if candidate.parent == fixture.dataset.resolve():
                    opened_dataset_files.append(candidate.name)
                    if candidate.name != "profile.txt":
                        raise AssertionError(f"sensor content was opened: {candidate.name}")
                return original_open(path, *args, **kwargs)

            with mock.patch("pathlib.Path.open", new=guarded_open):
                completion = GLOBALS["run_audit"](**fixture.kwargs())

            self.assertEqual(set(opened_dataset_files), {"profile.txt"})
            result = json.loads(
                (fixture.output / "label_audit.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["audit_scope"]["sensor_values_accessed"])
            self.assertFalse(result["audit_scope"]["sensor_content_hashes_computed"])
            self.assertFalse(result["dataset_file_metadata"]["content_read"])
            self.assertEqual(result["validations"]["shape"], [2205, 5])
            self.assertEqual(result["cycle_counts"]["all"]["cycle_count"], 2205)
            self.assertEqual(result["cycle_counts"]["stable"]["cycle_count"], 1900)
            self.assertEqual(
                result["cycle_counts"]["stable"]["by_valve_condition_percent"],
                {"100": 600, "90": 500, "80": 400, "73": 400},
            )
            self.assertEqual(result["episode_summaries"]["all"]["episode_count"], 5)
            self.assertEqual(result["episode_summaries"]["stable"]["episode_count"], 4)
            self.assertEqual(len(result["episodes"]), 5)
            self.assertEqual(result["inputs"]["archive"]["content_use"], "sha256_only")
            self.assertFalse(completion["sensor_values_accessed"])

    def test_result_and_completion_fingerprints_are_canonical(self) -> None:
        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            GLOBALS["run_audit"](**fixture.kwargs())
            result = json.loads(
                (fixture.output / "label_audit.json").read_text(encoding="utf-8")
            )
            fingerprint = result.pop("audit_fingerprint")
            self.assertEqual(fingerprint, GLOBALS["canonical_fingerprint"](result))
            completion = json.loads(
                (fixture.output / "completion.json").read_text(encoding="utf-8")
            )
            completion_fingerprint = completion.pop("completion_fingerprint")
            self.assertEqual(
                completion_fingerprint, GLOBALS["canonical_fingerprint"](completion)
            )
            self.assertEqual(completion["audit_fingerprint"], fingerprint)

    def test_completed_reentry_is_read_only_and_tampering_is_rejected(self) -> None:
        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            first = GLOBALS["run_audit"](**fixture.kwargs())
            artifacts = [fixture.output / "label_audit.json", fixture.output / "completion.json"]
            timestamps = {path.name: path.stat().st_mtime_ns for path in artifacts}
            second = GLOBALS["run_audit"](**fixture.kwargs())
            self.assertEqual(first, second)
            self.assertEqual(
                timestamps, {path.name: path.stat().st_mtime_ns for path in artifacts}
            )

            result_path = fixture.output / "label_audit.json"
            result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                GLOBALS["run_audit"](**fixture.kwargs())

    def test_partial_and_unexpected_output_are_never_overwritten(self) -> None:
        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            fixture.output.mkdir()
            partial = fixture.output / "label_audit.json"
            partial.write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partial"):
                GLOBALS["run_audit"](**fixture.kwargs())
            self.assertEqual(partial.read_text(encoding="utf-8"), "do not replace")

        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            GLOBALS["run_audit"](**fixture.kwargs())
            unexpected = fixture.output / "extra.txt"
            unexpected.write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                GLOBALS["run_audit"](**fixture.kwargs())

    def test_protocol_sidecar_and_frozen_declarations_are_enforced(self) -> None:
        with TemporaryDirectory() as folder:
            fixture = Fixture(Path(folder))
            fixture.sidecar.write_text("0" * 64 + "  protocol.json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                GLOBALS["build_audit"](
                    dataset_dir=fixture.dataset,
                    archive_path=fixture.archive,
                    protocol_path=fixture.protocol,
                    protocol_checksum_path=fixture.sidecar,
                )


if __name__ == "__main__":
    unittest.main()
