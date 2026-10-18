from __future__ import annotations

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamic_sequence_gpu.py"
GLOBALS = runpy.run_path(str(SCRIPT))


class DynamicSequenceRunnerTests(unittest.TestCase):
    def test_device_prescan_accepts_both_styles(self) -> None:
        requested = GLOBALS["_requested_device"]
        self.assertEqual(requested(["--device", "cuda"]), "cuda")
        self.assertEqual(requested(["--device=cuda"]), "cuda")
        self.assertIsNone(requested([]))

    def test_protocol_freezes_fallback_before_stage_a(self) -> None:
        protocol = GLOBALS["_protocol"](
            checksums={"x": "y"},
            source_hashes={"a.py": "abc"},
            nis_sha256="nis-sha",
            nis_fingerprint="nis-protocol",
            batch_size=32,
        )
        self.assertEqual(
            protocol["sequence_rule"]["stage_b"],
            "run fallback iff stage_a fails any gate",
        )
        self.assertEqual(
            protocol["stage_b_method"]["windows_samples"], [5, 20, 60]
        )
        self.assertEqual(
            protocol["splits"]["stage_a_audit"][1], list(range(221, 321))
        )
        self.assertEqual(
            protocol["splits"]["stage_b_audit_if_triggered"][1],
            list(range(321, 401)),
        )
        self.assertEqual(
            protocol["splits"]["final_healthy_reserved"][1],
            list(range(401, 501)),
        )
        self.assertEqual(protocol["gates"]["stage_a"]["fault_total"], 120)
        self.assertEqual(protocol["gates"]["stage_b"]["fault_total"], 150)

    def test_binomial_boundaries_are_exact(self) -> None:
        tail = GLOBALS["_binomial_upper_tail"]
        self.assertAlmostEqual(tail(100, 0.05, 10), 0.028188294163416, places=14)
        self.assertAlmostEqual(tail(80, 0.05, 8), 0.04659152748523252, places=14)

    def test_finite_calibration_predictive_boundaries_are_explicit(self) -> None:
        tail = GLOBALS["_beta_binomial_upper_tail"]
        self.assertAlmostEqual(tail(100, 5.0, 96.0, 10), 0.08199, places=4)
        self.assertAlmostEqual(tail(80, 5.0, 96.0, 8), 0.09662, places=4)

    def test_manifest_rejects_unidentified_nonempty_output(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "stale.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                GLOBALS["_load_or_create_manifest"](
                    folder / "manifest.json", {"schema_version": 1}
                )

    def test_atomic_npz_round_trip(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npz"
            expected = np.arange(20, dtype=float).reshape(5, 4)
            GLOBALS["_atomic_npz"](path, {"run": expected})
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["run"], expected)

    def test_completion_verifies_every_artifact_and_detects_tampering(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "one.txt").write_text("one", encoding="utf-8")
            (folder / "two.txt").write_text("two", encoding="utf-8")
            GLOBALS["_write_completion"](
                folder,
                "protocol",
                "decision",
                ("one.txt", "two.txt"),
            )
            self.assertTrue(GLOBALS["_verify_completion"](folder, "protocol"))
            (folder / "two.txt").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                GLOBALS["_verify_completion"](folder, "protocol")

    def test_prior_nis_must_record_failure(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "aggregate.json"
            value = {
                "protocol_fingerprint": "old",
                "pre_registered_gate_result": {
                    "decision": "FAIL_PRE_REGISTERED_GATE"
                },
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            digest, fingerprint = GLOBALS["_read_failed_nis"](path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(fingerprint, "old")
            value["pre_registered_gate_result"]["decision"] = "PASS"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed NIS"):
                GLOBALS["_read_failed_nis"](path)


if __name__ == "__main__":
    unittest.main()
