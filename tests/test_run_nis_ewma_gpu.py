from __future__ import annotations

from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_nis_ewma_gpu.py"
GLOBALS = runpy.run_path(str(SCRIPT))


class NISEWMAGPURunnerTests(unittest.TestCase):
    def test_device_prescan_accepts_both_cli_styles(self) -> None:
        requested = GLOBALS["_requested_device"]
        self.assertEqual(requested(["--device", "cuda"]), "cuda")
        self.assertEqual(requested(["--device=cuda"]), "cuda")
        self.assertIsNone(requested([]))

    def test_pre_registered_binomial_boundary_is_exact(self) -> None:
        upper_tail = GLOBALS["_binomial_upper_tail"]
        self.assertAlmostEqual(
            upper_tail(100, 0.05, 10),
            0.028188294163416,
            places=14,
        )

    def test_atomic_npz_round_trip_disallows_pickle(self) -> None:
        atomic_npz = GLOBALS["_atomic_npz"]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npz"
            expected = np.arange(12, dtype=float).reshape(3, 4)
            atomic_npz(path, {"healthy_001": expected})
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["healthy_001"], expected)

    def test_clean_json_converts_nonfinite_metrics_to_null(self) -> None:
        clean = GLOBALS["_clean_json"]
        self.assertEqual(
            clean({"finite": np.float64(1.5), "missing": float("nan")}),
            {"finite": 1.5, "missing": None},
        )


if __name__ == "__main__":
    unittest.main()
