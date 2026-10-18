from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


PROBE = Path(__file__).with_name("_directional_gpu_probe.py")


class DirectionalCUDATests(unittest.TestCase):
    def test_cpu_cuda_batch_reset_and_determinism(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PROBE)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode == 77:
            self.skipTest("CUDA is not available")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
