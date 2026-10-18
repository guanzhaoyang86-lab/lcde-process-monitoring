from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


PROBE = Path(__file__).with_name("_gpu_dynamic_probe.py")


class DynamicCUDATests(unittest.TestCase):
    def _run_probe(self, method: str) -> None:
        result = subprocess.run(
            [sys.executable, str(PROBE), method],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 77:
            self.skipTest("CUDA is not available")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_cva_cpu_cuda_and_duplicate_match(self) -> None:
        self._run_probe("cva")

    def test_multiscale_cpu_cuda_and_duplicate_match(self) -> None:
        self._run_probe("multiscale")


if __name__ == "__main__":
    unittest.main()
