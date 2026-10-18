from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rieth_pilot.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT))


class FittedSuiteCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = Namespace(
            design_runs=(1, 2),
            calibration_runs=(21, 22),
            baselines=("median", "locf"),
        )
        self.suite = SimpleNamespace(
            design_run_names=(
                "Rieth-fault_free_training-fault-00-run-001",
                "Rieth-fault_free_training-fault-00-run-002",
            ),
            calibration_run_names=(
                "Rieth-fault_free_testing-fault-00-run-021",
                "Rieth-fault_free_testing-fault-00-run-022",
            ),
            baselines=("median", "locf"),
        )

    def test_exact_checkpoint_split_matches(self) -> None:
        suite_matches = SCRIPT_GLOBALS["_suite_matches"]
        self.assertIs(suite_matches(self.suite, self.args), True)

    def test_changed_calibration_split_does_not_match(self) -> None:
        suite_matches = SCRIPT_GLOBALS["_suite_matches"]
        changed = SimpleNamespace(
            **{
                **vars(self.suite),
                "calibration_run_names": (
                    "Rieth-fault_free_testing-fault-00-run-021",
                    "Rieth-fault_free_testing-fault-00-run-023",
                ),
            }
        )
        self.assertIs(suite_matches(changed, self.args), False)

    def test_changed_baseline_set_does_not_match(self) -> None:
        suite_matches = SCRIPT_GLOBALS["_suite_matches"]
        changed = SimpleNamespace(**{**vars(self.suite), "baselines": ("median",)})
        self.assertIs(suite_matches(changed, self.args), False)

    def test_cuda_cli_prescan_accepts_both_argparse_styles(self) -> None:
        requests_cuda = SCRIPT_GLOBALS["_cli_requests_cuda"]
        self.assertTrue(requests_cuda(["--device", "cuda"]))
        self.assertTrue(requests_cuda(["--device=cuda"]))
        self.assertFalse(requests_cuda(["--device", "cpu"]))
        self.assertFalse(requests_cuda([]))

    def test_no_resume_cuda_requires_explicit_warm_start(self) -> None:
        parser = SCRIPT_GLOBALS["build_parser"]()
        validate = SCRIPT_GLOBALS["_validate_args"]
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "fitted_suite.pkl").touch()
            args = parser.parse_args(
                [
                    "unused-data",
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda",
                    "--no-resume",
                ]
            )
            with self.assertRaisesRegex(ValueError, "warm-start-suite"):
                validate(args)


if __name__ == "__main__":
    unittest.main()
