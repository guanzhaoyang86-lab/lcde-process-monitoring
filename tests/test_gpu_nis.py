from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mate_spc.datasets import SensorSequence
from mate_spc.gpu_nis import (
    score_complete_runs_cuda,
    validate_healthy_run_nis_cuda,
)


class _FakeFilter:
    def filter(self, values: np.ndarray, mask: np.ndarray) -> SimpleNamespace:
        del mask
        nis = np.mean(np.asarray(values, dtype=float) ** 2, axis=1)
        return SimpleNamespace(normalized_innovation=nis)


class _FakeMonitor:
    filter_model = _FakeFilter()


def _run(name: str, length: int = 5, *, fault: bool = False) -> SensorSequence:
    values = np.arange(length * 52, dtype=float).reshape(length, 52) / 100.0
    labels = np.zeros(length, dtype=bool)
    if fault:
        labels[-1] = True
    return SensorSequence(
        name=name,
        x=values,
        fault=labels,
        observed_mask=np.ones_like(values, dtype=bool),
    )


def _runtime(device: str) -> dict[str, object]:
    return {
        "backend": "pytorch_cuda_compact_batch",
        "resolved_device": device,
        "device_name": "mock GPU",
        "dtype": "float64",
    }


def _raw_result(values: np.ndarray, nis: np.ndarray) -> dict[str, object]:
    return {
        "nis": nis,
        "batch_size": values.shape[0],
        "sequence_length": values.shape[1],
        "elapsed_ms": 1.25,
        "fallback_sequence_count": 0,
        "factorization_failure_count": 0,
        "device": "cuda",
        "device_name": "mock GPU",
        "dtype": "float64",
    }


class CompleteRunNISTests(unittest.TestCase):
    @patch("mate_spc.gpu_nis.cuda_runtime_metadata", side_effect=_runtime)
    @patch("mate_spc.gpu_nis.score_incomplete_batch")
    def test_scores_equal_length_runs_in_audited_batches(
        self, backend: unittest.mock.Mock, _metadata: unittest.mock.Mock
    ) -> None:
        def score(
            _monitor: object,
            values: np.ndarray,
            masks: np.ndarray,
            **_kwargs: object,
        ) -> dict[str, object]:
            self.assertTrue(masks.all())
            nis = np.mean(values**2, axis=2)
            return _raw_result(values, nis)

        backend.side_effect = score
        runs = (_run("a"), _run("b"), _run("c"))
        result = score_complete_runs_cuda(_FakeMonitor(), runs, batch_size=2)
        self.assertEqual(tuple(result.by_run), ("a", "b", "c"))
        self.assertEqual(result.execution["kernel_calls"], 2)
        self.assertEqual(result.execution["sequences"], 3)
        self.assertEqual(result.execution["time_steps"], 15)
        self.assertEqual(result.sequence_length, 5)
        self.assertEqual(result.feature_count, 52)
        np.testing.assert_allclose(
            result.by_run["a"], np.mean(runs[0].x**2, axis=1)
        )

    @patch("mate_spc.gpu_nis.cuda_runtime_metadata", side_effect=_runtime)
    @patch("mate_spc.gpu_nis.score_incomplete_batch")
    def test_clips_only_tiny_negative_nis(
        self, backend: unittest.mock.Mock, _metadata: unittest.mock.Mock
    ) -> None:
        def score(
            _monitor: object,
            values: np.ndarray,
            _masks: np.ndarray,
            **_kwargs: object,
        ) -> dict[str, object]:
            nis = np.ones(values.shape[:2], dtype=float)
            nis[0, 0] = -5e-13
            return _raw_result(values, nis)

        backend.side_effect = score
        result = score_complete_runs_cuda(_FakeMonitor(), [_run("a")])
        self.assertEqual(result.by_run["a"][0], 0.0)

        def bad_score(
            _monitor: object,
            values: np.ndarray,
            _masks: np.ndarray,
            **_kwargs: object,
        ) -> dict[str, object]:
            nis = np.ones(values.shape[:2], dtype=float)
            nis[0, 0] = -2e-12
            return _raw_result(values, nis)

        backend.side_effect = bad_score
        with self.assertRaisesRegex(RuntimeError, "materially negative"):
            score_complete_runs_cuda(_FakeMonitor(), [_run("a")])

    def test_rejects_invalid_complete_run_groups(self) -> None:
        first = _run("a")
        with self.assertRaisesRegex(ValueError, "equal length"):
            score_complete_runs_cuda(_FakeMonitor(), [first, _run("b", 6)])
        with self.assertRaisesRegex(ValueError, "unique"):
            score_complete_runs_cuda(_FakeMonitor(), [first, _run("a")])
        incomplete = _run("incomplete")
        incomplete.observed_mask[0, 0] = False
        with self.assertRaisesRegex(ValueError, "fully observed"):
            score_complete_runs_cuda(_FakeMonitor(), [incomplete])
        exogenous = SensorSequence(
            name="exogenous",
            x=first.x,
            fault=first.fault,
            observed_mask=first.observed_mask,
            covariates=np.ones((first.x.shape[0], 1)),
        )
        with self.assertRaisesRegex(ValueError, "exogenous"):
            score_complete_runs_cuda(_FakeMonitor(), [exogenous])

    @patch("mate_spc.gpu_nis.cuda_runtime_metadata", side_effect=_runtime)
    @patch("mate_spc.gpu_nis.score_incomplete_batch")
    def test_validates_healthy_cpu_cuda_nis_and_duplicate_exactly(
        self, backend: unittest.mock.Mock, _metadata: unittest.mock.Mock
    ) -> None:
        def score(
            _monitor: object,
            values: np.ndarray,
            _masks: np.ndarray,
            **_kwargs: object,
        ) -> dict[str, object]:
            nis = np.mean(values**2, axis=2)
            return _raw_result(values, nis)

        backend.side_effect = score
        result = validate_healthy_run_nis_cuda(_FakeMonitor(), _run("healthy"))
        self.assertTrue(result["passed"])
        self.assertTrue(result["batch_duplicate_exact"])
        self.assertEqual(result["maximum_absolute"], 0.0)
        self.assertEqual(result["cuda_execution"]["sequences"], 2)

    @patch("mate_spc.gpu_nis.cuda_runtime_metadata", side_effect=_runtime)
    @patch("mate_spc.gpu_nis.score_incomplete_batch")
    def test_validation_rejects_faults_and_numeric_mismatch(
        self, backend: unittest.mock.Mock, _metadata: unittest.mock.Mock
    ) -> None:
        with self.assertRaisesRegex(ValueError, "healthy"):
            validate_healthy_run_nis_cuda(_FakeMonitor(), _run("fault", fault=True))
        backend.assert_not_called()

        def score(
            _monitor: object,
            values: np.ndarray,
            _masks: np.ndarray,
            **_kwargs: object,
        ) -> dict[str, object]:
            nis = np.mean(values**2, axis=2) + 1e-4
            return _raw_result(values, nis)

        backend.side_effect = score
        with self.assertRaisesRegex(RuntimeError, "validation failed"):
            validate_healthy_run_nis_cuda(_FakeMonitor(), _run("healthy"))


if __name__ == "__main__":
    unittest.main()
