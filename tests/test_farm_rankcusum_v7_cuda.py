from __future__ import annotations

import numpy as np
import pytest

from mate_spc_farm_v7.monitor import (
    FarmReference,
    MONITOR_START,
    score_runs_cpu,
    score_runs_cuda,
)


def _healthy_runs() -> np.ndarray:
    run = np.arange(60, dtype=np.float64)[:, None, None] * 1.0e-5
    time = np.arange(500, dtype=np.float64)[None, :, None] * 1.0e-3
    channel = np.arange(52, dtype=np.float64)[None, None, :] * 0.1
    return np.asarray(run + time + channel, dtype=np.float64)


def _test_runs() -> np.ndarray:
    channel = np.arange(52, dtype=np.float64)[None, None, :] * 0.1
    runs = np.broadcast_to(channel + 0.25, (3, 500, 52)).copy()
    runs[1, MONITOR_START:, :] = channel[0, 0, :] + 2.0
    runs[2, MONITOR_START:, :] = channel[0, 0, :] - 2.0
    return np.asarray(runs, dtype=np.float64)


def _cuda_or_skip() -> object:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


def test_cuda_authoritative_batch_is_repeatable_and_matches_cpu_float64() -> None:
    torch = _cuda_or_skip()
    reference = FarmReference.fit(_healthy_runs())
    runs = _test_runs()
    cpu = score_runs_cpu(reference, runs)
    first = score_runs_cuda(reference, runs, device=torch.device("cuda"), batch_size=1)
    second = score_runs_cuda(reference, runs, device="cuda", batch_size=2)
    assert first.source_device.startswith("cuda")
    assert first.paths.dtype == np.float64
    np.testing.assert_array_equal(first.paths, second.paths)
    np.testing.assert_array_equal(first.run_maxima, second.run_maxima)
    np.testing.assert_allclose(first.paths, cpu.paths, rtol=0.0, atol=2.0e-10)


def test_cuda_entrypoint_rejects_non_cuda_device_and_invalid_batch_size() -> None:
    _cuda_or_skip()
    reference = FarmReference.fit(_healthy_runs())
    runs = _test_runs()[:1]
    with pytest.raises(ValueError, match="CUDA device"):
        score_runs_cuda(reference, runs, device="cpu")
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError):
            score_runs_cuda(reference, runs, batch_size=invalid)  # type: ignore[arg-type]
