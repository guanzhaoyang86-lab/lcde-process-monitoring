"""Subprocess GPU probe that guarantees torch loads before NumPy on Windows."""

from __future__ import annotations

import sys

import torch
import numpy as np

from mate_spc.cva_mewma import CausalCVAMEWMAMonitor
from mate_spc.datasets import SensorSequence
from mate_spc.gpu_dynamic import (
    score_cva_runs_cuda,
    score_multiscale_runs_cuda,
)
from mate_spc.multiscale_sparse import CausalMultiScaleSparseMonitor


def _matrices(length: int = 80) -> list[tuple[str, np.ndarray]]:
    result: list[tuple[str, np.ndarray]] = []
    for run_id in range(1, 21):
        rng = np.random.default_rng(9000 + run_id)
        noise = rng.normal(size=(length, 52))
        values = np.empty_like(noise)
        values[0] = noise[0]
        for index in range(1, length):
            values[index] = 0.4 * values[index - 1] + noise[index]
        result.append((f"d-{run_id:02d}", values))
    return result


def _sequence(name: str, values: np.ndarray) -> SensorSequence:
    return SensorSequence(
        name=name,
        x=values.copy(),
        fault=np.zeros(values.shape[0], dtype=bool),
        observed_mask=np.ones_like(values, dtype=bool),
    )


def _assert_fields(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    references: tuple[tuple[str, np.ndarray], ...],
) -> None:
    for key, reference in references:
        np.testing.assert_allclose(first[key], reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(first[key], second[key])


def _run_cva(
    design: list[tuple[str, np.ndarray]],
    runs: tuple[SensorSequence, ...],
) -> None:
    monitor = CausalCVAMEWMAMonitor().fit_design(design)
    cpu = monitor.decision_path(runs[0].x)
    gpu = score_cva_runs_cuda(monitor, runs, batch_size=2)
    _assert_fields(
        gpu.by_run["gpu-a"],
        gpu.by_run["gpu-b"],
        (
            ("state_energy", cpu.state_energy),
            ("residual_energy", cpu.residual_energy),
            ("component_score", cpu.component_score),
            ("component_persistence", cpu.component_persistence),
            ("persistence", cpu.persistence),
        ),
    )
    assert gpu.execution["sequences"] == 2
    assert gpu.execution["cuda_event_elapsed_seconds"] > 0.0


def _run_multiscale(
    design: list[tuple[str, np.ndarray]],
    runs: tuple[SensorSequence, ...],
) -> None:
    monitor = CausalMultiScaleSparseMonitor().fit_design(design)
    cpu = monitor.decision_path(runs[0].x)
    gpu = score_multiscale_runs_cuda(monitor, runs, batch_size=2)
    _assert_fields(
        gpu.by_run["gpu-a"],
        gpu.by_run["gpu-b"],
        (
            ("component_score", cpu.component_score),
            ("component_persistence", cpu.component_persistence),
            ("persistence", cpu.persistence),
        ),
    )
    assert gpu.execution["sequences"] == 2
    assert gpu.execution["cuda_event_elapsed_seconds"] > 0.0


def main(argv: list[str] | None = None) -> int:
    tokens = sys.argv[1:] if argv is None else argv
    if not torch.cuda.is_available():
        return 77
    if len(tokens) != 1 or tokens[0] not in {"cva", "multiscale"}:
        raise ValueError("expected exactly one probe: cva or multiscale")
    design = _matrices()
    runs = (
        _sequence("gpu-a", design[0][1]),
        _sequence("gpu-b", design[0][1]),
    )
    if tokens[0] == "cva":
        _run_cva(design, runs)
    else:
        _run_multiscale(design, runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
