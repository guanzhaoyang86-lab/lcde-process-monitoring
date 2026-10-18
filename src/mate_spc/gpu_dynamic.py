"""Auditable float64 CUDA scoring for the two frozen dynamic monitors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .cva_mewma import CausalCVAMEWMAMonitor
from .datasets import SensorSequence
from .gpu_multirun import cuda_runtime_metadata
from .multiscale_sparse import CausalMultiScaleSparseMonitor


@dataclass(frozen=True)
class DynamicGPUResult:
    """Per-run score arrays plus explicit CUDA execution provenance."""

    by_run: dict[str, dict[str, np.ndarray]]
    execution: dict[str, object]
    runtime_environment: dict[str, object]
    sequence_length: int
    feature_count: int


def _validate_runs(
    runs: Iterable[SensorSequence],
    *,
    minimum_length: int,
) -> tuple[tuple[SensorSequence, ...], int]:
    normalized = tuple(runs)
    if not normalized:
        raise ValueError("at least one complete run is required")
    names = [str(run.name) for run in normalized]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("run names must be nonempty and unique")
    lengths: set[int] = set()
    for run in normalized:
        values = np.asarray(run.x, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 52
            or values.shape[0] < minimum_length
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"run {run.name!r} must be complete finite (time, 52) "
                f"with at least {minimum_length} samples"
            )
        if run.covariates is not None:
            raise ValueError(f"run {run.name!r} must not contain exogenous channels")
        if run.observed_mask is not None:
            mask = np.asarray(run.observed_mask, dtype=bool)
            if mask.shape != values.shape or not mask.all():
                raise ValueError(f"run {run.name!r} must be fully observed")
        lengths.add(values.shape[0])
    if len(lengths) != 1:
        raise ValueError("runs within one CUDA scoring call must have equal length")
    return normalized, lengths.pop()


def _positive_batch_size(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError("batch_size must be a positive integer")
    return result


def _torch_family_persistence(
    component: object,
    *,
    valid_start: int,
    window: int,
) -> tuple[object, object]:
    import torch

    if not bool(torch.isfinite(component[:, valid_start:]).all()):
        raise RuntimeError("CUDA component score contains non-finite eligible values")
    batch, length, count = component.shape
    first = valid_start + window - 1
    rolling = component[:, valid_start:].unfold(1, window, 1)
    component_valid = torch.min(rolling, dim=-1).values
    component_persistence = torch.full(
        (batch, length, count),
        -torch.inf,
        dtype=component.dtype,
        device=component.device,
    )
    component_persistence[:, first:] = component_valid
    envelope = torch.full(
        (batch, length),
        -torch.inf,
        dtype=component.dtype,
        device=component.device,
    )
    envelope[:, first:] = torch.max(component_valid, dim=2).values
    return component_persistence, envelope


def _execution(
    *,
    backend: str,
    device: str,
    device_name: str,
    calls: int,
    sequences: int,
    sequence_length: int,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "backend": backend,
        "device": device,
        "device_name": device_name,
        "dtype": "float64",
        "kernel_calls": calls,
        "sequences": sequences,
        "time_steps": sequences * sequence_length,
        "cuda_event_elapsed_seconds": elapsed_ms / 1000.0,
        "fallback_sequences": 0,
        "factorization_failures": 0,
        "parameter_fitting_backend": "numpy_cpu",
        "score_backend": backend,
        "tf32": False,
    }


def _score_batches(
    runs: tuple[SensorSequence, ...],
    sequence_length: int,
    *,
    device: str,
    batch_size: int,
    backend: str,
    scorer: Callable[[object], dict[str, object]],
) -> DynamicGPUResult:
    import torch

    runtime = cuda_runtime_metadata(device)
    target = torch.device(device)
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    by_run: dict[str, dict[str, np.ndarray]] = {}
    elapsed_ms = 0.0
    calls = 0
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            for start in range(0, len(runs), batch_size):
                chunk = runs[start : start + batch_size]
                values = torch.as_tensor(
                    np.stack([run.x for run in chunk]),
                    dtype=torch.float64,
                    device=target,
                )
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = scorer(values)
                end.record()
                end.synchronize()
                elapsed_ms += float(begin.elapsed_time(end))
                calls += 1
                arrays = {
                    key: value.detach().cpu().numpy().astype(np.float64, copy=False)
                    for key, value in result.items()
                }
                for position, run in enumerate(chunk):
                    by_run[str(run.name)] = {
                        key: array[position].copy() for key, array in arrays.items()
                    }
                del values, result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    if len(by_run) != len(runs):
        raise RuntimeError("CUDA dynamic scorer did not return every run")
    execution = _execution(
        backend=backend,
        device=str(target),
        device_name=str(runtime["device_name"]),
        calls=calls,
        sequences=len(runs),
        sequence_length=sequence_length,
        elapsed_ms=elapsed_ms,
    )
    return DynamicGPUResult(
        by_run=by_run,
        execution=execution,
        runtime_environment={**runtime, "backend": backend},
        sequence_length=sequence_length,
        feature_count=52,
    )


def score_cva_runs_cuda(
    monitor: CausalCVAMEWMAMonitor,
    runs: Iterable[SensorSequence],
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> DynamicGPUResult:
    """Score complete runs with a frozen CPU-fitted CVA projection on CUDA."""
    import torch

    if not isinstance(monitor, CausalCVAMEWMAMonitor) or not monitor.design_fitted_:
        raise ValueError("monitor must be a design-fitted CausalCVAMEWMAMonitor")
    batch_size = _positive_batch_size(batch_size)
    normalized, sequence_length = _validate_runs(
        runs,
        minimum_length=monitor.PAST_ORDER + monitor.PERSISTENCE_WINDOW,
    )
    target = torch.device(device)
    raw_center = torch.as_tensor(monitor.raw_center_, dtype=torch.float64, device=target)
    raw_scale = torch.as_tensor(monitor.raw_scale_, dtype=torch.float64, device=target)
    past_mean = torch.as_tensor(monitor.past_mean_, dtype=torch.float64, device=target)
    projection = torch.as_tensor(monitor.projection_, dtype=torch.float64, device=target)
    energy_center = torch.as_tensor(
        monitor.energy_center_, dtype=torch.float64, device=target
    )
    energy_scale = torch.as_tensor(
        monitor.energy_scale_, dtype=torch.float64, device=target
    )
    order = monitor.PAST_ORDER
    smoothing = monitor.EWMA_LAMBDA
    beta = 1.0 - smoothing
    rank = monitor.rank_
    valid_start = monitor.valid_start_

    def scorer(values: object) -> dict[str, object]:
        standardized = (values - raw_center) / raw_scale
        length = standardized.shape[1]
        past = torch.cat(
            [
                standardized[:, order - 1 - lag : length - lag]
                for lag in range(order)
            ],
            dim=2,
        )
        coordinates = (past - past_mean) @ projection
        steps = coordinates.shape[1]
        powers = beta ** torch.arange(
            steps, dtype=torch.float64, device=coordinates.device
        )
        ewma = (
            smoothing
            * torch.cumsum(coordinates / powers.view(1, -1, 1), dim=1)
            * powers.view(1, -1, 1)
        )
        state_valid = torch.sum(ewma[:, :, :rank] ** 2, dim=2)
        residual_valid = torch.sum(ewma[:, :, rank:] ** 2, dim=2)
        state = torch.full(
            (values.shape[0], length),
            torch.nan,
            dtype=torch.float64,
            device=values.device,
        )
        residual = torch.full_like(state, torch.nan)
        state[:, valid_start:] = state_valid
        residual[:, valid_start:] = residual_valid
        logged = torch.stack(
            (torch.log1p(state_valid), torch.log1p(residual_valid)), dim=2
        )
        component_valid = (logged - energy_center) / energy_scale
        component = torch.full(
            (values.shape[0], length, 2),
            torch.nan,
            dtype=torch.float64,
            device=values.device,
        )
        component[:, valid_start:] = component_valid
        component_persistence, persistence = _torch_family_persistence(
            component,
            valid_start=valid_start,
            window=monitor.PERSISTENCE_WINDOW,
        )
        return {
            "state_energy": state,
            "residual_energy": residual,
            "component_score": component,
            "component_persistence": component_persistence,
            "persistence": persistence,
        }

    return _score_batches(
        normalized,
        sequence_length,
        device=device,
        batch_size=batch_size,
        backend="pytorch_cuda_causal_cva_mewma",
        scorer=scorer,
    )


def score_multiscale_runs_cuda(
    monitor: CausalMultiScaleSparseMonitor,
    runs: Iterable[SensorSequence],
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> DynamicGPUResult:
    """Score the frozen multi-scale sparse family on CUDA."""
    import torch

    if (
        not isinstance(monitor, CausalMultiScaleSparseMonitor)
        or not monitor.design_fitted_
    ):
        raise ValueError(
            "monitor must be a design-fitted CausalMultiScaleSparseMonitor"
        )
    batch_size = _positive_batch_size(batch_size)
    normalized, sequence_length = _validate_runs(
        runs,
        minimum_length=max(monitor.WINDOWS) + monitor.PERSISTENCE_WINDOW,
    )
    target = torch.device(device)
    raw_center = torch.as_tensor(monitor.raw_center_, dtype=torch.float64, device=target)
    raw_scale = torch.as_tensor(monitor.raw_scale_, dtype=torch.float64, device=target)
    channel_center = {
        key: torch.as_tensor(value, dtype=torch.float64, device=target)
        for key, value in monitor.channel_center_.items()
    }
    channel_scale = {
        key: torch.as_tensor(value, dtype=torch.float64, device=target)
        for key, value in monitor.channel_scale_.items()
    }
    energy_center = {
        key: torch.as_tensor(value, dtype=torch.float64, device=target)
        for key, value in monitor.energy_center_.items()
    }
    energy_scale = {
        key: torch.as_tensor(value, dtype=torch.float64, device=target)
        for key, value in monitor.energy_scale_.items()
    }
    count = len(monitor.component_names_)
    valid_start = monitor.valid_start_

    def scorer(values: object) -> dict[str, object]:
        standardized = (values - raw_center) / raw_scale
        batch, length, _ = standardized.shape
        component = torch.full(
            (batch, length, count),
            torch.nan,
            dtype=torch.float64,
            device=values.device,
        )
        position = 0
        for window in monitor.WINDOWS:
            windows = standardized.unfold(1, window, 1)
            means = torch.mean(windows, dim=-1)
            centered = windows - means.unsqueeze(-1)
            variance = torch.sum(centered**2, dim=-1) / (window - 1)
            difference_energy = torch.mean(
                torch.diff(windows, dim=-1) ** 2, dim=-1
            )
            statistics = {
                "mean": means,
                "log_variance": torch.log(variance + monitor.EPSILON),
                "log_difference_energy": torch.log(
                    difference_energy + monitor.EPSILON
                ),
            }
            for statistic in monitor.STATISTICS:
                key = (window, statistic)
                normalized_statistic = (
                    statistics[statistic] - channel_center[key]
                ) / channel_scale[key]
                top = torch.topk(
                    normalized_statistic**2,
                    k=monitor.TOP_K,
                    dim=2,
                    largest=True,
                    sorted=False,
                ).values
                logged_energy = torch.log1p(torch.mean(top, dim=2))
                component[:, window - 1 :, position] = (
                    logged_energy - energy_center[key]
                ) / energy_scale[key]
                position += 1
        component_persistence, persistence = _torch_family_persistence(
            component,
            valid_start=valid_start,
            window=monitor.PERSISTENCE_WINDOW,
        )
        return {
            "component_score": component,
            "component_persistence": component_persistence,
            "persistence": persistence,
        }

    return _score_batches(
        normalized,
        sequence_length,
        device=device,
        batch_size=batch_size,
        backend="pytorch_cuda_multiscale_sparse_dynamic",
        scorer=scorer,
    )


__all__ = [
    "DynamicGPUResult",
    "score_cva_runs_cuda",
    "score_multiscale_runs_cuda",
]
