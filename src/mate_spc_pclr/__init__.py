"""Causal PCLR successor monitor.

The public names are loaded lazily. This keeps metadata-only
``mate_spc_pclr.data`` imports independent of PyTorch/CUDA and, on Windows,
allows ``mate_spc_pclr.backend`` to preload PyTorch before importing the
NumPy-based data and model modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BASELINE_START",
    "BASELINE_STOP",
    "CAUSAL_LAGS",
    "CONTROL_PAIRS",
    "FAULT_IDS",
    "FEATURE_COUNT",
    "HEALTHY_ID",
    "MONITOR_START",
    "MONITOR_STOP",
    "MULTISCALE_WINDOWS",
    "PCLRConfig",
    "PCLRFeatureModel",
    "PCLRFeaturePath",
    "PCLRHeadFoldFit",
    "PCLRHeads",
    "PCLRModel",
    "PCLRModelFoldFit",
    "PCLRMonitoringBatch",
    "PCLRMonitoringPath",
    "PCLRRunDiagnostics",
    "PCLR_FEATURE_COUNT",
    "PCLR_FEATURE_NAMES",
    "PERSISTENCE",
    "RUN_LENGTH",
    "SequenceGroupSplit",
    "assert_numpy_torch_logit_parity",
    "diagnostics_by_run",
    "first_episode_detection",
    "fit_pclr_heads_cuda",
    "fit_pclr_heads_group_cv_cuda",
    "fit_pclr_group_cv_cuda",
    "load_pclr_model",
    "logits_torch_float64",
    "monitor_batch_torch_float64",
    "page_cusum_numpy",
    "page_cusum_torch_float64",
    "pclr_feature_names",
    "run_score_diagnostics",
    "save_pclr_model",
    "sequence_group_folds",
    "sequence_group_splits",
]

_MODEL_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Resolve the historical package-level model API on first use."""

    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = import_module(f"{__name__}.model")
    value = getattr(model, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _MODEL_EXPORTS)
