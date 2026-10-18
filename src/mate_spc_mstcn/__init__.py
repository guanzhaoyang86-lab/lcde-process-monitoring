"""Lazy public surface for the experimental MSCD-TCN package.

Importing this package alone intentionally imports neither NumPy nor PyTorch.
On the supported Windows CUDA runtime, numeric runners must import the model
submodule before importing NumPy. Importing features first and then asking for
model numerics fails closed by design.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_FEATURE_EXPORTS = {
    "BASELINE_STOP",
    "CLASS_IDS",
    "CONTROL_PAIRS",
    "CONTROL_VARIABLE_PAIRS",
    "DYNAMIC_FEATURE_COUNT",
    "DYNAMIC_FEATURE_NAMES",
    "FAULT_IDS",
    "FEATURE_SCHEMA",
    "FLOW_RESPONSE_PAIRS",
    "HAAR_SCALES",
    "MONITOR_START",
    "PROCESS_FEATURE_COUNT",
    "RUN_LENGTH",
    "RobustRunTransform",
    "causal_haar",
    "dynamic_feature_names",
}
_MODEL_EXPORTS = {
    "CausalTrainingIndex",
    "DEFAULT_PARITY_ATOL",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "MSTCNConfig",
    "MSTCNModel",
    "MSTCNPath",
    "MSTCNTrainingConfig",
    "MSTCNTrainingResult",
    "MSTCNWindowScores",
    "QUICK_DIAGNOSTIC_TRAINING",
    "assert_cpu_cuda_inference_parity",
    "assert_window_sequence_inference_parity",
    "build_mstcn_network",
    "build_training_index",
    "causal_windows",
    "fit_mstcn_cuda",
    "load_mstcn_model",
    "mstcn_mil_loss_torch",
    "save_mstcn_model",
}
__all__ = sorted(_FEATURE_EXPORTS | _MODEL_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _FEATURE_EXPORTS:
        module = import_module(".features", __name__)
    elif name in _MODEL_EXPORTS:
        module = import_module(".model", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
