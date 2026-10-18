"""GPU MiniROCKET Log-Odds CUSUM v8.

PyTorch must be imported before NumPy in the pinned Windows CUDA environment.
Import it here so every package submodule inherits the safe DLL load order.
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401
except ImportError as error:  # pragma: no cover - environment preflight owns this
    raise RuntimeError("PyTorch is required for MiniROCKET v8") from error


METHOD_NAME = "GPU MiniROCKET Log-Odds CUSUM"
CANDIDATE_ID = "gpu-minirocket-logodds-cusum-v8-fixed"
SCHEMA_VERSION = 8


__all__ = ["CANDIDATE_ID", "METHOD_NAME", "SCHEMA_VERSION"]
