"""PCF-CycleRank-Scan v6.

The package initializer intentionally imports no numeric dependency.  The
Windows RTX 5090 runner must import PyTorch before NumPy/HDF5, so callers use
the explicit ``features``, ``model``, ``loss``, ``training``, ``scan``,
``evaluation``, and ``backend_v6`` submodules after runtime preflight.
"""

from __future__ import annotations


METHOD_NAME = "PCF-CycleRank-Scan"
CANDIDATE_ID = "pcf-cyclerank-scan-v6-fixed"
SCHEMA_VERSION = 6


__all__ = ["CANDIDATE_ID", "METHOD_NAME", "SCHEMA_VERSION"]
