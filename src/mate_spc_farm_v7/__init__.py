"""Run-calibrated FARM rank-CUSUM v7.

The initializer intentionally imports no numeric dependency.  Production
entrypoints can therefore perform their CUDA/runtime preflight before loading
NumPy or PyTorch through :mod:`mate_spc_farm_v7.monitor`.
"""

from __future__ import annotations


METHOD_NAME = "Run-Calibrated FARM Rank-CUSUM"
CANDIDATE_ID = "farm-rankcusum-v7-fixed"
SCHEMA_VERSION = 7


__all__ = ["CANDIDATE_ID", "METHOD_NAME", "SCHEMA_VERSION"]
