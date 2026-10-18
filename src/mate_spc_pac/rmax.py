"""Public API for the audited WS-PAC-RMax-CUSUM implementation."""

from .calibration import calibrate_ws_pac_rmax_paths
from .certificate import (
    STRICT_COMPARISON,
    PACCalibration,
    calibration_maxima_sha256,
)
from .common import (
    CLIP_LIMIT,
    GATE_DEFER,
    GATE_PASS,
    PERSISTENCE_WINDOW,
    ROBUST_SCALE_FACTOR,
    DeploymentSpec,
    PrefixApplicabilityConfig,
    PrefixGateResult,
)
from .core import (
    StandardizedLogitRun,
    WSPACRMaxModel,
    WSPACRMaxPath,
    page_cusum_numpy,
    score_ws_pac_rmax_numpy,
    score_ws_pac_rmax_runs_numpy,
)
from .probability import (
    CALIBRATION_CERTIFIED,
    MAX_CERTIFIABLE_SAMPLE_COUNT,
    InsufficientCalibrationRunsError,
    PACRank,
    minimum_pac_calibration_runs,
    pac_tolerance_rank,
)
from .serialization import (
    MODEL_SCHEMA,
    MODEL_SCHEMA_VERSION,
    load_ws_pac_rmax_model,
    save_ws_pac_rmax_model,
)

__all__ = [
    "CALIBRATION_CERTIFIED",
    "CLIP_LIMIT",
    "GATE_DEFER",
    "GATE_PASS",
    "MAX_CERTIFIABLE_SAMPLE_COUNT",
    "MODEL_SCHEMA",
    "MODEL_SCHEMA_VERSION",
    "PERSISTENCE_WINDOW",
    "ROBUST_SCALE_FACTOR",
    "STRICT_COMPARISON",
    "DeploymentSpec",
    "InsufficientCalibrationRunsError",
    "PACCalibration",
    "PACRank",
    "PrefixApplicabilityConfig",
    "PrefixGateResult",
    "StandardizedLogitRun",
    "WSPACRMaxModel",
    "WSPACRMaxPath",
    "calibrate_ws_pac_rmax_paths",
    "calibration_maxima_sha256",
    "load_ws_pac_rmax_model",
    "minimum_pac_calibration_runs",
    "pac_tolerance_rank",
    "page_cusum_numpy",
    "save_ws_pac_rmax_model",
    "score_ws_pac_rmax_numpy",
    "score_ws_pac_rmax_runs_numpy",
]
