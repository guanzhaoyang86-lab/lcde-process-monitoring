"""Stable orchestration-facing API for the v8 fixed candidate."""

from __future__ import annotations

# Import model first: it preserves the required torch-before-NumPy DLL order.
from .model import (
    FeatureStandardizer,
    LinearHeadState,
    RobustNormalizer,
    TrainingWindows,
    build_final_refit_training_windows,
    build_training_windows,
    fit_feature_standardizer,
    fit_feature_standardizer_final_refit,
    fit_linear_head,
    fit_linear_head_final_refit,
    linear_logits,
    normalize_runs,
    robust_fit_normalizer,
    robust_fit_normalizer_final_refit,
    set_determinism,
    standardize_features,
)
from .transform import (
    MiniRocketTransformState,
    fit_minirocket_transform,
    fit_minirocket_transform_final_refit,
    load_minirocket_state_npz,
    save_minirocket_state_npz,
    transform_windows,
)
from .monitor import (
    InnerThreshold,
    MonitorCalibration,
    SequentialScoreBatch,
    ThresholdDecisions,
    apply_threshold,
    compute_log_odds,
    fit_monitor_calibration,
    inner_threshold,
    scan_batch,
    scan_run,
)
from .evaluation import (
    DevelopmentGateReport,
    RunDecision,
    evaluate_gates,
    fold_roles,
)


__all__ = [
    "DevelopmentGateReport",
    "FeatureStandardizer",
    "InnerThreshold",
    "LinearHeadState",
    "MonitorCalibration",
    "MiniRocketTransformState",
    "RobustNormalizer",
    "RunDecision",
    "SequentialScoreBatch",
    "ThresholdDecisions",
    "TrainingWindows",
    "apply_threshold",
    "build_training_windows",
    "build_final_refit_training_windows",
    "compute_log_odds",
    "evaluate_gates",
    "fit_feature_standardizer",
    "fit_feature_standardizer_final_refit",
    "fit_linear_head",
    "fit_linear_head_final_refit",
    "fit_minirocket_transform",
    "fit_minirocket_transform_final_refit",
    "fit_monitor_calibration",
    "fold_roles",
    "inner_threshold",
    "linear_logits",
    "load_minirocket_state_npz",
    "normalize_runs",
    "robust_fit_normalizer",
    "robust_fit_normalizer_final_refit",
    "scan_batch",
    "scan_run",
    "save_minirocket_state_npz",
    "set_determinism",
    "standardize_features",
    "transform_windows",
]
