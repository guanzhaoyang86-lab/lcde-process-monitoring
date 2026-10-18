"""CUDA subprocess probe; torch must load before NumPy on Windows."""

from __future__ import annotations

import sys

import torch
import numpy as np

from mate_spc_directional.glr import (
    FAULT_IDS,
    DirectionalGLRModel,
    compare_directional_paths,
)
from mate_spc_directional.gpu import (
    fit_directional_glr_cuda,
    score_directional_glr_cuda,
    validate_directional_glr_cuda,
)


def _fixed_model() -> DirectionalGLRModel:
    return DirectionalGLRModel(
        fault_ids=FAULT_IDS,
        weights=np.array(
            [
                [0.5, -0.25, 0.1, 0.0],
                [0.0, 0.4, -0.2, 0.3],
                [-0.1, 0.2, 0.6, -0.4],
            ]
        ),
        bias=np.array([0.1, -0.2, 0.3]),
        normal_logit_center=np.array([0.05, -0.1, 0.2]),
        normal_logit_scale=np.array([0.8, 1.1, 1.4]),
        kappa=np.array([0.1, 0.25, 0.5]),
        threshold=2.0,
    )


def _scoring_features() -> np.ndarray:
    values = np.random.default_rng(8811).normal(size=(41, 4))
    values[:7] = np.nan
    return values


def _training_task() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    features: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    shifts = {
        0: np.zeros(4),
        3: np.array([2.0, 0.0, 0.0, 0.0]),
        9: np.array([0.0, 2.0, 0.0, 0.0]),
        15: np.array([0.0, 0.0, 2.0, 0.0]),
    }
    for position, label in enumerate((0, 3, 9, 15)):
        values = np.random.default_rng(9900 + position).normal(size=(500, 4))
        values[219:] += shifts[label]
        values[:63] = np.nan
        name = f"class-{label:02d}"
        features[name] = values
        labels[name] = label
    return features, labels


def _assert_scoring_invariance() -> None:
    model = _fixed_model()
    values = _scoring_features()
    validation = validate_directional_glr_cuda(
        model,
        values,
        valid_start=7,
        atol=1e-12,
    )
    assert validation["cpu_cuda_alarm_exact"]
    assert validation["duplicate_alarm_exact"]
    assert validation["batch_partition_alarm_exact"]
    assert validation["single_execution"]["fallback_sequences"] == 0
    assert validation["runtime_environment"]["cuda_available"]
    assert validation["runtime_environment"]["dtype"] == "float64"

    first = score_directional_glr_cuda(
        model,
        {"same": values},
        valid_start=7,
        batch_size=1,
    )
    second = score_directional_glr_cuda(
        model,
        {"same": values},
        valid_start=7,
        batch_size=1,
    )
    compare_directional_paths(
        first.by_run["same"],
        second.by_run["same"],
        atol=0.0,
    )


def _assert_fit_is_deterministic() -> None:
    features, labels = _training_task()
    common = {
        "ridge_lambda": 0.1,
        "kappa": 0.25,
        "max_iter": 50,
        "max_eval": 75,
        "tolerance_grad": 1e-10,
        "tolerance_change": 1e-12,
    }
    first, first_metadata = fit_directional_glr_cuda(features, labels, **common)
    second, second_metadata = fit_directional_glr_cuda(features, labels, **common)

    assert first_metadata["final_objective"] < first_metadata["initial_objective"]
    assert second_metadata["final_objective"] < second_metadata["initial_objective"]
    for field in (
        "weights",
        "bias",
        "normal_logit_center",
        "normal_logit_scale",
        "kappa",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first_metadata["optimizer_iterations"] == second_metadata["optimizer_iterations"]
    assert first_metadata["closure_evaluations"] == second_metadata["closure_evaluations"]


def main() -> int:
    if not torch.cuda.is_available():
        return 77
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    _assert_scoring_invariance()
    _assert_fit_is_deterministic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
