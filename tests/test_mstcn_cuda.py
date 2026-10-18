from __future__ import annotations

import torch
import numpy as np
import pytest

import mate_spc_mstcn.model as model


def _run(seed: int, class_id: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.2, size=(500, 52)).astype(np.float64)
    if class_id == 3:
        values[160:, 0] += 1.0
    elif class_id == 9:
        values[160:, 1] += rng.normal(scale=0.7, size=340)
    elif class_id == 15:
        actuator = np.sign(np.sin(np.arange(340) / 7.0))
        values[160:, 51] += actuator
        values[162:, 10] += 0.35 * actuator[:-2]
    return values


def _training_data() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    runs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for class_id in (0, 3, 9, 15):
        for replicate in range(2):
            name = f"class-{class_id}-run-{replicate}"
            runs[name] = _run(7000 + 10 * class_id + replicate, class_id)
            labels[name] = class_id
    return runs, labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_mil_fit_smoke_and_dual_head_parity() -> None:
    runs, labels = _training_data()
    config = model.MSTCNConfig(
        channels=2,
        dilations=(1,),
        hidden_size=4,
        mil_top_fraction=0.125,
    )
    result = model.fit_mstcn_cuda(
        runs,
        labels,
        config=config,
        seed=119,
        epochs=1,
        batch_size=8,
        examples_per_class_per_epoch=2,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        gradient_clip_norm=1.0,
        device="cuda",
    )
    assert len(result.loss_history) == 1
    assert np.isfinite(result.loss_history[0])
    assert result.metadata["positive_endpoint_labels_used"] is False
    assert result.metadata["mil_top_fraction_q"] == 0.125
    assert result.metadata["batch_balance"] == "eight_runs_exactly_two_per_class"
    assert result.metadata["mil_endpoint_count"] == 85
    assert result.metadata["training_config"] == {
        "epochs": 1,
        "batch_size": 8,
        "examples_per_class_per_epoch": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "device": "cuda",
    }

    probe = _run(9001, 9)
    maximum = model.assert_cpu_cuda_inference_parity(
        result.model,
        probe,
        endpoints=np.array([160, 240, 496], dtype=np.int64),
        batch_size=3,
        atol=1.0e-9,
    )
    assert maximum <= 1.0e-9
    sequence_error = model.assert_window_sequence_inference_parity(
        result.model,
        probe,
        endpoints=np.array([160, 240, 496], dtype=np.int64),
        device="cuda",
        batch_size=3,
        atol=1.0e-9,
    )
    assert sequence_error <= 1.0e-9
