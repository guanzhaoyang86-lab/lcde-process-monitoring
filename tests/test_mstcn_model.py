from __future__ import annotations

import json
import math

# Torch must precede NumPy on the supported Windows CUDA environment.
import torch
import numpy as np
import pytest

import mate_spc_mstcn.model as model
from mate_spc_mstcn.features import DYNAMIC_FEATURE_COUNT, RobustRunTransform


def _run(seed: int, *, fault_class: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(scale=0.25, size=(500, 52)).astype(np.float64)
    if fault_class == 3:
        values[160:, 0] += 1.2
    elif fault_class == 9:
        values[160:, 1] += rng.normal(scale=0.8, size=340)
    elif fault_class == 15:
        wave = np.sign(np.sin(np.arange(340) / 9.0))
        values[160:, 51] += wave
        values[162:, 10] += 0.4 * wave[:-2]
    return values


def _runs() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    runs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for class_id in (0, 3, 9, 15):
        for replicate in range(2):
            name = f"class-{class_id}-run-{replicate}"
            runs[name] = _run(1000 + class_id * 10 + replicate, fault_class=class_id)
            labels[name] = class_id
    return runs, labels


def _small_config(*, q: float = 0.125) -> model.MSTCNConfig:
    return model.MSTCNConfig(
        channels=3,
        dilations=(1,),
        hidden_size=5,
        mil_top_fraction=q,
    )


def _untrained_model(*, q: float = 0.125) -> model.MSTCNModel:
    runs, _ = _runs()
    transform = RobustRunTransform.fit(runs)
    return model.MSTCNModel.from_untrained(
        transform, config=_small_config(q=q), seed=72
    )


def test_frozen_default_architecture_and_q_specific_config() -> None:
    config = model.MSTCNConfig()
    assert config.input_features == DYNAMIC_FEATURE_COUNT == 512
    assert config.window_length == 128
    assert config.channels == 32
    assert config.kernel_size == 3
    assert config.dilations == (1, 2, 4, 8, 16)
    assert config.receptive_field == 125
    assert config.pool_scales == (8, 32, 96)
    assert config.hidden_size == 64
    assert config.class_ids == (0, 3, 9, 15)
    assert config.monitor_start == 160
    assert config.endpoint_stride == 4
    assert config.mil_top_fraction == 0.125
    assert "fault_train_start" not in vars(config)
    assert "early_bag_stop" not in vars(config)
    assert model.MSTCNConfig(mil_top_fraction=0.25).mil_top_fraction == 0.25
    with pytest.raises(ValueError, match="frozen q"):
        model.MSTCNConfig(mil_top_fraction=0.10)
    with pytest.raises(ValueError, match="positive integer"):
        model.MSTCNConfig(channels=True)


def test_training_index_is_run_level_full_bag_without_endpoint_targets() -> None:
    runs, labels = _runs()
    index = model.build_training_index(runs, labels)
    assert index.class_indices.shape == (8,)
    np.testing.assert_array_equal(
        np.bincount(index.class_indices, minlength=4), [2, 2, 2, 2]
    )
    np.testing.assert_array_equal(index.mil_endpoints, np.arange(160, 500, 4))
    assert index.mil_endpoints.size == 85
    assert index.mil_endpoints[-1] == 496
    assert not hasattr(index, "endpoint_labels")
    assert not hasattr(index, "run_indices")
    with pytest.raises(ValueError, match="exact float64"):
        bad = dict(runs)
        bad[next(iter(bad))] = bad[next(iter(bad))].astype(np.float32)
        model.build_training_index(bad, labels)


@pytest.mark.parametrize("q,top_count", [(0.125, 11), (0.25, 22)])
def test_mil_objective_matches_frozen_formula_and_only_top_tail_has_gradient(
    q: float, top_count: int
) -> None:
    generator = torch.Generator().manual_seed(91)
    detection = torch.randn(
        (8, 85), generator=generator, dtype=torch.float64, requires_grad=True
    )
    attribution = torch.randn(
        (8, 85, 3), generator=generator, dtype=torch.float64, requires_grad=True
    )
    classes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    loss, terms = model.mstcn_mil_loss_torch(
        detection, attribution, classes, top_fraction=q
    )

    values, indices = torch.topk(
        detection, top_count, dim=1, largest=True, sorted=False
    )
    evidence = values.mean(dim=1)
    pooled_attribution = torch.gather(
        attribution, 1, indices[:, :, None].expand(-1, -1, 3)
    ).mean(dim=1)
    reference = 0.5 * torch.nn.functional.softplus(evidence[:2]).mean()
    for class_index, rows in enumerate(((2, 3), (4, 5), (6, 7)), start=1):
        reference = reference + (
            torch.nn.functional.softplus(-evidence[list(rows)]).mean() / 6.0
        )
        target = torch.full((2,), class_index - 1, dtype=torch.int64)
        reference = reference + (
            0.25
            * torch.nn.functional.cross_entropy(
                pooled_attribution[list(rows)], target
            )
            / 3.0
        )
    torch.testing.assert_close(loss, reference, rtol=0.0, atol=1.0e-14)
    assert terms["top_count"] == top_count
    loss.backward()
    assert detection.grad is not None
    assert attribution.grad is not None
    assert int(torch.count_nonzero(detection.grad)) == 8 * top_count
    assert int(torch.count_nonzero(attribution.grad[:2])) == 0
    assert int(torch.count_nonzero(attribution.grad[2:])) == 6 * top_count * 3


def test_mil_objective_rejects_nonbalanced_batches() -> None:
    detection = torch.zeros((8, 85), dtype=torch.float64)
    attribution = torch.zeros((8, 85, 3), dtype=torch.float64)
    wrong = torch.tensor([0, 0, 0, 1, 2, 2, 3, 3], dtype=torch.int64)
    with pytest.raises(ValueError, match="exactly two"):
        model.mstcn_mil_loss_torch(
            detection, attribution, wrong, top_fraction=0.125
        )


def test_network_is_endpoint_causal_and_has_two_heads() -> None:
    config = _small_config()
    network = model.build_mstcn_network(config, seed=18, device="cpu")
    generator = torch.Generator().manual_seed(22)
    original = torch.randn(
        (1, 500, DYNAMIC_FEATURE_COUNT),
        generator=generator,
        dtype=torch.float64,
    )
    changed = original.clone()
    changed[:, 251:, :] += 100.0
    endpoints = torch.tensor([160, 200, 250], dtype=torch.int64)
    with torch.no_grad():
        left_detection, left_attribution = network.forward_endpoints(
            original, endpoints
        )
        right_detection, right_attribution = network.forward_endpoints(
            changed, endpoints
        )
    assert left_detection.shape == (1, 3)
    assert left_attribution.shape == (1, 3, 3)
    torch.testing.assert_close(
        left_detection, right_detection, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        left_attribution, right_attribution, rtol=0.0, atol=0.0
    )


def test_whole_sequence_matches_independent_causal_windows() -> None:
    runs, _ = _runs()
    fitted = model.MSTCNModel.from_untrained(
        RobustRunTransform.fit(runs),
        config=model.MSTCNConfig(),
        seed=72,
    )
    error = model.assert_window_sequence_inference_parity(
        fitted,
        _run(300),
        endpoints=np.array([160, 233, 344, 496], dtype=np.int64),
        device="cpu",
        batch_size=2,
        atol=1.0e-10,
    )
    assert error <= 1.0e-10


def test_detection_path_exposes_separate_evidence_and_joint_probabilities() -> None:
    path = model.MSTCNPath(
        np.array([160, 161], dtype=np.int64),
        np.array([-2.0, 2.0], dtype=np.float64),
        np.array([[2.0, 0.0, -1.0], [-1.0, 0.0, 2.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(path.detection_evidence, path.detection_logits)
    assert path.detection_probabilities[0] < 0.5
    assert path.detection_probabilities[1] > 0.5
    assert path.attribution_probabilities.shape == (2, 3)
    assert path.class_probabilities.shape == (2, 4)
    np.testing.assert_allclose(
        path.class_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-15
    )
    assert not path.class_probabilities.flags.writeable


def test_safe_npz_round_trip_binds_complete_q_config(tmp_path) -> None:
    fitted = _untrained_model(q=0.25)
    target = fitted.save_npz(
        tmp_path / "mstcn-q25.npz",
        metadata={"candidate": "q_1_4", "monitor_center": [0.0, 1.0]},
    )
    with np.load(target, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        config_document = json.loads(str(archive["config_json"].item()))
        assert set(config_document) == {
            "feature_schema",
            "input_features",
            "window_length",
            "channels",
            "kernel_size",
            "dilations",
            "pool_scales",
            "hidden_size",
            "class_ids",
            "monitor_start",
            "endpoint_stride",
            "mil_top_fraction",
        }
    restored, metadata = model.load_mstcn_model(target)
    assert restored.config == fitted.config
    assert metadata == {
        "candidate": "q_1_4",
        "monitor_center": [0.0, 1.0],
    }
    for name in fitted.state:
        np.testing.assert_array_equal(restored.state[name], fitted.state[name])
    before = fitted.score_run(_run(400), endpoints=[160, 496])
    after = restored.score_run(_run(400), endpoints=[160, 496])
    np.testing.assert_array_equal(after.detection_logits, before.detection_logits)
    np.testing.assert_array_equal(
        after.attribution_logits, before.attribution_logits
    )


def test_training_configuration_is_complete_and_frozen() -> None:
    config = model.MSTCNTrainingConfig()
    assert config.epochs == 30
    assert config.batch_size == 8
    assert config.learning_rate == 1.0e-3
    assert config.weight_decay == 1.0e-4
    assert config.gradient_clip_norm == 1.0
    with pytest.raises(ValueError, match="frozen at eight"):
        model.MSTCNTrainingConfig(batch_size=4)
    with pytest.raises(ValueError, match="must be even"):
        model.MSTCNTrainingConfig(examples_per_class_per_epoch=3)
