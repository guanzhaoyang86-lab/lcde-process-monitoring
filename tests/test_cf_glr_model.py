from __future__ import annotations

from mate_spc_cf_glr import model

import json

import numpy as np
import pytest


def _feature_batches(
    run_count: int,
    *,
    seed: int = 13,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        fault_id: rng.normal(size=(run_count, 500, dimension)).astype(np.float64)
        for fault_id, dimension in zip(
            model.FAULT_IDS, model.EXPERT_FEATURE_DIMS, strict=True
        )
    }


def test_frozen_config_geometry_and_strict_fault_ids() -> None:
    config = model.CFGLRConfig()
    assert config.feature_schema == "cf_glr_mechanism_features_v2"
    assert config.expert_feature_dims == (20, 20, 28)
    assert len(config.main_endpoints) == 85
    assert config.main_endpoints[0] == 160
    assert config.main_endpoints[-1] == 496
    assert config.prefix_endpoints == tuple(range(128, 160, 4))
    assert config.top_count == 9
    with pytest.raises(ValueError, match="feature_schema"):
        model.CFGLRConfig(feature_schema="cf_glr_mechanism_features_v1")
    with pytest.raises(ValueError, match="fault_id"):
        config.feature_dim(3.0)
    initialized = model.initialize_cf_glr_model()
    bad = _feature_batches(1)
    bad[3.0] = bad.pop(3)
    with pytest.raises(ValueError, match="fault id"):
        initialized.score_runs(bad)
    bad_bool = _feature_batches(1)
    bad_bool[True] = bad_bool.pop(3)
    with pytest.raises(ValueError, match="fault id"):
        initialized.score_runs(bad_bool)


def test_architecture_members_and_dense_score_shape() -> None:
    fitted = model.initialize_cf_glr_model()
    assert tuple(len(fitted.states[fault_id]) for fault_id in model.FAULT_IDS) == (1, 1, 3)
    assert fitted.states[3][0]["head.weight"].shape == (1, 20)
    assert fitted.states[15][0]["input.weight"].shape == (16, 28)
    features = _feature_batches(2)
    paths = fitted.score_runs(features, device="cpu", batch_size=127)
    assert paths.logits.shape == (2, 500, 3)
    assert paths.logits.dtype == np.float64
    assert not paths.logits.flags.writeable
    assert paths.for_fault(15).shape == (2, 500)
    one = fitted.score_run(
        {fault_id: values[0] for fault_id, values in features.items()}
    )
    np.testing.assert_allclose(one, paths.logits[0], rtol=0.0, atol=2.0e-15)
    assert not one.flags.writeable


def test_fit_feature_standardizers_is_balanced_and_uses_only_frozen_endpoints() -> None:
    config = model.CFGLRConfig()
    healthy: dict[int, np.ndarray] = {}
    faults: dict[int, np.ndarray] = {}
    endpoints = np.asarray(config.prefix_endpoints + config.main_endpoints)
    for fault_id, dimension in zip(
        config.fault_ids, config.expert_feature_dims, strict=True
    ):
        h = np.full((2, 500, dimension), 1.0e9, dtype=np.float64)
        f = np.full((2, 500, dimension), -1.0e9, dtype=np.float64)
        h[0, endpoints, :] = 1.0
        h[1, endpoints, :] = 3.0
        f[0, endpoints, :] = 5.0
        f[1, endpoints, :] = 7.0
        h[:, endpoints, 0] = 4.0
        f[:, endpoints, 0] = 4.0
        healthy[fault_id] = h
        faults[fault_id] = f
    centers, scales = model.fit_feature_standardizers(healthy, faults)
    for fault_id in config.fault_ids:
        np.testing.assert_allclose(centers[fault_id], 4.0, rtol=0.0, atol=0.0)
        assert scales[fault_id][0] == model.FEATURE_SCALE_FLOOR
        np.testing.assert_allclose(
            scales[fault_id][1:],
            2.0 * model.FEATURE_MAD_CONSISTENCY,
            rtol=0.0,
            atol=1.0e-15,
        )
        assert not centers[fault_id].flags.writeable
        assert not scales[fault_id].flags.writeable
    unbalanced = dict(faults)
    unbalanced[15] = unbalanced[15][:1]
    with pytest.raises(ValueError, match="same run count"):
        model.fit_feature_standardizers(healthy, unbalanced)


def test_binary_mil_loss_is_exact_top_ten_percent_with_prefix_auxiliary() -> None:
    torch = model._torch_module()
    healthy = torch.linspace(-2.0, 2.0, 170, dtype=torch.float64).reshape(2, 85)
    fault = torch.linspace(2.0, -2.0, 170, dtype=torch.float64).reshape(2, 85)
    prefix = torch.zeros((4, 8), dtype=torch.float64)
    healthy.requires_grad_()
    fault.requires_grad_()
    prefix.requires_grad_()
    parameter = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    loss, terms = model.binary_mil_loss_torch(
        healthy, fault, prefix, (parameter,)
    )
    healthy_bag = torch.topk(healthy, 9, dim=1).values.mean(dim=1)
    fault_bag = torch.topk(fault, 9, dim=1).values.mean(dim=1)
    expected = (
        0.5
        * torch.nn.functional.binary_cross_entropy_with_logits(
            healthy_bag, torch.zeros_like(healthy_bag)
        )
        + 0.5
        * torch.nn.functional.binary_cross_entropy_with_logits(
            fault_bag, torch.ones_like(fault_bag)
        )
        + 0.1
        * torch.nn.functional.binary_cross_entropy_with_logits(
            prefix.reshape(-1), torch.zeros(32, dtype=torch.float64)
        )
        + 1.0e-4 * torch.square(parameter).sum()
    )
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
    assert terms["top_count"] == 9
    loss.backward()
    assert int(torch.count_nonzero(healthy.grad)) == 18
    assert int(torch.count_nonzero(fault.grad)) == 18
    assert int(torch.count_nonzero(prefix.grad)) == 32
    assert parameter.grad is not None


def test_safe_npz_round_trip_binds_standardizers_and_manifest(tmp_path) -> None:
    initialized = model.initialize_cf_glr_model()
    centers = {
        fault_id: np.linspace(0.0, 1.0, dimension, dtype=np.float64)
        for fault_id, dimension in zip(
            model.FAULT_IDS, model.EXPERT_FEATURE_DIMS, strict=True
        )
    }
    scales = {
        fault_id: np.linspace(1.0, 2.0, dimension, dtype=np.float64)
        for fault_id, dimension in zip(
            model.FAULT_IDS, model.EXPERT_FEATURE_DIMS, strict=True
        )
    }
    fitted = model.CFGLRModel(initialized.config, initialized.states, centers, scales)
    target = fitted.save_npz(tmp_path / "cf-glr-v4.npz", metadata={"fold": 2})
    with np.load(target, allow_pickle=False) as archive:
        assert str(archive["schema"].item()) == model.MODEL_SCHEMA
        manifest = json.loads(str(archive["manifest_json"].item()))
        assert manifest["schema"] == "mate_spc_cf_glr_model_manifest_v1"
        assert len(manifest["feature_standardizer_entries"]) == 6
        assert (
            manifest["feature_standardization"]["scale_floor"]
            == model.FEATURE_SCALE_FLOOR
        )
    restored, metadata = model.load_cf_glr_model(target)
    assert metadata == {"fold": 2}
    features = _feature_batches(1)
    np.testing.assert_array_equal(
        fitted.score_runs(features).logits,
        restored.score_runs(features).logits,
    )
    for fault_id in model.FAULT_IDS:
        np.testing.assert_array_equal(
            restored.feature_centers[fault_id], centers[fault_id]
        )
        np.testing.assert_array_equal(restored.feature_scales[fault_id], scales[fault_id])


def test_training_contract_has_no_tunable_optimizer_fields() -> None:
    training = model.CFGLRTrainingConfig()
    assert training.epochs == 80
    assert training.learning_rate == 1.0e-3
    assert training.l2_coefficient == 1.0e-4
    assert training.prefix_negative_weight == 0.1
    assert training.device == "cuda"
    assert training.dtype == "float64"
    assert training.optimizer == "AdamW"
    assert "batch_size" not in vars(training)
    assert "gradient_clip" not in vars(training)
    with pytest.raises(ValueError, match="epochs"):
        model.CFGLRTrainingConfig(epochs=79)
    with pytest.raises(ValueError, match="learning_rate"):
        model.CFGLRTrainingConfig(learning_rate="0.001")  # type: ignore[arg-type]


@pytest.mark.skipif(
    model._torch is None or not model._torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fit_and_dense_cpu_cuda_parity_on_synthetic_bags() -> None:
    healthy = _feature_batches(1, seed=101)
    faults = _feature_batches(1, seed=102)
    for fault_id in model.FAULT_IDS:
        faults[fault_id][:, 160:, 0] += 2.0
    result = model.fit_cf_glr_cuda(healthy, faults)
    assert len(result.final_losses) == 5
    assert all(np.isfinite(value) for value in result.final_losses.values())
    error = model.assert_cpu_cuda_inference_parity(result.model, healthy)
    assert error <= model.DEFAULT_PARITY_ATOL
