from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

# The project's RTX-5090 Windows environment requires Torch to own DLL/OpenMP
# load order before NumPy.
torch = pytest.importorskip("torch")
import numpy as np

from mate_spc_cris_v9.model import (
    ADAMW_LEARNING_RATE,
    ADAMW_WEIGHT_DECAY,
    CONTEXT_LENGTH,
    CRISConfig,
    CRISModel,
    CRISTrainingConfig,
    DIAGONAL_SCALE_SEMANTICS,
    FORECAST_HORIZONS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    _network_class,
)


def _synthetic_runs(
    *, n_runs: int = 4, length: int = 96, readonly: bool = True
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    x_runs: list[np.ndarray] = []
    c_runs: list[np.ndarray] = []
    mask_runs: list[np.ndarray] = []
    delta_runs: list[np.ndarray] = []
    for run_index in range(n_runs):
        rng = np.random.default_rng(900 + run_index)
        time = np.arange(length, dtype=np.float32)
        c = np.column_stack(
            (
                np.sin(time / 9.0 + run_index * 0.1),
                np.cos(time / 13.0 - run_index * 0.05),
            )
        ).astype(np.float32)
        x = np.zeros((length, 2), dtype=np.float32)
        x[0] = (0.1 * run_index, -0.05 * run_index)
        for index in range(1, length):
            noise = rng.normal(0.0, 0.008, size=2)
            x[index, 0] = 0.78 * x[index - 1, 0] + 0.32 * c[index, 0] + noise[0]
            x[index, 1] = 0.70 * x[index - 1, 1] - 0.25 * c[index, 1] + noise[1]
        mask = np.ones_like(x, dtype=bool)
        delta = np.ones_like(x, dtype=np.float32)
        if readonly:
            for array in (x, c, mask, delta):
                array.setflags(write=False)
        x_runs.append(x)
        c_runs.append(c)
        mask_runs.append(mask)
        delta_runs.append(delta)
    return x_runs, c_runs, mask_runs, delta_runs


@pytest.fixture(scope="module")
def fitted_bundle() -> tuple[
    CRISModel,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    x, c, mask, delta = _synthetic_runs()
    config = CRISConfig(target_dim=2, covariate_dim=2, delta_time_dim=2)
    training = CRISTrainingConfig(epochs=12, batch_size=256, device="cpu", seed=2718)
    model = CRISModel.fit(
        x,
        c,
        mask,
        delta,
        config=config,
        training_config=training,
        metadata={"purpose": "unit-test"},
    )
    return model, x, c, mask, delta


def test_frozen_architecture_is_two_layer_hidden64_diagonal_student_t() -> None:
    config = CRISConfig(target_dim=3, covariate_dim=4, delta_time_dim=1)
    network = _network_class()(config)
    assert network.gru.num_layers == NUM_LAYERS == 2
    assert network.gru.hidden_size == HIDDEN_SIZE == 64
    assert network.gru.input_size == 3 + 4 + 3 + 1
    assert network.head.out_features == len(FORECAST_HORIZONS) * 3 * 2
    spec = config.architecture_spec()
    assert spec["context_length"] == CONTEXT_LENGTH == 60
    assert spec["horizons"] == [1, 5, 15]
    assert spec["degrees_of_freedom"] == 5.0
    assert spec["distribution_head"] == {
        "parameters": ["mu", "log_scale"],
        "scale_semantics": DIAGONAL_SCALE_SEMANTICS,
        "low_rank_covariance": False,
    }
    assert ADAMW_LEARNING_RATE == 3.0e-4
    assert ADAMW_WEIGHT_DECAY == 1.0e-4


def test_fixed_epoch_training_loss_decreases_on_synthetic_dynamics(fitted_bundle) -> None:
    model = fitted_bundle[0]
    assert model.loss_history.shape == (model.training_config.epochs,)
    assert np.isfinite(model.loss_history).all()
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.source_device == "cpu"
    assert model.attestation.dtype == "float32"
    assert model.attestation.deterministic_algorithms is True


def test_prediction_alignment_finite_positive_and_readonly(fitted_bundle) -> None:
    model, x, c, mask, delta = fitted_bundle
    prediction = model.predict(x[0], c[0], mask[0], delta[0], device="cpu", batch_size=7)
    expected_shape = (x[0].shape[0], 3, 2)
    assert prediction.mu.shape == expected_shape
    assert prediction.scale.shape == expected_shape
    assert prediction.valid.shape == expected_shape
    for horizon_index, horizon in enumerate(FORECAST_HORIZONS):
        first_target = CONTEXT_LENGTH - 1 + horizon
        assert not prediction.valid[:first_target, horizon_index].any()
        assert prediction.valid[first_target:, horizon_index].all()
    assert np.isfinite(prediction.mu[prediction.valid]).all()
    assert np.isfinite(prediction.scale[prediction.valid]).all()
    assert np.all(prediction.scale[prediction.valid] > 0.0)
    assert np.isnan(prediction.mu[~prediction.valid]).all()
    assert np.isnan(prediction.scale[~prediction.valid]).all()
    assert np.isfinite(prediction.log_scale[prediction.valid]).all()
    assert prediction.source_device == "cpu"
    assert not prediction.mu.flags.writeable
    assert not prediction.scale.flags.writeable
    assert not prediction.valid.flags.writeable


def test_causal_prefix_prediction_never_uses_values_after_anchor(fitted_bundle) -> None:
    model, x, c, mask, delta = fitted_bundle
    base = model.predict(x[0], c[0], mask[0], delta[0])
    cutoff = 72
    altered_x = np.array(x[0], copy=True)
    altered_c = np.array(c[0], copy=True)
    altered_mask = np.array(mask[0], copy=True)
    altered_delta = np.array(delta[0], copy=True)
    altered_x[cutoff + 1 :] += 100.0
    altered_c[cutoff + 1 :] -= 50.0
    altered_mask[cutoff + 1 :, 0] = False
    altered_delta[cutoff + 1 :] = 20.0
    altered = model.predict(altered_x, altered_c, altered_mask, altered_delta)
    target_indices = np.arange(x[0].shape[0])
    for horizon_index, horizon in enumerate(FORECAST_HORIZONS):
        causal = base.valid[:, horizon_index, 0] & (target_indices - horizon <= cutoff)
        np.testing.assert_array_equal(
            base.mu[causal, horizon_index], altered.mu[causal, horizon_index]
        )
        np.testing.assert_array_equal(
            base.scale[causal, horizon_index], altered.scale[causal, horizon_index]
        )


def test_innovations_follow_unit_variance_student_t_standardization(fitted_bundle) -> None:
    model, x, c, mask, delta = fitted_bundle
    prediction = model.predict(x[0], c[0], mask[0], delta[0])
    innovations = model.innovations(
        x[0], mask=mask[0], prediction=prediction
    )
    expected = np.full(prediction.mu.shape, np.nan, dtype=np.float32)
    broadcast_x = np.broadcast_to(x[0][:, None, :], prediction.mu.shape)
    expected[prediction.valid] = (
        np.sqrt((5.0 - 2.0) / 5.0)
        * (broadcast_x[prediction.valid] - prediction.mu[prediction.valid])
        / prediction.scale[prediction.valid]
    )
    np.testing.assert_allclose(
        innovations[prediction.valid], expected[prediction.valid], rtol=2.0e-6, atol=2.0e-6
    )
    assert np.isnan(innovations[~prediction.valid]).all()
    assert not innovations.flags.writeable


def test_no_pickle_npz_round_trip_preserves_cpu_predictions(
    fitted_bundle, tmp_path: Path
) -> None:
    model, x, c, mask, delta = fitted_bundle
    before = model.predict(x[1], c[1], mask[1], delta[1], batch_size=9)
    path = model.save_npz(tmp_path / "cris-v9.npz", metadata={"fold": 3})
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        architecture = json.loads(str(archive["architecture_json"].item()))
        assert architecture["distribution_head"]["low_rank_covariance"] is False
        assert architecture["distribution_head"]["scale_semantics"] == (
            DIAGONAL_SCALE_SEMANTICS
        )
    restored = CRISModel.load_npz(path)
    after = restored.predict(x[1], c[1], mask[1], delta[1], batch_size=11)
    np.testing.assert_array_equal(before.valid, after.valid)
    # Different inference batch partitions can change float32 GEMM accumulation
    # by a few ULPs even though every persisted tensor is byte-identical.
    np.testing.assert_allclose(
        before.mu[before.valid], after.mu[after.valid], atol=1.0e-6, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        before.scale[before.valid], after.scale[after.valid], atol=1.0e-6, rtol=1.0e-6
    )
    assert dict(restored.metadata) == {"fold": 3, "purpose": "unit-test"}
    assert not restored.normalization.target_center.flags.writeable
    assert all(not value.flags.writeable for value in restored.state_dict.values())


def test_npz_rejects_broadcastable_normalization_tamper(
    fitted_bundle, tmp_path: Path
) -> None:
    model = fitted_bundle[0]
    canonical = model.save_npz(tmp_path / "canonical.npz")
    with np.load(canonical, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["target_scale"] = np.ones(1, dtype=np.float32)
    damaged = tmp_path / "broadcast-scale.npz"
    np.savez_compressed(damaged, **payload)
    with pytest.raises(ValueError, match="target_scale has wrong dtype or shape"):
        CRISModel.load_npz(damaged)


def test_readonly_numpy_inputs_are_owned_before_torch(fitted_bundle) -> None:
    model, x, c, mask, delta = fitted_bundle
    assert all(not run.flags.writeable for runs in (x, c, mask, delta) for run in runs)
    prediction = model.predict(x[2], c[2], mask[2], delta[2])
    assert prediction.valid.any()
    # Fit itself received these same read-only arrays in the module fixture;
    # reaching this assertion under ``-W error`` proves no unsafe alias warning.
    assert np.isfinite(model.loss_history).all()


def test_masked_values_are_invariant_even_when_nan_or_extreme(fitted_bundle) -> None:
    model, x, c, _mask, delta = fitted_bundle
    first_x = np.array(x[0], copy=True)
    second_x = np.array(x[0], copy=True)
    mask = np.ones_like(first_x, dtype=bool)
    masked_rows = np.array([5, 18, 44, 70])
    mask[masked_rows, 0] = False
    first_x[masked_rows, 0] = np.nan
    second_x[masked_rows, 0] = 1.0e6
    first = model.predict(first_x, c[0], mask, delta[0])
    second = model.predict(second_x, c[0], mask, delta[0])
    np.testing.assert_array_equal(first.valid, second.valid)
    np.testing.assert_array_equal(first.mu[first.valid], second.mu[second.valid])
    np.testing.assert_array_equal(first.scale[first.valid], second.scale[second.valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_inference_parity_and_concrete_device_attestation(fitted_bundle) -> None:
    model, x, c, mask, delta = fitted_bundle
    cpu = model.predict(x[3], c[3], mask[3], delta[3], device="cpu", batch_size=13)
    cuda = model.predict(x[3], c[3], mask[3], delta[3], device="cuda", batch_size=13)
    np.testing.assert_array_equal(cpu.valid, cuda.valid)
    np.testing.assert_allclose(cpu.mu[cpu.valid], cuda.mu[cuda.valid], rtol=2.0e-4, atol=2.0e-4)
    np.testing.assert_allclose(
        cpu.scale[cpu.valid], cuda.scale[cuda.valid], rtol=2.0e-4, atol=2.0e-4
    )
    assert cuda.source_device.startswith("cuda:")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fresh_process_load_cuda_predict_reenables_determinism(
    fitted_bundle, tmp_path: Path
) -> None:
    model, x, c, mask, delta = fitted_bundle
    model_path = model.save_npz(tmp_path / "fresh-process.npz")
    input_path = tmp_path / "fresh-input.npz"
    np.savez_compressed(
        input_path,
        x=x[0],
        c=c[0],
        mask=mask[0],
        delta=delta[0],
    )
    root = Path(__file__).resolve().parents[1]
    program = """
import json, os, sys
import torch
import numpy as np
sys.path.insert(0, sys.argv[1])
from mate_spc_cris_v9.model import CRISModel
torch.use_deterministic_algorithms(False)
model = CRISModel.load_npz(sys.argv[2])
with np.load(sys.argv[3], allow_pickle=False) as data:
    first = model.predict(data['x'], data['c'], data['mask'], data['delta'], device='cuda')
    second = model.predict(data['x'], data['c'], data['mask'], data['delta'], device='cuda')
print(json.dumps({
    'deterministic': torch.are_deterministic_algorithms_enabled(),
    'workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
    'source': first.source_device,
    'repeat_exact': bool(np.array_equal(first.mu, second.mu, equal_nan=True)),
}))
"""
    environment = dict(os.environ)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(root / "src"),
            str(model_path),
            str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    evidence = json.loads(completed.stdout.strip())
    assert evidence == {
        "deterministic": True,
        "workspace": ":4096:8",
        "source": "cuda:0",
        "repeat_exact": True,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fit_is_deterministic_and_attested() -> None:
    x, c, mask, delta = _synthetic_runs(n_runs=2, length=78)
    config = CRISConfig(target_dim=2, covariate_dim=2, delta_time_dim=2)
    training = CRISTrainingConfig(epochs=2, batch_size=64, device="cuda", seed=31415)
    first = CRISModel.fit(
        x, c, mask, delta, config=config, training_config=training
    )
    second = CRISModel.fit(
        x, c, mask, delta, config=config, training_config=training
    )
    for name in first.state_dict:
        np.testing.assert_array_equal(first.state_dict[name], second.state_dict[name])
    np.testing.assert_array_equal(first.loss_history, second.loss_history)
    assert first.source_device.startswith("cuda:")
    assert first.attestation.cuda_runtime
    assert first.attestation.cuda_device_name


def test_invalid_training_hyperparameters_and_covariance_claim_fail_closed() -> None:
    with pytest.raises(ValueError, match="learning_rate is frozen"):
        CRISTrainingConfig(learning_rate=1.0e-3)
    with pytest.raises(ValueError, match="two-layer GRU"):
        CRISConfig(target_dim=1, covariate_dim=1, hidden_size=32)
    with pytest.raises(ValueError, match="horizons must be exactly"):
        CRISConfig(target_dim=1, covariate_dim=1, horizons=(1, 5))
