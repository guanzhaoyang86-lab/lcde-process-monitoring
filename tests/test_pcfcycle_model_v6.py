from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mate_spc_pcfcycle.model import (
    BUILD_SEED,
    DENSE_INFERENCE_ENDPOINTS,
    MATURE_FAULT_ENDPOINTS,
    MONITOR_TRAIN_ENDPOINTS,
    PREFIX_ENDPOINTS,
    TRAIN_ENDPOINTS,
    build_pcfcycle_model,
    load_pcfcycle_checkpoint,
    parameter_counts,
    residual_signature,
    save_pcfcycle_checkpoint,
    score_endpoint_pairs,
    state_sha256,
)


EXPECTED_PARAMETER_COUNTS = {
    "g_ab": 695_516,
    "g_ba": 2_832_052,
    "d_a": 179_137,
    "d_b": 171_393,
    "stage2_detector": 88_356,
    "cycle_generators": 3_527_568,
    "cycle_discriminators": 350_530,
    "total": 3_966_454,
}


def _pair(batch: int = 2, *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260910)
    raw = torch.randn((batch, 52, 128), generator=generator, device=device)
    features = torch.randn((batch, 604), generator=generator, device=device)
    return raw, features


def test_exact_architecture_geometry_signature_and_parameter_counts() -> None:
    model = build_pcfcycle_model().eval()
    raw, features = _pair()
    with torch.no_grad():
        output = model(raw, features)
        raw_patches = model.cycle.d_a(raw)
        feature_judgement = model.cycle.d_b(features)

    assert output.cycle.feature_prediction.shape == (2, 604)
    assert output.cycle.raw_prediction.shape == (2, 52, 128)
    assert output.cycle.raw_cycle.shape == raw.shape
    assert output.cycle.feature_cycle.shape == features.shape
    assert raw_patches.shape == (2, 1, 16)
    assert feature_judgement.shape == (2, 1)
    assert output.residual_signature.shape == (2, 656)
    assert output.detector.embedding.shape == (2, 32)
    assert output.detector.detection_logit.shape == (2,)
    assert output.detector.attribution_logits.shape == (2, 3)

    expected_raw = torch.log1p((raw - output.cycle.raw_cycle).square().mean(dim=-1))
    expected_feature = (features - output.cycle.feature_cycle).abs()
    torch.testing.assert_close(output.residual_signature[:, :52], expected_raw)
    torch.testing.assert_close(output.residual_signature[:, 52:], expected_feature)
    torch.testing.assert_close(
        residual_signature(raw, features, output.cycle), output.residual_signature
    )
    assert parameter_counts(model) == EXPECTED_PARAMETER_COUNTS
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_endpoint_contract_and_batched_scorer_restore_training_mode() -> None:
    assert TRAIN_ENDPOINTS == tuple(range(128, 497, 4)) and len(TRAIN_ENDPOINTS) == 93
    assert PREFIX_ENDPOINTS == tuple(range(128, 157, 4)) and len(PREFIX_ENDPOINTS) == 8
    assert MONITOR_TRAIN_ENDPOINTS == tuple(range(160, 497, 4))
    assert len(MONITOR_TRAIN_ENDPOINTS) == 85
    assert MATURE_FAULT_ENDPOINTS == tuple(range(192, 497, 4))
    assert len(MATURE_FAULT_ENDPOINTS) == 77
    assert DENSE_INFERENCE_ENDPOINTS == tuple(range(160, 500))
    assert len(DENSE_INFERENCE_ENDPOINTS) == 340

    model = build_pcfcycle_model()
    model.train()
    model.detector.eval()
    modes_before = tuple(module.training for module in model.modules())
    raw, features = _pair(batch=6)
    raw = raw.reshape(2, 3, 52, 128)
    features = features.reshape(2, 3, 604)
    scored = score_endpoint_pairs(model, raw, features, batch_size=2)
    assert tuple(module.training for module in model.modules()) == modes_before
    assert scored.embedding.shape == (2, 3, 32)
    assert scored.detection_logits.shape == (2, 3)
    assert scored.attribution_logits.shape == (2, 3, 3)

    model.eval()
    with torch.no_grad():
        direct = model(raw.reshape(6, 52, 128), features.reshape(6, 604)).detector
    torch.testing.assert_close(scored.embedding.reshape(6, 32), direct.embedding)
    torch.testing.assert_close(scored.detection_logits.reshape(6), direct.detection_logit)
    torch.testing.assert_close(scored.attribution_logits.reshape(6, 3), direct.attribution_logits)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_endpoint_scorer_rejects_empty_axes_early(shape: tuple[int, int]) -> None:
    runs, endpoints = shape
    model = build_pcfcycle_model()
    raw = torch.empty(runs, endpoints, 52, 128, dtype=torch.float32)
    features = torch.empty(runs, endpoints, 604, dtype=torch.float32)
    with pytest.raises(ValueError, match="nonempty"):
        score_endpoint_pairs(model, raw, features)


def test_builders_are_deterministic_without_consuming_caller_rng() -> None:
    torch.manual_seed(19)
    expected_next = torch.rand(5)
    torch.manual_seed(19)
    first = build_pcfcycle_model(seed=BUILD_SEED)
    observed_next = torch.rand(5)
    second = build_pcfcycle_model(seed=BUILD_SEED)
    different = build_pcfcycle_model(seed=BUILD_SEED + 1)
    torch.testing.assert_close(observed_next, expected_next, rtol=0.0, atol=0.0)
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0.0, atol=0.0)
    assert any(
        not torch.equal(tensor, different.state_dict()[name])
        for name, tensor in first.state_dict().items()
    )


def test_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    model = build_pcfcycle_model().eval()
    raw, features = _pair(batch=1)
    with torch.no_grad():
        expected = model(raw, features).detector.detection_logit
    destination = tmp_path / "pcfcycle-v6.pt"
    metadata = save_pcfcycle_checkpoint(model, destination, metadata={"fixture": True})
    assert metadata["schema"] == "mate_spc_pcfcycle_v6"
    assert metadata["parameter_counts"] == EXPECTED_PARAMETER_COUNTS
    assert metadata["state_sha256"] == state_sha256(model)
    loaded = load_pcfcycle_checkpoint(destination).eval()
    assert state_sha256(loaded) == state_sha256(model)
    with torch.no_grad():
        actual = loaded(raw, features).detector.detection_logit
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_input_guards_fail_closed() -> None:
    model = build_pcfcycle_model()
    raw, features = _pair(batch=1)
    with pytest.raises(ValueError, match="float32"):
        model(raw.double(), features)
    with pytest.raises(ValueError, match=r"\[batch,52,128\]"):
        model(raw[:, :, :-1], features)
    with pytest.raises(ValueError, match=r"\[batch,604\]"):
        model(raw, features[:, :-1])
    with pytest.raises(ValueError, match="finite"):
        bad = features.clone()
        bad[0, 0] = torch.nan
        model(raw, bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_synthetic_forward_compatibility() -> None:
    cpu = build_pcfcycle_model().eval()
    gpu = build_pcfcycle_model(device="cuda").eval()
    gpu.load_state_dict(cpu.state_dict(), strict=True)
    raw, features = _pair(batch=1)
    with torch.no_grad():
        expected = cpu(raw, features)
        actual = gpu(raw.cuda(), features.cuda())
    torch.testing.assert_close(
        actual.residual_signature.cpu(),
        expected.residual_signature,
        rtol=2.0e-3,
        atol=5.0e-4,
    )
    torch.testing.assert_close(
        actual.detector.detection_logit.cpu(),
        expected.detector.detection_logit,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
    torch.testing.assert_close(
        actual.detector.attribution_logits.cpu(),
        expected.detector.attribution_logits,
        rtol=2.0e-4,
        atol=2.0e-4,
    )
