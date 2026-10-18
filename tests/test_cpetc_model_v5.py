from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pytest

from mate_spc_cpetc.features import (
    AETrainingConfig,
    ChannelDenoisingAE,
    ChannelNormalizer,
    extract_inclusive_windows,
    extract_window_features,
    fit_loop_reference,
    loop_features,
)
from mate_spc_cpetc.model import (
    CPETCNet,
    MILTrainingConfig,
    fit_cpetc_model,
    load_cpetc_checkpoint,
    mil_loss,
    save_cpetc_checkpoint,
    score_cpetc_logits,
    _stack_runs,
)


@dataclass(frozen=True)
class DummyRun:
    x: np.ndarray
    source_idv: int


def test_network_shapes_and_channel_identity_are_preserved() -> None:
    torch.manual_seed(1)
    network = CPETCNet().double().eval()
    channels = torch.randn((2, 5, 52, 14), dtype=torch.float64)
    loops = torch.randn((2, 5, 12), dtype=torch.float64)
    baseline = network(channels, loops)
    changed = network(channels.flip(-2), loops)
    assert baseline.shape == (2, 5, 3)
    assert not torch.equal(baseline, changed)


def test_mil_pooling_is_exact_and_other_faults_are_excluded() -> None:
    labels = torch.tensor((0, 3, 9, 15), dtype=torch.long)
    logits = torch.zeros((4, 93, 3), dtype=torch.float64)
    logits[1, 8:, 0] = torch.arange(85, dtype=torch.float64)
    logits[1, :8, 0] = 2.0
    loss, detail = mil_loss(logits, labels)
    positive = torch.arange(76, 85, dtype=torch.float64).mean()
    expected_fault3 = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        positive.reshape(1), torch.ones(1, dtype=torch.float64)
    )
    expected_fault3 += 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    expected_fault3 += 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor((2.0,), dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    zero_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    expected_other = 1.1 * zero_bce
    expected_total = (expected_fault3 + expected_other + expected_other) / 3.0
    assert detail["fault_3_positive_bce"] == pytest.approx(
        float(torch.nn.functional.binary_cross_entropy_with_logits(
            positive.reshape(1), torch.ones(1, dtype=torch.float64)
        ))
    )
    changed = logits.clone()
    changed[2:, :, 0] = 1.0e6
    _, changed_detail = mil_loss(changed, labels)
    for key in ("fault_3_positive_bce", "fault_3_healthy_bce", "fault_3_prefix_bce"):
        assert changed_detail[key] == detail[key]
    torch.testing.assert_close(loss, expected_total)
    assert detail["loss"] == pytest.approx(float(expected_total), rel=0.0, abs=1.0e-15)


def _tiny_runs() -> list[DummyRun]:
    rng = np.random.default_rng(20260912)
    runs: list[DummyRun] = []
    for idv in (1, 3, 9, 15):
        x = rng.normal(size=(500, 52)).astype(np.float64)
        if idv != 1:
            x[160:, idv % 52] += 0.2
        x.setflags(write=False)
        runs.append(DummyRun(x, idv))
    return runs


def test_tiny_end_to_end_fit_score_and_checkpoint(tmp_path: Path) -> None:
    result = fit_cpetc_model(
        _tiny_runs(),
        device="cpu",
        ae_config=AETrainingConfig(epochs=1, batch_size=128),
        mil_config=MILTrainingConfig(epochs=1),
    )
    assert result.model.normalizer.median.dtype == torch.float64
    assert result.model.normalizer.scale.dtype == torch.float64
    assert {parameter.dtype for parameter in result.model.autoencoder.parameters()} == {
        torch.float32
    }
    assert {parameter.dtype for parameter in result.model.network.parameters()} == {torch.float32}
    assert all(not parameter.requires_grad for parameter in result.model.autoencoder.parameters())
    logits = score_cpetc_logits(result, _tiny_runs(), device="cpu")
    assert logits.shape == (4, 93, 3)
    assert logits.dtype == np.float64 and np.isfinite(logits).all()
    destination = tmp_path / "tiny.pt"
    metadata = save_cpetc_checkpoint(result, destination, metadata={"fixture": True})
    assert metadata["schema"] == "mate_spc_cpetc_v5_1"
    loaded = load_cpetc_checkpoint(destination)
    assert loaded.normalizer.median.dtype == torch.float64
    assert loaded.normalizer.scale.dtype == torch.float64
    assert {parameter.dtype for parameter in loaded.autoencoder.parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in loaded.network.parameters()} == {torch.float32}
    replay = score_cpetc_logits(loaded, _tiny_runs(), device="cpu")
    np.testing.assert_array_equal(replay, logits)


def test_float64_source_and_normalizer_preserve_sub_float32_perturbation() -> None:
    time = np.arange(500, dtype=np.float64)[:, None]
    channel = np.arange(52, dtype=np.float64)[None, :]
    # Binary-exact float32 grid: the perturbation is real in float64 but is
    # guaranteed to collapse when the old eager float32 source cast is used.
    baseline = 1.0 + time * (2.0**-18) + channel * (2.0**-20)
    shifted = baseline + 1.0e-10
    assert np.array_equal(baseline.astype(np.float32), shifted.astype(np.float32))

    raw0, _ = _stack_runs((DummyRun(baseline, 1),))
    raw1, _ = _stack_runs((DummyRun(shifted, 1),))
    assert raw0.dtype == torch.float64 and raw1.dtype == torch.float64
    assert bool(torch.any(raw0 != raw1).item())

    fit0 = ChannelNormalizer().fit(raw0)
    fit1 = ChannelNormalizer().fit(raw1)
    delta = fit1.median - fit0.median
    torch.testing.assert_close(
        delta,
        torch.full_like(delta, 1.0e-10),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_loop_float64_compute_and_float32_neural_boundary_shapes() -> None:
    generator = torch.Generator().manual_seed(20260912)
    raw = torch.randn((2, 500, 52), generator=generator, dtype=torch.float64)
    normalizer = ChannelNormalizer().fit(raw)
    normalized = normalizer.transform_runs(raw)
    reference = fit_loop_reference(normalized)
    windows = extract_inclusive_windows(raw)[:, :2]

    direct = loop_features(normalizer.transform_windows(windows[:, 0]), reference)
    assert direct.shape == (2, 12) and direct.dtype == torch.float64
    with pytest.raises(ValueError, match="float64"):
        loop_features(normalizer.transform_windows(windows[:, 0]).float(), reference)

    channels, loops = extract_window_features(
        windows,
        normalizer=normalizer,
        autoencoder=ChannelDenoisingAE(),
        loop_reference=reference,
        batch_size=2,
    )
    assert channels.shape == (2, 2, 52, 14)
    assert loops.shape == (2, 2, 12)
    assert channels.dtype == torch.float32 and loops.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_network_cpu_cuda_forward_parity() -> None:
    torch.manual_seed(81)
    cpu = CPETCNet().float().eval()
    gpu = CPETCNet().float().cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    channels = torch.randn((3, 7, 52, 14))
    loops = torch.randn((3, 7, 12))
    with torch.no_grad():
        expected = cpu(channels, loops)
        observed = gpu(channels.cuda(), loops.cuda()).cpu()
    assert float(torch.max(torch.abs(expected - observed))) <= 1.0e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tiny_full_cuda_fit_and_cpu_replay_parity() -> None:
    result = fit_cpetc_model(
        _tiny_runs(),
        device="cuda",
        ae_config=AETrainingConfig(epochs=1, batch_size=128),
        mil_config=MILTrainingConfig(epochs=1),
    )
    gpu = score_cpetc_logits(result, _tiny_runs(), device="cuda")
    cpu = score_cpetc_logits(result, _tiny_runs(), device="cpu")
    assert result.model.normalizer.median.dtype == torch.float64
    assert result.model.normalizer.scale.dtype == torch.float64
    assert {parameter.dtype for parameter in result.model.autoencoder.parameters()} == {
        torch.float32
    }
    assert {parameter.dtype for parameter in result.model.network.parameters()} == {torch.float32}
    assert float(np.max(np.abs(gpu - cpu))) <= 1.0e-5
