from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mate_spc_pcfcycle.loss import (
    ATTRIBUTION_WEIGHT,
    BALANCED_BCE_WEIGHT,
    PAIRED_RANK_WEIGHT,
    SUPCON_WEIGHT,
    balanced_binary_cross_entropy,
    composite_supervised_loss,
    discriminator_lsgan_loss,
    generator_lsgan_cycle_loss,
    mature_fault_attribution_loss,
    same_run_paired_rank_loss,
    supervised_contrastive_loss,
)
from mate_spc_pcfcycle.model import build_cycle_core


def _pair(batch: int = 1, *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260911)
    raw = torch.randn((batch, 52, 128), generator=generator, device=device)
    features = torch.randn((batch, 604), generator=generator, device=device)
    return raw, features


def test_discriminator_and_generator_losses_have_isolated_gradient_paths() -> None:
    core = build_cycle_core()
    raw, features = _pair()
    discriminator = discriminator_lsgan_loss(core, raw, features)
    torch.testing.assert_close(
        discriminator.total,
        discriminator.components["discriminator_raw"]
        + discriminator.components["discriminator_feature"],
    )
    discriminator.total.backward()
    assert all(parameter.grad is None for parameter in core.g_ab.parameters())
    assert all(parameter.grad is None for parameter in core.g_ba.parameters())
    assert any(parameter.grad is not None for parameter in core.d_a.parameters())
    assert any(parameter.grad is not None for parameter in core.d_b.parameters())

    core.zero_grad(set_to_none=True)
    generator = generator_lsgan_cycle_loss(core, raw, features)
    component_sum = sum(
        value for name, value in generator.components.items() if name != "total"
    )
    torch.testing.assert_close(generator.total, component_sum)
    generator.total.backward()
    assert any(parameter.grad is not None for parameter in core.g_ab.parameters())
    assert any(parameter.grad is not None for parameter in core.g_ba.parameters())
    assert all(parameter.grad is None for parameter in core.d_a.parameters())
    assert all(parameter.grad is None for parameter in core.d_b.parameters())
    assert all(parameter.requires_grad for parameter in core.d_a.parameters())
    assert all(parameter.requires_grad for parameter in core.d_b.parameters())


def test_balanced_bce_is_population_balanced_not_sample_balanced() -> None:
    logits = torch.tensor((-2.0, 0.0, 1.0, 2.0, 3.0), dtype=torch.float32)
    targets = torch.tensor((0, 0, 1, 1, 1), dtype=torch.long)
    observed = balanced_binary_cross_entropy(logits, targets)
    expected = 0.5 * F.binary_cross_entropy_with_logits(
        logits[:2], torch.zeros(2)
    ) + 0.5 * F.binary_cross_entropy_with_logits(logits[2:], torch.ones(3))
    torch.testing.assert_close(observed, expected)


def test_paired_rank_uses_midpoint_median_of_exactly_eight_prefixes() -> None:
    prefix = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    post = torch.full((1, 77), 5.0)
    observed = same_run_paired_rank_loss(post, prefix)
    expected = F.softplus(torch.tensor(1.0 - (5.0 - 3.5)))
    torch.testing.assert_close(observed, expected)
    with pytest.raises(ValueError, match="8 endpoints"):
        same_run_paired_rank_loss(post, prefix[:, :7])
    with pytest.raises(ValueError, match="77 mature"):
        same_run_paired_rank_loss(post[:, :-1], prefix)


def _four_class_embeddings(*, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.zeros((8, 32), dtype=torch.float32)
    for index in range(4):
        embeddings[2 * index : 2 * index + 2, index] = 1.0
    embeddings.requires_grad_(requires_grad)
    labels = torch.tensor((0, 0, 3, 3, 9, 9, 15, 15), dtype=torch.long)
    return embeddings, labels


def test_supervised_contrastive_is_four_class_temperature_point_one() -> None:
    embeddings, labels = _four_class_embeddings()
    observed = supervised_contrastive_loss(embeddings, labels)
    # Every anchor has one same-class logit exp(10) and six orthogonal logits
    # exp(0), so the exact common negative log-probability is known.
    expected = torch.log(torch.exp(torch.tensor(10.0)) + 6.0) - 10.0
    torch.testing.assert_close(observed, expected, rtol=1.0e-5, atol=1.0e-6)
    with pytest.raises(ValueError, match="four classes"):
        supervised_contrastive_loss(embeddings[:6], labels[:6])


def test_attribution_maps_fault_ids_to_head_order_3_9_15() -> None:
    logits = torch.tensor(
        ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        dtype=torch.float32,
    )
    ids = torch.tensor((3, 9, 15))
    observed = mature_fault_attribution_loss(logits, ids)
    expected = F.cross_entropy(logits, torch.tensor((0, 1, 2)))
    torch.testing.assert_close(observed, expected)


def test_composite_supervised_loss_has_exact_weights_and_backpropagates() -> None:
    detection = torch.linspace(-1.0, 1.0, 8, requires_grad=True)
    binary = torch.tensor((0, 0, 0, 0, 1, 1, 1, 1))
    post = torch.full((3, 77), 1.5, requires_grad=True)
    prefix = torch.zeros((3, 8), requires_grad=True)
    embeddings, labels = _four_class_embeddings(requires_grad=True)
    attribution = torch.randn((6, 3), generator=torch.Generator().manual_seed(7))
    attribution.requires_grad_(True)
    fault_ids = torch.tensor((3, 3, 9, 9, 15, 15))
    output = composite_supervised_loss(
        detection_logits=detection,
        binary_targets=binary,
        post_logits=post,
        prefix_logits=prefix,
        embeddings=embeddings,
        contrastive_class_labels=labels,
        attribution_logits=attribution,
        attribution_fault_ids=fault_ids,
    )
    expected = (
        BALANCED_BCE_WEIGHT * output.components["balanced_bce"]
        + PAIRED_RANK_WEIGHT * output.components["paired_rank"]
        + SUPCON_WEIGHT * output.components["supervised_contrastive"]
        + ATTRIBUTION_WEIGHT * output.components["mature_fault_attribution"]
    )
    torch.testing.assert_close(output.total, expected)
    output.total.backward()
    for value in (detection, post, prefix, embeddings, attribution):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_losses_are_cuda_compatible() -> None:
    core = build_cycle_core(device="cuda")
    raw, features = _pair(device="cuda")
    generator = generator_lsgan_cycle_loss(core, raw, features)
    assert generator.total.is_cuda and bool(torch.isfinite(generator.total).item())
    generator.total.backward()

    embeddings, labels = _four_class_embeddings(requires_grad=True)
    gpu_embeddings = embeddings.detach().cuda().requires_grad_(True)
    contrastive = supervised_contrastive_loss(gpu_embeddings, labels.cuda())
    assert contrastive.is_cuda and bool(torch.isfinite(contrastive).item())
    contrastive.backward()
    assert gpu_embeddings.grad is not None
