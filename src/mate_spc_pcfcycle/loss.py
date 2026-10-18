"""Protocol-bound losses for the two PCF-CycleRank v6 training stages."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import EMBEDDING_DIM, NETWORK_DTYPE, PCFCycleCore


CYCLE_RAW_WEIGHT = 10.0
CYCLE_FEATURE_WEIGHT = 10.0
PAIRED_FEATURE_WEIGHT = 5.0
PAIRED_RAW_WEIGHT = 5.0
BALANCED_BCE_WEIGHT = 0.40
PAIRED_RANK_WEIGHT = 0.30
SUPCON_WEIGHT = 0.20
ATTRIBUTION_WEIGHT = 0.10
SUPCON_TEMPERATURE = 0.10
FAULT_IDS = (3, 9, 15)
SUPCON_CLASS_IDS = (0, 3, 9, 15)


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    components: Mapping[str, Tensor]


def _require_float32(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != NETWORK_DTYPE:
        raise ValueError(f"{name} must use torch.float32")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_core(value: Any) -> PCFCycleCore:
    if not isinstance(value, PCFCycleCore):
        raise TypeError("core must be a PCFCycleCore")
    return value


@contextmanager
def _frozen_parameters(modules: Sequence[nn.Module]) -> Iterator[None]:
    parameters = [parameter for module in modules for parameter in module.parameters()]
    original = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original, strict=True):
            parameter.requires_grad_(requires_grad)


def discriminator_lsgan_loss(core: Any, raw: Any, features: Any) -> LossOutput:
    """Return the discriminator step; generated values are detached exactly once."""

    model = _require_core(core)
    a = _require_float32(raw, name="A")
    bfeat = _require_float32(features, name="Bfeat")
    # Shape/device validation is centralized in the complete core call.  Its
    # outputs are not retained, so the discriminator step does not build a
    # generator graph.
    with torch.no_grad():
        fake_features = model.g_ab(a)
        fake_raw = model.g_ba(bfeat)

    raw_real = model.d_a(a)
    raw_fake = model.d_a(fake_raw.detach())
    feature_real = model.d_b(bfeat)
    feature_fake = model.d_b(fake_features.detach())
    raw_domain = 0.5 * ((raw_real - 1.0).square().mean() + raw_fake.square().mean())
    feature_domain = 0.5 * (
        (feature_real - 1.0).square().mean() + feature_fake.square().mean()
    )
    total = raw_domain + feature_domain
    return LossOutput(
        total,
        {
            "discriminator_raw": raw_domain,
            "discriminator_feature": feature_domain,
            "total": total,
        },
    )


def generator_lsgan_cycle_loss(core: Any, raw: Any, features: Any) -> LossOutput:
    """Return the generator step with exact adversarial, cycle, and paired weights."""

    model = _require_core(core)
    a = _require_float32(raw, name="A")
    bfeat = _require_float32(features, name="Bfeat")
    fake_features = model.g_ab(a)
    fake_raw = model.g_ba(bfeat)
    raw_cycle = model.g_ba(fake_features)
    feature_cycle = model.g_ab(fake_raw)

    # Discriminator parameters are temporarily frozen while gradients still
    # propagate through their operations to both generators.
    with _frozen_parameters((model.d_a, model.d_b)):
        adversarial_raw = 0.5 * (model.d_a(fake_raw) - 1.0).square().mean()
        adversarial_feature = 0.5 * (model.d_b(fake_features) - 1.0).square().mean()

    cycle_raw = CYCLE_RAW_WEIGHT * F.l1_loss(raw_cycle, a)
    cycle_feature = CYCLE_FEATURE_WEIGHT * F.l1_loss(feature_cycle, bfeat)
    paired_feature = PAIRED_FEATURE_WEIGHT * F.l1_loss(fake_features, bfeat)
    paired_raw = PAIRED_RAW_WEIGHT * F.l1_loss(fake_raw, a)
    total = (
        adversarial_raw
        + adversarial_feature
        + cycle_raw
        + cycle_feature
        + paired_feature
        + paired_raw
    )
    return LossOutput(
        total,
        {
            "adversarial_raw": adversarial_raw,
            "adversarial_feature": adversarial_feature,
            "cycle_raw_weighted": cycle_raw,
            "cycle_feature_weighted": cycle_feature,
            "paired_feature_weighted": paired_feature,
            "paired_raw_weighted": paired_raw,
            "total": total,
        },
    )


def balanced_binary_cross_entropy(logits: Any, targets: Any) -> Tensor:
    """Give the healthy and pooled-fault populations exactly one half each."""

    values = _require_float32(logits, name="detection_logits")
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.squeeze(1)
    if values.ndim != 1:
        raise ValueError("detection_logits must have shape [samples] or [samples,1]")
    if not isinstance(targets, Tensor) or targets.ndim != 1 or targets.shape != values.shape:
        raise ValueError("binary_targets must align with detection_logits")
    labels = targets.to(device=values.device)
    if not bool(torch.all((labels == 0) | (labels == 1)).item()):
        raise ValueError("binary_targets must contain only zero and one")
    healthy = values[labels == 0]
    faulty = values[labels == 1]
    if healthy.numel() == 0 or faulty.numel() == 0:
        raise ValueError("balanced BCE requires both healthy and fault samples")
    return 0.5 * F.binary_cross_entropy_with_logits(healthy, torch.zeros_like(healthy)) + 0.5 * F.binary_cross_entropy_with_logits(
        faulty, torch.ones_like(faulty)
    )


def _row_midpoint_median(values: Tensor) -> Tensor:
    ordered = values.sort(dim=1).values
    count = ordered.shape[1]
    midpoint = count // 2
    if count % 2:
        return ordered[:, midpoint]
    return 0.5 * (ordered[:, midpoint - 1] + ordered[:, midpoint])


def same_run_paired_rank_loss(post_logits: Any, prefix_logits: Any) -> Tensor:
    """Rank every mature post-fault logit above its same-run prefix median by one."""

    post = _require_float32(post_logits, name="post_logits")
    prefix = _require_float32(prefix_logits, name="prefix_logits")
    if post.ndim != 2 or prefix.ndim != 2:
        raise ValueError("post_logits and prefix_logits must have shape [runs,endpoints]")
    if post.shape[0] != prefix.shape[0] or post.shape[0] < 1:
        raise ValueError("post_logits and prefix_logits must contain the same runs")
    if post.shape[1] != 77:
        raise ValueError("post_logits must contain the 77 mature endpoints 192..496 step 4")
    if prefix.shape[1] != 8:
        raise ValueError("prefix_logits must contain the 8 endpoints 128..156 step 4")
    baseline = _row_midpoint_median(prefix).unsqueeze(1)
    return F.softplus(1.0 - (post - baseline)).mean()


def supervised_contrastive_loss(
    embeddings: Any,
    class_labels: Any,
    *,
    temperature: float = SUPCON_TEMPERATURE,
) -> Tensor:
    """Four-class supervised contrastive loss over normal/IDV3/IDV9/IDV15."""

    values = _require_float32(embeddings, name="embeddings")
    if values.ndim != 2 or values.shape[1] != EMBEDDING_DIM:
        raise ValueError("embeddings must have shape [samples,32]")
    if not isinstance(class_labels, Tensor) or class_labels.ndim != 1:
        raise ValueError("class_labels must have shape [samples]")
    if class_labels.shape[0] != values.shape[0]:
        raise ValueError("class_labels must align with embeddings")
    if not isinstance(temperature, (float, int)) or isinstance(temperature, bool):
        raise TypeError("temperature must be numeric")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("temperature must be finite and positive")
    labels = class_labels.to(device=values.device, dtype=torch.long)
    observed = set(int(item) for item in labels.detach().cpu().tolist())
    if observed != set(SUPCON_CLASS_IDS):
        raise ValueError("SupCon requires exactly the four classes 0/3/9/15")
    for class_id in SUPCON_CLASS_IDS:
        if int((labels == class_id).sum().item()) < 2:
            raise ValueError("SupCon requires at least two samples from every class")

    normalized = F.normalize(values, p=2.0, dim=1, eps=1.0e-12)
    similarities = normalized @ normalized.transpose(0, 1) / tau
    sample_count = values.shape[0]
    self_mask = torch.eye(sample_count, dtype=torch.bool, device=values.device)
    denominator_logits = similarities.masked_fill(self_mask, -torch.inf)
    log_denominator = torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    log_probabilities = similarities - log_denominator
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    per_anchor = -(log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)) / positive_count
    return per_anchor.mean()


def mature_fault_attribution_loss(logits: Any, fault_ids: Any) -> Tensor:
    """Three-way attribution CE, defined only on mature fault instances."""

    values = _require_float32(logits, name="attribution_logits")
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("attribution_logits must have shape [samples,3]")
    if not isinstance(fault_ids, Tensor) or fault_ids.ndim != 1:
        raise ValueError("attribution_fault_ids must have shape [samples]")
    if fault_ids.shape[0] != values.shape[0] or values.shape[0] < 1:
        raise ValueError("attribution fault IDs must align and be nonempty")
    ids = fault_ids.to(device=values.device, dtype=torch.long)
    if not bool(torch.all((ids == 3) | (ids == 9) | (ids == 15)).item()):
        raise ValueError("attribution labels must be fault IDs 3/9/15")
    targets = torch.empty_like(ids)
    targets[ids == 3] = 0
    targets[ids == 9] = 1
    targets[ids == 15] = 2
    return F.cross_entropy(values, targets)


def composite_supervised_loss(
    *,
    detection_logits: Any,
    binary_targets: Any,
    post_logits: Any,
    prefix_logits: Any,
    embeddings: Any,
    contrastive_class_labels: Any,
    attribution_logits: Any,
    attribution_fault_ids: Any,
) -> LossOutput:
    """Exact 0.40/0.30/0.20/0.10 PCF-CycleRank stage-two objective."""

    balanced = balanced_binary_cross_entropy(detection_logits, binary_targets)
    paired_rank = same_run_paired_rank_loss(post_logits, prefix_logits)
    contrastive = supervised_contrastive_loss(
        embeddings,
        contrastive_class_labels,
        temperature=SUPCON_TEMPERATURE,
    )
    attribution = mature_fault_attribution_loss(attribution_logits, attribution_fault_ids)
    weighted_balanced = BALANCED_BCE_WEIGHT * balanced
    weighted_rank = PAIRED_RANK_WEIGHT * paired_rank
    weighted_contrastive = SUPCON_WEIGHT * contrastive
    weighted_attribution = ATTRIBUTION_WEIGHT * attribution
    total = weighted_balanced + weighted_rank + weighted_contrastive + weighted_attribution
    return LossOutput(
        total,
        {
            "balanced_bce": balanced,
            "paired_rank": paired_rank,
            "supervised_contrastive": contrastive,
            "mature_fault_attribution": attribution,
            "balanced_bce_weighted": weighted_balanced,
            "paired_rank_weighted": weighted_rank,
            "supervised_contrastive_weighted": weighted_contrastive,
            "mature_fault_attribution_weighted": weighted_attribution,
            "total": total,
        },
    )


__all__ = [
    "ATTRIBUTION_WEIGHT",
    "BALANCED_BCE_WEIGHT",
    "CYCLE_FEATURE_WEIGHT",
    "CYCLE_RAW_WEIGHT",
    "FAULT_IDS",
    "LossOutput",
    "PAIRED_FEATURE_WEIGHT",
    "PAIRED_RANK_WEIGHT",
    "PAIRED_RAW_WEIGHT",
    "SUPCON_CLASS_IDS",
    "SUPCON_TEMPERATURE",
    "SUPCON_WEIGHT",
    "balanced_binary_cross_entropy",
    "composite_supervised_loss",
    "discriminator_lsgan_loss",
    "generator_lsgan_cycle_loss",
    "mature_fault_attribution_loss",
    "same_run_paired_rank_loss",
    "supervised_contrastive_loss",
]
