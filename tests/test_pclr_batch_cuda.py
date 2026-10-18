from __future__ import annotations

import math

import torch
import numpy as np
import pytest

from mate_spc_pclr import model


def _heads(feature_count: int = 7) -> model.PCLRHeads:
    rng = np.random.default_rng(8117)
    return model.PCLRHeads(
        weights=rng.normal(0.0, 0.3, size=(3, feature_count)),
        bias=np.array([0.15, -0.2, 0.05], dtype=np.float64),
        healthy_logit_center=np.array([0.1, -0.3, 0.2], dtype=np.float64),
        healthy_logit_scale=np.array([0.8, 1.1, 1.4], dtype=np.float64),
        kappa=np.array([0.12, 0.25, 0.4], dtype=np.float64),
    )


def _numpy_reference(
    features: np.ndarray,
    heads: model.PCLRHeads,
    indices: np.ndarray,
    threshold: float | None,
) -> list[model.PCLRMonitoringPath]:
    return [
        model.page_cusum_numpy(
            heads.logits_numpy(features[run]),
            heads,
            local_indices=indices[run],
            threshold=threshold,
        )
        for run in range(features.shape[0])
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 6, 32])
def test_batched_cuda_monitor_matches_every_numpy_run(batch_size: int) -> None:
    rng = np.random.default_rng(20260307)
    features = rng.normal(size=(35, 19, 7)).astype(np.float64)
    indices = np.stack(
        [np.arange(100 + run, 119 + run, dtype=np.int64) for run in range(35)]
    )
    heads = _heads()
    threshold = 2.75
    expected = _numpy_reference(features, heads, indices, threshold)

    actual = model.monitor_batch_torch_float64(
        features,
        heads,
        local_indices=indices,
        threshold=threshold,
        device="cuda",
        batch_size=batch_size,
    )

    for run, reference in enumerate(expected):
        selected = actual.run(run)
        for name in (
            "raw_logits",
            "standardized_logits",
            "cusum",
            "component_persistence",
            "envelope",
        ):
            np.testing.assert_allclose(
                getattr(selected, name),
                getattr(reference, name),
                rtol=0.0,
                atol=1.0e-10,
            )
        for name in ("local_indices", "eligible", "alarm", "episode_start"):
            np.testing.assert_array_equal(
                getattr(selected, name), getattr(reference, name)
            )
        assert selected.threshold == reference.threshold
    np.testing.assert_allclose(
        actual.run_max,
        np.array([path.run_max for path in expected]),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        actual.component_max,
        np.stack([path.component_max for path in expected]),
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_cuda_monitor_is_repeatable_and_restores_backend_flags() -> None:
    rng = np.random.default_rng(9301)
    features = rng.normal(size=(33, 13, 7)).astype(np.float64)
    heads = _heads()
    prior = (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        bool(torch.backends.cudnn.deterministic),
        bool(torch.backends.cudnn.benchmark),
    )
    first = model.monitor_batch_torch_float64(
        features, heads, threshold=1.5, device="cuda", batch_size=32
    )
    second = model.monitor_batch_torch_float64(
        features, heads, threshold=1.5, device="cuda", batch_size=32
    )
    for name in (
        "raw_logits",
        "standardized_logits",
        "cusum",
        "component_persistence",
        "envelope",
        "eligible",
        "alarm",
        "episode_start",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert prior == (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        bool(torch.backends.cudnn.deterministic),
        bool(torch.backends.cudnn.benchmark),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_batched_cuda_page_has_no_ceiling_and_uses_strict_threshold() -> None:
    heads = model.PCLRHeads(
        weights=np.zeros((3, 2), dtype=np.float64),
        bias=np.array([1.0e6, -1.0, -1.0], dtype=np.float64),
        healthy_logit_center=np.zeros(3, dtype=np.float64),
        healthy_logit_scale=np.ones(3, dtype=np.float64),
        kappa=np.zeros(3, dtype=np.float64),
    )
    actual = model.monitor_batch_torch_float64(
        np.zeros((2, 5, 2), dtype=np.float64),
        heads,
        threshold=3.0e6,
        device="cuda",
        batch_size=1,
    )
    assert actual.cusum[0, -1, 0] == 5.0e6
    assert actual.envelope[0, -1] == 3.0e6
    assert not actual.alarm.any()
    np.testing.assert_array_equal(
        actual.eligible[0], np.array([False, False, True, True, True])
    )


def _reference_objective(
    records: tuple[tuple[str, int, np.ndarray], ...],
    prefix_records: dict[str, np.ndarray],
    weights: torch.Tensor,
    bias: torch.Tensor,
    *,
    ridge: float,
    point_weight: float,
    bag_weight: float,
    prefix_rank_weight: float,
    rank_margin: float,
    top_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    by_label = {
        label: [
            torch.as_tensor(matrix, dtype=torch.float64, device=weights.device)
            for _, run_label, matrix in records
            if run_label == label
        ]
        for label in (model.HEALTHY_ID, *model.FAULT_IDS)
    }
    by_label_prefix = {
        label: [
            torch.as_tensor(
                prefix_records[name], dtype=torch.float64, device=weights.device
            )
            for name, run_label, _ in records
            if run_label == label
        ]
        for label in (model.HEALTHY_ID, *model.FAULT_IDS)
    }

    def top_bag(logit: torch.Tensor) -> torch.Tensor:
        count = max(1, int(math.ceil(logit.shape[0] * top_fraction)))
        return torch.topk(logit, count, largest=True, sorted=False).values.mean()

    losses: list[torch.Tensor] = []
    for head, fault_id in enumerate(model.FAULT_IDS):
        negative_logits = [
            matrix @ weights[head] + bias[head]
            for matrix in by_label[model.HEALTHY_ID]
        ]
        positive_logits = [
            matrix @ weights[head] + bias[head] for matrix in by_label[fault_id]
        ]
        negative_point = torch.stack(
            [torch.nn.functional.softplus(logit).mean() for logit in negative_logits]
        ).mean()
        positive_point = torch.stack(
            [torch.nn.functional.softplus(-logit).mean() for logit in positive_logits]
        ).mean()
        negative_bag = torch.stack(
            [torch.nn.functional.softplus(top_bag(logit)) for logit in negative_logits]
        ).mean()
        positive_bag = torch.stack(
            [torch.nn.functional.softplus(-top_bag(logit)) for logit in positive_logits]
        ).mean()
        positive_prefix_logits = [
            matrix @ weights[head] + bias[head]
            for matrix in by_label_prefix[fault_id]
        ]
        rankings = [
            torch.nn.functional.softplus(
                rank_margin - (top_bag(post) - prefix.mean())
            )
            for post, prefix in zip(
                positive_logits, positive_prefix_logits, strict=True
            )
        ]
        losses.append(
            point_weight * 0.5 * (negative_point + positive_point)
            + bag_weight * 0.5 * (negative_bag + positive_bag)
            + prefix_rank_weight * torch.stack(rankings).mean()
            + 0.5 * ridge * torch.sum(weights[head] ** 2)
        )
    stacked = torch.stack(losses)
    return stacked.mean(), stacked


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_stacked_training_objective_preserves_loss_and_gradient(
    device_name: str,
) -> None:
    rng = np.random.default_rng(6049)
    features: dict[str, np.ndarray] = {}
    prefixes: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    feature_count = 5
    for label in (model.HEALTHY_ID, *model.FAULT_IDS):
        for run, length in enumerate((7, 7, 9), start=1):
            name = f"label-{label:02d}-run-{run:02d}"
            features[name] = rng.normal(size=(length, feature_count))
            prefixes[name] = rng.normal(
                size=(model.MONITOR_START - model.BASELINE_STOP, feature_count)
            )
            labels[name] = label
    records = model._validate_training_features(features, labels)
    prefix_records = model._validate_training_prefix_features(prefixes, records)
    target = torch.device(device_name)
    stacks = model._stack_training_features_torch(
        records, prefix_records, torch=torch, target=target
    )
    initial_weights = rng.normal(size=(3, feature_count))
    initial_bias = rng.normal(size=3)
    batched_weights = torch.tensor(
        initial_weights, dtype=torch.float64, device=target, requires_grad=True
    )
    batched_bias = torch.tensor(
        initial_bias, dtype=torch.float64, device=target, requires_grad=True
    )
    reference_weights = torch.tensor(
        initial_weights, dtype=torch.float64, device=target, requires_grad=True
    )
    reference_bias = torch.tensor(
        initial_bias, dtype=torch.float64, device=target, requires_grad=True
    )
    parameters = dict(
        ridge=0.037,
        point_weight=1.2,
        bag_weight=0.4,
        prefix_rank_weight=0.3,
        rank_margin=0.17,
        top_fraction=0.25,
    )
    batched_loss, batched_heads = model._pclr_training_objective_batched(
        torch, stacks, batched_weights, batched_bias, **parameters
    )
    reference_loss, reference_heads = _reference_objective(
        records,
        prefix_records,
        reference_weights,
        reference_bias,
        **parameters,
    )
    batched_gradient = torch.autograd.grad(
        batched_loss, (batched_weights, batched_bias)
    )
    reference_gradient = torch.autograd.grad(
        reference_loss, (reference_weights, reference_bias)
    )

    np.testing.assert_allclose(
        batched_loss.detach().cpu().numpy(),
        reference_loss.detach().cpu().numpy(),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        batched_heads.detach().cpu().numpy(),
        reference_heads.detach().cpu().numpy(),
        rtol=0.0,
        atol=1.0e-12,
    )
    for actual, expected in zip(batched_gradient, reference_gradient, strict=True):
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(),
            expected.detach().cpu().numpy(),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_batched_monitor_rejects_invalid_batch_contract_before_torch_work() -> None:
    heads = _heads()
    with pytest.raises(ValueError, match="run>=1"):
        model.monitor_batch_torch_float64(
            np.empty((0, 5, 7), dtype=np.float64), heads, device="cpu"
        )
    with pytest.raises(ValueError, match="batch_size"):
        model.monitor_batch_torch_float64(
            np.zeros((1, 5, 7), dtype=np.float64),
            heads,
            device="cpu",
            batch_size=0,
        )
    with pytest.raises(ValueError, match="consecutive"):
        model.monitor_batch_torch_float64(
            np.zeros((1, 5, 7), dtype=np.float64),
            heads,
            local_indices=np.array([0, 1, 3, 4, 5]),
            device="cpu",
        )
