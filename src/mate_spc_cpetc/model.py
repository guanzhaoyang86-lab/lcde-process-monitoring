"""Fixed three-expert MIL model for CP-ETC-Scan v5.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from .features import (
    AETrainingConfig,
    ChannelDenoisingAE,
    ChannelNormalizer,
    FAULT_IDS,
    LoopReferenceState,
    configure_determinism,
    extract_inclusive_windows,
    extract_window_features,
    fit_loop_reference,
    train_denoising_ae,
)


MODEL_SCHEMA = "mate_spc_cpetc_v5_1"
TRAINING_SEED = 20260912
RAW_DTYPE = torch.float64
NEURAL_DTYPE = torch.float32
RUN_FEATURE_BATCH_SIZE = 8


@dataclass(frozen=True)
class MILTrainingConfig:
    epochs: int = 60
    batch_runs_per_class: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = TRAINING_SEED

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_runs_per_class != 8:
            raise ValueError("MIL epochs must be positive and batch_runs_per_class fixed at 8")
        if self.learning_rate != 3.0e-4 or self.weight_decay != 1.0e-4:
            raise ValueError("MIL optimizer differs from the frozen protocol")
        if self.seed != TRAINING_SEED:
            raise ValueError("MIL seed differs from the frozen protocol")


class CPETCNet(nn.Module):
    """Channel-ID-preserving network with independent fault 3/9/15 heads."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_embedding = nn.Embedding(52, 4)
        self.shared_channel = nn.Sequential(nn.Linear(18, 8), nn.GELU(), nn.Linear(8, 4))
        self.channel_projection = nn.Sequential(nn.Linear(208, 32), nn.GELU())
        self.loop_projection = nn.Sequential(nn.Linear(12, 8), nn.GELU())
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(40, 16), nn.GELU(), nn.Linear(16, 1)) for _ in FAULT_IDS]
        )

    def forward(self, channel_values: Any, loop_values: Any) -> Tensor:
        channels = channel_values
        loops = loop_values
        if not isinstance(channels, Tensor) or not isinstance(loops, Tensor):
            raise TypeError("network inputs must be tensors")
        if channels.ndim < 3 or tuple(channels.shape[-2:]) != (52, 14):
            raise ValueError("channel features must end in shape [52,14]")
        if tuple(loops.shape) != tuple(channels.shape[:-2]) + (12,):
            raise ValueError("loop features must align and end in shape [12]")
        ids = torch.arange(52, device=channels.device)
        embedding = self.channel_embedding(ids)
        view = (1,) * (channels.ndim - 2) + (52, 4)
        embedding = embedding.reshape(view).expand(*channels.shape[:-2], 52, 4)
        encoded = self.shared_channel(torch.cat((channels, embedding), dim=-1))
        channel_path = self.channel_projection(encoded.reshape(*channels.shape[:-2], 208))
        loop_path = self.loop_projection(loops)
        fused = torch.cat((channel_path, loop_path), dim=-1)
        return torch.cat(tuple(head(fused) for head in self.heads), dim=-1)


def _top_mean(values: Tensor, count: int, *, dim: int) -> Tensor:
    if count < 1 or count > values.shape[dim]:
        raise ValueError("invalid frozen top-count")
    return torch.topk(values, count, dim=dim).values.mean(dim=dim)


def mil_loss(logits: Any, labels: Any) -> tuple[Tensor, dict[str, float]]:
    """Frozen one-vs-healthy MIL objective; other faults never become negatives."""

    scores = logits
    targets = labels
    if not isinstance(scores, Tensor) or scores.ndim != 3 or tuple(scores.shape[1:]) != (
        93,
        3,
    ):
        raise ValueError("logits must have shape [runs,93,3]")
    if not isinstance(targets, Tensor) or targets.shape != (scores.shape[0],):
        raise ValueError("labels must have shape [runs]")
    healthy_pool = _top_mean(scores, 10, dim=1)
    positive_pool = _top_mean(scores[:, 8:, :], 9, dim=1)
    prefix_pool = _top_mean(scores[:, :8, :], 1, dim=1)
    expert_losses: list[Tensor] = []
    details: dict[str, float] = {}
    for expert_index, fault_id in enumerate(FAULT_IDS):
        healthy = targets == 0
        positive = targets == fault_id
        if not bool(healthy.any().item()) or not bool(positive.any().item()):
            raise ValueError(f"expert {fault_id} needs healthy and matching-fault runs")
        healthy_loss = F.binary_cross_entropy_with_logits(
            healthy_pool[healthy, expert_index],
            torch.zeros_like(healthy_pool[healthy, expert_index]),
        )
        positive_loss = F.binary_cross_entropy_with_logits(
            positive_pool[positive, expert_index],
            torch.ones_like(positive_pool[positive, expert_index]),
        )
        prefix_loss = F.binary_cross_entropy_with_logits(
            prefix_pool[positive, expert_index],
            torch.zeros_like(prefix_pool[positive, expert_index]),
        )
        current = 0.5 * positive_loss + 0.5 * healthy_loss + 0.1 * prefix_loss
        expert_losses.append(current)
        details[f"fault_{fault_id}_positive_bce"] = float(positive_loss.detach().cpu())
        details[f"fault_{fault_id}_healthy_bce"] = float(healthy_loss.detach().cpu())
        details[f"fault_{fault_id}_prefix_bce"] = float(prefix_loss.detach().cpu())
    loss = torch.stack(expert_losses).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("MIL loss became nonfinite")
    details["loss"] = float(loss.detach().cpu())
    return loss, details


class CPETCModel(nn.Module):
    def __init__(
        self,
        *,
        normalizer: ChannelNormalizer | None = None,
        autoencoder: ChannelDenoisingAE | None = None,
        network: CPETCNet | None = None,
        loop_reference: LoopReferenceState | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = ChannelNormalizer() if normalizer is None else normalizer
        self.autoencoder = ChannelDenoisingAE() if autoencoder is None else autoencoder
        self.network = CPETCNet() if network is None else network
        self.loop_reference = loop_reference

    def require_fitted(self) -> LoopReferenceState:
        if not bool(self.normalizer.is_fitted.item()) or self.loop_reference is None:
            raise RuntimeError("CPETCModel is not fitted")
        if self.normalizer.median.dtype != RAW_DTYPE or self.normalizer.scale.dtype != RAW_DTYPE:
            raise RuntimeError("normalizer fitted state must remain float64")
        if next(self.autoencoder.parameters()).dtype != NEURAL_DTYPE:
            raise RuntimeError("autoencoder must remain float32")
        if next(self.network.parameters()).dtype != NEURAL_DTYPE:
            raise RuntimeError("classifier must remain float32")
        return self.loop_reference

    @torch.no_grad()
    def score_tensor(
        self,
        raw_runs: Any,
        *,
        run_batch_size: int = RUN_FEATURE_BATCH_SIZE,
        feature_batch_size: int = 512,
    ) -> Tensor:
        runs = raw_runs
        if not isinstance(runs, Tensor) or runs.ndim != 3 or tuple(runs.shape[1:]) != (
            500,
            52,
        ):
            raise ValueError("raw_runs must have shape [runs,500,52]")
        reference = self.require_fitted()
        device = next(self.network.parameters()).device
        raw64 = runs.to(device=device, dtype=RAW_DTYPE)
        channel_values, loop_values = _extract_run_features(
            self,
            raw64,
            reference,
            run_batch_size=run_batch_size,
            feature_batch_size=feature_batch_size,
        )
        self.network.eval()
        return self.network(channel_values, loop_values)


def _place_mixed_precision_model(model: CPETCModel, device: torch.device | str) -> CPETCModel:
    """Place fixed-statistic and neural paths without downcasting fitted state."""

    target = torch.device(device)
    model.normalizer.to(device=target)
    model.autoencoder.to(device=target, dtype=NEURAL_DTYPE)
    model.network.to(device=target, dtype=NEURAL_DTYPE)
    return model


@torch.no_grad()
def _extract_run_features(
    model: CPETCModel,
    raw_runs: Tensor,
    loop_reference: LoopReferenceState,
    *,
    run_batch_size: int = RUN_FEATURE_BATCH_SIZE,
    feature_batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    if raw_runs.ndim != 3 or tuple(raw_runs.shape[1:]) != (500, 52):
        raise ValueError("raw_runs must have shape [runs,500,52]")
    if raw_runs.dtype != RAW_DTYPE:
        raise ValueError("raw_runs must retain float64 source precision")
    if run_batch_size < 1 or feature_batch_size < 1:
        raise ValueError("feature batch sizes must be positive")
    channel_chunks: list[Tensor] = []
    loop_chunks: list[Tensor] = []
    for start in range(0, raw_runs.shape[0], run_batch_size):
        windows = extract_inclusive_windows(raw_runs[start : start + run_batch_size])
        channel_values, loop_values = extract_window_features(
            windows,
            normalizer=model.normalizer,
            autoencoder=model.autoencoder,
            loop_reference=loop_reference,
            batch_size=feature_batch_size,
        )
        channel_chunks.append(channel_values)
        loop_chunks.append(loop_values)
    channels = torch.cat(channel_chunks, dim=0)
    loops = torch.cat(loop_chunks, dim=0)
    if channels.dtype != NEURAL_DTYPE or loops.dtype != NEURAL_DTYPE:
        raise RuntimeError("fixed features must cross the classifier boundary as float32")
    return channels, loops


@dataclass(frozen=True)
class CPETCTrainingResult:
    model: CPETCModel
    ae_loss_history: tuple[float, ...]
    mil_loss_history: tuple[float, ...]
    training_run_count: int
    healthy_run_count: int
    requested_device: str
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, CPETCModel):
            raise TypeError("training result model must be CPETCModel")
        if not self.ae_loss_history or not self.mil_loss_history:
            raise ValueError("training histories cannot be empty")
        if any(not math.isfinite(value) for value in (*self.ae_loss_history, *self.mil_loss_history)):
            raise ValueError("training histories must be finite")


def _run_idv(run: Any) -> int:
    for name in ("source_idv", "idv"):
        if hasattr(run, name):
            return int(getattr(run, name))
    metadata = getattr(run, "metadata", None)
    if isinstance(metadata, Mapping):
        return int(metadata["source_idv"])
    raise ValueError("run has no source IDV")


def _stack_runs(runs: Sequence[Any]) -> tuple[Tensor, Tensor]:
    if not runs:
        raise ValueError("runs cannot be empty")
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    for run in runs:
        value = np.asarray(getattr(run, "x", None))
        if value.shape != (500, 52) or value.dtype != np.dtype(np.float64):
            raise ValueError("every run.x must be exact float64 [500,52]")
        if not np.isfinite(value).all():
            raise ValueError("run.x must be finite")
        idv = _run_idv(run)
        if idv not in (1, *FAULT_IDS):
            raise ValueError("run IDV is outside the frozen classes")
        arrays.append(value)
        labels.append(0 if idv == 1 else idv)
    return torch.from_numpy(np.stack(arrays)), torch.tensor(labels)


def fit_cpetc_model(
    runs: Sequence[Any],
    *,
    device: str = "cuda",
    ae_config: AETrainingConfig = AETrainingConfig(),
    mil_config: MILTrainingConfig = MILTrainingConfig(),
) -> CPETCTrainingResult:
    mil_config.validate()
    configure_determinism(TRAINING_SEED)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    raw_cpu, labels_cpu = _stack_runs(runs)
    counts = {label: int(torch.sum(labels_cpu == label)) for label in (0, *FAULT_IDS)}
    if len(set(counts.values())) != 1 or min(counts.values()) < 1:
        raise ValueError("training runs must be balanced across healthy/F3/F9/F15")
    target = torch.device(device)
    raw = raw_cpu.to(device=target, dtype=RAW_DTYPE)
    labels = labels_cpu.to(target)
    model = _place_mixed_precision_model(CPETCModel(), target)
    healthy = raw[labels == 0]
    model.normalizer.fit(healthy)
    normalized_healthy = model.normalizer.transform_runs(healthy)
    model.loop_reference = fit_loop_reference(normalized_healthy)
    healthy_windows = extract_inclusive_windows(normalized_healthy.to(dtype=NEURAL_DTYPE))
    ae_history = train_denoising_ae(model.autoencoder, healthy_windows, config=ae_config)
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)
    del healthy_windows

    channel_values, loop_values = _extract_run_features(
        model,
        raw,
        model.require_fitted(),
    )
    configure_determinism(mil_config.seed)
    model.network = CPETCNet().to(device=target, dtype=NEURAL_DTYPE)
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=mil_config.learning_rate,
        weight_decay=mil_config.weight_decay,
    )
    generator = torch.Generator(device=target)
    generator.manual_seed(mil_config.seed)
    class_indices = {
        label: torch.nonzero(labels == label, as_tuple=False).flatten()
        for label in (0, *FAULT_IDS)
    }
    step_count = math.ceil(counts[0] / mil_config.batch_runs_per_class)
    mil_history: list[float] = []
    model.network.train()
    for _ in range(mil_config.epochs):
        permutations = {
            label: indices[
                torch.randperm(indices.numel(), device=target, generator=generator)
            ]
            for label, indices in class_indices.items()
        }
        epoch_total = 0.0
        for step in range(step_count):
            start = step * mil_config.batch_runs_per_class
            stop = min(start + mil_config.batch_runs_per_class, counts[0])
            selected = torch.cat(
                tuple(permutations[label][start:stop] for label in (0, *FAULT_IDS))
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.network(channel_values[selected], loop_values[selected])
            loss, _ = mil_loss(logits, labels[selected])
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach().cpu())
        mil_history.append(epoch_total / step_count)
    model.network.eval()
    return CPETCTrainingResult(
        model,
        tuple(ae_history),
        tuple(mil_history),
        len(runs),
        counts[0],
        device,
        "float32",
    )


def _model_from(value: CPETCTrainingResult | CPETCModel) -> CPETCModel:
    if isinstance(value, CPETCTrainingResult):
        return value.model
    if isinstance(value, CPETCModel):
        return value
    raise TypeError("expected CPETCTrainingResult or CPETCModel")


def score_cpetc_logits(
    value: CPETCTrainingResult | CPETCModel,
    runs: Sequence[Any],
    *,
    device: str,
) -> np.ndarray:
    model = _place_mixed_precision_model(_model_from(value), torch.device(device))
    raw, _ = _stack_runs(runs)
    logits = model.score_tensor(raw.to(device=device, dtype=RAW_DTYPE))
    result = logits.detach().cpu().to(torch.float64).numpy()
    if result.shape != (len(runs), 93, 3) or not np.isfinite(result).all():
        raise FloatingPointError("model scoring produced invalid logits")
    return result


def component_audit_hashes(value: CPETCTrainingResult | CPETCModel) -> dict[str, str]:
    model = _model_from(value)
    output: dict[str, str] = {}
    for prefix, module in (
        ("normalizer", model.normalizer),
        ("autoencoder", model.autoencoder),
        ("network", model.network),
    ):
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        output[prefix] = digest.hexdigest()
    reference = model.require_fitted().to_document()
    output["loop_reference"] = hashlib.sha256(
        repr(sorted(reference.items())).encode("utf-8")
    ).hexdigest()
    return output


def save_cpetc_checkpoint(
    value: CPETCTrainingResult | CPETCModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = _model_from(value)
    histories = (
        {
            "ae": list(value.ae_loss_history),
            "mil": list(value.mil_loss_history),
            "training_run_count": value.training_run_count,
            "healthy_run_count": value.healthy_run_count,
            "requested_device": value.requested_device,
            "dtype": value.dtype,
        }
        if isinstance(value, CPETCTrainingResult)
        else None
    )
    document = {
        "schema": MODEL_SCHEMA,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "loop_reference": model.require_fitted().to_document(),
        "training": histories,
        "component_hashes": component_audit_hashes(model),
        "metadata": dict(metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(document, destination)
    return {
        "schema": MODEL_SCHEMA,
        "component_hashes": document["component_hashes"],
        "training": histories,
        "metadata": document["metadata"],
    }


def load_cpetc_checkpoint(path: str | Path, *, device: str = "cpu") -> CPETCModel:
    document = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(document, Mapping) or document.get("schema") != MODEL_SCHEMA:
        raise ValueError("checkpoint schema changed")
    reference = document["loop_reference"]
    state = LoopReferenceState(
        float(reference["dwell_threshold"]),
        tuple(tuple(float(item) for item in row) for row in reference["arx_coefficients"]),  # type: ignore[arg-type]
        tuple(float(item) for item in reference["residual_denominators"]),  # type: ignore[arg-type]
    )
    model = _place_mixed_precision_model(CPETCModel(loop_reference=state), device)
    model.load_state_dict(document["state_dict"], strict=True)
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if component_audit_hashes(model) != dict(document["component_hashes"]):
        raise ValueError("checkpoint component hashes changed")
    return model


__all__ = [
    "CPETCModel",
    "CPETCNet",
    "CPETCTrainingResult",
    "MILTrainingConfig",
    "MODEL_SCHEMA",
    "component_audit_hashes",
    "fit_cpetc_model",
    "load_cpetc_checkpoint",
    "mil_loss",
    "save_cpetc_checkpoint",
    "score_cpetc_logits",
]
