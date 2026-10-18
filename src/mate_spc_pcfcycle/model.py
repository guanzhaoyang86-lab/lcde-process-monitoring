"""Neural core for the frozen PCF-CycleRank v6 candidate.

The module deliberately contains no data loading or feature construction.  It
operates on the two protocol-bound tensors only: a paired raw residual window
``A`` with shape ``[batch, 52, 128]`` and its fixed feature representation
``Bfeat`` with shape ``[batch, 604]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


MODEL_SCHEMA = "mate_spc_pcfcycle_v6"
BUILD_SEED = 20260910
RAW_CHANNELS = 52
RAW_LENGTH = 128
FEATURE_DIM = 604
RESIDUAL_SIGNATURE_DIM = 656
EMBEDDING_DIM = 32
NETWORK_DTYPE = torch.float32
TRAIN_ENDPOINTS = tuple(range(128, 497, 4))
PREFIX_ENDPOINTS = tuple(range(128, 157, 4))
MONITOR_TRAIN_ENDPOINTS = tuple(range(160, 497, 4))
MATURE_FAULT_ENDPOINTS = tuple(range(192, 497, 4))
DENSE_INFERENCE_ENDPOINTS = tuple(range(160, 500))


def _require_float32(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != NETWORK_DTYPE:
        raise ValueError(f"{name} must use torch.float32")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    return value


def _validate_pair(raw: Any, features: Any) -> tuple[Tensor, Tensor]:
    a = _require_float32(raw, name="A")
    bfeat = _require_float32(features, name="Bfeat")
    if a.ndim != 3 or tuple(a.shape[1:]) != (RAW_CHANNELS, RAW_LENGTH):
        raise ValueError("A must have shape [batch,52,128]")
    if bfeat.ndim != 2 or bfeat.shape[1] != FEATURE_DIM:
        raise ValueError("Bfeat must have shape [batch,604]")
    if a.shape[0] != bfeat.shape[0]:
        raise ValueError("A and Bfeat batch dimensions must agree")
    if a.device != bfeat.device:
        raise ValueError("A and Bfeat must be on the same device")
    if not bool(torch.isfinite(a).all().item()) or not bool(torch.isfinite(bfeat).all().item()):
        raise ValueError("A and Bfeat must contain only finite values")
    return a, bfeat


class RawToFeatureGenerator(nn.Module):
    """Map a 52-by-128 paired raw residual window into 604 fixed features."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(52, 128, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, FEATURE_DIM),
        )

    def forward(self, raw: Any) -> Tensor:
        a = _require_float32(raw, name="A")
        if a.ndim != 3 or tuple(a.shape[1:]) != (RAW_CHANNELS, RAW_LENGTH):
            raise ValueError("A must have shape [batch,52,128]")
        encoded = self.temporal(a)
        if tuple(encoded.shape[1:]) != (256, 16):
            raise RuntimeError("raw encoder geometry drifted from [batch,256,16]")
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)
        return self.projection(pooled)


class FeatureToRawGenerator(nn.Module):
    """Map 604 fixed features back to a 52-by-128 paired raw residual window."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.GELU(),
            nn.Linear(512, 4096),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 52, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, features: Any) -> Tensor:
        bfeat = _require_float32(features, name="Bfeat")
        if bfeat.ndim != 2 or bfeat.shape[1] != FEATURE_DIM:
            raise ValueError("Bfeat must have shape [batch,604]")
        latent = self.projection(bfeat).reshape(bfeat.shape[0], 256, 16)
        reconstructed = self.temporal(latent)
        if tuple(reconstructed.shape[1:]) != (RAW_CHANNELS, RAW_LENGTH):
            raise RuntimeError("raw decoder geometry drifted from [batch,52,128]")
        return reconstructed


class RawPatchDiscriminator(nn.Module):
    """PatchGAN discriminator over paired raw residual windows."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(52, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, raw: Any) -> Tensor:
        a = _require_float32(raw, name="A")
        if a.ndim != 3 or tuple(a.shape[1:]) != (RAW_CHANNELS, RAW_LENGTH):
            raise ValueError("A must have shape [batch,52,128]")
        output = self.network(a)
        if tuple(output.shape[1:]) != (1, 16):
            raise RuntimeError("raw PatchGAN geometry drifted from [batch,1,16]")
        return output


class FeatureDiscriminator(nn.Module):
    """LSGAN discriminator over the 604-dimensional feature domain."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, features: Any) -> Tensor:
        bfeat = _require_float32(features, name="Bfeat")
        if bfeat.ndim != 2 or bfeat.shape[1] != FEATURE_DIM:
            raise ValueError("Bfeat must have shape [batch,604]")
        return self.network(bfeat)


@dataclass(frozen=True)
class CycleOutputs:
    feature_prediction: Tensor
    raw_prediction: Tensor
    raw_cycle: Tensor
    feature_cycle: Tensor


class PCFCycleCore(nn.Module):
    """Bidirectional generators and their two protocol-bound discriminators."""

    def __init__(self) -> None:
        super().__init__()
        self.g_ab = RawToFeatureGenerator()
        self.g_ba = FeatureToRawGenerator()
        self.d_a = RawPatchDiscriminator()
        self.d_b = FeatureDiscriminator()

    def forward(self, raw: Any, features: Any) -> CycleOutputs:
        a, bfeat = _validate_pair(raw, features)
        feature_prediction = self.g_ab(a)
        raw_prediction = self.g_ba(bfeat)
        return CycleOutputs(
            feature_prediction=feature_prediction,
            raw_prediction=raw_prediction,
            raw_cycle=self.g_ba(feature_prediction),
            feature_cycle=self.g_ab(raw_prediction),
        )


def residual_signature(
    raw: Any,
    features: Any,
    outputs: CycleOutputs,
) -> Tensor:
    """Return the exact 52 + 604 PCF-Cycle residual signature."""

    a, bfeat = _validate_pair(raw, features)
    if outputs.raw_cycle.shape != a.shape:
        raise ValueError("raw_cycle must match A")
    if outputs.feature_cycle.shape != bfeat.shape:
        raise ValueError("feature_cycle must match Bfeat")
    if outputs.raw_cycle.device != a.device or outputs.feature_cycle.device != a.device:
        raise ValueError("cycle outputs must be on the input device")
    raw_error = torch.log1p((a - outputs.raw_cycle).square().mean(dim=-1))
    feature_error = (bfeat - outputs.feature_cycle).abs()
    signature = torch.cat((raw_error, feature_error), dim=-1)
    if signature.shape != (a.shape[0], RESIDUAL_SIGNATURE_DIM):
        raise RuntimeError("residual signature geometry drifted from [batch,656]")
    return signature


@dataclass(frozen=True)
class DetectorOutputs:
    embedding: Tensor
    detection_logit: Tensor
    attribution_logits: Tensor


class Stage2Detector(nn.Module):
    """Frozen 656-to-32 residual encoder with separate detection/attribution heads."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(RESIDUAL_SIGNATURE_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, EMBEDDING_DIM),
            nn.GELU(),
        )
        self.detection_head = nn.Linear(EMBEDDING_DIM, 1)
        self.attribution_head = nn.Linear(EMBEDDING_DIM, 3)

    def forward(self, signature: Any) -> DetectorOutputs:
        value = _require_float32(signature, name="residual_signature")
        if value.ndim != 2 or value.shape[1] != RESIDUAL_SIGNATURE_DIM:
            raise ValueError("residual_signature must have shape [batch,656]")
        embedding = self.encoder(value)
        return DetectorOutputs(
            embedding=embedding,
            detection_logit=self.detection_head(embedding).squeeze(-1),
            attribution_logits=self.attribution_head(embedding),
        )


@dataclass(frozen=True)
class PCFCycleOutputs:
    cycle: CycleOutputs
    residual_signature: Tensor
    detector: DetectorOutputs


class PCFCycleModel(nn.Module):
    """Complete neural state for the two-stage PCF-CycleRank candidate."""

    def __init__(self, cycle: PCFCycleCore | None = None, detector: Stage2Detector | None = None):
        super().__init__()
        self.cycle = cycle if cycle is not None else PCFCycleCore()
        self.detector = detector if detector is not None else Stage2Detector()

    def forward(self, raw: Any, features: Any) -> PCFCycleOutputs:
        cycle = self.cycle(raw, features)
        signature = residual_signature(raw, features, cycle)
        return PCFCycleOutputs(cycle, signature, self.detector(signature))


@dataclass(frozen=True)
class EndpointScores:
    embedding: Tensor
    detection_logits: Tensor
    attribution_logits: Tensor


def score_endpoint_pairs(
    value: PCFCycleModel,
    raw_windows: Any,
    feature_vectors: Any,
    *,
    batch_size: int = 256,
) -> EndpointScores:
    """Score sparse training or dense inference endpoints in fixed batches.

    Inputs have shapes ``[runs,endpoints,52,128]`` and
    ``[runs,endpoints,604]``.  The endpoint axis may therefore be 93 for all
    training endpoints, 85 for sparse monitor endpoints, or 340 for dense
    inference without changing the neural model.
    """

    if not isinstance(value, PCFCycleModel):
        raise TypeError("value must be a PCFCycleModel")
    raw = _require_float32(raw_windows, name="raw_windows")
    features = _require_float32(feature_vectors, name="feature_vectors")
    if raw.ndim != 4 or tuple(raw.shape[2:]) != (RAW_CHANNELS, RAW_LENGTH):
        raise ValueError("raw_windows must have shape [runs,endpoints,52,128]")
    if features.ndim != 3 or features.shape[2] != FEATURE_DIM:
        raise ValueError("feature_vectors must have shape [runs,endpoints,604]")
    if tuple(raw.shape[:2]) != tuple(features.shape[:2]):
        raise ValueError("raw_windows and feature_vectors run/endpoint axes must agree")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError("raw_windows and feature_vectors must have nonempty run/endpoint axes")
    if raw.device != features.device:
        raise ValueError("raw_windows and feature_vectors must be on the same device")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        model_device = next(value.parameters()).device
    except StopIteration as error:  # pragma: no cover - architecture always has parameters
        raise RuntimeError("PCFCycleModel unexpectedly has no parameters") from error
    if raw.device != model_device:
        raise ValueError("endpoint tensors must already be on the model device")

    run_count, endpoint_count = raw.shape[:2]
    flat_raw = raw.reshape(-1, RAW_CHANNELS, RAW_LENGTH)
    flat_features = features.reshape(-1, FEATURE_DIM)
    embeddings: list[Tensor] = []
    detections: list[Tensor] = []
    attributions: list[Tensor] = []
    module_modes = tuple((module, module.training) for module in value.modules())
    value.eval()
    try:
        with torch.no_grad():
            for start in range(0, flat_raw.shape[0], batch_size):
                result = value(
                    flat_raw[start : start + batch_size],
                    flat_features[start : start + batch_size],
                )
                embeddings.append(result.detector.embedding)
                detections.append(result.detector.detection_logit)
                attributions.append(result.detector.attribution_logits)
    finally:
        for module, training in module_modes:
            module.training = training
    return EndpointScores(
        embedding=torch.cat(embeddings).reshape(run_count, endpoint_count, EMBEDDING_DIM),
        detection_logits=torch.cat(detections).reshape(run_count, endpoint_count),
        attribution_logits=torch.cat(attributions).reshape(run_count, endpoint_count, 3),
    )


def _seeded_build(factory: Any, *, seed: int, device: str | torch.device) -> Any:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = factory()
    return module.to(device=device, dtype=NETWORK_DTYPE)


def build_cycle_core(
    *, seed: int = BUILD_SEED, device: str | torch.device = "cpu"
) -> PCFCycleCore:
    return _seeded_build(PCFCycleCore, seed=seed, device=device)


def build_stage2_detector(
    *, seed: int = BUILD_SEED, device: str | torch.device = "cpu"
) -> Stage2Detector:
    return _seeded_build(Stage2Detector, seed=seed, device=device)


def build_pcfcycle_model(
    *, seed: int = BUILD_SEED, device: str | torch.device = "cpu"
) -> PCFCycleModel:
    cycle = build_cycle_core(seed=seed, device="cpu")
    detector = build_stage2_detector(seed=seed, device="cpu")
    return PCFCycleModel(cycle, detector).to(device=device, dtype=NETWORK_DTYPE)


def parameter_counts(value: PCFCycleModel) -> dict[str, int]:
    if not isinstance(value, PCFCycleModel):
        raise TypeError("value must be a PCFCycleModel")

    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    counts = {
        "g_ab": count(value.cycle.g_ab),
        "g_ba": count(value.cycle.g_ba),
        "d_a": count(value.cycle.d_a),
        "d_b": count(value.cycle.d_b),
        "stage2_detector": count(value.detector),
    }
    counts["cycle_generators"] = counts["g_ab"] + counts["g_ba"]
    counts["cycle_discriminators"] = counts["d_a"] + counts["d_b"]
    counts["total"] = sum(counts[key] for key in ("g_ab", "g_ba", "d_a", "d_b", "stage2_detector"))
    return counts


def state_sha256(value: PCFCycleModel) -> str:
    if not isinstance(value, PCFCycleModel):
        raise TypeError("value must be a PCFCycleModel")
    digest = hashlib.sha256()
    for name, tensor in sorted(value.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def save_pcfcycle_checkpoint(
    value: PCFCycleModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    build_seed: int = BUILD_SEED,
) -> dict[str, Any]:
    if not isinstance(value, PCFCycleModel):
        raise TypeError("value must be a PCFCycleModel")
    document = {
        "schema": MODEL_SCHEMA,
        "build_seed": int(build_seed),
        "network_dtype": "float32",
        "input_shapes": {"A": [RAW_CHANNELS, RAW_LENGTH], "Bfeat": [FEATURE_DIM]},
        "parameter_counts": parameter_counts(value),
        "state_sha256": state_sha256(value),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in value.state_dict().items()},
        "metadata": dict(metadata or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(document, destination)
    return {key: document[key] for key in document if key != "state_dict"}


def load_pcfcycle_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> PCFCycleModel:
    document = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(document, Mapping) or document.get("schema") != MODEL_SCHEMA:
        raise ValueError("checkpoint schema changed")
    if document.get("network_dtype") != "float32":
        raise ValueError("checkpoint network dtype changed")
    if document.get("input_shapes") != {
        "A": [RAW_CHANNELS, RAW_LENGTH],
        "Bfeat": [FEATURE_DIM],
    }:
        raise ValueError("checkpoint input geometry changed")
    seed = document.get("build_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("checkpoint build seed is invalid")
    model = build_pcfcycle_model(seed=seed, device="cpu")
    state = document.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state_dict is invalid")
    model.load_state_dict(state, strict=True)
    if parameter_counts(model) != document.get("parameter_counts"):
        raise ValueError("checkpoint parameter counts changed")
    if state_sha256(model) != document.get("state_sha256"):
        raise ValueError("checkpoint state hash changed")
    return model.to(device=device, dtype=NETWORK_DTYPE)


__all__ = [
    "BUILD_SEED",
    "CycleOutputs",
    "DENSE_INFERENCE_ENDPOINTS",
    "DetectorOutputs",
    "EMBEDDING_DIM",
    "EndpointScores",
    "FEATURE_DIM",
    "FeatureDiscriminator",
    "FeatureToRawGenerator",
    "MODEL_SCHEMA",
    "NETWORK_DTYPE",
    "MONITOR_TRAIN_ENDPOINTS",
    "MATURE_FAULT_ENDPOINTS",
    "PCFCycleCore",
    "PCFCycleModel",
    "PCFCycleOutputs",
    "RAW_CHANNELS",
    "RAW_LENGTH",
    "RESIDUAL_SIGNATURE_DIM",
    "PREFIX_ENDPOINTS",
    "RawPatchDiscriminator",
    "RawToFeatureGenerator",
    "Stage2Detector",
    "build_cycle_core",
    "build_pcfcycle_model",
    "build_stage2_detector",
    "load_pcfcycle_checkpoint",
    "parameter_counts",
    "residual_signature",
    "save_pcfcycle_checkpoint",
    "score_endpoint_pairs",
    "state_sha256",
    "TRAIN_ENDPOINTS",
]
