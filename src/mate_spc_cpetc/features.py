"""Frozen channel-preserving features for CP-ETC-Scan v5.1.

Torch is imported before NumPy because the supported Windows CUDA environment
requires Torch's DLLs to be loaded first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


RUN_LENGTH = 500
CHANNELS = 52
WINDOW_LENGTH = 128
WINDOW_ENDPOINTS = tuple(range(128, 497, 4))
PREFIX_COUNT = 8
MONITOR_COUNT = 85
MONITOR_ENDPOINTS = WINDOW_ENDPOINTS[PREFIX_COUNT:]
FAULT_IDS = (3, 9, 15)
LOOP_CHANNELS = (51, 10, 21)  # XMV11, XMEAS11, XMEAS22
PSD_BANDS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 65))
MAD_CONSISTENCY = 1.4826
SCALE_FLOOR = 1.0e-6
CLIP = 8.0

if len(WINDOW_ENDPOINTS) != 93 or len(MONITOR_ENDPOINTS) != MONITOR_COUNT:
    raise AssertionError("frozen CP-ETC window contract changed")


def _finite_tensor(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating torch.Tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def validate_runs(value: Any, *, name: str = "runs") -> Tensor:
    runs = _finite_tensor(value, name=name)
    if runs.ndim != 3 or tuple(runs.shape[1:]) != (RUN_LENGTH, CHANNELS):
        raise ValueError(f"{name} must have shape [runs,500,52]")
    if runs.shape[0] < 1:
        raise ValueError(f"{name} must be nonempty")
    return runs


def extract_inclusive_windows(runs: Any) -> Tensor:
    """Return ``[B,93,52,128]`` windows ``x[t-127:t+1]``."""

    values = validate_runs(runs)
    windows = torch.stack(
        [values[:, endpoint - 127 : endpoint + 1, :] for endpoint in WINDOW_ENDPOINTS],
        dim=1,
    )
    return windows.permute(0, 1, 3, 2).contiguous()


def split_prefix_monitor(windows: Any) -> tuple[Tensor, Tensor]:
    values = _finite_tensor(windows, name="windows")
    if values.ndim < 4 or tuple(values.shape[-3:]) != (93, 52, 128):
        raise ValueError("windows must end in shape [93,52,128]")
    return values[..., :PREFIX_COUNT, :, :], values[..., PREFIX_COUNT:, :, :]


class ChannelNormalizer(nn.Module):
    """Fit-only channel median/MAD state with immutable v5.1 constants."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("median", torch.zeros(CHANNELS, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(CHANNELS, dtype=torch.float64))
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, healthy_fit_runs: Any) -> "ChannelNormalizer":
        self._require_float64_state()
        values = validate_runs(healthy_fit_runs, name="healthy_fit_runs")
        flat = values.detach().to(device=self.median.device, dtype=torch.float64).reshape(
            -1, CHANNELS
        )
        median = torch.quantile(flat, 0.5, dim=0, interpolation="linear")
        mad = torch.quantile(
            torch.abs(flat - median), 0.5, dim=0, interpolation="linear"
        )
        scale = torch.clamp(MAD_CONSISTENCY * mad, min=SCALE_FLOOR)
        if not bool(torch.isfinite(median).all().item()) or not bool(
            torch.isfinite(scale).all().item()
        ):
            raise FloatingPointError("normalizer fit produced nonfinite state")
        self.median.copy_(median)
        self.scale.copy_(scale)
        self.is_fitted.fill_(True)
        return self

    def _require_fitted(self) -> None:
        if not bool(self.is_fitted.item()):
            raise RuntimeError("ChannelNormalizer.fit must be called first")

    def _require_float64_state(self) -> None:
        if self.median.dtype != torch.float64 or self.scale.dtype != torch.float64:
            raise RuntimeError("normalizer fitted state must remain float64")

    def transform_runs(self, runs: Any) -> Tensor:
        values = validate_runs(runs)
        self._require_fitted()
        self._require_float64_state()
        values64 = values.to(device=self.median.device, dtype=torch.float64)
        return torch.clamp((values64 - self.median) / self.scale, -CLIP, CLIP)

    def transform_windows(self, windows: Any) -> Tensor:
        values = _finite_tensor(windows, name="windows")
        if values.ndim < 3 or tuple(values.shape[-2:]) != (CHANNELS, WINDOW_LENGTH):
            raise ValueError("windows must end in shape [52,128]")
        self._require_fitted()
        self._require_float64_state()
        values64 = values.to(device=self.median.device, dtype=torch.float64)
        view = (1,) * (values.ndim - 2) + (CHANNELS, 1)
        return torch.clamp(
            (values64 - self.median.reshape(view)) / self.scale.reshape(view), -CLIP, CLIP
        )

    def to_document(self) -> dict[str, Any]:
        self._require_fitted()
        self._require_float64_state()
        return {
            "median": self.median.detach().cpu().numpy().tolist(),
            "scale": self.scale.detach().cpu().numpy().tolist(),
            "mad_consistency": MAD_CONSISTENCY,
            "scale_floor": SCALE_FLOOR,
            "clip": CLIP,
        }


class ChannelDenoisingAE(nn.Module):
    """Grouped autoencoder that never mixes identities of the 52 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = nn.Conv1d(52, 208, 7, stride=2, padding=3, groups=52)
        self.encoder2 = nn.Conv1d(208, 104, 5, stride=2, padding=2, groups=52)
        self.decoder1 = nn.ConvTranspose1d(104, 208, 4, stride=2, padding=1, groups=52)
        self.decoder2 = nn.ConvTranspose1d(208, 52, 4, stride=2, padding=1, groups=52)

    def encode(self, windows: Any) -> Tensor:
        values = _finite_tensor(windows, name="windows")
        if values.ndim != 3 or tuple(values.shape[1:]) != (52, 128):
            raise ValueError("AE windows must have shape [batch,52,128]")
        encoded = self.encoder2(F.gelu(self.encoder1(values)))
        if tuple(encoded.shape[1:]) != (104, 32):
            raise AssertionError("AE encoder shape changed")
        return encoded.reshape(values.shape[0], 52, 2, 32)

    def decode(self, code: Any) -> Tensor:
        values = _finite_tensor(code, name="code")
        if values.ndim != 4 or tuple(values.shape[1:]) != (52, 2, 32):
            raise ValueError("AE code must have shape [batch,52,2,32]")
        flat = values.reshape(values.shape[0], 104, 32)
        reconstruction = self.decoder2(F.gelu(self.decoder1(flat)))
        if tuple(reconstruction.shape[1:]) != (52, 128):
            raise AssertionError("AE decoder shape changed")
        return reconstruction

    def forward(self, windows: Any) -> tuple[Tensor, Tensor]:
        code = self.encode(windows)
        return self.decode(code), code


@dataclass(frozen=True)
class AETrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    mask_probability: float = 0.10
    noise_std: float = 0.05
    seed: int = 20260912


def configure_determinism(seed: int = 20260912) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_denoising_ae(
    model: ChannelDenoisingAE,
    healthy_windows: Any,
    *,
    config: AETrainingConfig = AETrainingConfig(),
) -> list[float]:
    values = _finite_tensor(healthy_windows, name="healthy_windows")
    if values.ndim < 3 or tuple(values.shape[-2:]) != (52, 128):
        raise ValueError("healthy_windows must end in shape [52,128]")
    samples = values.reshape(-1, 52, 128).detach()
    if samples.shape[0] < 1:
        raise ValueError("healthy_windows must be nonempty")
    configure_determinism(config.seed)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    samples = samples.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        order = torch.randperm(samples.shape[0], generator=generator, device=device)
        total = 0.0
        count = 0
        for start in range(0, samples.shape[0], config.batch_size):
            clean = samples[order[start : start + config.batch_size]]
            mask = (
                torch.rand(
                    (clean.shape[0], 1, clean.shape[2]),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                < config.mask_probability
            )
            noise = config.noise_std * torch.randn(
                clean.shape, generator=generator, device=device, dtype=dtype
            )
            corrupted = (~mask).to(dtype) * clean + noise
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(corrupted)
            loss = F.smooth_l1_loss(reconstruction, clean, beta=1.0)
            loss = loss + 0.1 * F.smooth_l1_loss(
                torch.diff(reconstruction, dim=-1), torch.diff(clean, dim=-1), beta=1.0
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("AE training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * clean.shape[0]
            count += int(clean.shape[0])
        losses.append(total / count)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return losses


@dataclass(frozen=True)
class LoopReferenceState:
    dwell_threshold: float
    arx_coefficients: tuple[tuple[float, float, float], tuple[float, float, float]]
    residual_denominators: tuple[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.dwell_threshold) or self.dwell_threshold < 1.0e-6:
            raise ValueError("dwell threshold must be finite and at least 1e-6")
        beta = np.asarray(self.arx_coefficients, dtype=np.float64)
        denominator = np.asarray(self.residual_denominators, dtype=np.float64)
        if beta.shape != (2, 3) or denominator.shape != (2,):
            raise ValueError("loop-reference shapes changed")
        if not np.isfinite(beta).all() or not np.isfinite(denominator).all():
            raise ValueError("loop-reference values must be finite")
        if np.any(denominator < 1.0e-8):
            raise ValueError("loop residual denominators must be at least 1e-8")

    def to_document(self) -> dict[str, Any]:
        return {
            "dwell_threshold": self.dwell_threshold,
            "arx_coefficients": [list(row) for row in self.arx_coefficients],
            "residual_denominators": list(self.residual_denominators),
            "dwell_quantile": 0.20,
            "ridge_diagonal": [0.0, 1.0e-4, 1.0e-4],
        }


@torch.no_grad()
def fit_loop_reference(normalized_healthy_runs: Any) -> LoopReferenceState:
    values = validate_runs(normalized_healthy_runs, name="normalized_healthy_runs")
    if values.dtype != torch.float64:
        raise ValueError("normalized_healthy_runs must retain float64 precision")
    work = values.detach().to(device="cpu", dtype=torch.float64)
    u = work[:, :, LOOP_CHANNELS[0]]
    dwell = max(
        float(
            np.quantile(
                np.abs(np.diff(u.numpy(), axis=1)).reshape(-1),
                0.20,
                method="linear",
            )
        ),
        1.0e-6,
    )
    betas: list[tuple[float, float, float]] = []
    denominators: list[float] = []
    for channel in LOOP_CHANNELS[1:]:
        y = work[:, :, channel]
        target = y[:, 1:].reshape(-1)
        design = torch.stack(
            (
                torch.ones_like(y[:, :-1]),
                y[:, :-1],
                u[:, :-1],
            ),
            dim=-1,
        ).reshape(-1, 3)
        gram = design.T @ design
        gram = gram + torch.diag(torch.tensor((0.0, 1.0e-4, 1.0e-4)))
        condition = float(torch.linalg.cond(gram))
        if not math.isfinite(condition) or condition > 1.0e12:
            raise ValueError("ARX ridge system exceeds the frozen condition-number guard")
        beta = torch.linalg.solve(gram, design.T @ target)
        if not bool(torch.isfinite(beta).all().item()):
            raise FloatingPointError("ARX fit produced nonfinite coefficients")
        residual = y[:, 1:] - (
            beta[0] + beta[1] * y[:, :-1] + beta[2] * u[:, :-1]
        )
        window_mse = torch.stack(
            [
                torch.mean(residual[:, endpoint - 127 : endpoint] ** 2, dim=1)
                for endpoint in WINDOW_ENDPOINTS
            ],
            dim=1,
        ).reshape(-1)
        denominator = max(float(torch.quantile(window_mse, 0.5)), 1.0e-8)
        betas.append(tuple(float(item) for item in beta.tolist()))
        denominators.append(denominator)
    return LoopReferenceState(dwell, tuple(betas), tuple(denominators))  # type: ignore[arg-type]


def _one_sided_psd(windows: Tensor) -> Tensor:
    if windows.ndim != 3 or tuple(windows.shape[1:]) != (52, 128):
        raise ValueError("PSD windows must have shape [batch,52,128]")
    centered = windows - windows.mean(dim=-1, keepdim=True)
    hann = torch.hann_window(128, periodic=True, device=windows.device, dtype=windows.dtype)
    spectrum = torch.fft.rfft(centered * hann, dim=-1)
    power = torch.abs(spectrum) ** 2 / torch.sum(hann**2)
    power = power.clone()
    power[:, :, 0] = 0.0
    power[:, :, 1:64] *= 2.0
    return power


def channel_features(
    normalized_windows: Any,
    reconstruction: Any,
    code: Any,
) -> Tensor:
    windows = _finite_tensor(normalized_windows, name="normalized_windows")
    rebuilt = _finite_tensor(reconstruction, name="reconstruction")
    latent = _finite_tensor(code, name="code")
    if windows.ndim != 3 or tuple(windows.shape[1:]) != (52, 128):
        raise ValueError("normalized_windows must have shape [batch,52,128]")
    if rebuilt.shape != windows.shape or tuple(latent.shape) != (
        windows.shape[0],
        52,
        2,
        32,
    ):
        raise ValueError("reconstruction or code shape changed")

    mean = windows.mean(dim=-1)
    variance = torch.mean((windows - mean.unsqueeze(-1)) ** 2, dim=-1)
    delta_energy = torch.mean(torch.diff(windows, dim=-1) ** 2, dim=-1)
    reconstruction_error = torch.mean((rebuilt - windows) ** 2, dim=-1)
    code_mean = latent.mean(dim=(-1, -2))
    code_rms = torch.sqrt(torch.mean(latent**2, dim=(-1, -2)))
    temporal = torch.stack(
        (
            mean,
            torch.log1p(variance),
            torch.log1p(delta_energy),
            torch.log1p(reconstruction_error),
            code_mean,
            code_rms,
        ),
        dim=-1,
    )

    power = _one_sided_psd(windows)
    total = torch.clamp(torch.sum(power[:, :, 1:65], dim=-1), min=1.0e-12)
    spectral: list[Tensor] = []
    for start, stop in PSD_BANDS:
        share = torch.sum(power[:, :, start:stop], dim=-1) / total
        spectral.append(torch.log(share + 1.0e-12))
    q = power[:, :, 1:65] / total.unsqueeze(-1)
    entropy = -torch.sum(q * torch.log(q + 1.0e-12), dim=-1) / math.log(64.0)
    frequencies = torch.arange(1, 65, device=windows.device, dtype=windows.dtype) / 64.0
    centroid = torch.sum(q * frequencies, dim=-1)
    spectral.extend((entropy, centroid))
    result = torch.cat((temporal, torch.stack(spectral, dim=-1)), dim=-1)
    if tuple(result.shape[1:]) != (52, 14) or not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("channel feature extraction failed")
    return result


def _batch_correlation(left: Tensor, right: Tensor) -> Tensor:
    left0 = left - left.mean(dim=-1, keepdim=True)
    right0 = right - right.mean(dim=-1, keepdim=True)
    numerator = torch.mean(left0 * right0, dim=-1)
    denominator = torch.sqrt(
        torch.mean(left0**2, dim=-1) * torch.mean(right0**2, dim=-1)
    )
    return numerator / torch.clamp(denominator, min=1.0e-8)


def _welch_coherence(u: Tensor, y: Tensor) -> Tensor:
    starts = tuple(range(0, 97, 16))
    u_segments = torch.stack([u[:, start : start + 32] for start in starts], dim=1)
    y_segments = torch.stack([y[:, start : start + 32] for start in starts], dim=1)
    u_segments = u_segments - u_segments.mean(dim=-1, keepdim=True)
    y_segments = y_segments - y_segments.mean(dim=-1, keepdim=True)
    hann = torch.hann_window(32, periodic=True, device=u.device, dtype=u.dtype)
    uf = torch.fft.rfft(u_segments * hann, dim=-1)
    yf = torch.fft.rfft(y_segments * hann, dim=-1)
    window_energy = torch.sum(hann**2)
    sxx = torch.mean(torch.abs(uf) ** 2 / window_energy, dim=1)
    syy = torch.mean(torch.abs(yf) ** 2 / window_energy, dim=1)
    sxy = torch.mean(uf * torch.conj(yf) / window_energy, dim=1)
    # Frozen one-sided convention: double interior bins k=1..15, but not
    # DC (k=0) or the even-length Nyquist bin (k=16).
    sxx = sxx.clone()
    syy = syy.clone()
    sxy = sxy.clone()
    sxx[:, 1:16] *= 2.0
    syy[:, 1:16] *= 2.0
    sxy[:, 1:16] *= 2.0
    coherence = torch.clamp(
        torch.abs(sxy) ** 2 / (sxx * syy + 1.0e-12), min=0.0, max=1.0
    )
    return torch.sum(coherence[:, 1:17] * sxx[:, 1:17], dim=-1) / (
        torch.sum(sxx[:, 1:17], dim=-1) + 1.0e-12
    )


def loop_features(normalized_windows: Any, state: LoopReferenceState) -> Tensor:
    windows = _finite_tensor(normalized_windows, name="normalized_windows")
    if windows.ndim != 3 or tuple(windows.shape[1:]) != (52, 128):
        raise ValueError("normalized_windows must have shape [batch,52,128]")
    if not isinstance(state, LoopReferenceState):
        raise TypeError("state must be LoopReferenceState")
    if windows.dtype != torch.float64:
        raise ValueError("loop features must be computed from float64 normalized windows")
    u = windows[:, LOOP_CHANNELS[0], :]
    du = torch.diff(u, dim=-1)
    absolute = torch.abs(du)
    tau = torch.as_tensor(state.dwell_threshold, device=u.device, dtype=u.dtype)
    dwell = torch.mean((absolute <= tau).to(u.dtype), dim=-1)
    preceding_dwell = torch.stack(
        (
            absolute[:, :-4] <= tau,
            absolute[:, 1:-3] <= tau,
            absolute[:, 2:-2] <= tau,
            absolute[:, 3:-1] <= tau,
        ),
        dim=-1,
    ).all(dim=-1)
    current = absolute[:, 4:]
    slip = torch.sum(
        preceding_dwell.to(u.dtype)
        * (current > tau).to(u.dtype)
        * torch.log1p((current / tau) ** 2),
        dim=-1,
    ) / 124.0
    reversal = torch.mean((du[:, 1:] * du[:, :-1] < 0.0).to(u.dtype), dim=-1)
    pieces: list[Tensor] = [dwell, slip, reversal]

    for response_index, channel in enumerate(LOOP_CHANNELS[1:]):
        y = windows[:, channel, :]
        area = torch.sum(
            0.5 * (u[:, 1:] + u[:, :-1]) * torch.diff(y, dim=-1), dim=-1
        )
        denominator = 127.0 * (
            torch.std(u, dim=-1, correction=0) * torch.std(y, dim=-1, correction=0)
            + 1.0e-8
        )
        area = area / denominator
        forward: list[Tensor] = []
        reverse: list[Tensor] = []
        for lag in range(1, 9):
            forward.append(torch.abs(_batch_correlation(u[:, :-lag], y[:, lag:])))
            reverse.append(torch.abs(_batch_correlation(y[:, :-lag], u[:, lag:])))
        asymmetry = torch.stack(forward, dim=-1).max(dim=-1).values - torch.stack(
            reverse, dim=-1
        ).max(dim=-1).values
        beta = torch.as_tensor(
            state.arx_coefficients[response_index], device=u.device, dtype=u.dtype
        )
        residual = y[:, 1:] - (beta[0] + beta[1] * y[:, :-1] + beta[2] * u[:, :-1])
        residual_ratio = torch.log1p(
            torch.mean(residual**2, dim=-1) / state.residual_denominators[response_index]
        )
        coherence = _welch_coherence(u, y)
        pieces.extend((area, asymmetry, residual_ratio, coherence))

    u_as_channels = torch.zeros(
        (u.shape[0], 52, 128), device=u.device, dtype=u.dtype
    )
    u_as_channels[:, 0, :] = u
    u_power = _one_sided_psd(u_as_channels)[:, 0, :]
    u_total = torch.clamp(torch.sum(u_power[:, 1:65], dim=-1), min=1.0e-12)
    concentration = torch.max(u_power[:, 1:65], dim=-1).values / u_total
    pieces.append(concentration)
    result = torch.stack(pieces, dim=-1)
    if result.shape[1] != 12 or not bool(torch.isfinite(result).all().item()):
        raise FloatingPointError("loop feature extraction failed")
    return result


@torch.no_grad()
def extract_window_features(
    raw_windows: Any,
    *,
    normalizer: ChannelNormalizer,
    autoencoder: ChannelDenoisingAE,
    loop_reference: LoopReferenceState,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    values = _finite_tensor(raw_windows, name="raw_windows")
    if values.ndim < 3 or tuple(values.shape[-2:]) != (52, 128):
        raise ValueError("raw_windows must end in shape [52,128]")
    leading = values.shape[:-2]
    flat = values.reshape(-1, 52, 128)
    channels: list[Tensor] = []
    loops: list[Tensor] = []
    autoencoder.eval()
    ae_parameter = next(autoencoder.parameters())
    if ae_parameter.dtype != torch.float32:
        raise RuntimeError("the AE inference boundary must remain float32")
    for start in range(0, flat.shape[0], batch_size):
        raw = flat[start : start + batch_size]
        normalized64 = normalizer.transform_windows(raw)
        normalized32 = normalized64.to(device=ae_parameter.device, dtype=torch.float32)
        reconstruction, code = autoencoder(normalized32)
        channels.append(channel_features(normalized32, reconstruction, code))
        # The loop branch is the only fixed-feature path whose arithmetic must
        # retain source precision. Cast exactly once at the neural boundary.
        loops.append(loop_features(normalized64, loop_reference).to(torch.float32))
    channel_result = torch.cat(channels, dim=0).reshape(*leading, 52, 14)
    loop_result = torch.cat(loops, dim=0).reshape(*leading, 12)
    return channel_result, loop_result


__all__ = [
    "AETrainingConfig",
    "CHANNELS",
    "CLIP",
    "ChannelDenoisingAE",
    "ChannelNormalizer",
    "FAULT_IDS",
    "LOOP_CHANNELS",
    "LoopReferenceState",
    "MONITOR_COUNT",
    "MONITOR_ENDPOINTS",
    "PREFIX_COUNT",
    "PSD_BANDS",
    "RUN_LENGTH",
    "WINDOW_ENDPOINTS",
    "WINDOW_LENGTH",
    "channel_features",
    "configure_determinism",
    "extract_inclusive_windows",
    "extract_window_features",
    "fit_loop_reference",
    "loop_features",
    "split_prefix_monitor",
    "train_denoising_ae",
    "validate_runs",
]
