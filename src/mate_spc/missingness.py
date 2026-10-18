"""Mechanism-driven missingness generators for sensor time series.

Masks use ``True`` for observed and ``False`` for missing.  The generators are
for controlled stress tests; they never alter the original data in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .utils import EPS, validate_complete


@dataclass(frozen=True)
class MissingnessConfig:
    """Configuration for one missingness stress scenario."""

    mechanism: str = "mcar"
    rate: float = 0.1
    min_block: int = 3
    max_block: int = 30
    channel_fraction: float = 0.3
    mnar_strength: float = 1.5

    def validate(self) -> None:
        allowed = {"mcar", "blackout", "multi_blackout", "mnar"}
        if self.mechanism not in allowed:
            raise ValueError(
                f"unknown mechanism {self.mechanism!r}; expected one of {sorted(allowed)}"
            )
        if not 0.0 <= self.rate < 1.0:
            raise ValueError("rate must lie in [0, 1)")
        if self.min_block < 1 or self.max_block < self.min_block:
            raise ValueError("block lengths must satisfy 1 <= min_block <= max_block")
        if not 0.0 < self.channel_fraction <= 1.0:
            raise ValueError("channel_fraction must lie in (0, 1]")
        if self.mnar_strength < 0.0:
            raise ValueError("mnar_strength must be nonnegative")

    def identifier(self) -> str:
        """Stable identifier that keeps distinct parameterizations separate."""
        return (
            f"{self.mechanism}|rate={self.rate:.8g}|min={self.min_block}"
            f"|max={self.max_block}|channels={self.channel_fraction:.8g}"
            f"|mnar={self.mnar_strength:.8g}"
        )


DEFAULT_SCENARIOS: tuple[MissingnessConfig, ...] = (
    MissingnessConfig("mcar", rate=0.10),
    MissingnessConfig("blackout", rate=0.10, min_block=3, max_block=20),
    MissingnessConfig(
        "multi_blackout", rate=0.15, min_block=5, max_block=40, channel_fraction=0.35
    ),
    MissingnessConfig(
        "multi_blackout", rate=0.05, min_block=1, max_block=8, channel_fraction=1.0
    ),
    MissingnessConfig("mnar", rate=0.10, mnar_strength=1.5),
)


def _ensure_initial_context(mask: np.ndarray) -> np.ndarray:
    """Keep the first row observed so every causal imputer has a valid state."""
    if mask.shape[0] > 0:
        mask[0, :] = True
    return mask


def _single_channel_blocks(
    n: int,
    d: int,
    target_cells: int,
    min_block: int,
    max_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((n, d), dtype=bool)
    missing = 0
    attempts = 0
    max_attempts = max(100, target_cells * 10)
    while missing < target_cells and attempts < max_attempts:
        attempts += 1
        channel = int(rng.integers(0, d))
        length = int(rng.integers(min_block, max_block + 1))
        length = min(length, max(n - 1, 1))
        start_max = max(n - length + 1, 2)
        start = int(rng.integers(1, start_max)) if start_max > 1 else 0
        stop = min(start + length, n)
        before = int((~mask[start:stop, channel]).sum())
        mask[start:stop, channel] = False
        missing += (stop - start) - before
    return mask


def _multi_channel_blocks(
    n: int,
    d: int,
    target_cells: int,
    min_block: int,
    max_block: int,
    channel_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((n, d), dtype=bool)
    missing = 0
    attempts = 0
    max_attempts = max(100, target_cells * 5)
    n_channels = min(d, max(1, int(np.ceil(channel_fraction * d))))
    while missing < target_cells and attempts < max_attempts:
        attempts += 1
        channels = rng.choice(d, size=n_channels, replace=False)
        length = int(rng.integers(min_block, max_block + 1))
        length = min(length, max(n - 1, 1))
        start_max = max(n - length + 1, 2)
        start = int(rng.integers(1, start_max)) if start_max > 1 else 0
        stop = min(start + length, n)
        region = mask[start:stop][:, channels]
        before = int((~region).sum())
        mask[np.ix_(np.arange(start, stop), channels)] = False
        missing += (stop - start) * n_channels - before
    return mask


def _mnar_probabilities(x: np.ndarray, target_rate: float, strength: float) -> np.ndarray:
    """Value-and-slope-dependent missing probabilities with calibrated mean rate."""
    center = np.median(x, axis=0)
    scale = np.median(np.abs(x - center), axis=0) * 1.4826
    fallback = np.std(x, axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, fallback)
    scale = np.where(scale > EPS, scale, 1.0)
    z = np.abs((x - center) / scale)
    slope = np.vstack([np.zeros((1, x.shape[1])), np.abs(np.diff(x, axis=0)) / scale])
    risk = strength * (0.65 * z + 0.35 * slope)

    # Find an intercept making the average logistic probability equal target_rate.
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        prob = 1.0 / (1.0 + np.exp(-np.clip(mid + risk, -50.0, 50.0)))
        if float(prob.mean()) > target_rate:
            hi = mid
        else:
            lo = mid
    return 1.0 / (1.0 + np.exp(-np.clip(0.5 * (lo + hi) + risk, -50.0, 50.0)))


def generate_observed_mask(
    x: np.ndarray,
    config: MissingnessConfig,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Generate an observed-value mask under a controlled missingness mechanism."""
    arr = validate_complete(x)
    config.validate()
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n, d = arr.shape
    target = int(round(config.rate * n * d))

    if config.rate == 0.0:
        return np.ones_like(arr, dtype=bool)
    if config.mechanism == "mcar":
        mask = rng.random((n, d)) >= config.rate
    elif config.mechanism == "blackout":
        mask = _single_channel_blocks(
            n, d, target, config.min_block, config.max_block, rng
        )
    elif config.mechanism == "multi_blackout":
        mask = _multi_channel_blocks(
            n,
            d,
            target,
            config.min_block,
            config.max_block,
            config.channel_fraction,
            rng,
        )
    else:
        probability = _mnar_probabilities(arr, config.rate, config.mnar_strength)
        mask = rng.random((n, d)) >= probability

    return _ensure_initial_context(mask)


def apply_mask(x: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Return a copy with unobserved cells replaced by NaN."""
    arr = validate_complete(x)
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.shape != arr.shape:
        raise ValueError("observed_mask must have the same shape as x")
    out = arr.copy()
    out[~mask] = np.nan
    return out


def gap_lengths(observed_mask: np.ndarray) -> np.ndarray:
    """Consecutive missing-run length for each time and sensor."""
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("observed_mask must be two-dimensional")
    gaps = np.zeros(mask.shape, dtype=int)
    for t in range(mask.shape[0]):
        if t == 0:
            gaps[t] = (~mask[t]).astype(int)
        else:
            gaps[t] = np.where(mask[t], 0, gaps[t - 1] + 1)
    return gaps


def severity_labels(
    observed_mask: np.ndarray,
    low_fraction: float = 0.2,
    medium_fraction: float = 0.4,
    short_gap: int = 2,
    medium_gap: int = 20,
) -> np.ndarray:
    """Map the current missing fraction and longest active gap to four strata.

    0 = complete, 1 = mild, 2 = moderate, 3 = severe.  These strata drive both
    task-aligned ensemble weights and missingness-conditional calibration.
    """
    mask = np.asarray(observed_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("observed_mask must be two-dimensional")
    if not 0.0 <= low_fraction <= medium_fraction <= 1.0:
        raise ValueError("invalid missing-fraction cut points")
    gaps = gap_lengths(mask)
    fraction = (~mask).mean(axis=1)
    longest = gaps.max(axis=1)
    labels = np.full(mask.shape[0], 3, dtype=int)
    labels[fraction == 0.0] = 0
    mild = (fraction > 0.0) & (fraction <= low_fraction) & (longest <= short_gap)
    labels[mild] = 1
    moderate = (
        (fraction > 0.0)
        & (fraction <= medium_fraction)
        & (longest <= medium_gap)
        & ~mild
    )
    labels[moderate] = 2
    return labels


def scenario_cycle(
    scenarios: Sequence[MissingnessConfig] | None,
    repeats: int,
) -> list[MissingnessConfig]:
    """Repeat a validated scenario list deterministically."""
    chosen = tuple(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    if not chosen:
        raise ValueError("at least one missingness scenario is required")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    for config in chosen:
        config.validate()
    return [chosen[i % len(chosen)] for i in range(repeats * len(chosen))]
