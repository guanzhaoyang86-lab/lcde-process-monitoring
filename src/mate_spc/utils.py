"""Numerical helpers kept dependency-light for reproducible experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np


EPS = 1e-10


def as_2d_float(x: np.ndarray | Iterable[Iterable[float]], name: str = "x") -> np.ndarray:
    """Return a finite-or-NaN 2-D float array and reject infinities."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least 2 rows and 1 column")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values")
    return arr


def validate_complete(x: np.ndarray, name: str = "x") -> np.ndarray:
    arr = as_2d_float(x, name)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be complete and finite")
    return arr


def robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise median and robust scale with safe fallbacks."""
    arr = validate_complete(x)
    location = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - location), axis=0)
    scale = 1.4826 * mad
    std = np.std(arr, axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, std)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return location, scale


def higher_quantile(values: np.ndarray, q: float) -> float:
    """Conservative empirical quantile compatible with old NumPy releases."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("cannot compute a quantile from no finite values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    try:
        return float(np.quantile(vals, q, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(vals, q, interpolation="higher"))


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Split-conformal upper quantile: ceil((n+1)(1-alpha))/n."""
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size == 0 or np.isnan(vals).any():
        raise ValueError("calibration scores are empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rank = int(np.ceil((vals.size + 1) * (1.0 - alpha)))
    if rank > vals.size:
        return float("inf")
    rank = max(rank, 1)
    return float(np.partition(vals, rank - 1)[rank - 1])


def ridge_solve(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """Solve multivariate ridge regression while leaving the intercept unpenalized."""
    a = np.asarray(design, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("incompatible design and target matrices")
    penalty = np.eye(a.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(a.T @ a + penalty) @ a.T @ b


def simplex_grid(n_weights: int, step: float) -> np.ndarray:
    """Enumerate a deterministic simplex grid whose rows sum exactly to one."""
    if n_weights < 1:
        raise ValueError("n_weights must be positive")
    if not 0.0 < step <= 1.0:
        raise ValueError("step must lie in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-8):
        raise ValueError("step must evenly divide 1.0")

    rows: list[list[int]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            visit(prefix + [value], remaining - value, slots - 1)

    visit([], units, n_weights)
    return np.asarray(rows, dtype=float) / units


def k_of_w(raw_alarm: np.ndarray, k: int, window: int) -> np.ndarray:
    """Causal k-of-w persistence rule."""
    alarm = np.asarray(raw_alarm, dtype=bool)
    if alarm.ndim != 1:
        raise ValueError("raw_alarm must be one-dimensional")
    if not 1 <= k <= window:
        raise ValueError("require 1 <= k <= window")
    counts = np.convolve(alarm.astype(int), np.ones(window, dtype=int), mode="full")
    counts = counts[: alarm.size]
    return counts >= k


def rolling_kth_largest(values: np.ndarray, k: int, window: int) -> np.ndarray:
    """Causal k-th-largest statistic over a trailing window.

    Infinite sentinels are intentional: ``-inf`` represents an unavailable
    sample and therefore cannot count towards a persisted high-score alarm.
    NaNs remain invalid because their ordering is undefined.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or np.isnan(arr).any():
        raise ValueError("values must be a one-dimensional array without NaNs")
    if not 1 <= k <= window:
        raise ValueError("require 1 <= k <= window")
    out = np.full(arr.size, -np.inf, dtype=float)
    for t in range(arr.size):
        start = max(0, t - window + 1)
        current = arr[start : t + 1]
        if current.size >= k:
            out[t] = float(np.partition(current, current.size - k)[current.size - k])
    return out


def ewma_with_resets(
    values: np.ndarray,
    reset: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    """Causal EWMA that abstains and forgets its state at reset samples."""
    arr = np.asarray(values, dtype=float)
    reset_mask = np.asarray(reset, dtype=bool)
    if arr.ndim != 1 or reset_mask.shape != arr.shape or np.isnan(arr).any():
        raise ValueError("values and reset must be aligned one-dimensional arrays without NaNs")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    out = np.full(arr.size, -np.inf, dtype=float)
    state: float | None = None
    for index, value in enumerate(arr):
        if reset_mask[index]:
            state = None
            continue
        state = float(value) if state is None else (
            smoothing * float(value) + (1.0 - smoothing) * state
        )
        out[index] = state
    return out


def segmented_rolling_kth_largest(
    values: np.ndarray,
    reset: np.ndarray,
    k: int,
    window: int,
) -> np.ndarray:
    """Apply a trailing k-th-largest rule independently between reset samples."""
    arr = np.asarray(values, dtype=float)
    reset_mask = np.asarray(reset, dtype=bool)
    if arr.ndim != 1 or reset_mask.shape != arr.shape or np.isnan(arr).any():
        raise ValueError("values and reset must be aligned one-dimensional arrays without NaNs")
    if not 1 <= k <= window:
        raise ValueError("require 1 <= k <= window")
    out = np.full(arr.size, -np.inf, dtype=float)
    start = 0
    for index in range(arr.size + 1):
        boundary = index == arr.size or reset_mask[index]
        if not boundary:
            continue
        if index > start:
            out[start:index] = rolling_kth_largest(arr[start:index], k, window)
        start = index + 1
    return out
