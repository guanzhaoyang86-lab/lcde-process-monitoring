"""Transparent point-imputation baselines used by the paper experiments."""

from __future__ import annotations

import numpy as np

from .utils import as_2d_float, validate_complete


def _validate_incomplete(
    x: np.ndarray,
    observed_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = as_2d_float(x, "x")
    finite = np.isfinite(arr)
    if observed_mask is None:
        mask = finite
    else:
        mask = np.asarray(observed_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("observed_mask must have the same shape as x")
        if np.any(mask & ~finite):
            raise ValueError("observed entries must be finite")
    return arr, mask


def median_impute(
    x: np.ndarray,
    reference: np.ndarray,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Column-median fill using a separate, complete training reference."""
    arr, mask = _validate_incomplete(x, observed_mask)
    ref = validate_complete(reference, "reference")
    if ref.shape[1] != arr.shape[1]:
        raise ValueError("reference and x must have the same number of columns")
    medians = np.median(ref, axis=0)
    return np.where(mask, arr, medians)


def locf_impute(
    x: np.ndarray,
    reference: np.ndarray,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Causal last-observation-carried-forward with training medians as fallback."""
    arr, mask = _validate_incomplete(x, observed_mask)
    ref = validate_complete(reference, "reference")
    if ref.shape[1] != arr.shape[1]:
        raise ValueError("reference and x must have the same number of columns")
    out = arr.copy()
    previous = np.median(ref, axis=0)
    for t in range(out.shape[0]):
        out[t, ~mask[t]] = previous[~mask[t]]
        previous = out[t]
    return out


def linear_interpolate_offline(
    x: np.ndarray,
    reference: np.ndarray,
    observed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Two-sided linear interpolation, reported only as an offline upper bound.

    Because it can use future samples, this baseline must not be described as an
    online deployable method.
    """
    arr, mask = _validate_incomplete(x, observed_mask)
    ref = validate_complete(reference, "reference")
    if ref.shape[1] != arr.shape[1]:
        raise ValueError("reference and x must have the same number of columns")
    out = arr.copy()
    index = np.arange(arr.shape[0])
    fallback = np.median(ref, axis=0)
    for column in range(arr.shape[1]):
        valid = mask[:, column]
        if valid.any():
            out[:, column] = np.interp(index, index[valid], arr[valid, column])
        else:
            out[:, column] = fallback[column]
    return out


def imputation_error(
    truth: np.ndarray,
    imputed: np.ndarray,
    observed_mask: np.ndarray,
    scale: np.ndarray | None = None,
) -> dict[str, float]:
    """MAE/RMSE on artificially hidden cells only."""
    actual = validate_complete(truth, "truth")
    estimate = validate_complete(imputed, "imputed")
    mask = np.asarray(observed_mask, dtype=bool)
    if actual.shape != estimate.shape or mask.shape != actual.shape:
        raise ValueError("truth, imputed, and observed_mask must have identical shapes")
    hidden = ~mask
    if not hidden.any():
        return {"mae": 0.0, "rmse": 0.0, "n_hidden": 0.0}
    error = estimate - actual
    if scale is not None:
        divisor = np.asarray(scale, dtype=float)
        if divisor.shape != (actual.shape[1],) or np.any(divisor <= 0):
            raise ValueError("scale must be a positive vector matching the feature count")
        error = error / divisor
    selected = error[hidden]
    return {
        "mae": float(np.mean(np.abs(selected))),
        "rmse": float(np.sqrt(np.mean(selected**2))),
        "n_hidden": float(selected.size),
    }

