"""Fixed preprocessing and path-level metrics for the SKAB external study."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np


WARMUP_ROWS = 300
STANDARD_DEVIATION_FLOOR = 1.0e-8


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _numeric_vector(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise TypeError(f"{name} must be a real numeric vector")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(result, dtype=np.float64, order="C", copy=True)


def _binary_vector(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if raw.dtype.kind == "b":
        result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    elif raw.dtype.kind in {"i", "u"} and np.isin(raw, (0, 1)).all():
        result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    else:
        raise ValueError(f"{name} must contain only boolean or integer zero/one values")
    return result


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def warmup_standardize(x: Any, warmup: int = WARMUP_ROWS) -> np.ndarray:
    """Standardize every row from statistics fitted on the first warmup rows.

    The population standard deviation is floored coordinate-wise at ``1e-8``.
    No later row contributes to either fitted statistic.
    """
    raw = np.asarray(x)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise TypeError("x must be a real numeric matrix")
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("x must have shape (nonempty_time, nonempty_features)")
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite values")
    count = _integer(warmup, name="warmup", minimum=1)
    if count > values.shape[0]:
        raise ValueError("warmup cannot exceed the number of rows in x")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            center = np.mean(values[:count], axis=0, dtype=np.float64)
            scale = np.std(values[:count], axis=0, ddof=0, dtype=np.float64)
            scale = np.maximum(scale, STANDARD_DEVIATION_FLOOR)
            standardized = (values - center[None, :]) / scale[None, :]
        except FloatingPointError as exc:
            raise ValueError("x is numerically unsafe to standardize") from exc
    if not np.isfinite(standardized).all():
        raise ValueError("standardization produced nonfinite values")
    return _readonly(standardized, dtype=np.float64)


def prepend_warmup(path: Any, total: int, warmup: int) -> np.ndarray:
    """Prepend exactly ``warmup`` zeros to a post-warmup score path."""
    values = _numeric_vector(path, name="path")
    total_count = _integer(total, name="total", minimum=1)
    warmup_count = _integer(warmup, name="warmup", minimum=0)
    if warmup_count >= total_count:
        raise ValueError("warmup must be smaller than total")
    expected = total_count - warmup_count
    if values.size != expected:
        raise ValueError(f"path must contain exactly total-warmup={expected} values")
    result = np.zeros(total_count, dtype=np.float64)
    result[warmup_count:] = values
    return _readonly(result, dtype=np.float64)


def binary_average_precision(y: Any, score: Any) -> float:
    """Compute non-interpolated binary average precision without sklearn.

    Equal-score samples are evaluated as one threshold group, so the result is
    invariant to the order of tied positive and negative observations.
    """
    labels = _binary_vector(y, name="y")
    scores = _numeric_vector(score, name="score")
    if labels.shape != scores.shape:
        raise ValueError("y and score must have identical shape")
    positive_count = int(np.sum(labels))
    if positive_count == 0:
        raise ValueError("y must contain at least one positive value")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order].astype(np.int64)
    cumulative_true = np.cumsum(ordered_labels, dtype=np.int64)
    cumulative_total = np.arange(1, labels.size + 1, dtype=np.int64)
    group_end = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    true_at_threshold = cumulative_true[group_end]
    total_at_threshold = cumulative_total[group_end]
    recall = true_at_threshold.astype(np.float64) / float(positive_count)
    precision = true_at_threshold.astype(np.float64) / total_at_threshold.astype(
        np.float64
    )
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision, dtype=np.float64))


def _timestamps_ns(value: Any, *, expected_length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "M" or raw.ndim != 1 or raw.shape != (expected_length,):
        raise ValueError("timestamps must be an aligned one-dimensional datetime64 array")
    if np.isnat(raw).any():
        raise ValueError("timestamps must not contain NaT")
    timestamps = np.asarray(raw, dtype="datetime64[ns]")
    if not np.array_equal(timestamps.astype(raw.dtype), raw):
        raise ValueError("timestamps must be exactly representable as datetime64[ns]")
    integer_ns = timestamps.astype(np.int64)
    if any(
        int(integer_ns[index]) <= int(integer_ns[index - 1])
        for index in range(1, integer_ns.size)
    ):
        raise ValueError("timestamps must be strictly increasing and unique")
    return integer_ns


def evaluate_skab_path(
    score: Any,
    threshold: Any,
    anomaly: Any,
    timestamps: Any,
) -> dict[str, bool | int | float | None]:
    """Evaluate one unadjusted SKAB path using post-event alarm rising edges.

    A pre-event alarm that remains active at the event onset does not count as
    detection.  Detection requires the first new ``False -> True`` transition
    at or after the first positive anomaly label.  Thresholding is strictly
    greater-than; equality never alarms.
    """
    scores = _numeric_vector(score, name="score")
    labels = _binary_vector(anomaly, name="anomaly")
    if labels.shape != scores.shape:
        raise ValueError("score and anomaly must have identical shape")
    if scores.size < 2:
        raise ValueError("a SKAB path must contain at least two timestamps")
    if not np.any(labels):
        raise ValueError("anomaly must contain at least one event sample")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (Real, np.integer, np.floating)
    ):
        raise TypeError("threshold must be a real number")
    cutoff = float(threshold)
    if not math.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    timestamp_ns = _timestamps_ns(timestamps, expected_length=scores.size)

    alarm = scores > cutoff
    rising = alarm & ~np.r_[False, alarm[:-1]]
    rising_indices = np.flatnonzero(rising)
    event_index = int(np.flatnonzero(labels)[0])
    post_event_rises = rising_indices[rising_indices >= event_index]
    detected = bool(post_event_rises.size)
    detection_index = int(post_event_rises[0]) if detected else None
    delay_seconds = (
        float(
            int(timestamp_ns[detection_index]) - int(timestamp_ns[event_index])
        )
        / 1.0e9
        if detection_index is not None
        else None
    )
    duration_seconds = float(int(timestamp_ns[-1]) - int(timestamp_ns[0])) / 1.0e9
    if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
        raise ValueError("timestamp duration must be finite and positive")

    true_positive = int(np.sum(alarm & labels))
    false_positive = int(np.sum(alarm & ~labels))
    false_negative = int(np.sum(~alarm & labels))
    f1_denominator = 2 * true_positive + false_positive + false_negative
    unadjusted_f1 = (
        0.0 if f1_denominator == 0 else 2.0 * true_positive / f1_denominator
    )
    episodes = int(rising_indices.size)
    return {
        "detected": detected,
        "delay_seconds": delay_seconds,
        "within_60": bool(detected and delay_seconds is not None and delay_seconds <= 60.0),
        "pre_event_false_alarm": bool(np.any(rising_indices < event_index)),
        "occupancy": float(np.mean(alarm, dtype=np.float64)),
        "episodes": episodes,
        "episodes_per_hour": float(episodes * 3600.0 / duration_seconds),
        "unadjusted_f1": float(unadjusted_f1),
        "average_precision": binary_average_precision(labels, scores),
    }


__all__ = [
    "STANDARD_DEVIATION_FLOOR",
    "WARMUP_ROWS",
    "binary_average_precision",
    "evaluate_skab_path",
    "prepend_warmup",
    "warmup_standardize",
]
