"""Shared, leakage-safe utilities for causal dynamic process monitors."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np


RIETH_FEATURE_COUNT = 52
ROBUST_SCALE_FACTOR = 1.4826


@dataclass(frozen=True)
class FamilyEnvelopePath:
    """Component scores, within-component persistence, and family envelope."""

    component_score: np.ndarray
    component_persistence: np.ndarray
    persistence: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    threshold: float | None


@dataclass(frozen=True)
class RunConformalSummary:
    """Auditable run-level conformal calibration for a fixed score family."""

    healthy_run_count: int
    alpha: float
    rank_one_based: int
    threshold: float
    rank_driver_run: str
    rank_driver_score: float
    run_maxima: tuple[tuple[str, float], ...]
    order_statistics: tuple[tuple[int, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "healthy_run_count": self.healthy_run_count,
            "alpha": self.alpha,
            "calibration_unit": "independent healthy run maximum",
            "conformal_rank_one_based": self.rank_one_based,
            "threshold": self.threshold,
            "alarm_comparison": "strictly_greater_than",
            "rank_driver": {
                "run": self.rank_driver_run,
                "score": self.rank_driver_score,
            },
            "order_statistics": {
                str(rank): value for rank, value in self.order_statistics
            },
            "run_maxima": [
                {"run": name, "maximum": value}
                for name, value in self.run_maxima
            ],
        }


def validate_named_arrays(
    runs: Iterable[tuple[str, np.ndarray]],
    *,
    label: str,
    required_count: int,
    minimum_length: int,
    feature_count: int = RIETH_FEATURE_COUNT,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Validate independent, complete multivariate runs without joining them."""
    records: list[tuple[str, np.ndarray]] = []
    for item in runs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{label} entries must be (run_name, matrix) tuples")
        name, values = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} run names must be nonempty strings")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != feature_count:
            raise ValueError(
                f"{label} run {name!r} must have shape (time, {feature_count})"
            )
        if matrix.shape[0] < minimum_length or not np.isfinite(matrix).all():
            raise ValueError(
                f"{label} run {name!r} must contain at least {minimum_length} "
                "complete finite samples"
            )
        records.append((name, matrix.copy()))
    if len(records) != required_count:
        raise ValueError(
            f"{label} requires exactly {required_count} independent runs; "
            f"received {len(records)}"
        )
    names = [name for name, _ in records]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} run names must be unique")
    return tuple(records)


def strict_robust_location_scale(
    values: np.ndarray,
    *,
    axis: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/MAD scale with per-coordinate sample-SD fallback and no masking."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("robust scaling requires nonempty finite values")
    center = np.median(array, axis=axis)
    expanded = center if axis is None else np.expand_dims(center, axis=axis)
    mad = np.median(np.abs(array - expanded), axis=axis)
    scale = ROBUST_SCALE_FACTOR * mad
    fallback = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(fallback):
        sample_sd = np.std(array, axis=axis, ddof=1)
        scale = np.where(fallback, sample_sd, scale)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("robust location/scale produced a non-finite value")
    if np.any(scale <= 0.0):
        indices = np.flatnonzero(np.asarray(scale).reshape(-1) <= 0.0).tolist()
        raise ValueError(f"design scale remains degenerate at coordinates {indices}")
    return (
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        np.asarray(fallback, dtype=bool),
    )


def family_persistence_path(
    component_score: np.ndarray,
    *,
    valid_start: int,
    window: int = 3,
    threshold: float | None = None,
) -> FamilyEnvelopePath:
    """Apply persistence within each mechanism before taking a family maximum."""
    score = np.asarray(component_score, dtype=np.float64)
    if score.ndim != 2 or score.shape[1] < 1:
        raise ValueError("component_score must have shape (time, components)")
    if (
        isinstance(valid_start, (bool, np.bool_))
        or not isinstance(valid_start, (int, np.integer))
        or valid_start < 0
    ):
        raise ValueError("valid_start must be a nonnegative integer")
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 1
    ):
        raise ValueError("window must be a positive integer")
    if valid_start + window > score.shape[0]:
        raise ValueError("score path is too short for the persistence window")
    if not np.isfinite(score[valid_start:]).all():
        raise ValueError("all component scores must be finite after valid_start")
    if threshold is not None and not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite when supplied")

    component_persistence = np.full_like(score, -np.inf)
    first = valid_start + window - 1
    for index in range(first, score.shape[0]):
        component_persistence[index] = np.min(
            score[index - window + 1 : index + 1], axis=0
        )
    persistence = np.full(score.shape[0], -np.inf, dtype=np.float64)
    persistence[first:] = np.max(component_persistence[first:], axis=1)
    eligible = np.zeros(score.shape[0], dtype=bool)
    eligible[first:] = True
    alarm = (
        np.zeros(score.shape[0], dtype=bool)
        if threshold is None
        else eligible & (persistence > float(threshold))
    )
    return FamilyEnvelopePath(
        component_score=score.copy(),
        component_persistence=component_persistence,
        persistence=persistence,
        eligible=eligible,
        alarm=alarm,
        threshold=None if threshold is None else float(threshold),
    )


def calibrate_run_envelopes(
    runs: Iterable[tuple[str, np.ndarray]],
    *,
    required_count: int = 100,
    alpha: float = 0.05,
) -> RunConformalSummary:
    """Calibrate one threshold from maxima of independent run envelopes."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    records: list[tuple[str, float]] = []
    for item in runs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("calibration entries must be (run_name, envelope) tuples")
        name, values = item
        if not isinstance(name, str) or not name:
            raise ValueError("calibration run names must be nonempty strings")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"calibration envelope for {name!r} is invalid")
        finite_indices = np.flatnonzero(np.isfinite(array))
        if finite_indices.size == 0:
            raise ValueError(f"calibration envelope for {name!r} is invalid")
        first = int(finite_indices[0])
        if (
            not np.isneginf(array[:first]).all()
            or not np.isfinite(array[first:]).all()
        ):
            raise ValueError(
                f"calibration envelope for {name!r} must be leading -inf "
                "followed by finite values"
            )
        records.append((name, float(np.max(array[first:]))))
    if len(records) != required_count:
        raise ValueError(
            f"calibration requires exactly {required_count} independent runs; "
            f"received {len(records)}"
        )
    if len({name for name, _ in records}) != len(records):
        raise ValueError("calibration run names must be unique")

    rank = int(ceil((required_count + 1) * (1.0 - float(alpha))))
    if rank > required_count:
        raise ValueError(
            "finite conformal threshold unavailable for this alpha/run count"
        )
    ordered = sorted(records, key=lambda item: (item[1], item[0]))
    driver_name, threshold = ordered[rank - 1]
    selected_ranks = tuple(sorted({1, 90, 95, rank, 99, required_count}))
    selected_ranks = tuple(value for value in selected_ranks if value <= required_count)
    return RunConformalSummary(
        healthy_run_count=required_count,
        alpha=float(alpha),
        rank_one_based=rank,
        threshold=float(threshold),
        rank_driver_run=driver_name,
        rank_driver_score=float(threshold),
        run_maxima=tuple(records),
        order_statistics=tuple(
            (position, float(ordered[position - 1][1]))
            for position in selected_ranks
        ),
    )


__all__ = [
    "FamilyEnvelopePath",
    "RIETH_FEATURE_COUNT",
    "RunConformalSummary",
    "calibrate_run_envelopes",
    "family_persistence_path",
    "strict_robust_location_scale",
    "validate_named_arrays",
]
