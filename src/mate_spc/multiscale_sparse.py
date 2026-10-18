"""Causal multi-scale sparse dynamic feature envelope detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .dynamic_common import (
    FamilyEnvelopePath,
    calibrate_run_envelopes,
    family_persistence_path,
    strict_robust_location_scale,
    validate_named_arrays,
)


@dataclass(frozen=True)
class MultiScaleSparseResult:
    """One run's fixed nine-mechanism sparse dynamic envelope."""

    family: FamilyEnvelopePath
    component_names: tuple[str, ...]

    @property
    def component_score(self) -> np.ndarray:
        return self.family.component_score

    @property
    def component_persistence(self) -> np.ndarray:
        return self.family.component_persistence

    @property
    def persistence(self) -> np.ndarray:
        return self.family.persistence

    @property
    def eligible(self) -> np.ndarray:
        return self.family.eligible

    @property
    def alarm(self) -> np.ndarray:
        return self.family.alarm

    @property
    def threshold(self) -> float | None:
        return self.family.threshold


def _rolling_statistics(
    standardized: np.ndarray,
    window: int,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Return causal channelwise mean, log-variance, and log-difference energy."""
    matrix = np.asarray(standardized, dtype=np.float64)
    length, dimension = matrix.shape
    if window < 2 or length < window:
        raise ValueError("rolling window must be at least two and fit in the run")
    windows = np.lib.stride_tricks.sliding_window_view(
        matrix, window_shape=window, axis=0
    )
    means = np.mean(windows, axis=-1)
    centered = windows - means[:, :, None]
    variance = np.sum(centered**2, axis=-1) / (window - 1)
    difference_energy = np.mean(np.diff(windows, axis=-1) ** 2, axis=-1)
    expected = (length - window + 1, dimension)
    if means.shape != expected or difference_energy.shape != expected:
        raise RuntimeError("rolling statistic alignment is invalid")
    return {
        "mean": means,
        "log_variance": np.log(variance + epsilon),
        "log_difference_energy": np.log(difference_energy + epsilon),
    }


class CausalMultiScaleSparseMonitor:
    """Fixed fallback detector for sparse mean, variance, and dynamic changes."""

    DESIGN_RUN_COUNT = 20
    CALIBRATION_RUN_COUNT = 100
    FEATURE_COUNT = 52
    WINDOWS = (5, 20, 60)
    STATISTICS = ("mean", "log_variance", "log_difference_energy")
    TOP_K = 3
    EPSILON = 1e-8
    PERSISTENCE_WINDOW = 3
    ALPHA = 0.05

    def __init__(self) -> None:
        self.design_fitted_ = False
        self.fitted_ = False

    @property
    def component_names_(self) -> tuple[str, ...]:
        return tuple(
            f"w{window}_{statistic}"
            for window in self.WINDOWS
            for statistic in self.STATISTICS
        )

    @property
    def valid_start_(self) -> int:
        return max(self.WINDOWS) - 1

    def fit_design(
        self,
        design_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "CausalMultiScaleSparseMonitor":
        """Freeze every channel/group normalization using healthy design only."""
        records = validate_named_arrays(
            design_runs,
            label="multi-scale design",
            required_count=self.DESIGN_RUN_COUNT,
            minimum_length=max(self.WINDOWS) + self.PERSISTENCE_WINDOW,
            feature_count=self.FEATURE_COUNT,
        )
        pooled_raw = np.concatenate([matrix for _, matrix in records], axis=0)
        raw_center, raw_scale, raw_fallback = strict_robust_location_scale(
            pooled_raw, axis=0
        )
        standardized = tuple(
            (name, (matrix - raw_center) / raw_scale)
            for name, matrix in records
        )
        by_run = {
            name: {
                (window, statistic): values
                for window in self.WINDOWS
                for statistic, values in _rolling_statistics(
                    matrix, window, self.EPSILON
                ).items()
            }
            for name, matrix in standardized
        }

        channel_center: dict[tuple[int, str], np.ndarray] = {}
        channel_scale: dict[tuple[int, str], np.ndarray] = {}
        channel_fallback: dict[tuple[int, str], np.ndarray] = {}
        energy_center: dict[tuple[int, str], float] = {}
        energy_scale: dict[tuple[int, str], float] = {}
        energy_fallback: dict[tuple[int, str], bool] = {}
        group_design_counts: dict[tuple[int, str], int] = {}
        for window in self.WINDOWS:
            for statistic in self.STATISTICS:
                key = (window, statistic)
                pooled = np.concatenate(
                    [by_run[name][key] for name, _ in standardized], axis=0
                )
                center, scale, fallback = strict_robust_location_scale(
                    pooled, axis=0
                )
                normalized = (pooled - center) / scale
                top = np.partition(
                    normalized**2,
                    normalized.shape[1] - self.TOP_K,
                    axis=1,
                )[
                    :, -self.TOP_K :
                ]
                logged_energy = np.log1p(np.mean(top, axis=1))
                e_center, e_scale, e_fallback = strict_robust_location_scale(
                    logged_energy, axis=None
                )
                channel_center[key] = center
                channel_scale[key] = scale
                channel_fallback[key] = fallback
                energy_center[key] = float(e_center)
                energy_scale[key] = float(e_scale)
                energy_fallback[key] = bool(e_fallback)
                group_design_counts[key] = int(pooled.shape[0])

        self.raw_center_ = raw_center
        self.raw_scale_ = raw_scale
        self.raw_scale_fallback_ = raw_fallback
        self.channel_center_ = channel_center
        self.channel_scale_ = channel_scale
        self.channel_scale_fallback_ = channel_fallback
        self.energy_center_ = energy_center
        self.energy_scale_ = energy_scale
        self.energy_scale_fallback_ = energy_fallback
        self.group_design_counts_ = group_design_counts
        self.design_run_names_ = tuple(name for name, _ in records)
        self.design_fitted_ = True
        self.fitted_ = False
        return self

    def _check_design(self) -> None:
        if not self.design_fitted_:
            raise RuntimeError("multi-scale design parameters have not been fitted")

    def _component_scores(self, values: np.ndarray) -> np.ndarray:
        self._check_design()
        matrix = np.asarray(values, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.FEATURE_COUNT
            or matrix.shape[0] < max(self.WINDOWS) + self.PERSISTENCE_WINDOW
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "multi-scale scoring requires a complete finite (time, 52) run"
            )
        standardized = (matrix - self.raw_center_) / self.raw_scale_
        component = np.full(
            (matrix.shape[0], len(self.component_names_)),
            np.nan,
            dtype=np.float64,
        )
        position = 0
        for window in self.WINDOWS:
            statistics = _rolling_statistics(standardized, window, self.EPSILON)
            for statistic in self.STATISTICS:
                key = (window, statistic)
                normalized = (
                    statistics[statistic] - self.channel_center_[key]
                ) / self.channel_scale_[key]
                top = np.partition(
                    normalized**2,
                    normalized.shape[1] - self.TOP_K,
                    axis=1,
                )[
                    :, -self.TOP_K :
                ]
                logged_energy = np.log1p(np.mean(top, axis=1))
                component[window - 1 :, position] = (
                    logged_energy - self.energy_center_[key]
                ) / self.energy_scale_[key]
                position += 1
        return component

    def path_from_component_scores(
        self,
        component_score: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> MultiScaleSparseResult:
        self._check_design()
        component = np.asarray(component_score, dtype=np.float64)
        if component.ndim != 2 or component.shape[1] != len(self.component_names_):
            raise ValueError("multi-scale component score matrix has invalid shape")
        family = family_persistence_path(
            component,
            valid_start=self.valid_start_,
            window=self.PERSISTENCE_WINDOW,
            threshold=threshold,
        )
        return MultiScaleSparseResult(
            family=family,
            component_names=self.component_names_,
        )

    def decision_path(
        self,
        values: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> MultiScaleSparseResult:
        component = self._component_scores(values)
        return self.path_from_component_scores(component, threshold=threshold)

    def calibrate_envelopes(
        self,
        calibration_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "CausalMultiScaleSparseMonitor":
        self._check_design()
        summary = calibrate_run_envelopes(
            calibration_runs,
            required_count=self.CALIBRATION_RUN_COUNT,
            alpha=self.ALPHA,
        )
        self.threshold_ = summary.threshold
        self.conformal_summary_ = summary
        self.calibration_run_names_ = tuple(
            name for name, _ in summary.run_maxima
        )
        self.fitted_ = True
        return self

    def calibrate(
        self,
        calibration_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "CausalMultiScaleSparseMonitor":
        records = validate_named_arrays(
            calibration_runs,
            label="multi-scale calibration",
            required_count=self.CALIBRATION_RUN_COUNT,
            minimum_length=max(self.WINDOWS) + self.PERSISTENCE_WINDOW,
            feature_count=self.FEATURE_COUNT,
        )
        return self.calibrate_envelopes(
            (name, self.decision_path(matrix).persistence)
            for name, matrix in records
        )

    def predict(self, values: np.ndarray) -> MultiScaleSparseResult:
        if not self.fitted_:
            raise RuntimeError("multi-scale monitor has not been calibrated")
        return self.decision_path(values, threshold=self.threshold_)

    @property
    def calibration_summary_(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("multi-scale monitor has not been calibrated")
        group_counts = {
            f"w{window}_{statistic}": self.group_design_counts_[(window, statistic)]
            for window in self.WINDOWS
            for statistic in self.STATISTICS
        }
        channel_fallback = {
            f"w{window}_{statistic}": int(
                np.sum(self.channel_scale_fallback_[(window, statistic)])
            )
            for window in self.WINDOWS
            for statistic in self.STATISTICS
        }
        energy_fallback = {
            f"w{window}_{statistic}": self.energy_scale_fallback_[
                (window, statistic)
            ]
            for window in self.WINDOWS
            for statistic in self.STATISTICS
        }
        return {
            "method": "causal_multiscale_sparse_dynamic_family_envelope_3of3",
            "design_run_count": self.DESIGN_RUN_COUNT,
            "feature_count": self.FEATURE_COUNT,
            "raw_normalization": "design median and 1.4826*MAD; SD fallback",
            "raw_scale_sd_fallback_count": int(np.sum(self.raw_scale_fallback_)),
            "windows_samples": list(self.WINDOWS),
            "windows_minutes": [3 * value for value in self.WINDOWS],
            "statistics": list(self.STATISTICS),
            "difference_energy_epsilon": self.EPSILON,
            "sparse_energy": "mean of largest 3 squared standardized channels",
            "top_k": self.TOP_K,
            "component_transform": "log1p sparse energy",
            "component_names": list(self.component_names_),
            "group_design_counts": group_counts,
            "channel_scale_sd_fallback_count": channel_fallback,
            "energy_scale_sd_fallback": energy_fallback,
            "family_rule": "3/3 persistence per component, then maximum",
            "valid_score_start_zero_based": self.valid_start_,
            "first_alarm_eligible_zero_based": (
                self.valid_start_ + self.PERSISTENCE_WINDOW - 1
            ),
            **self.conformal_summary_.as_dict(),
        }


__all__ = ["CausalMultiScaleSparseMonitor", "MultiScaleSparseResult"]
