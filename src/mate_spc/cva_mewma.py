"""Causal regularized CVA monitor with signed vector MEWMA dynamics."""

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
class CVAMEWMAResult:
    """One run's causal CVA state/residual decision path."""

    state_energy: np.ndarray
    residual_energy: np.ndarray
    family: FamilyEnvelopePath

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


def _past_blocks(values: np.ndarray, order: int, *, stop: int | None = None) -> np.ndarray:
    """Return rows [x_t, x_t-1, ...] without crossing a run boundary."""
    matrix = np.asarray(values, dtype=np.float64)
    end = matrix.shape[0] if stop is None else int(stop)
    return np.concatenate(
        [matrix[order - 1 - lag : end - lag] for lag in range(order)],
        axis=1,
    )


def _future_blocks(values: np.ndarray, past: int, future: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    length = matrix.shape[0]
    return np.concatenate(
        [
            matrix[past - 1 + lead : length - future + lead]
            for lead in range(1, future + 1)
        ],
        axis=1,
    )


def _symmetric_inverse_sqrt(covariance: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(
        0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    )
    if not np.isfinite(values).all() or float(values[0]) <= 0.0:
        raise ValueError("regularized covariance is not positive definite")
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


class CausalCVAMEWMAMonitor:
    """Frozen causal CVA detector; future samples are used only during design."""

    DESIGN_RUN_COUNT = 20
    CALIBRATION_RUN_COUNT = 100
    FEATURE_COUNT = 52
    PAST_ORDER = 5
    FUTURE_ORDER = 5
    RIDGE_FRACTION = 0.001
    CANONICAL_ENERGY = 0.90
    EWMA_LAMBDA = 0.2
    PERSISTENCE_WINDOW = 3
    ALPHA = 0.05

    def __init__(self) -> None:
        self.design_fitted_ = False
        self.fitted_ = False

    @property
    def valid_start_(self) -> int:
        return self.PAST_ORDER - 1

    def fit_design(
        self,
        design_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "CausalCVAMEWMAMonitor":
        """Fit scaling, regularized CVA subspaces, and energy normalization."""
        minimum = self.PAST_ORDER + self.FUTURE_ORDER
        records = validate_named_arrays(
            design_runs,
            label="CVA design",
            required_count=self.DESIGN_RUN_COUNT,
            minimum_length=minimum,
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

        past_rows: list[np.ndarray] = []
        future_rows: list[np.ndarray] = []
        for _, matrix in standardized:
            stop = matrix.shape[0] - self.FUTURE_ORDER
            past_rows.append(_past_blocks(matrix, self.PAST_ORDER, stop=stop))
            future_rows.append(
                _future_blocks(matrix, self.PAST_ORDER, self.FUTURE_ORDER)
            )
        past = np.concatenate(past_rows, axis=0)
        future = np.concatenate(future_rows, axis=0)
        if past.shape != future.shape or past.shape[0] <= past.shape[1]:
            raise ValueError("CVA design blocks have an invalid shape")

        past_mean = np.mean(past, axis=0)
        future_mean = np.mean(future, axis=0)
        past_zero = past - past_mean
        future_zero = future - future_mean
        denominator = past.shape[0] - 1
        covariance_past = (past_zero.T @ past_zero) / denominator
        covariance_future = (future_zero.T @ future_zero) / denominator
        covariance_future_past = (future_zero.T @ past_zero) / denominator
        dimension = covariance_past.shape[0]
        ridge_past = (
            self.RIDGE_FRACTION * float(np.trace(covariance_past)) / dimension
        )
        ridge_future = (
            self.RIDGE_FRACTION * float(np.trace(covariance_future)) / dimension
        )
        if ridge_past <= 0.0 or ridge_future <= 0.0:
            raise ValueError("CVA ridge magnitude must be positive")
        regularized_past = covariance_past + ridge_past * np.eye(dimension)
        regularized_future = covariance_future + ridge_future * np.eye(dimension)
        whiten_past = _symmetric_inverse_sqrt(regularized_past)
        whiten_future = _symmetric_inverse_sqrt(regularized_future)
        cross = whiten_future @ covariance_future_past @ whiten_past
        _, singular_values, right_transpose = np.linalg.svd(
            cross, full_matrices=True
        )
        canonical_energy = singular_values**2
        total_energy = float(np.sum(canonical_energy))
        if not np.isfinite(total_energy) or total_energy <= 0.0:
            raise ValueError("CVA canonical correlations are degenerate")
        cumulative = np.cumsum(canonical_energy) / total_energy
        rank = int(np.searchsorted(cumulative, self.CANONICAL_ENERGY) + 1)
        if not 1 <= rank < dimension:
            raise ValueError(
                f"design-only CVA rank {rank} does not split state and residual"
            )
        right_vectors = right_transpose.T
        projection = whiten_past @ right_vectors

        self.raw_center_ = raw_center
        self.raw_scale_ = raw_scale
        self.raw_scale_fallback_ = raw_fallback
        self.past_mean_ = past_mean
        self.projection_ = projection
        self.rank_ = rank
        self.singular_values_ = singular_values
        self.canonical_energy_fraction_ = float(cumulative[rank - 1])
        self.ridge_past_ = float(ridge_past)
        self.ridge_future_ = float(ridge_future)
        self.design_pair_count_ = int(past.shape[0])
        self.design_run_names_ = tuple(name for name, _ in records)

        design_log_energies: list[np.ndarray] = []
        for _, matrix in records:
            state, residual = self._energy_components(matrix)
            design_log_energies.append(
                np.column_stack(
                    (
                        np.log1p(state[self.valid_start_ :]),
                        np.log1p(residual[self.valid_start_ :]),
                    )
                )
            )
        pooled_energy = np.concatenate(design_log_energies, axis=0)
        energy_center, energy_scale, energy_fallback = (
            strict_robust_location_scale(pooled_energy, axis=0)
        )
        self.energy_center_ = energy_center
        self.energy_scale_ = energy_scale
        self.energy_scale_fallback_ = energy_fallback
        self.design_online_score_count_ = int(pooled_energy.shape[0])
        self.design_fitted_ = True
        self.fitted_ = False
        return self

    def _check_design(self) -> None:
        if not self.design_fitted_:
            raise RuntimeError("CVA design parameters have not been fitted")

    def _energy_components(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "projection_"):
            raise RuntimeError("CVA projection has not been fitted")
        matrix = np.asarray(values, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.FEATURE_COUNT
            or matrix.shape[0] < self.PAST_ORDER + self.PERSISTENCE_WINDOW
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("CVA scoring requires a complete finite (time, 52) run")
        standardized = (matrix - self.raw_center_) / self.raw_scale_
        past = _past_blocks(standardized, self.PAST_ORDER)
        coordinates = (past - self.past_mean_) @ self.projection_
        previous = np.zeros(coordinates.shape[1], dtype=np.float64)
        state_values = np.empty(coordinates.shape[0], dtype=np.float64)
        residual_values = np.empty(coordinates.shape[0], dtype=np.float64)
        smoothing = self.EWMA_LAMBDA
        for index, coordinate in enumerate(coordinates):
            previous = smoothing * coordinate + (1.0 - smoothing) * previous
            state_values[index] = float(np.sum(previous[: self.rank_] ** 2))
            residual_values[index] = float(np.sum(previous[self.rank_ :] ** 2))
        length = matrix.shape[0]
        state = np.full(length, np.nan, dtype=np.float64)
        residual = np.full(length, np.nan, dtype=np.float64)
        state[self.valid_start_ :] = state_values
        residual[self.valid_start_ :] = residual_values
        return state, residual

    def path_from_energies(
        self,
        state_energy: np.ndarray,
        residual_energy: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> CVAMEWMAResult:
        """Build the fixed two-mechanism family path from precomputed energies."""
        self._check_design()
        state = np.asarray(state_energy, dtype=np.float64)
        residual = np.asarray(residual_energy, dtype=np.float64)
        if state.ndim != 1 or residual.shape != state.shape:
            raise ValueError("state and residual energies must be equal-length vectors")
        valid = slice(self.valid_start_, None)
        if (
            state.size < self.PAST_ORDER + self.PERSISTENCE_WINDOW
            or not np.isfinite(state[valid]).all()
            or not np.isfinite(residual[valid]).all()
            or np.any(state[valid] < -1e-12)
            or np.any(residual[valid] < -1e-12)
        ):
            raise ValueError("CVA energy vectors are invalid")
        component = np.full((state.size, 2), np.nan, dtype=np.float64)
        logged = np.column_stack(
            (
                np.log1p(np.maximum(state[valid], 0.0)),
                np.log1p(np.maximum(residual[valid], 0.0)),
            )
        )
        component[valid] = (logged - self.energy_center_) / self.energy_scale_
        family = family_persistence_path(
            component,
            valid_start=self.valid_start_,
            window=self.PERSISTENCE_WINDOW,
            threshold=threshold,
        )
        return CVAMEWMAResult(
            state_energy=state.copy(),
            residual_energy=residual.copy(),
            family=family,
        )

    def decision_path(
        self,
        values: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> CVAMEWMAResult:
        """Score one run causally using only its current and previous samples."""
        self._check_design()
        state, residual = self._energy_components(values)
        return self.path_from_energies(state, residual, threshold=threshold)

    def calibrate_envelopes(
        self,
        calibration_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "CausalCVAMEWMAMonitor":
        """Freeze q96 from 100 independent healthy run-family maxima."""
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
    ) -> "CausalCVAMEWMAMonitor":
        """CPU reference calibration from raw healthy runs."""
        records = validate_named_arrays(
            calibration_runs,
            label="CVA calibration",
            required_count=self.CALIBRATION_RUN_COUNT,
            minimum_length=self.PAST_ORDER + self.PERSISTENCE_WINDOW,
            feature_count=self.FEATURE_COUNT,
        )
        return self.calibrate_envelopes(
            (name, self.decision_path(matrix).persistence)
            for name, matrix in records
        )

    def predict(self, values: np.ndarray) -> CVAMEWMAResult:
        if not self.fitted_:
            raise RuntimeError("CVA monitor has not been calibrated")
        return self.decision_path(values, threshold=self.threshold_)

    @property
    def calibration_summary_(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("CVA monitor has not been calibrated")
        return {
            "method": "causal_regularized_CVA_signed_vector_MEWMA_family_3of3",
            "design_run_count": self.DESIGN_RUN_COUNT,
            "design_pair_count": self.design_pair_count_,
            "design_online_score_count": self.design_online_score_count_,
            "feature_count": self.FEATURE_COUNT,
            "past_order": self.PAST_ORDER,
            "future_order_design_only": self.FUTURE_ORDER,
            "online_uses_future": False,
            "past_block_order": "current_to_oldest",
            "ridge_fraction_trace_per_dimension": self.RIDGE_FRACTION,
            "ridge_past": self.ridge_past_,
            "ridge_future": self.ridge_future_,
            "rank_rule": "minimum squared canonical correlation energy >= 0.90",
            "rank": self.rank_,
            "canonical_energy_fraction": self.canonical_energy_fraction_,
            "singular_values": self.singular_values_.tolist(),
            "raw_scale_sd_fallback_count": int(np.sum(self.raw_scale_fallback_)),
            "energy_scale_sd_fallback": self.energy_scale_fallback_.tolist(),
            "energy_center": self.energy_center_.tolist(),
            "energy_scale": self.energy_scale_.tolist(),
            "ewma_lambda": self.EWMA_LAMBDA,
            "ewma_initial_state": "zero vector independently for each run",
            "components": ["state_T2", "residual_Q"],
            "family_rule": "3/3 persistence per component, then maximum",
            "valid_score_start_zero_based": self.valid_start_,
            "first_alarm_eligible_zero_based": (
                self.valid_start_ + self.PERSISTENCE_WINDOW - 1
            ),
            **self.conformal_summary_.as_dict(),
        }


__all__ = ["CVAMEWMAResult", "CausalCVAMEWMAMonitor"]
