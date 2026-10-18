"""Causal linear-Gaussian reconstruction with posterior uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import EPS, as_2d_float, ridge_solve, robust_location_scale, validate_complete


def _positive_semidefinite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


def _shrink_covariance(residual: np.ndarray, shrinkage: float) -> np.ndarray:
    if residual.shape[0] < 2:
        return np.eye(residual.shape[1])
    covariance = np.cov(residual, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    scale = max(float(np.trace(shrunk) / shrunk.shape[0]), 1.0)
    return _positive_semidefinite(shrunk, floor=EPS * scale)


def _solve_psd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve a positive-semidefinite system with minimal diagonal stabilization."""
    symmetric = 0.5 * (matrix + matrix.T)
    jitter = max(float(np.trace(symmetric) / max(symmetric.shape[0], 1)), 1.0) * 1e-10
    stabilized = symmetric + np.eye(symmetric.shape[0]) * jitter
    try:
        factor = np.linalg.cholesky(stabilized)
        return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(stabilized, rcond=1e-10) @ right_hand_side


@dataclass(frozen=True)
class FilterResult:
    """Posterior summaries in standardized coordinates plus raw-scale reconstruction."""

    mean_z: np.ndarray
    covariance_z: np.ndarray
    reconstructed: np.ndarray
    normalized_innovation: np.ndarray
    observed_mask: np.ndarray


class LinearGaussianFilter:
    """VAR(1) state model observed through a changing, exact sensor subset.

    The filter is causal: at time ``t`` it uses only observations up to ``t``.
    Missing dimensions are conditioned on the currently observed dimensions and
    uncertainty from previous gaps is propagated through the state transition.
    """

    def __init__(self, ridge: float = 1e-3, shrinkage: float = 0.1) -> None:
        if ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        self.ridge = float(ridge)
        self.shrinkage = float(shrinkage)
        self.fitted_ = False

    def fit(
        self,
        x: np.ndarray,
        exogenous: np.ndarray | None = None,
        observed_mask: np.ndarray | None = None,
        run_start_mask: np.ndarray | None = None,
    ) -> "LinearGaussianFilter":
        arr = validate_complete(x, "x")
        mask = (
            np.ones_like(arr, dtype=bool)
            if observed_mask is None
            else np.asarray(observed_mask, dtype=bool)
        )
        if mask.shape != arr.shape:
            raise ValueError("observed_mask must have the same shape as x")
        complete_rows = mask.all(axis=1)
        starts = (
            np.zeros(arr.shape[0], dtype=bool)
            if run_start_mask is None
            else np.asarray(run_start_mask, dtype=bool)
        )
        if starts.shape != (arr.shape[0],):
            raise ValueError("run_start_mask must have one value per row")
        starts = starts.copy()
        starts[0] = True
        if int(complete_rows.sum()) < 5:
            raise ValueError("at least five complete observations are required")
        self.location_, self.scale_ = robust_location_scale(arr[complete_rows])
        z = (arr - self.location_) / self.scale_
        if exogenous is None:
            u = np.empty((arr.shape[0], 0), dtype=float)
            self.exogenous_location_ = np.empty(0, dtype=float)
            self.exogenous_scale_ = np.empty(0, dtype=float)
        else:
            u_raw = as_2d_float(exogenous, "exogenous")
            if u_raw.shape[0] != arr.shape[0]:
                raise ValueError("exogenous and x must contain the same number of rows")
            if not np.isfinite(u_raw[complete_rows]).all():
                raise ValueError("exogenous values must be finite on complete training rows")
            self.exogenous_location_, self.exogenous_scale_ = robust_location_scale(
                u_raw[complete_rows]
            )
            u = (u_raw - self.exogenous_location_) / self.exogenous_scale_
        self.n_exogenous_ = u.shape[1]

        valid_transition = complete_rows[:-1] & complete_rows[1:] & ~starts[1:]
        if int(valid_transition.sum()) < 4:
            raise ValueError("at least four adjacent complete transitions are required")
        self.training_transition_count_ = int(valid_transition.sum())
        design = np.column_stack([np.ones(z.shape[0] - 1), z[:-1], u[1:]])[
            valid_transition
        ]
        target = z[1:][valid_transition]
        coefficients = ridge_solve(design, target, self.ridge)
        self.intercept_ = coefficients[0]
        self.transition_ = coefficients[1 : 1 + z.shape[1]].T
        self.exogenous_coefficients_ = coefficients[1 + z.shape[1] :].T

        # Prevent explosive uncertainty when a fitted finite-sample VAR is unstable.
        radius = float(np.max(np.abs(np.linalg.eigvals(self.transition_))))
        if np.isfinite(radius) and radius > 0.995:
            self.transition_ *= 0.995 / radius

        previous = z[:-1][valid_transition]
        current_u = u[1:][valid_transition]
        predictions = self.intercept_ + previous @ self.transition_.T
        if self.n_exogenous_:
            predictions += current_u @ self.exogenous_coefficients_.T
        innovations = target - predictions
        self.innovation_covariance_ = _shrink_covariance(innovations, self.shrinkage)
        fitted_rows = z[complete_rows]
        centered = fitted_rows - np.mean(fitted_rows, axis=0)
        self.initial_covariance_ = _shrink_covariance(centered, self.shrinkage)
        self.dimension_ = arr.shape[1]
        self.training_run_count_ = int(starts.sum())
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("LinearGaussianFilter has not been fitted")

    def standardize(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        arr = as_2d_float(x, "x")
        if arr.shape[1] != self.dimension_:
            raise ValueError("x has a different number of features from the fitted model")
        return (arr - self.location_) / self.scale_

    def inverse_standardize(self, z: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(z, dtype=float) * self.scale_ + self.location_

    def standardize_exogenous(
        self,
        exogenous: np.ndarray | None,
        n_rows: int,
    ) -> np.ndarray:
        self._check_fitted()
        if self.n_exogenous_ == 0:
            if exogenous is not None and np.asarray(exogenous).size:
                raise ValueError("the fitted model does not use exogenous variables")
            return np.empty((n_rows, 0), dtype=float)
        if exogenous is None:
            raise ValueError("exogenous variables are required by the fitted model")
        raw = as_2d_float(exogenous, "exogenous")
        if raw.shape != (n_rows, self.n_exogenous_):
            raise ValueError("exogenous variables have an incompatible shape")
        # At deployment an actuator readout can be missing together with sensors.
        # Carry it forward causally, with the Phase-I location as the initial fallback.
        completed = raw.copy()
        previous = self.exogenous_location_.copy()
        for t in range(n_rows):
            missing = ~np.isfinite(completed[t])
            completed[t, missing] = previous[missing]
            previous = completed[t]
        return (completed - self.exogenous_location_) / self.exogenous_scale_

    def filter(
        self,
        x: np.ndarray,
        observed_mask: np.ndarray | None = None,
        exogenous: np.ndarray | None = None,
    ) -> FilterResult:
        """Filter an incomplete sequence and return every posterior mean/covariance."""
        self._check_fitted()
        arr = as_2d_float(x, "x")
        if arr.shape[1] != self.dimension_:
            raise ValueError("x has a different number of features from the fitted model")
        finite = np.isfinite(arr)
        if observed_mask is None:
            mask = finite
        else:
            mask = np.asarray(observed_mask, dtype=bool)
            if mask.shape != arr.shape:
                raise ValueError("observed_mask must have the same shape as x")
            if np.any(mask & ~finite):
                raise ValueError("observed entries must be finite")

        z_observed = (np.where(mask, arr, self.location_) - self.location_) / self.scale_
        n, d = arr.shape
        u = self.standardize_exogenous(exogenous, n)
        means = np.zeros((n, d), dtype=float)
        covariances = np.zeros((n, d, d), dtype=float)
        nis = np.zeros(n, dtype=float)

        previous_mean = np.zeros(d, dtype=float)
        previous_covariance = self.initial_covariance_.copy()

        for t in range(n):
            if t == 0:
                prior_mean = self.intercept_.copy()
                if self.n_exogenous_:
                    prior_mean += self.exogenous_coefficients_ @ u[t]
                prior_covariance = self.initial_covariance_.copy()
            else:
                prior_mean = self.intercept_ + self.transition_ @ previous_mean
                if self.n_exogenous_:
                    prior_mean += self.exogenous_coefficients_ @ u[t]
                prior_covariance = (
                    self.innovation_covariance_
                    + self.transition_ @ previous_covariance @ self.transition_.T
                )
                prior_covariance = _positive_semidefinite(prior_covariance)

            observed = np.flatnonzero(mask[t])
            missing = np.flatnonzero(~mask[t])
            posterior_mean = prior_mean.copy()
            posterior_covariance = prior_covariance.copy()

            if observed.size:
                innovation = z_observed[t, observed] - prior_mean[observed]
                observed_covariance = prior_covariance[np.ix_(observed, observed)]
                solved_innovation = _solve_psd(observed_covariance, innovation)
                nis[t] = float(innovation @ solved_innovation) / observed.size
                posterior_mean[observed] = z_observed[t, observed]

                if missing.size:
                    cross = prior_covariance[np.ix_(missing, observed)]
                    gain = _solve_psd(observed_covariance, cross.T).T
                    posterior_mean[missing] = prior_mean[missing] + gain @ innovation
                    conditional = (
                        prior_covariance[np.ix_(missing, missing)]
                        - gain @ prior_covariance[np.ix_(observed, missing)]
                    )
                    conditional = _positive_semidefinite(conditional)
                    posterior_covariance = np.zeros((d, d), dtype=float)
                    posterior_covariance[np.ix_(missing, missing)] = conditional
                else:
                    posterior_covariance = np.zeros((d, d), dtype=float)
            # With no observations, the posterior is exactly the propagated prior.

            means[t] = posterior_mean
            covariances[t] = posterior_covariance
            previous_mean = posterior_mean
            previous_covariance = posterior_covariance

        reconstructed = self.inverse_standardize(means)
        reconstructed[mask] = arr[mask]
        return FilterResult(means, covariances, reconstructed, nis, mask)
