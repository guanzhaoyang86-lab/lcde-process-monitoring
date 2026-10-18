"""Posterior-informed PCA evidence and block-conformal calibration."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .missingness import gap_lengths, severity_labels
from .state_space import FilterResult, LinearGaussianFilter
from .utils import EPS, conformal_quantile, higher_quantile, ridge_solve, validate_complete


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Inverse-CDF weighted quantile with deterministic tie handling."""
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 1 or w.shape != y.shape or y.size == 0:
        raise ValueError("weighted quantile inputs must be aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("weighted quantile inputs must be finite with nonnegative weights")
    total = float(w.sum())
    if total <= 0.0 or not 0.0 <= quantile <= 1.0:
        raise ValueError("invalid weighted quantile")
    order = np.argsort(y, kind="mergesort")
    cumulative = np.cumsum(w[order])
    index = min(int(np.searchsorted(cumulative, quantile * total, side="left")), y.size - 1)
    return float(y[order[index]])


@dataclass(frozen=True)
class ScoreResult:
    # Historical field name retained for API compatibility.  ``fused`` is the
    # instantaneous (not EWMA-smoothed) standardized evidence.
    fused: np.ndarray
    t2: np.ndarray
    spe: np.ndarray
    t2_location: np.ndarray
    t2_uncertainty: np.ndarray
    spe_location: np.ndarray
    spe_uncertainty: np.ndarray
    innovation: np.ndarray
    context: np.ndarray
    posterior_trace: np.ndarray
    location_energy: np.ndarray
    uncertainty_energy: np.ndarray
    uncertainty_fraction: np.ndarray
    context_features: np.ndarray
    filter_result: FilterResult

    @property
    def instantaneous(self) -> np.ndarray:
        return self.fused


def _expected_quadratic(
    mean: np.ndarray,
    covariance: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        _quadratic_at_mean(mean, matrix)
        + _quadratic_uncertainty(covariance, matrix),
        0.0,
    )


def _quadratic_at_mean(mean: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.einsum("ni,ij,nj->n", mean, matrix, mean, optimize=True)
    return np.maximum(values, 0.0)


def _quadratic_uncertainty(covariance: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.einsum("ij,nji->n", matrix, covariance, optimize=True)
    return np.maximum(values, 0.0)


class PosteriorIntegratedMonitor:
    """Dynamic PCA monitor that separates anomaly location from uncertainty."""

    def __init__(
        self,
        variance_explained: float = 0.9,
        max_components: int | None = None,
        ridge: float = 1e-3,
        covariance_shrinkage: float = 0.1,
        component_reference_quantile: float = 0.95,
    ) -> None:
        if not 0.0 < variance_explained <= 1.0:
            raise ValueError("variance_explained must lie in (0, 1]")
        if max_components is not None and max_components < 1:
            raise ValueError("max_components must be positive")
        if not 0.5 <= component_reference_quantile < 1.0:
            raise ValueError("component_reference_quantile must lie in [0.5, 1)")
        self.variance_explained = float(variance_explained)
        self.max_components = max_components
        self.component_reference_quantile = float(component_reference_quantile)
        self.filter_model = LinearGaussianFilter(ridge, covariance_shrinkage)
        self.fitted_ = False

    def fit(
        self,
        x: np.ndarray,
        exogenous: np.ndarray | None = None,
        observed_mask: np.ndarray | None = None,
        integrate_uncertainty: bool = False,
        run_start_mask: np.ndarray | None = None,
    ) -> "PosteriorIntegratedMonitor":
        arr = validate_complete(x, "x")
        mask = (
            np.ones_like(arr, dtype=bool)
            if observed_mask is None
            else np.asarray(observed_mask, dtype=bool)
        )
        if mask.shape != arr.shape:
            raise ValueError("observed_mask must have the same shape as x")
        fit_rows = mask.all(axis=1)
        if int(fit_rows.sum()) < 2:
            raise ValueError("at least two fully observed Phase-I rows are required")
        self.filter_model.fit(arr, exogenous, mask, run_start_mask)
        z = self.filter_model.standardize(arr)
        u = self.filter_model.standardize_exogenous(exogenous, arr.shape[0])
        z_fit = z[fit_rows]
        u_fit = u[fit_rows]
        if u.shape[1]:
            design = np.column_stack([np.ones(z_fit.shape[0]), u_fit])
            coefficients = ridge_solve(design, z_fit, ridge=1e-3)
            self.baseline_intercept_ = coefficients[0]
            self.baseline_coefficients_ = coefficients[1:].T
            baseline = self.baseline_intercept_ + u @ self.baseline_coefficients_.T
        else:
            self.baseline_intercept_ = np.mean(z_fit, axis=0)
            self.baseline_coefficients_ = np.empty((z.shape[1], 0), dtype=float)
            baseline = np.broadcast_to(self.baseline_intercept_, z.shape)
        residual_z = z_fit - baseline[fit_rows]
        _, singular_values, vt = np.linalg.svd(residual_z, full_matrices=False)
        # The PCA reference is estimated only from fully observed Phase-I rows;
        # use that same effective sample size in the covariance eigenvalues.
        eigenvalues = singular_values**2 / max(z_fit.shape[0] - 1, 1)
        total = float(eigenvalues.sum())
        if total <= EPS:
            n_components = 1
        else:
            cumulative = np.cumsum(eigenvalues) / total
            n_components = int(np.searchsorted(cumulative, self.variance_explained) + 1)
        if self.max_components is not None:
            n_components = min(n_components, self.max_components)
        # Preserve at least one residual dimension when possible so SPE remains meaningful.
        if z.shape[1] > 1:
            n_components = min(n_components, z.shape[1] - 1)
        n_components = max(1, n_components)

        self.loadings_ = vt[:n_components].T
        retained = np.maximum(eigenvalues[:n_components], EPS)
        self.t2_matrix_ = (self.loadings_ / retained) @ self.loadings_.T
        self.spe_matrix_ = np.eye(z.shape[1]) - self.loadings_ @ self.loadings_.T
        self.n_components_ = n_components

        # Freeze component and global score scales on genuinely complete healthy
        # rows.  At these rows U=0, so the UD and M+U fits share exactly the same
        # quadratic statistic; the ablation changes only how missingness
        # uncertainty enters downstream scores.
        t2 = _quadratic_at_mean(residual_z, self.t2_matrix_)
        spe = _quadratic_at_mean(residual_z, self.spe_matrix_)
        burn = min(
            max(5, z.shape[1]),
            max(z_fit.shape[0] // 5, 1),
            z_fit.shape[0] - 1,
        )
        reference_slice = slice(burn, None)
        q = self.component_reference_quantile
        t2_reference = max(higher_quantile(t2[reference_slice], q), EPS)
        spe_reference = max(higher_quantile(spe[reference_slice], q), EPS)
        self.component_scales_ = np.array([t2_reference, spe_reference])
        self.combined_matrix_ = self.t2_matrix_ / t2_reference + self.spe_matrix_ / spe_reference
        combined = _quadratic_at_mean(residual_z, self.combined_matrix_)
        transformed = np.log1p(combined[reference_slice])
        self.score_center_ = float(np.median(transformed))
        score_mad = 1.4826 * float(np.median(np.abs(transformed - self.score_center_)))
        score_std = float(np.std(transformed, ddof=1)) if transformed.size > 1 else 1.0
        self.score_scale_ = score_mad if score_mad > EPS else max(score_std, EPS)
        self.integrate_uncertainty_ = bool(integrate_uncertainty)
        self.dimension_ = arr.shape[1]
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("PosteriorIntegratedMonitor has not been fitted")

    def score(
        self,
        x: np.ndarray,
        observed_mask: np.ndarray | None = None,
        exogenous: np.ndarray | None = None,
        integrate_uncertainty: bool | None = None,
        elapsed_start: int = 0,
    ) -> ScoreResult:
        self._check_fitted()
        if (
            isinstance(elapsed_start, (bool, np.bool_))
            or not isinstance(elapsed_start, (int, np.integer))
            or elapsed_start < 0
        ):
            raise ValueError("elapsed_start must be a nonnegative integer")
        elapsed_start = int(elapsed_start)
        use_uncertainty = (
            self.integrate_uncertainty_
            if integrate_uncertainty is None
            else bool(integrate_uncertainty)
        )
        if use_uncertainty != self.integrate_uncertainty_:
            raise ValueError(
                "the uncertainty score mode is fitted; fit a separate monitor "
                "instead of overriding it during scoring"
            )
        filtered = self.filter_model.filter(x, observed_mask, exogenous)
        u = self.filter_model.standardize_exogenous(exogenous, filtered.mean_z.shape[0])
        baseline = self.baseline_intercept_ + u @ self.baseline_coefficients_.T
        monitoring_mean = filtered.mean_z - baseline
        t2_location = _quadratic_at_mean(monitoring_mean, self.t2_matrix_)
        spe_location = _quadratic_at_mean(monitoring_mean, self.spe_matrix_)
        t2_uncertainty = _quadratic_uncertainty(
            filtered.covariance_z, self.t2_matrix_
        )
        spe_uncertainty = _quadratic_uncertainty(
            filtered.covariance_z, self.spe_matrix_
        )
        location_energy = _quadratic_at_mean(monitoring_mean, self.combined_matrix_)
        uncertainty_energy = _quadratic_uncertainty(
            filtered.covariance_z, self.combined_matrix_
        )
        if use_uncertainty:
            t2 = t2_location + t2_uncertainty
            spe = spe_location + spe_uncertainty
            combined = location_energy + uncertainty_energy
        else:
            t2 = t2_location
            spe = spe_location
            combined = location_energy
        innovation = filtered.normalized_innovation
        instantaneous = (np.log1p(combined) - self.score_center_) / self.score_scale_
        context = severity_labels(filtered.observed_mask)
        posterior_trace = np.trace(filtered.covariance_z, axis1=1, axis2=2) / self.dimension_
        missing_fraction = (~filtered.observed_mask).mean(axis=1)
        longest_gap = gap_lengths(filtered.observed_mask).max(axis=1)
        total_energy = location_energy + uncertainty_energy
        uncertainty_fraction = np.divide(
            uncertainty_energy,
            total_energy,
            out=np.zeros_like(uncertainty_energy),
            where=total_energy > EPS,
        )
        context_features = np.column_stack(
            [
                missing_fraction,
                np.log1p(longest_gap),
                np.log1p(posterior_trace),
                np.log1p(uncertainty_energy),
                np.log1p(
                    elapsed_start
                    + np.arange(filtered.mean_z.shape[0], dtype=float)
                ),
            ]
        )
        return ScoreResult(
            fused=instantaneous,
            t2=t2,
            spe=spe,
            t2_location=t2_location,
            t2_uncertainty=t2_uncertainty,
            spe_location=spe_location,
            spe_uncertainty=spe_uncertainty,
            innovation=innovation,
            context=context,
            posterior_trace=posterior_trace,
            location_energy=location_energy,
            uncertainty_energy=uncertainty_energy,
            uncertainty_fraction=uncertainty_fraction,
            context_features=context_features,
            filter_result=filtered,
        )


class BlockMondrianCalibrator:
    """Context-stratified split-conformal thresholds based on healthy block maxima.

    Under exchangeability of non-overlapping healthy block maxima within each
    missingness context, the conformal quantile controls the probability that a
    block contains at least one threshold exceedance.  Time-series applications
    should choose a block longer than the dominant autocorrelation horizon.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        block_size: int = 60,
        n_contexts: int = 4,
        min_blocks: int = 8,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if block_size < 1 or n_contexts < 1 or min_blocks < 1:
            raise ValueError("block_size, n_contexts, and min_blocks must be positive")
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.n_contexts = int(n_contexts)
        self.min_blocks = int(min_blocks)
        self.fitted_ = False

    def _block_maxima(self, scores: np.ndarray, contexts: np.ndarray) -> list[list[float]]:
        values: list[list[float]] = [[] for _ in range(self.n_contexts)]
        n = scores.size
        for start in range(0, n, self.block_size):
            stop = min(start + self.block_size, n)
            if stop - start < max(2, self.block_size // 2):
                continue
            block_scores = scores[start:stop]
            block_contexts = contexts[start:stop]
            for group in range(self.n_contexts):
                selected = block_scores[block_contexts == group]
                if selected.size:
                    values[group].append(float(np.max(selected)))
        return values

    def fit(
        self,
        score_context_sequences: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> "BlockMondrianCalibrator":
        grouped: list[list[float]] = [[] for _ in range(self.n_contexts)]
        pooled: list[float] = []
        sequence_count = 0
        for scores, contexts in score_context_sequences:
            score_array = np.asarray(scores, dtype=float)
            context_array = np.asarray(contexts, dtype=int)
            if score_array.ndim != 1 or context_array.shape != score_array.shape:
                raise ValueError("scores and contexts must be aligned one-dimensional arrays")
            if np.any((context_array < 0) | (context_array >= self.n_contexts)):
                raise ValueError("context index outside calibrated range")
            maxima = self._block_maxima(score_array, context_array)
            for group, values in enumerate(maxima):
                grouped[group].extend(values)
                pooled.extend(values)
            sequence_count += 1
        if sequence_count == 0 or not pooled:
            raise ValueError("no complete calibration blocks were available")

        pooled_threshold = conformal_quantile(np.asarray(pooled), self.alpha)
        thresholds = np.empty(self.n_contexts, dtype=float)
        counts = np.empty(self.n_contexts, dtype=int)
        for group, values in enumerate(grouped):
            counts[group] = len(values)
            if len(values) >= self.min_blocks:
                # Never make a sparse context less conservative than the pooled chart.
                group_threshold = conformal_quantile(np.asarray(values), self.alpha)
                thresholds[group] = max(group_threshold, pooled_threshold)
            else:
                thresholds[group] = pooled_threshold
        self.thresholds_ = thresholds
        self.block_counts_ = counts
        self.pooled_threshold_ = pooled_threshold
        self.fitted_ = True
        return self

    def thresholds_for(self, contexts: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("BlockMondrianCalibrator has not been fitted")
        labels = np.asarray(contexts, dtype=int)
        if np.any((labels < 0) | (labels >= self.n_contexts)):
            raise ValueError("context index outside calibrated range")
        return self.thresholds_[labels]


@dataclass
class _TreeNode:
    feature: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    leaf_id: int = -1


class RegimeStandardizer:
    """Honest shallow CART regimes over mask, gap, and posterior uncertainty.

    The tree is deliberately tiny and is fitted only on a dedicated healthy
    design split.  Within each frozen leaf, scores are standardized by robust
    location and scale before final alarm calibration.
    """

    def __init__(
        self,
        max_depth: int = 2,
        min_leaf_fraction: float = 0.05,
        candidate_quantiles: int = 9,
        support_feature_count: int = 4,
    ) -> None:
        if max_depth < 0 or not 0.0 < min_leaf_fraction < 0.5:
            raise ValueError("invalid tree complexity")
        if candidate_quantiles < 2:
            raise ValueError("candidate_quantiles must be at least two")
        if support_feature_count < 1:
            raise ValueError("support_feature_count must be positive")
        self.max_depth = int(max_depth)
        self.min_leaf_fraction = float(min_leaf_fraction)
        self.candidate_quantiles = int(candidate_quantiles)
        self.support_feature_count = int(support_feature_count)
        self.fitted_ = False

    @staticmethod
    def _weighted_loss(y: np.ndarray, weights: np.ndarray) -> float:
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0
        mean = float(np.sum(weights * y) / total)
        return float(np.sum(weights * (y - mean) ** 2))

    def fit(
        self,
        features: np.ndarray,
        scores: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RegimeStandardizer":
        x = np.asarray(features, dtype=float)
        y = np.asarray(scores, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
            raise ValueError("features and scores are not aligned")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("regime design data must be finite")
        weights = (
            np.ones(y.size)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if (
            weights.shape != y.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or weights.sum() <= 0.0
        ):
            raise ValueError("invalid sample weights")
        min_leaf_rows = 5
        min_leaf_weight = self.min_leaf_fraction * float(weights.sum())
        quantiles = np.linspace(0.1, 0.9, self.candidate_quantiles)

        def grow(indices: np.ndarray, depth: int) -> _TreeNode:
            node = _TreeNode()
            if (
                depth >= self.max_depth
                or indices.size < 2 * min_leaf_rows
                or float(weights[indices].sum()) < 2.0 * min_leaf_weight
            ):
                return node
            parent_loss = self._weighted_loss(y[indices], weights[indices])
            best_gain = 0.0
            best: tuple[int, float, np.ndarray, np.ndarray] | None = None
            for feature in range(x.shape[1]):
                candidates = np.unique(
                    [
                        _weighted_quantile(
                            x[indices, feature], weights[indices], float(quantile)
                        )
                        for quantile in quantiles
                    ]
                )
                for threshold in candidates:
                    left = indices[x[indices, feature] <= threshold]
                    right = indices[x[indices, feature] > threshold]
                    if (
                        left.size < min_leaf_rows
                        or right.size < min_leaf_rows
                        or float(weights[left].sum()) < min_leaf_weight
                        or float(weights[right].sum()) < min_leaf_weight
                    ):
                        continue
                    loss = self._weighted_loss(y[left], weights[left]) + self._weighted_loss(
                        y[right], weights[right]
                    )
                    gain = parent_loss - loss
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best = (feature, float(threshold), left, right)
            if best is not None:
                node.feature, node.threshold, left, right = best
                node.left = grow(left, depth + 1)
                node.right = grow(right, depth + 1)
            return node

        self.root_ = grow(np.arange(y.size), 0)
        next_leaf = 0

        def assign(node: _TreeNode) -> None:
            nonlocal next_leaf
            if node.feature < 0:
                node.leaf_id = next_leaf
                next_leaf += 1
                return
            assert node.left is not None and node.right is not None
            assign(node.left)
            assign(node.right)

        assign(self.root_)
        self.n_leaves_ = next_leaf
        leaves = self._predict_leaves(x)
        global_center = _weighted_quantile(y, weights, 0.5)
        global_mad = 1.4826 * _weighted_quantile(
            np.abs(y - global_center), weights, 0.5
        )
        global_mean = float(np.sum(weights * y) / weights.sum())
        global_std = float(
            np.sqrt(np.sum(weights * (y - global_mean) ** 2) / weights.sum())
        )
        global_scale = global_mad if global_mad > EPS else max(global_std, EPS)
        centers = np.empty(next_leaf, dtype=float)
        scales = np.empty(next_leaf, dtype=float)
        counts = np.empty(next_leaf, dtype=int)
        masses = np.empty(next_leaf, dtype=float)
        for leaf in range(next_leaf):
            selected_mask = leaves == leaf
            selected = y[selected_mask]
            selected_weights = weights[selected_mask]
            counts[leaf] = selected.size
            masses[leaf] = float(selected_weights.sum())
            if selected.size:
                center = _weighted_quantile(selected, selected_weights, 0.5)
                mad = 1.4826 * _weighted_quantile(
                    np.abs(selected - center), selected_weights, 0.5
                )
                mean = float(np.sum(selected_weights * selected) / selected_weights.sum())
                std = float(
                    np.sqrt(
                        np.sum(selected_weights * (selected - mean) ** 2)
                        / selected_weights.sum()
                    )
                )
                centers[leaf] = center
                scales[leaf] = mad if mad > EPS else max(std, global_scale, EPS)
            else:
                centers[leaf] = global_center
                scales[leaf] = global_scale
        self.centers_ = centers
        self.scales_ = scales
        self.leaf_counts_ = counts
        self.leaf_weight_mass_ = masses
        self.n_support_features_ = min(self.support_feature_count, x.shape[1])
        self.support_lower_ = np.min(x[:, : self.n_support_features_], axis=0)
        self.support_upper_ = np.max(x[:, : self.n_support_features_], axis=0)
        self.n_features_in_ = x.shape[1]
        self.fitted_ = True
        return self

    def _predict_leaves(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        leaves = np.empty(x.shape[0], dtype=int)
        for row_index, row in enumerate(x):
            node = self.root_
            while node.feature >= 0:
                child = node.left if row[node.feature] <= node.threshold else node.right
                assert child is not None
                node = child
            leaves[row_index] = node.leaf_id
        return leaves

    def transform(
        self,
        features: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("RegimeStandardizer has not been fitted")
        x = np.asarray(features, dtype=float)
        y = np.asarray(scores, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or y.shape != (x.shape[0],):
            raise ValueError("invalid regime transformation inputs")
        leaves = self._predict_leaves(x)
        standardized = (y - self.centers_[leaves]) / self.scales_[leaves]
        tolerance = 1e-12
        support_x = x[:, : self.n_support_features_]
        unsupported = np.any(
            (support_x < self.support_lower_ - tolerance)
            | (support_x > self.support_upper_ + tolerance),
            axis=1,
        )
        return standardized, leaves, unsupported


class BlockConformalAlarmCalibrator:
    """Clustered robust threshold for the final persisted alarm.

    Repeated masks of the same underlying time block are augmentations, not new
    independent calibration samples.  They are first collapsed by a maximum
    within each base block and mechanism; the final conformity score is the
    maximum over the predeclared mechanism family.  Thus the effective sample
    size is the number of non-overlapping base blocks.
    """

    def __init__(self, alpha: float = 0.05, block_size: int = 60, min_blocks: int = 8) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if block_size < 1 or min_blocks < 1:
            raise ValueError("block_size and min_blocks must be positive")
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.min_blocks = int(min_blocks)
        self.fitted_ = False

    def _block_maxima(self, statistic: np.ndarray) -> np.ndarray:
        values = np.asarray(statistic, dtype=float)
        if values.ndim != 1 or np.isnan(values).any():
            raise ValueError("persistence statistic must be one-dimensional without NaNs")
        n_blocks = values.size // self.block_size
        if n_blocks == 0:
            return np.empty(0, dtype=float)
        complete = values[: n_blocks * self.block_size]
        return np.max(complete.reshape(n_blocks, self.block_size), axis=1)

    def fit(
        self,
        named_persistence_sequences: Iterable[tuple[str, np.ndarray]],
    ) -> "BlockConformalAlarmCalibrator":
        records = [
            (str(name), np.asarray(statistic, dtype=float))
            for name, statistic in named_persistence_sequences
        ]
        if not records:
            raise ValueError("no calibration sequences were supplied")
        lengths = {values.size for _, values in records}
        if len(lengths) != 1:
            raise ValueError("mask augmentations must share the same base timeline")
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for name, statistic in records:
            grouped[name].append(self._block_maxima(statistic))
        if not grouped or not all(grouped.values()):
            raise ValueError("no complete calibration blocks were available")
        mechanism_envelopes = {
            name: np.max(np.vstack(replicates), axis=0)
            for name, replicates in grouped.items()
        }
        block_counts_set = {values.size for values in mechanism_envelopes.values()}
        if len(block_counts_set) != 1 or not block_counts_set:
            raise ValueError("calibration mechanisms do not share aligned base blocks")
        n_blocks = next(iter(block_counts_set))
        finite_sample_minimum = int(np.ceil(1.0 / self.alpha)) - 1
        required = max(self.min_blocks, finite_sample_minimum)
        if n_blocks < required:
            raise ValueError(
                f"only {n_blocks} complete base blocks are available; "
                f"need at least {required} for alpha={self.alpha:g}"
            )

        mechanism_thresholds: dict[str, float] = {}
        block_counts: dict[str, int] = {}
        for name, values in mechanism_envelopes.items():
            block_counts[name] = values.size
            mechanism_thresholds[name] = conformal_quantile(values, self.alpha)
        family_envelope = np.max(np.vstack(list(mechanism_envelopes.values())), axis=0)
        family_threshold = conformal_quantile(family_envelope, self.alpha)
        self.mechanism_thresholds_ = mechanism_thresholds
        self.block_counts_ = block_counts
        self.augmentation_counts_ = {name: len(values) for name, values in grouped.items()}
        self.base_block_count_ = n_blocks
        self.base_run_count_ = None
        self.calibration_unit_ = "time_block"
        self.pooled_threshold_ = family_threshold  # backward-compatible summary key
        self.threshold_ = family_threshold
        self.fitted_ = True
        return self

    def fit_run_clusters(
        self,
        named_run_persistence_sequences: Iterable[
            tuple[str, str, np.ndarray]
        ],
    ) -> "BlockConformalAlarmCalibrator":
        """Calibrate on one worst-case score per independent healthy run.

        For each run and mechanism, repeated masks and all time points are first
        collapsed by a maximum.  Mechanisms are then collapsed within the run.
        The split-conformal sample size is therefore the number of healthy runs,
        never the number of time blocks or artificial masks.
        """
        records = [
            (str(name), str(run_id), np.asarray(statistic, dtype=float))
            for name, run_id, statistic in named_run_persistence_sequences
        ]
        if not records:
            raise ValueError("no run-level calibration sequences were supplied")
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for name, run_id, statistic in records:
            if statistic.ndim != 1 or np.isnan(statistic).any():
                raise ValueError(
                    "persistence statistic must be one-dimensional without NaNs"
                )
            if statistic.size == 0:
                raise ValueError("run-level calibration sequences cannot be empty")
            grouped[(run_id, name)].append(float(np.max(statistic)))

        run_ids = tuple(sorted({run_id for run_id, _ in grouped}))
        mechanisms = tuple(sorted({name for _, name in grouped}))
        for run_id in run_ids:
            available = {name for candidate, name in grouped if candidate == run_id}
            if available != set(mechanisms):
                raise ValueError(
                    f"calibration run {run_id} does not contain every mechanism"
                )
        replicate_counts: dict[str, set[int]] = defaultdict(set)
        run_mechanism_envelopes: dict[tuple[str, str], float] = {}
        for (run_id, name), values in grouped.items():
            replicate_counts[name].add(len(values))
            run_mechanism_envelopes[(run_id, name)] = max(values)
        if any(len(counts) != 1 for counts in replicate_counts.values()):
            raise ValueError("mask repeat counts differ between calibration runs")

        n_runs = len(run_ids)
        finite_sample_minimum = int(np.ceil(1.0 / self.alpha)) - 1
        required = max(self.min_blocks, finite_sample_minimum)
        if n_runs < required:
            raise ValueError(
                f"only {n_runs} independent calibration runs are available; "
                f"need at least {required} for alpha={self.alpha:g}"
            )
        mechanism_envelopes = {
            name: np.asarray(
                [
                    run_mechanism_envelopes[(run_id, name)]
                    for run_id in run_ids
                ],
                dtype=float,
            )
            for name in mechanisms
        }
        family_envelope = np.asarray(
            [
                max(
                    run_mechanism_envelopes[(run_id, name)]
                    for name in mechanisms
                )
                for run_id in run_ids
            ],
            dtype=float,
        )
        self.mechanism_thresholds_ = {
            name: conformal_quantile(values, self.alpha)
            for name, values in mechanism_envelopes.items()
        }
        self.block_counts_ = {name: n_runs for name in mechanisms}
        self.augmentation_counts_ = {
            name: next(iter(replicate_counts[name])) for name in mechanisms
        }
        self.base_block_count_ = n_runs
        self.base_run_count_ = n_runs
        self.calibration_unit_ = "healthy_run_maximum"
        self.pooled_threshold_ = conformal_quantile(family_envelope, self.alpha)
        self.threshold_ = self.pooled_threshold_
        self.fitted_ = True
        return self
