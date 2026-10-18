from __future__ import annotations

import gc
from dataclasses import replace
import unittest
import weakref

import numpy as np

from mate_spc import MAPIC, MAPICConfig
from mate_spc.missingness import (
    MissingnessConfig,
    apply_mask,
    generate_observed_mask,
    severity_labels,
)
from mate_spc.monitor import (
    BlockConformalAlarmCalibrator,
    RegimeStandardizer,
    ScoreResult,
    _expected_quadratic,
    _quadratic_at_mean,
    _quadratic_uncertainty,
)
from mate_spc.state_space import FilterResult, LinearGaussianFilter
from mate_spc.synthetic import make_demo_data, simulate_process
from mate_spc.utils import (
    conformal_quantile,
    ewma_with_resets,
    rolling_kth_largest,
    segmented_rolling_kth_largest,
)


class UtilityTests(unittest.TestCase):
    def test_ud_mapic_is_the_default_score_mode(self) -> None:
        self.assertFalse(MAPICConfig().integrate_uncertainty)
        self.assertTrue(MAPICConfig().use_posterior_quality_context)

    def test_exact_small_sample_conformal_boundary(self) -> None:
        values = np.arange(9, dtype=float)
        self.assertEqual(conformal_quantile(values, 0.1), 8.0)
        self.assertTrue(np.isinf(conformal_quantile(values[:-1], 0.1)))

    def test_rolling_kth_largest_matches_k_of_w_logic(self) -> None:
        values = np.array([0.0, 3.0, 2.0, 4.0, 0.0])
        statistic = rolling_kth_largest(values, k=2, window=3)
        expected = np.array([-np.inf, 0.0, 2.0, 3.0, 2.0])
        np.testing.assert_allclose(statistic, expected)

        unavailable = rolling_kth_largest(
            np.array([3.0, -np.inf, 4.0]), k=2, window=3
        )
        np.testing.assert_allclose(unavailable, np.array([-np.inf, -np.inf, 3.0]))

    def test_unknown_resets_ewma_and_persistence_window(self) -> None:
        values = np.array([10.0, 0.0, 999.0, 2.0, 4.0, 6.0])
        reset = np.array([False, False, True, False, False, False])
        ewma = ewma_with_resets(values, reset, smoothing=0.5)
        np.testing.assert_allclose(ewma, np.array([10.0, 5.0, -np.inf, 2.0, 3.0, 4.5]))
        persistence = segmented_rolling_kth_largest(ewma, reset, k=2, window=3)
        np.testing.assert_allclose(
            persistence,
            np.array([-np.inf, 5.0, -np.inf, -np.inf, 2.0, 3.0]),
        )

    def test_calibration_repeats_are_clustered_by_base_block(self) -> None:
        calibrator = BlockConformalAlarmCalibrator(alpha=0.2, block_size=10, min_blocks=4)
        first = np.arange(40, dtype=float)
        second = first + 100.0
        calibrator.fit([("same", first), ("same", second), ("clean", first)])
        self.assertEqual(calibrator.block_counts_["same"], 4)
        self.assertEqual(calibrator.augmentation_counts_["same"], 2)
        self.assertEqual(calibrator.base_block_count_, 4)

    def test_calibrator_rejects_insufficient_independent_blocks(self) -> None:
        calibrator = BlockConformalAlarmCalibrator(alpha=0.05, block_size=10)
        with self.assertRaisesRegex(ValueError, "need at least 19"):
            calibrator.fit([("clean", np.arange(100, dtype=float))])

    def test_weighted_regime_is_invariant_to_weight_splitting(self) -> None:
        rng = np.random.default_rng(2)
        features = rng.normal(size=(120, 4))
        scores = features[:, 0] ** 2 + 0.2 * features[:, 1]
        first = RegimeStandardizer(max_depth=2).fit(features, scores)
        duplicate_features = np.vstack([features, features[:60]])
        duplicate_scores = np.r_[scores, scores[:60]]
        duplicate_weights = np.r_[np.full(60, 0.5), np.ones(60), np.full(60, 0.5)]
        second = RegimeStandardizer(max_depth=2).fit(
            duplicate_features, duplicate_scores, duplicate_weights
        )
        transformed_first, leaves_first, _ = first.transform(features, scores)
        transformed_second, leaves_second, _ = second.transform(features, scores)
        np.testing.assert_array_equal(leaves_first, leaves_second)
        np.testing.assert_allclose(transformed_first, transformed_second, atol=1e-12)


class MissingnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = simulate_process(500, dimension=5, seed=1).x

    def test_masks_are_reproducible_and_non_mutating(self) -> None:
        config = MissingnessConfig("multi_blackout", 0.2, 3, 15, 0.4)
        first = generate_observed_mask(self.x, config, seed=9)
        second = generate_observed_mask(self.x, config, seed=9)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(first[0].all())
        incomplete = apply_mask(self.x, first)
        self.assertTrue(np.isnan(incomplete[~first]).all())
        self.assertTrue(np.isfinite(self.x).all())

    def test_mcar_rate_and_contexts(self) -> None:
        mask = generate_observed_mask(self.x, MissingnessConfig("mcar", 0.2), seed=4)
        self.assertLess(abs((~mask).mean() - 0.2), 0.03)
        labels = severity_labels(mask)
        self.assertEqual(labels.shape, (self.x.shape[0],))
        self.assertTrue(np.all((labels >= 0) & (labels <= 3)))


class PosteriorTests(unittest.TestCase):
    def test_filter_is_causal_preserves_observations_and_is_psd(self) -> None:
        train = simulate_process(300, dimension=4, seed=11)
        test = simulate_process(140, dimension=4, seed=12)
        model = LinearGaussianFilter().fit(train.x, train.covariates)
        mask = generate_observed_mask(
            test.x, MissingnessConfig("blackout", 0.15, 3, 12), seed=7
        )
        incomplete = apply_mask(test.x, mask)
        first = model.filter(incomplete, mask, test.covariates)
        changed = incomplete.copy()
        future = changed[80:].copy()
        future[mask[80:]] += 1000.0
        changed[80:] = future
        second = model.filter(changed, mask, test.covariates)
        np.testing.assert_allclose(first.mean_z[:80], second.mean_z[:80], atol=1e-12)
        np.testing.assert_allclose(first.reconstructed[mask], test.x[mask], atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(first.covariance_z)
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-9)

    def test_masked_phase1_release_values_do_not_affect_fit(self) -> None:
        data = simulate_process(180, dimension=4, seed=21)
        mask = np.ones_like(data.x, dtype=bool)
        mask[70] = False
        changed = data.x.copy()
        changed[70] += 1e6
        first = LinearGaussianFilter().fit(data.x, data.covariates, mask)
        second = LinearGaussianFilter().fit(changed, data.covariates, mask)
        np.testing.assert_allclose(first.location_, second.location_)
        np.testing.assert_allclose(first.transition_, second.transition_)
        np.testing.assert_allclose(first.innovation_covariance_, second.innovation_covariance_)

    def test_quadratic_expectation_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(5)
        mean = rng.normal(size=(1, 3))
        factor = rng.normal(size=(3, 3))
        covariance = factor @ factor.T * 0.2
        a_factor = rng.normal(size=(3, 3))
        matrix = a_factor @ a_factor.T
        analytic = _expected_quadratic(mean, covariance[None, :, :], matrix)[0]
        location = _quadratic_at_mean(mean, matrix)[0]
        uncertainty = _quadratic_uncertainty(covariance[None, :, :], matrix)[0]
        self.assertAlmostEqual(analytic, location + uncertainty)
        draws = rng.multivariate_normal(mean[0], covariance, size=50_000)
        empirical = np.mean(np.einsum("ni,ij,nj->n", draws, matrix, draws))
        self.assertLess(abs(analytic - empirical) / analytic, 0.02)


class EndToEndTests(unittest.TestCase):
    def test_fit_releases_full_posterior_tensors_between_augmentations(self) -> None:
        class TrackingMonitor:
            def __init__(self) -> None:
                self.covariances: list[weakref.ReferenceType[np.ndarray]] = []
                self.live_before_score: list[int] = []

            def fit(self, *args: object, **kwargs: object) -> "TrackingMonitor":
                return self

            def score(
                self,
                x: np.ndarray,
                observed_mask: np.ndarray | None = None,
                exogenous: np.ndarray | None = None,
                integrate_uncertainty: bool = True,
                elapsed_start: int = 0,
            ) -> ScoreResult:
                del exogenous, integrate_uncertainty, elapsed_start
                gc.collect()
                self.live_before_score.append(sum(ref() is not None for ref in self.covariances))
                values = np.asarray(x, dtype=float)
                n_rows, n_features = values.shape
                mask = (
                    np.isfinite(values)
                    if observed_mask is None
                    else np.asarray(observed_mask, dtype=bool)
                )
                covariance = np.zeros((n_rows, n_features, n_features), dtype=float)
                self.covariances.append(weakref.ref(covariance))
                reconstructed = np.where(mask, values, 0.0)
                filter_result = FilterResult(
                    mean_z=reconstructed,
                    covariance_z=covariance,
                    reconstructed=reconstructed,
                    normalized_innovation=np.zeros(n_rows),
                    observed_mask=mask,
                )
                time = np.arange(n_rows, dtype=float)
                fused = 1.0 + time / max(n_rows, 1)
                context_features = np.column_stack(
                    [
                        (~mask).mean(axis=1),
                        np.zeros(n_rows),
                        np.zeros(n_rows),
                        np.zeros(n_rows),
                        np.log1p(time),
                    ]
                )
                zeros = np.zeros(n_rows)
                return ScoreResult(
                    fused=fused,
                    t2=zeros,
                    spe=zeros,
                    t2_location=zeros,
                    t2_uncertainty=zeros,
                    spe_location=zeros,
                    spe_uncertainty=zeros,
                    innovation=zeros,
                    context=np.zeros(n_rows, dtype=int),
                    posterior_trace=zeros,
                    location_energy=zeros,
                    uncertainty_energy=zeros,
                    uncertainty_fraction=zeros,
                    context_features=context_features,
                    filter_result=filter_result,
                )

        rng = np.random.default_rng(123)
        train = rng.normal(size=(48, 6))
        calibration = rng.normal(size=(48, 6))
        config = MAPICConfig(
            alpha=0.5,
            block_size=8,
            min_calibration_blocks=2,
            calibration_repeats=2,
            scenarios=(MissingnessConfig("mcar", 0.1),),
        )
        model = MAPIC(config)
        tracker = TrackingMonitor()
        model.monitor = tracker
        model.fit(train, calibration)
        self.assertEqual(max(tracker.live_before_score), 0)

    def test_mapic_end_to_end_and_complete_data_equivalence(self) -> None:
        train, calibration, test = make_demo_data(220, 480, 180, dimension=4)
        config = MAPICConfig(
            alpha=0.2,
            block_size=12,
            min_calibration_blocks=4,
            calibration_repeats=1,
            persistence_k=2,
            persistence_window=3,
            seed=44,
        )
        model = MAPIC(config).fit(
            train.x, calibration.x, train.covariates, calibration.covariates
        )
        integrated_model = MAPIC(replace(config, integrate_uncertainty=True)).fit(
            train.x, calibration.x, train.covariates, calibration.covariates
        )
        complete_a = integrated_model.monitor.score(
            test.x, exogenous=test.covariates
        )
        complete_b = model.monitor.score(test.x, exogenous=test.covariates)
        np.testing.assert_allclose(complete_a.fused, complete_b.fused, atol=1e-10)
        np.testing.assert_allclose(
            integrated_model.monitor.combined_matrix_,
            model.monitor.combined_matrix_,
            atol=1e-12,
        )

        mask = generate_observed_mask(
            test.x, MissingnessConfig("multi_blackout", 0.15, 2, 8, 0.5), seed=3
        )
        result = model.predict(apply_mask(test.x, mask), mask, test.covariates)
        self.assertEqual(result.score.shape, (test.x.shape[0],))
        self.assertTrue(np.isfinite(result.reconstructed).all())
        self.assertTrue(np.isfinite(result.score).all())
        self.assertTrue(np.all(result.uncertainty_energy >= 0.0))
        self.assertTrue(
            np.all(
                (result.uncertainty_fraction >= 0.0)
                & (result.uncertainty_fraction <= 1.0)
            )
        )
        self.assertTrue(set(np.unique(result.status)).issubset({0, 1, 2}))

        mask_with_outage = mask.copy()
        mask_with_outage[50] = False
        incomplete_with_outage = apply_mask(test.x, mask_with_outage)
        outage_result = model.predict(
            incomplete_with_outage, mask_with_outage, test.covariates
        )
        self.assertTrue(outage_result.unknown[50])
        self.assertEqual(outage_result.status[50], 2)
        self.assertFalse(outage_result.alarm[50])
        self.assertTrue(np.isneginf(outage_result.ewma_score[50]))
        next_supported = np.flatnonzero(
            (~outage_result.unknown) & (np.arange(outage_result.unknown.size) > 50)
        )[0]
        self.assertAlmostEqual(
            outage_result.ewma_score[next_supported],
            outage_result.standardized_score[next_supported],
        )

        with self.assertRaisesRegex(ValueError, "fit a separate MAPIC model"):
            model.predict(
                apply_mask(test.x, mask),
                mask,
                test.covariates,
                integrate_uncertainty=True,
            )
        with self.assertRaisesRegex(ValueError, "fit a separate monitor"):
            model.monitor.score(
                apply_mask(test.x, mask),
                mask,
                test.covariates,
                integrate_uncertainty=True,
            )

        duplicate = MAPIC(config).fit(
            train.x, calibration.x, train.covariates, calibration.covariates
        )
        duplicate_result = duplicate.predict(apply_mask(test.x, mask), mask, test.covariates)
        np.testing.assert_allclose(result.score, duplicate_result.score)
        np.testing.assert_array_equal(result.status, duplicate_result.status)

        full = model.predict(test.x, covariates=test.covariates)
        prefix_length = 97
        prefix = model.predict(
            test.x[:prefix_length], covariates=test.covariates[:prefix_length]
        )
        np.testing.assert_allclose(full.score[:prefix_length], prefix.score)
        np.testing.assert_allclose(
            full.standardized_score[:prefix_length], prefix.standardized_score
        )
        np.testing.assert_allclose(full.ewma_score[:prefix_length], prefix.ewma_score)
        np.testing.assert_allclose(
            full.persistence_score[:prefix_length], prefix.persistence_score
        )
        np.testing.assert_array_equal(full.status[:prefix_length], prefix.status)
        np.testing.assert_array_equal(full.context[:prefix_length], prefix.context)

    def test_phase1_masked_release_values_do_not_change_full_mapic(self) -> None:
        train, calibration, test = make_demo_data(220, 480, 100, dimension=4)
        train_mask = np.ones_like(train.x, dtype=bool)
        calibration_mask = np.ones_like(calibration.x, dtype=bool)
        train_mask[60] = False
        calibration_mask[[80, 310]] = False
        changed_train = train.x.copy()
        changed_calibration = calibration.x.copy()
        changed_train[~train_mask] += 1e6
        changed_calibration[~calibration_mask] -= 1e6
        config = MAPICConfig(
            alpha=0.2,
            block_size=12,
            min_calibration_blocks=4,
            calibration_repeats=1,
            seed=9,
        )
        first = MAPIC(config).fit(
            train.x,
            calibration.x,
            train.covariates,
            calibration.covariates,
            train_mask,
            calibration_mask,
        )
        second = MAPIC(config).fit(
            changed_train,
            changed_calibration,
            train.covariates,
            calibration.covariates,
            train_mask,
            calibration_mask,
        )
        self.assertAlmostEqual(first.calibrator.threshold_, second.calibrator.threshold_)
        np.testing.assert_allclose(
            first.predict(test.x, covariates=test.covariates).score,
            second.predict(test.x, covariates=test.covariates).score,
        )


if __name__ == "__main__":
    unittest.main()
