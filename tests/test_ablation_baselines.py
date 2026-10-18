from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from mate_spc import MAPICConfig
from mate_spc.baselines import CalibratedPointMonitor
from mate_spc.experiment import run_stress_test
from mate_spc.missingness import MissingnessConfig, apply_mask, generate_observed_mask
from mate_spc.synthetic import make_demo_data


class BaselinePhaseIMaskTests(unittest.TestCase):
    def test_masked_release_values_do_not_change_fitted_baselines(self) -> None:
        train, calibration, test = make_demo_data(140, 240, 90, dimension=3)
        train_mask = np.ones_like(train.x, dtype=bool)
        calibration_mask = np.ones_like(calibration.x, dtype=bool)
        train_mask[[30, 31]] = False
        calibration_mask[[70, 150]] = False

        changed_train = train.x.copy()
        changed_calibration = calibration.x.copy()
        changed_train[~train_mask] += 1e6
        changed_calibration[~calibration_mask] -= 1e6
        config = MAPICConfig(
            alpha=0.25,
            block_size=12,
            min_calibration_blocks=3,
            calibration_repeats=1,
            scenarios=(MissingnessConfig("mnar", 0.15),),
            seed=17,
        )
        test_mask = generate_observed_mask(
            test.x, MissingnessConfig("multi_blackout", 0.15, 2, 6, 0.67), seed=5
        )
        incomplete_test = apply_mask(test.x, test_mask)

        for method in sorted(CalibratedPointMonitor.METHODS):
            with self.subTest(method=method):
                first = CalibratedPointMonitor(method, config).fit(
                    train.x,
                    calibration.x,
                    train.covariates,
                    calibration.covariates,
                    train_mask,
                    calibration_mask,
                )
                second = CalibratedPointMonitor(method, config).fit(
                    changed_train,
                    changed_calibration,
                    train.covariates,
                    calibration.covariates,
                    train_mask,
                    calibration_mask,
                )
                self.assertAlmostEqual(first.calibrator.threshold_, second.calibrator.threshold_)
                np.testing.assert_allclose(first.regime.centers_, second.regime.centers_)
                np.testing.assert_allclose(first.regime.scales_, second.regime.scales_)
                first_result = first.predict(incomplete_test, test_mask, test.covariates)
                second_result = second.predict(incomplete_test, test_mask, test.covariates)
                np.testing.assert_allclose(first_result.score, second_result.score)
                np.testing.assert_allclose(
                    first_result.reconstructed, second_result.reconstructed
                )
                np.testing.assert_array_equal(first_result.alarm, second_result.alarm)


class IndependentAblationTests(unittest.TestCase):
    def test_integrated_ablation_has_its_own_fit_and_threshold(self) -> None:
        train, calibration, test = make_demo_data(140, 240, 100, dimension=3)
        natural_mask = np.ones_like(test.x, dtype=bool)
        natural_mask[[20, 55]] = False
        test = replace(test, observed_mask=natural_mask)
        changed_x = test.x.copy()
        changed_x[~natural_mask] += 1e9
        changed_test = replace(test, x=changed_x)
        scenario = MissingnessConfig("mnar", 0.2, mnar_strength=1.5)
        config = MAPICConfig(
            alpha=0.25,
            block_size=12,
            min_calibration_blocks=3,
            calibration_repeats=1,
            scenarios=(scenario,),
            seed=9,
        )
        report = run_stress_test(
            train,
            calibration,
            test,
            config,
            scenarios=(scenario,),
            seeds=(4,),
            baseline_methods=(),
        )
        changed_report = run_stress_test(
            train,
            calibration,
            changed_test,
            config,
            scenarios=(scenario,),
            seeds=(4,),
            baseline_methods=(),
        )

        full_threshold = report["calibration_summary"]["threshold"]
        ablation_summary = report["ablation_calibration_summaries"][
            "MAPIC_integrated_expectation"
        ]
        self.assertNotAlmostEqual(full_threshold, ablation_summary["threshold"])
        self.assertEqual(
            report["calibration_summary"]["score_mode"],
            "uncertainty_decoupled_location_M",
        )
        self.assertEqual(
            ablation_summary["score_mode"], "posterior_expectation_M_plus_U"
        )
        note = report["records"][0]["methods"]["MAPIC_integrated_expectation"]["note"]
        self.assertIn("independently fitted", note)
        self.assertNotIn("frozen control limit", note)
        def normalize_nan(value: object) -> object:
            if isinstance(value, dict):
                return {key: normalize_nan(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalize_nan(item) for item in value]
            if isinstance(value, float) and np.isnan(value):
                return None
            return value

        self.assertEqual(
            normalize_nan(report["records"]), normalize_nan(changed_report["records"])
        )
        self.assertAlmostEqual(
            report["records"][0]["natural_missing_rate"],
            float((~natural_mask).mean()),
        )


if __name__ == "__main__":
    unittest.main()
