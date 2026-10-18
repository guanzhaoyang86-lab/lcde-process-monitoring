from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mate_spc_directional.features import (
    DirectionalFeatureTransformer,
    _pair_window_features,
    load_directional_feature_transformer,
)


def _base_transformer(seed: int = 1234) -> DirectionalFeatureTransformer:
    rng = np.random.default_rng(seed)
    return DirectionalFeatureTransformer(
        raw_center=np.zeros(52),
        raw_scale=np.ones(52),
        past_mean=np.zeros(260),
        residual_projection=rng.normal(scale=0.08, size=(260, 204)),
        cva_rank=56,
    )


def _ar_run(seed: int, *, length: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(length, 52))
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    for index in range(1, length):
        values[index] = 0.35 * values[index - 1] + innovations[index]
    return values


class DirectionalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = tuple(_ar_run(8000 + index) for index in range(4))
        cls.transformer = _base_transformer().fit(cls.design)

    def test_signed_residual_ewma_matches_hand_recursion(self) -> None:
        projection = np.zeros((260, 204))
        projection[0, 0] = 1.0
        transformer = DirectionalFeatureTransformer(
            raw_center=np.zeros(52),
            raw_scale=np.ones(52),
            past_mean=np.zeros(260),
            residual_projection=projection,
            cva_rank=56,
        )
        run = np.zeros((9, 52))
        run[:, 0] = np.arange(1.0, 10.0)

        actual = transformer.signed_residual_ewma(run)
        expected = []
        previous = 0.0
        for coordinate in run[4:, 0]:
            previous = 0.2 * coordinate + 0.8 * previous
            expected.append(previous)

        self.assertTrue(np.isnan(actual[:4]).all())
        np.testing.assert_allclose(actual[4:, 0], expected, rtol=0.0, atol=1e-14)
        np.testing.assert_array_equal(actual[4:, 1:], np.zeros((5, 203)))

    def test_positive_control_lag_means_controller_leads_response(self) -> None:
        rng = np.random.default_rng(4421)
        controller = np.cumsum(rng.normal(size=80))
        standardized = np.zeros((80, 52), dtype=float)
        standardized[:, 0] = controller
        standardized[2:, 1] = controller[:-2]

        features = _pair_window_features(
            standardized,
            controller_index=0,
            response_index=1,
            window=20,
        )

        np.testing.assert_allclose(features[:, 2], 1.0, rtol=0.0, atol=1e-13)
        self.assertLess(float(np.max(features[:, 1])), 0.99)

    def test_raw_count_valid_start_and_feature_name_order_are_frozen(self) -> None:
        raw = self.transformer._raw_features(self.design[0])
        transformed = self.transformer.transform(self.design[0])

        self.assertEqual(raw.shape, (100, 1080))
        self.assertEqual(self.transformer.valid_start_, 63)
        self.assertTrue(np.isnan(raw[:63]).all())
        self.assertTrue(np.isfinite(raw[63:]).all())
        self.assertEqual(
            transformed.shape,
            (100, self.transformer.output_feature_count_),
        )
        self.assertTrue(np.isnan(transformed[:63]).all())
        self.assertTrue(np.isfinite(transformed[63:]).all())

        names = self.transformer.RAW_FEATURE_NAMES
        self.assertEqual(len(names), 1080)
        self.assertEqual(len(set(names)), 1080)
        self.assertEqual(names[0], "residual_ewma_r000_rolling_sum_w01")
        self.assertEqual(names[203], "residual_ewma_r203_rolling_sum_w01")
        self.assertEqual(names[204], "residual_ewma_r000_rolling_sum_w05")
        self.assertEqual(names[815], "residual_ewma_r203_rolling_sum_w60")
        self.assertEqual(
            names[816],
            "control_u41_y01_w20_diff_corr_u_leads_y_lag0",
        )
        self.assertEqual(
            names[-1],
            "control_u51_y21_w60_diff_autocorr_maxabs_y",
        )
        expected_retained = tuple(
            name
            for name, retained in zip(names, self.transformer.keep_mask_)
            if retained
        )
        self.assertEqual(self.transformer.feature_names_, expected_retained)

    def test_future_changes_do_not_change_a_transformed_prefix(self) -> None:
        original = self.design[0].copy()
        changed = original.copy()
        changed[80:] += np.linspace(5.0, 25.0, 52)

        original_features = self.transformer.transform(original)
        changed_features = self.transformer.transform(changed)

        np.testing.assert_array_equal(
            original_features[:80],
            changed_features[:80],
        )
        self.assertGreater(
            float(np.max(np.abs(original_features[80:] - changed_features[80:]))),
            0.0,
        )

    def test_each_run_resets_state_and_names_cannot_change_values(self) -> None:
        first = self.design[1]
        second = self.design[2]
        individual_first = self.transformer.transform(first)
        individual_second = self.transformer.transform(second)
        batch_first, batch_second = self.transformer.transform_runs((first, second))

        np.testing.assert_array_equal(batch_first, individual_first)
        np.testing.assert_array_equal(batch_second, individual_second)
        repeated_a, repeated_b = self.transformer.transform_runs((second, second))
        np.testing.assert_array_equal(repeated_a, repeated_b)

        named = self.transformer.transform_named_runs(
            (("renamed-a", second), ("renamed-b", second))
        )
        self.assertEqual(tuple(name for name, _ in named), ("renamed-a", "renamed-b"))
        np.testing.assert_array_equal(named[0][1], named[1][1])

        residual = self.transformer.signed_residual_ewma(second)
        standardized = second / self.transformer.raw_scale
        first_past = np.concatenate(
            [standardized[4 - lag] for lag in range(5)],
            axis=0,
        )
        first_coordinate = (
            first_past - self.transformer.past_mean
        ) @ self.transformer.residual_projection
        np.testing.assert_allclose(
            residual[4],
            0.2 * first_coordinate,
            rtol=0.0,
            atol=1e-13,
        )

    def test_safe_npz_round_trip_and_bad_keys_are_rejected(self) -> None:
        metadata = {"phase": "feature-test", "design_runs": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "directional-features.npz"
            self.transformer.save_npz(path, metadata=metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(
                    any(np.asarray(archive[key]).dtype.kind == "O" for key in archive.files)
                )
            restored, restored_metadata = load_directional_feature_transformer(path)
            class_loaded = DirectionalFeatureTransformer.load_npz(path)

            bad_path = Path(directory) / "bad-keys.npz"
            np.savez(bad_path, schema=np.asarray("incomplete"))
            with self.assertRaisesRegex(ValueError, "archive keys"):
                load_directional_feature_transformer(bad_path)

        self.assertEqual(restored_metadata, metadata)
        self.assertTrue(restored.fitted_)
        self.assertEqual(restored.feature_names_, self.transformer.feature_names_)
        self.assertEqual(
            restored.dropped_feature_names_,
            self.transformer.dropped_feature_names_,
        )
        for field in (
            "raw_center",
            "raw_scale",
            "past_mean",
            "residual_projection",
            "residual_center_",
            "residual_scale_",
            "residual_scale_fallback_",
            "feature_center_",
            "feature_scale_",
            "feature_scale_fallback_",
            "keep_mask_",
        ):
            np.testing.assert_array_equal(
                getattr(restored, field),
                getattr(self.transformer, field),
            )
            self.assertFalse(getattr(restored, field).flags.writeable)
        expected = self.transformer.transform(self.design[0])
        np.testing.assert_array_equal(restored.transform(self.design[0]), expected)
        np.testing.assert_array_equal(class_loaded.transform(self.design[0]), expected)


if __name__ == "__main__":
    unittest.main()
