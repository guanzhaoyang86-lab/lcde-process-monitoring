from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mate_spc_directional.features import DirectionalFeatureTransformer
from mate_spc_hybrid.features import (
    ENERGY_FEATURE_NAMES,
    ENERGY_RAW_FEATURE_COUNT,
    HybridFeatureTransformer,
    load_hybrid_feature_transformer,
)


def _base_transformer(seed: int = 9102) -> DirectionalFeatureTransformer:
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


class HybridFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = tuple(_ar_run(9200 + index) for index in range(4))
        cls.directional = _base_transformer().fit(cls.design)
        cls.hybrid = HybridFeatureTransformer(cls.directional).fit(cls.design)

    def test_fixed_count_names_alignment_and_hand_calculation(self) -> None:
        run = self.design[0]
        raw = self.hybrid._raw_energy_features(run)

        self.assertEqual(ENERGY_RAW_FEATURE_COUNT, 768)
        self.assertEqual(len(ENERGY_FEATURE_NAMES), 768)
        self.assertEqual(len(set(ENERGY_FEATURE_NAMES)), 768)
        self.assertEqual(
            ENERGY_FEATURE_NAMES[0],
            "residual_ewma_z_r000_rolling_mean_square_w05",
        )
        self.assertEqual(
            ENERGY_FEATURE_NAMES[611],
            "residual_ewma_z_r203_rolling_mean_square_w60",
        )
        self.assertEqual(
            ENERGY_FEATURE_NAMES[612],
            "raw_z_x00_diff_rolling_mean_square_w05",
        )
        self.assertEqual(
            ENERGY_FEATURE_NAMES[-1],
            "raw_z_x51_diff_rolling_mean_square_w60",
        )
        self.assertEqual(raw.shape, (100, 768))
        self.assertEqual(self.hybrid.valid_start_, 63)
        self.assertTrue(np.isnan(raw[:63]).all())
        self.assertTrue(np.isfinite(raw[63:]).all())

        residual = self.directional.signed_residual_ewma(run)[4:]
        residual_z = (
            residual - self.directional.residual_center_
        ) / self.directional.residual_scale_
        np.testing.assert_allclose(
            raw[63, 0],
            np.mean(np.square(residual_z[55:60, 0])),
            rtol=0.0,
            atol=1e-13,
        )
        differences = np.diff(run, axis=0)
        raw_w20_x02 = 612 + 52 + 2
        np.testing.assert_allclose(
            raw[63, raw_w20_x02],
            np.mean(np.square(differences[43:63, 2])),
            rtol=0.0,
            atol=1e-13,
        )

    def test_preserves_directional_features_is_causal_and_resets_each_run(self) -> None:
        original = self.design[1].copy()
        changed = original.copy()
        changed[80:] += np.linspace(7.0, 19.0, 52)
        directional = self.directional.transform(original)
        output = self.hybrid.transform(original)
        changed_output = self.hybrid.transform(changed)

        self.assertEqual(
            output.shape,
            (100, self.hybrid.output_feature_count_),
        )
        np.testing.assert_array_equal(
            output[:, : self.hybrid.directional_feature_count_],
            directional,
        )
        self.assertEqual(
            self.hybrid.feature_names_[: self.hybrid.directional_feature_count_],
            self.directional.feature_names_,
        )
        np.testing.assert_array_equal(output[:80], changed_output[:80])
        self.assertGreater(float(np.max(np.abs(output[80:] - changed_output[80:]))), 0.0)
        self.assertLessEqual(float(np.max(np.abs(output[63:]))), 10.0)

        first, second = self.hybrid.transform_runs((self.design[1], self.design[2]))
        np.testing.assert_array_equal(first, self.hybrid.transform(self.design[1]))
        np.testing.assert_array_equal(second, self.hybrid.transform(self.design[2]))
        repeated_a, repeated_b = self.hybrid.transform_runs(
            (self.design[2], self.design[2])
        )
        np.testing.assert_array_equal(repeated_a, repeated_b)

    def test_degenerate_energy_coordinates_are_dropped(self) -> None:
        design = []
        for seed in range(4):
            run = _ar_run(9500 + seed)
            run[:, 0] = 0.0
            design.append(run)
        directional = _base_transformer(9500).fit(design)
        hybrid = HybridFeatureTransformer(directional).fit(design)

        for window in (5, 20, 60):
            self.assertIn(
                f"raw_z_x00_diff_rolling_mean_square_w{window:02d}",
                hybrid.dropped_energy_feature_names_,
            )
        self.assertEqual(
            hybrid.energy_feature_names_,
            tuple(
                name
                for name, retained in zip(ENERGY_FEATURE_NAMES, hybrid.energy_keep_mask_)
                if retained
            ),
        )

    def test_safe_self_contained_npz_round_trip_and_bad_keys_are_rejected(self) -> None:
        metadata = {"phase": "hybrid-test", "reference_runs": 4}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid-features.npz"
            self.hybrid.save_npz(path, metadata=metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(
                    any(np.asarray(archive[key]).dtype.kind == "O" for key in archive.files)
                )
            restored, restored_metadata = load_hybrid_feature_transformer(path)
            class_loaded = HybridFeatureTransformer.load_npz(path)

            bad_path = Path(directory) / "bad-keys.npz"
            np.savez(bad_path, schema=np.asarray("incomplete"))
            with self.assertRaisesRegex(ValueError, "archive keys"):
                load_hybrid_feature_transformer(bad_path)

        self.assertEqual(restored_metadata, metadata)
        self.assertEqual(restored.feature_names_, self.hybrid.feature_names_)
        self.assertEqual(
            restored.dropped_energy_feature_names_,
            self.hybrid.dropped_energy_feature_names_,
        )
        expected = self.hybrid.transform(self.design[3])
        np.testing.assert_array_equal(restored.transform(self.design[3]), expected)
        np.testing.assert_array_equal(class_loaded.transform(self.design[3]), expected)
        for field in (
            "energy_center_",
            "energy_scale_",
            "energy_scale_fallback_",
            "energy_keep_mask_",
        ):
            self.assertFalse(getattr(restored, field).flags.writeable)

    def test_unfitted_directional_transformer_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already be fitted"):
            HybridFeatureTransformer(_base_transformer(9999))


if __name__ == "__main__":
    unittest.main()
