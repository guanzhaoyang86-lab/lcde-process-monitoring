from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mate_spc_directional.glr import (
    FAULT_IDS,
    DirectionalGLRModel,
    conformal_run_threshold,
    load_directional_glr_model,
    page_cusum_numpy,
    save_directional_glr_model,
    score_directional_glr_numpy,
)


def _identity_model(*, threshold: float | None = None) -> DirectionalGLRModel:
    return DirectionalGLRModel(
        fault_ids=FAULT_IDS,
        weights=np.eye(3),
        bias=np.zeros(3),
        normal_logit_center=np.zeros(3),
        normal_logit_scale=np.ones(3),
        kappa=np.zeros(3),
        threshold=threshold,
    )


class DirectionalGLRTests(unittest.TestCase):
    def test_page_cusum_matches_hand_recursion(self) -> None:
        evidence = np.array(
            [
                [1.0, -1.0, 0.5],
                [1.0, 2.0, -1.0],
                [-4.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
            ]
        )
        actual = page_cusum_numpy(evidence, np.full(3, 0.5))
        expected = np.array(
            [
                [0.5, 0.0, 0.0],
                [1.0, 1.5, 0.0],
                [0.0, 2.0, 0.5],
                [2.5, 2.5, 1.0],
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_three_of_three_persistence_and_strict_threshold_are_exact(self) -> None:
        model = _identity_model(threshold=3.0)
        features = np.tile(np.array([3.0, -3.0, -3.0]), (4, 1))

        path = score_directional_glr_numpy(model, features, valid_start=0)

        np.testing.assert_array_equal(
            path.eligible,
            np.array([False, False, True, True]),
        )
        self.assertTrue(np.isneginf(path.envelope[:2]).all())
        np.testing.assert_allclose(path.envelope[2:], np.array([3.0, 6.0]))
        np.testing.assert_array_equal(
            path.alarm,
            np.array([False, False, False, True]),
        )
        np.testing.assert_allclose(path.component_persistence[2], [3.0, 0.0, 0.0])

    def test_evidence_cannot_alternate_heads_to_satisfy_persistence(self) -> None:
        model = _identity_model(threshold=1.0)
        features = np.array(
            [
                [2.0, -8.0, -8.0],
                [-8.0, 2.0, -8.0],
                [-8.0, -8.0, 2.0],
            ]
        )

        path = score_directional_glr_numpy(model, features, valid_start=0)

        np.testing.assert_array_equal(path.alarm, np.zeros(3, dtype=bool))
        np.testing.assert_allclose(path.component_persistence[2], np.zeros(3))
        self.assertEqual(path.envelope[2], 0.0)

    def test_q96_for_one_hundred_run_maxima(self) -> None:
        envelopes = {
            f"cal-{index:03d}": np.array([-np.inf, float(index)])
            for index in range(1, 101)
        }

        threshold, metadata = conformal_run_threshold(
            envelopes,
            alpha=0.05,
            required_count=100,
        )

        self.assertEqual(threshold, 96.0)
        self.assertEqual(metadata["rank_one_based"], 96)
        self.assertEqual(metadata["rank_driver_run"], "cal-096")
        self.assertEqual(metadata["comparison"], "strictly_greater_than")

    def test_safe_npz_round_trip_preserves_model_and_metadata(self) -> None:
        original = DirectionalGLRModel(
            fault_ids=FAULT_IDS,
            weights=np.arange(15, dtype=float).reshape(3, 5) / 7.0,
            bias=np.array([-0.5, 0.0, 0.5]),
            normal_logit_center=np.array([1.0, 2.0, 3.0]),
            normal_logit_scale=np.array([0.5, 1.5, 2.5]),
            kappa=np.array([0.1, 0.25, 0.5]),
            threshold=12.5,
        )
        metadata = {"phase": "synthetic-test", "folds": [0, 1, 2, 3, 4]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "directional-model.npz"
            save_directional_glr_model(original, path, metadata=metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(
                    any(np.asarray(archive[key]).dtype.kind == "O" for key in archive.files)
                )
            restored, restored_metadata = load_directional_glr_model(path)

        self.assertEqual(restored.fault_ids, original.fault_ids)
        self.assertEqual(restored.threshold, original.threshold)
        self.assertEqual(restored_metadata, metadata)
        for field in (
            "weights",
            "bias",
            "normal_logit_center",
            "normal_logit_scale",
            "kappa",
        ):
            np.testing.assert_array_equal(getattr(restored, field), getattr(original, field))
            self.assertFalse(getattr(restored, field).flags.writeable)


if __name__ == "__main__":
    unittest.main()
