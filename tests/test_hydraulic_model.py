from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:  # Torch must load before NumPy on Windows to avoid an OpenMP DLL collision.
    import torch as _torch
except (ImportError, OSError):
    _torch = None

import numpy as np

from mate_spc_hydraulic.model import (
    CV_FOLD_COUNT,
    CV_FOLD_SALT,
    FEATURE_COUNT,
    HEAD_COUNT,
    HEALTHY_LABEL,
    HYDRAULIC_SCORER_SPEC_SHA256,
    PHASE_COUNT,
    POSITIVE_LABELS,
    RIDGE_LAMBDA_GRID,
    ROBUST_SCALE_FACTOR,
    SCALE_FLOOR,
    HydraulicDirectionalModel,
    balanced_validation_log_loss_numpy,
    calibrate_hydraulic_pac,
    cell_balanced_cycle_weights,
    compare_hydraulic_paths,
    fit_healthy_phase_normalization,
    fit_hydraulic_directional_cuda,
    fit_hydraulic_grouped_cv_cuda,
    hydraulic_directional_model_sha256,
    load_hydraulic_model,
    predict_hydraulic_logits_numpy,
    protocol_grouped_fold_ids,
    save_hydraulic_model,
    score_hydraulic_cuda,
    score_hydraulic_numpy,
    validate_grouped_folds,
    validate_residual_features,
)


def _identity_model(
    *,
    threshold: float | None = None,
    kappa: float = 0.75,
) -> HydraulicDirectionalModel:
    weights = np.zeros((HEAD_COUNT, FEATURE_COUNT), dtype=np.float64)
    weights[:, :HEAD_COUNT] = np.eye(HEAD_COUNT)
    return HydraulicDirectionalModel(
        positive_labels=POSITIVE_LABELS,
        weights=weights,
        bias=np.zeros(HEAD_COUNT),
        phase_logit_center=np.zeros((PHASE_COUNT, HEAD_COUNT)),
        phase_logit_scale=np.ones((PHASE_COUNT, HEAD_COUNT)),
        kappa=np.full(HEAD_COUNT, kappa),
        threshold=threshold,
    )


def _complete_group_data(group_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    groups: list[str] = []
    for group in range(group_count):
        for label in (100, 90, 80, 73):
            labels.append(label)
            groups.append(f"C{group:03d}-P0-A100")
    return np.asarray(labels, dtype=np.int64), np.asarray(groups)


def _cuda_available() -> bool:
    return bool(_torch is not None and _torch.cuda.is_available())


class HydraulicInputAndWeightTests(unittest.TestCase):
    def test_residual_shape_and_finiteness_are_strict(self) -> None:
        valid = np.zeros((2, PHASE_COUNT, FEATURE_COUNT), dtype=np.float32)
        checked = validate_residual_features(valid)
        self.assertEqual(checked.shape, (2, PHASE_COUNT, FEATURE_COUNT))
        self.assertEqual(checked.dtype, np.float64)
        self.assertFalse(np.shares_memory(valid, checked))

        with self.assertRaisesRegex(ValueError, "shape"):
            validate_residual_features(np.zeros((2, PHASE_COUNT - 1, FEATURE_COUNT)))
        invalid = np.zeros((1, PHASE_COUNT, FEATURE_COUNT))
        invalid[0, 5, 7] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_residual_features(invalid)

    def test_each_observed_nuisance_class_cell_has_equal_total_mass(self) -> None:
        labels = np.asarray([100, 100, 90, 100, 90, 90, 80, 73])
        groups = np.asarray(["g0", "g0", "g0", "g1", "g1", "g1", "g1", "g1"])

        weights = cell_balanced_cycle_weights(
            labels,
            groups,
            positive_label=90,
        )

        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertEqual(weights[0], 0.125)
        self.assertEqual(weights[1], 0.125)
        self.assertEqual(weights[2], 0.25)
        self.assertEqual(weights[3], 0.25)
        self.assertEqual(weights[4], 0.125)
        self.assertEqual(weights[5], 0.125)
        np.testing.assert_array_equal(weights[6:], np.zeros(2))

    def test_labels_and_groups_reject_ambiguous_coercions(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            cell_balanced_cycle_weights(
                [100.0, 90.0],
                ["g0", "g0"],
                positive_label=90,
            )
        with self.assertRaisesRegex(ValueError, "string or integer"):
            cell_balanced_cycle_weights(
                [100, 90],
                [0.5, 0.5],
                positive_label=90,
            )
        with self.assertRaisesRegex(ValueError, "positive_label"):
            cell_balanced_cycle_weights(
                [100, 90],
                ["g0", "g0"],
                positive_label=100,
            )


class HydraulicNormalizationTests(unittest.TestCase):
    def test_phase_median_uses_only_healthy_cycles_and_global_head_floor(self) -> None:
        logits = np.empty((3, PHASE_COUNT, HEAD_COUNT), dtype=np.float64)
        phase = np.arange(PHASE_COUNT, dtype=np.float64)[:, None]
        logits[0] = phase
        logits[1] = phase + 2.0
        logits[2] = 1.0e6

        center, scale, floored = fit_healthy_phase_normalization(
            logits,
            [HEALTHY_LABEL, HEALTHY_LABEL, 90],
        )

        expected_center = np.broadcast_to(phase + 1.0, center.shape)
        healthy = logits[:2]
        global_center = np.median(healthy, axis=(0, 1))
        global_robust = ROBUST_SCALE_FACTOR * np.median(
            np.abs(healthy - global_center[None, None, :]),
            axis=(0, 1),
        )
        expected_floor = np.maximum(SCALE_FLOOR, 0.1 * global_robust)
        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(
            scale,
            np.broadcast_to(expected_floor[None, :], scale.shape),
        )
        self.assertTrue(floored.all())

    def test_collapsed_global_and_phase_mad_use_absolute_floor(self) -> None:
        logits = np.zeros((2, PHASE_COUNT, HEAD_COUNT))
        _, scale, floored = fit_healthy_phase_normalization(logits, [100, 100])

        np.testing.assert_array_equal(
            scale,
            np.full((PHASE_COUNT, HEAD_COUNT), SCALE_FLOOR),
        )
        self.assertTrue(floored.all())


class HydraulicScoringTests(unittest.TestCase):
    def test_logits_have_fixed_cycle_phase_head_shape(self) -> None:
        model = _identity_model()
        features = np.zeros((2, PHASE_COUNT, FEATURE_COUNT))
        features[:, :, :HEAD_COUNT] = 2.0

        logits = predict_hydraulic_logits_numpy(model, features)

        self.assertEqual(logits.shape, (2, PHASE_COUNT, HEAD_COUNT))
        np.testing.assert_array_equal(logits, np.full(logits.shape, 2.0))

    def test_page_resets_per_cycle_and_three_of_three_is_strict(self) -> None:
        model = _identity_model(threshold=1.0)
        features = np.zeros((2, PHASE_COUNT, FEATURE_COUNT))
        features[0, :, 0] = 1.75
        features[1, :, 0] = -8.0

        path = score_hydraulic_numpy(model, features)

        np.testing.assert_array_equal(path.eligible[:, :2], np.zeros((2, 2), dtype=bool))
        self.assertTrue(np.isneginf(path.envelope[:, :2]).all())
        self.assertEqual(path.cusum[0, 0, 0], 1.0)
        self.assertEqual(path.component_persistence[0, 2, 0], 1.0)
        self.assertFalse(path.alarm[0, 2])
        self.assertTrue(path.alarm[0, 3])
        np.testing.assert_array_equal(path.cusum[1, :, 0], np.zeros(PHASE_COUNT))
        self.assertFalse(path.alarm[1].any())

    def test_standardized_evidence_is_clipped_at_eight(self) -> None:
        model = _identity_model(kappa=0.0)
        features = np.zeros((1, PHASE_COUNT, FEATURE_COUNT))
        features[0, 0, 0] = 1000.0
        features[0, 1, 1] = -1000.0

        path = score_hydraulic_numpy(model, features)

        self.assertEqual(path.clipped_evidence[0, 0, 0], 8.0)
        self.assertEqual(path.clipped_evidence[0, 1, 1], -8.0)

    def test_cell_balanced_zero_logit_validation_loss_is_log_two(self) -> None:
        labels, groups = _complete_group_data()
        model = HydraulicDirectionalModel(
            positive_labels=POSITIVE_LABELS,
            weights=np.zeros((HEAD_COUNT, FEATURE_COUNT)),
            bias=np.zeros(HEAD_COUNT),
            phase_logit_center=np.zeros((PHASE_COUNT, HEAD_COUNT)),
            phase_logit_scale=np.ones((PHASE_COUNT, HEAD_COUNT)),
            kappa=np.full(HEAD_COUNT, 0.75),
        )
        features = np.zeros((labels.size, PHASE_COUNT, FEATURE_COUNT))

        loss = balanced_validation_log_loss_numpy(model, features, labels, groups)

        self.assertAlmostEqual(loss, float(np.log(2.0)), places=15)

    def test_pac_calibration_uses_complete_cycle_maxima(self) -> None:
        model = _identity_model(kappa=0.75)
        healthy = np.zeros((90, PHASE_COUNT, FEATURE_COUNT))

        calibrated, result = calibrate_hydraulic_pac(model, healthy)

        self.assertEqual(result.rank.sample_count, 90)
        self.assertEqual(result.sample_count, 90)
        self.assertEqual(result.epsilon, 0.05)
        self.assertEqual(result.delta, 0.05)
        self.assertEqual(result.threshold, 0.0)
        self.assertEqual(calibrated.threshold, 0.0)
        self.assertEqual(result.comparison, "strictly_greater_than")
        self.assertEqual(result.scorer_spec_sha256, HYDRAULIC_SCORER_SPEC_SHA256)
        self.assertEqual(
            result.model_sha256,
            hydraulic_directional_model_sha256(model.with_threshold(None)),
        )
        self.assertEqual(len(result.calibration_sha256), 64)
        certificate = result.as_dict()
        self.assertEqual(certificate["rank_one_based"], result.rank.rank_one_based)
        self.assertEqual(certificate["achieved_confidence"], result.achieved_confidence)
        self.assertEqual(len(certificate["run_maxima"]), 90)
        self.assertEqual(certificate["scorer_spec"]["cycle_horizon_seconds"], 60)

    def test_pac_calibration_rejects_nonprotocol_kappa(self) -> None:
        model = _identity_model(kappa=0.5)
        healthy = np.zeros((90, PHASE_COUNT, FEATURE_COUNT))

        with self.assertRaisesRegex(ValueError, "protocol kappa"):
            calibrate_hydraulic_pac(model, healthy)


class HydraulicGroupedCVTests(unittest.TestCase):
    def test_frozen_hash_order_is_deterministic_and_group_intact(self) -> None:
        labels, groups = _complete_group_data(group_count=6)
        folds = protocol_grouped_fold_ids(groups)
        second = protocol_grouped_fold_ids(groups[::-1])[::-1]

        np.testing.assert_array_equal(folds, second)
        self.assertEqual(set(folds.tolist()), set(range(CV_FOLD_COUNT)))
        for group in set(groups.tolist()):
            self.assertEqual(np.unique(folds[groups == group]).size, 1)
        checked = validate_grouped_folds(labels, groups, folds)
        np.testing.assert_array_equal(checked, folds)

        ordering = sorted(
            set(groups.tolist()),
            key=lambda group: (
                hashlib.sha256(f"{CV_FOLD_SALT}|{group}".encode()).hexdigest(),
                group,
            ),
        )
        for index, group in enumerate(ordering):
            self.assertEqual(int(folds[np.flatnonzero(groups == group)[0]]), index % 3)

    def test_group_split_is_rejected(self) -> None:
        labels, groups = _complete_group_data(group_count=3)
        folds = np.repeat(np.arange(3), 4)
        folds[0] = 1

        with self.assertRaisesRegex(ValueError, "split across folds"):
            validate_grouped_folds(labels, groups, folds)

    def test_exact_cv_ties_choose_larger_lambda_without_cuda(self) -> None:
        labels, groups = _complete_group_data(group_count=3)
        features = np.zeros((labels.size, PHASE_COUNT, FEATURE_COUNT))
        folds = protocol_grouped_fold_ids(groups)
        fixed_model = _identity_model()

        with mock.patch(
            "mate_spc_hydraulic.model.fit_hydraulic_directional_cuda",
            return_value=(fixed_model, {"mock": True}),
        ) as fit_mock, mock.patch(
            "mate_spc_hydraulic.model.balanced_validation_log_loss_numpy",
            return_value=0.5,
        ):
            model, result = fit_hydraulic_grouped_cv_cuda(
                features,
                labels,
                groups,
                fold_by_cycle=folds,
            )

        self.assertIs(model, fixed_model)
        self.assertEqual(result.selected_lambda, max(RIDGE_LAMBDA_GRID))
        self.assertEqual(len(result.folds), len(RIDGE_LAMBDA_GRID) * CV_FOLD_COUNT)
        self.assertEqual(fit_mock.call_count, len(RIDGE_LAMBDA_GRID) * 3 + 1)

    def test_cv_rejects_a_valid_but_nonprotocol_group_assignment(self) -> None:
        labels, groups = _complete_group_data(group_count=3)
        features = np.zeros((labels.size, PHASE_COUNT, FEATURE_COUNT))
        expected = protocol_grouped_fold_ids(groups)
        nonprotocol = expected.copy()
        nonprotocol[expected == 0] = 1
        nonprotocol[expected == 1] = 0

        with self.assertRaisesRegex(ValueError, "frozen SHA256"):
            fit_hydraulic_grouped_cv_cuda(
                features,
                labels,
                groups,
                fold_by_cycle=nonprotocol,
            )


class HydraulicSerializationTests(unittest.TestCase):
    @staticmethod
    def _valid_archive_payload(directory: str) -> dict[str, np.ndarray]:
        path = Path(directory) / "valid-source.npz"
        save_hydraulic_model(_identity_model(threshold=12.5), path, metadata={"ok": 1})
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}

    def test_safe_npz_round_trip_preserves_arrays_and_metadata(self) -> None:
        original = _identity_model(threshold=12.5)
        metadata = {"protocol": "hydraulic_padr_pac_v1", "folds": [0, 1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hydraulic-model.npz"
            save_hydraulic_model(original, path, metadata=metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(
                    any(np.asarray(archive[key]).dtype.kind == "O" for key in archive.files)
                )
            restored, restored_metadata = load_hydraulic_model(path)

        self.assertEqual(restored.positive_labels, original.positive_labels)
        self.assertEqual(restored.threshold, original.threshold)
        self.assertEqual(restored_metadata, metadata)
        for name in (
            "weights",
            "bias",
            "phase_logit_center",
            "phase_logit_scale",
            "kappa",
        ):
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
            self.assertFalse(getattr(restored, name).flags.writeable)

    def test_integer_identity_fields_cannot_be_float_coerced_or_reshaped(self) -> None:
        corruptions = (
            ("schema_version", np.asarray([1.9]), "schema_version"),
            ("schema_version", np.asarray([[1]], dtype=np.int64), "schema_version"),
            ("positive_labels", np.asarray([90.9, 80.0, 73.0]), "positive_labels"),
            (
                "positive_labels",
                np.asarray([[90, 80, 73]], dtype=np.int64),
                "positive_labels",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            original = self._valid_archive_payload(directory)
            for case, (field, value, pattern) in enumerate(corruptions):
                with self.subTest(field=field, case=case):
                    payload = {key: array.copy() for key, array in original.items()}
                    payload[field] = value
                    path = Path(directory) / f"bad-identity-{case}.npz"
                    np.savez(path, **payload)
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_hydraulic_model(path)

    def test_every_real_numeric_field_rejects_text_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = self._valid_archive_payload(directory)
            for field in (
                "weights",
                "bias",
                "phase_logit_center",
                "phase_logit_scale",
                "kappa",
                "threshold",
            ):
                with self.subTest(field=field):
                    payload = {key: array.copy() for key, array in original.items()}
                    payload[field] = np.full(original[field].shape, "1.0")
                    path = Path(directory) / f"bad-text-{field}.npz"
                    np.savez(path, **payload)
                    with self.assertRaisesRegex(ValueError, "real numeric dtype"):
                        load_hydraulic_model(path)

    def test_object_array_is_rejected_without_pickle_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self._valid_archive_payload(directory)
            payload["bias"] = np.asarray([object(), object(), object()], dtype=object)
            path = Path(directory) / "bad-object.npz"
            np.savez(path, **payload)

            with self.assertRaisesRegex(ValueError, "non-pickled real numeric"):
                load_hydraulic_model(path)

    def test_threshold_shape_and_finiteness_are_checked_before_coercion(self) -> None:
        corruptions = (
            (np.asarray([1.0, 2.0]), "zero or one"),
            (np.asarray([np.nan]), "finite"),
            (np.asarray([np.inf]), "finite"),
            (np.asarray([-np.inf]), "finite"),
        )
        with tempfile.TemporaryDirectory() as directory:
            original = self._valid_archive_payload(directory)
            for case, (value, pattern) in enumerate(corruptions):
                with self.subTest(case=case):
                    payload = {key: array.copy() for key, array in original.items()}
                    payload["threshold"] = value
                    path = Path(directory) / f"bad-threshold-{case}.npz"
                    np.savez(path, **payload)
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_hydraulic_model(path)

    def test_metadata_rejects_javascript_nonfinite_constants(self) -> None:
        invalid_json = (
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
        )
        with tempfile.TemporaryDirectory() as directory:
            original = self._valid_archive_payload(directory)
            for case, text in enumerate(invalid_json):
                with self.subTest(text=text):
                    payload = {key: array.copy() for key, array in original.items()}
                    payload["metadata_json"] = np.asarray(text)
                    path = Path(directory) / f"bad-json-{case}.npz"
                    np.savez(path, **payload)
                    with self.assertRaisesRegex(ValueError, "strict finite JSON"):
                        load_hydraulic_model(path)


@unittest.skipUnless(_cuda_available(), "CUDA is not available")
class HydraulicCUDATests(unittest.TestCase):
    def test_cuda_fit_is_reproducible_and_scoring_matches_numpy(self) -> None:
        rng = np.random.default_rng(20260907)
        labels_base, groups_base = _complete_group_data(group_count=3)
        labels = np.repeat(labels_base, 2)
        groups = np.repeat(groups_base, 2)
        features = rng.normal(
            scale=0.05,
            size=(labels.size, PHASE_COUNT, FEATURE_COUNT),
        )
        for cycle, label in enumerate(labels):
            if label == 100:
                features[cycle, :, :3] -= 1.5
            elif label == 90:
                features[cycle, :, 0] += 1.5
            elif label == 80:
                features[cycle, :, 1] += 1.5
            else:
                features[cycle, :, 2] += 1.5

        common = {
            "ridge_lambda": 1.0e-2,
            "max_iter": 40,
            "max_eval": 50,
            "history_size": 20,
        }
        first, first_metadata = fit_hydraulic_directional_cuda(
            features,
            labels,
            groups,
            **common,
        )
        second, _ = fit_hydraulic_directional_cuda(
            features,
            labels,
            groups,
            **common,
        )

        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.bias, second.bias)
        self.assertEqual(first.weights.dtype, np.float64)
        self.assertEqual(first_metadata["dtype"], "float64")
        numpy_path = score_hydraulic_numpy(first, features[:5])
        cuda_path, execution = score_hydraulic_cuda(first, features[:5], batch_size=2)
        comparison = compare_hydraulic_paths(numpy_path, cuda_path, atol=1.0e-10)
        self.assertTrue(comparison["eligible_exact"])
        self.assertTrue(comparison["alarm_exact"])
        self.assertEqual(execution["page_state_scope"], "reset_per_cycle")


if __name__ == "__main__":
    unittest.main()
