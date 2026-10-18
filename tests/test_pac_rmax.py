from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mate_spc_pac import (
    GATE_DEFER,
    GATE_PASS,
    DeploymentSpec,
    InsufficientCalibrationRunsError,
    PrefixApplicabilityConfig,
    StandardizedLogitRun,
    WSPACRMaxModel,
    calibrate_ws_pac_rmax_paths,
    load_ws_pac_rmax_model,
    minimum_pac_calibration_runs,
    pac_tolerance_rank,
    save_ws_pac_rmax_model,
    score_ws_pac_rmax_numpy,
)


UPSTREAM_SHA = "a" * 64
OTHER_SHA = "b" * 64


def _spec(
    head_ids: tuple[str, ...] = ("fault-3",),
    *,
    run_length: int = 6,
    valid_start: int = 0,
    monitor_start: int = 2,
) -> DeploymentSpec:
    return DeploymentSpec(
        run_length=run_length,
        valid_start=valid_start,
        monitor_start=monitor_start,
        head_ids=head_ids,
        upstream_artifact_sha256=UPSTREAM_SHA,
    )


def _gate(
    head_count: int,
    *,
    median_lower: float | np.ndarray = -10.0,
    median_upper: float | np.ndarray = 10.0,
    scale_lower: float | np.ndarray = 0.0,
    scale_upper: float | np.ndarray = 10.0,
    minimum: int = 2,
) -> PrefixApplicabilityConfig:
    return PrefixApplicabilityConfig(
        head_count=head_count,
        median_lower=median_lower,
        median_upper=median_upper,
        robust_scale_lower=scale_lower,
        robust_scale_upper=scale_upper,
        minimum_prefix_observations=minimum,
    )


def _model(
    spec: DeploymentSpec | None = None,
    *,
    gate: PrefixApplicabilityConfig | None = None,
    kappa: float | np.ndarray = 0.0,
) -> WSPACRMaxModel:
    deployment = _spec() if spec is None else spec
    return WSPACRMaxModel(
        deployment=deployment,
        prefix_gate=_gate(deployment.head_count) if gate is None else gate,
        kappa=kappa,
    )


def _run(
    values: np.ndarray,
    *,
    head_ids: tuple[str, ...] = ("fault-3",),
    upstream_sha256: str = UPSTREAM_SHA,
) -> StandardizedLogitRun:
    return StandardizedLogitRun(
        values=values,
        head_ids=head_ids,
        upstream_artifact_sha256=upstream_sha256,
    )


def _constant_path(model: WSPACRMaxModel, evidence: float):
    values = np.zeros((model.deployment.run_length, model.head_count), dtype=float)
    values[model.deployment.monitor_start :] = evidence
    return score_ws_pac_rmax_numpy(
        model,
        _run(values, head_ids=model.deployment.head_ids),
    )


class DeploymentBoundPathTests(unittest.TestCase):
    def test_asymmetric_transform_reset_and_per_head_kappa(self) -> None:
        spec = _spec(("fault-3", "fault-9"), run_length=7, monitor_start=3)
        model = _model(spec, gate=_gate(2, minimum=3), kappa=np.array([0.5, 1.0]))
        values = np.array(
            [
                [-2.0, 0.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [-3.0, 100.0],
                [2.0, 3.0],
                [2.0, 3.0],
                [2.0, 3.0],
            ]
        )

        path = score_ws_pac_rmax_numpy(model, _run(values, head_ids=spec.head_ids))

        scale = 1.4826 * 2.0
        self.assertEqual(path.status, GATE_PASS)
        self.assertEqual(path.transformed_evidence[3, 0], -3.0)
        self.assertAlmostEqual(path.transformed_evidence[4, 0], 2.0 / scale)
        np.testing.assert_array_equal(np.isnan(path.cusum[:3]), np.ones((3, 2), bool))
        np.testing.assert_allclose(path.clipped_evidence[3], np.array([-3.0, 8.0]))
        np.testing.assert_allclose(path.cusum[3, :], np.array([0.0, 7.0]))

    def test_deployment_rejects_horizon_head_order_and_upstream_mismatch(self) -> None:
        spec = _spec(("fault-3", "fault-9"))
        model = _model(spec)
        with self.assertRaisesRegex(ValueError, "horizon mismatch"):
            score_ws_pac_rmax_numpy(
                model,
                _run(np.zeros((7, 2)), head_ids=spec.head_ids),
            )
        with self.assertRaisesRegex(ValueError, "head_ids/order"):
            score_ws_pac_rmax_numpy(
                model,
                _run(np.zeros((6, 2)), head_ids=tuple(reversed(spec.head_ids))),
            )
        with self.assertRaisesRegex(ValueError, "upstream artifact"):
            score_ws_pac_rmax_numpy(
                model,
                _run(np.zeros((6, 2)), head_ids=spec.head_ids, upstream_sha256=OTHER_SHA),
            )

    def test_deployment_freezes_clip_and_persistence(self) -> None:
        with self.assertRaisesRegex(ValueError, "clip_limit is frozen"):
            DeploymentSpec(
                run_length=6,
                valid_start=0,
                monitor_start=2,
                head_ids=("fault-3",),
                upstream_artifact_sha256=UPSTREAM_SHA,
                clip_limit=7.0,
            )
        with self.assertRaisesRegex(ValueError, "persistence_window is frozen"):
            DeploymentSpec(
                run_length=6,
                valid_start=0,
                monitor_start=2,
                head_ids=("fault-3",),
                upstream_artifact_sha256=UPSTREAM_SHA,
                persistence_window=2,
            )

    def test_future_change_cannot_change_past_path(self) -> None:
        spec = _spec(("fault-3", "fault-9"), run_length=12)
        model = _model(spec, kappa=np.array([0.1, 0.2]))
        original = np.zeros((12, 2), dtype=float)
        original[2:, 0] = np.linspace(0.0, 4.5, 10)
        changed = original.copy()
        changed[9:] = 1000.0
        left = score_ws_pac_rmax_numpy(model, _run(original, head_ids=spec.head_ids))
        right = score_ws_pac_rmax_numpy(model, _run(changed, head_ids=spec.head_ids))
        for field in (
            "transformed_evidence",
            "clipped_evidence",
            "cusum",
            "component_persistence",
            "envelope",
            "eligible",
            "alarm",
        ):
            np.testing.assert_array_equal(getattr(left, field)[:9], getattr(right, field)[:9])

    def test_gate_defer_never_alarms(self) -> None:
        gate = _gate(1, median_upper=0.5)
        uncalibrated = _model(gate=gate)
        pass_paths = {
            f"pass-{index:03d}": _constant_path(uncalibrated, 1.0)
            for index in range(100)
        }
        certificate = calibrate_ws_pac_rmax_paths(pass_paths)
        model = uncalibrated.with_calibration(certificate)
        values = np.array([[2.0], [2.0], [100.0], [100.0], [100.0], [100.0]])

        path = score_ws_pac_rmax_numpy(model, _run(values))

        self.assertEqual(path.status, GATE_DEFER)
        self.assertFalse(path.eligible.any())
        self.assertFalse(path.alarm.any())
        self.assertGreater(path.envelope[-1], model.threshold)


class ConservativePACRankTests(unittest.TestCase):
    def test_required_n59_boundary_and_n100_documented_rank(self) -> None:
        self.assertEqual(minimum_pac_calibration_runs(epsilon=0.05, delta=0.05), 59)
        boundary = pac_tolerance_rank(59, epsilon=0.05, delta=0.05)
        documented = pac_tolerance_rank(100, epsilon=0.05, delta=0.05)

        self.assertEqual(boundary.rank_one_based, 59)
        self.assertLessEqual(boundary.upper_tail_probability_bound, 0.05)
        self.assertEqual(documented.rank_one_based, 99)
        self.assertAlmostEqual(documented.achieved_confidence, 0.962919, places=6)
        self.assertLessEqual(documented.upper_tail_probability_bound, 0.05)

        with self.assertRaises(InsufficientCalibrationRunsError) as caught:
            pac_tolerance_rank(58, epsilon=0.05, delta=0.05)
        self.assertEqual(caught.exception.minimum_count, 59)

    def test_tiny_supported_delta_uses_upper_tail_not_one_minus_delta(self) -> None:
        rank = pac_tolerance_rank(600, epsilon=0.05, delta=1e-12)

        self.assertEqual(rank.rank_one_based, 600)
        self.assertLessEqual(rank.upper_tail_probability_bound, 1e-12)
        self.assertGreater(rank.achieved_confidence, 0.999999999999)

    def test_tiny_epsilon_is_stable_and_unrepresentable_target_is_rejected(self) -> None:
        minimum = minimum_pac_calibration_runs(epsilon=1e-12, delta=0.05)
        self.assertGreater(minimum, 2_000_000_000_000)
        with self.assertRaises(InsufficientCalibrationRunsError):
            pac_tolerance_rank(100, epsilon=1e-12, delta=0.05)
        with self.assertRaisesRegex(ValueError, "binary64 range"):
            pac_tolerance_rank(100, epsilon=0.05, delta=1e-20)


class ConditionalCalibrationTests(unittest.TestCase):
    def test_certificate_reports_unconditional_pass_and_defer_counts(self) -> None:
        model = _model(gate=_gate(1, median_upper=0.5))
        paths = {
            f"pass-{index:03d}": _constant_path(model, (index + 1) / 100.0)
            for index in range(99)
        }
        deferred_values = np.array(
            [[2.0], [2.0], [100.0], [100.0], [100.0], [100.0]]
        )
        paths["defer-000"] = score_ws_pac_rmax_numpy(model, _run(deferred_values))

        certificate = calibrate_ws_pac_rmax_paths(
            paths,
            required_unconditional_count=100,
        )

        self.assertEqual(certificate.unconditional_sample_count, 100)
        self.assertEqual(certificate.pass_conditional_sample_count, 99)
        self.assertEqual(certificate.defer_sample_count, 1)
        self.assertEqual(certificate.deferred_runs, ("defer-000",))
        self.assertEqual(certificate.rank.sample_count, 99)
        self.assertEqual(len(certificate.calibration_maxima_sha256), 64)

    def test_too_many_defers_prevent_conditional_certificate(self) -> None:
        model = _model(gate=_gate(1, median_upper=0.5))
        paths = {
            f"pass-{index:03d}": _constant_path(model, 1.0)
            for index in range(58)
        }
        defer_values = np.array([[2.0], [2.0], [9.0], [9.0], [9.0], [9.0]])
        paths["defer-a"] = score_ws_pac_rmax_numpy(model, _run(defer_values))
        paths["defer-b"] = score_ws_pac_rmax_numpy(model, _run(defer_values))

        with self.assertRaises(InsufficientCalibrationRunsError) as caught:
            calibrate_ws_pac_rmax_paths(paths)

        self.assertEqual(caught.exception.pass_conditional_count, 58)
        self.assertEqual(caught.exception.unconditional_count, 60)
        self.assertEqual(caught.exception.deferred_runs, ("defer-a", "defer-b"))

    def test_tied_maxima_and_strict_threshold_do_not_alarm_at_equality(self) -> None:
        base = _model()
        paths = {
            f"healthy-{index:03d}": _constant_path(base, 1.0)
            for index in range(100)
        }
        certificate = calibrate_ws_pac_rmax_paths(paths)
        deployed = base.with_calibration(certificate)

        equal_path = _constant_path(deployed, 1.0)
        larger_path = _constant_path(deployed, 2.0)

        self.assertEqual(certificate.threshold, 2.0)
        self.assertFalse(equal_path.alarm.any())
        self.assertTrue(larger_path.alarm.any())
        self.assertEqual(certificate.comparison, "strictly_greater_than")

    def test_paths_from_different_specs_cannot_be_mixed(self) -> None:
        first = _model()
        second = _model(_spec(run_length=7))
        paths = {
            f"first-{index:03d}": _constant_path(first, 1.0)
            for index in range(59)
        }
        paths["other-spec"] = _constant_path(second, 1.0)
        with self.assertRaisesRegex(ValueError, "mix different deployment"):
            calibrate_ws_pac_rmax_paths(paths)

    def test_certificate_cannot_move_to_another_kappa_or_mix_its_paths(self) -> None:
        base = _model(kappa=0.0)
        changed = _model(kappa=0.5)
        paths = {
            f"base-{index:03d}": _constant_path(base, 1.0)
            for index in range(59)
        }
        certificate = calibrate_ws_pac_rmax_paths(paths)

        with self.assertRaisesRegex(ValueError, "another monitor configuration"):
            changed.with_calibration(certificate)
        paths["changed"] = _constant_path(changed, 1.0)
        with self.assertRaisesRegex(ValueError, "mix different monitor configurations"):
            calibrate_ws_pac_rmax_paths(paths)


def _tamper_archive(source: Path, target: Path, key: str, value: np.ndarray) -> None:
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload[key] = value
    np.savez_compressed(target, **payload)


class StrictSerializationTests(unittest.TestCase):
    def _calibrated_model(self) -> WSPACRMaxModel:
        base = _model()
        paths = {
            f"healthy-{index:03d}": _constant_path(base, (index + 1) / 100.0)
            for index in range(100)
        }
        return base.with_calibration(calibrate_ws_pac_rmax_paths(paths))

    def test_round_trip_binds_deployment_and_full_certificate(self) -> None:
        original = self._calibrated_model()
        metadata = {"phase": "unit-test", "seeds": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_ws_pac_rmax_model(original, path, metadata=metadata)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["run_length"].dtype, np.dtype(np.int64))
                self.assertEqual(archive["certificate_rank"].dtype, np.dtype(np.int64))
                self.assertFalse(
                    any(np.asarray(archive[key]).dtype.kind == "O" for key in archive.files)
                )
            restored, restored_metadata = load_ws_pac_rmax_model(path)

        self.assertEqual(restored_metadata, metadata)
        self.assertEqual(restored.deployment, original.deployment)
        self.assertEqual(restored.deployment.fingerprint, original.deployment.fingerprint)
        self.assertEqual(restored.calibration, original.calibration)
        self.assertEqual(
            restored.calibration.calibration_maxima_sha256,
            original.calibration.calibration_maxima_sha256,
        )

    def test_float_integer_field_is_rejected_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.npz"
            bad_run_length = Path(directory) / "bad-run-length.npz"
            bad_rank = Path(directory) / "bad-rank.npz"
            save_ws_pac_rmax_model(self._calibrated_model(), valid)
            _tamper_archive(
                valid,
                bad_run_length,
                "run_length",
                np.asarray([1.9], dtype=np.float64),
            )
            _tamper_archive(
                valid,
                bad_rank,
                "certificate_rank",
                np.asarray([1.9], dtype=np.float64),
            )

            with self.assertRaisesRegex(ValueError, "run_length must have dtype int64"):
                load_ws_pac_rmax_model(bad_run_length)
            with self.assertRaisesRegex(ValueError, "certificate_rank must have dtype int64"):
                load_ws_pac_rmax_model(bad_rank)

    def test_nonfinite_json_constant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.npz"
            bad = Path(directory) / "bad-json.npz"
            save_ws_pac_rmax_model(self._calibrated_model(), valid)
            _tamper_archive(valid, bad, "metadata_json", np.asarray('{"bad":NaN}'))

            with self.assertRaisesRegex(ValueError, "strict finite JSON"):
                load_ws_pac_rmax_model(bad)

    def test_tampered_maxima_fingerprint_or_confidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.npz"
            bad_fingerprint = Path(directory) / "bad-fingerprint.npz"
            bad_confidence = Path(directory) / "bad-confidence.npz"
            save_ws_pac_rmax_model(self._calibrated_model(), valid)
            _tamper_archive(
                valid,
                bad_fingerprint,
                "certificate_maxima_sha256",
                np.asarray("0" * 64),
            )
            _tamper_archive(
                valid,
                bad_confidence,
                "certificate_achieved_confidence",
                np.asarray([0.99], dtype=np.float64),
            )

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_ws_pac_rmax_model(bad_fingerprint)
            with self.assertRaisesRegex(ValueError, "fresh conservative calculation"):
                load_ws_pac_rmax_model(bad_confidence)


if __name__ == "__main__":
    unittest.main()
