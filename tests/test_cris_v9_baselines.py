from __future__ import annotations

import inspect

import numpy as np
import pytest

from mate_spc_cris_v9.baselines import (
    BaselineScore,
    BaselineThreshold,
    EnergyCUSUMBaseline,
    PCAT2QBaseline,
    PredictiveResidualCUSUMBaseline,
    STRICT_COMPARISON,
    apply_trajectory_threshold,
)


def _healthy_matrix_runs(
    *, count: int = 22, length: int = 80, readonly: bool = True
) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    for run_index in range(count):
        rng = np.random.default_rng(1000 + run_index)
        values = np.zeros((length, 4), dtype=np.float64)
        noise = rng.normal(0.0, 0.18, size=(length, 4))
        for time in range(1, length):
            values[time] = (
                np.array((0.74, 0.67, 0.59, 0.51)) * values[time - 1]
                + noise[time]
            )
        values[:, 2] += 0.35 * values[:, 0]
        values[:, 3] -= 0.25 * values[:, 1]
        if readonly:
            values.setflags(write=False)
        runs.append(values)
    return runs


def _healthy_innovation_runs(
    *, count: int = 22, length: int = 80, readonly: bool = True
) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    for run_index in range(count):
        rng = np.random.default_rng(2000 + run_index)
        values = rng.standard_t(df=5.0, size=(length, 3, 4)) * np.sqrt(3.0 / 5.0)
        values = np.asarray(values, dtype=np.float64)
        values[:15] = np.nan
        if readonly:
            values.setflags(write=False)
        runs.append(values)
    return runs


@pytest.fixture(scope="module")
def fitted_baselines():
    matrix_runs = _healthy_matrix_runs()
    innovation_runs = _healthy_innovation_runs()
    pca = PCAT2QBaseline.fit(matrix_runs[:10])
    energy = EnergyCUSUMBaseline.fit(matrix_runs[:10])
    predictive = PredictiveResidualCUSUMBaseline.fit(innovation_runs[:10])
    return pca, energy, predictive, matrix_runs, innovation_runs


def test_all_fit_apis_are_numeric_healthy_only_without_label_channels() -> None:
    for baseline_class in (
        PCAT2QBaseline,
        EnergyCUSUMBaseline,
        PredictiveResidualCUSUMBaseline,
    ):
        signature = inspect.signature(baseline_class.fit)
        assert tuple(signature.parameters) == ("healthy_fit_runs",)
        assert all(
            forbidden not in signature.parameters
            for forbidden in ("labels", "fault_labels", "fault_kind", "onset")
        )
    with pytest.raises(TypeError):
        PCAT2QBaseline.fit(_healthy_matrix_runs(count=2), fault_labels=[0, 1])


def test_pca_is_healthy_fit_t2_plus_q_with_readonly_state(fitted_baselines) -> None:
    pca, _energy, _predictive, matrix_runs, _innovation_runs = fitted_baselines
    assert 1 <= pca.components.shape[0] <= matrix_runs[0].shape[1]
    assert pca.components.shape[1] == matrix_runs[0].shape[1]
    assert np.all(pca.eigenvalues > 0.0)
    assert pca.t2_reference_mean > 0.0
    assert pca.q_reference_mean > 0.0
    for value in (pca.center, pca.scale, pca.components, pca.eigenvalues):
        assert not value.flags.writeable
    score = pca.score(matrix_runs[10])
    assert np.array_equal(score.point_score, score.path)
    assert np.isfinite(score.path).all()


def test_all_baselines_return_finite_readonly_paths(fitted_baselines) -> None:
    pca, energy, predictive, matrix_runs, innovation_runs = fitted_baselines
    cases = (
        (pca, matrix_runs[10]),
        (energy, matrix_runs[10]),
        (predictive, innovation_runs[10]),
    )
    for baseline, values in cases:
        score = baseline.score(values)
        assert score.path.shape == (values.shape[0],)
        assert score.point_score.shape == score.path.shape
        assert np.isfinite(score.point_score).all()
        assert np.isfinite(score.path).all()
        assert not score.point_score.flags.writeable
        assert not score.path.flags.writeable
        assert score.trajectory_maximum == float(np.max(score.path))
    predictive_score = predictive.score(innovation_runs[10])
    assert np.array_equal(predictive_score.path[:15], np.zeros(15))


def test_each_score_is_causal_under_future_perturbation(fitted_baselines) -> None:
    pca, energy, predictive, matrix_runs, innovation_runs = fitted_baselines
    cutoff = 43
    matrix = np.array(matrix_runs[11], copy=True)
    changed_matrix = np.array(matrix, copy=True)
    changed_matrix[cutoff + 1 :] += np.array((20.0, -30.0, 40.0, -50.0))
    innovations = np.array(innovation_runs[11], copy=True)
    changed_innovations = np.array(innovations, copy=True)
    changed_innovations[cutoff + 1 :] += 25.0
    for baseline in (pca, energy):
        before = baseline.score(matrix)
        after = baseline.score(changed_matrix)
        np.testing.assert_array_equal(
            before.point_score[: cutoff + 1], after.point_score[: cutoff + 1]
        )
        np.testing.assert_array_equal(before.path[: cutoff + 1], after.path[: cutoff + 1])
    before = predictive.score(innovations)
    after = predictive.score(changed_innovations)
    np.testing.assert_array_equal(before.point_score[: cutoff + 1], after.point_score[: cutoff + 1])
    np.testing.assert_array_equal(before.path[: cutoff + 1], after.path[: cutoff + 1])


def test_threshold_uses_one_maximum_from_every_complete_healthy_run(fitted_baselines) -> None:
    pca, energy, predictive, matrix_runs, innovation_runs = fitted_baselines
    cases = (
        (pca, matrix_runs[12:22]),
        (energy, matrix_runs[12:22]),
        (predictive, innovation_runs[12:22]),
    )
    for baseline, calibration_runs in cases:
        threshold = baseline.calibrate_trajectory_threshold(calibration_runs, alpha=0.10)
        expected = np.asarray(
            [baseline.score(run).trajectory_maximum for run in calibration_runs],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(threshold.calibration_maxima, expected)
        assert threshold.rank == 10
        assert threshold.threshold == float(np.max(expected))
        assert threshold.comparison == STRICT_COMPARISON
        assert not threshold.calibration_maxima.flags.writeable


def test_calibration_does_not_truncate_late_trajectory_samples(fitted_baselines) -> None:
    _pca, energy, _predictive, matrix_runs, _innovation_runs = fitted_baselines
    calibration = [np.array(run, copy=True) for run in matrix_runs[12:22]]
    calibration[3][-1] += 100.0
    threshold = energy.calibrate_trajectory_threshold(calibration, alpha=0.10)
    late_maximum = energy.score(calibration[3]).trajectory_maximum
    assert threshold.calibration_maxima[3] == late_maximum
    assert late_maximum == np.max(threshold.calibration_maxima)


def test_threshold_application_is_strict_and_ties_do_not_alarm() -> None:
    threshold = BaselineThreshold(
        baseline_name="energy_cusum",
        alpha=0.50,
        rank=1,
        threshold=2.0,
        calibration_maxima=np.asarray([2.0], dtype=np.float64),
    )
    score = BaselineScore(
        baseline_name="energy_cusum",
        point_score=np.asarray((0.0, 2.0, 2.0 + 1.0e-12), dtype=np.float64),
        path=np.asarray((0.0, 2.0, 2.0 + 1.0e-12), dtype=np.float64),
    )
    decision = apply_trajectory_threshold(score, threshold)
    np.testing.assert_array_equal(decision.alarm, np.asarray((False, False, True)))
    assert decision.first_alarm_index == 2
    assert decision.detected is True
    assert not decision.alarm.flags.writeable


def test_too_few_healthy_trajectories_for_requested_alpha_fails_closed(
    fitted_baselines,
) -> None:
    pca, _energy, _predictive, matrix_runs, _innovation_runs = fitted_baselines
    with pytest.raises(ValueError, match="unreachable"):
        pca.calibrate_trajectory_threshold(matrix_runs[12:17], alpha=0.01)


def test_predictive_baseline_is_only_squared_energy_plus_ordinary_cusum(
    fitted_baselines,
) -> None:
    _pca, _energy, predictive, _matrix_runs, innovation_runs = fitted_baselines
    run = innovation_runs[10]
    score = predictive.score(run)
    finite = np.isfinite(run)
    count = np.sum(finite, axis=(1, 2))
    raw_energy = np.zeros(run.shape[0], dtype=np.float64)
    available = count > 0
    raw_energy[available] = np.sum(
        np.square(np.where(finite, run, 0.0)), axis=(1, 2)
    )[available] / count[available]
    expected_increment = np.zeros(run.shape[0], dtype=np.float64)
    expected_increment[available] = (
        (raw_energy[available] - predictive.energy_center) / predictive.energy_scale
        - predictive.allowance
    )
    expected_path = np.empty(run.shape[0], dtype=np.float64)
    state = 0.0
    for index, increment in enumerate(expected_increment):
        state = max(0.0, state + float(increment))
        expected_path[index] = state
    np.testing.assert_allclose(score.point_score, expected_increment, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(score.path, expected_path, atol=0.0, rtol=0.0)


def test_baseline_threshold_cannot_be_applied_to_another_baseline(fitted_baselines) -> None:
    pca, energy, _predictive, matrix_runs, _innovation_runs = fitted_baselines
    threshold = pca.calibrate_trajectory_threshold(matrix_runs[12:22], alpha=0.10)
    with pytest.raises(ValueError, match="different baselines"):
        apply_trajectory_threshold(energy.score(matrix_runs[10]), threshold)
