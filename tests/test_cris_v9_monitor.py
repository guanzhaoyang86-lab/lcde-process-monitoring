from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from mate_spc_cris_v9.monitor import (
    ComponentKey,
    MonitorConfig,
    PointCalibrator,
    PointStatistics,
    UnreachableThresholdError,
    apply_threshold,
    build_point_statistics,
    calibrate_trajectory_threshold,
    hmm_log_forward,
    scan_innovations,
)


def _index(statistics: PointStatistics, key: ComponentKey) -> int:
    return statistics.components.index(key)


def _brute_hmm_prefix(e_values: np.ndarray, rho: float, recovery: float) -> np.ndarray:
    expected: list[float] = []
    for stop in range(1, e_values.size + 1):
        total = 0.0
        for states in itertools.product((0, 1), repeat=stop):
            probability = 1.0
            previous = 0
            emissions = 1.0
            for time, state in enumerate(states):
                if previous == 0:
                    probability *= rho if state == 1 else 1.0 - rho
                else:
                    probability *= 1.0 - recovery if state == 1 else recovery
                if state == 1:
                    emissions *= float(e_values[time])
                previous = state
            total += probability * emissions
        expected.append(math.log(total))
    return np.asarray(expected)


def test_moment_construction_is_directional_and_causal() -> None:
    config = MonitorConfig(top_k=1)
    innovations = np.asarray(
        [
            [[2.0, -1.0]],
            [[3.0, -2.0]],
            [[4.0, -3.0]],
        ],
        dtype=np.float64,
    )
    statistics = build_point_statistics(innovations, config)

    location_positive = _index(
        statistics, ComponentKey("location", 0, 4, "positive")
    )
    location_negative = _index(
        statistics, ComponentKey("location", 0, 4, "negative")
    )
    scale_positive = _index(
        statistics, ComponentKey("scale", 0, 4, "positive")
    )
    dependence_positive = _index(
        statistics, ComponentKey("dependence", 0, 4, "positive")
    )
    dependence_negative = _index(
        statistics, ComponentKey("dependence", 0, 4, "negative")
    )

    assert statistics.values[0, location_positive] == pytest.approx(2.0)
    assert statistics.values[0, location_negative] == pytest.approx(0.0)
    assert statistics.values[1, location_positive] == pytest.approx(5.0 / math.sqrt(2.0))
    assert statistics.values[1, location_negative] == pytest.approx(0.0)
    assert statistics.values[0, scale_positive] == pytest.approx(3.0 / math.sqrt(2.0))
    assert math.isnan(statistics.values[0, dependence_positive])
    assert statistics.values[1, dependence_positive] == pytest.approx(6.0)
    assert statistics.values[1, dependence_negative] == pytest.approx(0.0)

    changed = innovations.copy()
    changed[2] = 1.0e6
    changed_statistics = build_point_statistics(changed, config)
    np.testing.assert_array_equal(
        changed_statistics.values[:2], statistics.values[:2]
    )


def test_nan_is_unknown_immediately_and_partial_dimensions_are_fixed_topk() -> None:
    innovations = np.ones((4, 1, 3), dtype=np.float64)
    innovations[1, 0] = np.nan
    innovations[2, 0, 0] = np.nan
    statistics = build_point_statistics(innovations, MonitorConfig(top_k=2))
    location = [
        index
        for index, key in enumerate(statistics.components)
        if key.moment == "location"
    ]
    assert np.isnan(statistics.values[1, location]).all()
    assert np.isfinite(statistics.values[2, location]).all()


def test_topk_is_selected_by_absolute_magnitude_before_splitting_signs() -> None:
    innovations = np.asarray([[[-100.0, 3.0, 2.0]]], dtype=np.float64)
    statistics = build_point_statistics(innovations, MonitorConfig(top_k=2))
    positive = _index(
        statistics, ComponentKey("location", 0, 4, "positive")
    )
    negative = _index(
        statistics, ComponentKey("location", 0, 4, "negative")
    )
    # Selected coordinates are -100 and +3.  The unselected +2 must not enter
    # an independently chosen positive top-k set.
    assert statistics.values[0, positive] == pytest.approx(1.5)
    assert statistics.values[0, negative] == pytest.approx(50.0)


def test_upper_tail_conformal_ties_and_power_e_value() -> None:
    config = MonitorConfig(top_k=1)
    component_count = 3 * 1 * 3 * 2
    samples = tuple(
        np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
        for _ in range(component_count)
    )
    calibrator = PointCalibrator.from_sorted_samples(
        samples, horizon_count=1, dimension_count=2, config=config
    )
    statistics = PointStatistics(
        values=np.full((1, component_count), 2.0, dtype=np.float64),
        components=calibrator.components,
    )
    p_values = calibrator.p_values(statistics)
    e_values = calibrator.e_values(statistics)
    # Three calibration values (2,2,4) are >= the tied observation.
    np.testing.assert_allclose(p_values, 4.0 / 5.0)
    np.testing.assert_allclose(e_values, 0.5 * (4.0 / 5.0) ** -0.5)


def test_hmm_log_forward_matches_exhaustive_state_paths() -> None:
    e_values = np.asarray([2.0, 0.5, 3.0, 1.25], dtype=np.float64)
    rho = 0.10
    recovery = 0.50
    trace = hmm_log_forward(
        np.log(e_values), rho=rho, recovery=recovery, reset_gap=64
    )
    np.testing.assert_allclose(
        trace.score, _brute_hmm_prefix(e_values, rho, recovery), rtol=1e-13, atol=1e-13
    )


def test_hmm_unknown_pauses_and_long_gap_resets() -> None:
    compressed = hmm_log_forward(
        np.log(np.asarray([2.0, 3.0])), rho=0.02, recovery=0.20, reset_gap=5
    )
    short_gap = hmm_log_forward(
        np.asarray([math.log(2.0), np.nan, math.log(3.0)]),
        rho=0.02,
        recovery=0.20,
        reset_gap=5,
    )
    assert math.isnan(short_gap.score[1])
    assert short_gap.held_score[1] == pytest.approx(short_gap.held_score[0])
    assert short_gap.score[2] == pytest.approx(compressed.score[1])

    reset = hmm_log_forward(
        np.asarray([math.log(2.0), np.nan, np.nan, math.log(3.0)]),
        rho=0.02,
        recovery=0.20,
        reset_gap=2,
    )
    standalone = hmm_log_forward(
        np.asarray([math.log(3.0)]), rho=0.02, recovery=0.20, reset_gap=2
    )
    assert reset.reset.tolist() == [False, False, True, False]
    assert reset.score[3] == pytest.approx(standalone.score[0])


def test_scan_outputs_components_and_global_unknown_pause() -> None:
    rng = np.random.default_rng(20260909)
    config = MonitorConfig(top_k=2)
    healthy = [rng.normal(size=(12, 2, 4)) for _ in range(3)]
    calibrator = PointCalibrator.fit(healthy, config)
    target = rng.normal(size=(12, 2, 4))
    target[4:7] = np.nan
    result = scan_innovations(target, calibrator, config, threshold=0.0)

    assert result.score.shape == (12,)
    assert result.p_values.shape == (12, 3 * 2 * 3 * 2)
    assert result.component_scores.shape == (12, 3 * 2 * 3 * 2, 2)
    assert result.component_valid.shape == result.p_values.shape
    assert result.alarm is not None
    assert not result.valid[4:7].any()
    assert np.isnan(result.score[4:7]).all()
    assert not result.alarm[4:7].any()
    # The registered 64-sample reset is not reached by this short gap.
    assert np.isfinite(result.component_scores[6]).all()


def test_trajectory_rank_reachability_ties_and_strict_threshold() -> None:
    paths = [np.asarray([float(index), np.nan]) for index in range(20)]
    threshold = calibrate_trajectory_threshold(paths, alpha=0.05)
    assert threshold.ascending_rank == 20
    assert threshold.value == 19.0
    assert threshold.comparison == "strictly_greater_than"
    alarms = apply_threshold(
        np.asarray([np.nan, 18.0, 19.0, 19.000001]), threshold
    )
    assert alarms.tolist() == [False, False, False, True]

    with pytest.raises(UnreachableThresholdError, match="exceeds n=20"):
        calibrate_trajectory_threshold(paths, alpha=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"moment_weights": (0.0, 0.0, 0.0)},
        {"rolling_scales": (4, 8, 64)},
        {"eta": 0.4},
        {"dependence_lag": 2},
        {"reset_gap": 63},
    ],
)
def test_monitor_config_refuses_unregistered_method_changes(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MonitorConfig(**kwargs)
