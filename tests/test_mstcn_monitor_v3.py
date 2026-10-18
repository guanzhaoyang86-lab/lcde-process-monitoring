from __future__ import annotations

import numpy as np
import pytest

from mate_spc_mstcn.monitor import (
    FittedTopTailMonitor,
    RobustPointwiseStandardizer,
    causal_fault_attribution,
    first_persistent_exceedance,
    multiscale_top_tail_scan,
    strict_order_statistic,
)


def test_pointwise_standardizer_uses_frozen_q10_linear_mad_floor() -> None:
    healthy = np.asarray(
        [
            [0.0, 10.0, 2.0, 7.0],
            [1.0, 12.0, 2.0, 8.0],
            [2.0, 14.0, 2.0, 9.0],
            [100.0, 16.0, 2.0, 10.0],
        ],
        dtype=np.float64,
    )
    standardizer = RobustPointwiseStandardizer.fit(healthy)
    raw_scales = np.asarray([1.4826, 2.9652, 0.0, 1.4826], dtype=np.float64)
    expected_floor = np.quantile(raw_scales, 0.10, method="linear")
    assert standardizer.scale_floor == pytest.approx(expected_floor)
    np.testing.assert_array_equal(standardizer.center, [1.5, 13.0, 2.0, 8.5])
    assert standardizer.scale[2] == standardizer.scale_floor
    document = standardizer.to_document()
    assert document["floor_quantile"] == 0.10
    assert document["quantile_method"] == "linear"

    validation = np.full(4, 1.0e6, dtype=np.float64)
    transformed = standardizer.transform(validation)
    assert transformed.shape == (4,)
    assert standardizer.to_document() == document  # transform cannot update state
    restored = RobustPointwiseStandardizer.from_document(document)
    np.testing.assert_array_equal(restored.center, standardizer.center)
    np.testing.assert_array_equal(restored.scale, standardizer.scale)


def _identity_monitor(endpoints: np.ndarray) -> FittedTopTailMonitor:
    pointwise = RobustPointwiseStandardizer(
        np.zeros(endpoints.size, dtype=np.float64),
        np.ones(endpoints.size, dtype=np.float64),
        1.0,
    )
    return FittedTopTailMonitor(
        endpoints=endpoints,
        pointwise=pointwise,
        windows=(4, 8),
        top_fraction=0.25,
        branch_center=np.zeros(2, dtype=np.float64),
        branch_scale=np.ones(2, dtype=np.float64),
        branch_scale_floor=1.0,
    )


def test_fitted_scan_has_explicit_g_and_two_point_p_and_is_causal() -> None:
    endpoints = np.arange(160, 180, dtype=np.int64)
    evidence = np.arange(20, dtype=np.float64)
    monitor = _identity_monitor(endpoints)
    path = multiscale_top_tail_scan(evidence, monitor)

    assert np.isneginf(path.grid_values[:3]).all()
    assert path.grid_values[3] == 3.0  # W=4, top one of [0,1,2,3]
    assert np.isneginf(path.persistent_values[:4]).all()
    assert path.persistent_values[4] == 3.0
    assert path.run_maximum == 18.0  # max min(G[t-1], G[t]), not max G=19

    changed = evidence.copy()
    changed[12:] += 1.0e6
    changed_path = monitor.transform(changed)
    np.testing.assert_array_equal(changed_path.grid_values[:12], path.grid_values[:12])
    np.testing.assert_array_equal(
        changed_path.persistent_values[:12], path.persistent_values[:12]
    )


def test_fitted_monitor_is_training_only_single_rho_and_round_trips() -> None:
    rng = np.random.default_rng(7)
    endpoints = np.arange(160, 200, dtype=np.int64)
    healthy = rng.normal(size=(12, endpoints.size)).astype(np.float64)
    monitor = FittedTopTailMonitor.fit(
        healthy,
        endpoints,
        windows=(4, 8, 16),
        top_fraction=0.125,
    )
    assert monitor.top_fraction == 0.125
    before = monitor.to_document()
    validation = rng.normal(loc=20.0, size=endpoints.size).astype(np.float64)
    scored = monitor.transform(validation)
    assert np.isfinite(scored.run_maximum)
    assert monitor.to_document() == before

    restored = FittedTopTailMonitor.from_document(before)
    replay = restored.transform(validation)
    np.testing.assert_array_equal(replay.grid_values, scored.grid_values)
    np.testing.assert_array_equal(replay.persistent_values, scored.persistent_values)
    np.testing.assert_array_equal(replay.winning_windows, scored.winning_windows)


def test_rank_79_of_80_and_strict_persistent_alarm() -> None:
    maxima = np.arange(1, 81, dtype=np.float64)
    threshold = strict_order_statistic(maxima, rank=79)
    assert threshold == 79.0
    assert np.count_nonzero(maxima > threshold) == 1

    endpoints = np.arange(160, 166, dtype=np.int64)
    pointwise = RobustPointwiseStandardizer(
        np.zeros(6, dtype=np.float64), np.ones(6, dtype=np.float64), 1.0
    )
    monitor = FittedTopTailMonitor(
        endpoints,
        pointwise,
        (1,),
        1.0,
        np.zeros(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        1.0,
    )
    path = monitor.transform(
        np.asarray([0.0, 1.0, 1.0, 2.0, 2.0, 0.0], dtype=np.float64)
    )
    assert first_persistent_exceedance(path, 1.0) == 164
    assert first_persistent_exceedance(path, 2.0) is None


def test_fault_attribution_uses_no_logits_after_alarm() -> None:
    endpoints = np.arange(160, 165, dtype=np.int64)
    logits = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert causal_fault_attribution(logits, endpoints, 162) == 3
    assert causal_fault_attribution(logits, endpoints, None) is None


def test_monitor_rejects_malformed_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="at_least_2"):
        RobustPointwiseStandardizer.fit(np.zeros((1, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="exact float64"):
        RobustPointwiseStandardizer.fit(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="rank exceeds"):
        strict_order_statistic(np.arange(4, dtype=np.float64), rank=5)
    monitor = _identity_monitor(np.arange(160, 180, dtype=np.int64))
    broken = monitor.to_document()
    broken["top_fraction"] = [0.125, 0.25]
    with pytest.raises((TypeError, ValueError)):
        FittedTopTailMonitor.from_document(broken)

