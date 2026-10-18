from __future__ import annotations

import inspect

import numpy as np
import pytest

from mate_spc_cris_v9.baselines import PredictiveResidualCUSUMBaseline
from mate_spc_cris_v9.energy_anchor_monitor import (
    ENERGY_SCALE_FLOOR,
    FUSION_WEIGHT as IMPLEMENTED_FUSION_WEIGHT,
    ROBUST_MAD_FACTOR,
    ROBUST_SCALE_FLOOR,
    STRUCTURE_CLIP as IMPLEMENTED_STRUCTURE_CLIP,
    EnergyAnchoredHXMonitor,
    structure_products,
)


HORIZON_PAIRS = ((0, 1), (0, 2), (1, 2))
CHANNEL_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
STRUCTURAL_CHANNELS = 4
MAD_FACTOR = 1.4826
MAD_FLOOR = 1.0e-3
STRUCTURE_CLIP = 8.0
SCALE_FLOOR = 1.0e-8
CUSUM_ALLOWANCE = 0.5
FUSION_WEIGHT = 0.25


def _healthy_innovations(
    *, count: int = 7, length: int = 84, readonly: bool = True
) -> list[np.ndarray]:
    runs: list[np.ndarray] = []
    for run_index in range(count):
        rng = np.random.default_rng(7100 + run_index)
        values = rng.standard_t(df=5.0, size=(length, 3, 6)) * np.sqrt(3.0 / 5.0)
        values = np.asarray(values, dtype=np.float64)
        # The predictor has a horizon-dependent leading unavailable prefix.
        values[:60, :, :] = np.nan
        values[60:64, 1:, :] = np.nan
        values[64:74, 2:, :] = np.nan
        if readonly:
            values.setflags(write=False)
        runs.append(values)
    return runs


@pytest.fixture(scope="module")
def healthy_runs() -> list[np.ndarray]:
    return _healthy_innovations()


@pytest.fixture(scope="module")
def monitor(healthy_runs) -> EnergyAnchoredHXMonitor:
    return EnergyAnchoredHXMonitor.fit(
        healthy_runs[:5],
        include_horizon=True,
        include_cross_channel=True,
        structure_target_count=4,
    )


def _expected_h_products(run: np.ndarray) -> np.ndarray:
    # Exact frozen order: horizon-pair first, then the four liquid levels.
    return np.column_stack(
        [
            run[:, first_horizon, channel] * run[:, second_horizon, channel]
            for first_horizon, second_horizon in HORIZON_PAIRS
            for channel in range(STRUCTURAL_CHANNELS)
        ]
    )


def _expected_x_products(run: np.ndarray) -> np.ndarray:
    # Exact frozen order: liquid-level pair first, then three horizons.
    return np.column_stack(
        [
            run[:, horizon, first_channel] * run[:, horizon, second_channel]
            for first_channel, second_channel in CHANNEL_PAIRS
            for horizon in range(3)
        ]
    )


def test_energy_branch_exactly_reproduces_predictive_residual_cusum(
    healthy_runs,
) -> None:
    baseline = PredictiveResidualCUSUMBaseline.fit(healthy_runs[:5])
    anchored = EnergyAnchoredHXMonitor.fit(
        healthy_runs[:5],
        include_horizon=True,
        include_cross_channel=True,
        structure_target_count=4,
    )
    expected = baseline.score(healthy_runs[5])
    actual = anchored.score_components(healthy_runs[5])
    np.testing.assert_array_equal(actual.energy_increment, expected.point_score)
    np.testing.assert_array_equal(actual.energy_path, expected.path)
    assert anchored.energy_baseline.energy_center == baseline.energy_center
    assert anchored.energy_baseline.energy_scale == baseline.energy_scale
    assert anchored.energy_baseline.allowance == baseline.allowance


def test_h_and_x_product_dimensions_indices_and_formulas(monitor) -> None:
    time = np.arange(1, 6, dtype=np.float64)[:, None, None]
    horizon = (10.0 * np.arange(1, 4, dtype=np.float64))[None, :, None]
    channel = np.arange(1, 7, dtype=np.float64)[None, None, :]
    run = time * 100.0 + horizon + channel
    h_products = structure_products(
        run,
        include_horizon=True,
        include_cross_channel=False,
        structure_target_count=4,
    )
    x_products = structure_products(
        run,
        include_horizon=False,
        include_cross_channel=True,
        structure_target_count=4,
    )
    assert h_products.shape == (5, 12)
    assert x_products.shape == (5, 18)
    np.testing.assert_array_equal(h_products, _expected_h_products(run))
    np.testing.assert_array_equal(x_products, _expected_x_products(run))
    # Channels 4 and 5 are predictor auxiliaries: they remain in the radial
    # energy branch but must never enter H/X structure products.
    changed_auxiliary = np.array(run, copy=True)
    changed_auxiliary[:, :, 4:] += 1.0e6
    changed_h = structure_products(
        changed_auxiliary,
        include_horizon=True,
        include_cross_channel=False,
        structure_target_count=4,
    )
    changed_x = structure_products(
        changed_auxiliary,
        include_horizon=False,
        include_cross_channel=True,
        structure_target_count=4,
    )
    np.testing.assert_array_equal(h_products, changed_h)
    np.testing.assert_array_equal(x_products, changed_x)


def test_healthy_product_fit_uses_frozen_median_mad_then_structure_zscore(
    healthy_runs, monitor
) -> None:
    products: list[np.ndarray] = []
    for run in healthy_runs[:5]:
        h_products = _expected_h_products(run)
        x_products = _expected_x_products(run)
        combined = np.concatenate((h_products, x_products), axis=1)
        products.append(combined[np.isfinite(combined).all(axis=1)])
    joined = np.concatenate(products, axis=0)
    expected_center = np.median(joined, axis=0)
    expected_scale = np.maximum(
        MAD_FACTOR * np.median(np.abs(joined - expected_center[None, :]), axis=0),
        MAD_FLOOR,
    )
    standardized = np.clip(
        (joined - expected_center[None, :]) / expected_scale[None, :],
        -STRUCTURE_CLIP,
        STRUCTURE_CLIP,
    )
    structure_energy = np.mean(np.square(standardized), axis=1)
    expected_structure_center = float(np.mean(structure_energy))
    expected_structure_scale = max(float(np.std(structure_energy)), SCALE_FLOOR)

    np.testing.assert_allclose(monitor.product_center, expected_center, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(monitor.product_scale, expected_scale, atol=0.0, rtol=0.0)
    assert monitor.structure_energy_center == expected_structure_center
    assert monitor.structure_energy_scale == expected_structure_scale
    assert ROBUST_MAD_FACTOR == MAD_FACTOR
    assert ROBUST_SCALE_FLOOR == MAD_FLOOR
    assert ENERGY_SCALE_FLOOR == SCALE_FLOOR
    assert IMPLEMENTED_STRUCTURE_CLIP == STRUCTURE_CLIP
    assert monitor.allowance == CUSUM_ALLOWANCE
    assert IMPLEMENTED_FUSION_WEIGHT == FUSION_WEIGHT
    assert monitor.fusion_weight == FUSION_WEIGHT


def test_score_is_strictly_causal_under_future_innovation_changes(
    healthy_runs, monitor
) -> None:
    original = np.array(healthy_runs[5], copy=True)
    changed = np.array(original, copy=True)
    cutoff = 76
    changed[cutoff + 1 :] += np.linspace(5.0, 30.0, 18).reshape(3, 6)
    before = monitor.score_components(original)
    after = monitor.score_components(changed)
    for name in (
        "energy_increment",
        "energy_path",
        "structure_increment",
        "structure_path",
        "fused_path",
    ):
        np.testing.assert_array_equal(
            getattr(before, name)[: cutoff + 1],
            getattr(after, name)[: cutoff + 1],
        )


def test_nan_in_any_structural_liquid_level_holds_structure_state(
    healthy_runs, monitor
) -> None:
    run = np.array(healthy_runs[5], copy=True)
    missing_time = 78
    run[missing_time, 1, 2] = np.nan
    score = monitor.score_components(run)
    assert score.structure_increment[missing_time] == 0.0
    assert score.structure_path[missing_time] == score.structure_path[missing_time - 1]
    # A later fully observed row resumes the recurrence.
    assert np.isfinite(score.structure_increment[missing_time + 1])
    assert np.isfinite(score.structure_path).all()


def test_energy_anchored_fusion_is_exact_and_bounded(healthy_runs, monitor) -> None:
    score = monitor.score_components(healthy_runs[6])
    expected = score.energy_path + FUSION_WEIGHT * np.minimum(
        score.energy_path, score.structure_path
    )
    np.testing.assert_allclose(score.fused_path, expected, atol=0.0, rtol=0.0)
    assert np.all(score.fused_path >= score.energy_path)
    assert np.all(score.fused_path <= (1.0 + FUSION_WEIGHT) * score.energy_path)
    assert np.isfinite(score.fused_path).all()
    assert not score.fused_path.flags.writeable
    public_score = monitor.score(healthy_runs[6])
    np.testing.assert_array_equal(public_score.path, score.fused_path)


def test_fit_is_healthy_numeric_only_and_has_no_fault_label_channel(
    healthy_runs,
) -> None:
    signature = inspect.signature(EnergyAnchoredHXMonitor.fit)
    assert tuple(signature.parameters) == (
        "healthy_fit_runs",
        "include_horizon",
        "include_cross_channel",
        "structure_target_count",
    )
    for forbidden in ("labels", "fault_labels", "fault_kind", "onset"):
        assert forbidden not in signature.parameters
    with pytest.raises(TypeError):
        EnergyAnchoredHXMonitor.fit(
            healthy_runs[:2], fault_labels=("healthy", "fault")
        )
    with pytest.raises(ValueError, match="numeric"):
        EnergyAnchoredHXMonitor.fit(
            [np.full((80, 3, 6), "not-numeric", dtype="<U11")]
        )
    # The actual predictor output is (T,3,6); only its first four channels are
    # structural, while all six remain in the fitted radial-energy baseline.
    fitted = EnergyAnchoredHXMonitor.fit(healthy_runs[:2])
    assert fitted.energy_baseline.innovation_shape == (3, 6)
    assert fitted.structure_target_count == 4


@pytest.mark.parametrize(
    ("include_h", "include_x", "expected_features"),
    ((True, False, 12), (False, True, 18), (True, True, 30)),
)
def test_minimal_two_factor_ablation_geometry(
    healthy_runs, include_h: bool, include_x: bool, expected_features: int
) -> None:
    candidate = EnergyAnchoredHXMonitor.fit(
        healthy_runs[:5],
        include_horizon=include_h,
        include_cross_channel=include_x,
        structure_target_count=4,
    )
    assert candidate.product_center.shape == (expected_features,)
    assert candidate.product_scale.shape == (expected_features,)
    score = candidate.score(healthy_runs[5])
    assert np.isfinite(score.path).all()


def test_disabling_both_structure_factors_is_rejected(healthy_runs) -> None:
    with pytest.raises(ValueError, match="at least one structure product family"):
        EnergyAnchoredHXMonitor.fit(
            healthy_runs[:5],
            include_horizon=False,
            include_cross_channel=False,
        )
