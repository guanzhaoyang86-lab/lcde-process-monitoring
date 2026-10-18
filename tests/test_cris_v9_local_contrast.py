from __future__ import annotations

import numpy as np
import pytest

from mate_spc_cris_v9.local_contrast import (
    LocalContrastConfig,
    LocalContrastDualEnergyMonitor,
    REFERENCE_ROWS,
    WARMUP_ROWS,
    local_contrast_components,
)


def _case(seed: int = 7, length: int = 520, dimension: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(length, dimension))
    innovations = rng.normal(size=(length, 3, dimension))
    innovations[:75] = np.nan
    return x, innovations


def test_components_have_frozen_alignment_and_are_causal() -> None:
    x, innovations = _case()
    config = LocalContrastConfig(current_window=20)
    first = local_contrast_components(x, innovations, config)
    assert first.shape == (x.shape[0], 5)
    assert np.isnan(first[:REFERENCE_ROWS]).all()
    assert np.isfinite(first[REFERENCE_ROWS:]).all()

    changed_x = x.copy()
    changed_innovations = innovations.copy()
    changed_x[450:] += 50.0
    changed_innovations[450:] *= 10.0
    changed = local_contrast_components(changed_x, changed_innovations, config)
    np.testing.assert_array_equal(first[:450], changed[:450])


def test_dual_scan_reacts_to_sustained_raw_and_predictive_change() -> None:
    healthy_x, healthy_innovations = _case(8, length=700)
    config = LocalContrastConfig(
        current_window=20, channel_top_k=2, allowance=1.0, scan_horizon=60
    )
    monitor = LocalContrastDualEnergyMonitor.fit(
        healthy_x, healthy_innovations, config
    )
    test_x, test_innovations = _case(9, length=700)
    baseline = monitor.score(test_x, test_innovations)
    test_x[520:, :2] += 3.0
    test_innovations[520:, :, 2:4] *= 4.0
    shifted = monitor.score(test_x, test_innovations)
    assert np.max(shifted[540:]) > np.max(baseline[540:])
    assert np.all(shifted[:WARMUP_ROWS] == 0.0)


def test_modes_and_state_are_deterministic_and_read_only() -> None:
    x, innovations = _case(10, length=650)
    monitor = LocalContrastDualEnergyMonitor.fit(
        x, innovations, LocalContrastConfig()
    )
    for mode in ("dual", "raw_only", "predictive_only"):
        first = monitor.score(x, innovations, mode=mode)
        second = monitor.score(x, innovations, mode=mode)
        np.testing.assert_array_equal(first, second)
    assert not monitor.component_center.flags.writeable
    assert not monitor.component_scale.flags.writeable
    with pytest.raises(ValueError, match="mode"):
        monitor.score(x, innovations, mode="bad")  # type: ignore[arg-type]


def test_rejects_bad_geometry_and_config() -> None:
    x, innovations = _case()
    with pytest.raises(ValueError, match="reference_rows"):
        LocalContrastConfig(reference_rows=300)
    with pytest.raises(ValueError, match="current_window"):
        LocalContrastConfig(current_window=1)
    with pytest.raises(ValueError, match="aligned"):
        local_contrast_components(x, innovations[:-1], LocalContrastConfig())
