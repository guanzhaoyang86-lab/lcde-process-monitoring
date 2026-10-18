from __future__ import annotations

# Keep the supported Windows mixed-runtime import order stable during collection.
import torch  # noqa: F401
import numpy as np
import pytest

from mate_spc_cf_glr.features import (
    BASELINE_STOP,
    BROAD_PROCESS_CHANNELS,
    BROAD_PROCESS_WINDOWS,
    BROAD_STATISTIC_ORDER,
    BROAD_TOP_K,
    D_FEED_XMEAS_INDEX,
    EXPERT_FEATURE_COUNTS,
    F15_FEATURE_NAMES,
    F15_LAGS,
    F15_STATISTIC_ORDER,
    F15_WINDOWS,
    F3_FEATURE_NAMES,
    F9_FEATURE_NAMES,
    FEATURE_METADATA,
    FEATURE_SCHEMA,
    MONITOR_START,
    PREFIX_ENDPOINTS,
    ExpertFeaturePaths,
    expert_feature_names,
    extract_expert_feature_paths,
)


def _run(seed: int = 31) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(500, 52)).astype(np.float64)
    values[:, 1] += np.linspace(0.0, 1.0, 500, dtype=np.float64)
    values[:, 51] += 0.25 * np.sin(np.arange(500, dtype=np.float64) / 7.0)
    return values


def test_frozen_geometry_names_and_channel_provenance() -> None:
    assert FEATURE_SCHEMA == "cf_glr_mechanism_features_v2"
    assert D_FEED_XMEAS_INDEX == 1
    assert BROAD_PROCESS_WINDOWS == (8, 16, 32, 64)
    assert BROAD_PROCESS_CHANNELS == tuple(range(52))
    assert BROAD_TOP_K == 5
    assert F15_WINDOWS == (16, 32, 64, 128)
    assert F15_LAGS == (1, 2, 4, 8)
    assert BROAD_STATISTIC_ORDER == (
        "top5mean_abs_robust_location",
        "top5mean_robust_variance",
        "top5mean_delta_energy",
        "top5mean_abs_lag1_autocovariance",
        "top5mean_haar_energy",
    )
    assert F15_STATISTIC_ORDER == (
        "dwell_ratio",
        "slip_impulse_after_dwell4",
        "normalized_loop_area_u_y1",
        "normalized_loop_area_u_y2",
        "maxlag_corr_asymmetry_u_y1",
        "maxlag_corr_asymmetry_u_y2",
        "robust_ARX_residual_energy_proxy",
    )
    assert EXPERT_FEATURE_COUNTS == {3: 20, 9: 20, 15: 28}
    assert len(F3_FEATURE_NAMES) == len(set(F3_FEATURE_NAMES)) == 20
    assert len(F9_FEATURE_NAMES) == len(set(F9_FEATURE_NAMES)) == 20
    assert len(F15_FEATURE_NAMES) == len(set(F15_FEATURE_NAMES)) == 28
    assert F3_FEATURE_NAMES[:5] == tuple(
        f"fault3.{name}[window=8]" for name in BROAD_STATISTIC_ORDER
    )
    assert F15_FEATURE_NAMES[:7] == tuple(
        f"fault15.{name}[window=16]" for name in F15_STATISTIC_ORDER
    )
    assert F15_FEATURE_NAMES[-1] == (
        "fault15.robust_ARX_residual_energy_proxy[window=128]"
    )
    assert expert_feature_names(15) is F15_FEATURE_NAMES
    assert "unmeasured" in FEATURE_METADATA["disturbance_observability"]
    assert FEATURE_METADATA["broad_process_channel_indices"] == tuple(range(52))
    assert FEATURE_METADATA["broad_process_reduction"] == (
        "mean_of_five_largest_channel_magnitudes"
    )
    assert PREFIX_ENDPOINTS == tuple(range(128, 160, 4))


def test_paths_are_exact_float64_read_only_and_prefix_is_populated() -> None:
    paths = extract_expert_feature_paths(_run())
    assert isinstance(paths, ExpertFeaturePaths)
    assert paths.fault3.shape == (500, 20)
    assert paths.fault9.shape == (500, 20)
    assert paths.fault15.shape == (500, 28)
    for fault_id in (3, 9, 15):
        matrix = paths.for_fault(fault_id)
        assert matrix.dtype == np.dtype(np.float64)
        assert matrix.flags.c_contiguous
        assert not matrix.flags.writeable
        assert np.isfinite(matrix).all()
        np.testing.assert_array_equal(matrix[:BASELINE_STOP], 0.0)
        assert np.any(matrix[BASELINE_STOP:MONITOR_START] != 0.0)
        assert paths.by_fault[fault_id] is matrix
    np.testing.assert_array_equal(paths.fault3, paths.fault9)


def test_broad_experts_retain_downstream_channels_beyond_xmeas2() -> None:
    original = _run(42)
    downstream_changed = original.copy()
    downstream_changed[180:, 30] += 100.0
    left = extract_expert_feature_paths(original)
    right = extract_expert_feature_paths(downstream_changed)
    np.testing.assert_array_equal(left.fault3[:180], right.fault3[:180])
    assert np.any(left.fault3[180:] != right.fault3[180:])
    np.testing.assert_array_equal(left.fault9, left.fault3)
    np.testing.assert_array_equal(right.fault9, right.fault3)


def test_every_emitted_endpoint_is_strictly_causal() -> None:
    original = _run(55)
    prefix_changed = original.copy()
    prefix_changed[129:] += 11.0
    changed = original.copy()
    changed[251:] += np.linspace(1.0, 80.0, 249, dtype=np.float64)[:, None]
    left = extract_expert_feature_paths(original)
    prefix_right = extract_expert_feature_paths(prefix_changed)
    right = extract_expert_feature_paths(changed)
    for fault_id in (3, 9, 15):
        np.testing.assert_array_equal(
            left.for_fault(fault_id)[:129], prefix_right.for_fault(fault_id)[:129]
        )
        assert np.any(
            left.for_fault(fault_id)[129:MONITOR_START]
            != prefix_right.for_fault(fault_id)[129:MONITOR_START]
        )
        np.testing.assert_array_equal(
            left.for_fault(fault_id)[:251], right.for_fault(fault_id)[:251]
        )
        assert np.any(
            left.for_fault(fault_id)[251:] != right.for_fault(fault_id)[251:]
        )


def test_constant_run_is_finite_and_has_exact_dwell_semantics() -> None:
    paths = extract_expert_feature_paths(np.ones((500, 52), dtype=np.float64))
    for matrix in paths.by_fault.values():
        assert np.isfinite(matrix).all()
    for width in F15_WINDOWS:
        dwell_column = F15_FEATURE_NAMES.index(f"fault15.dwell_ratio[window={width}]")
        slip_column = F15_FEATURE_NAMES.index(
            f"fault15.slip_impulse_after_dwell4[window={width}]"
        )
        assert paths.fault15[MONITOR_START, dwell_column] == 1.0
        assert paths.fault15[MONITOR_START, slip_column] == 0.0


def test_release_after_four_dwell_steps_produces_a_positive_impulse() -> None:
    run = np.ones((500, 52), dtype=np.float64)
    run[150:, 51] = 2.0
    paths = extract_expert_feature_paths(run)
    for width in F15_WINDOWS:
        column = F15_FEATURE_NAMES.index(
            f"fault15.slip_impulse_after_dwell4[window={width}]"
        )
        assert paths.fault15[MONITOR_START, column] > 0.0


def test_invalid_inputs_and_fault_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="exact float64"):
        extract_expert_feature_paths(np.zeros((500, 52), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        extract_expert_feature_paths(np.zeros((499, 52), dtype=np.float64))
    invalid = np.zeros((500, 52), dtype=np.float64)
    invalid[250, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_expert_feature_paths(invalid)

    paths = extract_expert_feature_paths(np.zeros((500, 52), dtype=np.float64))
    for invalid_id in (0, 1, True, 3.0):
        with pytest.raises(ValueError, match="3, 9, or 15"):
            paths.for_fault(invalid_id)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="3, 9, or 15"):
            expert_feature_names(invalid_id)  # type: ignore[arg-type]
