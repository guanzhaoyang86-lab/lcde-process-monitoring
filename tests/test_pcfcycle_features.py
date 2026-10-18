from __future__ import annotations

# The supported Windows CUDA environment requires Torch to own DLL load order
# when these tests are collected alongside neural PCF-Cycle tests.
import torch  # noqa: F401
import numpy as np
import pytest

from mate_spc_pcfcycle.features import (
    A_SHAPE,
    BFeatureStandardizer,
    B_FIT_ROW_COUNT,
    B_FEATURE_COUNT,
    CHANNEL_FEATURE_COUNT,
    CHANNEL_SUMMARY_NAMES,
    CANDIDATE_NAME,
    FIT_HEALTHY_RUN_COUNT,
    HEALTH_FEATURE_NAMES,
    HealthyPrefixNormalizer,
    LOOP_FEATURE_NAMES,
    MONITOR_ENDPOINTS,
    NORMALIZATION_CLIP,
    NORMALIZATION_FLOOR,
    PCFCycleFeatureExtractor,
    REFERENCE_STOP,
    RFFT_BANDS,
    _welch_coherence,
    extract_pcfcycle_features,
    load_pcfcycle_feature_state,
    loop_summaries,
    paired_channel_summaries,
    save_pcfcycle_feature_state,
)


def _healthy_runs(seed: int = 410) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    runs: dict[str, np.ndarray] = {}
    for index in range(FIT_HEALTHY_RUN_COUNT):
        innovations = rng.normal(scale=0.22, size=(500, 52))
        values = np.empty_like(innovations, dtype=np.float64)
        values[0] = innovations[0]
        for row in range(1, 500):
            values[row] = 0.32 * values[row - 1] + innovations[row]
        values[:, 10] += 0.35 * np.roll(values[:, 51], 1)
        values[:, 21] -= 0.20 * np.roll(values[:, 51], 2)
        runs[f"healthy-{index:02d}"] = values.astype(np.float64, copy=False)
    return runs


@pytest.fixture(scope="module")
def fit_bundle() -> tuple[dict[str, np.ndarray], PCFCycleFeatureExtractor]:
    runs = _healthy_runs()
    return runs, PCFCycleFeatureExtractor.fit(runs)


@pytest.fixture(scope="module")
def b_fit_bundle() -> tuple[np.ndarray, BFeatureStandardizer]:
    rng = np.random.default_rng(778)
    rows = rng.normal(size=(B_FIT_ROW_COUNT, B_FEATURE_COUNT)).astype(np.float64)
    rows[:, 0] = 3.25
    return rows, BFeatureStandardizer.fit(rows)


def test_frozen_feature_geometry_and_exact_loop_order() -> None:
    assert CANDIDATE_NAME == "pcf-cyclerank-scan-v6-fixed"
    assert A_SHAPE == (52, 128)
    assert len(CHANNEL_SUMMARY_NAMES) == 11
    assert CHANNEL_FEATURE_COUNT == 572
    assert len(LOOP_FEATURE_NAMES) == 24
    assert len(HEALTH_FEATURE_NAMES) == 8
    assert B_FEATURE_COUNT == 604
    np.testing.assert_array_equal(MONITOR_ENDPOINTS, np.arange(160, 497, 4))
    assert RFFT_BANDS == ((1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 65))
    order = (
        "current_dwell",
        "delta_dwell",
        "current_stick_slip",
        "delta_stick_slip",
        "current_reversal",
        "delta_reversal",
        "current_oriented_area",
        "delta_oriented_area",
        "delta_lag_asymmetry",
        "log_arx_residual_ratio",
        "delta_welch_coherence",
        "delta_actuator_psd_concentration",
    )
    assert LOOP_FEATURE_NAMES == tuple(
        f"{pair}.{metric}"
        for pair in ("xmv11_xmeas11", "xmv11_xmeas22")
        for metric in order
    )


def test_normalizer_is_exact_pooled_prefix_mad_and_suffix_sealed() -> None:
    runs = _healthy_runs(511)
    fitted = HealthyPrefixNormalizer.fit(runs)
    pooled = np.concatenate([runs[name][:REFERENCE_STOP] for name in sorted(runs)])
    expected_center = np.median(pooled, axis=0)
    expected_scale = np.maximum(
        1.4826 * np.median(np.abs(pooled - expected_center), axis=0),
        NORMALIZATION_FLOOR,
    )
    np.testing.assert_array_equal(fitted.center, expected_center)
    np.testing.assert_array_equal(fitted.scale, expected_scale)

    # A nonfinite synthetic suffix proves fitting neither validates nor consumes it.
    sealed = {name: value.copy() for name, value in runs.items()}
    for value in sealed.values():
        value[REFERENCE_STOP:] = np.nan
    sealed_fit = HealthyPrefixNormalizer.fit(sealed)
    np.testing.assert_array_equal(sealed_fit.center, fitted.center)
    np.testing.assert_array_equal(sealed_fit.scale, fitted.scale)

    transformed = fitted.transform_run(runs["healthy-00"])
    assert transformed.dtype == np.dtype(np.float64)
    assert not transformed.flags.writeable
    assert np.max(transformed) <= NORMALIZATION_CLIP
    assert np.min(transformed) >= -NORMALIZATION_CLIP
    with pytest.raises(ValueError, match="exactly 60"):
        HealthyPrefixNormalizer.fit(list(runs.values())[:-1])
    with pytest.raises(ValueError, match="one of"):
        HealthyPrefixNormalizer.fit(runs, expected_run_count=80)


def test_final_refit_is_explicitly_restricted_to_100_healthy_runs() -> None:
    runs = list(_healthy_runs(612).values())
    final_runs = runs + runs[:40]
    final = PCFCycleFeatureExtractor.fit(final_runs, expected_run_count=100)
    assert final.normalizer.fit_run_count == 100
    assert final.health_model.fit_run_count == 100
    assert final.metadata["fit_access"] == "100_fit_healthy_prefixes_0_128_only"


def test_channel_and_loop_summaries_are_reference_relative_and_repeatable() -> None:
    rng = np.random.default_rng(91)
    reference = rng.normal(size=(52, 128)).astype(np.float64)
    current = np.stack((reference, reference + 0.05), axis=0)
    channel = paired_channel_summaries(current, reference)
    loops = loop_summaries(current, reference)
    assert channel.shape == (2, 52, 11)
    assert loops.shape == (2, 24)
    np.testing.assert_allclose(channel[0], 0.0, rtol=0.0, atol=0.0)
    # Raw-current fields are 0,2,4,6 in each loop block; all other fields are
    # reference-relative and vanish for an identical current/reference pair.
    relative = [
        index
        for block in (0, 12)
        for index in range(block, block + 12)
        if index - block not in (0, 2, 4, 6)
    ]
    np.testing.assert_allclose(loops[0, relative], 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_array_equal(channel, paired_channel_summaries(current, reference))
    np.testing.assert_array_equal(loops, loop_summaries(current, reference))
    assert np.isfinite(channel).all() and np.isfinite(loops).all()


def test_welch_coherence_uses_the_exact_periodic_hann() -> None:
    time = np.arange(128, dtype=np.float64)
    actuator = np.stack((np.sin(time / 5.0), np.cos(time / 9.0)))
    response = np.stack((np.sin((time - 2.0) / 5.0), np.cos((time + 3.0) / 9.0)))
    actual = _welch_coherence(actuator, response)

    length = 32
    taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)
    starts = range(0, 128 - length + 1, 16)
    left = np.stack([actuator[:, start : start + length] for start in starts], axis=1)
    right = np.stack([response[:, start : start + length] for start in starts], axis=1)
    left = (left - left.mean(axis=2, keepdims=True)) * taper
    right = (right - right.mean(axis=2, keepdims=True)) * taper
    left_fft = np.fft.rfft(left, axis=2)[..., 1:]
    right_fft = np.fft.rfft(right, axis=2)[..., 1:]
    pxx = np.mean(np.abs(left_fft) ** 2, axis=1)
    pyy = np.mean(np.abs(right_fft) ** 2, axis=1)
    pxy = np.mean(left_fft * np.conjugate(right_fft), axis=1)
    expected = np.mean(
        np.clip(np.abs(pxy) ** 2 / np.maximum(pxx * pyy, 1e-12), 0.0, 1.0),
        axis=1,
    )
    np.testing.assert_array_equal(actual, expected)


def test_fitted_health_model_is_finite_readonly_and_prefix_fitted(
    fit_bundle: tuple[dict[str, np.ndarray], PCFCycleFeatureExtractor],
) -> None:
    runs, extractor = fit_bundle
    model = extractor.health_model
    for value in vars(model).values():
        if isinstance(value, np.ndarray):
            assert value.dtype == np.dtype(np.float64)
            assert np.isfinite(value).all()
            assert not value.flags.writeable
    normalized = extractor.normalizer.transform_run(runs["healthy-00"])
    stats = model.statistics(normalized[:128])
    assert stats.shape == (8,)
    assert stats.dtype == np.dtype(np.float64)
    assert np.isfinite(stats).all()
    assert np.all(stats >= -1e-10)

    endpoints = np.array([160, 204, 248], dtype=np.int64)
    windows = np.stack(
        [normalized[value - 127 : value + 1] for value in endpoints], axis=0
    )
    brute = model.statistics_many(windows)
    path = model.statistics_path(normalized, endpoints)
    np.testing.assert_allclose(path, brute, rtol=0.0, atol=1.0e-9)


def test_full_feature_extraction_has_exact_layout_and_is_causal(
    fit_bundle: tuple[dict[str, np.ndarray], PCFCycleFeatureExtractor],
) -> None:
    runs, extractor = fit_bundle
    run = runs["healthy-01"].copy()
    endpoints = np.array([160, 204, 248], dtype=np.int64)
    features = extract_pcfcycle_features(extractor, run, endpoints=endpoints)
    assert features.a.shape == (3, 52, 128)
    assert features.b.shape == (3, 604)
    assert features.a.dtype == features.b.dtype == np.dtype(np.float64)
    assert not features.a.flags.writeable and not features.b.flags.writeable
    assert np.isfinite(features.a).all() and np.isfinite(features.b).all()

    normalized = extractor.normalizer.transform_run(run)
    np.testing.assert_array_equal(
        features.a[-1], normalized[121:249].T - normalized[:128].T
    )
    repeated = extractor.extract_run(run, endpoints=endpoints)
    np.testing.assert_array_equal(repeated.a, features.a)
    np.testing.assert_array_equal(repeated.b, features.b)

    changed = run.copy()
    changed[249:] += 1000.0
    causal = extractor.extract_run(changed, endpoints=endpoints)
    np.testing.assert_array_equal(causal.a, features.a)
    np.testing.assert_array_equal(causal.b, features.b)
    assert extractor.metadata["fit_access"] == "60_fit_healthy_prefixes_0_128_only"
    assert extractor.metadata["b_blocks"] == (572, 24, 8)


def test_entrypoints_reject_wrong_dtype_and_noncausal_endpoint(
    fit_bundle: tuple[dict[str, np.ndarray], PCFCycleFeatureExtractor],
) -> None:
    runs, extractor = fit_bundle
    with pytest.raises(ValueError, match="exact float64"):
        extractor.extract_run(runs["healthy-00"].astype(np.float32))
    with pytest.raises(ValueError, match="complete causal"):
        extractor.extract_run(
            runs["healthy-00"], endpoints=np.array([126], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="exact int64"):
        extractor.extract_run(
            runs["healthy-00"], endpoints=np.array([160], dtype=np.int32)
        )


def test_mandatory_b_standardizer_has_exact_fit_population_and_formula(
    b_fit_bundle: tuple[np.ndarray, BFeatureStandardizer],
) -> None:
    rows, fitted = b_fit_bundle
    assert fitted.center[0] == 3.25
    assert fitted.scale[0] == NORMALIZATION_FLOOR
    assert fitted.center[17] == np.median(rows[:, 17])
    assert fitted.scale[17] == max(
        1.4826 * np.median(np.abs(rows[:, 17] - fitted.center[17])),
        NORMALIZATION_FLOOR,
    )
    transformed = fitted.transform(rows[:3])
    assert transformed.shape == (3, 604)
    assert transformed.dtype == np.dtype(np.float64)
    assert not transformed.flags.writeable
    assert np.max(np.abs(transformed)) <= NORMALIZATION_CLIP
    np.testing.assert_array_equal(
        fitted.inverse_transform(np.zeros(604, dtype=np.float64)), fitted.center
    )
    assert fitted.metadata["fit_rows"] == 5100
    assert fitted.metadata["fit_run_count"] == 60
    assert fitted.metadata["mandatory_before_neural_stage"] is True
    with pytest.raises(ValueError, match="5100"):
        BFeatureStandardizer.fit(rows[:-1])

    final_rows = np.broadcast_to(
        np.arange(604, dtype=np.float64), (100 * MONITOR_ENDPOINTS.size, 604)
    )
    final = BFeatureStandardizer.fit(final_rows, expected_run_count=100)
    assert final.fit_run_count == 100
    assert final.metadata["fit_rows"] == 8500


def test_safe_feature_state_npz_roundtrip(
    tmp_path,
    fit_bundle: tuple[dict[str, np.ndarray], PCFCycleFeatureExtractor],
    b_fit_bundle: tuple[np.ndarray, BFeatureStandardizer],
) -> None:
    runs, extractor = fit_bundle
    _, standardizer = b_fit_bundle
    target = save_pcfcycle_feature_state(
        tmp_path / "pcfcycle-features.npz",
        extractor,
        standardizer,
        metadata={"stage": "synthetic_test", "fold": 2},
    )
    with np.load(target, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    restored, restored_b, metadata = load_pcfcycle_feature_state(target)
    assert restored_b is not None
    assert metadata == {"fold": 2, "stage": "synthetic_test"}
    assert restored.normalizer.fit_run_count == 60
    assert restored.health_model.fit_run_count == 60
    assert restored_b.fit_run_count == 60
    np.testing.assert_array_equal(restored.normalizer.center, extractor.normalizer.center)
    np.testing.assert_array_equal(
        restored.health_model.var_coefficients,
        extractor.health_model.var_coefficients,
    )
    np.testing.assert_array_equal(restored_b.center, standardizer.center)
    endpoints = np.array([160, 248], dtype=np.int64)
    before = extractor.extract_run(runs["healthy-02"], endpoints=endpoints)
    after = restored.extract_run(runs["healthy-02"], endpoints=endpoints)
    np.testing.assert_array_equal(after.a, before.a)
    np.testing.assert_allclose(after.b, before.b, rtol=0.0, atol=1.0e-9)
