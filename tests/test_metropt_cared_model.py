from __future__ import annotations

import json

import numpy as np
import pytest

from mate_spc_metropt.model import (
    ANALOG_COUNT,
    ENSEMBLE_SEEDS,
    FORECAST_HORIZONS,
    INPUT_FEATURE_COUNT,
    INPUT_WINDOW_MINUTES,
    MONITORING_RESIDUAL_COUNT,
    BlockConformalCalibration,
    CaredBCScorer,
    CaredEnsemble,
    RankNormalTransform,
    RobustScaler,
    align_forecasts_to_score_times,
    align_inference_forecasts_to_score_times,
    block_conformal_rank,
    build_causal_tcn,
    build_causal_input_windows,
    build_causal_windows,
    build_monitoring_residuals,
    calibrate_block_conformal,
    control_state_codes,
    daily_maxima,
    deterministic_state_balanced_indices,
    directional_glr_scores,
    dual_page_cusum,
    fit_cared_scorer,
    fit_f1_directional_glr,
    fit_state_conditional_reference,
    persistence_huber_loss,
    physical_causal_features,
    state_conditional_energy,
    train_cared_ensemble,
)


def _minute_series(count: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    analog = np.column_stack(
        [0.01 * (feature + 1) * index + np.sin(index / (feature + 2)) for feature in range(7)]
    )
    controls = np.zeros((count, 6), dtype=np.float64)
    controls[:, 0] = (index.astype(int) // 4) % 2
    controls[:, 1] = (index.astype(int) // 7) % 2
    controls[:, 2] = (index.astype(int) // 11) % 2
    controls[:, 3] = (index.astype(int) // 13) % 2
    controls[:, 4] = (index.astype(int) // 17) % 2
    controls[:, 5] = np.maximum(np.sin(index / 5.0), 0.0)
    timestamps = np.datetime64("2020-02-01T00:00") + np.arange(count).astype("timedelta64[m]")
    return analog, controls, timestamps.astype("datetime64[ns]")


def test_robust_scaler_roundtrip_and_constant_feature() -> None:
    values = np.column_stack((np.arange(20.0), np.ones(20)))
    scaler = RobustScaler.fit(values)
    transformed = scaler.transform(values)
    assert scaler.scale[1] == 1.0
    np.testing.assert_allclose(scaler.inverse_transform(transformed), values)


def test_window_construction_never_crosses_gap_day_or_ineligible_row() -> None:
    analog, controls, timestamps = _minute_series(75)
    timestamps = timestamps.copy()
    timestamps[45:] += np.timedelta64(10, "m")
    eligible = np.ones(75, dtype=bool)
    eligible[30] = False
    batch = build_causal_windows(
        analog,
        controls,
        timestamps,
        eligible=eligible,
        maximum_gap_seconds=5 * 60,
    )
    assert batch.inputs.shape[1:] == (10, 13)
    assert batch.targets.shape[1:] == (3, 7)
    for anchor, targets in zip(batch.anchor_indices, batch.target_indices):
        begin = anchor - INPUT_WINDOW_MINUTES + 1
        end = targets[-1]
        assert eligible[begin : end + 1].all()
        assert np.all(
            timestamps[begin : end + 1].astype("datetime64[D]")
            == timestamps[anchor].astype("datetime64[D]")
        )
        gaps = np.diff(timestamps[begin : end + 1].astype(np.int64)) / 1.0e9
        assert np.max(gaps) <= 5 * 60


def test_future_target_perturbation_does_not_change_existing_input_window() -> None:
    analog, controls, timestamps = _minute_series(50)
    original = build_causal_windows(analog, controls, timestamps)
    target_index = int(original.target_indices[0, -1])
    changed = analog.copy()
    changed[target_index] += 999.0
    rebuilt = build_causal_windows(changed, controls, timestamps)
    np.testing.assert_array_equal(rebuilt.inputs[0], original.inputs[0])
    assert not np.array_equal(rebuilt.targets[0], original.targets[0])


def test_online_scores_are_invariant_to_future_missingness() -> None:
    analog, controls, timestamps = _minute_series(90)
    eligible = np.ones(90, dtype=bool)
    inference = build_causal_input_windows(
        analog, controls, timestamps, eligible=eligible
    )
    predictions = np.repeat(inference.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1)
    aligned = align_inference_forecasts_to_score_times(
        inference,
        predictions,
        analog,
        controls,
        timestamps,
        eligible=eligible,
    )

    cutoff = 48
    changed_controls = controls.copy()
    changed_controls[cutoff + 1 :] = np.nan
    changed_eligible = eligible.copy()
    changed_eligible[cutoff + 1 :] = False
    changed_inference = build_causal_input_windows(
        analog,
        changed_controls,
        timestamps,
        eligible=changed_eligible,
    )
    changed_predictions = np.repeat(
        changed_inference.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1
    )
    changed_aligned = align_inference_forecasts_to_score_times(
        changed_inference,
        changed_predictions,
        analog,
        changed_controls,
        timestamps,
        eligible=changed_eligible,
    )

    original_prefix = aligned.score_indices <= cutoff
    changed_prefix = changed_aligned.score_indices <= cutoff
    np.testing.assert_array_equal(
        changed_aligned.score_indices[changed_prefix],
        aligned.score_indices[original_prefix],
    )
    np.testing.assert_array_equal(
        changed_aligned.residuals[changed_prefix],
        aligned.residuals[original_prefix],
    )
    rng = np.random.default_rng(480)
    fit = rng.normal(size=(600, MONITORING_RESIDUAL_COUNT))
    dev = rng.normal(size=(600, MONITORING_RESIDUAL_COUNT))
    f1 = rng.normal(size=(600, MONITORING_RESIDUAL_COUNT))
    f1[:, 0] += 2.0
    state_zero = np.zeros(600, dtype=np.int64)
    scorer = fit_cared_scorer(fit, state_zero, dev, state_zero, f1, state_zero).with_threshold(2.0)
    original_path = scorer.score(
        aligned.residuals,
        aligned.state_codes,
        aligned.score_timestamps,
        reset_before=aligned.reset_before,
    )
    changed_path = scorer.score(
        changed_aligned.residuals,
        changed_aligned.state_codes,
        changed_aligned.score_timestamps,
        reset_before=changed_aligned.reset_before,
    )
    np.testing.assert_array_equal(
        changed_path.envelope[changed_prefix], original_path.envelope[original_prefix]
    )
    np.testing.assert_array_equal(
        changed_path.alarm[changed_prefix], original_path.alarm[original_prefix]
    )


def test_training_and_inference_windows_never_cross_explicit_reset() -> None:
    analog, controls, timestamps = _minute_series(80)
    reset_before = np.zeros(80, dtype=bool)
    reset_before[[0, 35]] = True
    training = build_causal_windows(
        analog, controls, timestamps, reset_before=reset_before
    )
    inference = build_causal_input_windows(
        analog, controls, timestamps, reset_before=reset_before
    )

    for anchor, targets in zip(training.anchor_indices, training.target_indices):
        begin = int(anchor) - INPUT_WINDOW_MINUTES + 1
        assert not reset_before[begin + 1 : int(targets[-1]) + 1].any()
    for anchor in inference.anchor_indices:
        begin = int(anchor) - INPUT_WINDOW_MINUTES + 1
        assert not reset_before[begin + 1 : int(anchor) + 1].any()

    predictions = np.repeat(inference.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1)
    aligned = align_inference_forecasts_to_score_times(
        inference,
        predictions,
        analog,
        controls,
        timestamps,
        reset_before=reset_before,
    )
    assert not np.any((aligned.score_indices >= 35) & (aligned.score_indices < 49))
    assert aligned.reset_before[0]
    assert aligned.reset_before[np.flatnonzero(aligned.score_indices >= 49)[0]]


def test_three_forecast_horizons_align_to_one_score_timestamp_without_future_leakage() -> None:
    analog, controls, timestamps = _minute_series(60)
    windows = build_causal_windows(analog, controls, timestamps)
    predictions = np.repeat(windows.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1)
    aligned = align_forecasts_to_score_times(
        windows, predictions, analog, controls, timestamps
    )
    target = int(aligned.score_indices[0])
    for head, source_row in enumerate(aligned.source_window_rows[0]):
        assert windows.target_indices[source_row, head] == target
        expected = analog[target] - predictions[source_row, head]
        np.testing.assert_array_equal(
            aligned.prediction_residuals[0, head * 7 : (head + 1) * 7], expected
        )

    changed_analog = analog.copy()
    changed_analog[target + 1 :] += 500.0
    changed_windows = build_causal_windows(changed_analog, controls, timestamps)
    changed_predictions = np.repeat(
        changed_windows.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1
    )
    changed_aligned = align_forecasts_to_score_times(
        changed_windows, changed_predictions, changed_analog, controls, timestamps
    )
    through_target = changed_aligned.score_indices <= target
    np.testing.assert_array_equal(
        changed_aligned.residuals[through_target],
        aligned.residuals[aligned.score_indices <= target],
    )


def test_alignment_allows_unused_nan_gap_but_never_scores_through_it() -> None:
    analog, controls, timestamps = _minute_series(80)
    controls_with_gap = controls.copy()
    controls_with_gap[25] = np.nan
    eligible = np.ones(80, dtype=bool)
    eligible[25] = False
    windows = build_causal_windows(
        analog, controls_with_gap, timestamps, eligible=eligible
    )
    predictions = np.repeat(windows.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1)
    aligned = align_forecasts_to_score_times(
        windows, predictions, analog, controls_with_gap, timestamps
    )
    assert aligned.score_indices.size > 0
    assert not np.any(
        (aligned.score_indices - INPUT_WINDOW_MINUTES + 1 <= 25)
        & (aligned.score_indices >= 25)
    )

    clean_windows = build_causal_windows(analog, controls, timestamps)
    clean_predictions = np.repeat(
        clean_windows.inputs[:, -1:, :ANALOG_COUNT], 3, axis=1
    )
    clean_aligned = align_forecasts_to_score_times(
        clean_windows, clean_predictions, analog, controls, timestamps
    )
    used_target = int(clean_aligned.score_indices[0])
    used_gap = controls.copy()
    used_gap[used_target] = np.nan
    gap_aligned = align_forecasts_to_score_times(
        clean_windows, clean_predictions, analog, used_gap, timestamps
    )
    assert used_target not in gap_aligned.score_indices


def test_tcn_prefix_is_strictly_causal() -> None:
    torch = pytest.importorskip("torch")
    model = build_causal_tcn(seed=17, device="cpu", dtype="float64")
    model.eval()
    generator = torch.Generator().manual_seed(4)
    values = torch.randn(2, 10, INPUT_FEATURE_COUNT, generator=generator, dtype=torch.float64)
    perturbed = values.clone()
    perturbed[:, 7:] += 100.0
    with torch.no_grad():
        original = model.forward_sequence(values)
        changed = model.forward_sequence(perturbed)
    np.testing.assert_array_equal(original[:, :7].numpy(), changed[:, :7].numpy())
    assert not torch.equal(original[:, 7:], changed[:, 7:])
    assert tuple(model(values).shape) == (2, len(FORECAST_HORIZONS), ANALOG_COUNT)


def test_physical_features_residual_dimension_and_control_state() -> None:
    analog, controls, timestamps = _minute_series(40)
    batch = build_causal_windows(analog, controls, timestamps)
    prediction = batch.targets - 0.25
    residual = build_monitoring_residuals(prediction, batch.targets, batch.inputs)
    assert residual.shape == (batch.inputs.shape[0], MONITORING_RESIDUAL_COUNT)
    np.testing.assert_allclose(residual[:, :21], 0.25)
    physical = physical_causal_features(batch.inputs)
    np.testing.assert_array_equal(residual[:, 21:], physical)
    codes = control_state_codes(batch.inputs[:, :, ANALOG_COUNT:])
    expected_bits = batch.inputs[:, -1, ANALOG_COUNT : ANALOG_COUNT + 5] > 0.5
    expected = expected_bits @ (1 << np.arange(5))
    np.testing.assert_array_equal(codes, expected)


def test_state_shrinkage_energy_and_f1_glr_are_directional() -> None:
    rng = np.random.default_rng(12)
    healthy = rng.normal(size=(1200, 6))
    healthy_states = np.repeat([0, 3], 600)
    fault = rng.normal(size=(600, 6))
    fault[:, 0] += 3.0
    fault_states = np.repeat([0, 3], 300)
    reference = fit_state_conditional_reference(healthy, healthy_states)
    direction = fit_f1_directional_glr(
        reference, healthy, healthy_states, fault, fault_states
    )
    positive = rng.normal(size=(200, 6))
    positive[:, 0] += 2.0
    negative = rng.normal(size=(200, 6))
    negative[:, 0] -= 2.0
    states = np.zeros(200, dtype=np.int64)
    positive_glr = directional_glr_scores(positive, states, reference, direction)
    negative_glr = directional_glr_scores(negative, states, reference, direction)
    assert np.median(positive_glr) > np.median(negative_glr) + 1.0
    assert np.all(positive_glr >= 0.0)
    assert np.all(negative_glr >= 0.0)
    assert np.median(state_conditional_energy(positive, states, reference)) > 0.0
    # An unseen state is required to fall back to the global reference/direction.
    unseen = np.full(200, 31, dtype=np.int64)
    assert np.isfinite(directional_glr_scores(positive, unseen, reference, direction)).all()


def test_rare_state_pools_to_nearest_frequent_state_with_deterministic_tie() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(1300, 3))
    states = np.concatenate(
        (np.zeros(600, dtype=np.int64), np.ones(100, dtype=np.int64), np.full(600, 3))
    )
    reference = fit_state_conditional_reference(values, states)
    np.testing.assert_array_equal(reference.state_codes, [0, 3])
    np.testing.assert_array_equal(reference.counts, [700, 600])
    # State 1 is one bit from both 0 and 3; fixed-width lexical order selects 0.
    assert reference.row_for_state(1) == reference.row_for_state(0)


def test_state_reference_fails_closed_without_a_500_sample_state() -> None:
    rng = np.random.default_rng(171)
    values = rng.normal(size=(800, 4))
    states = np.repeat([0, 3], 400)
    with pytest.raises(ValueError, match="no control state reaches"):
        fit_state_conditional_reference(values, states)


def test_hamming_tie_uses_five_control_bit_tuple_lexical_order() -> None:
    rng = np.random.default_rng(172)
    values = rng.normal(size=(1300, 3))
    states = np.concatenate(
        (
            np.ones(600, dtype=np.int64),
            np.full(600, 2, dtype=np.int64),
            np.zeros(100, dtype=np.int64),
        )
    )
    reference = fit_state_conditional_reference(values, states)
    # State 0 ties at distance one from state 1=(1,0,0,0,0) and
    # state 2=(0,1,0,0,0).  Lexical order in control-column bit order picks 2,
    # unlike an implicit numeric-code tie break.
    destination = reference.state_codes[reference.row_for_state(0)]
    assert destination == 2
    np.testing.assert_array_equal(reference.counts, [600, 700])


def test_rank_normal_cusum_daily_reset_and_block_conformal() -> None:
    reference = np.column_stack((np.arange(1.0, 101.0), np.arange(101.0, 201.0)))
    ranker = RankNormalTransform.fit(reference)
    transformed = ranker.transform(np.asarray([[50.0, 150.0], [200.0, 300.0]]))
    assert abs(transformed[0, 0]) < 0.1
    assert transformed[1, 0] > transformed[0, 0]

    timestamps = np.asarray(
        [
            "2020-02-01T23:58",
            "2020-02-01T23:59",
            "2020-02-02T00:00",
            "2020-02-02T00:10",
        ],
        dtype="datetime64[ns]",
    )
    page = dual_page_cusum(np.ones((4, 2)), timestamps, maximum_gap_seconds=300)
    np.testing.assert_array_equal(page.reset, [True, False, True, True])
    np.testing.assert_allclose(page.cusum[:, 0], [0.5, 1.0, 0.5, 0.5])
    explicit = dual_page_cusum(
        np.ones((4, 2)),
        timestamps,
        maximum_gap_seconds=1000,
        reset_before=np.asarray([False, True, False, False]),
    )
    np.testing.assert_array_equal(explicit.reset, [True, True, True, False])
    np.testing.assert_allclose(explicit.cusum[:, 0], [0.5, 0.5, 0.5, 1.0])
    blocks = daily_maxima(page, timestamps)
    np.testing.assert_array_equal(blocks.dates, np.asarray(["2020-02-01", "2020-02-02"], dtype="datetime64[D]"))
    np.testing.assert_allclose(blocks.maxima, [1.0, 0.5])

    maxima = np.arange(1.0, 71.0)
    assert block_conformal_rank(70, alpha=0.05) == 68
    calibration = calibrate_block_conformal(maxima, alpha=0.05)
    assert isinstance(calibration, BlockConformalCalibration)
    assert calibration.rank_one_based == 68
    assert calibration.threshold == 68.0
    assert calibration.upper_alarm_probability_bound == pytest.approx(3.0 / 71.0)
    np.testing.assert_array_equal(calibration.alarm([68.0, 69.0]), [False, True])


def test_rank_normal_midrank_keeps_zero_atom_and_page_cusum_at_zero() -> None:
    reference = np.zeros((100, 2), dtype=np.float64)
    ranker = RankNormalTransform.fit(reference)
    evidence = ranker.transform(np.zeros((50, 2), dtype=np.float64))
    np.testing.assert_array_equal(evidence, np.zeros_like(evidence))
    timestamps = (
        np.datetime64("2020-02-01T00:00")
        + np.arange(50).astype("timedelta64[m]")
    ).astype("datetime64[ns]")
    page = dual_page_cusum(evidence, timestamps)
    np.testing.assert_array_equal(page.cusum, np.zeros_like(evidence))
    np.testing.assert_array_equal(page.envelope, np.zeros(50))


def test_statistical_scorer_and_pickle_free_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(33)
    fit = rng.normal(size=(1200, MONITORING_RESIDUAL_COUNT))
    fit_states = np.repeat([0, 1], 600)
    healthy = rng.normal(size=(700, MONITORING_RESIDUAL_COUNT))
    healthy_states = np.repeat([0, 1], 350)
    f1 = rng.normal(size=(550, MONITORING_RESIDUAL_COUNT))
    f1[:, 2] += 2.5
    f1_states = np.resize(np.asarray([0, 1]), 550)
    scorer = fit_cared_scorer(
        fit, fit_states, healthy, healthy_states, f1, f1_states
    )
    sample = rng.normal(size=(20, MONITORING_RESIDUAL_COUNT))
    sample[:, 2] += 1.5
    states = np.zeros(20, dtype=np.int64)
    timestamps = np.datetime64("2020-03-01T00:00") + np.arange(20).astype("timedelta64[m]")
    expected = scorer.score(sample, states, timestamps)
    path = tmp_path / "scorer.npz"
    scorer.save_npz(path, metadata={"role": "synthetic"})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata_json"].dtype.kind in "SU"
    restored, metadata = CaredBCScorer.load_npz(path)
    actual = restored.score(sample, states, timestamps)
    assert metadata == {"role": "synthetic"}
    np.testing.assert_array_equal(actual.energy, expected.energy)
    np.testing.assert_array_equal(actual.directional_glr, expected.directional_glr)
    np.testing.assert_array_equal(actual.cusum, expected.cusum)

    with np.load(path, allow_pickle=False) as archive:
        original_payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    original_document = json.loads(str(original_payload["metadata_json"].item()))
    corruptions = {
        "feature_count": MONITORING_RESIDUAL_COUNT - 1,
        "kappa": 0.6,
        "maximum_gap_seconds": 61.0,
        "reference_shrinkage_count": 499,
        "direction_shrinkage_count": 499,
        "rank_lower_clip": -3.0,
        "rank_upper_clip": 7.0,
    }
    for field, bad_value in corruptions.items():
        payload = {name: np.array(value, copy=True) for name, value in original_payload.items()}
        document = dict(original_document)
        document[field] = bad_value
        payload["metadata_json"] = np.asarray(
            json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
        np.savez_compressed(path, **payload)
        with pytest.raises(ValueError, match="frozen protocol"):
            CaredBCScorer.load_npz(path)

    mismatched = {name: np.array(value, copy=True) for name, value in original_payload.items()}
    mismatched["direction_state_codes"] = mismatched["direction_state_codes"][::-1]
    np.savez_compressed(path, **mismatched)
    with pytest.raises(ValueError, match="state codes"):
        CaredBCScorer.load_npz(path)


def test_fit_dev_and_f1_roles_are_separated() -> None:
    rng = np.random.default_rng(121)
    states = np.zeros(700, dtype=np.int64)
    fit = rng.normal(size=(700, 4))
    dev = rng.normal(size=(700, 4))
    f1 = rng.normal(size=(700, 4))
    f1[:, 0] += 2.0
    original = fit_cared_scorer(fit, states, dev, states, f1, states)

    changed_fit = fit.copy()
    changed_fit[:, 1] += 3.0
    fit_changed = fit_cared_scorer(changed_fit, states, dev, states, f1, states)
    assert not np.array_equal(
        fit_changed.reference.global_center, original.reference.global_center
    )

    changed_dev = dev.copy()
    changed_dev[:, 2] += 1.5
    dev_changed = fit_cared_scorer(fit, states, changed_dev, states, f1, states)
    np.testing.assert_array_equal(
        dev_changed.reference.global_center, original.reference.global_center
    )
    np.testing.assert_array_equal(
        dev_changed.reference.global_covariance, original.reference.global_covariance
    )
    assert not np.array_equal(
        dev_changed.direction.global_direction, original.direction.global_direction
    )
    assert not np.array_equal(
        dev_changed.rank_normalizer.sorted_reference,
        original.rank_normalizer.sorted_reference,
    )


def test_ensemble_npz_roundtrip_and_cpu_prediction(tmp_path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    input_scaler = RobustScaler.fit(rng.normal(size=(40, 10, 13)))
    target_scaler = RobustScaler.fit(rng.normal(size=(40, 3, 7)))
    ensemble = CaredEnsemble.from_untrained(input_scaler, target_scaler, dtype="float64")
    inputs = rng.normal(size=(3, 10, 13))
    expected = ensemble.predict(inputs, device="cpu", batch_size=2)
    path = tmp_path / "ensemble.npz"
    ensemble.save_npz(path, metadata={"seeds": list(ENSEMBLE_SEEDS)})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["metadata_json"].dtype.kind in "SU"
    restored, metadata = CaredEnsemble.load_npz(path)
    actual = restored.predict(inputs, device="cpu", batch_size=2)
    assert restored.seeds == ENSEMBLE_SEEDS
    assert metadata == {"seeds": list(ENSEMBLE_SEEDS)}
    np.testing.assert_array_equal(actual, expected)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    document = json.loads(str(payload["metadata_json"].item()))
    document["seeds"] = [17, 29, 99]
    payload["metadata_json"] = np.asarray(
        json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="seeds violate"):
        CaredEnsemble.load_npz(path)


def test_training_api_and_persistence_baseline_on_small_synthetic_data() -> None:
    pytest.importorskip("torch")
    analog, controls, timestamps = _minute_series(80)
    batch = build_causal_windows(analog, controls, timestamps)
    training = batch.subset(np.arange(45))
    validation = batch.subset(np.arange(45, batch.inputs.shape[0]))
    balanced = deterministic_state_balanced_indices(training.inputs, seed=17)
    balanced_codes = control_state_codes(
        training.inputs[balanced, :, ANALOG_COUNT:]
    )
    _, balanced_counts = np.unique(balanced_codes, return_counts=True)
    assert np.all(balanced_counts == balanced_counts[0])
    result = train_cared_ensemble(
        training,
        validation,
        device="cpu",
        dtype="float32",
        seeds=(17,),
        max_epochs=1,
        patience=1,
        batch_size=32,
    )
    assert len(result.histories) == 1
    assert len(result.histories[0].training_loss) == 1
    assert np.isfinite(result.ensemble_validation_huber_loss)
    assert result.persistence_validation_huber_loss == pytest.approx(
        persistence_huber_loss(validation, result.ensemble.target_scaler)
    )


def test_cpu_cuda_prediction_parity_if_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(91)
    input_scaler = RobustScaler.fit(rng.normal(size=(30, 10, 13)))
    target_scaler = RobustScaler.fit(rng.normal(size=(30, 3, 7)))
    ensemble = CaredEnsemble.from_untrained(input_scaler, target_scaler, dtype="float64")
    inputs = rng.normal(size=(5, 10, 13))
    cpu = ensemble.predict(inputs, device="cpu")
    cuda = ensemble.predict(inputs, device="cuda")
    np.testing.assert_allclose(cuda, cpu, rtol=0.0, atol=1.0e-10)
