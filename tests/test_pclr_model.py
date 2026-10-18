from __future__ import annotations

import numpy as np
import pytest

import mate_spc_pclr.model as pclr_model
from mate_spc_pclr import (
    BASELINE_STOP,
    CAUSAL_LAGS,
    CONTROL_PAIRS,
    FAULT_IDS,
    MONITOR_START,
    MONITOR_STOP,
    PCLRConfig,
    PCLRFeatureModel,
    PCLRHeads,
    PCLRModel,
    PCLR_FEATURE_COUNT,
    PERSISTENCE,
    assert_numpy_torch_logit_parity,
    diagnostics_by_run,
    first_episode_detection,
    fit_pclr_heads_cuda,
    load_pclr_model,
    page_cusum_numpy,
    run_score_diagnostics,
    sequence_group_folds,
    sequence_group_splits,
)


def _synthetic_run(seed: int, *, location: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.normal(scale=0.15, size=(500, 52))
    run = np.empty_like(innovations)
    run[0] = rng.normal(scale=0.1, size=52) + location
    for time in range(1, 500):
        run[time] = 0.72 * run[time - 1] + innovations[time] + 0.28 * location
    # Add gentle controller/response coupling without using future samples.
    for controller, response in CONTROL_PAIRS:
        run[1:, response] += 0.08 * (run[:-1, controller] - location)
    return run


@pytest.fixture(scope="module")
def feature_model() -> PCLRFeatureModel:
    runs = {
        f"healthy_{index}": _synthetic_run(100 + index, location=index * 0.2)
        for index in range(4)
    }
    return PCLRFeatureModel.fit(runs)


def _dummy_heads(feature_count: int, *, threshold: float | None = 5.0) -> PCLRHeads:
    rng = np.random.default_rng(812)
    return PCLRHeads(
        weights=rng.normal(scale=0.02, size=(3, feature_count)),
        bias=np.array([-0.1, 0.2, -0.3]),
        healthy_logit_center=np.array([-0.2, 0.1, -0.4]),
        healthy_logit_scale=np.array([0.8, 1.2, 0.6]),
        kappa=np.array([0.0, 0.25, 0.5]),
        threshold=threshold,
    )


def test_frozen_contract_and_feature_shape(feature_model: PCLRFeatureModel) -> None:
    assert CAUSAL_LAGS == (1, 2, 4, 8, 16, 32)
    assert CONTROL_PAIRS == (
        (41, 1),
        (42, 2),
        (43, 0),
        (44, 3),
        (45, 4),
        (46, 9),
        (47, 13),
        (48, 16),
        (49, 18),
        (50, 20),
        (51, 21),
    )
    assert np.all(feature_model.mad_floor > 0.0)
    path = feature_model.transform_run_detailed(_synthetic_run(23))
    assert path.values.shape == (MONITOR_STOP - MONITOR_START, PCLR_FEATURE_COUNT)
    assert np.array_equal(path.local_indices, np.arange(160, 500))
    assert path.prefix_values.shape == (32, PCLR_FEATURE_COUNT)
    assert np.array_equal(path.prefix_local_indices, np.arange(128, 160))
    assert np.isnan(path.residuals[: max(CAUSAL_LAGS)]).all()
    assert np.isfinite(path.residuals[max(CAUSAL_LAGS) :]).all()


def test_feature_path_is_causal_after_fixed_baseline(feature_model: PCLRFeatureModel) -> None:
    original = _synthetic_run(31)
    changed = original.copy()
    changed[250:] += np.linspace(10.0, 100.0, 250)[:, None]
    left = feature_model.transform_run_detailed(original)
    right = feature_model.transform_run_detailed(changed)
    np.testing.assert_array_equal(left.prefix_values, right.prefix_values)
    before_change = left.local_indices < 250
    np.testing.assert_array_equal(left.values[before_change], right.values[before_change])
    assert np.max(np.abs(left.values[~before_change] - right.values[~before_change])) > 1.0


def test_asinh_and_raw_logits_have_no_positive_hard_limit(
    feature_model: PCLRFeatureModel,
) -> None:
    run = _synthetic_run(44)
    run[200:, 0] = 1.0e200
    normalized, _, _ = feature_model.normalize_run(run)
    assert normalized[200, 0] > 100.0

    heads = PCLRHeads(
        weights=np.ones((3, 2)),
        bias=np.zeros(3),
        healthy_logit_center=np.zeros(3),
        healthy_logit_scale=np.ones(3),
        kappa=np.zeros(3),
    )
    low = heads.logits_numpy(np.array([[1.0e4, 1.0e4]]))
    high = heads.logits_numpy(np.array([[1.0e12, 1.0e12]]))
    assert np.all(high > low * 1.0e7)


def test_page_cusum_has_no_ceiling_and_uses_three_point_persistence() -> None:
    heads = PCLRHeads(
        weights=np.ones((3, 1)),
        bias=np.zeros(3),
        healthy_logit_center=np.zeros(3),
        healthy_logit_scale=np.ones(3),
        kappa=np.zeros(3),
        threshold=50.0,
    )
    raw = np.full((40, 3), 100.0)
    path = page_cusum_numpy(raw, heads)
    assert path.cusum[-1, 0] == 4000.0
    assert path.run_max == 3800.0
    assert not path.eligible[: PERSISTENCE - 1].any()
    assert path.alarm[PERSISTENCE - 1]
    assert path.episode_start[PERSISTENCE - 1]


def test_episode_start_at_onset_does_not_count_continuing_preonset_alarm() -> None:
    indices = np.arange(158, 166)
    continuing_then_new = np.array([False, True, True, True, False, True, True, False])
    assert first_episode_detection(
        continuing_then_new, onset=160, local_indices=indices
    ) == 163
    continuing_only = np.array([False, True, True, True, True, True, True, True])
    assert first_episode_detection(continuing_only, onset=160, local_indices=indices) is None


def test_sequence_group_folds_never_split_a_group() -> None:
    labels: dict[str, int] = {}
    groups: dict[str, str] = {}
    for group_index in range(6):
        for label in (0, *FAULT_IDS):
            name = f"g{group_index}_label{label}"
            labels[name] = label
            groups[name] = f"sequence_{group_index}"
    folds = sequence_group_folds(labels, groups, n_splits=3)
    for group in set(groups.values()):
        assert len({folds[name] for name in folds if groups[name] == group}) == 1
    splits = sequence_group_splits(labels, groups, n_splits=3)
    for split in splits:
        train_groups = {groups[name] for name in split.training_run_names}
        valid_groups = {groups[name] for name in split.validation_run_names}
        assert train_groups.isdisjoint(valid_groups)
        assert split.validation_run_names


def test_sequence_group_folds_reject_incomplete_or_misgrouped_crn_clusters() -> None:
    labels = {f"label{label}": label for label in (0, *FAULT_IDS)}
    incorrectly_separate = {name: name for name in labels}
    with pytest.raises(ValueError, match="each CRN group must contain exactly one"):
        sequence_group_folds(labels, incorrectly_separate, n_splits=2)


def test_five_fold_assignment_scales_to_exact_100_crn_groups() -> None:
    labels: dict[str, int] = {}
    groups: dict[str, int] = {}
    for group in range(201, 301):
        for label in (0, *FAULT_IDS):
            name = f"run{group}_label{label}"
            labels[name] = label
            groups[name] = group
    splits = sequence_group_splits(labels, groups, n_splits=5)
    assert len(splits) == 5
    assert {len(split.training_run_names) for split in splits} == {320}
    assert {len(split.validation_run_names) for split in splits} == {80}
    validation_groups = [
        {groups[name] for name in split.validation_run_names} for split in splits
    ]
    assert all(len(group_set) == 20 for group_set in validation_groups)
    assert set().union(*validation_groups) == set(range(201, 301))


def test_fold_safe_cv_fits_preprocessing_and_heads_on_training_groups_only(
    feature_model: PCLRFeatureModel, monkeypatch
) -> None:
    runs: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    groups: dict[str, int] = {}
    for group in range(5):
        for label in (0, *FAULT_IDS):
            name = f"group{group}_label{label}"
            runs[name] = np.full((500, 52), float(group * 100 + label))
            labels[name] = label
            groups[name] = group

    feature_fit_calls: list[tuple[str, ...]] = []
    head_fit_calls: list[tuple[str, ...]] = []

    def fake_feature_fit(cls, development_runs, *, config=None):  # type: ignore[no-untyped-def]
        feature_fit_calls.append(tuple(sorted(development_runs)))
        return feature_model

    shared_features = np.zeros((MONITOR_STOP - MONITOR_START, PCLR_FEATURE_COUNT))
    shared_prefix = np.zeros((MONITOR_START - BASELINE_STOP, PCLR_FEATURE_COUNT))

    def fake_transform_training_runs(self, selected_runs):  # type: ignore[no-untyped-def]
        return (
            {name: shared_features for name in sorted(selected_runs)},
            {name: shared_prefix for name in sorted(selected_runs)},
        )

    def fake_transform_runs(self, selected_runs):  # type: ignore[no-untyped-def]
        return {name: shared_features for name in sorted(selected_runs)}

    def fake_head_fit(features_by_run, labels_by_run, **kwargs):  # type: ignore[no-untyped-def]
        assert set(features_by_run) == set(labels_by_run)
        assert set(kwargs["prefix_features_by_run"]) == set(features_by_run)
        head_fit_calls.append(tuple(sorted(features_by_run)))
        return _dummy_heads(PCLR_FEATURE_COUNT, threshold=None), {
            "ridge_lambda": kwargs["ridge_lambda"]
        }

    monkeypatch.setattr(PCLRFeatureModel, "fit", classmethod(fake_feature_fit))
    monkeypatch.setattr(
        PCLRFeatureModel, "transform_training_runs", fake_transform_training_runs
    )
    monkeypatch.setattr(PCLRFeatureModel, "transform_runs", fake_transform_runs)
    monkeypatch.setattr(pclr_model, "fit_pclr_heads_cuda", fake_head_fit)

    fits = pclr_model.fit_pclr_group_cv_cuda(
        runs,
        labels,
        groups,
        ridge_lambdas=(0.01,),
        n_splits=5,
    )
    assert len(fits) == 5
    assert len(feature_fit_calls) == len(head_fit_calls) == 5
    for index, fit in enumerate(fits):
        training = set(fit.training_run_names)
        validation = set(fit.validation_run_names)
        assert training == set(feature_fit_calls[index]) == set(head_fit_calls[index])
        assert training.isdisjoint(validation)
        assert {groups[name] for name in training}.isdisjoint(
            {groups[name] for name in validation}
        )
        assert set(fit.metadata["feature_fit_run_names"]) == training
        assert set(fit.metadata["head_fit_run_names"]) == training
        assert set(fit.metadata["healthy_normalization_fit_run_names"]) == {
            name for name in training if labels[name] == 0
        }
        assert set(fit.validation_raw_logits) == validation

    with pytest.raises(RuntimeError, match="precomputed-feature PCLR CV is not fold-safe"):
        pclr_model.fit_pclr_heads_group_cv_cuda(
            {name: shared_features for name in runs},
            labels,
            groups,
            ridge_lambdas=(0.01,),
            n_splits=5,
        )


def test_numpy_torch_float64_inference_parity() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(98)
    features = rng.normal(size=(71, 17))
    heads = _dummy_heads(17)
    maximum = assert_numpy_torch_logit_parity(features, heads, atol=1.0e-10)
    assert maximum <= 1.0e-10


def test_torch_float64_page_path_matches_numpy_beyond_affine_logits() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(198)
    raw = rng.normal(loc=0.15, scale=1.3, size=(47, 3))
    heads = _dummy_heads(4, threshold=2.5)
    reference = page_cusum_numpy(raw, heads, local_indices=np.arange(160, 207))
    candidate = pclr_model.page_cusum_torch_float64(
        raw,
        heads,
        local_indices=np.arange(160, 207),
        device="cpu",
    )
    for name in (
        "raw_logits",
        "standardized_logits",
        "cusum",
        "component_persistence",
        "envelope",
    ):
        np.testing.assert_allclose(
            getattr(candidate, name), getattr(reference, name), rtol=0.0, atol=1.0e-12
        )
    for name in ("local_indices", "eligible", "alarm", "episode_start"):
        np.testing.assert_array_equal(getattr(candidate, name), getattr(reference, name))


def test_cuda_head_fit_smoke_preserves_equal_run_time_contract() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    rng = np.random.default_rng(991)
    features: dict[str, np.ndarray] = {}
    prefix_features: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for label_index, label in enumerate((0, *FAULT_IDS)):
        for run_index in range(2):
            name = f"label{label}_run{run_index}"
            matrix = rng.normal(scale=0.5, size=(16, 5))
            if label != 0:
                matrix[:, label_index - 1] += 1.5
                matrix[8:, label_index - 1] += 0.5
            features[name] = matrix
            prefix_features[name] = rng.normal(scale=0.5, size=(32, 5))
            labels[name] = label
    heads, metadata = fit_pclr_heads_cuda(
        features,
        labels,
        prefix_features_by_run=prefix_features,
        epochs=4,
        learning_rate=0.03,
        device="cuda",
    )
    assert heads.weights.shape == (3, 5)
    assert np.all(heads.healthy_logit_scale > 0.0)
    assert metadata["loss_weighting"] == "equal_heads_equal_classes_equal_runs_equal_times"
    assert metadata["prefix_rank_contract"].startswith("causal_local_128_160")
    assert metadata["dtype"] == "float64"
    healthy_logits = np.concatenate(
        [
            heads.logits_numpy(features[name])
            for name in sorted(features)
            if labels[name] == 0
        ],
        axis=0,
    )
    expected_center = np.median(healthy_logits, axis=0)
    expected_mad = 1.4826 * np.median(
        np.abs(healthy_logits - expected_center), axis=0
    )
    expected_sd = np.std(healthy_logits, axis=0, ddof=1)
    expected_scale = np.where(expected_mad <= 0.0, expected_sd, expected_mad)
    np.testing.assert_allclose(heads.healthy_logit_center, expected_center, atol=1.0e-12)
    np.testing.assert_allclose(heads.healthy_logit_scale, expected_scale, atol=1.0e-12)
    assert_numpy_torch_logit_parity(features["label0_run0"], heads, atol=1.0e-10)


def test_positive_prefix_rank_weight_requires_true_128_160_features() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    features = {
        f"label{label}": np.zeros((8, 3)) for label in (0, *FAULT_IDS)
    }
    labels = {f"label{label}": label for label in (0, *FAULT_IDS)}
    with pytest.raises(ValueError, match="prefix_features_by_run must exactly match"):
        fit_pclr_heads_cuda(features, labels, epochs=1)
    wrong_prefix = {name: np.zeros((31, 3)) for name in features}
    with pytest.raises(ValueError, match=r"shape \(32, 3\)"):
        fit_pclr_heads_cuda(
            features,
            labels,
            prefix_features_by_run=wrong_prefix,
            epochs=1,
        )


def test_top_tail_diagnostics_expose_run_and_component_maxima() -> None:
    heads = PCLRHeads(
        weights=np.ones((3, 1)),
        bias=np.zeros(3),
        healthy_logit_center=np.zeros(3),
        healthy_logit_scale=np.ones(3),
        kappa=np.array([0.0, 0.25, 0.5]),
    )
    raw = np.arange(1.0, 31.0).reshape(10, 3)
    path = page_cusum_numpy(raw, heads)
    one = run_score_diagnostics(path)
    many = diagnostics_by_run({"run_b": path, "run_a": path})
    assert one.run_max == path.run_max
    np.testing.assert_array_equal(one.component_max, path.component_max)
    assert one.component_top_tail_mean.shape == (3,)
    assert tuple(many) == ("run_a", "run_b")


def test_npz_round_trip_uses_only_non_object_arrays(
    feature_model: PCLRFeatureModel, tmp_path
) -> None:
    model = PCLRModel(feature_model, _dummy_heads(PCLR_FEATURE_COUNT, threshold=17.5))
    target = model.save_npz(tmp_path / "pclr_model.npz", metadata={"purpose": "unit-test"})
    with np.load(target, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    restored, metadata = load_pclr_model(target)
    assert metadata == {"purpose": "unit-test"}
    np.testing.assert_array_equal(restored.feature_model.mad_floor, feature_model.mad_floor)
    np.testing.assert_array_equal(
        restored.feature_model.var_coefficients, feature_model.var_coefficients
    )
    np.testing.assert_array_equal(restored.heads.weights, model.heads.weights)
    assert restored.heads.threshold == 17.5
    run = _synthetic_run(101)
    before = model.score_run_numpy(run)
    after = restored.score_run_numpy(run)
    np.testing.assert_array_equal(after.raw_logits, before.raw_logits)
    np.testing.assert_array_equal(after.cusum, before.cusum)


def test_npz_loader_rejects_dtype_coercion_and_noncanonical_threshold_sentinel(
    feature_model: PCLRFeatureModel, tmp_path
) -> None:
    model = PCLRModel(feature_model, _dummy_heads(PCLR_FEATURE_COUNT, threshold=None))
    target = model.save_npz(tmp_path / "canonical.npz")
    with np.load(target, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}

    wrong_dtype = dict(payload)
    wrong_dtype["fault_ids"] = wrong_dtype["fault_ids"].astype(np.float64)
    wrong_dtype_path = tmp_path / "wrong_dtype.npz"
    np.savez_compressed(wrong_dtype_path, **wrong_dtype)
    with pytest.raises(ValueError, match="invalid shape or dtype"):
        load_pclr_model(wrong_dtype_path)

    wrong_sentinel = dict(payload)
    wrong_sentinel["threshold"] = np.array(1.0, dtype=np.float64)
    wrong_sentinel_path = tmp_path / "wrong_sentinel.npz"
    np.savez_compressed(wrong_sentinel_path, **wrong_sentinel)
    with pytest.raises(ValueError, match="canonical zero sentinel"):
        load_pclr_model(wrong_sentinel_path)


def test_input_contract_rejects_wrong_run_shape(feature_model: PCLRFeatureModel) -> None:
    with pytest.raises(ValueError, match=r"shape \(500, 52\)"):
        feature_model.transform_run(np.zeros((499, 52)))
    with pytest.raises(ValueError, match="baseline slice"):
        PCLRConfig(baseline_stop=127)
    assert BASELINE_STOP == 128
