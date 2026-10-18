from __future__ import annotations

import json

import numpy as np
import pytest

from mate_spc_farm_v7.monitor import (
    CALIBRATION_RUN_COUNT,
    CANDIDATE_ID,
    CHANNEL_INDICES,
    DEFAULT_DRIFT,
    FIT_ROWS_PER_RUN,
    FarmDecisions,
    FarmReference,
    FarmScoreBatch,
    FarmThreshold,
    MONITOR_LENGTH,
    MONITOR_START,
    MONITOR_STOP,
    SELECTED_CHANNEL_COUNT,
    TOP_R,
    apply_threshold,
    calibrate_threshold,
    load_farm_reference_npz,
    save_farm_reference_npz,
    score_runs_cpu,
    theoretical_score_upper_bound,
)


def _healthy_runs(run_count: int = 60) -> np.ndarray:
    run = np.arange(run_count, dtype=np.float64)[:, None, None] * 1.0e-5
    time = np.arange(FIT_ROWS_PER_RUN, dtype=np.float64)[None, :, None] * 1.0e-3
    channel = np.arange(52, dtype=np.float64)[None, None, :] * 0.1
    return np.asarray(run + time + channel, dtype=np.float64)


@pytest.fixture(scope="module")
def development_reference() -> FarmReference:
    return FarmReference.fit(_healthy_runs())


def _scoring_runs() -> np.ndarray:
    channel = np.arange(52, dtype=np.float64)[None, None, :] * 0.1
    runs = np.broadcast_to(channel + 0.25, (4, 500, 52)).copy()
    runs[1, MONITOR_START:, :] = channel[0, 0, :] + 2.0
    runs[2, :MONITOR_START, :] = channel[0, 0, :] + 2.0
    runs[3, MONITOR_START:, :] = channel[0, 0, :] - 2.0
    return np.asarray(runs, dtype=np.float64)


def test_frozen_channels_and_constants_are_exact() -> None:
    assert CANDIDATE_ID == "farm-rankcusum-v7-fixed"
    assert CHANNEL_INDICES == tuple(range(41)) + (46, 47, 48, 50, 51)
    assert len(CHANNEL_INDICES) == SELECTED_CHANNEL_COUNT == 46
    assert DEFAULT_DRIFT == 1.3
    assert TOP_R == 4
    assert (MONITOR_START, MONITOR_STOP, MONITOR_LENGTH) == (160, 500, 340)


def test_fit_is_stable_sorted_and_ignores_the_six_excluded_channels() -> None:
    healthy = _healthy_runs()
    reference = FarmReference.fit(healthy, expected_run_count=60)
    selected = healthy[:, :, CHANNEL_INDICES].reshape(-1, SELECTED_CHANNEL_COUNT)
    expected = np.sort(selected, axis=0, kind="stable").T
    np.testing.assert_array_equal(reference.sorted_values, expected)
    assert reference.sorted_values.shape == (46, 30000)
    assert reference.sorted_values.dtype == np.float64
    assert not reference.sorted_values.flags.writeable

    changed = healthy.copy()
    changed[:, :, [41, 42, 43, 44, 45, 49]] += 1.0e6
    np.testing.assert_array_equal(
        FarmReference.fit(changed).sorted_values, reference.sorted_values
    )


def test_fit_count_is_strictly_sixty_or_one_hundred() -> None:
    final = FarmReference.fit(_healthy_runs(100), expected_run_count=100)
    assert final.fit_run_count == 100
    assert final.reference_count == 50000
    for invalid in (True, 59, 80, 101, 60.0):
        with pytest.raises(ValueError):
            FarmReference.fit(_healthy_runs(), expected_run_count=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly 60"):
        FarmReference.fit(_healthy_runs(59), expected_run_count=60)


def test_reference_fit_rejects_wrong_dtype_shape_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="float64"):
        FarmReference.fit(_healthy_runs().astype(np.float32))
    with pytest.raises(ValueError, match="shape"):
        FarmReference.fit(np.zeros((60, 499, 52), dtype=np.float64))
    nonfinite = _healthy_runs()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        FarmReference.fit(nonfinite)


def test_reference_state_and_no_pickle_npz_roundtrip(
    development_reference: FarmReference, tmp_path: pytest.TempPathFactory
) -> None:
    state = development_reference.state_dict()
    restored = FarmReference.from_state_dict(state)
    np.testing.assert_array_equal(restored.sorted_values, development_reference.sorted_values)
    assert restored.fit_run_count == 60
    path = tmp_path / "farm_reference.npz"
    save_farm_reference_npz(development_reference, path)
    loaded = load_farm_reference_npz(path)
    np.testing.assert_array_equal(loaded.sorted_values, development_reference.sorted_values)
    assert loaded.state_dict()["ecdf_search_side"] == "left"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("candidate_id", "different"),
        ("drift", 1.2),
        ("top_r", 5),
        ("channel_indices", list(range(46))),
        ("fit_run_count", 80),
        ("reference_count", 50000),
    ],
)
def test_reference_state_rejects_changed_contract(
    development_reference: FarmReference, field: str, bad_value: object
) -> None:
    state = development_reference.state_dict()
    state[field] = bad_value
    with pytest.raises(ValueError):
        FarmReference.from_state_dict(state)


def test_cpu_scan_resets_at_160_and_matches_exact_extreme_rank_recurrence(
    development_reference: FarmReference,
) -> None:
    scores = score_runs_cpu(development_reference, _scoring_runs())
    assert scores.paths.shape == (4, MONITOR_LENGTH)
    assert scores.endpoints[[0, -1]].tolist() == [160, 499]
    np.testing.assert_array_equal(scores.paths[0], np.zeros(MONITOR_LENGTH))
    np.testing.assert_array_equal(scores.paths[2], np.zeros(MONITOR_LENGTH))

    increment = math_log_increment = np.log(development_reference.reference_count + 2.0) - 1.3
    expected = TOP_R * increment * np.arange(1, MONITOR_LENGTH + 1, dtype=np.float64)
    np.testing.assert_allclose(scores.paths[1], expected, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(scores.paths[3], expected, rtol=0.0, atol=2.0e-10)
    assert math_log_increment > 0.0
    np.testing.assert_array_equal(scores.run_maxima, np.max(scores.paths, axis=1))
    assert not scores.paths.flags.writeable
    assert not scores.run_maxima.flags.writeable


def test_cpu_scoring_rejects_invalid_input(development_reference: FarmReference) -> None:
    with pytest.raises(ValueError, match="float64"):
        score_runs_cpu(development_reference, _scoring_runs().astype(np.float32))
    wrong = np.zeros((2, 500, 51), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        score_runs_cpu(development_reference, wrong)
    nonfinite = _scoring_runs()
    nonfinite[0, 200, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        score_runs_cpu(development_reference, nonfinite)


def test_rank20_threshold_roundtrip_and_strict_endpoint_decisions() -> None:
    maxima = np.arange(1.0, CALIBRATION_RUN_COUNT + 1.0, dtype=np.float64)
    threshold = calibrate_threshold(maxima, repeat_error=0.0)
    assert threshold.threshold == 20.0
    assert threshold.rank19 == 19.0
    assert threshold.top_gap == 1.0
    assert threshold.distinct_count == 20
    restored = FarmThreshold.from_document(threshold.to_document())
    assert restored == threshold

    paths = np.zeros((3, MONITOR_LENGTH), dtype=np.float64)
    paths[0, 50] = threshold.threshold
    paths[1, 0] = threshold.threshold + 1.0e-6
    paths[2, -1] = threshold.threshold + 1.0e-6
    decisions = apply_threshold(paths, threshold)
    assert decisions.alarms.tolist() == [False, True, True]
    assert decisions.alarm_endpoints.tolist() == [-1, 160, 499]
    assert decisions.delays.tolist() == [340, 0, 339]
    assert decisions.run_maxima.tolist() == [20.0, 20.000001, 20.000001]


def test_threshold_fails_closed_on_degeneracy_gap_and_unreachable_boundary() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        calibrate_threshold(np.zeros(20, dtype=np.float64), repeat_error=0.0)
    too_small_gap = np.arange(20, dtype=np.float64)
    too_small_gap[-1] = too_small_gap[-2] + 1.0e-7
    with pytest.raises(ValueError, match="tolerance"):
        calibrate_threshold(too_small_gap, repeat_error=1.0e-8)
    upper = theoretical_score_upper_bound(30000)
    unreachable = np.linspace(upper - 19.0, upper, 20, dtype=np.float64)
    with pytest.raises(ValueError, match="reachable"):
        calibrate_threshold(unreachable, repeat_error=0.0)


@pytest.mark.parametrize("invalid", [True, 19, 21, 20.0])
def test_threshold_rejects_invalid_calibration_count(invalid: object) -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(
            np.arange(20, dtype=np.float64), expected_run_count=invalid  # type: ignore[arg-type]
        )


def test_threshold_rejects_boolean_nonfinite_and_tampered_documents() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(np.ones(20, dtype=np.bool_))
    with pytest.raises(ValueError):
        calibrate_threshold(np.full(20, np.nan, dtype=np.float64))
    threshold = calibrate_threshold(np.arange(1.0, 21.0, dtype=np.float64))
    document = threshold.to_document()
    document["numeric_tolerance"] = threshold.tolerance + 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        FarmThreshold.from_document(document)


def test_score_batch_reference_count_must_match_threshold() -> None:
    score = FarmScoreBatch(
        paths=np.zeros((1, MONITOR_LENGTH), dtype=np.float64),
        source_device="cpu",
        reference_count=50000,
    )
    threshold = calibrate_threshold(np.arange(1.0, 21.0, dtype=np.float64))
    with pytest.raises(ValueError, match="reference counts"):
        apply_threshold(score, threshold)


def test_score_and_decision_metadata_reject_noncanonical_or_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="source_device"):
        FarmScoreBatch(
            paths=np.zeros((1, MONITOR_LENGTH), dtype=np.float64),
            source_device="cuda:gpu0",
            reference_count=30000,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        FarmDecisions(
            alarms=np.asarray([False], dtype=np.bool_),
            alarm_endpoints=np.asarray([-1], dtype=np.int64),
            delays=np.asarray([340], dtype=np.int64),
            run_maxima=np.asarray([2.0], dtype=np.float64),
            threshold=1.0,
        )


def test_npz_archive_is_json_metadata_without_pickle(
    development_reference: FarmReference, tmp_path: pytest.TempPathFactory
) -> None:
    path = tmp_path / "reference.npz"
    save_farm_reference_npz(development_reference, path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata_json", "sorted_values"}
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["candidate_id"] == CANDIDATE_ID
        assert metadata["fit_run_count"] == 60
        assert archive["sorted_values"].dtype == np.float64
