from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Keep the verified Windows DLL load order in the test process too.
import torch
import numpy as np
import pytest

from mate_spc_minirocket_v8.evaluation import (
    CONDITION_IDS,
    FAULT_IDS,
    FOLD_RUN_IDS,
    RunDecision,
    evaluate_gates,
    fold_roles,
)
from mate_spc_minirocket_v8.model import (
    CLASS_EXAMPLE_COUNTS,
    CLASS_WEIGHTS,
    FEATURE_STD_FLOOR,
    GLOBAL_SEED,
    _fit_feature_moments,
    _torch_linear_fit,
    _training_metadata,
    normalize_runs,
    robust_fit_normalizer,
    set_determinism,
)
from mate_spc_minirocket_v8.monitor import (
    CALIBRATION_SCALE_FLOOR,
    InnerThreshold,
    MonitorCalibration,
    apply_threshold,
    compute_log_odds,
    fit_monitor_calibration,
    inner_threshold,
    scan_batch,
    scan_run,
)
from mate_spc_minirocket_v8.transform import (
    _fit_transform_impl,
    load_minirocket_state_npz,
    save_minirocket_state_npz,
    transform_windows,
)


def test_frozen_weighting_equalizes_total_endpoint_weight() -> None:
    totals = np.asarray(CLASS_EXAMPLE_COUNTS, dtype=np.float64) * np.asarray(
        CLASS_WEIGHTS, dtype=np.float64
    )
    np.testing.assert_allclose(totals, np.full(4, 5_100.0), rtol=0.0, atol=1e-12)
    assert sum(CLASS_EXAMPLE_COUNTS) == 20_400


def test_robust_normalizer_uses_all_fit_rows_mad_floor_and_clip() -> None:
    fit = np.zeros((60, 500, 52), dtype=np.float64)
    fit[:30, :, 0] = -2.0
    fit[30:, :, 0] = 2.0
    fit[:, :, 1] = 7.0
    state = robust_fit_normalizer(fit)
    assert state.location.dtype == np.float64
    assert state.location[0] == 0.0
    assert state.scale[0] == pytest.approx(1.4826 * 2.0)
    assert state.location[1] == 7.0
    assert state.scale[1] == 1e-6
    assert not state.location.flags.writeable

    runs = np.zeros((1, 500, 52), dtype=np.float64)
    runs[:, :, 0] = 1e9
    runs[:, :, 1] = -1e9
    normalized = normalize_runs(runs, state)
    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 500, 52)
    assert float(normalized[0, 0, 0]) == 10.0
    assert float(normalized[0, 0, 1]) == -10.0


def test_training_metadata_uses_real_noncontiguous_ids_and_exact_order() -> None:
    run_ids = np.asarray(fold_roles(0).fit, dtype=np.int64)
    assert run_ids.shape == (60,)
    assert not np.array_equal(run_ids, np.arange(1, 61, dtype=np.int64))
    labels, sources, runs, endpoints = _training_metadata(run_ids)
    assert labels.shape == sources.shape == runs.shape == endpoints.shape == (20_400,)
    assert (labels[0], sources[0], runs[0], endpoints[0]) == (0, 1, run_ids[0], 160)
    assert (labels[84], sources[84], runs[84], endpoints[84]) == (
        0,
        1,
        run_ids[0],
        496,
    )
    assert (labels[85], sources[85], runs[85], endpoints[85]) == (
        0,
        1,
        run_ids[1],
        160,
    )
    assert (labels[5_100], sources[5_100], runs[5_100], endpoints[5_100]) == (
        0,
        3,
        run_ids[0],
        128,
    )
    assert (labels[6_540], sources[6_540], runs[6_540], endpoints[6_540]) == (
        1,
        3,
        run_ids[0],
        192,
    )
    assert (labels[-1], sources[-1], runs[-1], endpoints[-1]) == (
        3,
        15,
        run_ids[-1],
        496,
    )

    invalid = run_ids.copy()
    invalid[[0, 1]] = invalid[[1, 0]]
    with pytest.raises(ValueError, match="strictly ascending"):
        _training_metadata(invalid)
    with pytest.raises(ValueError, match="int64"):
        _training_metadata(run_ids.astype(np.int32))


def test_feature_moments_are_float64_population_statistics_with_floor() -> None:
    values = np.asarray(
        [[1.0, 5.0, -2.0], [2.0, 5.0, 1.0], [4.0, 5.0, 7.0]],
        dtype=np.float32,
    )
    mean, scale = _fit_feature_moments(values)
    assert mean.dtype == scale.dtype == np.float64
    np.testing.assert_allclose(mean, values.mean(axis=0, dtype=np.float64), rtol=0, atol=0)
    expected = values.std(axis=0, dtype=np.float64, ddof=0)
    expected[1] = FEATURE_STD_FLOOR
    np.testing.assert_allclose(scale, expected, rtol=0, atol=1e-15)


def test_vendored_transform_state_roundtrip_preserves_exact_cpu_ppvs(
    tmp_path: Path,
) -> None:
    windows = np.random.default_rng(843).normal(size=(8, 52, 128)).astype(np.float32)
    set_determinism(GLOBAL_SEED)
    state = _fit_transform_impl(
        windows, device="cpu", enforce_frozen_count=False
    )
    python_rng_before = __import__("random").getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()
    first = transform_windows(windows[:2], state, device="cpu")
    assert __import__("random").getstate() == python_rng_before
    assert numpy_rng_before[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(numpy_rng_before[1], np.random.get_state()[1])
    assert numpy_rng_before[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_rng_before, torch.random.get_rng_state())
    assert first.shape == (2, 9_996)
    assert first.dtype == np.float32
    assert np.all((first >= 0.0) & (first <= 1.0))
    destination = tmp_path / "minirocket_state.npz"
    save_minirocket_state_npz(destination, state)
    restored = load_minirocket_state_npz(destination)
    second = transform_windows(windows[:2], restored, device="cpu")
    np.testing.assert_array_equal(first, second)
    with np.load(destination, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_log_odds_uses_stable_fault_logsumexp_minus_health() -> None:
    logits = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1000.0, 999.0, 998.0, 997.0]],
        dtype=np.float32,
    )
    q = compute_log_odds(logits)
    assert q.dtype == np.float64
    assert q[0] == pytest.approx(np.log(3.0), abs=1e-15)
    expected = 999.0 + np.log(1.0 + np.exp(-1.0) + np.exp(-2.0)) - 1000.0
    assert q[1] == pytest.approx(expected, abs=1e-15)
    assert np.isfinite(q).all()
    with pytest.raises(ValueError, match="dimension 4"):
        compute_log_odds(np.zeros((2, 3), dtype=np.float64))


def test_fit_monitor_calibration_removes_each_run_prefix_level() -> None:
    dense_pattern = np.linspace(-2.0, 3.0, 340, dtype=np.float64)
    complete = np.empty((60, 348), dtype=np.float64)
    for run in range(60):
        offset = 100.0 * run
        complete[run, :8] = offset
        complete[run, 8:] = offset + dense_pattern
    state = fit_monitor_calibration(complete)
    expected_center = float(np.median(dense_pattern))
    expected_scale = max(
        1.4826 * float(np.median(np.abs(dense_pattern - expected_center))),
        CALIBRATION_SCALE_FLOOR,
    )
    assert state.center == pytest.approx(expected_center, abs=1e-12)
    assert state.scale == pytest.approx(expected_scale, abs=1e-12)
    assert state.pool_size == 20_400


def test_scan_has_exact_cusum_and_first_16_point_intermittent_semantics() -> None:
    calibration = MonitorCalibration(center=0.0, scale=1.0)
    prefix = np.zeros(8, dtype=np.float64)
    q = np.full(340, 0.5, dtype=np.float64)
    path = scan_run(q, prefix, calibration)
    assert path.shape == (340,)
    assert path[0] == pytest.approx(0.25)
    assert path[14] == pytest.approx(3.75)
    assert path[15] == pytest.approx(4.0)

    intermittent_q = np.full(340, -10.0, dtype=np.float64)
    intermittent_q[[0, 2, 4, 6]] = 2.0
    intermittent_path = scan_run(intermittent_q, prefix, calibration)
    # At endpoint 175, the inclusive 160..175 window contains four positive
    # values of two, so I=0.5*(2+2+2+2)=4 while C has reset to zero.
    assert intermittent_path[15] == pytest.approx(4.0)
    batch = scan_batch(
        np.stack((q, intermittent_q)), np.zeros((2, 8)), calibration
    )
    np.testing.assert_array_equal(batch.paths[0], path)
    np.testing.assert_array_equal(batch.paths[1], intermittent_path)
    np.testing.assert_array_equal(batch.endpoints, np.arange(160, 500))


def test_inner_rank20_threshold_and_strict_crossing_are_exact() -> None:
    threshold = inner_threshold(np.arange(20, dtype=np.float64), repeat_error=0.0)
    assert threshold.threshold == 19.0
    assert threshold.rank19 == 18.0
    assert threshold.distinct_count == 20
    assert threshold.numeric_tolerance == pytest.approx(1e-4)

    paths = np.zeros((3, 340), dtype=np.float64)
    paths[0, 10] = 19.0  # equality is negative
    paths[1, 0] = 19.5
    paths[2, -1] = 20.0
    decisions = apply_threshold(paths, threshold)
    np.testing.assert_array_equal(decisions.alarms, [False, True, True])
    np.testing.assert_array_equal(decisions.alarm_endpoints, [-1, 160, 499])
    np.testing.assert_array_equal(decisions.delays, [340, 0, 339])

    with pytest.raises(ValueError, match="distinct"):
        inner_threshold(np.repeat(np.arange(10, dtype=np.float64), 2), 0.0)
    with pytest.raises(ValueError, match="replay-separation"):
        InnerThreshold(
            threshold=2.0, rank19=1.0, distinct_count=20, repeat_error=0.01
        )


def _outer_rows(
    *,
    healthy_alarms: set[int] | None = None,
    misses: dict[int, set[int]] | None = None,
    endpoint: int = 160,
) -> list[RunDecision]:
    healthy_alarms = set() if healthy_alarms is None else healthy_alarms
    misses = {} if misses is None else misses
    result: list[RunDecision] = []
    for source in CONDITION_IDS:
        for run in range(1, 101):
            if source == 1:
                alarm_endpoint = endpoint if run in healthy_alarms else None
            else:
                alarm_endpoint = None if run in misses.get(source, set()) else endpoint
            result.append(RunDecision(source, run, float(run), alarm_endpoint))
    return result


def test_v8_gates_use_one_shared_300_fault_delay_pool() -> None:
    healthy = {run for fold in range(5) for run in FOLD_RUN_IDS[fold][:2]}
    ten_misses_each = {
        source: {run for fold in range(5) for run in FOLD_RUN_IDS[fold][:2]}
        for source in FAULT_IDS
    }
    passing = evaluate_gates(
        _outer_rows(healthy_alarms=healthy, misses=ten_misses_each), [True] * 5
    )
    assert passing.all_pass
    assert passing.decision == "GO"
    assert passing.aggregate_metrics.pooled_fault_delays.runs == 300
    assert passing.aggregate_metrics.pooled_fault_delays.median_delay_samples == 0.0
    assert (
        passing.aggregate_metrics.pooled_fault_delays.nearest_rank_p90_delay_samples
        == 0.0
    )

    thirty_one_misses = {source: set(values) for source, values in ten_misses_each.items()}
    extra = next(run for run in FOLD_RUN_IDS[0] if run not in thirty_one_misses[15])
    thirty_one_misses[15].add(extra)
    failing = evaluate_gates(_outer_rows(misses=thirty_one_misses), [True] * 5)
    assert all(
        failing.check(f"fault_{source}_overall_detection") for source in FAULT_IDS
    )
    assert all(
        failing.check(f"fault_{source}_each_fold_detection") for source in FAULT_IDS
    )
    assert failing.check("pooled_fault_median_delay")
    assert not failing.check("pooled_fault_p90_delay")
    assert failing.decision == "DEVELOPMENT_NO_GO"


def test_one_invalid_numeric_fold_forces_development_no_go() -> None:
    report = evaluate_gates(_outer_rows(), [True, True, False, True, True])
    assert not report.check("numeric_all_folds")
    assert not report.all_pass
    for invalid in ([True] * 4, [1] * 5, [np.bool_(True)] * 5):
        with pytest.raises(ValueError, match="five exact booleans"):
            evaluate_gates(_outer_rows(), invalid)  # type: ignore[arg-type]


def _toy_linear_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [-2.0, -2.0, 0.0],
            [-2.0, -1.0, 0.0],
            [-1.0, -2.0, 0.0],
            [-1.0, -1.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, -1.0, 0.0],
            [1.0, -2.0, 0.0],
            [1.0, -1.0, 0.0],
            [-2.0, 2.0, 0.0],
            [-2.0, 1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.arange(4, dtype=np.int64), 4)
    return features, labels


def test_zero_initialized_full_batch_lbfgs_toy_fit_is_deterministic_on_cpu() -> None:
    features, labels = _toy_linear_data()
    set_determinism(GLOBAL_SEED)
    first = _torch_linear_fit(
        features, labels, device="cpu", enforce_frozen_shape=False
    )
    set_determinism(GLOBAL_SEED)
    second = _torch_linear_fit(
        features, labels, device="cpu", enforce_frozen_shape=False
    )
    assert first[0].shape == (4, 3)
    assert first[1].shape == (4,)
    assert first[3] < first[2]
    assert first[4] >= 0.0
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:5] == second[2:5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_toy_lbfgs_supports_deterministic_cuda_replay() -> None:
    features, labels = _toy_linear_data()
    set_determinism(GLOBAL_SEED)
    first = _torch_linear_fit(
        features, labels, device="cuda", enforce_frozen_shape=False
    )
    set_determinism(GLOBAL_SEED)
    second = _torch_linear_fit(
        features, labels, device="cuda", enforce_frozen_shape=False
    )
    assert np.max(np.abs(first[0] - second[0])) <= 1e-6
    assert np.max(np.abs(first[1] - second[1])) <= 1e-6
    assert first[3] <= first[2] + 1e-10


def test_clean_subprocess_imports_torch_before_numpy_and_fits_toy_head() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = """
from mate_spc_minirocket_v8.model import _torch_linear_fit
import numpy as np
x = np.asarray([[-1.,-1.],[-2.,-1.],[1.,-1.],[2.,-1.],[-1.,1.],[-2.,1.],[1.,1.],[2.,1.]], dtype=np.float32)
y = np.repeat(np.arange(4, dtype=np.int64), 2)
r = _torch_linear_fit(x, y, device='cpu', enforce_frozen_shape=False)
assert r[3] <= r[2] + 1e-10
print('clean-v8-import-fit-ok')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "clean-v8-import-fit-ok" in completed.stdout
