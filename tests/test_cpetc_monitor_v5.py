from __future__ import annotations

import numpy as np
import pytest

from mate_spc_cpetc.monitor import (
    CALIBRATION_RUNS,
    MISS_DELAY,
    MONITOR_COUNT,
    MONITOR_ENDPOINTS,
    ExpertTailCalibrator,
    apply_strict_threshold,
    detection_delay,
    fit_loo_threshold,
    fused_monitor_path,
    leave_one_out_run_maxima,
    monitor_path_from_pvalues,
)


def _calibration_logits(run_count: int = 2) -> np.ndarray:
    endpoint = np.linspace(-2.0, 2.0, MONITOR_COUNT, dtype=np.float64)
    logits = np.empty((run_count, MONITOR_COUNT, 3), dtype=np.float64)
    for run in range(run_count):
        logits[run, :, 0] = endpoint + 0.1 * run
        logits[run, :, 1] = endpoint**3 + 0.2 * run
        logits[run, :, 2] = np.sin(endpoint) + 0.3 * run
    return logits


def _pvalues_for_evidence(evidence: np.ndarray) -> np.ndarray:
    assert evidence.shape == (MONITOR_COUNT,)
    pvalues = np.ones((MONITOR_COUNT, 3), dtype=np.float64)
    pvalues[:, 0] = np.exp(-evidence) / 3.0
    return pvalues


def test_empirical_survival_is_monotone_tie_inclusive_and_bounded() -> None:
    calibration = _calibration_logits()
    calibration[:, 20:25, 0] = 0.5
    fitted = ExpertTailCalibrator.fit(calibration)

    query = np.zeros((MONITOR_COUNT, 3), dtype=np.float64)
    query[:, 1:] = calibration[0, :, 1:]
    query[0, 0] = -1.0e6
    query[1, 0] = 0.5
    query[2, 0] = 1.0e6
    pvalues = fitted.pvalues(query)

    sample_count = calibration.shape[0] * MONITOR_COUNT
    tied_count = int(np.count_nonzero(calibration[:, :, 0] >= 0.5))
    assert pvalues[0, 0] == 1.0
    assert pvalues[1, 0] == (1.0 + tied_count) / (sample_count + 1.0)
    assert pvalues[2, 0] == 1.0 / (sample_count + 1.0)
    assert pvalues[0, 0] >= pvalues[1, 0] >= pvalues[2, 0]
    assert np.all((pvalues > 0.0) & (pvalues <= 1.0))
    assert not pvalues.flags.writeable


def test_leave_one_out_fully_excludes_the_target_run() -> None:
    rng = np.random.default_rng(20260912)
    logits = rng.normal(size=(CALIBRATION_RUNS, MONITOR_COUNT, 3)).astype(np.float64)
    maxima = leave_one_out_run_maxima(logits)

    manual_calibrator = ExpertTailCalibrator.fit(logits[1:])
    manual_maximum = fused_monitor_path(logits[0], manual_calibrator).run_maximum
    assert maxima[0] == manual_maximum

    changed = logits.copy()
    changed[0] += 1.0e4
    changed_calibrator = ExpertTailCalibrator.fit(changed[1:])
    np.testing.assert_array_equal(
        changed_calibrator.sorted_logits,
        manual_calibrator.sorted_logits,
    )


def test_empirical_tail_is_invariant_to_positive_expertwise_affine_scaling() -> None:
    calibration = _calibration_logits(4)
    query = calibration[0].copy()
    baseline = ExpertTailCalibrator.fit(calibration).pvalues(query)

    scale = np.asarray([2.0, 0.5, 7.0], dtype=np.float64)
    shift = np.asarray([10.0, -3.0, 0.25], dtype=np.float64)
    transformed_calibration = calibration * scale[None, None, :] + shift[None, None, :]
    transformed_query = query * scale[None, :] + shift[None, :]
    transformed = ExpertTailCalibrator.fit(transformed_calibration).pvalues(
        transformed_query
    )
    np.testing.assert_array_equal(transformed, baseline)


def test_bonferroni_fusion_evidence_and_deterministic_winner() -> None:
    pvalues = np.ones((MONITOR_COUNT, 3), dtype=np.float64)
    pvalues[0] = np.asarray([0.10, 0.20, 0.30])
    pvalues[1] = np.asarray([0.01, 0.01, 0.50])
    path = monitor_path_from_pvalues(pvalues)

    assert path.fused_pvalues[0] == pytest.approx(0.30)
    assert path.evidence[0] == pytest.approx(-np.log(0.30))
    assert path.fused_pvalues[1] == pytest.approx(0.03)
    assert path.evidence[1] == pytest.approx(-np.log(0.03))
    assert path.winning_expert_indices[0] == 0
    assert path.winning_expert_indices[1] == 0


def test_latest_eight_top_two_scan_is_causal() -> None:
    evidence = np.zeros(MONITOR_COUNT, dtype=np.float64)
    evidence[:9] = np.arange(1.0, 10.0, dtype=np.float64)
    path = monitor_path_from_pvalues(_pvalues_for_evidence(evidence))

    assert not path.eligible[0]
    assert path.scan_scores[0] == 0.0
    assert path.scan_scores[1] == pytest.approx(3.0)
    assert path.scan_scores[7] == pytest.approx(15.0)
    # At position 8, the latest-eight window is positions 1..8: 8 + 9.
    assert path.scan_scores[8] == pytest.approx(17.0)

    changed = evidence.copy()
    changed[20] = 100.0
    changed_path = monitor_path_from_pvalues(_pvalues_for_evidence(changed))
    np.testing.assert_array_equal(changed_path.scan_scores[:20], path.scan_scores[:20])


def test_alarm_comparison_is_strict_and_delay_uses_frozen_miss_value() -> None:
    evidence = np.ones(MONITOR_COUNT, dtype=np.float64)
    path = monitor_path_from_pvalues(_pvalues_for_evidence(evidence))
    assert path.run_maximum == pytest.approx(2.0)

    tied = apply_strict_threshold(path, 2.0)
    assert not tied.alarm
    assert tied.delay == MISS_DELAY == 340
    assert detection_delay(None) == 340

    above = apply_strict_threshold(path, 1.99)
    assert above.alarm
    assert above.first_alarm_position == 1
    assert above.first_alarm_endpoint == MONITOR_ENDPOINTS[1] == 164
    assert above.attributed_fault == 3
    assert above.delay == 4


def test_loo_threshold_is_the_maximum_and_reports_nondegeneracy() -> None:
    rng = np.random.default_rng(78143)
    logits = rng.normal(size=(CALIBRATION_RUNS, MONITOR_COUNT, 3)).astype(np.float64)
    calibration = fit_loo_threshold(logits)

    assert calibration.threshold == float(np.max(calibration.run_maxima))
    assert calibration.rank19 == float(np.sort(calibration.run_maxima)[-2])
    assert calibration.adjacent_tail_gap > calibration.numerical_tolerance
    assert calibration.distinct_maxima >= 19
    assert calibration.nondegenerate


def test_monitor_rejects_nonfinite_but_constant_ecdf_remains_defined() -> None:
    broken = _calibration_logits()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite exact float64"):
        ExpertTailCalibrator.fit(broken)

    constant = ExpertTailCalibrator.fit(
        np.zeros((2, MONITOR_COUNT, 3), dtype=np.float64)
    )
    tied = constant.pvalues(np.zeros((MONITOR_COUNT, 3), dtype=np.float64))
    above = constant.pvalues(np.ones((MONITOR_COUNT, 3), dtype=np.float64))
    np.testing.assert_array_equal(tied, np.ones_like(tied))
    np.testing.assert_array_equal(above, np.full_like(above, 1.0 / 171.0))

    malformed_p = np.ones((MONITOR_COUNT, 3), dtype=np.float64)
    malformed_p[0, 0] = 0.0
    with pytest.raises(ValueError, match=r"in \(0,1\]"):
        monitor_path_from_pvalues(malformed_p)

    with pytest.raises(ValueError, match="frozen monitor endpoint"):
        detection_delay(161)


def test_degenerate_loo_threshold_can_be_inspected_but_not_accepted() -> None:
    # Continuous per-expert logits keep every ECDF valid, while identical runs make
    # all leave-one-out run maxima equal and therefore invalidate the threshold.
    one_run = _calibration_logits(1)[0]
    identical = np.repeat(one_run[None, :, :], CALIBRATION_RUNS, axis=0)
    result = fit_loo_threshold(identical, require_nondegenerate=False)
    assert not result.nondegenerate
    with pytest.raises(ValueError, match="threshold is degenerate"):
        result.require_nondegenerate()
    with pytest.raises(ValueError, match="threshold is degenerate"):
        fit_loo_threshold(identical)
