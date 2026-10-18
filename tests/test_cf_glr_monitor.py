from __future__ import annotations

import numpy as np
import pytest

from mate_spc_cf_glr.monitor import (
    AlarmDecision,
    FittedExpertStandardizer,
    apply_strict_threshold,
    fit_rank20_threshold,
    fused_monitor_path,
)


def _identity_standardizer() -> FittedExpertStandardizer:
    return FittedExpertStandardizer(
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        1.0e-12,
    )


def test_standardizer_is_fit_only_on_monitored_points_and_round_trips() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(5, 500, 3)).astype(np.float64)
    logits[:, :160, :] = 1.0e12
    expected_values = logits[:, 160:, :].reshape(-1, 3)
    expected_center = np.median(expected_values, axis=0)
    raw_scale = 1.4826 * np.median(
        np.abs(expected_values - expected_center[None, :]), axis=0
    )
    expected_floor = max(
        1.0e-12,
        float(np.quantile(raw_scale[raw_scale > 1.0e-12], 0.1, method="linear")),
    )
    standardizer = FittedExpertStandardizer.fit(logits)
    np.testing.assert_array_equal(standardizer.center, expected_center)
    np.testing.assert_array_equal(
        standardizer.scale, np.maximum(raw_scale, expected_floor)
    )
    restored = FittedExpertStandardizer.from_document(standardizer.to_document())
    np.testing.assert_array_equal(restored.center, standardizer.center)
    np.testing.assert_array_equal(restored.scale, standardizer.scale)
    transformed = restored.transform(logits[0])
    assert transformed.shape == (500, 3)
    assert transformed.dtype == np.float64
    assert not transformed.flags.writeable


def test_union_then_two_point_persistence_and_causal_attribution() -> None:
    logits = np.zeros((500, 3), dtype=np.float64)
    logits[160, 0] = 5.0
    logits[161, 2] = 6.0
    path = fused_monitor_path(logits, _identity_standardizer())
    assert path.union_score[160] == 5.0
    assert path.union_score[161] == 6.0
    assert path.persistent_score[161] == 5.0
    assert path.run_maximum == 5.0
    assert not path.eligible[160]
    assert path.eligible[161]
    decision = apply_strict_threshold(path, 4.9)
    assert decision.alarm
    assert decision.first_alarm_index == 161
    assert decision.attributed_fault == 15
    tied = apply_strict_threshold(path, 5.0)
    assert not tied.alarm
    assert tied.first_alarm_index is None
    assert tied.attributed_fault is None


def test_rank20_is_exact_maximum_and_threshold_is_strict() -> None:
    logits = np.zeros((20, 500, 3), dtype=np.float64)
    for run in range(20):
        logits[run, 160:162, 0] = float(run + 1)
    standardizer = _identity_standardizer()
    calibration = fit_rank20_threshold(logits, standardizer)
    assert calibration.threshold == 20.0
    assert calibration.rank19 == 19.0
    assert calibration.adjacent_tail_gap == 1.0
    assert calibration.distinct_maxima == 20
    assert calibration.nondegenerate
    assert calibration.to_document()["comparison"] == "strictly_greater_than"
    equal = logits[-1].copy()
    assert not apply_strict_threshold(
        fused_monitor_path(equal, standardizer), calibration.threshold
    ).alarm
    above = equal.copy()
    above[160:162, 0] = 20.25
    assert apply_strict_threshold(
        fused_monitor_path(above, standardizer), calibration.threshold
    ).alarm
    with pytest.raises(ValueError, match="exactly 20"):
        fit_rank20_threshold(logits[:19], standardizer)


def test_monitor_rejects_malformed_nonfinite_and_float_fault_id() -> None:
    with pytest.raises(ValueError, match="zero MAD"):
        FittedExpertStandardizer.fit(np.zeros((2, 500, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="exact float64"):
        fused_monitor_path(
            np.zeros((500, 3), dtype=np.float32), _identity_standardizer()
        )
    broken = np.zeros((500, 3), dtype=np.float64)
    broken[200, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fused_monitor_path(broken, _identity_standardizer())
    with pytest.raises(ValueError, match="attributed_fault"):
        AlarmDecision(True, 161, 3.0, 2.0, 1.0)
    path = fused_monitor_path(np.zeros((500, 3), dtype=np.float64), _identity_standardizer())
    with pytest.raises(ValueError, match="threshold"):
        apply_strict_threshold(path, "1.0")  # type: ignore[arg-type]
