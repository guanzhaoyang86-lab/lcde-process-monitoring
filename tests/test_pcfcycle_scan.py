from __future__ import annotations

import numpy as np
import pytest

from mate_spc_pcfcycle.scan import (
    CALIBRATOR_SCHEMA,
    DENSE_ENDPOINTS,
    PREFIX_ENDPOINTS,
    RunRelativeLogitCalibrator,
    multiscale_unbounded_scan,
    scan_many,
    validate_inner_threshold,
)


def _logits(runs: int = 60) -> np.ndarray:
    columns = len(PREFIX_ENDPOINTS) + len(DENSE_ENDPOINTS)
    values = np.zeros((runs, columns), dtype=np.float64)
    for run in range(runs):
        values[run, : len(PREFIX_ENDPOINTS)] = 10.0 * run + np.linspace(-0.1, 0.1, 8)
        values[run, len(PREFIX_ENDPOINTS) :] = 10.0 * run + np.linspace(-1.0, 1.0, 340)
    return values


def test_calibrator_is_invariant_to_per_run_logit_offsets() -> None:
    logits = _logits()
    fitted = RunRelativeLogitCalibrator.fit(logits, expected_run_count=60)
    offsets = np.linspace(-11.0, 27.0, logits.shape[0])[:, None]
    shifted = logits + offsets
    np.testing.assert_allclose(fitted.transform(logits), fitted.transform(shifted), atol=1e-12)


def test_calibrator_document_roundtrip_is_exact() -> None:
    fitted = RunRelativeLogitCalibrator.fit(_logits())
    restored = RunRelativeLogitCalibrator.from_document(fitted.to_document())
    assert restored == fitted
    assert fitted.fit_run_count == 60
    assert fitted.to_document()["schema"] == CALIBRATOR_SCHEMA


def test_final_refit_count_is_serialized_and_transform_count_is_unrestricted() -> None:
    fitted = RunRelativeLogitCalibrator.fit(_logits(100), expected_run_count=100)
    assert fitted.fit_run_count == 100
    assert fitted.to_document()["fit_run_count"] == 100
    assert fitted.transform(_logits(3)).shape == (3, len(DENSE_ENDPOINTS))


@pytest.mark.parametrize("expected", [True, 59, 80, 101, 60.0])
def test_calibrator_rejects_invalid_expected_fit_count(expected: object) -> None:
    with pytest.raises(ValueError):
        RunRelativeLogitCalibrator.fit(_logits(60), expected_run_count=expected)  # type: ignore[arg-type]


def test_calibrator_rejects_fit_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="exactly 60"):
        RunRelativeLogitCalibrator.fit(_logits(59), expected_run_count=60)


@pytest.mark.parametrize(
    ("field", "value"),
    [("drift", 0.3), ("fit_run_count", 80), ("fit_run_count", True)],
)
def test_calibrator_document_rejects_changed_frozen_values(field: str, value: object) -> None:
    document = RunRelativeLogitCalibrator.fit(_logits()).to_document()
    document[field] = value
    with pytest.raises(ValueError):
        RunRelativeLogitCalibrator.from_document(document)


def test_calibrator_document_rejects_legacy_schema() -> None:
    document = RunRelativeLogitCalibrator.fit(_logits()).to_document()
    document["schema"] = "pcf_cyclerank_run_relative_logit_calibrator_v1"
    with pytest.raises(ValueError, match="schema"):
        RunRelativeLogitCalibrator.from_document(document)


def test_fit_health_center_is_removed_after_run_prefix_baseline() -> None:
    logits = _logits()
    fitted = RunRelativeLogitCalibrator.fit(logits)
    transformed = fitted.transform(logits)
    assert np.median(transformed) == pytest.approx(0.0, abs=1e-12)
    assert fitted.to_document()["prefix_endpoints"] == list(range(128, 160, 4))


def test_constant_evidence_has_exact_four_sample_branch_and_persistence() -> None:
    evidence = np.ones(340, dtype=np.float64)
    path = multiscale_unbounded_scan(evidence)
    assert path.raw_endpoints[0] == 163
    assert path.endpoints[0] == 164
    assert path.raw_scores[0] == pytest.approx(1.5)
    assert path.scores[0] == pytest.approx(1.5)
    assert path.run_maximum > path.scores[0]


def test_scan_is_not_finitely_capped() -> None:
    low = np.zeros(340, dtype=np.float64)
    high = low.copy()
    high[100:164] = 1.0e6
    assert multiscale_unbounded_scan(high).run_maximum > 1.0e6
    assert multiscale_unbounded_scan(high).run_maximum > multiscale_unbounded_scan(low).run_maximum


def test_strict_persistent_decision_does_not_alarm_on_equality() -> None:
    path = multiscale_unbounded_scan(np.ones(340, dtype=np.float64))
    threshold = path.run_maximum
    assert path.first_exceedance(threshold) is None
    assert path.first_exceedance(threshold - 1e-12) is not None


def test_scan_many_matches_individual_paths() -> None:
    values = np.vstack((np.zeros(340), np.ones(340))).astype(np.float64)
    paths = scan_many(values)
    assert len(paths) == 2
    np.testing.assert_array_equal(paths[1].scores, multiscale_unbounded_scan(values[1]).scores)


@pytest.mark.parametrize("drift", [-99.0, 0.0, 0.3, float("nan")])
def test_all_public_scan_entries_lock_frozen_drift(drift: float) -> None:
    with pytest.raises(ValueError, match="frozen value"):
        multiscale_unbounded_scan(np.zeros(340), drift=drift)
    with pytest.raises(ValueError, match="frozen value"):
        scan_many(np.zeros((2, 340)), drift=drift)


def test_inner_threshold_is_rank20_and_requires_nondegenerate_top_gap() -> None:
    audit = validate_inner_threshold(np.arange(20, dtype=np.float64), repeat_error=0.0)
    assert audit.threshold == 19.0
    assert audit.rank19 == 18.0
    assert audit.distinct_count == 20
    assert audit.valid
    tied = validate_inner_threshold(np.concatenate((np.arange(18.0), [18.0, 18.0])), repeat_error=0.0)
    assert tied.threshold == tied.rank19
    assert not tied.valid


@pytest.mark.parametrize(
    "value",
    [np.zeros(339), np.full(340, np.nan), np.zeros((2, 340))],
)
def test_single_scan_rejects_wrong_or_nonfinite_geometry(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        multiscale_unbounded_scan(value)


def test_inner_threshold_rejects_wrong_count_and_nonfinite_error() -> None:
    with pytest.raises(ValueError):
        validate_inner_threshold(np.arange(19.0), repeat_error=0.0)
    with pytest.raises(ValueError):
        validate_inner_threshold(np.arange(20.0), repeat_error=float("nan"))
