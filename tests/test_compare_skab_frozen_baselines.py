from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_skab_frozen_baselines_gpu")


def _timestamps(size: int) -> np.ndarray:
    return np.datetime64("2020-01-01T00:00:00") + np.arange(size).astype("timedelta64[s]")


def test_alarm_after_anomaly_ends_is_not_active_event_detection(runner) -> None:
    scores = np.zeros(400)
    scores[350:] = 2.0
    labels = np.zeros(400, dtype=bool)
    labels[310:330] = True

    result = runner._supplement(scores, 1.0, labels, _timestamps(400))

    assert result["active_event_detected"] is False
    assert result["active_event_delay_seconds"] is None
    assert result["active_event_within60"] is False
    assert result["postwarmup_f1"] == 0.0


def test_detection_delay_preserves_onset_before_monitoring_start(runner) -> None:
    scores = np.zeros(400)
    scores[310:335] = 2.0
    labels = np.zeros(400, dtype=bool)
    labels[104:340] = True

    result = runner._supplement(scores, 1.0, labels, _timestamps(400))

    assert result["active_event_detected"] is True
    assert result["active_event_delay_seconds"] == pytest.approx(206.0)
    assert result["active_event_within60"] is False
