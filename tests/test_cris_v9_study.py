from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from mate_spc_cris_v9.monitor import (
    MonitorConfig,
    PointCalibrator,
    TrajectoryThreshold,
)
from mate_spc_cris_v9.simulator import CRISBatch
from mate_spc_cris_v9.study import (
    CandidateDevelopment,
    candidate_specs,
    development_go,
    load_study_charter,
    model_arrays,
    seeds_for_role,
    select_candidate,
    simulate_role,
    summarize_paths,
)


PROJECT = Path(__file__).resolve().parents[1]
CHARTER_PATH = PROJECT / "protocols" / "cris_v9_four_tank_study_charter.json"


def _charter():
    return load_study_charter(CHARTER_PATH)


def _batch(kind: str, masks: list[np.ndarray]) -> CRISBatch:
    count = len(masks)
    length = int(masks[0].size)
    return CRISBatch(
        x=np.ones((count, length, 4), dtype=np.float64),
        c=np.ones((count, length, 4), dtype=np.float64),
        observed_mask=np.ones((count, length, 4), dtype=np.bool_),
        fault_mask=np.stack(masks),
        onset=np.full(count, 5, dtype=np.int64),
        fault_kind=(kind,) * count,
        seed=np.arange(1, count + 1, dtype=np.int64),
    )


def test_charter_roles_are_disjoint_and_confirmation_generation_is_guarded() -> None:
    charter = _charter()
    fit = seeds_for_role(charter, "healthy_model_fit")
    final = seeds_for_role(charter, "confirmation_healthy")
    assert len(fit) == 400
    assert len(final) == 200
    assert set(fit).isdisjoint(final)
    with pytest.raises(PermissionError, match="committed global claim"):
        simulate_role(charter, "confirmation_healthy")


def test_model_arrays_monitor_commands_but_condition_only_on_setpoints() -> None:
    batch = _batch("healthy", [np.zeros(8, dtype=np.bool_)])
    x, c, mask = model_arrays(batch)
    assert x.shape == (1, 8, 6)
    assert c.shape == (1, 8, 2)
    assert mask.shape == x.shape
    np.testing.assert_array_equal(x[:, :, :4], batch.x)
    np.testing.assert_array_equal(x[:, :, 4:], batch.c[:, :, 2:])
    np.testing.assert_array_equal(c, batch.c[:, :, :2])


def test_summary_uses_episode_alarm_onsets_not_inactive_occupancy() -> None:
    charter = _charter()
    fault = np.zeros(14, dtype=np.bool_)
    fault[5:7] = True
    batch = _batch("intermittent_pump", [fault])
    # One continuous pre-effect alarm crosses into the event but has no new
    # onset there.  A later inactive-gap onset lies beyond the fixed grace.
    score = np.zeros(14, dtype=np.float64)
    score[3:6] = 2.0
    score[13] = 2.0
    result = summarize_paths((score,), batch, 1.0, charter)
    assert result["recall"] == 0.0
    assert result["pre_effect_false_alarm_runs"] == 1
    assert result["inactive_gap_alarm_onsets"] == 1


def _candidate_result(index: int, *, health: float, covariate: float, recalls: tuple[float, ...]):
    specs = candidate_specs(_charter())
    config = specs[index].monitor_config
    rng = np.random.default_rng(10 + index)
    calibrator = PointCalibrator.fit(
        [rng.normal(size=(80, 3, 6)) for _ in range(2)], config
    )
    threshold = TrajectoryThreshold(
        value=0.0,
        alpha=0.05,
        calibration_run_count=20,
        ascending_rank=20,
        calibration_maxima=np.zeros(20, dtype=np.float64),
    )
    summaries: dict[str, dict[str, float | int | str]] = {
        "healthy": {"condition": "healthy", "alarm_rate": health},
        "covariate_shift_only": {
            "condition": "covariate_shift_only",
            "alarm_rate": covariate,
        },
    }
    for kind, recall in zip(
        ("sensor_bias", "intermittent_pump", "valve_stiction", "dependence_shift"),
        recalls,
        strict=True,
    ):
        summaries[kind] = {
            "condition": kind,
            "recall": recall,
            "median_relative_delay": 0.1,
            "pre_effect_false_alarm_runs": 0,
        }
    return CandidateDevelopment(
        specs[index], calibrator, threshold, MappingProxyType(summaries)
    )


def test_selection_and_confirmation_go_are_distinct_hard_gates() -> None:
    charter = _charter()
    first = _candidate_result(0, health=0.05, covariate=0.05, recalls=(0.9,) * 4)
    second = _candidate_result(1, health=0.05, covariate=0.05, recalls=(0.95,) * 4)
    third = _candidate_result(2, health=0.20, covariate=0.05, recalls=(1.0,) * 4)
    selected = select_candidate((first, second, third), charter)
    assert selected is second
    assert development_go(selected, charter)

    weak = _candidate_result(1, health=0.05, covariate=0.05, recalls=(0.9, 0.4, 0.9, 0.9))
    assert not development_go(weak, charter)
