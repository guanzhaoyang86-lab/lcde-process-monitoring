from __future__ import annotations

import numpy as np
import pytest

from mate_spc_pcfcycle.evaluation import (
    CONDITION_IDS,
    FOLD_RUN_IDS,
    RunDecision,
    development_gates,
    exact_one_sided_clopper_pearson_upper,
    fold_roles,
    nearest_rank,
)


def _passing_rows() -> list[RunDecision]:
    rows: list[RunDecision] = []
    for idv in CONDITION_IDS:
        for run in range(1, 101):
            endpoint = None if idv == 1 else 200
            rows.append(RunDecision(idv, run, float(run), endpoint))
    return rows


def test_collision_safe_folds_partition_every_run() -> None:
    values = [run for fold in range(5) for run in FOLD_RUN_IDS[fold]]
    assert len(values) == len(set(values)) == 100
    assert set(values) == set(range(1, 101))
    owner = {run: fold for fold, ids in FOLD_RUN_IDS.items() for run in ids}
    for component in ((14, 43, 98), (6, 76), (9, 88), (24, 33)):
        assert len({owner[run] for run in component}) == 1


def test_nested_fold_roles_have_exact_disjoint_sizes() -> None:
    for outer in range(5):
        roles = fold_roles(outer)
        assert {name: len(values) for name, values in roles.items()} == {
            "fit": 60,
            "inner_healthy": 20,
            "outer": 20,
        }
        assert not (set(roles["fit"]) & set(roles["inner_healthy"]))
        assert not (set(roles["fit"]) & set(roles["outer"]))
        assert not (set(roles["inner_healthy"]) & set(roles["outer"]))


def test_run_decision_uses_strict_fault_delay_semantics() -> None:
    assert RunDecision(3, 1, 2.0, None).delay == 340
    assert RunDecision(15, 2, 3.0, 164).delay == 4
    assert RunDecision(1, 3, 1.0, None).delay is None


def test_passing_rows_pass_all_development_gates() -> None:
    report = development_gates(_passing_rows(), numeric_folds_valid=[True] * 5)
    assert report["all_pass"]
    assert report["aggregate_metrics"]["faults"]["15"]["detection_rate"] == 1.0


def test_per_fold_fault_gate_and_numeric_gate_are_enforced() -> None:
    rows = _passing_rows()
    failed_runs = set(FOLD_RUN_IDS[0][:7])
    rows = [
        RunDecision(row.source_idv, row.simulation_run, row.run_maximum, None)
        if row.source_idv == 9 and row.simulation_run in failed_runs
        else row
        for row in rows
    ]
    report = development_gates(rows, numeric_folds_valid=[True] * 5)
    assert not report["checks"]["fault_9_each_fold_detection"]
    assert not report["all_pass"]
    numeric = development_gates(_passing_rows(), numeric_folds_valid=[True] * 4 + [False])
    assert not numeric["checks"]["numeric_all_folds"]


def test_nearest_rank_is_noninterpolated() -> None:
    assert nearest_rank([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0


def test_clopper_pearson_matches_closed_form_for_zero_events() -> None:
    observed = exact_one_sided_clopper_pearson_upper(0, 100)
    assert observed == pytest.approx(1.0 - 0.05 ** (1.0 / 100.0), abs=1e-12)
    assert exact_one_sided_clopper_pearson_upper(100, 100) == 1.0


def test_clopper_pearson_known_threshold_statements() -> None:
    assert exact_one_sided_clopper_pearson_upper(4, 100) < 0.10
    assert exact_one_sided_clopper_pearson_upper(5, 100) > 0.10
    assert exact_one_sided_clopper_pearson_upper(1, 100) < 0.05
    assert exact_one_sided_clopper_pearson_upper(2, 100) > 0.05


def test_evaluation_rejects_incomplete_outer_rows() -> None:
    with pytest.raises(ValueError):
        development_gates(_passing_rows()[:-1], numeric_folds_valid=[True] * 5)


def test_folds_reject_non_integer_identifier() -> None:
    with pytest.raises(ValueError):
        fold_roles(True)
    with pytest.raises(ValueError):
        fold_roles(5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_idv": True},
        {"source_idv": 3.0},
        {"simulation_run": True},
        {"simulation_run": 1.0},
        {"alarm_endpoint": True},
        {"alarm_endpoint": 164.0},
        {"attributed_fault": True},
        {"attributed_fault": 3.0},
    ],
)
def test_run_decision_rejects_boolean_and_noninteger_identifiers(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "source_idv": 3,
        "simulation_run": 1,
        "run_maximum": 1.0,
        "alarm_endpoint": 164,
        "attributed_fault": 3,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match="integer"):
        RunDecision(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("flags", [[1] * 5, [0.0] * 5, ["false"] * 5])
def test_development_gates_rejects_nonboolean_numeric_flags(flags: list[object]) -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        development_gates(_passing_rows(), numeric_folds_valid=flags)  # type: ignore[arg-type]
