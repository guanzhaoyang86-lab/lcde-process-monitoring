from __future__ import annotations

import math

import numpy as np
import pytest

from mate_spc_farm_v7.evaluation import (
    CONDITION_IDS,
    FAULT_IDS,
    FOLD_RUN_IDS,
    MISS_DELAY,
    RunDecision,
    aggregate_metrics,
    development_gates,
    fold_roles,
    nearest_rank,
)


def _rows(
    *,
    healthy_alarms: set[int] | None = None,
    fault_misses: dict[int, set[int]] | None = None,
    fault_endpoint: int = 200,
) -> list[RunDecision]:
    healthy_alarms = set() if healthy_alarms is None else healthy_alarms
    fault_misses = {} if fault_misses is None else fault_misses
    result: list[RunDecision] = []
    for source_idv in CONDITION_IDS:
        for simulation_run in range(1, 101):
            if source_idv == 1:
                endpoint = 200 if simulation_run in healthy_alarms else None
            else:
                endpoint = (
                    None
                    if simulation_run in fault_misses.get(source_idv, set())
                    else fault_endpoint
                )
            result.append(
                RunDecision(
                    source_idv=source_idv,
                    simulation_run=simulation_run,
                    run_maximum=float(simulation_run),
                    alarm_endpoint=endpoint,
                )
            )
    return result


def test_frozen_folds_partition_runs_and_keep_seed_components_together() -> None:
    flattened = [run for fold in range(5) for run in FOLD_RUN_IDS[fold]]
    assert len(flattened) == len(set(flattened)) == 100
    assert set(flattened) == set(range(1, 101))
    owner = {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}
    for component in ((14, 43, 98), (6, 76), (9, 88), (24, 33)):
        assert len({owner[run] for run in component}) == 1


def test_nested_roles_are_exact_disjoint_and_deterministic() -> None:
    for outer_fold in range(5):
        first = fold_roles(outer_fold)
        second = fold_roles(outer_fold)
        assert first == second
        assert (len(first.fit), len(first.inner_healthy), len(first.outer)) == (60, 20, 20)
        assert not set(first.fit) & set(first.inner_healthy)
        assert not set(first.fit) & set(first.outer)
        assert not set(first.inner_healthy) & set(first.outer)
        assert set(first.fit) | set(first.inner_healthy) | set(first.outer) == set(
            range(1, 101)
        )


@pytest.mark.parametrize("value", [True, np.bool_(False), 0.0, "0", None])
def test_fold_roles_rejects_boolean_and_noninteger_identifiers(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        fold_roles(value)  # type: ignore[arg-type]


def test_farm_alarm_endpoint_can_start_at_onset_and_miss_is_340() -> None:
    assert RunDecision(3, 1, 1.0, 160).delay == 0
    assert RunDecision(9, 2, 1.0, 499).delay == 339
    assert RunDecision(15, 3, 1.0, None).delay == MISS_DELAY == 340
    assert RunDecision(1, 4, 1.0, 160).delay is None
    with pytest.raises(ValueError, match="160..499"):
        RunDecision(3, 1, 1.0, 159)
    with pytest.raises(ValueError, match="160..499"):
        RunDecision(3, 1, 1.0, 500)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_idv", True, "integer"),
        ("source_idv", 3.0, "integer"),
        ("simulation_run", np.bool_(True), "integer"),
        ("simulation_run", 1.0, "integer"),
        ("run_maximum", False, "real number"),
        ("run_maximum", "1.0", "real number"),
        ("run_maximum", math.inf, "finite"),
        ("run_maximum", math.nan, "finite"),
        ("run_maximum", -1.0, "nonnegative"),
        ("alarm_endpoint", True, "integer"),
        ("alarm_endpoint", 160.0, "integer"),
    ],
)
def test_run_decision_rejects_wrong_types_and_nonfinite_values(
    field: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "source_idv": 3,
        "simulation_run": 1,
        "run_maximum": 1.0,
        "alarm_endpoint": 160,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        RunDecision(**kwargs)  # type: ignore[arg-type]


def test_nearest_rank_is_noninterpolated_and_fail_closed() -> None:
    assert nearest_rank([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0
    assert nearest_rank(range(1, 101), 0.90) == 90.0
    for values in ([], [1.0, math.nan], [False, 1.0], ["1.0"]):
        with pytest.raises(ValueError):
            nearest_rank(values, 0.90)  # type: ignore[arg-type]
    for probability in (True, 0.0, 1.01, math.inf, "0.9"):
        with pytest.raises(ValueError):
            nearest_rank([1.0], probability)  # type: ignore[arg-type]


def test_complete_passing_rows_produce_typed_metrics_and_pass_every_gate() -> None:
    rows = _rows()
    metrics = aggregate_metrics(rows)
    assert metrics.healthy.runs == 100
    assert metrics.healthy.false_alarms == 0
    assert metrics.fault(15).detections == 100
    assert metrics.fault(15).median_delay_samples == 40.0
    assert metrics.fault(15).nearest_rank_p90_delay_samples == 40.0

    report = development_gates(rows, numeric_folds_valid=[True] * 5)
    assert report.all_pass
    assert all(report.check(name) for name, _ in report.checks)
    document = report.to_document()
    assert document["all_pass"] is True
    assert document["aggregate_metrics"]["faults"]["9"]["detection_rate"] == 1.0


def test_all_gates_are_and_combined_at_exact_inclusive_boundaries() -> None:
    # Two healthy alarms per fold is 10 overall and passes both inclusive limits.
    healthy_alarms = {
        run
        for fold in range(5)
        for run in FOLD_RUN_IDS[fold][:2]
    }
    # Two misses per fold leaves 90 prompt detections and therefore also makes
    # the non-interpolated p90 delay pass.
    misses = {
        fault: {
            run
            for fold in range(5)
            for run in FOLD_RUN_IDS[fold][:2]
        }
        for fault in FAULT_IDS
    }
    report = development_gates(
        _rows(healthy_alarms=healthy_alarms, fault_misses=misses),
        numeric_folds_valid=[True] * 5,
    )
    assert report.all_pass
    assert report.aggregate_metrics.healthy.false_alarms == 10
    assert all(report.aggregate_metrics.fault(fault).detections == 90 for fault in FAULT_IDS)

    # At exactly 85 detections the inclusive detection check passes, while the
    # independent p90-delay check makes the logically-ANDed report fail.
    exact_detection_boundary = {
        fault: {
            run
            for fold in range(5)
            for run in FOLD_RUN_IDS[fold][:3]
        }
        for fault in FAULT_IDS
    }
    boundary_report = development_gates(
        _rows(fault_misses=exact_detection_boundary),
        numeric_folds_valid=[True] * 5,
    )
    assert all(
        boundary_report.check(f"fault_{fault}_overall_detection")
        for fault in FAULT_IDS
    )
    assert not boundary_report.all_pass


def test_healthy_overall_and_per_fold_limits_are_independent() -> None:
    eleven = {
        run
        for fold in range(5)
        for run in FOLD_RUN_IDS[fold][: (3 if fold == 0 else 2)]
    }
    overall_failure = development_gates(
        _rows(healthy_alarms=eleven), numeric_folds_valid=[True] * 5
    )
    assert not overall_failure.check("healthy_overall")
    assert overall_failure.check("healthy_each_fold")
    assert not overall_failure.all_pass

    clustered = set(FOLD_RUN_IDS[0][:5])
    fold_failure = development_gates(
        _rows(healthy_alarms=clustered), numeric_folds_valid=[True] * 5
    )
    assert fold_failure.check("healthy_overall")
    assert not fold_failure.check("healthy_each_fold")
    assert not fold_failure.all_pass


def test_fault_overall_and_per_fold_detection_limits_are_independent() -> None:
    # Seven misses in one fold: overall 93% passes, fold 13/20 fails.
    misses = {9: set(FOLD_RUN_IDS[0][:7])}
    fold_failure = development_gates(
        _rows(fault_misses=misses), numeric_folds_valid=[True] * 5
    )
    assert fold_failure.check("fault_9_overall_detection")
    assert not fold_failure.check("fault_9_each_fold_detection")
    assert not fold_failure.all_pass

    # Four misses in every fold: each fold 16/20 passes but overall 80% fails.
    misses = {
        9: {
            run
            for fold in range(5)
            for run in FOLD_RUN_IDS[fold][:4]
        }
    }
    overall_failure = development_gates(
        _rows(fault_misses=misses), numeric_folds_valid=[True] * 5
    )
    assert not overall_failure.check("fault_9_overall_detection")
    assert overall_failure.check("fault_9_each_fold_detection")
    assert not overall_failure.all_pass


def test_delay_gates_include_misses_and_use_nearest_rank_p90() -> None:
    # 89 prompt alarms and 11 misses: median passes, nearest-rank p90 is a miss.
    misses = {15: set(range(90, 101))}
    report = development_gates(
        _rows(fault_misses=misses, fault_endpoint=160),
        numeric_folds_valid=[True] * 5,
    )
    assert report.aggregate_metrics.fault(15).median_delay_samples == 0.0
    assert report.aggregate_metrics.fault(15).nearest_rank_p90_delay_samples == 340.0
    assert report.check("fault_15_median_delay")
    assert not report.check("fault_15_p90_delay")
    assert not report.all_pass


@pytest.mark.parametrize(
    "flags",
    [
        [True] * 4,
        [True] * 6,
        [1] * 5,
        [np.bool_(True)] * 5,
        ["true"] * 5,
    ],
)
def test_numeric_fold_flags_require_exactly_five_exact_booleans(
    flags: list[object],
) -> None:
    with pytest.raises(ValueError):
        development_gates(_rows(), numeric_folds_valid=flags)  # type: ignore[arg-type]


def test_false_numeric_fold_flag_forces_no_go() -> None:
    report = development_gates(_rows(), numeric_folds_valid=[True] * 4 + [False])
    assert not report.check("numeric_all_folds")
    assert not report.all_pass


def test_aggregate_requires_exactly_one_row_per_condition_and_run() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="exactly 400"):
        aggregate_metrics(rows[:-1])
    duplicate = rows.copy()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(ValueError, match="exactly once"):
        aggregate_metrics(duplicate)
    wrong_type: list[object] = rows.copy()
    wrong_type[0] = {"source_idv": 1}
    with pytest.raises(ValueError, match="RunDecision"):
        aggregate_metrics(wrong_type)  # type: ignore[arg-type]
