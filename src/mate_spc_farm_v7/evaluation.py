"""Fail-closed run-level evaluation for FARM Rank-CUSUM v7.

This module deliberately contains no data-loading code.  It receives one outer
decision for every condition/run identity and applies the frozen five-fold
development gates.  Missed fault runs are assigned ``MISS_DELAY`` before delay
summaries are calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

import numpy as np


FAULT_IDS: tuple[int, ...] = (3, 9, 15)
CONDITION_IDS: tuple[int, ...] = (1, *FAULT_IDS)
FAULT_ONSET = 160
MONITOR_END = 499
MISS_DELAY = 340

# Reuse the collision-safe v5.1/v6 partition exactly.  In particular, known
# shared-seed connected components must never cross an outer boundary.
FOLD_RUN_IDS: dict[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)):
        raise RuntimeError("v7 fold identifiers changed")
    flattened = tuple(run for fold in range(5) for run in FOLD_RUN_IDS[fold])
    if len(flattened) != 100 or set(flattened) != set(range(1, 101)):
        raise RuntimeError("v7 folds must partition Run1..100 exactly")
    owner = {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}
    components = ((14, 43, 98), (6, 76), (9, 88), (24, 33))
    if any(len({owner[run] for run in component}) != 1 for component in components):
        raise RuntimeError("a shared-seed connected component crosses v7 folds")


_validate_folds()


def _strict_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{name} must be an integer (boolean values are invalid)")
    return int(value)


def _strict_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number (boolean values are invalid)")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FoldRoles:
    """Disjoint run identities assigned to one nested outer fold."""

    fit: tuple[int, ...]
    inner_healthy: tuple[int, ...]
    outer: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("fit", self.fit),
            ("inner_healthy", self.inner_healthy),
            ("outer", self.outer),
        ):
            if type(values) is not tuple:
                raise ValueError(f"{name} must be a tuple")
            for index, value in enumerate(values):
                run = _strict_integer(value, name=f"{name}[{index}]")
                if not 1 <= run <= 100:
                    raise ValueError(f"{name} run identifiers must be in 1..100")
        if len(self.fit) != 60 or len(self.inner_healthy) != 20 or len(self.outer) != 20:
            raise ValueError("fold roles must have exact sizes 60/20/20")
        groups = (set(self.fit), set(self.inner_healthy), set(self.outer))
        if any(len(group) != expected for group, expected in zip(groups, (60, 20, 20), strict=True)):
            raise ValueError("fold roles contain duplicate run identifiers")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("fold roles must be pairwise disjoint")
        if set.union(*groups) != set(range(1, 101)):
            raise ValueError("fold roles must cover Run1..100 exactly")

    def to_document(self) -> dict[str, list[int]]:
        return {
            "fit": list(self.fit),
            "inner_healthy": list(self.inner_healthy),
            "outer": list(self.outer),
        }


def fold_roles(outer_fold: int) -> FoldRoles:
    """Return the immutable 60-fit/20-inner/20-outer assignment."""

    outer = _strict_integer(outer_fold, name="outer_fold")
    if outer not in FOLD_RUN_IDS:
        raise ValueError("outer_fold must be in 0..4")
    inner = (outer + 1) % 5
    fit = tuple(
        sorted(
            run
            for fold in range(5)
            if fold not in (outer, inner)
            for run in FOLD_RUN_IDS[fold]
        )
    )
    return FoldRoles(fit=fit, inner_healthy=FOLD_RUN_IDS[inner], outer=FOLD_RUN_IDS[outer])


@dataclass(frozen=True, slots=True)
class RunDecision:
    """One thresholded outer decision.

    ``alarm_endpoint`` is ``None`` for a miss/no-alarm.  A FARM scan starts at
    the fault-onset-aligned endpoint 160, so a detected fault delay can be zero.
    """

    source_idv: int
    simulation_run: int
    run_maximum: float
    alarm_endpoint: int | None

    def __post_init__(self) -> None:
        source_idv = _strict_integer(self.source_idv, name="source_idv")
        simulation_run = _strict_integer(self.simulation_run, name="simulation_run")
        run_maximum = _strict_finite_real(self.run_maximum, name="run_maximum")
        if source_idv not in CONDITION_IDS:
            raise ValueError("source_idv must be one of 1/3/9/15")
        if not 1 <= simulation_run <= 100:
            raise ValueError("simulation_run must be in 1..100")
        if run_maximum < 0.0:
            raise ValueError("run_maximum must be nonnegative")
        alarm_endpoint: int | None = None
        if self.alarm_endpoint is not None:
            alarm_endpoint = _strict_integer(self.alarm_endpoint, name="alarm_endpoint")
            if not FAULT_ONSET <= alarm_endpoint <= MONITOR_END:
                raise ValueError(
                    f"alarm_endpoint must be in {FAULT_ONSET}..{MONITOR_END}"
                )
        object.__setattr__(self, "source_idv", source_idv)
        object.__setattr__(self, "simulation_run", simulation_run)
        object.__setattr__(self, "run_maximum", run_maximum)
        object.__setattr__(self, "alarm_endpoint", alarm_endpoint)

    @property
    def alarm(self) -> bool:
        return self.alarm_endpoint is not None

    @property
    def delay(self) -> int | None:
        if self.source_idv == 1:
            return None
        if self.alarm_endpoint is None:
            return MISS_DELAY
        return self.alarm_endpoint - FAULT_ONSET

    def to_document(self) -> dict[str, object]:
        return {
            "source_idv": self.source_idv,
            "simulation_run": self.simulation_run,
            "run_maximum": self.run_maximum,
            "alarm": self.alarm,
            "alarm_endpoint": self.alarm_endpoint,
            "delay_samples": self.delay,
        }


@dataclass(frozen=True, slots=True)
class HealthyMetrics:
    runs: int
    false_alarms: int

    def __post_init__(self) -> None:
        runs = _strict_integer(self.runs, name="runs")
        false_alarms = _strict_integer(self.false_alarms, name="false_alarms")
        if runs < 1 or not 0 <= false_alarms <= runs:
            raise ValueError(
                "healthy counts must satisfy runs >= 1 and 0 <= false_alarms <= runs"
            )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "false_alarms", false_alarms)

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.runs

    def to_document(self) -> dict[str, int | float]:
        return {
            "runs": self.runs,
            "false_alarms": self.false_alarms,
            "false_alarm_rate": self.false_alarm_rate,
        }


@dataclass(frozen=True, slots=True)
class FaultMetrics:
    source_idv: int
    runs: int
    detections: int
    median_delay_samples: float
    nearest_rank_p90_delay_samples: float

    def __post_init__(self) -> None:
        source_idv = _strict_integer(self.source_idv, name="source_idv")
        runs = _strict_integer(self.runs, name="runs")
        detections = _strict_integer(self.detections, name="detections")
        median_delay = _strict_finite_real(
            self.median_delay_samples, name="median_delay_samples"
        )
        p90_delay = _strict_finite_real(
            self.nearest_rank_p90_delay_samples,
            name="nearest_rank_p90_delay_samples",
        )
        if source_idv not in FAULT_IDS:
            raise ValueError("source_idv must be one of 3/9/15")
        if runs < 1 or not 0 <= detections <= runs:
            raise ValueError(
                "fault counts must satisfy runs >= 1 and 0 <= detections <= runs"
            )
        if not 0.0 <= median_delay <= MISS_DELAY or not 0.0 <= p90_delay <= MISS_DELAY:
            raise ValueError(f"delay metrics must be in 0..{MISS_DELAY}")
        object.__setattr__(self, "source_idv", source_idv)
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "median_delay_samples", median_delay)
        object.__setattr__(self, "nearest_rank_p90_delay_samples", p90_delay)

    @property
    def detection_rate(self) -> float:
        return self.detections / self.runs

    def to_document(self) -> dict[str, int | float]:
        return {
            "runs": self.runs,
            "detections": self.detections,
            "detection_rate": self.detection_rate,
            "median_delay_samples": self.median_delay_samples,
            "nearest_rank_p90_delay_samples": self.nearest_rank_p90_delay_samples,
        }


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    healthy: HealthyMetrics
    faults: tuple[FaultMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.healthy, HealthyMetrics):
            raise ValueError("healthy must be HealthyMetrics")
        if type(self.faults) is not tuple or any(
            not isinstance(item, FaultMetrics) for item in self.faults
        ):
            raise ValueError("faults must be a tuple of FaultMetrics")
        if tuple(item.source_idv for item in self.faults) != FAULT_IDS:
            raise ValueError("fault metrics must be ordered as Fault 3/9/15")

    def fault(self, source_idv: int) -> FaultMetrics:
        fault = _strict_integer(source_idv, name="source_idv")
        for item in self.faults:
            if item.source_idv == fault:
                return item
        raise ValueError("source_idv must be one of 3/9/15")

    def to_document(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy.to_document(),
            "faults": {str(item.source_idv): item.to_document() for item in self.faults},
        }


@dataclass(frozen=True, slots=True)
class DevelopmentGateReport:
    checks: tuple[tuple[str, bool], ...]
    per_fold_healthy_false_alarms: tuple[int, ...]
    per_fold_fault_detection_rates: tuple[tuple[int, tuple[float, ...]], ...]
    aggregate_metrics: AggregateMetrics

    def __post_init__(self) -> None:
        if type(self.checks) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in self.checks
        ):
            raise ValueError("checks must be (name, boolean) tuples")
        if any(not isinstance(name, str) or not name for name, _ in self.checks):
            raise ValueError("gate check names must be nonempty strings")
        names = tuple(name for name, _ in self.checks)
        if len(names) != len(set(names)):
            raise ValueError("gate check names must be unique")
        if any(type(passed) is not bool for _, passed in self.checks):
            raise ValueError("gate check results must be exact booleans")
        if len(self.per_fold_healthy_false_alarms) != 5:
            raise ValueError("healthy fold metrics must contain five values")
        for index, value in enumerate(self.per_fold_healthy_false_alarms):
            count = _strict_integer(value, name=f"healthy_fold[{index}]")
            if not 0 <= count <= 20:
                raise ValueError("healthy fold counts must be in 0..20")
        if tuple(fault for fault, _ in self.per_fold_fault_detection_rates) != FAULT_IDS:
            raise ValueError("fault fold metrics must be ordered as Fault 3/9/15")
        if any(len(rates) != 5 for _, rates in self.per_fold_fault_detection_rates):
            raise ValueError("each fault must contain five fold detection rates")
        for fault, rates in self.per_fold_fault_detection_rates:
            _strict_integer(fault, name="fault fold source_idv")
            for index, value in enumerate(rates):
                rate = _strict_finite_real(value, name=f"fault_{fault}_fold[{index}]")
                if not 0.0 <= rate <= 1.0:
                    raise ValueError("fault fold detection rates must be in [0,1]")
        if not isinstance(self.aggregate_metrics, AggregateMetrics):
            raise ValueError("aggregate_metrics must be AggregateMetrics")

    @property
    def all_pass(self) -> bool:
        return all(passed for _, passed in self.checks)

    def check(self, name: str) -> bool:
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        for candidate, passed in self.checks:
            if candidate == name:
                return passed
        raise KeyError(name)

    def to_document(self) -> dict[str, Any]:
        return {
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "per_fold_healthy_false_alarms": list(
                self.per_fold_healthy_false_alarms
            ),
            "per_fold_fault_detection_rates": {
                str(fault): list(rates)
                for fault, rates in self.per_fold_fault_detection_rates
            },
            "aggregate_metrics": self.aggregate_metrics.to_document(),
        }


def nearest_rank(values: Sequence[float], probability: float) -> float:
    """Return the non-interpolated nearest-rank quantile."""

    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise ValueError("values must be an iterable of finite real numbers") from error
    if not raw_values:
        raise ValueError("values must be nonempty")
    array = np.asarray(
        [
            _strict_finite_real(value, name=f"values[{index}]")
            for index, value in enumerate(raw_values)
        ],
        dtype=np.float64,
    )
    p = _strict_finite_real(probability, name="probability")
    if not 0.0 < p <= 1.0:
        raise ValueError("probability must be in (0,1]")
    rank = int(math.ceil(p * array.size))
    return float(np.sort(array)[rank - 1])


def _validate_complete_rows(rows: Iterable[RunDecision]) -> tuple[RunDecision, ...]:
    try:
        records = tuple(rows)
    except TypeError as error:
        raise ValueError("outer decisions must be an iterable") from error
    if any(not isinstance(row, RunDecision) for row in records):
        raise ValueError("all outer decisions must be RunDecision instances")
    if len(records) != 400:
        raise ValueError("aggregate evaluation requires exactly 400 outer decisions")
    identities = {(row.source_idv, row.simulation_run) for row in records}
    expected = {(idv, run) for idv in CONDITION_IDS for run in range(1, 101)}
    if identities != expected:
        raise ValueError("outer decisions must cover each condition/run exactly once")
    return tuple(sorted(records, key=lambda row: (row.source_idv, row.simulation_run)))


def aggregate_metrics(rows: Iterable[RunDecision]) -> AggregateMetrics:
    """Aggregate a complete set of 400 outer decisions."""

    records = _validate_complete_rows(rows)
    healthy_rows = tuple(row for row in records if row.source_idv == 1)
    healthy = HealthyMetrics(
        runs=len(healthy_rows), false_alarms=sum(row.alarm for row in healthy_rows)
    )
    fault_metrics: list[FaultMetrics] = []
    for fault in FAULT_IDS:
        selected = tuple(row for row in records if row.source_idv == fault)
        delays = tuple(float(row.delay) for row in selected if row.delay is not None)
        fault_metrics.append(
            FaultMetrics(
                source_idv=fault,
                runs=len(selected),
                detections=sum(row.alarm for row in selected),
                median_delay_samples=float(np.median(np.asarray(delays))),
                nearest_rank_p90_delay_samples=nearest_rank(delays, 0.90),
            )
        )
    return AggregateMetrics(healthy=healthy, faults=tuple(fault_metrics))


def development_gates(
    rows: Iterable[RunDecision], *, numeric_folds_valid: Sequence[bool]
) -> DevelopmentGateReport:
    """Apply every frozen development gate with logical AND semantics."""

    records = _validate_complete_rows(rows)
    try:
        supplied_flags = tuple(numeric_folds_valid)
    except TypeError as error:
        raise ValueError("numeric_folds_valid must be an iterable") from error
    if len(supplied_flags) != 5:
        raise ValueError("numeric_folds_valid must contain exactly five flags")
    if any(type(value) is not bool for value in supplied_flags):
        raise ValueError("numeric_folds_valid entries must be exact booleans")

    metrics = aggregate_metrics(records)
    run_to_fold = {run: fold for fold, ids in FOLD_RUN_IDS.items() for run in ids}
    healthy_by_fold: list[int] = []
    fault_by_fold: dict[int, list[float]] = {fault: [] for fault in FAULT_IDS}
    for fold in range(5):
        selected_fold = tuple(
            row for row in records if run_to_fold[row.simulation_run] == fold
        )
        healthy_by_fold.append(
            sum(row.alarm for row in selected_fold if row.source_idv == 1)
        )
        for fault in FAULT_IDS:
            detected = sum(
                row.alarm for row in selected_fold if row.source_idv == fault
            )
            fault_by_fold[fault].append(detected / 20.0)

    checks: list[tuple[str, bool]] = [
        ("numeric_all_folds", all(supplied_flags)),
        ("healthy_overall", metrics.healthy.false_alarms <= 10),
        ("healthy_each_fold", all(value <= 4 for value in healthy_by_fold)),
    ]
    for fault in FAULT_IDS:
        item = metrics.fault(fault)
        checks.extend(
            (
                (f"fault_{fault}_overall_detection", item.detections >= 85),
                (
                    f"fault_{fault}_each_fold_detection",
                    all(rate >= 0.70 for rate in fault_by_fold[fault]),
                ),
                (f"fault_{fault}_median_delay", item.median_delay_samples <= 140.0),
                (
                    f"fault_{fault}_p90_delay",
                    item.nearest_rank_p90_delay_samples <= 220.0,
                ),
            )
        )

    return DevelopmentGateReport(
        checks=tuple(checks),
        per_fold_healthy_false_alarms=tuple(healthy_by_fold),
        per_fold_fault_detection_rates=tuple(
            (fault, tuple(fault_by_fold[fault])) for fault in FAULT_IDS
        ),
        aggregate_metrics=metrics,
    )


__all__ = [
    "AggregateMetrics",
    "CONDITION_IDS",
    "DevelopmentGateReport",
    "FAULT_IDS",
    "FAULT_ONSET",
    "FOLD_RUN_IDS",
    "FaultMetrics",
    "FoldRoles",
    "HealthyMetrics",
    "MISS_DELAY",
    "MONITOR_END",
    "RunDecision",
    "aggregate_metrics",
    "development_gates",
    "fold_roles",
    "nearest_rank",
]
