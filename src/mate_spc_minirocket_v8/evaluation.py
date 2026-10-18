"""Run-level metrics and frozen development gates for MiniROCKET v8."""

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

FOLD_RUN_IDS: dict[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)):
        raise RuntimeError("v8 fold identifiers changed")
    flattened = tuple(run for fold in range(5) for run in FOLD_RUN_IDS[fold])
    if len(flattened) != 100 or set(flattened) != set(range(1, 101)):
        raise RuntimeError("v8 folds must partition Run1..100 exactly")
    owner = {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}
    for component in ((14, 43, 98), (6, 76), (9, 88), (24, 33)):
        if len({owner[run] for run in component}) != 1:
            raise RuntimeError("a shared-seed component crosses v8 outer folds")


_validate_folds()


def _strict_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{name} must be an integer (boolean values are invalid)")
    return int(value)


def _strict_float(value: object, *, name: str) -> float:
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
    fit: tuple[int, ...]
    inner_healthy: tuple[int, ...]
    outer: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values, count in (
            ("fit", self.fit, 60),
            ("inner_healthy", self.inner_healthy, 20),
            ("outer", self.outer, 20),
        ):
            if type(values) is not tuple or len(values) != count:
                raise ValueError(f"{name} must be an exact {count}-run tuple")
            if len(set(values)) != count:
                raise ValueError(f"{name} contains duplicate runs")
            for index, value in enumerate(values):
                run = _strict_int(value, name=f"{name}[{index}]")
                if not 1 <= run <= 100:
                    raise ValueError("fold run IDs must be in 1..100")
        fit, inner, outer = set(self.fit), set(self.inner_healthy), set(self.outer)
        if fit & inner or fit & outer or inner & outer:
            raise ValueError("fold roles must be pairwise disjoint")
        if fit | inner | outer != set(range(1, 101)):
            raise ValueError("fold roles must cover Run1..100")

    def to_document(self) -> dict[str, list[int]]:
        return {
            "fit": list(self.fit),
            "inner_healthy": list(self.inner_healthy),
            "outer": list(self.outer),
        }


def fold_roles(outer_fold: int) -> FoldRoles:
    fold = _strict_int(outer_fold, name="outer_fold")
    if fold not in FOLD_RUN_IDS:
        raise ValueError("outer_fold must be in 0..4")
    inner = (fold + 1) % 5
    fit = tuple(
        sorted(
            run
            for candidate in range(5)
            if candidate not in (fold, inner)
            for run in FOLD_RUN_IDS[candidate]
        )
    )
    return FoldRoles(fit=fit, inner_healthy=FOLD_RUN_IDS[inner], outer=FOLD_RUN_IDS[fold])


@dataclass(frozen=True, slots=True)
class RunDecision:
    source_idv: int
    simulation_run: int
    run_maximum: float
    alarm_endpoint: int | None

    def __post_init__(self) -> None:
        source = _strict_int(self.source_idv, name="source_idv")
        run = _strict_int(self.simulation_run, name="simulation_run")
        maximum = _strict_float(self.run_maximum, name="run_maximum")
        if source not in CONDITION_IDS:
            raise ValueError("source_idv must be one of 1/3/9/15")
        if not 1 <= run <= 100:
            raise ValueError("simulation_run must be in 1..100")
        if maximum < 0.0:
            raise ValueError("run_maximum must be nonnegative")
        endpoint = None
        if self.alarm_endpoint is not None:
            endpoint = _strict_int(self.alarm_endpoint, name="alarm_endpoint")
            if not FAULT_ONSET <= endpoint <= MONITOR_END:
                raise ValueError("alarm_endpoint must be in 160..499")
        object.__setattr__(self, "source_idv", source)
        object.__setattr__(self, "simulation_run", run)
        object.__setattr__(self, "run_maximum", maximum)
        object.__setattr__(self, "alarm_endpoint", endpoint)

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


def nearest_rank(values: Sequence[float], probability: float) -> float:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise ValueError("values must be an iterable") from error
    if not raw:
        raise ValueError("values must be nonempty")
    array = np.asarray(
        [_strict_float(item, name=f"values[{index}]") for index, item in enumerate(raw)],
        dtype=np.float64,
    )
    p = _strict_float(probability, name="probability")
    if not 0.0 < p <= 1.0:
        raise ValueError("probability must be in (0,1]")
    rank = int(math.ceil(p * array.size))
    return float(np.sort(array, kind="stable")[rank - 1])


@dataclass(frozen=True, slots=True)
class HealthyMetrics:
    runs: int
    false_alarms: int

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
class PooledFaultDelayMetrics:
    runs: int
    median_delay_samples: float
    nearest_rank_p90_delay_samples: float

    def to_document(self) -> dict[str, int | float]:
        return {
            "runs": self.runs,
            "median_delay_samples": self.median_delay_samples,
            "nearest_rank_p90_delay_samples": self.nearest_rank_p90_delay_samples,
        }


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    healthy: HealthyMetrics
    faults: tuple[FaultMetrics, ...]
    pooled_fault_delays: PooledFaultDelayMetrics

    def fault(self, source_idv: int) -> FaultMetrics:
        source = _strict_int(source_idv, name="source_idv")
        for item in self.faults:
            if item.source_idv == source:
                return item
        raise ValueError("source_idv must be one of 3/9/15")

    def to_document(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy.to_document(),
            "faults": {str(item.source_idv): item.to_document() for item in self.faults},
            "pooled_fault_delays": self.pooled_fault_delays.to_document(),
        }


def _complete_rows(rows: Iterable[RunDecision]) -> tuple[RunDecision, ...]:
    try:
        records = tuple(rows)
    except TypeError as error:
        raise ValueError("rows must be iterable") from error
    if len(records) != 400:
        raise ValueError("evaluation requires exactly 400 outer decisions")
    if any(not isinstance(row, RunDecision) for row in records):
        raise ValueError("all rows must be RunDecision instances")
    identities = {(row.source_idv, row.simulation_run) for row in records}
    expected = {(condition, run) for condition in CONDITION_IDS for run in range(1, 101)}
    if identities != expected:
        raise ValueError("rows must cover each condition/run exactly once")
    return tuple(sorted(records, key=lambda row: (row.source_idv, row.simulation_run)))


def aggregate_metrics(rows: Iterable[RunDecision]) -> AggregateMetrics:
    records = _complete_rows(rows)
    healthy_rows = tuple(row for row in records if row.source_idv == 1)
    healthy = HealthyMetrics(
        runs=len(healthy_rows), false_alarms=sum(row.alarm for row in healthy_rows)
    )
    faults: list[FaultMetrics] = []
    pooled_delays: list[float] = []
    for source in FAULT_IDS:
        selected = tuple(row for row in records if row.source_idv == source)
        delays = [float(row.delay) for row in selected if row.delay is not None]
        pooled_delays.extend(delays)
        faults.append(
            FaultMetrics(
                source_idv=source,
                runs=len(selected),
                detections=sum(row.alarm for row in selected),
                median_delay_samples=float(np.median(np.asarray(delays, dtype=np.float64))),
                nearest_rank_p90_delay_samples=nearest_rank(delays, 0.90),
            )
        )
    pooled = PooledFaultDelayMetrics(
        runs=len(pooled_delays),
        median_delay_samples=float(
            np.median(np.asarray(pooled_delays, dtype=np.float64))
        ),
        nearest_rank_p90_delay_samples=nearest_rank(pooled_delays, 0.90),
    )
    if pooled.runs != 300:
        raise RuntimeError("v8 pooled fault-delay count changed")
    return AggregateMetrics(
        healthy=healthy, faults=tuple(faults), pooled_fault_delays=pooled
    )


@dataclass(frozen=True, slots=True)
class DevelopmentGateReport:
    checks: tuple[tuple[str, bool], ...]
    per_fold_healthy_false_alarms: tuple[int, ...]
    per_fold_fault_detection_rates: tuple[tuple[int, tuple[float, ...]], ...]
    aggregate_metrics: AggregateMetrics

    @property
    def all_pass(self) -> bool:
        return all(passed for _, passed in self.checks)

    @property
    def decision(self) -> str:
        return "GO" if self.all_pass else "DEVELOPMENT_NO_GO"

    def check(self, name: str) -> bool:
        for candidate, passed in self.checks:
            if candidate == name:
                return passed
        raise KeyError(name)

    def to_document(self) -> dict[str, Any]:
        return {
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "decision": self.decision,
            "per_fold_healthy_false_alarms": list(self.per_fold_healthy_false_alarms),
            "per_fold_fault_detection_rates": {
                str(source): list(rates)
                for source, rates in self.per_fold_fault_detection_rates
            },
            "aggregate_metrics": self.aggregate_metrics.to_document(),
        }


def evaluate_gates(
    rows: Iterable[RunDecision], numeric_folds_valid: Sequence[bool]
) -> DevelopmentGateReport:
    """Apply all frozen v8 gates with uncompensated logical-AND semantics."""

    records = _complete_rows(rows)
    try:
        flags = tuple(numeric_folds_valid)
    except TypeError as error:
        raise ValueError("numeric_folds_valid must be iterable") from error
    if len(flags) != 5 or any(type(flag) is not bool for flag in flags):
        raise ValueError("numeric_folds_valid must contain five exact booleans")
    metrics = aggregate_metrics(records)
    run_to_fold = {run: fold for fold, run_ids in FOLD_RUN_IDS.items() for run in run_ids}
    healthy_by_fold: list[int] = []
    fault_by_fold: dict[int, list[float]] = {source: [] for source in FAULT_IDS}
    for fold in range(5):
        selected = tuple(
            row for row in records if run_to_fold[row.simulation_run] == fold
        )
        healthy_by_fold.append(
            sum(row.alarm for row in selected if row.source_idv == 1)
        )
        for source in FAULT_IDS:
            detections = sum(
                row.alarm for row in selected if row.source_idv == source
            )
            fault_by_fold[source].append(detections / 20.0)

    checks: list[tuple[str, bool]] = [
        ("numeric_all_folds", all(flags)),
        ("healthy_overall", metrics.healthy.false_alarms <= 10),
        ("healthy_each_fold", all(count <= 4 for count in healthy_by_fold)),
    ]
    for source in FAULT_IDS:
        checks.extend(
            (
                (
                    f"fault_{source}_overall_detection",
                    metrics.fault(source).detections >= 85,
                ),
                (
                    f"fault_{source}_each_fold_detection",
                    all(rate >= 0.70 for rate in fault_by_fold[source]),
                ),
            )
        )
    checks.extend(
        (
            (
                "pooled_fault_median_delay",
                metrics.pooled_fault_delays.median_delay_samples <= 140.0,
            ),
            (
                "pooled_fault_p90_delay",
                metrics.pooled_fault_delays.nearest_rank_p90_delay_samples <= 220.0,
            ),
        )
    )
    return DevelopmentGateReport(
        checks=tuple(checks),
        per_fold_healthy_false_alarms=tuple(healthy_by_fold),
        per_fold_fault_detection_rates=tuple(
            (source, tuple(fault_by_fold[source])) for source in FAULT_IDS
        ),
        aggregate_metrics=metrics,
    )


# Compatibility with the neighboring v7 naming, while keeping evaluate_gates
# as the stable v8 API promised to the orchestration layer.
development_gates = evaluate_gates


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
    "PooledFaultDelayMetrics",
    "RunDecision",
    "aggregate_metrics",
    "development_gates",
    "evaluate_gates",
    "fold_roles",
    "nearest_rank",
]
