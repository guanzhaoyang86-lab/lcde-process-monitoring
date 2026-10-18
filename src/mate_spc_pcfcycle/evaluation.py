"""Run-level evaluation and immutable fold roles for v6."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FAULT_IDS = (3, 9, 15)
CONDITION_IDS = (1, *FAULT_IDS)
MISS_DELAY = 340
FAULT_ONSET = 160

# Exact seed-connected v5.1 folds.  Reusing these blocks prevents every known
# duplicated RNG-seed owner from crossing an outer boundary.
FOLD_RUN_IDS: Mapping[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}


def _validate_folds() -> None:
    if set(FOLD_RUN_IDS) != set(range(5)):
        raise RuntimeError("v6 fold identifiers changed")
    flat = tuple(run for fold in range(5) for run in FOLD_RUN_IDS[fold])
    if len(flat) != 100 or set(flat) != set(range(1, 101)):
        raise RuntimeError("v6 folds must partition Run1..100")
    components = ((14, 43, 98), (6, 76), (9, 88), (24, 33))
    owner = {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}
    if any(len({owner[run] for run in component}) != 1 for component in components):
        raise RuntimeError("a shared-seed connected component crosses v6 folds")


_validate_folds()


def _strict_integer_identifier(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def fold_roles(outer_fold: int) -> dict[str, tuple[int, ...]]:
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, (int, np.integer)):
        raise ValueError("outer_fold must be an integer")
    outer = int(outer_fold)
    if outer not in FOLD_RUN_IDS:
        raise ValueError("outer_fold must be in 0..4")
    inner = (outer + 1) % 5
    fit = tuple(
        run
        for fold in range(5)
        if fold not in (outer, inner)
        for run in FOLD_RUN_IDS[fold]
    )
    return {
        "fit": tuple(sorted(fit)),
        "inner_healthy": FOLD_RUN_IDS[inner],
        "outer": FOLD_RUN_IDS[outer],
    }


@dataclass(frozen=True)
class RunDecision:
    source_idv: int
    simulation_run: int
    run_maximum: float
    alarm_endpoint: int | None
    attributed_fault: int | None = None

    def __post_init__(self) -> None:
        source_idv = _strict_integer_identifier(self.source_idv, name="source_idv")
        simulation_run = _strict_integer_identifier(
            self.simulation_run, name="simulation_run"
        )
        if source_idv not in CONDITION_IDS:
            raise ValueError("source_idv must be one of 1/3/9/15")
        if not 1 <= simulation_run <= 100:
            raise ValueError("simulation_run must be in 1..100")
        if not math.isfinite(float(self.run_maximum)):
            raise ValueError("run_maximum must be finite")
        alarm_endpoint = None
        if self.alarm_endpoint is not None:
            alarm_endpoint = _strict_integer_identifier(
                self.alarm_endpoint, name="alarm_endpoint"
            )
            if not 164 <= alarm_endpoint <= 499:
                raise ValueError("alarm_endpoint must be in 164..499")
        attributed_fault = None
        if self.attributed_fault is not None:
            attributed_fault = _strict_integer_identifier(
                self.attributed_fault, name="attributed_fault"
            )
            if attributed_fault not in FAULT_IDS:
                raise ValueError("attributed_fault must be 3/9/15 or None")
        object.__setattr__(self, "source_idv", source_idv)
        object.__setattr__(self, "simulation_run", simulation_run)
        object.__setattr__(self, "alarm_endpoint", alarm_endpoint)
        object.__setattr__(self, "attributed_fault", attributed_fault)

    @property
    def alarm(self) -> bool:
        return self.alarm_endpoint is not None

    @property
    def delay(self) -> int | None:
        if self.source_idv == 1:
            return None
        return MISS_DELAY if self.alarm_endpoint is None else int(self.alarm_endpoint) - FAULT_ONSET

    def to_document(self) -> dict[str, object]:
        return {
            "source_idv": self.source_idv,
            "simulation_run": self.simulation_run,
            "run_maximum": float(self.run_maximum),
            "alarm": self.alarm,
            "alarm_endpoint": self.alarm_endpoint,
            "delay_samples": self.delay,
            "attributed_fault": self.attributed_fault,
        }


def nearest_rank(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    p = float(probability)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite vector")
    if not 0.0 < p <= 1.0 or not math.isfinite(p):
        raise ValueError("probability must be finite in (0,1]")
    rank = int(math.ceil(p * array.size))
    return float(np.sort(array)[rank - 1])


def _validate_complete_rows(rows: Iterable[RunDecision]) -> tuple[RunDecision, ...]:
    result = tuple(rows)
    if len(result) != 400:
        raise ValueError("aggregate evaluation requires 400 outer decisions")
    identities = {(row.source_idv, row.simulation_run) for row in result}
    expected = {(idv, run) for idv in CONDITION_IDS for run in range(1, 101)}
    if identities != expected:
        raise ValueError("outer decisions do not cover each condition/run exactly once")
    return tuple(sorted(result, key=lambda row: (row.source_idv, row.simulation_run)))


def aggregate_metrics(rows: Iterable[RunDecision]) -> dict[str, Any]:
    records = _validate_complete_rows(rows)
    healthy = tuple(row for row in records if row.source_idv == 1)
    faults: dict[str, Any] = {}
    for fault in FAULT_IDS:
        selected = tuple(row for row in records if row.source_idv == fault)
        delays = [float(row.delay) for row in selected]
        detected = sum(row.alarm for row in selected)
        faults[str(fault)] = {
            "runs": 100,
            "detections": detected,
            "detection_rate": detected / 100.0,
            "median_delay_samples": float(np.median(delays)),
            "nearest_rank_p90_delay_samples": nearest_rank(delays, 0.90),
        }
    false_alarms = sum(row.alarm for row in healthy)
    return {
        "healthy": {
            "runs": 100,
            "false_alarms": false_alarms,
            "false_alarm_rate": false_alarms / 100.0,
        },
        "faults": faults,
    }


def development_gates(
    rows: Iterable[RunDecision], *, numeric_folds_valid: Sequence[bool]
) -> dict[str, Any]:
    records = _validate_complete_rows(rows)
    supplied_flags = tuple(numeric_folds_valid)
    if len(supplied_flags) != 5:
        raise ValueError("numeric_folds_valid must contain five flags")
    if any(not isinstance(value, (bool, np.bool_)) for value in supplied_flags):
        raise ValueError("numeric_folds_valid entries must be boolean")
    valid = tuple(bool(value) for value in supplied_flags)
    metrics = aggregate_metrics(records)
    run_to_fold = {run: fold for fold, ids in FOLD_RUN_IDS.items() for run in ids}
    per_fold_healthy = []
    per_fold_fault: dict[str, list[float]] = {str(fault): [] for fault in FAULT_IDS}
    for fold in range(5):
        fold_rows = tuple(row for row in records if run_to_fold[row.simulation_run] == fold)
        per_fold_healthy.append(sum(row.alarm for row in fold_rows if row.source_idv == 1))
        for fault in FAULT_IDS:
            selected = tuple(row for row in fold_rows if row.source_idv == fault)
            per_fold_fault[str(fault)].append(sum(row.alarm for row in selected) / 20.0)
    checks: dict[str, bool] = {
        "numeric_all_folds": all(valid),
        "healthy_overall": metrics["healthy"]["false_alarms"] <= 10,
        "healthy_each_fold": all(value <= 4 for value in per_fold_healthy),
    }
    for fault in FAULT_IDS:
        item = metrics["faults"][str(fault)]
        checks[f"fault_{fault}_overall_detection"] = item["detection_rate"] >= 0.85
        checks[f"fault_{fault}_each_fold_detection"] = all(
            value >= 0.70 for value in per_fold_fault[str(fault)]
        )
        checks[f"fault_{fault}_median_delay"] = item["median_delay_samples"] <= 140.0
        checks[f"fault_{fault}_p90_delay"] = (
            item["nearest_rank_p90_delay_samples"] <= 220.0
        )
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "per_fold_healthy_false_alarms": per_fold_healthy,
        "per_fold_fault_detection_rates": per_fold_fault,
        "aggregate_metrics": metrics,
    }


def exact_one_sided_clopper_pearson_upper(
    successes: int, trials: int, *, confidence: float = 0.95
) -> float:
    """Exact one-sided binomial upper confidence bound without SciPy."""

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("counts must be integers")
    if not isinstance(successes, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError("counts must be integers")
    k, n = int(successes), int(trials)
    level = float(confidence)
    if n < 1 or k < 0 or k > n:
        raise ValueError("counts must satisfy 0 <= successes <= trials")
    if not 0.0 < level < 1.0 or not math.isfinite(level):
        raise ValueError("confidence must be finite in (0,1)")
    if k == n:
        return 1.0
    alpha = 1.0 - level

    def cdf(probability: float) -> float:
        return sum(
            math.comb(n, index)
            * probability**index
            * (1.0 - probability) ** (n - index)
            for index in range(k + 1)
        )

    lower, upper = 0.0, 1.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if cdf(middle) > alpha:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


__all__ = [
    "CONDITION_IDS",
    "FAULT_IDS",
    "FAULT_ONSET",
    "FOLD_RUN_IDS",
    "MISS_DELAY",
    "RunDecision",
    "aggregate_metrics",
    "development_gates",
    "exact_one_sided_clopper_pearson_upper",
    "fold_roles",
    "nearest_rank",
]
