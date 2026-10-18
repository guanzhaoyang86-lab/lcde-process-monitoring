"""Validation and aggregation for classic TEP missingness stress runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Iterable


_FAULT_NAME = re.compile(r"^TEP-fault-(\d+)-test$")
_CALIBRATION_RTOL = 1e-10
_CALIBRATION_ATOL = 1e-9


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite_values(values: Iterable[object]) -> list[float]:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    return finite


def _mean(values: Iterable[object]) -> float | None:
    finite = _finite_values(values)
    return sum(finite) / len(finite) if finite else None


def _median(values: Iterable[object]) -> float | None:
    finite = _finite_values(values)
    return float(median(finite)) if finite else None


def _numerically_compatible(left: object, right: object) -> bool:
    """Compare fitted summaries while tolerating harmless BLAS round-off."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=_CALIBRATION_RTOL,
            abs_tol=_CALIBRATION_ATOL,
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _numerically_compatible(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numerically_compatible(a, b) for a, b in zip(left, right)
        )
    return left == right


def _dataset_signature(
    metadata: object, *, include_name: bool
) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata must be an object")
    fields = ["rows", "features", "fault_samples", "sample_period_seconds"]
    if include_name:
        fields.insert(0, "name")
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(f"dataset metadata is missing fields: {missing}")
    return {field: metadata[field] for field in fields}


def _method_summary(entries: list[dict[str, object]]) -> dict[str, float | int | None]:
    monitoring = [entry["monitoring"] for entry in entries]
    if not all(isinstance(value, dict) for value in monitoring):
        raise ValueError("every method entry must contain a monitoring object")
    monitors = [value for value in monitoring if isinstance(value, dict)]
    event_count = sum(int(value["event_count"]) for value in monitors)
    events_detected = sum(int(value["events_detected"]) for value in monitors)
    false_episodes = sum(int(value["false_alarm_episodes"]) for value in monitors)
    normal_hours = sum(float(value["monitored_normal_hours"]) for value in monitors)
    delays = [
        float(value["median_delay_samples"])
        for value in monitors
        if value.get("median_delay_samples") is not None
    ]
    imputation_sse = 0.0
    imputation_count = 0.0
    for entry in entries:
        imputation = entry.get("imputation")
        if not isinstance(imputation, dict):
            continue
        count = float(imputation.get("n_hidden", 0.0))
        rmse = imputation.get("rmse")
        if count > 0.0 and rmse is not None and math.isfinite(float(rmse)):
            imputation_sse += float(rmse) ** 2 * count
            imputation_count += count
    return {
        "detected_event_evaluations": events_detected,
        "event_evaluation_count": event_count,
        "event_evaluation_recall": (
            events_detected / event_count if event_count else None
        ),
        "median_delay_samples_among_detected": _median(delays),
        "false_alarm_episodes": false_episodes,
        "false_alarm_episodes_per_100h": (
            100.0 * false_episodes / normal_hours if normal_hours > 0.0 else None
        ),
        "monitored_normal_hours": normal_hours,
        "mean_point_f1": _mean(value.get("point_f1") for value in monitors),
        "mean_decision_coverage": _mean(
            value.get("decision_coverage") for value in monitors
        ),
        "mean_unknown_rate": _mean(entry.get("unknown_rate") for entry in entries),
        "pooled_scale_normalized_imputation_rmse": (
            math.sqrt(imputation_sse / imputation_count)
            if imputation_count > 0.0
            else None
        ),
        "imputation_hidden_cells": int(imputation_count),
        "case_count": len(entries),
    }


def _summaries(
    records: list[dict[str, object]], methods: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    return {
        method: _method_summary(
            [
                record["methods"][method]
                for record in records
                if isinstance(record.get("methods"), dict)
            ]
        )
        for method in methods
    }


def aggregate_tep_fault_files(
    paths: Iterable[str | Path],
    *,
    expected_faults: Iterable[int] = range(1, 22),
) -> dict[str, object]:
    """Validate compatible per-fault reports and return publication-safe summaries."""
    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("no TEP fault reports were supplied")
    expected = tuple(sorted({int(value) for value in expected_faults}))
    if not expected or expected[0] < 1:
        raise ValueError("expected_faults must contain positive fault identifiers")

    reports: dict[int, tuple[Path, dict[str, object]]] = {}
    for path in resolved:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        test = report.get("test")
        if not isinstance(test, dict):
            raise ValueError(f"{path.name}: missing test metadata")
        match = _FAULT_NAME.fullmatch(str(test.get("name", "")))
        if match is None:
            raise ValueError(f"{path.name}: invalid TEP test name")
        fault_id = int(match.group(1))
        if fault_id in reports:
            raise ValueError(f"duplicate fault report: {fault_id}")
        records = report.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path.name}: records must be a non-empty list")
        reports[fault_id] = (path, report)

    input_root = resolved[0].parent
    if any(path.parent != input_root for path in resolved):
        raise ValueError("all TEP fault reports must be in one input directory")

    actual = tuple(sorted(reports))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"fault coverage mismatch; missing={missing}, extra={extra}")

    first = reports[expected[0]][1]
    reference_configuration = _canonical(first.get("configuration"))
    reference_protocol = _canonical(first.get("evaluation_protocol"))
    reference_method = first.get("method")
    reference_train = _dataset_signature(first.get("train"), include_name=True)
    reference_calibration = _dataset_signature(
        first.get("calibration"), include_name=True
    )
    reference_test = _dataset_signature(first.get("test"), include_name=False)
    reference_calibration_summary = first.get("calibration_summary")
    reference_ablation_summaries = first.get(
        "ablation_calibration_summaries"
    )
    if not isinstance(reference_calibration_summary, dict):
        raise ValueError("reference report has no fitted calibration summary")
    if not isinstance(reference_ablation_summaries, dict):
        raise ValueError("reference report has no fitted ablation summaries")
    reference_records = first["records"]
    if not isinstance(reference_records, list):
        raise ValueError("reference records are invalid")
    reference_pairs = tuple(
        sorted((str(record["scenario_id"]), int(record["seed"])) for record in reference_records)
    )
    if len(reference_pairs) != len(set(reference_pairs)):
        raise ValueError("reference report contains duplicate scenario/seed cases")
    first_methods = reference_records[0].get("methods")
    if not isinstance(first_methods, dict) or not first_methods:
        raise ValueError("reference report has no methods")
    methods = tuple(first_methods)
    scenario_definitions: dict[str, object] = {}
    all_records: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    by_fault: dict[str, object] = {}

    for fault_id in expected:
        path, report = reports[fault_id]
        if _canonical(report.get("configuration")) != reference_configuration:
            raise ValueError(f"fault {fault_id}: configuration mismatch")
        if _canonical(report.get("evaluation_protocol")) != reference_protocol:
            raise ValueError(f"fault {fault_id}: evaluation protocol mismatch")
        if report.get("method") != reference_method:
            raise ValueError(f"fault {fault_id}: method label mismatch")
        if (
            _dataset_signature(report.get("train"), include_name=True)
            != reference_train
        ):
            raise ValueError(f"fault {fault_id}: training metadata mismatch")
        if (
            _dataset_signature(report.get("calibration"), include_name=True)
            != reference_calibration
        ):
            raise ValueError(f"fault {fault_id}: calibration metadata mismatch")
        if (
            _dataset_signature(report.get("test"), include_name=False)
            != reference_test
        ):
            raise ValueError(f"fault {fault_id}: test layout mismatch")
        if not _numerically_compatible(
            report.get("calibration_summary"),
            reference_calibration_summary,
        ):
            raise ValueError(f"fault {fault_id}: fitted calibration mismatch")
        if not _numerically_compatible(
            report.get("ablation_calibration_summaries"),
            reference_ablation_summaries,
        ):
            raise ValueError(f"fault {fault_id}: fitted ablation mismatch")
        records = report["records"]
        if not isinstance(records, list):
            raise ValueError(f"fault {fault_id}: invalid records")
        pairs = tuple(
            sorted((str(record["scenario_id"]), int(record["seed"])) for record in records)
        )
        if pairs != reference_pairs:
            raise ValueError(f"fault {fault_id}: scenario/seed design mismatch")
        for record in records:
            record_methods = record.get("methods")
            if not isinstance(record_methods, dict) or tuple(record_methods) != methods:
                raise ValueError(f"fault {fault_id}: method set or order mismatch")
            scenario_id = str(record["scenario_id"])
            definition = record.get("scenario")
            previous = scenario_definitions.setdefault(scenario_id, definition)
            if _canonical(previous) != _canonical(definition):
                raise ValueError(f"fault {fault_id}: scenario definition mismatch")
        all_records.extend(records)
        by_fault[f"{fault_id:02d}"] = {
            "test": report["test"],
            "methods": _summaries(records, methods),
        }
        files.append(
            {
                "fault_id": fault_id,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    by_scenario: dict[str, object] = {}
    for scenario_id in sorted(scenario_definitions):
        selected = [
            record for record in all_records if str(record["scenario_id"]) == scenario_id
        ]
        by_scenario[scenario_id] = {
            "scenario": scenario_definitions[scenario_id],
            "methods": _summaries(selected, methods),
        }

    return {
        "schema_version": 1,
        "source": "classic single-realization Tennessee Eastman Process",
        "independence_warning": (
            "Each fault has one process trajectory. Scenario/seed cases are "
            "repeated missingness views of that trajectory, not independent "
            "process runs; the complete-input reference is consequently "
            "repeated across those views."
        ),
        "analysis_unit": {
            "name": "event evaluation under a missingness stress view",
            "distinct_process_trajectories": len(expected),
            "stress_views": len(all_records),
        },
        "delay_metric": (
            "median first-alarm delay in samples among detected event "
            "evaluations only"
        ),
        "imputation_metric": {
            "name": "robust-scale-normalized RMSE",
            "normalization": "residual divided by Phase-I feature scale",
            "scope": "originally observed, artificially hidden cells only",
        },
        "calibration_compatibility": {
            "fitted_summaries_checked": True,
            "relative_tolerance": _CALIBRATION_RTOL,
            "absolute_tolerance": _CALIBRATION_ATOL,
        },
        "fault_ids": list(expected),
        "fault_count": len(expected),
        "records_per_fault": len(reference_pairs),
        "total_cases": len(all_records),
        "scenario_seed_design": [
            {"scenario_id": scenario_id, "seed": seed}
            for scenario_id, seed in reference_pairs
        ],
        "configuration": first["configuration"],
        "evaluation_protocol": first["evaluation_protocol"],
        "train": first["train"],
        "calibration": first["calibration"],
        "test_layout": reference_test,
        "methods": list(methods),
        "files": files,
        "overall": _summaries(all_records, methods),
        "by_fault": by_fault,
        "by_scenario": by_scenario,
    }


def write_tep_aggregate(
    report: dict[str, object], json_path: str | Path, csv_path: str | Path
) -> None:
    """Write the full aggregate JSON and a compact fault-by-method CSV."""
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = [
        "fault_id",
        "method",
        "detected_event_evaluations",
        "event_evaluation_count",
        "event_evaluation_recall",
        "median_delay_samples_among_detected",
        "false_alarm_episodes",
        "false_alarm_episodes_per_100h",
        "monitored_normal_hours",
        "mean_point_f1",
        "mean_decision_coverage",
        "mean_unknown_rate",
        "pooled_scale_normalized_imputation_rmse",
        "imputation_hidden_cells",
        "case_count",
    ]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        by_fault = report["by_fault"]
        if not isinstance(by_fault, dict):
            raise ValueError("aggregate report has invalid by_fault data")
        for fault_id, fault_report in by_fault.items():
            if not isinstance(fault_report, dict):
                raise ValueError("aggregate report has invalid fault data")
            method_reports = fault_report["methods"]
            if not isinstance(method_reports, dict):
                raise ValueError("aggregate report has invalid method data")
            for method, summary in method_reports.items():
                if not isinstance(summary, dict):
                    raise ValueError("aggregate report has invalid method summary")
                writer.writerow({"fault_id": fault_id, "method": method, **summary})


def _display(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_tep_markdown(report: dict[str, object], path: str | Path) -> None:
    """Write a compact human-readable report without changing the JSON source."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fault_ids = report["fault_ids"]
    if not isinstance(fault_ids, list) or not fault_ids:
        raise ValueError("aggregate report has invalid fault_ids")
    expected_range = list(range(int(fault_ids[0]), int(fault_ids[-1]) + 1))
    fault_label = (
        f"{fault_ids[0]}–{fault_ids[-1]}"
        if fault_ids == expected_range
        else ", ".join(str(value) for value in fault_ids)
    )
    lines = [
        f"# Classic TEP faults {fault_label} missingness stress report",
        "",
        (
            f"Faults: {report['fault_count']}; cases: {report['total_cases']}; "
            f"cases per fault: {report['records_per_fault']}."
        ),
        "",
        f"> {report['independence_warning']}",
        "",
        "## Overall method summary",
        "",
        (
            "| Method | Detected stress views | View recall | "
            "Median delay among detections (samples) | False episodes | "
            "False/100h | Mean F1 | Coverage | "
            "Robust-scale-normalized imputation RMSE |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = report["overall"]
    if not isinstance(overall, dict):
        raise ValueError("aggregate report has invalid overall data")
    for method, summary in overall.items():
        if not isinstance(summary, dict):
            raise ValueError("aggregate report has invalid overall summary")
        detected = (
            f"{summary['detected_event_evaluations']}/"
            f"{summary['event_evaluation_count']}"
        )
        lines.append(
            f"| {method} | {detected} | "
            f"{_display(summary['event_evaluation_recall'])} | "
            f"{_display(summary['median_delay_samples_among_detected'], 1)} | "
            f"{summary['false_alarm_episodes']} | "
            f"{_display(summary['false_alarm_episodes_per_100h'], 3)} | "
            f"{_display(summary['mean_point_f1'])} | "
            f"{_display(summary['mean_decision_coverage'])} | "
            f"{_display(summary['pooled_scale_normalized_imputation_rmse'])} |"
        )

    lines.extend(
        [
            "",
            "## UD-MAPIC by fault",
            "",
            (
                "| Fault | Detected stress views | View recall | "
                "Median delay among detections (samples) | False episodes | "
                "Mean F1 | Coverage |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_fault = report["by_fault"]
    if not isinstance(by_fault, dict):
        raise ValueError("aggregate report has invalid by_fault data")
    for fault_id, fault_report in by_fault.items():
        if not isinstance(fault_report, dict):
            raise ValueError("aggregate report has invalid fault data")
        methods = fault_report["methods"]
        if not isinstance(methods, dict):
            raise ValueError("aggregate report has invalid fault methods")
        summary = methods["UD-MAPIC"]
        if not isinstance(summary, dict):
            raise ValueError("aggregate report has invalid UD-MAPIC summary")
        detected = (
            f"{summary['detected_event_evaluations']}/"
            f"{summary['event_evaluation_count']}"
        )
        lines.append(
            f"| {fault_id} | {detected} | "
            f"{_display(summary['event_evaluation_recall'])} | "
            f"{_display(summary['median_delay_samples_among_detected'], 1)} | "
            f"{summary['false_alarm_episodes']} | "
            f"{_display(summary['mean_point_f1'])} | "
            f"{_display(summary['mean_decision_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## UD-MAPIC by missingness scenario",
            "",
            (
                "| Scenario | Detected stress views | View recall | "
                "Median delay among detections (samples) | False episodes | "
                "False/100h | Mean F1 | Coverage |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_scenario = report["by_scenario"]
    if not isinstance(by_scenario, dict):
        raise ValueError("aggregate report has invalid by_scenario data")
    for scenario_id, scenario_report in by_scenario.items():
        if not isinstance(scenario_report, dict):
            raise ValueError("aggregate report has invalid scenario data")
        methods = scenario_report["methods"]
        if not isinstance(methods, dict):
            raise ValueError("aggregate report has invalid scenario methods")
        summary = methods["UD-MAPIC"]
        if not isinstance(summary, dict):
            raise ValueError("aggregate report has invalid scenario summary")
        detected = (
            f"{summary['detected_event_evaluations']}/"
            f"{summary['event_evaluation_count']}"
        )
        lines.append(
            f"| {_markdown_cell(scenario_id)} | {detected} | "
            f"{_display(summary['event_evaluation_recall'])} | "
            f"{_display(summary['median_delay_samples_among_detected'], 1)} | "
            f"{summary['false_alarm_episodes']} | "
            f"{_display(summary['false_alarm_episodes_per_100h'], 3)} | "
            f"{_display(summary['mean_point_f1'])} | "
            f"{_display(summary['mean_decision_coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "- These are classic single-realization TEP stress tests. Mask seeds "
                "within a fault are repeated missingness views, not independent runs."
            ),
            (
                "- Detection delay is conditional on detection; always interpret it "
                "together with view-level event recall."
            ),
            (
                "- The complete-input reference repeats the same result over the 15 "
                "missingness views of each fault; it is not 15 independent trials."
            ),
            (
                "- Use the Rieth multi-run TEP dataset before computing inferential "
                "confidence intervals or making publication-level superiority claims."
            ),
            (
                "- Offline linear interpolation is noncausal and is included only as "
                "a reference."
            ),
            "",
        ]
    )
    target.write_text("\n".join(lines), encoding="utf-8")
