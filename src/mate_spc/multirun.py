"""Boundary-safe fitting, evaluation, and aggregation for multi-run studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import numpy as np

from .baselines import CalibratedPointMonitor
from .datasets import SensorSequence
from .experiment import _config_dict, _monitoring_metrics, _sequence_summary
from .imputation import imputation_error
from .missingness import MissingnessConfig, apply_mask, generate_observed_mask
from .model import MAPIC, MAPICConfig


@dataclass(frozen=True)
class FittedMultiRunSuite:
    proposed: MAPIC
    integrated: MAPIC
    mask_only: MAPIC
    baselines: dict[str, CalibratedPointMonitor]
    config: MAPICConfig
    design_run_names: tuple[str, ...]
    calibration_run_names: tuple[str, ...]


def fit_multirun_suite(
    train_runs: Iterable[SensorSequence],
    calibration_runs: Iterable[SensorSequence],
    config: MAPICConfig,
    baseline_methods: tuple[str, ...] = (
        "median",
        "locf",
        "kalman_mean",
        "linear_offline",
    ),
    progress: Callable[[str], None] | None = None,
) -> FittedMultiRunSuite:
    """Fit all methods once using independent Phase-I run collections."""
    train = tuple(train_runs)
    calibration = tuple(calibration_runs)
    if not train or not calibration:
        raise ValueError("multi-run fitting requires design and calibration runs")

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    ud_config = replace(config, integrate_uncertainty=False)
    announce("fitting UD-MAPIC")
    proposed = MAPIC(ud_config).fit_runs(train, calibration)
    announce("fitting M+U ablation")
    integrated = MAPIC(
        replace(config, integrate_uncertainty=True)
    ).fit_runs(train, calibration)
    announce("fitting mask-only ablation")
    mask_only = MAPIC(
        replace(
            config,
            integrate_uncertainty=False,
            use_posterior_quality_context=False,
        )
    ).fit_runs(train, calibration)
    baselines: dict[str, CalibratedPointMonitor] = {}
    for method in baseline_methods:
        announce(f"fitting baseline {method}")
        baselines[method] = CalibratedPointMonitor(
            method, ud_config
        ).fit_runs(train, calibration)
    return FittedMultiRunSuite(
        proposed=proposed,
        integrated=integrated,
        mask_only=mask_only,
        baselines=baselines,
        config=ud_config,
        design_run_names=tuple(run.name for run in train),
        calibration_run_names=tuple(run.name for run in calibration),
    )


def multirun_suite_summary(suite: FittedMultiRunSuite) -> dict[str, object]:
    """Return JSON-ready Phase-I provenance without serializing fitted arrays."""
    return {
        "method": "UD-MAPIC",
        "configuration": _config_dict(suite.config),
        "design_runs": list(suite.design_run_names),
        "calibration_runs": list(suite.calibration_run_names),
        "calibration_summary": suite.proposed.calibration_summary_,
        "ablation_calibration_summaries": {
            "MAPIC_integrated_expectation": (
                suite.integrated.calibration_summary_
            ),
            "UD-MAPIC_mask_only_context": (
                suite.mask_only.calibration_summary_
            ),
        },
        "baseline_calibration_summaries": {
            method: model.calibration_summary_
            for method, model in suite.baselines.items()
        },
    }


def evaluate_multirun_stress(
    suite: FittedMultiRunSuite,
    test: SensorSequence,
    scenarios: Iterable[MissingnessConfig],
    seeds: Iterable[int] = (11, 29, 47),
    warning_horizon: int = 0,
    refractory: int = 0,
) -> dict[str, object]:
    """Evaluate one independent fault run with frozen fitted models."""
    if not np.isfinite(test.x).all():
        raise ValueError("controlled stress testing requires complete test truth")
    if test.x.shape[1] != suite.proposed.n_features_in_:
        raise ValueError("test feature count differs from the fitted suite")
    test_base_mask = (
        np.ones_like(test.x, dtype=bool)
        if test.observed_mask is None
        else np.asarray(test.observed_mask, dtype=bool)
    )
    if test_base_mask.shape != test.x.shape:
        raise ValueError("test observed mask must match its data array")
    proposed = suite.proposed
    oracle = proposed.predict(test.x, test_base_mask, test.covariates)
    stress_reference = oracle.reconstructed
    records: list[dict[str, object]] = []
    for scenario in tuple(scenarios):
        for seed in tuple(seeds):
            artificial_mask = generate_observed_mask(
                stress_reference, scenario, int(seed)
            )
            mask = test_base_mask & artificial_mask
            artificially_hidden = test_base_mask & ~artificial_mask
            imputation_evaluation_mask = ~artificially_hidden
            incomplete = apply_mask(test.x, mask)
            mapic_result = proposed.predict(
                incomplete, mask, test.covariates
            )
            integrated = suite.integrated.predict(
                incomplete, mask, test.covariates
            )
            mask_only = suite.mask_only.predict(
                incomplete, mask, test.covariates
            )
            record: dict[str, object] = {
                "mechanism": scenario.mechanism,
                "scenario_id": scenario.identifier(),
                "scenario": asdict(scenario),
                "target_missing_rate": scenario.rate,
                "realized_missing_rate": float((~mask).mean()),
                "natural_missing_rate": float((~test_base_mask).mean()),
                "realized_artificial_missing_rate_on_observed_cells": (
                    float(artificially_hidden.sum() / test_base_mask.sum())
                    if test_base_mask.any()
                    else 0.0
                ),
                "seed": int(seed),
                "methods": {
                    "same_chart_base_observed_input": {
                        "monitoring": _monitoring_metrics(
                            oracle.alarm,
                            test,
                            warning_horizon,
                            refractory,
                            decision_mask=~oracle.unknown,
                        ),
                        "unknown_rate": float(oracle.unknown.mean()),
                    },
                    "UD-MAPIC": {
                        "monitoring": _monitoring_metrics(
                            mapic_result.alarm,
                            test,
                            warning_horizon,
                            refractory,
                            decision_mask=~mapic_result.unknown,
                        ),
                        "imputation": imputation_error(
                            test.x,
                            mapic_result.reconstructed,
                            imputation_evaluation_mask,
                            scale=proposed.monitor.filter_model.scale_,
                        ),
                        "unknown_rate": float(
                            mapic_result.unknown.mean()
                        ),
                        "imputation_scope": (
                            "originally observed, artificially hidden cells "
                            "only; residuals normalized by Phase-I feature "
                            "scale"
                        ),
                    },
                    "MAPIC_integrated_expectation": {
                        "monitoring": _monitoring_metrics(
                            integrated.alarm,
                            test,
                            warning_horizon,
                            refractory,
                            decision_mask=~integrated.unknown,
                        ),
                        "unknown_rate": float(integrated.unknown.mean()),
                    },
                    "UD-MAPIC_mask_only_context": {
                        "monitoring": _monitoring_metrics(
                            mask_only.alarm,
                            test,
                            warning_horizon,
                            refractory,
                            decision_mask=~mask_only.unknown,
                        ),
                        "unknown_rate": float(mask_only.unknown.mean()),
                    },
                },
            }
            method_table = record["methods"]
            assert isinstance(method_table, dict)
            for method, baseline in suite.baselines.items():
                result = baseline.predict(
                    incomplete, mask, test.covariates
                )
                method_table[method] = {
                    "monitoring": _monitoring_metrics(
                        result.alarm,
                        test,
                        warning_horizon,
                        refractory,
                        decision_mask=~result.unknown,
                    ),
                    "imputation": imputation_error(
                        test.x,
                        result.reconstructed,
                        imputation_evaluation_mask,
                        scale=proposed.monitor.filter_model.scale_,
                    ),
                    "unknown_rate": float(result.unknown.mean()),
                    "imputation_scope": (
                        "originally observed, artificially hidden cells only; "
                        "residuals normalized by Phase-I feature scale"
                    ),
                    "causal": method != "linear_offline",
                }
            records.append(record)
    return {
        "schema_version": 1,
        "method": "UD-MAPIC",
        "configuration": _config_dict(suite.config),
        "evaluation_protocol": {
            "warning_horizon_samples": int(warning_horizon),
            "refractory_samples": int(refractory),
            "calibration_unit": "healthy_run_maximum",
        },
        "test": _sequence_summary(test),
        "test_metadata": {
            key: value
            for key, value in test.metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "records": records,
    }


def _finite(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def _method_summary(
    entries: list[dict[str, object]],
) -> dict[str, float | int | None]:
    monitoring = [entry["monitoring"] for entry in entries]
    if not all(isinstance(value, dict) for value in monitoring):
        raise ValueError("method entries must contain monitoring metrics")
    monitors = [value for value in monitoring if isinstance(value, dict)]
    event_count = sum(int(value["event_count"]) for value in monitors)
    detected = sum(int(value["events_detected"]) for value in monitors)
    false_episodes = sum(
        int(value["false_alarm_episodes"]) for value in monitors
    )
    normal_hours = sum(
        float(value["monitored_normal_hours"]) for value in monitors
    )
    delays = _finite(
        value.get("median_delay_samples") for value in monitors
    )
    imputation_sse = 0.0
    imputation_count = 0.0
    for entry in entries:
        imputation = entry.get("imputation")
        if not isinstance(imputation, dict):
            continue
        count = float(imputation.get("n_hidden", 0.0))
        rmse = imputation.get("rmse")
        if count > 0.0 and rmse is not None:
            imputation_sse += float(rmse) ** 2 * count
            imputation_count += count
    f1 = _finite(value.get("point_f1") for value in monitors)
    coverage = _finite(
        value.get("decision_coverage") for value in monitors
    )
    unknown = _finite(entry.get("unknown_rate") for entry in entries)
    return {
        "detected_event_evaluations": detected,
        "event_evaluation_count": event_count,
        "event_evaluation_recall": (
            detected / event_count if event_count else None
        ),
        "median_delay_samples_among_detected": (
            float(median(delays)) if delays else None
        ),
        "false_alarm_episodes": false_episodes,
        "false_alarm_episodes_per_100h": (
            100.0 * false_episodes / normal_hours
            if normal_hours > 0.0
            else None
        ),
        "mean_point_f1": sum(f1) / len(f1) if f1 else None,
        "mean_decision_coverage": (
            sum(coverage) / len(coverage) if coverage else None
        ),
        "mean_unknown_rate": (
            sum(unknown) / len(unknown) if unknown else None
        ),
        "pooled_scale_normalized_imputation_rmse": (
            math.sqrt(imputation_sse / imputation_count)
            if imputation_count > 0.0
            else None
        ),
        "imputation_hidden_cells": int(imputation_count),
        "case_count": len(entries),
    }


def _method_entries(
    reports: Iterable[dict[str, object]], method: str
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for report in reports:
        records = report.get("records")
        if not isinstance(records, list):
            raise ValueError("multi-run report has invalid records")
        for record in records:
            methods = record.get("methods")
            if not isinstance(methods, dict):
                raise ValueError("multi-run record has invalid methods")
            entry = methods.get(method)
            if not isinstance(entry, dict):
                raise ValueError(f"multi-run record lacks method {method}")
            entries.append(entry)
    return entries


def _percentile_interval(
    values: list[float], confidence: float
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    tail = 0.5 * (1.0 - confidence)
    return {
        "lower": float(np.quantile(array, tail)),
        "upper": float(np.quantile(array, 1.0 - tail)),
        "replicates": int(array.size),
        "confidence": float(confidence),
    }


def aggregate_multirun_reports(
    reports: Iterable[dict[str, object]],
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260907,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Aggregate stress views and bootstrap at the independent run level."""
    values = tuple(reports)
    if not values:
        raise ValueError("no multi-run reports were supplied")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    reference_configuration = values[0].get("configuration")
    reference_protocol = values[0].get("evaluation_protocol")
    reference_records = values[0].get("records")
    if not isinstance(reference_records, list) or not reference_records:
        raise ValueError("reference multi-run report has no records")
    first_methods = reference_records[0].get("methods")
    if not isinstance(first_methods, dict) or not first_methods:
        raise ValueError("reference multi-run report has no methods")
    methods = tuple(first_methods)
    scenario_seed_design = tuple(
        sorted(
            (str(record["scenario_id"]), int(record["seed"]))
            for record in reference_records
        )
    )
    by_fault_reports: dict[int, list[dict[str, object]]] = {}
    seen_runs: set[tuple[int, int]] = set()
    protocol_fingerprints: set[str] = set()
    for report in values:
        if report.get("configuration") != reference_configuration:
            raise ValueError("multi-run configuration mismatch")
        if report.get("evaluation_protocol") != reference_protocol:
            raise ValueError("multi-run evaluation protocol mismatch")
        fingerprint = report.get("protocol_fingerprint")
        if fingerprint is not None:
            protocol_fingerprints.add(str(fingerprint))
        metadata = report.get("test_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("multi-run report has no test metadata")
        fault_id = int(metadata["fault_id"])
        run_id = int(metadata["simulation_run"])
        key = (fault_id, run_id)
        if key in seen_runs:
            raise ValueError(f"duplicate fault/run report: {key}")
        seen_runs.add(key)
        records = report.get("records")
        if not isinstance(records, list):
            raise ValueError("multi-run report has invalid records")
        design = tuple(
            sorted(
                (str(record["scenario_id"]), int(record["seed"]))
                for record in records
            )
        )
        if design != scenario_seed_design:
            raise ValueError(f"fault/run {key}: scenario/seed mismatch")
        for record in records:
            record_methods = record.get("methods")
            if not isinstance(record_methods, dict):
                raise ValueError(f"fault/run {key}: invalid method table")
            if tuple(record_methods) != methods:
                raise ValueError(f"fault/run {key}: method set mismatch")
        by_fault_reports.setdefault(fault_id, []).append(report)
    if len(protocol_fingerprints) > 1:
        raise ValueError("multi-run protocol fingerprint mismatch")
    for fault_id in by_fault_reports:
        by_fault_reports[fault_id].sort(
            key=lambda report: int(report["test_metadata"]["simulation_run"])
        )

    overall = {
        method: _method_summary(_method_entries(values, method))
        for method in methods
    }
    by_fault: dict[str, object] = {}
    by_run: dict[str, object] = {}
    for fault_id, fault_reports in sorted(by_fault_reports.items()):
        by_fault[f"{fault_id:02d}"] = {
            "run_count": len(fault_reports),
            "methods": {
                method: _method_summary(
                    _method_entries(fault_reports, method)
                )
                for method in methods
            },
        }
        for report in fault_reports:
            metadata = report["test_metadata"]
            assert isinstance(metadata, dict)
            run_id = int(metadata["simulation_run"])
            run_key = f"fault-{fault_id:02d}-run-{run_id:03d}"
            method_summaries = {
                method: _method_summary(
                    _method_entries((report,), method)
                )
                for method in methods
            }
            by_run[run_key] = {
                "fault_id": fault_id,
                "simulation_run": run_id,
                "methods": {
                    method: {
                        **summary,
                        "any_view_detected": (
                            summary["detected_event_evaluations"] > 0
                        ),
                        "all_views_detected": (
                            summary["detected_event_evaluations"]
                            == summary["event_evaluation_count"]
                        ),
                    }
                    for method, summary in method_summaries.items()
                },
            }

    metrics = (
        "event_evaluation_recall",
        "median_delay_samples_among_detected",
        "false_alarm_episodes_per_100h",
        "mean_point_f1",
        "mean_decision_coverage",
    )
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_values = {
        method: {metric: [] for metric in metrics} for method in methods
    }
    comparison_methods = tuple(
        method
        for method in (
            "UD-MAPIC_mask_only_context",
            "MAPIC_integrated_expectation",
        )
        if method in methods
    )
    paired_values = {
        comparator: {metric: [] for metric in metrics}
        for comparator in comparison_methods
    }
    for _ in range(bootstrap_replicates):
        sampled: list[dict[str, object]] = []
        for fault_reports in by_fault_reports.values():
            indices = rng.integers(0, len(fault_reports), len(fault_reports))
            sampled.extend(fault_reports[int(index)] for index in indices)
        summaries = {
            method: _method_summary(_method_entries(sampled, method))
            for method in methods
        }
        for method, summary in summaries.items():
            for metric in metrics:
                value = summary[metric]
                if value is not None:
                    bootstrap_values[method][metric].append(float(value))
        for comparator in comparison_methods:
            for metric in metrics:
                left = summaries["UD-MAPIC"][metric]
                right = summaries[comparator][metric]
                if left is not None and right is not None:
                    paired_values[comparator][metric].append(
                        float(left) - float(right)
                    )

    intervals = {
        method: {
            metric: _percentile_interval(samples, confidence)
            for metric, samples in metric_values.items()
            if samples
        }
        for method, metric_values in bootstrap_values.items()
    }
    paired_comparisons = {
        comparator: {
            metric: {
                "estimate": (
                    float(overall["UD-MAPIC"][metric])
                    - float(overall[comparator][metric])
                    if overall["UD-MAPIC"][metric] is not None
                    and overall[comparator][metric] is not None
                    else None
                ),
                "bootstrap_interval": _percentile_interval(
                    samples, confidence
                ),
            }
            for metric, samples in metric_values.items()
            if samples
        }
        for comparator, metric_values in paired_values.items()
    }
    return {
        "schema_version": 1,
        "source": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": "10.7910/DVN/6C3JR1",
        "protocol_fingerprint": (
            next(iter(protocol_fingerprints))
            if protocol_fingerprints
            else None
        ),
        "analysis_unit": "independent fault process run",
        "nested_stress_views_per_run": len(scenario_seed_design),
        "run_count": len(values),
        "fault_ids": sorted(by_fault_reports),
        "run_counts_by_fault": {
            f"{fault_id:02d}": len(fault_reports)
            for fault_id, fault_reports in sorted(by_fault_reports.items())
        },
        "configuration": reference_configuration,
        "evaluation_protocol": reference_protocol,
        "methods": list(methods),
        "scenario_seed_design": [
            {"scenario_id": scenario_id, "seed": seed}
            for scenario_id, seed in scenario_seed_design
        ],
        "overall": overall,
        "by_fault": by_fault,
        "by_run": by_run,
        "run_clustered_bootstrap": {
            "sampling": "stratified resampling of process runs within fault",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "intervals": intervals,
            "paired_ud_minus_comparator": paired_comparisons,
        },
        "interpretation_warning": (
            "Mask/scenario views are nested within process runs. Confidence "
            "intervals resample runs, never individual views or time points."
        ),
    }


def _display(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _estimate_interval(
    estimate: object,
    interval: object,
    digits: int = 4,
) -> str:
    if not isinstance(interval, dict):
        return "—"
    lower = interval.get("lower")
    upper = interval.get("upper")
    if lower is None or upper is None:
        return "—"
    return (
        f"{_display(estimate, digits)} "
        f"[{_display(lower, digits)}, {_display(upper, digits)}]"
    )


def write_multirun_markdown(
    report: dict[str, object], path: str | Path
) -> None:
    """Write the compact pilot report used for human review."""
    lines = [
        "# Rieth multi-run TEP missingness pilot",
        "",
        (
            f"Independent fault runs: {report['run_count']}; "
            f"faults: {', '.join(str(v) for v in report['fault_ids'])}; "
            f"nested stress views/run: "
            f"{report['nested_stress_views_per_run']}."
        ),
        "",
        f"> {report['interpretation_warning']}",
        "",
        "## Overall descriptive results",
        "",
        (
            "| Method | Detected views | View recall | Conditional median "
            "delay (samples) | False/100h | Mean F1 | Coverage |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    overall = report["overall"]
    if not isinstance(overall, dict):
        raise ValueError("invalid multi-run overall summary")
    for method, summary in overall.items():
        if not isinstance(summary, dict):
            raise ValueError("invalid multi-run method summary")
        detected = (
            f"{summary['detected_event_evaluations']}/"
            f"{summary['event_evaluation_count']}"
        )
        lines.append(
            f"| {method} | {detected} | "
            f"{_display(summary['event_evaluation_recall'])} | "
            f"{_display(summary['median_delay_samples_among_detected'], 1)} | "
            f"{_display(summary['false_alarm_episodes_per_100h'], 3)} | "
            f"{_display(summary['mean_point_f1'])} | "
            f"{_display(summary['mean_decision_coverage'])} |"
        )

    bootstrap = report.get("run_clustered_bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("invalid multi-run bootstrap summary")
    intervals = bootstrap.get("intervals")
    if not isinstance(intervals, dict):
        raise ValueError("invalid multi-run bootstrap intervals")
    confidence = 100.0 * float(bootstrap["confidence"])
    lines.extend(
        [
            "",
            f"## Run-clustered {confidence:g}% bootstrap intervals",
            "",
            (
                "| Method | View recall [CI] | Conditional median delay "
                "[CI] | Mean F1 [CI] |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for method, summary in overall.items():
        if not isinstance(summary, dict):
            raise ValueError("invalid multi-run method summary")
        method_intervals = intervals.get(method)
        if not isinstance(method_intervals, dict):
            raise ValueError(f"missing bootstrap intervals for {method}")
        recall_interval = _estimate_interval(
            summary["event_evaluation_recall"],
            method_intervals.get("event_evaluation_recall"),
        )
        delay_interval = _estimate_interval(
            summary["median_delay_samples_among_detected"],
            method_intervals.get("median_delay_samples_among_detected"),
            1,
        )
        f1_interval = _estimate_interval(
            summary["mean_point_f1"],
            method_intervals.get("mean_point_f1"),
        )
        lines.append(
            f"| {method} | "
            f"{recall_interval} | {delay_interval} | {f1_interval} |"
        )

    paired = bootstrap.get("paired_ud_minus_comparator")
    if not isinstance(paired, dict):
        raise ValueError("invalid paired bootstrap comparisons")
    lines.extend(
        [
            "",
            "## Paired UD-MAPIC differences",
            "",
            (
                "| Comparator | Δ view recall [CI] | Δ conditional median "
                "delay [CI] | Δ mean F1 [CI] |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for comparator, metrics in paired.items():
        if not isinstance(metrics, dict):
            raise ValueError("invalid paired comparison metrics")

        def paired_cell(metric: str, digits: int = 4) -> str:
            entry = metrics.get(metric)
            if not isinstance(entry, dict):
                return "—"
            return _estimate_interval(
                entry.get("estimate"),
                entry.get("bootstrap_interval"),
                digits,
            )

        lines.append(
            f"| {comparator} | "
            f"{paired_cell('event_evaluation_recall')} | "
            f"{paired_cell('median_delay_samples_among_detected', 1)} | "
            f"{paired_cell('mean_point_f1')} |"
        )
    lines.extend(
        [
            "",
            (
                "Positive recall/F1 differences favor UD-MAPIC; negative delay "
                "differences favor UD-MAPIC. These are paired, run-clustered "
                "pilot intervals, not final confirmatory claims."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## UD-MAPIC by fault",
            "",
            (
                "| Fault | Runs | Detected views | View recall | "
                "Conditional median delay | False/100h | Mean F1 |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_fault = report["by_fault"]
    if not isinstance(by_fault, dict):
        raise ValueError("invalid multi-run fault summary")
    for fault_id, fault_report in by_fault.items():
        if not isinstance(fault_report, dict):
            raise ValueError("invalid multi-run fault report")
        methods = fault_report["methods"]
        if not isinstance(methods, dict):
            raise ValueError("invalid multi-run fault methods")
        summary = methods["UD-MAPIC"]
        detected = (
            f"{summary['detected_event_evaluations']}/"
            f"{summary['event_evaluation_count']}"
        )
        lines.append(
            f"| {fault_id} | {fault_report['run_count']} | {detected} | "
            f"{_display(summary['event_evaluation_recall'])} | "
            f"{_display(summary['median_delay_samples_among_detected'], 1)} | "
            f"{_display(summary['false_alarm_episodes_per_100h'], 3)} | "
            f"{_display(summary['mean_point_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Statistical boundary",
            "",
            (
                "- Bootstrap intervals resample complete runs within each "
                "fault; masks and time points never become independent units."
            ),
            (
                "- Detection delay is conditional on detection and must be "
                "read together with recall."
            ),
            (
                "- A small pilot validates the protocol and estimates compute "
                "cost; it is not the final publication analysis."
            ),
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
