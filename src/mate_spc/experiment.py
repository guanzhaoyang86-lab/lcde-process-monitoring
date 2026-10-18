"""Reproducible stress-test orchestration and JSON-ready result tables."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Iterable

import numpy as np

from .baselines import CalibratedPointMonitor
from .datasets import SensorSequence
from .imputation import imputation_error
from .metrics import event_metrics
from .missingness import MissingnessConfig, apply_mask, generate_observed_mask
from .model import MAPIC, MAPICConfig


def _monitoring_metrics(
    alarm: np.ndarray,
    sequence: SensorSequence,
    warning_horizon: int,
    refractory: int,
    fault: np.ndarray | None = None,
    decision_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    result = event_metrics(
        alarm,
        sequence.fault if fault is None else fault,
        sequence.sample_period_seconds,
        warning_horizon=warning_horizon,
        refractory=refractory,
        decision_mask=decision_mask,
    )
    return result.as_dict()


def run_stress_test(
    train: SensorSequence,
    calibration: SensorSequence,
    test: SensorSequence,
    config: MAPICConfig,
    scenarios: Iterable[MissingnessConfig],
    seeds: Iterable[int] = (11, 29, 47),
    baseline_methods: tuple[str, ...] = ("median", "locf", "kalman_mean", "linear_offline"),
    warning_horizon: int = 0,
    refractory: int = 0,
) -> dict[str, object]:
    """Fit once, then evaluate independent structured-missingness stress masks."""
    if not np.isfinite(train.x).all() or not np.isfinite(calibration.x).all():
        raise ValueError("Phase-I training and calibration sequences must be complete")
    if not np.isfinite(test.x).all():
        raise ValueError("controlled stress testing requires complete test ground truth")
    if train.fault.any() or calibration.fault.any():
        raise ValueError("Phase-I train and calibration sequences must be healthy")
    if train.x.shape[1] != calibration.x.shape[1] or train.x.shape[1] != test.x.shape[1]:
        raise ValueError("train, calibration, and test feature counts differ")
    if any((train.feature_names, calibration.feature_names, test.feature_names)) and not (
        train.feature_names == calibration.feature_names == test.feature_names
    ):
        raise ValueError("train, calibration, and test feature schemas differ")
    if any((train.covariate_names, calibration.covariate_names, test.covariate_names)) and not (
        train.covariate_names == calibration.covariate_names == test.covariate_names
    ):
        raise ValueError("train, calibration, and test covariate schemas differ")
    periods = np.asarray(
        [
            train.sample_period_seconds,
            calibration.sample_period_seconds,
            test.sample_period_seconds,
        ],
        dtype=float,
    )
    if (
        not np.isfinite(periods).all()
        or np.any(periods <= 0.0)
        or not np.allclose(periods, periods[0], rtol=0.0, atol=1e-9)
    ):
        raise ValueError("train, calibration, and test sample periods must match")
    seeds = tuple(seeds)

    ud_config = replace(config, integrate_uncertainty=False)
    proposed = MAPIC(ud_config).fit(
        train.x,
        calibration.x,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
    )
    integrated_config = replace(config, integrate_uncertainty=True)
    integrated_model = MAPIC(integrated_config).fit(
        train.x,
        calibration.x,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
    )
    mask_only_config = replace(
        config,
        integrate_uncertainty=False,
        use_posterior_quality_context=False,
    )
    mask_only_model = MAPIC(mask_only_config).fit(
        train.x,
        calibration.x,
        train.covariates,
        calibration.covariates,
        train.observed_mask,
        calibration.observed_mask,
    )
    baselines = {
        method: CalibratedPointMonitor(method, ud_config).fit(
            train.x,
            calibration.x,
            train.covariates,
            calibration.covariates,
            train.observed_mask,
            calibration.observed_mask,
        )
        for method in baseline_methods
    }
    test_base_mask = (
        np.ones_like(test.x, dtype=bool)
        if test.observed_mask is None
        else np.asarray(test.observed_mask, dtype=bool)
    )
    if test_base_mask.shape != test.x.shape:
        raise ValueError("test observed mask must match its data array")
    # Natural gaps are part of the deployment input, not artificial ground
    # truth.  A causal reconstruction supplies safe values only for generating
    # additional stress masks (especially MNAR); it is never scored as truth.
    oracle = proposed.predict(test.x, test_base_mask, test.covariates)
    stress_reference = oracle.reconstructed
    records: list[dict[str, object]] = []
    for scenario in scenarios:
        for seed in seeds:
            artificial_mask = generate_observed_mask(stress_reference, scenario, seed)
            mask = test_base_mask & artificial_mask
            artificially_hidden = test_base_mask & ~artificial_mask
            imputation_evaluation_mask = ~artificially_hidden
            incomplete = apply_mask(test.x, mask)
            mapic_result = proposed.predict(incomplete, mask, test.covariates)
            integrated = integrated_model.predict(
                incomplete, mask, test.covariates
            )
            mask_only = mask_only_model.predict(incomplete, mask, test.covariates)
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
                        "unknown_rate": float(mapic_result.unknown.mean()),
                        "imputation_scope": (
                            "originally observed, artificially hidden cells only; "
                            "residuals normalized by Phase-I feature scale"
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
                        "note": (
                            "independently fitted posterior-expectation ablation using "
                            "M + tr(A P) throughout score scaling, regime design, and "
                            "conformal calibration"
                        ),
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
                        "note": (
                            "independently fitted ablation retaining M but removing "
                            "posterior trace and U from quality governance"
                        ),
                    },
                },
            }
            method_table = record["methods"]
            assert isinstance(method_table, dict)
            for method, baseline in baselines.items():
                baseline_result = baseline.predict(incomplete, mask, test.covariates)
                method_table[method] = {
                    "monitoring": _monitoring_metrics(
                        baseline_result.alarm,
                        test,
                        warning_horizon,
                        refractory,
                        decision_mask=~baseline_result.unknown,
                    ),
                    "imputation": imputation_error(
                        test.x,
                        baseline_result.reconstructed,
                        imputation_evaluation_mask,
                        scale=proposed.monitor.filter_model.scale_,
                    ),
                    "unknown_rate": float(baseline_result.unknown.mean()),
                    "imputation_scope": (
                        "originally observed, artificially hidden cells only; "
                        "residuals normalized by Phase-I feature scale"
                    ),
                    "causal": method != "linear_offline",
                    "quality_context": "mask_only_without_posterior_covariance",
                }
            records.append(record)

    return {
        "method": "UD-MAPIC",
        "configuration": _config_dict(ud_config),
        "evaluation_protocol": {
            "warning_horizon_samples": int(warning_horizon),
            "refractory_samples": int(refractory),
        },
        "train": _sequence_summary(train),
        "calibration": _sequence_summary(calibration),
        "test": _sequence_summary(test),
        "calibration_summary": proposed.calibration_summary_,
        "ablation_calibration_summaries": {
            "MAPIC_integrated_expectation": integrated_model.calibration_summary_,
            "UD-MAPIC_mask_only_context": mask_only_model.calibration_summary_,
        },
        "records": records,
    }


def evaluate_natural_missingness(
    model: MAPIC | CalibratedPointMonitor,
    sequence: SensorSequence,
    warning_horizon: int = 0,
    refractory: int = 0,
) -> dict[str, object]:
    """Evaluate real timing gaps without pretending interpolated releases are truth."""
    if sequence.observed_mask is None:
        raise ValueError("sequence has no natural observed mask")
    result = model.predict(
        sequence.x,
        sequence.observed_mask,
        sequence.covariates,
    )
    visible_fault = np.asarray(
        sequence.metadata.get("visible_fault", sequence.fault), dtype=bool
    )
    visible_metrics = _monitoring_metrics(
        result.alarm,
        sequence,
        warning_horizon,
        refractory,
        fault=visible_fault,
        decision_mask=~result.unknown,
    )
    non_normal_metrics = _monitoring_metrics(
        result.alarm,
        sequence,
        warning_horizon,
        refractory,
        fault=sequence.fault,
        decision_mask=~result.unknown,
    )
    return {
        "monitoring": visible_metrics,
        "visible_effect_monitoring": visible_metrics,
        "non_normal_episode_monitoring": non_normal_metrics,
        "natural_missing_rate": float((~sequence.observed_mask).mean()),
        "unknown_rate": float(result.unknown.mean()),
        "imputation": None,
        "imputation_note": "No ground truth exists at naturally interpolated timestamps.",
    }


def _sequence_summary(sequence: SensorSequence) -> dict[str, object]:
    return {
        "name": sequence.name,
        "rows": int(sequence.x.shape[0]),
        "features": int(sequence.x.shape[1]),
        "fault_samples": int(sequence.fault.sum()),
        "sample_period_seconds": sequence.sample_period_seconds,
    }


def _config_dict(config: MAPICConfig) -> dict[str, object]:
    result = asdict(config)
    result["scenarios"] = [asdict(item) for item in config.scenarios]
    return result
