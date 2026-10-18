"""Evaluate all six pre-confirmation-frozen methods on the same confirmation runs."""

from __future__ import annotations

import torch  # noqa: F401  # Windows torch-first DLL loading contract

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from mate_spc_cris_v9.model import CRISModel
from mate_spc_cris_v9.monitor import (
    PointCalibrator,
    calibrate_trajectory_threshold,
    scan_innovations,
)
from mate_spc_cris_v9.study import (
    FAULT_KINDS,
    candidate_specs,
    evaluate_baselines,
    innovation_runs,
    load_study_charter,
    sha256_file,
    simulate_role,
    summarize_paths,
)


PROJECT = Path(__file__).resolve().parents[1]
CHARTER_PATH = PROJECT / "protocols" / "cris_v9_four_tank_study_charter.json"
OUTPUT_ROOT = PROJECT / "outputs" / "cris_v9_four_tank"
MODEL_PATH = OUTPUT_ROOT / "cris_v9_model.npz"
PRIOR_SELECTED_RESULT = (
    OUTPUT_ROOT / "predictive_residual_cusum_confirmation.json"
)
RESULT_PATH = OUTPUT_ROOT / "all_frozen_methods_confirmation.json"
CONFIRMATION_ROLES = (
    "confirmation_healthy",
    "confirmation_sensor_bias",
    "confirmation_intermittent_pump",
    "confirmation_valve_stiction",
    "confirmation_dependence_shift",
    "confirmation_covariate_shift_only",
)


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done == 1 or done % 50 == 0 or done == total:
            print(f"[innovations] {label}: {done}/{total}", flush=True)

    return report


def _write(document: dict[str, Any]) -> None:
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(RESULT_PATH)


def _scorecard(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    recalls = [float(summaries[kind]["recall"]) for kind in FAULT_KINDS]
    delays = [
        float(summaries[kind]["median_relative_delay"]) for kind in FAULT_KINDS
    ]
    healthy = float(summaries["healthy"]["alarm_rate"])
    covariate = float(summaries["covariate_shift_only"]["alarm_rate"])
    return {
        "macro_fault_recall": float(np.mean(recalls)),
        "minimum_fault_recall": min(recalls),
        "worst_median_relative_delay": max(delays),
        "mean_median_relative_delay": float(np.mean(delays)),
        "healthy_alarm_rate": healthy,
        "covariate_shift_alarm_rate": covariate,
        "maximum_negative_control_alarm_rate": max(healthy, covariate),
        "both_negative_controls_at_or_below_0_10": (
            healthy <= 0.10 and covariate <= 0.10
        ),
    }


def _ranking_key(item: dict[str, Any]) -> tuple[Any, ...]:
    score = item["scorecard"]
    return (
        not score["both_negative_controls_at_or_below_0_10"],
        -score["minimum_fault_recall"],
        -score["macro_fault_recall"],
        score["worst_median_relative_delay"],
        score["maximum_negative_control_alarm_rate"],
        item["frozen_order"],
    )


def main() -> int:
    started = time.perf_counter()
    if RESULT_PATH.exists():
        raise FileExistsError(f"comparison result already exists: {RESULT_PATH}")
    charter = load_study_charter(CHARTER_PATH)
    model = CRISModel.load_npz(MODEL_PATH)
    print(f"[gpu] {model.attestation.cuda_device_name}", flush=True)

    fit_batch = simulate_role(charter, "healthy_model_fit")
    reference_batch = simulate_role(charter, "healthy_point_reference")
    reference_innovations = innovation_runs(
        model,
        reference_batch,
        charter,
        device="cuda",
        progress=_progress("healthy_point_reference"),
    )
    threshold_batch = simulate_role(charter, "healthy_threshold_calibration")
    threshold_innovations = innovation_runs(
        model,
        threshold_batch,
        charter,
        device="cuda",
        progress=_progress("healthy_threshold_calibration"),
    )

    confirmation: dict[str, tuple[Any, tuple[np.ndarray, ...]]] = {}
    for role in CONFIRMATION_ROLES:
        print(f"[simulate] {role}", flush=True)
        batch = simulate_role(charter, role, confirmation_authorized=True)
        innovations = innovation_runs(
            model,
            batch,
            charter,
            device="cuda",
            progress=_progress(role),
        )
        confirmation[role] = (batch, innovations)

    methods: list[dict[str, Any]] = []
    for frozen_order, spec in enumerate(candidate_specs(charter)):
        print(f"[method] {spec.candidate_id}", flush=True)
        calibrator = PointCalibrator.fit(
            reference_innovations, spec.monitor_config
        )
        calibration_paths: list[np.ndarray] = []
        for index, run in enumerate(threshold_innovations):
            calibration_paths.append(scan_innovations(run, calibrator).score)
            if index == 0 or (index + 1) % 50 == 0:
                print(
                    f"[calibration-scan] {spec.candidate_id}: "
                    f"{index + 1}/{len(threshold_innovations)}",
                    flush=True,
                )
        threshold = calibrate_trajectory_threshold(
            calibration_paths, alpha=float(charter["method"]["trajectory_alpha"])
        )
        summaries: dict[str, dict[str, Any]] = {}
        for role, (batch, innovations) in confirmation.items():
            paths: list[np.ndarray] = []
            for index, run in enumerate(innovations):
                paths.append(scan_innovations(run, calibrator).score)
                if index == 0 or (index + 1) % 50 == 0:
                    print(
                        f"[confirmation-scan] {spec.candidate_id} {role}: "
                        f"{index + 1}/{len(innovations)}",
                        flush=True,
                    )
            summaries[batch.fault_kind[0]] = summarize_paths(
                paths, batch, threshold, charter
            )
        methods.append(
            {
                "method": spec.candidate_id,
                "family": "cris_hmm_scan",
                "frozen_order": frozen_order,
                "threshold": {
                    "value": threshold.value,
                    "alpha": threshold.alpha,
                    "ascending_rank": threshold.ascending_rank,
                    "calibration_run_count": threshold.calibration_run_count,
                },
                "summaries": summaries,
                "scorecard": _scorecard(summaries),
            }
        )

    print("[method] three frozen baselines", flush=True)
    baselines = evaluate_baselines(
        fit_batch,
        reference_innovations,
        threshold_batch,
        threshold_innovations,
        confirmation,
        charter,
    )
    for offset, result in enumerate(baselines, start=len(methods)):
        summaries = {
            name: dict(value) for name, value in result.summaries.items()
        }
        methods.append(
            {
                "method": result.baseline.name,
                "family": "frozen_baseline",
                "frozen_order": offset,
                "threshold": {
                    "value": result.threshold.threshold,
                    "alpha": result.threshold.alpha,
                    "ascending_rank": result.threshold.rank,
                    "calibration_run_count": int(
                        result.threshold.calibration_maxima.size
                    ),
                },
                "summaries": summaries,
                "scorecard": _scorecard(summaries),
            }
        )
        print(f"[baseline-complete] {result.baseline.name}", flush=True)

    prior = json.loads(PRIOR_SELECTED_RESULT.read_text(encoding="ascii"))
    predictive = next(
        item for item in methods if item["method"] == "predictive_residual_cusum"
    )
    if predictive["summaries"] != prior["summaries"]:
        raise RuntimeError(
            "recomputed predictive-residual confirmation differs from its "
            "previously persisted result"
        )
    ordered = sorted(methods, key=_ranking_key)
    ranking = [item["method"] for item in ordered]
    document = {
        "schema": "all_six_frozen_methods_same_confirmation_comparison",
        "created_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "comparison_status": (
            "all methods and hyperparameters were frozen before any "
            "confirmation trajectory was generated; no confirmation tuning"
        ),
        "protocol_id": charter["protocol_id"],
        "protocol_sha256": sha256_file(CHARTER_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "device": {
            "source_device": model.source_device,
            "cuda_device_name": model.attestation.cuda_device_name,
            "torch_version": model.attestation.torch_version,
            "cuda_runtime": model.attestation.cuda_runtime,
        },
        "ranking_rule": (
            "negative-controls<=0.10 first; maximize minimum recall; maximize "
            "macro recall; minimize worst median relative delay; minimize "
            "maximum negative-control alarm rate; frozen order"
        ),
        "ranking": ranking,
        "methods": methods,
        "predictive_result_exactly_reproduces_prior_confirmation": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(document)
    print(f"[result] ranking={ranking}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
