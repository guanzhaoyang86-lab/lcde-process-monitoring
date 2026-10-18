"""Run the complete CRIS-v9 four-tank experiment on CUDA.

The script intentionally keeps governance minimal: one fixed method, one
development pass, and confirmation only after the development GO gate.
"""

from __future__ import annotations

# On this Windows host PyTorch must be imported before NumPy/pandas-backed
# project modules so its DLL directory is registered first.
import torch  # noqa: F401

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from mate_spc_cris_v9.evaluation import EpisodeOutcome
from mate_spc_cris_v9.monitor import apply_threshold, scan_innovations
from mate_spc_cris_v9.study import (
    FAULT_KINDS,
    CandidateDevelopment,
    BaselineDevelopment,
    candidate_specs,
    development_go,
    diagnostic_best,
    evaluate_baselines,
    evaluate_candidate,
    fit_model,
    innovation_runs,
    load_study_charter,
    model_arrays,
    select_candidate,
    sha256_file,
    simulate_role,
    summarize_paths,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CHARTER = PROJECT / "protocols" / "cris_v9_four_tank_study_charter.json"
DEFAULT_OUTPUT = PROJECT / "outputs" / "cris_v9_four_tank"
DEVELOPMENT_ROLES = (
    "development_healthy",
    "development_sensor_bias",
    "development_intermittent_pump",
    "development_valve_stiction",
    "development_dependence_shift",
    "development_covariate_shift_only",
)
CONFIRMATION_ROLES = (
    "confirmation_healthy",
    "confirmation_sensor_bias",
    "confirmation_intermittent_pump",
    "confirmation_valve_stiction",
    "confirmation_dependence_shift",
    "confirmation_covariate_shift_only",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(path)


def _innovation_progress(role: str):
    def report(done: int, total: int) -> None:
        if done == 1 or done % 50 == 0 or done == total:
            print(f"[innovations] {role}: {done}/{total}", flush=True)

    return report


def _candidate_progress(candidate_id: str):
    last: dict[str, int] = {}

    def report(role: str, done: int, total: int) -> None:
        if done == 1 or done % 50 == 0 or done == total:
            if last.get(role) != done:
                print(f"[scan] {candidate_id} {role}: {done}/{total}", flush=True)
                last[role] = done

    return report


def _make_role_data(
    roles: tuple[str, ...],
    charter: dict[str, Any] | Any,
    model: Any,
    *,
    confirmation_authorized: bool,
) -> dict[str, tuple[Any, tuple[np.ndarray, ...]]]:
    result: dict[str, tuple[Any, tuple[np.ndarray, ...]]] = {}
    for role in roles:
        print(f"[simulate] {role}", flush=True)
        batch = simulate_role(
            charter, role, confirmation_authorized=confirmation_authorized
        )
        innovations = innovation_runs(
            model,
            batch,
            charter,
            device="cuda",
            progress=_innovation_progress(role),
        )
        result[role] = (batch, innovations)
    return result


def _candidate_result_document(
    results: tuple[CandidateDevelopment, ...],
    selected: CandidateDevelopment | None,
) -> dict[str, Any]:
    best = diagnostic_best(results)
    return {
        "candidates": [item.to_document() for item in results],
        "selected_candidate": None if selected is None else selected.spec.candidate_id,
        "diagnostic_best_candidate": best.spec.candidate_id,
    }


def _score_selected(
    selected: CandidateDevelopment,
    data: dict[str, tuple[Any, tuple[np.ndarray, ...]]],
    charter: Any,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for role, (batch, innovations) in data.items():
        paths = tuple(
            scan_innovations(run, selected.calibrator).score for run in innovations
        )
        summaries[batch.fault_kind[0]] = summarize_paths(
            paths, batch, selected.threshold, charter
        )
        print(f"[confirm-scan] {role}: {len(paths)}/{len(paths)}", flush=True)
    return summaries


def _score_frozen_baselines(
    baselines: tuple[BaselineDevelopment, ...],
    data: dict[str, tuple[Any, tuple[np.ndarray, ...]]],
    charter: Any,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for fitted in baselines:
        summaries: dict[str, dict[str, Any]] = {}
        for _role, (batch, innovations) in data.items():
            if fitted.baseline.name == "predictive_residual_cusum":
                values = innovations
            else:
                targets, _covariates, _mask = model_arrays(batch)
                values = tuple(targets)
            paths = tuple(fitted.baseline.score(run).path for run in values)
            summaries[batch.fault_kind[0]] = summarize_paths(
                paths, batch, fitted.threshold.threshold, charter
            )
        documents.append(
            {
                "baseline": fitted.baseline.name,
                "threshold": {
                    "value": fitted.threshold.threshold,
                    "alpha": fitted.threshold.alpha,
                    "calibration_run_count": int(
                        fitted.threshold.calibration_maxima.size
                    ),
                    "ascending_rank": fitted.threshold.rank,
                },
                "summaries": summaries,
            }
        )
        print(f"[confirm-baseline] {fitted.baseline.name}", flush=True)
    return documents


def _repeatability(
    selected: CandidateDevelopment,
    model: Any,
    batch: Any,
    charter: Any,
) -> dict[str, Any]:
    targets, covariates, mask = model_arrays(batch)
    kwargs = {
        "device": "cuda",
        "batch_size": int(charter["method"]["inference_batch_size"]),
    }
    first = model.innovations(targets[0], covariates[0], mask[0], **kwargs)
    second = model.innovations(targets[0], covariates[0], mask[0], **kwargs)
    finite = np.isfinite(first) & np.isfinite(second)
    max_error = float(np.max(np.abs(first[finite] - second[finite])))
    same_nan = bool(np.array_equal(np.isnan(first), np.isnan(second)))
    first_score = scan_innovations(
        first, selected.calibrator, threshold=selected.threshold
    )
    second_score = scan_innovations(
        second, selected.calibrator, threshold=selected.threshold
    )
    score_finite = first_score.valid & second_score.valid
    score_error = float(
        np.max(np.abs(first_score.score[score_finite] - second_score.score[score_finite]))
    )
    return {
        "innovation_max_abs_error": max_error,
        "innovation_nan_mask_exact": same_nan,
        "score_max_abs_error": score_error,
        "score_valid_mask_exact": bool(
            np.array_equal(first_score.valid, second_score.valid)
        ),
        "alarm_decisions_exact": bool(
            np.array_equal(first_score.alarm, second_score.alarm)
        ),
        "source_device": model.source_device,
        "cuda_device_name": model.attestation.cuda_device_name,
        "torch_version": model.attestation.torch_version,
        "cuda_runtime": model.attestation.cuda_runtime,
    }


def _confirmation_go(
    summaries: dict[str, dict[str, Any]],
    repeatability: dict[str, Any],
    charter: Any,
) -> tuple[bool, list[str]]:
    gate = charter["confirmation_go_gates"]
    failures: list[str] = []
    healthy = summaries["healthy"]
    if int(healthy["alarm_runs"]) > int(gate["healthy_alarm_runs_max"]):
        failures.append("healthy_alarm_runs")
    if float(healthy["one_sided_cp95_upper"]) > float(
        gate["healthy_one_sided_cp95_upper_max"]
    ):
        failures.append("healthy_cp95_upper")
    if summaries["covariate_shift_only"]["alarm_rate"] > float(
        gate["covariate_shift_only_alarm_rate_max"]
    ):
        failures.append("covariate_shift_only_alarm_rate")
    for kind in FAULT_KINDS:
        recall_gate = (
            float(gate["intermittent_pump_recall_min"])
            if kind == "intermittent_pump"
            else float(gate[f"{kind}_recall_min"])
        )
        if summaries[kind]["recall"] < recall_gate:
            failures.append(f"{kind}_recall")
        if summaries[kind]["median_relative_delay"] > float(
            gate["each_fault_median_relative_delay_max"]
        ):
            failures.append(f"{kind}_median_relative_delay")
    if repeatability["score_max_abs_error"] > float(
        gate["cuda_repeat_score_atol"]
    ):
        failures.append("cuda_repeat_score")
    if not repeatability["alarm_decisions_exact"]:
        failures.append("cuda_alarm_decisions")
    return not failures, failures


def _superiority(
    cris: dict[str, dict[str, Any]],
    baselines: list[dict[str, Any]],
    charter: Any,
) -> dict[str, Any]:
    rule = charter["superiority_gate"]
    by_baseline: dict[str, Any] = {}
    for item in baselines:
        other = item["summaries"]
        gains = {
            kind: float(cris[kind]["recall"]) - float(other[kind]["recall"])
            for kind in FAULT_KINDS
        }
        improved = [
            kind
            for kind, gain in gains.items()
            if gain >= float(rule["recall_gain_min"])
        ]
        health_ok = (
            cris["healthy"]["alarm_rate"] <= other["healthy"]["alarm_rate"]
        )
        by_baseline[item["baseline"]] = {
            "recall_gains": gains,
            "families_with_required_gain": improved,
            "cris_health_alarm_rate_not_larger": health_ok,
            "passes": health_ok
            and len(improved)
            >= int(rule["minimum_fault_families_with_recall_gain"]),
        }
    return by_baseline


def run(charter_path: Path, output_root: Path) -> int:
    started = time.perf_counter()
    charter = load_study_charter(charter_path)
    output_root.mkdir(parents=True, exist_ok=True)
    print("[stage] simulate healthy model-fit role", flush=True)
    fit_batch = simulate_role(charter, "healthy_model_fit")
    print("[stage] train frozen CRIS-v9 model on CUDA", flush=True)
    model = fit_model(fit_batch, charter, device="cuda")
    model_path = model.save_npz(output_root / "cris_v9_model.npz")
    print(
        f"[gpu] {model.attestation.cuda_device_name}; "
        f"loss {model.loss_history[0]:.6f} -> {model.loss_history[-1]:.6f}",
        flush=True,
    )

    print("[stage] point-reference innovations", flush=True)
    reference_batch = simulate_role(charter, "healthy_point_reference")
    reference_innovations = innovation_runs(
        model,
        reference_batch,
        charter,
        device="cuda",
        progress=_innovation_progress("healthy_point_reference"),
    )
    print("[stage] trajectory-threshold innovations", flush=True)
    threshold_batch = simulate_role(charter, "healthy_threshold_calibration")
    threshold_innovations = innovation_runs(
        model,
        threshold_batch,
        charter,
        device="cuda",
        progress=_innovation_progress("healthy_threshold_calibration"),
    )
    development_data = _make_role_data(
        DEVELOPMENT_ROLES,
        charter,
        model,
        confirmation_authorized=False,
    )

    candidate_results: list[CandidateDevelopment] = []
    for spec in candidate_specs(charter):
        print(f"[stage] candidate {spec.candidate_id}", flush=True)
        candidate_results.append(
            evaluate_candidate(
                spec,
                reference_innovations,
                threshold_innovations,
                development_data,
                charter,
                progress=_candidate_progress(spec.candidate_id),
            )
        )
    candidates = tuple(candidate_results)
    selected = select_candidate(candidates, charter)
    selected_go = selected is not None and development_go(selected, charter)

    print("[stage] frozen baselines", flush=True)
    baseline_results = evaluate_baselines(
        fit_batch,
        reference_innovations,
        threshold_batch,
        threshold_innovations,
        development_data,
        charter,
    )
    development_document = {
        "schema": "cris_v9_four_tank_development_result",
        "created_utc": _now(),
        "protocol_id": charter["protocol_id"],
        "protocol_sha256": sha256_file(charter_path),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "device": {
            "source_device": model.source_device,
            "cuda_device_name": model.attestation.cuda_device_name,
            "torch_version": model.attestation.torch_version,
            "cuda_runtime": model.attestation.cuda_runtime,
        },
        "training_loss": model.loss_history.tolist(),
        "development": _candidate_result_document(candidates, selected),
        "selected_passes_all_development_gates": selected_go,
        "baselines": [item.to_document() for item in baseline_results],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "development_result.json", development_document)
    print(
        f"[development] selected="
        f"{None if selected is None else selected.spec.candidate_id}; "
        f"GO={selected_go}",
        flush=True,
    )
    if not selected_go or selected is None:
        print("[result] DEVELOPMENT_NO_GO; confirmation not generated", flush=True)
        return 2

    print("[stage] development GO; generate independent confirmation roles", flush=True)
    confirmation_data = _make_role_data(
        CONFIRMATION_ROLES,
        charter,
        model,
        confirmation_authorized=True,
    )
    confirmation_summaries = _score_selected(selected, confirmation_data, charter)
    repeatability = _repeatability(
        selected, model, confirmation_data["confirmation_healthy"][0], charter
    )
    baseline_confirmation = _score_frozen_baselines(
        baseline_results, confirmation_data, charter
    )
    go, failures = _confirmation_go(
        confirmation_summaries, repeatability, charter
    )
    confirmation_document = {
        "schema": "cris_v9_four_tank_confirmation_result",
        "created_utc": _now(),
        "protocol_id": charter["protocol_id"],
        "protocol_sha256": sha256_file(charter_path),
        "model_sha256": sha256_file(model_path),
        "selected_candidate": selected.spec.candidate_id,
        "summaries": confirmation_summaries,
        "cuda_repeatability": repeatability,
        "baselines": baseline_confirmation,
        "superiority": _superiority(
            confirmation_summaries, baseline_confirmation, charter
        ),
        "confirmation_go": go,
        "failed_gates": failures,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "confirmation_result.json", confirmation_document)
    print(f"[result] CONFIRMATION_GO={go}; failed_gates={failures}", flush=True)
    return 0 if go else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charter", type=Path, default=DEFAULT_CHARTER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return run(args.charter.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
