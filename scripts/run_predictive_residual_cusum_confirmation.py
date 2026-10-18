"""Independent confirmation of the strongest frozen development method."""

from __future__ import annotations

import torch  # noqa: F401  # must precede NumPy-backed project imports on Windows

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from mate_spc_cris_v9.baselines import PredictiveResidualCUSUMBaseline
from mate_spc_cris_v9.model import CRISModel
from mate_spc_cris_v9.study import (
    FAULT_KINDS,
    innovation_runs,
    load_study_charter,
    model_arrays,
    sha256_file,
    simulate_role,
    summarize_paths,
)


PROJECT = Path(__file__).resolve().parents[1]
CHARTER = PROJECT / "protocols" / "cris_v9_four_tank_study_charter.json"
OUTPUT = PROJECT / "outputs" / "cris_v9_four_tank"
MODEL = OUTPUT / "cris_v9_model.npz"
RESULT = OUTPUT / "predictive_residual_cusum_confirmation.json"
CONFIRMATION_ROLES = (
    "confirmation_healthy",
    "confirmation_sensor_bias",
    "confirmation_intermittent_pump",
    "confirmation_valve_stiction",
    "confirmation_dependence_shift",
    "confirmation_covariate_shift_only",
)


def _progress(role: str):
    def report(done: int, total: int) -> None:
        if done == 1 or done % 50 == 0 or done == total:
            print(f"[innovations] {role}: {done}/{total}", flush=True)

    return report


def _write(document: dict[str, Any]) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(RESULT)


def _go(summaries: dict[str, dict[str, Any]], repeat: dict[str, Any], charter: Any):
    gate = charter["confirmation_go_gates"]
    failures: list[str] = []
    healthy = summaries["healthy"]
    if healthy["alarm_runs"] > gate["healthy_alarm_runs_max"]:
        failures.append("healthy_alarm_runs")
    if (
        healthy["one_sided_cp95_upper"]
        > gate["healthy_one_sided_cp95_upper_max"]
    ):
        failures.append("healthy_cp95_upper")
    if (
        summaries["covariate_shift_only"]["alarm_rate"]
        > gate["covariate_shift_only_alarm_rate_max"]
    ):
        failures.append("covariate_shift_only_alarm_rate")
    for kind in FAULT_KINDS:
        required = (
            gate["intermittent_pump_recall_min"]
            if kind == "intermittent_pump"
            else gate[f"{kind}_recall_min"]
        )
        if summaries[kind]["recall"] < required:
            failures.append(f"{kind}_recall")
        if (
            summaries[kind]["median_relative_delay"]
            > gate["each_fault_median_relative_delay_max"]
        ):
            failures.append(f"{kind}_median_relative_delay")
    if repeat["score_max_abs_error"] > gate["cuda_repeat_score_atol"]:
        failures.append("cuda_repeat_score")
    if not repeat["alarm_decisions_exact"]:
        failures.append("cuda_alarm_decisions")
    return not failures, failures


def main() -> int:
    started = time.perf_counter()
    if RESULT.exists():
        raise FileExistsError(f"confirmation result already exists: {RESULT}")
    charter = load_study_charter(CHARTER)
    model = CRISModel.load_npz(MODEL)
    print(
        f"[gpu] loaded {model.attestation.cuda_device_name}; "
        "freeze method=predictive_residual_cusum",
        flush=True,
    )

    reference = simulate_role(charter, "healthy_point_reference")
    reference_innovations = innovation_runs(
        model,
        reference,
        charter,
        device="cuda",
        progress=_progress("healthy_point_reference"),
    )
    baseline = PredictiveResidualCUSUMBaseline.fit(reference_innovations)

    calibration = simulate_role(charter, "healthy_threshold_calibration")
    calibration_innovations = innovation_runs(
        model,
        calibration,
        charter,
        device="cuda",
        progress=_progress("healthy_threshold_calibration"),
    )
    threshold = baseline.calibrate_trajectory_threshold(
        calibration_innovations, alpha=float(charter["method"]["trajectory_alpha"])
    )
    print(
        f"[calibration] rank={threshold.rank}/{threshold.calibration_maxima.size}; "
        f"threshold={threshold.threshold:.9f}",
        flush=True,
    )

    summaries: dict[str, dict[str, Any]] = {}
    first_healthy_batch = None
    for role in CONFIRMATION_ROLES:
        print(f"[simulate-confirmation] {role}", flush=True)
        batch = simulate_role(charter, role, confirmation_authorized=True)
        innovations = innovation_runs(
            model,
            batch,
            charter,
            device="cuda",
            progress=_progress(role),
        )
        paths = tuple(baseline.score(run).path for run in innovations)
        summaries[batch.fault_kind[0]] = summarize_paths(
            paths, batch, threshold.threshold, charter
        )
        if role == "confirmation_healthy":
            first_healthy_batch = batch

    if first_healthy_batch is None:
        raise RuntimeError("confirmation healthy role was not evaluated")
    targets, covariates, mask = model_arrays(first_healthy_batch)
    kwargs = {
        "device": "cuda",
        "batch_size": int(charter["method"]["inference_batch_size"]),
    }
    first = model.innovations(targets[0], covariates[0], mask[0], **kwargs)
    second = model.innovations(targets[0], covariates[0], mask[0], **kwargs)
    first_path = baseline.score(first).path
    second_path = baseline.score(second).path
    repeat = {
        "innovation_arrays_exact": bool(np.array_equal(first, second, equal_nan=True)),
        "score_max_abs_error": float(np.max(np.abs(first_path - second_path))),
        "alarm_decisions_exact": bool(
            np.array_equal(
                first_path > threshold.threshold,
                second_path > threshold.threshold,
            )
        ),
        "source_device": model.source_device,
        "cuda_device_name": model.attestation.cuda_device_name,
    }
    go, failures = _go(summaries, repeat, charter)
    document = {
        "schema": "predictive_residual_cusum_independent_confirmation",
        "created_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "selection_disclosure": (
            "selected as the strongest of six fixed CRIS/baseline methods on "
            "development data; confirmation seeds were not previously generated"
        ),
        "protocol_id": charter["protocol_id"],
        "protocol_sha256": sha256_file(CHARTER),
        "model_sha256": sha256_file(MODEL),
        "method": {
            "id": baseline.name,
            "energy_center": baseline.energy_center,
            "energy_scale": baseline.energy_scale,
            "allowance": baseline.allowance,
        },
        "threshold": {
            "value": threshold.threshold,
            "alpha": threshold.alpha,
            "ascending_rank": threshold.rank,
            "calibration_run_count": int(threshold.calibration_maxima.size),
        },
        "summaries": summaries,
        "cuda_repeatability": repeat,
        "confirmation_go": go,
        "failed_gates": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(document)
    print(f"[result] CONFIRMATION_GO={go}; failed_gates={failures}", flush=True)
    return 0 if go else 3


if __name__ == "__main__":
    raise SystemExit(main())
