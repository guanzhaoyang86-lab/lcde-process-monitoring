"""Development-only threshold sensitivity for the SKAB EA-HX path."""

from __future__ import annotations

import torch  # noqa: F401

import json
from pathlib import Path

import numpy as np

from mate_spc_cris_v9.energy_anchor_monitor import EnergyAnchoredHXMonitor
from mate_spc_cris_v9.model import CRISModel
from mate_spc_cris_v9.skab import (
    load_features,
    load_labels,
    load_preaccess_charter,
    partition_bounds,
)
from mate_spc_cris_v9.skab_external import (
    WARMUP_ROWS,
    evaluate_skab_path,
    prepend_warmup,
    warmup_standardize,
)


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data" / "skab-v0.9.0" / "SKAB-0.9.0" / "data"
OUTPUT = PROJECT / "outputs" / "ea_hx_skab"
QUANTILES = (0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)


def _innovations(model: CRISModel, x: np.ndarray) -> np.ndarray:
    return model.innovations(
        x,
        np.zeros((x.shape[0], 1), dtype=np.float32),
        np.ones(x.shape, dtype=np.bool_),
        device="cuda",
        batch_size=2048,
    )


def _episodes(path: np.ndarray, threshold: float) -> int:
    alarm = path > threshold
    return int(np.sum(alarm & ~np.r_[False, alarm[:-1]]))


def main() -> int:
    charter = load_preaccess_charter(
        PROJECT / "protocols" / "cris_v9_skab_preaccess_charter.json"
    )
    healthy = load_features(ROOT / "anomaly-free/anomaly-free.csv", dataset_root=ROOT)
    bounds = partition_bounds(healthy.x.shape[0])
    standardized = {
        name: warmup_standardize(healthy.x[first:last])
        for name, (first, last) in bounds.items()
    }
    model = CRISModel.load_npz(OUTPUT / "ea_hx_skab_model.npz")
    reference = _innovations(model, standardized["point_score_reference"])
    monitor = EnergyAnchoredHXMonitor.fit((reference[WARMUP_ROWS:],))
    calibration = _innovations(model, standardized["trajectory_threshold_blocks"])
    calibration_path = monitor.score(calibration[WARMUP_ROWS:]).path
    audit = _innovations(model, standardized["healthy_development_audit"])
    audit_subpath = monitor.score(audit[WARMUP_ROWS:]).path

    development = json.loads((OUTPUT / "development_result.json").read_text())
    run_ids = [item["run_id"] for item in development["development"]["per_run"]["ea_hx_cusum"]]
    paths: dict[str, np.ndarray] = {}
    labels = {}
    timestamps = {}
    for run_id in run_ids:
        score_file = OUTPUT / "scores" / "development" / (run_id.replace("/", "__") + ".npz")
        with np.load(score_file, allow_pickle=False) as archive:
            paths[run_id] = np.array(archive["ea_hx_cusum"], copy=True)
        source = ROOT / run_id
        labels[run_id] = load_labels(source, dataset_root=ROOT).anomaly
        timestamps[run_id] = load_features(source, dataset_root=ROOT).timestamps

    rows = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(calibration_path, quantile, method="higher"))
        outcomes = [
            evaluate_skab_path(paths[name], threshold, labels[name], timestamps[name])
            for name in run_ids
        ]
        delays = [float(item["delay_seconds"]) for item in outcomes if item["detected"]]
        rows.append(
            {
                "calibration_quantile": quantile,
                "threshold": threshold,
                "detected": sum(bool(item["detected"]) for item in outcomes),
                "within_60": sum(bool(item["within_60"]) for item in outcomes),
                "pre_event_false_alarm_experiments": sum(
                    bool(item["pre_event_false_alarm"]) for item in outcomes
                ),
                "median_delay": float(np.median(delays)) if delays else None,
                "healthy_audit_episodes": _episodes(audit_subpath, threshold),
            }
        )
    direction = []
    for run_id in run_ids:
        source = ROOT / run_id
        features = load_features(source, dataset_root=ROOT)
        x = warmup_standardize(features.x)
        run_innovations = _innovations(model, x)
        increments = monitor.energy_baseline.score(
            run_innovations[WARMUP_ROWS:]
        ).point_score
        event = int(np.flatnonzero(labels[run_id])[0]) - WARMUP_ROWS
        anomaly_after_warmup = labels[run_id][WARMUP_ROWS:]
        direction.append(
            {
                "run_id": run_id,
                "pre_mean_increment": float(np.mean(increments[:event])),
                "anomaly_mean_increment": float(
                    np.mean(increments[anomaly_after_warmup])
                ),
            }
        )
    print(json.dumps({"thresholds": rows, "energy_direction": direction}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
