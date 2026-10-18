"""Compare existing SKAB methods with fixed thresholds; no new model selection."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # Windows torch-first DLL loading contract

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mate_spc_cris_v9.baselines import (
    EnergyCUSUMBaseline, PCAT2QBaseline, PredictiveResidualCUSUMBaseline,
)
from mate_spc_cris_v9.energy_anchor_monitor import EnergyAnchoredHXMonitor
from mate_spc_cris_v9.model import CRISModel
from mate_spc_cris_v9.skab import (
    load_features, load_labels, load_preaccess_charter, partition_bounds,
    sha256_file, validate_score_label_binding,
)
from mate_spc_cris_v9.skab_external import (
    WARMUP_ROWS, binary_average_precision, evaluate_skab_path, warmup_standardize,
)
import run_ea_hx_skab_gpu as existing


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "skab_frozen_baseline_comparison"
LCDE_ROOT = PROJECT / "outputs" / "lcde_skab_v2_self_calibrated"
BASELINE_ROOTS = {
    "global": PROJECT / "outputs" / "ea_hx_skab",
    "self_starting": PROJECT / "outputs" / "ea_hx_skab_self_starting",
}


def _global_methods(fit_x, innovations):
    reference = (innovations[WARMUP_ROWS:],)
    return {
        "ea_hx_cusum": EnergyAnchoredHXMonitor.fit(reference),
        "predictive_residual_cusum": PredictiveResidualCUSUMBaseline.fit(reference),
        "pca_t2_plus_q": PCAT2QBaseline.fit((fit_x[WARMUP_ROWS:],)),
        "energy_cusum": EnergyCUSUMBaseline.fit((fit_x[WARMUP_ROWS:],)),
    }


def _supplement(score, threshold, labels, timestamps):
    """Explicit active-event metrics and a common post-commissioning AP/F1 mask."""
    all_labels = np.asarray(labels, dtype=bool)
    all_timestamps = np.asarray(timestamps)
    true_onset = int(np.flatnonzero(all_labels)[0])
    commissioning_anomaly_count = int(np.sum(all_labels[:WARMUP_ROWS]))
    score = np.asarray(score)[WARMUP_ROWS:]
    labels = all_labels[WARMUP_ROWS:]
    timestamps = all_timestamps[WARMUP_ROWS:]
    alarm = score > threshold
    rises = alarm & ~np.r_[False, alarm[:-1]]
    active_rises = np.flatnonzero(rises & labels)
    detection = int(active_rises[0]) if active_rises.size else None
    delay = None if detection is None else float(
        (timestamps[detection] - all_timestamps[true_onset]) / np.timedelta64(1, "s")
    )
    tp = int(np.sum(alarm & labels))
    fp = int(np.sum(alarm & ~labels))
    fn = int(np.sum(~alarm & labels))
    return {
        "active_event_detected": detection is not None,
        "active_event_delay_seconds": delay,
        "active_event_within60": delay is not None and delay <= 60,
        "postwarmup_f1": float(2 * tp / (2 * tp + fp + fn)),
        "postwarmup_ap": binary_average_precision(labels, score),
        "healthy_sample_false_positive_rate": float(np.mean(alarm[~labels])),
        "healthy_alarm_episodes": int(np.sum(rises & ~labels)),
        "postwarmup_anomaly_prevalence": float(np.mean(labels)),
        "commissioning_anomaly_rows": commissioning_anomaly_count,
    }


def _aggregate(records):
    summary = existing._aggregate(records)
    delays = [r["active_event_delay_seconds"] for r in records if r["active_event_detected"]]
    summary.update({
        "active_event_detected_runs": sum(r["active_event_detected"] for r in records),
        "active_event_within60_runs": sum(r["active_event_within60"] for r in records),
        "active_event_median_delay_seconds": float(np.median(delays)) if delays else None,
        "mean_postwarmup_f1": float(np.mean([r["postwarmup_f1"] for r in records])),
        "mean_postwarmup_ap": float(np.mean([r["postwarmup_ap"] for r in records])),
        "mean_healthy_sample_false_positive_rate": float(np.mean([
            r["healthy_sample_false_positive_rate"] for r in records
        ])),
    })
    return summary


def _paired_intervals(records):
    proposed = records["lcde_dual"]
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, len(proposed), size=(10000, len(proposed)))
    output = {}
    for name, comparator in records.items():
        if name == "lcde_dual":
            continue
        assert [r["run_id"] for r in comparator] == [r["run_id"] for r in proposed]
        comparison = {}
        for metric in ("postwarmup_ap", "postwarmup_f1", "active_event_detected"):
            delta = np.array([float(a[metric]) - float(b[metric])
                              for a, b in zip(proposed, comparator, strict=True)])
            comparison[metric] = {
                "mean_difference_lcde_minus_comparator": float(np.mean(delta)),
                "paired_run_bootstrap_95_percentile_interval": np.quantile(
                    np.mean(delta[indices], axis=1), [0.025, 0.975]
                ).tolist(),
            }
        output[name] = comparison
    return output


def main():
    result_path = OUTPUT / "comparison_result.json"
    if result_path.exists():
        print(f"[existing-result] {result_path}", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for predictive-model inference")
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    charter = load_preaccess_charter(existing.CHARTER_PATH)
    documents = {
        kind: json.loads((root / "development_result.json").read_text())
        for kind, root in BASELINE_ROOTS.items()
    }
    model_hash = sha256_file(existing.MODEL_PATH)
    if any(doc["model_sha256"] != model_hash for doc in documents.values()):
        raise ValueError("baseline predictive model hash mismatch")
    model = CRISModel.load_npz(existing.MODEL_PATH)
    healthy = load_features(
        existing.DATASET_ROOT / "anomaly-free/anomaly-free.csv",
        dataset_root=existing.DATASET_ROOT,
    )
    segments = {}
    for name, (start, stop) in partition_bounds(len(healthy.x)).items():
        if name in {"healthy_model_fit", "point_score_reference", "trajectory_threshold_blocks"}:
            x = warmup_standardize(healthy.x[start:stop])
            segments[name] = (x, existing._innovations(model, x)
                              if name != "healthy_model_fit" else None)
    global_methods = _global_methods(segments["healthy_model_fit"][0],
                                     segments["point_score_reference"][1])
    calibration_x, calibration_i = segments["trajectory_threshold_blocks"]
    thresholds = {}
    for kind in BASELINE_ROOTS:
        methods = global_methods if kind == "global" else existing._fit_methods(
            calibration_x, calibration_i
        )
        paths = existing._score_paths(methods, calibration_x, calibration_i)
        for name, score in paths.items():
            frozen_threshold = documents[kind]["thresholds"][name]
            np.testing.assert_allclose(np.max(score), frozen_threshold, rtol=1e-7, atol=1e-8)
            thresholds[f"{kind}/{name}"] = frozen_threshold

    # Reconstruct both published baseline variants against every saved development
    # path. This detects changed implementations without selecting on test labels.
    replay_errors = {f"{kind}/{name}": 0.0 for kind in BASELINE_ROOTS
                     for name in existing.METHOD_ORDER}
    for source in existing._corrected_paths(charter, "development"):
        features = load_features(source, dataset_root=existing.DATASET_ROOT)
        x = warmup_standardize(features.x)
        innovations = existing._innovations(model, x)
        for kind, root in BASELINE_ROOTS.items():
            methods = global_methods if kind == "global" else existing._fit_methods(x, innovations)
            paths = existing._score_paths(methods, x, innovations)
            saved = root / "scores/development" / (features.run_id.replace("/", "__") + ".npz")
            with np.load(saved, allow_pickle=False) as archive:
                assert str(archive["source_sha256"].item()) == features.source_sha256
                for name, score in paths.items():
                    np.testing.assert_allclose(score, archive[name], rtol=1e-7, atol=1e-8)
                    key = f"{kind}/{name}"
                    replay_errors[key] = max(replay_errors[key], float(np.max(np.abs(score - archive[name]))))
    print(f"[development-replay] all 18 x 8 paths match; max error={max(replay_errors.values()):.3g}", flush=True)

    lcde_result = json.loads((LCDE_ROOT / "final_result.json").read_text())
    lcde_manifest = json.loads((LCDE_ROOT / "final_score_manifest.json").read_text())
    assert sha256_file(LCDE_ROOT / "final_score_manifest.json") == lcde_result["score_manifest_sha256"]
    lcde_records = {r["run_id"]: r for r in lcde_manifest["records"]}
    for mode, threshold in lcde_result["thresholds"].items():
        thresholds[f"lcde_{mode}"] = threshold
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scored = []
    for source in existing._corrected_paths(charter, "final"):
        features = load_features(source, dataset_root=existing.DATASET_ROOT)
        x = warmup_standardize(features.x)
        innovations = existing._innovations(model, x)
        paths = {}
        for kind in BASELINE_ROOTS:
            methods = global_methods if kind == "global" else existing._fit_methods(x, innovations)
            paths.update({f"{kind}/{name}": score for name, score in
                          existing._score_paths(methods, x, innovations).items()})
        record = lcde_records[features.run_id]
        score_file = PROJECT / record["score_file"]
        assert features.source_sha256 == record["source_sha256"]
        assert sha256_file(score_file) == record["score_file_sha256"]
        with np.load(score_file, allow_pickle=False) as archive:
            for mode in ("dual", "raw_only", "predictive_only"):
                paths[f"lcde_{mode}"] = np.array(archive[mode], copy=True)
        target = OUTPUT / (features.run_id.replace("/", "__") + ".npz")
        np.savez_compressed(target, **paths)
        scored.append((features, paths))
        print(f"[final-score] {features.run_id}", flush=True)

    records = {name: [] for name in thresholds}
    for features, paths in scored:
        labels = load_labels(existing.DATASET_ROOT / features.run_id, dataset_root=existing.DATASET_ROOT)
        validate_score_label_binding({"run_id": features.run_id,
            "source_sha256": features.source_sha256, "row_count": len(features.x)}, labels)
        for name, score in paths.items():
            metrics = evaluate_skab_path(score, thresholds[name], labels.anomaly, features.timestamps)
            metrics.update(_supplement(score, thresholds[name], labels.anomaly, features.timestamps))
            metrics["run_id"] = features.run_id
            records[name].append(metrics)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "purpose": "fixed-method comparison; engineering gates are descriptive only",
        "interpretation": "Previously evaluated holdout; no method or threshold tuned in this comparison. Bootstrap intervals exploratory, no multiplicity correction.",
        "metric_note": "Legacy detected permits any rising edge after onset, even after recovery. Active-event detection requires the rising edge to coincide with an anomaly label. AP/F1 supplemental scores exclude first 300 commissioning rows for every method.",
        "model_sha256": model_hash,
        "development_replay_max_errors": replay_errors,
        "thresholds": thresholds,
        "aggregate": {name: _aggregate(rows) for name, rows in records.items()},
        "clean_commissioning_sensitivity": {
            "exclusion_basis": "first 300 rows contain an anomaly label; identical exclusion for every method, reported alongside all 16",
            "excluded_run_ids": [r["run_id"] for r in records["lcde_dual"]
                                 if r["commissioning_anomaly_rows"] > 0],
            "aggregate": {name: _aggregate([r for r in rows if r["commissioning_anomaly_rows"] == 0])
                          for name, rows in records.items()},
        },
        "per_run": records,
        "paired_comparisons": _paired_intervals(records),
    }
    existing._write(result_path, result)
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print(f"[result] {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
