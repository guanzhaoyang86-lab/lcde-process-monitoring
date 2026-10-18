"""External SKAB v0.9.0 evaluation for the frozen EA-HX-CUSUM monitor."""

from __future__ import annotations

import torch  # noqa: F401  # Windows torch-first DLL loading contract

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from mate_spc_cris_v9.baselines import (
    EnergyCUSUMBaseline,
    PCAT2QBaseline,
    PredictiveResidualCUSUMBaseline,
)
from mate_spc_cris_v9.energy_anchor_monitor import EnergyAnchoredHXMonitor
from mate_spc_cris_v9.model import CRISConfig, CRISModel, CRISTrainingConfig
from mate_spc_cris_v9.skab import (
    SENSOR_COLUMNS,
    assigned_paths,
    bind_scores_to_source,
    load_features,
    load_labels,
    load_preaccess_charter,
    partition_bounds,
    sha256_file,
    validate_score_label_binding,
)
from mate_spc_cris_v9.skab_external import (
    WARMUP_ROWS,
    evaluate_skab_path,
    prepend_warmup,
    warmup_standardize,
)


PROJECT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT / "data" / "skab-v0.9.0" / "SKAB-0.9.0" / "data"
ARCHIVE_PATH = PROJECT / "data" / "skab-v0.9.0.zip"
CHARTER_PATH = PROJECT / "protocols" / "cris_v9_skab_preaccess_charter.json"
OUTPUT_ROOT = PROJECT / "outputs" / "ea_hx_skab_self_starting"
MODEL_PATH = PROJECT / "outputs" / "ea_hx_skab" / "ea_hx_skab_model.npz"
DEVELOPMENT_RESULT = OUTPUT_ROOT / "development_result.json"
FINAL_RESULT = OUTPUT_ROOT / "final_result.json"
SCORE_ROOT = OUTPUT_ROOT / "scores"

# The v0.9.0 tag uses zero-based valve filenames; the pre-download charter
# mistakenly listed those two terminal files as if numbering were one-based.
PATH_CORRECTIONS = {
    "valve1/16.csv": "valve1/0.csv",
    "valve2/4.csv": "valve2/0.csv",
}

METHOD_ORDER = (
    "ea_hx_cusum",
    "predictive_residual_cusum",
    "pca_t2_plus_q",
    "energy_cusum",
)


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(path)


def _dummy_covariates(length: int) -> np.ndarray:
    return np.zeros((length, 1), dtype=np.float32)


def _innovations(model: CRISModel, x: np.ndarray) -> np.ndarray:
    return model.innovations(
        x,
        _dummy_covariates(x.shape[0]),
        np.ones(x.shape, dtype=np.bool_),
        device="cuda",
        batch_size=2048,
    )


def _corrected_paths(charter: Mapping[str, Any], role: str) -> tuple[Path, ...]:
    raw = assigned_paths(DATASET_ROOT, charter, role=role)
    corrected: list[Path] = []
    for path in raw:
        relative = path.relative_to(DATASET_ROOT.resolve()).as_posix()
        mapped = PATH_CORRECTIONS.get(relative, relative)
        corrected.append((DATASET_ROOT / mapped).resolve())
    paths = tuple(corrected)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing assigned SKAB files: {missing}")
    return paths


def _load_or_train_model(fit_x: np.ndarray) -> CRISModel:
    if MODEL_PATH.exists():
        model = CRISModel.load_npz(MODEL_PATH)
        if model.config.target_dim != len(SENSOR_COLUMNS) or model.config.covariate_dim != 1:
            raise ValueError("saved SKAB model dimensions are incompatible")
        print(f"[model-cache] {MODEL_PATH}", flush=True)
        return model
    print("[train] 40 deterministic epochs on RTX GPU", flush=True)
    model = CRISModel.fit(
        fit_x,
        _dummy_covariates(fit_x.shape[0]),
        np.ones(fit_x.shape, dtype=np.bool_),
        config=CRISConfig(target_dim=len(SENSOR_COLUMNS), covariate_dim=1),
        training_config=CRISTrainingConfig(
            epochs=40,
            batch_size=1024,
            learning_rate=3.0e-4,
            weight_decay=1.0e-4,
            gradient_clip_norm=1.0,
            seed=20260915,
            device="cuda",
        ),
        metadata={
            "dataset": "SKAB-v0.9.0",
            "preprocessing": "first-300-row within-run standardization",
            "targets": list(SENSOR_COLUMNS),
            "covariate": "constant_zero_dummy",
        },
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model.save_npz(MODEL_PATH)
    print(f"[model-saved] {MODEL_PATH}", flush=True)
    return model


def _fit_methods(
    local_x: np.ndarray,
    local_innovations: np.ndarray,
) -> dict[str, Any]:
    reference = (local_innovations[:WARMUP_ROWS],)
    return {
        "ea_hx_cusum": EnergyAnchoredHXMonitor.fit(reference),
        "predictive_residual_cusum": PredictiveResidualCUSUMBaseline.fit(reference),
        "pca_t2_plus_q": PCAT2QBaseline.fit((local_x[:WARMUP_ROWS],)),
        "energy_cusum": EnergyCUSUMBaseline.fit((local_x[:WARMUP_ROWS],)),
    }


def _score_paths(
    methods: Mapping[str, Any],
    x: np.ndarray,
    innovations: np.ndarray,
) -> dict[str, np.ndarray]:
    total = x.shape[0]
    post_innovations = innovations[WARMUP_ROWS:]
    post_targets = x[WARMUP_ROWS:]
    output: dict[str, np.ndarray] = {}
    for name, method in methods.items():
        values = post_innovations if name in {
            "ea_hx_cusum",
            "predictive_residual_cusum",
        } else post_targets
        output[name] = prepend_warmup(
            method.score(values).path, total=total, warmup=WARMUP_ROWS
        )
    return output


def _calibrate_thresholds(
    calibration_x: np.ndarray,
    calibration_innovations: np.ndarray,
) -> dict[str, float]:
    methods = _fit_methods(calibration_x, calibration_innovations)
    paths = _score_paths(methods, calibration_x, calibration_innovations)
    return {name: float(np.max(path)) for name, path in paths.items()}


def _rising_edges(alarm: np.ndarray) -> np.ndarray:
    return alarm & ~np.r_[False, alarm[:-1]]


def _healthy_summary(
    paths: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    timestamps: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    duration = float(
        (timestamps[-1] - timestamps[0]).astype("timedelta64[ns]").astype(np.int64)
    ) / 1.0e9
    output: dict[str, dict[str, float | int]] = {}
    for name, path in paths.items():
        alarm = np.asarray(path > thresholds[name], dtype=np.bool_)
        episodes = int(np.sum(_rising_edges(alarm)))
        output[name] = {
            "alarm_episodes": episodes,
            "alarm_occupancy": float(np.mean(alarm, dtype=np.float64)),
            "episodes_per_hour": float(episodes * 3600.0 / duration),
        }
    return output


def _persist_score_paths(
    role: str,
    run_id: str,
    source_sha256: str,
    paths: Mapping[str, np.ndarray],
    model_sha256: str,
) -> Path:
    directory = SCORE_ROOT / role
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (run_id.replace("/", "__") + ".npz")
    np.savez_compressed(
        target,
        run_id=np.asarray(run_id),
        source_sha256=np.asarray(source_sha256),
        model_sha256=np.asarray(model_sha256),
        **{name: np.asarray(paths[name], dtype=np.float64) for name in METHOD_ORDER},
    )
    return target


def _nearest_rank(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = max(1, int(math.ceil(probability * ordered.size)))
    return float(ordered[rank - 1])


def _aggregate(per_run: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delays = [float(item["delay_seconds"]) for item in per_run if item["detected"]]
    return {
        "runs": len(per_run),
        "detected_runs": sum(bool(item["detected"]) for item in per_run),
        "event_recall": float(np.mean([item["detected"] for item in per_run])),
        "within_60_runs": sum(bool(item["within_60"]) for item in per_run),
        "within_60_rate": float(np.mean([item["within_60"] for item in per_run])),
        "pre_event_false_alarm_experiments": sum(
            bool(item["pre_event_false_alarm"]) for item in per_run
        ),
        "conditional_median_delay_seconds": (
            float(np.median(np.asarray(delays, dtype=np.float64))) if delays else None
        ),
        "conditional_nearest_rank_p90_delay_seconds": _nearest_rank(delays, 0.90),
        "mean_alarm_occupancy": float(
            np.mean([item["occupancy"] for item in per_run])
        ),
        "mean_alarm_episodes_per_hour": float(
            np.mean([item["episodes_per_hour"] for item in per_run])
        ),
        "mean_unadjusted_f1": float(
            np.mean([item["unadjusted_f1"] for item in per_run])
        ),
        "mean_average_precision": float(
            np.mean([item["average_precision"] for item in per_run])
        ),
    }


def _evaluate_files(
    role: str,
    paths: Sequence[Path],
    model: CRISModel,
    thresholds: Mapping[str, float],
    model_sha256: str,
) -> dict[str, Any]:
    scored: list[tuple[Any, dict[str, np.ndarray], Path]] = []
    for index, path in enumerate(paths, start=1):
        features = load_features(path, dataset_root=DATASET_ROOT)
        if features.x.shape[0] <= WARMUP_ROWS:
            raise ValueError(f"SKAB run is shorter than warmup: {features.run_id}")
        x = warmup_standardize(features.x)
        innovations = _innovations(model, x)
        methods = _fit_methods(x, innovations)
        score_paths = _score_paths(methods, x, innovations)
        score_file = _persist_score_paths(
            role,
            features.run_id,
            features.source_sha256,
            score_paths,
            model_sha256,
        )
        scored.append((features, score_paths, score_file))
        print(f"[score-{role}] {index}/{len(paths)} {features.run_id}", flush=True)

    per_method: dict[str, list[dict[str, Any]]] = {name: [] for name in METHOD_ORDER}
    # Labels are opened only after every score path in this role is persisted.
    for features, score_paths, score_file in scored:
        metadata = bind_scores_to_source(features, score_file)
        labels = load_labels(DATASET_ROOT / features.run_id, dataset_root=DATASET_ROOT)
        validate_score_label_binding(metadata, labels)
        for name in METHOD_ORDER:
            metrics = evaluate_skab_path(
                score_paths[name],
                thresholds[name],
                labels.anomaly,
                features.timestamps,
            )
            metrics.update({
                "run_id": features.run_id,
                "group": features.run_id.split("/", 1)[0],
            })
            per_method[name].append(metrics)

    aggregate = {name: _aggregate(values) for name, values in per_method.items()}
    return {"aggregate": aggregate, "per_run": per_method}


def _development_passes(
    aggregate: Mapping[str, Any], healthy: Mapping[str, Any]
) -> tuple[bool, dict[str, bool]]:
    proposed = aggregate["ea_hx_cusum"]
    checks = {
        "within_60_at_least_14_of_18": proposed["within_60_runs"] >= 14,
        "pre_event_false_alarm_experiments_at_most_3": (
            proposed["pre_event_false_alarm_experiments"] <= 3
        ),
        "conditional_median_delay_at_most_30_seconds": (
            proposed["conditional_median_delay_seconds"] is not None
            and proposed["conditional_median_delay_seconds"] <= 30.0
        ),
        "healthy_audit_alarm_episodes_at_most_1": (
            healthy["ea_hx_cusum"]["alarm_episodes"] <= 1
        ),
    }
    return all(checks.values()), checks


def main() -> int:
    started = time.perf_counter()
    if DEVELOPMENT_RESULT.exists() or FINAL_RESULT.exists():
        raise FileExistsError("SKAB EA-HX result already exists")
    charter = load_preaccess_charter(CHARTER_PATH)
    development_paths = _corrected_paths(charter, "development")
    final_paths = _corrected_paths(charter, "final")
    if set(development_paths).intersection(final_paths):
        raise ValueError("corrected SKAB development/final paths overlap")

    healthy_path = DATASET_ROOT / "anomaly-free" / "anomaly-free.csv"
    healthy = load_features(healthy_path, dataset_root=DATASET_ROOT)
    bounds = partition_bounds(healthy.x.shape[0])
    segments = {
        name: warmup_standardize(healthy.x[first:last])
        for name, (first, last) in bounds.items()
    }
    model = _load_or_train_model(segments["healthy_model_fit"])
    model_sha256 = sha256_file(MODEL_PATH)
    print(f"[gpu] {model.attestation.cuda_device_name}", flush=True)
    calibration_innovations = _innovations(
        model, segments["trajectory_threshold_blocks"]
    )
    thresholds = _calibrate_thresholds(
        segments["trajectory_threshold_blocks"],
        calibration_innovations,
    )

    audit_first, audit_last = bounds["healthy_development_audit"]
    audit_features = type(healthy)(
        run_id=healthy.run_id,
        timestamps=healthy.timestamps[audit_first:audit_last],
        x=healthy.x[audit_first:audit_last],
        feature_names=healthy.feature_names,
        sample_period_seconds=healthy.sample_period_seconds,
        source_sha256=healthy.source_sha256,
    )
    audit_innovations = _innovations(model, segments["healthy_development_audit"])
    audit_methods = _fit_methods(
        segments["healthy_development_audit"], audit_innovations
    )
    audit_paths = _score_paths(
        audit_methods, segments["healthy_development_audit"], audit_innovations
    )
    healthy_summary = _healthy_summary(
        audit_paths, thresholds, audit_features.timestamps
    )

    development = _evaluate_files(
        "development",
        development_paths,
        model,
        thresholds,
        model_sha256,
    )
    development_go, checks = _development_passes(
        development["aggregate"], healthy_summary
    )
    development_document = {
        "schema": "ea_hx_cusum_skab_development",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "name": "SKAB",
            "version": "v0.9.0",
            "archive_sha256": sha256_file(ARCHIVE_PATH),
            "path_corrections": PATH_CORRECTIONS,
        },
        "device": model.attestation.cuda_device_name,
        "model_sha256": model_sha256,
        "training_loss_first": float(model.loss_history[0]),
        "training_loss_last": float(model.loss_history[-1]),
        "preprocessing": {
            "within_run_warmup_standardization_rows": WARMUP_ROWS,
            "within_run_residual_calibration_rows": WARMUP_ROWS,
            "covariates": "one constant-zero compatibility channel",
            "monitoring_starts_after_warmup": True,
        },
        "thresholds": thresholds,
        "threshold_calibration": (
            "strict maximum of one held-out continuous healthy partition; "
            "descriptive external calibration, no independent-run guarantee"
        ),
        "healthy_audit": healthy_summary,
        "development": development,
        "development_checks": checks,
        "development_go": development_go,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write(DEVELOPMENT_RESULT, development_document)
    print(f"[development-go] {development_go}", flush=True)
    if not development_go:
        return 2

    final = _evaluate_files(
        "final", final_paths, model, thresholds, model_sha256
    )
    final_document = {
        "schema": "ea_hx_cusum_skab_final",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "EA-HX-CUSUM",
        "dataset": "SKAB-v0.9.0",
        "device": model.attestation.cuda_device_name,
        "model_sha256": model_sha256,
        "thresholds": thresholds,
        "development_result": str(DEVELOPMENT_RESULT.resolve()),
        "final": final,
        "elapsed_seconds_including_development": time.perf_counter() - started,
    }
    _write(FINAL_RESULT, final_document)
    print(f"[result] {FINAL_RESULT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
