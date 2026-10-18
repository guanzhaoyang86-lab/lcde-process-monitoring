"""Development-only grouped separability audit for the MSCD-TCN feature family.

This script is deliberately restricted to the already-consumed Extended TEP
Mode 1, 50% magnitude development cells.  It never constructs an access plan
for the 25% calibration cells or the 75% confirmation cells.  The diagnostic
uses a shallow ExtraTrees classifier on causal multi-scale summaries; it is
not the final sequential monitor.  Its purpose is to establish whether the
new variance/covariance and valve-stiction features carry out-of-group signal
before spending time fitting a temporal neural network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from mate_spc_mstcn.features import (
    CLASS_IDS,
    DYNAMIC_FEATURE_COUNT,
    RobustRunTransform,
)
from mate_spc_pclr.backend_v2 import load_numeric_runs


DEVELOPMENT_MAGNITUDE = 50
DEVELOPMENT_RUNS = tuple(range(1, 101))
SOURCE_IDVS = (1, 3, 9, 15)
HEALTHY_WINDOW = (100, 600)
FAULT_WINDOW = (440, 940)
ENDPOINTS = (191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 499)
SUMMARY_WINDOWS = (8, 32, 64)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _development_access_plan() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for simulation_run in DEVELOPMENT_RUNS:
        rows.append(
            {
                "idv": 1,
                "simulation_run": simulation_run,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "window_half_open": list(HEALTHY_WINDOW),
                "role": "healthy",
            }
        )
        rows.extend(
            {
                "idv": source_id,
                "simulation_run": simulation_run,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "window_half_open": list(FAULT_WINDOW),
                "role": f"fault-{source_id}",
            }
            for source_id in SOURCE_IDVS[1:]
        )
    return tuple(rows)


def _load_fold_map(protocol_path: Path) -> dict[int, int]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = payload["development"]["run_to_fold"]
    result = {int(key): int(value) for key, value in raw.items()}
    if set(result) != set(DEVELOPMENT_RUNS) or set(result.values()) != set(range(5)):
        raise ValueError("development protocol does not contain the canonical five-fold map")
    return result


def _label_for_run(run: Any) -> int:
    source_id = int(run.source_idv)
    if source_id not in SOURCE_IDVS:
        raise ValueError("unexpected source ID in the development-only diagnostic")
    return 0 if source_id == 1 else source_id


def _summary_rows(path: np.ndarray) -> np.ndarray:
    if path.shape != (500, DYNAMIC_FEATURE_COUNT):
        raise ValueError("dynamic feature geometry changed")
    rows: list[np.ndarray] = []
    for endpoint in ENDPOINTS:
        pieces = [path[endpoint]]
        stop = endpoint + 1
        for width in SUMMARY_WINDOWS:
            start = stop - width
            segment = path[start:stop]
            pieces.append(np.mean(segment, axis=0))
            pieces.append(np.std(segment, axis=0, ddof=0))
        rows.append(np.concatenate(pieces))
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite causal summary features")
    return matrix


def _fold_arrays(
    groups: Mapping[int, Any],
    fold_map: Mapping[int, int],
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_runs = [run for run in DEVELOPMENT_RUNS if fold_map[run] != fold]
    valid_runs = [run for run in DEVELOPMENT_RUNS if fold_map[run] == fold]
    training_raw = {
        condition.name: condition.x
        for simulation_run in train_runs
        for condition in groups[simulation_run].conditions
    }
    transform = RobustRunTransform.fit(training_raw)

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    valid_x: list[np.ndarray] = []
    valid_y: list[int] = []
    valid_group: list[int] = []
    valid_endpoint: list[int] = []
    for simulation_run in DEVELOPMENT_RUNS:
        is_validation = simulation_run in valid_runs
        for condition in groups[simulation_run].conditions:
            rows = _summary_rows(transform.transform_run(condition.x))
            label = _label_for_run(condition)
            if is_validation:
                valid_x.append(rows)
                valid_y.extend([label] * len(ENDPOINTS))
                valid_group.extend([simulation_run] * len(ENDPOINTS))
                valid_endpoint.extend(ENDPOINTS)
            else:
                train_x.append(rows)
                train_y.extend([label] * len(ENDPOINTS))
    return (
        np.concatenate(train_x, axis=0),
        np.asarray(train_y, dtype=np.int64),
        np.concatenate(valid_x, axis=0),
        np.asarray(valid_y, dtype=np.int64),
        np.asarray(valid_group, dtype=np.int64),
        np.asarray(valid_endpoint, dtype=np.int64),
    )


def run_diagnostic(data_dir: Path, protocol_path: Path) -> dict[str, Any]:
    source = data_dir / "TEP_Mode1.h5"
    groups = load_numeric_runs(source, _development_access_plan())
    if tuple(sorted(groups)) != DEVELOPMENT_RUNS:
        raise ValueError("development loader returned non-canonical groups")
    fold_map = _load_fold_map(protocol_path)

    all_truth: list[int] = []
    all_prediction: list[int] = []
    all_probability: list[np.ndarray] = []
    all_group: list[int] = []
    all_endpoint: list[int] = []
    binary_truth_by_fault: dict[int, list[int]] = {
        fault_id: [] for fault_id in CLASS_IDS[1:]
    }
    binary_score_by_fault: dict[int, list[float]] = {
        fault_id: [] for fault_id in CLASS_IDS[1:]
    }
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train_x, train_y, valid_x, valid_y, valid_group, valid_endpoint = _fold_arrays(
            groups, fold_map, fold
        )
        classifier = ExtraTreesClassifier(
            n_estimators=384,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260908 + fold,
            n_jobs=-1,
        )
        classifier.fit(train_x, train_y)
        if tuple(int(value) for value in classifier.classes_) != CLASS_IDS:
            raise RuntimeError("classifier class order changed")
        probability = classifier.predict_proba(valid_x)
        prediction = classifier.classes_[np.argmax(probability, axis=1)].astype(np.int64)
        cm = confusion_matrix(valid_y, prediction, labels=list(CLASS_IDS))
        recalls = {
            str(label): float(cm[index, index] / max(1, np.sum(cm[index])))
            for index, label in enumerate(CLASS_IDS)
        }
        fold_binary_auc: dict[str, float] = {}
        for fault_id in CLASS_IDS[1:]:
            training_pair = np.isin(train_y, (0, fault_id))
            validation_pair = np.isin(valid_y, (0, fault_id))
            binary_classifier = ExtraTreesClassifier(
                n_estimators=384,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=20261908 + 100 * fold + fault_id,
                n_jobs=-1,
            )
            binary_classifier.fit(train_x[training_pair], train_y[training_pair])
            if tuple(int(value) for value in binary_classifier.classes_) != (0, fault_id):
                raise RuntimeError("binary classifier class order changed")
            score = binary_classifier.predict_proba(valid_x[validation_pair])[:, 1]
            binary_truth = (valid_y[validation_pair] == fault_id).astype(np.int64)
            fold_binary_auc[str(fault_id)] = float(roc_auc_score(binary_truth, score))
            binary_truth_by_fault[fault_id].extend(binary_truth.tolist())
            binary_score_by_fault[fault_id].extend(score.tolist())
        fold_rows.append(
            {
                "fold": fold,
                "training_group_count": len({run for run in DEVELOPMENT_RUNS if fold_map[run] != fold}),
                "validation_group_count": len(set(valid_group.tolist())),
                "window_recall_by_class": recalls,
                "binary_head_fault_vs_healthy_auc": fold_binary_auc,
                "confusion_matrix_rows_truth_columns_prediction": cm.tolist(),
            }
        )
        all_truth.extend(valid_y.tolist())
        all_prediction.extend(prediction.tolist())
        all_probability.append(probability)
        all_group.extend(valid_group.tolist())
        all_endpoint.extend(valid_endpoint.tolist())

    truth = np.asarray(all_truth, dtype=np.int64)
    prediction = np.asarray(all_prediction, dtype=np.int64)
    probability = np.concatenate(all_probability, axis=0)
    groups_array = np.asarray(all_group, dtype=np.int64)
    endpoint_array = np.asarray(all_endpoint, dtype=np.int64)
    cm = confusion_matrix(truth, prediction, labels=list(CLASS_IDS))
    recalls = {
        str(label): float(cm[index, index] / max(1, np.sum(cm[index])))
        for index, label in enumerate(CLASS_IDS)
    }
    early = endpoint_array <= 287
    early_cm = confusion_matrix(truth[early], prediction[early], labels=list(CLASS_IDS))
    early_recalls = {
        str(label): float(early_cm[index, index] / max(1, np.sum(early_cm[index])))
        for index, label in enumerate(CLASS_IDS)
    }
    fault_probability = 1.0 - probability[:, 0]
    binary_auc_by_fault: dict[str, float] = {}
    anomaly_detection_rate_by_fault: dict[str, float] = {}
    for fault_id in CLASS_IDS[1:]:
        pair = np.isin(truth, (0, fault_id))
        binary_auc_by_fault[str(fault_id)] = float(
            roc_auc_score((truth[pair] == fault_id).astype(np.int64), fault_probability[pair])
        )
        fault_rows = truth == fault_id
        anomaly_detection_rate_by_fault[str(fault_id)] = float(
            np.mean(prediction[fault_rows] != 0)
        )
    endpoint_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        selected = endpoint_array == endpoint
        selected_truth = truth[selected]
        selected_prediction = prediction[selected]
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "post_fault_delay_samples": endpoint - 160,
                "healthy_false_positive_rate": float(
                    np.mean(selected_prediction[selected_truth == 0] != 0)
                ),
                "fault_detection_rate": {
                    str(fault_id): float(
                        np.mean(selected_prediction[selected_truth == fault_id] != 0)
                    )
                    for fault_id in CLASS_IDS[1:]
                },
            }
        )
    binary_head_auc = {
        str(fault_id): float(
            roc_auc_score(
                np.asarray(binary_truth_by_fault[fault_id], dtype=np.int64),
                np.asarray(binary_score_by_fault[fault_id], dtype=np.float64),
            )
        )
        for fault_id in CLASS_IDS[1:]
    }
    return {
        "schema": "mstcn_v3_development_separability_diagnostic",
        "schema_version": 1,
        "claim_scope": "already_consumed_mode1_magnitude50_development_only_not_confirmation",
        "source": str(source.resolve()),
        "development_magnitude_percent": DEVELOPMENT_MAGNITUDE,
        "simulation_runs": [min(DEVELOPMENT_RUNS), max(DEVELOPMENT_RUNS)],
        "fold_count": 5,
        "feature_family": "causal_MSCD_TCN_dynamic_features_with_multiscale_summary",
        "summary_windows": list(SUMMARY_WINDOWS),
        "endpoints": list(ENDPOINTS),
        "classifier": {
            "name": "ExtraTreesClassifier",
            "n_estimators": 384,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        "folds": fold_rows,
        "oof_window_recall_by_class": recalls,
        "oof_early_window_recall_by_class": early_recalls,
        "oof_binary_fault_vs_healthy_auc_by_fault": binary_auc_by_fault,
        "oof_dedicated_binary_head_auc_by_fault": binary_head_auc,
        "oof_anomaly_detection_rate_by_fault": anomaly_detection_rate_by_fault,
        "oof_endpoint_detection": endpoint_rows,
        "oof_confusion_matrix_rows_truth_columns_prediction": cm.tolist(),
        "oof_sample_count": int(truth.size),
        "oof_group_count": int(np.unique(groups_array).size),
        "oof_mean_true_class_probability": float(
            np.mean(
                probability[
                    np.arange(truth.size),
                    np.searchsorted(np.asarray(CLASS_IDS, dtype=np.int64), truth),
                ]
            )
        ),
        "interpretation": "feature_learnability_only; not an online alarm or false-alarm estimate",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=_project_root() / "protocols" / "pclr_cusum_development_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_project_root()
        / "outputs"
        / "mstcn_spc_v3_diagnostics"
        / "separability.json",
    )
    args = parser.parse_args()
    result = run_diagnostic(args.data_dir.resolve(), args.protocol.resolve())
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
