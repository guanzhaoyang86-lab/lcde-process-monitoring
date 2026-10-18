"""Run the frozen weakly supervised directional GLR-CUSUM experiment on CUDA.

The workflow is intentionally split into development and audit phases.  Every
manifest that freezes a phase is written before that phase opens any new Rieth
run values.  New fitted models use numeric NPZ plus JSON metadata; the one
legacy pickle dependency is loaded only after its hash has been verified
against the completed predecessor experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# This Windows environment must initialize CUDA before NumPy/MKL is imported.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc.cva_mewma import CausalCVAMEWMAMonitor
from mate_spc.datasets import (
    RIETH_FEATURE_COLUMNS,
    RiethTEPArchive,
    SensorSequence,
)
from mate_spc_directional.features import (
    CONTROL_LAGS,
    CONTROL_PAIRS,
    CONTROL_WINDOWS,
    EWMA_LAMBDA,
    OUTPUT_CLIP,
    RESIDUAL_DIMENSION,
    RESIDUAL_WINDOWS,
    DirectionalFeatureTransformer,
    load_directional_feature_transformer,
)
from mate_spc_directional.glr import (
    FAULT_IDS,
    NORMAL_ID,
    PERSISTENCE_WINDOW,
    TRAIN_START,
    TRAIN_STOP,
    Z_CLIP,
    DirectionalGLRModel,
    DirectionalGLRPath,
    conformal_run_threshold,
    load_directional_glr_model,
)
from mate_spc_directional.gpu import (
    DirectionalGLRFoldFit,
    DirectionalGLRGPUResult,
    fit_directional_glr_cuda,
    fit_directional_glr_grid_cuda,
    score_directional_glr_cuda,
    validate_directional_glr_cuda,
)


DATASET_DOI = "10.7910/DVN/6C3JR1"
FAULT_START = 160
DESIGN_HEALTH_RUNS = tuple(range(201, 301))
DEVELOPMENT_RUNS = tuple(range(1, 101))
CALIBRATION_RUNS = tuple(range(301, 401))
AUDIT_RUNS = tuple(range(101, 201))
FAULT_RUNS_FORBIDDEN_AFTER_FAILURE = tuple(range(201, 401))
FINAL_RUNS = tuple(range(401, 501))
DEVELOPMENT_STOP = 500
EXPECTED_TRAIN_LENGTH = 500
EXPECTED_TEST_LENGTH = 960
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KAPPAS = (0.1, 0.25, 0.5, 0.75)
FOLDS = tuple(range(5))
ALPHA = 0.05
CALIBRATION_RANK = 96
AUDIT_HEALTH_MAXIMUM = 9
AUDIT_PER_FAULT_MINIMUM = 60
AUDIT_TOTAL_MINIMUM = 210
CPU_GPU_TOLERANCE = 1e-10
LBFGS_MAX_ITER = 1000
LBFGS_MAX_EVAL = 1250
LBFGS_TOLERANCE_GRAD = 1e-5
LBFGS_TOLERANCE_CHANGE = 1e-9
LBFGS_HISTORY_SIZE = 100
OLD_DECISION = "FAIL_BOTH_FROZEN_STAGES_STOP_SEARCH"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            _clean_json(value),
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=path.suffix,
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".tmp",
            prefix=f".{path.name}-",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen directional density-ratio GLR-CUSUM experiment"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("development", "audit", "all"),
        default="all",
    )
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cva-model",
        type=Path,
        help="fitted_stage_a_cva.pkl from the completed predecessor experiment",
    )
    parser.add_argument(
        "--cva-completion",
        type=Path,
        help="completion.json that authenticates --cva-model",
    )
    return parser


def _source_hashes(project_root: Path) -> dict[str, str]:
    new_package = sorted(
        (project_root / "src" / "mate_spc_directional").glob("*.py")
    )
    new_tests = sorted((project_root / "tests").glob("*directional*.py"))
    dependencies = (
        project_root / "scripts" / "run_directional_glr_gpu.py",
        project_root / "src" / "mate_spc" / "cva_mewma.py",
        project_root / "src" / "mate_spc" / "dynamic_common.py",
        project_root / "src" / "mate_spc" / "datasets.py",
        project_root / "pyproject.toml",
    )
    paths = tuple(sorted(set(new_package + new_tests + list(dependencies))))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol source files are missing: {missing}")
    required_new = {"__init__.py", "features.py", "glr.py", "gpu.py"}
    found_new = {path.name for path in new_package}
    if not required_new.issubset(found_new):
        raise FileNotFoundError(
            "directional package is incomplete: "
            f"missing={sorted(required_new - found_new)}"
        )
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in paths
    }


def _read_json_with_fingerprint(
    path: Path,
    *,
    fingerprint_field: str,
) -> tuple[dict[str, object], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    stored = value.get(fingerprint_field)
    body = dict(value)
    body.pop(fingerprint_field, None)
    if not isinstance(stored, str) or stored != _fingerprint(body):
        raise ValueError(f"{path} fingerprint is corrupt")
    return value, stored


def _load_or_create_manifest(
    path: Path,
    protocol: dict[str, object],
    *,
    require_empty_parent: bool,
) -> str:
    fingerprint = _fingerprint(protocol)
    if path.exists():
        current, stored = _read_json_with_fingerprint(
            path,
            fingerprint_field="protocol_fingerprint",
        )
        body = dict(current)
        body.pop("protocol_fingerprint")
        if stored != fingerprint or body != protocol:
            raise ValueError(f"{path} contains a different frozen protocol")
        return fingerprint
    if require_empty_parent and path.parent.exists() and tuple(path.parent.iterdir()):
        raise ValueError(
            "output directory is nonempty but has no development manifest; "
            "use a fresh directory"
        )
    _atomic_json(path, {**protocol, "protocol_fingerprint": fingerprint})
    return fingerprint


def _completion_body(
    root: Path,
    phase_fingerprint: str,
    decision: str,
    artifact_names: Iterable[str],
) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for name in sorted(set(artifact_names)):
        target = root / name
        if not target.is_file():
            raise FileNotFoundError(f"cannot complete: artifact is missing: {name}")
        artifacts[name] = {
            "size_bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
        }
    return {
        "schema_version": 1,
        "phase_fingerprint": phase_fingerprint,
        "decision": decision,
        "artifacts": artifacts,
    }


def _write_completion(
    path: Path,
    root: Path,
    phase_fingerprint: str,
    decision: str,
    artifact_names: Iterable[str],
) -> None:
    body = _completion_body(root, phase_fingerprint, decision, artifact_names)
    _atomic_json(
        path,
        {**body, "completion_fingerprint": _fingerprint(body)},
    )


def _verify_completion(
    path: Path,
    root: Path,
    phase_fingerprint: str | None = None,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    value, _ = _read_json_with_fingerprint(
        path,
        fingerprint_field="completion_fingerprint",
    )
    if phase_fingerprint is not None and value.get("phase_fingerprint") != (
        phase_fingerprint
    ):
        raise ValueError(f"{path} belongs to another phase protocol")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{path} has no completed artifacts")
    for relative, expected in artifacts.items():
        target = root / relative
        if not target.is_file() or not isinstance(expected, dict):
            raise ValueError(f"completed artifact is missing: {relative}")
        if target.stat().st_size != int(expected.get("size_bytes", -1)):
            raise ValueError(f"completed artifact size changed: {relative}")
        if _sha256_file(target) != expected.get("sha256"):
            raise ValueError(f"completed artifact hash changed: {relative}")
    return value


def _artifact_hash(completion: Mapping[str, object], name: str) -> str:
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("completion has no artifact map")
    record = artifacts.get(name)
    if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"completion does not authenticate artifact {name!r}")
    return str(record["sha256"])


def _old_cva_reference(
    model_path: Path,
    completion_path: Path,
    *,
    load_model: bool,
) -> tuple[dict[str, object], CausalCVAMEWMAMonitor | None]:
    completion, _ = _read_json_with_fingerprint(
        completion_path,
        fingerprint_field="completion_fingerprint",
    )
    if completion.get("decision") != OLD_DECISION:
        raise ValueError("legacy CVA must come from the recorded failed sequence")
    if model_path.parent.resolve() != completion_path.parent.resolve():
        raise ValueError("legacy CVA model and completion must share one directory")
    expected_sha = _artifact_hash(completion, model_path.name)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    observed_sha = _sha256_file(model_path)
    artifacts = completion["artifacts"]
    expected_size = int(artifacts[model_path.name]["size_bytes"])
    if model_path.stat().st_size != expected_size or observed_sha != expected_sha:
        raise ValueError("legacy CVA pickle does not match its completion manifest")
    reference = {
        "model_path": model_path.name,
        "model_sha256": observed_sha,
        "model_size_bytes": model_path.stat().st_size,
        "completion_path": completion_path.name,
        "completion_sha256": _sha256_file(completion_path),
        "completion_fingerprint": completion["completion_fingerprint"],
        "predecessor_protocol_fingerprint": completion["protocol_fingerprint"],
        "required_predecessor_decision": OLD_DECISION,
        "pickle_policy": "hash_verified_before_loading; no new pickle is written",
    }
    if not load_model:
        return reference, None
    with model_path.open("rb") as stream:
        monitor = pickle.load(stream)
    if not isinstance(monitor, CausalCVAMEWMAMonitor):
        raise TypeError("authenticated legacy pickle is not a CVA monitor")
    required = ("raw_center_", "raw_scale_", "past_mean_", "projection_", "rank_")
    if not bool(getattr(monitor, "design_fitted_", False)) or any(
        not hasattr(monitor, name) for name in required
    ):
        raise ValueError("authenticated CVA monitor lacks frozen design parameters")
    if int(monitor.rank_) != 56:
        raise ValueError(f"directional protocol requires CVA rank 56, got {monitor.rank_}")
    expected_shapes = {
        "raw_center_": (52,),
        "raw_scale_": (52,),
        "past_mean_": (260,),
        "projection_": (260, 260),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(getattr(monitor, name), dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"legacy CVA field {name} is invalid")
    return reference, monitor


def _development_protocol(
    *,
    checksums: dict[str, str],
    source_hashes: dict[str, str],
    cva_reference: dict[str, object],
    batch_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "weakly_supervised_method_development",
        "dataset": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": DATASET_DOI,
        "dataset_md5": checksums,
        "source_sha256": source_hashes,
        "legacy_cva": cva_reference,
        "execution": {
            "required_device": "cuda",
            "fit_dtype": "float64",
            "score_dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "cpu_gpu_tolerance": CPU_GPU_TOLERANCE,
        },
        "method": {
            "name": "control_loop_directional_density_ratio_page_cusum_3of3",
            "feature_transformer": {
                "source_cva_rank": 56,
                "signed_residual_dimension": RESIDUAL_DIMENSION,
                "signed_residual_ewma_lambda": EWMA_LAMBDA,
                "ewma_initial_state": "zero independently for each run",
                "residual_windows_samples": list(RESIDUAL_WINDOWS),
                "residual_window_statistic": (
                    "signed standardized EWMA residual rolling sum divided by "
                    "square root of window"
                ),
                "control_pairs_zero_based_xmv_xmeas": [
                    list(pair) for pair in CONTROL_PAIRS
                ],
                "control_pair_assumption": (
                    "externally frozen engineering assumption in the fixed 52-column "
                    "Rieth layout; pairs are not inferred from audit data"
                ),
                "pair_51_21_note": (
                    "frozen auxiliary physical response for XMV11, not a controller "
                    "pair inferred from Rieth labels"
                ),
                "control_windows_samples": list(CONTROL_WINDOWS),
                "control_lags_samples": list(CONTROL_LAGS),
                "control_statistics_per_pair_window": [
                    "difference_correlation_u_leads_y_lag_0",
                    "difference_correlation_u_leads_y_lag_1",
                    "difference_correlation_u_leads_y_lag_2",
                    "difference_correlation_u_leads_y_lag_4",
                    "difference_correlation_u_leads_y_lag_8",
                    "normalized_directed_hysteresis_area",
                    "absolute_normalized_hysteresis_area",
                    "log_difference_energy_controller",
                    "log_difference_energy_response",
                    "maximum_absolute_difference_autocorrelation_controller",
                    "maximum_absolute_difference_autocorrelation_response",
                ],
                "control_statistic_coordinate_count": 11,
                "raw_feature_count": 1080,
                "valid_start_zero_based": 63,
                "design_scaling": (
                    "median and 1.4826*MAD; sample-SD fallback; drop combined raw "
                    "coordinates still degenerate"
                ),
                "output_clip": [-OUTPUT_CLIP, OUTPUT_CLIP],
                "fit_data": "healthy transformer-design runs only",
                "run_boundary_rule": "all states and windows reset per run",
            },
            "classifier": {
                "heads": "three one-vs-normal balanced-prior logistic log-odds",
                "fault_head_order": list(FAULT_IDS),
                "training_time_slice_zero_based_half_open": [
                    TRAIN_START,
                    TRAIN_STOP,
                ],
                "ridge_lambda_grid": list(RIDGE_LAMBDAS),
                "loss_weighting": "equal heads, classes, runs, and time samples",
                "bias_penalized": False,
                "optimizer": {
                    "backend": "pytorch_cuda_float64",
                    "algorithm": "joint_three_head_LBFGS_strong_wolfe",
                    "initial_weights_and_bias": "zero",
                    "max_iter": LBFGS_MAX_ITER,
                    "max_eval": LBFGS_MAX_EVAL,
                    "tolerance_grad": LBFGS_TOLERANCE_GRAD,
                    "tolerance_change": LBFGS_TOLERANCE_CHANGE,
                    "history_size": LBFGS_HISTORY_SIZE,
                    "require_convergence": True,
                    "tf32": False,
                },
                "normal_logit_location": "training-normal pooled median per head",
                "normal_logit_scale": (
                    "1.4826*MAD with sample-SD fallback per head"
                ),
                "standardized_logit_clip": [-Z_CLIP, Z_CLIP],
            },
            "sequential_statistic": {
                "page_reference_allowance_grid": list(KAPPAS),
                "page_initial_state": "zero independently per run and head",
                "persistence": "3_of_3 separately per head then family maximum",
                "alarm_comparison": "strictly_greater_than",
                "fault_onset_zero_based": FAULT_START,
            },
        },
        "development_splits": {
            "transformer_design": ["fault_free_training", list(DESIGN_HEALTH_RUNS)],
            "classifier_normal": ["fault_free_testing", list(DEVELOPMENT_RUNS)],
            "classifier_faults": [
                "faulty_testing",
                list(FAULT_IDS),
                list(DEVELOPMENT_RUNS),
            ],
            "development_truncation_half_open": [0, DEVELOPMENT_STOP],
            "fold": "(simulation_run - 1) % 5",
            "fold_ids": list(FOLDS),
            "per_fold_per_class_runs": 20,
        },
        "development_selection": {
            "oof_healthy_threshold": (
                "pool exactly 100 out-of-fold healthy run envelope maxima and "
                "take order statistic 96 at alpha 0.05"
            ),
            "selection_key_maximized_lexicographically": [
                "minimum_detected_runs_across_fault_classes",
                "total_detected_fault_runs",
                "negative_restricted_mean_delay_miss_as_340",
                "larger_ridge_lambda",
                "larger_kappa",
            ],
            "final_fit": "refit selected ridge on all 100 runs per class",
            "development_threshold_reused_in_audit": False,
        },
        "predeclared_audit": {
            "calibration": ["fault_free_testing", list(CALIBRATION_RUNS)],
            "method_specific_healthy": ["fault_free_testing", list(AUDIT_RUNS)],
            "method_specific_faults": [
                "faulty_testing",
                list(FAULT_IDS),
                list(AUDIT_RUNS),
            ],
            "calibration_rank_one_based": CALIBRATION_RANK,
            "all_sequence_truncation_half_open": [0, DEVELOPMENT_STOP],
            "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
            "detected_runs_per_fault_minimum": AUDIT_PER_FAULT_MINIMUM,
            "detected_fault_runs_total_minimum": AUDIT_TOTAL_MINIMUM,
            "all_gates_required": True,
        },
        "reserved": {
            "fault_runs_201_400_if_audit_fails": "do_not_open_for_more_tuning",
            "final_healthy_401_500": "do_not_select_score_or_inspect",
            "final_all_faults_401_500": "do_not_select_score_or_inspect",
        },
        "data_access_semantics": {
            "archive_format": "partition-level RData decoded before row selection",
            "model_visible_rows": "only explicit load_runs selections",
            "all_test_rows_500_959": "discarded before feature transformation",
            "reserved_claim": "runs 401-500 are not selected, scored, or inspected",
        },
        "forbidden_after_manifest": [
            "feature family or control pairs",
            "training time interval",
            "ridge or kappa grids",
            "fold assignment or selection objective",
            "development and audit splits",
            "calibration rule or audit gates",
        ],
    }


def _audit_base_protocol(
    *,
    development_fingerprint: str,
    development_completion_sha256: str,
    transformer_sha256: str,
    model_sha256: str,
    selected: Mapping[str, object],
    checksums: dict[str, str],
    source_hashes: dict[str, str],
    batch_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "frozen_method_specific_audit_base",
        "development_protocol_fingerprint": development_fingerprint,
        "development_completion_sha256": development_completion_sha256,
        "transformer_sha256": transformer_sha256,
        "uncalibrated_model_sha256": model_sha256,
        "selected_hyperparameters": {
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
        },
        "dataset_doi": DATASET_DOI,
        "dataset_md5": checksums,
        "source_sha256": source_hashes,
        "execution": {
            "required_device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "cpu_gpu_tolerance": CPU_GPU_TOLERANCE,
        },
        "ordered_access": [
            "write this base manifest before calibration values are loaded",
            "load and score only healthy calibration runs 301-400",
            "freeze q96 threshold and pass CPU/CUDA validation",
            "write audit_open_manifest.json",
            "only then load method-specific audit runs 101-200",
        ],
        "calibration": {
            "split": ["fault_free_testing", list(CALIBRATION_RUNS)],
            "sequence_truncation_half_open": [0, DEVELOPMENT_STOP],
            "unit": "independent healthy run family-envelope maximum",
            "alpha": ALPHA,
            "rank_one_based": CALIBRATION_RANK,
            "comparison": "strictly_greater_than",
        },
        "audit": {
            "healthy": ["fault_free_testing", list(AUDIT_RUNS)],
            "faults": ["faulty_testing", list(FAULT_IDS), list(AUDIT_RUNS)],
            "sequence_truncation_half_open": [0, DEVELOPMENT_STOP],
            "fault_onset_zero_based": FAULT_START,
            "pre_fault_episode_continuing_into_fault": "not_credited",
            "miss_delay_samples": DEVELOPMENT_STOP - FAULT_START,
            "gates": {
                "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
                "per_fault_detected_runs_minimum": AUDIT_PER_FAULT_MINIMUM,
                "total_fault_detected_runs_minimum": AUDIT_TOTAL_MINIMUM,
                "all_required": True,
            },
        },
        "stop_rule": {
            "on_failure": "do not open faulty runs 201-400 for model search",
            "final_runs": "do not select, score, or inspect runs 401-500",
        },
        "rdata_decode_caveat": (
            "pyreadr decodes a complete RData partition before load_runs returns "
            "only selected run rows"
        ),
    }


def _audit_open_protocol(
    *,
    base_fingerprint: str,
    calibration: Mapping[str, object],
    calibration_sha256: str,
    calibration_scores_sha256: str,
    validation_sha256: str,
    calibrated_model_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "audit_open_authorization",
        "audit_base_fingerprint": base_fingerprint,
        "threshold": calibration["threshold"],
        "alpha": ALPHA,
        "rank_one_based": CALIBRATION_RANK,
        "comparison": "strictly_greater_than",
        "calibration_sha256": calibration_sha256,
        "calibration_scores_sha256": calibration_scores_sha256,
        "cuda_validation_sha256": validation_sha256,
        "calibrated_model_sha256": calibrated_model_sha256,
        "validation_required_before_audit": True,
        "audit_splits_now_authorized": {
            "healthy": ["fault_free_testing", list(AUDIT_RUNS)],
            "faults": ["faulty_testing", list(FAULT_IDS), list(AUDIT_RUNS)],
        },
        "gates": {
            "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
            "per_fault_detected_runs_minimum": AUDIT_PER_FAULT_MINIMUM,
            "total_fault_detected_runs_minimum": AUDIT_TOTAL_MINIMUM,
            "all_required": True,
        },
        "created_before_method_specific_audit_load": True,
    }


def _run_ids(
    runs: Iterable[SensorSequence],
    expected: tuple[int, ...],
    *,
    partition: str,
    expected_length: int,
    label: str,
) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(int(run.metadata["simulation_run"]) for run in normalized)
    if actual != expected:
        raise RuntimeError(f"{label} run IDs differ from the frozen split")
    for run in normalized:
        if run.metadata.get("partition") != partition:
            raise RuntimeError(f"{label} partition metadata is invalid")
        if int(run.metadata.get("fault_id", -1)) != 0:
            raise RuntimeError(f"{label} contains nonzero fault metadata")
        if np.asarray(run.x).shape != (expected_length, 52):
            raise RuntimeError(f"{label} run {run.name} has an unexpected shape")
        if tuple(run.feature_names) != tuple(RIETH_FEATURE_COLUMNS):
            raise RuntimeError(f"{label} run {run.name} has unexpected feature names")
        if np.asarray(run.fault, dtype=bool).any():
            raise RuntimeError(f"{label} run {run.name} is not healthy")
    return normalized


def _fault_run_ids(
    runs: Iterable[SensorSequence],
    expected_runs: tuple[int, ...],
    *,
    expected_length: int,
    label: str,
) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(
        (int(run.metadata["fault_id"]), int(run.metadata["simulation_run"]))
        for run in normalized
    )
    expected = tuple(
        (fault_id, run_id)
        for fault_id in FAULT_IDS
        for run_id in expected_runs
    )
    if actual != expected:
        raise RuntimeError(f"{label} fault IDs/runs differ from the frozen split")
    for run in normalized:
        if run.metadata.get("partition") != "faulty_testing":
            raise RuntimeError(f"{label} partition metadata is invalid")
        if np.asarray(run.x).shape != (expected_length, 52):
            raise RuntimeError(f"{label} run {run.name} has an unexpected shape")
        if tuple(run.feature_names) != tuple(RIETH_FEATURE_COLUMNS):
            raise RuntimeError(f"{label} run {run.name} has unexpected feature names")
        fault = np.asarray(run.fault, dtype=bool)
        if fault.shape != (expected_length,) or fault[:FAULT_START].any() or not (
            fault[FAULT_START:].all()
        ):
            raise RuntimeError(f"{label} run {run.name} has an unexpected fault mask")
    return normalized


def _truncate_run(run: SensorSequence, stop: int) -> SensorSequence:
    values = np.asarray(run.x[:stop], dtype=np.float64)
    if values.shape != (stop, 52) or not np.isfinite(values).all():
        raise RuntimeError(f"{run.name} is invalid inside the frozen truncation")
    observed = None
    if run.observed_mask is not None:
        observed = np.asarray(run.observed_mask[:stop], dtype=bool)
    return replace(
        run,
        x=np.array(values, copy=True),
        fault=np.array(run.fault[:stop], dtype=bool, copy=True),
        observed_mask=None if observed is None else np.array(observed, copy=True),
    )


def _validate_full_values(runs: Iterable[SensorSequence]) -> None:
    for run in runs:
        values = np.asarray(run.x, dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{run.name} contains non-finite values")


def _named_values(runs: Iterable[SensorSequence]) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple((run.name, np.asarray(run.x, dtype=np.float64)) for run in runs)


def _feature_mapping(
    transformer: DirectionalFeatureTransformer,
    runs: Iterable[SensorSequence],
) -> dict[str, np.ndarray]:
    return {
        name: features
        for name, features in transformer.transform_named_runs(_named_values(runs))
    }


def _unicode_array(values: Sequence[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(tuple(values), dtype=f"<U{max(width, 1)}")


def _fold_for_run(run: SensorSequence) -> int:
    return (int(run.metadata["simulation_run"]) - 1) % len(FOLDS)


def _episode_starts(alarm: np.ndarray) -> list[int]:
    values = np.asarray(alarm, dtype=bool)
    if values.ndim != 1:
        raise ValueError("alarm path must be one-dimensional")
    padded = np.concatenate(([False], values))
    return np.flatnonzero(padded[1:] & ~padded[:-1]).astype(int).tolist()


def _health_records(
    runs: Iterable[SensorSequence],
    paths: Mapping[str, DirectionalGLRPath],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        path = paths[run.name]
        alarm = np.asarray(path.eligible, dtype=bool) & (
            np.asarray(path.envelope, dtype=np.float64) > threshold
        )
        starts = _episode_starts(alarm)
        eligible_values = path.envelope[path.eligible]
        records.append(
            {
                "run": run.name,
                "simulation_run": int(run.metadata["simulation_run"]),
                "any_alarm": bool(starts),
                "alarm_episode_count": len(starts),
                "first_alarm_index": starts[0] if starts else None,
                "maximum_envelope": float(np.max(eligible_values)),
            }
        )
    return records


def _fault_records(
    runs: Iterable[SensorSequence],
    paths: Mapping[str, DirectionalGLRPath],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        path = paths[run.name]
        alarm = np.asarray(path.eligible, dtype=bool) & (
            np.asarray(path.envelope, dtype=np.float64) > threshold
        )
        starts = _episode_starts(alarm)
        post_starts = [index for index in starts if index >= FAULT_START]
        first = post_starts[0] if post_starts else None
        fault_id = int(run.metadata["fault_id"])
        head = None
        if first is not None:
            head_index = int(np.argmax(path.component_persistence[first]))
            head = int(path.fault_ids[head_index])
        miss_delay = int(run.x.shape[0] - FAULT_START)
        records.append(
            {
                "run": run.name,
                "fault_id": fault_id,
                "simulation_run": int(run.metadata["simulation_run"]),
                "detected": first is not None,
                "first_fault_alarm_episode_index": first,
                "delay_samples": None if first is None else first - FAULT_START,
                "restricted_delay_samples_miss_at_run_end": (
                    miss_delay if first is None else first - FAULT_START
                ),
                "pre_fault_alarm_episode_count": sum(
                    index < FAULT_START for index in starts
                ),
                "attributed_fault_head_at_first_alarm": head,
                "attribution_matches_fault": head == fault_id,
                "maximum_post_fault_envelope": float(
                    np.max(path.envelope[FAULT_START:])
                ),
            }
        )
    return records


def _fault_aggregate(
    records: Sequence[Mapping[str, object]],
    fault_id: int | None = None,
) -> dict[str, object]:
    selected = [
        row
        for row in records
        if fault_id is None or int(row["fault_id"]) == fault_id
    ]
    if not selected:
        raise ValueError("fault aggregate is empty")
    detected = sum(bool(row["detected"]) for row in selected)
    delays = [
        int(row["delay_samples"])
        for row in selected
        if row["delay_samples"] is not None
    ]
    restricted = [
        int(row["restricted_delay_samples_miss_at_run_end"])
        for row in selected
    ]
    return {
        "run_count": len(selected),
        "detected_runs": detected,
        "detection_rate": detected / len(selected),
        "conditional_median_delay_samples": (
            float(np.median(delays)) if delays else None
        ),
        "restricted_mean_delay_samples": float(np.mean(restricted)),
        "runs_with_pre_fault_alarm": sum(
            int(row["pre_fault_alarm_episode_count"]) > 0 for row in selected
        ),
        "attribution_matches_fault_runs": sum(
            bool(row["attribution_matches_fault"]) for row in selected
        ),
    }


def _candidate_summary(
    *,
    ridge_lambda: float,
    kappa: float,
    threshold: float,
    calibration: Mapping[str, object],
    health_records: Sequence[Mapping[str, object]],
    fault_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_fault = {
        f"{fault_id:02d}": _fault_aggregate(fault_records, fault_id)
        for fault_id in FAULT_IDS
    }
    overall = _fault_aggregate(fault_records)
    class_hits = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"])
        for fault_id in FAULT_IDS
    ]
    selection_key = (
        min(class_hits),
        int(overall["detected_runs"]),
        -float(overall["restricted_mean_delay_samples"]),
        float(ridge_lambda),
        float(kappa),
    )
    return {
        "ridge_lambda": float(ridge_lambda),
        "kappa": float(kappa),
        "threshold": float(threshold),
        "healthy_alarm_runs": sum(bool(row["any_alarm"]) for row in health_records),
        "healthy_run_count": len(health_records),
        "calibration_rank_one_based": calibration["rank_one_based"],
        "calibration_rank_driver_run": calibration["rank_driver_run"],
        "by_fault": by_fault,
        "overall": overall,
        "minimum_detected_runs_across_faults": min(class_hits),
        "selection_key": list(selection_key),
    }


def _selection_key(candidate: Mapping[str, object]) -> tuple[float, ...]:
    value = candidate.get("selection_key")
    if not isinstance(value, Sequence) or len(value) != 5:
        raise ValueError("candidate has an invalid selection key")
    return tuple(float(item) for item in value)


def _evaluate_oof_grid(
    fits: Sequence[DirectionalGLRFoldFit],
    features: Mapping[str, np.ndarray],
    labels: Mapping[str, int],
    runs_by_name: Mapping[str, SensorSequence],
    *,
    valid_start: int,
    device: str,
    batch_size: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, DirectionalGLRPath],
    dict[str, object],
    list[dict[str, object]],
]:
    expected_names = set(features)
    candidates: list[dict[str, object]] = []
    best_paths: dict[str, DirectionalGLRPath] | None = None
    best_candidate: dict[str, object] | None = None
    executions: list[dict[str, object]] = []
    for ridge_lambda in RIDGE_LAMBDAS:
        ridge_fits = [
            fit for fit in fits if float(fit.ridge_lambda) == ridge_lambda
        ]
        if tuple(sorted(int(fit.held_out_fold) for fit in ridge_fits)) != FOLDS:
            raise RuntimeError(f"ridge {ridge_lambda} lacks exactly five OOF fits")
        held_out_union = {
            name for fit in ridge_fits for name in fit.held_out_run_names
        }
        if held_out_union != expected_names:
            raise RuntimeError("OOF fits do not partition every development run")
        for kappa in KAPPAS:
            paths: dict[str, DirectionalGLRPath] = {}
            for fit in ridge_fits:
                held_out = {
                    name: features[name] for name in fit.held_out_run_names
                }
                started = time.perf_counter()
                scored = score_directional_glr_cuda(
                    fit.model.with_kappa(kappa),
                    held_out,
                    valid_start=valid_start,
                    device=device,
                    batch_size=batch_size,
                )
                if set(paths).intersection(scored.by_run):
                    raise RuntimeError("a development run received two OOF scores")
                paths.update(scored.by_run)
                executions.append(
                    {
                        "purpose": "oof_grid_scoring",
                        "ridge_lambda": ridge_lambda,
                        "kappa": kappa,
                        "held_out_fold": fit.held_out_fold,
                        "wall_seconds": time.perf_counter() - started,
                        **scored.execution,
                    }
                )
            if set(paths) != expected_names:
                raise RuntimeError("OOF scoring did not return every development run")
            healthy_envelopes = {
                name: path.envelope
                for name, path in paths.items()
                if int(labels[name]) == NORMAL_ID
            }
            threshold, calibration = conformal_run_threshold(
                healthy_envelopes,
                alpha=ALPHA,
                required_count=len(DEVELOPMENT_RUNS),
            )
            healthy_runs = [
                runs_by_name[name]
                for name in sorted(paths)
                if int(labels[name]) == NORMAL_ID
            ]
            fault_runs = [
                runs_by_name[name]
                for name in sorted(paths)
                if int(labels[name]) != NORMAL_ID
            ]
            candidate = _candidate_summary(
                ridge_lambda=ridge_lambda,
                kappa=kappa,
                threshold=threshold,
                calibration=calibration,
                health_records=_health_records(
                    healthy_runs,
                    paths,
                    threshold=threshold,
                ),
                fault_records=_fault_records(
                    fault_runs,
                    paths,
                    threshold=threshold,
                ),
            )
            candidates.append(candidate)
            if best_candidate is None or _selection_key(candidate) > _selection_key(
                best_candidate
            ):
                best_candidate = candidate
                best_paths = paths
    if best_candidate is None or best_paths is None:
        raise RuntimeError("development grid produced no candidate")
    return candidates, best_paths, best_candidate, executions


def _fold_fit_records(
    fits: Sequence[DirectionalGLRFoldFit],
) -> list[dict[str, object]]:
    return [
        {
            "ridge_lambda": fit.ridge_lambda,
            "held_out_fold": fit.held_out_fold,
            "training_run_names": list(fit.training_run_names),
            "held_out_run_names": list(fit.held_out_run_names),
            "convergence": fit.convergence,
        }
        for fit in fits
    ]


def _pack_paths(
    runs: Sequence[SensorSequence],
    paths: Mapping[str, DirectionalGLRPath],
    *,
    threshold: float,
) -> dict[str, np.ndarray]:
    names = [run.name for run in runs]
    ordered = [paths[name] for name in names]
    return {
        "run_names": _unicode_array(names),
        "fault_ids": np.asarray(
            [int(run.metadata["fault_id"]) for run in runs],
            dtype=np.int64,
        ),
        "simulation_runs": np.asarray(
            [int(run.metadata["simulation_run"]) for run in runs],
            dtype=np.int64,
        ),
        "logits": np.stack([path.logits for path in ordered]),
        "standardized_logits": np.stack(
            [path.standardized_logits for path in ordered]
        ),
        "cusum": np.stack([path.cusum for path in ordered]),
        "component_persistence": np.stack(
            [path.component_persistence for path in ordered]
        ),
        "envelope": np.stack([path.envelope for path in ordered]),
        "eligible": np.stack([path.eligible for path in ordered]),
        "alarm": np.stack(
            [path.eligible & (path.envelope > threshold) for path in ordered]
        ),
        "threshold": np.asarray([threshold], dtype=np.float64),
    }


def _development_report(aggregate: Mapping[str, object]) -> str:
    selected = aggregate["selected_candidate"]
    lines = [
        "# Directional GLR-CUSUM development",
        "",
        f"Protocol fingerprint: {aggregate['protocol_fingerprint']}.",
        "",
        "The feature transformer used only healthy FF-training runs 201-300. ",
        "The classifier and every hyperparameter comparison used grouped five-fold ",
        "out-of-fold predictions from FF-testing/fault runs 1-100, truncated to ",
        "samples 0:500.",
        "",
        "## Frozen selection",
        "",
        f"- Ridge lambda: {selected['ridge_lambda']}.",
        f"- Page kappa: {selected['kappa']}.",
        f"- OOF healthy q96 threshold: {selected['threshold']:.6g}.",
        f"- Minimum class hits: {selected['minimum_detected_runs_across_faults']}.",
        f"- Total fault hits: {selected['overall']['detected_runs']}/300.",
        f"- Restricted mean delay: "
        f"{selected['overall']['restricted_mean_delay_samples']:.3f} samples.",
        "",
        "The OOF development threshold is not reused for audit. Audit requires a ",
        "fresh q96 threshold from healthy FF-testing runs 301-400.",
        "",
        "Runs 401-500 were not selected, scored, or inspected. Because Rieth data ",
        "are stored by RData partition, pyreadr still decodes a complete partition ",
        "before the loader returns only selected rows.",
        "",
    ]
    return "\n".join(lines)


def _audit_summary(
    health_records: Sequence[Mapping[str, object]],
    fault_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    health_alarms = sum(bool(row["any_alarm"]) for row in health_records)
    by_fault = {
        f"{fault_id:02d}": _fault_aggregate(fault_records, fault_id)
        for fault_id in FAULT_IDS
    }
    overall = _fault_aggregate(fault_records)
    per_fault = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"])
        for fault_id in FAULT_IDS
    ]
    health_pass = health_alarms <= AUDIT_HEALTH_MAXIMUM
    per_fault_pass = min(per_fault) >= AUDIT_PER_FAULT_MINIMUM
    total_pass = int(overall["detected_runs"]) >= AUDIT_TOTAL_MINIMUM
    passed = health_pass and per_fault_pass and total_pass
    return {
        "passed": passed,
        "decision": (
            "PASS_ALL_FROZEN_AUDIT_GATES"
            if passed
            else "FAIL_FROZEN_AUDIT_STOP_METHOD_SEARCH"
        ),
        "healthy": {
            "run_count": len(health_records),
            "runs_with_any_alarm": health_alarms,
            "alarm_episodes": sum(
                int(row["alarm_episode_count"]) for row in health_records
            ),
            "gate_maximum": AUDIT_HEALTH_MAXIMUM,
            "passed": health_pass,
            "records": list(health_records),
        },
        "faults": {
            "run_count": len(fault_records),
            "overall": overall,
            "by_fault": by_fault,
            "minimum_detected_runs_across_faults": min(per_fault),
            "per_fault_gate_minimum": AUDIT_PER_FAULT_MINIMUM,
            "total_gate_minimum": AUDIT_TOTAL_MINIMUM,
            "per_fault_passed": per_fault_pass,
            "total_passed": total_pass,
            "records": list(fault_records),
        },
        "all_gates_required": True,
    }


def _audit_report(aggregate: Mapping[str, object]) -> str:
    result = aggregate["audit"]
    health = result["healthy"]
    faults = result["faults"]
    lines = [
        "# Frozen directional GLR-CUSUM audit",
        "",
        f"Audit-open fingerprint: {aggregate['audit_open_fingerprint']}.",
        "",
        "Calibration on healthy FF-testing runs 301-400 and CPU/CUDA validation ",
        "were completed before the method-specific audit splits were loaded.",
        "",
        "Every calibration and audit run was truncated to samples 0:500 before ",
        "feature transformation; rows 500:960 were neither transformed nor scored.",
        "",
        "| Gate | Result | Required | Pass |",
        "|---|---:|---:|---:|",
        f"| Healthy runs with alarm | {health['runs_with_any_alarm']}/100 | <= 9 | "
        f"{str(health['passed']).lower()} |",
        f"| Total fault runs detected | {faults['overall']['detected_runs']}/300 | "
        f">= 210 | {str(faults['total_passed']).lower()} |",
        f"| Minimum detected in one fault | "
        f"{faults['minimum_detected_runs_across_faults']}/100 | >= 60 | "
        f"{str(faults['per_fault_passed']).lower()} |",
        "",
        f"Decision: {result['decision']}.",
        "",
        "| Fault | Detected | Median delay | Restricted mean delay |",
        "|---:|---:|---:|---:|",
    ]
    for fault_id in FAULT_IDS:
        item = faults["by_fault"][f"{fault_id:02d}"]
        median = item["conditional_median_delay_samples"]
        median_text = "-" if median is None else f"{median:.1f}"
        lines.append(
            f"| {fault_id} | {item['detected_runs']}/100 | {median_text} | "
            f"{item['restricted_mean_delay_samples']:.1f} |"
        )
    execution = aggregate["gpu_execution"]
    lines.extend(
        [
            "",
            "## Execution and boundary",
            "",
            f"- Device: {execution['device_name']}; float64; TF32 false.",
            f"- Audit scoring CUDA-event time: "
            f"{execution['audit_cuda_event_seconds']:.3f} seconds.",
            "- Runs 401-500 were not selected, scored, or inspected.",
            "- If this audit failed, faulty runs 201-400 remain unopened and must ",
            "not be used to continue tuning this method.",
            "- RData caveat: pyreadr decodes a complete partition before selected ",
            "rows are returned.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_cva_paths(
    project_root: Path,
    model: Path | None,
    completion: Path | None,
) -> tuple[Path, Path]:
    default_root = project_root / "outputs" / "dynamic_sequence_gpu_v1"
    model_path = model or default_root / "fitted_stage_a_cva.pkl"
    completion_path = completion or model_path.parent / "completion.json"
    return model_path.resolve(), completion_path.resolve()


def _development_artifacts() -> tuple[str, ...]:
    return (
        "development_manifest.json",
        "directional_transformer.npz",
        "directional_glr_uncalibrated.npz",
        "development_candidates.json",
        "development_fit_metadata.json",
        "development_scores.npz",
        "development_aggregate.json",
        "development_gpu_execution.json",
        "DEVELOPMENT_REPORT.md",
    )


def _run_development(
    args: argparse.Namespace,
    *,
    project_root: Path,
    archive: RiethTEPArchive,
    checksums: dict[str, str],
    cva_model_path: Path,
    cva_completion_path: Path,
) -> dict[str, object]:
    cva_reference, _ = _old_cva_reference(
        cva_model_path,
        cva_completion_path,
        load_model=False,
    )
    protocol = _development_protocol(
        checksums=checksums,
        source_hashes=_source_hashes(project_root),
        cva_reference=cva_reference,
        batch_size=args.batch_size,
    )
    manifest_path = args.output_dir / "development_manifest.json"
    fingerprint = _load_or_create_manifest(
        manifest_path,
        protocol,
        require_empty_parent=True,
    )
    completed = _verify_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
    )
    if completed is not None:
        print("directional development is already complete", flush=True)
        return completed

    _, cva = _old_cva_reference(
        cva_model_path,
        cva_completion_path,
        load_model=True,
    )
    assert cva is not None
    print("loading healthy feature-design runs FF-training 201..300", flush=True)
    design = _run_ids(
        archive.load_runs(
            "fault_free_training",
            {0: DESIGN_HEALTH_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DESIGN_HEALTH_RUNS,
        partition="fault_free_training",
        expected_length=EXPECTED_TRAIN_LENGTH,
        label="transformer design",
    )
    _validate_full_values(design)
    fit_started = time.perf_counter()
    transformer = DirectionalFeatureTransformer.from_cva_monitor(cva).fit(
        _named_values(design)
    )
    transformer_fit_seconds = time.perf_counter() - fit_started

    print("loading development healthy/fault runs 1..100", flush=True)
    healthy_full = _run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: DEVELOPMENT_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DEVELOPMENT_RUNS,
        partition="fault_free_testing",
        expected_length=EXPECTED_TEST_LENGTH,
        label="development healthy",
    )
    fault_full = _fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: DEVELOPMENT_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DEVELOPMENT_RUNS,
        expected_length=EXPECTED_TEST_LENGTH,
        label="development faults",
    )
    healthy = tuple(_truncate_run(run, DEVELOPMENT_STOP) for run in healthy_full)
    faults = tuple(_truncate_run(run, DEVELOPMENT_STOP) for run in fault_full)
    all_runs = healthy + faults
    transform_started = time.perf_counter()
    features = _feature_mapping(transformer, all_runs)
    transform_seconds = time.perf_counter() - transform_started
    labels = {
        run.name: int(run.metadata["fault_id"])
        for run in all_runs
    }
    folds = {run.name: _fold_for_run(run) for run in all_runs}
    runs_by_name = {run.name: run for run in all_runs}
    counts = {
        label: sum(value == label for value in labels.values())
        for label in (NORMAL_ID, *FAULT_IDS)
    }
    if counts != {0: 100, 3: 100, 9: 100, 15: 100}:
        raise RuntimeError(f"development classes are not balanced: {counts}")

    print("fitting 5 ridge values x 5 grouped CUDA folds", flush=True)
    grid_started = time.perf_counter()
    fits = fit_directional_glr_grid_cuda(
        features,
        labels,
        ridge_lambdas=RIDGE_LAMBDAS,
        fold_by_run=folds,
        kappa=0.0,
        device=args.device,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        require_convergence=True,
    )
    grid_fit_seconds = time.perf_counter() - grid_started
    candidates, selected_paths, selected, score_executions = _evaluate_oof_grid(
        fits,
        features,
        labels,
        runs_by_name,
        valid_start=transformer.valid_start_,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        "selected ridge="
        f"{selected['ridge_lambda']} kappa={selected['kappa']} "
        f"min-class={selected['minimum_detected_runs_across_faults']} "
        f"total={selected['overall']['detected_runs']}",
        flush=True,
    )
    final_fit_started = time.perf_counter()
    final_model, final_fit_metadata = fit_directional_glr_cuda(
        features,
        labels,
        ridge_lambda=float(selected["ridge_lambda"]),
        kappa=float(selected["kappa"]),
        device=args.device,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        require_convergence=True,
    )
    final_fit_seconds = time.perf_counter() - final_fit_started
    transformer_path = args.output_dir / "directional_transformer.npz"
    transformer.save_npz(
        transformer_path,
        metadata={
            "protocol_fingerprint": fingerprint,
            "legacy_cva_sha256": cva_reference["model_sha256"],
            "design_partition": "fault_free_training",
            "design_runs": list(DESIGN_HEALTH_RUNS),
            "valid_start_zero_based": transformer.valid_start_,
            "raw_feature_count": transformer.RAW_FEATURE_COUNT,
            "retained_feature_count": transformer.output_feature_count_,
        },
    )
    model_path = args.output_dir / "directional_glr_uncalibrated.npz"
    final_model.save_npz(
        model_path,
        metadata={
            "protocol_fingerprint": fingerprint,
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
            "audit_threshold": None,
            "development_oof_threshold_not_for_audit": selected["threshold"],
            "feature_transformer_sha256": _sha256_file(transformer_path),
            "fit": final_fit_metadata,
        },
    )
    fit_metadata = {
        "protocol_fingerprint": fingerprint,
        "transformer_fit_seconds": transformer_fit_seconds,
        "development_transform_seconds": transform_seconds,
        "grid_fit_wall_seconds": grid_fit_seconds,
        "grid_fits": _fold_fit_records(fits),
        "final_refit_wall_seconds": final_fit_seconds,
        "final_refit": final_fit_metadata,
    }
    candidate_value = {
        "protocol_fingerprint": fingerprint,
        "selection_rule": (
            "maximize (minimum class hits, total hits, -restricted mean delay, "
            "ridge lambda, kappa)"
        ),
        "candidates": candidates,
        "selected": selected,
    }
    gpu_execution = {
        "protocol_fingerprint": fingerprint,
        "grid_fit_cuda_event_seconds": sum(
            float(fit.convergence["cuda_event_elapsed_seconds"]) for fit in fits
        ),
        "oof_scoring_cuda_event_seconds": sum(
            float(item["cuda_event_elapsed_seconds"]) for item in score_executions
        ),
        "final_fit_cuda_event_seconds": final_fit_metadata[
            "cuda_event_elapsed_seconds"
        ],
        "oof_scoring_calls": score_executions,
        "device_name": final_fit_metadata["device_name"],
        "dtype": "float64",
        "tf32": False,
    }
    aggregate = {
        "schema_version": 1,
        "analysis_status": "weakly_supervised_method_development",
        "protocol_fingerprint": fingerprint,
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "feature_transformer": {
            "valid_start_zero_based": transformer.valid_start_,
            "raw_feature_count": transformer.RAW_FEATURE_COUNT,
            "retained_feature_count": transformer.output_feature_count_,
            "dropped_feature_count": len(transformer.dropped_feature_names_),
        },
        "final_model": {
            "feature_count": final_model.feature_count,
            "fault_ids": list(final_model.fault_ids),
            "threshold": final_model.threshold,
            "safe_serialization": "numeric_npz_and_json_allow_pickle_false",
        },
        "gpu_execution": gpu_execution,
        "data_boundary": {
            "development_test_rows_500_959_discarded_before_transform": True,
            "audit_runs_101_200_not_loaded": True,
            "calibration_runs_301_400_not_loaded": True,
            "fault_runs_201_400_not_loaded": True,
            "final_runs_401_500_not_selected_scored_or_inspected": True,
            "archive_decode_caveat": (
                "pyreadr decodes a complete RData partition before selected rows "
                "are returned"
            ),
        },
    }
    _atomic_json(args.output_dir / "development_candidates.json", candidate_value)
    _atomic_json(args.output_dir / "development_fit_metadata.json", fit_metadata)
    _atomic_json(args.output_dir / "development_gpu_execution.json", gpu_execution)
    _atomic_json(args.output_dir / "development_aggregate.json", aggregate)
    _atomic_npz(
        args.output_dir / "development_scores.npz",
        _pack_paths(all_runs, selected_paths, threshold=float(selected["threshold"])),
    )
    _atomic_text(
        args.output_dir / "DEVELOPMENT_REPORT.md",
        _development_report(aggregate),
    )
    _write_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
        "DEVELOPMENT_COMPLETE_METHOD_FROZEN_FOR_AUDIT",
        _development_artifacts(),
    )
    return _verify_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
    ) or {}


def _load_completed_development(
    output_dir: Path,
    *,
    current_source_hashes: Mapping[str, str],
) -> tuple[dict[str, object], str, dict[str, object]]:
    manifest, fingerprint = _read_json_with_fingerprint(
        output_dir / "development_manifest.json",
        fingerprint_field="protocol_fingerprint",
    )
    if manifest.get("source_sha256") != dict(current_source_hashes):
        raise ValueError("source files changed after the development manifest")
    completion = _verify_completion(
        output_dir / "development_completion.json",
        output_dir,
        fingerprint,
    )
    if completion is None:
        raise RuntimeError("audit requires a completed development phase")
    candidates = json.loads(
        (output_dir / "development_candidates.json").read_text(encoding="utf-8")
    )
    selected = candidates.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("development selection artifact is invalid")
    return completion, fingerprint, selected


def _calibration_artifacts_valid(
    output_dir: Path,
    open_manifest: Mapping[str, object],
) -> None:
    pairs = {
        "calibration.json": open_manifest["calibration_sha256"],
        "calibration_scores.npz": open_manifest["calibration_scores_sha256"],
        "cuda_validation.json": open_manifest["cuda_validation_sha256"],
        "directional_glr_calibrated.npz": open_manifest[
            "calibrated_model_sha256"
        ],
    }
    for name, expected in pairs.items():
        path = output_dir / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"audit-open artifact changed: {name}")


def _audit_artifacts() -> tuple[str, ...]:
    return _development_artifacts() + (
        "development_completion.json",
        "audit_base_manifest.json",
        "calibration.json",
        "calibration_scores.npz",
        "cuda_validation.json",
        "directional_glr_calibrated.npz",
        "audit_open_manifest.json",
        "audit_result.json",
        "audit_scores.npz",
        "gpu_execution.json",
        "aggregate.json",
        "REPORT.md",
    )


def _run_audit(
    args: argparse.Namespace,
    *,
    project_root: Path,
    archive: RiethTEPArchive,
    checksums: dict[str, str],
) -> dict[str, object]:
    source_hashes = _source_hashes(project_root)
    development, development_fingerprint, selected = _load_completed_development(
        args.output_dir,
        current_source_hashes=source_hashes,
    )
    transformer_hash = _artifact_hash(development, "directional_transformer.npz")
    model_hash = _artifact_hash(development, "directional_glr_uncalibrated.npz")
    base_protocol = _audit_base_protocol(
        development_fingerprint=development_fingerprint,
        development_completion_sha256=_sha256_file(
            args.output_dir / "development_completion.json"
        ),
        transformer_sha256=transformer_hash,
        model_sha256=model_hash,
        selected=selected,
        checksums=checksums,
        source_hashes=source_hashes,
        batch_size=args.batch_size,
    )
    base_fingerprint = _load_or_create_manifest(
        args.output_dir / "audit_base_manifest.json",
        base_protocol,
        require_empty_parent=False,
    )
    completion_path = args.output_dir / "completion.json"
    open_path = args.output_dir / "audit_open_manifest.json"
    if completion_path.exists():
        open_manifest, open_fingerprint = _read_json_with_fingerprint(
            open_path,
            fingerprint_field="protocol_fingerprint",
        )
        if open_manifest.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("completed audit belongs to another base protocol")
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        completed = _verify_completion(
            completion_path,
            args.output_dir,
            open_fingerprint,
        )
        if completed is None:
            raise RuntimeError("audit completion disappeared during verification")
        print("directional audit is already complete", flush=True)
        return completed

    transformer, transformer_metadata = load_directional_feature_transformer(
        args.output_dir / "directional_transformer.npz"
    )
    model, model_metadata = load_directional_glr_model(
        args.output_dir / "directional_glr_uncalibrated.npz"
    )
    if model.threshold is not None:
        raise ValueError("development model must be uncalibrated")
    if model.feature_count != transformer.output_feature_count_:
        raise ValueError("transformer and GLR model feature counts differ")

    if open_path.exists():
        open_manifest, open_fingerprint = _read_json_with_fingerprint(
            open_path,
            fingerprint_field="protocol_fingerprint",
        )
        if open_manifest.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("audit-open manifest belongs to another base protocol")
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        calibrated_model, calibrated_metadata = load_directional_glr_model(
            args.output_dir / "directional_glr_calibrated.npz"
        )
        calibration = json.loads(
            (args.output_dir / "calibration.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (args.output_dir / "cuda_validation.json").read_text(encoding="utf-8")
        )
        frozen_thresholds = (
            open_manifest.get("threshold"),
            calibration.get("threshold"),
            calibrated_model.threshold,
        )
        if (
            any(value is None for value in frozen_thresholds)
            or len({float(value) for value in frozen_thresholds}) != 1
            or not bool(validation.get("passed"))
        ):
            raise ValueError("existing calibration, model, and validation disagree")
        if calibrated_metadata.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("calibrated model metadata belongs to another audit")
        calibration_execution = calibration["gpu_execution"]
        print("verified existing audit-open freeze; calibration not reopened", flush=True)
    else:
        print("loading healthy calibration runs FF-testing 301..400", flush=True)
        calibration_full = _run_ids(
            archive.load_runs(
                "fault_free_testing",
                {0: CALIBRATION_RUNS},
                fault_start=FAULT_START,
                split_manipulated=False,
            ),
            CALIBRATION_RUNS,
            partition="fault_free_testing",
            expected_length=EXPECTED_TEST_LENGTH,
            label="audit calibration",
        )
        calibration_runs = tuple(
            _truncate_run(run, DEVELOPMENT_STOP) for run in calibration_full
        )
        transform_started = time.perf_counter()
        calibration_features = _feature_mapping(transformer, calibration_runs)
        calibration_transform_seconds = time.perf_counter() - transform_started
        score_started = time.perf_counter()
        calibration_scored = score_directional_glr_cuda(
            model,
            calibration_features,
            valid_start=transformer.valid_start_,
            device=args.device,
            batch_size=args.batch_size,
        )
        calibration_wall_seconds = time.perf_counter() - score_started
        threshold, conformal = conformal_run_threshold(
            {
                name: path.envelope
                for name, path in calibration_scored.by_run.items()
            },
            alpha=ALPHA,
            required_count=len(CALIBRATION_RUNS),
        )
        calibrated_model = model.with_threshold(threshold)
        representative_name = calibration_runs[0].name
        validation = validate_directional_glr_cuda(
            calibrated_model,
            calibration_features[representative_name],
            valid_start=transformer.valid_start_,
            device=args.device,
            atol=CPU_GPU_TOLERANCE,
        )
        validation = {
            "passed": True,
            "representative_run": representative_name,
            "audit_base_fingerprint": base_fingerprint,
            **validation,
        }
        calibration_execution = {
            "feature_transform_seconds": calibration_transform_seconds,
            "score_wall_seconds": calibration_wall_seconds,
            **calibration_scored.execution,
        }
        calibration = {
            "audit_base_fingerprint": base_fingerprint,
            "method": "directional_density_ratio_page_cusum",
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
            "sequence_truncation_half_open": [0, DEVELOPMENT_STOP],
            "valid_start_zero_based": transformer.valid_start_,
            "first_alarm_eligible_zero_based": (
                transformer.valid_start_ + PERSISTENCE_WINDOW - 1
            ),
            **conformal,
            "gpu_execution": calibration_execution,
        }
        calibrated_metadata = {
            **model_metadata,
            "audit_base_fingerprint": base_fingerprint,
            "threshold_calibration": conformal,
            "transformer_metadata": transformer_metadata,
        }
        calibrated_path = args.output_dir / "directional_glr_calibrated.npz"
        calibrated_model.save_npz(
            calibrated_path,
            metadata=calibrated_metadata,
        )
        _atomic_json(args.output_dir / "calibration.json", calibration)
        _atomic_json(args.output_dir / "cuda_validation.json", validation)
        _atomic_npz(
            args.output_dir / "calibration_scores.npz",
            _pack_paths(
                calibration_runs,
                calibration_scored.by_run,
                threshold=threshold,
            ),
        )
        open_protocol = _audit_open_protocol(
            base_fingerprint=base_fingerprint,
            calibration=calibration,
            calibration_sha256=_sha256_file(args.output_dir / "calibration.json"),
            calibration_scores_sha256=_sha256_file(
                args.output_dir / "calibration_scores.npz"
            ),
            validation_sha256=_sha256_file(
                args.output_dir / "cuda_validation.json"
            ),
            calibrated_model_sha256=_sha256_file(calibrated_path),
        )
        open_fingerprint = _load_or_create_manifest(
            open_path,
            open_protocol,
            require_empty_parent=False,
        )
        open_manifest = json.loads(open_path.read_text(encoding="utf-8"))
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        print(
            f"audit opened after q96={threshold:.6g} and CPU/CUDA validation",
            flush=True,
        )

    print("loading method-specific audit healthy/fault runs 101..200", flush=True)
    audit_healthy_full = _run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: AUDIT_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        AUDIT_RUNS,
        partition="fault_free_testing",
        expected_length=EXPECTED_TEST_LENGTH,
        label="method-specific healthy audit",
    )
    audit_faults_full = _fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: AUDIT_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        AUDIT_RUNS,
        expected_length=EXPECTED_TEST_LENGTH,
        label="method-specific fault audit",
    )
    audit_healthy = tuple(
        _truncate_run(run, DEVELOPMENT_STOP) for run in audit_healthy_full
    )
    audit_faults = tuple(
        _truncate_run(run, DEVELOPMENT_STOP) for run in audit_faults_full
    )
    audit_runs = audit_healthy + audit_faults
    transform_started = time.perf_counter()
    audit_features = _feature_mapping(transformer, audit_runs)
    audit_transform_seconds = time.perf_counter() - transform_started
    score_started = time.perf_counter()
    audit_scored = score_directional_glr_cuda(
        calibrated_model,
        audit_features,
        valid_start=transformer.valid_start_,
        device=args.device,
        batch_size=args.batch_size,
    )
    audit_wall_seconds = time.perf_counter() - score_started
    threshold = float(calibrated_model.threshold)
    result = _audit_summary(
        _health_records(
            audit_healthy,
            audit_scored.by_run,
            threshold=threshold,
        ),
        _fault_records(
            audit_faults,
            audit_scored.by_run,
            threshold=threshold,
        ),
    )
    execution = {
        "device_name": audit_scored.execution["device_name"],
        "dtype": "float64",
        "tf32": False,
        "calibration": calibration_execution,
        "audit": {
            "feature_transform_seconds": audit_transform_seconds,
            "score_wall_seconds": audit_wall_seconds,
            **audit_scored.execution,
        },
        "audit_cuda_event_seconds": audit_scored.execution[
            "cuda_event_elapsed_seconds"
        ],
    }
    aggregate = {
        "schema_version": 1,
        "analysis_status": "frozen_method_specific_audit",
        "development_protocol_fingerprint": development_fingerprint,
        "audit_base_fingerprint": base_fingerprint,
        "audit_open_fingerprint": open_fingerprint,
        "selected_hyperparameters": {
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
            "threshold": threshold,
        },
        "calibration": calibration,
        "cuda_validation": validation,
        "audit": result,
        "gpu_execution": execution,
        "data_boundary": {
            "all_selected_test_runs_truncated_to_0_500_before_transform": True,
            "rows_500_959_not_transformed_or_scored": True,
            "method_specific_healthy_101_200_consumed": True,
            "method_specific_faults_101_200_consumed": True,
            "faults_201_400_not_selected_scored_or_inspected": True,
            "final_healthy_401_500_not_selected_scored_or_inspected": True,
            "final_all_faults_401_500_not_selected_scored_or_inspected": True,
            "archive_decode_caveat": (
                "pyreadr decodes a complete RData partition before selected rows "
                "are returned"
            ),
        },
        "safe_model_serialization": "numeric_npz_and_json_allow_pickle_false",
    }
    _atomic_json(args.output_dir / "audit_result.json", result)
    _atomic_json(args.output_dir / "gpu_execution.json", execution)
    _atomic_json(args.output_dir / "aggregate.json", aggregate)
    _atomic_npz(
        args.output_dir / "audit_scores.npz",
        _pack_paths(audit_runs, audit_scored.by_run, threshold=threshold),
    )
    _atomic_text(args.output_dir / "REPORT.md", _audit_report(aggregate))
    _write_completion(
        args.output_dir / "completion.json",
        args.output_dir,
        open_fingerprint,
        str(result["decision"]),
        _audit_artifacts(),
    )
    print(f"audit decision: {result['decision']}", flush=True)
    return _verify_completion(
        args.output_dir / "completion.json",
        args.output_dir,
        open_fingerprint,
    ) or {}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    args.output_dir = args.output_dir.resolve()
    cva_model, cva_completion = _resolve_cva_paths(
        project_root,
        args.cva_model,
        args.cva_completion,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = RiethTEPArchive(args.data_dir, verify_checksums=True)
    checksums = archive.verify(
        ("fault_free_training", "fault_free_testing", "faulty_testing")
    )
    if args.phase in {"development", "all"}:
        _run_development(
            args,
            project_root=project_root,
            archive=archive,
            checksums=checksums,
            cva_model_path=cva_model,
            cva_completion_path=cva_completion,
        )
    if args.phase in {"audit", "all"}:
        _run_audit(
            args,
            project_root=project_root,
            archive=archive,
            checksums=checksums,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
