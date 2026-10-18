"""Run the frozen hybrid directional-energy GLR-CUSUM experiment on CUDA.

This is a sequential exploratory follow-up to the completed directional
experiment.  It authenticates that experiment's failed completion before it
loads the frozen directional transformer, appends variance-sensitive energy
features, and fits three joint one-vs-normal density-ratio heads.  Development,
calibration, and audit are deliberately separated by hash-authenticated
manifests so that audit values cannot be opened before the method is frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# On this Windows host CUDA must be initialized before NumPy/MKL is imported.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_directional_glr_gpu as base  # noqa: E402

from mate_spc.datasets import (  # noqa: E402
    RiethTEPArchive,
    SensorSequence,
)
from mate_spc_directional.features import (  # noqa: E402
    load_directional_feature_transformer,
)
from mate_spc_directional.glr import (  # noqa: E402
    FAULT_IDS,
    NORMAL_ID,
    PERSISTENCE_WINDOW,
    TRAIN_START,
    TRAIN_STOP,
    Z_CLIP,
    DirectionalGLRPath,
    conformal_run_threshold,
    load_directional_glr_model,
)
from mate_spc_directional.gpu import (  # noqa: E402
    DirectionalGLRFoldFit,
    fit_directional_glr_cuda,
    fit_directional_glr_grid_cuda,
    score_directional_glr_cuda,
    validate_directional_glr_cuda,
)
from mate_spc_hybrid.features import (  # noqa: E402
    ENERGY_FEATURE_NAMES,
    ENERGY_RAW_FEATURE_COUNT,
    ENERGY_WINDOWS,
    HybridFeatureTransformer,
    load_hybrid_feature_transformer,
)


DATASET_DOI = "10.7910/DVN/6C3JR1"
ANALYSIS_STATUS = "sequential_exploratory_new_method"
FAULT_START = 160
REFERENCE_HEALTH_RUNS = tuple(range(201, 301))
DEVELOPMENT_RUNS = tuple(range(1, 201))
CALIBRATION_RUNS = tuple(range(301, 401))
AUDIT_RUNS = tuple(range(201, 301))
FAULT_RUNS_FORBIDDEN_AFTER_FAILURE = tuple(range(301, 401))
FINAL_RUNS = tuple(range(401, 501))
SEQUENCE_STOP = 500
EXPECTED_TRAIN_LENGTH = 500
EXPECTED_TEST_LENGTH = 960
RIDGE_LAMBDAS = (1e-3, 1e-2, 1e-1)
KAPPAS = (0.5, 0.75, 1.0, 1.25)
FOLDS = tuple(range(5))
ALPHA = 0.05
DEVELOPMENT_RANK = 191
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
DIRECTIONAL_FAILURE_DECISION = "FAIL_FROZEN_AUDIT_STOP_METHOD_SEARCH"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen hybrid directional-energy GLR-CUSUM experiment"
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
        "--directional-transformer",
        type=Path,
        help="directional_transformer.npz from the completed failed experiment",
    )
    parser.add_argument(
        "--directional-completion",
        type=Path,
        help="FAIL completion.json authenticating --directional-transformer",
    )
    return parser


def _source_hashes(project_root: Path) -> dict[str, str]:
    hybrid_package = sorted((project_root / "src" / "mate_spc_hybrid").glob("*.py"))
    directional_package = sorted(
        (project_root / "src" / "mate_spc_directional").glob("*.py")
    )
    hybrid_tests = sorted((project_root / "tests").glob("*hybrid*.py"))
    dependencies = (
        project_root / "scripts" / "run_hybrid_energy_glr_gpu.py",
        project_root / "scripts" / "run_directional_glr_gpu.py",
        project_root / "src" / "mate_spc" / "cva_mewma.py",
        project_root / "src" / "mate_spc" / "dynamic_common.py",
        project_root / "src" / "mate_spc" / "datasets.py",
        project_root / "pyproject.toml",
    )
    paths = tuple(
        sorted(
            set(hybrid_package + directional_package + hybrid_tests + list(dependencies))
        )
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hybrid protocol source files are missing: {missing}")
    required = {"__init__.py", "features.py"}
    found = {path.name for path in hybrid_package}
    if not required.issubset(found):
        raise FileNotFoundError(
            f"hybrid package is incomplete: missing={sorted(required - found)}"
        )
    return {
        path.relative_to(project_root).as_posix(): base._sha256_file(path)
        for path in paths
    }


def _resolve_directional_paths(
    project_root: Path,
    transformer: Path | None,
    completion: Path | None,
) -> tuple[Path, Path]:
    default_root = project_root / "outputs" / "directional_glr_gpu_v1"
    transformer_path = transformer or default_root / "directional_transformer.npz"
    completion_path = completion or transformer_path.parent / "completion.json"
    return transformer_path.resolve(), completion_path.resolve()


def _directional_reference(
    transformer_path: Path,
    completion_path: Path,
    *,
    load_transformer: bool,
) -> tuple[dict[str, object], object | None]:
    completion, _ = base._read_json_with_fingerprint(
        completion_path,
        fingerprint_field="completion_fingerprint",
    )
    if completion.get("decision") != DIRECTIONAL_FAILURE_DECISION:
        raise ValueError(
            "the parent directional transformer must come from its recorded "
            "failed frozen audit"
        )
    if transformer_path.parent.resolve() != completion_path.parent.resolve():
        raise ValueError("directional transformer and completion must share a directory")
    expected_sha = base._artifact_hash(completion, transformer_path.name)
    artifacts = completion.get("artifacts")
    record = artifacts.get(transformer_path.name) if isinstance(artifacts, Mapping) else None
    if not transformer_path.is_file() or not isinstance(record, Mapping):
        raise FileNotFoundError("authenticated directional transformer is missing")
    observed_sha = base._sha256_file(transformer_path)
    if (
        transformer_path.stat().st_size != int(record.get("size_bytes", -1))
        or observed_sha != expected_sha
    ):
        raise ValueError("directional transformer does not match its FAIL completion")
    reference = {
        "analysis_status": ANALYSIS_STATUS,
        "transformer_path": transformer_path.name,
        "transformer_sha256": observed_sha,
        "transformer_size_bytes": transformer_path.stat().st_size,
        "completion_path": completion_path.name,
        "completion_sha256": base._sha256_file(completion_path),
        "completion_fingerprint": completion["completion_fingerprint"],
        "parent_phase_fingerprint": completion["phase_fingerprint"],
        "required_parent_decision": DIRECTIONAL_FAILURE_DECISION,
        "serialization_policy": "numeric_npz_loaded_with_allow_pickle_false",
    }
    if not load_transformer:
        return reference, None
    directional, metadata = load_directional_feature_transformer(transformer_path)
    if directional.valid_start_ != 63 or directional.cva_rank != 56:
        raise ValueError("authenticated directional transformer violates frozen geometry")
    if directional.output_feature_count_ < 1:
        raise ValueError("authenticated directional transformer has no retained features")
    reference["embedded_metadata_sha256"] = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return reference, directional


def _development_protocol(
    *,
    checksums: dict[str, str],
    source_hashes: dict[str, str],
    directional_reference: dict[str, object],
    batch_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "dataset": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": DATASET_DOI,
        "dataset_md5": checksums,
        "source_sha256": source_hashes,
        "authenticated_parent_directional_transformer": directional_reference,
        "execution": {
            "required_device": "cuda",
            "fit_dtype": "float64",
            "score_dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "cpu_gpu_tolerance": CPU_GPU_TOLERANCE,
        },
        "method": {
            "name": "hybrid_directional_energy_density_ratio_page_cusum_3of3",
            "feature_transformer": {
                "composition": (
                    "all retained authenticated directional coordinates followed by "
                    "retained independently scaled energy coordinates"
                ),
                "directional_transformer_refit": False,
                "directional_valid_start_zero_based": 63,
                "energy_windows_samples": list(ENERGY_WINDOWS),
                "energy_raw_feature_count": ENERGY_RAW_FEATURE_COUNT,
                "required_retained_energy_feature_count": ENERGY_RAW_FEATURE_COUNT,
                "energy_feature_names": list(ENERGY_FEATURE_NAMES),
                "energy_statistics": [
                    "rolling mean square of each standardized signed CVA residual EWMA",
                    "rolling mean square of each standardized raw-variable first difference",
                ],
                "energy_scaling": (
                    "healthy-reference median and 1.4826*MAD; sample-SD fallback; "
                    "drop energy coordinates still degenerate"
                ),
                "energy_scaler_fit_data": [
                    "fault_free_training",
                    list(REFERENCE_HEALTH_RUNS),
                ],
                "output_clip": [-10.0, 10.0],
                "run_boundary_rule": "all states and windows reset per run",
            },
            "classifier": {
                "heads": "three joint one-vs-normal balanced-prior ridge-logistic log-odds",
                "fault_head_order": list(FAULT_IDS),
                "training_time_slice_zero_based_half_open": [TRAIN_START, TRAIN_STOP],
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
                "normal_logit_scale": "1.4826*MAD with sample-SD fallback per head",
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
            "energy_scaler_reference": [
                "fault_free_training",
                list(REFERENCE_HEALTH_RUNS),
            ],
            "classifier_normal": ["fault_free_testing", list(DEVELOPMENT_RUNS)],
            "classifier_faults": [
                "faulty_testing",
                list(FAULT_IDS),
                list(DEVELOPMENT_RUNS),
            ],
            "all_test_sequences_truncated_half_open": [0, SEQUENCE_STOP],
            "fold": "(simulation_run - 1) % 5",
            "fold_ids": list(FOLDS),
            "per_fold_per_class_runs": 40,
            "grouping": "matched normal and all three faults share the same fold",
        },
        "development_selection": {
            "oof_healthy_threshold": (
                "pool exactly 200 out-of-fold healthy run envelope maxima and "
                "take order statistic 191 at alpha 0.05"
            ),
            "selection_key_maximized_lexicographically": [
                "minimum_detected_runs_across_fault_classes",
                "fault_9_detected_runs",
                "total_detected_fault_runs",
                "negative_overall_restricted_mean_delay_miss_as_340",
                "larger_ridge_lambda",
                "larger_kappa",
            ],
            "final_fit": "refit selected ridge on all 200 runs per class",
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
            "all_sequence_truncation_half_open": [0, SEQUENCE_STOP],
            "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
            "detected_runs_per_fault_minimum": AUDIT_PER_FAULT_MINIMUM,
            "detected_fault_runs_total_minimum": AUDIT_TOTAL_MINIMUM,
            "all_gates_required": True,
            "provenance_note": (
                "fault runs 201-300 were not previously selected or scored; the "
                "healthy split is method-specific but not globally unseen"
            ),
        },
        "reserved": {
            "fault_runs_after_failure": [
                list(FAULT_RUNS_FORBIDDEN_AFTER_FAILURE),
                "do_not_open_for_more_tuning",
            ],
            "final_healthy": [
                list(FINAL_RUNS),
                "do_not_select_score_or_inspect",
            ],
            "final_all_faults": [
                list(FINAL_RUNS),
                "do_not_select_score_or_inspect",
            ],
        },
        "data_access_semantics": {
            "archive_format": "partition-level RData decoded before row selection",
            "model_visible_rows": "only explicit load_runs selections",
            "all_test_rows_500_959": "discarded before feature transformation",
            "reserved_claim": "runs 401-500 are not selected, scored, or inspected",
        },
        "forbidden_after_manifest": [
            "feature family or parent directional transformer",
            "training time interval",
            "ridge or kappa grids",
            "fold assignment or selection objective",
            "development, calibration, and audit splits",
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
        "analysis_status": ANALYSIS_STATUS,
        "phase": "frozen_method_specific_audit_base",
        "development_protocol_fingerprint": development_fingerprint,
        "development_completion_sha256": development_completion_sha256,
        "hybrid_transformer_sha256": transformer_sha256,
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
            "only then load method-specific healthy/fault runs 201-300",
        ],
        "calibration": {
            "split": ["fault_free_testing", list(CALIBRATION_RUNS)],
            "sequence_truncation_half_open": [0, SEQUENCE_STOP],
            "unit": "independent healthy run family-envelope maximum",
            "alpha": ALPHA,
            "rank_one_based": CALIBRATION_RANK,
            "comparison": "strictly_greater_than",
        },
        "audit": {
            "healthy": ["fault_free_testing", list(AUDIT_RUNS)],
            "faults": ["faulty_testing", list(FAULT_IDS), list(AUDIT_RUNS)],
            "sequence_truncation_half_open": [0, SEQUENCE_STOP],
            "fault_onset_zero_based": FAULT_START,
            "pre_fault_episode_continuing_into_fault": "not_credited",
            "miss_delay_samples": SEQUENCE_STOP - FAULT_START,
            "gates": {
                "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
                "per_fault_detected_runs_minimum": AUDIT_PER_FAULT_MINIMUM,
                "total_fault_detected_runs_minimum": AUDIT_TOTAL_MINIMUM,
                "all_required": True,
            },
            "provenance": {
                "faults_201_300_previously_selected_or_scored": False,
                "healthy_201_300_globally_unseen": False,
                "interpretation": "method-specific sequential exploratory audit",
            },
        },
        "stop_rule": {
            "on_failure": "do not open faulty runs 301-400 for model search",
            "final_runs": "do not select, score, or inspect runs 401-500",
        },
        "rdata_decode_caveat": (
            "pyreadr decodes a complete partition before load_runs returns only "
            "selected run rows"
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
        "analysis_status": ANALYSIS_STATUS,
        "phase": "audit_open_authorization",
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
        "audit_fault_provenance": "faults_201_300_not_previously_selected_or_scored",
        "healthy_provenance": "method_specific_not_globally_unseen",
        "gates": {
            "healthy_alarm_runs_maximum": AUDIT_HEALTH_MAXIMUM,
            "per_fault_detected_runs_minimum": AUDIT_PER_FAULT_MINIMUM,
            "total_fault_detected_runs_minimum": AUDIT_TOTAL_MINIMUM,
            "all_required": True,
        },
        "created_before_method_specific_audit_load": True,
    }


def _feature_mapping(
    transformer: HybridFeatureTransformer,
    runs: Iterable[SensorSequence],
) -> dict[str, np.ndarray]:
    return {
        name: features
        for name, features in transformer.transform_named_runs(base._named_values(runs))
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
        f"{fault_id:02d}": base._fault_aggregate(fault_records, fault_id)
        for fault_id in FAULT_IDS
    }
    overall = base._fault_aggregate(fault_records)
    class_hits = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"]) for fault_id in FAULT_IDS
    ]
    selection_key = (
        min(class_hits),
        int(by_fault["09"]["detected_runs"]),
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
        "fault_9_detected_runs": int(by_fault["09"]["detected_runs"]),
        "selection_key": list(selection_key),
    }


def _selection_key(candidate: Mapping[str, object]) -> tuple[float, ...]:
    value = candidate.get("selection_key")
    if not isinstance(value, Sequence) or len(value) != 6:
        raise ValueError("candidate has an invalid six-coordinate selection key")
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
        held_out_names = [name for fit in ridge_fits for name in fit.held_out_run_names]
        if len(held_out_names) != len(set(held_out_names)) or set(held_out_names) != (
            expected_names
        ):
            raise RuntimeError("OOF fits do not form an exact development partition")
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
                        "purpose": "hybrid_oof_grid_scoring",
                        "ridge_lambda": ridge_lambda,
                        "kappa": kappa,
                        "held_out_fold": fit.held_out_fold,
                        "wall_seconds": time.perf_counter() - started,
                        **scored.execution,
                    }
                )
            if set(paths) != expected_names:
                raise RuntimeError("OOF scoring did not return every development run")
            threshold, calibration = conformal_run_threshold(
                {
                    name: path.envelope
                    for name, path in paths.items()
                    if int(labels[name]) == NORMAL_ID
                },
                alpha=ALPHA,
                required_count=len(DEVELOPMENT_RUNS),
            )
            if int(calibration["rank_one_based"]) != DEVELOPMENT_RANK:
                raise RuntimeError("development conformal threshold is not q191")
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
                health_records=base._health_records(
                    healthy_runs,
                    paths,
                    threshold=threshold,
                ),
                fault_records=base._fault_records(
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
        raise RuntimeError("hybrid development grid produced no candidate")
    return candidates, best_paths, best_candidate, executions


def _development_report(aggregate: Mapping[str, object]) -> str:
    selected = aggregate["selected_candidate"]
    return "\n".join(
        [
            "# Hybrid directional-energy GLR-CUSUM development",
            "",
            f"Analysis status: {ANALYSIS_STATUS}.",
            f"Protocol fingerprint: {aggregate['protocol_fingerprint']}.",
            "",
            "The completed failed directional experiment authenticated the frozen ",
            "directional transformer. Only the appended energy scaler used healthy ",
            "FF-training runs 201-300. The classifier and every hyperparameter ",
            "comparison used grouped five-fold out-of-fold predictions from ",
            "FF-testing/fault runs 1-200, truncated to samples 0:500.",
            "",
            "## Frozen selection",
            "",
            f"- Ridge lambda: {selected['ridge_lambda']}.",
            f"- Page kappa: {selected['kappa']}.",
            f"- OOF healthy q191 threshold: {selected['threshold']:.6g}.",
            f"- Minimum class hits: {selected['minimum_detected_runs_across_faults']}.",
            f"- Fault 9 hits: {selected['fault_9_detected_runs']}/200.",
            f"- Total fault hits: {selected['overall']['detected_runs']}/600.",
            f"- Restricted mean delay: "
            f"{selected['overall']['restricted_mean_delay_samples']:.3f} samples.",
            "",
            "The OOF threshold is not reused for audit. Audit requires a fresh q96 ",
            "threshold from healthy FF-testing runs 301-400.",
            "",
            "Runs 401-500 were not selected, scored, or inspected. Rieth RData files ",
            "are nevertheless decoded at partition level before selected rows return.",
            "",
        ]
    )


def _audit_report(aggregate: Mapping[str, object]) -> str:
    result = aggregate["audit"]
    health = result["healthy"]
    faults = result["faults"]
    lines = [
        "# Frozen hybrid directional-energy GLR-CUSUM audit",
        "",
        f"Analysis status: {ANALYSIS_STATUS}.",
        f"Audit-open fingerprint: {aggregate['audit_open_fingerprint']}.",
        "",
        "Calibration on healthy FF-testing runs 301-400 and CPU/CUDA validation ",
        "were completed before method-specific audit healthy/fault runs 201-300 ",
        "were loaded. Audit faults 201-300 had not previously been selected or ",
        "scored; the healthy split is method-specific but not globally unseen.",
        "",
        "Every calibration and audit run was truncated to samples 0:500 before ",
        "feature transformation; rows 500:960 were not transformed or scored.",
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
            "- If this audit failed, faulty runs 301-400 remain unopened and must ",
            "not be used to continue tuning this method.",
            "- RData caveat: pyreadr decodes a complete partition before selected ",
            "rows are returned.",
            "",
        ]
    )
    return "\n".join(lines)


def _development_artifacts() -> tuple[str, ...]:
    return (
        "development_manifest.json",
        "hybrid_transformer.npz",
        "hybrid_glr_uncalibrated.npz",
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
    directional_transformer_path: Path,
    directional_completion_path: Path,
) -> dict[str, object]:
    parent_reference, _ = _directional_reference(
        directional_transformer_path,
        directional_completion_path,
        load_transformer=False,
    )
    protocol = _development_protocol(
        checksums=checksums,
        source_hashes=_source_hashes(project_root),
        directional_reference=parent_reference,
        batch_size=args.batch_size,
    )
    manifest_path = args.output_dir / "development_manifest.json"
    fingerprint = base._load_or_create_manifest(
        manifest_path,
        protocol,
        require_empty_parent=True,
    )
    completed = base._verify_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
    )
    if completed is not None:
        print("hybrid development is already complete", flush=True)
        return completed

    loaded_reference, directional = _directional_reference(
        directional_transformer_path,
        directional_completion_path,
        load_transformer=True,
    )
    if loaded_reference["transformer_sha256"] != parent_reference["transformer_sha256"]:
        raise RuntimeError("parent directional transformer changed during development open")
    assert directional is not None

    print("loading hybrid energy-scaler reference FF-training 201..300", flush=True)
    reference = base._run_ids(
        archive.load_runs(
            "fault_free_training",
            {0: REFERENCE_HEALTH_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        REFERENCE_HEALTH_RUNS,
        partition="fault_free_training",
        expected_length=EXPECTED_TRAIN_LENGTH,
        label="hybrid energy-scaler reference",
    )
    base._validate_full_values(reference)
    transformer_started = time.perf_counter()
    transformer = HybridFeatureTransformer(directional).fit(base._named_values(reference))
    transformer_fit_seconds = time.perf_counter() - transformer_started
    if transformer.energy_feature_count_ != ENERGY_RAW_FEATURE_COUNT:
        raise RuntimeError(
            "the frozen hybrid method requires all 768 energy coordinates to be "
            "nondegenerate on its healthy reference split"
        )

    print("loading hybrid development healthy/fault runs 1..200", flush=True)
    healthy_full = base._run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: DEVELOPMENT_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DEVELOPMENT_RUNS,
        partition="fault_free_testing",
        expected_length=EXPECTED_TEST_LENGTH,
        label="hybrid development healthy",
    )
    fault_full = base._fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: DEVELOPMENT_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DEVELOPMENT_RUNS,
        expected_length=EXPECTED_TEST_LENGTH,
        label="hybrid development faults",
    )
    healthy = tuple(base._truncate_run(run, SEQUENCE_STOP) for run in healthy_full)
    faults = tuple(base._truncate_run(run, SEQUENCE_STOP) for run in fault_full)
    all_runs = healthy + faults
    transform_started = time.perf_counter()
    features = _feature_mapping(transformer, all_runs)
    transform_seconds = time.perf_counter() - transform_started
    labels = {run.name: int(run.metadata["fault_id"]) for run in all_runs}
    folds = {
        run.name: (int(run.metadata["simulation_run"]) - 1) % len(FOLDS)
        for run in all_runs
    }
    runs_by_name = {run.name: run for run in all_runs}
    counts = {
        label: sum(value == label for value in labels.values())
        for label in (NORMAL_ID, *FAULT_IDS)
    }
    expected_counts = {0: 200, 3: 200, 9: 200, 15: 200}
    if counts != expected_counts:
        raise RuntimeError(f"hybrid development classes are not balanced: {counts}")
    for run_id in DEVELOPMENT_RUNS:
        matched_folds = {
            folds[run.name]
            for run in all_runs
            if int(run.metadata["simulation_run"]) == run_id
        }
        if matched_folds != {(run_id - 1) % len(FOLDS)}:
            raise RuntimeError("matched development runs do not share their frozen fold")

    print("fitting 3 ridge values x 5 grouped CUDA folds", flush=True)
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
        f"fault9={selected['fault_9_detected_runs']} "
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

    transformer_path = args.output_dir / "hybrid_transformer.npz"
    transformer.save_npz(
        transformer_path,
        metadata={
            "analysis_status": ANALYSIS_STATUS,
            "protocol_fingerprint": fingerprint,
            "parent_directional_transformer_sha256": parent_reference[
                "transformer_sha256"
            ],
            "parent_directional_completion_sha256": parent_reference[
                "completion_sha256"
            ],
            "energy_scaler_partition": "fault_free_training",
            "energy_scaler_runs": list(REFERENCE_HEALTH_RUNS),
            "valid_start_zero_based": transformer.valid_start_,
            "directional_feature_count": transformer.directional_feature_count_,
            "energy_raw_feature_count": ENERGY_RAW_FEATURE_COUNT,
            "retained_energy_feature_count": transformer.energy_feature_count_,
            "hybrid_feature_count": transformer.output_feature_count_,
        },
    )
    model_path = args.output_dir / "hybrid_glr_uncalibrated.npz"
    final_model.save_npz(
        model_path,
        metadata={
            "analysis_status": ANALYSIS_STATUS,
            "protocol_fingerprint": fingerprint,
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
            "audit_threshold": None,
            "development_oof_threshold_not_for_audit": selected["threshold"],
            "feature_transformer_sha256": base._sha256_file(transformer_path),
            "fit": final_fit_metadata,
        },
    )
    fit_metadata = {
        "analysis_status": ANALYSIS_STATUS,
        "protocol_fingerprint": fingerprint,
        "transformer_fit_seconds": transformer_fit_seconds,
        "development_transform_seconds": transform_seconds,
        "grid_fit_wall_seconds": grid_fit_seconds,
        "grid_fits": base._fold_fit_records(fits),
        "final_refit_wall_seconds": final_fit_seconds,
        "final_refit": final_fit_metadata,
    }
    candidate_value = {
        "analysis_status": ANALYSIS_STATUS,
        "protocol_fingerprint": fingerprint,
        "selection_rule": (
            "maximize (minimum class hits, fault 9 hits, total hits, negative "
            "restricted mean delay, ridge lambda, kappa)"
        ),
        "candidates": candidates,
        "selected": selected,
    }
    gpu_execution = {
        "analysis_status": ANALYSIS_STATUS,
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
        "analysis_status": ANALYSIS_STATUS,
        "phase": "development",
        "protocol_fingerprint": fingerprint,
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "feature_transformer": {
            "valid_start_zero_based": transformer.valid_start_,
            "directional_feature_count": transformer.directional_feature_count_,
            "energy_raw_feature_count": ENERGY_RAW_FEATURE_COUNT,
            "retained_energy_feature_count": transformer.energy_feature_count_,
            "dropped_energy_feature_count": len(
                transformer.dropped_energy_feature_names_
            ),
            "hybrid_feature_count": transformer.output_feature_count_,
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
            "development_runs_1_200_consumed": True,
            "audit_faults_201_300_not_loaded": True,
            "calibration_runs_301_400_not_loaded": True,
            "fault_runs_301_400_not_loaded": True,
            "final_runs_401_500_not_selected_scored_or_inspected": True,
            "archive_decode_caveat": (
                "pyreadr decodes a complete RData partition before selected rows "
                "are returned"
            ),
        },
    }
    base._atomic_json(args.output_dir / "development_candidates.json", candidate_value)
    base._atomic_json(args.output_dir / "development_fit_metadata.json", fit_metadata)
    base._atomic_json(args.output_dir / "development_gpu_execution.json", gpu_execution)
    base._atomic_json(args.output_dir / "development_aggregate.json", aggregate)
    base._atomic_npz(
        args.output_dir / "development_scores.npz",
        base._pack_paths(
            all_runs,
            selected_paths,
            threshold=float(selected["threshold"]),
        ),
    )
    base._atomic_text(
        args.output_dir / "DEVELOPMENT_REPORT.md",
        _development_report(aggregate),
    )
    base._write_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
        "DEVELOPMENT_COMPLETE_METHOD_FROZEN_FOR_AUDIT",
        _development_artifacts(),
    )
    del features, selected_paths, fits
    gc.collect()
    return base._verify_completion(
        args.output_dir / "development_completion.json",
        args.output_dir,
        fingerprint,
    ) or {}


def _load_completed_development(
    output_dir: Path,
    *,
    current_source_hashes: Mapping[str, str],
) -> tuple[dict[str, object], str, dict[str, object]]:
    manifest, fingerprint = base._read_json_with_fingerprint(
        output_dir / "development_manifest.json",
        fingerprint_field="protocol_fingerprint",
    )
    if manifest.get("analysis_status") != ANALYSIS_STATUS:
        raise ValueError("development manifest has the wrong analysis status")
    if manifest.get("source_sha256") != dict(current_source_hashes):
        raise ValueError("source files changed after the hybrid development manifest")
    completion = base._verify_completion(
        output_dir / "development_completion.json",
        output_dir,
        fingerprint,
    )
    if completion is None:
        raise RuntimeError("hybrid audit requires a completed development phase")
    candidates = json.loads(
        (output_dir / "development_candidates.json").read_text(encoding="utf-8")
    )
    selected = candidates.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("hybrid development selection artifact is invalid")
    return completion, fingerprint, selected


def _calibration_artifacts_valid(
    output_dir: Path,
    open_manifest: Mapping[str, object],
) -> None:
    pairs = {
        "calibration.json": open_manifest["calibration_sha256"],
        "calibration_scores.npz": open_manifest["calibration_scores_sha256"],
        "cuda_validation.json": open_manifest["cuda_validation_sha256"],
        "hybrid_glr_calibrated.npz": open_manifest["calibrated_model_sha256"],
    }
    for name, expected in pairs.items():
        path = output_dir / name
        if not path.is_file() or base._sha256_file(path) != expected:
            raise ValueError(f"hybrid audit-open artifact changed: {name}")


def _audit_artifacts() -> tuple[str, ...]:
    return _development_artifacts() + (
        "development_completion.json",
        "audit_base_manifest.json",
        "calibration.json",
        "calibration_scores.npz",
        "cuda_validation.json",
        "hybrid_glr_calibrated.npz",
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
    transformer_hash = base._artifact_hash(development, "hybrid_transformer.npz")
    model_hash = base._artifact_hash(development, "hybrid_glr_uncalibrated.npz")
    base_protocol = _audit_base_protocol(
        development_fingerprint=development_fingerprint,
        development_completion_sha256=base._sha256_file(
            args.output_dir / "development_completion.json"
        ),
        transformer_sha256=transformer_hash,
        model_sha256=model_hash,
        selected=selected,
        checksums=checksums,
        source_hashes=source_hashes,
        batch_size=args.batch_size,
    )
    base_fingerprint = base._load_or_create_manifest(
        args.output_dir / "audit_base_manifest.json",
        base_protocol,
        require_empty_parent=False,
    )
    completion_path = args.output_dir / "completion.json"
    open_path = args.output_dir / "audit_open_manifest.json"
    if completion_path.exists():
        open_manifest, open_fingerprint = base._read_json_with_fingerprint(
            open_path,
            fingerprint_field="protocol_fingerprint",
        )
        if open_manifest.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("completed hybrid audit belongs to another base protocol")
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        completed = base._verify_completion(
            completion_path,
            args.output_dir,
            open_fingerprint,
        )
        if completed is None:
            raise RuntimeError("hybrid audit completion disappeared during verification")
        print("hybrid audit is already complete", flush=True)
        return completed

    transformer, transformer_metadata = load_hybrid_feature_transformer(
        args.output_dir / "hybrid_transformer.npz"
    )
    if transformer.energy_feature_count_ != ENERGY_RAW_FEATURE_COUNT:
        raise ValueError("completed hybrid transformer does not retain 768 energies")
    model, model_metadata = load_directional_glr_model(
        args.output_dir / "hybrid_glr_uncalibrated.npz"
    )
    if model.threshold is not None:
        raise ValueError("hybrid development model must be uncalibrated")
    if model.feature_count != transformer.output_feature_count_:
        raise ValueError("hybrid transformer and GLR model feature counts differ")

    if open_path.exists():
        open_manifest, open_fingerprint = base._read_json_with_fingerprint(
            open_path,
            fingerprint_field="protocol_fingerprint",
        )
        if open_manifest.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("hybrid audit-open manifest belongs to another base protocol")
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        calibrated_model, calibrated_metadata = load_directional_glr_model(
            args.output_dir / "hybrid_glr_calibrated.npz"
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
            raise ValueError("existing hybrid calibration, model, and validation disagree")
        if calibrated_metadata.get("audit_base_fingerprint") != base_fingerprint:
            raise ValueError("calibrated hybrid model metadata belongs to another audit")
        calibration_execution = calibration["gpu_execution"]
        print("verified existing hybrid audit-open freeze", flush=True)
    else:
        print("loading healthy calibration runs FF-testing 301..400", flush=True)
        calibration_full = base._run_ids(
            archive.load_runs(
                "fault_free_testing",
                {0: CALIBRATION_RUNS},
                fault_start=FAULT_START,
                split_manipulated=False,
            ),
            CALIBRATION_RUNS,
            partition="fault_free_testing",
            expected_length=EXPECTED_TEST_LENGTH,
            label="hybrid audit calibration",
        )
        calibration_runs = tuple(
            base._truncate_run(run, SEQUENCE_STOP) for run in calibration_full
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
        if int(conformal["rank_one_based"]) != CALIBRATION_RANK:
            raise RuntimeError("audit calibration conformal threshold is not q96")
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
            "analysis_status": ANALYSIS_STATUS,
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
            "analysis_status": ANALYSIS_STATUS,
            "audit_base_fingerprint": base_fingerprint,
            "method": "hybrid_directional_energy_density_ratio_page_cusum",
            "ridge_lambda": selected["ridge_lambda"],
            "kappa": selected["kappa"],
            "sequence_truncation_half_open": [0, SEQUENCE_STOP],
            "valid_start_zero_based": transformer.valid_start_,
            "first_alarm_eligible_zero_based": (
                transformer.valid_start_ + PERSISTENCE_WINDOW - 1
            ),
            **conformal,
            "gpu_execution": calibration_execution,
        }
        calibrated_metadata = {
            **model_metadata,
            "analysis_status": ANALYSIS_STATUS,
            "audit_base_fingerprint": base_fingerprint,
            "threshold_calibration": conformal,
            "transformer_metadata": transformer_metadata,
        }
        calibrated_path = args.output_dir / "hybrid_glr_calibrated.npz"
        calibrated_model.save_npz(calibrated_path, metadata=calibrated_metadata)
        base._atomic_json(args.output_dir / "calibration.json", calibration)
        base._atomic_json(args.output_dir / "cuda_validation.json", validation)
        base._atomic_npz(
            args.output_dir / "calibration_scores.npz",
            base._pack_paths(calibration_runs, calibration_scored.by_run, threshold=threshold),
        )
        open_protocol = _audit_open_protocol(
            base_fingerprint=base_fingerprint,
            calibration=calibration,
            calibration_sha256=base._sha256_file(
                args.output_dir / "calibration.json"
            ),
            calibration_scores_sha256=base._sha256_file(
                args.output_dir / "calibration_scores.npz"
            ),
            validation_sha256=base._sha256_file(
                args.output_dir / "cuda_validation.json"
            ),
            calibrated_model_sha256=base._sha256_file(calibrated_path),
        )
        open_fingerprint = base._load_or_create_manifest(
            open_path,
            open_protocol,
            require_empty_parent=False,
        )
        open_manifest = json.loads(open_path.read_text(encoding="utf-8"))
        _calibration_artifacts_valid(args.output_dir, open_manifest)
        print(
            f"hybrid audit opened after q96={threshold:.6g} and CPU/CUDA validation",
            flush=True,
        )

    print("loading method-specific audit healthy/fault runs 201..300", flush=True)
    audit_healthy_full = base._run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: AUDIT_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        AUDIT_RUNS,
        partition="fault_free_testing",
        expected_length=EXPECTED_TEST_LENGTH,
        label="hybrid method-specific healthy audit",
    )
    audit_faults_full = base._fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: AUDIT_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        AUDIT_RUNS,
        expected_length=EXPECTED_TEST_LENGTH,
        label="hybrid method-specific fault audit",
    )
    audit_healthy = tuple(
        base._truncate_run(run, SEQUENCE_STOP) for run in audit_healthy_full
    )
    audit_faults = tuple(
        base._truncate_run(run, SEQUENCE_STOP) for run in audit_faults_full
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
    result = {
        "analysis_status": ANALYSIS_STATUS,
        **base._audit_summary(
            base._health_records(
                audit_healthy,
                audit_scored.by_run,
                threshold=threshold,
            ),
            base._fault_records(
                audit_faults,
                audit_scored.by_run,
                threshold=threshold,
            ),
        ),
    }
    execution = {
        "analysis_status": ANALYSIS_STATUS,
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
        "analysis_status": ANALYSIS_STATUS,
        "phase": "frozen_method_specific_audit",
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
        "data_provenance": {
            "audit_faults_201_300_previously_selected_or_scored": False,
            "audit_healthy_201_300_globally_unseen": False,
            "audit_interpretation": "method-specific sequential exploratory audit",
        },
        "data_boundary": {
            "all_selected_test_runs_truncated_to_0_500_before_transform": True,
            "rows_500_959_not_transformed_or_scored": True,
            "method_specific_healthy_201_300_consumed": True,
            "method_specific_faults_201_300_consumed": True,
            "faults_301_400_not_selected_scored_or_inspected": True,
            "final_healthy_401_500_not_selected_scored_or_inspected": True,
            "final_all_faults_401_500_not_selected_scored_or_inspected": True,
            "archive_decode_caveat": (
                "pyreadr decodes a complete RData partition before selected rows "
                "are returned"
            ),
        },
        "safe_model_serialization": "numeric_npz_and_json_allow_pickle_false",
    }
    base._atomic_json(args.output_dir / "audit_result.json", result)
    base._atomic_json(args.output_dir / "gpu_execution.json", execution)
    base._atomic_json(args.output_dir / "aggregate.json", aggregate)
    base._atomic_npz(
        args.output_dir / "audit_scores.npz",
        base._pack_paths(audit_runs, audit_scored.by_run, threshold=threshold),
    )
    base._atomic_text(args.output_dir / "REPORT.md", _audit_report(aggregate))
    base._write_completion(
        args.output_dir / "completion.json",
        args.output_dir,
        open_fingerprint,
        str(result["decision"]),
        _audit_artifacts(),
    )
    print(f"hybrid audit decision: {result['decision']}", flush=True)
    return base._verify_completion(
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
    directional_transformer, directional_completion = _resolve_directional_paths(
        project_root,
        args.directional_transformer,
        args.directional_completion,
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
            directional_transformer_path=directional_transformer,
            directional_completion_path=directional_completion,
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
