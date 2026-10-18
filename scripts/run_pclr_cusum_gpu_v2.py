#!/usr/bin/env python3
"""Fail-closed governance runner for severity-stratified PCLR-CUSUM v2.

This module deliberately contains orchestration and validation only.  Numeric
HDF5 access and model math live behind the delayed ``mate_spc_pclr.backend_v2``
interface.  That package must be present before a real protocol is frozen.

The irreversible order is::

    freeze-development -> develop -> freeze-confirmation -> calibrate -> confirm

No command in this file is run on import.  Metadata-only invocations do not
import h5py, torch, or the numeric backend; numeric CLI invocations preload
torch before NumPy to satisfy the target Windows CUDA runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def _numeric_cli_requested(tokens: Sequence[str]) -> bool:
    return bool(tokens) and tokens[0] in {"develop", "calibrate", "confirm"}


# Windows CUDA builds in the target environment must load torch DLLs before
# NumPy/MKL. Metadata-only freeze commands intentionally skip this import.
if _numeric_cli_requested(sys.argv[1:]):
    import torch as _torch_preload  # noqa: F401

import numpy as np


SCHEMA_VERSION = 2
METHOD = "Prefix-Conditioned Loop-aware Signed Residual Density-Ratio CUSUM"
METHOD_SHORT = "PCLR-CUSUM"
DATA_FILE = "TEP_Mode1.h5"
MODE1_SIZE_BYTES = 23_924_710_848
MODE1_MD5 = "c79ecc6d12a5ccb49caffcd8f531feae"
MODE1_SHA256 = "d6b0ed0ef3974ff571afd6717ba0418a5b8eb58a7822c84a05735572d52b3294"
DESIGN_DOCUMENT = "docs/pclr_cusum_successor_v2.md"
IMPLEMENTATION_FILES = (
    DESIGN_DOCUMENT,
    "scripts/run_pclr_cusum_gpu_v2.py",
    "src/mate_spc_pclr/__init__.py",
    "src/mate_spc_pclr/data_v2.py",
    "src/mate_spc_pclr/model.py",
    "src/mate_spc_pclr/backend_v2.py",
)

FAULT_IDS = (3, 9, 15)
ALL_IDS = (1, 3, 9, 15)
CONSUMED_MAGNITUDE = 100
CONSUMED_RUNS = tuple(range(1, 201))
DEVELOPMENT_MAGNITUDE = 50
CALIBRATION_MAGNITUDE = 25
CONFIRMATION_MAGNITUDE = 75
DEVELOPMENT_RUNS = tuple(range(1, 101))
CALIBRATION_RUNS = tuple(range(1, 101))
CONFIRMATION_RUNS = tuple(range(1, 101))
FOLD_COUNT = 5
FOLD_SALT = "pclr-cusum-v2-mag50-balanced-block-folds-2026-09-08"

SEQUENCE_LENGTH = 500
BASELINE_WINDOW = (0, 128)
WARMUP_WINDOW = (128, 160)
MONITORED_WINDOW = (160, 500)
HEALTHY_SOURCE_WINDOW = (100, 600)
FAULT_SOURCE_WINDOW = (440, 940)
FAULT_START = 160
MISS_DELAY = 340
PERSISTENCE = 3
FIRST_ALARM_ELIGIBLE = FAULT_START + PERSISTENCE - 1
LAGS = (1, 2, 4, 8, 16, 32)
LOOP_PAIRS = (
    (41, 1), (42, 2), (43, 0), (44, 3), (45, 4), (46, 9),
    (47, 13), (48, 16), (49, 18), (50, 20), (51, 21),
)

VARX_RIDGES = (1.0e-3, 1.0e-2, 1.0e-1)
CLASSIFIER_RIDGES = (1.0e-3, 1.0e-2, 1.0e-1)
PAGE_DRIFTS = (0.0, 0.25, 0.5)
FINAL_REFIT_STEPS = 500
GRID = tuple(
    {
        "candidate_id": f"v{vi}-c{ci}-d{di}",
        "varx_ridge": vr,
        "classifier_ridge": cr,
        "page_drift": drift,
    }
    for vi, vr in enumerate(VARX_RIDGES)
    for ci, cr in enumerate(CLASSIFIER_RIDGES)
    for di, drift in enumerate(PAGE_DRIFTS)
)

PARITY_ATOL = 1e-10
FLOAT_TOLERANCE = 1e-12
CALIBRATION_RANK = 99
CALIBRATION_EPSILON = 0.10
CALIBRATION_DELTA = 0.01
MIN_UNIQUE_MAXIMA = 95
TOP_TAIL_RANK = 95
TOP_TAIL_TOLERANCE_MULTIPLIER = 100.0
KNOWN_NUMERICAL_LIMITS = (0.0, 3153.75)

DEVELOPMENT_GATES = {
    "minimum_detection_rate_each_fault": 0.85,
    "maximum_median_delay_each_fault": 140.0,
    "maximum_p90_delay_each_fault": 220.0,
    "minimum_detection_rate_each_fold_fault": 0.70,
    "maximum_oof_healthy_alarm_runs_total": 10,
    "maximum_oof_healthy_alarm_runs_each_fold": 4,
}
DEVELOPMENT_THRESHOLD_RANK = 79
DEVELOPMENT_HEALTHY_COUNT_PER_FOLD = 80
DEVELOPMENT_MIN_UNIQUE_MAXIMA = 76
CONFIRMATION_GATES = {
    "maximum_healthy_alarm_runs": 4,
    "minimum_detected_runs_each_fault": 90,
    "minimum_detected_runs_total": 270,
    "maximum_median_delay_each_fault": 120.0,
    "maximum_p90_delay_each_fault": 180.0,
    "miss_delay_samples": MISS_DELAY,
}

DEFAULT_DEVELOPMENT_PROTOCOL = Path("protocols/pclr_cusum_development_v2.json")
DEFAULT_CONFIRMATION_PROTOCOL = Path("protocols/pclr_cusum_confirmation_v2.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/pclr_cusum_successor_v2")


@dataclass(frozen=True)
class BundlePaths:
    protocol: Path
    sidecar: Path
    commit: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while True:
            block = stream.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_record(path: Path, *, include_md5: bool = False) -> dict[str, Any]:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"required file is not regular: {path}")
    if include_md5:
        sha256 = hashlib.sha256()
        md5 = hashlib.md5()
        with path.open("rb") as stream:
            while True:
                block = stream.read(16 * 1024 * 1024)
                if not block:
                    break
                sha256.update(block)
                md5.update(block)
        result: dict[str, Any] = {
            "size_bytes": int(stat.st_size),
            "sha256": sha256.hexdigest(),
            "md5": md5.hexdigest(),
        }
    else:
        result = {"size_bytes": int(stat.st_size), "sha256": _hash_file(path, "sha256")}
    return result


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_batch_size(value: object) -> int:
    """Fail before a phase claim or any HDF5 access on invalid batching."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("batch_size must be a positive integer")
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    safe: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("NPZ field names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"object arrays are forbidden: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"non-finite numeric array is forbidden: {name}")
        safe[name] = array
    with path.open("wb") as stream:
        np.savez_compressed(stream, **safe)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"unsafe or unreadable NPZ: {path}") from error
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError(f"object arrays are forbidden: {path}")
    return arrays


def _require_exact_arrays(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray], *, label: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label} NPZ field set changed")
    for name in expected:
        if actual[name].dtype != expected[name].dtype or not np.array_equal(actual[name], expected[name]):
            raise ValueError(f"{label} NPZ derived field changed: {name}")


def _bundle_paths(protocol_path: Path) -> BundlePaths:
    protocol_path = protocol_path.resolve()
    return BundlePaths(
        protocol=protocol_path,
        sidecar=protocol_path.with_name(protocol_path.name + ".sha256"),
        commit=protocol_path.with_name(protocol_path.name + ".commit.json"),
    )


def _publish_protocol_bundle(protocol_path: Path, payload: Mapping[str, Any]) -> str:
    """Publish protocol, sidecar, then a commit marker; any partial state fails closed."""
    bundle = _bundle_paths(protocol_path)
    bundle.protocol.parent.mkdir(parents=True, exist_ok=True)
    freeze_claim = bundle.protocol.with_name(bundle.protocol.name + ".freeze-claim")
    try:
        claim_descriptor = os.open(freeze_claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError("protocol freeze is already in progress; refusing concurrent publication") from error
    os.close(claim_descriptor)
    protocol_bytes = _canonical_json_bytes(payload)
    protocol_sha = _sha256_bytes(protocol_bytes)
    sidecar_bytes = f"{protocol_sha}  {bundle.protocol.name}\n".encode("utf-8")
    commit = {
        "schema": "pclr_protocol_atomic_commit",
        "schema_version": SCHEMA_VERSION,
        "protocol_name": bundle.protocol.name,
        "protocol_sha256": protocol_sha,
        "sidecar_name": bundle.sidecar.name,
        "sidecar_sha256": _sha256_bytes(sidecar_bytes),
    }
    temp_paths: list[Path] = []
    try:
        if any(path.exists() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
            raise FileExistsError("protocol bundle already exists or is partial; refusing overwrite")
        for destination, content in (
            (bundle.protocol, protocol_bytes),
            (bundle.sidecar, sidecar_bytes),
            (bundle.commit, _canonical_json_bytes(commit)),
        ):
            descriptor, raw_path = tempfile.mkstemp(
                dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
            )
            temp = Path(raw_path)
            temp_paths.append(temp)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temp_paths[0], bundle.protocol)
        temp_paths.pop(0)
        os.replace(temp_paths[0], bundle.sidecar)
        temp_paths.pop(0)
        os.replace(temp_paths[0], bundle.commit)
        temp_paths.pop(0)
    finally:
        for temp in temp_paths:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        try:
            freeze_claim.unlink()
        except FileNotFoundError:
            pass
    return protocol_sha


def _load_protocol(protocol_path: Path, expected_sha256: str, *, schema: str) -> tuple[dict[str, Any], str]:
    expected_sha256 = _require_sha256(expected_sha256.lower(), label="protocol SHA-256")
    bundle = _bundle_paths(protocol_path)
    if not all(path.is_file() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
        raise ValueError("protocol bundle is absent or partially published")
    actual_sha = _hash_file(bundle.protocol)
    if actual_sha != expected_sha256:
        raise ValueError("protocol digest does not match the required command-line digest")
    expected_sidecar = f"{actual_sha}  {bundle.protocol.name}\n"
    if bundle.sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError("protocol SHA-256 sidecar is invalid")
    commit = _strict_json(bundle.commit)
    if commit != {
        "schema": "pclr_protocol_atomic_commit",
        "schema_version": SCHEMA_VERSION,
        "protocol_name": bundle.protocol.name,
        "protocol_sha256": actual_sha,
        "sidecar_name": bundle.sidecar.name,
        "sidecar_sha256": _hash_file(bundle.sidecar),
    }:
        raise ValueError("protocol atomic commit marker is invalid")
    protocol = _strict_json(bundle.protocol)
    if protocol.get("schema") != schema or protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected protocol schema")
    return protocol, actual_sha


def _implementation_records(project_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in IMPLEMENTATION_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"PCLR implementation is incomplete; create {relative} before freezing"
            )
        records[relative] = _hash_record(path)
    return records


def _source_path(data_dir: Path) -> Path:
    return data_dir.resolve() / DATA_FILE


def _source_record(data_dir: Path) -> dict[str, Any]:
    path = _source_path(data_dir)
    record = _hash_record(path, include_md5=True)
    record["name"] = DATA_FILE
    record["absolute_path"] = str(path)
    if (
        record["size_bytes"] != MODE1_SIZE_BYTES
        or record["md5"] != MODE1_MD5
        or record["sha256"] != MODE1_SHA256
    ):
        raise ValueError("source is not the byte-exact official DTU/Figshare TEP Mode 1 archive")
    return record


def _source_metadata_record(data_dir: Path) -> dict[str, Any]:
    """Read topology/labels only; PCLRMode1Archive guarantees no numeric decode."""
    from mate_spc_pclr.data_v2 import PCLRMode1Archive

    summary = PCLRMode1Archive(_source_path(data_dir)).validate_topology()
    return {
        "labels": list(summary.labels),
        "selected_run_count": int(summary.selected_run_count),
        "counts_by_partition": dict(summary.counts_by_partition),
        "counts_by_status": dict(summary.counts_by_status),
        "counts_by_magnitude": {
            str(key): int(value)
            for key, value in summary.counts_by_magnitude.items()
        },
        "unique_seed_count": int(summary.unique_seed_count),
        "seed_sets_pairwise_disjoint": bool(summary.seed_sets_pairwise_disjoint),
        "run_seed_attributes_decoded": True,
        "processdata_or_idv_init_values_decoded": False,
    }


def _fold_assignment() -> dict[str, int]:
    ranked = sorted(
        DEVELOPMENT_RUNS,
        key=lambda run: hashlib.sha256(f"{FOLD_SALT}|{run}".encode("ascii")).hexdigest(),
    )
    assignment = {str(run): index % FOLD_COUNT for index, run in enumerate(ranked)}
    if sorted(assignment) != sorted(str(run) for run in DEVELOPMENT_RUNS):
        raise AssertionError("development fold assignment is incomplete")
    counts = [sum(fold == index for fold in assignment.values()) for index in range(FOLD_COUNT)]
    if counts != [20] * FOLD_COUNT:
        raise AssertionError("development folds must contain exactly 20 balanced blocks each")
    return dict(sorted(assignment.items(), key=lambda item: int(item[0])))


def _grid_payload() -> list[dict[str, Any]]:
    return [dict(candidate) for candidate in GRID]


def _fold_specs() -> list[dict[str, Any]]:
    assignment = _fold_assignment()
    return [
        {
            "fold": fold,
            "training_runs": [run for run in DEVELOPMENT_RUNS if assignment[str(run)] != fold],
            "validation_runs": [run for run in DEVELOPMENT_RUNS if assignment[str(run)] == fold],
        }
        for fold in range(FOLD_COUNT)
    ]


def _fold_fit_contract() -> dict[str, Any]:
    return {
        "input_to_each_fold_fit": "raw_500_by_52_runs_from_training_balanced_blocks_only",
        "fold_local_components_refit": [
            "MAD_scale_floors",
            "VARX_coefficients",
            "feature_standardizers",
            "density_ratio_heads",
        ],
        "validation_transformed_only_after_fold_local_fit": True,
        "full_development_prefit_or_feature_reuse": False,
        "training_group_count_per_fold": 80,
        "validation_group_count_per_fold": 20,
        "conditions_per_group": list(ALL_IDS),
        "grouping_semantics": "balanced_stratification_block_not_common_random_numbers",
    }


def _selection_digest(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _monitor_contract() -> dict[str, Any]:
    return {
        "method": METHOD,
        "sequence_length": SEQUENCE_LENGTH,
        "commissioning_baseline_half_open": list(BASELINE_WINDOW),
        "causal_warmup_half_open": list(WARMUP_WINDOW),
        "monitored_half_open": list(MONITORED_WINDOW),
        "local_fault_start_zero_based": FAULT_START,
        "miss_delay_samples": MISS_DELAY,
        "sample_minutes": 3,
        "lags": list(LAGS),
        "xmv_xmeas_zero_based_pairs": [list(pair) for pair in LOOP_PAIRS],
        "heads": list(FAULT_IDS),
        "transform": "per-run baseline median/MAD scale floor then asinh",
        "supervised_training_objective": {
            "point_evidence": True,
            "top_quartile_bag_evidence": True,
            "ranking": "causal_local_128_160_mean_vs_post_160_500_top_quartile",
            "prefix_feature_rows_per_run": 32,
            "ranking_disabled_if_prefix_features_absent": "fail_closed",
        },
        "positive_hard_clip": None,
        "page_recursion": "one-sided, reset at zero, no finite ceiling",
        "persistence": "three-sample envelope defining alarm episodes",
        "persistence_samples": PERSISTENCE,
        "first_alarm_eligible_local_zero_based": FIRST_ALARM_ELIGIBLE,
        "threshold_comparison": "strictly_greater_than",
    }


def _governance_contract(output_root: Path) -> dict[str, Any]:
    return {
        "bound_output_root": str(output_root.resolve()),
        "consumed_diagnostic_only": {
            "magnitude_percent": CONSUMED_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "runs": [1, 200],
        },
        "development": {
            "magnitude_percent": DEVELOPMENT_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "runs": [1, 100],
        },
        "calibration": {
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "prefault_source_idv": 1,
            "runs": [1, 100],
        },
        "quarantined_no_access_v2": {
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "idvs": list(FAULT_IDS),
            "runs": [1, 100],
        },
        "final_confirmation": {
            "magnitude_percent": CONFIRMATION_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "runs": [1, 100],
        },
        "same_run_number_across_idvs_is_only_a_balanced_stratification_block": True,
        "rng_seed_is_the_independent_simulation_identity": True,
        "fault_total_270_of_300_must_not_be_treated_as_binomial_n_300": True,
        "forbidden_consumed_cells": {
            "magnitude_percent": CONSUMED_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "runs": [1, 200],
        },
        "final_confirmation_is_one_time_for_pass_or_fail": True,
    }


def build_development_protocol(project_root: Path, data_dir: Path, output_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    design_record = _hash_record(project_root / DESIGN_DOCUMENT)
    implementation = _implementation_records(project_root)
    if implementation[DESIGN_DOCUMENT] != design_record:
        raise AssertionError("design document binding mismatch")
    folds = _fold_assignment()
    grid = _grid_payload()
    return {
        "schema": "pclr_development_protocol",
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "v2_frozen_before_any_lower_magnitude_processdata_or_idv_init_numeric_access",
        "method": METHOD_SHORT,
        "design_document": {"path": DESIGN_DOCUMENT, **design_record},
        "implementation_records": implementation,
        "source_record": _source_record(data_dir),
        "source_metadata_record": _source_metadata_record(data_dir),
        "governance": _governance_contract(output_root),
        "monitor": _monitor_contract(),
        "development": {
            "source_ids": list(ALL_IDS),
            "fault_magnitude_percent": DEVELOPMENT_MAGNITUDE,
            "simulation_runs": list(DEVELOPMENT_RUNS),
            "grouping": "run_number_balanced_stratification_block_not_CRN; rng_seed_is_simulation_identity",
            "healthy_source": {
                "idv": 1,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "window_half_open": list(HEALTHY_SOURCE_WINDOW),
            },
            "fault_source": {
                "idvs": list(FAULT_IDS),
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "window_half_open": list(FAULT_SOURCE_WINDOW),
            },
            "fold_count": FOLD_COUNT,
            "fold_salt": FOLD_SALT,
            "run_to_fold": folds,
            "run_to_fold_sha256": _selection_digest(folds),
            "fold_specs": _fold_specs(),
            "fold_fit_contract": _fold_fit_contract(),
            "fold_threshold": {
                "training_healthy_count": DEVELOPMENT_HEALTHY_COUNT_PER_FOLD,
                "ascending_order_statistic_rank": DEVELOPMENT_THRESHOLD_RANK,
                "rule": "79th_ascending_of_80_training_healthy_run_maxima_noninterpolated",
                "comparison": "strictly_greater_than",
                "validation_healthy_used_for_threshold": False,
                "minimum_numerically_distinct_maxima": DEVELOPMENT_MIN_UNIQUE_MAXIMA,
                "q79_strictly_below_q80": True,
                "numeric_parity_atol": PARITY_ATOL,
                "nondegeneracy_tail_rule": "q80_minus_q76_strictly_greater_than_100_times_max(observed_full_path_CPU_CUDA_error,observed_run_maxima_CPU_CUDA_error,float_tolerance,parity_atol)",
                "known_numerical_or_algorithmic_limits": list(KNOWN_NUMERICAL_LIMITS),
            },
            "fixed_grid": grid,
            "fixed_grid_sha256": _selection_digest(grid),
            "selection_order": [
                "maximize_minimum_fault_detection_rate",
                "minimize_worst_fault_nearest_rank_p90_delay",
                "minimize_out_of_fold_healthy_alarm_runs",
                "maximize_classifier_ridge",
                "maximize_varx_ridge",
                "maximize_page_drift",
                "canonical_candidate_id",
            ],
            "gates": DEVELOPMENT_GATES,
            "final_refit_steps": FINAL_REFIT_STEPS,
            "calibration_for_early_stopping": False,
            "cpu_cuda_float64_parity_atol": PARITY_ATOL,
        },
    }


def freeze_development(project_root: Path, data_dir: Path, output_root: Path, protocol_path: Path) -> str:
    _assert_phase_unstarted(output_root.resolve(), "develop")
    return _publish_protocol_bundle(
        protocol_path,
        build_development_protocol(project_root, data_dir, output_root),
    )


def _verify_live_inputs(project_root: Path, data_dir: Path, protocol: Mapping[str, Any]) -> None:
    if _implementation_records(project_root) != protocol["implementation_records"]:
        raise ValueError("implementation or design document changed after protocol freeze")
    if _source_record(data_dir) != protocol["source_record"]:
        raise ValueError("source identity changed after protocol freeze")
    if _source_metadata_record(data_dir) != protocol["source_metadata_record"]:
        raise ValueError("source HDF5 metadata/topology changed after protocol freeze")


def _assert_development_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject a self-consistent digest whose scientific contract is not v2."""
    if protocol.get("method") != METHOD_SHORT or protocol.get("monitor") != _monitor_contract():
        raise ValueError("development method/monitor contract changed")
    design = protocol.get("design_document")
    implementation = protocol.get("implementation_records")
    if not isinstance(design, dict) or not isinstance(implementation, dict):
        raise ValueError("development implementation binding is absent")
    if design.get("path") != DESIGN_DOCUMENT:
        raise ValueError("development design-document path changed")
    if implementation.get(DESIGN_DOCUMENT) != {
        "size_bytes": design.get("size_bytes"), "sha256": design.get("sha256")
    }:
        raise ValueError("development design-document SHA-256 binding changed")
    development = protocol.get("development", {})
    fixed = {
        "source_ids": list(ALL_IDS),
        "fault_magnitude_percent": DEVELOPMENT_MAGNITUDE,
        "simulation_runs": list(DEVELOPMENT_RUNS),
        "grouping": "run_number_balanced_stratification_block_not_CRN; rng_seed_is_simulation_identity",
        "healthy_source": {
            "idv": 1,
            "magnitude_percent": DEVELOPMENT_MAGNITUDE,
            "window_half_open": list(HEALTHY_SOURCE_WINDOW),
        },
        "fault_source": {
            "idvs": list(FAULT_IDS),
            "magnitude_percent": DEVELOPMENT_MAGNITUDE,
            "window_half_open": list(FAULT_SOURCE_WINDOW),
        },
        "fold_count": FOLD_COUNT,
        "fold_salt": FOLD_SALT,
        "run_to_fold": _fold_assignment(),
        "run_to_fold_sha256": _selection_digest(_fold_assignment()),
        "fold_specs": _fold_specs(),
        "fold_fit_contract": _fold_fit_contract(),
        "fold_threshold": {
            "training_healthy_count": DEVELOPMENT_HEALTHY_COUNT_PER_FOLD,
            "ascending_order_statistic_rank": DEVELOPMENT_THRESHOLD_RANK,
            "rule": "79th_ascending_of_80_training_healthy_run_maxima_noninterpolated",
            "comparison": "strictly_greater_than",
            "validation_healthy_used_for_threshold": False,
            "minimum_numerically_distinct_maxima": DEVELOPMENT_MIN_UNIQUE_MAXIMA,
            "q79_strictly_below_q80": True,
            "numeric_parity_atol": PARITY_ATOL,
            "nondegeneracy_tail_rule": "q80_minus_q76_strictly_greater_than_100_times_max(observed_full_path_CPU_CUDA_error,observed_run_maxima_CPU_CUDA_error,float_tolerance,parity_atol)",
            "known_numerical_or_algorithmic_limits": list(KNOWN_NUMERICAL_LIMITS),
        },
        "fixed_grid": _grid_payload(),
        "fixed_grid_sha256": _selection_digest(_grid_payload()),
        "selection_order": [
            "maximize_minimum_fault_detection_rate",
            "minimize_worst_fault_nearest_rank_p90_delay",
            "minimize_out_of_fold_healthy_alarm_runs",
            "maximize_classifier_ridge",
            "maximize_varx_ridge",
            "maximize_page_drift",
            "canonical_candidate_id",
        ],
        "gates": DEVELOPMENT_GATES,
        "final_refit_steps": FINAL_REFIT_STEPS,
        "calibration_for_early_stopping": False,
        "cpu_cuda_float64_parity_atol": PARITY_ATOL,
    }
    if development != fixed:
        raise ValueError("development split/grid/selection contract changed")
    governance = protocol.get("governance", {})
    bound = Path(str(governance.get("bound_output_root", "")))
    if governance != _governance_contract(bound):
        raise ValueError("development partition governance changed")


def _assert_confirmation_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("method") != METHOD_SHORT or protocol.get("monitor") != _monitor_contract():
        raise ValueError("confirmation method/monitor contract changed")
    design = protocol.get("design_document")
    implementation = protocol.get("implementation_records")
    if not isinstance(design, dict) or not isinstance(implementation, dict):
        raise ValueError("confirmation implementation binding is absent")
    if design.get("path") != DESIGN_DOCUMENT or implementation.get(DESIGN_DOCUMENT) != {
        "size_bytes": design.get("size_bytes"),
        "sha256": design.get("sha256"),
    }:
        raise ValueError("confirmation design-document SHA-256 binding changed")
    governance = protocol.get("governance", {})
    bound = Path(str(governance.get("bound_output_root", "")))
    if governance != _governance_contract(bound):
        raise ValueError("confirmation partition governance changed")
    calibration = protocol.get("calibration", {})
    required_calibration = {
        "healthy_source_idv": 1,
        "fault_magnitude_percent": CALIBRATION_MAGNITUDE,
        "simulation_runs": list(CALIBRATION_RUNS),
        "source_window_half_open": list(HEALTHY_SOURCE_WINDOW),
        "healthy_window_excludes_first_affected_global_index": 600,
        "pseudo_onset_local_zero_based": FAULT_START,
        "rank": CALIBRATION_RANK,
        "rule": "99th_ascending_of_100_run_maxima_noninterpolated",
        "comparison": "strictly_greater_than",
        "epsilon": CALIBRATION_EPSILON,
        "delta": CALIBRATION_DELTA,
        "certificate_assumption": "100_distinct_RNG_seed_IDV1_prefault_simulations_from_one_fixed_Mode1_nominal_distribution",
        "minimum_numerically_distinct_maxima": MIN_UNIQUE_MAXIMA,
        "q99_strictly_below_q100": True,
        "top_tail": {"lower_rank": TOP_TAIL_RANK, "upper_rank": 100, "safety_multiplier": TOP_TAIL_TOLERANCE_MULTIPLIER, "reference": "max(observed_full_path_CPU_CUDA_error,float_tolerance,parity_atol)", "comparison": "spread_strictly_greater_than_multiplier_times_reference"},
        "float_tolerance": FLOAT_TOLERANCE,
        "parity_atol": PARITY_ATOL,
        "known_numerical_or_algorithmic_limits": list(KNOWN_NUMERICAL_LIMITS),
    }
    if calibration != required_calibration:
        raise ValueError("calibration order-statistic/non-degeneracy contract changed")
    required_confirmation = {
        "fault_magnitude_percent": CONFIRMATION_MAGNITUDE,
        "healthy": {"source_idv": 1, "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_runs": list(CONFIRMATION_RUNS), "source_window_half_open": list(HEALTHY_SOURCE_WINDOW), "excludes_first_affected_global_index": 600, "pseudo_onset_local_zero_based": FAULT_START},
        "faults": {"source_idvs": list(FAULT_IDS), "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_runs": list(CONFIRMATION_RUNS), "source_window_half_open": list(FAULT_SOURCE_WINDOW)},
        "local_fault_start_zero_based": FAULT_START,
        "detection_requires_new_alarm_episode_start_at_or_after": FAULT_START,
        "gates": CONFIRMATION_GATES,
        "one_time_even_if_decision_is_fail": True,
    }
    if protocol.get("confirmation") != required_confirmation:
        raise ValueError("one-time final confirmation contract changed")
    binding = protocol.get("development_binding", {})
    selected = next(
        (row for row in GRID if row["candidate_id"] == binding.get("selected_candidate_id")),
        None,
    ) if isinstance(binding, dict) else None
    if selected is None or binding.get("selected_params") != {
        key: selected[key] for key in ("varx_ridge", "classifier_ridge", "page_drift")
    }:
        raise ValueError("confirmation selected-model binding is outside the fixed grid")
    required_binding_fields = {
        "protocol_sha256",
        "completion_sha256",
        "result_sha256",
        "selected_candidate_id",
        "selected_params",
        "model_record",
        "model_absolute_path",
    }
    if not isinstance(binding, dict) or set(binding) != required_binding_fields:
        raise ValueError("confirmation development binding fields changed")
    for field in ("protocol_sha256", "completion_sha256", "result_sha256"):
        _require_sha256(str(binding[field]), label=f"development binding {field}")
    model_record = binding.get("model_record")
    if (
        not isinstance(model_record, dict)
        or set(model_record) != {"size_bytes", "sha256"}
        or isinstance(model_record.get("size_bytes"), bool)
        or not isinstance(model_record.get("size_bytes"), int)
        or model_record["size_bytes"] <= 0
    ):
        raise ValueError("confirmation model hash record is invalid")
    _require_sha256(str(model_record["sha256"]), label="confirmation model SHA-256")
    model_path = Path(str(binding.get("model_absolute_path", "")))
    expected_model_path = bound.resolve() / "develop" / "pclr_model.npz"
    if not model_path.is_absolute() or model_path.resolve() != expected_model_path:
        raise ValueError("confirmation model path is not the bound development artifact")


def _assert_bound_output(output_root: Path, protocol: Mapping[str, Any]) -> None:
    expected = protocol["governance"]["bound_output_root"]
    if str(output_root.resolve()) != expected:
        raise ValueError("output root differs from the frozen one-time destination")


def _assert_no_partial(output_root: Path, phase: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    prefix = f".{phase}.inprogress-"
    leftovers = [path for path in output_root.iterdir() if path.name.startswith(prefix)]
    if leftovers:
        raise RuntimeError(f"partial {phase} staging exists; fail-closed: {leftovers[0]}")


def _assert_phase_unstarted(output_root: Path, phase: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_no_partial(output_root, phase)
    if (output_root / phase).exists() or (output_root / f".{phase}.claim").exists():
        raise RuntimeError(f"{phase} has already started or completed")


def _claim_phase(output_root: Path, phase: str) -> tuple[Path, Path]:
    """Atomically claim the sole execution before any source numeric access."""
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / phase
    if target.exists():
        raise RuntimeError(f"{phase} target already exists and was not accepted as complete")
    claim = output_root / f".{phase}.claim"
    try:
        claim.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"{phase} has already been claimed; concurrent/repeated access is forbidden") from error
    _assert_no_partial(output_root, phase)
    # Interruption leaves both claim and staging fail-closed. Success removes the
    # claim only after the completed target has been atomically published.
    staging = Path(tempfile.mkdtemp(dir=output_root, prefix=f".{phase}.inprogress-"))
    return target, staging


def _assert_completed_clean(output_root: Path, phase: str) -> None:
    _assert_no_partial(output_root, phase)
    claim = output_root / f".{phase}.claim"
    if claim.exists():
        raise RuntimeError(f"completed {phase} coexists with an unfinished execution claim")


def _artifact_records(directory: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {name: _hash_record(directory / name) for name in names}


def _finalize_phase(
    target: Path,
    staging: Path,
    *,
    phase: str,
    protocol_sha256: str,
    artifact_names: Sequence[str],
    semantic_sha256: str,
) -> dict[str, Any]:
    artifacts = _artifact_records(staging, artifact_names)
    completion = {
        "schema": "pclr_phase_completion",
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "semantic_replay_sha256": semantic_sha256,
        "artifacts": artifacts,
    }
    _write_json(staging / "completion.json", completion)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing {phase} target")
    os.replace(staging, target)
    claim = target.parent / f".{phase}.claim"
    try:
        claim.rmdir()
    except OSError as error:
        raise RuntimeError(f"{phase} target published but execution claim could not be cleared; fail-closed") from error
    return completion


def _validate_parity(
    parity: Mapping[str, Any],
    *,
    require_alarm: bool = True,
    require_attribution: bool = False,
    enforce: bool = True,
) -> dict[str, Any]:
    if not isinstance(parity, Mapping):
        raise ValueError("CPU/CUDA parity record must be a mapping")
    for field in ("eligibility_exact", "alarm_exact", "attribution_exact"):
        if not isinstance(parity.get(field), (bool, np.bool_)):
            raise ValueError(f"CPU/CUDA parity field must be boolean: {field}")
    max_error = float(parity.get("max_abs_score_error", math.inf))
    if not math.isfinite(max_error) or max_error < 0.0:
        raise ValueError("CPU/CUDA numeric-path error is absent or invalid")
    result = {
        "eligibility_exact": bool(parity["eligibility_exact"]),
        "alarm_exact": bool(parity["alarm_exact"]),
        "max_abs_score_error": max_error,
        "attribution_exact": bool(parity["attribution_exact"]),
    }
    passed = (
        result["eligibility_exact"]
        and (result["alarm_exact"] or not require_alarm)
        and (result["attribution_exact"] or not require_attribution)
        and max_error <= PARITY_ATOL
    )
    if enforce and not passed:
        raise ValueError("CPU/CUDA parity failed the frozen exactness/tolerance contract")
    return result


def _nearest_rank(values: Sequence[int | float], probability: float) -> float:
    if not values:
        raise ValueError("nearest-rank statistic requires values")
    ordered = sorted(float(value) for value in values)
    rank = int(math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _normalize_candidate(candidate: Mapping[str, Any], fold_map: Mapping[str, int]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", ""))
    frozen = next((row for row in GRID if row["candidate_id"] == candidate_id), None)
    if frozen is None:
        raise ValueError(f"candidate outside frozen grid: {candidate_id}")
    if dict(candidate.get("params", {})) != {
        key: frozen[key] for key in ("varx_ridge", "classifier_ridge", "page_drift")
    }:
        raise ValueError(f"candidate parameters changed: {candidate_id}")
    expected_fold_fits = [
        {
            **spec,
            "candidate_id": candidate_id,
            "fit_from_raw_training_groups_only": True,
            "fold_local_components_refit": _fold_fit_contract()["fold_local_components_refit"],
        }
        for spec in _fold_specs()
    ]
    if candidate.get("fold_fits") != expected_fold_fits:
        raise ValueError(f"candidate {candidate_id} lacks fold-local full-pipeline fit lineage")
    raw_fold_calibrations = candidate.get("fold_calibrations")
    if not isinstance(raw_fold_calibrations, list) or len(raw_fold_calibrations) != FOLD_COUNT:
        raise ValueError(f"candidate {candidate_id} lacks five fold threshold records")
    fold_calibrations: list[dict[str, Any]] = []
    for spec in _fold_specs():
        matches = [row for row in raw_fold_calibrations if int(row.get("fold", -1)) == spec["fold"]]
        if len(matches) != 1:
            raise ValueError("fold threshold records must identify every fold exactly once")
        replay = _development_fold_threshold_summary(matches[0], spec)
        if replay != matches[0]:
            raise ValueError("fold threshold semantic replay failed")
        fold_calibrations.append(replay)
    healthy_rows = candidate.get("healthy_records")
    if not isinstance(healthy_rows, list) or len(healthy_rows) != len(DEVELOPMENT_RUNS):
        raise ValueError(f"candidate {candidate_id} must contain 100 OOF healthy records")
    normalized_healthy: list[dict[str, Any]] = []
    healthy_seen: set[int] = set()
    for row in healthy_rows:
        run = int(row["simulation_run"])
        fold = int(row["fold"])
        alarmed_raw = row["alarmed"]
        if not isinstance(alarmed_raw, (bool, np.bool_)):
            raise ValueError("OOF healthy alarm flag must be boolean")
        if run not in DEVELOPMENT_RUNS or fold != int(fold_map[str(run)]) or run in healthy_seen:
            raise ValueError("OOF healthy record violates the frozen grouped split")
        healthy_seen.add(run)
        normalized_healthy.append({"simulation_run": run, "fold": fold, "alarmed": bool(alarmed_raw)})
    rows = candidate.get("records")
    if not isinstance(rows, list) or len(rows) != len(FAULT_IDS) * len(DEVELOPMENT_RUNS):
        raise ValueError(f"candidate {candidate_id} must contain 300 held-out fault records")
    seen: set[tuple[int, int]] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        fault = int(row["fault_id"])
        run = int(row["simulation_run"])
        fold = int(row["fold"])
        if not isinstance(row["detected"], (bool, np.bool_)):
            raise ValueError("fault detection flag must be boolean")
        detected = bool(row["detected"])
        delay = int(row["restricted_delay_samples"])
        if fault not in FAULT_IDS or run not in DEVELOPMENT_RUNS:
            raise ValueError("development record accesses an unauthorized fault or run")
        if fold != int(fold_map[str(run)]):
            raise ValueError("same-numbered IDVs did not retain their frozen group fold")
        if (fault, run) in seen:
            raise ValueError("duplicate development held-out record")
        if not (0 <= delay <= MISS_DELAY) or (not detected and delay != MISS_DELAY):
            raise ValueError("invalid restricted development delay")
        seen.add((fault, run))
        normalized_rows.append({
            "fault_id": fault,
            "simulation_run": run,
            "fold": fold,
            "detected": detected,
            "restricted_delay_samples": delay,
        })
    parity = _validate_parity(
        candidate.get("parity", {}),
        require_alarm=True,
        require_attribution=True,
        enforce=False,
    )
    per_fault: dict[str, Any] = {}
    fold_fault: dict[str, Any] = {}
    for fault in FAULT_IDS:
        fault_rows = [row for row in normalized_rows if row["fault_id"] == fault]
        delays = [row["restricted_delay_samples"] for row in fault_rows]
        per_fault[str(fault)] = {
            "detection_rate": sum(row["detected"] for row in fault_rows) / 100.0,
            "median_delay_samples": float(statistics.median(delays)),
            "p90_delay_samples_nearest_rank": _nearest_rank(delays, 0.90),
        }
        for fold in range(FOLD_COUNT):
            subset = [row for row in fault_rows if row["fold"] == fold]
            if len(subset) != 20:
                raise ValueError("each fold/fault must contain exactly 20 held-out runs")
            fold_fault[f"fold-{fold}/fault-{fault}"] = {
                "detection_rate": sum(row["detected"] for row in subset) / 20.0
            }
    healthy_by_fold = {
        str(fold): sum(row["alarmed"] for row in normalized_healthy if row["fold"] == fold)
        for fold in range(FOLD_COUNT)
    }
    if any(sum(row["fold"] == fold for row in normalized_healthy) != 20 for fold in range(FOLD_COUNT)):
        raise ValueError("each fold must contain exactly 20 OOF healthy runs")
    healthy_total = sum(row["alarmed"] for row in normalized_healthy)
    preselection_gates = {
        "each_fault_detection_rate_at_least_0_85": all(
            row["detection_rate"] >= DEVELOPMENT_GATES["minimum_detection_rate_each_fault"]
            for row in per_fault.values()
        ),
        "each_fault_median_delay_at_most_140": all(
            row["median_delay_samples"] <= DEVELOPMENT_GATES["maximum_median_delay_each_fault"]
            for row in per_fault.values()
        ),
        "each_fault_p90_delay_at_most_220": all(
            row["p90_delay_samples_nearest_rank"] <= DEVELOPMENT_GATES["maximum_p90_delay_each_fault"]
            for row in per_fault.values()
        ),
        "each_fold_fault_detection_rate_at_least_0_70": all(
            row["detection_rate"] >= DEVELOPMENT_GATES["minimum_detection_rate_each_fold_fault"]
            for row in fold_fault.values()
        ),
        "all_fold_q79_thresholds_nondegenerate": all(row["decision"] == "PASS" for row in fold_calibrations),
        "oof_healthy_alarm_runs_total_at_most_10": healthy_total <= DEVELOPMENT_GATES["maximum_oof_healthy_alarm_runs_total"],
        "oof_healthy_alarm_runs_each_fold_at_most_4": all(value <= DEVELOPMENT_GATES["maximum_oof_healthy_alarm_runs_each_fold"] for value in healthy_by_fold.values()),
        "full_cuda_numpy_float64_parity": (
            parity["eligibility_exact"]
            and parity["alarm_exact"]
            and parity["attribution_exact"]
            and parity["max_abs_score_error"] <= PARITY_ATOL
        ),
    }
    summary = {
        "candidate_id": candidate_id,
        "params": dict(candidate["params"]),
        "records": sorted(normalized_rows, key=lambda row: (row["fault_id"], row["simulation_run"])),
        "per_fault": per_fault,
        "fold_fault": fold_fault,
        "healthy_records": sorted(normalized_healthy, key=lambda row: row["simulation_run"]),
        "oof_healthy_alarm_runs": healthy_total,
        "oof_healthy_alarm_runs_by_fold": healthy_by_fold,
        "fold_fits": expected_fold_fits,
        "fold_calibrations": fold_calibrations,
        "preselection_gates": preselection_gates,
        "selection_eligible": all(preselection_gates.values()),
        "parity": parity,
    }
    summary["minimum_fault_detection_rate"] = min(
        row["detection_rate"] for row in per_fault.values()
    )
    summary["worst_fault_p90_delay_samples"] = max(
        row["p90_delay_samples_nearest_rank"] for row in per_fault.values()
    )
    return summary


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    params = candidate["params"]
    return (
        -float(candidate["minimum_fault_detection_rate"]),
        float(candidate["worst_fault_p90_delay_samples"]),
        int(candidate["oof_healthy_alarm_runs"]),
        -float(params["classifier_ridge"]),
        -float(params["varx_ridge"]),
        -float(params["page_drift"]),
        str(candidate["candidate_id"]),
    )


def _select_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [candidate for candidate in candidates if candidate.get("selection_eligible") is True]
    if not eligible:
        raise RuntimeError("no candidate passed every frozen development gate; development is a terminal no-go")
    return min(eligible, key=_candidate_key)


def _development_decision(candidate: Mapping[str, Any]) -> dict[str, Any]:
    per_fault = candidate["per_fault"]
    fold_fault = candidate["fold_fault"]
    gates = {
        "each_fault_detection_rate_at_least_0_85": all(
            row["detection_rate"] >= DEVELOPMENT_GATES["minimum_detection_rate_each_fault"]
            for row in per_fault.values()
        ),
        "each_fault_median_delay_at_most_140": all(
            row["median_delay_samples"] <= DEVELOPMENT_GATES["maximum_median_delay_each_fault"]
            for row in per_fault.values()
        ),
        "each_fault_p90_delay_at_most_220": all(
            row["p90_delay_samples_nearest_rank"] <= DEVELOPMENT_GATES["maximum_p90_delay_each_fault"]
            for row in per_fault.values()
        ),
        "each_fold_fault_detection_rate_at_least_0_70": all(
            row["detection_rate"] >= DEVELOPMENT_GATES["minimum_detection_rate_each_fold_fault"]
            for row in fold_fault.values()
        ),
        "all_fold_q79_thresholds_nondegenerate": bool(candidate["preselection_gates"]["all_fold_q79_thresholds_nondegenerate"]),
        "oof_healthy_alarm_runs_total_at_most_10": bool(candidate["preselection_gates"]["oof_healthy_alarm_runs_total_at_most_10"]),
        "oof_healthy_alarm_runs_each_fold_at_most_4": bool(candidate["preselection_gates"]["oof_healthy_alarm_runs_each_fold_at_most_4"]),
        "full_cuda_numpy_float64_parity": bool(candidate["preselection_gates"]["full_cuda_numpy_float64_parity"]),
    }
    return {"decision": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def _development_no_go_result(
    protocol_sha: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible_count = sum(
        candidate.get("selection_eligible") is True for candidate in candidates
    )
    if eligible_count != 0:
        raise ValueError("DEVELOPMENT_NO_GO requires exactly zero eligible candidates")
    return {
        "schema": "pclr_development_result",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "decision": "DEVELOPMENT_NO_GO",
        "development_magnitude_percent": DEVELOPMENT_MAGNITUDE,
        "reason": "zero_selection_eligible_candidates",
        "candidate_count": len(candidates),
        "selection_eligible_candidate_count": 0,
        "selected_model_generated": False,
        "confirmation_freeze_permitted": False,
        "rerun_or_retuning_under_same_v2_protocol_forbidden": True,
    }


def _selected_development_result(
    protocol_sha: str,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    decision = _development_decision(selected)
    return {
        "schema": "pclr_development_result",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "selected_candidate_id": selected["candidate_id"],
        "development_magnitude_percent": DEVELOPMENT_MAGNITUDE,
        "selected_params": selected["params"],
        "selection_key": list(_candidate_key(selected)),
        "selected_summary": {
            key: selected[key]
            for key in (
                "per_fault",
                "fold_fault",
                "fold_calibrations",
                "oof_healthy_alarm_runs",
                "oof_healthy_alarm_runs_by_fold",
                "preselection_gates",
                "parity",
                "minimum_fault_detection_rate",
                "worst_fault_p90_delay_samples",
            )
        },
        **decision,
        "final_refit_steps": FINAL_REFIT_STEPS,
        "selected_model_generated": True,
        "confirmation_freeze_permitted": decision["decision"] == "PASS",
    }


def _development_access_plan(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for run in DEVELOPMENT_RUNS:
        plan.append({"idv": 1, "magnitude_percent": DEVELOPMENT_MAGNITUDE, "simulation_run": run, "window_half_open": list(HEALTHY_SOURCE_WINDOW), "role": "healthy"})
        for fault in FAULT_IDS:
            plan.append({"idv": fault, "magnitude_percent": DEVELOPMENT_MAGNITUDE, "simulation_run": run, "window_half_open": list(FAULT_SOURCE_WINDOW), "role": f"fault-{fault}"})
    if protocol["development"]["run_to_fold"] != _fold_assignment():
        raise ValueError("development fold contract changed")
    return plan


def _load_backend() -> ModuleType:
    try:
        return importlib.import_module("mate_spc_pclr.backend_v2")
    except ImportError as error:
        raise RuntimeError(
            "numeric PCLR v2 backend is not implemented; expected src/mate_spc_pclr/backend_v2.py"
        ) from error


def _backend_function(backend: ModuleType, name: str) -> Any:
    function = getattr(backend, name, None)
    if not callable(function):
        raise RuntimeError(f"PCLR backend is missing required callable: {name}")
    return function


def _development_fold_threshold_summary(
    outcome: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    cpu = np.asarray(outcome.get("training_healthy_cpu_run_maxima"))
    cuda = np.asarray(outcome.get("training_healthy_cuda_run_maxima"))
    expected_shape = (DEVELOPMENT_HEALTHY_COUNT_PER_FOLD,)
    if cpu.shape != expected_shape or cuda.shape != expected_shape:
        raise ValueError("fold threshold requires exactly 80 training-healthy maxima per path")
    if cpu.dtype != np.dtype(np.float64) or cuda.dtype != np.dtype(np.float64):
        raise ValueError("fold training-healthy maxima must be float64")
    if not np.isfinite(cpu).all() or not np.isfinite(cuda).all():
        raise ValueError("fold training-healthy maxima must be finite")
    ordered = np.sort(cpu)
    q79 = float(ordered[DEVELOPMENT_THRESHOLD_RANK - 1])
    q80 = float(ordered[-1])
    q76 = float(ordered[DEVELOPMENT_MIN_UNIQUE_MAXIMA - 1])
    supplied = outcome.get("operational_threshold_q79")
    if isinstance(supplied, bool) or float(supplied) != q79:
        raise ValueError("backend did not use the exact noninterpolated CPU q79 threshold")
    numeric_error = float(np.max(np.abs(cpu - cuda)))
    parity_record = outcome.get("parity")
    if isinstance(parity_record, Mapping):
        full_path_error = float(parity_record.get("max_abs_score_error", math.inf))
    else:
        full_path_error = float(
            outcome.get("max_abs_full_path_cpu_cuda_error", math.inf)
        )
    if not math.isfinite(full_path_error) or full_path_error < 0.0:
        raise ValueError("fold full-path CPU/CUDA error is absent or invalid")
    distinct = _distinct_count(cpu, FLOAT_TOLERANCE)
    reference = max(numeric_error, full_path_error, FLOAT_TOLERANCE, PARITY_ATOL)
    spread = q80 - q76
    known_limit_hit = any(
        bool(np.any(np.isclose(cpu, limit, rtol=0.0, atol=FLOAT_TOLERANCE)))
        for limit in KNOWN_NUMERICAL_LIMITS
    )
    gates = {
        "cpu_cuda_maxima_error_at_most_1e_10": numeric_error <= PARITY_ATOL,
        "at_least_76_numerically_distinct_maxima": distinct >= DEVELOPMENT_MIN_UNIQUE_MAXIMA,
        "q79_strictly_below_q80": q79 < q80,
        "q80_minus_q76_above_100_times_numeric_reference": spread > TOP_TAIL_TOLERANCE_MULTIPLIER * reference,
        "no_maximum_equals_known_numerical_or_algorithmic_limit": not known_limit_hit,
        "no_top_finite_tie": int(np.count_nonzero(np.isclose(cpu, q80, rtol=0.0, atol=FLOAT_TOLERANCE))) == 1,
    }
    return {
        "fold": int(spec["fold"]),
        "training_runs": list(spec["training_runs"]),
        "validation_runs_excluded_from_threshold": list(spec["validation_runs"]),
        "training_healthy_cpu_run_maxima": cpu.tolist(),
        "training_healthy_cuda_run_maxima": cuda.tolist(),
        "operational_threshold_q79": q79,
        "q80": q80,
        "q76": q76,
        "numerically_distinct_maxima": distinct,
        "max_abs_cpu_cuda_maxima_error": numeric_error,
        "max_abs_full_path_cpu_cuda_error": full_path_error,
        "tail_numeric_reference": reference,
        "tail_spread_q80_minus_q76": spread,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }


def _evaluate_grid_fold_safe(
    backend: ModuleType,
    groups: Any,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Leakage-free factorized CV: 15 feature fits, 45 head fits, 135 scores."""
    if not isinstance(groups, Mapping) or set(groups) != set(DEVELOPMENT_RUNS):
        raise ValueError("development backend must return a balanced-block mapping for runs 1--100")
    fit_feature = _backend_function(backend, "fit_fold_feature_model")
    transform = _backend_function(backend, "transform_fold_groups")
    fit_heads = _backend_function(backend, "fit_fold_heads")
    score = _backend_function(backend, "score_fold_candidate")
    accumulated: dict[str, dict[str, Any]] = {}
    for frozen in GRID:
        accumulated[str(frozen["candidate_id"])] = {
            "candidate_id": frozen["candidate_id"],
            "params": {key: frozen[key] for key in ("varx_ridge", "classifier_ridge", "page_drift")},
            "records": [],
            "healthy_records": [],
            "fold_calibrations": [],
            "parities": [],
            "fold_fits": [],
        }
    for spec in _fold_specs():
        training = {run: groups[run] for run in spec["training_runs"]}
        validation = {run: groups[run] for run in spec["validation_runs"]}
        if set(training) & set(validation) or set(training) | set(validation) != set(DEVELOPMENT_RUNS):
            raise AssertionError("internal grouped-fold partition is invalid")
        for varx_ridge in VARX_RIDGES:
            # This fit receives no validation object and cannot prefit on all 100 groups.
            feature_model = fit_feature(
                train_groups=training,
                varx_ridge=varx_ridge,
                fold=int(spec["fold"]),
                device="cuda",
                dtype="float64",
            )
            train_features = transform(
                feature_model=feature_model,
                groups=training,
                role="training",
                fold=int(spec["fold"]),
                device="cuda",
                dtype="float64",
                batch_size=int(batch_size),
            )
            validation_features = transform(
                feature_model=feature_model,
                groups=validation,
                role="validation",
                fold=int(spec["fold"]),
                device="cuda",
                dtype="float64",
                batch_size=int(batch_size),
            )
            for classifier_ridge in CLASSIFIER_RIDGES:
                # Heads receive training features only; validation labels cannot leak.
                heads = fit_heads(
                    train_features=train_features,
                    classifier_ridge=classifier_ridge,
                    fold=int(spec["fold"]),
                    device="cuda",
                    dtype="float64",
                )
                for page_drift in PAGE_DRIFTS:
                    frozen = next(
                        row for row in GRID
                        if row["varx_ridge"] == varx_ridge
                        and row["classifier_ridge"] == classifier_ridge
                        and row["page_drift"] == page_drift
                    )
                    outcome = score(
                        heads=heads,
                        validation_features=validation_features,
                        page_drift=page_drift,
                        development_threshold_rank=DEVELOPMENT_THRESHOLD_RANK,
                        threshold_source="80_training_healthy_run_maxima_only",
                        threshold_rule="noninterpolated_ascending_order_statistic",
                        threshold_comparison="strictly_greater_than",
                        fold=int(spec["fold"]),
                        device="cuda",
                        dtype="float64",
                    )
                    if not isinstance(outcome, Mapping) or not isinstance(outcome.get("records"), list) or not isinstance(outcome.get("healthy_records"), list):
                        raise ValueError("score_fold_candidate must return held-out fault/healthy records and parity")
                    if len(outcome["records"]) != len(FAULT_IDS) * 20:
                        raise ValueError("each candidate/fold must return 60 held-out fault records")
                    if len(outcome["healthy_records"]) != 20:
                        raise ValueError("each candidate/fold must return 20 held-out healthy records")
                    slot = accumulated[str(frozen["candidate_id"])]
                    slot["records"].extend(outcome["records"])
                    slot["healthy_records"].extend(outcome["healthy_records"])
                    slot["fold_calibrations"].append(
                        _development_fold_threshold_summary(outcome, spec)
                    )
                    slot["parities"].append(
                        _validate_parity(
                            outcome.get("parity", {}),
                            require_alarm=True,
                            require_attribution=True,
                            enforce=False,
                        )
                    )
                    slot["fold_fits"].append({
                        **spec,
                        "candidate_id": frozen["candidate_id"],
                        "fit_from_raw_training_groups_only": True,
                        "fold_local_components_refit": _fold_fit_contract()["fold_local_components_refit"],
                    })
    candidates: list[dict[str, Any]] = []
    for frozen in GRID:
        slot = accumulated[str(frozen["candidate_id"])]
        parities = slot.pop("parities")
        slot["fold_fits"] = sorted(slot["fold_fits"], key=lambda row: row["fold"])
        slot["fold_calibrations"] = sorted(slot["fold_calibrations"], key=lambda row: row["fold"])
        slot["parity"] = {
            "eligibility_exact": all(row["eligibility_exact"] for row in parities),
            "alarm_exact": all(row["alarm_exact"] for row in parities),
            "max_abs_score_error": max(row["max_abs_score_error"] for row in parities),
            "attribution_exact": all(row["attribution_exact"] for row in parities),
        }
        candidates.append(slot)
    return candidates


def _cuda_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for governed CUDA execution") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; governed numeric phases require CUDA")
    tensor = torch.tensor([2.0, 3.0], dtype=torch.float64, device="cuda")
    probe = float((tensor[0] + tensor[1]).cpu().item())
    if probe != 5.0:
        raise RuntimeError("CUDA float64 probe failed")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "requested_device": "cuda",
        "numeric_dtype": "float64",
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "device_name": str(properties.name),
        "float64_probe": probe,
    }


def _base_intent(phase: str, protocol_sha256: str, access_plan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "pclr_numeric_access_intent",
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "access_plan": list(access_plan),
        "numeric_read_permitted_only_after_this_file_and_phase_manifest": True,
        "consumed_magnitude100_idv1_3_9_15_runs1_200_forbidden": True,
    }


def _phase_manifest(
    phase: str,
    protocol_sha256: str,
    intent_sha256: str,
    protocol: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    model_record: Mapping[str, Any] | None = None,
    calibration_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_runtime(runtime)
    value: dict[str, Any] = {
        "schema": "pclr_phase_manifest",
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "access_intent_sha256": intent_sha256,
        "source_record": protocol["source_record"],
        "source_metadata_record": protocol["source_metadata_record"],
        "implementation_records": protocol["implementation_records"],
        "runtime": dict(runtime),
    }
    if model_record is not None:
        value["model_record"] = dict(model_record)
    if calibration_binding is not None:
        value["calibration_binding"] = dict(calibration_binding)
    return value


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    required = {
        "requested_device",
        "numeric_dtype",
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "float64_probe",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required:
        raise ValueError("runtime record fields differ from the frozen CUDA contract")
    if runtime["requested_device"] != "cuda" or runtime["numeric_dtype"] != "float64":
        raise ValueError("governed numeric phases require CUDA float64")
    if not all(
        isinstance(runtime[field], str) and bool(runtime[field])
        for field in ("torch_version", "torch_cuda_version", "device_name")
    ):
        raise ValueError("runtime version/device identity is incomplete")
    if isinstance(runtime["float64_probe"], bool) or float(runtime["float64_probe"]) != 5.0:
        raise ValueError("runtime CUDA float64 probe is invalid")


def _validate_model_arrays(value: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("backend must return nonempty numeric final-model arrays")
    arrays: dict[str, np.ndarray] = {}
    for name, raw in value.items():
        array = np.asarray(raw)
        if array.dtype.kind not in "biufcUS" or array.dtype.kind == "O":
            raise ValueError(f"model field must be numeric or fixed-width text: {name}")
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"model field is non-finite: {name}")
        arrays[str(name)] = array
    return arrays


def _validate_pclr_model_file(
    path: Path,
    selected_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the core archive schema and the frozen final-refit semantics."""

    expected_params = {
        "varx_ridge": float(selected_params["varx_ridge"]),
        "classifier_ridge": float(selected_params["classifier_ridge"]),
        "page_drift": float(selected_params["page_drift"]),
    }
    if expected_params not in [
        {key: row[key] for key in expected_params}
        for row in GRID
    ]:
        raise ValueError("final model hyperparameters are outside the frozen grid")

    from mate_spc_pclr.model import PCLRConfig, load_pclr_model

    model, metadata = load_pclr_model(path)
    if model.heads.threshold is not None:
        raise ValueError("development model must remain uncalibrated")
    if tuple(model.heads.fault_ids) != FAULT_IDS:
        raise ValueError("development model fault heads changed")
    expected_feature_config = PCLRConfig(
        var_ridge_alpha=expected_params["varx_ridge"]
    )
    if model.feature_model.config != expected_feature_config:
        raise ValueError(
            "development model feature configuration differs from the frozen contract"
        )
    expected_kappa = np.full(len(FAULT_IDS), expected_params["page_drift"], dtype=np.float64)
    if not np.array_equal(model.heads.kappa, expected_kappa):
        raise ValueError("development model Page drift differs from selection")

    required_metadata = {
        "backend": "mate_spc_pclr.backend_v2",
        "fit_all_components_from_scratch": True,
        "development_group_count": len(DEVELOPMENT_RUNS),
        "development_run_count": len(DEVELOPMENT_RUNS) * len(ALL_IDS),
        "development_runs": [min(DEVELOPMENT_RUNS), max(DEVELOPMENT_RUNS)],
        "development_magnitude_percent": DEVELOPMENT_MAGNITUDE,
        "grouping_semantics": "balanced_stratification_block_not_crn",
        "selected_params": expected_params,
        "final_refit_steps": FINAL_REFIT_STEPS,
        "steps": FINAL_REFIT_STEPS,
    }
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in required_metadata.items()
    ):
        raise ValueError("development model refit metadata differs from the frozen contract")
    head_fit = metadata.get("head_fit")
    required_head_fit = {
        "requested_device": "cuda",
        "dtype": "float64",
        "epochs": FINAL_REFIT_STEPS,
        "ridge_lambda": expected_params["classifier_ridge"],
        "prefix_rank_contract": "causal_local_128_160_mean_vs_post_160_500_top_quartile",
    }
    if not isinstance(head_fit, dict) or any(
        head_fit.get(key) != value for key, value in required_head_fit.items()
    ):
        raise ValueError("development model head-fit metadata differs from selection")
    prefix_weight = head_fit.get("prefix_rank_weight")
    if isinstance(prefix_weight, bool) or not isinstance(prefix_weight, (int, float)):
        raise ValueError("development model prefix-ranking weight is invalid")
    if not math.isfinite(float(prefix_weight)) or float(prefix_weight) <= 0.0:
        raise ValueError("development model must use the frozen causal prefix ranking")
    return metadata


def run_development(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    batch_size = _require_positive_batch_size(batch_size)
    protocol, protocol_sha = _load_protocol(
        protocol_path, protocol_sha256, schema="pclr_development_protocol"
    )
    _assert_development_protocol(protocol)
    _assert_bound_output(output_root, protocol)
    target = output_root.resolve() / "develop"
    if target.exists():
        _assert_completed_clean(output_root.resolve(), "develop")
        return _verify_development_bundle(target, protocol, protocol_sha)
    target, staging = _claim_phase(output_root.resolve(), "develop")
    _verify_live_inputs(project_root.resolve(), data_dir, protocol)
    runtime = _cuda_preflight()
    access_plan = _development_access_plan(protocol)
    intent = _base_intent("develop", protocol_sha, access_plan)
    _write_json(staging / "access_intent.json", intent)
    manifest = _phase_manifest(
        "develop", protocol_sha, _hash_file(staging / "access_intent.json"), protocol, runtime
    )
    _write_json(staging / "phase_manifest.json", manifest)

    # The first operation capable of decoding source numeric values occurs here.
    backend = _load_backend()
    data = _backend_function(backend, "load_numeric_runs")(
        _source_path(data_dir), access_plan
    )
    raw_candidates = _evaluate_grid_fold_safe(backend, data, batch_size=batch_size)
    if not isinstance(raw_candidates, list) or len(raw_candidates) != len(GRID):
        raise ValueError("backend must evaluate every and only frozen grid candidate")
    normalized = [_normalize_candidate(candidate, _fold_assignment()) for candidate in raw_candidates]
    if {row["candidate_id"] for row in normalized} != {row["candidate_id"] for row in GRID}:
        raise ValueError("backend candidate IDs do not equal the frozen grid")
    _write_json(staging / "development_cv.json", {"candidates": normalized})
    if not any(candidate["selection_eligible"] is True for candidate in normalized):
        result = _development_no_go_result(protocol_sha, normalized)
        _write_json(staging / "development_result.json", result)
        semantic = _selection_digest(
            {"intent": intent, "manifest": manifest, "result": result}
        )
        _finalize_phase(
            target,
            staging,
            phase="develop",
            protocol_sha256=protocol_sha,
            artifact_names=(
                "access_intent.json",
                "phase_manifest.json",
                "development_cv.json",
                "development_result.json",
            ),
            semantic_sha256=semantic,
        )
        return _verify_development_bundle(target, protocol, protocol_sha)
    selected = _select_candidate(normalized)
    model_raw = _backend_function(backend, "refit_selected_model")(
        raw_development_groups=data,
        selected_params=selected["params"],
        steps=FINAL_REFIT_STEPS,
        fit_all_components_from_scratch=True,
        device="cuda",
        dtype="float64",
    )
    model_arrays = _validate_model_arrays(model_raw)
    result = _selected_development_result(protocol_sha, selected)
    _write_json(staging / "development_result.json", result)
    model_path = staging / "pclr_model.npz"
    _write_npz(model_path, model_arrays)
    _validate_pclr_model_file(model_path, selected["params"])
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    _finalize_phase(
        target,
        staging,
        phase="develop",
        protocol_sha256=protocol_sha,
        artifact_names=("access_intent.json", "phase_manifest.json", "development_cv.json", "development_result.json", "pclr_model.npz"),
        semantic_sha256=semantic,
    )
    return _verify_development_bundle(target, protocol, protocol_sha)


def _verify_artifacts(target: Path, completion: Mapping[str, Any], expected_names: Sequence[str]) -> None:
    actual_entries = {path.name for path in target.iterdir()}
    expected_entries = set(expected_names) | {"completion.json"}
    if actual_entries != expected_entries or any(not path.is_file() for path in target.iterdir()):
        raise ValueError("phase directory contains an untracked or missing artifact")
    if completion.get("artifacts") != _artifact_records(target, expected_names):
        raise ValueError("phase artifact hash/size records changed")


def _verify_development_bundle(target: Path, protocol: Mapping[str, Any], protocol_sha: str) -> dict[str, Any]:
    _assert_development_protocol(protocol)
    completion = _strict_json(target / "completion.json")
    if completion.get("phase") != "develop" or completion.get("protocol_sha256") != protocol_sha:
        raise ValueError("development completion binding is invalid")
    base_names = (
        "access_intent.json",
        "phase_manifest.json",
        "development_cv.json",
        "development_result.json",
    )
    completion_artifacts = completion.get("artifacts")
    if not isinstance(completion_artifacts, dict):
        raise ValueError("development completion artifact index is invalid")
    artifact_keys = set(completion_artifacts)
    if artifact_keys == set(base_names):
        model_present = False
        names = base_names
    elif artifact_keys == set(base_names) | {"pclr_model.npz"}:
        model_present = True
        names = (*base_names, "pclr_model.npz")
    else:
        raise ValueError("development completion has an invalid artifact variant")
    _verify_artifacts(target, completion, names)
    intent = _strict_json(target / "access_intent.json")
    manifest = _strict_json(target / "phase_manifest.json")
    expected_intent = _base_intent("develop", protocol_sha, _development_access_plan(protocol))
    if intent != expected_intent:
        raise ValueError("development access intent semantic replay failed")
    if manifest != _phase_manifest("develop", protocol_sha, _hash_file(target / "access_intent.json"), protocol, manifest["runtime"]):
        raise ValueError("development phase manifest semantic replay failed")
    cv = _strict_json(target / "development_cv.json")
    candidates = [_normalize_candidate(row, _fold_assignment()) for row in cv.get("candidates", [])]
    if len(candidates) != len(GRID):
        raise ValueError("development CV artifact is incomplete")
    if {row["candidate_id"] for row in candidates} != {
        row["candidate_id"] for row in GRID
    }:
        raise ValueError("development CV candidate IDs differ from the frozen grid")
    eligible = [
        candidate
        for candidate in candidates
        if candidate["selection_eligible"] is True
    ]
    if not eligible:
        if model_present:
            raise ValueError("zero-eligible DEVELOPMENT_NO_GO must not contain a model")
        selected = None
        expected_result = _development_no_go_result(protocol_sha, candidates)
    else:
        if not model_present:
            raise ValueError("eligible development selection is missing its final model")
        selected = _select_candidate(candidates)
        expected_result = _selected_development_result(protocol_sha, selected)
    result = _strict_json(target / "development_result.json")
    if result != expected_result:
        raise ValueError("development result semantic replay failed")
    if selected is not None:
        _validate_model_arrays(_safe_npz(target / "pclr_model.npz"))
        _validate_pclr_model_file(target / "pclr_model.npz", selected["params"])
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    if completion.get("semantic_replay_sha256") != semantic:
        raise ValueError("development completion semantic digest changed")
    return result


def build_confirmation_protocol(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    development_protocol_path: Path,
    development_protocol_sha256: str,
) -> dict[str, Any]:
    development_protocol, development_sha = _load_protocol(
        development_protocol_path,
        development_protocol_sha256,
        schema="pclr_development_protocol",
    )
    _assert_development_protocol(development_protocol)
    _assert_bound_output(output_root, development_protocol)
    development_target = output_root.resolve() / "develop"
    _assert_completed_clean(output_root.resolve(), "develop")
    result = _verify_development_bundle(development_target, development_protocol, development_sha)
    if result["decision"] != "PASS" or result["confirmation_freeze_permitted"] is not True:
        raise RuntimeError("development go/no-go did not pass; confirmation freeze is forbidden")
    _assert_phase_unstarted(output_root.resolve(), "calibrate")
    _assert_phase_unstarted(output_root.resolve(), "confirm")
    _verify_live_inputs(project_root.resolve(), data_dir, development_protocol)
    model_path = development_target / "pclr_model.npz"
    model_record = _hash_record(model_path)
    return {
        "schema": "pclr_confirmation_protocol",
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "frozen_after_passed_development_before_calibration_or_confirmation_access",
        "method": METHOD_SHORT,
        "design_document": development_protocol["design_document"],
        "implementation_records": development_protocol["implementation_records"],
        "source_record": development_protocol["source_record"],
        "source_metadata_record": development_protocol["source_metadata_record"],
        "governance": development_protocol["governance"],
        "monitor": development_protocol["monitor"],
        "development_binding": {
            "protocol_sha256": development_sha,
            "completion_sha256": _hash_file(development_target / "completion.json"),
            "result_sha256": _hash_file(development_target / "development_result.json"),
            "selected_candidate_id": result["selected_candidate_id"],
            "selected_params": result["selected_params"],
            "model_record": model_record,
            "model_absolute_path": str(model_path.resolve()),
        },
        "calibration": {
            "healthy_source_idv": 1,
            "fault_magnitude_percent": CALIBRATION_MAGNITUDE,
            "simulation_runs": list(CALIBRATION_RUNS),
            "source_window_half_open": list(HEALTHY_SOURCE_WINDOW),
            "healthy_window_excludes_first_affected_global_index": 600,
            "pseudo_onset_local_zero_based": FAULT_START,
            "rank": CALIBRATION_RANK,
            "rule": "99th_ascending_of_100_run_maxima_noninterpolated",
            "comparison": "strictly_greater_than",
            "epsilon": CALIBRATION_EPSILON,
            "delta": CALIBRATION_DELTA,
            "certificate_assumption": "100_distinct_RNG_seed_IDV1_prefault_simulations_from_one_fixed_Mode1_nominal_distribution",
            "minimum_numerically_distinct_maxima": MIN_UNIQUE_MAXIMA,
            "q99_strictly_below_q100": True,
            "top_tail": {"lower_rank": TOP_TAIL_RANK, "upper_rank": 100, "safety_multiplier": TOP_TAIL_TOLERANCE_MULTIPLIER, "reference": "max(observed_full_path_CPU_CUDA_error,float_tolerance,parity_atol)", "comparison": "spread_strictly_greater_than_multiplier_times_reference"},
            "float_tolerance": FLOAT_TOLERANCE,
            "parity_atol": PARITY_ATOL,
            "known_numerical_or_algorithmic_limits": list(KNOWN_NUMERICAL_LIMITS),
        },
        "confirmation": {
            "fault_magnitude_percent": CONFIRMATION_MAGNITUDE,
            "healthy": {"source_idv": 1, "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_runs": list(CONFIRMATION_RUNS), "source_window_half_open": list(HEALTHY_SOURCE_WINDOW), "excludes_first_affected_global_index": 600, "pseudo_onset_local_zero_based": FAULT_START},
            "faults": {"source_idvs": list(FAULT_IDS), "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_runs": list(CONFIRMATION_RUNS), "source_window_half_open": list(FAULT_SOURCE_WINDOW)},
            "local_fault_start_zero_based": FAULT_START,
            "detection_requires_new_alarm_episode_start_at_or_after": FAULT_START,
            "gates": CONFIRMATION_GATES,
            "one_time_even_if_decision_is_fail": True,
        },
    }


def freeze_confirmation(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    development_protocol_path: Path,
    development_protocol_sha256: str,
    confirmation_protocol_path: Path,
) -> str:
    payload = build_confirmation_protocol(
        project_root,
        data_dir,
        output_root,
        development_protocol_path,
        development_protocol_sha256,
    )
    return _publish_protocol_bundle(confirmation_protocol_path, payload)


def _model_record(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return dict(protocol["development_binding"]["model_record"])


def _verify_confirmation_live_inputs(project_root: Path, data_dir: Path, protocol: Mapping[str, Any]) -> Path:
    if _implementation_records(project_root.resolve()) != protocol["implementation_records"]:
        raise ValueError("confirmation implementation changed after freeze")
    if _source_record(data_dir) != protocol["source_record"]:
        raise ValueError("confirmation source changed after freeze")
    if _source_metadata_record(data_dir) != protocol["source_metadata_record"]:
        raise ValueError("confirmation source metadata/topology changed after freeze")
    model_path = Path(protocol["development_binding"]["model_absolute_path"])
    if _hash_record(model_path) != _model_record(protocol):
        raise ValueError("frozen PCLR model changed")
    _validate_pclr_model_file(model_path, protocol["development_binding"]["selected_params"])
    return model_path


def _calibration_access_plan() -> list[dict[str, Any]]:
    return [
        {"idv": 1, "magnitude_percent": CALIBRATION_MAGNITUDE, "simulation_run": run, "window_half_open": list(HEALTHY_SOURCE_WINDOW), "role": "healthy_calibration"}
        for run in CALIBRATION_RUNS
    ]


def _confirmation_access_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"idv": 1, "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_run": run, "window_half_open": list(HEALTHY_SOURCE_WINDOW), "role": "healthy_confirmation"}
        for run in CONFIRMATION_RUNS
    ]
    rows.extend(
        {"idv": fault, "magnitude_percent": CONFIRMATION_MAGNITUDE, "simulation_run": run, "window_half_open": list(FAULT_SOURCE_WINDOW), "role": f"fault-{fault}"}
        for fault in FAULT_IDS
        for run in CONFIRMATION_RUNS
    )
    return rows


def _replay_page_persistence(
    standardized_logits: np.ndarray, page_drift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = standardized_logits.shape[0]
    cusum = np.zeros_like(standardized_logits)
    state = np.zeros((count, len(FAULT_IDS)), dtype=np.float64)
    for time in range(FAULT_START, SEQUENCE_LENGTH):
        state = np.maximum(0.0, state + standardized_logits[:, time] - page_drift)
        cusum[:, time] = state
    persistent = np.zeros_like(cusum)
    for time in range(FIRST_ALARM_ELIGIBLE, SEQUENCE_LENGTH):
        persistent[:, time] = np.min(
            cusum[:, time - PERSISTENCE + 1 : time + 1], axis=1
        )
    envelope = np.max(persistent, axis=2)
    eligible = np.zeros((count, SEQUENCE_LENGTH), dtype=np.bool_)
    eligible[:, FIRST_ALARM_ELIGIBLE:] = True
    return cusum, persistent, envelope, eligible


def _validate_score_payload(
    value: Mapping[str, Any], *, count: int, require_attribution: bool, page_drift: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not math.isfinite(float(page_drift)) or float(page_drift) < 0.0:
        raise ValueError("frozen Page drift must be finite and nonnegative")
    arrays: dict[str, np.ndarray] = {}
    replay_errors: dict[str, float] = {}
    for prefix in ("cpu", "cuda"):
        standardized = np.asarray(value[f"{prefix}_standardized_logits"])
        cusum = np.asarray(value[f"{prefix}_cusum"])
        persistent = np.asarray(value[f"{prefix}_component_persistence"])
        scores = np.asarray(value[f"{prefix}_scores"])
        eligible = np.asarray(value[f"{prefix}_eligible"])
        component_shape = (count, SEQUENCE_LENGTH, len(FAULT_IDS))
        if any(array.shape != component_shape or array.dtype != np.dtype(np.float64) for array in (standardized, cusum, persistent)):
            raise ValueError(f"{prefix} component paths must be float64 [{count},500,3]")
        if scores.shape != (count, SEQUENCE_LENGTH) or scores.dtype != np.dtype(np.float64):
            raise ValueError(f"{prefix} scores must be float64 [{count},500]")
        if eligible.shape != scores.shape or eligible.dtype != np.dtype(np.bool_):
            raise ValueError(f"{prefix} eligibility must be bool [{count},500]")
        if not all(np.isfinite(array).all() for array in (standardized, cusum, persistent, scores)):
            raise ValueError("all standardized-logit/Page/persistence/envelope paths must be finite")
        if np.any(cusum < 0.0) or np.any(persistent < 0.0) or np.any(scores < 0.0):
            raise ValueError("Page, persistence, and envelope paths must be nonnegative")
        expected_cusum, expected_persistent, expected_scores, expected_eligible = _replay_page_persistence(
            standardized, float(page_drift)
        )
        tolerance = FLOAT_TOLERANCE if prefix == "cpu" else PARITY_ATOL
        errors = {
            "cusum": float(np.max(np.abs(cusum - expected_cusum))),
            "persistence": float(np.max(np.abs(persistent - expected_persistent))),
            "envelope": float(np.max(np.abs(scores - expected_scores))),
        }
        if max(errors.values()) > tolerance:
            raise ValueError(f"{prefix} Page/persistence/envelope semantic replay failed")
        if not np.array_equal(eligible, expected_eligible):
            raise ValueError(f"eligibility geometry differs from local [{FIRST_ALARM_ELIGIBLE},500)")
        for name, array in (
            ("standardized_logits", standardized),
            ("cusum", cusum),
            ("component_persistence", persistent),
            ("scores", scores),
            ("eligible", eligible),
        ):
            arrays[f"{prefix}_{name}"] = array
        replay_errors[prefix] = max(errors.values())
        if require_attribution:
            attribution = np.asarray(FAULT_IDS, dtype=np.int64)[np.argmax(persistent, axis=2)]
            arrays[f"{prefix}_attribution"] = attribution.astype(np.int64, copy=False)
    for name in ("source_idvs", "simulation_runs", "magnitude_percent", "rng_seeds"):
        array = np.asarray(value.get(name))
        if array.shape != (count,) or array.dtype != np.dtype(np.int64):
            raise ValueError(f"{name} must be exact int64 [{count}]")
        arrays[name] = array
    if np.any(arrays["rng_seeds"] <= 0) or np.unique(arrays["rng_seeds"]).size != count:
        raise ValueError("each scored row must have one distinct positive RNG seed")
    numeric_fields = ("standardized_logits", "cusum", "component_persistence", "scores")
    maximums = {
        field: float(np.max(np.abs(arrays[f"cpu_{field}"] - arrays[f"cuda_{field}"])))
        for field in numeric_fields
    }
    max_error = max(maximums.values())
    parity = {
        "max_abs_score_error": max_error,
        "numeric_stage_max_abs_errors": maximums,
        "semantic_replay_max_abs_errors": replay_errors,
        "eligibility_exact": bool(np.array_equal(arrays["cpu_eligible"], arrays["cuda_eligible"])),
        "attribution_exact": (bool(np.array_equal(arrays["cpu_attribution"], arrays["cuda_attribution"])) if require_attribution else True),
    }
    return arrays, parity


def _numeric_stage_parity_errors(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, float]:
    fields = ("standardized_logits", "cusum", "component_persistence", "scores")
    errors: dict[str, float] = {}
    for field in fields:
        cpu = np.asarray(arrays[f"cpu_{field}"])
        cuda = np.asarray(arrays[f"cuda_{field}"])
        if cpu.shape != cuda.shape or cpu.dtype != np.dtype(np.float64) or cuda.dtype != np.dtype(np.float64):
            raise ValueError(f"CPU/CUDA {field} paths must have matching float64 geometry")
        if not np.isfinite(cpu).all() or not np.isfinite(cuda).all():
            raise ValueError(f"CPU/CUDA {field} paths must be finite")
        errors[field] = float(np.max(np.abs(cpu - cuda)))
    return errors


def _run_maxima(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    masked = np.where(eligible, scores, -np.inf)
    maxima = np.max(masked, axis=1)
    if not np.isfinite(maxima).all():
        raise ValueError("each run must contain finite eligible monitor scores")
    return maxima.astype(np.float64, copy=False)


def _distinct_count(values: np.ndarray, tolerance: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        return 0
    return 1 + int(np.count_nonzero(np.diff(ordered) > tolerance))


def _binomial_upper_tail(n: int, success_probability: float, at_least: int) -> float:
    return float(sum(
        math.comb(n, count)
        * success_probability ** count
        * (1.0 - success_probability) ** (n - count)
        for count in range(at_least, n + 1)
    ))


def _calibration_summary(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not np.array_equal(arrays["source_idvs"], np.ones(100, dtype=np.int64)):
        raise ValueError("calibration identity must contain only IDV1 pre-fault controls")
    if not np.array_equal(arrays["simulation_runs"], np.asarray(CALIBRATION_RUNS, dtype=np.int64)):
        raise ValueError("calibration simulation-run order changed")
    if not np.array_equal(
        arrays["magnitude_percent"],
        np.full(100, CALIBRATION_MAGNITUDE, dtype=np.int64),
    ):
        raise ValueError("calibration magnitude identity changed")
    cpu_maxima = _run_maxima(arrays["cpu_scores"], arrays["cpu_eligible"])
    cuda_maxima = _run_maxima(arrays["cuda_scores"], arrays["cuda_eligible"])
    stage_errors = _numeric_stage_parity_errors(arrays)
    max_error = max(stage_errors.values())
    score_error = stage_errors["scores"]
    threshold = float(np.sort(cpu_maxima)[CALIBRATION_RANK - 1])
    q100 = float(np.sort(cpu_maxima)[99])
    q95 = float(np.sort(cpu_maxima)[TOP_TAIL_RANK - 1])
    cpu_alarm = arrays["cpu_eligible"] & (arrays["cpu_scores"] > threshold)
    cuda_alarm = arrays["cuda_eligible"] & (arrays["cuda_scores"] > threshold)
    known_limit_hit = any(
        bool(np.any(np.isclose(cpu_maxima, limit, rtol=0.0, atol=FLOAT_TOLERANCE)))
        for limit in KNOWN_NUMERICAL_LIMITS
    )
    distinct = _distinct_count(cpu_maxima, FLOAT_TOLERANCE)
    tail_spread = q100 - q95
    tail_reference = max(max_error, FLOAT_TOLERANCE, PARITY_ATOL)
    required_tail_spread = TOP_TAIL_TOLERANCE_MULTIPLIER * tail_reference
    gates = {
        "all_paths_finite": True,
        "cuda_numpy_all_numeric_stage_error_at_most_1e_10": max_error <= PARITY_ATOL,
        "cuda_numpy_eligibility_exact": bool(np.array_equal(arrays["cpu_eligible"], arrays["cuda_eligible"])),
        "cuda_numpy_alarm_exact": bool(np.array_equal(cpu_alarm, cuda_alarm)),
        "at_least_95_numerically_distinct_maxima": distinct >= MIN_UNIQUE_MAXIMA,
        "q99_strictly_below_q100": threshold < q100,
        "top_tail_spread_above_100_times_numeric_error_reference": tail_spread > required_tail_spread,
        "no_maximum_equals_known_numerical_or_algorithmic_limit": not known_limit_hit,
        "no_top_finite_tie": int(np.count_nonzero(np.isclose(cpu_maxima, q100, rtol=0.0, atol=FLOAT_TOLERANCE))) == 1,
    }
    certificate_failure = _binomial_upper_tail(100, 1.0 - CALIBRATION_EPSILON, CALIBRATION_RANK)
    certificate = {
        "sample_count": 100,
        "ascending_order_statistic_rank": CALIBRATION_RANK,
        "epsilon": CALIBRATION_EPSILON,
        "delta": CALIBRATION_DELTA,
        "failure_probability_binomial_tail": certificate_failure,
        "failure_probability_strictly_below_delta": certificate_failure < CALIBRATION_DELTA,
        "comparison": "strictly_greater_than",
    }
    gates["order_statistic_certificate_below_delta"] = certificate["failure_probability_strictly_below_delta"]
    result = {
        "schema": "pclr_calibration_result",
        "schema_version": SCHEMA_VERSION,
        "source_magnitude_percent": CALIBRATION_MAGNITUDE,
        "source_role": "IDV1_pre_fault_Mode1_control_window",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "operational_threshold_q99": threshold,
        "q100": q100,
        "q95": q95,
        "top_tail_spread_q100_minus_q95": tail_spread,
        "top_tail_numeric_error_reference": tail_reference,
        "required_top_tail_spread_strict_lower_bound": required_tail_spread,
        "numerically_distinct_maxima": distinct,
        "max_abs_cpu_cuda_score_error": score_error,
        "max_abs_cpu_cuda_full_path_error": max_error,
        "numeric_stage_max_abs_errors": stage_errors,
        "gates": gates,
        "certificate": certificate,
        "confirmation_access_permitted": all(gates.values()),
    }
    evidence = dict(arrays)
    evidence.update({
        "cpu_run_maxima": cpu_maxima,
        "cuda_run_maxima": cuda_maxima,
        "cpu_alarm": cpu_alarm,
        "cuda_alarm": cuda_alarm,
        "operational_threshold_q99": np.asarray([threshold], dtype=np.float64),
    })
    return result, evidence


def run_calibration(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    batch_size = _require_positive_batch_size(batch_size)
    protocol, protocol_sha = _load_protocol(protocol_path, protocol_sha256, schema="pclr_confirmation_protocol")
    _assert_confirmation_protocol(protocol)
    _assert_bound_output(output_root, protocol)
    target = output_root.resolve() / "calibrate"
    if target.exists():
        _assert_completed_clean(output_root.resolve(), "calibrate")
        return _verify_calibration_bundle(target, protocol, protocol_sha)
    target, staging = _claim_phase(output_root.resolve(), "calibrate")
    model_path = _verify_confirmation_live_inputs(project_root, data_dir, protocol)
    runtime = _cuda_preflight()
    plan = _calibration_access_plan()
    intent = _base_intent("calibrate", protocol_sha, plan)
    _write_json(staging / "access_intent.json", intent)
    manifest = _phase_manifest("calibrate", protocol_sha, _hash_file(staging / "access_intent.json"), protocol, runtime, model_record=_model_record(protocol))
    _write_json(staging / "phase_manifest.json", manifest)

    backend = _load_backend()
    data = _backend_function(backend, "load_numeric_runs")(_source_path(data_dir), plan)
    payload = _backend_function(backend, "score_paths")(
        model_path=model_path, data=data, device="cuda", dtype="float64", batch_size=int(batch_size)
    )
    arrays, parity = _validate_score_payload(
        payload,
        count=100,
        require_attribution=False,
        page_drift=float(protocol["development_binding"]["selected_params"]["page_drift"]),
    )
    parity["alarm_exact"] = True  # recalculated and validated in _calibration_summary
    _validate_parity(parity)
    result, evidence = _calibration_summary(arrays)
    result["protocol_sha256"] = protocol_sha
    result["model_sha256"] = _model_record(protocol)["sha256"]
    _write_json(staging / "calibration_result.json", result)
    _write_npz(staging / "calibration_evidence.npz", evidence)
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    _finalize_phase(target, staging, phase="calibrate", protocol_sha256=protocol_sha, artifact_names=("access_intent.json", "phase_manifest.json", "calibration_result.json", "calibration_evidence.npz"), semantic_sha256=semantic)
    return _verify_calibration_bundle(target, protocol, protocol_sha)


def _verify_calibration_bundle(target: Path, protocol: Mapping[str, Any], protocol_sha: str) -> dict[str, Any]:
    _assert_confirmation_protocol(protocol)
    completion = _strict_json(target / "completion.json")
    if completion.get("phase") != "calibrate" or completion.get("protocol_sha256") != protocol_sha:
        raise ValueError("calibration completion binding is invalid")
    names = ("access_intent.json", "phase_manifest.json", "calibration_result.json", "calibration_evidence.npz")
    _verify_artifacts(target, completion, names)
    intent = _strict_json(target / "access_intent.json")
    expected_intent = _base_intent("calibrate", protocol_sha, _calibration_access_plan())
    if intent != expected_intent:
        raise ValueError("calibration access intent semantic replay failed")
    manifest = _strict_json(target / "phase_manifest.json")
    expected_manifest = _phase_manifest("calibrate", protocol_sha, _hash_file(target / "access_intent.json"), protocol, manifest["runtime"], model_record=_model_record(protocol))
    if manifest != expected_manifest:
        raise ValueError("calibration manifest semantic replay failed")
    arrays = _safe_npz(target / "calibration_evidence.npz")
    base, parity = _validate_score_payload(
        arrays,
        count=100,
        require_attribution=False,
        page_drift=float(protocol["development_binding"]["selected_params"]["page_drift"]),
    )
    parity["alarm_exact"] = True
    _validate_parity(parity)
    replay, expected_evidence = _calibration_summary(base)
    _require_exact_arrays(arrays, expected_evidence, label="calibration evidence")
    replay["protocol_sha256"] = protocol_sha
    replay["model_sha256"] = _model_record(protocol)["sha256"]
    result = _strict_json(target / "calibration_result.json")
    if replay != result:
        raise ValueError("calibration result semantic replay failed")
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    if completion.get("semantic_replay_sha256") != semantic:
        raise ValueError("calibration completion semantic digest changed")
    return result


def _calibration_binding(output_root: Path, protocol: Mapping[str, Any], protocol_sha: str) -> dict[str, Any]:
    target = output_root.resolve() / "calibrate"
    _assert_completed_clean(output_root.resolve(), "calibrate")
    result = _verify_calibration_bundle(target, protocol, protocol_sha)
    if result["decision"] != "PASS" or result["confirmation_access_permitted"] is not True:
        raise RuntimeError("calibration failed non-degeneracy; final confirmation access is forbidden")
    return {
        "completion_sha256": _hash_file(target / "completion.json"),
        "result_sha256": _hash_file(target / "calibration_result.json"),
        "operational_threshold_q99": result["operational_threshold_q99"],
        "rank": CALIBRATION_RANK,
        "comparison": "strictly_greater_than",
    }


def _episode_first_start(alarm: np.ndarray) -> int | None:
    starts = np.flatnonzero(alarm & np.concatenate((np.asarray([True]), ~alarm[:-1])))
    starts = starts[starts >= FAULT_START]
    return int(starts[0]) if starts.size else None


def _confirmation_summary(
    arrays: Mapping[str, np.ndarray], threshold: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_idvs = np.asarray(
        [1] * 100 + [fault for fault in FAULT_IDS for _ in range(100)],
        dtype=np.int64,
    )
    if not np.array_equal(arrays["source_idvs"], expected_idvs):
        raise ValueError("confirmation IDV order changed")
    if not np.array_equal(
        arrays["simulation_runs"],
        np.asarray(list(CONFIRMATION_RUNS) * len(ALL_IDS), dtype=np.int64),
    ):
        raise ValueError("confirmation simulation-run order changed")
    if not np.array_equal(
        arrays["magnitude_percent"],
        np.full(400, CONFIRMATION_MAGNITUDE, dtype=np.int64),
    ):
        raise ValueError("confirmation magnitude identity changed")
    cpu_alarm = arrays["cpu_eligible"] & (arrays["cpu_scores"] > threshold)
    cuda_alarm = arrays["cuda_eligible"] & (arrays["cuda_scores"] > threshold)
    stage_errors = _numeric_stage_parity_errors(arrays)
    max_error = max(stage_errors.values())
    parity = {
        "max_abs_score_error": max_error,
        "numeric_stage_max_abs_errors": stage_errors,
        "eligibility_exact": bool(np.array_equal(arrays["cpu_eligible"], arrays["cuda_eligible"])),
        "alarm_exact": bool(np.array_equal(cpu_alarm, cuda_alarm)),
        "attribution_exact": bool(np.array_equal(arrays["cpu_attribution"], arrays["cuda_attribution"])),
    }
    _validate_parity(parity)
    if not parity["attribution_exact"]:
        raise ValueError("CPU/CUDA attribution arrays differ")
    healthy_alarm_count = sum(_episode_first_start(cpu_alarm[index]) is not None for index in range(100))
    per_fault: dict[str, Any] = {}
    total_detected = 0
    for fault_index, fault in enumerate(FAULT_IDS):
        offset = 100 + fault_index * 100
        delays: list[int] = []
        detected = 0
        attributed = 0
        for local in range(100):
            index = offset + local
            start = _episode_first_start(cpu_alarm[index])
            if start is None:
                delays.append(MISS_DELAY)
                continue
            detected += 1
            delays.append(start - FAULT_START)
            if int(arrays["cpu_attribution"][index, start]) == fault:
                attributed += 1
        total_detected += detected
        per_fault[str(fault)] = {
            "detected_runs": detected,
            "detection_rate": detected / 100.0,
            "median_restricted_delay_samples": float(statistics.median(delays)),
            "p90_restricted_delay_samples_nearest_rank": _nearest_rank(delays, 0.90),
            "attribution_matches_fault_runs": attributed,
        }
    gates = {
        "healthy_alarm_runs_at_most_4": healthy_alarm_count <= 4,
        "each_fault_detected_runs_at_least_90": all(row["detected_runs"] >= 90 for row in per_fault.values()),
        "total_detected_runs_at_least_270": total_detected >= 270,
        "each_fault_median_delay_at_most_120": all(row["median_restricted_delay_samples"] <= 120 for row in per_fault.values()),
        "each_fault_p90_delay_at_most_180": all(row["p90_restricted_delay_samples_nearest_rank"] <= 180 for row in per_fault.values()),
        "full_cuda_numpy_float64_parity": True,
    }
    result = {
        "schema": "pclr_confirmation_result",
        "schema_version": SCHEMA_VERSION,
        "source_magnitude_percent": CONFIRMATION_MAGNITUDE,
        "claim_scope": "prospective_severity_shift_confirmation_with_independent_RNG_seed_simulations_within_Extended_TEP_Mode1",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "operational_threshold_q99": float(threshold),
        "healthy_alarm_runs": int(healthy_alarm_count),
        "healthy_run_count": 100,
        "faults": per_fault,
        "total_detected_fault_runs": int(total_detected),
        "gates": gates,
        "parity": parity,
        "final_rows_irreversibly_consumed": True,
        "rerun_or_retuning_after_pass_or_fail_forbidden": True,
    }
    evidence = dict(arrays)
    evidence.update({
        "cpu_alarm": cpu_alarm,
        "cuda_alarm": cuda_alarm,
        "operational_threshold_q99": np.asarray([threshold], dtype=np.float64),
    })
    return result, evidence


def run_confirmation(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    *,
    batch_size: int,
) -> dict[str, Any]:
    batch_size = _require_positive_batch_size(batch_size)
    protocol, protocol_sha = _load_protocol(protocol_path, protocol_sha256, schema="pclr_confirmation_protocol")
    _assert_confirmation_protocol(protocol)
    _assert_bound_output(output_root, protocol)
    target = output_root.resolve() / "confirm"
    if target.exists():
        _assert_completed_clean(output_root.resolve(), "confirm")
        return _verify_confirmation_bundle(target, protocol, protocol_sha, output_root)
    calibration_binding = _calibration_binding(output_root, protocol, protocol_sha)
    target, staging = _claim_phase(output_root.resolve(), "confirm")
    model_path = _verify_confirmation_live_inputs(project_root, data_dir, protocol)
    runtime = _cuda_preflight()
    plan = _confirmation_access_plan()
    intent = _base_intent("confirm", protocol_sha, plan)
    intent["one_time_final_access_even_if_result_fails"] = True
    _write_json(staging / "access_intent.json", intent)
    manifest = _phase_manifest("confirm", protocol_sha, _hash_file(staging / "access_intent.json"), protocol, runtime, model_record=_model_record(protocol), calibration_binding=calibration_binding)
    _write_json(staging / "phase_manifest.json", manifest)

    backend = _load_backend()
    data = _backend_function(backend, "load_numeric_runs")(_source_path(data_dir), plan)
    payload = _backend_function(backend, "score_paths")(
        model_path=model_path, data=data, device="cuda", dtype="float64", batch_size=int(batch_size)
    )
    arrays, _ = _validate_score_payload(
        payload,
        count=400,
        require_attribution=True,
        page_drift=float(protocol["development_binding"]["selected_params"]["page_drift"]),
    )
    result, evidence = _confirmation_summary(arrays, float(calibration_binding["operational_threshold_q99"]))
    result["protocol_sha256"] = protocol_sha
    result["model_sha256"] = _model_record(protocol)["sha256"]
    result["calibration_binding"] = calibration_binding
    _write_json(staging / "confirmation_result.json", result)
    _write_npz(staging / "confirmation_evidence.npz", evidence)
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    _finalize_phase(target, staging, phase="confirm", protocol_sha256=protocol_sha, artifact_names=("access_intent.json", "phase_manifest.json", "confirmation_result.json", "confirmation_evidence.npz"), semantic_sha256=semantic)
    return _verify_confirmation_bundle(target, protocol, protocol_sha, output_root)


def _verify_confirmation_bundle(target: Path, protocol: Mapping[str, Any], protocol_sha: str, output_root: Path) -> dict[str, Any]:
    _assert_confirmation_protocol(protocol)
    completion = _strict_json(target / "completion.json")
    if completion.get("phase") != "confirm" or completion.get("protocol_sha256") != protocol_sha:
        raise ValueError("confirmation completion binding is invalid")
    names = ("access_intent.json", "phase_manifest.json", "confirmation_result.json", "confirmation_evidence.npz")
    _verify_artifacts(target, completion, names)
    calibration_binding = _calibration_binding(output_root, protocol, protocol_sha)
    intent = _strict_json(target / "access_intent.json")
    expected_intent = _base_intent("confirm", protocol_sha, _confirmation_access_plan())
    expected_intent["one_time_final_access_even_if_result_fails"] = True
    if intent != expected_intent:
        raise ValueError("confirmation access intent semantic replay failed")
    manifest = _strict_json(target / "phase_manifest.json")
    expected_manifest = _phase_manifest("confirm", protocol_sha, _hash_file(target / "access_intent.json"), protocol, manifest["runtime"], model_record=_model_record(protocol), calibration_binding=calibration_binding)
    if manifest != expected_manifest:
        raise ValueError("confirmation manifest semantic replay failed")
    arrays = _safe_npz(target / "confirmation_evidence.npz")
    base, _ = _validate_score_payload(
        arrays,
        count=400,
        require_attribution=True,
        page_drift=float(protocol["development_binding"]["selected_params"]["page_drift"]),
    )
    replay, expected_evidence = _confirmation_summary(
        base, float(calibration_binding["operational_threshold_q99"])
    )
    _require_exact_arrays(arrays, expected_evidence, label="confirmation evidence")
    replay["protocol_sha256"] = protocol_sha
    replay["model_sha256"] = _model_record(protocol)["sha256"]
    replay["calibration_binding"] = calibration_binding
    result = _strict_json(target / "confirmation_result.json")
    if replay != result:
        raise ValueError("confirmation result semantic replay failed")
    semantic = _selection_digest({"intent": intent, "manifest": manifest, "result": result})
    if completion.get("semantic_replay_sha256") != semantic:
        raise ValueError("confirmation completion semantic digest changed")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_dev = subparsers.add_parser("freeze-development")
    freeze_dev.add_argument("data_dir", type=Path)
    freeze_dev.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    freeze_dev.add_argument("--protocol", type=Path, default=DEFAULT_DEVELOPMENT_PROTOCOL)

    develop = subparsers.add_parser("develop")
    develop.add_argument("data_dir", type=Path)
    develop.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    develop.add_argument("--protocol", type=Path, default=DEFAULT_DEVELOPMENT_PROTOCOL)
    develop.add_argument("--protocol-sha256", required=True)
    develop.add_argument("--batch-size", type=int, default=32)

    freeze_confirm = subparsers.add_parser("freeze-confirmation")
    freeze_confirm.add_argument("data_dir", type=Path)
    freeze_confirm.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    freeze_confirm.add_argument("--development-protocol", type=Path, default=DEFAULT_DEVELOPMENT_PROTOCOL)
    freeze_confirm.add_argument("--development-protocol-sha256", required=True)
    freeze_confirm.add_argument("--protocol", type=Path, default=DEFAULT_CONFIRMATION_PROTOCOL)

    for command in ("calibrate", "confirm"):
        numeric = subparsers.add_parser(command)
        numeric.add_argument("data_dir", type=Path)
        numeric.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        numeric.add_argument("--protocol", type=Path, default=DEFAULT_CONFIRMATION_PROTOCOL)
        numeric.add_argument("--protocol-sha256", required=True)
        numeric.add_argument("--batch-size", type=_positive_batch_size_arg, default=32)
    return parser


def _positive_batch_size_arg(value: str) -> int:
    try:
        return _require_positive_batch_size(int(value))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("batch size must be a positive integer") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = _project_root()
    if args.command == "freeze-development":
        digest = freeze_development(root, args.data_dir, args.output_root, args.protocol)
        print(digest)
    elif args.command == "develop":
        result = run_development(root, args.data_dir, args.output_root, args.protocol, args.protocol_sha256, batch_size=args.batch_size)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "freeze-confirmation":
        digest = freeze_confirmation(root, args.data_dir, args.output_root, args.development_protocol, args.development_protocol_sha256, args.protocol)
        print(digest)
    elif args.command == "calibrate":
        result = run_calibration(root, args.data_dir, args.output_root, args.protocol, args.protocol_sha256, batch_size=args.batch_size)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "confirm":
        result = run_confirmation(root, args.data_dir, args.output_root, args.protocol, args.protocol_sha256, batch_size=args.batch_size)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
