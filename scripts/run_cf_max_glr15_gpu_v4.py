#!/usr/bin/env python3
"""Governed CUDA runner for the fixed CF-Max-GLR15 v4 experiment.

The module intentionally imports only the Python standard library.  Numeric
imports happen lazily through ``mate_spc_cf_glr.backend_v4`` so that PyTorch is
the first native numeric runtime loaded on Windows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence
import uuid


SCHEMA_VERSION = 1
ALL_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
RUNS = tuple(range(1, 101))
FOLD_COUNT = 5
FOLD_SALT = "cf-max-glr15-v4-mode1-mag50-three-way-2026-09-08"
DEVELOPMENT_MAGNITUDE = 50
CALIBRATION_MAGNITUDE = 25
CONFIRMATION_MAGNITUDE = 75
HEALTHY_SOURCE_WINDOW = (100, 600)
FAULT_SOURCE_WINDOW = (440, 940)
MONITOR_START = 160
RUN_LENGTH = 500
MISS_DELAY = RUN_LENGTH - MONITOR_START
PARITY_ATOL = 1.0e-9
FLOAT_TOLERANCE = 1.0e-12
INNER_CALIBRATION_RANK = 20
FORMAL_CALIBRATION_RANK = 98
UNIQUE_CANDIDATE_ID = "cf-max-glr15-fixed"

DEVELOPMENT_GATES = {
    "healthy_alarm_runs_max": 10,
    "healthy_alarm_runs_per_fold_max": 4,
    "each_fault_detection_rate_min": 0.85,
    "each_fold_each_fault_detection_rate_min": 0.70,
    "each_fault_median_delay_max": 140,
    "each_fault_nearest_rank_p90_delay_max": 220,
}
CONFIRMATION_GATES = {
    "healthy_alarm_runs_max": 4,
    "each_fault_detected_runs_min": 90,
    "each_fault_median_delay_max": 140,
    "each_fault_nearest_rank_p90_delay_max": 220,
}

DEFAULT_OUTPUT_ROOT = Path("outputs/cf_max_glr15_v4")
DEFAULT_FORMAL_PROTOCOL = Path("protocols/cf_max_glr15_formal_v4.json")
MODE1_FILE_NAME = "TEP_Mode1.h5"
MODE1_SIZE_BYTES = 23_924_710_848
MODE1_MD5 = "c79ecc6d12a5ccb49caffcd8f531feae"
MODE1_SHA256 = "d6b0ed0ef3974ff571afd6717ba0418a5b8eb58a7822c84a05735572d52b3294"

IMPLEMENTATION_FILES = (
    "scripts/run_cf_max_glr15_gpu_v4.py",
    "docs/cf_max_glr15_v4.md",
    "src/mate_spc_cf_glr/features.py",
    "src/mate_spc_cf_glr/model.py",
    "src/mate_spc_cf_glr/monitor.py",
    "src/mate_spc_cf_glr/backend_v4.py",
    "src/mate_spc_pclr/data_v2.py",
)


class NumericBackend(Protocol):
    def cuda_preflight(self) -> Mapping[str, Any]: ...

    def development_iteration(
        self,
        *,
        source_path: Path,
        access_plan: Sequence[Mapping[str, Any]],
        fold_map: Mapping[int, int],
        config: Mapping[str, Any],
        model_path: Path,
        monitor_path: Path,
        device: str,
        dtype: str,
    ) -> Mapping[str, Any]: ...

    def verify_bound_artifacts(
        self, *, model_path: Path, monitor_path: Path, config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def calibrate(
        self,
        *,
        source_path: Path,
        access_plan: Sequence[Mapping[str, Any]],
        model_path: Path,
        monitor_path: Path,
        config: Mapping[str, Any],
        device: str,
        dtype: str,
    ) -> Mapping[str, Any]: ...

    def confirm(
        self,
        *,
        source_path: Path,
        access_plan: Sequence[Mapping[str, Any]],
        model_path: Path,
        monitor_path: Path,
        config: Mapping[str, Any],
        threshold: float,
        device: str,
        dtype: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ProtocolBundle:
    protocol: Path
    sidecar: Path
    commit: Path
    claim: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, absolute: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "size_bytes": int(path.stat().st_size),
        "sha256": _hash_file(path),
    }
    if absolute:
        record["absolute_path"] = str(path.resolve())
    return record


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    if _canonical_json_bytes(value) != path.read_bytes():
        raise ValueError(f"JSON is not canonical: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def _claim_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _bundle_paths(protocol_path: Path) -> ProtocolBundle:
    return ProtocolBundle(
        protocol_path,
        protocol_path.with_suffix(protocol_path.suffix + ".sha256"),
        protocol_path.with_suffix(protocol_path.suffix + ".commit.json"),
        protocol_path.with_suffix(protocol_path.suffix + ".claim.json"),
    )


def _publish_protocol(protocol_path: Path, payload: Mapping[str, Any]) -> str:
    bundle = _bundle_paths(protocol_path)
    if any(path.exists() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
        raise FileExistsError("formal protocol bundle already exists")
    _claim_file(
        bundle.claim,
        {"schema": "cf_max_glr15_protocol_claim_v1", "claimed_at_utc": _now()},
    )
    encoded = _canonical_json_bytes(payload)
    digest = _sha256_bytes(encoded)
    _atomic_write(bundle.protocol, encoded)
    _atomic_write(bundle.sidecar, f"{digest}  {bundle.protocol.name}\n".encode("ascii"))
    _write_json(
        bundle.commit,
        {
            "schema": "cf_max_glr15_protocol_commit_v1",
            "protocol_file": bundle.protocol.name,
            "protocol_sha256": digest,
            "sidecar_sha256": _hash_file(bundle.sidecar),
        },
    )
    return digest


def _load_protocol(protocol_path: Path) -> tuple[dict[str, Any], str]:
    bundle = _bundle_paths(protocol_path)
    if not all(path.is_file() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
        raise FileNotFoundError("formal protocol bundle is incomplete")
    protocol = _strict_json(bundle.protocol)
    digest = _hash_file(bundle.protocol)
    if bundle.sidecar.read_text(encoding="ascii") != f"{digest}  {bundle.protocol.name}\n":
        raise ValueError("protocol SHA-256 sidecar differs")
    commit = _strict_json(bundle.commit)
    if commit != {
        "schema": "cf_max_glr15_protocol_commit_v1",
        "protocol_file": bundle.protocol.name,
        "protocol_sha256": digest,
        "sidecar_sha256": _hash_file(bundle.sidecar),
    }:
        raise ValueError("protocol commit differs")
    return protocol, digest


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _implementation_records(project_root: Path) -> dict[str, dict[str, Any]]:
    return {name: _file_record(project_root / name) for name in IMPLEMENTATION_FILES}


def _source_path(data_dir: Path) -> Path:
    path = data_dir / MODE1_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_record(data_dir: Path) -> dict[str, Any]:
    path = _source_path(data_dir)
    if int(path.stat().st_size) != MODE1_SIZE_BYTES:
        raise ValueError("Mode1 archive size differs from the official source")
    record = _file_record(path, absolute=True)
    record["md5"] = _hash_file(path, "md5")
    if record["sha256"] != MODE1_SHA256 or record["md5"] != MODE1_MD5:
        raise ValueError("Mode1 archive digest differs from the official source")
    return record


def _fold_assignment() -> dict[int, int]:
    ranked = sorted(
        RUNS,
        key=lambda run: hashlib.sha256(f"{FOLD_SALT}|{run}".encode("ascii")).hexdigest(),
    )
    result = {run: index % FOLD_COUNT for index, run in enumerate(ranked)}
    if [sum(fold == value for value in result.values()) for fold in range(FOLD_COUNT)] != [20] * 5:
        raise AssertionError("canonical folds must contain 20 run blocks each")
    return dict(sorted(result.items()))


def _fold_roles(outer_fold: int, fold_map: Mapping[int, int]) -> dict[str, list[int]]:
    if type(outer_fold) is not int or outer_fold not in range(FOLD_COUNT):
        raise ValueError("outer_fold must be 0..4")
    calibration_fold = (outer_fold + 1) % FOLD_COUNT
    roles = {
        "fit": sorted(
            run for run, fold in fold_map.items() if fold not in (outer_fold, calibration_fold)
        ),
        "calibration": sorted(run for run, fold in fold_map.items() if fold == calibration_fold),
        "test": sorted(run for run, fold in fold_map.items() if fold == outer_fold),
    }
    if [len(roles[key]) for key in ("fit", "calibration", "test")] != [60, 20, 20]:
        raise AssertionError("nested roles must be exactly 60/20/20")
    if set(roles["fit"]) & set(roles["calibration"]):
        raise AssertionError("fit/calibration roles overlap")
    if (set(roles["fit"]) | set(roles["calibration"])) & set(roles["test"]):
        raise AssertionError("outer test role overlaps")
    return roles


def _development_access_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        for idv in ALL_IDVS:
            rows.append(
                {
                    "idv": idv,
                    "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                    "simulation_run": run,
                    "window_half_open": list(
                        HEALTHY_SOURCE_WINDOW if idv == 1 else FAULT_SOURCE_WINDOW
                    ),
                    "role": "healthy_development" if idv == 1 else f"fault-{idv}_development",
                }
            )
    return rows


def _calibration_access_plan() -> list[dict[str, Any]]:
    return [
        {
            "idv": 1,
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "simulation_run": run,
            "window_half_open": list(HEALTHY_SOURCE_WINDOW),
            "role": "healthy_calibration_prefault_only",
        }
        for run in RUNS
    ]


def _confirmation_access_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        for idv in ALL_IDVS:
            rows.append(
                {
                    "idv": idv,
                    "magnitude_percent": CONFIRMATION_MAGNITUDE,
                    "simulation_run": run,
                    "window_half_open": list(
                        HEALTHY_SOURCE_WINDOW if idv == 1 else FAULT_SOURCE_WINDOW
                    ),
                    "role": "healthy_confirmation" if idv == 1 else f"fault-{idv}_confirmation",
                }
            )
    return rows


def _validate_access_plan(phase: str, value: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "development": _development_access_plan(),
        "calibrate": _calibration_access_plan(),
        "confirm": _confirmation_access_plan(),
    }
    if phase not in expected or list(value) != expected[phase]:
        raise ValueError(f"{phase} access plan differs from its exact allocation")
    for row in value:
        numeric = (
            row["idv"], row["magnitude_percent"], row["simulation_run"],
            *row["window_half_open"],
        )
        if any(type(item) is not int for item in numeric):
            raise ValueError("access plan numeric identities must be plain integers")
        if row["magnitude_percent"] == 25 and row["idv"] != 1:
            raise ValueError("magnitude-25 fault cells are quarantined")


def default_config() -> dict[str, Any]:
    return {
        "schema": "cf_max_glr15_complete_config",
        "schema_version": 1,
        "candidate": {"candidate_id": UNIQUE_CANDIDATE_ID, "candidate_count": 1},
        "features": {
            "schema": "cf_glr_mechanism_features_v2",
            "dimensions_by_fault": {"3": 20, "9": 20, "15": 28},
            "canonical_zero_half_open": [0, 128],
            "prefix_feature_half_open": [128, 160],
            "monitor_half_open": [160, 500],
            "causal_no_future_rows": True,
        },
        "experts": {
            "3": {"architecture": "linear", "hidden_width": 0, "seeds": [20260909]},
            "9": {"architecture": "linear", "hidden_width": 0, "seeds": [20260909]},
            "15": {
                "architecture": "gelu_mlp_28x16x1",
                "hidden_width": 16,
                "seeds": [20260909, 20260910, 20260911],
                "ensemble": "arithmetic_mean_endpoint_logits",
            },
        },
        "training": {
            "optimizer": "AdamW",
            "epochs": 80,
            "learning_rate": 0.001,
            "batching": "full_batch_per_expert_all_fit_healthy_and_matching_fault_bags",
            "weight_decay": 0.0,
            "gradient_clip": None,
            "dtype": "float64",
            "device": "cuda",
            "mil_endpoints_half_open_step": [160, 500, 4],
            "mil_top_fraction": 0.10,
            "mil_top_count": 9,
            "prefix_negative_endpoints_half_open_step": [128, 160, 4],
            "prefix_negative_endpoint_count": 8,
            "healthy_bag_bce_weight": 0.5,
            "fault_bag_bce_weight": 0.5,
            "prefix_negative_weight": 0.1,
            "l2_weight": 0.0001,
            "validation_or_early_stopping": False,
        },
        "monitor": {
            "fit": "per_expert_global_median_MAD_on_fit_healthy_monitored_points",
            "fusion": "U_t=max_k_standardized_logit",
            "persistence": "P_t=min(U_t_minus_1,U_t)",
            "run_maximum": "max_t_P_t",
            "inner_calibration_rank": 20,
            "inner_calibration_size": 20,
            "formal_calibration_rank": 98,
            "formal_calibration_size": 100,
            "comparison": "strictly_greater_than",
            "attribution": "argmax_expert_at_first_alarm_lowest_fault_id_tie",
        },
        "nested_development": {
            "fold_count": 5,
            "fold_salt": FOLD_SALT,
            "outer_test": "F_j",
            "inner_calibration": "F_(j+1)_mod_5_healthy_only",
            "fit": "remaining_three_folds",
            "fit_runs_per_class": 60,
            "calibration_healthy_runs": 20,
            "outer_test_runs_per_class": 20,
        },
        "development_gates": dict(DEVELOPMENT_GATES),
        "confirmation_gates": dict(CONFIRMATION_GATES),
    }


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("config must be an object")
    try:
        normalized = json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError("config must be finite canonical JSON") from error
    if _canonical_json_bytes(normalized) != _canonical_json_bytes(default_config()):
        raise ValueError("config differs from the exact audited v4 specification")
    return normalized


def _load_config(path: Path | None) -> dict[str, Any]:
    return validate_config(default_config() if path is None else _strict_json(path))


def _load_backend() -> ModuleType:
    if "numpy" in sys.modules and "torch" not in sys.modules:
        raise RuntimeError("numeric backend must import PyTorch before NumPy")
    root = _project_root()
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import importlib

    return importlib.import_module("mate_spc_cf_glr.backend_v4")


def _runtime_preflight(backend: NumericBackend) -> dict[str, Any]:
    value = dict(backend.cuda_preflight())
    required = {"cuda_available", "device_type", "dtype", "torch_version", "cuda_version", "device_name"}
    if set(value) != required:
        raise ValueError("CUDA preflight fields changed")
    if value["cuda_available"] is not True or value["device_type"] != "cuda" or value["dtype"] != "float64":
        raise RuntimeError("v4 requires available CUDA float64")
    return value


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("nearest-rank input is empty")
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def _finite_json_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_row(row: Mapping[str, Any], *, expected_idv: int, expected_run: int) -> dict[str, Any]:
    required = {
        "idv", "simulation_run", "run_maximum", "parity_max_abs_error",
        "alarm", "first_alarm_index", "attribution",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise ValueError("scored row fields changed")
    if type(row["idv"]) is not int or type(row["simulation_run"]) is not int:
        raise ValueError("scored identities must be exact integers")
    if row["idv"] != expected_idv or row["simulation_run"] != expected_run:
        raise ValueError("scored row identity differs from requested cell")
    maximum = _finite_json_number(row["run_maximum"], name="run maximum")
    parity = _finite_json_number(row["parity_max_abs_error"], name="row parity")
    if parity < 0.0 or parity > PARITY_ATOL:
        raise ValueError("scored row has invalid maximum/parity")
    if type(row["alarm"]) is not bool:
        raise ValueError("alarm must be a boolean")
    first = row["first_alarm_index"]
    attribution = row["attribution"]
    if row["alarm"]:
        if type(first) is not int or first not in range(MONITOR_START + 1, RUN_LENGTH):
            raise ValueError("alarmed row lacks a legal first alarm")
        if type(attribution) is not int or attribution not in FAULT_IDVS:
            raise ValueError("alarmed row lacks a legal expert attribution")
    elif first is not None or attribution is not None:
        raise ValueError("non-alarm row must not carry alarm/attribution")
    return {
        "idv": row["idv"], "simulation_run": row["simulation_run"],
        "run_maximum": maximum, "parity_max_abs_error": parity,
        "alarm": row["alarm"], "first_alarm_index": first, "attribution": attribution,
    }


def _development_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for fold in folds for row in fold["rows"]]
    healthy = [row for row in rows if row["idv"] == 1]
    faults: dict[str, Any] = {}
    gates: dict[str, bool] = {
        "calibration_nondegenerate": all(
            fold["calibration_nondegenerate"] is True for fold in folds
        ),
        "healthy_total": sum(row["alarm"] for row in healthy) <= DEVELOPMENT_GATES["healthy_alarm_runs_max"],
        "healthy_each_fold": all(
            sum(row["alarm"] for row in fold["rows"] if row["idv"] == 1)
            <= DEVELOPMENT_GATES["healthy_alarm_runs_per_fold_max"]
            for fold in folds
        ),
    }
    fold_fault_rates: list[float] = []
    for fault in FAULT_IDVS:
        selected = [row for row in rows if row["idv"] == fault]
        delays = [
            row["first_alarm_index"] - MONITOR_START if row["alarm"] else MISS_DELAY
            for row in selected
        ]
        detected = sum(row["alarm"] for row in selected)
        rate = detected / len(selected)
        median = float(statistics.median(delays))
        p90 = _nearest_rank(delays, 0.90)
        per_fold = []
        for fold in folds:
            chosen = [row for row in fold["rows"] if row["idv"] == fault]
            fold_rate = sum(row["alarm"] for row in chosen) / len(chosen)
            per_fold.append(fold_rate)
            fold_fault_rates.append(fold_rate)
        faults[str(fault)] = {
            "detected_runs": detected, "detection_rate": rate,
            "median_delay": median, "nearest_rank_p90_delay": p90,
            "fold_detection_rates": per_fold,
        }
        gates[f"fault_{fault}_overall"] = rate >= DEVELOPMENT_GATES["each_fault_detection_rate_min"]
        gates[f"fault_{fault}_folds"] = min(per_fold) >= DEVELOPMENT_GATES["each_fold_each_fault_detection_rate_min"]
        gates[f"fault_{fault}_median"] = median <= DEVELOPMENT_GATES["each_fault_median_delay_max"]
        gates[f"fault_{fault}_p90"] = p90 <= DEVELOPMENT_GATES["each_fault_nearest_rank_p90_delay_max"]
    return {
        "healthy_alarm_runs": sum(row["alarm"] for row in healthy),
        "faults": faults,
        "minimum_fold_by_fault_detection_rate": min(fold_fault_rates),
        "gate_checks": gates,
        "all_gates_pass": all(gates.values()),
    }


def _validate_development_result(
    raw: Mapping[str, Any], config: Mapping[str, Any], fold_map: Mapping[int, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "schema", "config", "candidate_id", "folds", "runtime",
        "selected_candidate_id", "model_metadata", "model_parity_max_abs_error",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("development backend result fields changed")
    if raw["schema"] != "cf_max_glr15_development_backend_v1":
        raise ValueError("development backend schema changed")
    if _canonical_json_bytes(raw["config"]) != _canonical_json_bytes(config):
        raise ValueError("backend echoed a different config")
    if raw["candidate_id"] != UNIQUE_CANDIDATE_ID:
        raise ValueError("v4 must evaluate exactly its unique candidate")
    folds = raw["folds"]
    if not isinstance(folds, list) or len(folds) != FOLD_COUNT:
        raise ValueError("development must contain five outer folds")
    normalized_folds: list[dict[str, Any]] = []
    for outer, fold in enumerate(folds):
        fields = {
            "outer_fold", "calibration_fold", "fit_run_ids_by_class",
            "calibration_healthy_run_ids", "excluded_calibration_fault_run_ids_by_class",
            "test_run_ids_by_class", "calibration_healthy_maxima", "threshold",
            "rank19", "distinct_maxima", "adjacent_tail_gap",
            "calibration_nondegenerate",
            "rows", "score_parity_max_abs_error", "discrete_parity_equal",
        }
        if not isinstance(fold, Mapping) or set(fold) != fields:
            raise ValueError("outer fold fields changed")
        roles = _fold_roles(outer, fold_map)
        expected_by_class_fit = {str(idv): roles["fit"] for idv in ALL_IDVS}
        expected_by_class_cal_fault = {str(idv): roles["calibration"] for idv in FAULT_IDVS}
        expected_by_class_test = {str(idv): roles["test"] for idv in ALL_IDVS}
        if (
            fold["outer_fold"] != outer
            or fold["calibration_fold"] != (outer + 1) % FOLD_COUNT
            or fold["fit_run_ids_by_class"] != expected_by_class_fit
            or fold["calibration_healthy_run_ids"] != roles["calibration"]
            or fold["excluded_calibration_fault_run_ids_by_class"] != expected_by_class_cal_fault
            or fold["test_run_ids_by_class"] != expected_by_class_test
        ):
            raise ValueError("outer fold role identities differ from canonical 60/20/20 split")
        maxima = fold["calibration_healthy_maxima"]
        if not isinstance(maxima, list) or len(maxima) != 20:
            raise ValueError("inner calibration must have 20 healthy maxima")
        parsed_maxima = [
            _finite_json_number(value, name="inner calibration maximum")
            for value in maxima
        ]
        threshold = _finite_json_number(fold["threshold"], name="inner threshold")
        if threshold != sorted(parsed_maxima)[INNER_CALIBRATION_RANK - 1]:
            raise ValueError("inner threshold must be rank20/max of 20 healthy runs")
        parity = _finite_json_number(
            fold["score_parity_max_abs_error"], name="outer fold parity"
        )
        if not 0.0 <= parity <= PARITY_ATOL:
            raise ValueError("outer fold parity failed")
        tolerance = max(FLOAT_TOLERANCE, parity)
        ordered_maxima = sorted(parsed_maxima)
        rank19 = _finite_json_number(fold["rank19"], name="inner rank19")
        distinct = fold["distinct_maxima"]
        gap = _finite_json_number(
            fold["adjacent_tail_gap"], name="inner adjacent tail gap"
        )
        observed_distinct = 1 + sum(
            current - previous > tolerance
            for previous, current in zip(ordered_maxima, ordered_maxima[1:])
        )
        nondegenerate = (
            type(distinct) is int
            and distinct == observed_distinct
            and rank19 == ordered_maxima[18]
            and gap == threshold - rank19
            and distinct >= 19
            and threshold > rank19
            and gap > 100.0 * tolerance
        )
        if fold["calibration_nondegenerate"] is not nondegenerate:
            raise ValueError("inner calibration degeneracy flag is inconsistent")
        if fold["discrete_parity_equal"] is not True:
            raise ValueError("outer fold CPU/CUDA alarm outputs differ")
        raw_rows = fold["rows"]
        if not isinstance(raw_rows, list) or len(raw_rows) != 80:
            raise ValueError("outer test must contain 20 runs per class")
        by_identity = {(row.get("idv"), row.get("simulation_run")): row for row in raw_rows if isinstance(row, Mapping)}
        if len(by_identity) != 80:
            raise ValueError("outer test row identities are duplicated/malformed")
        rows = [
            _validate_row(by_identity[(idv, run)], expected_idv=idv, expected_run=run)
            for run in roles["test"] for idv in ALL_IDVS
        ]
        normalized_folds.append({
            **dict(fold),
            "calibration_healthy_maxima": parsed_maxima,
            "threshold": threshold,
            "rank19": rank19,
            "distinct_maxima": distinct,
            "adjacent_tail_gap": gap,
            "calibration_nondegenerate": nondegenerate,
            "rows": rows,
            "score_parity_max_abs_error": parity,
        })
    metrics = _development_metrics(normalized_folds)
    selected = raw["selected_candidate_id"]
    if metrics["all_gates_pass"]:
        if selected != UNIQUE_CANDIDATE_ID or raw["model_metadata"] is None:
            raise ValueError("PASS backend did not select/save the unique candidate")
        model_parity = _finite_json_number(
            raw["model_parity_max_abs_error"], name="final refit parity"
        )
        if not 0.0 <= model_parity <= PARITY_ATOL:
            raise ValueError("final refit parity failed")
    elif selected is not None or raw["model_metadata"] is not None or raw["model_parity_max_abs_error"] is not None:
        raise ValueError("NO_GO backend must not expose a selected/saved model")
    normalized = dict(raw)
    normalized["folds"] = normalized_folds
    return normalized, metrics


def _iteration_dir(output_root: Path, iteration_id: str) -> Path:
    if not iteration_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in iteration_id):
        raise ValueError("iteration_id must use only letters, digits, hyphen, underscore")
    return output_root / "development_iterations" / iteration_id


def _write_access_intent(directory: Path, *, phase: str, source: Mapping[str, Any], plan: Sequence[Mapping[str, Any]], protocol_sha256: str | None = None) -> None:
    _write_json(
        directory / "access_intent.json",
        {
            "schema": "cf_max_glr15_numeric_access_intent_v1",
            "phase": phase,
            "written_before_numeric_access": True,
            "source": dict(source),
            "protocol_sha256": protocol_sha256,
            "access_plan": list(plan),
        },
    )


def _completion(directory: Path, *, phase: str, names: Sequence[str], binding_sha256: str) -> None:
    artifacts = {name: _file_record(directory / name) for name in names}
    _write_json(
        directory / "completion.json",
        {
            "schema": "cf_max_glr15_phase_completion_v1",
            "phase": phase,
            "binding_sha256": binding_sha256,
            "artifacts": artifacts,
        },
    )


def run_development_iteration(
    *, data_dir: Path, output_root: Path, iteration_id: str,
    config_path: Path | None = None, backend: NumericBackend | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    root = _project_root()
    source = _source_record(data_dir)
    implementation = _implementation_records(root)
    fold_map = _fold_assignment()
    plan = _development_access_plan()
    _validate_access_plan("development", plan)
    directory = _iteration_dir(output_root, iteration_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    _write_json(directory / "claim.json", {"schema": "cf_max_glr15_development_claim_v1", "iteration_id": iteration_id, "claimed_at_utc": _now()})
    _write_json(directory / "config.json", config)
    manifest = {
        "schema": "cf_max_glr15_iteration_manifest_v1",
        "iteration_id": iteration_id,
        "config_sha256": _config_sha256(config),
        "source": source,
        "implementation": implementation,
        "fold_map": {str(key): value for key, value in fold_map.items()},
        "candidate_id": UNIQUE_CANDIDATE_ID,
    }
    _write_json(directory / "iteration_manifest.json", manifest)
    _write_access_intent(directory, phase="development", source=source, plan=plan)
    numeric = _load_backend() if backend is None else backend
    runtime = _runtime_preflight(numeric)
    _write_json(directory / "runtime.json", runtime)
    model_path = directory / "model.npz"
    monitor_path = directory / "monitor.json"
    raw = numeric.development_iteration(
        source_path=_source_path(data_dir), access_plan=plan, fold_map=fold_map,
        config=config, model_path=model_path, monitor_path=monitor_path,
        device="cuda", dtype="float64",
    )
    normalized, metrics = _validate_development_result(raw, config, fold_map)
    decision = "DEVELOPMENT_PASS" if metrics["all_gates_pass"] else "DEVELOPMENT_NO_GO"
    if decision == "DEVELOPMENT_PASS":
        if not model_path.is_file() or not monitor_path.is_file():
            raise FileNotFoundError("PASS requires safe model and monitor artifacts")
        numeric.verify_bound_artifacts(model_path=model_path, monitor_path=monitor_path, config=config)
    elif model_path.exists() or monitor_path.exists():
        raise ValueError("NO_GO must not create model/monitor artifacts")
    result = {**normalized, "metrics": metrics, "decision": decision}
    _write_json(directory / "development_result.json", result)
    names = ["claim.json", "config.json", "iteration_manifest.json", "access_intent.json", "runtime.json", "development_result.json"]
    if decision == "DEVELOPMENT_PASS":
        names.extend(["model.npz", "monitor.json"])
    _completion(directory, phase="development-iteration", names=names, binding_sha256=_sha256_bytes(_canonical_json_bytes(manifest)))
    return result


def _verify_iteration(output_root: Path, iteration_id: str, backend: NumericBackend) -> tuple[dict[str, Any], dict[str, Any], Path]:
    directory = _iteration_dir(output_root, iteration_id)
    completion = _strict_json(directory / "completion.json")
    if completion.get("schema") != "cf_max_glr15_phase_completion_v1" or completion.get("phase") != "development-iteration":
        raise ValueError("development completion receipt changed")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("development artifact receipt is malformed")
    for name, expected in artifacts.items():
        if _file_record(directory / name) != expected:
            raise ValueError(f"development artifact changed: {name}")
    config = validate_config(_strict_json(directory / "config.json"))
    manifest = _strict_json(directory / "iteration_manifest.json")
    manifest_fields = {
        "schema", "iteration_id", "config_sha256", "source",
        "implementation", "fold_map", "candidate_id",
    }
    if (
        set(manifest) != manifest_fields
        or manifest["schema"] != "cf_max_glr15_iteration_manifest_v1"
        or manifest["iteration_id"] != iteration_id
        or manifest["config_sha256"] != _config_sha256(config)
        or manifest["candidate_id"] != UNIQUE_CANDIDATE_ID
        or completion.get("binding_sha256")
        != _sha256_bytes(_canonical_json_bytes(manifest))
    ):
        raise ValueError("development manifest semantic binding changed")
    try:
        fold_map = {
            int(key): value for key, value in manifest["fold_map"].items()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("development fold map is malformed") from error
    if fold_map != _fold_assignment():
        raise ValueError("development fold map differs from the canonical split")
    result = _strict_json(directory / "development_result.json")
    backend_fields = {
        "schema", "config", "candidate_id", "folds", "runtime",
        "selected_candidate_id", "model_metadata", "model_parity_max_abs_error",
    }
    if not backend_fields.issubset(result):
        raise ValueError("development result lacks backend fields")
    replayed, replayed_metrics = _validate_development_result(
        {key: result[key] for key in backend_fields}, config, fold_map
    )
    expected_decision = (
        "DEVELOPMENT_PASS"
        if replayed_metrics["all_gates_pass"]
        else "DEVELOPMENT_NO_GO"
    )
    if (
        result.get("decision") != expected_decision
        or result.get("metrics") != replayed_metrics
        or any(result[key] != replayed[key] for key in backend_fields)
    ):
        raise ValueError("development result fails deterministic semantic replay")
    if expected_decision != "DEVELOPMENT_PASS":
        raise ValueError("only an audited PASS iteration can be frozen")
    verified = backend.verify_bound_artifacts(
        model_path=directory / "model.npz",
        monitor_path=directory / "monitor.json",
        config=config,
    )
    if (
        isinstance(verified, Mapping)
        and "model_metadata" in verified
        and verified["model_metadata"] != result["model_metadata"]
    ):
        raise ValueError("development result metadata differs from the bound model")
    return config, result, directory


def freeze_formal(
    *, data_dir: Path, output_root: Path, iteration_id: str,
    protocol_path: Path, backend: NumericBackend | None = None,
) -> str:
    freeze_claim = output_root / ".formal_freeze.claim.json"
    numeric = _load_backend() if backend is None else backend
    _runtime_preflight(numeric)
    config, result, directory = _verify_iteration(output_root, iteration_id, numeric)
    source = _source_record(data_dir)
    manifest = _strict_json(directory / "iteration_manifest.json")
    if manifest.get("source") != source or manifest.get("implementation") != _implementation_records(_project_root()):
        raise ValueError("live source/implementation differs from PASS iteration")
    protocol = {
        "schema": "cf_max_glr15_formal_protocol_v1",
        "created_at_utc": _now(),
        "output_root_absolute": str(output_root.resolve()),
        "source": source,
        "implementation": manifest["implementation"],
        "config": config,
        "config_sha256": _config_sha256(config),
        "development_binding": {
            "iteration_id": iteration_id,
            "decision": result["decision"],
            "candidate_id": UNIQUE_CANDIDATE_ID,
            "model": _file_record(directory / "model.npz"),
            "monitor": _file_record(directory / "monitor.json"),
            "development_result": _file_record(directory / "development_result.json"),
        },
        "partitions": {
            "development": "Mode1 magnitude50 IDV1/3/9/15 runs1-100 consumed/repeatable",
            "calibration": "Mode1 magnitude25 IDV1 pre-fault runs1-100 one-time",
            "quarantined": "Mode1 magnitude25 IDV3/9/15 never accessed",
            "confirmation": "Mode1 magnitude75 IDV1/3/9/15 runs1-100 one-time",
            "magnitude100": "already consumed diagnostic-only and prohibited",
        },
        "calibration_rule": {
            "rank": 98, "sample_size": 100, "comparison": "strictly_greater_than",
            "finite_sample_marginal_exceedance_bound": "3/101",
        },
        "development_gates": dict(DEVELOPMENT_GATES),
        "confirmation_gates": dict(CONFIRMATION_GATES),
    }
    # Claim only after every read-only audit has passed. A typo, stale source,
    # or invalid PASS receipt must not irreversibly consume the one formal slot.
    _claim_file(
        freeze_claim,
        {
            "schema": "cf_max_glr15_formal_freeze_claim_v1",
            "iteration_id": iteration_id,
            "protocol_absolute": str(protocol_path.resolve()),
            "claimed_at_utc": _now(),
        },
    )
    return _publish_protocol(protocol_path, protocol)


def _assert_live_protocol(protocol_path: Path, data_dir: Path, output_root: Path) -> tuple[dict[str, Any], str, Path, Path]:
    protocol, digest = _load_protocol(protocol_path)
    if protocol.get("schema") != "cf_max_glr15_formal_protocol_v1":
        raise ValueError("formal protocol schema changed")
    if protocol.get("output_root_absolute") != str(output_root.resolve()):
        raise ValueError("formal protocol is bound to another output root")
    if protocol.get("source") != _source_record(data_dir):
        raise ValueError("formal source changed")
    if protocol.get("implementation") != _implementation_records(_project_root()):
        raise ValueError("formal implementation changed")
    config = validate_config(protocol.get("config", {}))
    if protocol.get("config_sha256") != _config_sha256(config):
        raise ValueError("formal config binding changed")
    iteration = protocol["development_binding"]["iteration_id"]
    directory = _iteration_dir(output_root, iteration)
    for key, name in (("model", "model.npz"), ("monitor", "monitor.json"), ("development_result", "development_result.json")):
        if protocol["development_binding"][key] != _file_record(directory / name):
            raise ValueError(f"bound development {key} changed")
    return protocol, digest, directory / "model.npz", directory / "monitor.json"


def _claim_phase(output_root: Path, phase: str, protocol_sha256: str) -> Path:
    directory = output_root / phase
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    _write_json(directory / "claim.json", {"schema": "cf_max_glr15_formal_phase_claim_v1", "phase": phase, "protocol_sha256": protocol_sha256, "claimed_at_utc": _now()})
    return directory


def _validate_calibration(raw: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"schema", "config", "rows", "model_parity_max_abs_error", "discrete_parity_equal"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw["schema"] != "cf_max_glr15_calibration_backend_v1":
        raise ValueError("calibration backend result changed")
    if _canonical_json_bytes(raw["config"]) != _canonical_json_bytes(config):
        raise ValueError("calibration echoed a different config")
    if raw["discrete_parity_equal"] is not True:
        raise ValueError("calibration CPU/CUDA ordering differs")
    parity = _finite_json_number(
        raw["model_parity_max_abs_error"], name="calibration parity"
    )
    if not 0.0 <= parity <= PARITY_ATOL:
        raise ValueError("calibration parity failed")
    rows = raw["rows"]
    expected_fields = {"idv", "simulation_run", "run_maximum", "parity_max_abs_error"}
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("calibration must return 100 rows")
    by_run: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields or type(row["idv"]) is not int or row["idv"] != 1 or type(row["simulation_run"]) is not int:
            raise ValueError("calibration row identity/fields changed")
        maximum = _finite_json_number(
            row["run_maximum"], name="calibration run maximum"
        )
        error = _finite_json_number(
            row["parity_max_abs_error"], name="calibration row parity"
        )
        if not 0 <= error <= PARITY_ATOL:
            raise ValueError("calibration row is nonfinite or violates parity")
        if row["simulation_run"] in by_run:
            raise ValueError("duplicate calibration run")
        by_run[row["simulation_run"]] = {**dict(row), "run_maximum": maximum, "parity_max_abs_error": error}
    if set(by_run) != set(RUNS):
        raise ValueError("calibration run identities differ")
    ordered_rows = [by_run[run] for run in RUNS]
    maxima = sorted(row["run_maximum"] for row in ordered_rows)
    tolerance = max(FLOAT_TOLERANCE, parity)
    distinct = 1 + sum(current - previous > tolerance for previous, current in zip(maxima, maxima[1:]))
    threshold = maxima[FORMAL_CALIBRATION_RANK - 1]
    checks = {
        "finite_100": len(maxima) == 100 and all(math.isfinite(value) for value in maxima),
        "distinct_at_least_96": distinct >= 96,
        "rank98_below_rank100": threshold < maxima[99],
    }
    summary = {
        "decision": "PASS" if all(checks.values()) else "INVALID_CALIBRATION",
        "threshold": threshold,
        "rank": FORMAL_CALIBRATION_RANK,
        "sample_size": 100,
        "comparison": "strictly_greater_than",
        "finite_sample_marginal_exceedance_bound": "3/101",
        "distinct_count": distinct,
        "checks": checks,
    }
    normalized = dict(raw)
    normalized["rows"] = ordered_rows
    normalized["model_parity_max_abs_error"] = parity
    return normalized, summary


def run_calibration(
    *, data_dir: Path, output_root: Path, protocol_path: Path,
    backend: NumericBackend | None = None,
) -> dict[str, Any]:
    protocol, digest, model_path, monitor_path = _assert_live_protocol(protocol_path, data_dir, output_root)
    directory = _claim_phase(output_root, "calibrate", digest)
    plan = _calibration_access_plan()
    _validate_access_plan("calibrate", plan)
    _write_access_intent(directory, phase="calibrate", source=protocol["source"], plan=plan, protocol_sha256=digest)
    numeric = _load_backend() if backend is None else backend
    runtime = _runtime_preflight(numeric)
    numeric.verify_bound_artifacts(model_path=model_path, monitor_path=monitor_path, config=protocol["config"])
    _write_json(directory / "runtime.json", runtime)
    raw = numeric.calibrate(source_path=_source_path(data_dir), access_plan=plan, model_path=model_path, monitor_path=monitor_path, config=protocol["config"], device="cuda", dtype="float64")
    normalized, summary = _validate_calibration(raw, protocol["config"])
    result = {**normalized, "summary": summary, "protocol_sha256": digest}
    _write_json(directory / "calibration_result.json", result)
    _completion(directory, phase="calibrate", names=["claim.json", "access_intent.json", "runtime.json", "calibration_result.json"], binding_sha256=digest)
    return result


def _verified_calibration(output_root: Path, protocol_sha256: str) -> tuple[dict[str, Any], float]:
    directory = output_root / "calibrate"
    completion = _strict_json(directory / "completion.json")
    if completion.get("phase") != "calibrate" or completion.get("binding_sha256") != protocol_sha256:
        raise ValueError("calibration completion binding changed")
    for name, record in completion["artifacts"].items():
        if _file_record(directory / name) != record:
            raise ValueError(f"calibration artifact changed: {name}")
    result = _strict_json(directory / "calibration_result.json")
    if result.get("protocol_sha256") != protocol_sha256:
        raise ValueError("calibration result protocol binding changed")
    if result.get("summary", {}).get("decision") != "PASS":
        raise RuntimeError("invalid calibration is consumed and forbids confirmation")
    threshold = _finite_json_number(
        result["summary"]["threshold"], name="calibration threshold"
    )
    return result, threshold


def _confirmation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    healthy = [row for row in rows if row["idv"] == 1]
    faults: dict[str, Any] = {}
    checks = {"healthy": sum(row["alarm"] for row in healthy) <= CONFIRMATION_GATES["healthy_alarm_runs_max"]}
    for fault in FAULT_IDVS:
        chosen = [row for row in rows if row["idv"] == fault]
        detected = sum(row["alarm"] for row in chosen)
        delays = [row["first_alarm_index"] - MONITOR_START if row["alarm"] else MISS_DELAY for row in chosen]
        median = float(statistics.median(delays))
        p90 = _nearest_rank(delays, 0.90)
        faults[str(fault)] = {"detected_runs": detected, "detection_rate": detected / 100.0, "median_delay": median, "nearest_rank_p90_delay": p90}
        checks[f"fault_{fault}_detected"] = detected >= CONFIRMATION_GATES["each_fault_detected_runs_min"]
        checks[f"fault_{fault}_median"] = median <= CONFIRMATION_GATES["each_fault_median_delay_max"]
        checks[f"fault_{fault}_p90"] = p90 <= CONFIRMATION_GATES["each_fault_nearest_rank_p90_delay_max"]
    return {"healthy_alarm_runs": sum(row["alarm"] for row in healthy), "faults": faults, "gate_checks": checks, "all_gates_pass": all(checks.values())}


def run_confirmation(
    *, data_dir: Path, output_root: Path, protocol_path: Path,
    backend: NumericBackend | None = None,
) -> dict[str, Any]:
    protocol, digest, model_path, monitor_path = _assert_live_protocol(protocol_path, data_dir, output_root)
    _, threshold = _verified_calibration(output_root, digest)
    directory = _claim_phase(output_root, "confirm", digest)
    plan = _confirmation_access_plan()
    _validate_access_plan("confirm", plan)
    _write_access_intent(directory, phase="confirm", source=protocol["source"], plan=plan, protocol_sha256=digest)
    numeric = _load_backend() if backend is None else backend
    runtime = _runtime_preflight(numeric)
    numeric.verify_bound_artifacts(model_path=model_path, monitor_path=monitor_path, config=protocol["config"])
    _write_json(directory / "runtime.json", runtime)
    raw = numeric.confirm(source_path=_source_path(data_dir), access_plan=plan, model_path=model_path, monitor_path=monitor_path, config=protocol["config"], threshold=threshold, device="cuda", dtype="float64")
    required = {"schema", "config", "threshold", "rows", "model_parity_max_abs_error", "discrete_parity_equal"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw["schema"] != "cf_max_glr15_confirmation_backend_v1":
        raise ValueError("confirmation backend result changed")
    raw_threshold = _finite_json_number(
        raw["threshold"], name="confirmation threshold"
    )
    if _canonical_json_bytes(raw["config"]) != _canonical_json_bytes(protocol["config"]) or raw_threshold != threshold or raw["discrete_parity_equal"] is not True:
        raise ValueError("confirmation config/threshold/parity binding changed")
    parity = _finite_json_number(
        raw["model_parity_max_abs_error"], name="confirmation parity"
    )
    if not 0 <= parity <= PARITY_ATOL:
        raise ValueError("confirmation parity failed")
    raw_rows = raw["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 400:
        raise ValueError("confirmation must return 400 rows")
    by_identity = {(row.get("idv"), row.get("simulation_run")): row for row in raw_rows if isinstance(row, Mapping)}
    if len(by_identity) != 400:
        raise ValueError("confirmation row identities duplicated/malformed")
    rows = [_validate_row(by_identity[(idv, run)], expected_idv=idv, expected_run=run) for run in RUNS for idv in ALL_IDVS]
    metrics = _confirmation_metrics(rows)
    result = {**dict(raw), "rows": rows, "model_parity_max_abs_error": parity, "metrics": metrics, "decision": "PASS" if metrics["all_gates_pass"] else "FAIL", "protocol_sha256": digest}
    _write_json(directory / "confirmation_result.json", result)
    _completion(directory, phase="confirm", names=["claim.json", "access_intent.json", "runtime.json", "confirmation_result.json"], binding_sha256=digest)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("development-iteration", "freeze-formal", "calibrate", "confirm"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--iteration-id")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_FORMAL_PROTOCOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "development-iteration":
        if not args.iteration_id:
            raise ValueError("development-iteration requires --iteration-id")
        result = run_development_iteration(data_dir=args.data_dir, output_root=args.output_root, iteration_id=args.iteration_id, config_path=args.config)
        print(result["decision"])
    elif args.command == "freeze-formal":
        if not args.iteration_id:
            raise ValueError("freeze-formal requires --iteration-id")
        print(freeze_formal(data_dir=args.data_dir, output_root=args.output_root, iteration_id=args.iteration_id, protocol_path=args.protocol))
    elif args.command == "calibrate":
        print(run_calibration(data_dir=args.data_dir, output_root=args.output_root, protocol_path=args.protocol)["summary"]["decision"])
    else:
        print(run_confirmation(data_dir=args.data_dir, output_root=args.output_root, protocol_path=args.protocol)["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
