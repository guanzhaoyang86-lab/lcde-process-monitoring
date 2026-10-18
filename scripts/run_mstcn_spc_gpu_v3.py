#!/usr/bin/env python3
"""Fail-closed GPU governance runner for MSCD-TCN-SPC v3.

The command surface deliberately separates repeatable method development from
the irreversible formal audit::

    development-iteration  (Mode 1, 50%, repeatable)
    freeze-formal          (metadata/hash only)
    calibrate              (Mode 1, 25%, IDV1 pre-fault only, once)
    confirm                (Mode 1, 75%, IDV1/3/9/15, once)

The runner owns partition governance, immutable protocol publication, phase
claims, artifact binding, configuration validation, threshold construction,
and metric replay.  Numeric HDF5 access and model computation are delayed
behind an injectable backend.  Importing this file never opens an HDF5 file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


SCHEMA_VERSION = 1
METHOD = "Multi-Scale Causal Dynamics TCN with top-tail sequential scan"
METHOD_SHORT = "MSCD-TCN-SPC-v3"
DATA_FILE = "TEP_Mode1.h5"
MODE1_SIZE_BYTES = 23_924_710_848
MODE1_MD5 = "c79ecc6d12a5ccb49caffcd8f531feae"
MODE1_SHA256 = "d6b0ed0ef3974ff571afd6717ba0418a5b8eb58a7822c84a05735572d52b3294"

DESIGN_DOCUMENT = "docs/mstcn_spc_v3.md"
IMPLEMENTATION_FILES = (
    DESIGN_DOCUMENT,
    "scripts/run_mstcn_spc_gpu_v3.py",
    "src/mate_spc_mstcn/features.py",
    "src/mate_spc_mstcn/model.py",
    "src/mate_spc_mstcn/monitor.py",
    "src/mate_spc_mstcn/backend_v3.py",
)

ALL_IDS = (1, 3, 9, 15)
FAULT_IDS = (3, 9, 15)
DEVELOPMENT_MAGNITUDE = 50
CALIBRATION_MAGNITUDE = 25
CONFIRMATION_MAGNITUDE = 75
CONSUMED_MAGNITUDE = 100
DEVELOPMENT_RUNS = tuple(range(1, 101))
CALIBRATION_RUNS = tuple(range(1, 101))
CONFIRMATION_RUNS = tuple(range(1, 101))
CONSUMED_RUNS = tuple(range(1, 201))

SOURCE_HEALTHY_WINDOW = (100, 600)
SOURCE_FAULT_WINDOW = (440, 940)
SEQUENCE_LENGTH = 500
BASELINE_WINDOW = (0, 128)
MONITORED_WINDOW = (160, 500)
FAULT_START = 160
MISS_DELAY = 340
PERSISTENCE = 2
FOLD_COUNT = 5
FOLD_SALT = "mstcn-spc-v3-mode1-mag50-block-folds-2026-09-08"

PARITY_ATOL = 1.0e-9
FLOAT_TOLERANCE = 1.0e-12
CALIBRATION_RANK = 98
CALIBRATION_EPSILON = 0.10
CALIBRATION_DELTA = 0.01
MIN_UNIQUE_MAXIMA = 96
TOP_TAIL_LOWER_RANK = 96

DEVELOPMENT_GATES = {
    "maximum_oof_healthy_alarm_runs": 10,
    "maximum_oof_healthy_alarm_runs_each_fold": 4,
    "minimum_detection_rate_each_fault": 0.85,
    "minimum_detection_rate_each_fold_fault": 0.70,
    "maximum_median_delay_samples_each_fault": 140.0,
    "maximum_p90_delay_samples_each_fault": 220.0,
}
CONFIRMATION_GATES = {
    "maximum_healthy_alarm_runs": 4,
    "minimum_detected_runs_each_fault": 90,
    "minimum_detected_runs_total": 270,
    "maximum_median_delay_samples_each_fault": 140.0,
    "maximum_p90_delay_samples_each_fault": 220.0,
    "miss_delay_samples": MISS_DELAY,
}

DEFAULT_OUTPUT_ROOT = Path("outputs/mstcn_spc_v3")
DEFAULT_FORMAL_PROTOCOL = Path("protocols/mstcn_spc_formal_v3.json")


@dataclass(frozen=True)
class BundlePaths:
    protocol: Path
    sidecar: Path
    commit: Path


class NumericBackend(Protocol):
    """Minimal high-level interface required from ``backend_v3``.

    A backend may be injected in tests.  The production backend must postpone
    opening ``source_path`` until the runner has durably written access intent.
    """

    def cuda_preflight(self) -> Mapping[str, Any]: ...

    def development_iteration(
        self,
        *,
        source_path: Path,
        access_plan: Sequence[Mapping[str, Any]],
        fold_map: Mapping[int, int],
        config: Mapping[str, Any],
        model_path: Path,
        device: str,
        dtype: str,
    ) -> Mapping[str, Any]: ...

    def calibrate(
        self,
        *,
        source_path: Path,
        access_plan: Sequence[Mapping[str, Any]],
        model_path: Path,
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
        config: Mapping[str, Any],
        threshold: float,
        device: str,
        dtype: str,
    ) -> Mapping[str, Any]: ...


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hash_record(path: Path, *, include_md5: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required regular file is absent: {path}")
    record: dict[str, Any] = {
        "size_bytes": int(path.stat().st_size),
        "sha256": _hash_file(path, "sha256"),
    }
    if include_md5:
        record["md5"] = _hash_file(path, "md5")
    return record


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {token}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(raw)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def _require_sha256(value: str, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _bundle_paths(protocol_path: Path) -> BundlePaths:
    protocol = protocol_path.resolve()
    return BundlePaths(
        protocol,
        protocol.with_name(protocol.name + ".sha256"),
        protocol.with_name(protocol.name + ".commit.json"),
    )


def _publish_protocol_bundle(protocol_path: Path, payload: Mapping[str, Any]) -> str:
    """Publish protocol, sidecar, then commit marker under an exclusive claim."""

    bundle = _bundle_paths(protocol_path)
    bundle.protocol.parent.mkdir(parents=True, exist_ok=True)
    claim = bundle.protocol.with_name(bundle.protocol.name + ".freeze-claim")
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError("formal freeze is already claimed") from error
    os.close(descriptor)
    temporary: list[Path] = []
    try:
        if any(path.exists() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
            raise FileExistsError("formal protocol bundle exists or is partial; refusing overwrite")
        protocol_bytes = _canonical_json_bytes(payload)
        protocol_sha = _sha256_bytes(protocol_bytes)
        sidecar_bytes = f"{protocol_sha}  {bundle.protocol.name}\n".encode("utf-8")
        commit = {
            "schema": "mstcn_spc_protocol_atomic_commit",
            "schema_version": SCHEMA_VERSION,
            "protocol_name": bundle.protocol.name,
            "protocol_sha256": protocol_sha,
            "sidecar_name": bundle.sidecar.name,
            "sidecar_sha256": _sha256_bytes(sidecar_bytes),
        }
        contents = (
            (bundle.protocol, protocol_bytes),
            (bundle.sidecar, sidecar_bytes),
            (bundle.commit, _canonical_json_bytes(commit)),
        )
        for destination, content in contents:
            descriptor, raw = tempfile.mkstemp(
                dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
            )
            temp = Path(raw)
            temporary.append(temp)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for destination in (bundle.protocol, bundle.sidecar, bundle.commit):
            os.replace(temporary.pop(0), destination)
        return protocol_sha
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
        claim.unlink(missing_ok=True)


def _load_protocol(
    protocol_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    expected = _require_sha256(expected_sha256, label="protocol SHA-256")
    bundle = _bundle_paths(protocol_path)
    if not all(path.is_file() for path in (bundle.protocol, bundle.sidecar, bundle.commit)):
        raise ValueError("formal protocol bundle is absent or partially published")
    actual = _hash_file(bundle.protocol)
    if actual != expected:
        raise ValueError("formal protocol digest does not match the required digest")
    if bundle.sidecar.read_text(encoding="utf-8") != f"{actual}  {bundle.protocol.name}\n":
        raise ValueError("formal protocol sidecar is invalid")
    commit = _strict_json(bundle.commit)
    expected_commit = {
        "schema": "mstcn_spc_protocol_atomic_commit",
        "schema_version": SCHEMA_VERSION,
        "protocol_name": bundle.protocol.name,
        "protocol_sha256": actual,
        "sidecar_name": bundle.sidecar.name,
        "sidecar_sha256": _hash_file(bundle.sidecar),
    }
    if commit != expected_commit:
        raise ValueError("formal protocol commit marker is invalid")
    protocol = _strict_json(bundle.protocol)
    if protocol.get("schema") != "mstcn_spc_formal_protocol" or protocol.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        raise ValueError("unexpected formal protocol schema")
    return protocol, actual


def _implementation_records(project_root: Path) -> dict[str, dict[str, Any]]:
    return {
        relative: _hash_record(project_root.resolve() / relative)
        for relative in IMPLEMENTATION_FILES
    }


def _source_path(data_dir: Path) -> Path:
    return data_dir.resolve() / DATA_FILE


def _source_record(data_dir: Path) -> dict[str, Any]:
    source = _source_path(data_dir)
    record = {
        "name": DATA_FILE,
        "absolute_path": str(source),
        **_hash_record(source, include_md5=True),
    }
    if (
        record["size_bytes"] != MODE1_SIZE_BYTES
        or record["sha256"] != MODE1_SHA256
        or record["md5"] != MODE1_MD5
    ):
        raise ValueError("source is not the byte-exact official DTU Mode 1 archive")
    return record


def _source_metadata_record(data_dir: Path) -> dict[str, Any]:
    """Validate labels/topology only; the adapter contract forbids numeric decode."""

    from mate_spc_pclr.data_v2 import PCLRMode1Archive

    summary = PCLRMode1Archive(_source_path(data_dir)).validate_topology()
    return {
        "labels": list(summary.labels),
        "selected_run_count": int(summary.selected_run_count),
        "counts_by_partition": dict(summary.counts_by_partition),
        "counts_by_status": dict(summary.counts_by_status),
        "counts_by_magnitude": {
            str(key): int(value) for key, value in summary.counts_by_magnitude.items()
        },
        "unique_seed_count": int(summary.unique_seed_count),
        "seed_sets_pairwise_disjoint": bool(summary.seed_sets_pairwise_disjoint),
        "run_seed_attributes_decoded": True,
        "processdata_or_idv_init_values_decoded": False,
    }


def _fold_assignment() -> dict[int, int]:
    ranked = sorted(
        DEVELOPMENT_RUNS,
        key=lambda run: hashlib.sha256(f"{FOLD_SALT}|{run}".encode("ascii")).hexdigest(),
    )
    assignment = {run: index % FOLD_COUNT for index, run in enumerate(ranked)}
    if [sum(value == fold for value in assignment.values()) for fold in range(FOLD_COUNT)] != [
        20
    ] * FOLD_COUNT:
        raise AssertionError("development folds must contain 20 balanced blocks each")
    return dict(sorted(assignment.items()))


def _development_access_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in DEVELOPMENT_RUNS:
        rows.append(
            {
                "idv": 1,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "simulation_run": run,
                "window_half_open": list(SOURCE_HEALTHY_WINDOW),
                "role": "healthy_development",
            }
        )
        rows.extend(
            {
                "idv": fault,
                "magnitude_percent": DEVELOPMENT_MAGNITUDE,
                "simulation_run": run,
                "window_half_open": list(SOURCE_FAULT_WINDOW),
                "role": f"fault-{fault}_development",
            }
            for fault in FAULT_IDS
        )
    return rows


def _calibration_access_plan() -> list[dict[str, Any]]:
    return [
        {
            "idv": 1,
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "simulation_run": run,
            "window_half_open": list(SOURCE_HEALTHY_WINDOW),
            "role": "healthy_calibration_prefault_only",
        }
        for run in CALIBRATION_RUNS
    ]


def _confirmation_access_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in CONFIRMATION_RUNS:
        rows.append(
            {
                "idv": 1,
                "magnitude_percent": CONFIRMATION_MAGNITUDE,
                "simulation_run": run,
                "window_half_open": list(SOURCE_HEALTHY_WINDOW),
                "role": "healthy_confirmation",
            }
        )
        rows.extend(
            {
                "idv": fault,
                "magnitude_percent": CONFIRMATION_MAGNITUDE,
                "simulation_run": run,
                "window_half_open": list(SOURCE_FAULT_WINDOW),
                "role": f"fault-{fault}_confirmation",
            }
            for fault in FAULT_IDS
        )
    return rows


def _validate_access_plan(phase: str, plan: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        "development": _development_access_plan,
        "calibrate": _calibration_access_plan,
        "confirm": _confirmation_access_plan,
    }
    if phase not in expected or list(plan) != expected[phase]():
        raise ValueError(f"{phase} access plan differs from the hard-coded partition")
    for row in plan:
        magnitude = int(row["magnitude_percent"])
        if magnitude == CONSUMED_MAGNITUDE:
            raise ValueError("100% cells are diagnostic-only and forbidden to v3 fitting/audit")
        if magnitude == CALIBRATION_MAGNITUDE and int(row["idv"]) != 1:
            raise ValueError("25% fault cells are quarantined and forbidden")


def default_model_config() -> dict[str, Any]:
    """Return the exact four-candidate configuration approved by method audit."""

    q_values = [0.125, 0.25]
    rho_values = [0.125, 0.25]
    return {
        "schema": "mstcn_spc_complete_model_config",
        "schema_version": SCHEMA_VERSION,
        "feature_transform": {
            "feature_schema": "mstcn_causal_haar_loop_v2",
            "input_features": 512,
            "commissioning_baseline_half_open": [0, 128],
            "fit_scope": "fold_training_run_prefixes_only",
            "causal_no_future_rows": True,
        },
        "architecture": {
            "window_length": 128,
            "channels": 32,
            "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16],
            "pool_scales": [8, 32, 96],
            "pooling": "TCN_endpoint_state_plus_trailing_mean_RMS_of_1x1_pointwise_projection",
            "hidden_size": 64,
            "detector_output_count": 1,
            "attribution_class_ids": [3, 9, 15],
        },
        "endpoints": {
            "dense_half_open_step": [160, 500, 1],
            "mil_half_open_step": [160, 500, 4],
        },
        "training": {
            "seed": 20260908,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 30,
            "bag_batch_size": 8,
            "class_balanced_runs_per_class_per_batch": 2,
            "examples_per_class_per_epoch_cap": 100,
            "fold_training_runs_per_class": 80,
            "final_refit_runs_per_class": 100,
            "gradient_clip_norm": 1.0,
            "healthy_detection_loss_weight": 0.5,
            "each_fault_detection_loss_weight": 0.16666666666666666,
            "attribution_loss_weight": 0.25,
            "mil_top_fractions_q": q_values,
            "models_per_q_per_fold": 1,
            "validation_early_stopping": False,
            "device": "cuda",
            "dtype": "float64",
        },
        "scan": {
            "evidence": "detector_logit",
            "window_samples_W": [16, 32, 64, 128],
            "tail_fractions_rho": rho_values,
            "pointwise_standardizer": "robust_healthy_training_only",
            "per_window_studentizer": "robust_healthy_training_only",
            "aggregation": "maximum_over_W_of_trailing_top_rho_studentized_mean",
            "persistence_samples": 2,
            "threshold_comparison": "strictly_greater_than",
            "attribution": "argmax_cumulative_three_class_logits_through_first_alarm",
        },
        "candidate_grid": [
            {
                "candidate_id": f"q{int(q * 1000):03d}-rho{int(rho * 1000):03d}",
                "mil_top_fraction_q": q,
                "scan_tail_fraction_rho": rho,
            }
            for q in q_values
            for rho in rho_values
        ],
        "selection_order": [
            "maximize_minimum_fault_detection_rate",
            "maximize_minimum_fold_by_fault_detection_rate",
            "minimize_worst_fault_nearest_rank_p90_delay",
            "minimize_worst_fault_median_delay",
            "minimize_oof_healthy_alarm_runs",
            "prefer_larger_rho",
            "prefer_larger_q",
            "canonical_candidate_id",
        ],
    }


def validate_model_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all fields against the pre-registered exact specification.

    The strict equality is intentional: a formal audit must never obtain a new
    hyperparameter through an omitted field, backend default, or edited JSON.
    This implementation uses only the standard library and is therefore safe
    in metadata-only processes where neither NumPy nor Torch may be imported.
    """

    if not isinstance(value, Mapping):
        raise ValueError("model config must be an object")
    try:
        normalized = json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise ValueError("model config must be finite canonical JSON") from error
    expected = default_model_config()
    if _canonical_json_bytes(normalized) != _canonical_json_bytes(expected):
        raise ValueError("model config differs from the exact audited v3 specification")
    if len(normalized["candidate_grid"]) != 4:
        raise AssertionError("audited q-by-rho grid must contain exactly four candidates")
    return normalized


def _load_config(path: Path | None) -> dict[str, Any]:
    return validate_model_config(default_model_config() if path is None else _strict_json(path))


def _runtime_preflight(backend: NumericBackend) -> dict[str, Any]:
    raw = dict(backend.cuda_preflight())
    required = {
        "cuda_available",
        "device_type",
        "dtype",
        "torch_version",
        "cuda_version",
        "device_name",
    }
    if set(raw) != required:
        raise ValueError("backend CUDA preflight fields changed")
    if raw["cuda_available"] is not True or raw["device_type"] != "cuda" or raw[
        "dtype"
    ] != "float64":
        raise RuntimeError("MSCD-TCN formal numeric phases require CUDA float64")
    if not all(isinstance(raw[name], str) and raw[name] for name in required - {"cuda_available"}):
        raise ValueError("backend CUDA preflight contains invalid text")
    return raw


def _load_backend() -> ModuleType:
    try:
        return importlib.import_module("mate_spc_mstcn.backend_v3")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "production numeric backend mate_spc_mstcn.backend_v3 is not installed; "
            "governance can still be tested with an injected backend"
        ) from error


def _phase_paths(output_root: Path, phase: str) -> tuple[Path, Path, Path]:
    root = output_root.resolve()
    target = root / phase
    staging = root / f".{phase}.staging"
    claim = root / f".{phase}.claim"
    return target, staging, claim


def _assert_phase_unstarted(output_root: Path, phase: str) -> None:
    target, staging, claim = _phase_paths(output_root, phase)
    if target.exists() or staging.exists() or claim.exists():
        raise FileExistsError(f"{phase} already exists, is partial, or is claimed")


def _claim_phase(output_root: Path, phase: str) -> tuple[Path, Path, Path]:
    target, staging, claim = _phase_paths(output_root, phase)
    output_root.resolve().mkdir(parents=True, exist_ok=True)
    _assert_phase_unstarted(output_root, phase)
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(f"{phase} was concurrently claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical_json_bytes({"phase": phase, "pid": os.getpid()}))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        staging.mkdir()
    except Exception:
        claim.unlink(missing_ok=True)
        raise
    return target, staging, claim


def _artifact_records(directory: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {name: _hash_record(directory / name) for name in names}


def _finalize_phase(
    target: Path,
    staging: Path,
    claim: Path,
    *,
    phase: str,
    binding_sha256: str,
    artifact_names: Sequence[str],
) -> None:
    completion = {
        "schema": "mstcn_spc_phase_completion",
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "binding_sha256": binding_sha256,
        "artifacts": _artifact_records(staging, artifact_names),
    }
    _write_json(staging / "completion.json", completion)
    try:
        os.replace(staging, target)
    except Exception:
        raise
    else:
        claim.unlink(missing_ok=True)


def _write_access_intent(
    staging: Path,
    *,
    phase: str,
    binding_sha256: str,
    access_plan: Sequence[Mapping[str, Any]],
) -> Path:
    _validate_access_plan("development" if phase == "development-iteration" else phase, access_plan)
    path = staging / "access_intent.json"
    _write_json(
        path,
        {
            "schema": "mstcn_spc_access_intent",
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "binding_sha256": binding_sha256,
            "written_before_numeric_source_access": True,
            "access_plan": list(access_plan),
        },
    )
    return path


def _validate_parity(value: Any) -> float:
    error = float(value)
    if not math.isfinite(error) or error < 0.0 or error > PARITY_ATOL:
        raise ValueError(f"CPU/CUDA float64 parity exceeds {PARITY_ATOL:g}")
    return error


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("nearest-rank input cannot be empty")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("metric values must be finite")
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _validate_run_rows(
    rows: Any,
    *,
    expected_idvs: Sequence[int],
    expected_runs: Sequence[int],
    require_alarm_fields: bool,
) -> list[dict[str, Any]]:
    canonical_runs = tuple(int(run) for run in expected_runs)
    if (
        not canonical_runs
        or len(set(canonical_runs)) != len(canonical_runs)
        or not isinstance(rows, list)
        or len(rows) != len(expected_idvs) * len(canonical_runs)
    ):
        raise ValueError("backend returned the wrong number of run rows")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("backend run row must be an object")
        base = {"idv", "simulation_run", "run_maximum", "parity_max_abs_error"}
        required = base | ({"alarm", "first_alarm_index", "attribution"} if require_alarm_fields else set())
        if set(raw) != required:
            raise ValueError("backend run row fields changed")
        identity = (int(raw["idv"]), int(raw["simulation_run"]))
        if identity in identities:
            raise ValueError("backend returned a duplicate run identity")
        identities.add(identity)
        maximum = float(raw["run_maximum"])
        if not math.isfinite(maximum):
            raise ValueError("backend run maximum is nonfinite")
        row: dict[str, Any] = {
            "idv": identity[0],
            "simulation_run": identity[1],
            "run_maximum": maximum,
            "parity_max_abs_error": _validate_parity(raw["parity_max_abs_error"]),
        }
        if require_alarm_fields:
            if raw["alarm"] not in (True, False):
                raise ValueError("alarm must be boolean")
            alarm = bool(raw["alarm"])
            first = raw["first_alarm_index"]
            if alarm:
                if isinstance(first, bool) or not isinstance(first, int) or not FAULT_START <= first < SEQUENCE_LENGTH:
                    raise ValueError("alarmed run must have a valid post-onset first alarm")
            elif first is not None:
                raise ValueError("non-alarmed run must have null first alarm")
            attribution = raw["attribution"]
            if alarm and (isinstance(attribution, bool) or attribution not in FAULT_IDS):
                raise ValueError("alarmed run attribution must be 3, 9, or 15")
            if not alarm and attribution is not None:
                raise ValueError("non-alarmed run attribution must be null")
            row.update(alarm=alarm, first_alarm_index=first, attribution=attribution)
        normalized.append(row)
    expected = {(idv, run) for idv in expected_idvs for run in canonical_runs}
    if identities != expected:
        raise ValueError("backend run identities differ from the frozen partition")
    return sorted(normalized, key=lambda row: (row["idv"], row["simulation_run"]))


def _validate_development_result(
    raw: Mapping[str, Any],
    config: Mapping[str, Any],
    model_path: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "config", "candidates", "selected_candidate_id", "runtime",
        "model_metadata", "model_parity_max_abs_error",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("development backend payload fields changed")
    if validate_model_config(raw["config"]) != dict(config):
        raise ValueError("backend used a different model configuration")
    if dict(raw["runtime"]) != dict(runtime):
        raise ValueError("development backend runtime differs from CUDA preflight")
    candidates = raw["candidates"]
    frozen_grid = config["candidate_grid"]
    if not isinstance(candidates, list) or len(candidates) != len(frozen_grid):
        raise ValueError("development backend must return all four frozen candidates")
    fold_map = _fold_assignment()
    normalized_candidates: list[dict[str, Any]] = []
    for expected_candidate, candidate in zip(frozen_grid, candidates, strict=True):
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "candidate_id", "mil_top_fraction_q", "scan_tail_fraction_rho", "folds"
        }:
            raise ValueError("development candidate fields changed")
        identity = {
            "candidate_id": candidate["candidate_id"],
            "mil_top_fraction_q": candidate["mil_top_fraction_q"],
            "scan_tail_fraction_rho": candidate["scan_tail_fraction_rho"],
        }
        if identity != expected_candidate:
            raise ValueError("development candidates differ from the frozen q-by-rho grid")
        folds = candidate["folds"]
        if not isinstance(folds, list) or len(folds) != FOLD_COUNT:
            raise ValueError("each candidate must return exactly five folds")
        all_rows: list[dict[str, Any]] = []
        normalized_folds: list[dict[str, Any]] = []
        fold_checks: dict[str, bool] = {}
        fold_fault_detection_rates: dict[str, float] = {}
        for fold in range(FOLD_COUNT):
            item = folds[fold]
            if not isinstance(item, Mapping) or set(item) != {
                "fold", "training_healthy_maxima", "rows",
                "score_parity_max_abs_error", "discrete_parity_equal",
            } or item["fold"] != fold:
                raise ValueError("candidate folds are missing, reordered, or malformed")
            parity = _validate_parity(item["score_parity_max_abs_error"])
            if item["discrete_parity_equal"] is not True:
                raise ValueError("CPU/CUDA discrete outputs must be identical")
            maxima = item["training_healthy_maxima"]
            if not isinstance(maxima, list) or len(maxima) != 80:
                raise ValueError("each fold needs 80 training-healthy run maxima")
            ordered = sorted(float(value) for value in maxima)
            if any(not math.isfinite(value) for value in ordered):
                raise ValueError("training-healthy maximum is nonfinite")
            tolerance = max(FLOAT_TOLERANCE, parity)
            distinct = 1 + sum(
                current - previous > tolerance
                for previous, current in zip(ordered, ordered[1:])
            )
            threshold = ordered[78]
            q80 = ordered[79]
            adjacent_tail_gap = q80 - threshold
            threshold_checks = {
                "at_least_76_distinct": distinct >= 76,
                "rank79_strictly_below_rank80": threshold < q80,
                "adjacent_tail_gap_nondegenerate": adjacent_tail_gap > 100.0 * tolerance,
            }
            expected_runs = sorted(run for run, assigned in fold_map.items() if assigned == fold)
            rows = _validate_run_rows(
                item["rows"], expected_idvs=ALL_IDS,
                expected_runs=expected_runs, require_alarm_fields=True,
            )
            for row in rows:
                if bool(row["alarm"]) != (float(row["run_maximum"]) > threshold):
                    raise ValueError("development alarm disagrees with rank-79 strict threshold")
            healthy_alarms = sum(row["alarm"] for row in rows if row["idv"] == 1)
            fold_checks[f"fold_{fold}_healthy_alarms"] = (
                healthy_alarms
                <= DEVELOPMENT_GATES["maximum_oof_healthy_alarm_runs_each_fold"]
            )
            for fault in FAULT_IDS:
                fault_rows = [row for row in rows if row["idv"] == fault]
                rate = sum(row["alarm"] for row in fault_rows) / len(fault_rows)
                fold_fault_detection_rates[f"fold_{fold}_fault_{fault}"] = rate
                fold_checks[f"fold_{fold}_fault_{fault}_detection"] = (
                    rate >= DEVELOPMENT_GATES["minimum_detection_rate_each_fold_fault"]
                )
            fold_checks.update(
                {
                    f"fold_{fold}_{name}": passed
                    for name, passed in threshold_checks.items()
                }
            )
            all_rows.extend(rows)
            normalized_folds.append(
                {
                    "fold": fold,
                    "training_healthy_maxima": ordered,
                    "threshold_rank79": threshold,
                    "rank80": q80,
                    "distinct_maxima": distinct,
                    "adjacent_tail_gap_rank80_minus_rank79": adjacent_tail_gap,
                    "score_parity_max_abs_error": parity,
                    "discrete_parity_equal": True,
                    "threshold_checks": threshold_checks,
                    "rows": rows,
                }
            )
        if len({(row["idv"], row["simulation_run"]) for row in all_rows}) != 400:
            raise ValueError("candidate OOF rows are not one-to-one")
        metrics = _summary_metrics(all_rows, gates=DEVELOPMENT_GATES, development=True)
        metrics["fold_gate_checks"] = fold_checks
        metrics["fold_fault_detection_rates"] = fold_fault_detection_rates
        metrics["all_gates_pass"] = metrics["all_gates_pass"] and all(fold_checks.values())
        normalized_candidates.append(
            {
                **identity,
                "folds": normalized_folds,
                "metrics": metrics,
                "selection_eligible": metrics["all_gates_pass"],
            }
        )
    eligible = [row for row in normalized_candidates if row["selection_eligible"]]
    selected = min(eligible, key=_candidate_selection_key) if eligible else None
    expected_selected = None if selected is None else selected["candidate_id"]
    if raw["selected_candidate_id"] != expected_selected:
        raise ValueError("backend selected candidate differs from deterministic OOF selection")
    if selected is None:
        if model_path.exists() or raw["model_metadata"] is not None or raw["model_parity_max_abs_error"] is not None:
            raise ValueError("DEVELOPMENT_NO_GO must not emit a model")
    else:
        if not model_path.is_file() or not isinstance(raw["model_metadata"], Mapping):
            raise ValueError("passed development selection must emit one final model")
        _validate_parity(raw["model_parity_max_abs_error"])
    return {
        "config": dict(config),
        "candidates": normalized_candidates,
        "selected_candidate_id": expected_selected,
        "selected_candidate": None if selected is None else {
            key: selected[key]
            for key in ("candidate_id", "mil_top_fraction_q", "scan_tail_fraction_rho")
        },
        "runtime": dict(raw["runtime"]),
        "model_metadata": None if selected is None else dict(raw["model_metadata"]),
        "model_parity_max_abs_error": (
            None if selected is None else float(raw["model_parity_max_abs_error"])
        ),
        "decision": "PASS" if selected is not None else "DEVELOPMENT_NO_GO",
    }


def _candidate_selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    minimum_detection = min(
        metrics["faults"][str(fault)]["detection_rate"] for fault in FAULT_IDS
    )
    worst_p90 = max(
        metrics["faults"][str(fault)]["p90_delay_samples_nearest_rank"]
        for fault in FAULT_IDS
    )
    minimum_fold_detection = min(metrics["fold_fault_detection_rates"].values())
    worst_median = max(
        metrics["faults"][str(fault)]["median_delay_samples"] for fault in FAULT_IDS
    )
    return (
        -minimum_detection,
        -minimum_fold_detection,
        worst_p90,
        worst_median,
        metrics["healthy_alarm_runs"],
        -float(candidate["scan_tail_fraction_rho"]),
        -float(candidate["mil_top_fraction_q"]),
        candidate["candidate_id"],
    )


def _summary_metrics(
    rows: Sequence[Mapping[str, Any]], *, gates: Mapping[str, Any], development: bool
) -> dict[str, Any]:
    healthy = [row for row in rows if int(row["idv"]) == 1]
    healthy_alarms = sum(bool(row["alarm"]) for row in healthy)
    faults: dict[str, dict[str, Any]] = {}
    gate_checks: dict[str, bool] = {}
    for fault in FAULT_IDS:
        selected = [row for row in rows if int(row["idv"]) == fault]
        detected = [row for row in selected if bool(row["alarm"])]
        delays = [
            int(row["first_alarm_index"]) - FAULT_START if row["alarm"] else MISS_DELAY
            for row in selected
        ]
        attribution_correct = sum(
            bool(row["alarm"]) and int(row["attribution"]) == fault for row in selected
        )
        metrics = {
            "run_count": len(selected),
            "detected_runs": len(detected),
            "detection_rate": len(detected) / len(selected),
            "median_delay_samples": float(statistics.median(delays)),
            "p90_delay_samples_nearest_rank": _nearest_rank(delays, 0.90),
            "attribution_accuracy_all_runs": attribution_correct / len(selected),
            "attribution_accuracy_detected_runs": (
                attribution_correct / len(detected) if detected else 0.0
            ),
        }
        faults[str(fault)] = metrics
        if development:
            gate_checks[f"fault_{fault}_detection"] = (
                metrics["detection_rate"] >= gates["minimum_detection_rate_each_fault"]
            )
            gate_checks[f"fault_{fault}_median_delay"] = (
                metrics["median_delay_samples"]
                <= gates["maximum_median_delay_samples_each_fault"]
            )
            gate_checks[f"fault_{fault}_p90_delay"] = (
                metrics["p90_delay_samples_nearest_rank"]
                <= gates["maximum_p90_delay_samples_each_fault"]
            )
        else:
            gate_checks[f"fault_{fault}_detected_runs"] = (
                metrics["detected_runs"] >= gates["minimum_detected_runs_each_fault"]
            )
            gate_checks[f"fault_{fault}_median_delay"] = (
                metrics["median_delay_samples"]
                <= gates["maximum_median_delay_samples_each_fault"]
            )
            gate_checks[f"fault_{fault}_p90_delay"] = (
                metrics["p90_delay_samples_nearest_rank"]
                <= gates["maximum_p90_delay_samples_each_fault"]
            )
    if development:
        gate_checks["healthy_alarm_runs"] = healthy_alarms <= gates[
            "maximum_oof_healthy_alarm_runs"
        ]
    else:
        gate_checks["healthy_alarm_runs"] = healthy_alarms <= gates[
            "maximum_healthy_alarm_runs"
        ]
        gate_checks["detected_runs_total"] = sum(
            value["detected_runs"] for value in faults.values()
        ) >= gates["minimum_detected_runs_total"]
    return {
        "healthy_run_count": len(healthy),
        "healthy_alarm_runs": healthy_alarms,
        "faults": faults,
        "gate_checks": gate_checks,
        "all_gates_pass": all(gate_checks.values()),
        "attribution_is_separate_from_primary_detection_gate": True,
    }


def _verify_model_file(
    model_path: Path,
    config: Mapping[str, Any],
    selected_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Use the package safe loader and require the exact frozen architecture."""

    from mate_spc_mstcn.model import load_mstcn_model
    from mate_spc_mstcn.monitor import FittedTopTailMonitor

    model, metadata = load_mstcn_model(model_path)
    if not isinstance(selected_candidate, Mapping) or set(selected_candidate) != {
        "candidate_id", "mil_top_fraction_q", "scan_tail_fraction_rho"
    }:
        raise ValueError("selected candidate identity is absent or malformed")
    actual = asdict(model.config)
    expected_fields = {
        "feature_schema", "input_features", "window_length", "channels",
        "kernel_size", "dilations", "pool_scales", "hidden_size",
        "class_ids", "monitor_start", "endpoint_stride", "mil_top_fraction",
    }
    if set(actual) != expected_fields:
        raise ValueError("serialized model config field set differs from schema-v2")
    expected = {
        "feature_schema": config["feature_transform"]["feature_schema"],
        "input_features": config["feature_transform"]["input_features"],
        "window_length": config["architecture"]["window_length"],
        "channels": config["architecture"]["channels"],
        "kernel_size": config["architecture"]["kernel_size"],
        "dilations": config["architecture"]["dilations"],
        "pool_scales": config["architecture"]["pool_scales"],
        "hidden_size": config["architecture"]["hidden_size"],
        "class_ids": [0, *config["architecture"]["attribution_class_ids"]],
        "monitor_start": config["endpoints"]["dense_half_open_step"][0],
        "endpoint_stride": config["endpoints"]["mil_half_open_step"][2],
        "mil_top_fraction": float(selected_candidate["mil_top_fraction_q"]),
    }
    for key, value in expected.items():
        observed = actual.get(key)
        if isinstance(observed, tuple):
            observed = list(observed)
        if observed != value:
            raise ValueError(f"serialized model field {key!r} differs from complete config")
    if model.seed != config["training"]["seed"]:
        raise ValueError("serialized model seed differs from complete config")
    metadata_keys = {
        "schema", "backend_schema", "complete_model_config_sha256",
        "selected_candidate", "fitted_monitor", "training", "loss_history",
        "monitor_fit_scope", "held_out_25_or_75_percent_accessed",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_keys:
        raise ValueError("bound model metadata field set differs from schema-v1")
    if (
        metadata["schema"] != "mstcn_spc_bound_model_metadata_v1"
        or metadata["backend_schema"] != "mstcn_spc_backend_v3"
        or metadata["complete_model_config_sha256"] != _config_sha256(config)
        or metadata["selected_candidate"] != dict(selected_candidate)
        or metadata["monitor_fit_scope"]
        != "all_100_magnitude_50_development_healthy_runs_only"
        or metadata["held_out_25_or_75_percent_accessed"] is not False
    ):
        raise ValueError("bound model metadata identity/scope changed")

    training = metadata["training"]
    training_keys = {
        "method", "objective", "mil_top_fraction_q",
        "positive_endpoint_labels_used", "detection_target",
        "attribution_target", "loss_formula", "architecture_pooling",
        "requested_device", "dtype", "seed", "model_config",
        "training_config", "training_run_names", "training_run_count_by_class",
        "mil_endpoints_half_open_stride", "mil_endpoint_count",
        "dense_inference_endpoints_half_open_stride", "batch_balance",
        "validation_or_early_stopping_used", "window_contract",
        "preprocessor_fit_scope",
    }
    if not isinstance(training, Mapping) or set(training) != training_keys:
        raise ValueError("bound training metadata field set changed")
    expected_training_config = {
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["bag_batch_size"],
        "examples_per_class_per_epoch": config["training"][
            "examples_per_class_per_epoch_cap"
        ],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        "device": "cuda",
    }
    normalized_actual_config = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in actual.items()
    }
    expected_run_names = sorted(
        f"m50-idv{idv:02d}-run{run:03d}"
        for idv in ALL_IDS
        for run in DEVELOPMENT_RUNS
    )
    if (
        training["method"] != "Detection-first Multi-Scale Causal Dynamics TCN"
        or training["objective"] != "q-specific full-monitoring run-level MIL"
        or float(training["mil_top_fraction_q"])
        != float(selected_candidate["mil_top_fraction_q"])
        or training["positive_endpoint_labels_used"] is not False
        or training["detection_target"] != "fault_vs_healthy"
        or training["attribution_target"] != "fault_3_vs_9_vs_15_fault_runs_only"
        or training["loss_formula"]
        != "0.5*mean_H softplus(A) + (1/6)*sum_k mean_Fk softplus(-A) + 0.25*(1/3)*sum_k mean_Fk CE(C,k)"
        or training["architecture_pooling"]
        != "TCN endpoint state plus 8/32/96 mean-RMS of pointwise projection"
        or training["requested_device"] != "cuda"
        or training["dtype"] != "float64"
        or training["seed"] != config["training"]["seed"]
        or training["model_config"] != normalized_actual_config
        or training["training_config"] != expected_training_config
        or training["training_run_names"] != expected_run_names
        or training["training_run_count_by_class"]
        != {"0": 100, "3": 100, "9": 100, "15": 100}
        or training["mil_endpoints_half_open_stride"] != [160, 500, 4]
        or training["mil_endpoint_count"] != 85
        or training["dense_inference_endpoints_half_open_stride"] != [160, 500, 1]
        or training["batch_balance"] != "eight_runs_exactly_two_per_class"
        or training["validation_or_early_stopping_used"] is not False
        or training["window_contract"]
        != "each output ends at its named endpoint and contains no future row"
        or training["preprocessor_fit_scope"]
        != "training_run_commissioning_prefixes_only"
    ):
        raise ValueError("bound training metadata differs from the audited final refit")

    history = metadata["loss_history"]
    if (
        not isinstance(history, list)
        or len(history) != config["training"]["epochs"]
        or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in history)
    ):
        raise ValueError("bound final-refit loss history must contain 30 finite epochs")
    monitor = FittedTopTailMonitor.from_document(metadata["fitted_monitor"])
    if (
        monitor.endpoints.tolist() != list(range(160, 500))
        or list(monitor.windows) != config["scan"]["window_samples_W"]
        or monitor.top_fraction != float(selected_candidate["scan_tail_fraction_rho"])
        or monitor.persistence != config["scan"]["persistence_samples"]
    ):
        raise ValueError("bound fitted monitor geometry differs from selected rho/config")
    return metadata


def run_development_iteration(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    iteration_id: str,
    config: Mapping[str, Any],
    *,
    backend: NumericBackend | None = None,
    source_record: Mapping[str, Any] | None = None,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one repeatable 50%-only grouped development iteration."""

    if not iteration_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in iteration_id):
        raise ValueError("iteration_id must use only ASCII letters, digits, '-' or '_'")
    complete_config = validate_model_config(config)
    actual_backend = _load_backend() if backend is None else backend
    runtime = _runtime_preflight(actual_backend)
    development_root = output_root.resolve() / "development_iterations"
    target, staging, claim = _claim_phase(development_root, iteration_id)
    iteration_manifest = {
        "schema": "mstcn_spc_development_iteration_manifest",
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_SHORT,
        "iteration_id": iteration_id,
        "config": complete_config,
        "source_record": dict(source_record or _source_record(data_dir)),
        "implementation_records": _implementation_records(project_root),
    }
    binding = _sha256_bytes(_canonical_json_bytes(iteration_manifest))
    access_plan = _development_access_plan()
    try:
        _write_access_intent(
            staging,
            phase="development-iteration",
            binding_sha256=binding,
            access_plan=access_plan,
        )
        _write_json(staging / "config.json", complete_config)
        _write_json(staging / "iteration_manifest.json", iteration_manifest)
        _write_json(staging / "runtime.json", runtime)
        model_path = staging / "mstcn_model.npz"
        raw = actual_backend.development_iteration(
            source_path=_source_path(data_dir),
            access_plan=access_plan,
            fold_map=_fold_assignment(),
            config=complete_config,
            model_path=model_path,
            device="cuda",
            dtype="float64",
        )
        result = _validate_development_result(raw, complete_config, model_path, runtime)
        if result["decision"] == "PASS":
            if verify_model is None:
                verified_metadata = _verify_model_file(
                    model_path, complete_config, result["selected_candidate"]
                )
            else:
                verified_metadata = verify_model(model_path, complete_config)
            if dict(verified_metadata) != result["model_metadata"]:
                raise ValueError("backend payload metadata differs from bound model metadata")
        _write_json(staging / "development_result.json", result)
        names: tuple[str, ...] = (
            "access_intent.json",
            "config.json",
            "iteration_manifest.json",
            "runtime.json",
            "development_result.json",
        )
        if result["decision"] == "PASS":
            names = (*names, "mstcn_model.npz")
        _finalize_phase(
            target,
            staging,
            claim,
            phase="development-iteration",
            binding_sha256=binding,
            artifact_names=names,
        )
    except Exception:
        # Retain both claim and staging: a failed attempt is visible and cannot
        # be silently rerun under the same identity.
        raise
    return _strict_json(target / "development_result.json")


def _partition_contract(output_root: Path) -> dict[str, Any]:
    return {
        "bound_output_root": str(output_root.resolve()),
        "development_repeatable": {
            "magnitude_percent": DEVELOPMENT_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "simulation_runs": [1, 100],
        },
        "calibration_one_time": {
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "idv": 1,
            "simulation_runs": [1, 100],
            "source_window_half_open": list(SOURCE_HEALTHY_WINDOW),
            "pre_fault_only": True,
        },
        "quarantined_never_access": {
            "magnitude_percent": CALIBRATION_MAGNITUDE,
            "idvs": list(FAULT_IDS),
            "simulation_runs": [1, 100],
        },
        "confirmation_one_time": {
            "magnitude_percent": CONFIRMATION_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "simulation_runs": [1, 100],
        },
        "consumed_diagnostic_only": {
            "magnitude_percent": CONSUMED_MAGNITUDE,
            "idvs": list(ALL_IDS),
            "simulation_runs": [1, 200],
        },
        "same_run_number_across_idvs_is_balancing_not_CRN": True,
        "rng_seed_is_independent_simulation_identity": True,
        "calibration_and_confirmation_may_not_tune_model_or_scan": True,
        "confirmation_is_one_time_even_on_failure": True,
    }


def _monitor_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": METHOD,
        "sequence_length": SEQUENCE_LENGTH,
        "baseline_half_open": list(BASELINE_WINDOW),
        "monitored_half_open": list(MONITORED_WINDOW),
        "fault_start_zero_based": FAULT_START,
        "miss_delay_samples": MISS_DELAY,
        "sample_minutes": 3,
        "complete_model_config": validate_model_config(config),
        "cpu_cuda_float64_parity_atol": PARITY_ATOL,
        "primary_endpoint": "fault_detection",
        "secondary_endpoint": "fault_attribution_descriptive_not_detection_gate",
    }


def _calibration_contract() -> dict[str, Any]:
    return {
        "run_count": 100,
        "ascending_order_statistic_rank": CALIBRATION_RANK,
        "rule": "98th_ascending_of_100_run_maxima_noninterpolated",
        "comparison": "strictly_greater_than",
        "epsilon": CALIBRATION_EPSILON,
        "delta": CALIBRATION_DELTA,
        "minimum_numerically_distinct_maxima": MIN_UNIQUE_MAXIMA,
        "rank98_strictly_below_rank100": True,
        "tail_diagnostic_lower_rank": TOP_TAIL_LOWER_RANK,
        "tail_spread_is_acceptance_gate": False,
        "float_tolerance": FLOAT_TOLERANCE,
        "parity_atol": PARITY_ATOL,
        "certificate_scope": "fixed_monitor_run_level_false_alarm_probability_below_0.10_with_confidence_at_least_0.99_under_exchangeability",
        "finite_sample_marginal_exceedance_bound": "3/101_under_exchangeability_with_strict_greater_than",
        "does_not_certify_false_alarm_probability_below_0.05": True,
    }


def _verify_iteration_bundle(
    iteration_dir: Path,
    output_root: Path,
    *,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    root = output_root.resolve() / "development_iterations"
    iteration = iteration_dir.resolve()
    if iteration.parent != root:
        raise ValueError("selected iteration must be a direct child of the bound development root")
    completion = _strict_json(iteration / "completion.json")
    expected_names = {
        "access_intent.json",
        "config.json",
        "iteration_manifest.json",
        "runtime.json",
        "mstcn_model.npz",
        "development_result.json",
    }
    if (
        completion.get("schema") != "mstcn_spc_phase_completion"
        or completion.get("schema_version") != SCHEMA_VERSION
        or completion.get("phase") != "development-iteration"
        or set(completion.get("artifacts", {})) != expected_names
        or completion["artifacts"] != _artifact_records(iteration, sorted(expected_names))
        or {path.name for path in iteration.iterdir()} != expected_names | {"completion.json"}
    ):
        raise ValueError("selected development iteration is incomplete or mutated")
    config = validate_model_config(_strict_json(iteration / "config.json"))
    iteration_manifest = _strict_json(iteration / "iteration_manifest.json")
    if set(iteration_manifest) != {
        "schema", "schema_version", "method", "iteration_id", "config",
        "source_record", "implementation_records",
    } or iteration_manifest != {
        "schema": "mstcn_spc_development_iteration_manifest",
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_SHORT,
        "iteration_id": iteration.name,
        "config": config,
        "source_record": iteration_manifest.get("source_record"),
        "implementation_records": iteration_manifest.get("implementation_records"),
    }:
        raise ValueError("development iteration manifest fields changed")
    if _sha256_bytes(_canonical_json_bytes(iteration_manifest)) != completion.get(
        "binding_sha256"
    ):
        raise ValueError("development iteration manifest binding changed")
    result = _strict_json(iteration / "development_result.json")
    if result.get("config") != config or result.get("decision") != "PASS":
        raise RuntimeError("only a passed development iteration may enter formal freeze")
    model_path = iteration / "mstcn_model.npz"
    replay_candidates: list[dict[str, Any]] = []
    for candidate in result.get("candidates", []):
        replay_candidates.append(
            {
                key: candidate[key]
                for key in (
                    "candidate_id", "mil_top_fraction_q", "scan_tail_fraction_rho"
                )
            }
            | {
                "folds": [
                    {
                        "fold": fold["fold"],
                        "training_healthy_maxima": fold["training_healthy_maxima"],
                        "rows": fold["rows"],
                        "score_parity_max_abs_error": fold[
                            "score_parity_max_abs_error"
                        ],
                        "discrete_parity_equal": fold["discrete_parity_equal"],
                    }
                    for fold in candidate["folds"]
                ]
            }
        )
    replayed = _validate_development_result(
        {
            "config": config,
            "candidates": replay_candidates,
            "selected_candidate_id": result.get("selected_candidate_id"),
            "runtime": result.get("runtime"),
            "model_metadata": result.get("model_metadata"),
            "model_parity_max_abs_error": result.get("model_parity_max_abs_error"),
        },
        config,
        model_path,
        result.get("runtime"),
    )
    if replayed != result:
        raise ValueError("development result semantic replay failed")
    selected_rows = [
        row for row in result.get("candidates", [])
        if row.get("candidate_id") == result.get("selected_candidate_id")
    ]
    if (
        len(selected_rows) != 1
        or selected_rows[0].get("selection_eligible") is not True
        or selected_rows[0].get("metrics", {}).get("all_gates_pass") is not True
    ):
        raise ValueError("development PASS is inconsistent with selected candidate gates")
    # The default verifier imports the model (Torch first on Windows), validates
    # its safe NPZ schema, and then topology-only HDF5 validation may import
    # NumPy.  It never decodes processdata/idv_init values.
    if verify_model is None:
        verified_metadata = _verify_model_file(
            model_path, config, result["selected_candidate"]
        )
    else:
        verified_metadata = verify_model(model_path, config)
    if dict(verified_metadata) != result["model_metadata"]:
        raise ValueError("development result metadata differs from bound model metadata")
    return completion, config, model_path, iteration_manifest


def build_formal_protocol(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    selected_iteration: Path,
    *,
    source_record: Mapping[str, Any] | None = None,
    source_metadata_record: Mapping[str, Any] | None = None,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build (but do not publish) the immutable formal audit protocol."""

    _assert_phase_unstarted(output_root, "calibrate")
    _assert_phase_unstarted(output_root, "confirm")
    completion, config, model_path, iteration_manifest = _verify_iteration_bundle(
        selected_iteration, output_root, verify_model=verify_model
    )
    iteration = selected_iteration.resolve()
    actual_source = dict(source_record or _source_record(data_dir))
    metadata = dict(source_metadata_record or _source_metadata_record(data_dir))
    implementation = _implementation_records(project_root)
    if iteration_manifest["source_record"] != actual_source:
        raise ValueError("development source record differs at formal freeze")
    if iteration_manifest["implementation_records"] != implementation:
        raise ValueError("implementation changed between development and formal freeze")
    return {
        "schema": "mstcn_spc_formal_protocol",
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "frozen_after_development_pass_before_any_v3_25_or_75_percent_numeric_access",
        "method": METHOD_SHORT,
        "implementation_records": implementation,
        "source_record": actual_source,
        "source_metadata_record": metadata,
        "partition_governance": _partition_contract(output_root),
        "monitor": _monitor_contract(config),
        "development_binding": {
            "iteration_name": iteration.name,
            "iteration_absolute_path": str(iteration),
            "completion_record": _hash_record(iteration / "completion.json"),
            "result_record": _hash_record(iteration / "development_result.json"),
            "config_record": _hash_record(iteration / "config.json"),
            "iteration_manifest_record": _hash_record(
                iteration / "iteration_manifest.json"
            ),
            "model_absolute_path": str(model_path.resolve()),
            "model_record": _hash_record(model_path),
            "iteration_binding_sha256": completion["binding_sha256"],
            "selected_candidate": _strict_json(iteration / "development_result.json")[
                "selected_candidate"
            ],
        },
        "calibration": _calibration_contract(),
        "confirmation": {
            "gates": CONFIRMATION_GATES,
            "one_time_even_if_decision_is_fail": True,
            "detection_requires_new_alarm_at_or_after": FAULT_START,
            "threshold_source": "completed_bound_calibration_only",
        },
    }


def freeze_formal(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    selected_iteration: Path,
    protocol_path: Path,
    **testing_injections: Any,
) -> str:
    return _publish_protocol_bundle(
        protocol_path,
        build_formal_protocol(
            project_root,
            data_dir,
            output_root,
            selected_iteration,
            **testing_injections,
        ),
    )


def _assert_formal_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "schema_version", "analysis_status", "method",
        "implementation_records", "source_record", "source_metadata_record",
        "partition_governance", "monitor", "development_binding",
        "calibration", "confirmation",
    }
    if set(protocol) != required:
        raise ValueError("formal protocol fields changed")
    if (
        protocol["schema"] != "mstcn_spc_formal_protocol"
        or protocol["schema_version"] != SCHEMA_VERSION
        or protocol["method"] != METHOD_SHORT
        or protocol["analysis_status"]
        != "frozen_after_development_pass_before_any_v3_25_or_75_percent_numeric_access"
    ):
        raise ValueError("formal protocol identity changed")
    config = validate_model_config(protocol["monitor"]["complete_model_config"])
    if protocol["monitor"] != _monitor_contract(config):
        raise ValueError("formal monitor contract changed")
    bound = Path(str(protocol["partition_governance"].get("bound_output_root", "")))
    if protocol["partition_governance"] != _partition_contract(bound):
        raise ValueError("formal partition governance changed")
    if protocol["calibration"] != _calibration_contract():
        raise ValueError("formal calibration contract changed")
    if protocol["confirmation"] != {
        "gates": CONFIRMATION_GATES,
        "one_time_even_if_decision_is_fail": True,
        "detection_requires_new_alarm_at_or_after": FAULT_START,
        "threshold_source": "completed_bound_calibration_only",
    }:
        raise ValueError("formal confirmation contract changed")
    binding = protocol["development_binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "iteration_name", "iteration_absolute_path", "completion_record",
        "result_record", "config_record", "iteration_manifest_record",
        "model_absolute_path", "model_record", "iteration_binding_sha256",
        "selected_candidate",
    }:
        raise ValueError("formal development binding fields changed")
    selected = binding["selected_candidate"]
    if not isinstance(selected, Mapping) or dict(selected) not in config["candidate_grid"]:
        raise ValueError("formal selected candidate is outside the frozen grid")
    _require_sha256(binding["iteration_binding_sha256"], label="iteration binding SHA-256")
    return config


def _assert_bound_output(output_root: Path, protocol: Mapping[str, Any]) -> None:
    bound = Path(protocol["partition_governance"]["bound_output_root"])
    if output_root.resolve() != bound.resolve():
        raise ValueError("output root differs from the path bound at formal freeze")


def _verify_formal_live_inputs(
    project_root: Path,
    data_dir: Path,
    protocol: Mapping[str, Any],
    *,
    source_record: Mapping[str, Any] | None = None,
    source_metadata_record: Mapping[str, Any] | None = None,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = _assert_formal_protocol(protocol)
    if _implementation_records(project_root) != protocol["implementation_records"]:
        raise ValueError("implementation or design changed after formal freeze")
    if dict(source_record or _source_record(data_dir)) != protocol["source_record"]:
        raise ValueError("source bytes changed after formal freeze")
    if dict(source_metadata_record or _source_metadata_record(data_dir)) != protocol[
        "source_metadata_record"
    ]:
        raise ValueError("source topology changed after formal freeze")
    binding = protocol["development_binding"]
    model_path = Path(binding["model_absolute_path"])
    if _hash_record(model_path) != binding["model_record"]:
        raise ValueError("frozen model changed")
    iteration = Path(binding["iteration_absolute_path"])
    for name, key in (
        ("completion.json", "completion_record"),
        ("development_result.json", "result_record"),
        ("config.json", "config_record"),
        ("iteration_manifest.json", "iteration_manifest_record"),
    ):
        if _hash_record(iteration / name) != binding[key]:
            raise ValueError("selected development artifact changed after formal freeze")
    if verify_model is None:
        verified_metadata = _verify_model_file(
            model_path, config, binding["selected_candidate"]
        )
    else:
        verified_metadata = verify_model(model_path, config)
    development_result = _strict_json(iteration / "development_result.json")
    if dict(verified_metadata) != development_result.get("model_metadata"):
        raise ValueError("live model metadata differs from frozen development result")
    return model_path, config


def _calibration_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    maxima = sorted(float(row["run_maximum"]) for row in rows)
    if len(maxima) != 100:
        raise ValueError("calibration requires exactly 100 maxima")
    q98 = maxima[CALIBRATION_RANK - 1]
    q100 = maxima[-1]
    tolerance = max(
        FLOAT_TOLERANCE,
        PARITY_ATOL,
        max(float(row["parity_max_abs_error"]) for row in rows),
    )
    distinct = 1 + sum(
        current - previous > tolerance for previous, current in zip(maxima, maxima[1:])
    )
    tail_spread_diagnostic = q100 - maxima[TOP_TAIL_LOWER_RANK - 1]
    checks = {
        "minimum_distinct_maxima": distinct >= MIN_UNIQUE_MAXIMA,
        "rank98_strictly_below_rank100": q98 < q100,
    }
    return {
        "decision": "PASS" if all(checks.values()) else "CALIBRATION_INVALID",
        "threshold": q98 if all(checks.values()) else None,
        "rank": CALIBRATION_RANK,
        "run_count": len(maxima),
        "minimum": maxima[0],
        "q95": maxima[TOP_TAIL_LOWER_RANK - 1],
        "q98": q98,
        "q100": q100,
        "numerically_distinct_maxima": distinct,
        "tolerance": tolerance,
        "rank100_minus_rank96_diagnostic_only": tail_spread_diagnostic,
        "checks": checks,
        "certificate": _calibration_contract()["certificate_scope"],
        "finite_sample_marginal_exceedance_bound": "3/101",
        "does_not_certify_false_alarm_probability_below_0.05": True,
    }


def run_calibration(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    *,
    backend: NumericBackend | None = None,
    source_record: Mapping[str, Any] | None = None,
    source_metadata_record: Mapping[str, Any] | None = None,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    # Importing the production backend first establishes torch-before-NumPy DLL
    # order on Windows.  Protocol and path checks still occur before HDF5 access.
    actual_backend = _load_backend() if backend is None else backend
    protocol, protocol_sha = _load_protocol(protocol_path, protocol_sha256)
    _assert_bound_output(output_root, protocol)
    _assert_phase_unstarted(output_root, "confirm")
    model_path, config = _verify_formal_live_inputs(
        project_root, data_dir, protocol, source_record=source_record,
        source_metadata_record=source_metadata_record, verify_model=verify_model,
    )
    runtime = _runtime_preflight(actual_backend)
    target, staging, claim = _claim_phase(output_root, "calibrate")
    access_plan = _calibration_access_plan()
    _write_access_intent(
        staging, phase="calibrate", binding_sha256=protocol_sha, access_plan=access_plan
    )
    _write_json(staging / "runtime.json", runtime)
    raw = actual_backend.calibrate(
        source_path=_source_path(data_dir), access_plan=access_plan,
        model_path=model_path, config=config, device="cuda", dtype="float64",
    )
    if not isinstance(raw, Mapping) or set(raw) != {
        "config", "rows", "model_parity_max_abs_error", "discrete_parity_equal"
    }:
        raise ValueError("calibration backend payload fields changed")
    if validate_model_config(raw["config"]) != config:
        raise ValueError("calibration backend used a different model config")
    _validate_parity(raw["model_parity_max_abs_error"])
    if raw["discrete_parity_equal"] is not True:
        raise ValueError("calibration CPU/CUDA discrete outputs differ")
    rows = _validate_run_rows(
        raw["rows"], expected_idvs=(1,), expected_runs=CALIBRATION_RUNS,
        require_alarm_fields=False,
    )
    result = {
        "schema": "mstcn_spc_calibration_result",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "summary": _calibration_summary(rows),
        "rows": rows,
    }
    _write_json(staging / "calibration_result.json", result)
    _finalize_phase(
        target, staging, claim, phase="calibrate", binding_sha256=protocol_sha,
        artifact_names=("access_intent.json", "runtime.json", "calibration_result.json"),
    )
    return _strict_json(target / "calibration_result.json")


def _verify_calibration(output_root: Path, protocol_sha: str) -> tuple[dict[str, Any], float]:
    target = output_root.resolve() / "calibrate"
    completion = _strict_json(target / "completion.json")
    names = ("access_intent.json", "runtime.json", "calibration_result.json")
    if (
        completion.get("phase") != "calibrate"
        or completion.get("binding_sha256") != protocol_sha
        or completion.get("artifacts") != _artifact_records(target, names)
        or {path.name for path in target.iterdir()} != set(names) | {"completion.json"}
    ):
        raise ValueError("calibration bundle is incomplete or mutated")
    result = _strict_json(target / "calibration_result.json")
    rows = _validate_run_rows(
        result.get("rows"), expected_idvs=(1,), expected_runs=CALIBRATION_RUNS,
        require_alarm_fields=False,
    )
    expected = _calibration_summary(rows)
    if result != {
        "schema": "mstcn_spc_calibration_result",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "summary": expected,
        "rows": rows,
    }:
        raise ValueError("calibration semantic replay failed")
    if expected["decision"] != "PASS" or expected["threshold"] is None:
        raise RuntimeError("invalid calibration forbids confirmation")
    return result, float(expected["threshold"])


def run_confirmation(
    project_root: Path,
    data_dir: Path,
    output_root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    *,
    backend: NumericBackend | None = None,
    source_record: Mapping[str, Any] | None = None,
    source_metadata_record: Mapping[str, Any] | None = None,
    verify_model: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    actual_backend = _load_backend() if backend is None else backend
    protocol, protocol_sha = _load_protocol(protocol_path, protocol_sha256)
    _assert_bound_output(output_root, protocol)
    _, threshold = _verify_calibration(output_root, protocol_sha)
    model_path, config = _verify_formal_live_inputs(
        project_root, data_dir, protocol, source_record=source_record,
        source_metadata_record=source_metadata_record, verify_model=verify_model,
    )
    runtime = _runtime_preflight(actual_backend)
    target, staging, claim = _claim_phase(output_root, "confirm")
    access_plan = _confirmation_access_plan()
    _write_access_intent(
        staging, phase="confirm", binding_sha256=protocol_sha, access_plan=access_plan
    )
    _write_json(staging / "runtime.json", runtime)
    raw = actual_backend.confirm(
        source_path=_source_path(data_dir), access_plan=access_plan,
        model_path=model_path, config=config, threshold=threshold,
        device="cuda", dtype="float64",
    )
    if not isinstance(raw, Mapping) or set(raw) != {
        "config", "threshold", "rows", "model_parity_max_abs_error",
        "discrete_parity_equal",
    }:
        raise ValueError("confirmation backend payload fields changed")
    if validate_model_config(raw["config"]) != config or float(raw["threshold"]) != threshold:
        raise ValueError("confirmation backend config/threshold binding changed")
    _validate_parity(raw["model_parity_max_abs_error"])
    if raw["discrete_parity_equal"] is not True:
        raise ValueError("confirmation CPU/CUDA discrete outputs differ")
    rows = _validate_run_rows(
        raw["rows"], expected_idvs=ALL_IDS, expected_runs=CONFIRMATION_RUNS,
        require_alarm_fields=True,
    )
    for row in rows:
        if bool(row["alarm"]) != (float(row["run_maximum"]) > threshold):
            raise ValueError("confirmation alarm disagrees with strict frozen threshold")
    metrics = _summary_metrics(rows, gates=CONFIRMATION_GATES, development=False)
    result = {
        "schema": "mstcn_spc_confirmation_result",
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_sha,
        "calibration_result_record": _hash_record(
            output_root.resolve() / "calibrate" / "calibration_result.json"
        ),
        "threshold": threshold,
        "metrics": metrics,
        "decision": "PASS" if metrics["all_gates_pass"] else "FAIL",
        "rows": rows,
    }
    _write_json(staging / "confirmation_result.json", result)
    _finalize_phase(
        target, staging, claim, phase="confirm", binding_sha256=protocol_sha,
        artifact_names=("access_intent.json", "runtime.json", "confirmation_result.json"),
    )
    return _strict_json(target / "confirmation_result.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    develop = sub.add_parser("development-iteration")
    develop.add_argument("data_dir", type=Path)
    develop.add_argument("--iteration-id", required=True)
    develop.add_argument("--config", type=Path)
    develop.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    freeze = sub.add_parser("freeze-formal")
    freeze.add_argument("data_dir", type=Path)
    freeze.add_argument("--selected-iteration", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_FORMAL_PROTOCOL)
    for command in ("calibrate", "confirm"):
        numeric = sub.add_parser(command)
        numeric.add_argument("data_dir", type=Path)
        numeric.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        numeric.add_argument("--protocol", type=Path, default=DEFAULT_FORMAL_PROTOCOL)
        numeric.add_argument("--protocol-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "development-iteration":
        result = run_development_iteration(
            PROJECT_ROOT, args.data_dir, args.output_root, args.iteration_id,
            _load_config(args.config),
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "freeze-formal":
        print(freeze_formal(
            PROJECT_ROOT, args.data_dir, args.output_root,
            args.selected_iteration, args.protocol,
        ))
    elif args.command == "calibrate":
        result = run_calibration(
            PROJECT_ROOT, args.data_dir, args.output_root,
            args.protocol, args.protocol_sha256,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "confirm":
        result = run_confirmation(
            PROJECT_ROOT, args.data_dir, args.output_root,
            args.protocol, args.protocol_sha256,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
