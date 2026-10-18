"""Fail-closed Mode-2 development backend for PCF-CycleRank-Scan v6.

PyTorch is intentionally imported before NumPy and before any feature/data
module.  The public development entry point validates all 400 selectors and
the collision-safe fold map before constructing the Mode-2 archive.  There is
no import or path to Mode-1 or Rieth data in this module.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from mate_spc_cpetc.data_v5 import (
    ALLOWED_DUPLICATE_SEED_OWNERS,
    CONDITION_IDVS,
    MODE2_SHA256,
    OFFICIAL_MODE2_PATH,
    REQUIRED_UNIQUE_SEED_COUNT,
)

from . import CANDIDATE_ID
from .evaluation import FOLD_RUN_IDS, RunDecision, development_gates, fold_roles
from .features import (
    BFeatureStandardizer,
    PCFCycleFeatureExtractor,
    save_pcfcycle_feature_state,
)
from .model import (
    BUILD_SEED,
    DENSE_INFERENCE_ENDPOINTS,
    MONITOR_TRAIN_ENDPOINTS,
    PREFIX_ENDPOINTS,
    TRAIN_ENDPOINTS,
    PCFCycleModel,
    parameter_counts,
)
from .scan import (
    RunRelativeLogitCalibrator,
    ThresholdAudit,
    scan_many,
    validate_inner_threshold,
)


BACKEND_SCHEMA = "pcf_cyclerank_v6_development_result"
PROTOCOL_SCHEMA = "pcf_cyclerank_v6_preregistered_development_protocol"
PROTOCOL_ID = "pcf-cyclerank-scan-v6-mode2-development"
PROTOCOL_SCHEMA_VERSION = 1
CANDIDATE_COUNT = 1
GO_GATES_SHA256 = "16b640574d4c3b728c8c67a6fede3f1938f8c850fbc589d5ed582afca0b700d0"
SOURCE_FILE_SIZE = 23_897_125_720
SOURCE_MODE = 2
MAGNITUDE_PERCENT = 50
FOLD_COUNT = 5
CUDA_REPEAT_ATOL = 1.0e-6
FOLD_MAP_SHA256 = "64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13"
STAGE1_EPOCHS = 120
STAGE2_EPOCHS = 80


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if hasattr(value, "to_document"):
        return _plain_json(value.to_document())
    if hasattr(value, "to_metadata"):
        return _plain_json(value.to_metadata())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain_json(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"required artifact was not created: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _configure_determinism() -> None:
    random.seed(BUILD_SEED)
    np.random.seed(BUILD_SEED)
    torch.manual_seed(BUILD_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BUILD_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def cuda_preflight() -> Mapping[str, Any]:
    """Return the authoritative CUDA environment after deterministic setup."""

    _configure_determinism()
    available = bool(torch.cuda.is_available())
    properties = torch.cuda.get_device_properties(0) if available else None
    name = torch.cuda.get_device_name(0) if available else None
    report = {
        "cuda_available": available,
        "device_type": "cuda" if available else None,
        "device_index": 0 if available else None,
        "device_name": name,
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if available else None
        ),
        "device_total_memory": (
            int(properties.total_memory) if properties is not None else None
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "network_dtype": "float32",
        "feature_and_scan_dtype": "float64",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_repeat_atol": CUDA_REPEAT_ATOL,
        "rtx_5090": bool(name is not None and "RTX 5090" in name.upper()),
    }
    report["passed"] = bool(
        report["cuda_available"]
        and report["rtx_5090"]
        and report["deterministic_algorithms"]
        and not report["deterministic_warn_only"]
        and report["cudnn_deterministic"]
        and not report["cudnn_benchmark"]
        and report["cublas_workspace_config"] == ":4096:8"
    )
    return report


def _require_formal_cuda(device: str) -> Mapping[str, Any]:
    if device != "cuda":
        raise ValueError("formal v6 development requires device='cuda'")
    environment = dict(cuda_preflight())
    if not environment["passed"]:
        raise RuntimeError("formal v6 CUDA preflight did not pass")
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is required for formal v6 development")
    if not environment["rtx_5090"]:
        raise RuntimeError("formal v6 development is bound to an RTX 5090")
    if not environment["deterministic_algorithms"]:
        raise RuntimeError("deterministic CUDA algorithms are required")
    if environment["cublas_workspace_config"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8'")
    return environment


def _candidate_epoch(candidate: Mapping[str, Any], stage: str) -> int:
    direct = candidate.get(f"{stage}_epochs")
    if direct is not None:
        return int(direct)
    stage_document = candidate.get(stage)
    if isinstance(stage_document, Mapping) and "epochs" in stage_document:
        return int(stage_document["epochs"])
    training = candidate.get("training")
    if isinstance(training, Mapping):
        direct = training.get(f"{stage}_epochs")
        if direct is not None:
            return int(direct)
        stage_document = training.get(stage)
        if isinstance(stage_document, Mapping) and "epochs" in stage_document:
            return int(stage_document["epochs"])
    raise ValueError(f"protocol candidate is missing {stage} epochs")


def _candidate_endpoints(candidate: Mapping[str, Any], name: str) -> tuple[int, ...]:
    aliases = {
        "prefix": (
            "training_prefix",
            "prefix_endpoints",
            "fault_prefix_negative_endpoints",
        ),
        "monitor": (
            "training_monitor",
            "monitor_training_endpoints",
            "training_monitor_endpoints",
        ),
        "dense": ("dense_inference_monitor", "dense_inference_endpoints"),
    }
    containers = [candidate]
    for key in ("endpoints", "endpoint_contract"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in aliases[name]:
            if key in container:
                raw = container[key]
                if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                    raise ValueError(f"protocol {key} must be a sequence")
                if len(raw) == 3:
                    start, stop, step = (int(item) for item in raw)
                    if step <= 0 or stop < start or (stop - start) % step:
                        raise ValueError(f"protocol {key} endpoint triplet is invalid")
                    return tuple(range(start, stop + 1, step))
                return tuple(int(item) for item in raw)
    raise ValueError(f"protocol candidate is missing {name} endpoints")


def validate_protocol(protocol_doc: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the complete preregistered v6 semantic contract in memory."""

    if not isinstance(protocol_doc, Mapping):
        raise ValueError("protocol_doc must be a mapping")
    document = _plain_json(dict(protocol_doc))
    required_top = {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_state",
        "created_utc",
        "candidate",
        "candidate_count",
        "source",
        "folds",
        "access_plan",
        "development_go_gates",
        "governance",
        "reporting",
        "bindings",
        "numeric_backend_contract",
    }
    if set(document) != required_top:
        raise ValueError("v6 protocol top-level fields changed")
    if document["schema"] != PROTOCOL_SCHEMA or document["schema_version"] != 1:
        raise ValueError("v6 protocol schema changed")
    if document["protocol_id"] != PROTOCOL_ID:
        raise ValueError("v6 protocol ID changed")
    if document["candidate_count"] != CANDIDATE_COUNT:
        raise ValueError("v6 protocol must contain exactly one candidate")
    if not isinstance(document["protocol_state"], str) or not document["protocol_state"]:
        raise ValueError("v6 protocol state is missing")
    if not isinstance(document["created_utc"], str) or not document["created_utc"]:
        raise ValueError("v6 protocol creation timestamp is missing")

    candidate = document["candidate"]
    if not isinstance(candidate, Mapping) or candidate.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("v6 candidate identity changed")
    if candidate.get("candidate_count") != CANDIDATE_COUNT:
        raise ValueError("v6 candidate count changed")
    if int(candidate.get("seed", -1)) != BUILD_SEED:
        raise ValueError("v6 candidate seed changed")
    expected_data = {
        "mode": SOURCE_MODE,
        "fault_regime": "SingleFault",
        "magnitude_percent": MAGNITUDE_PERCENT,
        "idvs": list(CONDITION_IDVS),
        "runs_per_idv": 100,
        "healthy_window_zero_based_half_open": [100, 600],
        "fault_window_zero_based_half_open": [440, 940],
        "fault_local_onset_index": 160,
        "miss_delay_samples": 340,
    }
    if candidate.get("data") != expected_data:
        raise ValueError("v6 candidate data contract changed")
    if _candidate_epoch(candidate, "stage1") != STAGE1_EPOCHS:
        raise ValueError("v6 stage1 epochs changed")
    if _candidate_epoch(candidate, "stage2") != STAGE2_EPOCHS:
        raise ValueError("v6 stage2 epochs changed")
    if _candidate_endpoints(candidate, "prefix") != tuple(PREFIX_ENDPOINTS):
        raise ValueError("v6 prefix endpoints changed")
    if _candidate_endpoints(candidate, "monitor") != tuple(MONITOR_TRAIN_ENDPOINTS):
        raise ValueError("v6 monitor-training endpoints changed")
    if _candidate_endpoints(candidate, "dense") != tuple(DENSE_INFERENCE_ENDPOINTS):
        raise ValueError("v6 dense-inference endpoints changed")

    expected_source = {
        "relative_path": "data/extended_tep_dtu_mode2/TEP_Mode2.h5",
        "filename": "TEP_Mode2.h5",
        "mode": SOURCE_MODE,
        "fault_regime": "SingleFault",
        "magnitude_percent": MAGNITUDE_PERCENT,
        "idvs": list(CONDITION_IDVS),
        "sha256": MODE2_SHA256,
        "size_bytes": SOURCE_FILE_SIZE,
        "hdf5_opened_during_freeze": False,
        "mode1_access_authorized": False,
        "rieth_access_authorized": False,
    }
    if document["source"] != expected_source:
        raise ValueError("v6 Mode2 source binding changed")

    folds = document["folds"]
    canonical_fold_map = _canonical_fold_map()
    protocol_outer_roles = []
    for outer_fold in range(FOLD_COUNT):
        inner_fold = (outer_fold + 1) % FOLD_COUNT
        excluded = set(FOLD_RUN_IDS[outer_fold]) | set(FOLD_RUN_IDS[inner_fold])
        protocol_outer_roles.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fit_run_ids_per_class": [
                    run for run in range(1, 101) if run not in excluded
                ],
                "inner_healthy_run_ids": list(FOLD_RUN_IDS[inner_fold]),
                "inner_fault_values_authorized": False,
                "outer_run_ids_per_class": list(FOLD_RUN_IDS[outer_fold]),
            }
        )
    expected_folds = {
        "source": "exact v5.1 collision-safe seed-connected assignment",
        "fold_run_ids": {
            str(fold): list(FOLD_RUN_IDS[fold]) for fold in range(FOLD_COUNT)
        },
        "run_to_fold": {
            str(run): canonical_fold_map[run] for run in sorted(canonical_fold_map)
        },
        "run_to_fold_canonical_json_sha256": FOLD_MAP_SHA256,
        "cross_fold_seed_collision_count": 0,
        "outer_roles": protocol_outer_roles,
    }
    if folds != _plain_json(expected_folds):
        raise ValueError("v6 fold contract changed")
    access = document["access_plan"]
    expected_access = {
        "record_count": 400,
        "selector_count": 400,
        "canonical_sha256": ACCESS_PLAN_SHA256,
        "exact_selector_fields": list(development_access_plan()[0]),
        "wildcards_allowed": False,
    }
    if access != expected_access:
        raise ValueError("v6 protocol access-plan binding changed")
    if _sha256_document(document["development_go_gates"]) != GO_GATES_SHA256:
        raise ValueError("v6 development GO gates changed")
    governance = document["governance"]
    if (
        not isinstance(governance, Mapping)
        or governance.get("candidate_count") != CANDIDATE_COUNT
        or governance.get("mode1_access_authorized") is not False
        or governance.get("rieth_access_authorized") is not False
        or governance.get("inner_fault_numeric_access_authorized") is not False
    ):
        raise ValueError("v6 development governance changed")
    numeric = document["numeric_backend_contract"]
    expected_numeric = {
        "module": "mate_spc_pcfcycle.backend_v6",
        "required_functions": [
            "cuda_preflight",
            "validate_protocol",
            "run_development",
        ],
        "backend_import_before_access_intent": False,
        "result_schema": BACKEND_SCHEMA,
        "cuda_repeat_atol": CUDA_REPEAT_ATOL,
        "cpu_drift_is_go_gate": False,
        "authoritative_repeat_scope": (
            "per-fold inner-healthy frozen-model score -> frozen-calibrator "
            "transform -> scan -> decision double replay"
        ),
    }
    if numeric != expected_numeric:
        raise ValueError("numeric backend contract changed")
    return {
        "valid": True,
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "canonical_document_sha256": _sha256_document(document),
        "fold_map_sha256": FOLD_MAP_SHA256,
    }


def development_access_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run in range(1, 101):
        for idv in CONDITION_IDVS:
            group = (
                f"Mode2/SingleFault/SimulationCompleted/IDV{idv}/"
                f"Mode2_IDVInfo_{idv}_50/Run{run}"
            )
            result.append(
                {
                    "mode": 2,
                    "fault_regime": "SingleFault",
                    "status": "SimulationCompleted",
                    "magnitude_percent": 50,
                    "idv": idv,
                    "simulation_run": run,
                    "group_path": group,
                    "processdata_path": f"{group}/processdata",
                    "idv_init_path": f"{group}/idv_init",
                    "window_zero_based_half_open": (
                        [100, 600] if idv == 1 else [440, 940]
                    ),
                    "processdata_row_zero_based_half_open": (
                        [100, 600] if idv == 1 else [440, 940]
                    ),
                    "processdata_column_zero_based_half_open": [0, 53],
                    "idv_init_row_zero_based_half_open": [0, 3],
                    "idv_init_column_zero_based_half_open": [0, 28],
                    "group_attributes_authorized": ["seed"],
                    "dataset_metadata_authorized": [
                        "object_type",
                        "shape",
                        "dtype",
                    ],
                    "root_dataset_value_reads_authorized": [],
                    "processdata_values_outside_row_window_authorized": False,
                    "role": "healthy_control" if idv == 1 else f"fault_{idv}",
                }
            )
    return result


_EXPECTED_ACCESS_PLAN = tuple(development_access_plan())
ACCESS_PLAN_SHA256 = _sha256_document(_EXPECTED_ACCESS_PLAN)


def _validate_access_plan(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("access_plan must be a sequence")
    try:
        candidate = [_plain_json(dict(row)) for row in value]
    except (TypeError, ValueError) as error:
        raise ValueError("access_plan is malformed") from error
    expected = [dict(row) for row in _EXPECTED_ACCESS_PLAN]
    if candidate != expected:
        raise ValueError("access_plan differs from the exact 400 Mode2 selectors")
    if _sha256_document(candidate) != ACCESS_PLAN_SHA256:
        raise ValueError("access_plan hash changed")
    return candidate


def _canonical_fold_map() -> dict[int, int]:
    return {
        run: fold for fold, run_ids in FOLD_RUN_IDS.items() for run in run_ids
    }


def _validate_fold_map(value: Mapping[int, int] | Mapping[str, int]) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("fold_map must be a mapping")
    if any(isinstance(key, bool) or isinstance(item, bool) for key, item in value.items()):
        raise ValueError("boolean fold entries are forbidden")
    try:
        candidate = {int(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("fold_map contains noninteger entries") from error
    if candidate != _canonical_fold_map():
        raise ValueError("fold_map differs from the exact collision-safe allocation")
    payload = {str(key): candidate[key] for key in sorted(candidate)}
    if _sha256_document(payload) != FOLD_MAP_SHA256:
        raise ValueError("fold_map hash changed")
    return candidate


@dataclass(frozen=True)
class _NumericRun:
    """Narrow local run object built from one authorized 500-row slice."""

    name: str
    x: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def source_idv(self) -> int:
        return int(self.metadata["source_idv"])

    @property
    def simulation_run(self) -> int:
        return int(self.metadata["simulation_run"])

    @property
    def rng_seed(self) -> int:
        return int(self.metadata["rng_seed"])


@dataclass(frozen=True)
class _NumericBlock:
    simulation_run: int
    conditions: tuple[_NumericRun, ...]


def _read_seed_attribute(group: Any, *, group_path: str) -> int:
    attrs = getattr(group, "attrs", None)
    if attrs is None or "seed" not in attrs:
        raise ValueError(f"{group_path}: exact seed attribute is missing")
    raw = np.asarray(attrs["seed"])
    if raw.shape != (1,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{group_path}: seed must be one numeric element")
    value = float(raw[0])
    if not math.isfinite(value) or value <= 0.0 or not value.is_integer():
        raise ValueError(f"{group_path}: seed must be a positive integer value")
    return int(value)


def _require_dataset_metadata(
    dataset: Any,
    *,
    path: str,
    shape: tuple[int, ...],
    h5py_module: Any | None,
) -> None:
    if h5py_module is not None and not isinstance(dataset, h5py_module.Dataset):
        raise ValueError(f"{path}: object must be an HDF5 dataset")
    if tuple(getattr(dataset, "shape", ())) != shape:
        raise ValueError(f"{path}: exact shape must be {shape}")
    if np.dtype(getattr(dataset, "dtype", None)) != np.dtype(np.float64):
        raise ValueError(f"{path}: dtype must be exact float64")


def _read_exact_selector_blocks(
    handle: Any,
    access_plan: Sequence[Mapping[str, Any]],
    *,
    h5py_module: Any | None = None,
) -> tuple[Mapping[str, Any], tuple[_NumericBlock, ...]]:
    """Materialize only the 400 frozen value slices, never root labels/full time."""

    plan = _validate_access_plan(access_plan)
    by_identity: dict[tuple[int, int], _NumericRun] = {}
    owners_by_seed: dict[int, list[tuple[int, int]]] = {}
    for selector in plan:
        run_id = int(selector["simulation_run"])
        idv = int(selector["idv"])
        group_path = str(selector["group_path"])
        processdata_path = str(selector["processdata_path"])
        idv_init_path = str(selector["idv_init_path"])
        for path in (group_path, processdata_path, idv_init_path):
            if path not in handle:
                raise ValueError(f"authorized exact HDF5 path is missing: {path}")
        group = handle[group_path]
        if h5py_module is not None and not isinstance(group, h5py_module.Group):
            raise ValueError(f"{group_path}: object must be an HDF5 group")
        seed = _read_seed_attribute(group, group_path=group_path)
        processdata = handle[processdata_path]
        idv_init = handle[idv_init_path]
        _require_dataset_metadata(
            processdata,
            path=processdata_path,
            shape=(2000, 54),
            h5py_module=h5py_module,
        )
        _require_dataset_metadata(
            idv_init,
            path=idv_init_path,
            shape=(3, 28),
            h5py_module=h5py_module,
        )
        start, stop = (int(value) for value in selector["window_zero_based_half_open"])
        values_with_time = np.array(
            processdata[start:stop, 0:53],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if values_with_time.shape != (500, 53) or not np.isfinite(
            values_with_time
        ).all():
            raise ValueError(f"{processdata_path}: authorized value slice changed")
        expected_time = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
        if not np.allclose(
            values_with_time[:, 0], expected_time, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"{processdata_path}: authorized-window time changed")
        declared = np.asarray(idv_init[0:3, 0:28], dtype=np.float64)
        expected_declared = np.zeros((3, 28), dtype=np.float64)
        expected_declared[:, idv - 1] = (0.0, 0.5, 30.0)
        if (
            declared.shape != (3, 28)
            or not np.isfinite(declared).all()
            or not np.allclose(declared, expected_declared, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"{idv_init_path}: exact fault declaration changed")
        x = np.ascontiguousarray(values_with_time[:, 1:53], dtype=np.float64)
        x.setflags(write=False)
        metadata = {
            "source_idv": idv,
            "simulation_run": run_id,
            "rng_seed": seed,
            "fault_id": 0 if idv == 1 else idv,
            "group_key": run_id,
            "fold_id": _canonical_fold_map()[run_id],
            "source_group_path": group_path,
            "processdata_path": processdata_path,
            "normalized_window_half_open": [start, stop],
        }
        run = _NumericRun(
            name=f"PCFCycle-v6-Mode2-IDV{idv}-Run{run_id}",
            x=x,
            metadata=metadata,
        )
        identity = (run_id, idv)
        if identity in by_identity:
            raise RuntimeError("duplicate exact-selector identity")
        by_identity[identity] = run
        owners_by_seed.setdefault(seed, []).append(identity)

    observed_duplicates = {
        seed: tuple(sorted(owners))
        for seed, owners in owners_by_seed.items()
        if len(owners) > 1
    }
    if (
        len(by_identity) != 400
        or len(owners_by_seed) != REQUIRED_UNIQUE_SEED_COUNT
        or observed_duplicates != dict(ALLOWED_DUPLICATE_SEED_OWNERS)
    ):
        raise ValueError("Mode2 exact seed-owner topology changed")
    blocks = tuple(
        _NumericBlock(
            simulation_run=run_id,
            conditions=tuple(by_identity[(run_id, idv)] for idv in CONDITION_IDVS),
        )
        for run_id in range(1, 101)
    )
    topology = {
        "selected_run_count": 400,
        "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {str(idv): 100 for idv in CONDITION_IDVS},
        "unique_seed_count": len(owners_by_seed),
        "duplicate_seed_value_count": len(observed_duplicates),
        "duplicate_seed_owners": {
            str(seed): [list(owner) for owner in owners]
            for seed, owners in observed_duplicates.items()
        },
        "cross_fold_seed_collision_count": 0,
        "numeric_value_access": {
            "processdata_slices": 400,
            "processdata_rows_per_slice": 500,
            "processdata_columns_half_open": [0, 53],
            "idv_init_slices": 400,
            "root_dataset_value_reads": 0,
            "processdata_values_outside_authorized_windows": 0,
        },
    }
    return topology, blocks


def _authenticate_mode2_source() -> None:
    path = Path(OFFICIAL_MODE2_PATH)
    if not path.is_file() or path.name != "TEP_Mode2.h5":
        raise FileNotFoundError(path)
    if path.stat().st_size != SOURCE_FILE_SIZE:
        raise ValueError("official Mode2 source size changed")
    if _sha256_file(path) != MODE2_SHA256:
        raise ValueError("official Mode2 source SHA-256 changed")


def _load_numeric_blocks(
    access_plan: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    """Open the official archive only after validating all exact selectors."""

    plan = _validate_access_plan(access_plan)
    _authenticate_mode2_source()
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("formal v6 Mode2 loading requires h5py") from error
    with h5py.File(OFFICIAL_MODE2_PATH, "r") as handle:
        return _read_exact_selector_blocks(handle, plan, h5py_module=h5py)


def _run_id(run: Any) -> int:
    return int(run.simulation_run)


def _run_idv(run: Any) -> int:
    return int(run.source_idv)


def _class_id(run: Any) -> int:
    return 0 if _run_idv(run) == 1 else _run_idv(run)


def _validate_blocks(blocks: Sequence[Any], fold_map: Mapping[int, int]) -> tuple[Any, ...]:
    if len(blocks) != 100:
        raise ValueError("v6 development requires exactly 100 balanced blocks")
    ordered = tuple(sorted(blocks, key=lambda item: int(item.simulation_run)))
    if tuple(int(item.simulation_run) for item in ordered) != tuple(range(1, 101)):
        raise ValueError("development blocks must be exactly Run1..100")
    seen: set[tuple[int, int]] = set()
    seed_folds: dict[int, set[int]] = {}
    for block in ordered:
        run_id = int(block.simulation_run)
        conditions = tuple(block.conditions)
        if tuple(_run_idv(run) for run in conditions) != CONDITION_IDVS:
            raise ValueError("each block must contain IDV1/3/9/15 once in order")
        if hasattr(block, "fold_id") and int(block.fold_id) != fold_map[run_id]:
            raise ValueError("block fold identity differs from the frozen fold map")
        for run in conditions:
            identity = (_run_idv(run), _run_id(run))
            if _run_id(run) != run_id or identity in seen:
                raise ValueError("duplicate or inconsistent development run identity")
            seen.add(identity)
            if hasattr(run, "x"):
                values = np.asarray(run.x)
                if (
                    values.shape != (500, 52)
                    or values.dtype != np.dtype(np.float64)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("development x must be finite exact float64 500x52")
            if hasattr(run, "rng_seed"):
                seed_folds.setdefault(int(run.rng_seed), set()).add(fold_map[run_id])
    if seen != {(idv, run) for run in range(1, 101) for idv in CONDITION_IDVS}:
        raise ValueError("development run identities are incomplete")
    if any(len(owners) != 1 for owners in seed_folds.values()):
        raise ValueError("an RNG-seed collision crosses a frozen outer fold")
    return ordered


def _select_runs(
    blocks: Sequence[Any], run_ids: Sequence[int], idvs: Sequence[int]
) -> tuple[Any, ...]:
    selected_runs = set(int(value) for value in run_ids)
    selected_idvs = tuple(int(value) for value in idvs)
    result = tuple(
        run
        for block in blocks
        if int(block.simulation_run) in selected_runs
        for run in block.conditions
        if _run_idv(run) in selected_idvs
    )
    if len(result) != len(selected_runs) * len(selected_idvs):
        raise RuntimeError("fold run selection count changed")
    return result


@dataclass(frozen=True)
class FoldFitState:
    extractor: PCFCycleFeatureExtractor
    b_standardizer: BFeatureStandardizer
    model: PCFCycleModel
    stage1_history: Any
    stage2_history: Any
    runs_per_class: int


@dataclass(frozen=True)
class EndpointScorePaths:
    runs: tuple[Any, ...]
    endpoints: tuple[int, ...]
    embeddings: np.ndarray
    detection_logits: np.ndarray
    attribution_logits: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.runs)
        endpoint_count = len(self.endpoints)
        arrays = (
            (self.embeddings, (count, endpoint_count, 32), "embeddings"),
            (self.detection_logits, (count, endpoint_count), "detection_logits"),
            (
                self.attribution_logits,
                (count, endpoint_count, 3),
                "attribution_logits",
            ),
        )
        for value, shape, name in arrays:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.dtype(np.float64):
                raise ValueError(f"{name} must be exact float64 with shape {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")


def _run_values(run: Any) -> np.ndarray:
    values = np.asarray(run.x)
    if (
        values.shape != (500, 52)
        or values.dtype != np.dtype(np.float64)
        or not np.isfinite(values).all()
    ):
        raise ValueError("run.x must be finite exact float64 500x52")
    return values


def _extract_numpy_pairs(
    extractor: PCFCycleFeatureExtractor,
    b_standardizer: BFeatureStandardizer | None,
    runs: Sequence[Any],
    endpoints: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    endpoint_array = np.asarray(tuple(endpoints), dtype=np.int64)
    extracted = [
        extractor.extract_run(_run_values(run), endpoints=endpoint_array) for run in runs
    ]
    raw = np.stack([value.a for value in extracted], axis=0)
    b_raw = np.stack([value.b for value in extracted], axis=0)
    b_values = (
        b_raw
        if b_standardizer is None
        else b_standardizer.transform(b_raw.reshape(-1, b_raw.shape[-1])).reshape(
            b_raw.shape
        )
    )
    return raw, b_values


def _device_tensors(
    raw: np.ndarray, b_values: np.ndarray, *, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(device=device),
        torch.from_numpy(np.asarray(b_values, dtype=np.float32)).to(device=device),
    )


def _fit_fold_state(
    fit_runs: Sequence[Any], *, device: str, expected_run_count: int
) -> FoldFitState:
    from . import training
    from .model import build_pcfcycle_model

    by_class = {
        class_id: tuple(run for run in fit_runs if _class_id(run) == class_id)
        for class_id in (0, 3, 9, 15)
    }
    if any(len(value) != expected_run_count for value in by_class.values()):
        raise ValueError("fit set must contain the expected run count per class")
    healthy = by_class[0]
    extractor = PCFCycleFeatureExtractor.fit(
        {str(run.name): _run_values(run) for run in healthy},
        expected_run_count=expected_run_count,
    )
    healthy_raw, healthy_b_raw = _extract_numpy_pairs(
        extractor, None, healthy, MONITOR_TRAIN_ENDPOINTS
    )
    b_standardizer = BFeatureStandardizer.fit(
        healthy_b_raw.reshape(-1, healthy_b_raw.shape[-1]),
        expected_run_count=expected_run_count,
    )
    healthy_b = b_standardizer.transform(
        healthy_b_raw.reshape(-1, healthy_b_raw.shape[-1])
    ).reshape(healthy_b_raw.shape)
    model = build_pcfcycle_model(seed=BUILD_SEED, device=device)
    stage1_raw, stage1_b = _device_tensors(healthy_raw, healthy_b, device=device)
    stage1 = training.fit_stage1(
        model,
        stage1_raw,
        stage1_b,
        seed=BUILD_SEED,
        runs_per_class=expected_run_count,
    )
    del stage1_raw, stage1_b, healthy_raw, healthy_b_raw, healthy_b

    raw_by_class: dict[int, torch.Tensor] = {}
    b_by_class: dict[int, torch.Tensor] = {}
    for class_id, class_runs in by_class.items():
        raw, b_values = _extract_numpy_pairs(
            extractor, b_standardizer, class_runs, TRAIN_ENDPOINTS
        )
        raw_by_class[class_id], b_by_class[class_id] = _device_tensors(
            raw, b_values, device=device
        )
    signatures = training.precompute_stage2_signatures(
        model,
        raw_by_class,
        b_by_class,
        runs_per_class=expected_run_count,
    )
    del raw_by_class, b_by_class
    stage2 = training.fit_stage2(
        model,
        signatures,
        seed=BUILD_SEED,
        runs_per_class=expected_run_count,
    )
    if len(stage1.epochs) != STAGE1_EPOCHS or len(stage2.epochs) != STAGE2_EPOCHS:
        raise RuntimeError("production training did not retain exact 120/80 epochs")
    return FoldFitState(
        extractor,
        b_standardizer,
        model,
        stage1,
        stage2,
        expected_run_count,
    )


def _score_runs(
    state: FoldFitState,
    runs: Sequence[Any],
    endpoints: Sequence[int],
    *,
    device: str,
    run_batch_size: int = 2,
) -> EndpointScorePaths:
    from .training import infer_endpoint_scores

    selected = tuple(runs)
    endpoint_tuple = tuple(int(value) for value in endpoints)
    embeddings: list[np.ndarray] = []
    detection: list[np.ndarray] = []
    attribution: list[np.ndarray] = []
    for start in range(0, len(selected), run_batch_size):
        chunk = selected[start : start + run_batch_size]
        raw, b_values = _extract_numpy_pairs(
            state.extractor, state.b_standardizer, chunk, endpoint_tuple
        )
        raw_tensor, b_tensor = _device_tensors(raw, b_values, device=device)
        scores = infer_endpoint_scores(state.model, raw_tensor, b_tensor)
        embeddings.append(scores.embedding.detach().cpu().numpy().astype(np.float64))
        detection.append(
            scores.detection_logits.detach().cpu().numpy().astype(np.float64)
        )
        attribution.append(
            scores.attribution_logits.detach().cpu().numpy().astype(np.float64)
        )
        del raw_tensor, b_tensor
    return EndpointScorePaths(
        selected,
        endpoint_tuple,
        np.concatenate(embeddings, axis=0),
        np.concatenate(detection, axis=0),
        np.concatenate(attribution, axis=0),
    )


_SCAN_ENDPOINTS = tuple(PREFIX_ENDPOINTS) + tuple(DENSE_INFERENCE_ENDPOINTS)


def _fit_scan_calibrator(
    state: FoldFitState,
    fit_healthy: Sequence[Any],
    *,
    device: str,
    expected_run_count: int,
) -> RunRelativeLogitCalibrator:
    scores = _score_runs(state, fit_healthy, _SCAN_ENDPOINTS, device=device)
    return RunRelativeLogitCalibrator.fit(
        scores.detection_logits, expected_run_count=expected_run_count
    )


def _maximum_path_difference(left: EndpointScorePaths, right: EndpointScorePaths) -> float:
    if tuple((_run_idv(x), _run_id(x)) for x in left.runs) != tuple(
        (_run_idv(x), _run_id(x)) for x in right.runs
    ) or left.endpoints != right.endpoints:
        raise ValueError("score path identities differ")
    return max(
        float(np.max(np.abs(left.embeddings - right.embeddings))),
        float(np.max(np.abs(left.detection_logits - right.detection_logits))),
        float(np.max(np.abs(left.attribution_logits - right.attribution_logits))),
    )


def _audit_inner(
    state: FoldFitState,
    calibrator: RunRelativeLogitCalibrator,
    inner_healthy: Sequence[Any],
    *,
    device: str,
) -> tuple[ThresholdAudit, dict[str, Any]]:
    first = _score_runs(state, inner_healthy, _SCAN_ENDPOINTS, device=device)
    second = _score_runs(state, inner_healthy, _SCAN_ENDPOINTS, device=device)
    first_relative = calibrator.transform(first.detection_logits)
    second_relative = calibrator.transform(second.detection_logits)
    first_scans = scan_many(first_relative, drift=calibrator.drift)
    second_scans = scan_many(second_relative, drift=calibrator.drift)
    continuous_error = max(
        _maximum_path_difference(first, second),
        float(np.max(np.abs(first_relative - second_relative))),
        max(
            float(np.max(np.abs(left.raw_scores - right.raw_scores)))
            for left, right in zip(first_scans, second_scans)
        ),
        max(
            float(np.max(np.abs(left.scores - right.scores)))
            for left, right in zip(first_scans, second_scans)
        ),
    )
    first_maxima = np.asarray([path.run_maximum for path in first_scans])
    second_maxima = np.asarray([path.run_maximum for path in second_scans])
    threshold = validate_inner_threshold(first_maxima, repeat_error=continuous_error)
    first_alarm = tuple(path.first_exceedance(threshold.threshold) for path in first_scans)
    second_alarm = tuple(path.first_exceedance(threshold.threshold) for path in second_scans)
    discrete_equal = bool(
        np.array_equal(first_maxima, second_maxima)
        and np.array_equal(
            np.argmax(first.attribution_logits, axis=2),
            np.argmax(second.attribution_logits, axis=2),
        )
        and first_alarm == second_alarm
    )
    return threshold, {
        "schema": "pcf_cyclerank_authoritative_cuda_repeat_v1",
        "scope": "per_fold_inner_complete_score_scan_replay",
        "device": device,
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": continuous_error,
        "discrete_equal": discrete_equal,
        "passes": bool(continuous_error <= CUDA_REPEAT_ATOL and discrete_equal),
        "inner_run_maxima": first_maxima.tolist(),
    }


def _history_metadata(value: Any) -> Mapping[str, Any]:
    return value.to_metadata() if hasattr(value, "to_metadata") else dict(value)


def _save_model_atomic(state: FoldFitState, path: Path, metadata: Mapping[str, Any]) -> None:
    from .training import PCFCycleTrainingResult

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        result = PCFCycleTrainingResult(
            state.model, state.stage1_history, state.stage2_history
        )
        result.save(temporary, extra_metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_fold_artifacts(
    state: FoldFitState,
    calibrator: RunRelativeLogitCalibrator,
    threshold: ThresholdAudit,
    fold_dir: Path,
    *,
    outer_fold: int,
    roles: Mapping[str, Sequence[int]],
) -> Mapping[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=False)
    feature_path = fold_dir / "feature_state.npz"
    model_path = fold_dir / "model.pt"
    calibrator_path = fold_dir / "scan_calibrator.json"
    threshold_path = fold_dir / "inner_threshold.json"
    common = {
        "candidate_id": CANDIDATE_ID,
        "outer_fold": outer_fold,
        "roles": _plain_json(roles),
        "frozen_before_outer_scoring": True,
    }
    save_pcfcycle_feature_state(
        feature_path,
        state.extractor,
        state.b_standardizer,
        metadata=common,
    )
    _save_model_atomic(state, model_path, common)
    _atomic_json(calibrator_path, calibrator.to_document())
    _atomic_json(threshold_path, threshold.to_document())
    records = {
        "feature_state": _file_record(feature_path),
        "model": _file_record(model_path),
        "scan_calibrator": _file_record(calibrator_path),
        "inner_threshold": _file_record(threshold_path),
    }
    manifest = {
        "schema": "pcf_cyclerank_frozen_fold_artifacts_v1",
        **common,
        "parameter_counts": parameter_counts(state.model),
        "stage1_history": _history_metadata(state.stage1_history),
        "stage2_history": _history_metadata(state.stage2_history),
        "artifacts": records,
    }
    manifest_path = fold_dir / "freeze_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**records, "freeze_manifest": _file_record(manifest_path)}


def _decisions_from_scores(
    scores: EndpointScorePaths,
    calibrator: RunRelativeLogitCalibrator,
    threshold: float,
) -> tuple[tuple[RunDecision, ...], list[dict[str, Any]]]:
    relative = calibrator.transform(scores.detection_logits)
    scans = scan_many(relative, drift=calibrator.drift)
    fault_lookup = (3, 9, 15)
    decisions: list[RunDecision] = []
    replay: list[dict[str, Any]] = []
    prefix_count = len(PREFIX_ENDPOINTS)
    for index, (run, path) in enumerate(zip(scores.runs, scans)):
        alarm_endpoint = path.first_exceedance(threshold)
        attributed = None
        if alarm_endpoint is not None:
            dense_index = int(alarm_endpoint) - DENSE_INFERENCE_ENDPOINTS[0]
            logits_index = prefix_count + dense_index
            attributed = fault_lookup[
                int(np.argmax(scores.attribution_logits[index, logits_index]))
            ]
        decision = RunDecision(
            source_idv=_run_idv(run),
            simulation_run=_run_id(run),
            run_maximum=path.run_maximum,
            alarm_endpoint=alarm_endpoint,
            attributed_fault=attributed,
        )
        decisions.append(decision)
        replay.append(
            {
                "identity": {
                    "source_idv": _run_idv(run),
                    "simulation_run": _run_id(run),
                },
                "detection_logits_prefix_then_dense": (
                    scores.detection_logits[index].tolist()
                ),
                "attribution_logits_prefix_then_dense": (
                    scores.attribution_logits[index].tolist()
                ),
                "run_relative_logits": relative[index].tolist(),
                "raw_scan_endpoints": path.raw_endpoints.tolist(),
                "raw_scan_scores": path.raw_scores.tolist(),
                "persistent_endpoints": path.endpoints.tolist(),
                "persistent_scores": path.scores.tolist(),
                "decision": decision.to_document(),
            }
        )
    return tuple(decisions), replay


def _score_outer(
    state: FoldFitState,
    calibrator: RunRelativeLogitCalibrator,
    threshold: float,
    outer_runs: Sequence[Any],
    *,
    device: str,
) -> tuple[tuple[RunDecision, ...], list[dict[str, Any]], EndpointScorePaths]:
    scores = _score_runs(state, outer_runs, _SCAN_ENDPOINTS, device=device)
    decisions, replay = _decisions_from_scores(scores, calibrator, threshold)
    return decisions, replay, scores


def _cross_device_report(
    state: FoldFitState,
    authoritative: EndpointScorePaths,
    calibrator: RunRelativeLogitCalibrator,
    threshold: float,
) -> Mapping[str, Any]:
    cpu_model = PCFCycleModel()
    cpu_model.load_state_dict(
        {name: tensor.detach().cpu() for name, tensor in state.model.state_dict().items()},
        strict=True,
    )
    cpu_state = replace(state, model=cpu_model.to(dtype=torch.float32, device="cpu"))
    cpu = _score_runs(
        cpu_state, authoritative.runs, authoritative.endpoints, device="cpu"
    )
    maximum_error = _maximum_path_difference(authoritative, cpu)
    left, _ = _decisions_from_scores(authoritative, calibrator, threshold)
    right, _ = _decisions_from_scores(cpu, calibrator, threshold)
    discrete_equal = tuple(
        (item.alarm_endpoint, item.attributed_fault) for item in left
    ) == tuple((item.alarm_endpoint, item.attributed_fault) for item in right)
    return {
        "is_go_gate": False,
        "authoritative_device": "cuda",
        "comparison_device": "cpu",
        "maximum_absolute_error": maximum_error,
        "discrete_equal": discrete_equal,
    }


def _fold_metrics(rows: Sequence[RunDecision]) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "healthy_false_alarms": sum(row.alarm for row in rows if row.source_idv == 1),
        "faults": {},
    }
    for fault in (3, 9, 15):
        selected = tuple(row for row in rows if row.source_idv == fault)
        delays = np.asarray([row.delay for row in selected], dtype=np.float64)
        result["faults"][str(fault)] = {
            "detections": sum(row.alarm for row in selected),
            "detection_rate": sum(row.alarm for row in selected) / 20.0,
            "median_delay_samples": float(np.median(delays)),
        }
    return result


def _save_outer_result(
    fold_dir: Path,
    *,
    outer_fold: int,
    decisions: Sequence[RunDecision],
    replay: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    path = fold_dir / "outer_result.json"
    document = {
        "schema": "pcf_cyclerank_outer_fold_result_v1",
        "outer_fold": outer_fold,
        "authoritative_device": "cuda",
        "scored_once_after_freeze": True,
        "metrics": _fold_metrics(decisions),
        "decisions": [item.to_document() for item in decisions],
        "replay": list(replay),
    }
    _atomic_json(path, document)
    return _file_record(path)


def _execute_fold(
    blocks: Sequence[Any],
    *,
    outer_fold: int,
    artifact_root: Path,
    device: str,
) -> tuple[dict[str, Any], tuple[RunDecision, ...]]:
    roles = fold_roles(outer_fold)
    fit_runs = _select_runs(blocks, roles["fit"], CONDITION_IDVS)
    fit_healthy = _select_runs(blocks, roles["fit"], (1,))
    inner_healthy = _select_runs(blocks, roles["inner_healthy"], (1,))
    outer_runs = _select_runs(blocks, roles["outer"], CONDITION_IDVS)
    state = _fit_fold_state(fit_runs, device=device, expected_run_count=60)
    calibrator = _fit_scan_calibrator(
        state, fit_healthy, device=device, expected_run_count=60
    )
    threshold, repeat = _audit_inner(
        state, calibrator, inner_healthy, device=device
    )
    fold_dir = artifact_root / f"fold_{outer_fold}"
    artifacts = _save_fold_artifacts(
        state,
        calibrator,
        threshold,
        fold_dir,
        outer_fold=outer_fold,
        roles=roles,
    )
    decisions, replay, authoritative = _score_outer(
        state,
        calibrator,
        threshold.threshold,
        outer_runs,
        device=device,
    )
    outer_artifact = _save_outer_result(
        fold_dir,
        outer_fold=outer_fold,
        decisions=decisions,
        replay=replay,
    )
    cross_device = _cross_device_report(
        state, authoritative, calibrator, threshold.threshold
    )
    numeric_valid = bool(threshold.valid and repeat["passes"])
    document = {
        "outer_fold": outer_fold,
        "roles": _plain_json(roles),
        "fit_counts": {"runs_per_class": 60, "total_runs": 240},
        "inner_healthy_count": 20,
        "outer_counts": {"runs_per_class": 20, "total_runs": 80},
        "threshold": threshold.to_document(),
        "authoritative_cuda_repeat": repeat,
        "cross_device_report_only": cross_device,
        "numeric_valid": numeric_valid,
        "artifacts_frozen_before_outer": artifacts,
        "outer_result_append_only": outer_artifact,
        "metrics": _fold_metrics(decisions),
        "decisions": [item.to_document() for item in decisions],
    }
    return document, decisions


def _run_final_refit(
    blocks: Sequence[Any], *, artifact_root: Path, device: str
) -> Mapping[str, Any]:
    all_runs = _select_runs(blocks, range(1, 101), CONDITION_IDVS)
    state = _fit_fold_state(all_runs, device=device, expected_run_count=100)
    all_healthy = _select_runs(blocks, range(1, 101), (1,))
    calibrator = _fit_scan_calibrator(
        state, all_healthy, device=device, expected_run_count=100
    )
    final_dir = artifact_root / "final_refit"
    final_dir.mkdir(parents=True, exist_ok=False)
    feature_path = final_dir / "feature_state.npz"
    model_path = final_dir / "model.pt"
    calibrator_path = final_dir / "scan_calibrator.json"
    metadata = {
        "candidate_id": CANDIDATE_ID,
        "runs_per_class": 100,
        "scan_calibrator_fit_healthy_runs": 100,
        "created_only_after_development_go": True,
        "target_domain_calibration_not_performed": True,
    }
    save_pcfcycle_feature_state(
        feature_path, state.extractor, state.b_standardizer, metadata=metadata
    )
    _save_model_atomic(state, model_path, metadata)
    _atomic_json(calibrator_path, calibrator.to_document())
    manifest = {
        "schema": "pcf_cyclerank_final_mode2_refit_v1",
        **metadata,
        "parameter_counts": parameter_counts(state.model),
        "feature_state": _file_record(feature_path),
        "model": _file_record(model_path),
        "scan_calibrator": _file_record(calibrator_path),
    }
    manifest_path = final_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": _file_record(manifest_path)}


def _validate_protocol_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("protocol_sha256 must be lowercase hexadecimal SHA-256")
    return value


def _numeric_no_go_result(
    *,
    protocol_sha256: str,
    environment: Mapping[str, Any],
    topology: Mapping[str, Any],
    completed_folds: Sequence[Mapping[str, Any]],
    failed_fold: int,
    error: BaseException,
) -> dict[str, Any]:
    folds = list(completed_folds)
    folds.append(
        {
            "outer_fold": failed_fold,
            "numeric_valid": False,
            "status": "FOLD_NUMERIC_NO_GO",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    for fold in range(failed_fold + 1, FOLD_COUNT):
        folds.append(
            {
                "outer_fold": fold,
                "numeric_valid": False,
                "status": "NOT_RUN_AFTER_NUMERIC_NO_GO",
            }
        )
    return {
        "schema": BACKEND_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "source_sha256": MODE2_SHA256,
        "source_size_bytes": SOURCE_FILE_SIZE,
        "fold_map_sha256": FOLD_MAP_SHA256,
        "access_plan_sha256": ACCESS_PLAN_SHA256,
        "topology": _plain_json(topology),
        "environment": _plain_json(environment),
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
        "mode1_access_count": 0,
        "rieth_accessed": False,
        "rieth_access_count": 0,
        "decision": "DEVELOPMENT_NO_GO",
        "selected_candidate_id": None,
        "folds": folds,
        "gates": {
            "all_pass": False,
            "checks": {"numeric_all_folds": False},
            "aggregate_metrics": None,
        },
        "aggregate": None,
        "authoritative_cuda_repeat": {
            "scope": "per_fold_inner_complete_score_scan_replay",
            "discrete_equal": False,
            "passes": False,
            "atol": CUDA_REPEAT_ATOL,
        },
        "cross_device_report_only": {"is_go_gate": False, "folds": []},
        "final_refit": None,
    }


def run_development(
    *,
    access_plan: Sequence[Mapping[str, Any]],
    fold_map: Mapping[int, int],
    artifact_root: str | os.PathLike[str],
    protocol_sha256: str,
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Run the five preregistered outer folds and optional GO-only final refit."""

    # These checks are intentionally complete before _load_numeric_blocks can
    # construct an archive or import h5py through its lazy context manager.
    validated_access = _validate_access_plan(access_plan)
    validated_folds = _validate_fold_map(fold_map)
    protocol_hash = _validate_protocol_sha256(protocol_sha256)
    root = Path(artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("artifact_root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    environment = _require_formal_cuda(device)
    topology, raw_blocks = _load_numeric_blocks(validated_access)
    blocks = _validate_blocks(raw_blocks, validated_folds)

    fold_documents: list[dict[str, Any]] = []
    decisions: list[RunDecision] = []
    numeric_errors = (FloatingPointError, np.linalg.LinAlgError)
    for outer_fold in range(FOLD_COUNT):
        try:
            document, fold_decisions = _execute_fold(
                blocks,
                outer_fold=outer_fold,
                artifact_root=root,
                device=device,
            )
        except numeric_errors as error:
            result = _numeric_no_go_result(
                protocol_sha256=protocol_hash,
                environment=environment,
                topology=topology,
                completed_folds=fold_documents,
                failed_fold=outer_fold,
                error=error,
            )
            _atomic_json(root / "development_decision.json", result)
            return result
        fold_documents.append(document)
        decisions.extend(fold_decisions)

    gate_document = development_gates(
        decisions,
        numeric_folds_valid=[bool(item["numeric_valid"]) for item in fold_documents],
    )
    decision = (
        "DEVELOPMENT_GO"
        if gate_document["all_pass"]
        else "DEVELOPMENT_NO_GO"
    )
    repeats = [item["authoritative_cuda_repeat"] for item in fold_documents]
    authoritative = {
        "scope": "per_fold_inner_complete_score_scan_replay",
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": max(
            float(item["maximum_absolute_error"]) for item in repeats
        ),
        "discrete_equal": all(bool(item["discrete_equal"]) for item in repeats),
        "passes": all(bool(item["passes"]) for item in repeats),
        "folds": repeats,
    }
    cross_device_folds = [item["cross_device_report_only"] for item in fold_documents]
    cross_device = {
        "is_go_gate": False,
        "maximum_absolute_error": max(
            float(item["maximum_absolute_error"]) for item in cross_device_folds
        ),
        "discrete_equal": all(
            bool(item["discrete_equal"]) for item in cross_device_folds
        ),
        "folds": cross_device_folds,
    }
    result: dict[str, Any] = {
        "schema": BACKEND_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_hash,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "source_sha256": MODE2_SHA256,
        "source_size_bytes": SOURCE_FILE_SIZE,
        "fold_map_sha256": FOLD_MAP_SHA256,
        "access_plan_sha256": ACCESS_PLAN_SHA256,
        "topology": _plain_json(topology),
        "environment": _plain_json(environment),
        "mode1_accessed": False,
        "mode1_hdf5_opens_or_value_reads": 0,
        "mode1_access_count": 0,
        "rieth_accessed": False,
        "rieth_access_count": 0,
        "decision": decision,
        "selected_candidate_id": (
            CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None
        ),
        "folds": fold_documents,
        "gates": _plain_json(gate_document),
        "aggregate": _plain_json(gate_document["aggregate_metrics"]),
        "authoritative_cuda_repeat": authoritative,
        "cross_device_report_only": cross_device,
        "final_refit": None,
    }
    # Persist the immutable development conclusion before a potentially long
    # GO-only final refit.  A refit infrastructure failure cannot erase it.
    _atomic_json(root / "development_decision.json", result)
    if decision == "DEVELOPMENT_GO":
        result["final_refit"] = _plain_json(
            _run_final_refit(blocks, artifact_root=root, device=device)
        )
        _atomic_json(root / "final_refit_result.json", result["final_refit"])
    return result


__all__ = [
    "ACCESS_PLAN_SHA256",
    "BACKEND_SCHEMA",
    "CANDIDATE_COUNT",
    "CUDA_REPEAT_ATOL",
    "FOLD_MAP_SHA256",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "cuda_preflight",
    "development_access_plan",
    "run_development",
    "validate_protocol",
]
