"""Fail-closed Mode-2 development backend for FARM-RankCUSUM v7.

The backend accepts only the frozen 400 Mode-2/magnitude-50 selectors.  It
validates the complete selector list and collision-safe fold map before the
first HDF5 open, then reads only each selector's authorized 500-row window.
Root labels and processdata values outside those windows are never decoded.

Protocol identity deliberately uses ``protocol_canonical_json_sha256``.  A
file-byte SHA-256 (which includes the final LF) is a separate runner/sidecar
identity and is never compared with the backend's parsed-document hash.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from mate_spc.hash_contract import (
    canonical_json_file_bytes,
    canonical_json_sha256,
    sha256_file,
)
from mate_spc_cpetc.data_v5 import (
    ALLOWED_DUPLICATE_SEED_OWNERS,
    CONDITION_IDVS,
    FOLD_RUN_IDS,
    MODE2_SHA256,
    OFFICIAL_MODE2_PATH,
    REQUIRED_UNIQUE_SEED_COUNT,
)


PROTOCOL_SCHEMA = "farm_rankcusum_v7_preregistered_development_protocol"
PROTOCOL_ID = "farm-rankcusum-v7-mode2-development"
BACKEND_RESULT_SCHEMA = "farm_rankcusum_v7_development_result"
CANDIDATE_ID = "farm-rankcusum-v7-fixed"
CANDIDATE_COUNT = 1
SOURCE_SIZE_BYTES = 23_897_125_720
CUDA_REPEAT_ATOL = 1.0e-6
FOLD_COUNT = 5
REFERENCE_CHANNEL_COUNT = 46
SEQUENCE_LENGTH = 500
FEATURE_COUNT = 52
FIT_HEALTHY_RUNS = 60
INNER_HEALTHY_RUNS = 20
FINAL_HEALTHY_RUNS = 100
MONITOR_LENGTH = 340


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
    raise TypeError(f"cannot serialize {type(value).__name__} to strict JSON")


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(_plain_json(value))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _replace_json(path: Path, value: Any) -> None:
    """Atomically create or replace the single mutable access-state record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(_plain_json(value))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _AccessState:
    """Crash-visible state machine for the numeric-access boundary."""

    def __init__(self, path: Path, *, protocol_canonical_json_sha256: str) -> None:
        self.path = path
        self.value: dict[str, Any] = {
            "schema": "farm_rankcusum_v7_backend_access_state",
            "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
            "hdf5_open_started": False,
            "hdf5_open_completed": False,
            "processdata_slices_read": 0,
            "idv_init_slices_read": 0,
            "numeric_slices_read": 0,
            "training_or_fit_started": False,
            "backend_completed": False,
            "mode1_opens": 0,
        }
        _replace_json(self.path, self.value)

    def update(self, **changes: Any) -> None:
        unknown = set(changes) - set(self.value)
        if unknown:
            raise KeyError(f"unknown access-state fields: {sorted(unknown)}")
        candidate = dict(self.value)
        candidate.update(changes)
        if candidate["mode1_opens"] != 0:
            raise RuntimeError("Mode1 opens are forbidden")
        if candidate["numeric_slices_read"] != (
            candidate["processdata_slices_read"]
            + candidate["idv_init_slices_read"]
        ):
            raise RuntimeError("numeric slice counters disagree")
        self.value = candidate
        _replace_json(self.path, self.value)

    def record_processdata_slice(self) -> None:
        self.update(
            processdata_slices_read=self.value["processdata_slices_read"] + 1,
            numeric_slices_read=self.value["numeric_slices_read"] + 1,
        )

    def record_idv_init_slice(self) -> None:
        self.update(
            idv_init_slices_read=self.value["idv_init_slices_read"] + 1,
            numeric_slices_read=self.value["numeric_slices_read"] + 1,
        )


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact was not created: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_bytes_sha256": sha256_file(path),
    }


def _configure_determinism() -> None:
    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260910)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def cuda_preflight() -> Mapping[str, Any]:
    """Configure and report the authoritative RTX-5090 CUDA runtime."""

    _configure_determinism()
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    properties = torch.cuda.get_device_properties(0) if available else None
    report = {
        "cuda_available": available,
        "device_name": name,
        "device_index": 0 if available else None,
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
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "authoritative_cuda_repeat_atol": CUDA_REPEAT_ATOL,
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


def _require_cuda(device: str) -> Mapping[str, Any]:
    if device != "cuda":
        raise ValueError("formal FARM v7 development requires device='cuda'")
    report = dict(cuda_preflight())
    if report["passed"] is not True:
        raise RuntimeError("formal FARM v7 CUDA preflight did not pass")
    return report


def development_access_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run_id in range(1, 101):
        for idv in CONDITION_IDVS:
            group = (
                f"Mode2/SingleFault/SimulationCompleted/IDV{idv}/"
                f"Mode2_IDVInfo_{idv}_50/Run{run_id}"
            )
            window = [100, 600] if idv == 1 else [440, 940]
            result.append(
                {
                    "mode": 2,
                    "fault_regime": "SingleFault",
                    "status": "SimulationCompleted",
                    "magnitude_percent": 50,
                    "idv": idv,
                    "simulation_run": run_id,
                    "group_path": group,
                    "processdata_path": f"{group}/processdata",
                    "idv_init_path": f"{group}/idv_init",
                    "window_zero_based_half_open": window,
                    "processdata_row_zero_based_half_open": window,
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
ACCESS_PLAN_CANONICAL_JSON_SHA256 = canonical_json_sha256(_EXPECTED_ACCESS_PLAN)


def _validate_access_plan(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("access_plan must be a sequence")
    candidate = _plain_json(list(value))
    if candidate != list(_EXPECTED_ACCESS_PLAN):
        raise ValueError("access_plan differs from the exact 400 FARM selectors")
    if canonical_json_sha256(candidate) != ACCESS_PLAN_CANONICAL_JSON_SHA256:
        raise ValueError("access_plan canonical JSON SHA-256 changed")
    return candidate


def _canonical_fold_map() -> dict[int, int]:
    return {
        run_id: fold
        for fold, run_ids in FOLD_RUN_IDS.items()
        for run_id in run_ids
    }


FOLD_MAP_CANONICAL_JSON_SHA256 = canonical_json_sha256(
    {str(run): fold for run, fold in sorted(_canonical_fold_map().items())}
)


def _validate_fold_map(value: Mapping[int, int] | Mapping[str, int]) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("fold_map must be a mapping")
    if any(isinstance(key, bool) or isinstance(item, bool) for key, item in value.items()):
        raise ValueError("boolean fold entries are forbidden")
    try:
        candidate = {int(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("fold_map entries must be integers") from error
    if candidate != _canonical_fold_map():
        raise ValueError("fold_map differs from the collision-safe allocation")
    return candidate


def validate_protocol(
    protocol_doc: Mapping[str, Any],
    *,
    expected_canonical_json_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate the parsed protocol in the canonical-JSON hash domain only."""

    if not isinstance(protocol_doc, Mapping):
        raise ValueError("protocol_doc must be a mapping")
    document = _plain_json(dict(protocol_doc))
    observed_hash = canonical_json_sha256(document)
    if expected_canonical_json_sha256 is not None:
        if (
            not isinstance(expected_canonical_json_sha256, str)
            or len(expected_canonical_json_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_canonical_json_sha256
            )
        ):
            raise ValueError("expected_canonical_json_sha256 must be lowercase SHA-256")
        if observed_hash != expected_canonical_json_sha256:
            raise ValueError("protocol canonical_json_sha256 mismatch")
    if (
        document.get("schema") != PROTOCOL_SCHEMA
        or document.get("schema_version") != 1
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("candidate_count") != CANDIDATE_COUNT
    ):
        raise ValueError("FARM v7 protocol identity changed")
    candidate = document.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("candidate_id") != CANDIDATE_ID
        or candidate.get("candidate_count") != CANDIDATE_COUNT
    ):
        raise ValueError("FARM v7 candidate identity changed")
    data = candidate.get("data")
    if not isinstance(data, Mapping) or (
        data.get("mode"),
        data.get("fault_regime"),
        data.get("magnitude_percent"),
        data.get("idvs"),
        data.get("runs_per_idv"),
        data.get("run_length"),
        data.get("feature_count"),
        data.get("rows_per_run"),
        data.get("process_channel_count"),
        data.get("healthy_window_zero_based_half_open"),
        data.get("fault_window_zero_based_half_open"),
        data.get("fault_local_onset_index"),
        data.get("monitor_endpoints_inclusive"),
        data.get("miss_delay_samples"),
    ) != (
        2,
        "SingleFault",
        50,
        [1, 3, 9, 15],
        100,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
        [100, 600],
        [440, 940],
        160,
        [160, 499],
        340,
    ):
        raise ValueError("FARM v7 candidate data contract changed")
    channels = candidate.get("channels")
    expected_channels = [*range(41), 46, 47, 48, 50, 51]
    if not isinstance(channels, Mapping) or (
        channels.get("zero_based_indices"),
        channels.get("count"),
    ) != (expected_channels, REFERENCE_CHANNEL_COUNT):
        raise ValueError("FARM v7 channel contract changed")
    reference = candidate.get("reference")
    if not isinstance(reference, Mapping) or (
        reference.get("selected_channel_count"),
        reference.get("fit_healthy_runs_per_fold"),
        reference.get("fit_reference_values_per_channel"),
        reference.get("fault_runs_used"),
        reference.get("allowed_healthy_run_counts"),
        reference.get("development_healthy_run_count"),
        reference.get("go_only_refit_healthy_run_count"),
        reference.get("rows_per_run"),
        reference.get("development_reference_shape"),
        reference.get("go_only_refit_reference_shape"),
        reference.get("sort"),
        reference.get("dtype"),
        reference.get("normalization"),
        reference.get("fault_values_used"),
    ) != (
        REFERENCE_CHANNEL_COUNT,
        FIT_HEALTHY_RUNS,
        30_000,
        0,
        [60, 100],
        FIT_HEALTHY_RUNS,
        100,
        SEQUENCE_LENGTH,
        [REFERENCE_CHANNEL_COUNT, 30_000],
        [REFERENCE_CHANNEL_COUNT, 50_000],
        "stable ascending per channel",
        "float64",
        False,
        False,
    ):
        raise ValueError("FARM v7 healthy reference contract changed")
    monitor = candidate.get("monitor")
    if not isinstance(monitor, Mapping) or (
        monitor.get("searchsorted_side"),
        monitor.get("posterior_ecdf"),
        monitor.get("two_sided"),
        monitor.get("k"),
        monitor.get("top_r"),
        monitor.get("reset_endpoint"),
        monitor.get("dense_endpoints_inclusive"),
        monitor.get("alarm_comparison"),
        monitor.get("persistence"),
        monitor.get("dtype"),
    ) != (
        "left",
        "(searchsorted_left+1)/(N+2)",
        True,
        1.3,
        4,
        160,
        [160, 499],
        ">",
        1,
        "float64",
    ):
        raise ValueError("FARM v7 monitor contract changed")
    threshold = candidate.get("inner_threshold")
    if not isinstance(threshold, Mapping) or (
        threshold.get("healthy_runs"),
        threshold.get("fault_inner_values_used"),
        threshold.get("ascending_rank"),
        threshold.get("comparison"),
        threshold.get("minimum_exact_distinct_maxima"),
        threshold.get("tail_gap_multiplier"),
        threshold.get("repeat_error_floor"),
        threshold.get("reachable_upper_bound"),
    ) != (
        INNER_HEALTHY_RUNS,
        False,
        20,
        ">",
        18,
        100.0,
        1.0e-10,
        "340*4*(log(N+2)-1.3)",
    ):
        raise ValueError("FARM v7 inner-threshold contract changed")
    determinism = candidate.get("determinism")
    if not isinstance(determinism, Mapping) or (
        determinism.get("authoritative_device"),
        determinism.get("authoritative_repeat_scope"),
        determinism.get("authoritative_repeat_atol"),
        determinism.get("authoritative_discrete_exact"),
        determinism.get("cross_device_cpu_cuda"),
    ) != (
        "CUDA RTX 5090",
        "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        CUDA_REPEAT_ATOL,
        True,
        "report_only_not_a_go_gate",
    ):
        raise ValueError("FARM v7 determinism contract changed")
    final_refit = candidate.get("go_only_final_refit")
    if not isinstance(final_refit, Mapping) or (
        final_refit.get("healthy_runs"),
        final_refit.get("fault_runs_used"),
        final_refit.get("target_domain_calibration_performed"),
        final_refit.get("mode1_access_authorized"),
    ) != (100, 0, False, False):
        raise ValueError("FARM v7 final-refit contract changed")
    source = document.get("source")
    if not isinstance(source, Mapping) or (
        source.get("sha256"),
        source.get("size_bytes"),
        source.get("mode"),
        source.get("magnitude_percent"),
        source.get("hdf5_opened_during_freeze"),
        source.get("mode1_access_authorized"),
        source.get("rieth_access_authorized"),
    ) != (MODE2_SHA256, SOURCE_SIZE_BYTES, 2, 50, False, False, False):
        raise ValueError("FARM v7 Mode2 source binding changed")
    access = document.get("access_plan")
    if not isinstance(access, Mapping) or (
        access.get("record_count"),
        access.get("selector_count"),
        access.get("canonical_json_sha256"),
        access.get("wildcards_allowed"),
    ) != (400, 400, ACCESS_PLAN_CANONICAL_JSON_SHA256, False):
        raise ValueError("FARM v7 exact access-plan binding changed")
    folds = document.get("folds")
    expected_folds = {
        str(fold): list(FOLD_RUN_IDS[fold]) for fold in range(FOLD_COUNT)
    }
    if not isinstance(folds, Mapping) or (
        folds.get("fold_run_ids") != expected_folds
        or folds.get("run_to_fold_canonical_json_sha256")
        != FOLD_MAP_CANONICAL_JSON_SHA256
    ):
        raise ValueError("FARM v7 fold binding changed")
    governance = document.get("governance")
    if (
        not isinstance(governance, Mapping)
        or governance.get("mode1_access_authorized") is not False
        or governance.get("rieth_access_authorized") is not False
        or governance.get("inner_fault_numeric_access_authorized") is not False
        or governance.get("fault_fit_or_threshold_authorized") is not False
    ):
        raise ValueError("FARM v7 governance binding changed")
    numeric = document.get("numeric_backend_contract")
    if not isinstance(numeric, Mapping) or (
        numeric.get("module"),
        numeric.get("required_functions"),
        numeric.get("hdf5_import_or_open_before_intent"),
        numeric.get("result_schema"),
        numeric.get("protocol_identity_field"),
        numeric.get("cuda_repeat_atol"),
        numeric.get("cpu_drift_is_go_gate"),
    ) != (
        "mate_spc_farm_v7.backend",
        ["cuda_preflight", "validate_protocol", "run_development"],
        False,
        BACKEND_RESULT_SCHEMA,
        "protocol_canonical_json_sha256",
        CUDA_REPEAT_ATOL,
        False,
    ):
        raise ValueError("FARM v7 numeric backend contract changed")
    return {
        "valid": True,
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "canonical_json_sha256": observed_hash,
        "protocol_canonical_json_sha256": observed_hash,
        "fold_map_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "access_plan_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "hash_domain": "canonical_json_without_file_terminator",
        "file_bytes_sha256_compared": False,
    }


@dataclass(frozen=True)
class NumericRun:
    name: str
    x: np.ndarray
    source_idv: int
    simulation_run: int
    rng_seed: int


@dataclass(frozen=True)
class NumericBlock:
    simulation_run: int
    conditions: tuple[NumericRun, ...]


def _read_seed(group: Any, *, path: str) -> int:
    attrs = getattr(group, "attrs", None)
    if attrs is None or "seed" not in attrs:
        raise ValueError(f"{path}: exact seed attribute is missing")
    value = np.asarray(attrs["seed"])
    if value.shape != (1,) or value.dtype.kind not in "iuf":
        raise ValueError(f"{path}: seed must be one numeric element")
    scalar = float(value[0])
    if not math.isfinite(scalar) or scalar <= 0.0 or not scalar.is_integer():
        raise ValueError(f"{path}: seed must be a positive integer")
    return int(scalar)


def _validate_dataset(
    value: Any,
    *,
    path: str,
    shape: tuple[int, ...],
    h5py_module: Any | None,
) -> None:
    if h5py_module is not None and not isinstance(value, h5py_module.Dataset):
        raise ValueError(f"{path}: object must be an HDF5 dataset")
    if tuple(getattr(value, "shape", ())) != shape:
        raise ValueError(f"{path}: exact shape must be {shape}")
    if np.dtype(getattr(value, "dtype", None)) != np.dtype(np.float64):
        raise ValueError(f"{path}: dtype must be exact float64")


def _read_exact_blocks(
    handle: Any,
    access_plan: Sequence[Mapping[str, Any]],
    *,
    h5py_module: Any | None = None,
    access_state: _AccessState | None = None,
) -> tuple[Mapping[str, Any], tuple[NumericBlock, ...]]:
    plan = _validate_access_plan(access_plan)
    by_identity: dict[tuple[int, int], NumericRun] = {}
    seed_owners: dict[int, list[tuple[int, int]]] = {}
    for selector in plan:
        run_id = int(selector["simulation_run"])
        idv = int(selector["idv"])
        group_path = str(selector["group_path"])
        process_path = str(selector["processdata_path"])
        idv_init_path = str(selector["idv_init_path"])
        for path in (group_path, process_path, idv_init_path):
            if path not in handle:
                raise ValueError(f"authorized exact HDF5 path is missing: {path}")
        group = handle[group_path]
        if h5py_module is not None and not isinstance(group, h5py_module.Group):
            raise ValueError(f"{group_path}: object must be an HDF5 group")
        seed = _read_seed(group, path=group_path)
        processdata = handle[process_path]
        idv_init = handle[idv_init_path]
        _validate_dataset(
            processdata,
            path=process_path,
            shape=(2000, 54),
            h5py_module=h5py_module,
        )
        _validate_dataset(
            idv_init,
            path=idv_init_path,
            shape=(3, 28),
            h5py_module=h5py_module,
        )
        start, stop = (int(item) for item in selector["window_zero_based_half_open"])
        decoded = np.array(
            processdata[start:stop, 0:53],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if access_state is not None:
            access_state.record_processdata_slice()
        if decoded.shape != (SEQUENCE_LENGTH, 53) or not np.isfinite(decoded).all():
            raise ValueError(f"{process_path}: authorized slice changed")
        expected_time = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
        if not np.allclose(decoded[:, 0], expected_time, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{process_path}: authorized-window time changed")
        declaration = np.asarray(idv_init[0:3, 0:28], dtype=np.float64)
        if access_state is not None:
            access_state.record_idv_init_slice()
        expected = np.zeros((3, 28), dtype=np.float64)
        expected[:, idv - 1] = (0.0, 0.5, 30.0)
        if (
            declaration.shape != (3, 28)
            or not np.isfinite(declaration).all()
            or not np.allclose(declaration, expected, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"{idv_init_path}: fault declaration changed")
        x = np.ascontiguousarray(decoded[:, 1:53], dtype=np.float64)
        x.setflags(write=False)
        identity = (run_id, idv)
        if identity in by_identity:
            raise RuntimeError("duplicate exact-selector identity")
        by_identity[identity] = NumericRun(
            name=f"FARM-v7-Mode2-IDV{idv}-Run{run_id}",
            x=x,
            source_idv=idv,
            simulation_run=run_id,
            rng_seed=seed,
        )
        seed_owners.setdefault(seed, []).append(identity)
    duplicate_owners = {
        seed: tuple(sorted(owners))
        for seed, owners in seed_owners.items()
        if len(owners) > 1
    }
    if (
        len(by_identity) != 400
        or len(seed_owners) != REQUIRED_UNIQUE_SEED_COUNT
        or duplicate_owners != dict(ALLOWED_DUPLICATE_SEED_OWNERS)
    ):
        raise ValueError("Mode2 seed-owner topology changed")
    blocks = tuple(
        NumericBlock(
            simulation_run=run_id,
            conditions=tuple(by_identity[(run_id, idv)] for idv in CONDITION_IDVS),
        )
        for run_id in range(1, 101)
    )
    topology = {
        "selected_run_count": 400,
        "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {str(idv): 100 for idv in CONDITION_IDVS},
        "unique_seed_count": len(seed_owners),
        "duplicate_seed_value_count": len(duplicate_owners),
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


def _authenticate_source() -> None:
    path = Path(OFFICIAL_MODE2_PATH)
    if not path.is_file() or path.name != "TEP_Mode2.h5":
        raise FileNotFoundError(path)
    if path.stat().st_size != SOURCE_SIZE_BYTES:
        raise ValueError("official Mode2 source size changed")
    if sha256_file(path) != MODE2_SHA256:
        raise ValueError("official Mode2 source SHA-256 changed")


def _load_numeric_blocks(
    access_plan: Sequence[Mapping[str, Any]],
    *,
    access_state: _AccessState | None = None,
) -> tuple[Mapping[str, Any], tuple[NumericBlock, ...]]:
    plan = _validate_access_plan(access_plan)
    _authenticate_source()
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("formal FARM v7 loading requires h5py") from error
    if access_state is not None:
        access_state.update(hdf5_open_started=True)
    with h5py.File(OFFICIAL_MODE2_PATH, "r") as handle:
        if access_state is not None:
            access_state.update(hdf5_open_completed=True)
        return _read_exact_blocks(
            handle,
            plan,
            h5py_module=h5py,
            access_state=access_state,
        )


def _validate_blocks(
    blocks: Sequence[NumericBlock], fold_map: Mapping[int, int]
) -> tuple[NumericBlock, ...]:
    if len(blocks) != 100:
        raise ValueError("FARM v7 requires exactly 100 balanced blocks")
    ordered = tuple(sorted(blocks, key=lambda item: int(item.simulation_run)))
    if tuple(item.simulation_run for item in ordered) != tuple(range(1, 101)):
        raise ValueError("blocks must be exactly Run1..100")
    identities: set[tuple[int, int]] = set()
    seed_folds: dict[int, set[int]] = {}
    for block in ordered:
        if tuple(run.source_idv for run in block.conditions) != CONDITION_IDVS:
            raise ValueError("each block must contain IDV1/3/9/15 in order")
        for run in block.conditions:
            identity = (run.source_idv, run.simulation_run)
            if run.simulation_run != block.simulation_run or identity in identities:
                raise ValueError("duplicate or inconsistent block identity")
            values = np.asarray(run.x)
            if (
                values.shape != (SEQUENCE_LENGTH, FEATURE_COUNT)
                or values.dtype != np.dtype(np.float64)
                or not np.isfinite(values).all()
            ):
                raise ValueError("run values must be finite float64 [500,52]")
            identities.add(identity)
            seed_folds.setdefault(run.rng_seed, set()).add(
                fold_map[run.simulation_run]
            )
    if identities != {
        (idv, run_id) for idv in CONDITION_IDVS for run_id in range(1, 101)
    }:
        raise ValueError("development identities are incomplete")
    if any(len(owner_folds) != 1 for owner_folds in seed_folds.values()):
        raise ValueError("a repeated seed crosses a frozen fold")
    return ordered


def _select_runs(
    blocks: Sequence[NumericBlock],
    run_ids: Sequence[int],
    idvs: Sequence[int],
) -> tuple[NumericRun, ...]:
    selected_runs = set(int(value) for value in run_ids)
    selected_idvs = tuple(int(value) for value in idvs)
    result = tuple(
        run
        for block in blocks
        if block.simulation_run in selected_runs
        for run in block.conditions
        if run.source_idv in selected_idvs
    )
    if len(result) != len(selected_runs) * len(selected_idvs):
        raise RuntimeError("fold run selection count changed")
    return result


def _values(runs: Sequence[NumericRun]) -> np.ndarray:
    result = np.stack([np.asarray(run.x, dtype=np.float64) for run in runs], axis=0)
    if result.shape != (len(runs), SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError("run tensor geometry changed")
    if result.dtype != np.dtype(np.float64) or not np.isfinite(result).all():
        raise ValueError("run tensor must be finite exact float64")
    return result


class _FoldNumericNoGo(RuntimeError):
    """A preregistered numeric gate failed after a valid healthy-only fit."""


def _monitor_api() -> Any:
    from . import monitor

    required = (
        "FarmReference",
        "apply_threshold",
        "calibrate_threshold",
        "load_farm_reference_npz",
        "save_farm_reference_npz",
        "score_runs_cpu",
        "score_runs_cuda",
    )
    if any(not hasattr(monitor, name) for name in required):
        raise RuntimeError("FARM monitor high-level API is incomplete")
    return monitor


def _evaluation_api() -> Any:
    from . import evaluation

    required = ("RunDecision", "development_gates", "fold_roles")
    if any(not hasattr(evaluation, name) for name in required):
        raise RuntimeError("FARM evaluation high-level API is incomplete")
    return evaluation


def _score_cuda(reference: Any, runs: Sequence[NumericRun], *, device: str) -> Any:
    monitor = _monitor_api()
    score = monitor.score_runs_cuda(
        reference,
        _values(runs),
        device=device,
        batch_size=64,
    )
    paths = np.asarray(score.paths)
    maxima = np.asarray(score.run_maxima)
    if (
        paths.shape != (len(runs), MONITOR_LENGTH)
        or paths.dtype != np.dtype(np.float64)
        or maxima.shape != (len(runs),)
        or maxima.dtype != np.dtype(np.float64)
        or not np.isfinite(paths).all()
        or not np.isfinite(maxima).all()
        or not np.array_equal(maxima, np.max(paths, axis=1))
    ):
        raise FloatingPointError("CUDA FARM score batch contract changed")
    return score


def _score_cpu(reference: Any, runs: Sequence[NumericRun]) -> Any:
    monitor = _monitor_api()
    score = monitor.score_runs_cpu(reference, _values(runs))
    paths = np.asarray(score.paths)
    maxima = np.asarray(score.run_maxima)
    if (
        paths.shape != (len(runs), MONITOR_LENGTH)
        or paths.dtype != np.dtype(np.float64)
        or maxima.shape != (len(runs),)
        or maxima.dtype != np.dtype(np.float64)
        or not np.isfinite(paths).all()
        or not np.isfinite(maxima).all()
        or not np.array_equal(maxima, np.max(paths, axis=1))
    ):
        raise FloatingPointError("CPU FARM score batch contract changed")
    return score


def _continuous_score_error(first: Any, second: Any) -> float:
    return max(
        float(np.max(np.abs(first.paths - second.paths))),
        float(np.max(np.abs(first.run_maxima - second.run_maxima))),
    )


def _decision_discrete_equal(first: Any, second: Any) -> bool:
    return bool(
        np.array_equal(first.alarms, second.alarms)
        and np.array_equal(first.alarm_endpoints, second.alarm_endpoints)
        and np.array_equal(first.delays, second.delays)
    )


def _audit_inner(
    reference: Any,
    inner_healthy: Sequence[NumericRun],
    *,
    device: str,
) -> tuple[Any, Mapping[str, Any]]:
    if len(inner_healthy) != INNER_HEALTHY_RUNS or any(
        run.source_idv != 1 for run in inner_healthy
    ):
        raise ValueError("inner threshold must use exactly 20 healthy runs")
    monitor = _monitor_api()
    first = _score_cuda(reference, inner_healthy, device=device)
    second = _score_cuda(reference, inner_healthy, device=device)
    continuous_error = _continuous_score_error(first, second)
    try:
        threshold = monitor.calibrate_threshold(
            first.run_maxima,
            expected_run_count=INNER_HEALTHY_RUNS,
            repeat_error=continuous_error,
            reference_count=reference.reference_count,
        )
    except ValueError as error:
        raise _FoldNumericNoGo(
            f"inner threshold numeric gate failed: {error}"
        ) from error
    first_decisions = monitor.apply_threshold(first, threshold)
    second_decisions = monitor.apply_threshold(second, threshold)
    discrete_equal = _decision_discrete_equal(first_decisions, second_decisions)
    repeat = {
        "schema": "farm_rankcusum_v7_authoritative_cuda_repeat",
        "scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        "device": device,
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": continuous_error,
        "discrete_equal": discrete_equal,
        "passes": bool(continuous_error <= CUDA_REPEAT_ATOL and discrete_equal),
        "inner_healthy_run_count": len(inner_healthy),
        "first_run_maxima": first.run_maxima.tolist(),
        "second_run_maxima": second.run_maxima.tolist(),
    }
    return threshold, repeat


def _save_reference_artifact(reference: Any, path: Path) -> Mapping[str, Any]:
    monitor = _monitor_api()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    monitor.save_farm_reference_npz(reference, path)
    restored = monitor.load_farm_reference_npz(path)
    if (
        restored.fit_run_count != reference.fit_run_count
        or not np.array_equal(restored.sorted_values, reference.sorted_values)
    ):
        raise RuntimeError("saved FARM reference failed exact roundtrip")
    return _file_record(path)


def _freeze_fold_artifacts(
    reference: Any,
    threshold: Any,
    fold_dir: Path,
    *,
    outer_fold: int,
    roles: Any,
    protocol_canonical_json_sha256: str,
) -> Mapping[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=False)
    reference_path = fold_dir / "healthy_reference.npz"
    threshold_path = fold_dir / "inner_threshold.json"
    reference_record = _save_reference_artifact(reference, reference_path)
    _atomic_json(threshold_path, threshold.to_document())
    threshold_record = _file_record(threshold_path)
    manifest = {
        "schema": "farm_rankcusum_v7_frozen_fold_artifacts",
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID,
        "outer_fold": outer_fold,
        "fold_roles": roles.to_document(),
        "fit_healthy_run_count": FIT_HEALTHY_RUNS,
        "fit_fault_run_count": 0,
        "threshold_healthy_run_count": INNER_HEALTHY_RUNS,
        "threshold_fault_run_count": 0,
        "reference_shape": list(reference.sorted_values.shape),
        "frozen_before_outer_scoring": True,
        "reference": reference_record,
        "threshold": threshold_record,
    }
    manifest_path = fold_dir / "freeze_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "reference": reference_record,
        "threshold": threshold_record,
        "freeze_manifest": _file_record(manifest_path),
    }


def _evaluation_rows(
    runs: Sequence[NumericRun], score: Any, decisions: Any
) -> tuple[Any, ...]:
    evaluation = _evaluation_api()
    if len(runs) != len(decisions.alarms):
        raise ValueError("outer decision count changed")
    if not np.array_equal(score.run_maxima, decisions.run_maxima):
        raise FloatingPointError("threshold decisions changed FARM run maxima")
    rows = []
    for index, run in enumerate(runs):
        alarm_endpoint = (
            int(decisions.alarm_endpoints[index])
            if bool(decisions.alarms[index])
            else None
        )
        rows.append(
            evaluation.RunDecision(
                source_idv=run.source_idv,
                simulation_run=run.simulation_run,
                run_maximum=float(score.run_maxima[index]),
                alarm_endpoint=alarm_endpoint,
            )
        )
    return tuple(rows)


def _fold_metrics(rows: Sequence[Any]) -> Mapping[str, Any]:
    healthy = tuple(row for row in rows if row.source_idv == 1)
    faults: dict[str, Any] = {}
    for fault in (3, 9, 15):
        selected = tuple(row for row in rows if row.source_idv == fault)
        delays = np.asarray([row.delay for row in selected], dtype=np.float64)
        detections = sum(row.alarm for row in selected)
        faults[str(fault)] = {
            "runs": len(selected),
            "detections": detections,
            "detection_rate": detections / len(selected),
            "median_delay_samples": float(np.median(delays)),
        }
    return {
        "healthy": {
            "runs": len(healthy),
            "false_alarms": sum(row.alarm for row in healthy),
        },
        "faults": faults,
    }


def _outer_replay(
    runs: Sequence[NumericRun], score: Any, decisions: Any
) -> list[dict[str, Any]]:
    return [
        {
            "identity": {
                "source_idv": run.source_idv,
                "simulation_run": run.simulation_run,
            },
            "score_path_endpoints_160_499": score.paths[index].tolist(),
            "run_maximum": float(score.run_maxima[index]),
            "alarm": bool(decisions.alarms[index]),
            "alarm_endpoint": (
                int(decisions.alarm_endpoints[index])
                if bool(decisions.alarms[index])
                else None
            ),
            "delay_samples": (
                None
                if run.source_idv == 1
                else int(decisions.delays[index])
            ),
        }
        for index, run in enumerate(runs)
    ]


def _save_outer_result(
    fold_dir: Path,
    *,
    outer_fold: int,
    rows: Sequence[Any],
    replay: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    path = fold_dir / "outer_result.json"
    _atomic_json(
        path,
        {
            "schema": "farm_rankcusum_v7_outer_fold_result",
            "outer_fold": outer_fold,
            "authoritative_device": "cuda",
            "scored_once_after_freeze": True,
            "metrics": _fold_metrics(rows),
            "decisions": [row.to_document() for row in rows],
            "replay": list(replay),
        },
    )
    return _file_record(path)


def _cpu_report(
    reference: Any,
    outer_runs: Sequence[NumericRun],
    authoritative_score: Any,
    threshold: Any,
    authoritative_decisions: Any,
) -> Mapping[str, Any]:
    monitor = _monitor_api()
    try:
        cpu_score = _score_cpu(reference, outer_runs)
        cpu_decisions = monitor.apply_threshold(cpu_score, threshold)
        return {
            "is_go_gate": False,
            "authoritative_device": "cuda",
            "comparison_device": "cpu",
            "maximum_absolute_error": _continuous_score_error(
                authoritative_score, cpu_score
            ),
            "discrete_equal": _decision_discrete_equal(
                authoritative_decisions, cpu_decisions
            ),
            "completed": True,
        }
    except Exception as error:  # report-only comparison must not become a GO gate
        return {
            "is_go_gate": False,
            "authoritative_device": "cuda",
            "comparison_device": "cpu",
            "completed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _execute_fold(
    blocks: Sequence[NumericBlock],
    *,
    outer_fold: int,
    artifact_root: Path,
    protocol_canonical_json_sha256: str,
    device: str,
) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    monitor = _monitor_api()
    evaluation = _evaluation_api()
    roles = evaluation.fold_roles(outer_fold)
    fit_healthy = _select_runs(blocks, roles.fit, (1,))
    inner_healthy = _select_runs(blocks, roles.inner_healthy, (1,))
    outer_runs = _select_runs(blocks, roles.outer, CONDITION_IDVS)
    if any(run.source_idv != 1 for run in (*fit_healthy, *inner_healthy)):
        raise RuntimeError("fault values entered healthy-only fit or threshold")
    reference = monitor.FarmReference.fit(
        _values(fit_healthy), expected_run_count=FIT_HEALTHY_RUNS
    )
    if reference.sorted_values.shape != (REFERENCE_CHANNEL_COUNT, 30_000):
        raise RuntimeError("fold healthy reference must have shape [46,30000]")
    threshold, repeat = _audit_inner(
        reference, inner_healthy, device=device
    )
    fold_dir = artifact_root / f"fold_{outer_fold}"
    frozen = _freeze_fold_artifacts(
        reference,
        threshold,
        fold_dir,
        outer_fold=outer_fold,
        roles=roles,
        protocol_canonical_json_sha256=protocol_canonical_json_sha256,
    )
    authoritative_score = _score_cuda(reference, outer_runs, device=device)
    authoritative_decisions = monitor.apply_threshold(
        authoritative_score, threshold
    )
    rows = _evaluation_rows(
        outer_runs, authoritative_score, authoritative_decisions
    )
    replay = _outer_replay(
        outer_runs, authoritative_score, authoritative_decisions
    )
    outer_record = _save_outer_result(
        fold_dir,
        outer_fold=outer_fold,
        rows=rows,
        replay=replay,
    )
    cpu = _cpu_report(
        reference,
        outer_runs,
        authoritative_score,
        threshold,
        authoritative_decisions,
    )
    numeric_valid = bool(repeat["passes"])
    document = {
        "outer_fold": outer_fold,
        "roles": roles.to_document(),
        "fit_healthy_run_count": len(fit_healthy),
        "fit_fault_run_count": 0,
        "inner_threshold_healthy_run_count": len(inner_healthy),
        "inner_threshold_fault_run_count": 0,
        "outer_run_count_per_class": 20,
        "threshold": threshold.to_document(),
        "authoritative_cuda_repeat": repeat,
        "cross_device_report_only": cpu,
        "numeric_valid": numeric_valid,
        "artifacts_frozen_before_outer": frozen,
        "outer_result_append_only": outer_record,
        "metrics": _fold_metrics(rows),
        "decisions": [row.to_document() for row in rows],
    }
    return document, rows


def _run_final_refit(
    blocks: Sequence[NumericBlock],
    *,
    artifact_root: Path,
    protocol_canonical_json_sha256: str,
) -> Mapping[str, Any]:
    monitor = _monitor_api()
    healthy = _select_runs(blocks, range(1, 101), (1,))
    if len(healthy) != FINAL_HEALTHY_RUNS or any(
        run.source_idv != 1 for run in healthy
    ):
        raise RuntimeError("final refit must use exactly 100 healthy runs")
    reference = monitor.FarmReference.fit(
        _values(healthy), expected_run_count=FINAL_HEALTHY_RUNS
    )
    if reference.sorted_values.shape != (REFERENCE_CHANNEL_COUNT, 50_000):
        raise RuntimeError("final healthy reference must have shape [46,50000]")
    final_dir = artifact_root / "final_refit"
    final_dir.mkdir(parents=True, exist_ok=False)
    reference_record = _save_reference_artifact(
        reference, final_dir / "healthy_reference.npz"
    )
    manifest = {
        "schema": "farm_rankcusum_v7_final_mode2_refit",
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID,
        "fit_healthy_run_count": FINAL_HEALTHY_RUNS,
        "fit_fault_run_count": 0,
        "reference_shape": [REFERENCE_CHANNEL_COUNT, 50_000],
        "created_only_after_development_go": True,
        "target_domain_threshold_not_fitted": True,
        "reference": reference_record,
    }
    manifest_path = final_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest": _file_record(manifest_path)}


def _validate_protocol_hash(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "protocol_canonical_json_sha256 must be lowercase hexadecimal SHA-256"
        )
    return value


def _finalize_access_state(state: _AccessState) -> Mapping[str, Any]:
    state.update(backend_completed=True)
    return {
        "final_state": _plain_json(state.value),
        "artifact": _file_record(state.path),
    }


def _persist_development_decision_before_final_refit(
    artifact_root: Path,
    *,
    protocol_canonical_json_sha256: str,
    decision: str,
    folds: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    aggregate: Mapping[str, Any] | None,
    authoritative_cuda_repeat: Mapping[str, Any],
    cross_device_report_only: Mapping[str, Any],
) -> Mapping[str, Any]:
    if decision not in {"DEVELOPMENT_GO", "DEVELOPMENT_NO_GO"}:
        raise ValueError("development decision must be GO or NO_GO")
    if len(folds) != FOLD_COUNT:
        raise ValueError("pre-refit decision must bind exactly five folds")
    document = {
        "schema": "farm_rankcusum_v7_development_decision_before_final_refit",
        "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "source_file_bytes_sha256": MODE2_SHA256,
        "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "development_decision_determined": True,
        "decision": decision,
        "selected_candidate_id": (
            CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None
        ),
        "folds": _plain_json(folds),
        "gates": _plain_json(gates),
        "aggregate": _plain_json(aggregate),
        "authoritative_cuda_repeat": _plain_json(authoritative_cuda_repeat),
        "cross_device_report_only": _plain_json(cross_device_report_only),
        "mode1_accessed": False,
        "mode1_opens": 0,
        "rieth_accessed": False,
        "rieth_opens": 0,
        "fault_fit_run_count": 0,
        "fault_threshold_run_count": 0,
        "final_refit_permitted": decision == "DEVELOPMENT_GO",
        "final_refit_started": False,
        "final_model_usable": False,
    }
    path = artifact_root / "development_decision_before_final_refit.json"
    _atomic_json(path, document)
    return _file_record(path)


def _numeric_no_go_result(
    *,
    artifact_root: Path,
    protocol_canonical_json_sha256: str,
    environment: Mapping[str, Any],
    topology: Mapping[str, Any],
    access_state: _AccessState,
    completed_folds: Sequence[Mapping[str, Any]],
    failed_fold: int,
    error: BaseException,
) -> Mapping[str, Any]:
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
    gates = {
        "all_pass": False,
        "checks": {"numeric_all_folds": False},
        "aggregate_metrics": None,
    }
    repeat = {
        "scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": None,
        "discrete_equal": False,
        "passes": False,
        "folds": [],
    }
    cpu_report = {"is_go_gate": False, "folds": []}
    decision_record = _persist_development_decision_before_final_refit(
        artifact_root,
        protocol_canonical_json_sha256=protocol_canonical_json_sha256,
        decision="DEVELOPMENT_NO_GO",
        folds=folds,
        gates=gates,
        aggregate=None,
        authoritative_cuda_repeat=repeat,
        cross_device_report_only=cpu_report,
    )
    state_record = _finalize_access_state(access_state)
    return {
        "schema": BACKEND_RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "source_file_bytes_sha256": MODE2_SHA256,
        "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "topology": _plain_json(topology),
        "environment": _plain_json(environment),
        "backend_access_state": state_record,
        "mode1_accessed": False,
        "mode1_opens": 0,
        "rieth_accessed": False,
        "rieth_opens": 0,
        "fault_fit_run_count": 0,
        "fault_threshold_run_count": 0,
        "decision": "DEVELOPMENT_NO_GO",
        "selected_candidate_id": None,
        "folds": folds,
        "gates": gates,
        "aggregate": None,
        "authoritative_cuda_repeat": repeat,
        "cross_device_report_only": cpu_report,
        "development_decision_before_final_refit": decision_record,
        "final_refit": None,
    }


def run_development(
    *,
    access_plan: Sequence[Mapping[str, Any]],
    fold_map: Mapping[int, int],
    artifact_root: str | os.PathLike[str],
    protocol_canonical_json_sha256: str,
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Execute the five frozen FARM development folds and GO-only refit."""

    validated_access = _validate_access_plan(access_plan)
    validated_folds = _validate_fold_map(fold_map)
    protocol_hash = _validate_protocol_hash(protocol_canonical_json_sha256)
    root = Path(artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("artifact_root must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    access_state = _AccessState(
        root / "backend_access_state.json",
        protocol_canonical_json_sha256=protocol_hash,
    )
    environment = _require_cuda(device)
    topology, raw_blocks = _load_numeric_blocks(
        validated_access, access_state=access_state
    )
    blocks = _validate_blocks(raw_blocks, validated_folds)
    access_state.update(training_or_fit_started=True)

    fold_documents: list[Mapping[str, Any]] = []
    all_rows: list[Any] = []
    for outer_fold in range(FOLD_COUNT):
        try:
            document, rows = _execute_fold(
                blocks,
                outer_fold=outer_fold,
                artifact_root=root,
                protocol_canonical_json_sha256=protocol_hash,
                device=device,
            )
        except (_FoldNumericNoGo, FloatingPointError, np.linalg.LinAlgError) as error:
            result = _numeric_no_go_result(
                artifact_root=root,
                protocol_canonical_json_sha256=protocol_hash,
                environment=environment,
                topology=topology,
                access_state=access_state,
                completed_folds=fold_documents,
                failed_fold=outer_fold,
                error=error,
            )
            _atomic_json(root / "development_result.json", result)
            return result
        fold_documents.append(document)
        all_rows.extend(rows)

    evaluation = _evaluation_api()
    gate_report = evaluation.development_gates(
        all_rows,
        numeric_folds_valid=[
            bool(document["numeric_valid"]) for document in fold_documents
        ],
    )
    gate_document = gate_report.to_document()
    decision = (
        "DEVELOPMENT_GO" if gate_report.all_pass else "DEVELOPMENT_NO_GO"
    )
    repeat_folds = [
        document["authoritative_cuda_repeat"] for document in fold_documents
    ]
    repeat = {
        "scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": max(
            float(item["maximum_absolute_error"]) for item in repeat_folds
        ),
        "discrete_equal": all(bool(item["discrete_equal"]) for item in repeat_folds),
        "passes": all(bool(item["passes"]) for item in repeat_folds),
        "folds": repeat_folds,
    }
    cpu_folds = [document["cross_device_report_only"] for document in fold_documents]
    completed_cpu = [item for item in cpu_folds if item.get("completed") is True]
    cpu_report = {
        "is_go_gate": False,
        "completed_fold_count": len(completed_cpu),
        "maximum_absolute_error": (
            max(float(item["maximum_absolute_error"]) for item in completed_cpu)
            if completed_cpu
            else None
        ),
        "discrete_equal": bool(
            len(completed_cpu) == FOLD_COUNT
            and all(bool(item["discrete_equal"]) for item in completed_cpu)
        ),
        "folds": cpu_folds,
    }
    decision_record = _persist_development_decision_before_final_refit(
        root,
        protocol_canonical_json_sha256=protocol_hash,
        decision=decision,
        folds=fold_documents,
        gates=gate_document,
        aggregate=gate_document["aggregate_metrics"],
        authoritative_cuda_repeat=repeat,
        cross_device_report_only=cpu_report,
    )
    final_refit = None
    if decision == "DEVELOPMENT_GO":
        final_refit = _plain_json(
            _run_final_refit(
                blocks,
                artifact_root=root,
                protocol_canonical_json_sha256=protocol_hash,
            )
        )
    state_record = _finalize_access_state(access_state)
    result = {
        "schema": BACKEND_RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": CANDIDATE_COUNT,
        "source_file_bytes_sha256": MODE2_SHA256,
        "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "topology": _plain_json(topology),
        "environment": _plain_json(environment),
        "backend_access_state": state_record,
        "mode1_accessed": False,
        "mode1_opens": 0,
        "rieth_accessed": False,
        "rieth_opens": 0,
        "fault_fit_run_count": 0,
        "fault_threshold_run_count": 0,
        "decision": decision,
        "selected_candidate_id": (
            CANDIDATE_ID if decision == "DEVELOPMENT_GO" else None
        ),
        "folds": fold_documents,
        "gates": gate_document,
        "aggregate": gate_document["aggregate_metrics"],
        "authoritative_cuda_repeat": repeat,
        "cross_device_report_only": cpu_report,
        "development_decision_before_final_refit": decision_record,
        "final_refit": final_refit,
    }
    _atomic_json(root / "development_result.json", result)
    return result


__all__ = [
    "ACCESS_PLAN_CANONICAL_JSON_SHA256",
    "BACKEND_RESULT_SCHEMA",
    "CANDIDATE_ID",
    "FOLD_MAP_CANONICAL_JSON_SHA256",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "cuda_preflight",
    "development_access_plan",
    "run_development",
    "validate_protocol",
]
