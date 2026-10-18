"""Strict row-level reader for the Extended TEP Mode 1 HDF5 archive.

Raw row zero is the simulator initial state. The canonical timeline is
``processdata[1:, 1:53]``: 2,000 samples and the original 41 XMEAS plus the
first 11 XMV channels. Only frozen external-audit selections are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .rieth_training import ExternalRiethRun, FEATURE_COLUMNS


DATASET_DOI = "10.11583/DTU.13385936.v1"
MODE1_FILE_ID = 26_003_087
MODE1_FILE_NAME = "TEP_Mode1.h5"
MODE1_SIZE_BYTES = 23_924_710_848
MODE1_MD5 = "c79ecc6d12a5ccb49caffcd8f531feae"
README_FILE_ID = 26_775_368
README_MD5 = "963359b135036c0b19951929798a8b46"

# Exact order stored in the official DTU/Figshare ``TEP_Mode1.h5`` archive.
# Repeated display names are intentional: columns 1..41 are measurements and
# columns 42..53 are manipulated variables, so several feed labels recur.
PROCESSDATA_LABELS = (
    "Time",
    "A Feed",
    "D Feed",
    "E Feed",
    "A and C Feed",
    "Recycle Flow",
    "Reactor Feed Rate",
    "Reactor Pressure",
    "Reactor Level",
    "Reactor Temperature",
    "Purge Rate",
    "Product Sep Temp",
    "Product Sep Level",
    "Product Sep Pressure",
    "Product Sep Underflow",
    "Stripper Level",
    "Stripper Pressure",
    "Stripper Underflow",
    "Stripper Temp",
    "Stripper Steam Flow",
    "Compressor Work",
    "Reactor Coolant Temp",
    "Separator Coolant Temp",
    "Component A to Reactor",
    "Component B to Reactor",
    "Component C to Reactor",
    "Component D to Reactor",
    "Component E to Reactor",
    "Component F to Reactor",
    "Component A in Purge",
    "Component B in Purge",
    "Component C in Purge",
    "Component D in Purge",
    "Component E in Purge",
    "Component F in Purge",
    "Component G in Purge",
    "Component H in Purge",
    "Component D in Product",
    "Component E in Product",
    "Component F in Product",
    "Component G in Product",
    "Component H in Product",
    "D feed",
    "E Feed",
    "A Feed",
    "A and C Feed",
    "Recycle",
    "Purge",
    "Separator",
    "Stripper",
    "Steam",
    "Reactor Coolant",
    "Condenser Coolant",
    "Agitator",
)

CALIBRATION_RUNS = tuple(range(1, 101))
AUDIT_RUNS = tuple(range(101, 201))
AUDIT_FAULT_IDS = (3, 9, 15)
NORMALIZED_LENGTH = 2_000
SEQUENCE_LENGTH = 500
CALIBRATION_WINDOW = (0, 500)
FAULT_WINDOW = (440, 940)
LOCAL_FAULT_START = 160


@dataclass(frozen=True)
class LocatedRun:
    """The unique physical HDF5 location selected for one simulation run."""

    status: str
    idv: int
    group_path: str
    processdata_path: str
    idv_init_path: str


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Extended TEP loading requires h5py; install h5py>=3.10"
        ) from error
    return h5py


def _base_path(status: str, idv: int, simulation_run: int) -> str:
    return (
        f"Mode1/SingleFault/{status}/IDV{idv}/"
        f"Mode1_IDVInfo_{idv}_100/Run{simulation_run}"
    )


def locate_unique_run(handle: Any, *, idv: int, simulation_run: int) -> LocatedRun:
    """Find a run in exactly one of SimulationCompleted/SimulationStopped."""

    if isinstance(idv, bool) or not isinstance(idv, (int, np.integer)) or not 1 <= idv <= 28:
        raise ValueError("IDV must be an integer in 1..28")
    if (
        isinstance(simulation_run, bool)
        or not isinstance(simulation_run, (int, np.integer))
        or not 1 <= simulation_run <= 500
    ):
        raise ValueError("simulation run must be an integer in 1..500")

    h5py = _require_h5py()
    matches: list[LocatedRun] = []
    for status in ("SimulationCompleted", "SimulationStopped"):
        group_path = _base_path(status, int(idv), int(simulation_run))
        if group_path not in handle:
            continue
        group = handle[group_path]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{group_path} exists but is not an HDF5 group")
        processdata_path = f"{group_path}/processdata"
        if processdata_path not in handle:
            raise ValueError(f"{group_path} has no exact processdata dataset")
        dataset = handle[processdata_path]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{processdata_path} is not an HDF5 dataset")
        idv_init_path = f"{group_path}/idv_init"
        if idv_init_path not in handle:
            raise ValueError(f"{group_path} has no exact idv_init dataset")
        idv_init = handle[idv_init_path]
        if not isinstance(idv_init, h5py.Dataset):
            raise ValueError(f"{idv_init_path} is not an HDF5 dataset")
        matches.append(
            LocatedRun(status, int(idv), group_path, processdata_path, idv_init_path)
        )
    if len(matches) != 1:
        raise ValueError(
            f"IDV{idv} Run{simulation_run} must occur in exactly one completion status; "
            f"found {len(matches)}"
        )
    return matches[0]


def _validate_idv_init(handle: Any, located: LocatedRun) -> None:
    """Authenticate a run's fault declaration before reading processdata values."""

    h5py = _require_h5py()
    dataset = handle[located.idv_init_path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{located.idv_init_path} is not an HDF5 dataset")
    if dataset.shape != (3, 28):
        raise ValueError(
            f"{located.idv_init_path}: expected exact shape (3, 28), got {dataset.shape}"
        )
    if np.dtype(dataset.dtype) != np.dtype(np.float64):
        raise ValueError(f"{located.idv_init_path}: idv_init must have exact float64 dtype")
    values = np.asarray(dataset[...])
    if values.shape != (3, 28) or not np.isfinite(values).all():
        raise ValueError(f"{located.idv_init_path}: idv_init must be finite")
    expected = np.zeros((3, 28), dtype=np.float64)
    expected[:, located.idv - 1] = np.asarray([0.0, 1.0, 30.0], dtype=np.float64)
    if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{located.idv_init_path}: must declare only IDV{located.idv} as [0, 1, 30]"
        )


def load_normalized_processdata(handle: Any, located: LocatedRun) -> np.ndarray:
    """Load and validate the frozen 2,000 by 52 canonical projection."""

    _validate_idv_init(handle, located)
    dataset = handle[located.processdata_path]
    if dataset.ndim != 2:
        raise ValueError(f"{located.processdata_path}: processdata must be two-dimensional")
    if dataset.shape[0] != NORMALIZED_LENGTH + 1:
        raise ValueError(
            f"{located.processdata_path}: expected 2001 raw rows, got {dataset.shape[0]}"
        )
    if dataset.shape[1] != 54:
        raise ValueError(
            f"{located.processdata_path}: expected exactly 54 raw columns, got {dataset.shape[1]}"
        )
    if np.dtype(dataset.dtype) != np.dtype(np.float64):
        raise ValueError(
            f"{located.processdata_path}: processdata must have exact float64 dtype"
        )
    values = np.asarray(dataset[1:, 1:53], dtype=np.float64)
    if values.shape != (NORMALIZED_LENGTH, len(FEATURE_COLUMNS)):
        raise ValueError(f"{located.processdata_path}: canonical projection shape changed")
    if not np.isfinite(values).all():
        raise ValueError(f"{located.processdata_path}: canonical projection is non-finite")
    values = np.ascontiguousarray(values)
    values.setflags(write=False)
    return values


def validate_processdata_labels(handle: Any) -> tuple[str, ...]:
    """Authenticate the 54-column order before any selected numeric read."""

    h5py = _require_h5py()
    if "Processdata_Labels" not in handle:
        raise ValueError("HDF5 root has no exact Processdata_Labels dataset")
    dataset = handle["Processdata_Labels"]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError("Processdata_Labels is not an HDF5 dataset")
    raw = np.asarray(dataset[...]).reshape(-1)
    if raw.size != 54:
        raise ValueError(f"Processdata_Labels must contain exactly 54 labels, got {raw.size}")
    labels = tuple(
        value.decode("utf-8", errors="strict")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in raw.tolist()
    )
    if labels != PROCESSDATA_LABELS:
        mismatch = next(
            (index, expected, observed)
            for index, (expected, observed) in enumerate(
                zip(PROCESSDATA_LABELS, labels, strict=True)
            )
            if expected != observed
        )
        index, expected, observed = mismatch
        raise ValueError(
            "Processdata_Labels do not match the frozen official sequence: "
            f"index {index} expected {expected!r}, got {observed!r}"
        )
    return labels


def _validate_authorized_time_mapping(dataset: Any, start: int, stop: int) -> tuple[float, float]:
    indices = [start + 1, start + 2, stop]
    times = np.asarray(dataset[indices, 0], dtype=np.float64)
    if times.shape != (3,) or not np.isfinite(times).all():
        raise ValueError("authorized processdata time anchors are invalid")
    expected_period_hours = 0.05
    if not np.isclose(times[1] - times[0], expected_period_hours, rtol=0.0, atol=1e-10):
        raise ValueError("processdata time step is not three minutes")
    expected_span = (stop - start - 1) * expected_period_hours
    if not np.isclose(times[2] - times[0], expected_span, rtol=0.0, atol=1e-8):
        raise ValueError("processdata time anchors do not match normalized row mapping")
    return float(times[0]), float(times[2])


def _materialize_run(
    handle: Any,
    *,
    idv: int,
    simulation_run: int,
    window: tuple[int, int],
    partition: str,
    reported_fault_id: int,
) -> ExternalRiethRun:
    authorized = {
        "external_calibration_IDV1_prefault": (
            idv == 1
            and reported_fault_id == 0
            and window == CALIBRATION_WINDOW
            and simulation_run in CALIBRATION_RUNS
        ),
        "external_audit_healthy_IDV1_prefault": (
            idv == 1
            and reported_fault_id == 0
            and window == CALIBRATION_WINDOW
            and simulation_run in AUDIT_RUNS
        ),
        "external_audit_fault_window": (
            idv in AUDIT_FAULT_IDS
            and reported_fault_id == idv
            and window == FAULT_WINDOW
            and simulation_run in AUDIT_RUNS
        ),
    }
    if authorized.get(partition) is not True:
        raise ValueError("requested IDV/run/window/fault/partition combination is not authorized")
    located = locate_unique_run(handle, idv=idv, simulation_run=simulation_run)
    start, stop = window
    if stop - start != SEQUENCE_LENGTH or not 0 <= start < stop <= NORMALIZED_LENGTH:
        raise ValueError("frozen Extended TEP window is invalid")
    _validate_idv_init(handle, located)
    dataset = handle[located.processdata_path]
    if dataset.shape != (NORMALIZED_LENGTH + 1, 54):
        raise ValueError(f"{located.processdata_path}: expected exact shape (2001, 54)")
    if np.dtype(dataset.dtype) != np.dtype(np.float64):
        raise ValueError(
            f"{located.processdata_path}: processdata must have exact float64 dtype"
        )
    values = np.array(
        dataset[start + 1 : stop + 1, 1:53], dtype=np.float64, copy=True, order="C"
    )
    if values.shape != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
        raise ValueError(f"{located.processdata_path}: authorized window shape changed")
    if not np.isfinite(values).all():
        raise ValueError(f"{located.processdata_path}: authorized window is non-finite")
    time_start, time_stop = _validate_authorized_time_mapping(dataset, start, stop)
    values.setflags(write=False)
    fault = np.zeros(SEQUENCE_LENGTH, dtype=bool)
    if reported_fault_id:
        fault[LOCAL_FAULT_START:] = True
    fault.setflags(write=False)
    name = (
        f"ExtendedTEP-Mode1-{partition}-fault-{reported_fault_id:02d}-"
        f"source-IDV-{idv:02d}-run-{simulation_run:03d}"
    )
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "dataset": "Extended_TEP_Mode1",
            "dataset_doi": DATASET_DOI,
            "partition": partition,
            "fault_id": int(reported_fault_id),
            "source_idv": int(idv),
            "simulation_run": int(simulation_run),
            "source_status": located.status,
            "source_group_path": located.group_path,
            "processdata_path": located.processdata_path,
            "raw_row_zero_dropped": True,
            "raw_column_half_open_projection": [1, 53],
            "normalized_source_length": NORMALIZED_LENGTH,
            "normalized_window_half_open": [start, stop],
            "source_time_hours_first_last": [time_start, time_stop],
            "sample_period_hours": 0.05,
            "fault_start_zero_based": LOCAL_FAULT_START if reported_fault_id else None,
        }
    )
    return ExternalRiethRun(
        name=name,
        x=values,
        fault=fault,
        feature_names=FEATURE_COLUMNS,
        metadata=metadata,
    )


class ExtendedTEPMode1Archive:
    """Phase-scoped loader whose public methods expose only frozen selections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _open(self) -> Any:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        return _require_h5py().File(self.path, "r")

    def load_calibration_runs(self) -> tuple[ExternalRiethRun, ...]:
        with self._open() as handle:
            validate_processdata_labels(handle)
            runs = tuple(
                _materialize_run(
                    handle,
                    idv=1,
                    simulation_run=run,
                    window=CALIBRATION_WINDOW,
                    partition="external_calibration_IDV1_prefault",
                    reported_fault_id=0,
                )
                for run in CALIBRATION_RUNS
            )
        if len(runs) != 100 or len({run.name for run in runs}) != 100:
            raise RuntimeError("Extended TEP calibration identity set changed")
        return runs

    def load_audit_runs(
        self,
    ) -> tuple[tuple[ExternalRiethRun, ...], tuple[ExternalRiethRun, ...]]:
        with self._open() as handle:
            validate_processdata_labels(handle)
            healthy = tuple(
                _materialize_run(
                    handle,
                    idv=1,
                    simulation_run=run,
                    window=CALIBRATION_WINDOW,
                    partition="external_audit_healthy_IDV1_prefault",
                    reported_fault_id=0,
                )
                for run in AUDIT_RUNS
            )
            faults = tuple(
                _materialize_run(
                    handle,
                    idv=fault_id,
                    simulation_run=run,
                    window=FAULT_WINDOW,
                    partition="external_audit_fault_window",
                    reported_fault_id=fault_id,
                )
                for fault_id in AUDIT_FAULT_IDS
                for run in AUDIT_RUNS
            )
        if len(healthy) != 100 or len(faults) != 300:
            raise RuntimeError("Extended TEP audit identity set changed")
        if len({run.name for run in healthy + faults}) != 400:
            raise RuntimeError("Extended TEP audit contains duplicate run identities")
        return healthy, faults
