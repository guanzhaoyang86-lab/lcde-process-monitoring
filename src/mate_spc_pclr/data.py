"""Fail-closed data contract for the PCLR Extended TEP Mode 1 study.

This module is deliberately independent from the frozen external-validation
loader.  It exposes only the rows assigned to the successor PCLR study and
never materializes a complete ``processdata`` dataset.

The raw simulator initial-state row is excluded.  A normalized row ``k`` maps
to raw row ``k + 1`` and the 52 model inputs are raw columns ``1:53``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DATASET_DOI = "10.11583/DTU.13385936.v1"
MODE1_FILE_NAME = "TEP_Mode1.h5"

# Exact order in the official DTU/Figshare TEP_Mode1.h5 archive.  Repeated
# display labels and the lower-case ``feed`` at index 42 are intentional.
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
FEATURE_NAMES = PROCESSDATA_LABELS[1:53]

DEVELOPMENT = "development"
CALIBRATION = "calibration"
FINAL = "final"
PARTITIONS = (DEVELOPMENT, CALIBRATION, FINAL)

CONDITION_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
DEVELOPMENT_RUNS = tuple(range(201, 301))
CALIBRATION_RUNS = tuple(range(301, 401))
FINAL_RUNS = tuple(range(401, 501))

NORMALIZED_LENGTH = 2_000
SEQUENCE_LENGTH = 500
# Healthy IDV1 uses normalized [100, 600): the last included row is global
# index 599, so the first affected global row 600 is excluded by construction.
HEALTHY_WINDOW = (100, 600)
FAULT_WINDOW = (440, 940)
LOCAL_FAULT_START = 160
SAMPLE_PERIOD_HOURS = 0.05

BASELINE = (0, 128)
WARMUP = (128, 160)
MONITOR = (160, 500)
SEGMENTS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"baseline": BASELINE, "warmup": WARMUP, "monitor": MONITOR}
)


@dataclass(frozen=True)
class AuthorizedSelection:
    """A request proven to belong to the immutable PCLR allocation."""

    partition: str
    idv: int
    simulation_run: int
    reported_fault_id: int
    window: tuple[int, int]


@dataclass(frozen=True)
class LocatedRun:
    """The sole physical HDF5 location for one simulator run."""

    status: str
    idv: int
    simulation_run: int
    group_path: str
    processdata_path: str
    idv_init_path: str


@dataclass(frozen=True)
class PCLRRun:
    """One immutable 500-by-52 input sequence and its study identity."""

    name: str
    x: np.ndarray
    fault: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def simulation_run(self) -> int:
        return int(self.metadata["simulation_run"])

    @property
    def source_idv(self) -> int:
        return int(self.metadata["source_idv"])

    @property
    def group_key(self) -> int:
        return int(self.metadata["group_key"])

    def segment(self, name: str) -> np.ndarray:
        """Return a read-only view of baseline, warmup, or monitoring rows."""

        if name not in SEGMENTS:
            raise ValueError(f"unknown PCLR segment {name!r}")
        start, stop = SEGMENTS[name]
        return self.x[start:stop]


@dataclass(frozen=True)
class DevelopmentGroup:
    """The four IDV conditions coupled by a common simulation-run seed."""

    simulation_run: int
    conditions: tuple[PCLRRun, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulation_run, bool)
            or not isinstance(self.simulation_run, (int, np.integer))
            or int(self.simulation_run) not in DEVELOPMENT_RUNS
        ):
            raise ValueError("development group key must be a run in 201..300")
        ids = tuple(run.source_idv for run in self.conditions)
        if ids != CONDITION_IDVS:
            raise ValueError(
                "development group must contain IDV1/3/9/15 exactly once in that order"
            )
        if any(
            run.simulation_run != int(self.simulation_run)
            or run.group_key != int(self.simulation_run)
            or run.metadata.get("partition") != DEVELOPMENT
            for run in self.conditions
        ):
            raise ValueError(
                "all four development conditions must share the simulation_run group key"
            )

    def by_idv(self, idv: int) -> PCLRRun:
        """Select one condition while retaining explicit paired-run semantics."""

        if idv not in CONDITION_IDVS:
            raise ValueError("development condition IDV must be one of 1, 3, 9, 15")
        return self.conditions[CONDITION_IDVS.index(idv)]


@dataclass(frozen=True)
class TopologySummary:
    """Metadata-only validation result; no numeric run values are retained."""

    labels: tuple[str, ...]
    selected_run_count: int
    counts_by_partition: Mapping[str, int]
    counts_by_status: Mapping[str, int]


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("PCLR Extended TEP loading requires h5py>=3.10") from error
    return h5py


def _require_plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def authorize_request(
    *, partition: str, idv: int, simulation_run: int
) -> AuthorizedSelection:
    """Fail closed before a file is opened or any HDF5 value is decoded.

    Runs 1--200 remain permanently unavailable to PCLR.  Fault rows in the
    301--400 calibration block are likewise impossible to request.
    """

    idv_int = _require_plain_int(idv, name="IDV")
    run_int = _require_plain_int(simulation_run, name="simulation_run")
    if run_int <= 200:
        raise ValueError("simulation runs 1..200 are permanently embargoed for PCLR")
    if run_int > 500:
        raise ValueError("simulation_run must not exceed 500")
    if partition not in PARTITIONS:
        raise ValueError(f"unknown PCLR partition {partition!r}")
    if run_int in CALIBRATION_RUNS and idv_int in FAULT_IDVS:
        raise ValueError("fault runs 301..400 are forbidden in calibration")

    if partition == DEVELOPMENT:
        permitted = run_int in DEVELOPMENT_RUNS and idv_int in CONDITION_IDVS
    elif partition == CALIBRATION:
        permitted = run_int in CALIBRATION_RUNS and idv_int == 1
    else:
        permitted = run_int in FINAL_RUNS and idv_int in CONDITION_IDVS
    if not permitted:
        raise ValueError("requested partition/IDV/simulation_run is not authorized")

    fault_id = 0 if idv_int == 1 else idv_int
    window = HEALTHY_WINDOW if idv_int == 1 else FAULT_WINDOW
    return AuthorizedSelection(partition, idv_int, run_int, fault_id, window)


def iter_authorized_selections(
    partitions: Iterable[str] = PARTITIONS,
) -> tuple[AuthorizedSelection, ...]:
    """Enumerate the immutable allocation without touching an HDF5 file."""

    requested = tuple(partitions)
    if len(set(requested)) != len(requested) or any(p not in PARTITIONS for p in requested):
        raise ValueError("partitions must be unique members of the PCLR allocation")
    selections: list[AuthorizedSelection] = []
    if DEVELOPMENT in requested:
        selections.extend(
            authorize_request(partition=DEVELOPMENT, idv=idv, simulation_run=run)
            for run in DEVELOPMENT_RUNS
            for idv in CONDITION_IDVS
        )
    if CALIBRATION in requested:
        selections.extend(
            authorize_request(partition=CALIBRATION, idv=1, simulation_run=run)
            for run in CALIBRATION_RUNS
        )
    if FINAL in requested:
        selections.extend(
            authorize_request(partition=FINAL, idv=idv, simulation_run=run)
            for run in FINAL_RUNS
            for idv in CONDITION_IDVS
        )
    return tuple(selections)


def _base_path(status: str, idv: int, simulation_run: int) -> str:
    return (
        f"Mode1/SingleFault/{status}/IDV{idv}/"
        f"Mode1_IDVInfo_{idv}_100/Run{simulation_run}"
    )


def locate_unique_run(handle: Any, selection: AuthorizedSelection) -> LocatedRun:
    """Resolve an authorized run in exactly one completion-status branch."""

    # Re-authorize even dataclass instances constructed directly by callers.
    canonical = authorize_request(
        partition=selection.partition,
        idv=selection.idv,
        simulation_run=selection.simulation_run,
    )
    if selection != canonical:
        raise ValueError("AuthorizedSelection fields do not match the canonical allocation")

    h5py = _require_h5py()
    matches: list[LocatedRun] = []
    for status in ("SimulationCompleted", "SimulationStopped"):
        group_path = _base_path(status, selection.idv, selection.simulation_run)
        if group_path not in handle:
            continue
        group = handle[group_path]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{group_path} exists but is not an HDF5 group")
        processdata_path = f"{group_path}/processdata"
        idv_init_path = f"{group_path}/idv_init"
        if processdata_path not in handle:
            raise ValueError(f"{group_path} has no exact processdata dataset")
        if idv_init_path not in handle:
            raise ValueError(f"{group_path} has no exact idv_init dataset")
        processdata = handle[processdata_path]
        idv_init = handle[idv_init_path]
        if not isinstance(processdata, h5py.Dataset):
            raise ValueError(f"{processdata_path} is not an HDF5 dataset")
        if not isinstance(idv_init, h5py.Dataset):
            raise ValueError(f"{idv_init_path} is not an HDF5 dataset")
        matches.append(
            LocatedRun(
                status=status,
                idv=selection.idv,
                simulation_run=selection.simulation_run,
                group_path=group_path,
                processdata_path=processdata_path,
                idv_init_path=idv_init_path,
            )
        )
    if len(matches) != 1:
        raise ValueError(
            f"IDV{selection.idv} Run{selection.simulation_run} must occur in exactly "
            f"one completion status; found {len(matches)}"
        )
    return matches[0]


def validate_processdata_labels(handle: Any) -> tuple[str, ...]:
    """Decode and exactly match the official 54 string labels."""

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
            "Processdata_Labels do not match the official sequence: "
            f"index {index} expected {expected!r}, got {observed!r}"
        )
    return labels


def _validate_dataset_metadata(handle: Any, located: LocatedRun) -> None:
    processdata = handle[located.processdata_path]
    idv_init = handle[located.idv_init_path]
    if processdata.shape != (NORMALIZED_LENGTH + 1, 54):
        raise ValueError(
            f"{located.processdata_path}: expected exact shape (2001, 54), "
            f"got {processdata.shape}"
        )
    if np.dtype(processdata.dtype) != np.dtype(np.float64):
        raise ValueError(f"{located.processdata_path}: processdata must be exact float64")
    if idv_init.shape != (3, 28):
        raise ValueError(
            f"{located.idv_init_path}: expected exact shape (3, 28), got {idv_init.shape}"
        )
    if np.dtype(idv_init.dtype) != np.dtype(np.float64):
        raise ValueError(f"{located.idv_init_path}: idv_init must be exact float64")


def _validate_idv_init_values(handle: Any, located: LocatedRun) -> None:
    values = np.asarray(handle[located.idv_init_path][...])
    if values.shape != (3, 28) or values.dtype != np.dtype(np.float64):
        raise ValueError(f"{located.idv_init_path}: decoded idv_init contract changed")
    if not np.isfinite(values).all():
        raise ValueError(f"{located.idv_init_path}: idv_init must be finite")
    expected = np.zeros((3, 28), dtype=np.float64)
    expected[:, located.idv - 1] = (0.0, 1.0, 30.0)
    if not np.array_equal(values, expected):
        raise ValueError(
            f"{located.idv_init_path}: must declare only IDV{located.idv} as [0, 1, 30]"
        )


def _validate_time_values(times: np.ndarray, *, path: str) -> tuple[float, float]:
    if times.shape != (SEQUENCE_LENGTH,) or not np.isfinite(times).all():
        raise ValueError(f"{path}: authorized time slice must be 500 finite values")
    elapsed = times - times[0]
    expected = np.arange(SEQUENCE_LENGTH, dtype=np.float64) * SAMPLE_PERIOD_HOURS
    if not np.allclose(elapsed, expected, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path}: time must advance exactly every three minutes")
    return float(times[0]), float(times[-1])


def _materialize_run(
    handle: Any, *, partition: str, idv: int, simulation_run: int
) -> PCLRRun:
    """Read one authorized window, never an entire processdata dataset."""

    # This is intentionally the first operation: unauthorized calls cannot
    # inspect topology, decode labels, or trigger any dataset access.
    selection = authorize_request(
        partition=partition, idv=idv, simulation_run=simulation_run
    )
    located = locate_unique_run(handle, selection)
    _validate_dataset_metadata(handle, located)
    _validate_idv_init_values(handle, located)

    start, stop = selection.window
    raw_start, raw_stop = start + 1, stop + 1
    dataset = handle[located.processdata_path]
    values = np.array(
        dataset[raw_start:raw_stop, 1:53], dtype=np.float64, copy=True, order="C"
    )
    times = np.asarray(dataset[raw_start:raw_stop, 0], dtype=np.float64)
    if values.shape != (SEQUENCE_LENGTH, len(FEATURE_NAMES)):
        raise ValueError(f"{located.processdata_path}: authorized window shape changed")
    if values.dtype != np.dtype(np.float64) or not np.isfinite(values).all():
        raise ValueError(
            f"{located.processdata_path}: authorized values must be finite float64"
        )
    time_first, time_last = _validate_time_values(times, path=located.processdata_path)

    values.setflags(write=False)
    fault = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    if selection.reported_fault_id:
        fault[LOCAL_FAULT_START:] = True
    fault.setflags(write=False)
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "dataset": "Extended_TEP_Mode1",
            "dataset_doi": DATASET_DOI,
            "study": "PCLR_successor",
            "partition": selection.partition,
            "fault_id": selection.reported_fault_id,
            "source_idv": selection.idv,
            "simulation_run": selection.simulation_run,
            "group_key": selection.simulation_run,
            "group_key_name": "simulation_run",
            "source_status": located.status,
            "source_group_path": located.group_path,
            "processdata_path": located.processdata_path,
            "raw_row_zero_dropped": True,
            "raw_column_half_open_projection": (1, 53),
            "normalized_window_half_open": selection.window,
            "sample_period_hours": SAMPLE_PERIOD_HOURS,
            "source_time_hours_first_last": (time_first, time_last),
            "fault_start_zero_based": (
                LOCAL_FAULT_START if selection.reported_fault_id else None
            ),
            "pseudo_onset_zero_based": LOCAL_FAULT_START,
            "segments_half_open": SEGMENTS,
        }
    )
    name = (
        f"PCLR-ExtendedTEP-Mode1-{selection.partition}-fault-"
        f"{selection.reported_fault_id:02d}-IDV-{selection.idv:02d}-"
        f"run-{selection.simulation_run:03d}"
    )
    return PCLRRun(name, values, fault, FEATURE_NAMES, metadata)


def _validate_development_groups(runs: Sequence[PCLRRun]) -> tuple[DevelopmentGroup, ...]:
    expected_count = len(DEVELOPMENT_RUNS) * len(CONDITION_IDVS)
    if len(runs) != expected_count:
        raise RuntimeError(f"development allocation must contain {expected_count} runs")
    by_run: dict[int, list[PCLRRun]] = {run: [] for run in DEVELOPMENT_RUNS}
    for item in runs:
        if item.simulation_run not in by_run:
            raise RuntimeError("development result contains an out-of-allocation group key")
        by_run[item.simulation_run].append(item)
    if any(
        len(by_run[run]) != len(CONDITION_IDVS)
        or {item.source_idv for item in by_run[run]} != set(CONDITION_IDVS)
        for run in DEVELOPMENT_RUNS
    ):
        raise RuntimeError("development groups must each contain exactly four conditions")
    groups = tuple(
        DevelopmentGroup(
            simulation_run=run,
            conditions=tuple(
                next(item for item in by_run[run] if item.source_idv == idv)
                for idv in CONDITION_IDVS
            ),
        )
        for run in DEVELOPMENT_RUNS
    )
    return groups


def validate_archive_topology(handle: Any) -> TopologySummary:
    """Validate all 900 allocated paths without reading numeric run values.

    Label strings are archive metadata and are decoded.  For ``processdata``
    and ``idv_init`` this function accesses only object identity, shape, and
    dtype; their values are intentionally untouched.
    """

    labels = validate_processdata_labels(handle)
    selections = iter_authorized_selections()
    partition_counts = {partition: 0 for partition in PARTITIONS}
    status_counts = {"SimulationCompleted": 0, "SimulationStopped": 0}
    development_ids: dict[int, set[int]] = {run: set() for run in DEVELOPMENT_RUNS}
    for selection in selections:
        located = locate_unique_run(handle, selection)
        _validate_dataset_metadata(handle, located)
        partition_counts[selection.partition] += 1
        status_counts[located.status] += 1
        if selection.partition == DEVELOPMENT:
            development_ids[selection.simulation_run].add(selection.idv)
    if any(ids != set(CONDITION_IDVS) for ids in development_ids.values()):
        raise RuntimeError("development topology lost four-condition run grouping")
    return TopologySummary(
        labels=labels,
        selected_run_count=len(selections),
        counts_by_partition=MappingProxyType(partition_counts),
        counts_by_status=MappingProxyType(status_counts),
    )


class PCLRMode1Archive:
    """Single-file archive exposing only development/calibration/final PCLR rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.name != MODE1_FILE_NAME:
            raise ValueError(f"PCLR requires the single official file {MODE1_FILE_NAME}")

    def _open(self) -> AbstractContextManager[Any]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        return _require_h5py().File(self.path, "r")

    def validate_topology(self) -> TopologySummary:
        with self._open() as handle:
            return validate_archive_topology(handle)

    def load_run(self, *, partition: str, idv: int, simulation_run: int) -> PCLRRun:
        # Authorize before _open so an embargoed request cannot even open HDF5.
        selection = authorize_request(
            partition=partition, idv=idv, simulation_run=simulation_run
        )
        with self._open() as handle:
            validate_processdata_labels(handle)
            return _materialize_run(
                handle,
                partition=selection.partition,
                idv=selection.idv,
                simulation_run=selection.simulation_run,
            )

    def load_development_groups(self) -> tuple[DevelopmentGroup, ...]:
        selections = iter_authorized_selections((DEVELOPMENT,))
        with self._open() as handle:
            validate_processdata_labels(handle)
            runs = tuple(
                _materialize_run(
                    handle,
                    partition=item.partition,
                    idv=item.idv,
                    simulation_run=item.simulation_run,
                )
                for item in selections
            )
        return _validate_development_groups(runs)

    def load_calibration_runs(self) -> tuple[PCLRRun, ...]:
        selections = iter_authorized_selections((CALIBRATION,))
        with self._open() as handle:
            validate_processdata_labels(handle)
            runs = tuple(
                _materialize_run(
                    handle,
                    partition=item.partition,
                    idv=item.idv,
                    simulation_run=item.simulation_run,
                )
                for item in selections
            )
        if len(runs) != 100 or any(run.source_idv != 1 for run in runs):
            raise RuntimeError("calibration must contain only 100 healthy IDV1 runs")
        return runs

    def load_final_runs(self) -> tuple[tuple[PCLRRun, ...], tuple[PCLRRun, ...]]:
        selections = iter_authorized_selections((FINAL,))
        with self._open() as handle:
            validate_processdata_labels(handle)
            runs = tuple(
                _materialize_run(
                    handle,
                    partition=item.partition,
                    idv=item.idv,
                    simulation_run=item.simulation_run,
                )
                for item in selections
            )
        healthy = tuple(run for run in runs if run.source_idv == 1)
        faults = tuple(run for run in runs if run.source_idv in FAULT_IDVS)
        if len(healthy) != 100 or len(faults) != 300:
            raise RuntimeError("final allocation must contain 100 healthy and 300 fault runs")
        return healthy, faults
