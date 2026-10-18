"""Fail-closed data contract for the lower-magnitude PCLR Mode 1 study.

The immutable allocation is identified by both fault magnitude and simulation
run number:

* development: magnitude 50, IDV1/3/9/15, Run1..100;
* calibration: magnitude 25, IDV1 pre-fault rows only, Run1..100;
* final: magnitude 75, IDV1/3/9/15, Run1..100.

Magnitude-25 fault branches are quarantined.  The magnitude-100 target
branches Run1..200 were consumed by earlier work and are never exposed here.
Repeated run numbers across conditions are balanced-stratification block
labels only.  Every allocated HDF5 group must carry its own globally unique
positive integer ``seed`` attribute.

For lower-magnitude branches ``processdata`` has exactly 2,000 rows.  A
normalized row ``k`` is raw row ``k`` and the 52 inputs are columns ``1:53``.
Topology validation reads labels, object metadata, and the run-group seed
attribute, but never decodes ``processdata`` or ``idv_init`` values.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import FEATURE_NAMES, PROCESSDATA_LABELS


DATASET_DOI = "10.11583/DTU.13385936.v1"
MODE1_FILE_NAME = "TEP_Mode1.h5"

DEVELOPMENT = "development"
CALIBRATION = "calibration"
FINAL = "final"
PARTITIONS = (DEVELOPMENT, CALIBRATION, FINAL)

CONDITION_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
DEVELOPMENT_RUNS = tuple(range(1, 101))
CALIBRATION_RUNS = tuple(range(1, 101))
FINAL_RUNS = tuple(range(1, 101))
PARTITION_MAGNITUDES: Mapping[str, int] = MappingProxyType(
    {DEVELOPMENT: 50, CALIBRATION: 25, FINAL: 75}
)

QUARANTINED_MAGNITUDE_PERCENT = 25
QUARANTINED_FAULT_IDVS = FAULT_IDVS
QUARANTINED_RUNS = tuple(range(1, 101))
CONSUMED_MAGNITUDE_PERCENT = 100
CONSUMED_TARGET_RUNS = tuple(range(1, 201))

NORMALIZED_LENGTH = 2_000
SEQUENCE_LENGTH = 500
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
    """A request proven to belong to the immutable lower-magnitude allocation."""

    partition: str
    idv: int
    simulation_run: int
    fault_magnitude_percent: int
    reported_fault_id: int
    window: tuple[int, int]


@dataclass(frozen=True)
class LocatedRun:
    """The sole completed HDF5 location and validated seed for one run."""

    status: str
    idv: int
    simulation_run: int
    fault_magnitude_percent: int
    rng_seed: int
    group_path: str
    processdata_path: str
    idv_init_path: str


@dataclass(frozen=True)
class PCLRRun:
    """One immutable 500-by-52 sequence and its study identity."""

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

    @property
    def fault_magnitude_percent(self) -> int:
        return int(self.metadata["fault_magnitude_percent"])

    @property
    def rng_seed(self) -> int:
        return int(self.metadata["rng_seed"])

    def segment(self, name: str) -> np.ndarray:
        """Return a read-only view of baseline, warmup, or monitoring rows."""

        if name not in SEGMENTS:
            raise ValueError(f"unknown PCLR segment {name!r}")
        start, stop = SEGMENTS[name]
        return self.x[start:stop]


@dataclass(frozen=True)
class DevelopmentGroup:
    """Four conditions sharing one balanced-stratification block label.

    ``simulation_run`` aligns class counts for grouped cross-validation.  The
    four conditions have distinct ``rng_seed`` values and are not a paired
    stochastic sample.
    """

    simulation_run: int
    conditions: tuple[PCLRRun, ...]

    def __post_init__(self) -> None:
        block = _require_plain_int(self.simulation_run, name="development block")
        if block not in DEVELOPMENT_RUNS:
            raise ValueError("development stratification block must be a run in 1..100")
        ids = tuple(run.source_idv for run in self.conditions)
        if ids != CONDITION_IDVS:
            raise ValueError(
                "development block must contain IDV1/3/9/15 exactly once in that order"
            )
        if any(
            run.simulation_run != block
            or run.group_key != block
            or run.metadata.get("partition") != DEVELOPMENT
            or run.fault_magnitude_percent != PARTITION_MAGNITUDES[DEVELOPMENT]
            for run in self.conditions
        ):
            raise ValueError(
                "all four conditions must match the balanced simulation_run block"
            )
        seeds = tuple(run.rng_seed for run in self.conditions)
        if len(set(seeds)) != len(seeds):
            raise ValueError("conditions in a development block must have distinct rng_seed values")

    @property
    def stratified_block(self) -> int:
        return int(self.simulation_run)

    def by_idv(self, idv: int) -> PCLRRun:
        if idv not in CONDITION_IDVS:
            raise ValueError("development condition IDV must be one of 1, 3, 9, 15")
        return self.conditions[CONDITION_IDVS.index(idv)]


# Both names refer to the same backend-compatible value object.  "Block" is
# the scientifically precise term; "Group" preserves the existing API shape.
DevelopmentBlock = DevelopmentGroup


@dataclass(frozen=True)
class TopologySummary:
    """Metadata-only validation result; no run dataset values are retained."""

    labels: tuple[str, ...]
    selected_run_count: int
    counts_by_partition: Mapping[str, int]
    counts_by_status: Mapping[str, int]
    counts_by_magnitude: Mapping[int, int]
    unique_seed_count: int
    seed_sets_pairwise_disjoint: bool


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
    """Authorize an exact allocation member without opening the HDF5 file."""

    if partition not in PARTITIONS:
        raise ValueError(f"unknown PCLR v2 partition {partition!r}")
    idv_int = _require_plain_int(idv, name="IDV")
    run_int = _require_plain_int(simulation_run, name="simulation_run")
    if run_int not in range(1, 101):
        raise ValueError(
            "simulation_run must be in 1..100 for the allocated lower-magnitude "
            "study; magnitude-100 target Run1..200 remains permanently consumed"
        )
    if partition == CALIBRATION and idv_int in QUARANTINED_FAULT_IDVS:
        raise ValueError("magnitude-25 fault IDV3/9/15 branches are quarantined")

    permitted_idvs = (1,) if partition == CALIBRATION else CONDITION_IDVS
    if idv_int not in permitted_idvs:
        raise ValueError("requested partition/IDV/simulation_run is not authorized")

    magnitude = PARTITION_MAGNITUDES[partition]
    reported_fault_id = 0 if idv_int == 1 else idv_int
    window = HEALTHY_WINDOW if idv_int == 1 else FAULT_WINDOW
    return AuthorizedSelection(
        partition=partition,
        idv=idv_int,
        simulation_run=run_int,
        fault_magnitude_percent=magnitude,
        reported_fault_id=reported_fault_id,
        window=window,
    )


def iter_authorized_selections(
    partitions: Iterable[str] = PARTITIONS,
) -> tuple[AuthorizedSelection, ...]:
    """Enumerate the immutable 900-run allocation without opening HDF5."""

    requested = tuple(partitions)
    if len(set(requested)) != len(requested) or any(p not in PARTITIONS for p in requested):
        raise ValueError("partitions must be unique members of the PCLR v2 allocation")
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


def _base_path(
    status: str, idv: int, fault_magnitude_percent: int, simulation_run: int
) -> str:
    return (
        f"Mode1/SingleFault/{status}/IDV{idv}/"
        f"Mode1_IDVInfo_{idv}_{fault_magnitude_percent}/Run{simulation_run}"
    )


def _read_rng_seed(group: Any, *, group_path: str) -> int:
    if "seed" not in group.attrs:
        raise ValueError(f"{group_path}: Run group has no exact seed attribute")
    raw = np.asarray(group.attrs["seed"])
    if raw.shape != (1,):
        raise ValueError(f"{group_path}: seed attribute must contain exactly one element")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{group_path}: seed attribute must be numeric")
    try:
        value = float(raw[0])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{group_path}: seed attribute must be numeric") from error
    if not np.isfinite(value):
        raise ValueError(f"{group_path}: seed attribute must be finite")
    if value <= 0.0 or not value.is_integer():
        raise ValueError(f"{group_path}: seed attribute must be a positive integer value")
    return int(value)


def locate_unique_run(handle: Any, selection: AuthorizedSelection) -> LocatedRun:
    """Resolve an authorized run that exists only under SimulationCompleted."""

    canonical = authorize_request(
        partition=selection.partition,
        idv=selection.idv,
        simulation_run=selection.simulation_run,
    )
    if selection != canonical:
        raise ValueError("AuthorizedSelection fields do not match the canonical allocation")

    h5py = _require_h5py()
    paths = {
        status: _base_path(
            status,
            selection.idv,
            selection.fault_magnitude_percent,
            selection.simulation_run,
        )
        for status in ("SimulationCompleted", "SimulationStopped")
    }
    observed = tuple(status for status, path in paths.items() if path in handle)
    if observed != ("SimulationCompleted",):
        raise ValueError(
            f"IDV{selection.idv} magnitude{selection.fault_magnitude_percent} "
            f"Run{selection.simulation_run} must exist only in SimulationCompleted; "
            f"observed statuses {observed}"
        )
    group_path = paths["SimulationCompleted"]
    group = handle[group_path]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"{group_path} exists but is not an HDF5 group")
    processdata_path = f"{group_path}/processdata"
    idv_init_path = f"{group_path}/idv_init"
    if processdata_path not in handle:
        raise ValueError(f"{group_path} has no exact processdata dataset")
    if idv_init_path not in handle:
        raise ValueError(f"{group_path} has no exact idv_init dataset")
    if not isinstance(handle[processdata_path], h5py.Dataset):
        raise ValueError(f"{processdata_path} is not an HDF5 dataset")
    if not isinstance(handle[idv_init_path], h5py.Dataset):
        raise ValueError(f"{idv_init_path} is not an HDF5 dataset")
    return LocatedRun(
        status="SimulationCompleted",
        idv=selection.idv,
        simulation_run=selection.simulation_run,
        fault_magnitude_percent=selection.fault_magnitude_percent,
        rng_seed=_read_rng_seed(group, group_path=group_path),
        group_path=group_path,
        processdata_path=processdata_path,
        idv_init_path=idv_init_path,
    )


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
    if processdata.shape != (NORMALIZED_LENGTH, 54):
        raise ValueError(
            f"{located.processdata_path}: expected exact shape (2000, 54), "
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
    expected[:, located.idv - 1] = (
        0.0,
        located.fault_magnitude_percent / 100.0,
        30.0,
    )
    if not np.array_equal(values, expected):
        raise ValueError(
            f"{located.idv_init_path}: must declare only IDV{located.idv} as "
            f"[0, {located.fault_magnitude_percent / 100.0:g}, 30]"
        )


def _validate_time_values(
    times: np.ndarray, *, path: str, normalized_window: tuple[int, int]
) -> tuple[float, float]:
    if times.shape != (SEQUENCE_LENGTH,) or not np.isfinite(times).all():
        raise ValueError(f"{path}: authorized time slice must be 500 finite values")
    start, stop = normalized_window
    expected = (np.arange(start, stop, dtype=np.float64) + 1.0) * SAMPLE_PERIOD_HOURS
    if not np.allclose(times, expected, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{path}: lower-magnitude absolute time must equal (raw_row + 1) * 0.05 hours"
        )
    # Persist canonical decimal-hour endpoints instead of platform-dependent
    # binary spellings such as 5.050000000000001.
    return round(float(times[0]), 12), round(float(times[-1]), 12)


def _materialize_run(
    handle: Any, *, partition: str, idv: int, simulation_run: int
) -> PCLRRun:
    """Read exactly one authorized 500-row window."""

    # Authorization is deliberately first: invalid requests cannot inspect the
    # archive, including its group topology and seed attributes.
    selection = authorize_request(
        partition=partition, idv=idv, simulation_run=simulation_run
    )
    located = locate_unique_run(handle, selection)
    _validate_dataset_metadata(handle, located)
    _validate_idv_init_values(handle, located)

    start, stop = selection.window
    dataset = handle[located.processdata_path]
    values = np.array(
        dataset[start:stop, 1:53], dtype=np.float64, copy=True, order="C"
    )
    times = np.asarray(dataset[start:stop, 0], dtype=np.float64)
    if values.shape != (SEQUENCE_LENGTH, len(FEATURE_NAMES)):
        raise ValueError(f"{located.processdata_path}: authorized window shape changed")
    if values.dtype != np.dtype(np.float64) or not np.isfinite(values).all():
        raise ValueError(
            f"{located.processdata_path}: authorized values must be finite float64"
        )
    time_first, time_last = _validate_time_values(
        times,
        path=located.processdata_path,
        normalized_window=selection.window,
    )

    values.setflags(write=False)
    fault = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    if selection.reported_fault_id:
        fault[LOCAL_FAULT_START:] = True
    fault.setflags(write=False)
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "dataset": "Extended_TEP_Mode1",
            "dataset_doi": DATASET_DOI,
            "study": "PCLR_successor_v2_lower_magnitude",
            "partition": selection.partition,
            "fault_id": selection.reported_fault_id,
            "source_idv": selection.idv,
            "simulation_run": selection.simulation_run,
            "fault_magnitude_percent": selection.fault_magnitude_percent,
            "rng_seed": located.rng_seed,
            "group_key": selection.simulation_run,
            "group_key_name": "simulation_run_balanced_stratification_block",
            "grouping_semantics": "balanced_stratification_block_only",
            "source_status": located.status,
            "source_group_path": located.group_path,
            "processdata_path": located.processdata_path,
            "raw_row_zero_dropped": False,
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
        f"PCLR-v2-ExtendedTEP-Mode1-{selection.partition}-magnitude-"
        f"{selection.fault_magnitude_percent:02d}-fault-"
        f"{selection.reported_fault_id:02d}-IDV-{selection.idv:02d}-"
        f"run-{selection.simulation_run:03d}"
    )
    return PCLRRun(name, values, fault, FEATURE_NAMES, metadata)


def _validate_development_groups(runs: Sequence[PCLRRun]) -> tuple[DevelopmentGroup, ...]:
    expected_count = len(DEVELOPMENT_RUNS) * len(CONDITION_IDVS)
    if len(runs) != expected_count:
        raise RuntimeError(f"development allocation must contain {expected_count} runs")
    by_block: dict[int, list[PCLRRun]] = {run: [] for run in DEVELOPMENT_RUNS}
    for item in runs:
        if item.simulation_run not in by_block:
            raise RuntimeError("development result contains an invalid stratification block")
        by_block[item.simulation_run].append(item)
    if any(
        len(by_block[block]) != len(CONDITION_IDVS)
        or {item.source_idv for item in by_block[block]} != set(CONDITION_IDVS)
        for block in DEVELOPMENT_RUNS
    ):
        raise RuntimeError(
            "each development stratification block must contain exactly four conditions"
        )
    groups = tuple(
        DevelopmentGroup(
            simulation_run=block,
            conditions=tuple(
                next(item for item in by_block[block] if item.source_idv == idv)
                for idv in CONDITION_IDVS
            ),
        )
        for block in DEVELOPMENT_RUNS
    )
    seeds = tuple(item.rng_seed for item in runs)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("development rng_seed values must be globally unique")
    return groups


def validate_archive_topology(handle: Any) -> TopologySummary:
    """Validate all 900 allocated paths without decoding run dataset values."""

    labels = validate_processdata_labels(handle)
    selections = iter_authorized_selections()
    partition_counts = {partition: 0 for partition in PARTITIONS}
    status_counts = {"SimulationCompleted": 0, "SimulationStopped": 0}
    magnitude_counts = {25: 0, 50: 0, 75: 0}
    seeds_by_partition: dict[str, set[int]] = {partition: set() for partition in PARTITIONS}
    owner_by_seed: dict[int, AuthorizedSelection] = {}
    development_ids: dict[int, set[int]] = {run: set() for run in DEVELOPMENT_RUNS}
    for selection in selections:
        located = locate_unique_run(handle, selection)
        _validate_dataset_metadata(handle, located)
        if located.rng_seed in owner_by_seed:
            previous = owner_by_seed[located.rng_seed]
            raise ValueError(
                "allocated rng_seed values must be globally unique; "
                f"seed {located.rng_seed} is reused by "
                f"{previous.partition}/IDV{previous.idv}/Run{previous.simulation_run} and "
                f"{selection.partition}/IDV{selection.idv}/Run{selection.simulation_run}"
            )
        owner_by_seed[located.rng_seed] = selection
        seeds_by_partition[selection.partition].add(located.rng_seed)
        partition_counts[selection.partition] += 1
        status_counts[located.status] += 1
        magnitude_counts[selection.fault_magnitude_percent] += 1
        if selection.partition == DEVELOPMENT:
            development_ids[selection.simulation_run].add(selection.idv)
    if any(ids != set(CONDITION_IDVS) for ids in development_ids.values()):
        raise RuntimeError("development topology lost a balanced four-condition block")
    seed_sets = tuple(seeds_by_partition[p] for p in PARTITIONS)
    pairwise_disjoint = all(
        seed_sets[left].isdisjoint(seed_sets[right])
        for left in range(len(seed_sets))
        for right in range(left + 1, len(seed_sets))
    )
    if not pairwise_disjoint:
        raise RuntimeError("partition rng_seed sets must be pairwise disjoint")
    return TopologySummary(
        labels=labels,
        selected_run_count=len(selections),
        counts_by_partition=MappingProxyType(partition_counts),
        counts_by_status=MappingProxyType(status_counts),
        counts_by_magnitude=MappingProxyType(magnitude_counts),
        unique_seed_count=len(owner_by_seed),
        seed_sets_pairwise_disjoint=True,
    )


class PCLRMode1Archive:
    """Single-file archive exposing only the immutable PCLR v2 allocation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.name != MODE1_FILE_NAME:
            raise ValueError(f"PCLR v2 requires the single official file {MODE1_FILE_NAME}")

    def _open(self) -> AbstractContextManager[Any]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        return _require_h5py().File(self.path, "r")

    def validate_topology(self) -> TopologySummary:
        with self._open() as handle:
            return validate_archive_topology(handle)

    def load_run(self, *, partition: str, idv: int, simulation_run: int) -> PCLRRun:
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
        if len(runs) != 100 or any(
            run.source_idv != 1 or run.fault_magnitude_percent != 25 for run in runs
        ):
            raise RuntimeError("calibration must contain only 100 magnitude-25 IDV1 runs")
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
        if len(healthy) != 100 or len(faults) != 300 or any(
            run.fault_magnitude_percent != 75 for run in runs
        ):
            raise RuntimeError(
                "final allocation must contain 100 healthy and 300 fault magnitude-75 runs"
            )
        return healthy, faults


__all__ = [
    "AuthorizedSelection",
    "BASELINE",
    "CALIBRATION",
    "CALIBRATION_RUNS",
    "CONDITION_IDVS",
    "CONSUMED_MAGNITUDE_PERCENT",
    "CONSUMED_TARGET_RUNS",
    "DATASET_DOI",
    "DEVELOPMENT",
    "DEVELOPMENT_RUNS",
    "DevelopmentBlock",
    "DevelopmentGroup",
    "FAULT_IDVS",
    "FAULT_WINDOW",
    "FEATURE_NAMES",
    "FINAL",
    "FINAL_RUNS",
    "HEALTHY_WINDOW",
    "LOCAL_FAULT_START",
    "LocatedRun",
    "MODE1_FILE_NAME",
    "MONITOR",
    "NORMALIZED_LENGTH",
    "PARTITIONS",
    "PARTITION_MAGNITUDES",
    "PCLRMode1Archive",
    "PCLRRun",
    "PROCESSDATA_LABELS",
    "QUARANTINED_FAULT_IDVS",
    "QUARANTINED_MAGNITUDE_PERCENT",
    "QUARANTINED_RUNS",
    "SAMPLE_PERIOD_HOURS",
    "SEGMENTS",
    "SEQUENCE_LENGTH",
    "TopologySummary",
    "WARMUP",
    "authorize_request",
    "iter_authorized_selections",
    "locate_unique_run",
    "validate_archive_topology",
    "validate_processdata_labels",
]
