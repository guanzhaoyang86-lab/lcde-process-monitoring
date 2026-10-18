"""Fail-closed Mode 2 development data contract for CP-ETC-Scan v5.

This module exposes one and only one allocation from the official Extended
Tennessee Eastman archive: Mode 2, magnitude 50, IDV1/3/9/15, Run1..100.
IDV1 contributes a wholly pre-fault healthy window; IDV3/9/15 contribute the
frozen fault-centred window.  A repeated ``simulation_run`` is merely a
balanced-stratification block label.  It is not evidence of paired random
seeds.  CP-ETC v5.1 exactly binds the five disclosed seed collisions and
keeps every seed-connected run component within one frozen fold.

The production archive location and SHA-256 are fixed in code.  There is no
runtime path, digest, mode, magnitude, status, IDV, run, or window override.
The private synthetic-fixture constructor is intentionally unavailable from
``__all__`` and exists only so tests can exercise the HDF5 contract without
opening the real archive.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from mate_spc_pclr.data import FEATURE_NAMES, PROCESSDATA_LABELS


DATASET_DOI = "10.11583/DTU.13385936.v1"
MODE2_FILE_NAME = "TEP_Mode2.h5"
MODE2_SHA256 = "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
PROTOCOL_ID = "cp-etc-scan-v5.1-mode2-development"
PROTOCOL_SHA256 = "55cd6dcb09409f53db4f078f2441f18a7659c1ce97b7165876d3f2a891a37923"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODE2_RELATIVE_PATH = Path("data") / "extended_tep_dtu_mode2" / MODE2_FILE_NAME
OFFICIAL_MODE2_PATH = (PROJECT_ROOT / MODE2_RELATIVE_PATH).resolve()

SOURCE_MODE = 2
DEVELOPMENT_PARTITION = "cross_operating_mode_development"
CONDITION_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
FAULT_MAGNITUDE_PERCENT = 50
DEVELOPMENT_RUNS = tuple(range(1, 101))

REQUIRED_UNIQUE_SEED_COUNT = 395
ALLOWED_DUPLICATE_SEED_OWNERS: Mapping[
    int, tuple[tuple[int, int], ...]
] = MappingProxyType(
    {
        1_313_107: ((14, 1), (43, 9)),
        1_314_539: ((9, 9), (88, 15)),
        1_315_245: ((6, 3), (76, 15)),
        1_319_584: ((43, 3), (98, 9)),
        1_319_919: ((24, 3), (33, 15)),
    }
)
FOLD_RUN_IDS: Mapping[int, tuple[int, ...]] = MappingProxyType(
    {
        0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
        1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
        2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
        3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
        4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
    }
)
RUN_TO_FOLD: Mapping[int, int] = MappingProxyType(
    {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}
)
if (
    tuple(sorted(RUN_TO_FOLD)) != DEVELOPMENT_RUNS
    or len(RUN_TO_FOLD) != sum(len(runs) for runs in FOLD_RUN_IDS.values())
    or tuple(len(FOLD_RUN_IDS[fold]) for fold in range(5)) != (20, 20, 20, 20, 20)
):
    raise RuntimeError("frozen v5.1 seed-component fold allocation is malformed")

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
class DevelopmentAllocation:
    """The complete immutable Mode 2 development allocation."""

    source_mode: int
    partition: str
    condition_idvs: tuple[int, ...]
    fault_magnitude_percent: int
    simulation_runs: tuple[int, ...]
    healthy_window: tuple[int, int]
    fault_window: tuple[int, int]


MODE2_DEVELOPMENT_ALLOCATION = DevelopmentAllocation(
    source_mode=SOURCE_MODE,
    partition=DEVELOPMENT_PARTITION,
    condition_idvs=CONDITION_IDVS,
    fault_magnitude_percent=FAULT_MAGNITUDE_PERCENT,
    simulation_runs=DEVELOPMENT_RUNS,
    healthy_window=HEALTHY_WINDOW,
    fault_window=FAULT_WINDOW,
)


@dataclass(frozen=True)
class AuthorizedSelection:
    """One selection proven to be a member of the frozen allocation."""

    idv: int
    simulation_run: int
    reported_fault_id: int
    window: tuple[int, int]


@dataclass(frozen=True)
class LocatedRun:
    """The sole completed HDF5 location and authenticated seed for one run."""

    status: str
    idv: int
    simulation_run: int
    rng_seed: int
    group_path: str
    processdata_path: str
    idv_init_path: str


@dataclass(frozen=True)
class CPETCDevelopmentRun:
    """One immutable 500-by-52 cross-operating-mode development sequence."""

    name: str
    x: np.ndarray
    fault: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def source_idv(self) -> int:
        return int(self.metadata["source_idv"])

    @property
    def simulation_run(self) -> int:
        return int(self.metadata["simulation_run"])

    @property
    def group_key(self) -> int:
        return int(self.metadata["group_key"])

    @property
    def rng_seed(self) -> int:
        return int(self.metadata["rng_seed"])

    @property
    def fault_id(self) -> int:
        return int(self.metadata["fault_id"])

    def segment(self, name: str) -> np.ndarray:
        """Return a read-only view of the frozen baseline/warmup/monitor segment."""

        if name not in SEGMENTS:
            raise ValueError(f"unknown CP-ETC v5 segment {name!r}")
        start, stop = SEGMENTS[name]
        result = self.x[start:stop]
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class DevelopmentBlock:
    """Four class-balanced conditions sharing only a stratification label."""

    simulation_run: int
    conditions: tuple[CPETCDevelopmentRun, ...]

    def __post_init__(self) -> None:
        block = _require_plain_int(self.simulation_run, name="development block")
        if block not in DEVELOPMENT_RUNS:
            raise ValueError("development stratification block must be a run in 1..100")
        if tuple(item.source_idv for item in self.conditions) != CONDITION_IDVS:
            raise ValueError("development block must contain IDV1/3/9/15 once in order")
        if any(
            item.simulation_run != block
            or item.group_key != block
            or item.metadata.get("partition") != DEVELOPMENT_PARTITION
            for item in self.conditions
        ):
            raise ValueError("all conditions must carry the same balanced block label")
        seeds = tuple(item.rng_seed for item in self.conditions)
        if len(set(seeds)) != len(seeds):
            raise ValueError("conditions in one block must have distinct rng_seed values")

    @property
    def stratified_block(self) -> int:
        return int(self.simulation_run)

    @property
    def fold_id(self) -> int:
        return int(RUN_TO_FOLD[self.simulation_run])

    def by_idv(self, idv: int) -> CPETCDevelopmentRun:
        idv_int = _require_plain_int(idv, name="IDV")
        if idv_int not in CONDITION_IDVS:
            raise ValueError("development IDV must be one of 1, 3, 9, 15")
        return self.conditions[CONDITION_IDVS.index(idv_int)]


DevelopmentGroup = DevelopmentBlock


@dataclass(frozen=True)
class TopologySummary:
    """Metadata-only result; it retains no processdata or idv_init values."""

    labels: tuple[str, ...]
    selected_run_count: int
    counts_by_status: Mapping[str, int]
    counts_by_idv: Mapping[int, int]
    unique_seed_count: int
    duplicate_seed_value_count: int
    duplicate_seed_owners: Mapping[int, tuple[tuple[int, int], ...]]
    cross_fold_seed_collision_count: int
    source_mode: int
    fault_magnitude_percent: int
    data_role: str


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("CP-ETC v5 Mode 2 loading requires h5py>=3.10") from error
    return h5py


def _require_plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def authorize_development_allocation(
    allocation: DevelopmentAllocation = MODE2_DEVELOPMENT_ALLOCATION,
) -> DevelopmentAllocation:
    """Reject any allocation change without inspecting or opening the archive."""

    if not isinstance(allocation, DevelopmentAllocation):
        raise ValueError("CP-ETC v5 allocation must be a DevelopmentAllocation")
    if allocation != MODE2_DEVELOPMENT_ALLOCATION:
        raise ValueError(
            "only Mode2/magnitude50/IDV1,3,9,15/Run1..100 frozen development is authorized"
        )
    return MODE2_DEVELOPMENT_ALLOCATION


def authorize_request(*, idv: int, simulation_run: int) -> AuthorizedSelection:
    """Authorize one exact allocation member without opening HDF5."""

    idv_int = _require_plain_int(idv, name="IDV")
    run_int = _require_plain_int(simulation_run, name="simulation_run")
    if idv_int not in CONDITION_IDVS:
        raise ValueError("only Mode 2 development IDV1/3/9/15 is authorized")
    if run_int not in DEVELOPMENT_RUNS:
        raise ValueError("only Mode 2 development Run1..100 is authorized")
    return AuthorizedSelection(
        idv=idv_int,
        simulation_run=run_int,
        reported_fault_id=0 if idv_int == 1 else idv_int,
        window=HEALTHY_WINDOW if idv_int == 1 else FAULT_WINDOW,
    )


def fold_for_run(simulation_run: int) -> int:
    """Return the frozen seed-component fold for an authorized run ID."""

    run_int = _require_plain_int(simulation_run, name="simulation_run")
    if run_int not in DEVELOPMENT_RUNS:
        raise ValueError("only Mode 2 development Run1..100 is authorized")
    return int(RUN_TO_FOLD[run_int])


def iter_authorized_selections(
    allocation: DevelopmentAllocation = MODE2_DEVELOPMENT_ALLOCATION,
) -> tuple[AuthorizedSelection, ...]:
    """Enumerate exactly 400 selections after authenticating the allocation."""

    canonical = authorize_development_allocation(allocation)
    return tuple(
        authorize_request(idv=idv, simulation_run=run)
        for run in canonical.simulation_runs
        for idv in canonical.condition_idvs
    )


def _base_path(status: str, idv: int, simulation_run: int) -> str:
    return (
        f"Mode2/SingleFault/{status}/IDV{idv}/"
        f"Mode2_IDVInfo_{idv}_{FAULT_MAGNITUDE_PERCENT}/Run{simulation_run}"
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

    canonical = authorize_request(idv=selection.idv, simulation_run=selection.simulation_run)
    if selection != canonical:
        raise ValueError("AuthorizedSelection fields do not match the canonical allocation")
    h5py = _require_h5py()
    paths = {
        status: _base_path(status, selection.idv, selection.simulation_run)
        for status in ("SimulationCompleted", "SimulationStopped")
    }
    observed = tuple(status for status, path in paths.items() if path in handle)
    if observed != ("SimulationCompleted",):
        raise ValueError(
            f"Mode2 IDV{selection.idv} magnitude50 Run{selection.simulation_run} "
            "must exist only in SimulationCompleted; "
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
        rng_seed=_read_rng_seed(group, group_path=group_path),
        group_path=group_path,
        processdata_path=processdata_path,
        idv_init_path=idv_init_path,
    )


def validate_processdata_labels(handle: Any) -> tuple[str, ...]:
    """Decode and exactly authenticate the official 54-column order."""

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
    expected[:, located.idv - 1] = (0.0, 0.5, 30.0)
    if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{located.idv_init_path}: must declare only IDV{located.idv} as [0, 0.5, 30]"
        )


def _validate_full_time_column(dataset: Any, *, path: str) -> np.ndarray:
    """Authenticate all 2,000 absolute timestamps during numeric materialization."""

    times = np.asarray(dataset[:, 0])
    if times.shape != (NORMALIZED_LENGTH,):
        raise ValueError(f"{path}: complete time column must contain exactly 2000 values")
    if times.dtype != np.dtype(np.float64):
        raise ValueError(f"{path}: complete time column must be exact float64")
    if not np.isfinite(times).all():
        raise ValueError(f"{path}: complete time column must be finite")
    expected = (
        np.arange(NORMALIZED_LENGTH, dtype=np.float64) + 1.0
    ) * SAMPLE_PERIOD_HOURS
    if not np.allclose(times, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{path}: complete Mode 2 time column (absolute time) must equal "
            "(raw_row + 1) * 0.05 hours"
        )
    times.setflags(write=False)
    return times


def _validate_time_values(
    times: np.ndarray, *, path: str, normalized_window: tuple[int, int]
) -> tuple[float, float]:
    if times.shape != (SEQUENCE_LENGTH,) or not np.isfinite(times).all():
        raise ValueError(f"{path}: authorized time slice must be 500 finite values")
    start, stop = normalized_window
    expected = (np.arange(start, stop, dtype=np.float64) + 1.0) * SAMPLE_PERIOD_HOURS
    if not np.allclose(times, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{path}: Mode 2 absolute time must equal (raw_row + 1) * 0.05 hours"
        )
    return round(float(times[0]), 12), round(float(times[-1]), 12)


def _materialize_run(
    handle: Any,
    selection: AuthorizedSelection,
    *,
    located: LocatedRun | None = None,
) -> CPETCDevelopmentRun:
    """Read only one authorized 500-row projection, after fault authentication."""

    canonical = authorize_request(idv=selection.idv, simulation_run=selection.simulation_run)
    if selection != canonical:
        raise ValueError("AuthorizedSelection fields do not match the canonical allocation")
    source = locate_unique_run(handle, canonical) if located is None else located
    if source.idv != canonical.idv or source.simulation_run != canonical.simulation_run:
        raise ValueError("located run identity does not match the authorized selection")
    _validate_dataset_metadata(handle, source)
    # Authenticate the declared fault before decoding any processdata values.
    _validate_idv_init_values(handle, source)
    start, stop = canonical.window
    dataset = handle[source.processdata_path]
    complete_times = _validate_full_time_column(dataset, path=source.processdata_path)
    values = np.array(dataset[start:stop, 1:53], dtype=np.float64, copy=True, order="C")
    times = complete_times[start:stop]
    if values.shape != (SEQUENCE_LENGTH, len(FEATURE_NAMES)):
        raise ValueError(f"{source.processdata_path}: authorized window shape changed")
    if values.dtype != np.dtype(np.float64) or not np.isfinite(values).all():
        raise ValueError(f"{source.processdata_path}: authorized values must be finite float64")
    time_first, time_last = _validate_time_values(
        times, path=source.processdata_path, normalized_window=canonical.window
    )
    values.setflags(write=False)
    fault = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    if canonical.reported_fault_id:
        fault[LOCAL_FAULT_START:] = True
    fault.setflags(write=False)
    metadata: Mapping[str, Any] = MappingProxyType(
        {
            "dataset": "Extended_TEP_Mode2",
            "dataset_doi": DATASET_DOI,
            "study": "CP_ETC_Scan_v5_1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "partition": DEVELOPMENT_PARTITION,
            "data_role": "cross_operating_mode_development_only",
            "source_operating_mode": SOURCE_MODE,
            "intended_target_operating_mode": 1,
            "not_formal_calibration_or_final_confirmation": True,
            "fault_id": canonical.reported_fault_id,
            "source_idv": canonical.idv,
            "simulation_run": canonical.simulation_run,
            "fault_magnitude_percent": FAULT_MAGNITUDE_PERCENT,
            "rng_seed": source.rng_seed,
            "group_key": canonical.simulation_run,
            "fold_id": fold_for_run(canonical.simulation_run),
            "group_key_name": "simulation_run_balanced_stratification_block",
            "grouping_semantics": "balanced_stratification_block_only_not_paired_seed",
            "seed_contract": "v5.1_exact_395_unique_and_5_disclosed_collisions",
            "source_status": source.status,
            "source_group_path": source.group_path,
            "processdata_path": source.processdata_path,
            "raw_row_zero_dropped": False,
            "raw_column_half_open_projection": (1, 53),
            "normalized_window_half_open": canonical.window,
            "sample_period_hours": SAMPLE_PERIOD_HOURS,
            "source_time_hours_first_last": (time_first, time_last),
            "fault_start_zero_based": (
                LOCAL_FAULT_START if canonical.reported_fault_id else None
            ),
            "pseudo_onset_zero_based": LOCAL_FAULT_START,
            "segments_half_open": SEGMENTS,
        }
    )
    name = (
        "CPETC-v5-ExtendedTEP-Mode2-cross-mode-development-magnitude-50-"
        f"fault-{canonical.reported_fault_id:02d}-IDV-{canonical.idv:02d}-"
        f"run-{canonical.simulation_run:03d}"
    )
    return CPETCDevelopmentRun(name, values, fault, FEATURE_NAMES, metadata)


def _scan_topology(
    handle: Any,
    selections: Sequence[AuthorizedSelection],
) -> tuple[TopologySummary, Mapping[tuple[int, int], LocatedRun]]:
    labels = validate_processdata_labels(handle)
    locations: dict[tuple[int, int], LocatedRun] = {}
    owners_by_seed: dict[int, list[AuthorizedSelection]] = {}
    counts_by_idv = {idv: 0 for idv in CONDITION_IDVS}
    for selection in selections:
        located = locate_unique_run(handle, selection)
        _validate_dataset_metadata(handle, located)
        owners_by_seed.setdefault(located.rng_seed, []).append(selection)
        key = (selection.simulation_run, selection.idv)
        if key in locations:
            raise RuntimeError("duplicate authorized Mode 2 development identity")
        locations[key] = located
        counts_by_idv[selection.idv] += 1
    if len(locations) != 400 or any(count != 100 for count in counts_by_idv.values()):
        raise RuntimeError("Mode 2 development topology must contain 100 runs per condition")
    observed_duplicates = {
        seed: tuple(sorted((owner.simulation_run, owner.idv) for owner in owners))
        for seed, owners in owners_by_seed.items()
        if len(owners) > 1
    }
    if (
        len(owners_by_seed) != REQUIRED_UNIQUE_SEED_COUNT
        or observed_duplicates != dict(ALLOWED_DUPLICATE_SEED_OWNERS)
    ):
        raise ValueError(
            "Mode 2 seed owners must exactly match the frozen v5.1 contract: "
            "395 unique values and the five disclosed two-owner collisions"
        )
    cross_fold_collisions = sum(
        len({fold_for_run(owner.simulation_run) for owner in owners}) > 1
        for owners in owners_by_seed.values()
    )
    if cross_fold_collisions:
        raise ValueError("v5.1 seed-connected owners must never cross frozen folds")
    summary = TopologySummary(
        labels=labels,
        selected_run_count=len(locations),
        counts_by_status=MappingProxyType(
            {"SimulationCompleted": len(locations), "SimulationStopped": 0}
        ),
        counts_by_idv=MappingProxyType(counts_by_idv),
        unique_seed_count=len(owners_by_seed),
        duplicate_seed_value_count=len(observed_duplicates),
        duplicate_seed_owners=MappingProxyType(observed_duplicates),
        cross_fold_seed_collision_count=0,
        source_mode=SOURCE_MODE,
        fault_magnitude_percent=FAULT_MAGNITUDE_PERCENT,
        data_role="cross_operating_mode_development_only",
    )
    return summary, MappingProxyType(locations)


def validate_archive_topology(handle: Any) -> TopologySummary:
    """Validate all 400 selected paths without decoding run dataset values."""

    selections = iter_authorized_selections()
    summary, _ = _scan_topology(handle, selections)
    return summary


def _build_development_blocks(
    runs: Sequence[CPETCDevelopmentRun],
) -> tuple[DevelopmentBlock, ...]:
    if len(runs) != 400:
        raise RuntimeError("Mode 2 development allocation must materialize exactly 400 runs")
    by_key = {(item.simulation_run, item.source_idv): item for item in runs}
    if len(by_key) != 400:
        raise RuntimeError("Mode 2 development result contains duplicate identities")
    blocks = tuple(
        DevelopmentBlock(
            simulation_run=run,
            conditions=tuple(by_key[(run, idv)] for idv in CONDITION_IDVS),
        )
        for run in DEVELOPMENT_RUNS
    )
    owners_by_seed: dict[int, list[CPETCDevelopmentRun]] = {}
    for item in runs:
        owners_by_seed.setdefault(item.rng_seed, []).append(item)
    observed_duplicates = {
        seed: tuple(sorted((item.simulation_run, item.source_idv) for item in owners))
        for seed, owners in owners_by_seed.items()
        if len(owners) > 1
    }
    if (
        len(owners_by_seed) != REQUIRED_UNIQUE_SEED_COUNT
        or observed_duplicates != dict(ALLOWED_DUPLICATE_SEED_OWNERS)
    ):
        raise RuntimeError("materialized runs violate the exact v5.1 seed-owner contract")
    if any(
        len({fold_for_run(item.simulation_run) for item in owners}) != 1
        for owners in owners_by_seed.values()
    ):
        raise RuntimeError("materialized seed-connected owners cross frozen folds")
    return blocks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Mode2DevelopmentArchive:
    """Archive facade fixed to the official Mode 2 development source."""

    def __init__(self) -> None:
        self.path = OFFICIAL_MODE2_PATH
        self._test_only_hash_bypass = False

    @classmethod
    def _for_synthetic_test(cls, path: str | Path) -> "Mode2DevelopmentArchive":
        """Create an explicit test fixture; never call this from an experiment runner."""

        fixture = cls.__new__(cls)
        fixture.path = Path(path)
        if fixture.path.name != MODE2_FILE_NAME:
            raise ValueError(f"synthetic fixture must retain the filename {MODE2_FILE_NAME}")
        fixture._test_only_hash_bypass = True
        return fixture

    def _authenticate_source(self) -> None:
        if self.path.name != MODE2_FILE_NAME:
            raise ValueError(f"CP-ETC v5 requires the single official file {MODE2_FILE_NAME}")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self._test_only_hash_bypass:
            return
        if self.path.resolve() != OFFICIAL_MODE2_PATH:
            raise ValueError(
                "production Mode 2 source must use the frozen project data path "
                f"{OFFICIAL_MODE2_PATH}"
            )
        observed = _sha256_file(self.path)
        if observed != MODE2_SHA256:
            raise ValueError(
                "official TEP_Mode2.h5 SHA-256 mismatch: "
                f"expected {MODE2_SHA256}, got {observed}"
            )

    @contextmanager
    def _open(self) -> Iterator[Any]:
        self._authenticate_source()
        with _require_h5py().File(self.path, "r") as handle:
            yield handle

    def validate_topology(
        self,
        allocation: DevelopmentAllocation = MODE2_DEVELOPMENT_ALLOCATION,
    ) -> TopologySummary:
        # Allocation authorization is intentionally before filesystem/HDF5 access.
        canonical = authorize_development_allocation(allocation)
        selections = iter_authorized_selections(canonical)
        with self._open() as handle:
            summary, _ = _scan_topology(handle, selections)
        return summary

    def load_development_blocks(
        self,
        allocation: DevelopmentAllocation = MODE2_DEVELOPMENT_ALLOCATION,
    ) -> tuple[DevelopmentBlock, ...]:
        """Load the sole authorized allocation as 100 immutable balanced blocks."""

        # Allocation authorization is intentionally before filesystem/HDF5 access.
        canonical = authorize_development_allocation(allocation)
        selections = iter_authorized_selections(canonical)
        with self._open() as handle:
            _, locations = _scan_topology(handle, selections)
            runs = tuple(
                _materialize_run(
                    handle,
                    selection,
                    located=locations[(selection.simulation_run, selection.idv)],
                )
                for selection in selections
            )
        return _build_development_blocks(runs)


__all__ = [
    "ALLOWED_DUPLICATE_SEED_OWNERS",
    "AuthorizedSelection",
    "BASELINE",
    "CONDITION_IDVS",
    "CPETCDevelopmentRun",
    "DATASET_DOI",
    "DEVELOPMENT_PARTITION",
    "DEVELOPMENT_RUNS",
    "DevelopmentAllocation",
    "DevelopmentBlock",
    "DevelopmentGroup",
    "FAULT_IDVS",
    "FAULT_MAGNITUDE_PERCENT",
    "FAULT_WINDOW",
    "FOLD_RUN_IDS",
    "FEATURE_NAMES",
    "HEALTHY_WINDOW",
    "LOCAL_FAULT_START",
    "LocatedRun",
    "MODE2_DEVELOPMENT_ALLOCATION",
    "MODE2_FILE_NAME",
    "MODE2_RELATIVE_PATH",
    "MODE2_SHA256",
    "MONITOR",
    "Mode2DevelopmentArchive",
    "NORMALIZED_LENGTH",
    "OFFICIAL_MODE2_PATH",
    "PROTOCOL_ID",
    "PROTOCOL_SHA256",
    "PROCESSDATA_LABELS",
    "REQUIRED_UNIQUE_SEED_COUNT",
    "RUN_TO_FOLD",
    "SAMPLE_PERIOD_HOURS",
    "SEGMENTS",
    "SEQUENCE_LENGTH",
    "SOURCE_MODE",
    "TopologySummary",
    "WARMUP",
    "authorize_development_allocation",
    "authorize_request",
    "fold_for_run",
    "iter_authorized_selections",
    "locate_unique_run",
    "validate_archive_topology",
    "validate_processdata_labels",
]
