from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import mate_spc_pclr.data_v2 as data  # noqa: E402


def _raw(*, offset: float = 0.0) -> np.ndarray:
    values = np.arange(2000 * 54, dtype=np.float64).reshape(2000, 54)
    values[:, 1:] += offset
    values[:, 0] = (np.arange(2000, dtype=np.float64) + 1.0) * data.SAMPLE_PERIOD_HOURS
    return values


def _idv_init(idv: int, magnitude: int) -> np.ndarray:
    values = np.zeros((3, 28), dtype=np.float64)
    values[:, idv - 1] = (0.0, magnitude / 100.0, 30.0)
    return values


def _add_labels(handle: h5py.File) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "Processdata_Labels",
        data=np.asarray(data.PROCESSDATA_LABELS, dtype=object),
        dtype=string_type,
    )


def _add_run(
    handle: h5py.File,
    *,
    idv: int,
    magnitude: int,
    run: int,
    seed: object = np.asarray([12345.0]),
    status: str = "SimulationCompleted",
    processdata: np.ndarray | None = None,
    idv_init: np.ndarray | None = None,
) -> str:
    group_path = data._base_path(status, idv, magnitude, run)
    group = handle.require_group(group_path)
    if seed is not None:
        group.attrs["seed"] = seed
    group.create_dataset("processdata", data=_raw() if processdata is None else processdata)
    group.create_dataset(
        "idv_init",
        data=_idv_init(idv, magnitude) if idv_init is None else idv_init,
    )
    return group_path


def _source(tmp_path: Path) -> Path:
    return tmp_path / data.MODE1_FILE_NAME


def _dummy_run(*, block: int, idv: int, seed: int) -> data.PCLRRun:
    x = np.zeros((500, 52), dtype=np.float64)
    x.setflags(write=False)
    fault = np.zeros(500, dtype=np.bool_)
    if idv != 1:
        fault[160:] = True
    fault.setflags(write=False)
    return data.PCLRRun(
        name=f"dummy-{block}-{idv}",
        x=x,
        fault=fault,
        feature_names=data.FEATURE_NAMES,
        metadata=MappingProxyType(
            {
                "partition": data.DEVELOPMENT,
                "source_idv": idv,
                "simulation_run": block,
                "group_key": block,
                "fault_magnitude_percent": 50,
                "rng_seed": seed,
            }
        ),
    )


def test_allocation_is_exact_and_magnitude_identified() -> None:
    assert data.PARTITION_MAGNITUDES == {
        data.DEVELOPMENT: 50,
        data.CALIBRATION: 25,
        data.FINAL: 75,
    }
    assert data.DEVELOPMENT_RUNS == data.CALIBRATION_RUNS == data.FINAL_RUNS
    assert data.DEVELOPMENT_RUNS == tuple(range(1, 101))
    assert data.CONSUMED_MAGNITUDE_PERCENT == 100
    assert data.CONSUMED_TARGET_RUNS == tuple(range(1, 201))

    selections = data.iter_authorized_selections()
    assert len(selections) == 900
    assert sum(item.partition == data.DEVELOPMENT for item in selections) == 400
    assert sum(item.partition == data.CALIBRATION for item in selections) == 100
    assert sum(item.partition == data.FINAL for item in selections) == 400
    assert {
        item.idv for item in selections if item.partition == data.CALIBRATION
    } == {1}
    assert {
        item.fault_magnitude_percent
        for item in selections
        if item.partition == data.CALIBRATION
    } == {25}


def test_calibration_faults_are_quarantined_and_no_magnitude_override_exists() -> None:
    for idv in data.FAULT_IDVS:
        with pytest.raises(ValueError, match="magnitude-25.*quarantined"):
            data.authorize_request(
                partition=data.CALIBRATION, idv=idv, simulation_run=1
            )
    assert "fault_magnitude_percent" not in inspect.signature(
        data.PCLRMode1Archive.load_run
    ).parameters


@pytest.mark.parametrize(
    ("partition", "idv", "run", "message"),
    [
        ("audit", 1, 1, "unknown PCLR v2 partition"),
        (data.DEVELOPMENT, 2, 1, "not authorized"),
        (data.FINAL, 1, 0, "permanently consumed"),
        (data.FINAL, 1, 101, "permanently consumed"),
    ],
)
def test_unauthorized_request_is_rejected_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
    idv: int,
    run: int,
    message: str,
) -> None:
    archive = data.PCLRMode1Archive(_source(tmp_path))
    opened = False

    def forbidden_open(self: data.PCLRMode1Archive):  # type: ignore[no-untyped-def]
        nonlocal opened
        opened = True
        raise AssertionError("unauthorized request opened HDF5")

    monkeypatch.setattr(data.PCLRMode1Archive, "_open", forbidden_open)
    with pytest.raises(ValueError, match=message):
        archive.load_run(partition=partition, idv=idv, simulation_run=run)
    assert opened is False


def test_lower_magnitude_rows_windows_fault_labels_and_metadata(tmp_path: Path) -> None:
    source = _source(tmp_path)
    healthy_raw = _raw(offset=10.0)
    fault_raw = _raw(offset=30.0)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=1,
            magnitude=50,
            run=1,
            seed=np.asarray([347639.0]),
            processdata=healthy_raw,
        )
        _add_run(
            handle,
            idv=3,
            magnitude=50,
            run=1,
            seed=np.asarray([89123.0]),
            processdata=fault_raw,
        )

    archive = data.PCLRMode1Archive(source)
    healthy = archive.load_run(partition=data.DEVELOPMENT, idv=1, simulation_run=1)
    fault = archive.load_run(partition=data.DEVELOPMENT, idv=3, simulation_run=1)

    np.testing.assert_array_equal(healthy.x, healthy_raw[100:600, 1:53])
    np.testing.assert_array_equal(fault.x, fault_raw[440:940, 1:53])
    assert healthy.x.shape == fault.x.shape == (500, 52)
    assert healthy.x.flags.writeable is False
    assert not healthy.fault.any()
    assert not fault.fault[:160].any()
    assert fault.fault[160:].all()
    assert healthy.metadata["raw_row_zero_dropped"] is False
    assert healthy.metadata["normalized_window_half_open"] == (100, 600)
    assert healthy.metadata["source_time_hours_first_last"] == (5.05, 30.0)
    assert fault.metadata["normalized_window_half_open"] == (440, 940)
    assert fault.metadata["source_time_hours_first_last"] == (22.05, 47.0)
    assert fault.metadata["fault_start_zero_based"] == 160
    assert fault.metadata["fault_magnitude_percent"] == 50
    assert fault.metadata["rng_seed"] == 89123
    assert fault.metadata["grouping_semantics"] == "balanced_stratification_block_only"
    assert fault.segment("baseline").shape == (128, 52)
    assert fault.segment("warmup").shape == (32, 52)
    assert fault.segment("monitor").shape == (340, 52)


def test_absolute_time_axis_includes_expected_local_onset(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = _raw()
    assert raw[600, 0] == pytest.approx(30.05)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=15,
            magnitude=75,
            run=1,
            seed=np.asarray([1001.0]),
            processdata=raw,
        )
    run = data.PCLRMode1Archive(source).load_run(
        partition=data.FINAL, idv=15, simulation_run=1
    )
    assert run.metadata["source_time_hours_first_last"] == (22.05, 47.0)
    assert raw[440 + data.LOCAL_FAULT_START, 0] == pytest.approx(30.05)


def test_shifted_absolute_time_fails_even_when_spacing_is_correct(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = _raw()
    raw[:, 0] -= data.SAMPLE_PERIOD_HOURS
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=1,
            magnitude=25,
            run=1,
            seed=np.asarray([1002.0]),
            processdata=raw,
        )
    with pytest.raises(ValueError, match="absolute time"):
        data.PCLRMode1Archive(source).load_run(
            partition=data.CALIBRATION, idv=1, simulation_run=1
        )


def test_idv_init_uses_partition_magnitude_and_is_checked_before_processdata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[tuple[str, object]] = []

    class SpyDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.shape = (3, 28) if kind == "idv_init" else (2000, 54)
            self.dtype = np.dtype(np.float64)

        def __getitem__(self, key: object) -> np.ndarray:
            accesses.append((self.kind, key))
            if self.kind == "processdata":
                raise AssertionError("invalid idv_init must stop before processdata")
            values = _idv_init(15, 50)
            values[1, 14] = 0.75
            return values

    located = data.LocatedRun(
        status="SimulationCompleted",
        idv=15,
        simulation_run=1,
        fault_magnitude_percent=50,
        rng_seed=5001,
        group_path="synthetic/group",
        processdata_path="synthetic/group/processdata",
        idv_init_path="synthetic/group/idv_init",
    )
    processdata = SpyDataset("processdata")
    idv_init = SpyDataset("idv_init")
    monkeypatch.setattr(data, "locate_unique_run", lambda *args, **kwargs: located)
    with pytest.raises(ValueError, match=r"\[0, 0.5, 30\]"):
        data._materialize_run(
            {
                located.processdata_path: processdata,
                located.idv_init_path: idv_init,
            },
            partition=data.DEVELOPMENT,
            idv=15,
            simulation_run=1,
        )
    assert accesses == [("idv_init", Ellipsis)]


@pytest.mark.parametrize(
    ("seed", "message"),
    [
        (None, "no exact seed"),
        (np.asarray(1.0), "exactly one element"),
        (np.asarray([1.0, 2.0]), "exactly one element"),
        (np.asarray([np.nan]), "finite"),
        (np.asarray([0.0]), "positive integer"),
        (np.asarray([-1.0]), "positive integer"),
        (np.asarray([1.5]), "positive integer"),
        (np.asarray([b"2"]), "numeric"),
    ],
)
def test_seed_attribute_is_strict(seed: object, message: str) -> None:
    class Group:
        attrs = {} if seed is None else {"seed": seed}

    with pytest.raises(ValueError, match=message):
        data._read_rng_seed(Group(), group_path="synthetic/run")
    assert data._read_rng_seed(
        type("ValidGroup", (), {"attrs": {"seed": np.asarray([347639.0])}})(),
        group_path="synthetic/run",
    ) == 347639


def test_only_completed_location_is_accepted(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with h5py.File(source, "w") as handle:
        _add_run(
            handle,
            idv=9,
            magnitude=50,
            run=1,
            seed=np.asarray([2001.0]),
            status="SimulationStopped",
        )
    selection = data.authorize_request(
        partition=data.DEVELOPMENT, idv=9, simulation_run=1
    )
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="only in SimulationCompleted"):
            data.locate_unique_run(handle, selection)


def test_development_group_is_balanced_block_with_distinct_seeds() -> None:
    conditions = tuple(
        _dummy_run(block=25, idv=idv, seed=5000 + index)
        for index, idv in enumerate(data.CONDITION_IDVS)
    )
    group = data.DevelopmentGroup(25, conditions)
    assert data.DevelopmentBlock is data.DevelopmentGroup
    assert group.stratified_block == 25
    assert group.by_idv(9).source_idv == 9
    assert {run.group_key for run in group.conditions} == {25}
    assert len({run.rng_seed for run in group.conditions}) == 4

    duplicate = list(conditions)
    duplicate[-1] = _dummy_run(block=25, idv=15, seed=5000)
    with pytest.raises(ValueError, match="distinct rng_seed"):
        data.DevelopmentGroup(25, tuple(duplicate))


def _fake_topology(
    *, duplicate_seed: bool = False,
) -> tuple[dict[str, object], type[Any], list[int]]:
    numeric_reads = [0]

    class FakeGroup:
        def __init__(self, seed: int) -> None:
            self.attrs = {"seed": np.asarray([float(seed)])}

    class FakeDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            if kind == "processdata":
                self.shape = (2000, 54)
                self.dtype = np.dtype(np.float64)
            elif kind == "idv_init":
                self.shape = (3, 28)
                self.dtype = np.dtype(np.float64)
            else:
                self.shape = (54,)
                self.dtype = np.dtype(object)

        def __getitem__(self, key: object) -> np.ndarray:
            if self.kind != "labels":
                numeric_reads[0] += 1
                raise AssertionError("topology validation decoded a run dataset")
            return np.asarray(data.PROCESSDATA_LABELS, dtype=object)

    class FakeH5py:
        Group = FakeGroup
        Dataset = FakeDataset

    objects: dict[str, object] = {"Processdata_Labels": FakeDataset("labels")}
    selections = data.iter_authorized_selections()
    for index, selection in enumerate(selections):
        seed = index + 1
        if duplicate_seed and index == len(selections) - 1:
            seed = 1
        group_path = data._base_path(
            "SimulationCompleted",
            selection.idv,
            selection.fault_magnitude_percent,
            selection.simulation_run,
        )
        objects[group_path] = FakeGroup(seed)
        objects[f"{group_path}/processdata"] = FakeDataset("processdata")
        objects[f"{group_path}/idv_init"] = FakeDataset("idv_init")
    return objects, FakeH5py, numeric_reads


def test_topology_validates_900_completed_globally_unique_seeds_without_run_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, fake_h5py, numeric_reads = _fake_topology()
    monkeypatch.setattr(data, "_require_h5py", lambda: fake_h5py)
    summary = data.validate_archive_topology(objects)
    assert summary.selected_run_count == 900
    assert dict(summary.counts_by_partition) == {
        data.DEVELOPMENT: 400,
        data.CALIBRATION: 100,
        data.FINAL: 400,
    }
    assert dict(summary.counts_by_status) == {
        "SimulationCompleted": 900,
        "SimulationStopped": 0,
    }
    assert dict(summary.counts_by_magnitude) == {25: 100, 50: 400, 75: 400}
    assert summary.unique_seed_count == 900
    assert summary.seed_sets_pairwise_disjoint is True
    assert numeric_reads == [0]


def test_topology_rejects_any_allocated_seed_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, fake_h5py, numeric_reads = _fake_topology(duplicate_seed=True)
    monkeypatch.setattr(data, "_require_h5py", lambda: fake_h5py)
    with pytest.raises(ValueError, match="globally unique"):
        data.validate_archive_topology(objects)
    assert numeric_reads == [0]


def test_numeric_access_is_exactly_idv_then_raw_equal_normalized_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[tuple[str, object]] = []

    class SpyDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.shape = (3, 28) if kind == "idv_init" else (2000, 54)
            self.dtype = np.dtype(np.float64)

        def __getitem__(self, key: object) -> np.ndarray:
            accesses.append((self.kind, key))
            if self.kind == "idv_init":
                return _idv_init(15, 75)
            if isinstance(key, tuple) and key[1] == 0:
                return (np.arange(440, 940, dtype=np.float64) + 1.0) * 0.05
            return np.ones((500, 52), dtype=np.float64)

    located = data.LocatedRun(
        status="SimulationCompleted",
        idv=15,
        simulation_run=1,
        fault_magnitude_percent=75,
        rng_seed=9001,
        group_path="synthetic/group",
        processdata_path="synthetic/group/processdata",
        idv_init_path="synthetic/group/idv_init",
    )
    processdata = SpyDataset("processdata")
    idv_init = SpyDataset("idv_init")
    monkeypatch.setattr(data, "locate_unique_run", lambda *args, **kwargs: located)
    run = data._materialize_run(
        {located.processdata_path: processdata, located.idv_init_path: idv_init},
        partition=data.FINAL,
        idv=15,
        simulation_run=1,
    )
    assert accesses == [
        ("idv_init", Ellipsis),
        ("processdata", (slice(440, 940), slice(1, 53))),
        ("processdata", (slice(440, 940), 0)),
    ]
    assert run.x.shape == (500, 52)
    assert run.fault_magnitude_percent == 75
    assert run.rng_seed == 9001


def test_exact_lower_magnitude_dataset_shapes_and_dtypes(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = _raw()[:-1]
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=3,
            magnitude=50,
            run=1,
            seed=np.asarray([3001.0]),
            processdata=raw,
        )
    with pytest.raises(ValueError, match=r"expected exact shape \(2000, 54\)"):
        data.PCLRMode1Archive(source).load_run(
            partition=data.DEVELOPMENT, idv=3, simulation_run=1
        )


def test_archive_requires_exact_official_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single official file TEP_Mode1.h5"):
        data.PCLRMode1Archive(tmp_path / "copy.h5")
    assert data.PCLRMode1Archive(_source(tmp_path)).path.name == data.MODE1_FILE_NAME

