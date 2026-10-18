from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import mate_spc_pclr.data as data  # noqa: E402


def _raw(*, offset: float = 0.0) -> np.ndarray:
    values = np.arange(2001 * 54, dtype=np.float64).reshape(2001, 54)
    values[:, 1:] += offset
    values[:, 0] = np.arange(2001, dtype=np.float64) * data.SAMPLE_PERIOD_HOURS
    return values


def _idv_init(idv: int) -> np.ndarray:
    values = np.zeros((3, 28), dtype=np.float64)
    values[:, idv - 1] = (0.0, 1.0, 30.0)
    return values


def _add_labels(handle: h5py.File, labels: tuple[str, ...] = data.PROCESSDATA_LABELS) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "Processdata_Labels", data=np.asarray(labels, dtype=object), dtype=string_type
    )


def _add_run(
    handle: h5py.File,
    *,
    idv: int,
    run: int,
    status: str = "SimulationCompleted",
    processdata: np.ndarray | None = None,
    idv_init: np.ndarray | None = None,
) -> str:
    group_path = data._base_path(status, idv, run)
    group = handle.require_group(group_path)
    group.create_dataset("processdata", data=_raw() if processdata is None else processdata)
    group.create_dataset("idv_init", data=_idv_init(idv) if idv_init is None else idv_init)
    return group_path


def _source(tmp_path: Path) -> Path:
    return tmp_path / data.MODE1_FILE_NAME


def _dummy_run(*, simulation_run: int, idv: int, partition: str = data.DEVELOPMENT) -> data.PCLRRun:
    x = np.zeros((500, 52), dtype=np.float64)
    x.setflags(write=False)
    fault = np.zeros(500, dtype=np.bool_)
    if idv != 1:
        fault[160:] = True
    fault.setflags(write=False)
    return data.PCLRRun(
        name=f"dummy-{simulation_run}-{idv}",
        x=x,
        fault=fault,
        feature_names=data.FEATURE_NAMES,
        metadata=MappingProxyType(
            {
                "partition": partition,
                "source_idv": idv,
                "simulation_run": simulation_run,
                "group_key": simulation_run,
            }
        ),
    )


def test_allocation_is_exact_disjoint_and_and_embargoes_old_runs() -> None:
    assert data.DEVELOPMENT_RUNS == tuple(range(201, 301))
    assert data.CALIBRATION_RUNS == tuple(range(301, 401))
    assert data.FINAL_RUNS == tuple(range(401, 501))
    assert not (set(data.DEVELOPMENT_RUNS) & set(data.CALIBRATION_RUNS))
    assert not (set(data.CALIBRATION_RUNS) & set(data.FINAL_RUNS))

    selections = data.iter_authorized_selections()
    assert len(selections) == 900
    assert sum(item.partition == data.DEVELOPMENT for item in selections) == 400
    assert sum(item.partition == data.CALIBRATION for item in selections) == 100
    assert sum(item.partition == data.FINAL for item in selections) == 400
    assert all(item.simulation_run > 200 for item in selections)
    assert {
        item.idv for item in selections if item.partition == data.CALIBRATION
    } == {1}


@pytest.mark.parametrize("run", [-1, 0, 1, 100, 200])
def test_runs_at_or_below_200_are_permanently_rejected(run: int) -> None:
    with pytest.raises(ValueError, match="permanently embargoed"):
        data.authorize_request(partition=data.DEVELOPMENT, idv=1, simulation_run=run)


@pytest.mark.parametrize("partition", data.PARTITIONS)
@pytest.mark.parametrize("idv", data.FAULT_IDVS)
def test_calibration_fault_rows_301_to_400_are_explicitly_forbidden(
    partition: str, idv: int
) -> None:
    with pytest.raises(ValueError, match="fault runs 301..400 are forbidden"):
        data.authorize_request(partition=partition, idv=idv, simulation_run=350)


@pytest.mark.parametrize(
    ("partition", "idv", "run"),
    [
        (data.DEVELOPMENT, 1, 301),
        (data.DEVELOPMENT, 2, 250),
        (data.CALIBRATION, 1, 401),
        (data.FINAL, 1, 400),
        (data.FINAL, 2, 450),
        ("audit", 1, 450),
    ],
)
def test_other_cross_partition_or_wrong_idv_requests_fail_closed(
    partition: str, idv: int, run: int
) -> None:
    with pytest.raises(ValueError, match="not authorized|unknown PCLR partition"):
        data.authorize_request(partition=partition, idv=idv, simulation_run=run)


def test_authorization_precedes_even_opening_the_hdf5_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = data.PCLRMode1Archive(_source(tmp_path))
    opened = False

    def forbidden_open(self: data.PCLRMode1Archive):  # type: ignore[no-untyped-def]
        nonlocal opened
        opened = True
        raise AssertionError("unauthorized request opened the archive")

    monkeypatch.setattr(data.PCLRMode1Archive, "_open", forbidden_open)
    with pytest.raises(ValueError, match="permanently embargoed"):
        archive.load_run(partition=data.DEVELOPMENT, idv=3, simulation_run=200)
    assert opened is False


def test_archive_requires_one_exact_mode1_file_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single official file TEP_Mode1.h5"):
        data.PCLRMode1Archive(tmp_path)
    with pytest.raises(ValueError, match="single official file TEP_Mode1.h5"):
        data.PCLRMode1Archive(tmp_path / "copy.h5")
    archive = data.PCLRMode1Archive(_source(tmp_path))
    assert archive.path.name == "TEP_Mode1.h5"


def test_healthy_and_fault_windows_and_segments_are_exact(tmp_path: Path) -> None:
    source = _source(tmp_path)
    healthy_raw = _raw(offset=10.0)
    fault_raw = _raw(offset=30.0)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=1, run=201, processdata=healthy_raw)
        _add_run(handle, idv=3, run=201, processdata=fault_raw)

    archive = data.PCLRMode1Archive(source)
    healthy = archive.load_run(partition=data.DEVELOPMENT, idv=1, simulation_run=201)
    fault = archive.load_run(partition=data.DEVELOPMENT, idv=3, simulation_run=201)

    np.testing.assert_array_equal(healthy.x, healthy_raw[101:601, 1:53])
    np.testing.assert_array_equal(fault.x, fault_raw[441:941, 1:53])
    assert healthy.x.shape == fault.x.shape == (500, 52)
    assert healthy.x.dtype == fault.x.dtype == np.float64
    assert healthy.x.flags.writeable is False
    assert not healthy.fault.any()
    assert not fault.fault[:160].any()
    assert fault.fault[160:].all()
    assert healthy.metadata["normalized_window_half_open"] == (100, 600)
    assert healthy.metadata["pseudo_onset_zero_based"] == 160
    assert fault.metadata["normalized_window_half_open"] == (440, 940)
    assert fault.metadata["fault_start_zero_based"] == 160
    assert fault.segment("baseline").shape == (128, 52)
    assert fault.segment("warmup").shape == (32, 52)
    assert fault.segment("monitor").shape == (340, 52)
    assert fault.segment("monitor").flags.writeable is False


def test_development_group_requires_four_conditions_with_same_run_key() -> None:
    conditions = tuple(_dummy_run(simulation_run=225, idv=idv) for idv in data.CONDITION_IDVS)
    group = data.DevelopmentGroup(225, conditions)
    assert group.by_idv(9).source_idv == 9
    assert {run.group_key for run in group.conditions} == {225}

    mismatched = list(conditions)
    mismatched[-1] = _dummy_run(simulation_run=226, idv=15)
    with pytest.raises(ValueError, match="share the simulation_run group key"):
        data.DevelopmentGroup(225, tuple(mismatched))
    with pytest.raises(ValueError, match="exactly once in that order"):
        data.DevelopmentGroup(225, tuple(reversed(conditions)))


def test_labels_require_the_exact_official_54_item_sequence(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
    with h5py.File(source, "r") as handle:
        labels = data.validate_processdata_labels(handle)
    assert labels == data.PROCESSDATA_LABELS
    assert len(data.FEATURE_NAMES) == 52
    assert labels.count("A Feed") == 2

    invalid = list(data.PROCESSDATA_LABELS)
    invalid[42] = "D Feed"
    with h5py.File(source, "w") as handle:
        _add_labels(handle, tuple(invalid))
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="index 42 expected 'D feed'"):
            data.validate_processdata_labels(handle)


def test_duplicate_completion_status_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=9, run=250, status="SimulationCompleted")
        _add_run(handle, idv=9, run=250, status="SimulationStopped")
    with pytest.raises(ValueError, match="exactly one completion status; found 2"):
        data.PCLRMode1Archive(source).load_run(
            partition=data.DEVELOPMENT, idv=9, simulation_run=250
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("process_shape", r"expected exact shape \(2001, 54\)"),
        ("process_dtype", "processdata must be exact float64"),
        ("idv_shape", r"expected exact shape \(3, 28\)"),
        ("idv_dtype", "idv_init must be exact float64"),
        ("idv_other_fault", "must declare only IDV15"),
        ("time_step", "time must advance exactly every three minutes"),
        ("nonfinite", "authorized values must be finite float64"),
    ],
)
def test_shape_dtype_time_one_hot_and_finiteness_are_strict(
    tmp_path: Path, kind: str, message: str
) -> None:
    source = _source(tmp_path)
    raw = _raw()
    init = _idv_init(15)
    if kind == "process_shape":
        raw = raw[:-1]
    elif kind == "process_dtype":
        raw = raw.astype(np.float32)
    elif kind == "idv_shape":
        init = init[:2]
    elif kind == "idv_dtype":
        init = init.astype(np.float32)
    elif kind == "idv_other_fault":
        init[1, 8] = 1.0
    elif kind == "time_step":
        raw[700, 0] += 0.01
    elif kind == "nonfinite":
        raw[700, 10] = np.nan
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=15, run=250, processdata=raw, idv_init=init)
    with pytest.raises(ValueError, match=message):
        data.PCLRMode1Archive(source).load_run(
            partition=data.DEVELOPMENT, idv=15, simulation_run=250
        )


def test_numeric_access_is_idv_then_only_the_authorized_processdata_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[tuple[str, object]] = []

    class SpyDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.shape = (3, 28) if kind == "idv_init" else (2001, 54)
            self.dtype = np.dtype(np.float64)

        def __getitem__(self, key: object) -> np.ndarray:
            accesses.append((self.kind, key))
            if self.kind == "idv_init":
                return _idv_init(15)
            if isinstance(key, tuple) and key[1] == 0:
                return np.arange(500, dtype=np.float64) * data.SAMPLE_PERIOD_HOURS
            return np.ones((500, 52), dtype=np.float64)

    located = data.LocatedRun(
        "SimulationCompleted",
        15,
        250,
        "synthetic/group",
        "synthetic/group/processdata",
        "synthetic/group/idv_init",
    )
    processdata = SpyDataset("processdata")
    idv_init = SpyDataset("idv_init")
    monkeypatch.setattr(data, "locate_unique_run", lambda *args, **kwargs: located)
    run = data._materialize_run(
        {located.processdata_path: processdata, located.idv_init_path: idv_init},
        partition=data.DEVELOPMENT,
        idv=15,
        simulation_run=250,
    )
    assert accesses == [
        ("idv_init", Ellipsis),
        ("processdata", (slice(441, 941), slice(1, 53))),
        ("processdata", (slice(441, 941), 0)),
    ]
    assert run.x.shape == (500, 52)


def test_unauthorized_materialization_touches_no_handle() -> None:
    class ExplodingHandle:
        def __contains__(self, key: object) -> bool:
            raise AssertionError("unauthorized request inspected HDF5")

        def __getitem__(self, key: object) -> object:
            raise AssertionError("unauthorized request read HDF5")

    with pytest.raises(ValueError, match="permanently embargoed"):
        data._materialize_run(
            ExplodingHandle(), partition=data.DEVELOPMENT, idv=1, simulation_run=199
        )


def test_topology_validation_never_decodes_numeric_run_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numeric_reads = 0

    class FakeGroup:
        pass

    class FakeDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            if kind == "processdata":
                self.shape = (2001, 54)
                self.dtype = np.dtype(np.float64)
            elif kind == "idv_init":
                self.shape = (3, 28)
                self.dtype = np.dtype(np.float64)
            else:
                self.shape = (54,)
                self.dtype = np.dtype(object)

        def __getitem__(self, key: object) -> np.ndarray:
            nonlocal numeric_reads
            if self.kind != "labels":
                numeric_reads += 1
                raise AssertionError("metadata-only validation decoded numeric values")
            return np.asarray(data.PROCESSDATA_LABELS, dtype=object)

    class FakeH5py:
        Group = FakeGroup
        Dataset = FakeDataset

    objects: dict[str, object] = {"Processdata_Labels": FakeDataset("labels")}
    for selection in data.iter_authorized_selections():
        group_path = data._base_path(
            "SimulationCompleted", selection.idv, selection.simulation_run
        )
        objects[group_path] = FakeGroup()
        objects[f"{group_path}/processdata"] = FakeDataset("processdata")
        objects[f"{group_path}/idv_init"] = FakeDataset("idv_init")
    monkeypatch.setattr(data, "_require_h5py", lambda: FakeH5py)

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
    assert numeric_reads == 0


def test_pclr_loader_has_no_dependency_on_frozen_external_loader() -> None:
    assert "mate_spc_external" not in Path(data.__file__).read_text(encoding="utf-8")
