from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

import torch  # noqa: F401

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import mate_spc_cpetc.data_v5 as data  # noqa: E402


def _raw(*, offset: float = 0.0) -> np.ndarray:
    values = np.zeros((2000, 54), dtype=np.float64)
    values[:, 0] = (np.arange(2000, dtype=np.float64) + 1.0) * data.SAMPLE_PERIOD_HOURS
    values[:, 1:] = np.arange(53, dtype=np.float64)[None, :] + offset
    return values


def _idv_init(idv: int) -> np.ndarray:
    values = np.zeros((3, 28), dtype=np.float64)
    values[:, idv - 1] = (0.0, 0.5, 30.0)
    return values


def _add_labels(handle: h5py.File, labels: tuple[str, ...] = data.PROCESSDATA_LABELS) -> None:
    string_type = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(
        "Processdata_Labels",
        data=np.asarray(labels, dtype=object),
        dtype=string_type,
    )


def _add_run(
    handle: h5py.File,
    *,
    idv: int,
    run: int,
    seed: object = np.asarray([12345.0]),
    status: str = "SimulationCompleted",
    processdata: np.ndarray | None = None,
    idv_init: np.ndarray | None = None,
) -> str:
    path = data._base_path(status, idv, run)
    group = handle.require_group(path)
    if seed is not None:
        group.attrs["seed"] = seed
    group.create_dataset("processdata", data=_raw() if processdata is None else processdata)
    group.create_dataset("idv_init", data=_idv_init(idv) if idv_init is None else idv_init)
    return path


def _source(tmp_path: Path) -> Path:
    return tmp_path / data.MODE2_FILE_NAME


def _archive(source: Path) -> data.Mode2DevelopmentArchive:
    return data.Mode2DevelopmentArchive._for_synthetic_test(source)


def _dummy_run(*, block: int, idv: int, seed: int) -> data.CPETCDevelopmentRun:
    x = np.zeros((500, 52), dtype=np.float64)
    x.setflags(write=False)
    fault = np.zeros(500, dtype=np.bool_)
    if idv != 1:
        fault[160:] = True
    fault.setflags(write=False)
    return data.CPETCDevelopmentRun(
        name=f"dummy-{block}-{idv}",
        x=x,
        fault=fault,
        feature_names=data.FEATURE_NAMES,
        metadata=MappingProxyType(
            {
                "partition": data.DEVELOPMENT_PARTITION,
                "source_idv": idv,
                "simulation_run": block,
                "group_key": block,
                "rng_seed": seed,
                "fault_id": 0 if idv == 1 else idv,
            }
        ),
    )


def test_frozen_allocation_path_digest_and_private_test_bypass() -> None:
    allocation = data.MODE2_DEVELOPMENT_ALLOCATION
    assert allocation.source_mode == 2
    assert allocation.condition_idvs == (1, 3, 9, 15)
    assert allocation.fault_magnitude_percent == 50
    assert allocation.simulation_runs == tuple(range(1, 101))
    assert allocation.healthy_window == (100, 600)
    assert allocation.fault_window == (440, 940)
    assert data.MODE2_RELATIVE_PATH.as_posix() == (
        "data/extended_tep_dtu_mode2/TEP_Mode2.h5"
    )
    assert data.OFFICIAL_MODE2_PATH == (data.PROJECT_ROOT / data.MODE2_RELATIVE_PATH).resolve()
    assert data.MODE2_SHA256 == (
        "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
    )
    assert data.PROTOCOL_ID == "cp-etc-scan-v5.1-mode2-development"
    assert data.PROTOCOL_SHA256 == (
        "55cd6dcb09409f53db4f078f2441f18a7659c1ce97b7165876d3f2a891a37923"
    )
    assert tuple(inspect.signature(data.Mode2DevelopmentArchive).parameters) == ()
    assert "_for_synthetic_test" not in data.__all__


@pytest.mark.parametrize(
    ("idv", "run", "message"),
    [
        (2, 1, "IDV1/3/9/15"),
        (3, 0, "Run1..100"),
        (9, 101, "Run1..100"),
        (True, 1, "IDV must be an integer"),
    ],
)
def test_run_authorizer_rejects_every_overreach(idv: object, run: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        data.authorize_request(idv=idv, simulation_run=run)  # type: ignore[arg-type]


def test_forged_allocation_is_rejected_before_hdf5_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(_source(tmp_path))
    opened = False

    @contextmanager
    def forbidden_open(self: data.Mode2DevelopmentArchive) -> Iterator[object]:
        nonlocal opened
        opened = True
        raise AssertionError("forged allocation opened HDF5")
        yield object()  # pragma: no cover

    monkeypatch.setattr(data.Mode2DevelopmentArchive, "_open", forbidden_open)
    forged = replace(data.MODE2_DEVELOPMENT_ALLOCATION, fault_magnitude_percent=75)
    with pytest.raises(ValueError, match="only Mode2/magnitude50"):
        archive.load_development_blocks(forged)
    assert opened is False


def test_production_source_hash_is_checked_before_hdf5_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    source.write_bytes(b"not-the-official-archive")
    monkeypatch.setattr(data, "OFFICIAL_MODE2_PATH", source.resolve())
    archive = data.Mode2DevelopmentArchive()
    opened = False

    class ForbiddenH5py:
        @staticmethod
        def File(*args: object, **kwargs: object) -> object:
            nonlocal opened
            opened = True
            raise AssertionError("digest mismatch reached h5py.File")

    monkeypatch.setattr(data, "_require_h5py", lambda: ForbiddenH5py)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        archive.validate_topology()
    assert opened is False


def test_exact_mode2_paths_and_completed_only_status(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with h5py.File(source, "w") as handle:
        stopped_path = _add_run(
            handle,
            idv=9,
            run=7,
            seed=np.asarray([7009.0]),
            status="SimulationStopped",
        )
    assert stopped_path == (
        "Mode2/SingleFault/SimulationStopped/IDV9/Mode2_IDVInfo_9_50/Run7"
    )
    selection = data.authorize_request(idv=9, simulation_run=7)
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="only in SimulationCompleted"):
            data.locate_unique_run(handle, selection)

    source.unlink()
    with h5py.File(source, "w") as handle:
        _add_run(handle, idv=9, run=7, seed=np.asarray([7009.0]))
        _add_run(
            handle,
            idv=9,
            run=7,
            seed=np.asarray([7010.0]),
            status="SimulationStopped",
        )
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="only in SimulationCompleted"):
            data.locate_unique_run(handle, selection)


def _run_materialization(
    source: Path, *, idv: int, run: int
) -> data.CPETCDevelopmentRun:
    selection = data.authorize_request(idv=idv, simulation_run=run)
    with h5py.File(source, "r") as handle:
        data.validate_processdata_labels(handle)
        return data._materialize_run(handle, selection)


def test_windows_time_fault_mask_readonly_and_cross_mode_metadata(tmp_path: Path) -> None:
    source = _source(tmp_path)
    healthy_raw = _raw(offset=10.0)
    fault_raw = _raw(offset=30.0)
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=1,
            run=1,
            seed=np.asarray([1001.0]),
            processdata=healthy_raw,
        )
        _add_run(
            handle,
            idv=15,
            run=1,
            seed=np.asarray([1015.0]),
            processdata=fault_raw,
        )
    healthy = _run_materialization(source, idv=1, run=1)
    fault = _run_materialization(source, idv=15, run=1)

    np.testing.assert_array_equal(healthy.x, healthy_raw[100:600, 1:53])
    np.testing.assert_array_equal(fault.x, fault_raw[440:940, 1:53])
    assert healthy.x.shape == fault.x.shape == (500, 52)
    assert healthy.x.flags.writeable is fault.x.flags.writeable is False
    assert healthy.fault.flags.writeable is fault.fault.flags.writeable is False
    assert not healthy.fault.any()
    assert not fault.fault[: data.LOCAL_FAULT_START].any()
    assert fault.fault[data.LOCAL_FAULT_START :].all()
    assert healthy.metadata["source_time_hours_first_last"] == (5.05, 30.0)
    assert fault.metadata["source_time_hours_first_last"] == (22.05, 47.0)
    assert fault.metadata["fault_start_zero_based"] == 160
    assert fault.metadata["normalized_window_half_open"] == (440, 940)
    assert healthy.metadata["normalized_window_half_open"] == (100, 600)
    assert fault.metadata["source_operating_mode"] == 2
    assert fault.metadata["intended_target_operating_mode"] == 1
    assert fault.metadata["data_role"] == "cross_operating_mode_development_only"
    assert fault.metadata["not_formal_calibration_or_final_confirmation"] is True
    assert fault.metadata["grouping_semantics"] == (
        "balanced_stratification_block_only_not_paired_seed"
    )
    assert fault.metadata["rng_seed"] == 1015
    assert fault.segment("baseline").shape == (128, 52)
    assert fault.segment("warmup").shape == (32, 52)
    assert fault.segment("monitor").shape == (340, 52)
    with pytest.raises(ValueError):
        fault.x[0, 0] = 999.0
    with pytest.raises(ValueError):
        fault.fault[0] = True
    with pytest.raises(TypeError):
        fault.metadata["rng_seed"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("processdata", "idv_init", "message"),
    [
        (np.zeros((1999, 54), dtype=np.float64), _idv_init(3), r"shape \(2000, 54\)"),
        (np.zeros((2000, 53), dtype=np.float64), _idv_init(3), r"shape \(2000, 54\)"),
        (_raw().astype(np.float32), _idv_init(3), "processdata must be exact float64"),
        (_raw(), np.zeros((2, 28), dtype=np.float64), r"shape \(3, 28\)"),
        (_raw(), _idv_init(3).astype(np.float32), "idv_init must be exact float64"),
    ],
)
def test_processdata_and_idv_init_shape_dtype_fail_closed(
    tmp_path: Path,
    processdata: np.ndarray,
    idv_init: np.ndarray,
    message: str,
) -> None:
    source = _source(tmp_path)
    with h5py.File(source, "w") as handle:
        path = _add_run(
            handle,
            idv=3,
            run=1,
            seed=np.asarray([3001.0]),
            processdata=processdata,
            idv_init=idv_init,
        )
    selection = data.authorize_request(idv=3, simulation_run=1)
    with h5py.File(source, "r") as handle:
        located = data.locate_unique_run(handle, selection)
        assert located.group_path == path
        with pytest.raises(ValueError, match=message):
            data._validate_dataset_metadata(handle, located)


def test_labels_order_is_exact(tmp_path: Path) -> None:
    source = _source(tmp_path)
    changed = list(data.PROCESSDATA_LABELS)
    changed[7], changed[8] = changed[8], changed[7]
    with h5py.File(source, "w") as handle:
        _add_labels(handle, tuple(changed))
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="official sequence"):
            data.validate_processdata_labels(handle)


def test_idv_init_is_authenticated_before_processdata_values() -> None:
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
            invalid = _idv_init(15)
            invalid[1, 14] = 0.75
            return invalid

    selection = data.authorize_request(idv=15, simulation_run=1)
    located = data.LocatedRun(
        status="SimulationCompleted",
        idv=15,
        simulation_run=1,
        rng_seed=15001,
        group_path="synthetic/group",
        processdata_path="synthetic/group/processdata",
        idv_init_path="synthetic/group/idv_init",
    )
    handle = {
        located.processdata_path: SpyDataset("processdata"),
        located.idv_init_path: SpyDataset("idv_init"),
    }
    with pytest.raises(ValueError, match=r"\[0, 0.5, 30\]"):
        data._materialize_run(handle, selection, located=located)
    assert accesses == [("idv_init", Ellipsis)]


def test_shifted_or_nonfinite_authorized_time_fails(tmp_path: Path) -> None:
    source = _source(tmp_path)
    shifted = _raw()
    shifted[:, 0] -= 0.05
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=1, run=1, seed=np.asarray([111.0]), processdata=shifted)
    with pytest.raises(ValueError, match="absolute time"):
        _run_materialization(source, idv=1, run=1)

    source.unlink()
    invalid = _raw()
    invalid[150, 0] = np.nan
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=1, run=1, seed=np.asarray([112.0]), processdata=invalid)
    with pytest.raises(ValueError, match="complete time column must be finite"):
        _run_materialization(source, idv=1, run=1)


@pytest.mark.parametrize(
    ("row", "replacement"),
    [
        (0, 0.0),
        (1000, 50.5),
        (1999, 99.95),
    ],
)
def test_complete_time_axis_rejects_start_step_or_end_change(
    tmp_path: Path, row: int, replacement: float
) -> None:
    source = _source(tmp_path)
    raw = _raw()
    raw[row, 0] = replacement
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=3, run=1, seed=np.asarray([301.0]), processdata=raw)
    with pytest.raises(ValueError, match="complete Mode 2 time column"):
        _run_materialization(source, idv=3, run=1)


def test_time_change_outside_authorized_500_rows_is_still_rejected(tmp_path: Path) -> None:
    source = _source(tmp_path)
    raw = _raw()
    raw[1000, 0] += 0.01  # IDV1 authorizes only raw rows 100:600.
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(handle, idv=1, run=1, seed=np.asarray([101.0]), processdata=raw)
    with pytest.raises(ValueError, match="complete Mode 2 time column"):
        _run_materialization(source, idv=1, run=1)


@pytest.mark.parametrize(
    ("delta", "accepted"),
    [(0.5e-12, True), (2.0e-12, False)],
)
def test_idv_init_uses_full_matrix_absolute_tolerance(
    tmp_path: Path, delta: float, accepted: bool
) -> None:
    source = _source(tmp_path)
    init = _idv_init(3)
    init[1, 2] += delta
    with h5py.File(source, "w") as handle:
        _add_labels(handle)
        _add_run(
            handle,
            idv=3,
            run=1,
            seed=np.asarray([302.0]),
            idv_init=init,
        )
    if accepted:
        run = _run_materialization(source, idv=3, run=1)
        assert run.source_idv == 3
    else:
        with pytest.raises(ValueError, match=r"\[0, 0.5, 30\]"):
            _run_materialization(source, idv=3, run=1)


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
def test_seed_is_exactly_one_positive_integer(seed: object, message: str) -> None:
    class Group:
        attrs = {} if seed is None else {"seed": seed}

    with pytest.raises(ValueError, match=message):
        data._read_rng_seed(Group(), group_path="synthetic/run")
    valid = type("ValidGroup", (), {"attrs": {"seed": np.asarray([9876.0])}})()
    assert data._read_rng_seed(valid, group_path="synthetic/run") == 9876


def _fake_topology(
    *, wrong_seed_contract: bool = False
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
    duplicate_owner_seed = {
        owner: seed
        for seed, owners in data.ALLOWED_DUPLICATE_SEED_OWNERS.items()
        for owner in owners
    }
    for index, selection in enumerate(selections):
        owner = (selection.simulation_run, selection.idv)
        seed = duplicate_owner_seed.get(owner, 2_000_000 + index)
        if wrong_seed_contract and owner == (14, 1):
            seed = 9_999_999
        path = data._base_path("SimulationCompleted", selection.idv, selection.simulation_run)
        objects[path] = FakeGroup(seed)
        objects[f"{path}/processdata"] = FakeDataset("processdata")
        objects[f"{path}/idv_init"] = FakeDataset("idv_init")
    return objects, FakeH5py, numeric_reads


def test_topology_is_metadata_only_and_matches_exact_v5_1_seed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, fake_h5py, numeric_reads = _fake_topology()
    monkeypatch.setattr(data, "_require_h5py", lambda: fake_h5py)
    summary = data.validate_archive_topology(objects)
    assert summary.selected_run_count == 400
    assert summary.unique_seed_count == 395
    assert summary.duplicate_seed_value_count == 5
    assert dict(summary.duplicate_seed_owners) == dict(data.ALLOWED_DUPLICATE_SEED_OWNERS)
    assert summary.cross_fold_seed_collision_count == 0
    assert dict(summary.counts_by_status) == {
        "SimulationCompleted": 400,
        "SimulationStopped": 0,
    }
    assert dict(summary.counts_by_idv) == {1: 100, 3: 100, 9: 100, 15: 100}
    assert summary.source_mode == 2
    assert summary.fault_magnitude_percent == 50
    assert summary.data_role == "cross_operating_mode_development_only"
    assert numeric_reads == [0]


def test_topology_rejects_any_change_to_frozen_seed_owners_without_numeric_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects, fake_h5py, numeric_reads = _fake_topology(wrong_seed_contract=True)
    monkeypatch.setattr(data, "_require_h5py", lambda: fake_h5py)
    with pytest.raises(ValueError, match="exactly match"):
        data.validate_archive_topology(objects)
    assert numeric_reads == [0]


def test_block_semantics_are_balanced_not_seed_paired() -> None:
    conditions = tuple(
        _dummy_run(block=25, idv=idv, seed=5000 + index)
        for index, idv in enumerate(data.CONDITION_IDVS)
    )
    block = data.DevelopmentBlock(25, conditions)
    assert data.DevelopmentGroup is data.DevelopmentBlock
    assert block.stratified_block == 25
    assert block.fold_id == data.fold_for_run(25) == 4
    assert block.by_idv(9).source_idv == 9
    assert {item.group_key for item in block.conditions} == {25}
    assert len({item.rng_seed for item in block.conditions}) == 4

    duplicate = list(conditions)
    duplicate[-1] = _dummy_run(block=25, idv=15, seed=5000)
    with pytest.raises(ValueError, match="distinct rng_seed"):
        data.DevelopmentBlock(25, tuple(duplicate))


def test_frozen_folds_keep_every_seed_component_together() -> None:
    assert {fold: len(runs) for fold, runs in data.FOLD_RUN_IDS.items()} == {
        0: 20,
        1: 20,
        2: 20,
        3: 20,
        4: 20,
    }
    assert tuple(sorted(data.RUN_TO_FOLD)) == tuple(range(1, 101))
    assert {data.fold_for_run(run) for run in (14, 43, 98)} == {0}
    assert {data.fold_for_run(run) for run in (6, 76)} == {1}
    assert {data.fold_for_run(run) for run in (9, 88)} == {2}
    assert {data.fold_for_run(run) for run in (24, 33)} == {3}
    for owners in data.ALLOWED_DUPLICATE_SEED_OWNERS.values():
        assert len({data.fold_for_run(run) for run, _ in owners}) == 1


def _full_virtual_archive(tmp_path: Path) -> Path:
    """Build 400 synthetic paths backed by one small virtual source dataset."""

    source = _source(tmp_path)
    raw = _raw(offset=42.0)
    with h5py.File(source, "w", libver="latest") as handle:
        _add_labels(handle)
        handle.create_dataset("_synthetic_processdata_template", data=raw)
        virtual_source = h5py.VirtualSource(
            str(source.resolve()), "_synthetic_processdata_template", shape=raw.shape
        )
        duplicate_owner_seed = {
            owner: seed
            for seed, owners in data.ALLOWED_DUPLICATE_SEED_OWNERS.items()
            for owner in owners
        }
        for index, selection in enumerate(data.iter_authorized_selections(), start=1):
            path = data._base_path(
                "SimulationCompleted", selection.idv, selection.simulation_run
            )
            group = handle.require_group(path)
            owner = (selection.simulation_run, selection.idv)
            seed = duplicate_owner_seed.get(owner, 2_000_000 + index)
            group.attrs["seed"] = np.asarray([float(seed)])
            layout = h5py.VirtualLayout(shape=raw.shape, dtype=np.float64)
            layout[:, :] = virtual_source
            group.create_virtual_dataset("processdata", layout)
            group.create_dataset("idv_init", data=_idv_init(selection.idv))
    return source


def test_load_development_blocks_end_to_end_on_synthetic_h5(tmp_path: Path) -> None:
    source = _full_virtual_archive(tmp_path)
    archive = _archive(source)
    summary = archive.validate_topology()
    blocks = archive.load_development_blocks()

    assert summary.selected_run_count == 400
    assert len(blocks) == 100
    assert tuple(block.simulation_run for block in blocks) == tuple(range(1, 101))
    assert all(tuple(item.source_idv for item in block.conditions) == (1, 3, 9, 15) for block in blocks)
    runs = tuple(item for block in blocks for item in block.conditions)
    assert len(runs) == 400
    assert len({item.rng_seed for item in runs}) == 395
    assert {block.fold_id for block in blocks} == {0, 1, 2, 3, 4}
    assert all(item.x.shape == (500, 52) and not item.x.flags.writeable for item in runs)
    assert all(not item.fault.flags.writeable for item in runs)
    assert blocks[0].by_idv(1).fault_id == 0
    assert blocks[-1].by_idv(15).fault_id == 15
    assert blocks[0].by_idv(1).metadata["source_time_hours_first_last"] == (5.05, 30.0)
    assert blocks[0].by_idv(3).metadata["source_time_hours_first_last"] == (22.05, 47.0)
