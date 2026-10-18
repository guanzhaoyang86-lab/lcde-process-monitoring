from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
import mate_spc_external.extended_tep_mode1 as loader  # noqa: E402

from mate_spc_external.extended_tep_mode1 import (  # noqa: E402
    CALIBRATION_WINDOW,
    FAULT_WINDOW,
    LOCAL_FAULT_START,
    _base_path,
    _materialize_run,
    load_normalized_processdata,
    locate_unique_run,
)


def _raw(rows: int = 2001, columns: int = 54) -> np.ndarray:
    values = np.arange(rows * columns, dtype=np.float64).reshape(rows, columns)
    values[:, 0] = np.arange(rows, dtype=np.float64) * 0.05
    return values


def _valid_idv_init(idv: int) -> np.ndarray:
    values = np.zeros((3, 28), dtype=np.float64)
    values[:, idv - 1] = [0.0, 1.0, 30.0]
    return values


def _nonfinite_idv_init(idv: int) -> np.ndarray:
    values = _valid_idv_init(idv)
    values[0, 0] = np.nan
    return values


def _wrong_target_idv_init(idv: int) -> np.ndarray:
    values = _valid_idv_init(idv)
    values[2, idv - 1] = 29.9
    return values


def _add_run(
    handle: h5py.File,
    *,
    status: str = "SimulationCompleted",
    idv: int = 1,
    run: int = 1,
    values: np.ndarray | None = None,
    idv_init: np.ndarray | None = None,
) -> str:
    group_path = _base_path(status, idv, run)
    group = handle.require_group(group_path)
    group.create_dataset("processdata", data=_raw() if values is None else values)
    group.create_dataset(
        "idv_init", data=_valid_idv_init(idv) if idv_init is None else idv_init
    )
    return group_path


def test_projection_drops_initial_row_and_last_xmv(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.h5"
    raw = _raw()
    with h5py.File(source, "w") as handle:
        group_path = _add_run(handle, values=raw)
    with h5py.File(source, "r") as handle:
        located = locate_unique_run(handle, idv=1, simulation_run=1)
        values = load_normalized_processdata(handle, located)
    assert located.status == "SimulationCompleted"
    assert located.group_path == group_path
    assert values.shape == (2000, 52)
    assert values.dtype == np.float64
    assert values.flags.writeable is False
    np.testing.assert_array_equal(values, raw[1:, 1:53])


def test_frozen_calibration_and_fault_windows(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.h5"
    raw = _raw()
    with h5py.File(source, "w") as handle:
        _add_run(handle, idv=1, run=7, values=raw)
        _add_run(handle, idv=3, run=107, values=raw + 3.0)
    with h5py.File(source, "r") as handle:
        calibration = _materialize_run(
            handle,
            idv=1,
            simulation_run=7,
            window=CALIBRATION_WINDOW,
            partition="external_calibration_IDV1_prefault",
            reported_fault_id=0,
        )
        fault = _materialize_run(
            handle,
            idv=3,
            simulation_run=107,
            window=FAULT_WINDOW,
            partition="external_audit_fault_window",
            reported_fault_id=3,
        )
    np.testing.assert_array_equal(calibration.x, raw[1:501, 1:53])
    assert not calibration.fault.any()
    np.testing.assert_array_equal(fault.x, (raw + 3.0)[441:941, 1:53])
    assert not fault.fault[:LOCAL_FAULT_START].any()
    assert fault.fault[LOCAL_FAULT_START:].all()
    assert fault.metadata["normalized_window_half_open"] == [440, 940]
    assert fault.metadata["source_idv"] == 3
    assert fault.metadata["simulation_run"] == 107


def test_stopped_run_is_accepted_but_duplicate_status_is_rejected(tmp_path: Path) -> None:
    stopped_source = tmp_path / "stopped.h5"
    with h5py.File(stopped_source, "w") as handle:
        _add_run(handle, status="SimulationStopped", idv=9, run=150)
    with h5py.File(stopped_source, "r") as handle:
        located = locate_unique_run(handle, idv=9, simulation_run=150)
    assert located.status == "SimulationStopped"

    duplicate_source = tmp_path / "duplicate.h5"
    with h5py.File(duplicate_source, "w") as handle:
        _add_run(handle, status="SimulationCompleted", idv=9, run=150)
        _add_run(handle, status="SimulationStopped", idv=9, run=150)
    with h5py.File(duplicate_source, "r") as handle:
        with pytest.raises(ValueError, match="exactly one completion status; found 2"):
            locate_unique_run(handle, idv=9, simulation_run=150)


def test_missing_run_and_processdata_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "missing.h5"
    with h5py.File(source, "w") as handle:
        handle.require_group(_base_path("SimulationCompleted", 15, 101))
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="no exact processdata dataset"):
            locate_unique_run(handle, idv=15, simulation_run=101)
        with pytest.raises(ValueError, match="found 0"):
            locate_unique_run(handle, idv=15, simulation_run=102)


def test_missing_idv_init_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "missing_idv_init.h5"
    with h5py.File(source, "w") as handle:
        group = handle.require_group(_base_path("SimulationCompleted", 3, 101))
        group.create_dataset("processdata", data=_raw())
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match="no exact idv_init dataset"):
            locate_unique_run(handle, idv=3, simulation_run=101)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (_raw(rows=2000), "expected 2001 raw rows"),
        (_raw(columns=52), "expected exactly 54 raw columns"),
    ],
)
def test_short_or_narrow_processdata_fails_closed(
    tmp_path: Path, values: np.ndarray, message: str
) -> None:
    source = tmp_path / "invalid_shape.h5"
    with h5py.File(source, "w") as handle:
        _add_run(handle, values=values)
    with h5py.File(source, "r") as handle:
        located = locate_unique_run(handle, idv=1, simulation_run=1)
        with pytest.raises(ValueError, match=message):
            load_normalized_processdata(handle, located)


def test_nonfinite_projected_value_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "nonfinite.h5"
    values = _raw()
    values[900, 20] = np.nan
    with h5py.File(source, "w") as handle:
        _add_run(handle, values=values)
    with h5py.File(source, "r") as handle:
        located = locate_unique_run(handle, idv=1, simulation_run=1)
        with pytest.raises(ValueError, match="canonical projection is non-finite"):
            load_normalized_processdata(handle, located)


def test_processdata_requires_exact_float64(tmp_path: Path) -> None:
    source = tmp_path / "float32_processdata.h5"
    with h5py.File(source, "w") as handle:
        _add_run(handle, values=_raw().astype(np.float32))
    with h5py.File(source, "r") as handle:
        located = locate_unique_run(handle, idv=1, simulation_run=1)
        with pytest.raises(ValueError, match="exact float64 dtype"):
            load_normalized_processdata(handle, located)


@pytest.mark.parametrize(
    ("idv", "idv_init", "message"),
    [
        (3, _valid_idv_init(3)[:2], "expected exact shape"),
        (9, _valid_idv_init(9).astype(np.float32), "exact float64 dtype"),
        (15, _nonfinite_idv_init(15), "must be finite"),
        (1, _wrong_target_idv_init(1), "must declare only IDV1"),
        (
            3,
            np.where(
                np.indices((3, 28))[1] == 8,
                np.asarray([0.0, 1.0, 30.0])[:, None],
                _valid_idv_init(3),
            ),
            "must declare only IDV3",
        ),
    ],
)
def test_idv_init_validation_fails_closed(
    tmp_path: Path, idv: int, idv_init: np.ndarray, message: str
) -> None:
    source = tmp_path / f"invalid_idv_init_{idv}.h5"
    with h5py.File(source, "w") as handle:
        _add_run(handle, idv=idv, idv_init=idv_init)
    with h5py.File(source, "r") as handle:
        located = locate_unique_run(handle, idv=idv, simulation_run=1)
        with pytest.raises(ValueError, match=message):
            load_normalized_processdata(handle, located)


def test_materialize_reads_only_the_authorized_window(monkeypatch: pytest.MonkeyPatch) -> None:
    access_order: list[tuple[str, object]] = []

    class SpyDataset:
        def __init__(self, kind: str) -> None:
            self.kind = kind
            self.shape = (3, 28) if kind == "idv_init" else (2001, 54)
            self.dtype = np.dtype(np.float64)

        def __getitem__(self, key: object) -> np.ndarray:
            access_order.append((self.kind, key))
            if self.kind == "idv_init":
                return _valid_idv_init(15)
            if isinstance(key[0], list):
                return np.asarray([0.05 * index for index in key[0]], dtype=np.float64)
            return np.ones((500, 52), dtype=np.float64)

    located = loader.LocatedRun(
        "SimulationCompleted",
        15,
        "synthetic/group",
        "synthetic/group/processdata",
        "synthetic/group/idv_init",
    )
    processdata = SpyDataset("processdata")
    idv_init = SpyDataset("idv_init")
    monkeypatch.setattr(loader, "locate_unique_run", lambda *args, **kwargs: located)
    monkeypatch.setattr(
        loader,
        "_require_h5py",
        lambda: type("FakeH5py", (), {"Dataset": SpyDataset}),
    )
    run = loader._materialize_run(
        {located.processdata_path: processdata, located.idv_init_path: idv_init},
        idv=15,
        simulation_run=125,
        window=loader.FAULT_WINDOW,
        partition="external_audit_fault_window",
        reported_fault_id=15,
    )
    assert access_order[0] == ("idv_init", Ellipsis)
    assert access_order[1] == ("processdata", (slice(441, 941), slice(1, 53)))
    assert access_order[2] == ("processdata", ([441, 442, 940], 0))
    assert run.x.shape == (500, 52)


@pytest.mark.parametrize(
    ("idv", "simulation_run", "window", "partition", "reported_fault_id"),
    [
        (3, 101, loader.CALIBRATION_WINDOW, "external_audit_healthy_IDV1_prefault", 0),
        (3, 101, loader.FAULT_WINDOW, "external_audit_fault_window", 9),
        (1, 101, loader.CALIBRATION_WINDOW, "external_calibration_IDV1_prefault", 0),
    ],
)
def test_materialize_rejects_unauthorized_identity_window_combinations(
    idv: int,
    simulation_run: int,
    window: tuple[int, int],
    partition: str,
    reported_fault_id: int,
) -> None:
    with pytest.raises(ValueError, match="combination is not authorized"):
        loader._materialize_run(
            {},
            idv=idv,
            simulation_run=simulation_run,
            window=window,
            partition=partition,
            reported_fault_id=reported_fault_id,
        )


def test_processdata_labels_match_exact_official_sequence_with_legal_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "labels.h5"
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "Processdata_Labels",
            data=np.asarray(loader.PROCESSDATA_LABELS, dtype=object),
            dtype=string_type,
        )
    with h5py.File(source, "r") as handle:
        labels = loader.validate_processdata_labels(handle)
    assert labels == loader.PROCESSDATA_LABELS
    assert labels.count("A Feed") == 2
    assert labels.count("E Feed") == 2
    assert labels.count("A and C Feed") == 2

    with h5py.File(source, "a") as handle:
        del handle["Processdata_Labels"]
        invalid = list(loader.PROCESSDATA_LABELS)
        invalid[42] = "D Feed"
        handle.create_dataset(
            "Processdata_Labels", data=np.asarray(invalid, dtype=object), dtype=string_type
        )
    with h5py.File(source, "r") as handle:
        with pytest.raises(
            ValueError,
            match="index 42 expected 'D feed', got 'D Feed'",
        ):
            loader.validate_processdata_labels(handle)


@pytest.mark.parametrize(
    ("index", "replacement"),
    [(1, "A Feed "), (43, "E feed"), (53, "Condenser Coolant")],
)
def test_processdata_labels_reject_any_order_spelling_or_whitespace_change(
    tmp_path: Path, index: int, replacement: str
) -> None:
    source = tmp_path / f"labels_invalid_{index}.h5"
    labels = list(loader.PROCESSDATA_LABELS)
    labels[index] = replacement
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "Processdata_Labels",
            data=np.asarray(labels, dtype=object),
            dtype=string_type,
        )
    with h5py.File(source, "r") as handle:
        with pytest.raises(ValueError, match=f"index {index} expected"):
            loader.validate_processdata_labels(handle)
