"""Synthetic-only tests for the FARM Rank-CUSUM v7 backend."""

from __future__ import annotations

import torch

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mate_spc.hash_contract import canonical_json_file_bytes, canonical_json_sha256
from mate_spc_cpetc.data_v5 import ALLOWED_DUPLICATE_SEED_OWNERS
from mate_spc_farm_v7 import backend
from mate_spc_farm_v7.evaluation import FOLD_RUN_IDS, RunDecision
from mate_spc_farm_v7.monitor import FarmScoreBatch
from scripts import run_farm_rankcusum_gpu_v7 as runner


def _fold_map() -> dict[int, int]:
    return {
        run_id: fold
        for fold, run_ids in FOLD_RUN_IDS.items()
        for run_id in run_ids
    }


def _protocol() -> dict[str, Any]:
    return {
        "schema": backend.PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": backend.PROTOCOL_ID,
        "candidate_count": 1,
        "candidate": deepcopy(runner.METHOD_CONFIG),
        "source": {
            "sha256": backend.MODE2_SHA256,
            "size_bytes": backend.SOURCE_SIZE_BYTES,
            "mode": 2,
            "magnitude_percent": 50,
            "hdf5_opened_during_freeze": False,
            "mode1_access_authorized": False,
            "rieth_access_authorized": False,
        },
        "access_plan": {
            "record_count": 400,
            "selector_count": 400,
            "canonical_json_sha256": (
                backend.ACCESS_PLAN_CANONICAL_JSON_SHA256
            ),
            "wildcards_allowed": False,
        },
        "folds": {
            "fold_run_ids": {
                str(fold): list(FOLD_RUN_IDS[fold]) for fold in range(5)
            },
            "run_to_fold_canonical_json_sha256": (
                backend.FOLD_MAP_CANONICAL_JSON_SHA256
            ),
        },
        "governance": {
            "mode1_access_authorized": False,
            "rieth_access_authorized": False,
            "inner_fault_numeric_access_authorized": False,
            "fault_fit_or_threshold_authorized": False,
        },
        "numeric_backend_contract": {
            "module": "mate_spc_farm_v7.backend",
            "required_functions": [
                "cuda_preflight",
                "validate_protocol",
                "run_development",
            ],
            "hdf5_import_or_open_before_intent": False,
            "result_schema": backend.BACKEND_RESULT_SCHEMA,
            "protocol_identity_field": "protocol_canonical_json_sha256",
            "cuda_repeat_atol": 1.0e-6,
            "cpu_drift_is_go_gate": False,
        },
    }


def _identity_blocks() -> tuple[backend.NumericBlock, ...]:
    shared = {
        idv: np.full((500, 52), float(idv), dtype=np.float64)
        for idv in (1, 3, 9, 15)
    }
    for values in shared.values():
        values.setflags(write=False)
    return tuple(
        backend.NumericBlock(
            simulation_run=run_id,
            conditions=tuple(
                backend.NumericRun(
                    name=f"IDV{idv}-Run{run_id}",
                    x=shared[idv],
                    source_idv=idv,
                    simulation_run=run_id,
                    rng_seed=10_000_000 + 100 * run_id + idv,
                )
                for idv in (1, 3, 9, 15)
            ),
        )
        for run_id in range(1, 101)
    )


def _topology() -> dict[str, Any]:
    return {
        "selected_run_count": 400,
        "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {"1": 100, "3": 100, "9": 100, "15": 100},
        "unique_seed_count": 395,
        "duplicate_seed_value_count": 5,
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


def _passing_rows(outer_fold: int, *, fault15_miss: bool = False) -> tuple[Any, ...]:
    rows = []
    for run_id in FOLD_RUN_IDS[outer_fold]:
        rows.append(RunDecision(1, run_id, 1.0, None))
        rows.append(RunDecision(3, run_id, 3.0, 160))
        rows.append(RunDecision(9, run_id, 3.0, 160))
        rows.append(
            RunDecision(15, run_id, 1.0 if fault15_miss else 3.0, None if fault15_miss else 160)
        )
    return tuple(rows)


def _repeat() -> dict[str, Any]:
    return {
        "scope": "per_fold_inner_healthy_complete_cuda_score_and_decision_replay",
        "atol": backend.CUDA_REPEAT_ATOL,
        "maximum_absolute_error": 0.0,
        "discrete_equal": True,
        "passes": True,
    }


def test_protocol_uses_canonical_json_hash_not_file_bytes_hash() -> None:
    protocol = _protocol()
    canonical_hash = canonical_json_sha256(protocol)
    file_hash = hashlib.sha256(canonical_json_file_bytes(protocol)).hexdigest()
    assert canonical_hash != file_hash
    result = backend.validate_protocol(
        protocol, expected_canonical_json_sha256=canonical_hash
    )
    assert result["protocol_canonical_json_sha256"] == canonical_hash
    assert result["file_bytes_sha256_compared"] is False
    with pytest.raises(ValueError, match="canonical_json_sha256 mismatch"):
        backend.validate_protocol(
            protocol, expected_canonical_json_sha256=file_hash
        )


def test_backend_accepts_actual_runner_protocol_document() -> None:
    protocol = runner._protocol_document({})
    canonical_hash = canonical_json_sha256(protocol)
    result = backend.validate_protocol(
        protocol, expected_canonical_json_sha256=canonical_hash
    )
    assert result["valid"] is True
    assert result["protocol_canonical_json_sha256"] == canonical_hash
    assert (
        result["fold_map_canonical_json_sha256"]
        == runner.FOLD_MAP_SHA256
    )
    assert (
        result["access_plan_canonical_json_sha256"]
        == runner.ACCESS_PLAN_SHA256
    )


def test_protocol_semantics_and_exact_selectors_fail_closed() -> None:
    changed = deepcopy(_protocol())
    changed["candidate"]["reference"]["fault_runs_used"] = 1
    with pytest.raises(ValueError, match="healthy reference"):
        backend.validate_protocol(changed)
    plan = backend.development_access_plan()
    assert len(plan) == 400
    assert backend._validate_access_plan(plan) == plan
    changed_plan = deepcopy(plan)
    changed_plan[0]["processdata_row_zero_based_half_open"] = [0, 500]
    with pytest.raises(ValueError, match="exact 400"):
        backend._validate_access_plan(changed_plan)
    changed_fold = _fold_map()
    changed_fold[1] = 0
    with pytest.raises(ValueError, match="collision-safe"):
        backend._validate_fold_map(changed_fold)


def test_access_state_is_atomic_and_phase_distinguishing(tmp_path: Path) -> None:
    path = tmp_path / "backend_access_state.json"
    state = backend._AccessState(
        path, protocol_canonical_json_sha256="a" * 64
    )
    initial = json.loads(path.read_text(encoding="utf-8"))
    assert initial["hdf5_open_started"] is False
    assert initial["numeric_slices_read"] == 0
    assert initial["training_or_fit_started"] is False
    state.update(hdf5_open_started=True)
    state.update(hdf5_open_completed=True)
    state.record_processdata_slice()
    state.record_idv_init_slice()
    state.update(training_or_fit_started=True, backend_completed=True)
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["hdf5_open_started"] is True
    assert final["hdf5_open_completed"] is True
    assert final["numeric_slices_read"] == 2
    assert final["processdata_slices_read"] == 1
    assert final["idv_init_slices_read"] == 1
    assert final["training_or_fit_started"] is True
    assert final["backend_completed"] is True
    assert final["mode1_opens"] == 0


class _FakeGroup:
    def __init__(self, seed: int) -> None:
        self.attrs = {"seed": np.asarray([seed], dtype=np.int64)}


class _FakeDataset:
    def __init__(self, *, kind: str, idv: int) -> None:
        self.kind = kind
        self.idv = idv
        self.shape = (2000, 54) if kind == "processdata" else (3, 28)
        self.dtype = np.dtype(np.float64)
        self.calls: list[Any] = []

    def __getitem__(self, key: Any) -> np.ndarray:
        self.calls.append(key)
        if self.kind == "processdata":
            rows, columns = key
            assert columns == slice(0, 53)
            start, stop = int(rows.start), int(rows.stop)
            result = np.zeros((stop - start, 53), dtype=np.float64)
            result[:, 0] = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
            return result
        assert key == (slice(0, 3), slice(0, 28))
        result = np.zeros((3, 28), dtype=np.float64)
        result[:, self.idv - 1] = (0.0, 0.5, 30.0)
        return result


class _FakeHandle:
    def __init__(self, plan: list[dict[str, Any]]) -> None:
        self.objects: dict[str, Any] = {}
        self.accessed_paths: list[str] = []
        duplicate_seed_by_owner = {
            owner: seed
            for seed, owners in ALLOWED_DUPLICATE_SEED_OWNERS.items()
            for owner in owners
        }
        for selector in plan:
            owner = (selector["simulation_run"], selector["idv"])
            seed = duplicate_seed_by_owner.get(
                owner, 10_000_000 + 100 * owner[0] + owner[1]
            )
            self.objects[selector["group_path"]] = _FakeGroup(seed)
            self.objects[selector["processdata_path"]] = _FakeDataset(
                kind="processdata", idv=selector["idv"]
            )
            self.objects[selector["idv_init_path"]] = _FakeDataset(
                kind="idv_init", idv=selector["idv"]
            )

    def __contains__(self, path: str) -> bool:
        return path in self.objects

    def __getitem__(self, path: str) -> Any:
        self.accessed_paths.append(path)
        return self.objects[path]


class _SliceCounter:
    def __init__(self) -> None:
        self.processdata = 0
        self.idv_init = 0

    def record_processdata_slice(self) -> None:
        self.processdata += 1

    def record_idv_init_slice(self) -> None:
        self.idv_init += 1


def test_exact_reader_only_reads_authorized_slices_and_updates_each_counter() -> None:
    plan = backend.development_access_plan()
    handle = _FakeHandle(plan)
    counter = _SliceCounter()
    topology, blocks = backend._read_exact_blocks(
        handle, plan, access_state=counter  # type: ignore[arg-type]
    )
    assert len(blocks) == 100
    assert topology == _topology()
    assert (counter.processdata, counter.idv_init) == (400, 400)
    assert "Processdata_Labels" not in handle.accessed_paths
    for selector in plan:
        processdata = handle.objects[selector["processdata_path"]]
        idv_init = handle.objects[selector["idv_init_path"]]
        start, stop = selector["window_zero_based_half_open"]
        assert processdata.calls == [(slice(start, stop), slice(0, 53))]
        assert idv_init.calls == [(slice(0, 3), slice(0, 28))]


def _inner_score(maxima: np.ndarray) -> FarmScoreBatch:
    paths = np.zeros((20, 340), dtype=np.float64)
    paths[:, -1] = maxima
    return FarmScoreBatch(paths=paths, source_device="cuda:0", reference_count=30_000)


def test_inner_cuda_complete_double_replay_is_numeric_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maxima = np.arange(1.0, 21.0, dtype=np.float64)
    scores = [_inner_score(maxima), _inner_score(maxima)]
    monkeypatch.setattr(backend, "_score_cuda", lambda *args, **kwargs: scores.pop(0))
    reference = SimpleNamespace(reference_count=30_000)
    inner = tuple(
        SimpleNamespace(source_idv=1, simulation_run=index + 1)
        for index in range(20)
    )
    threshold, repeat = backend._audit_inner(reference, inner, device="cuda")
    assert threshold.threshold == 20.0
    assert repeat["passes"] is True
    assert repeat["discrete_equal"] is True
    assert repeat["scope"] == (
        "per_fold_inner_healthy_complete_cuda_score_and_decision_replay"
    )


def test_fold_uses_only_healthy_fit_and_threshold_and_freezes_before_outer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocks = _identity_blocks()
    events: list[str] = []
    fit_values: list[np.ndarray] = []
    reference = SimpleNamespace(
        sorted_values=SimpleNamespace(shape=(46, 30_000)),
        reference_count=30_000,
    )

    class FakeReference:
        @classmethod
        def fit(cls, values: np.ndarray, *, expected_run_count: int) -> Any:
            fit_values.append(values)
            assert expected_run_count == 60
            return reference

    paths = np.zeros((80, 340), dtype=np.float64)
    score = FarmScoreBatch(paths=paths, source_device="cuda:0", reference_count=30_000)
    decisions = SimpleNamespace(
        alarms=np.zeros(80, dtype=np.bool_),
        alarm_endpoints=np.full(80, -1, dtype=np.int64),
        delays=np.full(80, 340, dtype=np.int64),
        run_maxima=np.zeros(80, dtype=np.float64),
    )
    fake_monitor = SimpleNamespace(
        FarmReference=FakeReference,
        apply_threshold=lambda score_value, threshold: decisions,
    )
    threshold = SimpleNamespace(
        threshold=1.0,
        to_document=lambda: {"threshold": 1.0},
    )
    monkeypatch.setattr(backend, "_monitor_api", lambda: fake_monitor)
    monkeypatch.setattr(
        backend,
        "_audit_inner",
        lambda ref, inner, *, device: (
            threshold,
            _repeat(),
        )
        if len(inner) == 20 and all(run.source_idv == 1 for run in inner)
        else (_ for _ in ()).throw(AssertionError("fault entered threshold")),
    )
    monkeypatch.setattr(
        backend,
        "_freeze_fold_artifacts",
        lambda *args, **kwargs: events.append("freeze") or {},
    )
    monkeypatch.setattr(
        backend,
        "_score_cuda",
        lambda *args, **kwargs: events.append("outer") or score,
    )
    monkeypatch.setattr(backend, "_save_outer_result", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        backend,
        "_cpu_report",
        lambda *args, **kwargs: {"is_go_gate": False, "completed": False},
    )
    backend._execute_fold(
        blocks,
        outer_fold=0,
        artifact_root=tmp_path,
        protocol_canonical_json_sha256="a" * 64,
        device="cuda",
    )
    assert events == ["freeze", "outer"]
    assert len(fit_values) == 1
    assert fit_values[0].shape == (60, 500, 52)
    assert np.all(fit_values[0] == 1.0)


def _patch_run_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[backend.NumericBlock, ...]:
    blocks = _identity_blocks()
    monkeypatch.setattr(
        backend,
        "_require_cuda",
        lambda device: {
            "passed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5090",
        },
    )

    def fake_load(plan: Any, *, access_state: Any) -> Any:
        access_state.update(hdf5_open_started=True, hdf5_open_completed=True)
        access_state.update(
            processdata_slices_read=400,
            idv_init_slices_read=400,
            numeric_slices_read=800,
        )
        return _topology(), blocks

    monkeypatch.setattr(backend, "_load_numeric_blocks", fake_load)
    monkeypatch.setattr(backend, "_validate_blocks", lambda value, fold_map: tuple(value))
    return blocks


@pytest.mark.parametrize(
    ("fault15_miss", "expected"),
    [(False, "DEVELOPMENT_GO"), (True, "DEVELOPMENT_NO_GO")],
)
def test_run_development_go_only_final_refit_and_access_state_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault15_miss: bool,
    expected: str,
) -> None:
    _patch_run_inputs(monkeypatch)

    def fake_fold(
        blocks: Any,
        *,
        outer_fold: int,
        artifact_root: Path,
        protocol_canonical_json_sha256: str,
        device: str,
    ) -> Any:
        rows = _passing_rows(outer_fold, fault15_miss=fault15_miss)
        return {
            "outer_fold": outer_fold,
            "numeric_valid": True,
            "authoritative_cuda_repeat": _repeat(),
            "cross_device_report_only": {
                "is_go_gate": False,
                "completed": True,
                "maximum_absolute_error": 0.0,
                "discrete_equal": True,
            },
        }, rows

    monkeypatch.setattr(backend, "_execute_fold", fake_fold)
    refit_calls: list[int] = []

    def fake_final_refit(
        blocks: Any,
        *,
        artifact_root: Path,
        protocol_canonical_json_sha256: str,
    ) -> dict[str, Any]:
        refit_calls.append(len(blocks))
        assert (
            artifact_root / "development_decision_before_final_refit.json"
        ).is_file()
        return {
            "schema": "farm_rankcusum_v7_final_mode2_refit",
            "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
            "candidate_id": backend.CANDIDATE_ID,
            "fit_healthy_run_count": 100,
            "fit_fault_run_count": 0,
            "reference_shape": [46, 50_000],
            "created_only_after_development_go": True,
            "target_domain_threshold_not_fitted": True,
            "reference": {
                "path": "C:/synthetic/healthy_reference.npz",
                "file_bytes_sha256": "0" * 64,
                "size_bytes": 1,
            },
            "manifest": {
                "path": "C:/synthetic/manifest.json",
                "file_bytes_sha256": "1" * 64,
                "size_bytes": 1,
            },
        }

    monkeypatch.setattr(
        backend,
        "_run_final_refit",
        fake_final_refit,
    )
    result = backend.run_development(
        access_plan=backend.development_access_plan(),
        fold_map=_fold_map(),
        artifact_root=tmp_path / expected.lower(),
        protocol_canonical_json_sha256="b" * 64,
    )
    assert result["decision"] == expected
    assert result["fault_fit_run_count"] == 0
    assert result["fault_threshold_run_count"] == 0
    assert refit_calls == ([] if fault15_miss else [100])
    assert (result["final_refit"] is not None) == (not fault15_miss)
    state = result["backend_access_state"]["final_state"]
    assert state["numeric_slices_read"] == 800
    assert state["training_or_fit_started"] is True
    assert state["backend_completed"] is True
    assert state["mode1_opens"] == 0
    record = result["backend_access_state"]["artifact"]
    assert record["file_bytes_sha256"] == hashlib.sha256(
        Path(record["path"]).read_bytes()
    ).hexdigest()
    decision_record = result["development_decision_before_final_refit"]
    decision_path = Path(decision_record["path"])
    decision_document = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision_path.name == "development_decision_before_final_refit.json"
    assert decision_document["decision"] == expected
    assert len(decision_document["folds"]) == 5
    assert decision_document["gates"] == result["gates"]
    assert decision_document["aggregate"] == result["aggregate"]
    assert decision_document["authoritative_cuda_repeat"] == result[
        "authoritative_cuda_repeat"
    ]
    assert decision_document["cross_device_report_only"] == result[
        "cross_device_report_only"
    ]
    assert decision_document["final_refit_started"] is False
    assert decision_document["final_model_usable"] is False
    assert decision_record["file_bytes_sha256"] == hashlib.sha256(
        decision_path.read_bytes()
    ).hexdigest()
    assert (
        runner._validate_development_result(
            result, protocol_canonical_json_sha256="b" * 64
        )
        == result
    )


def test_numeric_threshold_failure_completes_no_go_without_final_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_run_inputs(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_execute_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            backend._FoldNumericNoGo("synthetic degeneracy")
        ),
    )
    final_called: list[bool] = []
    monkeypatch.setattr(
        backend,
        "_run_final_refit",
        lambda *args, **kwargs: final_called.append(True),
    )
    result = backend.run_development(
        access_plan=backend.development_access_plan(),
        fold_map=_fold_map(),
        artifact_root=tmp_path / "numeric-no-go",
        protocol_canonical_json_sha256="c" * 64,
    )
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert len(result["folds"]) == 5
    assert result["final_refit"] is None
    assert final_called == []
    assert not (tmp_path / "numeric-no-go" / "final_refit").exists()
    decision_path = (
        tmp_path
        / "numeric-no-go"
        / "development_decision_before_final_refit.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "DEVELOPMENT_NO_GO"
    assert len(decision["folds"]) == 5
    assert decision["final_refit_permitted"] is False


def test_go_refit_failure_preserves_pre_refit_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_run_inputs(monkeypatch)

    def fake_fold(
        blocks: Any,
        *,
        outer_fold: int,
        artifact_root: Path,
        protocol_canonical_json_sha256: str,
        device: str,
    ) -> Any:
        return {
            "outer_fold": outer_fold,
            "numeric_valid": True,
            "authoritative_cuda_repeat": _repeat(),
            "cross_device_report_only": {
                "is_go_gate": False,
                "completed": True,
                "maximum_absolute_error": 0.0,
                "discrete_equal": True,
            },
        }, _passing_rows(outer_fold)

    monkeypatch.setattr(backend, "_execute_fold", fake_fold)

    def fail_final_refit(*args: Any, artifact_root: Path, **kwargs: Any) -> Any:
        assert (
            artifact_root / "development_decision_before_final_refit.json"
        ).is_file()
        raise OSError("synthetic final-refit write failure")

    monkeypatch.setattr(backend, "_run_final_refit", fail_final_refit)
    artifact_root = tmp_path / "refit-failure"
    with pytest.raises(OSError, match="synthetic final-refit"):
        backend.run_development(
            access_plan=backend.development_access_plan(),
            fold_map=_fold_map(),
            artifact_root=artifact_root,
            protocol_canonical_json_sha256="d" * 64,
        )
    decision_path = (
        artifact_root / "development_decision_before_final_refit.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "DEVELOPMENT_GO"
    assert decision["development_decision_determined"] is True
    assert decision["final_refit_started"] is False
    assert decision["final_model_usable"] is False
    assert len(decision["folds"]) == 5
    assert not (artifact_root / "development_result.json").exists()
