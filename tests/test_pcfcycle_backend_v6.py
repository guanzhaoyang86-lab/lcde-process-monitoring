"""Synthetic-only contract tests for the v6 development backend."""

from __future__ import annotations

import torch

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_pcf_cyclerank_gpu_v6 as runner
from mate_spc_cpetc.data_v5 import ALLOWED_DUPLICATE_SEED_OWNERS
from mate_spc_pcfcycle import backend_v6 as backend
from mate_spc_pcfcycle.evaluation import FOLD_RUN_IDS, RunDecision


def _fold_map() -> dict[int, int]:
    return {
        run_id: fold
        for fold, run_ids in FOLD_RUN_IDS.items()
        for run_id in run_ids
    }


def _identity_blocks() -> tuple[Any, ...]:
    return tuple(
        SimpleNamespace(
            simulation_run=run_id,
            conditions=tuple(
                SimpleNamespace(source_idv=idv, simulation_run=run_id)
                for idv in (1, 3, 9, 15)
            ),
        )
        for run_id in range(1, 101)
    )


def _repeat_report() -> dict[str, Any]:
    return {
        "scope": "per_fold_inner_complete_score_scan_replay",
        "atol": backend.CUDA_REPEAT_ATOL,
        "maximum_absolute_error": 0.0,
        "discrete_equal": True,
        "passes": True,
    }


def _mock_fold_document(
    blocks: tuple[Any, ...], outer_fold: int, *, successful: bool
) -> tuple[dict[str, Any], tuple[RunDecision, ...]]:
    outer = set(FOLD_RUN_IDS[outer_fold])
    rows = []
    for block in blocks:
        if block.simulation_run not in outer:
            continue
        for run in block.conditions:
            alarm = 164 if run.source_idv != 1 and (successful or run.source_idv != 15) else None
            rows.append(
                RunDecision(
                    source_idv=run.source_idv,
                    simulation_run=run.simulation_run,
                    run_maximum=2.0 if alarm is not None else 0.0,
                    alarm_endpoint=alarm,
                    attributed_fault=run.source_idv if alarm is not None else None,
                )
            )
    document = {
        "outer_fold": outer_fold,
        "numeric_valid": True,
        "authoritative_cuda_repeat": _repeat_report(),
        "cross_device_report_only": {
            "is_go_gate": False,
            "maximum_absolute_error": 0.0,
            "discrete_equal": True,
        },
    }
    return document, tuple(rows)


def _patch_development_inputs(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    blocks = _identity_blocks()
    topology = {
        "selected_run_count": 400,
        "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {"1": 100, "3": 100, "9": 100, "15": 100},
        "unique_seed_count": 395,
        "duplicate_seed_value_count": 5,
        "duplicate_seed_owners": {
            str(seed): [list(owner) for owner in owners]
            for seed, owners in ALLOWED_DUPLICATE_SEED_OWNERS.items()
        },
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
    monkeypatch.setattr(
        backend,
        "_require_formal_cuda",
        lambda device: {
            "passed": True,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 5090",
        },
    )
    monkeypatch.setattr(
        backend,
        "_load_numeric_blocks",
        lambda plan: (topology, blocks),
    )
    return blocks


def test_runner_protocol_document_validates_without_hdf5() -> None:
    document = runner._protocol_document({})
    validated = backend.validate_protocol(document)
    assert validated["valid"] is True
    assert validated["candidate_id"] == backend.CANDIDATE_ID
    assert backend.ACCESS_PLAN_SHA256 == runner.ACCESS_PLAN_SHA256

    changed = deepcopy(document)
    changed["candidate"]["stage2"]["epochs"] = 81
    with pytest.raises(ValueError, match="stage2 epochs"):
        backend.validate_protocol(changed)

    changed = deepcopy(document)
    changed["numeric_backend_contract"]["cpu_drift_is_go_gate"] = True
    with pytest.raises(ValueError, match="numeric backend contract"):
        backend.validate_protocol(changed)


def test_cuda_preflight_exposes_runner_pass_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=32 * 1024**3),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = backend.cuda_preflight()
    assert report["passed"] is True
    assert report["cuda_available"] is True
    assert report["rtx_5090"] is True


def test_exact_access_plan_and_fold_map_fail_closed() -> None:
    plan = backend.development_access_plan()
    assert len(plan) == 400
    assert backend._validate_access_plan(plan) == plan
    assert backend._validate_fold_map(_fold_map()) == _fold_map()
    changed = deepcopy(plan)
    changed[0]["processdata_row_zero_based_half_open"] = [0, 500]
    with pytest.raises(ValueError, match="exact 400"):
        backend._validate_access_plan(changed)
    changed_folds = _fold_map()
    changed_folds[1] = 0
    with pytest.raises(ValueError, match="collision-safe"):
        backend._validate_fold_map(changed_folds)


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
            assert isinstance(rows, slice) and isinstance(columns, slice)
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


def test_exact_selector_reader_never_reads_root_labels_or_outside_window() -> None:
    plan = backend.development_access_plan()
    handle = _FakeHandle(plan)
    topology, blocks = backend._read_exact_selector_blocks(handle, plan)
    assert len(blocks) == 100
    assert topology["unique_seed_count"] == 395
    assert topology["numeric_value_access"]["root_dataset_value_reads"] == 0
    assert "Processdata_Labels" not in handle.accessed_paths
    for selector in plan:
        processdata = handle.objects[selector["processdata_path"]]
        idv_init = handle.objects[selector["idv_init_path"]]
        start, stop = selector["window_zero_based_half_open"]
        assert processdata.calls == [(slice(start, stop), slice(0, 53))]
        assert idv_init.calls == [(slice(0, 3), slice(0, 28))]


def test_freeze_artifacts_precedes_outer_score(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    blocks = _identity_blocks()
    threshold = SimpleNamespace(
        threshold=1.0,
        valid=True,
        to_document=lambda: {"threshold": 1.0, "valid": True},
    )
    monkeypatch.setattr(backend, "_fit_fold_state", lambda *args, **kwargs: object())
    monkeypatch.setattr(backend, "_fit_scan_calibrator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        backend,
        "_audit_inner",
        lambda *args, **kwargs: (threshold, _repeat_report()),
    )
    monkeypatch.setattr(
        backend,
        "_save_fold_artifacts",
        lambda *args, **kwargs: events.append("freeze") or {},
    )
    _, decisions = _mock_fold_document(blocks, 0, successful=True)
    monkeypatch.setattr(
        backend,
        "_score_outer",
        lambda *args, **kwargs: (
            events.append("outer") or decisions,
            [],
            object(),
        ),
    )
    monkeypatch.setattr(backend, "_save_outer_result", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        backend,
        "_cross_device_report",
        lambda *args, **kwargs: {
            "is_go_gate": False,
            "maximum_absolute_error": 0.0,
            "discrete_equal": True,
        },
    )
    backend._execute_fold(
        blocks, outer_fold=0, artifact_root=tmp_path, device="cuda"
    )
    assert events == ["freeze", "outer"]


@pytest.mark.parametrize(
    ("successful", "expected_decision"),
    [(True, "DEVELOPMENT_GO"), (False, "DEVELOPMENT_NO_GO")],
)
def test_run_development_schema_and_go_only_refit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    successful: bool,
    expected_decision: str,
) -> None:
    blocks = _patch_development_inputs(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_execute_fold",
        lambda unused, *, outer_fold, artifact_root, device: _mock_fold_document(
            blocks, outer_fold, successful=successful
        ),
    )
    refit_calls: list[int] = []
    monkeypatch.setattr(
        backend,
        "_run_final_refit",
        lambda unused, *, artifact_root, device: refit_calls.append(len(unused))
        or {
            "schema": "pcf_cyclerank_final_mode2_refit_v1",
            "candidate_id": backend.CANDIDATE_ID,
            "runs_per_class": 100,
            "scan_calibrator_fit_healthy_runs": 100,
            "created_only_after_development_go": True,
            "target_domain_calibration_not_performed": True,
            "parameter_counts": {"total": 1},
            "feature_state": {
                "path": "mock/feature_state.npz",
                "sha256": "1" * 64,
                "size_bytes": 1,
            },
            "model": {
                "path": "mock/model.pt",
                "sha256": "2" * 64,
                "size_bytes": 1,
            },
            "scan_calibrator": {
                "path": "mock/scan_calibrator.json",
                "sha256": "3" * 64,
                "size_bytes": 1,
            },
            "manifest": {
                "path": "mock/manifest.json",
                "sha256": "4" * 64,
                "size_bytes": 1,
            },
        },
    )
    protocol_sha256 = "a" * 64
    result = backend.run_development(
        access_plan=backend.development_access_plan(),
        fold_map=_fold_map(),
        artifact_root=tmp_path / expected_decision.lower(),
        protocol_sha256=protocol_sha256,
    )
    assert result["decision"] == expected_decision
    assert result["mode1_hdf5_opens_or_value_reads"] == 0
    assert result["authoritative_cuda_repeat"]["scope"] == (
        "per_fold_inner_complete_score_scan_replay"
    )
    assert refit_calls == ([100] if successful else [])
    validated = runner._validate_development_result(
        result, protocol_sha256=protocol_sha256
    )
    assert validated["decision"] == expected_decision


def test_numeric_failure_is_completed_development_no_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_development_inputs(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_execute_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FloatingPointError("synthetic numeric guard")
        ),
    )
    protocol_sha256 = "b" * 64
    result = backend.run_development(
        access_plan=backend.development_access_plan(),
        fold_map=_fold_map(),
        artifact_root=tmp_path / "numeric_no_go",
        protocol_sha256=protocol_sha256,
    )
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert len(result["folds"]) == 5
    assert result["final_refit"] is None
    runner._validate_development_result(result, protocol_sha256=protocol_sha256)


def test_final_refit_fits_and_serializes_100_healthy_calibrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocks = _identity_blocks()
    state = SimpleNamespace(
        extractor=object(),
        b_standardizer=object(),
        model=object(),
    )
    monkeypatch.setattr(backend, "_fit_fold_state", lambda *args, **kwargs: state)
    calibrator_calls: list[tuple[int, int]] = []
    calibrator = SimpleNamespace(
        to_document=lambda: {
            "schema": "pcf_cyclerank_run_relative_logit_calibrator_v2",
            "fit_run_count": 100,
        }
    )

    def fake_fit_calibrator(
        unused_state: Any,
        healthy: tuple[Any, ...],
        *,
        device: str,
        expected_run_count: int,
    ) -> Any:
        calibrator_calls.append((len(healthy), expected_run_count))
        return calibrator

    writes: list[tuple[str, Any]] = []
    monkeypatch.setattr(backend, "_fit_scan_calibrator", fake_fit_calibrator)
    monkeypatch.setattr(backend, "save_pcfcycle_feature_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_save_model_atomic", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_atomic_json",
        lambda path, value: writes.append((Path(path).name, value)),
    )
    monkeypatch.setattr(
        backend,
        "_file_record",
        lambda path: {"path": str(path), "sha256": "0" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(backend, "parameter_counts", lambda model: {"total": 1})
    result = backend._run_final_refit(
        blocks, artifact_root=tmp_path, device="cuda"
    )
    assert calibrator_calls == [(100, 100)]
    assert result["scan_calibrator_fit_healthy_runs"] == 100
    assert "scan_calibrator" in result
    calibrator_write = next(value for name, value in writes if name == "scan_calibrator.json")
    assert calibrator_write["fit_run_count"] == 100
