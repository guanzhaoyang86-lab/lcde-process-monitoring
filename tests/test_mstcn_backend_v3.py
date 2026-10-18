from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

# The supported Windows CUDA runtime requires Torch to own the DLL/OpenMP
# load order before NumPy, matching the production backend contract.
import torch  # noqa: F401
import numpy as np
import pytest

import scripts.run_mstcn_spc_gpu_v3 as runner
from mate_spc_mstcn import backend_v3 as backend
from mate_spc_mstcn.model import MSTCNPath
from mate_spc_mstcn.monitor import FittedTopTailMonitor


def _numeric_runs(plan: tuple[dict[str, object], ...]) -> tuple[backend.NumericRun, ...]:
    shared = np.zeros((500, 52), dtype=np.float64)
    return tuple(
        backend.NumericRun(
            int(row["idv"]),
            int(row["simulation_run"]),
            int(row["magnitude_percent"]),
            index + 1,
            shared,
        )
        for index, row in enumerate(plan)
    )


def _paths(runs: list[backend.NumericRun] | tuple[backend.NumericRun, ...]) -> tuple[MSTCNPath, ...]:
    endpoints = backend.MONITOR_ENDPOINTS
    result = []
    for run in runs:
        evidence = np.zeros(endpoints.size, dtype=np.float64)
        attribution = np.zeros((endpoints.size, 3), dtype=np.float64)
        result.append(MSTCNPath(endpoints, evidence, attribution))
    return tuple(result)


def test_backend_and_runner_exact_config_and_plans_match() -> None:
    assert backend._expected_config() == runner.default_model_config()
    assert backend._canonical_fold_map() == runner._fold_assignment()
    assert list(backend._PLANS["development"]) == runner._development_access_plan()
    assert list(backend._PLANS["calibrate"]) == runner._calibration_access_plan()
    assert list(backend._PLANS["confirm"]) == runner._confirmation_access_plan()


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("architecture", "channels", 32.0),
        ("architecture", "detector_output_count", True),
        ("training", "epochs", 30.0),
        ("feature_transform", "causal_no_future_rows", 1),
    ],
)
def test_backend_exact_config_rejects_json_type_drift(
    section: str, field: str, replacement: object
) -> None:
    config = runner.default_model_config()
    config[section][field] = replacement
    with pytest.raises(ValueError, match="exact audited"):
        backend._validate_config(config)


def test_backend_rejects_json_type_aliases_in_exact_config() -> None:
    config = runner.default_model_config()
    type_aliases = (
        ("architecture", "channels", 32.0),
        ("architecture", "detector_output_count", True),
        ("training", "epochs", 30.0),
        ("feature_transform", "causal_no_future_rows", 1),
    )
    for section, field, alias in type_aliases:
        malformed = copy.deepcopy(config)
        malformed[section][field] = alias
        with pytest.raises(ValueError, match="exact audited"):
            backend._validate_config(malformed)


def test_malformed_plan_fails_before_hdf5_import_or_open(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = [dict(row) for row in backend._PLANS["calibrate"]]
    plan[0]["idv"] = 3

    def forbidden() -> object:
        raise AssertionError("h5py must not be imported for an unauthorized plan")

    monkeypatch.setattr(backend, "_require_h5py", forbidden)
    with pytest.raises(ValueError, match="not one exact canonical"):
        backend.load_numeric_runs("TEP_Mode1.h5", plan)


def test_development_no_go_returns_four_complete_candidates_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _numeric_runs(backend._PLANS["development"])
    fitted_q: list[float] = []

    def fake_fit(selected: list[backend.NumericRun], config: object, q: float) -> object:
        fitted_q.append(q)
        return SimpleNamespace(model=object(), metadata={"q": q}, loss_history=(1.0,))

    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **_: None)
    monkeypatch.setattr(
        backend,
        "_torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )
    monkeypatch.setattr(backend, "load_numeric_runs", lambda *_: runs)
    monkeypatch.setattr(backend, "_fit_model", fake_fit)
    monkeypatch.setattr(backend, "_score_runs", lambda model, selected, **_: _paths(selected))
    monkeypatch.setattr(
        backend,
        "cuda_preflight",
        lambda: {
            "cuda_available": True,
            "device_type": "cuda",
            "dtype": "float64",
            "torch_version": "synthetic",
            "cuda_version": "synthetic",
            "device_name": "synthetic",
        },
    )
    model_path = tmp_path / "must_not_exist.npz"
    result = backend.development_iteration(
        source_path=Path("TEP_Mode1.h5"),
        access_plan=list(backend._PLANS["development"]),
        fold_map=backend._canonical_fold_map(),
        config=runner.default_model_config(),
        model_path=model_path,
        device="cuda",
        dtype="float64",
    )
    assert result["selected_candidate_id"] is None
    assert result["model_metadata"] is None
    assert result["model_parity_max_abs_error"] is None
    assert not model_path.exists()
    assert len(result["candidates"]) == 4
    assert all(len(candidate["folds"]) == 5 for candidate in result["candidates"])
    assert all(
        len(fold["rows"]) == 80
        and len(fold["training_healthy_maxima"]) == 80
        and fold["discrete_parity_equal"] is True
        for candidate in result["candidates"]
        for fold in candidate["folds"]
    )
    assert fitted_q == [q for _fold in range(5) for q in (0.125, 0.25)]


def test_calibrate_and_confirm_use_frozen_monitor_transform_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(99)
    healthy_fit = rng.normal(size=(100, 340)).astype(np.float64)
    monitor = FittedTopTailMonitor.fit(
        healthy_fit,
        backend.MONITOR_ENDPOINTS,
        windows=backend.SCAN_WINDOWS,
        top_fraction=0.125,
    )
    marker_model = object()

    def fake_load(model_path: Path, config: object) -> tuple[object, FittedTopTailMonitor, dict]:
        return marker_model, monitor, {}

    def fake_load_runs(source_path: Path, access_plan: object) -> tuple[backend.NumericRun, ...]:
        _, normalized = backend._validate_access_plan(access_plan)
        return _numeric_runs(normalized)

    def fake_score(model: object, runs: object, **kwargs: object) -> tuple[MSTCNPath, ...]:
        paths = []
        for run in runs:
            offset = 0.001 * run.simulation_run + (5.0 if run.idv != 1 else 0.0)
            evidence = np.linspace(-1.0, 1.0, 340, dtype=np.float64) + offset
            attribution = np.zeros((340, 3), dtype=np.float64)
            attribution[:, (3, 9, 15).index(run.idv) if run.idv != 1 else 0] = 1.0
            paths.append(MSTCNPath(backend.MONITOR_ENDPOINTS, evidence, attribution))
        return tuple(paths)

    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **_: None)
    monkeypatch.setattr(backend, "_load_bound_model", fake_load)
    monkeypatch.setattr(backend, "load_numeric_runs", fake_load_runs)
    monkeypatch.setattr(backend, "_score_runs", fake_score)
    config = runner.default_model_config()

    calibration = backend.calibrate(
        source_path=Path("TEP_Mode1.h5"),
        access_plan=list(backend._PLANS["calibrate"]),
        model_path=Path("model.npz"),
        config=config,
        device="cuda",
        dtype="float64",
    )
    assert len(calibration["rows"]) == 100
    assert calibration["discrete_parity_equal"] is True
    threshold = sorted(row["run_maximum"] for row in calibration["rows"])[97]

    confirmation = backend.confirm(
        source_path=Path("TEP_Mode1.h5"),
        access_plan=list(backend._PLANS["confirm"]),
        model_path=Path("model.npz"),
        config=config,
        threshold=threshold,
        device="cuda",
        dtype="float64",
    )
    assert len(confirmation["rows"]) == 400
    assert confirmation["discrete_parity_equal"] is True
    assert all(
        row["alarm"] == (row["run_maximum"] > threshold)
        for row in confirmation["rows"]
    )
