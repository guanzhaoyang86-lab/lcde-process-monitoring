from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch  # Must precede NumPy and the numeric backend on Windows.
import numpy as np
import pytest

from mate_spc_cf_glr import backend_v4 as backend
from mate_spc_cf_glr.model import (
    CFGLRTrainingConfig,
    CFGLRTrainingResult,
    initialize_cf_glr_model,
)
from mate_spc_cf_glr.monitor import FittedExpertStandardizer


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_cf_max_glr15_gpu_v4.py"
)


def _runner():
    name = "run_cf_max_glr15_backend_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_backend_import_is_torch_first_and_hdf5_lazy():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys;"
        f"sys.path.insert(0,r'{source}');"
        "import torch;"
        "import mate_spc_cf_glr.backend_v4;"
        "assert 'torch' in sys.modules and 'h5py' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_backend_runner_config_plans_and_folds_match_exactly():
    runner = _runner()
    assert backend._expected_config() == runner.default_config()
    assert backend._canonical_fold_map() == runner._fold_assignment()
    assert list(backend._PLANS["development"]) == runner._development_access_plan()
    assert list(backend._PLANS["calibrate"]) == runner._calibration_access_plan()
    assert list(backend._PLANS["confirm"]) == runner._confirmation_access_plan()
    assert backend._expected_config()["features"]["schema"].endswith("_v2")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "schema_version", True),
        ("features", "causal_no_future_rows", 1),
        ("training", "epochs", 80.0),
        ("training", "mil_top_fraction", 1),
    ],
)
def test_backend_config_rejects_json_numeric_type_aliases(section, key, value):
    config = backend._expected_config()
    target = config if section is None else config[section]
    target[key] = value
    with pytest.raises(ValueError, match="exact audited"):
        backend._validate_config(config)


def test_malformed_plan_fails_before_archive_construction(monkeypatch):
    plan = [dict(row) for row in backend._PLANS["calibrate"]]
    plan[0] = {**plan[0], "idv": 3}
    opened = False

    class ForbiddenArchive:
        def __init__(self, _path):
            nonlocal opened
            opened = True

    monkeypatch.setattr(backend, "PCLRMode1Archive", ForbiddenArchive)
    with pytest.raises(ValueError, match="canonical"):
        backend.load_numeric_runs(Path("never-opened.h5"), plan)
    assert opened is False


@dataclass(frozen=True)
class FakeRun:
    source_idv: int
    simulation_run: int
    fault_magnitude_percent: int = 50

    @property
    def name(self) -> str:
        fault = 0 if self.source_idv == 1 else self.source_idv
        return (
            "PCLR-v2-ExtendedTEP-Mode1-development-magnitude-50-"
            f"fault-{fault:02d}-IDV-{self.source_idv:02d}-"
            f"run-{self.simulation_run:03d}"
        )


def _all_fake_runs():
    return tuple(
        FakeRun(idv, run) for run in range(1, 101) for idv in (1, 3, 9, 15)
    )


def test_bound_model_and_monitor_safe_roundtrip(tmp_path: Path):
    model = initialize_cf_glr_model()
    losses = {
        "fault_3_member_0": 0.1,
        "fault_9_member_0": 0.2,
        "fault_15_member_0": 0.3,
        "fault_15_member_1": 0.4,
        "fault_15_member_2": 0.5,
    }
    result = CFGLRTrainingResult(model, losses, CFGLRTrainingConfig())
    standardizer = FittedExpertStandardizer(
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        1.0e-12,
    )
    model_path = tmp_path / "model.npz"
    monitor_path = tmp_path / "monitor.json"
    metadata = backend._save_bound_artifacts(
        result,
        standardizer,
        config=backend._expected_config(),
        fit_runs=_all_fake_runs(),
        model_path=model_path,
        monitor_path=monitor_path,
    )
    verified = backend.verify_bound_artifacts(
        model_path=model_path,
        monitor_path=monitor_path,
        config=backend._expected_config(),
    )
    assert verified["model_metadata"] == metadata
    assert np.array_equal(verified["standardizer"].center, np.zeros(3))
    with np.load(model_path, allow_pickle=False) as archive:
        assert all(array.dtype != np.dtype(object) for array in archive.values())


def test_synthetic_development_has_exact_three_way_roles_and_no_model_on_no_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    runs = _all_fake_runs()
    fitted_roles: list[dict[int, list[int]]] = []

    class FakeModel:
        pass

    class FakeFit:
        model = FakeModel()

    def fake_fit(selected, _cache=None):
        role = {
            idv: sorted(run.simulation_run for run in selected if run.source_idv == idv)
            for idv in (1, 3, 9, 15)
        }
        fitted_roles.append(role)
        return FakeFit()

    def fake_scores(_model, selected, *, device, cache=None):
        values = np.empty((len(selected), 500, 3), dtype=np.float64)
        for index, run in enumerate(selected):
            base = run.simulation_run * 0.01
            values[index] = base + np.arange(500)[:, None] * 1.0e-5
            if run.source_idv != 1:
                values[index, 161:, :] += 2.0
        return values

    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **_kwargs: None)
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
    monkeypatch.setattr(backend, "load_numeric_runs", lambda *_args, **_kwargs: runs)
    monkeypatch.setattr(backend, "_build_feature_cache", lambda _runs: {})
    monkeypatch.setattr(backend, "_fit_model", fake_fit)
    monkeypatch.setattr(backend, "_score_model", fake_scores)
    monkeypatch.setattr(backend, "_development_pass", lambda _folds: False)
    monkeypatch.setattr(backend._torch.cuda, "empty_cache", lambda: None)

    result = backend.development_iteration(
        source_path=Path("synthetic.h5"),
        access_plan=list(backend._PLANS["development"]),
        fold_map=backend._canonical_fold_map(),
        config=backend._expected_config(),
        model_path=tmp_path / "model.npz",
        monitor_path=tmp_path / "monitor.json",
        device="cuda",
        dtype="float64",
    )
    assert result["selected_candidate_id"] is None
    assert len(result["folds"]) == 5
    assert len(fitted_roles) == 5
    for outer, roles in enumerate(fitted_roles):
        expected = {
            run for run, fold in backend._canonical_fold_map().items()
            if fold not in (outer, (outer + 1) % 5)
        }
        assert all(set(roles[idv]) == expected for idv in (1, 3, 9, 15))
        assert all(len(roles[idv]) == 60 for idv in roles)
    assert not (tmp_path / "model.npz").exists()
    assert not (tmp_path / "monitor.json").exists()
