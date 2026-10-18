from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import torch  # Must precede NumPy on Windows.
import numpy as np
import pytest

from mate_spc_cpetc import backend_v5 as backend
from mate_spc_cpetc.monitor import ExpertTailCalibrator
from mate_spc_cpetc.model import CPETCModel


def test_backend_import_is_torch_first_and_hdf5_lazy():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys;"
        f"sys.path.insert(0,r'{source}');"
        "import torch;"
        "import mate_spc_cpetc.backend_v5;"
        "assert 'torch' in sys.modules and 'h5py' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_exact_v51_fold_hash_roles_counts_and_disjointness():
    mapping = backend._canonical_fold_map()
    assert backend._fold_map_sha256(mapping) == backend.EXPECTED_FOLD_MAP_SHA256
    assert sorted(mapping) == list(range(1, 101))
    for outer in range(5):
        roles = backend.fold_roles(outer)
        assert len(roles["fit"]) == 60
        assert len(roles["inner_healthy"]) == 20
        assert len(roles["outer"]) == 20
        assert not (set(roles["fit"]) & set(roles["inner_healthy"]))
        assert not (set(roles["fit"]) & set(roles["outer"]))
        assert not (set(roles["inner_healthy"]) & set(roles["outer"]))
        assert set().union(*map(set, roles.values())) == set(range(1, 101))


def test_exact_parameter_counts_are_frozen_and_reportable():
    model = CPETCModel()
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)
    assert backend._parameter_counts(model) == {
        "classifier_exact": 9207,
        "autoencoder_exact": 6604,
        "total_exact": 15811,
        "trainable_at_checkpoint": 9207,
    }


def test_malformed_plan_fails_before_archive_construction(monkeypatch):
    plan = backend.development_access_plan()
    plan[0] = {**plan[0], "idv": 3}
    constructed = False

    class ForbiddenArchive:
        def __init__(self):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(backend, "Mode2DevelopmentArchive", ForbiddenArchive)
    with pytest.raises(ValueError, match="frozen Mode2 allocation"):
        backend.load_numeric_blocks(plan)
    assert constructed is False


def test_operational_threshold_uses_one_fit60_ecdf_not_loo(monkeypatch):
    rng = np.random.default_rng(20260912)
    fit = rng.normal(size=(60, 85, 3)).astype(np.float64)
    inner = rng.normal(size=(20, 85, 3)).astype(np.float64)
    calibrator = ExpertTailCalibrator.fit(fit)
    import mate_spc_cpetc.monitor as monitor
    monkeypatch.setattr(
        monitor,
        "leave_one_out_run_maxima",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LOO forbidden")),
    )
    result = backend._operational_threshold(inner, calibrator)
    assert result["operational_loo"] is False
    assert result["procedure"] == "fixed_fit_healthy_ecdf_then_inner_rank20"
    assert len(result["run_maxima"]) == 20
    assert result["threshold"] == max(result["run_maxima"])
    assert result["rank19"] == sorted(result["run_maxima"])[-2]
    assert calibrator.calibration_run_count == 60
    assert calibrator.samples_per_expert == 5100


def _decision(alarm: bool, delay: int = 0) -> dict[str, object]:
    return {
        "alarm": alarm,
        "first_alarm_position": 1 if alarm else None,
        "first_alarm_endpoint": 164 if alarm else None,
        "attributed_fault": 3 if alarm else None,
        "delay_samples": delay if alarm else 340,
        "delay_hours": 0.05 * (delay if alarm else 340),
        "run_maximum": 3.0 if alarm else 1.0,
        "threshold": 2.0,
    }


def _passing_fold(index: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in range(20):
        winners = ([3, 9, 15] * 29)[:85]
        rows.append({
            "idv": 1, "simulation_run": index * 20 + run + 1,
            "winning_experts": winners, "decision": _decision(False),
        })
        for fault in (3, 9, 15):
            rows.append({
                "idv": fault, "simulation_run": index * 20 + run + 1,
                "winning_experts": winners, "decision": _decision(True, 4),
            })
    fold: dict[str, object] = {
        "outer_rows": rows,
        "inner_threshold": {"nondegenerate": True},
        "parity": {"passes": True},
    }
    fold["metrics"] = backend._fold_metrics(rows)
    return fold


def test_all_preregistered_gates_are_jointly_required():
    folds = [_passing_fold(index) for index in range(5)]
    aggregate = backend._aggregate_metrics(folds)
    report = backend._gate_report(folds, aggregate)
    assert report["all_pass"] is True
    folds[0]["inner_threshold"]["nondegenerate"] = False
    assert backend._gate_report(folds, aggregate)["all_pass"] is False
    folds[0]["inner_threshold"]["nondegenerate"] = True
    folds[0]["parity"]["passes"] = False
    assert backend._gate_report(folds, aggregate)["all_pass"] is False


@dataclass(frozen=True)
class FakeRun:
    source_idv: int
    simulation_run: int

    @property
    def name(self) -> str:
        return f"synthetic-idv-{self.source_idv}-run-{self.simulation_run}"

    @property
    def fault_id(self) -> int:
        return 0 if self.source_idv == 1 else self.source_idv

    @property
    def rng_seed(self) -> int:
        return self.source_idv * 1000 + self.simulation_run


@dataclass(frozen=True)
class FakeBlock:
    simulation_run: int
    conditions: tuple[FakeRun, ...]


class FakeFit:
    def __init__(self, count: int):
        self.training_run_count = count
        self.healthy_run_count = count // 4
        self.requested_device = "cuda"
        self.dtype = "float32"
        self.ae_loss_history = (1.0,) * 40
        self.mil_loss_history = (1.0,) * 60


def test_synthetic_development_exact_roles_no_formal_model_on_no_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    blocks = tuple(
        FakeBlock(run, tuple(FakeRun(idv, run) for idv in (1, 3, 9, 15)))
        for run in range(1, 101)
    )
    fitted_roles: list[dict[int, set[int]]] = []

    def fake_fit(runs, *, device):
        assert device == "cuda"
        fitted_roles.append({
            idv: {_run.simulation_run for _run in runs if _run.source_idv == idv}
            for idv in (1, 3, 9, 15)
        })
        return FakeFit(len(runs))

    def fake_scores(_fitted, runs, *, device):
        values = np.empty((len(runs), 93, 3), dtype=np.float64)
        endpoints = np.arange(93, dtype=np.float64)[:, None]
        for index, run in enumerate(runs):
            values[index] = (
                run.simulation_run * 0.017 + endpoints * 0.003
                + np.asarray((0.0, 0.011, 0.023))[None, :]
            )
            if run.source_idv != 1:
                values[index, 8:] += 2.0
        return values

    monkeypatch.setattr(backend, "_require_cuda", lambda _device=None, **_kw: None)
    monkeypatch.setattr(backend, "cuda_preflight", lambda: {"cuda_available": True})
    monkeypatch.setattr(
        backend, "load_numeric_blocks", lambda _plan: ({"synthetic": True}, blocks)
    )
    monkeypatch.setattr(backend, "_fit_model", fake_fit)
    monkeypatch.setattr(backend, "_score_model", fake_scores)
    monkeypatch.setattr(
        backend, "_parameter_counts",
        lambda _fitted: {
            "classifier_exact": 9207, "autoencoder_exact": 6604,
            "total_exact": 15811, "trainable_at_checkpoint": 9207,
        },
    )
    monkeypatch.setattr(
        backend, "_save_fold_artifacts",
        lambda **_kwargs: {"ecdf_sha256": "e" * 64, "threshold_sha256": "t" * 64},
    )
    monkeypatch.setattr(
        backend, "_gate_report", lambda _folds, _aggregate: {"all_pass": False}
    )
    monkeypatch.setattr(backend.torch.cuda, "empty_cache", lambda: None)

    result = backend.development_iteration(
        access_plan=backend.development_access_plan(),
        fold_map=backend._canonical_fold_map(),
        artifact_root=tmp_path / "artifacts",
        device="cuda",
    )
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert result["selected_candidate_id"] is None
    assert result["formal_mode2_refit"] is None
    assert len(fitted_roles) == 5
    for outer, roles in enumerate(fitted_roles):
        expected = set(backend.fold_roles(outer)["fit"])
        assert all(roles[idv] == expected for idv in roles)
        assert all(len(roles[idv]) == 60 for idv in roles)
    identities = {
        (row["idv"], row["simulation_run"])
        for fold in result["folds"] for row in fold["outer_rows"]
    }
    assert len(identities) == 400
    assert not (tmp_path / "artifacts" / "formal_mode2_refit").exists()


def test_arx_numeric_guard_is_durable_development_no_go_before_outer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    blocks = tuple(
        FakeBlock(run, tuple(FakeRun(idv, run) for idv in (1, 3, 9, 15)))
        for run in range(1, 101)
    )
    scored = False

    def forbidden_score(*_args, **_kwargs):
        nonlocal scored
        scored = True
        raise AssertionError("outer scoring is forbidden after ARX numeric NO-GO")

    monkeypatch.setattr(backend, "_require_cuda", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backend, "cuda_preflight", lambda: {"cuda_available": True})
    monkeypatch.setattr(
        backend, "load_numeric_blocks", lambda _plan: ({"synthetic": True}, blocks)
    )
    monkeypatch.setattr(
        backend, "_fit_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("ARX ridge system exceeds the frozen condition-number guard")
        ),
    )
    monkeypatch.setattr(backend, "_score_model", forbidden_score)
    result = backend.development_iteration(
        access_plan=backend.development_access_plan(),
        fold_map=backend._canonical_fold_map(),
        artifact_root=tmp_path / "numeric-no-go",
        device="cuda",
    )
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert result["termination"]["kind"] == "FOLD_NUMERIC_NO_GO"
    assert result["termination"]["outer_data_scored"] is False
    assert scored is False
    assert Path(result["termination"]["record_path"]).is_file()
