from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

# The Windows CUDA environment must load PyTorch before NumPy.
pytest.importorskip("torch")

import numpy as np  # noqa: E402

import mate_spc_pclr.backend_v2 as backend  # noqa: E402
from mate_spc_pclr import data_v2 as data  # noqa: E402
from mate_spc_pclr.model import (  # noqa: E402
    CAUSAL_LAGS,
    FEATURE_COUNT,
    PCLR_FEATURE_COUNT,
    PCLRConfig,
    PCLRFeatureModel,
    PCLRHeads,
    load_pclr_model,
)


def _run(
    simulation_run: int,
    idv: int,
    *,
    partition: str = data.DEVELOPMENT,
    magnitude_percent: int | None = None,
    rng_seed: int | None = None,
) -> data.PCLRRun:
    magnitude = (
        int(data.PARTITION_MAGNITUDES[partition])
        if magnitude_percent is None
        else int(magnitude_percent)
    )
    seed = (
        magnitude * 100_000 + idv * 1_000 + simulation_run
        if rng_seed is None
        else int(rng_seed)
    )
    values = np.zeros((500, 52), dtype=np.float64)
    values[:, 0] = np.linspace(0.0, 1.0, 500)
    values.setflags(write=False)
    fault = np.zeros(500, dtype=np.bool_)
    if idv != 1:
        fault[160:] = True
    fault.setflags(write=False)
    return data.PCLRRun(
        name=(
            f"synthetic-{partition}-magnitude-{magnitude}-"
            f"idv-{idv}-run-{simulation_run}"
        ),
        x=values,
        fault=fault,
        feature_names=data.FEATURE_NAMES,
        metadata=MappingProxyType(
            {
                "partition": partition,
                "source_idv": idv,
                "simulation_run": simulation_run,
                "group_key": simulation_run,
                "fault_magnitude_percent": magnitude,
                "rng_seed": seed,
            }
        ),
    )


def _group(simulation_run: int) -> data.DevelopmentGroup:
    return data.DevelopmentGroup(
        simulation_run,
        tuple(_run(simulation_run, idv) for idv in data.CONDITION_IDVS),
    )


def _dummy_feature_model(*, ridge: float = 0.1) -> PCLRFeatureModel:
    rows = 1 + FEATURE_COUNT * len(CAUSAL_LAGS)
    return PCLRFeatureModel(
        mad_floor=np.ones(FEATURE_COUNT, dtype=np.float64),
        var_coefficients=np.zeros((rows, FEATURE_COUNT), dtype=np.float64),
        config=PCLRConfig(var_ridge_alpha=ridge),
    )


def _heads() -> PCLRHeads:
    return PCLRHeads(
        weights=np.zeros((3, PCLR_FEATURE_COUNT), dtype=np.float64),
        bias=np.zeros(3, dtype=np.float64),
        healthy_logit_center=np.zeros(3, dtype=np.float64),
        healthy_logit_scale=np.ones(3, dtype=np.float64),
        kappa=np.zeros(3, dtype=np.float64),
    )


def test_canonical_plans_have_exact_magnitude_aware_contract() -> None:
    assert len(backend._PLANS[data.DEVELOPMENT]) == 400
    assert len(backend._PLANS[data.CALIBRATION]) == 100
    assert len(backend._PLANS[data.FINAL]) == 400
    required = {
        "idv",
        "simulation_run",
        "magnitude_percent",
        "window_half_open",
        "role",
    }
    for partition, magnitude in (
        (data.DEVELOPMENT, 50),
        (data.CALIBRATION, 25),
        (data.FINAL, 75),
    ):
        plan = backend._PLANS[partition]
        assert all(set(row) == required for row in plan)
        assert {int(row["magnitude_percent"]) for row in plan} == {magnitude}
    first_block = backend._PLANS[data.DEVELOPMENT][:4]
    assert [(row["simulation_run"], row["idv"]) for row in first_block] == [
        (1, 1),
        (1, 3),
        (1, 9),
        (1, 15),
    ]
    assert [row["idv"] for row in backend._PLANS[data.CALIBRATION]] == [1] * 100


def test_malformed_plan_is_rejected_before_archive_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    class ForbiddenArchive:
        def __init__(self, path: Path) -> None:
            del path
            nonlocal constructed
            constructed = True
            raise AssertionError("malformed plan reached the HDF5 boundary")

    monkeypatch.setattr(backend, "PCLRMode1Archive", ForbiddenArchive)
    missing_magnitude = [dict(row) for row in backend._PLANS[data.CALIBRATION]]
    missing_magnitude[0].pop("magnitude_percent")
    with pytest.raises(ValueError, match="fields differ"):
        backend.load_numeric_runs(data.MODE1_FILE_NAME, missing_magnitude)
    wrong_magnitude = [dict(row) for row in backend._PLANS[data.CALIBRATION]]
    wrong_magnitude[0]["magnitude_percent"] = 50
    with pytest.raises(ValueError, match="exact canonical"):
        backend.load_numeric_runs(data.MODE1_FILE_NAME, wrong_magnitude)
    assert constructed is False


def test_development_load_preserves_block_api_and_distinct_seed_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized: list[tuple[str, int, int]] = []

    class FakeArchive:
        def __init__(self, path: Path) -> None:
            assert path.name == data.MODE1_FILE_NAME

        def _open(self):  # type: ignore[no-untyped-def]
            return nullcontext(object())

    def fake_materialize(
        handle: object, *, partition: str, idv: int, simulation_run: int
    ) -> data.PCLRRun:
        del handle
        materialized.append((partition, idv, simulation_run))
        return _run(simulation_run, idv, partition=partition)

    monkeypatch.setattr(backend, "PCLRMode1Archive", FakeArchive)
    monkeypatch.setattr(backend.pclr_data, "validate_processdata_labels", lambda handle: ())
    monkeypatch.setattr(backend.pclr_data, "_materialize_run", fake_materialize)
    plan = list(backend._PLANS[data.DEVELOPMENT])
    result = backend.load_numeric_runs(data.MODE1_FILE_NAME, plan)
    assert isinstance(result, MappingProxyType)
    assert tuple(result) == tuple(range(1, 101))
    assert tuple(item.source_idv for item in result[1].conditions) == (1, 3, 9, 15)
    assert len({item.rng_seed for item in result[1].conditions}) == 4
    assert materialized == [
        (data.DEVELOPMENT, int(row["idv"]), int(row["simulation_run"]))
        for row in plan
    ]


def test_numeric_batch_binds_magnitude_order_and_rng_identity() -> None:
    runs = (
        _run(2, 1, partition=data.CALIBRATION),
        _run(1, 1, partition=data.CALIBRATION),
    )
    batch = backend.NumericRunBatch(
        data.CALIBRATION,
        runs,
        ((25, 1, 2), (25, 1, 1)),
    )
    assert batch.access_keys == ((25, 1, 2), (25, 1, 1))
    with pytest.raises(ValueError, match="magnitude-aware"):
        backend.NumericRunBatch(
            data.CALIBRATION,
            runs,
            ((50, 1, 2), (25, 1, 1)),
        )
    duplicate_identity = (
        _run(1, 1, partition=data.CALIBRATION, rng_seed=7),
        _run(2, 1, partition=data.CALIBRATION, rng_seed=7),
    )
    with pytest.raises(ValueError, match="RNG seeds must be distinct"):
        backend.NumericRunBatch(
            data.CALIBRATION,
            duplicate_identity,
            ((25, 1, 1), (25, 1, 2)),
        )


def test_refit_metadata_binds_v2_magnitude_and_grouping_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    dummy = _dummy_feature_model()
    monkeypatch.setattr(
        backend.PCLRFeatureModel,
        "fit",
        classmethod(
            lambda cls, runs, config: PCLRFeatureModel(
                dummy.mad_floor, dummy.var_coefficients, config
            )
        ),
    )
    monitoring = np.zeros((340, PCLR_FEATURE_COUNT), dtype=np.float64)
    prefix = np.zeros((32, PCLR_FEATURE_COUNT), dtype=np.float64)
    monkeypatch.setattr(
        backend.PCLRFeatureModel,
        "transform_training_runs",
        lambda self, runs: (
            {name: monitoring for name in runs},
            {name: prefix for name in runs},
        ),
    )

    def fake_head_fit(features, labels, **kwargs):  # type: ignore[no-untyped-def]
        del features, labels
        return _heads().with_kappa(kwargs["kappa"]), {"synthetic": True}

    monkeypatch.setattr(backend, "fit_pclr_heads_cuda", fake_head_fit)
    arrays = backend.refit_selected_model(
        raw_development_groups={1: _group(1)},
        selected_params={
            "varx_ridge": 0.1,
            "classifier_ridge": 0.01,
            "page_drift": 0.5,
        },
        steps=7,
        fit_all_components_from_scratch=True,
        device="cuda",
        dtype="float64",
    )
    path = tmp_path / "pclr-v2.npz"
    np.savez_compressed(path, **arrays)
    _, metadata = load_pclr_model(path)
    assert metadata["backend"] == "mate_spc_pclr.backend_v2"
    assert metadata["development_magnitude_percent"] == 50
    assert metadata["grouping_semantics"] == "balanced_stratification_block_not_crn"
    assert metadata["development_runs"] == [1, 1]
    assert json.loads(str(arrays["metadata_json"].item())) == metadata


def test_score_paths_appends_identity_arrays_in_exact_run_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    runs = (
        _run(9, 15, partition=data.FINAL),
        _run(3, 1, partition=data.FINAL),
        _run(7, 3, partition=data.FINAL),
    )
    batch = backend.NumericRunBatch(
        data.FINAL,
        runs,
        ((75, 15, 9), (75, 1, 3), (75, 3, 7)),
    )
    sentinel = {"cpu_scores": np.zeros((3, 500), dtype=np.float64)}
    captured: dict[str, object] = {}

    def fake_score_paths(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return dict(sentinel)

    numeric_core = backend._load_numeric_core()
    monkeypatch.setattr(numeric_core, "score_paths", fake_score_paths)
    result = backend.score_paths(
        model_path="synthetic.npz",
        data=batch,
        device="cuda",
        dtype="float64",
        batch_size=2,
    )
    np.testing.assert_array_equal(result["source_idvs"], [15, 1, 3])
    np.testing.assert_array_equal(result["simulation_runs"], [9, 3, 7])
    np.testing.assert_array_equal(result["magnitude_percent"], [75, 75, 75])
    np.testing.assert_array_equal(
        result["rng_seeds"], [run.rng_seed for run in runs]
    )
    assert all(result[name].dtype == np.dtype(np.int64) for name in (
        "source_idvs",
        "simulation_runs",
        "magnitude_percent",
        "rng_seeds",
    ))
    legacy = captured["data"]
    assert legacy.access_keys == ((15, 9), (1, 3), (3, 7))


def test_backend_v2_source_is_archive_path_free_and_reuses_core_model() -> None:
    source = Path(backend.__file__).read_text(encoding="utf-8")
    assert "data/extended_tep_dtu_mode1" not in source
    assert "TEP_Mode1.h5" not in source
    assert "h5py.File" not in source
    assert "from .model import" in source
    assert source.index("import torch as _torch") < source.index("import numpy as np")


def _fresh_import(code: str) -> subprocess.CompletedProcess[str]:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_root = str(project_root / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else source_root + os.pathsep + inherited
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_numpy_first_metadata_import_is_clean_and_does_not_load_numeric_core() -> None:
    completed = _fresh_import(
        "import sys; import numpy; import mate_spc_pclr.backend_v2 as b; "
        "assert b._torch is None; "
        "assert 'mate_spc_pclr.backend' not in sys.modules; print('clean')"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "clean"
    assert completed.stderr == ""


def test_fresh_numeric_import_preloads_torch_before_numpy() -> None:
    completed = _fresh_import(
        "import sys; import mate_spc_pclr.backend_v2 as b; "
        "assert b._torch is not None; assert 'torch' in sys.modules; "
        "assert 'numpy' in sys.modules"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
