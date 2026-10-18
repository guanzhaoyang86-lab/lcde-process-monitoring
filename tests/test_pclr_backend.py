from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

# This environment's Windows CUDA stack requires PyTorch's runtime to be
# loaded before NumPy's BLAS runtime.
torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

import mate_spc_pclr.backend as backend  # noqa: E402
from mate_spc_pclr import data  # noqa: E402
from mate_spc_pclr.model import (  # noqa: E402
    CAUSAL_LAGS,
    FEATURE_COUNT,
    PCLR_FEATURE_COUNT,
    PCLRConfig,
    PCLRFeatureModel,
    PCLRHeads,
    PCLRModel,
    load_pclr_model,
)


def _run(simulation_run: int, idv: int, *, partition: str = data.DEVELOPMENT) -> data.PCLRRun:
    seed = simulation_run * 100 + idv
    rng = np.random.default_rng(seed)
    time = np.arange(500, dtype=np.float64)[:, None]
    channels = np.arange(52, dtype=np.float64)[None, :]
    values = (
        0.05 * rng.standard_normal((500, 52))
        + np.sin(time / (11.0 + channels / 10.0))
        + 0.001 * channels
    )
    if idv != 1:
        values[160:, (idv * 3) % 52] += 1.0 + idv / 10.0
    values = np.asarray(values, dtype=np.float64, order="C")
    values.setflags(write=False)
    fault = np.zeros(500, dtype=np.bool_)
    if idv != 1:
        fault[160:] = True
    fault.setflags(write=False)
    return data.PCLRRun(
        name=f"synthetic-{partition}-idv-{idv:02d}-run-{simulation_run:03d}",
        x=values,
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


def _group(simulation_run: int) -> data.DevelopmentGroup:
    return data.DevelopmentGroup(
        simulation_run,
        tuple(_run(simulation_run, idv) for idv in data.CONDITION_IDVS),
    )


def _dummy_feature_model(*, ridge: float = 1.0e-2) -> PCLRFeatureModel:
    rows = 1 + FEATURE_COUNT * len(CAUSAL_LAGS)
    return PCLRFeatureModel(
        mad_floor=np.ones(FEATURE_COUNT, dtype=np.float64),
        var_coefficients=np.zeros((rows, FEATURE_COUNT), dtype=np.float64),
        config=PCLRConfig(var_ridge_alpha=ridge),
    )


def _heads(feature_count: int = PCLR_FEATURE_COUNT) -> PCLRHeads:
    weights = np.zeros((3, feature_count), dtype=np.float64)
    weights[:, 0] = 1.0
    return PCLRHeads(
        weights=weights,
        bias=np.zeros(3, dtype=np.float64),
        healthy_logit_center=np.zeros(3, dtype=np.float64),
        healthy_logit_scale=np.ones(3, dtype=np.float64),
        kappa=np.zeros(3, dtype=np.float64),
    )


def test_malformed_access_plan_is_rejected_before_archive_construction(
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
    partial = list(backend._PLANS[data.CALIBRATION])[:-1]
    with pytest.raises(ValueError, match="exact canonical"):
        backend.load_numeric_runs(data.MODE1_FILE_NAME, partial)
    assert constructed is False


def test_load_numeric_runs_follows_exact_plan_and_returns_run_keyed_groups(
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
    assert tuple(result) == data.DEVELOPMENT_RUNS
    assert result[201].by_idv(15).source_idv == 15
    assert materialized == [
        (data.DEVELOPMENT, int(row["idv"]), int(row["simulation_run"]))
        for row in plan
    ]


def test_fold_feature_fit_and_transforms_are_group_disjoint_and_prefix_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    training = {201: _group(201), 202: _group(202)}
    validation = {203: _group(203)}
    fitted = backend.fit_fold_feature_model(
        train_groups=training,
        varx_ridge=0.1,
        fold=2,
        device="cuda",
        dtype="float64",
    )
    assert fitted.training_group_keys == (201, 202)
    assert fitted.model.config.var_ridge_alpha == 0.1
    train_features = backend.transform_fold_groups(
        feature_model=fitted,
        groups=training,
        role="training",
        fold=2,
        device="cuda",
        dtype="float64",
        batch_size=2,
    )
    validation_features = backend.transform_fold_groups(
        feature_model=fitted,
        groups=validation,
        role="validation",
        fold=2,
        device="cuda",
        dtype="float64",
        batch_size=2,
    )
    assert len(train_features.monitoring) == 8
    assert len(validation_features.monitoring) == 4
    assert {value.shape for value in train_features.monitoring.values()} == {
        (340, PCLR_FEATURE_COUNT)
    }
    assert {value.shape for value in train_features.prefixes.values()} == {
        (32, PCLR_FEATURE_COUNT)
    }
    with pytest.raises(ValueError, match="overlap"):
        backend.transform_fold_groups(
            feature_model=fitted,
            groups={202: training[202]},
            role="validation",
            fold=2,
            device="cuda",
            dtype="float64",
            batch_size=1,
        )


def test_head_fit_receives_true_prefixes_and_only_training_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    feature_fit = backend.FoldFeatureFit(_dummy_feature_model(), 0, 0.01, (201,))
    monitoring = {
        f"r-{label}": np.zeros((340, PCLR_FEATURE_COUNT), dtype=np.float64)
        for label in (0, 3, 9, 15)
    }
    prefixes = {
        name: np.zeros((32, PCLR_FEATURE_COUNT), dtype=np.float64)
        for name in monitoring
    }
    transformed = backend.FoldFeatures(
        feature_fit,
        0,
        "training",
        (201,),
        MappingProxyType(monitoring),
        MappingProxyType(prefixes),
        MappingProxyType({f"r-{label}": label for label in (0, 3, 9, 15)}),
        MappingProxyType({f"r-{label}": 201 for label in (0, 3, 9, 15)}),
        2,
    )
    captured: dict[str, object] = {}

    def fake_fit(features, labels, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        assert features is transformed.monitoring
        assert labels is transformed.labels
        assert kwargs["prefix_features_by_run"] is transformed.prefixes
        return _heads(), {
            "prefix_rank_weight": 0.10,
            "prefix_rank_contract": "causal_local_128_160_mean_vs_post_160_500_top_quartile",
        }

    monkeypatch.setattr(backend, "fit_pclr_heads_cuda", fake_fit)
    result = backend.fit_fold_heads(
        train_features=transformed,
        classifier_ridge=0.1,
        fold=0,
        device="cuda",
        dtype="float64",
    )
    assert result.training_features is transformed
    assert captured["epochs"] == backend.CV_HEAD_EPOCHS
    assert captured["kappa"] == 0.0


def test_fold_score_uses_only_80_training_healthy_maxima_q79_and_core_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    batch_calls: list[tuple[int, int]] = []
    real_batch_monitor = backend.monitor_batch_torch_float64

    def tracked_batch_monitor(features, heads, **kwargs):  # type: ignore[no-untyped-def]
        batch_calls.append((int(np.asarray(features).shape[0]), int(kwargs["batch_size"])))
        return real_batch_monitor(features, heads, **kwargs)

    monkeypatch.setattr(backend, "monitor_batch_torch_float64", tracked_batch_monitor)
    feature_fit = backend.FoldFeatureFit(
        _dummy_feature_model(), 1, 0.01, tuple(range(201, 281))
    )
    train_monitoring = {
        f"healthy-{run}": np.full((340, 1), (run - 200) / 100.0, dtype=np.float64)
        for run in range(201, 281)
    }
    empty_prefix = np.zeros((32, 1), dtype=np.float64)
    training = backend.FoldFeatures(
        feature_fit,
        1,
        "training",
        tuple(range(201, 281)),
        MappingProxyType(train_monitoring),
        MappingProxyType({name: empty_prefix for name in train_monitoring}),
        MappingProxyType({name: 0 for name in train_monitoring}),
        MappingProxyType(
            {name: int(name.rsplit("-", 1)[1]) for name in train_monitoring}
        ),
        16,
    )
    fitted = backend.FoldHeads(
        _heads(feature_count=1),
        feature_fit,
        1,
        0.01,
        training,
        MappingProxyType({}),
    )
    validation_monitoring: dict[str, np.ndarray] = {}
    validation_labels: dict[str, int] = {}
    validation_runs: dict[str, int] = {}
    for run in range(281, 301):
        healthy_name = f"healthy-{run}"
        validation_monitoring[healthy_name] = np.zeros((340, 1), dtype=np.float64)
        validation_labels[healthy_name] = 0
        validation_runs[healthy_name] = run
        for offset, fault in enumerate((3, 9, 15)):
            name = f"fault-{fault}-{run}"
            validation_monitoring[name] = np.full(
                (340, 1), 1000.0 + offset, dtype=np.float64
            )
            validation_labels[name] = fault
            validation_runs[name] = run
    validation = backend.FoldFeatures(
        feature_fit,
        1,
        "validation",
        tuple(range(281, 301)),
        MappingProxyType(validation_monitoring),
        MappingProxyType({name: empty_prefix for name in validation_monitoring}),
        MappingProxyType(validation_labels),
        MappingProxyType(validation_runs),
        16,
    )
    outcome = backend.score_fold_candidate(
        heads=fitted,
        validation_features=validation,
        page_drift=0.25,
        development_threshold_rank=79,
        threshold_source="80_training_healthy_run_maxima_only",
        threshold_rule="noninterpolated_ascending_order_statistic",
        threshold_comparison="strictly_greater_than",
        fold=1,
        device="cpu",
        dtype="float64",
    )
    assert outcome["training_healthy_cpu_run_maxima"].shape == (80,)
    assert outcome["training_healthy_cpu_run_maxima"].dtype == np.float64
    assert outcome["operational_threshold_q79"] == np.sort(
        outcome["training_healthy_cpu_run_maxima"]
    )[78]
    assert len(outcome["healthy_records"]) == 20
    assert outcome["healthy_records"][0] == {
        "simulation_run": 281,
        "fold": 1,
        "alarmed": False,
    }
    assert len(outcome["records"]) == 60
    assert all(row["detected"] for row in outcome["records"])
    assert all(row["restricted_delay_samples"] == 2 for row in outcome["records"])
    assert outcome["parity"]["alarm_exact"] is True
    assert outcome["parity"]["max_abs_score_error"] <= backend.PARITY_ATOL
    assert batch_calls == [(80, 16), (80, 16)]


def test_refit_returns_exact_safe_npz_payload_loadable_without_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend, "_require_cuda_float64", lambda **kwargs: None)
    dummy = _dummy_feature_model(ridge=0.1)
    monkeypatch.setattr(
        backend.PCLRFeatureModel,
        "fit",
        classmethod(lambda cls, runs, config: PCLRFeatureModel(dummy.mad_floor, dummy.var_coefficients, config)),
    )
    monitor = np.zeros((340, PCLR_FEATURE_COUNT), dtype=np.float64)
    prefix = np.zeros((32, PCLR_FEATURE_COUNT), dtype=np.float64)
    monkeypatch.setattr(
        backend.PCLRFeatureModel,
        "transform_training_runs",
        lambda self, runs: (
            {name: monitor for name in runs},
            {name: prefix for name in runs},
        ),
    )
    captured: dict[str, object] = {}

    def fake_fit(features, labels, **kwargs):  # type: ignore[no-untyped-def]
        del features, labels
        captured.update(kwargs)
        return _heads().with_kappa(kwargs["kappa"]), {"synthetic": True}

    monkeypatch.setattr(backend, "fit_pclr_heads_cuda", fake_fit)
    arrays = backend.refit_selected_model(
        raw_development_groups={201: _group(201)},
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
    path = tmp_path / "pclr_model.npz"
    np.savez_compressed(path, **arrays)
    model, metadata = load_pclr_model(path)
    assert model.feature_model.config.var_ridge_alpha == 0.1
    np.testing.assert_array_equal(model.heads.kappa, np.full(3, 0.5))
    assert model.heads.threshold is None
    assert metadata["fit_all_components_from_scratch"] is True
    assert metadata["development_runs"] == [201, 201]
    assert metadata["final_refit_steps"] == 7
    assert captured["epochs"] == 7
    assert captured["prefix_features_by_run"] is not None
    assert all(np.asarray(value).dtype.kind != "O" for value in arrays.values())


def test_score_paths_runs_full_cuda_float64_and_pads_core_monitor_geometry(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = PCLRModel(_dummy_feature_model(), _heads().with_kappa(0.25))
    model_path = tmp_path / "pclr_model.npz"
    model.save_npz(model_path, metadata={"synthetic": True})
    run = _run(301, 1, partition=data.CALIBRATION)
    batch = backend.NumericRunBatch(data.CALIBRATION, (run,), ((1, 301),))
    result = backend.score_paths(
        model_path=model_path,
        data=batch,
        device="cuda",
        dtype="float64",
        batch_size=1,
    )
    assert result["cpu_standardized_logits"].shape == (1, 500, 3)
    assert result["cpu_scores"].shape == (1, 500)
    assert not result["cpu_eligible"][0, :162].any()
    assert result["cpu_eligible"][0, 162:].all()
    assert not result["cpu_cusum"][0, :160].any()
    for field in ("standardized_logits", "cusum", "component_persistence", "scores"):
        assert np.max(
            np.abs(result[f"cpu_{field}"] - result[f"cuda_{field}"])
        ) <= backend.PARITY_ATOL
    np.testing.assert_array_equal(result["cpu_eligible"], result["cuda_eligible"])


def test_backend_source_has_no_real_archive_path_or_import_time_hdf5_read() -> None:
    source = Path(backend.__file__).read_text(encoding="utf-8")
    assert "data/extended_tep_dtu_mode1" not in source
    assert "TEP_Mode1.h5" not in source
    assert "h5py.File" not in source
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


def test_fresh_process_direct_backend_import_preloads_torch_before_numpy() -> None:
    completed = _fresh_import(
        "import sys; import mate_spc_pclr.backend as b; "
        "assert b._torch is not None; assert 'torch' in sys.modules; "
        "assert 'numpy' in sys.modules"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fresh_process_metadata_data_import_does_not_force_torch_or_cuda() -> None:
    completed = _fresh_import(
        "import sys; import mate_spc_pclr.data; "
        "assert 'numpy' in sys.modules; assert 'torch' not in sys.modules"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
