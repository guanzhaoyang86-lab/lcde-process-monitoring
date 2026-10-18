from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_mstcn_spc_gpu_v3.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_mstcn_spc_gpu_v3_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture
def synthetic_workspace(tmp_path: Path, runner):
    project = tmp_path / "project"
    for relative in runner.IMPLEMENTATION_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "outputs" / "v3"
    protocol = tmp_path / "protocols" / "formal.json"
    source_record = {
        "name": runner.DATA_FILE,
        "absolute_path": str((data / runner.DATA_FILE).resolve()),
        "size_bytes": 123,
        "sha256": "1" * 64,
        "md5": "2" * 32,
    }
    metadata_record = {
        "synthetic": True,
        "numeric_values_decoded": False,
        "selected_run_count": 900,
    }
    return project, data, output, protocol, source_record, metadata_record


class SyntheticBackend:
    def __init__(self, runner, output_root: Path):
        self.runner = runner
        self.output_root = output_root
        self.calls: list[str] = []

    @staticmethod
    def _runtime():
        return {
            "cuda_available": True,
            "device_type": "cuda",
            "dtype": "float64",
            "torch_version": "synthetic",
            "cuda_version": "synthetic",
            "device_name": "synthetic RTX 5090",
        }

    def cuda_preflight(self):
        return self._runtime()

    def _development_rows(self, runs):
        rows = []
        for idv in self.runner.ALL_IDS:
            for run in runs:
                alarm = idv != 1
                rows.append(
                    {
                        "idv": idv,
                        "simulation_run": run,
                        "run_maximum": 1.0 if alarm else 0.5,
                        "parity_max_abs_error": 0.0,
                        "alarm": alarm,
                        "first_alarm_index": 180 if alarm else None,
                        "attribution": idv if alarm else None,
                    }
                )
        return rows

    def development_iteration(
        self, *, source_path, access_plan, fold_map, config, model_path, device, dtype
    ):
        assert device == "cuda" and dtype == "float64"
        assert len(access_plan) == 400
        assert {row["magnitude_percent"] for row in access_plan} == {50}
        assert (model_path.parent / "access_intent.json").is_file()
        self.calls.append("development")
        candidates = []
        for identity in config["candidate_grid"]:
            folds = []
            for fold in range(5):
                runs = sorted(run for run, assigned in fold_map.items() if assigned == fold)
                folds.append(
                    {
                        "fold": fold,
                        "training_healthy_maxima": [index / 100.0 for index in range(80)],
                        "rows": self._development_rows(runs),
                        "score_parity_max_abs_error": 0.0,
                        "discrete_parity_equal": True,
                    }
                )
            candidates.append({**identity, "folds": folds})
        model_path.write_bytes(b"synthetic safe model")
        return {
            "config": copy.deepcopy(config),
            "candidates": candidates,
            "selected_candidate_id": "q250-rho250",
            "runtime": self._runtime(),
            "model_metadata": {"synthetic": True},
            "model_parity_max_abs_error": 0.0,
        }

    def calibrate(self, *, source_path, access_plan, model_path, config, device, dtype):
        assert device == "cuda" and dtype == "float64"
        assert len(access_plan) == 100
        assert {row["idv"] for row in access_plan} == {1}
        assert {row["magnitude_percent"] for row in access_plan} == {25}
        assert (self.output_root / ".calibrate.staging" / "access_intent.json").is_file()
        self.calls.append("calibrate")
        return {
            "config": copy.deepcopy(config),
            "model_parity_max_abs_error": 0.0,
            "discrete_parity_equal": True,
            "rows": [
                {
                    "idv": 1,
                    "simulation_run": run,
                    "run_maximum": run / 100.0,
                    "parity_max_abs_error": 0.0,
                }
                for run in range(1, 101)
            ],
        }

    def confirm(
        self, *, source_path, access_plan, model_path, config, threshold, device, dtype
    ):
        assert device == "cuda" and dtype == "float64"
        assert threshold == pytest.approx(0.98)
        assert len(access_plan) == 400
        assert {row["magnitude_percent"] for row in access_plan} == {75}
        assert (self.output_root / ".confirm.staging" / "access_intent.json").is_file()
        self.calls.append("confirm")
        rows = []
        for idv in self.runner.ALL_IDS:
            for run in range(1, 101):
                alarm = idv != 1
                rows.append(
                    {
                        "idv": idv,
                        "simulation_run": run,
                        "run_maximum": 1.5 if alarm else 0.5,
                        "parity_max_abs_error": 0.0,
                        "alarm": alarm,
                        "first_alarm_index": 180 if alarm else None,
                        "attribution": idv if alarm else None,
                    }
                )
        return {
            "config": copy.deepcopy(config),
            "threshold": threshold,
            "rows": rows,
            "model_parity_max_abs_error": 0.0,
            "discrete_parity_equal": True,
        }


def _verify_synthetic_model(path: Path, config):
    assert path.read_bytes() == b"synthetic safe model"
    assert config["feature_transform"]["input_features"] == 512
    return {"synthetic": True}


def _development_and_freeze(runner, workspace):
    project, data, output, protocol, source, metadata = workspace
    backend = SyntheticBackend(runner, output)
    result = runner.run_development_iteration(
        project,
        data,
        output,
        "iteration-001",
        runner.default_model_config(),
        backend=backend,
        source_record=source,
        verify_model=_verify_synthetic_model,
    )
    assert result["decision"] == "PASS"
    assert result["selected_candidate_id"] == "q250-rho250"
    iteration = output / "development_iterations" / "iteration-001"
    digest = runner.freeze_formal(
        project,
        data,
        output,
        iteration,
        protocol,
        source_record=source,
        source_metadata_record=metadata,
        verify_model=_verify_synthetic_model,
    )
    return backend, digest


def test_metadata_only_import_does_not_load_numpy_or_torch():
    code = (
        "import runpy,sys;"
        f"runpy.run_path({str(RUNNER_PATH)!r},run_name='metadata_probe');"
        "assert 'numpy' not in sys.modules;"
        "assert 'torch' not in sys.modules;"
        "assert 'mate_spc_mstcn.backend_v3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_exact_config_and_four_candidate_grid(runner):
    config = runner.validate_model_config(runner.default_model_config())
    assert config["feature_transform"] == {
        "feature_schema": "mstcn_causal_haar_loop_v2",
        "input_features": 512,
        "commissioning_baseline_half_open": [0, 128],
        "fit_scope": "fold_training_run_prefixes_only",
        "causal_no_future_rows": True,
    }
    assert len(config["candidate_grid"]) == 4
    assert config["training"]["dtype"] == "float64"
    assert config["scan"]["persistence_samples"] == 2
    mutated = copy.deepcopy(config)
    mutated["training"]["epochs"] = 31
    with pytest.raises(ValueError, match="exact audited"):
        runner.validate_model_config(mutated)
    missing = copy.deepcopy(config)
    del missing["training"]["healthy_detection_loss_weight"]
    with pytest.raises(ValueError, match="exact audited"):
        runner.validate_model_config(missing)

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
            runner.validate_model_config(malformed)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("architecture", "channels", 32.0),
        ("architecture", "detector_output_count", True),
        ("training", "epochs", 30.0),
        ("feature_transform", "causal_no_future_rows", 1),
    ],
)
def test_exact_config_rejects_json_numeric_type_drift(
    runner, section, field, replacement
):
    mutated = runner.default_model_config()
    mutated[section][field] = replacement
    with pytest.raises(ValueError, match="exact audited"):
        runner.validate_model_config(mutated)


def test_access_plans_are_disjoint_and_quarantine_25_percent_faults(runner):
    development = runner._development_access_plan()
    calibration = runner._calibration_access_plan()
    confirmation = runner._confirmation_access_plan()
    assert len(development) == 400 and {row["magnitude_percent"] for row in development} == {50}
    assert len(calibration) == 100
    assert {(row["magnitude_percent"], row["idv"]) for row in calibration} == {(25, 1)}
    assert all(row["window_half_open"] == [100, 600] for row in calibration)
    assert len(confirmation) == 400 and {row["magnitude_percent"] for row in confirmation} == {75}
    for phase, plan in (
        ("development", development),
        ("calibrate", calibration),
        ("confirm", confirmation),
    ):
        runner._validate_access_plan(phase, plan)
    poisoned = copy.deepcopy(calibration)
    poisoned[0]["idv"] = 3
    with pytest.raises(ValueError):
        runner._validate_access_plan("calibrate", poisoned)


def test_repeatable_development_uses_distinct_iteration_ids(runner, synthetic_workspace):
    project, data, output, _, source, _ = synthetic_workspace
    first = SyntheticBackend(runner, output)
    result = runner.run_development_iteration(
        project, data, output, "a", runner.default_model_config(),
        backend=first, source_record=source, verify_model=_verify_synthetic_model,
    )
    assert result["decision"] == "PASS"
    with pytest.raises(FileExistsError):
        runner.run_development_iteration(
            project, data, output, "a", runner.default_model_config(),
            backend=first, source_record=source, verify_model=_verify_synthetic_model,
        )
    second = SyntheticBackend(runner, output)
    runner.run_development_iteration(
        project, data, output, "b", runner.default_model_config(),
        backend=second, source_record=source, verify_model=_verify_synthetic_model,
    )
    assert (output / "development_iterations" / "a" / "completion.json").is_file()
    assert (output / "development_iterations" / "b" / "completion.json").is_file()
    assert not (output / "calibrate").exists()


def test_development_no_go_emits_no_model(runner, synthetic_workspace):
    project, data, output, _, source, _ = synthetic_workspace

    class NoGoBackend(SyntheticBackend):
        def development_iteration(self, **kwargs):
            raw = super().development_iteration(**kwargs)
            kwargs["model_path"].unlink()
            for candidate in raw["candidates"]:
                for fold in candidate["folds"]:
                    for row in fold["rows"]:
                        if row["idv"] == 9:
                            row.update(
                                run_maximum=0.5,
                                alarm=False,
                                first_alarm_index=None,
                                attribution=None,
                            )
            raw["selected_candidate_id"] = None
            raw["model_metadata"] = None
            raw["model_parity_max_abs_error"] = None
            return raw

    result = runner.run_development_iteration(
        project, data, output, "no-go", runner.default_model_config(),
        backend=NoGoBackend(runner, output), source_record=source,
        verify_model=_verify_synthetic_model,
    )
    target = output / "development_iterations" / "no-go"
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert result["selected_candidate_id"] is None
    assert not (target / "mstcn_model.npz").exists()
    assert "mstcn_model.npz" not in json.loads(
        (target / "completion.json").read_text(encoding="utf-8")
    )["artifacts"]


def test_formal_protocol_is_atomic_bound_and_tamper_evident(runner, synthetic_workspace):
    _, digest = _development_and_freeze(runner, synthetic_workspace)
    _, _, output, protocol, _, _ = synthetic_workspace
    loaded, actual = runner._load_protocol(protocol, digest)
    assert actual == digest
    assert loaded["partition_governance"]["bound_output_root"] == str(output.resolve())
    assert loaded["development_binding"]["selected_candidate"]["candidate_id"] == "q250-rho250"
    assert "src/mate_spc_mstcn/backend_v3.py" in loaded["implementation_records"]
    assert "src/mate_spc_mstcn/monitor.py" in loaded["implementation_records"]
    protocol.write_bytes(protocol.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        runner._load_protocol(protocol, digest)


def test_calibration_rank98_and_confirmation_end_to_end(runner, synthetic_workspace):
    backend, digest = _development_and_freeze(runner, synthetic_workspace)
    project, data, output, protocol, source, metadata = synthetic_workspace
    calibration = runner.run_calibration(
        project,
        data,
        output,
        protocol,
        digest,
        backend=backend,
        source_record=source,
        source_metadata_record=metadata,
        verify_model=_verify_synthetic_model,
    )
    assert calibration["summary"]["decision"] == "PASS"
    assert calibration["summary"]["threshold"] == pytest.approx(0.98)
    assert calibration["summary"]["finite_sample_marginal_exceedance_bound"] == "3/101"
    confirmation = runner.run_confirmation(
        project,
        data,
        output,
        protocol,
        digest,
        backend=backend,
        source_record=source,
        source_metadata_record=metadata,
        verify_model=_verify_synthetic_model,
    )
    assert confirmation["decision"] == "PASS"
    assert confirmation["metrics"]["healthy_alarm_runs"] == 0
    assert all(
        confirmation["metrics"]["faults"][str(fault)]["detected_runs"] == 100
        for fault in runner.FAULT_IDS
    )
    assert backend.calls == ["development", "calibrate", "confirm"]
    with pytest.raises(FileExistsError):
        runner.run_confirmation(
            project, data, output, protocol, digest, backend=backend,
            source_record=source, source_metadata_record=metadata,
            verify_model=_verify_synthetic_model,
        )
    assert backend.calls == ["development", "calibrate", "confirm"]


def test_invalid_calibration_is_consumed_and_blocks_confirmation(runner, synthetic_workspace):
    backend, digest = _development_and_freeze(runner, synthetic_workspace)
    project, data, output, protocol, source, metadata = synthetic_workspace

    class DegenerateBackend(SyntheticBackend):
        def calibrate(self, **kwargs):
            raw = super().calibrate(**kwargs)
            for row in raw["rows"]:
                row["run_maximum"] = 1.0
            return raw

    degenerate = DegenerateBackend(runner, output)
    result = runner.run_calibration(
        project, data, output, protocol, digest, backend=degenerate,
        source_record=source, source_metadata_record=metadata,
        verify_model=_verify_synthetic_model,
    )
    assert result["summary"]["decision"] == "CALIBRATION_INVALID"
    assert (output / "calibrate" / "completion.json").is_file()
    with pytest.raises(RuntimeError, match="forbids confirmation"):
        runner.run_confirmation(
            project, data, output, protocol, digest, backend=backend,
            source_record=source, source_metadata_record=metadata,
            verify_model=_verify_synthetic_model,
        )


def test_phase_claim_is_exclusive_and_partial_attempt_is_visible(runner, tmp_path: Path):
    root = tmp_path / "out"
    _, staging, claim = runner._claim_phase(root, "calibrate")
    assert staging.is_dir() and claim.is_file()
    with pytest.raises(FileExistsError):
        runner._claim_phase(root, "calibrate")


def test_bound_model_metadata_and_embedded_monitor_are_fully_validated(
    runner, tmp_path: Path
):
    # Torch-first matters on the target Windows environment.
    import torch  # noqa: F401
    import numpy as np
    from dataclasses import asdict

    from mate_spc_mstcn.features import RobustRunTransform
    from mate_spc_mstcn.model import MSTCNConfig, MSTCNModel
    from mate_spc_mstcn.monitor import FittedTopTailMonitor

    config = runner.default_model_config()
    selected = {
        "candidate_id": "q250-rho250",
        "mil_top_fraction_q": 0.25,
        "scan_tail_fraction_rho": 0.25,
    }
    architecture = MSTCNConfig(mil_top_fraction=0.25)
    model = MSTCNModel.from_untrained(
        RobustRunTransform(np.ones(52, dtype=np.float64)),
        config=architecture,
        seed=config["training"]["seed"],
    )
    rng = np.random.default_rng(7)
    monitor = FittedTopTailMonitor.fit(
        rng.normal(size=(100, 340)).astype(np.float64),
        np.arange(160, 500, dtype=np.int64),
        windows=(16, 32, 64, 128),
        top_fraction=0.25,
    )
    model_config = asdict(architecture)
    for name in ("dilations", "pool_scales", "class_ids"):
        model_config[name] = list(model_config[name])
    training_metadata = {
        "method": "Detection-first Multi-Scale Causal Dynamics TCN",
        "objective": "q-specific full-monitoring run-level MIL",
        "mil_top_fraction_q": 0.25,
        "positive_endpoint_labels_used": False,
        "detection_target": "fault_vs_healthy",
        "attribution_target": "fault_3_vs_9_vs_15_fault_runs_only",
        "loss_formula": "0.5*mean_H softplus(A) + (1/6)*sum_k mean_Fk softplus(-A) + 0.25*(1/3)*sum_k mean_Fk CE(C,k)",
        "architecture_pooling": "TCN endpoint state plus 8/32/96 mean-RMS of pointwise projection",
        "requested_device": "cuda",
        "dtype": "float64",
        "seed": config["training"]["seed"],
        "model_config": model_config,
        "training_config": {
            "epochs": 30,
            "batch_size": 8,
            "examples_per_class_per_epoch": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "device": "cuda",
        },
        "training_run_names": sorted(
            f"m50-idv{idv:02d}-run{run:03d}"
            for idv in runner.ALL_IDS
            for run in runner.DEVELOPMENT_RUNS
        ),
        "training_run_count_by_class": {"0": 100, "3": 100, "9": 100, "15": 100},
        "mil_endpoints_half_open_stride": [160, 500, 4],
        "mil_endpoint_count": 85,
        "dense_inference_endpoints_half_open_stride": [160, 500, 1],
        "batch_balance": "eight_runs_exactly_two_per_class",
        "validation_or_early_stopping_used": False,
        "window_contract": "each output ends at its named endpoint and contains no future row",
        "preprocessor_fit_scope": "training_run_commissioning_prefixes_only",
    }
    metadata = {
        "schema": "mstcn_spc_bound_model_metadata_v1",
        "backend_schema": "mstcn_spc_backend_v3",
        "complete_model_config_sha256": runner._config_sha256(config),
        "selected_candidate": selected,
        "fitted_monitor": monitor.to_document(),
        "training": training_metadata,
        "loss_history": [1.0] * 30,
        "monitor_fit_scope": "all_100_magnitude_50_development_healthy_runs_only",
        "held_out_25_or_75_percent_accessed": False,
    }
    path = tmp_path / "bound_model.npz"
    model.save_npz(path, metadata=metadata)
    assert runner._verify_model_file(path, config, selected) == metadata
    poisoned = copy.deepcopy(metadata)
    poisoned["fitted_monitor"]["top_fraction"] = 0.125
    model.save_npz(path, metadata=poisoned)
    with pytest.raises(ValueError, match="selected rho"):
        runner._verify_model_file(path, config, selected)
