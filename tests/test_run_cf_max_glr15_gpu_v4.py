from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_cf_max_glr15_gpu_v4.py"
)


@pytest.fixture
def runner():
    name = "run_cf_max_glr15_gpu_v4_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / runner.MODE1_FILE_NAME
    source.write_bytes(b"synthetic-metadata-only")
    output = tmp_path / "output"
    protocol = tmp_path / "formal.json"

    def source_record(selected: Path):
        path = selected / runner.MODE1_FILE_NAME
        return {
            "absolute_path": str(path.resolve()),
            "size_bytes": len(path.read_bytes()),
            "sha256": "1" * 64,
            "md5": "2" * 32,
        }

    monkeypatch.setattr(runner, "_source_record", source_record)
    monkeypatch.setattr(
        runner,
        "_implementation_records",
        lambda _root: {"synthetic.py": {"size_bytes": 1, "sha256": "3" * 64}},
    )
    return data_dir, output, protocol


class SyntheticBackend:
    def __init__(self):
        self.calls: list[str] = []

    def cuda_preflight(self):
        return {
            "cuda_available": True,
            "device_type": "cuda",
            "dtype": "float64",
            "torch_version": "synthetic",
            "cuda_version": "synthetic",
            "device_name": "synthetic-5090",
        }

    def verify_bound_artifacts(self, *, model_path, monitor_path, config):
        assert model_path.is_file() and monitor_path.is_file()
        return {"verified": True}

    def development_iteration(
        self, *, source_path, access_plan, fold_map, config, model_path,
        monitor_path, device, dtype,
    ):
        assert device == "cuda" and dtype == "float64"
        assert (model_path.parent / "access_intent.json").is_file()
        self.calls.append("development")
        folds = []
        for outer in range(5):
            roles = {
                "fit": sorted(
                    run for run, fold in fold_map.items()
                    if fold not in (outer, (outer + 1) % 5)
                ),
                "calibration": sorted(
                    run for run, fold in fold_map.items()
                    if fold == (outer + 1) % 5
                ),
                "test": sorted(
                    run for run, fold in fold_map.items() if fold == outer
                ),
            }
            maxima = [float(index) for index in range(20)]
            rows = []
            for run in roles["test"]:
                for idv in (1, 3, 9, 15):
                    alarm = idv != 1
                    rows.append({
                        "idv": idv,
                        "simulation_run": run,
                        "run_maximum": 0.0 if idv == 1 else 20.0,
                        "parity_max_abs_error": 0.0,
                        "alarm": alarm,
                        "first_alarm_index": 170 if alarm else None,
                        "attribution": idv if alarm else None,
                    })
            folds.append({
                "outer_fold": outer,
                "calibration_fold": (outer + 1) % 5,
                "fit_run_ids_by_class": {
                    str(idv): roles["fit"] for idv in (1, 3, 9, 15)
                },
                "calibration_healthy_run_ids": roles["calibration"],
                "excluded_calibration_fault_run_ids_by_class": {
                    str(idv): roles["calibration"] for idv in (3, 9, 15)
                },
                "test_run_ids_by_class": {
                    str(idv): roles["test"] for idv in (1, 3, 9, 15)
                },
                "calibration_healthy_maxima": maxima,
                "threshold": 19.0,
                "rank19": 18.0,
                "distinct_maxima": 20,
                "adjacent_tail_gap": 1.0,
                "calibration_nondegenerate": True,
                "rows": rows,
                "score_parity_max_abs_error": 0.0,
                "discrete_parity_equal": True,
            })
        model_path.write_bytes(b"safe-synthetic-model")
        monitor_path.write_text("{}\n", encoding="utf-8")
        return {
            "schema": "cf_max_glr15_development_backend_v1",
            "config": copy.deepcopy(config),
            "candidate_id": "cf-max-glr15-fixed",
            "folds": folds,
            "runtime": self.cuda_preflight(),
            "selected_candidate_id": "cf-max-glr15-fixed",
            "model_metadata": {"synthetic": True},
            "model_parity_max_abs_error": 0.0,
        }

    def calibrate(
        self, *, source_path, access_plan, model_path, monitor_path, config,
        device, dtype,
    ):
        assert (model_path.parents[2] / "calibrate" / "access_intent.json").is_file()
        self.calls.append("calibrate")
        return {
            "schema": "cf_max_glr15_calibration_backend_v1",
            "config": copy.deepcopy(config),
            "rows": [
                {
                    "idv": 1,
                    "simulation_run": run,
                    "run_maximum": run / 100.0,
                    "parity_max_abs_error": 0.0,
                }
                for run in range(1, 101)
            ],
            "model_parity_max_abs_error": 0.0,
            "discrete_parity_equal": True,
        }

    def confirm(
        self, *, source_path, access_plan, model_path, monitor_path, config,
        threshold, device, dtype,
    ):
        assert (model_path.parents[2] / "confirm" / "access_intent.json").is_file()
        self.calls.append("confirm")
        rows = []
        for run in range(1, 101):
            for idv in (1, 3, 9, 15):
                alarm = idv != 1
                rows.append({
                    "idv": idv,
                    "simulation_run": run,
                    "run_maximum": 0.0 if idv == 1 else threshold + 1.0,
                    "parity_max_abs_error": 0.0,
                    "alarm": alarm,
                    "first_alarm_index": 170 if alarm else None,
                    "attribution": idv if alarm else None,
                })
        return {
            "schema": "cf_max_glr15_confirmation_backend_v1",
            "config": copy.deepcopy(config),
            "threshold": threshold,
            "rows": rows,
            "model_parity_max_abs_error": 0.0,
            "discrete_parity_equal": True,
        }


def test_metadata_only_import_loads_neither_numpy_nor_torch():
    code = (
        "import importlib.util,sys;"
        f"p=r'{RUNNER_PATH}';"
        "s=importlib.util.spec_from_file_location('v4_meta',p);"
        "m=importlib.util.module_from_spec(s);sys.modules['v4_meta']=m;"
        "s.loader.exec_module(m);"
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("features", "dimensions_by_fault", "3"), 20.0),
        (("training", "epochs"), 80.0),
        (("features", "causal_no_future_rows"), 1),
    ],
)
def test_config_rejects_json_type_drift(runner, path, value):
    config = runner.default_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="exact audited"):
        runner.validate_config(config)


def test_three_way_fold_roles_and_allocations_are_exact(runner):
    folds = runner._fold_assignment()
    assert sorted(folds) == list(range(1, 101))
    for outer in range(5):
        roles = runner._fold_roles(outer, folds)
        assert [len(roles[key]) for key in ("fit", "calibration", "test")] == [60, 20, 20]
        assert not (set(roles["fit"]) & set(roles["calibration"]))
        assert not (set(roles["test"]) & (set(roles["fit"]) | set(roles["calibration"])))
    assert len(runner._development_access_plan()) == 400
    assert len(runner._calibration_access_plan()) == 100
    assert len(runner._confirmation_access_plan()) == 400
    poisoned = runner._calibration_access_plan()
    poisoned[0] = {**poisoned[0], "idv": 3}
    with pytest.raises(ValueError):
        runner._validate_access_plan("calibrate", poisoned)


def test_official_source_identity_fails_before_hashing_small_file(runner, tmp_path):
    (tmp_path / runner.MODE1_FILE_NAME).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="size differs"):
        runner._source_record(tmp_path)


def test_complete_pass_freeze_calibrate_confirm_is_one_time(
    runner, workspace,
):
    data_dir, output, protocol = workspace
    backend = SyntheticBackend()
    development = runner.run_development_iteration(
        data_dir=data_dir,
        output_root=output,
        iteration_id="dev-pass",
        backend=backend,
    )
    assert development["decision"] == "DEVELOPMENT_PASS"
    assert development["metrics"]["all_gates_pass"] is True
    digest = runner.freeze_formal(
        data_dir=data_dir,
        output_root=output,
        iteration_id="dev-pass",
        protocol_path=protocol,
        backend=backend,
    )
    assert len(digest) == 64
    calibration = runner.run_calibration(
        data_dir=data_dir,
        output_root=output,
        protocol_path=protocol,
        backend=backend,
    )
    assert calibration["summary"]["decision"] == "PASS"
    assert calibration["summary"]["threshold"] == pytest.approx(0.98)
    confirmation = runner.run_confirmation(
        data_dir=data_dir,
        output_root=output,
        protocol_path=protocol,
        backend=backend,
    )
    assert confirmation["decision"] == "PASS"
    assert backend.calls == ["development", "calibrate", "confirm"]
    with pytest.raises(FileExistsError):
        runner.run_confirmation(
            data_dir=data_dir,
            output_root=output,
            protocol_path=protocol,
            backend=backend,
        )


def test_freeze_semantically_replays_result_before_claim(runner, workspace):
    data_dir, output, protocol = workspace
    backend = SyntheticBackend()
    runner.run_development_iteration(
        data_dir=data_dir,
        output_root=output,
        iteration_id="tampered",
        backend=backend,
    )
    directory = output / "development_iterations" / "tampered"
    result_path = directory / "development_result.json"
    result = runner._strict_json(result_path)
    result["metrics"]["healthy_alarms_total"] = 0
    result_path.write_bytes(runner._canonical_json_bytes(result))
    completion_path = directory / "completion.json"
    completion = runner._strict_json(completion_path)
    completion["artifacts"]["development_result.json"] = runner._file_record(
        result_path
    )
    completion_path.write_bytes(runner._canonical_json_bytes(completion))

    with pytest.raises(ValueError, match="semantic replay"):
        runner.freeze_formal(
            data_dir=data_dir,
            output_root=output,
            iteration_id="tampered",
            protocol_path=protocol,
            backend=backend,
        )
    assert not protocol.exists()
    assert not (output / ".formal_freeze.claim.json").exists()


def test_development_no_go_creates_no_model(runner, workspace):
    data_dir, output, _ = workspace

    class NoGo(SyntheticBackend):
        def development_iteration(self, **kwargs):
            raw = super().development_iteration(**kwargs)
            kwargs["model_path"].unlink()
            kwargs["monitor_path"].unlink()
            for fold in raw["folds"]:
                for row in fold["rows"]:
                    if row["idv"] != 1:
                        row.update({
                            "run_maximum": 0.0,
                            "alarm": False,
                            "first_alarm_index": None,
                            "attribution": None,
                        })
            raw["selected_candidate_id"] = None
            raw["model_metadata"] = None
            raw["model_parity_max_abs_error"] = None
            return raw

    result = runner.run_development_iteration(
        data_dir=data_dir,
        output_root=output,
        iteration_id="dev-no-go",
        backend=NoGo(),
    )
    directory = output / "development_iterations" / "dev-no-go"
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert not (directory / "model.npz").exists()
    assert not (directory / "monitor.json").exists()


def test_degenerate_inner_calibration_forces_no_go(runner, workspace):
    data_dir, output, _ = workspace

    class Degenerate(SyntheticBackend):
        def development_iteration(self, **kwargs):
            raw = super().development_iteration(**kwargs)
            kwargs["model_path"].unlink()
            kwargs["monitor_path"].unlink()
            fold = raw["folds"][0]
            fold["calibration_healthy_maxima"] = [0.0] * 20
            fold["threshold"] = 0.0
            fold["rank19"] = 0.0
            fold["distinct_maxima"] = 1
            fold["adjacent_tail_gap"] = 0.0
            fold["calibration_nondegenerate"] = False
            raw["selected_candidate_id"] = None
            raw["model_metadata"] = None
            raw["model_parity_max_abs_error"] = None
            return raw

    result = runner.run_development_iteration(
        data_dir=data_dir,
        output_root=output,
        iteration_id="degenerate",
        backend=Degenerate(),
    )
    assert result["decision"] == "DEVELOPMENT_NO_GO"
    assert result["metrics"]["gate_checks"]["calibration_nondegenerate"] is False
