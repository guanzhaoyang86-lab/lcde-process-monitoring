"""Synthetic-only tests for the staged Hydraulic GPU experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mate_spc_hydraulic.model import (
    DEFAULT_KAPPA,
    POSITIVE_LABELS,
    HydraulicDirectionalModel,
    score_hydraulic_numpy,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hydraulic_padr_pac_gpu.py"
SPEC = importlib.util.spec_from_file_location("run_hydraulic_padr_pac_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _protocol() -> dict[str, object]:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "protocols"
            / "hydraulic_padr_pac_v1_3_analysis.json"
        ).read_text(encoding="utf-8")
    )


def _synthetic_audit(protocol: dict[str, object]) -> dict[str, object]:
    split = protocol["split"]
    groups = (
        split["development_groups"]
        + split["calibration_groups"]
        + split["confirmation_groups"]
    )
    episodes: list[dict[str, object]] = []
    cursor = 1
    episode_id = 1
    for group in groups:
        cooler, pump, accumulator = group.split("-")
        for label in (100, 90, 80, 73):
            length = 19 if group == "C100-P0-A090" and label == 100 else 10
            profile = {
                "cooler_condition_percent": int(cooler[1:]),
                "valve_condition_percent": label,
                "internal_pump_leakage": int(pump[1:]),
                "hydraulic_accumulator_bar": int(accumulator[1:]),
                "stable_flag": 0,
            }
            episodes.append(
                {
                    "episode_id": episode_id,
                    "start_row_1based": cursor,
                    "end_row_1based_inclusive": cursor + length - 1,
                    "length_cycles": length,
                    "profile": profile,
                }
            )
            cursor += length
            episode_id += 1
    return {"episodes": episodes}


def _preflight() -> object:
    protocol = _protocol()
    extracted = {
        f"sensor-{index:02d}.txt": {"size_bytes": index + 1, "sha256": "d" * 64}
        for index in range(19)
    }
    extracted["profile.txt"] = {
        "size_bytes": 1,
        "sha256": runner.PROFILE_SHA256,
    }
    frozen_inputs = {
        "analysis_protocol": {"sha256": runner.ANALYSIS_PROTOCOL_SHA256},
        "parent_analysis_protocol": {
            "sha256": runner.PARENT_ANALYSIS_PROTOCOL_SHA256
        },
        "aborted_parent_attempt": {
            "sha256": runner.ABORTED_PARENT_RECORD_SHA256
        },
        "base_analysis_protocol": {
            "sha256": runner.BASE_ANALYSIS_PROTOCOL_SHA256
        },
        "aborted_v1_attempt": {"sha256": runner.ABORTED_V1_RECORD_SHA256},
        "acquisition_protocol": {"sha256": runner.ACQUISITION_PROTOCOL_SHA256},
        "data_file_manifest": {"sha256": runner.DATA_FILE_MANIFEST_SHA256},
        "source_archive": {"sha256": runner.ARCHIVE_SHA256},
        "extracted_files": extracted,
        "label_audit": {
            "label_audit": {
                "audit_fingerprint": runner.LABEL_AUDIT_FINGERPRINT,
            }
        },
    }
    return runner.Preflight(
        analysis_protocol=protocol,
        frozen_inputs=frozen_inputs,
        source_sha256={"synthetic.py": "b" * 64},
        records=runner._build_cycle_records(protocol, _synthetic_audit(protocol)),
    )


def _loader_metadata(indices: list[int]) -> dict[str, object]:
    return {
        "parsed_cycle_indices": indices,
        "unrequested_rows_byte_scanned_for_boundaries": True,
        "unrequested_rows_decoded": False,
        "unrequested_numeric_values_parsed": False,
        "output_shape": [len(indices), 60, 44],
    }


def _zero_model() -> HydraulicDirectionalModel:
    return HydraulicDirectionalModel(
        positive_labels=POSITIVE_LABELS,
        weights=np.zeros((3, 88)),
        bias=np.zeros(3),
        phase_logit_center=np.zeros((60, 3)),
        phase_logit_scale=np.ones((60, 3)),
        kappa=np.full(3, DEFAULT_KAPPA),
        threshold=None,
    )


@dataclass(frozen=True)
class _FakeCV:
    fold_groups: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    selected_lambda: float = 1.0

    @property
    def final_fit(self) -> dict[str, object]:
        return {
            "device": "cuda:0",
            "dtype": "float64",
            "tf32": False,
            "converged": True,
            "ridge_lambda": self.selected_lambda,
            "development_cycle_count": 720,
            "nuisance_group_count": 18,
        }

    def as_dict(self) -> dict[str, object]:
        losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        return {
            "selected_lambda": self.selected_lambda,
            "mean_validation_loss_by_lambda": [
                {"ridge_lambda": ridge, "mean_validation_log_loss": loss}
                for ridge, loss in zip(runner.RIDGE_LAMBDA_GRID, losses, strict=True)
            ],
            "folds": [
                {
                    "ridge_lambda": ridge,
                    "held_out_fold": fold,
                    "training_cycle_count": 480,
                    "held_out_cycle_count": 240,
                    "held_out_nuisance_groups": list(self.fold_groups[fold]),
                    "validation_log_loss": loss,
                    "convergence": {
                        "converged": True,
                        "dtype": "float64",
                        "tf32": False,
                    },
                }
                for ridge, loss in zip(runner.RIDGE_LAMBDA_GRID, losses, strict=True)
                for fold in range(3)
            ],
            "final_fit": self.final_fit,
            "selection_scope": "development_grouped_cv_only",
        }


def _fake_gpu(model, features, *, device, batch_size):
    return score_hydraulic_numpy(model, features), {
        "device": device,
        "device_name": "synthetic CUDA",
        "dtype": "float64",
        "tf32": False,
        "batch_size": batch_size,
    }


def _patch_development_dependencies(monkeypatch, load_calls: list[list[int]]) -> None:
    def fake_load(_data_dir, indices):
        selected = list(indices)
        load_calls.append(selected)
        values = np.zeros((len(selected), 60, 44), dtype=np.float64)
        # Nonzero deterministic cycle variation keeps robust scales valid.
        values += np.arange(len(selected), dtype=np.float64)[:, None, None] * 1.0e-3
        return values, _loader_metadata(selected)

    monkeypatch.setattr(runner, "load_hydraulic_cycles", fake_load)
    def fake_fit(_features, _labels, groups, **kwargs):
        assert kwargs["max_iter"] == runner.OPTIMIZER_MAX_ITER == 2000
        assert kwargs["max_eval"] == runner.OPTIMIZER_MAX_EVAL == 2500
        assert kwargs["tolerance_grad"] == runner.OPTIMIZER_TOLERANCE_GRAD
        assert kwargs["tolerance_change"] == runner.OPTIMIZER_TOLERANCE_CHANGE
        assert kwargs["history_size"] == runner.OPTIMIZER_HISTORY_SIZE
        assert kwargs["require_convergence"] is True
        folds = np.asarray(kwargs["fold_by_cycle"])
        group_values = np.asarray(groups)
        fold_groups = tuple(
            tuple(
                sorted(
                    f"string:{group}"
                    for group in set(group_values[folds == fold].tolist())
                )
            )
            for fold in range(3)
        )
        return _zero_model(), _FakeCV(fold_groups=fold_groups)

    monkeypatch.setattr(runner, "fit_hydraulic_grouped_cv_cuda", fake_fit)
    monkeypatch.setattr(runner, "score_hydraulic_cuda", _fake_gpu)


def _args(tmp_path: Path, phase: str = "development") -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp_path / "never-read-real-data",
        output_dir=tmp_path / "out",
        batch_size=17,
        device="cuda",
        phase=phase,
        access_ledger=tmp_path / Path(runner.ACCESS_LEDGER_RELATIVE).name,
    )


def test_cycle_plan_has_exact_group_counts_and_disjoint_phases() -> None:
    preflight = _preflight()
    records = preflight.records
    assert {name: len({r.nuisance_group for r in rows}) for name, rows in records.items()} == {
        "development": 18,
        "calibration": 9,
        "confirmation": 9,
    }
    assert {
        name: {label: sum(r.valve_label == label for r in rows) for label in (100, 90, 80, 73)}
        for name, rows in records.items()
    } == runner.EXPECTED_CYCLE_COUNTS
    index_sets = [{record.index for record in records[name]} for name in records]
    assert not index_sets[0] & index_sets[1]
    assert not index_sets[0] & index_sets[2]
    assert not index_sets[1] & index_sets[2]


def test_v1_3_is_strictly_a_validator_representation_amendment() -> None:
    root = Path(__file__).resolve().parents[1]
    amended = _protocol()
    parent = json.loads(
        (root / "protocols" / "hydraulic_padr_pac_v1_2_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads(
        (root / "protocols" / "hydraulic_padr_pac_v1_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    aborted_path = (
        root / "outputs" / "hydraulic_padr_pac_gpu_v1_2" / "ABORTED.json"
    )
    aborted_v1_path = root / "outputs" / "hydraulic_padr_pac_gpu_v1" / "ABORTED.json"
    aborted = json.loads(aborted_path.read_text(encoding="utf-8"))
    aborted_v1 = json.loads(aborted_v1_path.read_text(encoding="utf-8"))

    assert runner._sha256_file(
        root / "protocols" / "hydraulic_padr_pac_v1_3_analysis.json"
    ) == runner.ANALYSIS_PROTOCOL_SHA256
    assert runner._sha256_file(
        root / "protocols" / "hydraulic_padr_pac_v1_2_analysis.json"
    ) == runner.PARENT_ANALYSIS_PROTOCOL_SHA256
    assert runner._sha256_file(
        root / "protocols" / "hydraulic_padr_pac_v1_analysis.json"
    ) == runner.BASE_ANALYSIS_PROTOCOL_SHA256
    assert runner._sha256_file(aborted_path) == runner.ABORTED_PARENT_RECORD_SHA256
    assert runner._sha256_file(aborted_v1_path) == runner.ABORTED_V1_RECORD_SHA256
    runner._validate_protocol(amended, parent, aborted, base, aborted_v1)

    for section in (
        "inputs",
        "estimand",
        "split",
        "features",
        "directional_model",
        "online_statistic",
        "calibration",
        "confirmation",
    ):
        assert amended[section] == parent[section]
    assert amended["amendment"]["only_change"] == {
        "confirmation_authentication.expected_cv_held_out_nuisance_groups": {
            "from": "{raw_protocol_group}",
            "to": "string:{raw_protocol_group}",
            "reason": (
                "mate_spc_hydraulic.model._normalize_nuisance_groups canonically "
                "prefixes nonempty string identifiers with string: before CV results "
                "are serialized"
            ),
            "training_scoring_threshold_gate_or_numeric_result_changed": False,
        }
    }
    assert amended["frozen_parent_development"]["development_completion_sha256"] == (
        runner.PARENT_DEVELOPMENT_COMPLETION_SHA256
    )
    assert amended["frozen_before_sensor_access"] is False
    assert amended["frozen_after_development_sensor_access"] is True
    assert amended["frozen_after_healthy_calibration_sensor_access"] is True
    assert amended["amendment_frozen_before_confirmation_sensor_access"] is True


def test_cv_authentication_expects_real_model_string_prefix() -> None:
    preflight = _preflight()
    records = preflight.records["development"]
    _, _, groups, _ = runner._records_arrays(records)
    folds = runner.protocol_grouped_fold_ids(groups)
    canonical = {
        fold: tuple(
            sorted(f"string:{group}" for group in set(groups[folds == fold].tolist()))
        )
        for fold in range(3)
    }
    cv = _FakeCV(tuple(canonical[fold] for fold in range(3))).as_dict()
    assert runner._selected_lambda_from_cv(
        cv, expected_fold_groups=canonical
    ) == 1.0
    assert all(
        group.startswith("string:")
        for item in cv["folds"]
        for group in item["held_out_nuisance_groups"]
    )

    raw = {
        fold: tuple(value.removeprefix("string:") for value in canonical[fold])
        for fold in range(3)
    }
    with pytest.raises(ValueError, match="fold scope"):
        runner._selected_lambda_from_cv(cv, expected_fold_groups=raw)


def test_raw_pac_threshold_and_effective_boundary_are_strictly_bound() -> None:
    raw = 250.44148069867225
    expected = math.nextafter(raw + runner.CPU_CUDA_ATOL, math.inf)
    record = runner._alarm_boundary_record(raw)
    assert record["raw_pac_threshold"] == raw
    assert record["numerical_alarm_guard"] == runner.CPU_CUDA_ATOL == 1.0e-10
    assert record["effective_alarm_boundary"] == expected
    assert expected > raw
    assert runner._validate_alarm_boundary_record(
        record, expected_raw_pac_threshold=raw
    ) == record

    for key, value in (
        ("raw_pac_threshold", math.nextafter(raw, math.inf)),
        ("numerical_alarm_guard", 1.0e-12),
        ("effective_alarm_boundary", math.nextafter(expected, math.inf)),
    ):
        tampered = dict(record)
        tampered[key] = value
        with pytest.raises(ValueError, match="boundary binding"):
            runner._validate_alarm_boundary_record(
                tampered, expected_raw_pac_threshold=raw
            )


def test_development_loads_no_calibration_fault_or_confirmation_cycle(
    tmp_path: Path, monkeypatch
) -> None:
    preflight = _preflight()
    args = _args(tmp_path)
    args.output_dir.mkdir()
    calls: list[list[int]] = []
    _patch_development_dependencies(monkeypatch, calls)
    unguarded_load = runner.load_hydraulic_cycles

    def manifest_guarded_load(data_dir, indices):
        assert (args.output_dir / runner.DEVELOPMENT_MANIFEST).is_file()
        assert (args.output_dir / runner.DEVELOPMENT_ACCESS).is_file()
        return unguarded_load(data_dir, indices)

    monkeypatch.setattr(runner, "load_hydraulic_cycles", manifest_guarded_load)

    completion = runner._run_development(args, preflight)

    expected = [r.index for r in preflight.records["development"]] + [
        r.index
        for r in preflight.records["calibration"]
        if r.valve_label == runner.HEALTHY_LABEL
    ]
    forbidden = {
        r.index
        for r in preflight.records["calibration"]
        if r.valve_label != runner.HEALTHY_LABEL
    } | {r.index for r in preflight.records["confirmation"]}
    assert calls == [expected]
    assert not set(calls[0]) & forbidden
    assert completion["decision"] == runner.DEVELOPMENT_DECISION
    manifest = json.loads((args.output_dir / runner.DEVELOPMENT_MANIFEST).read_text())
    assert manifest["execution"]["optimizer"] == {
        "name": "PyTorch CUDA LBFGS strong_wolfe",
        "max_iter": 2000,
        "max_eval": 2500,
        "tolerance_grad": 1.0e-7,
        "tolerance_change": 1.0e-10,
        "history_size": 100,
    }
    assert manifest["cycle_plan"]["confirmation_cycles_forbidden"] is True
    assert (
        manifest["cycle_plan"]["calibration_healthy_only"]["cycle_count"]
        == runner.PAC_CALIBRATION_COUNT
    )
    result = json.loads((args.output_dir / "development_result.json").read_text())
    certificate = json.loads(
        (args.output_dir / "calibration_certificate.json").read_text()
    )
    model, metadata = runner.load_hydraulic_model(
        args.output_dir / "calibrated_model.npz"
    )
    boundary = result["alarm_boundary"]
    assert certificate["threshold"] == boundary["raw_pac_threshold"]
    assert boundary["numerical_alarm_guard"] == runner.NUMERICAL_ALARM_GUARD
    assert model.threshold == boundary["effective_alarm_boundary"]
    assert model.threshold > certificate["threshold"]
    assert metadata["raw_pac_threshold"] == certificate["threshold"]
    assert metadata["numerical_alarm_guard"] == runner.NUMERICAL_ALARM_GUARD
    assert metadata["effective_alarm_boundary"] == model.threshold


def test_completed_development_reentry_does_not_load_or_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    preflight = _preflight()
    args = _args(tmp_path)
    args.output_dir.mkdir()
    calls: list[list[int]] = []
    _patch_development_dependencies(monkeypatch, calls)
    runner._run_development(args, preflight)
    before = {
        path.name: (path.stat().st_mtime_ns, runner._sha256_file(path))
        for path in args.output_dir.iterdir()
    }
    monkeypatch.setattr(
        runner,
        "load_hydraulic_cycles",
        lambda *_args, **_kwargs: pytest.fail("completed reentry touched sensor data"),
    )

    runner._run_development(args, preflight)

    after = {
        path.name: (path.stat().st_mtime_ns, runner._sha256_file(path))
        for path in args.output_dir.iterdir()
    }
    assert after == before


def test_confirmation_authenticates_sources_then_loads_only_held_out_once(
    tmp_path: Path, monkeypatch
) -> None:
    preflight = _preflight()
    args = _args(tmp_path)
    args.output_dir.mkdir()
    calls: list[list[int]] = []
    _patch_development_dependencies(monkeypatch, calls)
    runner._run_development(args, preflight)

    changed = runner.Preflight(
        analysis_protocol=preflight.analysis_protocol,
        frozen_inputs=preflight.frozen_inputs,
        source_sha256={"synthetic.py": "c" * 64},
        records=preflight.records,
    )
    with pytest.raises(ValueError, match="sources differ"):
        runner._run_confirmation(args, changed)
    assert len(calls) == 1
    assert not (args.output_dir / runner.CONFIRMATION_MANIFEST).exists()

    unguarded_load = runner.load_hydraulic_cycles
    score_calls: list[int] = []

    def manifest_guarded_load(data_dir, indices):
        assert (args.output_dir / runner.CONFIRMATION_MANIFEST).is_file()
        assert (args.output_dir / runner.CONFIRMATION_ACCESS).is_file()
        return unguarded_load(data_dir, indices)

    def counted_gpu(model, features, *, device, batch_size):
        score_calls.append(len(features))
        return _fake_gpu(model, features, device=device, batch_size=batch_size)

    monkeypatch.setattr(runner, "load_hydraulic_cycles", manifest_guarded_load)
    monkeypatch.setattr(runner, "score_hydraulic_cuda", counted_gpu)
    completion = runner._run_confirmation(args, preflight)

    expected = [record.index for record in preflight.records["confirmation"]]
    assert calls[1:] == [expected]
    assert score_calls == [369]
    assert completion["decision"] == runner.FAIL_DECISION
    result = json.loads((args.output_dir / "confirmation_result.json").read_text())
    assert result["healthy"]["cycle_count"] == 99
    assert result["faults"]["cycle_count"] == 270
    assert result["governance"]["same_data_retuning_permitted"] is False

    before = {
        path.name: (path.stat().st_mtime_ns, runner._sha256_file(path))
        for path in args.output_dir.iterdir()
    }
    monkeypatch.setattr(
        runner,
        "load_hydraulic_cycles",
        lambda *_args, **_kwargs: pytest.fail("completed confirmation touched data"),
    )
    monkeypatch.setattr(
        runner,
        "load_hydraulic_model",
        lambda *_args, **_kwargs: pytest.fail("completed confirmation touched model"),
    )
    runner._run_confirmation(args, preflight)
    after = {
        path.name: (path.stat().st_mtime_ns, runner._sha256_file(path))
        for path in args.output_dir.iterdir()
    }
    assert after == before
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda *_args, **_kwargs: pytest.fail("completed reentry ran preflight"),
    )
    monkeypatch.setattr(runner, "_resolve_paths", lambda _args, _root: None)
    monkeypatch.setattr(
        runner,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda _argv: _args(tmp_path, "confirmation")
        ),
    )
    assert runner.main([]) == 0


def test_hash_mismatch_fails_before_loader(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    records = []
    for index in range(20):
        name = "profile.txt" if index == 19 else f"sensor-{index:02d}.txt"
        payload = b"profile" if name == "profile.txt" else f"value-{index}".encode()
        (dataset / name).write_bytes(payload)
        records.append(
            {
                "name": name,
                "size_bytes": len(payload),
                "sha256": runner._sha256_file(dataset / name),
            }
        )
    # The official profile anchor is intentionally absent in this synthetic fixture,
    # so validation must fail closed after checking all file records.
    with pytest.raises(ValueError, match="profile.txt"):
        runner._verify_data_files(dataset, records)
    (dataset / "sensor-00.txt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity check failed"):
        runner._verify_data_files(dataset, records)


def test_json_reader_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        runner._read_json(path)
    with pytest.raises(ValueError, match="non-finite JSON number"):
        runner._json_bytes({"value": float("inf")})


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 99), (1, 99), (3, 99), (98, 99), (99, 99)],
)
def test_clopper_pearson_upper_is_exact(successes: int, trials: int) -> None:
    upper = runner.clopper_pearson_upper(successes, trials)
    assert successes / trials <= upper <= 1.0
    if successes == trials:
        assert upper == 1.0
    else:
        assert runner._binomial_cdf_leq(successes, trials, upper) == pytest.approx(
            0.05, rel=2e-12, abs=2e-12
        )
    if successes == 0:
        assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / trials), rel=1e-13)


def _path_for_gate(healthy_alarm_count: int, detected_per_severity: int):
    preflight = _preflight()
    records = preflight.records["confirmation"]
    alarm = np.zeros((len(records), 60), dtype=bool)
    component = np.full((len(records), 60, 3), -np.inf)
    envelope = np.full((len(records), 60), -np.inf)
    label_seen = {100: 0, 90: 0, 80: 0, 73: 0}
    for row, record in enumerate(records):
        ordinal = label_seen[record.valve_label]
        label_seen[record.valve_label] += 1
        limit = healthy_alarm_count if record.valve_label == 100 else detected_per_severity
        if ordinal < limit:
            alarm[row, 2] = True
            head = 0 if record.valve_label == 100 else POSITIVE_LABELS.index(record.valve_label)
            component[row, 2, head] = 2.0
            envelope[row, 2] = 2.0
    return records, SimpleNamespace(
        alarm=alarm,
        component_persistence=component,
        envelope=envelope,
    )


def test_frozen_gate_decision_pass_and_fail() -> None:
    records, passing_path = _path_for_gate(3, 90)
    passing, _ = runner._confirmation_metrics(records, passing_path)
    assert passing["decision"] == runner.PASS_DECISION
    assert passing["healthy"]["exact_one_sided_95pct_ucb"] <= 0.08

    records, failing_path = _path_for_gate(4, 90)
    failing, _ = runner._confirmation_metrics(records, failing_path)
    assert failing["decision"] == runner.FAIL_DECISION
    assert not failing["gates"]["healthy_alarm_count"]["passed"]

    records, weak_path = _path_for_gate(0, 71)
    weak, _ = runner._confirmation_metrics(records, weak_path)
    assert weak["decision"] == runner.FAIL_DECISION
    assert not weak["gates"]["minimum_recall_each_severity"]["passed"]
