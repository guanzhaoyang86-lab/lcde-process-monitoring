from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

import mate_spc_external.rieth_training as rieth
from mate_spc_directional.glr import DirectionalGLRPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_hybrid_pac_rmax_external_gpu.py"
RUNNER = runpy.run_path(str(SCRIPT))
PROTOCOL_SHA = "a" * 64
DEPLOYMENT_SHA = "b" * 64
MONITOR_SHA = "c" * 64


def _write_json(path: Path, value: object) -> str:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_stub() -> dict[str, object]:
    return {
        "monitor_binding": {
            "deployment_spec_sha256": DEPLOYMENT_SHA,
            "monitor_configuration_sha256": MONITOR_SHA,
        }
    }


def _calibration_intent() -> dict[str, object]:
    return RUNNER["_calibration_access_intent"](PROTOCOL_SHA, _protocol_stub())


def _audit_binding(threshold: float = 10.0) -> dict[str, object]:
    return {
        "calibration_completion_sha256": "d" * 64,
        "calibration_completion_fingerprint": "e" * 64,
        "pac_certificate_sha256": "f" * 64,
        "calibration_evidence_sha256": "1" * 64,
        "operational_threshold": threshold,
        "operational_threshold_binary64_hex": threshold.hex(),
        "threshold_source": "CPU_NumPy_reference",
        "calibration_decision": RUNNER["CALIBRATION_DECISION"],
    }


def _audit_intent() -> dict[str, object]:
    return RUNNER["_audit_access_intent"](PROTOCOL_SHA, _audit_binding())


def _mint(tmp_path: Path, intent: dict[str, object]) -> rieth.RiethAccessGrant:
    path = tmp_path / "access_intent.json"
    digest = _write_json(path, intent)
    return rieth.load_access_grant(
        path,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_intent_sha256=digest,
    )


def _authorize_calibration(grant: rieth.RiethAccessGrant) -> dict[str, object]:
    return grant.authorization_for(
        "fault_free_training",
        {0: tuple(range(301, 401))},
        expected_length=500,
        fault_start=20,
    )


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fault_id, run_id in ((3, 1), (3, 2), (9, 99)):
        for sample in range(1, 5):
            row: dict[str, float | int] = {
                "faultNumber": fault_id,
                "simulationRun": run_id,
                "sample": sample,
            }
            for feature_index, name in enumerate(rieth.FEATURE_COLUMNS):
                row[name] = fault_id * 10_000 + run_id * 100 + sample + feature_index / 100
            rows.append(row)
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=11).reset_index(drop=True)
    return frame[
        list(rieth.METADATA_COLUMNS) + list(reversed(rieth.FEATURE_COLUMNS))
    ]


def _path(maximum: float, *, threshold: float | None = None) -> DirectionalGLRPath:
    length = RUNNER["SEQUENCE_LENGTH"]
    valid_start = RUNNER["VALID_START"]
    first = RUNNER["FIRST_ALARM_ELIGIBLE"]
    component_shape = (length, len(RUNNER["FAULT_IDS"]))
    numerical = []
    for _ in range(4):
        values = np.full(component_shape, np.nan, dtype=np.float64)
        values[valid_start:] = 0.0
        numerical.append(values)
    persistence = np.full(component_shape, -np.inf, dtype=np.float64)
    persistence[first:] = maximum
    envelope = np.full(length, -np.inf, dtype=np.float64)
    envelope[first:] = maximum
    eligible = np.zeros(length, dtype=bool)
    eligible[first:] = True
    alarm = np.zeros(length, dtype=bool)
    if threshold is not None:
        alarm = eligible & (envelope > threshold)
    return DirectionalGLRPath(
        fault_ids=RUNNER["FAULT_IDS"],
        logits=numerical[0],
        standardized_logits=numerical[1],
        clipped_evidence=numerical[2],
        cusum=numerical[3],
        component_persistence=persistence,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm,
        valid_start=valid_start,
        threshold=threshold,
    )


def _evidence_arrays(phase: str, threshold: float) -> dict[str, np.ndarray]:
    length = RUNNER["SEQUENCE_LENGTH"]
    valid_start = RUNNER["VALID_START"]
    first = RUNNER["FIRST_ALARM_ELIGIBLE"]
    faults = RUNNER["FAULT_IDS"]
    if phase == "calibration":
        run_ids = list(RUNNER["CALIBRATION_RUNS"])
        fault_ids = [0] * 100
        partitions = ["fault_free_training"] * 100
    else:
        run_ids = list(RUNNER["AUDIT_RUNS"]) * 4
        fault_ids = [0] * 100 + [fault for fault in faults for _ in RUNNER["AUDIT_RUNS"]]
        partitions = ["fault_free_training"] * 100 + ["faulty_training"] * 300
    names = [
        f"Rieth-{partition}-fault-{fault_id:02d}-run-{run_id:03d}"
        for partition, fault_id, run_id in zip(partitions, fault_ids, run_ids)
    ]
    count = len(names)
    component_shape = (count, length, len(faults))
    component_values: dict[str, np.ndarray] = {}
    for key in ("logits", "standardized_logits", "clipped_evidence", "cusum"):
        values = np.full(component_shape, np.nan, dtype=np.float64)
        values[:, valid_start:] = 0.0
        component_values[key] = values
    persistence = np.full(component_shape, -np.inf, dtype=np.float64)
    persistence[:, first:] = 0.0
    envelope = np.full((count, length), -np.inf, dtype=np.float64)
    envelope[:, first:] = 0.0
    eligible = np.zeros((count, length), dtype=bool)
    eligible[:, first:] = True
    return {
        "run_names": np.asarray(names),
        "partitions": np.asarray(partitions),
        "fault_ids": np.asarray(fault_ids, dtype=np.int64),
        "simulation_runs": np.asarray(run_ids, dtype=np.int64),
        **component_values,
        "component_persistence": persistence,
        "envelope": envelope,
        "eligible": eligible,
        "alarm": eligible & (envelope > threshold) if phase == "audit" else np.zeros_like(eligible),
        "operational_threshold": np.asarray([threshold], dtype=np.float64),
        "fault_start_zero_based": np.asarray([RUNNER["FAULT_START"]], dtype=np.int64),
    }


def _save_evidence(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 401.5, "401"])
def test_selection_identifiers_are_strict_integers(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rieth._normalize_selection({0: [bad]})  # type: ignore[list-item]
    with pytest.raises((TypeError, ValueError)):
        rieth._normalize_selection({bad: [401]})  # type: ignore[dict-item]


@pytest.mark.parametrize("field", ["schema_version", "expected_run_length", "fault_start_zero_based"])
def test_intent_rejects_boolean_integer_fields(tmp_path: Path, field: str) -> None:
    intent = _calibration_intent()
    if field == "schema_version":
        intent[field] = True
    else:
        intent["authorized_reads"][0][field] = True  # type: ignore[index]
    path = tmp_path / "intent.json"
    digest = _write_json(path, intent)
    if field == "schema_version":
        with pytest.raises(ValueError):
            rieth.load_access_grant(
                path,
                expected_protocol_sha256=PROTOCOL_SHA,
                expected_intent_sha256=digest,
            )
    else:
        grant = rieth.load_access_grant(
            path,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_intent_sha256=digest,
        )
        with pytest.raises(PermissionError):
            _authorize_calibration(grant)


def test_intent_rejects_open_phase_binding_and_float_run_id(tmp_path: Path) -> None:
    open_binding = _calibration_intent()
    open_binding["phase_binding"]["extra"] = "forbidden"  # type: ignore[index]
    path = tmp_path / "open.json"
    digest = _write_json(path, open_binding)
    with pytest.raises(ValueError):
        rieth.load_access_grant(
            path,
            expected_protocol_sha256=PROTOCOL_SHA,
            expected_intent_sha256=digest,
        )

    float_run = _calibration_intent()
    float_run["authorized_reads"][0]["simulation_runs_by_fault"]["0"][0] = 301.0  # type: ignore[index]
    other = tmp_path / "float.json"
    other_digest = _write_json(other, float_run)
    grant = rieth.load_access_grant(
        other,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_intent_sha256=other_digest,
    )
    with pytest.raises(PermissionError):
        _authorize_calibration(grant)


def test_grant_is_sealed_tamper_evident_ordered_and_attempt_once(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        rieth.RiethAccessGrant(
            protocol_sha256=PROTOCOL_SHA,
            intent_path=tmp_path / "never.json",
            intent_sha256="2" * 64,
            intent=_calibration_intent(),
            _seal=object(),
        )

    wrong_order = _mint(tmp_path, _audit_intent())
    with pytest.raises(PermissionError):
        wrong_order.authorization_for(
            "faulty_training",
            {fault: tuple(range(401, 501)) for fault in RUNNER["FAULT_IDS"]},
            expected_length=500,
            fault_start=20,
        )
    with pytest.raises(PermissionError):
        wrong_order.authorization_for(
            "fault_free_training", {0: tuple(range(401, 501))},
            expected_length=500, fault_start=20,
        )

    repeat_dir = tmp_path / "repeat"
    repeat_dir.mkdir()
    repeat = _mint(repeat_dir, _audit_intent())
    repeat.authorization_for(
        "fault_free_training", {0: tuple(range(401, 501))},
        expected_length=500, fault_start=20,
    )
    with pytest.raises(PermissionError):
        repeat.authorization_for(
            "fault_free_training", {0: tuple(range(401, 501))},
            expected_length=500, fault_start=20,
        )

    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    complete = _mint(complete_dir, _audit_intent())
    complete.authorization_for(
        "fault_free_training", {0: tuple(range(401, 501))},
        expected_length=500, fault_start=20,
    )
    complete.authorization_for(
        "faulty_training",
        {fault: tuple(range(401, 501)) for fault in RUNNER["FAULT_IDS"]},
        expected_length=500,
        fault_start=20,
    )
    complete.assert_fully_consumed()
    with pytest.raises(PermissionError):
        complete.authorization_for(
            "fault_free_training", {0: tuple(range(401, 501))},
            expected_length=500, fault_start=20,
        )


def test_grant_reauthenticates_intent_and_invalidates_on_tamper(tmp_path: Path) -> None:
    path = tmp_path / "access_intent.json"
    intent = _calibration_intent()
    digest = _write_json(path, intent)
    grant = rieth.load_access_grant(
        path,
        expected_protocol_sha256=PROTOCOL_SHA,
        expected_intent_sha256=digest,
    )
    intent["created_before_external_numeric_decode"] = False
    _write_json(path, intent)
    with pytest.raises(PermissionError):
        _authorize_calibration(grant)
    with pytest.raises(PermissionError):
        _authorize_calibration(grant)


def test_loader_rejects_grant_subclass_and_failed_attempt_is_consumed(tmp_path: Path) -> None:
    class FakeGrant(rieth.RiethAccessGrant):
        pass

    archive = rieth.SafeRiethTrainingArchive(tmp_path)
    fake = object.__new__(FakeGrant)
    with pytest.raises(PermissionError):
        archive.load_runs(
            fake, "fault_free_training", {0: tuple(range(301, 401))},
            expected_length=500, fault_start=20,
        )

    grant = _mint(tmp_path, _calibration_intent())
    with pytest.raises(ValueError, match="disappeared or changed size"):
        archive.load_runs(
            grant, "fault_free_training", {0: tuple(range(301, 401))},
            expected_length=500, fault_start=20,
        )
    with pytest.raises(PermissionError, match="invalidated"):
        _authorize_calibration(grant)


def test_frame_conversion_is_canonical_and_rejects_duplicate_columns() -> None:
    frame = _synthetic_frame()
    runs = rieth._frame_to_runs(
        frame,
        partition="faulty_training",
        selections={3: [2, 1]},
        source_name="synthetic.RData",
        source_md5="0" * 32,
        expected_length=4,
        fault_start=2,
    )
    assert [run.metadata["simulation_run"] for run in runs] == [2, 1]
    assert all(run.x.shape == (4, 52) for run in runs)
    assert all(run.feature_names == rieth.FEATURE_COLUMNS for run in runs)
    np.testing.assert_array_equal(runs[0].fault, [False, False, True, True])
    expected_first = 3 * 10_000 + 2 * 100 + 1
    assert runs[0].x[0, 0] == expected_first
    assert runs[0].x[0, -1] == expected_first + 51 / 100

    duplicate = frame.copy()
    duplicate.insert(
        len(duplicate.columns), rieth.FEATURE_COLUMNS[0], 0.0, allow_duplicates=True
    )
    with pytest.raises(ValueError, match="duplicate column"):
        rieth._frame_to_runs(
            duplicate,
            partition="faulty_training",
            selections={3: [1, 2]},
            source_name="synthetic.RData",
            source_md5="0" * 32,
            expected_length=4,
            fault_start=2,
        )


@pytest.mark.parametrize("name,value", [("expected_length", True), ("fault_start", False)])
def test_frame_conversion_rejects_boolean_lengths(name: str, value: bool) -> None:
    arguments = {"expected_length": 4, "fault_start": 2}
    arguments[name] = value
    with pytest.raises(ValueError, match="must be an integer"):
        rieth._frame_to_runs(
            _synthetic_frame(),
            partition="faulty_training",
            selections={3: [1, 2]},
            source_name="synthetic.RData",
            source_md5="0" * 32,
            **arguments,
        )


def test_frozen_external_source_hashes_are_bound() -> None:
    assert rieth.EXTERNAL_RIETH_FILES["fault_free_training"] == {
        "file_name": "TEP_FaultFree_Training.RData",
        "md5": "ec126484534331f85001d8c4ebce6d17",
        "sha256": "12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940",
        "size_bytes": 24_678_017,
    }
    assert rieth.EXTERNAL_RIETH_FILES["faulty_training"] == {
        "file_name": "TEP_Faulty_Training.RData",
        "md5": "c5f594d54c47e620ff877feb58407fda",
        "sha256": "058df8bd90c5f575f44d2f6085bdd3bc3662e3e9cc25a50e34af42be3748952b",
        "size_bytes": 494_063_194,
    }


def test_protocol_and_manifests_precede_every_numeric_decode() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name in ("_run_calibration", "_run_audit"):
        function = functions[function_name]
        manifest_lines = []
        decode_lines = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            called = ast.unparse(node.func)
            if called == "_write_json" and node.args and "phase_manifest.json" in ast.unparse(node.args[0]):
                manifest_lines.append(node.lineno)
            if called.endswith(".load_runs"):
                decode_lines.append(node.lineno)
        assert len(manifest_lines) == 1
        assert decode_lines
        assert manifest_lines[0] < min(decode_lines)
    for function_name in ("freeze_protocol", "build_protocol_template"):
        assert not any(
            isinstance(node, ast.Call) and ast.unparse(node.func).endswith(".load_runs")
            for node in ast.walk(functions[function_name])
        )


def test_pac_q99_is_second_largest_and_cpu_cuda_driver_exact() -> None:
    names = [
        f"Rieth-fault_free_training-fault-00-run-{run:03d}"
        for run in RUNNER["CALIBRATION_RUNS"]
    ]
    cpu = {name: _path(float(index + 1)) for index, name in enumerate(names)}
    gpu = {name: _path(float(index + 1)) for index, name in enumerate(names)}
    certificate, parity = RUNNER["_calibration_parity"](
        cpu,
        gpu,
        _protocol_stub(),
        {
            "atol": RUNNER["CPU_GPU_TOLERANCE"],
            "run_count": 100,
            "all_numeric_paths_within_tolerance": True,
            "all_eligibility_arrays_exact": True,
            "all_alarm_arrays_exact": True,
            "maximum_errors": {},
            "records": [],
        },
    )
    assert certificate.threshold == 99.0
    assert certificate.rank.rank_one_based == 99
    assert certificate.rank_driver_run == names[-2]
    assert certificate.rank.achieved_confidence == pytest.approx(0.9629187906726446)
    assert certificate.rank.minimum_sample_count == 59
    assert parity["cpu_q99_driver_run"] == parity["cuda_q99_driver_run"] == names[-2]
    assert parity["q99_threshold_max_abs_error"] == 0.0


def _healthy_records(alarms: int) -> list[dict[str, object]]:
    return [{"any_alarm": index < alarms} for index in range(100)]


def _fault_records(delays: dict[int, int]) -> list[dict[str, object]]:
    return [
        {
            "fault_id": fault_id,
            "detected": True,
            "restricted_delay_samples_miss_at_run_end": delays[fault_id],
        }
        for fault_id in RUNNER["FAULT_IDS"]
        for _ in range(100)
    ]


def _parity_stub() -> dict[str, object]:
    return {
        "all_numeric_paths_within_tolerance": True,
        "all_alarm_arrays_exact": True,
    }


def test_success_gate_uses_each_fault_median_and_dynamic_cp_bound() -> None:
    passing = RUNNER["_summarize_audit"](
        _healthy_records(0),
        _fault_records({3: 100, 9: 100, 15: 100}),
        _parity_stub(),
        10.0,
    )
    assert passing["passed"] is True
    assert passing["healthy"]["empirical_cp95_upper_below_0_05"] is True

    one_fault_fails = RUNNER["_summarize_audit"](
        _healthy_records(4),
        _fault_records({3: 100, 9: 100, 15: 130}),
        _parity_stub(),
        10.0,
    )
    assert one_fault_fails["faults"]["overall_median_restricted_delay_samples"] == 100.0
    assert one_fault_fails["faults"]["per_fault"]["15"]["passed_median_gate"] is False
    assert one_fault_fails["passed"] is False
    assert one_fault_fails["healthy"]["empirical_cp95_upper_below_0_10"] is True


def test_evidence_schema_and_strict_alarm_replay(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.npz"
    calibration = _evidence_arrays("calibration", 10.0)
    _save_evidence(calibration_path, calibration)
    RUNNER["_validate_evidence"](
        calibration_path, phase="calibration", operational_threshold=10.0
    )

    bad_onset = dict(calibration)
    bad_onset["fault_start_zero_based"] = np.asarray([21], dtype=np.int64)
    bad_path = tmp_path / "bad-onset.npz"
    _save_evidence(bad_path, bad_onset)
    with pytest.raises(ValueError, match="fault onset"):
        RUNNER["_validate_evidence"](
            bad_path, phase="calibration", operational_threshold=10.0
        )

    audit = _evidence_arrays("audit", 10.0)
    audit["alarm"][0, RUNNER["FIRST_ALARM_ELIGIBLE"]] = True
    audit_path = tmp_path / "audit.npz"
    _save_evidence(audit_path, audit)
    with pytest.raises(ValueError, match="strict-threshold replay"):
        RUNNER["_validate_evidence"](
            audit_path, phase="audit", operational_threshold=10.0
        )


def test_completed_reentry_returns_before_archive_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ForbiddenArchive:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("completed reentry must not construct an RData archive")

    calibration_sentinel = {"already": "calibrated"}
    calibration_function = RUNNER["_run_calibration"]
    monkeypatch.setitem(calibration_function.__globals__, "SafeRiethTrainingArchive", ForbiddenArchive)
    monkeypatch.setitem(
        calibration_function.__globals__,
        "_verify_calibration_bundle",
        lambda *_args, **_kwargs: calibration_sentinel,
    )
    returned = calibration_function(
        project_root=tmp_path,
        data_dir=tmp_path,
        output_root=tmp_path / "output",
        protocol_path=tmp_path / "protocol.json",
        protocol={},
        protocol_sha256=PROTOCOL_SHA,
        device="cuda",
        batch_size=8,
    )
    assert returned is calibration_sentinel

    calibration = {
        "threshold": 10.0,
        "completion_sha256": "d" * 64,
        "completion": {"completion_fingerprint": "e" * 64},
        "certificate_sha256": "f" * 64,
        "evidence_sha256": "1" * 64,
    }
    audit_sentinel = {"already": "audited"}
    audit_function = RUNNER["_run_audit"]
    monkeypatch.setitem(audit_function.__globals__, "SafeRiethTrainingArchive", ForbiddenArchive)
    monkeypatch.setitem(
        audit_function.__globals__,
        "_verify_calibration_bundle",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setitem(
        audit_function.__globals__,
        "_verify_audit_bundle",
        lambda *_args, **_kwargs: audit_sentinel,
    )
    returned = audit_function(
        project_root=tmp_path,
        data_dir=tmp_path,
        output_root=tmp_path / "output",
        protocol_path=tmp_path / "protocol.json",
        protocol={},
        protocol_sha256=PROTOCOL_SHA,
        device="cuda",
        batch_size=8,
    )
    assert returned is audit_sentinel


def test_partial_phase_blocks_reentry_before_archive_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    (output / "calibration").mkdir(parents=True)

    class ForbiddenArchive:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("partial phase must block before archive construction")

    function = RUNNER["_run_calibration"]
    monkeypatch.setitem(function.__globals__, "SafeRiethTrainingArchive", ForbiddenArchive)
    with pytest.raises(RuntimeError, match="must not be reopened"):
        function(
            project_root=tmp_path,
            data_dir=tmp_path,
            output_root=output,
            protocol_path=tmp_path / "protocol.json",
            protocol={},
            protocol_sha256=PROTOCOL_SHA,
            device="cuda",
            batch_size=8,
        )
