from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from mate_spc_directional.glr import DirectionalGLRPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_extended_tep_mode1_q100_gpu.py"
RUNNER = runpy.run_path(str(SCRIPT))
PROTOCOL_SHA = "a" * 64
DEPLOYMENT_SHA = "b" * 64
MONITOR_SHA = "c" * 64


def _protocol_stub() -> dict[str, Any]:
    return {
        "schema": RUNNER["SCHEMA"],
        "schema_version": RUNNER["SCHEMA_VERSION"],
        "analysis_status": RUNNER["ANALYSIS_STATUS"],
        "method": RUNNER["METHOD"],
        "dataset": {
            "source_records": {
                RUNNER["DATA_FILE"]: {
                    "size_bytes": RUNNER["MODE1_SIZE_BYTES"],
                    "md5": RUNNER["MODE1_MD5"],
                    "sha256": "d" * 64,
                }
            }
        },
        "monitor_binding": {
            "deployment_spec_sha256": DEPLOYMENT_SHA,
            "monitor_configuration_sha256": MONITOR_SHA,
            "monitor_configuration": {
                "threshold_rule": "PAC_RMax_maximum_q100_of_100_external_run_maxima",
                "epsilon": RUNNER["PAC_EPSILON"],
                "delta": RUNNER["PAC_DELTA"],
                "comparison": "strictly_greater_than",
            },
        },
        "external_calibration": {
            "source_idv": 1,
            "magnitude_percent": 100,
            "simulation_runs": list(RUNNER["CALIBRATION_RUNS"]),
            "normalized_window_half_open": list(RUNNER["CALIBRATION_WINDOW"]),
            "run_count": 100,
            "rank_rule": "maximum_order_statistic_no_interpolation",
            "rank": {
                "rank_one_based": RUNNER["PAC_RANK"],
                "epsilon": RUNNER["PAC_EPSILON"],
                "delta": RUNNER["PAC_DELTA"],
            },
            "numeric_threshold_unknown_at_protocol_freeze": True,
            "completion_required_before_audit_open": True,
        },
        "external_audit": {
            "healthy": {
                "source_idv": 1,
                "simulation_runs": list(RUNNER["AUDIT_RUNS"]),
                "normalized_window_half_open": list(RUNNER["CALIBRATION_WINDOW"]),
            },
            "faults": {
                "fault_ids": list(RUNNER["AUDIT_FAULT_IDS"]),
                "simulation_runs": list(RUNNER["AUDIT_RUNS"]),
                "normalized_window_half_open": list(RUNNER["FAULT_WINDOW"]),
                "local_fault_start_zero_based": RUNNER["FAULT_START"],
            },
            "fault_detection_requires_new_alarm_episode_start_at_or_after": RUNNER[
                "FAULT_START"
            ],
            "pre_fault_alarm_continuing_across_onset_is_not_a_detection": True,
        },
        "success_gates": {
            "healthy_alarm_runs_maximum": RUNNER["HEALTHY_ALARM_RUNS_MAXIMUM"],
            "per_fault_detected_runs_minimum": RUNNER[
                "PER_FAULT_DETECTIONS_MINIMUM"
            ],
            "total_fault_detected_runs_minimum": RUNNER[
                "TOTAL_FAULT_DETECTIONS_MINIMUM"
            ],
            "per_fault_median_restricted_delay_samples_maximum": RUNNER[
                "MEDIAN_RESTRICTED_DELAY_MAXIMUM"
            ],
            "per_fault_restricted_p90_delay_samples_maximum": RUNNER[
                "PER_FAULT_RESTRICTED_P90_MAXIMUM"
            ],
            "restricted_delay_miss_imputation_samples": RUNNER["MISS_DELAY"],
            "pac_primary_claim": "synthetic protocol claim",
        },
        "forbidden_actions": list(RUNNER["FORBIDDEN_ACTIONS"]),
    }


def _path(
    maximum: float = 0.0,
    *,
    threshold: float | None = None,
    alarm: np.ndarray | None = None,
) -> DirectionalGLRPath:
    length = RUNNER["SEQUENCE_LENGTH"]
    valid_start = RUNNER["VALID_START"]
    first = RUNNER["FIRST_ALARM_ELIGIBLE"]
    fault_ids = tuple(RUNNER["FAULT_IDS"])
    shape = (length, len(fault_ids))
    numeric: list[np.ndarray] = []
    for _ in range(4):
        values = np.full(shape, np.nan, dtype=np.float64)
        values[valid_start:] = 0.0
        numeric.append(values)
    persistence = np.full(shape, -np.inf, dtype=np.float64)
    persistence[first:] = maximum
    envelope = np.full(length, -np.inf, dtype=np.float64)
    envelope[first:] = maximum
    eligible = np.zeros(length, dtype=bool)
    eligible[first:] = True
    if alarm is None:
        alarm_values = (
            eligible & (envelope > threshold)
            if threshold is not None
            else np.zeros(length, dtype=bool)
        )
    else:
        alarm_values = np.asarray(alarm, dtype=bool)
    return DirectionalGLRPath(
        fault_ids=fault_ids,
        logits=numeric[0],
        standardized_logits=numeric[1],
        clipped_evidence=numeric[2],
        cusum=numeric[3],
        component_persistence=persistence,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm_values,
        valid_start=valid_start,
        threshold=threshold,
    )


def _run(name: str, fault_id: int, simulation_run: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        metadata={"fault_id": fault_id, "simulation_run": simulation_run},
    )


def _parity_stub(*, eligibility: bool = True) -> dict[str, Any]:
    return {
        "atol": RUNNER["CPU_GPU_TOLERANCE"],
        "run_count": 400,
        "all_numeric_paths_within_tolerance": True,
        "all_eligibility_arrays_exact": eligibility,
        "all_alarm_arrays_exact": True,
        "maximum_errors": {},
        "records": [],
    }


def _healthy_records(alarms: int) -> list[dict[str, Any]]:
    return [
        {
            "run": f"healthy-{index:03d}",
            "simulation_run": index + 101,
            "any_alarm": index < alarms,
            "alarm_episode_count": int(index < alarms),
            "first_alarm_index": 100 if index < alarms else None,
            "maximum_envelope": 0.0,
        }
        for index in range(100)
    ]


def _fault_records(
    *,
    detected_per_fault: int = 100,
    delay: int = 120,
    delay_overrides: dict[int, list[int]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fault_id in RUNNER["AUDIT_FAULT_IDS"]:
        delays = (
            delay_overrides[fault_id]
            if delay_overrides is not None and fault_id in delay_overrides
            else [delay] * detected_per_fault
            + [RUNNER["MISS_DELAY"]] * (100 - detected_per_fault)
        )
        if len(delays) != 100:
            raise AssertionError("synthetic fault delays must contain exactly 100 values")
        for index, restricted in enumerate(delays):
            detected = index < detected_per_fault and restricted != RUNNER["MISS_DELAY"]
            records.append(
                {
                    "run": f"fault-{fault_id:02d}-{index:03d}",
                    "fault_id": fault_id,
                    "simulation_run": index + 101,
                    "detected": detected,
                    "first_fault_alarm_episode_index": (
                        RUNNER["FAULT_START"] + restricted if detected else None
                    ),
                    "delay_samples": restricted if detected else None,
                    "restricted_delay_samples_miss_at_run_end": restricted,
                    "pre_fault_alarm_episode_count": 0,
                    "pre_fault_interval_used_as_performance_evidence": False,
                    "attributed_fault_head_at_first_alarm": fault_id if detected else None,
                    "attribution_matches_fault": detected,
                    "maximum_post_fault_envelope": 0.0,
                }
            )
    return records


def _evidence_arrays(phase: str, threshold: float) -> dict[str, np.ndarray]:
    names, partitions, fault_ids, simulation_runs = RUNNER[
        "_expected_identity_arrays"
    ](phase)
    count = len(names)
    length = RUNNER["SEQUENCE_LENGTH"]
    valid_start = RUNNER["VALID_START"]
    first = RUNNER["FIRST_ALARM_ELIGIBLE"]
    component_shape = (count, length, len(RUNNER["FAULT_IDS"]))
    components: dict[str, np.ndarray] = {}
    for key in ("logits", "standardized_logits", "clipped_evidence", "cusum"):
        values = np.full(component_shape, np.nan, dtype=np.float64)
        values[:, valid_start:] = 0.0
        components[key] = values
    persistence = np.full(component_shape, -np.inf, dtype=np.float64)
    persistence[:, first:] = 0.0
    envelope = np.full((count, length), -np.inf, dtype=np.float64)
    envelope[:, first:] = 0.0
    eligible = np.zeros((count, length), dtype=bool)
    eligible[:, first:] = True
    alarm = (
        eligible & (envelope > threshold)
        if phase == "audit"
        else np.zeros_like(eligible)
    )
    return {
        "run_names": names,
        "partitions": partitions,
        "fault_ids": fault_ids,
        "simulation_runs": simulation_runs,
        **components,
        "component_persistence": persistence,
        "envelope": envelope,
        "eligible": eligible,
        "alarm": alarm,
        "operational_threshold": np.asarray([threshold], dtype=np.float64),
        "fault_start_zero_based": np.asarray(
            [RUNNER["FAULT_START"]], dtype=np.int64
        ),
    }


def _save_evidence(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _runtime_stub() -> dict[str, Any]:
    return {
        "device": "cuda",
        "device_name": "synthetic RTX",
        "compute_capability": [12, 0],
        "total_memory_bytes": 24_000_000_000,
        "torch_version": "synthetic",
        "torch_cuda_version": "12.8",
        "cuda_available": True,
        "dtype": "float64",
        "tf32": False,
        "deterministic_algorithms": True,
        "probe_sum_of_squares": 5.0,
    }


def _write_provenance_fixture(
    root: Path,
    *,
    phase: str,
    protocol: dict[str, Any],
    protocol_sha256: str,
    recorded_binding: dict[str, Any] | None,
    mutate_intent: Callable[[dict[str, Any]], None] | None = None,
    source_records: dict[str, Any] | None = None,
    mutate_data_access: Callable[[dict[str, Any]], None] | None = None,
    execution_runtime: dict[str, Any] | None = None,
) -> None:
    root.mkdir()
    ext = RUNNER["ext"]
    intent = RUNNER["_expected_access_intent"](
        phase=phase,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        calibration_binding=recorded_binding,
    )
    if mutate_intent is not None:
        mutate_intent(intent)
    ext._write_json(root / "access_intent.json", intent)
    runtime = _runtime_stub()
    ext._write_json(root / "runtime_preflight.json", runtime)
    manifest = RUNNER["_phase_manifest"](
        phase=phase,
        protocol_sha256=protocol_sha256,
        intent_sha256=ext._sha256_file(root / "access_intent.json"),
        runtime_sha256=ext._sha256_file(root / "runtime_preflight.json"),
        source_records=(
            protocol["dataset"]["source_records"]
            if source_records is None
            else source_records
        ),
        calibration_binding=recorded_binding,
    )
    ext._write_json(root / "phase_manifest.json", manifest)
    data_access = RUNNER["_expected_data_access_audit"](
        phase=phase,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        intent_sha256=ext._sha256_file(root / "access_intent.json"),
        phase_manifest_sha256=ext._sha256_file(root / "phase_manifest.json"),
        calibration_binding=recorded_binding,
    )
    if mutate_data_access is not None:
        mutate_data_access(data_access)
    ext._write_json(root / "data_access_audit.json", data_access)
    bound_runtime = runtime if execution_runtime is None else execution_runtime
    count = 100 if phase == "calibration" else 400
    execution = {
        "schema": RUNNER["SCHEMA"],
        "schema_version": RUNNER["SCHEMA_VERSION"],
        "phase": phase,
        "protocol_sha256": protocol_sha256,
        "runtime": bound_runtime,
        "score": {
            "backend": "pytorch_cuda_float64_with_independent_numpy_reference",
            "device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "deterministic_algorithms": True,
            "sequence_count": count,
            "sequence_length": RUNNER["SEQUENCE_LENGTH"],
            "batch_size": 32,
            "cuda_calls": [{
                "device": "cuda",
                "device_name": bound_runtime["device_name"],
                "dtype": "float64",
                "tf32": False,
            }],
        },
    }
    ext._write_json(root / "gpu_execution.json", execution)


def _forbidden_source(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("this branch must not hash, inspect, construct, or open the HDF source")


def _call_calibration(function: Callable[..., Any], tmp_path: Path) -> Any:
    return function(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        output_root=tmp_path / "output",
        protocol_path=tmp_path / "protocol.json",
        protocol=_protocol_stub(),
        protocol_sha256=PROTOCOL_SHA,
        device="cuda",
        batch_size=8,
    )


def _call_audit(function: Callable[..., Any], tmp_path: Path) -> Any:
    return function(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        output_root=tmp_path / "output",
        protocol_path=tmp_path / "protocol.json",
        protocol=_protocol_stub(),
        protocol_sha256=PROTOCOL_SHA,
        device="cuda",
        batch_size=8,
    )


def test_frozen_split_geometry_and_identity_sets_are_exact() -> None:
    assert RUNNER["CALIBRATION_RUNS"] == tuple(range(1, 101))
    assert RUNNER["AUDIT_RUNS"] == tuple(range(101, 201))
    assert tuple(RUNNER["FAULT_IDS"]) == tuple(RUNNER["AUDIT_FAULT_IDS"]) == (
        3,
        9,
        15,
    )
    assert RUNNER["CALIBRATION_WINDOW"] == (0, 500)
    assert RUNNER["FAULT_WINDOW"] == (440, 940)
    assert RUNNER["FAULT_START"] == 160
    assert RUNNER["VALID_START"] == 63
    assert RUNNER["FIRST_ALARM_ELIGIBLE"] == 65
    assert RUNNER["MISS_DELAY"] == 340

    calibration = RUNNER["_expected_identity_arrays"]("calibration")
    audit = RUNNER["_expected_identity_arrays"]("audit")
    assert len(calibration[0]) == 100
    assert len(set(calibration[0].tolist())) == 100
    assert len(audit[0]) == 400
    assert len(set(audit[0].tolist())) == 400
    assert not set(calibration[3].tolist()).intersection(RUNNER["AUDIT_RUNS"])


def test_q100_is_dynamic_maximum_with_exact_certificate() -> None:
    names = RUNNER["_expected_identity_arrays"]("calibration")[0].tolist()
    cpu = {name: _path(float(index + 1)) for index, name in enumerate(names)}
    gpu = {name: _path(float(index + 1)) for index, name in enumerate(names)}
    base_parity = {
        "atol": RUNNER["CPU_GPU_TOLERANCE"],
        "run_count": 100,
        "all_numeric_paths_within_tolerance": True,
        "all_eligibility_arrays_exact": True,
        "all_alarm_arrays_exact": True,
        "maximum_errors": {},
        "records": [],
    }
    certificate, parity = RUNNER["_calibration_parity"](
        cpu, gpu, _protocol_stub(), base_parity
    )
    assert certificate.threshold == 100.0
    assert certificate.rank.rank_one_based == 100
    assert certificate.rank.epsilon == 0.05
    assert certificate.rank.delta == 0.01
    assert certificate.rank.achieved_confidence >= 0.99
    assert certificate.rank.upper_tail_probability_bound <= 0.01
    assert certificate.rank.minimum_sample_count == 90
    assert certificate.rank_driver_run == names[-1]
    assert parity["cpu_q100_threshold"] == parity["cuda_q100_threshold"] == 100.0
    assert parity["q100_driver_run_exact"] is True
    assert parity["q100_rank_exact"] is True


def test_calibration_parity_requires_exact_eligibility() -> None:
    names = RUNNER["_expected_identity_arrays"]("calibration")[0].tolist()
    paths = {name: _path(float(index)) for index, name in enumerate(names)}
    parity = {
        "all_numeric_paths_within_tolerance": True,
        "all_eligibility_arrays_exact": False,
        "all_alarm_arrays_exact": True,
    }
    with pytest.raises(AssertionError, match="q100 calibration parity"):
        RUNNER["_calibration_parity"](
            paths, paths, _protocol_stub(), parity
        )


def test_episode_crossing_onset_is_not_a_detection_but_new_episode_is() -> None:
    crossing = np.zeros(RUNNER["SEQUENCE_LENGTH"], dtype=bool)
    crossing[150:170] = True
    later = np.zeros(RUNNER["SEQUENCE_LENGTH"], dtype=bool)
    later[150:170] = True
    later[180:190] = True
    runs = (_run("crossing", 15, 101), _run("later", 15, 102))
    records = RUNNER["_fault_records"](
        runs,
        {"crossing": _path(alarm=crossing), "later": _path(alarm=later)},
    )
    assert records[0]["detected"] is False
    assert records[0]["first_fault_alarm_episode_index"] is None
    assert records[0]["restricted_delay_samples_miss_at_run_end"] == 340
    assert records[0]["pre_fault_alarm_episode_count"] == 1
    assert records[1]["detected"] is True
    assert records[1]["first_fault_alarm_episode_index"] == 180
    assert records[1]["delay_samples"] == 20


def test_episode_start_exactly_at_onset_counts_with_zero_delay() -> None:
    alarm = np.zeros(RUNNER["SEQUENCE_LENGTH"], dtype=bool)
    alarm[RUNNER["FAULT_START"] :] = True
    record = RUNNER["_fault_records"](
        (_run("onset", 3, 101),), {"onset": _path(alarm=alarm)}
    )[0]
    assert record["detected"] is True
    assert record["first_fault_alarm_episode_index"] == RUNNER["FAULT_START"]
    assert record["delay_samples"] == 0


def test_evidence_accepts_exact_identities_and_strict_threshold(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.npz"
    calibration = _evidence_arrays("calibration", 10.0)
    _save_evidence(calibration_path, calibration)
    observed = RUNNER["_validate_evidence"](
        calibration_path, phase="calibration", operational_threshold=10.0
    )
    np.testing.assert_array_equal(
        observed["run_names"],
        RUNNER["_expected_identity_arrays"]("calibration")[0],
    )

    audit = _evidence_arrays("audit", 10.0)
    first = RUNNER["FIRST_ALARM_ELIGIBLE"]
    audit["envelope"][0, first] = 10.0
    audit["envelope"][0, first + 1] = np.nextafter(10.0, np.inf)
    audit["alarm"] = audit["eligible"] & (audit["envelope"] > 10.0)
    audit_path = tmp_path / "audit.npz"
    _save_evidence(audit_path, audit)
    validated = RUNNER["_validate_evidence"](
        audit_path, phase="audit", operational_threshold=10.0
    )
    assert not bool(validated["alarm"][0, first])
    assert bool(validated["alarm"][0, first + 1])


@pytest.mark.parametrize(
    ("field", "left", "right", "message"),
    [
        ("run_names", 0, 1, "exact run names"),
        ("partitions", 0, 100, "exact partitions"),
        ("fault_ids", 0, 100, "exact fault IDs"),
        ("simulation_runs", 0, 1, "exact simulation runs"),
    ],
)
def test_evidence_rejects_identity_or_order_changes(
    tmp_path: Path, field: str, left: int, right: int, message: str
) -> None:
    arrays = _evidence_arrays("audit", 10.0)
    arrays[field] = arrays[field].copy()
    arrays[field][left], arrays[field][right] = (
        arrays[field][right],
        arrays[field][left],
    )
    path = tmp_path / f"bad-{field}.npz"
    _save_evidence(path, arrays)
    with pytest.raises(ValueError, match=message):
        RUNNER["_validate_evidence"](
            path, phase="audit", operational_threshold=10.0
        )


def test_evidence_rejects_onset_eligibility_and_alarm_corruption(tmp_path: Path) -> None:
    corruptions: list[tuple[str, dict[str, np.ndarray], str]] = []
    onset = _evidence_arrays("audit", 10.0)
    onset["fault_start_zero_based"] = np.asarray([159], dtype=np.int64)
    corruptions.append(("onset", onset, "fault onset"))
    eligibility = _evidence_arrays("audit", 10.0)
    eligibility["eligible"][0, RUNNER["FIRST_ALARM_ELIGIBLE"]] = False
    corruptions.append(("eligibility", eligibility, "eligibility mask"))
    alarm = _evidence_arrays("audit", 10.0)
    alarm["alarm"][0, RUNNER["FIRST_ALARM_ELIGIBLE"]] = True
    corruptions.append(("alarm", alarm, "strict-threshold replay"))
    for name, arrays, message in corruptions:
        path = tmp_path / f"bad-{name}.npz"
        _save_evidence(path, arrays)
        with pytest.raises(ValueError, match=message):
            RUNNER["_validate_evidence"](
                path, phase="audit", operational_threshold=10.0
            )


@pytest.mark.parametrize("threshold", [-1.0, np.inf, -np.inf, np.nan])
def test_evidence_rejects_nonfinite_or_negative_threshold(
    tmp_path: Path, threshold: float
) -> None:
    arrays = _evidence_arrays("audit", threshold)
    path = tmp_path / "invalid-threshold.npz"
    _save_evidence(path, arrays)
    with pytest.raises(ValueError, match="threshold"):
        RUNNER["_validate_evidence"](
            path, phase="audit", operational_threshold=threshold
        )


def test_summary_boundary_and_each_failure_gate() -> None:
    summarize = RUNNER["_summarize_audit"]
    boundary = summarize(
        _healthy_records(4),
        _fault_records(detected_per_fault=90, delay=120),
        _parity_stub(),
        10.0,
    )
    assert boundary["passed"] is True
    assert boundary["decision"] == RUNNER["PASS_DECISION"]
    assert boundary["healthy"]["one_sided_clopper_pearson_95_upper"] < 0.10
    assert boundary["healthy"]["empirical_cp95_upper_below_0_05"] is False
    assert boundary["faults"]["detected_runs"] == 270

    too_many_healthy = summarize(
        _healthy_records(5),
        _fault_records(detected_per_fault=100, delay=10),
        _parity_stub(),
        10.0,
    )
    assert too_many_healthy["passed"] is False

    low_detection = summarize(
        _healthy_records(0),
        _fault_records(detected_per_fault=89, delay=10),
        _parity_stub(),
        10.0,
    )
    assert low_detection["passed"] is False

    slow_median = {fault: [121] * 100 for fault in RUNNER["AUDIT_FAULT_IDS"]}
    slow = summarize(
        _healthy_records(0),
        _fault_records(delay_overrides=slow_median),
        _parity_stub(),
        10.0,
    )
    assert slow["passed"] is False
    assert slow["faults"]["per_fault"]["15"]["passed_median_gate"] is False

    slow_p90 = {
        fault: [0] * 89 + [181] * 11 for fault in RUNNER["AUDIT_FAULT_IDS"]
    }
    p90 = summarize(
        _healthy_records(0),
        _fault_records(delay_overrides=slow_p90),
        _parity_stub(),
        10.0,
    )
    assert p90["passed"] is False
    assert p90["faults"]["per_fault"]["09"]["passed_p90_gate"] is False

    parity_failure = summarize(
        _healthy_records(0),
        _fault_records(detected_per_fault=100, delay=10),
        _parity_stub(eligibility=False),
        10.0,
    )
    assert parity_failure["passed"] is False
    assert parity_failure["parity_gate_passed"] is False


def test_completed_phase_reentry_never_touches_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration_sentinel = {"already": "calibrated"}
    calibration_function = RUNNER["_run_calibration"]
    monkeypatch.setitem(
        calibration_function.__globals__,
        "_verify_calibration_bundle",
        lambda *_args, **_kwargs: calibration_sentinel,
    )
    for name in (
        "_verify_incomplete_phase_inputs",
        "_source_records",
        "_label_and_topology_record",
        "ExtendedTEPMode1Archive",
    ):
        monkeypatch.setitem(calibration_function.__globals__, name, _forbidden_source)
    assert _call_calibration(calibration_function, tmp_path) is calibration_sentinel

    calibration = {
        "threshold": 10.0,
        "threshold_binary64_hex": (10.0).hex(),
        "completion_sha256": "d" * 64,
        "completion": {"completion_fingerprint": "e" * 64},
        "certificate_sha256": "f" * 64,
        "evidence_sha256": "1" * 64,
    }
    audit_sentinel = {"already": "audited"}
    audit_function = RUNNER["_run_audit"]
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
    for name in (
        "_verify_incomplete_phase_inputs",
        "_source_records",
        "_label_and_topology_record",
        "ExtendedTEPMode1Archive",
    ):
        monkeypatch.setitem(audit_function.__globals__, name, _forbidden_source)
    assert _call_audit(audit_function, tmp_path) is audit_sentinel


def test_phase_provenance_accepts_exact_calibration_replay(tmp_path: Path) -> None:
    protocol = _protocol_stub()
    target = tmp_path / "calibration"
    _write_provenance_fixture(
        target,
        phase="calibration",
        protocol=protocol,
        protocol_sha256=PROTOCOL_SHA,
        recorded_binding=None,
    )
    execution = RUNNER["_verify_phase_provenance"](
        target,
        phase="calibration",
        protocol=protocol,
        protocol_sha256=PROTOCOL_SHA,
        calibration_binding=None,
    )
    assert execution["runtime"] == _runtime_stub()


@pytest.mark.parametrize("corruption", ["intent", "source", "access", "runtime"])
def test_phase_provenance_rejects_consistently_rehashed_wrong_references(
    tmp_path: Path, corruption: str
) -> None:
    protocol = _protocol_stub()
    kwargs: dict[str, Any] = {}
    if corruption == "intent":
        kwargs["mutate_intent"] = lambda value: value["authorized_reads"].__setitem__(
            "source_idv", 2
        )
    elif corruption == "source":
        kwargs["source_records"] = {"wrong-source.h5": {"sha256": "0" * 64}}
    elif corruption == "access":
        kwargs["mutate_data_access"] = lambda value: value.__setitem__(
            "selected_runs", [2, 101]
        )
    else:
        wrong_runtime = _runtime_stub()
        wrong_runtime["device_name"] = "wrong GPU reference"
        kwargs["execution_runtime"] = wrong_runtime
    target = tmp_path / corruption
    _write_provenance_fixture(
        target,
        phase="calibration",
        protocol=protocol,
        protocol_sha256=PROTOCOL_SHA,
        recorded_binding=None,
        **kwargs,
    )
    with pytest.raises(ValueError):
        RUNNER["_verify_phase_provenance"](
            target,
            phase="calibration",
            protocol=protocol,
            protocol_sha256=PROTOCOL_SHA,
            calibration_binding=None,
        )


def test_audit_provenance_rejects_self_consistent_wrong_calibration_binding(
    tmp_path: Path,
) -> None:
    binding = {
        "calibration_completion_sha256": "1" * 64,
        "calibration_completion_fingerprint": "2" * 64,
        "pac_certificate_sha256": "3" * 64,
        "calibration_evidence_sha256": "4" * 64,
        "operational_threshold": 10.0,
        "operational_threshold_binary64_hex": (10.0).hex(),
        "threshold_source": "CPU_NumPy_reference",
        "calibration_decision": RUNNER["CALIBRATION_DECISION"],
    }
    wrong_binding = copy.deepcopy(binding)
    wrong_binding["calibration_completion_sha256"] = "9" * 64
    target = tmp_path / "audit"
    _write_provenance_fixture(
        target,
        phase="audit",
        protocol=_protocol_stub(),
        protocol_sha256=PROTOCOL_SHA,
        recorded_binding=wrong_binding,
    )
    with pytest.raises(ValueError, match="access intent"):
        RUNNER["_verify_phase_provenance"](
            target,
            phase="audit",
            protocol=_protocol_stub(),
            protocol_sha256=PROTOCOL_SHA,
            calibration_binding=binding,
        )


def test_partial_and_staging_calibration_block_before_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["_run_calibration"]
    for name in (
        "_verify_incomplete_phase_inputs",
        "_source_records",
        "_label_and_topology_record",
        "ExtendedTEPMode1Archive",
    ):
        monkeypatch.setitem(function.__globals__, name, _forbidden_source)

    partial = tmp_path / "output" / "calibration"
    partial.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="must not be reopened"):
        _call_calibration(function, tmp_path)

    partial.rmdir()
    staging = tmp_path / "output" / ".calibration.inprogress-crash"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="must not be reopened"):
        _call_calibration(function, tmp_path)


def test_audit_without_calibration_blocks_before_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["_run_audit"]
    for name in (
        "_verify_incomplete_phase_inputs",
        "_source_records",
        "_label_and_topology_record",
        "ExtendedTEPMode1Archive",
    ):
        monkeypatch.setitem(function.__globals__, name, _forbidden_source)
    with pytest.raises(RuntimeError, match="completed q100 calibration"):
        _call_audit(function, tmp_path)


def test_partial_or_staging_audit_blocks_before_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["_run_audit"]
    calibration = {
        "threshold": 10.0,
        "threshold_binary64_hex": (10.0).hex(),
        "completion_sha256": "d" * 64,
        "completion": {"completion_fingerprint": "e" * 64},
        "certificate_sha256": "f" * 64,
        "evidence_sha256": "1" * 64,
    }
    monkeypatch.setitem(
        function.__globals__,
        "_verify_calibration_bundle",
        lambda *_args, **_kwargs: calibration,
    )
    for name in (
        "_verify_incomplete_phase_inputs",
        "_source_records",
        "_label_and_topology_record",
        "ExtendedTEPMode1Archive",
    ):
        monkeypatch.setitem(function.__globals__, name, _forbidden_source)

    partial = tmp_path / "output" / "audit"
    partial.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="must not be reopened"):
        _call_audit(function, tmp_path)

    partial.rmdir()
    staging = tmp_path / "output" / ".audit.inprogress-crash"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="must not be reopened"):
        _call_audit(function, tmp_path)


def test_phase_manifest_precedes_every_numeric_archive_read() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for function_name, loader_method in (
        ("_run_calibration", "load_calibration_runs"),
        ("_run_audit", "load_audit_runs"),
    ):
        function = functions[function_name]
        manifest_lines: list[int] = []
        numeric_lines: list[int] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            called = ast.unparse(node.func)
            if (
                called.endswith("._write_json")
                and node.args
                and "phase_manifest.json" in ast.unparse(node.args[0])
            ):
                manifest_lines.append(node.lineno)
            if called.endswith(f".{loader_method}"):
                numeric_lines.append(node.lineno)
        assert len(manifest_lines) == 1
        assert len(numeric_lines) == 1
        assert manifest_lines[0] < numeric_lines[0]


def test_protocol_contract_accepts_exact_frozen_design() -> None:
    RUNNER["_assert_protocol_contract"](_protocol_stub())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["external_calibration"].__setitem__(
            "simulation_runs", list(range(2, 102))
        ),
        lambda p: p["external_calibration"].__setitem__(
            "normalized_window_half_open", [1, 501]
        ),
        lambda p: p["external_calibration"]["rank"].__setitem__(
            "rank_one_based", 99
        ),
        lambda p: p["external_audit"]["faults"].__setitem__(
            "local_fault_start_zero_based", 159
        ),
        lambda p: p["monitor_binding"]["monitor_configuration"].__setitem__(
            "comparison", "greater_than_or_equal"
        ),
        lambda p: p["success_gates"].__setitem__(
            "healthy_alarm_runs_maximum", 5
        ),
        lambda p: p.__setitem__("forbidden_actions", []),
    ],
)
def test_protocol_contract_rejects_material_changes(
    mutate: Callable[[dict[str, Any]], None]
) -> None:
    protocol = _protocol_stub()
    mutate(protocol)
    with pytest.raises(ValueError):
        RUNNER["_assert_protocol_contract"](protocol)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["external_audit"]["healthy"].__setitem__("source_idv", 2),
        lambda p: p["external_audit"]["healthy"].__setitem__(
            "normalized_window_half_open", [1, 501]
        ),
        lambda p: p["external_audit"].__setitem__(
            "pre_fault_alarm_continuing_across_onset_is_not_a_detection", False
        ),
    ],
)
def test_protocol_contract_rejects_healthy_or_episode_semantic_changes(
    mutate: Callable[[dict[str, Any]], None]
) -> None:
    protocol = _protocol_stub()
    mutate(protocol)
    with pytest.raises(ValueError):
        RUNNER["_assert_protocol_contract"](protocol)


def _freeze_with_synthetic_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template: dict[str, Any] | None = None,
) -> tuple[Path, Path, str]:
    protocol = tmp_path / "protocol.json"
    sidecar = protocol.with_suffix(".sha256")
    frozen = {"synthetic": "protocol", "nested": {"value": 1}} if template is None else template
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: frozen,
    )
    digest = function(tmp_path, tmp_path / "data", protocol)
    return protocol, sidecar, digest


def _freeze_temporaries(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if ".freeze-" in path.name]


def test_freeze_prepares_fsynced_temps_then_publishes_exact_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = {"synthetic": "protocol", "unicode": "外部确认", "value": [1, 2, 3]}
    protocol, sidecar, digest = _freeze_with_synthetic_template(
        tmp_path, monkeypatch, template
    )

    assert RUNNER["ext"]._sha256_file(protocol) == digest
    assert RUNNER["ext"]._strict_json(protocol) == template
    assert sidecar.read_text(encoding="utf-8") == f"{digest}  protocol.json\n"
    assert _freeze_temporaries(tmp_path) == []


def test_freeze_text_write_failure_removes_only_owned_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: {"synthetic": "protocol"},
    )

    def fail_after_closing(descriptor: int, _value: str) -> None:
        os.close(descriptor)
        raise OSError("injected sidecar write failure")

    monkeypatch.setitem(
        function.__globals__, "_write_fsynced_text_descriptor", fail_after_closing
    )
    protocol = tmp_path / "protocol.json"
    with pytest.raises(OSError, match="injected sidecar"):
        function(tmp_path, tmp_path / "data", protocol)

    assert not protocol.exists()
    assert not protocol.with_suffix(".sha256").exists()
    assert _freeze_temporaries(tmp_path) == []


def test_freeze_second_publish_failure_rolls_back_protocol_and_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: {"synthetic": "protocol"},
    )
    real_replace = function.__globals__["os"].replace
    calls = 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected commit-marker publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(function.__globals__["os"], "replace", fail_second_replace)
    protocol = tmp_path / "protocol.json"
    with pytest.raises(OSError, match="commit-marker"):
        function(tmp_path, tmp_path / "data", protocol)

    assert calls == 2
    assert not protocol.exists()
    assert not protocol.with_suffix(".sha256").exists()
    assert _freeze_temporaries(tmp_path) == []


def test_freeze_first_publish_failure_removes_both_owned_temporaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: {"synthetic": "protocol"},
    )

    def fail_first_replace(_source: object, _destination: object) -> None:
        raise OSError("injected protocol publish failure")

    monkeypatch.setattr(function.__globals__["os"], "replace", fail_first_replace)
    protocol = tmp_path / "protocol.json"
    with pytest.raises(OSError, match="protocol publish"):
        function(tmp_path, tmp_path / "data", protocol)

    assert not protocol.exists()
    assert not protocol.with_suffix(".sha256").exists()
    assert _freeze_temporaries(tmp_path) == []


def test_freeze_rollback_never_deletes_an_unowned_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: {"synthetic": "protocol"},
    )

    def create_unowned_target_then_fail(
        _source: object, destination: object
    ) -> None:
        Path(destination).write_text("racing-owner-bytes", encoding="utf-8")
        raise OSError("injected racing publication")

    monkeypatch.setattr(
        function.__globals__["os"], "replace", create_unowned_target_then_fail
    )
    protocol = tmp_path / "protocol.json"
    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        function(tmp_path, tmp_path / "data", protocol)

    assert protocol.read_text(encoding="utf-8") == "racing-owner-bytes"
    assert not protocol.with_suffix(".sha256").exists()
    assert _freeze_temporaries(tmp_path) == []


@pytest.mark.parametrize("preexisting", ["protocol", "sidecar"])
def test_freeze_never_overwrites_or_deletes_preexisting_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: str
) -> None:
    protocol = tmp_path / "protocol.json"
    sidecar = protocol.with_suffix(".sha256")
    protected = protocol if preexisting == "protocol" else sidecar
    protected.write_text("preexisting-owner-bytes", encoding="utf-8")
    function = RUNNER["freeze_protocol"]
    monkeypatch.setitem(
        function.__globals__,
        "build_protocol_template",
        lambda *_args, **_kwargs: pytest.fail(
            "template construction must not start with a preexisting target"
        ),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        function(tmp_path, tmp_path / "data", protocol)

    assert protected.read_text(encoding="utf-8") == "preexisting-owner-bytes"
    other = sidecar if protected == protocol else protocol
    assert not other.exists()
    assert _freeze_temporaries(tmp_path) == []


def test_protocol_without_commit_marker_is_fail_closed(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    digest = RUNNER["ext"]._sha256_file(protocol)

    with pytest.raises(ValueError, match="sidecar"):
        RUNNER["_load_protocol"](protocol, digest)
