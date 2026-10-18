from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest

from mate_spc_pac import pac_tolerance_rank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_hybrid_pac_q100_holdout_gpu.py"
SPEC = importlib.util.spec_from_file_location("q100_holdout_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _healthy(count: int = 0) -> list[dict[str, object]]:
    return [
        {
            "run": f"h-{index}",
            "simulation_run": 121 + index,
            "any_alarm": index < count,
            "alarm_episode_count": int(index < count),
            "first_alarm_index": 100 if index < count else None,
            "maximum_envelope": 300.0 if index < count else 100.0,
        }
        for index in range(80)
    ]


def _faults(*, detected: int = 100, delay: int = 80) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fault_id in runner.FAULT_IDS:
        for index in range(100):
            hit = index < detected
            rows.append(
                {
                    "run": f"f-{fault_id}-{index}",
                    "fault_id": fault_id,
                    "simulation_run": 301 + index,
                    "detected": hit,
                    "first_fault_alarm_episode_index": runner.FAULT_START + delay if hit else None,
                    "delay_samples": delay if hit else None,
                    "restricted_delay_samples_miss_at_run_end": delay if hit else runner.MISS_DELAY,
                    "pre_fault_alarm_episode_count": 0,
                    "pre_fault_interval_used_as_performance_evidence": False,
                    "attributed_fault_head_at_first_alarm": fault_id if hit else None,
                    "attribution_matches_fault": hit,
                    "maximum_post_fault_envelope": 1000.0 if hit else 100.0,
                }
            )
    return rows


PARITY = {
    "all_numeric_paths_within_tolerance": True,
    "all_alarm_arrays_exact": True,
}


def test_split_and_horizon_are_exactly_frozen() -> None:
    assert runner.HEALTHY_RUNS == tuple(range(121, 201))
    assert runner.FAULT_RUNS == tuple(range(301, 401))
    assert runner.FAULT_START == 160
    assert runner.SEQUENCE_LENGTH == 500
    assert runner.MISS_DELAY == 340
    assert len(runner.HEALTHY_RUNS) + 3 * len(runner.FAULT_RUNS) == 380


def test_directional_base_helpers_are_directly_bound_without_ext_base() -> None:
    assert runner.base.__name__ == "run_directional_glr_gpu"
    for name in ("_run_ids", "_fault_run_ids", "_validate_full_values", "_truncate_run"):
        assert callable(getattr(runner.base, name))
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "base"
        and isinstance(node.value, ast.Name)
        and node.value.id == "ext"
        for node in ast.walk(tree)
    )


def test_q100_delta01_certificate_rank_is_the_maximum() -> None:
    rank = pac_tolerance_rank(100, epsilon=0.05, delta=0.01)
    assert rank.rank_one_based == 100
    assert rank.achieved_confidence >= 0.99
    assert rank.upper_tail_probability_bound <= 0.01
    assert rank.minimum_sample_count == 90


def test_q100_threshold_is_bitwise_frozen() -> None:
    assert runner.Q100_THRESHOLD == 236.3968052224969
    assert runner.Q100_THRESHOLD.hex() == runner.Q100_THRESHOLD_HEX


def test_protocol_recomputes_q100_from_safe_consumed_evidence_only() -> None:
    protocol = runner.build_protocol_template(PROJECT_ROOT)
    revised = protocol["revised_pac_calibration"]
    assert revised["rank_one_based"] == 100
    assert revised["threshold"] == runner.Q100_THRESHOLD
    assert revised["certificate"]["rank"]["delta"] == 0.01
    assert revised["certificate"]["rank"]["achieved_confidence"] >= 0.99
    assert revised["rank_driver_run"] == "Rieth-fault_free_training-fault-00-run-338"
    assert revised["canonical_calibration_maxima_sha256"] == runner.Q100_CALIBRATION_MAXIMA_SHA256
    assert revised["calibration_completion"]["decision"] == runner.ext.CALIBRATION_DECISION
    assert revised["failed_q99_audit_completion"]["decision"] == runner.ext.FAIL_DECISION
    data = protocol["row_level_holdout_audit_data"]
    assert data["healthy"]["simulation_runs"] == list(range(121, 201))
    assert data["faults"]["simulation_runs"] == list(range(301, 401))
    assert protocol["governance"]["q100_rule_selected_after_q99_result_was_observed"] is True
    assert protocol["analysis_status"] == runner.ANALYSIS_STATUS
    assert protocol["governance"]["not_an_independent_external_or_final_confirmation"] is True


def test_protocol_does_not_reuse_old_401_500_rows() -> None:
    protocol = runner.build_protocol_template(PROJECT_ROOT)
    selected = (
        protocol["row_level_holdout_audit_data"]["healthy"]["simulation_runs"]
        + protocol["row_level_holdout_audit_data"]["faults"]["simulation_runs"]
    )
    assert not set(selected).intersection(range(401, 501))
    assert protocol["row_level_holdout_audit_data"]["faults"]["fault_start_zero_based"] == 160


def test_protocol_discloses_both_material_posthoc_q100_replays() -> None:
    protocol = runner.build_protocol_template(PROJECT_ROOT)
    evidence = protocol["posthoc_design_evidence"]
    assert evidence["materially_influenced_most_likely_to_succeed_selection"] is True
    old = evidence["old_hybrid_testing_audit_q100_replay"]
    external = evidence["external_training_partition_audit_q100_replay"]
    assert old["evidence"]["sha256"] == runner.OLD_HYBRID_AUDIT_SCORES_SHA256
    assert external["evidence"]["sha256"] == runner.EXTERNAL_AUDIT_EVIDENCE_SHA256
    assert (old["healthy_alarm_runs"], external["healthy_alarm_runs"]) == (1, 4)
    assert old["faults"]["09"]["median_delay"] == 100.0
    assert external["faults"]["15"]["p90_delay"] == 139.0


def test_protocol_binds_aborted_v11_and_limits_v12_to_wiring_fix() -> None:
    protocol = runner.build_protocol_template(PROJECT_ROOT)
    assert protocol["schema_version"] == 2
    failure = protocol["governance"]["v1_1_infrastructure_failure"]
    assert failure["superseded_protocol"]["sha256"] == runner.ABORTED_V1_1_PROTOCOL_SHA256
    assert failure["failure_record"]["sha256"] == runner.ABORTED_V1_1_FAILURE_RECORD_SHA256
    assert failure["archive_load_runs_call_count"] == 0
    assert failure["source_raw_byte_hashing_and_MD5_verification_occurred"] is True
    assert failure["pyreadr_import_or_read_r_occurred"] is False
    assert failure["holdout_numeric_results_observed"] is False
    assert failure["v1_2_change_scope"] == "implementation namespace wiring only"
    assert failure["algorithm_unchanged"] is True
    assert failure["q100_threshold_and_certificate_unchanged"] is True
    assert failure["row_level_split_unchanged"] is True
    assert failure["success_gates_unchanged"] is True


def test_episode_start_semantics_exclude_persistent_prefault_alarm() -> None:
    alarm = np.zeros(500, dtype=bool)
    alarm[150:170] = True
    alarm[180:183] = True
    starts = runner._episode_starts(alarm)
    assert starts.tolist() == [150, 180]
    assert starts[starts >= runner.FAULT_START].tolist() == [180]


def test_exact_boundary_summary_passes_all_frozen_gates() -> None:
    result = runner._summary(_healthy(3), _faults(detected=90, delay=120), PARITY)
    assert result["passed"] is True
    assert result["decision"] == runner.PASS_DECISION
    assert result["analysis_status"] == runner.ANALYSIS_STATUS
    assert result["phase"] == runner.ANALYSIS_STATUS
    assert result["healthy"]["one_sided_clopper_pearson_95_upper"] < 0.10
    assert result["faults"]["detected_runs"] == 270
    assert result["faults"]["attribution_is_report_only"] is True
    assert "does not establish p<0.05" in result["healthy_CP95_interpretation"]


def test_four_healthy_alarms_fail_without_changing_other_gates() -> None:
    result = runner._summary(_healthy(4), _faults(), PARITY)
    assert result["passed"] is False
    assert result["decision"] == runner.FAIL_DECISION
    assert result["healthy"]["passed"] is False


def test_delay_and_parity_gates_are_required() -> None:
    slow = runner._summary(_healthy(0), _faults(delay=181), PARITY)
    assert slow["passed"] is False
    bad_parity = runner._summary(
        _healthy(0), _faults(),
        {"all_numeric_paths_within_tolerance": True, "all_alarm_arrays_exact": False},
    )
    assert bad_parity["passed"] is False


def _evidence_arrays() -> dict[str, np.ndarray]:
    count = 380
    names = (
        [f"Rieth-fault_free_training-fault-00-run-{run:03d}" for run in runner.HEALTHY_RUNS]
        + [
            f"Rieth-faulty_testing-fault-{fault:02d}-run-{run:03d}"
            for fault in runner.FAULT_IDS
            for run in runner.FAULT_RUNS
        ]
    )
    faults = [0] * 80 + [fault for fault in runner.FAULT_IDS for _ in runner.FAULT_RUNS]
    runs = list(runner.HEALTHY_RUNS) + list(runner.FAULT_RUNS) * 3
    partitions = ["fault_free_training"] * 80 + ["faulty_testing"] * 300
    vector = np.zeros((count, 500), dtype=np.float64)
    vector[:, : runner.FIRST_ALARM_ELIGIBLE] = -np.inf
    heads = np.zeros((count, 500, 3), dtype=np.float64)
    eligible = np.zeros((count, 500), dtype=bool)
    eligible[:, runner.FIRST_ALARM_ELIGIBLE :] = True
    return {
        "run_names": np.asarray(names),
        "partitions": np.asarray(partitions),
        "fault_ids": np.asarray(faults, dtype=np.int64),
        "simulation_runs": np.asarray(runs, dtype=np.int64),
        "logits": heads.copy(),
        "standardized_logits": heads.copy(),
        "clipped_evidence": heads.copy(),
        "cusum": heads.copy(),
        "component_persistence": heads.copy(),
        "envelope": vector.copy(),
        "eligible": eligible,
        "alarm": np.zeros((count, 500), dtype=bool),
        "operational_threshold": np.asarray([runner.Q100_THRESHOLD]),
        "fault_start_zero_based": np.asarray([runner.FAULT_START], dtype=np.int64),
    }


def test_evidence_validator_enforces_exact_ids_and_strict_threshold() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "evidence.npz"
        np.savez_compressed(path, **_evidence_arrays())
        arrays = runner._validate_evidence(path)
        assert arrays["fault_ids"].shape == (380,)

        changed = _evidence_arrays()
        changed["simulation_runs"][0] = 999
        np.savez_compressed(path, **changed)
        with pytest.raises(ValueError, match="run identities"):
            runner._validate_evidence(path)


def test_evidence_validator_rejects_object_arrays() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "unsafe.npz"
        arrays = _evidence_arrays()
        arrays["run_names"] = np.asarray([object()] * 380, dtype=object)
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match="Object arrays|unsafe"):
            runner._validate_evidence(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("fault_ids", lambda a: a.astype(np.int32), "identities must be int64"),
        ("eligible", lambda a: a.astype(np.uint8), "eligible/alarm fields must be bool"),
        ("operational_threshold", lambda a: np.asarray([a[0], a[0]]), "shape changed"),
        ("logits", lambda a: a[:, :-1, :], "shape changed"),
    ],
)
def test_evidence_validator_rejects_every_shape_and_dtype_contract(
    field: str, replacement, message: str
) -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bad.npz"
        arrays = _evidence_arrays()
        arrays[field] = replacement(arrays[field])
        np.savez_compressed(path, **arrays)
        with pytest.raises(ValueError, match=message):
            runner._validate_evidence(path)


def test_evidence_validator_rejects_name_eligibility_and_envelope_corruption() -> None:
    corruptions = []
    names = _evidence_arrays()
    names["run_names"][0] = "wrong"
    corruptions.append((names, "exact run names"))
    eligible = _evidence_arrays()
    eligible["eligible"][0, 0] = True
    corruptions.append((eligible, "eligibility mask"))
    preeligible = _evidence_arrays()
    preeligible["envelope"][0, 0] = 0.0
    corruptions.append((preeligible, "pre-eligible envelope"))
    nonfinite = _evidence_arrays()
    nonfinite["envelope"][0, runner.FIRST_ALARM_ELIGIBLE] = np.nan
    corruptions.append((nonfinite, "eligible envelope"))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bad.npz"
        for arrays, message in corruptions:
            np.savez_compressed(path, **arrays)
            with pytest.raises(ValueError, match=message):
                runner._validate_evidence(path)


def test_freeze_writes_one_immutable_protocol_and_sidecar() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "protocol.json"
        digest = runner.freeze_protocol(PROJECT_ROOT, path)
        assert runner.ext._sha256_file(path) == digest
        assert path.with_suffix(".sha256").read_text(encoding="utf-8") == (
            f"{digest}  {path.name}\n"
        )
        with pytest.raises(FileExistsError):
            runner.freeze_protocol(PROJECT_ROOT, path)
