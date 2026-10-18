from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_pclr_cusum_gpu as runner


def _candidate(candidate: dict[str, object], *, delay: int = 10, misses: int = 0) -> dict[str, object]:
    records: list[dict[str, object]] = []
    healthy_records: list[dict[str, object]] = []
    for fault in runner.FAULT_IDS:
        fault_rows = []
        for run in runner.DEVELOPMENT_RUNS:
            fold = runner._fold_assignment()[str(run)]
            fault_rows.append((run, fold))
        # Spread misses deterministically across the fixed folds.
        ordered = sorted(fault_rows, key=lambda item: (item[1], item[0]))
        missed = {run for run, _ in ordered[:misses]}
        for run, fold in fault_rows:
            detected = run not in missed
            records.append(
                {
                    "fault_id": fault,
                    "simulation_run": run,
                    "fold": fold,
                    "detected": detected,
                    "restricted_delay_samples": delay if detected else runner.MISS_DELAY,
                }
            )
    for run in runner.DEVELOPMENT_RUNS:
        healthy_records.append(
            {
                "simulation_run": run,
                "fold": runner._fold_assignment()[str(run)],
                "alarmed": False,
            }
        )
    fold_calibrations = []
    for spec in runner._fold_specs():
        maxima = np.arange(1.0, 81.0, dtype=np.float64)
        fold_calibrations.append(
            runner._development_fold_threshold_summary(
                {
                    "training_healthy_cpu_run_maxima": maxima,
                    "training_healthy_cuda_run_maxima": maxima.copy(),
                    "operational_threshold_q79": 79.0,
                    "parity": {
                        "max_abs_score_error": 0.0,
                    },
                },
                spec,
            )
        )
    return {
        "candidate_id": candidate["candidate_id"],
        "params": {
            "varx_ridge": candidate["varx_ridge"],
            "classifier_ridge": candidate["classifier_ridge"],
            "page_drift": candidate["page_drift"],
        },
        "records": records,
        "healthy_records": healthy_records,
        "fold_calibrations": fold_calibrations,
        "fold_fits": [
            {
                **spec,
                "candidate_id": candidate["candidate_id"],
                "fit_from_raw_training_groups_only": True,
                "fold_local_components_refit": runner._fold_fit_contract()["fold_local_components_refit"],
            }
            for spec in runner._fold_specs()
        ],
        "parity": {
            "eligibility_exact": True,
            "alarm_exact": True,
            "max_abs_score_error": 0.0,
            "attribution_exact": True,
        },
    }


def _score_arrays(count: int) -> dict[str, np.ndarray]:
    eligible = np.zeros((count, runner.SEQUENCE_LENGTH), dtype=np.bool_)
    eligible[:, runner.FIRST_ALARM_ELIGIBLE :] = True
    scores = np.zeros((count, runner.SEQUENCE_LENGTH), dtype=np.float64)
    result = {
        "cpu_scores": scores,
        "cuda_scores": scores.copy(),
        "cpu_eligible": eligible,
        "cuda_eligible": eligible.copy(),
    }
    component_shape = (
        count,
        runner.SEQUENCE_LENGTH,
        len(runner.FAULT_IDS),
    )
    for field in ("standardized_logits", "cusum", "component_persistence"):
        result[f"cpu_{field}"] = np.zeros(component_shape, dtype=np.float64)
        result[f"cuda_{field}"] = np.zeros(component_shape, dtype=np.float64)
    return result


def _full_score_payload(standardized: np.ndarray, page_drift: float = 0.0) -> dict[str, np.ndarray]:
    cusum, persistent, scores, eligible = runner._replay_page_persistence(
        standardized, page_drift
    )
    payload: dict[str, np.ndarray] = {}
    for prefix in ("cpu", "cuda"):
        payload[f"{prefix}_standardized_logits"] = standardized.copy()
        payload[f"{prefix}_cusum"] = cusum.copy()
        payload[f"{prefix}_component_persistence"] = persistent.copy()
        payload[f"{prefix}_scores"] = scores.copy()
        payload[f"{prefix}_eligible"] = eligible.copy()
    return payload


def _synthetic_model_arrays(selected_params: dict[str, object]) -> dict[str, np.ndarray]:
    from mate_spc_pclr.backend import _model_arrays
    from mate_spc_pclr.model import (
        CAUSAL_LAGS,
        FEATURE_COUNT,
        PCLR_FEATURE_COUNT,
        PCLRConfig,
        PCLRFeatureModel,
        PCLRHeads,
        PCLRModel,
    )

    params = {key: float(selected_params[key]) for key in selected_params}
    feature_model = PCLRFeatureModel(
        mad_floor=np.ones(FEATURE_COUNT, dtype=np.float64),
        var_coefficients=np.zeros(
            (1 + FEATURE_COUNT * len(CAUSAL_LAGS), FEATURE_COUNT),
            dtype=np.float64,
        ),
        config=PCLRConfig(var_ridge_alpha=params["varx_ridge"]),
    )
    heads = PCLRHeads(
        weights=np.zeros((len(runner.FAULT_IDS), PCLR_FEATURE_COUNT), dtype=np.float64),
        bias=np.zeros(len(runner.FAULT_IDS), dtype=np.float64),
        healthy_logit_center=np.zeros(len(runner.FAULT_IDS), dtype=np.float64),
        healthy_logit_scale=np.ones(len(runner.FAULT_IDS), dtype=np.float64),
        kappa=np.full(
            len(runner.FAULT_IDS), params["page_drift"], dtype=np.float64
        ),
        threshold=None,
    )
    return _model_arrays(
        PCLRModel(feature_model, heads),
        metadata={
            "backend": "mate_spc_pclr.backend",
            "fit_all_components_from_scratch": True,
            "development_group_count": len(runner.DEVELOPMENT_RUNS),
            "development_run_count": len(runner.DEVELOPMENT_RUNS)
            * len(runner.ALL_IDS),
            "development_runs": [
                min(runner.DEVELOPMENT_RUNS),
                max(runner.DEVELOPMENT_RUNS),
            ],
            "selected_params": params,
            "final_refit_steps": runner.FINAL_REFIT_STEPS,
            "steps": runner.FINAL_REFIT_STEPS,
            "head_fit": {
                "requested_device": "cuda",
                "dtype": "float64",
                "epochs": runner.FINAL_REFIT_STEPS,
                "ridge_lambda": params["classifier_ridge"],
                "prefix_rank_contract": "causal_local_128_160_mean_vs_post_160_500_top_quartile",
                "prefix_rank_weight": 0.1,
            },
        },
    )


def _minimal_protocol(output_root: Path) -> dict[str, object]:
    protocol = {
        "schema": "pclr_development_protocol",
        "schema_version": runner.SCHEMA_VERSION,
        "method": runner.METHOD_SHORT,
        "monitor": runner._monitor_contract(),
        "design_document": {"path": runner.DESIGN_DOCUMENT, "size_bytes": 1, "sha256": "c" * 64},
        "source_record": {"name": runner.DATA_FILE, "size_bytes": 1, "sha256": "a" * 64, "md5": "b" * 32},
        "source_metadata_record": {"synthetic": True, "numeric_values_decoded": False},
        "implementation_records": {runner.DESIGN_DOCUMENT: {"size_bytes": 1, "sha256": "c" * 64}},
        "governance": runner._governance_contract(output_root),
        "development": {
            "source_ids": list(runner.ALL_IDS),
            "simulation_runs": list(runner.DEVELOPMENT_RUNS),
            "grouping": "simulation_run_shared_across_all_four_IDVs",
            "healthy_source": {"idv": 1, "window_half_open": list(runner.HEALTHY_SOURCE_WINDOW)},
            "fault_source": {"idvs": list(runner.FAULT_IDS), "window_half_open": list(runner.FAULT_SOURCE_WINDOW)},
            "fold_count": runner.FOLD_COUNT,
            "fold_salt": runner.FOLD_SALT,
            "run_to_fold": runner._fold_assignment(),
            "run_to_fold_sha256": runner._selection_digest(runner._fold_assignment()),
            "fold_specs": runner._fold_specs(),
            "fold_fit_contract": runner._fold_fit_contract(),
            "fold_threshold": {
                "training_healthy_count": runner.DEVELOPMENT_HEALTHY_COUNT_PER_FOLD,
                "ascending_order_statistic_rank": runner.DEVELOPMENT_THRESHOLD_RANK,
                "rule": "79th_ascending_of_80_training_healthy_run_maxima_noninterpolated",
                "comparison": "strictly_greater_than",
                "validation_healthy_used_for_threshold": False,
                "minimum_numerically_distinct_maxima": runner.DEVELOPMENT_MIN_UNIQUE_MAXIMA,
                "q79_strictly_below_q80": True,
                "numeric_parity_atol": runner.PARITY_ATOL,
                "nondegeneracy_tail_rule": "q80_minus_q76_strictly_greater_than_100_times_max(observed_full_path_CPU_CUDA_error,observed_run_maxima_CPU_CUDA_error,float_tolerance,parity_atol)",
                "known_numerical_or_algorithmic_limits": list(runner.KNOWN_NUMERICAL_LIMITS),
            },
            "fixed_grid": runner._grid_payload(),
            "fixed_grid_sha256": runner._selection_digest(runner._grid_payload()),
            "selection_order": [
                "maximize_minimum_fault_detection_rate",
                "minimize_worst_fault_nearest_rank_p90_delay",
                "minimize_out_of_fold_healthy_alarm_runs",
                "maximize_classifier_ridge",
                "maximize_varx_ridge",
                "maximize_page_drift",
                "canonical_candidate_id",
            ],
            "gates": runner.DEVELOPMENT_GATES,
            "final_refit_steps": runner.FINAL_REFIT_STEPS,
            "calibration_for_early_stopping": False,
            "cpu_cuda_float64_parity_atol": runner.PARITY_ATOL,
        },
    }
    return protocol


def _minimal_confirmation_protocol(output_root: Path) -> dict[str, object]:
    development = _minimal_protocol(output_root)
    selected = runner.GRID[0]
    return {
        "method": runner.METHOD_SHORT,
        "monitor": runner._monitor_contract(),
        "design_document": development["design_document"],
        "implementation_records": development["implementation_records"],
        "governance": runner._governance_contract(output_root),
        "development_binding": {
            "protocol_sha256": "a" * 64,
            "completion_sha256": "b" * 64,
            "result_sha256": "c" * 64,
            "selected_candidate_id": selected["candidate_id"],
            "selected_params": {
                key: selected[key]
                for key in ("varx_ridge", "classifier_ridge", "page_drift")
            },
            "model_record": {"size_bytes": 1, "sha256": "d" * 64},
            "model_absolute_path": str(
                (output_root.resolve() / "develop" / "pclr_model.npz").resolve()
            ),
        },
        "calibration": {
            "healthy_source_idv": 1,
            "simulation_runs": list(runner.CALIBRATION_RUNS),
            "source_window_half_open": list(runner.HEALTHY_SOURCE_WINDOW),
            "healthy_window_excludes_first_affected_global_index": 600,
            "pseudo_onset_local_zero_based": runner.FAULT_START,
            "rank": runner.CALIBRATION_RANK,
            "rule": "99th_ascending_of_100_run_maxima_noninterpolated",
            "comparison": "strictly_greater_than",
            "epsilon": runner.CALIBRATION_EPSILON,
            "delta": runner.CALIBRATION_DELTA,
            "certificate_assumption": "100_IID_simulation_run_groups_from_one_fixed_healthy_distribution",
            "minimum_numerically_distinct_maxima": runner.MIN_UNIQUE_MAXIMA,
            "q99_strictly_below_q100": True,
            "top_tail": {
                "lower_rank": runner.TOP_TAIL_RANK,
                "upper_rank": 100,
                "safety_multiplier": runner.TOP_TAIL_TOLERANCE_MULTIPLIER,
                "reference": "max(observed_full_path_CPU_CUDA_error,float_tolerance,parity_atol)",
                "comparison": "spread_strictly_greater_than_multiplier_times_reference",
            },
            "float_tolerance": runner.FLOAT_TOLERANCE,
            "parity_atol": runner.PARITY_ATOL,
            "known_numerical_or_algorithmic_limits": list(
                runner.KNOWN_NUMERICAL_LIMITS
            ),
        },
        "confirmation": {
            "healthy": {
                "source_idv": 1,
                "simulation_runs": list(runner.CONFIRMATION_RUNS),
                "source_window_half_open": list(runner.HEALTHY_SOURCE_WINDOW),
                "excludes_first_affected_global_index": 600,
                "pseudo_onset_local_zero_based": runner.FAULT_START,
            },
            "faults": {
                "source_idvs": list(runner.FAULT_IDS),
                "simulation_runs": list(runner.CONFIRMATION_RUNS),
                "source_window_half_open": list(runner.FAULT_SOURCE_WINDOW),
            },
            "local_fault_start_zero_based": runner.FAULT_START,
            "detection_requires_new_alarm_episode_start_at_or_after": runner.FAULT_START,
            "gates": runner.CONFIRMATION_GATES,
            "one_time_even_if_decision_is_fail": True,
        },
    }


def test_partition_and_window_contract_excludes_all_consumed_rows() -> None:
    assert runner.HEALTHY_SOURCE_WINDOW == (100, 600)
    assert runner.FAULT_SOURCE_WINDOW == (440, 940)
    assert runner.FAULT_START == 160
    assert runner.MISS_DELAY == 340
    assert max(runner.CONSUMED_RUNS) == 200
    assert min(runner.DEVELOPMENT_RUNS) == 201
    assert min(runner.CALIBRATION_RUNS) == 301
    assert min(runner.CONFIRMATION_RUNS) == 401
    plans = (
        runner._development_access_plan(_minimal_protocol(Path("synthetic"))),
        runner._calibration_access_plan(),
        runner._confirmation_access_plan(),
    )
    assert all(row["simulation_run"] > 200 for plan in plans for row in plan)
    healthy = [row for plan in plans for row in plan if row["idv"] == 1]
    assert healthy and all(row["window_half_open"] == [100, 600] for row in healthy)
    faults = [row for plan in plans for row in plan if row["idv"] in runner.FAULT_IDS]
    assert faults and all(row["window_half_open"] == [440, 940] for row in faults)


def test_fixed_hash_folds_group_all_four_idvs_and_balance_groups() -> None:
    assignment = runner._fold_assignment()
    assert len(assignment) == 100
    assert [sum(value == fold for value in assignment.values()) for fold in range(5)] == [20] * 5
    plan = runner._development_access_plan(_minimal_protocol(Path("synthetic")))
    for run in runner.DEVELOPMENT_RUNS:
        rows = [row for row in plan if row["simulation_run"] == run]
        assert [row["idv"] for row in rows] == [1, 3, 9, 15]
        assert len({assignment[str(row["simulation_run"])] for row in rows}) == 1


def test_protocol_assertions_bind_design_sha_and_final_model_path(
    tmp_path: Path,
) -> None:
    development = _minimal_protocol(tmp_path / "output")
    runner._assert_development_protocol(development)
    development["design_document"] = {
        **development["design_document"],
        "path": "docs/changed.md",
    }
    with pytest.raises(ValueError, match="design-document path"):
        runner._assert_development_protocol(development)

    confirmation = _minimal_confirmation_protocol(tmp_path / "output")
    runner._assert_confirmation_protocol(confirmation)
    confirmation["development_binding"] = {
        **confirmation["development_binding"],
        "model_absolute_path": str(tmp_path / "elsewhere.npz"),
    }
    with pytest.raises(ValueError, match="model path"):
        runner._assert_confirmation_protocol(confirmation)


def test_candidate_selection_uses_detection_then_p90_then_stronger_ridges() -> None:
    common = {
        "minimum_fault_detection_rate": 0.9,
        "worst_fault_p90_delay_samples": 100.0,
        "oof_healthy_alarm_runs": 0,
    }
    weak = {**common, "candidate_id": "weak", "params": {"classifier_ridge": 0.01, "varx_ridge": 0.01, "page_drift": 0.25}}
    strong = {**common, "candidate_id": "strong", "params": {"classifier_ridge": 0.1, "varx_ridge": 0.1, "page_drift": 0.5}}
    lower_detection = {**strong, "candidate_id": "lower", "minimum_fault_detection_rate": 0.89, "worst_fault_p90_delay_samples": 1.0}
    slower = {**strong, "candidate_id": "slower", "worst_fault_p90_delay_samples": 101.0}
    assert min((weak, strong), key=runner._candidate_key) is strong
    assert min((strong, lower_detection), key=runner._candidate_key) is strong
    assert min((strong, slower), key=runner._candidate_key) is strong


def test_development_normalization_replays_grouping_and_pass_gate() -> None:
    normalized = runner._normalize_candidate(_candidate(runner.GRID[0]), runner._fold_assignment())
    decision = runner._development_decision(normalized)
    assert decision["decision"] == "PASS"
    assert all(value is True for value in decision["gates"].values())
    corrupted = _candidate(runner.GRID[0])
    corrupted["records"][0]["fold"] = (corrupted["records"][0]["fold"] + 1) % 5
    with pytest.raises(ValueError, match="group fold"):
        runner._normalize_candidate(corrupted, runner._fold_assignment())


def test_development_scientific_failure_is_a_completed_no_go() -> None:
    # 20 misses/fault leaves 0.80 overall detection and violates the 0.85 gate.
    normalized = runner._normalize_candidate(
        _candidate(runner.GRID[0], misses=20), runner._fold_assignment()
    )
    decision = runner._development_decision(normalized)
    assert decision["decision"] == "FAIL"
    assert decision["gates"]["each_fault_detection_rate_at_least_0_85"] is False


def test_atomic_protocol_bundle_requires_protocol_sidecar_and_commit(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    digest = runner._publish_protocol_bundle(protocol, {"schema": "demo", "schema_version": 1})
    assert len(digest) == 64
    bundle = runner._bundle_paths(protocol)
    assert bundle.protocol.is_file() and bundle.sidecar.is_file() and bundle.commit.is_file()
    loaded, actual = runner._load_protocol(protocol, digest, schema="demo")
    assert actual == digest and loaded["schema"] == "demo"
    with pytest.raises(FileExistsError):
        runner._publish_protocol_bundle(protocol, {"schema": "demo", "schema_version": 1})
    bundle.commit.unlink()
    with pytest.raises(ValueError, match="partially"):
        runner._load_protocol(protocol, digest, schema="demo")


def test_calibration_is_noninterpolated_q99_strict_greater_and_certified() -> None:
    arrays = _score_arrays(100)
    maxima = np.arange(1.0, 101.0, dtype=np.float64)
    arrays["cpu_scores"][:, -1] = maxima
    arrays["cuda_scores"][:, -1] = maxima
    result, evidence = runner._calibration_summary(arrays)
    assert result["decision"] == "PASS"
    assert result["operational_threshold_q99"] == 99.0
    assert result["q100"] == 100.0
    assert result["numerically_distinct_maxima"] == 100
    assert result["certificate"]["epsilon"] == 0.10
    assert result["certificate"]["delta"] == 0.01
    assert result["certificate"]["failure_probability_binomial_tail"] < 0.01
    assert evidence["cpu_alarm"][98, -1] == np.bool_(False)
    assert evidence["cpu_alarm"][99, -1] == np.bool_(True)


def test_calibration_fails_closed_on_ceiling_ties_and_degeneracy() -> None:
    arrays = _score_arrays(100)
    arrays["cpu_scores"][:, -1] = runner.KNOWN_NUMERICAL_LIMITS[1]
    arrays["cuda_scores"][:, -1] = runner.KNOWN_NUMERICAL_LIMITS[1]
    result, _ = runner._calibration_summary(arrays)
    assert result["decision"] == "FAIL"
    assert result["gates"]["at_least_95_numerically_distinct_maxima"] is False
    assert result["gates"]["q99_strictly_below_q100"] is False
    assert result["gates"]["no_maximum_equals_known_numerical_or_algorithmic_limit"] is False
    assert result["gates"]["no_top_finite_tie"] is False


def test_calibration_tail_must_clear_100_times_parity_atol() -> None:
    arrays = _score_arrays(100)
    maxima = 1.0 + np.arange(100, dtype=np.float64) * 2.0e-12
    arrays["cpu_scores"][:, -1] = maxima
    arrays["cuda_scores"][:, -1] = maxima
    result, _ = runner._calibration_summary(arrays)
    assert result["numerically_distinct_maxima"] == 100
    assert result["q100"] > result["operational_threshold_q99"]
    assert result["top_tail_numeric_error_reference"] == runner.PARITY_ATOL
    assert result["decision"] == "FAIL"
    assert result["gates"]["top_tail_spread_above_100_times_numeric_error_reference"] is False


def test_calibration_tail_reference_uses_worst_full_path_stage_error() -> None:
    arrays = _score_arrays(100)
    maxima = np.arange(1.0, 101.0, dtype=np.float64)
    arrays["cpu_scores"][:, -1] = maxima
    arrays["cuda_scores"][:, -1] = maxima
    arrays["cuda_cusum"][0, 200, 0] = 2.0e-3
    result, _ = runner._calibration_summary(arrays)
    assert result["max_abs_cpu_cuda_score_error"] == 0.0
    assert result["max_abs_cpu_cuda_full_path_error"] == pytest.approx(2.0e-3)
    assert result["top_tail_numeric_error_reference"] == pytest.approx(2.0e-3)
    assert (
        result["gates"]["cuda_numpy_all_numeric_stage_error_at_most_1e_10"]
        is False
    )
    assert result["decision"] == "FAIL"


def test_calibration_rejects_even_one_maximum_at_page_zero_floor() -> None:
    arrays = _score_arrays(100)
    maxima = np.arange(100, dtype=np.float64)
    arrays["cpu_scores"][:, -1] = maxima
    arrays["cuda_scores"][:, -1] = maxima
    result, _ = runner._calibration_summary(arrays)
    assert result["gates"]["no_maximum_equals_known_numerical_or_algorithmic_limit"] is False


def test_score_payload_requires_full_float64_cuda_numpy_geometry() -> None:
    arrays = _full_score_payload(
        np.zeros((100, runner.SEQUENCE_LENGTH, len(runner.FAULT_IDS)), dtype=np.float64)
    )
    validated, parity = runner._validate_score_payload(
        arrays, count=100, require_attribution=False, page_drift=0.0
    )
    assert validated["cpu_scores"].dtype == np.float64
    assert parity["eligibility_exact"] is True
    bad = dict(arrays)
    bad["cuda_eligible"] = bad["cuda_eligible"].copy()
    bad["cuda_eligible"][0, 0] = True
    with pytest.raises(ValueError, match="eligibility geometry"):
        runner._validate_score_payload(
            bad, count=100, require_attribution=False, page_drift=0.0
        )


@pytest.mark.parametrize("burst_length", [1, 2])
def test_one_or_two_sample_exceedance_cannot_survive_three_sample_persistence(
    burst_length: int,
) -> None:
    logits = np.zeros((1, runner.SEQUENCE_LENGTH, len(runner.FAULT_IDS)), dtype=np.float64)
    logits[0, 200 : 200 + burst_length, 0] = 20.0
    logits[0, 200 + burst_length, 0] = -100.0
    payload = _full_score_payload(logits)
    arrays, _ = runner._validate_score_payload(
        payload, count=1, require_attribution=False, page_drift=0.0
    )
    assert not np.any(arrays["cpu_scores"] > 10.0)


def test_three_sample_exceedance_defines_a_persistent_envelope_alarm() -> None:
    logits = np.zeros((1, runner.SEQUENCE_LENGTH, len(runner.FAULT_IDS)), dtype=np.float64)
    logits[0, 200:203, 0] = 20.0
    payload = _full_score_payload(logits)
    arrays, _ = runner._validate_score_payload(
        payload, count=1, require_attribution=False, page_drift=0.0
    )
    alarm = arrays["cpu_eligible"] & (arrays["cpu_scores"] > 10.0)
    assert runner._episode_first_start(alarm[0]) == 202


def test_confirmation_gate_uses_new_episode_start_and_restricted_delays() -> None:
    arrays = _score_arrays(400)
    attribution = np.full((400, runner.SEQUENCE_LENGTH), 3, dtype=np.int64)
    for fault_index, fault in enumerate(runner.FAULT_IDS):
        offset = 100 + fault_index * 100
        arrays["cpu_scores"][offset : offset + 100, 170:173] = 11.0
        arrays["cuda_scores"][offset : offset + 100, 170:173] = 11.0
        attribution[offset : offset + 100] = fault
    arrays["cpu_attribution"] = attribution
    arrays["cuda_attribution"] = attribution.copy()
    result, _ = runner._confirmation_summary(arrays, threshold=10.0)
    assert result["decision"] == "PASS"
    assert result["healthy_alarm_runs"] == 0
    assert result["total_detected_fault_runs"] == 300
    for fault in runner.FAULT_IDS:
        row = result["faults"][str(fault)]
        assert row["detected_runs"] == 100
        assert row["median_restricted_delay_samples"] == 10.0
        assert row["p90_restricted_delay_samples_nearest_rank"] == 10.0


def test_confirmation_fail_is_terminal_scientific_result() -> None:
    arrays = _score_arrays(400)
    attribution = np.full((400, runner.SEQUENCE_LENGTH), 3, dtype=np.int64)
    arrays["cpu_attribution"] = attribution
    arrays["cuda_attribution"] = attribution.copy()
    result, _ = runner._confirmation_summary(arrays, threshold=10.0)
    assert result["decision"] == "FAIL"
    assert result["final_rows_irreversibly_consumed"] is True
    assert result["rerun_or_retuning_after_pass_or_fail_forbidden"] is True
    assert all(row["median_restricted_delay_samples"] == runner.MISS_DELAY for row in result["faults"].values())


def test_partial_staging_blocks_phase(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / ".confirm.inprogress-interrupted").mkdir()
    with pytest.raises(RuntimeError, match="fail-closed"):
        runner._assert_no_partial(output, "confirm")


def test_atomic_phase_claim_blocks_a_concurrent_second_execution(tmp_path: Path) -> None:
    output = tmp_path / "output"
    target, staging = runner._claim_phase(output, "confirm")
    assert not target.exists()
    assert staging.is_dir()
    assert (output / ".confirm.claim").is_dir()
    with pytest.raises(RuntimeError, match="already been claimed"):
        runner._claim_phase(output, "confirm")


@pytest.mark.parametrize("residual", [".confirm.claim", ".confirm.inprogress-orphan"])
def test_completed_target_with_claim_or_staging_residual_fails_closed(
    tmp_path: Path, residual: str
) -> None:
    output = tmp_path / "output"
    (output / "confirm").mkdir(parents=True)
    (output / residual).mkdir()
    with pytest.raises(RuntimeError):
        runner._assert_completed_clean(output, "confirm")


def test_runtime_replay_rejects_self_signed_non_cuda_manifest() -> None:
    with pytest.raises(ValueError, match="runtime"):
        runner._validate_runtime({})
    bad = {
        "requested_device": "cpu",
        "numeric_dtype": "float64",
        "torch_version": "x",
        "torch_cuda_version": "x",
        "device_name": "x",
        "float64_probe": 5.0,
    }
    with pytest.raises(ValueError, match="CUDA"):
        runner._validate_runtime(bad)


def test_development_writes_intent_and_manifest_before_backend_numeric_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    protocol_path = tmp_path / "development.json"
    protocol = _minimal_protocol(output)
    digest = runner._publish_protocol_bundle(protocol_path, protocol)
    observed: list[str] = []

    def load_numeric_runs(source: Path, plan: list[dict[str, object]]) -> object:
        staging = next(output.glob(".develop.inprogress-*"))
        assert (staging / "access_intent.json").is_file()
        assert (staging / "phase_manifest.json").is_file()
        observed.append("numeric-read")
        return {run: {"simulation_run": run, "idvs": runner.ALL_IDS} for run in runner.DEVELOPMENT_RUNS}

    fit_calls: list[tuple[str, int, float | None]] = []

    def fit_feature(**kwargs: object) -> dict[str, object]:
        assert len(kwargs["train_groups"]) == 80
        fit_calls.append(("feature", int(kwargs["fold"]), float(kwargs["varx_ridge"])))
        return {"fold": kwargs["fold"], "ridge": kwargs["varx_ridge"]}

    def transform(**kwargs: object) -> dict[str, object]:
        expected = 80 if kwargs["role"] == "training" else 20
        assert len(kwargs["groups"]) == expected
        return {"groups": kwargs["groups"], "role": kwargs["role"], "fold": kwargs["fold"]}

    def fit_heads(**kwargs: object) -> dict[str, object]:
        assert kwargs["train_features"]["role"] == "training"
        fit_calls.append(("heads", int(kwargs["fold"]), float(kwargs["classifier_ridge"])))
        return {"fold": kwargs["fold"]}

    def score_fold(**kwargs: object) -> dict[str, object]:
        fold = int(kwargs["fold"])
        fit_calls.append(("score", fold, float(kwargs["page_drift"])))
        full = _candidate(runner.GRID[0])
        records = [row for row in full["records"] if row["fold"] == fold]
        healthy = [row for row in full["healthy_records"] if row["fold"] == fold]
        calibration = full["fold_calibrations"][fold]
        return {
            "records": records,
            "healthy_records": healthy,
            "training_healthy_cpu_run_maxima": np.asarray(calibration["training_healthy_cpu_run_maxima"], dtype=np.float64),
            "training_healthy_cuda_run_maxima": np.asarray(calibration["training_healthy_cuda_run_maxima"], dtype=np.float64),
            "operational_threshold_q79": calibration["operational_threshold_q79"],
            "parity": full["parity"],
        }

    backend = SimpleNamespace(
        load_numeric_runs=load_numeric_runs,
        fit_fold_feature_model=fit_feature,
        transform_fold_groups=transform,
        fit_fold_heads=fit_heads,
        score_fold_candidate=score_fold,
        refit_selected_model=lambda **kwargs: _synthetic_model_arrays(
            kwargs["selected_params"]
        ),
    )
    monkeypatch.setattr(runner, "_verify_live_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "_cuda_preflight", lambda: {"requested_device": "cuda", "numeric_dtype": "float64", "torch_version": "synthetic", "torch_cuda_version": "synthetic", "device_name": "synthetic", "float64_probe": 5.0})
    monkeypatch.setattr(runner, "_load_backend", lambda: backend)

    first = runner.run_development(
        tmp_path, tmp_path, output, protocol_path, digest, batch_size=4
    )
    assert first["decision"] == "PASS"
    assert observed == ["numeric-read"]
    assert sum(row[0] == "feature" for row in fit_calls) == 15
    assert sum(row[0] == "heads" for row in fit_calls) == 45
    assert sum(row[0] == "score" for row in fit_calls) == 135

    # Completed reentry verifies and returns before live-source verification or backend import.
    monkeypatch.setattr(runner, "_verify_live_inputs", lambda *args: (_ for _ in ()).throw(AssertionError("live H5 touched")))
    monkeypatch.setattr(runner, "_load_backend", lambda: (_ for _ in ()).throw(AssertionError("backend imported")))
    second = runner.run_development(
        tmp_path, tmp_path, output, protocol_path, digest, batch_size=4
    )
    assert second == first
    assert observed == ["numeric-read"]


def test_zero_eligible_development_is_completed_no_go_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    protocol_path = tmp_path / "development.json"
    protocol = _minimal_protocol(output)
    digest = runner._publish_protocol_bundle(protocol_path, protocol)
    candidates = []
    for frozen in runner.GRID:
        candidate = _candidate(frozen)
        for record in candidate["healthy_records"]:
            record["alarmed"] = True
        candidates.append(candidate)
    observed: list[str] = []

    def load_numeric_runs(source: Path, plan: list[dict[str, object]]) -> object:
        staging = next(output.glob(".develop.inprogress-*"))
        assert (staging / "access_intent.json").is_file()
        assert (staging / "phase_manifest.json").is_file()
        observed.append("numeric-read")
        return {"synthetic": True}

    backend = SimpleNamespace(
        load_numeric_runs=load_numeric_runs,
        refit_selected_model=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("zero-eligible development must not refit a model")
        ),
    )
    monkeypatch.setattr(runner, "_verify_live_inputs", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_cuda_preflight",
        lambda: {
            "requested_device": "cuda",
            "numeric_dtype": "float64",
            "torch_version": "synthetic",
            "torch_cuda_version": "synthetic",
            "device_name": "synthetic",
            "float64_probe": 5.0,
        },
    )
    monkeypatch.setattr(runner, "_load_backend", lambda: backend)
    monkeypatch.setattr(
        runner,
        "_evaluate_grid_fold_safe",
        lambda *args, **kwargs: candidates,
    )

    first = runner.run_development(
        tmp_path, tmp_path, output, protocol_path, digest, batch_size=4
    )
    assert first["decision"] == "DEVELOPMENT_NO_GO"
    assert first["reason"] == "zero_selection_eligible_candidates"
    assert first["selected_model_generated"] is False
    assert first["confirmation_freeze_permitted"] is False
    assert not (output / "develop" / "pclr_model.npz").exists()
    assert not (output / ".develop.claim").exists()
    assert not list(output.glob(".develop.inprogress-*"))

    monkeypatch.setattr(
        runner,
        "_verify_live_inputs",
        lambda *args: (_ for _ in ()).throw(AssertionError("live H5 touched")),
    )
    monkeypatch.setattr(
        runner,
        "_load_backend",
        lambda: (_ for _ in ()).throw(AssertionError("backend imported")),
    )
    second = runner.run_development(
        tmp_path, tmp_path, output, protocol_path, digest, batch_size=4
    )
    assert second == first
    assert observed == ["numeric-read"]
    with pytest.raises(RuntimeError, match="go/no-go did not pass"):
        runner.build_confirmation_protocol(
            tmp_path,
            tmp_path,
            output,
            protocol_path,
            digest,
        )


def test_model_archive_semantics_bind_selected_hyperparameters(tmp_path: Path) -> None:
    params = {
        key: runner.GRID[0][key]
        for key in ("varx_ridge", "classifier_ridge", "page_drift")
    }
    valid = _synthetic_model_arrays(params)
    path = tmp_path / "model.npz"
    runner._write_npz(path, valid)
    metadata = runner._validate_pclr_model_file(path, params)
    assert metadata["selected_params"] == params

    tampered = dict(valid)
    raw_metadata = json.loads(str(tampered["metadata_json"].item()))
    raw_metadata["selected_params"]["classifier_ridge"] = 0.1
    tampered["metadata_json"] = np.array(
        json.dumps(
            raw_metadata,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        dtype=np.str_,
    )
    bad_path = tmp_path / "tampered.npz"
    runner._write_npz(bad_path, tampered)
    with pytest.raises(ValueError, match="refit metadata"):
        runner._validate_pclr_model_file(bad_path, params)

    config_tampered = dict(valid)
    raw_config = json.loads(str(config_tampered["config_json"].item()))
    raw_config["mad_floor_quantile"] = 0.20
    config_tampered["config_json"] = np.array(
        json.dumps(
            raw_config,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        dtype=np.str_,
    )
    config_path = tmp_path / "config-tampered.npz"
    runner._write_npz(config_path, config_tampered)
    with pytest.raises(ValueError, match="feature configuration"):
        runner._validate_pclr_model_file(config_path, params)


def test_confirmation_protocol_records_exact_order_statistic_and_one_time_gate() -> None:
    assert runner.CALIBRATION_RANK == 99
    assert runner.CALIBRATION_EPSILON == 0.10
    assert runner.CALIBRATION_DELTA == 0.01
    assert runner.CONFIRMATION_GATES == {
        "maximum_healthy_alarm_runs": 4,
        "minimum_detected_runs_each_fault": 90,
        "minimum_detected_runs_total": 270,
        "maximum_median_delay_each_fault": 120.0,
        "maximum_p90_delay_each_fault": 180.0,
        "miss_delay_samples": 340,
    }


@pytest.mark.parametrize("phase", ["calibrate", "confirm"])
@pytest.mark.parametrize("decision", ["PASS", "FAIL"])
def test_completed_calibration_and_confirmation_return_before_live_h5_or_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    decision: str,
) -> None:
    output = tmp_path / "output"
    (output / phase).mkdir(parents=True)
    protocol_path = tmp_path / "confirmation.json"
    protocol = {
        "schema": "pclr_confirmation_protocol",
        "schema_version": runner.SCHEMA_VERSION,
        "governance": runner._governance_contract(output),
    }
    digest = runner._publish_protocol_bundle(protocol_path, protocol)
    monkeypatch.setattr(runner, "_assert_confirmation_protocol", lambda value: None)
    monkeypatch.setattr(
        runner,
        "_verify_confirmation_live_inputs",
        lambda *args: (_ for _ in ()).throw(AssertionError("live H5 touched")),
    )
    monkeypatch.setattr(
        runner,
        "_load_backend",
        lambda: (_ for _ in ()).throw(AssertionError("backend imported")),
    )
    expected = {"decision": decision, "phase": phase}
    if phase == "calibrate":
        monkeypatch.setattr(runner, "_verify_calibration_bundle", lambda *args: expected)
        actual = runner.run_calibration(
            tmp_path, tmp_path, output, protocol_path, digest, batch_size=1
        )
    else:
        monkeypatch.setattr(runner, "_verify_confirmation_bundle", lambda *args: expected)
        monkeypatch.setattr(
            runner,
            "_calibration_binding",
            lambda *args: (_ for _ in ()).throw(AssertionError("calibration reopened")),
        )
        actual = runner.run_confirmation(
            tmp_path, tmp_path, output, protocol_path, digest, batch_size=1
        )
    assert actual == expected


def test_numeric_cli_subprocess_preloads_torch_before_numpy() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "develop", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_cli_exposes_only_governed_phase_commands() -> None:
    parser = runner._build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers_action.choices) == {
        "freeze-development",
        "develop",
        "freeze-confirmation",
        "calibrate",
        "confirm",
    }
