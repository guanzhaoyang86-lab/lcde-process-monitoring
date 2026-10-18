"""Develop then freeze/test LC-DE on the experiment-level SKAB split."""

from __future__ import annotations

import torch  # noqa: F401  # Windows torch-first DLL contract

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from mate_spc_cris_v9.local_contrast import (
    LocalContrastConfig,
    LocalContrastDualEnergyMonitor,
)
from mate_spc_cris_v9.model import CRISModel
from mate_spc_cris_v9.skab import (
    assigned_paths,
    load_features,
    load_labels,
    load_preaccess_charter,
    partition_bounds,
    sha256_file,
    validate_score_label_binding,
)
from mate_spc_cris_v9.skab_external import evaluate_skab_path, warmup_standardize


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "data" / "skab-v0.9.0" / "SKAB-0.9.0" / "data"
CHARTER_PATH = PROJECT / "protocols" / "cris_v9_skab_preaccess_charter.json"
MODEL_PATH = PROJECT / "outputs" / "ea_hx_skab" / "ea_hx_skab_model.npz"
OUTPUT_ROOT = PROJECT / "outputs" / "lcde_skab_v2_self_calibrated"
DEVELOPMENT_RESULT = OUTPUT_ROOT / "development_result.json"
FROZEN_JSON = OUTPUT_ROOT / "frozen_method.json"
FROZEN_NPZ = OUTPUT_ROOT / "frozen_method.npz"
COMPARATIVE_PROTOCOL = OUTPUT_ROOT / "comparative_freeze_protocol.json"
FINAL_RESULT = OUTPUT_ROOT / "final_result.json"
FINAL_SCORE_MANIFEST = OUTPUT_ROOT / "final_score_manifest.json"
DEVELOPMENT_SCORE_ROOT = OUTPUT_ROOT / "development_scores"
FINAL_SCORE_ROOT = OUTPUT_ROOT / "final_scores"
BASELINE_DEVELOPMENT_RESULT = PROJECT / "outputs" / "ea_hx_skab" / "development_result.json"
METHOD_PATH = PROJECT / "src" / "mate_spc_cris_v9" / "local_contrast.py"
RUNNER_PATH = Path(__file__).resolve()

PATH_CORRECTIONS = {
    "valve1/16.csv": "valve1/0.csv",
    "valve2/4.csv": "valve2/0.csv",
}
MODES = ("dual", "raw_only", "predictive_only")


def _implementation_hashes() -> dict[str, str]:
    return {
        "runner_sha256": sha256_file(RUNNER_PATH),
        "method_sha256": sha256_file(METHOD_PATH),
        "charter_sha256": sha256_file(CHARTER_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _resolved_split_manifest(charter: Mapping[str, Any]) -> dict[str, Any]:
    roles: dict[str, list[dict[str, Any]]] = {}
    absolute_by_role: dict[str, set[str]] = {}
    for role in ("development", "final"):
        records: list[dict[str, Any]] = []
        absolute_paths: set[str] = set()
        for virtual_path in charter["split_rule"][f"{role}_files"]:
            resolved_relative = PATH_CORRECTIONS.get(virtual_path, virtual_path)
            path = (DATA_ROOT / resolved_relative).resolve()
            try:
                path.relative_to(DATA_ROOT.resolve())
            except ValueError as error:
                raise ValueError("resolved SKAB path escapes dataset root") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            absolute = str(path).casefold()
            if absolute in absolute_paths:
                raise ValueError(f"duplicate resolved {role} path: {path}")
            absolute_paths.add(absolute)
            records.append(
                {
                    "charter_run_id": virtual_path,
                    "resolved_run_id": resolved_relative,
                    "source_sha256": sha256_file(path),
                }
            )
        roles[role] = records
        absolute_by_role[role] = absolute_paths
    if absolute_by_role["development"] & absolute_by_role["final"]:
        raise ValueError("resolved development and final SKAB paths overlap")
    if len(absolute_by_role["development"] | absolute_by_role["final"]) != 34:
        raise ValueError("resolved SKAB split must contain 34 unique experiments")
    return {
        "path_corrections": dict(PATH_CORRECTIONS),
        "roles": roles,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(path)


def _paths(charter: Mapping[str, Any], role: str) -> tuple[Path, ...]:
    # Keep assigned_paths as the charter/path traversal validator, then resolve
    # the two upstream SKAB tag renames explicitly and auditably.
    assigned = assigned_paths(DATA_ROOT, charter, role=role)
    result: list[Path] = []
    for path in assigned:
        relative = path.relative_to(DATA_ROOT.resolve()).as_posix()
        result.append((DATA_ROOT / PATH_CORRECTIONS.get(relative, relative)).resolve())
    manifest = _resolved_split_manifest(charter)
    expected = [item["resolved_run_id"] for item in manifest["roles"][role]]
    observed = [path.relative_to(DATA_ROOT.resolve()).as_posix() for path in result]
    if observed != expected:
        raise RuntimeError("resolved SKAB paths do not match authenticated manifest")
    return tuple(result)


def _dummy(length: int) -> np.ndarray:
    return np.zeros((length, 1), dtype=np.float32)


def _innovations(model: CRISModel, x: np.ndarray) -> np.ndarray:
    return model.innovations(
        x,
        _dummy(x.shape[0]),
        np.ones(x.shape, dtype=bool),
        device="cuda",
        batch_size=2048,
    )


def _healthy_segments(model: CRISModel) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    healthy = load_features(DATA_ROOT / "anomaly-free" / "anomaly-free.csv", dataset_root=DATA_ROOT)
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (first, last) in partition_bounds(healthy.x.shape[0]).items():
        x = warmup_standardize(healthy.x[first:last])
        output[name] = (x, _innovations(model, x))
    return output


def _load_feature_runs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        features = load_features(path, dataset_root=DATA_ROOT)
        output.append(
            {
                "run_id": features.run_id,
                "source_sha256": features.source_sha256,
                "timestamps": features.timestamps,
                "x": warmup_standardize(features.x),
            }
        )
    return output


def _load_runs(
    paths: Sequence[Path], model: CRISModel, *, label: str
) -> list[dict[str, Any]]:
    output = _load_feature_runs(paths)
    for index, run in enumerate(output, start=1):
        run["innovations"] = _innovations(model, run["x"])
        print(f"[{label}-innovation] {index}/{len(paths)} {run['run_id']}", flush=True)
    return output


def _candidate_configs() -> tuple[LocalContrastConfig, ...]:
    return tuple(
        LocalContrastConfig(
            current_window=window,
            channel_top_k=top_k,
            allowance=allowance,
            scan_horizon=horizon,
        )
        for window, top_k, allowance, horizon in itertools.product(
            (10, 20, 40), (1, 2), (0.5, 1.0), (30, 60)
        )
    )


def _candidate_id(config: LocalContrastConfig) -> str:
    allowance = str(config.allowance).replace(".", "p")
    return (
        f"w{config.current_window}_k{config.channel_top_k}_"
        f"a{allowance}_s{config.scan_horizon}"
    )


def _episodes(path: np.ndarray, threshold: float) -> int:
    alarm = np.asarray(path > threshold, dtype=bool)
    return int(np.count_nonzero(alarm & ~np.r_[False, alarm[:-1]]))


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delays = [float(item["delay_seconds"]) for item in records if item["detected"]]
    ordered = np.sort(np.asarray(delays, dtype=np.float64)) if delays else np.asarray([])
    p90 = None
    if ordered.size:
        p90 = float(ordered[max(0, int(math.ceil(0.90 * ordered.size)) - 1)])
    return {
        "runs": len(records),
        "detected_runs": sum(bool(item["detected"]) for item in records),
        "event_recall": float(np.mean([item["detected"] for item in records])),
        "within_60_runs": sum(bool(item["within_60"]) for item in records),
        "pre_event_false_alarm_runs": sum(
            bool(item["pre_event_false_alarm"]) for item in records
        ),
        "conditional_median_delay_seconds": (
            None if not delays else float(np.median(np.asarray(delays)))
        ),
        "conditional_nearest_rank_p90_delay_seconds": p90,
        "mean_unadjusted_f1": float(np.mean([item["unadjusted_f1"] for item in records])),
        "mean_average_precision": float(np.mean([item["average_precision"] for item in records])),
    }


def _development_go(aggregate: Mapping[str, Any], healthy_episodes: int) -> tuple[bool, dict[str, bool]]:
    checks = {
        "detected_within_60_at_least_14_of_18": aggregate["within_60_runs"] >= 14,
        "pre_event_false_alarm_runs_at_most_3": aggregate["pre_event_false_alarm_runs"] <= 3,
        "conditional_median_delay_at_most_30_seconds": (
            aggregate["conditional_median_delay_seconds"] is not None
            and aggregate["conditional_median_delay_seconds"] <= 30.0
        ),
        "healthy_audit_alarm_episodes_at_most_1": healthy_episodes <= 1,
    }
    return all(checks.values()), checks


def _selection_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["aggregate"]
    strict_go, _ = _development_go(metrics, int(item["healthy_audit_episodes"]))
    constraint_violations = (
        max(0, int(metrics["pre_event_false_alarm_runs"]) - 3)
        + max(0, int(item["healthy_audit_episodes"]) - 1)
    )
    delay = metrics["conditional_median_delay_seconds"]
    return (
        not strict_go,
        constraint_violations,
        -int(metrics["within_60_runs"]),
        -int(metrics["detected_runs"]),
        math.inf if delay is None else float(delay),
        -float(metrics["mean_average_precision"]),
        int(item["config"]["current_window"]),
        int(item["config"]["scan_horizon"]),
    )


def _persist_candidate_paths(
    runs: Sequence[Mapping[str, Any]],
    paths_by_candidate: Sequence[Sequence[np.ndarray]],
    candidate_ids: Sequence[str],
) -> None:
    DEVELOPMENT_SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    for run_index, run in enumerate(runs):
        target = DEVELOPMENT_SCORE_ROOT / (run["run_id"].replace("/", "__") + ".npz")
        temporary = target.with_suffix(".tmp.npz")
        matrix = np.stack(
            [paths_by_candidate[index][run_index] for index in range(len(candidate_ids))],
            axis=1,
        )
        np.savez_compressed(
            temporary,
            run_id=np.asarray(run["run_id"]),
            source_sha256=np.asarray(run["source_sha256"]),
            candidate_ids=np.asarray(candidate_ids),
            score_paths=matrix,
        )
        temporary.replace(target)


def _development_cache_valid(
    document: Mapping[str, Any],
    bindings: Mapping[str, str],
    split_manifest: Mapping[str, Any],
) -> bool:
    return (
        document.get("schema") == "lcde_skab_development"
        and document.get("artifact_bindings") == dict(bindings)
        and document.get("resolved_split_manifest_sha256")
        == _canonical_sha256(split_manifest)
        and document.get("candidate_count") == len(_candidate_configs())
        and isinstance(document.get("selected"), dict)
        and document.get("selected_candidate_id")
        == document["selected"].get("candidate_id")
    )


def _save_frozen(
    state: LocalContrastDualEnergyMonitor,
    selected: Mapping[str, Any],
    threshold_x: np.ndarray,
    threshold_innovations: np.ndarray,
    development: Mapping[str, Any],
    *,
    freeze_basis: str,
    comparison: Mapping[str, Any] | None = None,
) -> None:
    mode_thresholds = {
        mode: float(np.max(state.score(threshold_x, threshold_innovations, mode=mode)))
        for mode in MODES
    }
    if not math.isclose(
        mode_thresholds["dual"],
        float(selected["threshold"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("reconstructed LC-DE threshold differs from development")
    bindings = _implementation_hashes()
    split_manifest = development["resolved_split_manifest"]
    if development.get("artifact_bindings") != bindings:
        raise ValueError("development artifact is stale relative to current implementation")
    if development.get("resolved_split_manifest_sha256") != _canonical_sha256(split_manifest):
        raise ValueError("development split manifest hash mismatch")

    temporary = FROZEN_NPZ.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        component_center=state.component_center,
        component_scale=state.component_scale,
        config_json=np.asarray(json.dumps(state.config.as_dict(), sort_keys=True)),
        mode_names=np.asarray(MODES),
        mode_thresholds=np.asarray([mode_thresholds[mode] for mode in MODES]),
        model_sha256=np.asarray(bindings["model_sha256"]),
        method_sha256=np.asarray(bindings["method_sha256"]),
        charter_sha256=np.asarray(bindings["charter_sha256"]),
        development_result_sha256=np.asarray(sha256_file(DEVELOPMENT_RESULT)),
    )
    temporary.replace(FROZEN_NPZ)
    _write_json(
        FROZEN_JSON,
        {
            "schema": "lcde_skab_frozen_method",
            "candidate_id": selected["candidate_id"],
            "config": state.config.as_dict(),
            "component_names": [
                "raw_level_energy",
                "raw_mean_shift",
                "raw_log_variance_ratio",
                "raw_log_difference_energy_ratio",
                "predictive_log_energy_ratio",
            ],
            "fusion": "mean of largest two positive robust component z-scores",
            "scan": "maximum ending-subarray CUSUM restricted to scan_horizon",
            "thresholds": mode_thresholds,
            "freeze_basis": freeze_basis,
            "strict_development_gate_passed": bool(development["go"]),
            "comparison": comparison,
            "artifact_bindings": bindings,
            "resolved_split_manifest": split_manifest,
            "resolved_split_manifest_sha256": _canonical_sha256(split_manifest),
            "state_npz_sha256": sha256_file(FROZEN_NPZ),
            "development_result_sha256": sha256_file(DEVELOPMENT_RESULT),
        },
    )


def develop() -> bool:
    charter = load_preaccess_charter(CHARTER_PATH)
    bindings = _implementation_hashes()
    split_manifest = _resolved_split_manifest(charter)
    if DEVELOPMENT_RESULT.exists():
        document = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
        if _development_cache_valid(document, bindings, split_manifest):
            print(f"[development-cache] {document['decision']}", flush=True)
            return bool(document["go"])
        print("[development-cache] stale artifact ignored", flush=True)
    started = time.perf_counter()
    model = CRISModel.load_npz(MODEL_PATH)
    print(f"[gpu] {model.attestation.cuda_device_name}", flush=True)
    healthy = _healthy_segments(model)
    runs = _load_runs(_paths(charter, "development"), model, label="development")

    reference_x, reference_innovations = healthy["point_score_reference"]
    threshold_x, threshold_innovations = healthy["trajectory_threshold_blocks"]
    audit_x, audit_innovations = healthy["healthy_development_audit"]
    configs = _candidate_configs()
    candidate_ids = tuple(_candidate_id(config) for config in configs)
    candidate_paths: list[list[np.ndarray]] = []
    states: list[LocalContrastDualEnergyMonitor] = []
    thresholds: list[float] = []
    healthy_paths: list[np.ndarray] = []
    for index, config in enumerate(configs, start=1):
        state = LocalContrastDualEnergyMonitor.fit(
            reference_x, reference_innovations, config
        )
        threshold_path = state.score(threshold_x, threshold_innovations)
        threshold = float(np.max(threshold_path))
        audit_path = state.score(audit_x, audit_innovations)
        paths = [state.score(run["x"], run["innovations"]) for run in runs]
        states.append(state)
        thresholds.append(threshold)
        healthy_paths.append(audit_path)
        candidate_paths.append(paths)
        print(f"[candidate-score] {index}/{len(configs)} {candidate_ids[index - 1]}", flush=True)

    # Every score exists on disk before development labels are opened.
    _persist_candidate_paths(runs, candidate_paths, candidate_ids)
    records_by_candidate: list[dict[str, Any]] = []
    labels = [load_labels(DATA_ROOT / run["run_id"], dataset_root=DATA_ROOT) for run in runs]
    for candidate_index, (config, threshold) in enumerate(zip(configs, thresholds, strict=True)):
        per_run: list[dict[str, Any]] = []
        for run, label, path in zip(runs, labels, candidate_paths[candidate_index], strict=True):
            validate_score_label_binding(
                {
                    "run_id": run["run_id"],
                    "source_sha256": run["source_sha256"],
                    "row_count": int(path.size),
                },
                label,
            )
            metrics = evaluate_skab_path(path, threshold, label.anomaly, run["timestamps"])
            metrics["run_id"] = run["run_id"]
            per_run.append(metrics)
        aggregate = _aggregate(per_run)
        audit_episodes = _episodes(healthy_paths[candidate_index], threshold)
        records_by_candidate.append(
            {
                "candidate_id": candidate_ids[candidate_index],
                "config": config.as_dict(),
                "threshold": threshold,
                "healthy_audit_episodes": audit_episodes,
                "healthy_audit_occupancy": float(np.mean(healthy_paths[candidate_index] > threshold)),
                "aggregate": aggregate,
                "per_run": per_run,
            }
        )

    selected = min(records_by_candidate, key=_selection_key)
    selected_index = candidate_ids.index(selected["candidate_id"])
    go, checks = _development_go(
        selected["aggregate"], int(selected["healthy_audit_episodes"])
    )
    document = {
        "schema": "lcde_skab_development",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_seconds": time.perf_counter() - started,
        "device": model.attestation.cuda_device_name,
        "model_sha256": bindings["model_sha256"],
        "artifact_bindings": bindings,
        "resolved_split_manifest": split_manifest,
        "resolved_split_manifest_sha256": _canonical_sha256(split_manifest),
        "candidate_count": len(configs),
        "selection_rule": "feasibility_then_within60_then_recall_then_delay_then_AP_then_simplicity",
        "selected_candidate_id": selected["candidate_id"],
        "selected": selected,
        "checks": checks,
        "go": go,
        "decision": "DEVELOPMENT_GO_FREEZE" if go else "DEVELOPMENT_NO_GO",
        "candidates": records_by_candidate,
    }
    _write_json(DEVELOPMENT_RESULT, document)
    if go:
        _save_frozen(
            states[selected_index],
            selected,
            threshold_x,
            threshold_innovations,
            document,
            freeze_basis="ORIGINAL_CHARTER_DEVELOPMENT_GO",
        )
    print(f"[development] {document['decision']} {selected['aggregate']}", flush=True)
    return go


def freeze_comparative() -> None:
    """Freeze LC-DE under an explicit post-hoc Pareto rule, never as charter GO."""

    develop()
    development = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    charter = load_preaccess_charter(CHARTER_PATH)
    bindings = _implementation_hashes()
    split_manifest = _resolved_split_manifest(charter)
    if not _development_cache_valid(development, bindings, split_manifest):
        raise ValueError("cannot comparatively freeze a stale development artifact")
    baseline_document = json.loads(
        BASELINE_DEVELOPMENT_RESULT.read_text(encoding="utf-8")
    )
    if baseline_document.get("model_sha256") != bindings["model_sha256"]:
        raise ValueError("baseline development result uses a different predictive model")
    baselines = baseline_document["development"]["aggregate"]
    selected = development["selected"]
    metrics = selected["aggregate"]
    best_detected = max(int(item["detected_runs"]) for item in baselines.values())
    best_f1 = max(float(item["mean_unadjusted_f1"]) for item in baselines.values())
    best_ap = max(float(item["mean_average_precision"]) for item in baselines.values())
    equal_recall = [
        item
        for item in baselines.values()
        if int(item["detected_runs"]) >= int(metrics["detected_runs"])
    ]
    best_equal_recall_delay = min(
        float(item["conditional_median_delay_seconds"])
        for item in equal_recall
        if item["conditional_median_delay_seconds"] is not None
    )
    best_equal_recall_within60 = max(
        int(item["within_60_runs"]) for item in equal_recall
    )
    checks = {
        "event_detection_no_worse_than_best_baseline": (
            int(metrics["detected_runs"]) >= best_detected
        ),
        "mean_f1_at_least_10_percent_above_best_baseline": (
            float(metrics["mean_unadjusted_f1"]) >= 1.10 * best_f1
        ),
        "mean_ap_within_0p01_of_best_baseline": (
            float(metrics["mean_average_precision"]) >= best_ap - 0.01
        ),
        "delay_better_than_equal_recall_baseline": (
            metrics["conditional_median_delay_seconds"] is not None
            and float(metrics["conditional_median_delay_seconds"])
            < best_equal_recall_delay
        ),
        "within60_no_worse_than_equal_recall_baseline": (
            int(metrics["within_60_runs"]) >= best_equal_recall_within60
        ),
        "pre_event_false_alarm_runs_at_most_3": (
            int(metrics["pre_event_false_alarm_runs"]) <= 3
        ),
        "healthy_audit_alarm_episodes_at_most_1": (
            int(selected["healthy_audit_episodes"]) <= 1
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"LC-DE failed comparative freeze rule: {checks}")
    comparison = {
        "rule_status": "POST_HOC_USER_AUTHORIZED_NOT_PREREGISTERED",
        "checks": checks,
        "lcde": metrics,
        "baselines": baselines,
        "baseline_result_sha256": sha256_file(BASELINE_DEVELOPMENT_RESULT),
    }
    protocol = {
        "schema": "lcde_skab_posthoc_comparative_freeze",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": "FROZEN_BEFORE_HELDOUT_LABEL_ACCESS",
        "disclosure": (
            "The original SKAB development gate failed. The held-out evaluation is "
            "a post-hoc comparative audit authorized by the user, not the original "
            "preregistered confirmation. No held-out labels informed this freeze."
        ),
        "candidate_id": selected["candidate_id"],
        "config": selected["config"],
        "comparison": comparison,
        "artifact_bindings": bindings,
        "development_result_sha256": sha256_file(DEVELOPMENT_RESULT),
        "resolved_split_manifest": split_manifest,
        "resolved_split_manifest_sha256": _canonical_sha256(split_manifest),
    }
    _write_json(COMPARATIVE_PROTOCOL, protocol)
    comparison = {
        **comparison,
        "comparative_protocol_sha256": sha256_file(COMPARATIVE_PROTOCOL),
    }

    model = CRISModel.load_npz(MODEL_PATH)
    healthy = _healthy_segments(model)
    reference_x, reference_innovations = healthy["point_score_reference"]
    threshold_x, threshold_innovations = healthy["trajectory_threshold_blocks"]
    state = LocalContrastDualEnergyMonitor.fit(
        reference_x,
        reference_innovations,
        LocalContrastConfig(**selected["config"]),
    )
    _save_frozen(
        state,
        selected,
        threshold_x,
        threshold_innovations,
        development,
        freeze_basis=(
            "POST_HOC_COMPARATIVE_PARETO_FREEZE_AUTHORIZED_BY_USER; "
            "NOT_ORIGINAL_CHARTER_GO"
        ),
        comparison=comparison,
    )
    print(f"[freeze-comparative] {selected['candidate_id']} {checks}", flush=True)


def _load_frozen() -> tuple[LocalContrastDualEnergyMonitor, dict[str, float], dict[str, Any]]:
    document = json.loads(FROZEN_JSON.read_text(encoding="utf-8"))
    if document.get("schema") != "lcde_skab_frozen_method":
        raise ValueError("invalid frozen LC-DE schema")
    if sha256_file(FROZEN_NPZ) != document["state_npz_sha256"]:
        raise ValueError("frozen LC-DE state hash mismatch")
    bindings = _implementation_hashes()
    if document.get("artifact_bindings") != bindings:
        raise ValueError("frozen LC-DE implementation binding mismatch")
    if sha256_file(DEVELOPMENT_RESULT) != document.get("development_result_sha256"):
        raise ValueError("frozen LC-DE development result hash mismatch")
    charter = load_preaccess_charter(CHARTER_PATH)
    split_manifest = _resolved_split_manifest(charter)
    if (
        document.get("resolved_split_manifest_sha256")
        != _canonical_sha256(split_manifest)
        or document.get("resolved_split_manifest") != split_manifest
    ):
        raise ValueError("frozen LC-DE split manifest mismatch")
    comparison = document.get("comparison")
    if comparison is not None:
        if (
            sha256_file(COMPARATIVE_PROTOCOL)
            != comparison.get("comparative_protocol_sha256")
            or sha256_file(BASELINE_DEVELOPMENT_RESULT)
            != comparison.get("baseline_result_sha256")
        ):
            raise ValueError("frozen LC-DE comparative evidence hash mismatch")
    with np.load(FROZEN_NPZ, allow_pickle=False) as archive:
        config = LocalContrastConfig(**json.loads(str(np.asarray(archive["config_json"]).item())))
        state = LocalContrastDualEnergyMonitor(
            config=config,
            component_center=np.array(archive["component_center"], dtype=np.float64),
            component_scale=np.array(archive["component_scale"], dtype=np.float64),
        )
        names = tuple(str(value) for value in archive["mode_names"].tolist())
        thresholds = {
            name: float(value)
            for name, value in zip(names, archive["mode_thresholds"].tolist(), strict=True)
        }
        model_hash = str(np.asarray(archive["model_sha256"]).item())
        method_hash = str(np.asarray(archive["method_sha256"]).item())
        charter_hash = str(np.asarray(archive["charter_sha256"]).item())
        development_hash = str(
            np.asarray(archive["development_result_sha256"]).item()
        )
    development = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    if (
        names != MODES
        or model_hash != bindings["model_sha256"]
        or method_hash != bindings["method_sha256"]
        or charter_hash != bindings["charter_sha256"]
        or development_hash != document["development_result_sha256"]
        or config.as_dict() != document.get("config")
        or document.get("candidate_id") != development.get("selected_candidate_id")
        or thresholds != document.get("thresholds")
    ):
        raise ValueError("frozen LC-DE bundle is incompatible")
    return state, thresholds, document


def _write_final_score_manifest(
    scored: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for item in scored:
        run = item["run"]
        score_file = Path(item["score_file"]).resolve()
        try:
            relative_score_file = score_file.relative_to(PROJECT).as_posix()
        except ValueError as error:
            raise ValueError("final score file escapes project") from error
        records.append(
            {
                "run_id": run["run_id"],
                "source_sha256": run["source_sha256"],
                "row_count": int(run["x"].shape[0]),
                "score_file": relative_score_file,
                "score_file_sha256": sha256_file(score_file),
            }
        )
    manifest = {
        "schema": "lcde_skab_final_score_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": "ALL_SCORES_FROZEN_BEFORE_ANY_HELDOUT_LABEL_ACCESS",
        "labels_opened_when_written": False,
        "frozen_method_sha256": sha256_file(FROZEN_JSON),
        "frozen_state_sha256": frozen["state_npz_sha256"],
        "artifact_bindings": _implementation_hashes(),
        "records": records,
    }
    _write_json(FINAL_SCORE_MANIFEST, manifest)
    return manifest


def _load_final_score_manifest(
    expected_runs: Sequence[Mapping[str, Any]], frozen: Mapping[str, Any]
) -> list[dict[str, Any]]:
    manifest = json.loads(FINAL_SCORE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "lcde_skab_final_score_manifest"
        or manifest.get("state")
        != "ALL_SCORES_FROZEN_BEFORE_ANY_HELDOUT_LABEL_ACCESS"
        or manifest.get("labels_opened_when_written") is not False
        or manifest.get("frozen_method_sha256") != sha256_file(FROZEN_JSON)
        or manifest.get("frozen_state_sha256") != frozen["state_npz_sha256"]
        or manifest.get("artifact_bindings") != _implementation_hashes()
    ):
        raise ValueError("invalid or stale final score manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(expected_runs):
        raise ValueError("final score manifest run count mismatch")
    output: list[dict[str, Any]] = []
    for run, record in zip(expected_runs, records, strict=True):
        expected_identity = (
            run["run_id"],
            run["source_sha256"],
            int(run["x"].shape[0]),
        )
        observed_identity = (
            record.get("run_id"),
            record.get("source_sha256"),
            record.get("row_count"),
        )
        if observed_identity != expected_identity:
            raise ValueError("final score manifest source binding mismatch")
        score_file = (PROJECT / str(record["score_file"])).resolve()
        try:
            score_file.relative_to(FINAL_SCORE_ROOT.resolve())
        except ValueError as error:
            raise ValueError("final score manifest path escapes output root") from error
        if sha256_file(score_file) != record.get("score_file_sha256"):
            raise ValueError("final score file hash mismatch")
        with np.load(score_file, allow_pickle=False) as archive:
            if set(archive.files) != {
                "run_id",
                "source_sha256",
                "frozen_state_sha256",
                *MODES,
            }:
                raise ValueError("final score archive schema mismatch")
            archive_identity = (
                str(np.asarray(archive["run_id"]).item()),
                str(np.asarray(archive["source_sha256"]).item()),
                str(np.asarray(archive["frozen_state_sha256"]).item()),
            )
            if archive_identity != (
                run["run_id"],
                run["source_sha256"],
                frozen["state_npz_sha256"],
            ):
                raise ValueError("final score archive source binding mismatch")
            paths = {
                mode: np.array(archive[mode], dtype=np.float64, copy=True)
                for mode in MODES
            }
        if any(
            path.shape != (expected_identity[2],) or not np.isfinite(path).all()
            for path in paths.values()
        ):
            raise ValueError("invalid final score path geometry or values")
        output.append({"run": run, "paths": paths, "score_file": str(score_file)})
    return output


def _final_cache_valid(document: Mapping[str, Any]) -> bool:
    return (
        document.get("schema") == "lcde_skab_final"
        and document.get("artifact_bindings") == _implementation_hashes()
        and document.get("frozen_method_sha256") == sha256_file(FROZEN_JSON)
        and document.get("frozen_state_sha256") == sha256_file(FROZEN_NPZ)
        and document.get("score_manifest_sha256") == sha256_file(FINAL_SCORE_MANIFEST)
    )


def final() -> None:
    if FINAL_RESULT.exists():
        document = json.loads(FINAL_RESULT.read_text(encoding="utf-8"))
        if not _final_cache_valid(document):
            raise RuntimeError("stale final artifact found; refusing a second held-out run")
        print(f"[final-cache] {document['decision']} {FINAL_RESULT}", flush=True)
        return
    started = time.perf_counter()
    charter = load_preaccess_charter(CHARTER_PATH)
    model = CRISModel.load_npz(MODEL_PATH)
    state, thresholds, frozen = _load_frozen()
    final_paths = _paths(charter, "final")
    if FINAL_SCORE_MANIFEST.exists():
        runs = _load_feature_runs(final_paths)
        scored = _load_final_score_manifest(runs, frozen)
        print("[final-score-manifest] resuming immutable scores", flush=True)
    else:
        runs = _load_runs(final_paths, model, label="final")
        generated: list[dict[str, Any]] = []
        FINAL_SCORE_ROOT.mkdir(parents=True, exist_ok=True)
        for run in runs:
            paths = {
                mode: state.score(run["x"], run["innovations"], mode=mode)
                for mode in MODES
            }
            target = FINAL_SCORE_ROOT / (run["run_id"].replace("/", "__") + ".npz")
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                run_id=np.asarray(run["run_id"]),
                source_sha256=np.asarray(run["source_sha256"]),
                frozen_state_sha256=np.asarray(frozen["state_npz_sha256"]),
                **paths,
            )
            temporary.replace(target)
            generated.append({"run": run, "paths": paths, "score_file": str(target)})
        _write_final_score_manifest(generated, frozen)
        # Evaluation consumes only the persisted, authenticated score files.
        scored = _load_final_score_manifest(runs, frozen)
    print("[final-score] every path authenticated; opening labels once", flush=True)

    per_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for item in scored:
        run = item["run"]
        labels = load_labels(DATA_ROOT / run["run_id"], dataset_root=DATA_ROOT)
        validate_score_label_binding(
            {
                "run_id": run["run_id"],
                "source_sha256": run["source_sha256"],
                "row_count": int(item["paths"]["dual"].size),
            },
            labels,
        )
        for mode in MODES:
            metrics = evaluate_skab_path(
                item["paths"][mode], thresholds[mode], labels.anomaly, run["timestamps"]
            )
            metrics["run_id"] = run["run_id"]
            per_mode[mode].append(metrics)
    aggregate = {mode: _aggregate(records) for mode, records in per_mode.items()}
    proposed = aggregate["dual"]
    success = (
        proposed["detected_runs"] >= 12
        and proposed["pre_event_false_alarm_runs"] <= 3
        and proposed["conditional_median_delay_seconds"] is not None
        and proposed["conditional_median_delay_seconds"] <= 30.0
        and proposed["conditional_nearest_rank_p90_delay_seconds"] is not None
        and proposed["conditional_nearest_rank_p90_delay_seconds"] <= 60.0
    )
    document = {
        "schema": "lcde_skab_final",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_seconds": time.perf_counter() - started,
        "device": model.attestation.cuda_device_name,
        "artifact_bindings": _implementation_hashes(),
        "frozen_method_sha256": sha256_file(FROZEN_JSON),
        "frozen_state_sha256": sha256_file(FROZEN_NPZ),
        "score_manifest_sha256": sha256_file(FINAL_SCORE_MANIFEST),
        "freeze_basis": frozen["freeze_basis"],
        "strict_development_gate_passed": frozen["strict_development_gate_passed"],
        "audit_interpretation": (
            "original_preregistered_confirmation"
            if frozen["strict_development_gate_passed"]
            else "posthoc_heldout_comparative_audit_not_original_preregistered_confirmation"
        ),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "per_run": per_mode,
        "success_reference": {
            "detected_runs_min": 12,
            "denominator": 16,
            "pre_event_false_alarm_runs_max": 3,
            "conditional_median_delay_seconds_max": 30.0,
            "conditional_nearest_rank_p90_delay_seconds_max": 60.0,
        },
        "passed": success,
        "decision": (
            "PASS_FROZEN_FINAL"
            if success and frozen["strict_development_gate_passed"]
            else "PASS_POSTHOC_HELDOUT_REFERENCE"
            if success
            else "FAIL_POSTHOC_HELDOUT_REFERENCE_NO_RETUNING"
            if not frozen["strict_development_gate_passed"]
            else "FAIL_FROZEN_FINAL_NO_RETUNING"
        ),
    }
    _write_json(FINAL_RESULT, document)
    print(f"[final] {document['decision']} {aggregate}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("develop", "freeze-comparative", "final", "all", "comparative-all"),
    )
    args = parser.parse_args()
    if args.phase == "comparative-all":
        develop()
        freeze_comparative()
        final()
        return 0
    if args.phase == "freeze-comparative":
        freeze_comparative()
        return 0
    if args.phase in {"develop", "all"}:
        go = develop()
        if not go:
            return 2
    if args.phase in {"final", "all"}:
        final()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
