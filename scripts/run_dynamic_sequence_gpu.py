"""Run the frozen sequential CVA then multi-scale GPU development experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Callable, Iterable


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# This Windows environment must initialize the CUDA runtime before NumPy/MKL.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc.cva_mewma import CausalCVAMEWMAMonitor
from mate_spc.datasets import RiethTEPArchive, SensorSequence
from mate_spc.gpu_dynamic import (
    DynamicGPUResult,
    score_cva_runs_cuda,
    score_multiscale_runs_cuda,
)
from mate_spc.gpu_multirun import cuda_runtime_metadata
from mate_spc.metrics import contiguous_regions, event_metrics
from mate_spc.multiscale_sparse import CausalMultiScaleSparseMonitor


DESIGN_RUNS = tuple(range(1, 21))
CALIBRATION_RUNS = tuple(range(21, 121))
FAULT_IDS = (3, 9, 15)
FAULT_START = 160
STAGE_A_AUDIT_RUNS = tuple(range(221, 321))
STAGE_A_FAULT_RUNS = tuple(range(11, 51))
STAGE_B_AUDIT_RUNS = tuple(range(321, 401))
STAGE_B_FAULT_RUNS = tuple(range(51, 101))
FINAL_RUNS = tuple(range(401, 501))
DATASET_DOI = "10.7910/DVN/6C3JR1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            _clean_json(value),
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".pkl",
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".npz",
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=path.suffix,
        prefix=f".{path.stem}-",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen sequential causal dynamic Rieth experiment on CUDA"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nis-result", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _source_hashes(project_root: Path) -> dict[str, str]:
    source_paths = sorted((project_root / "src" / "mate_spc").glob("*.py"))
    extras = (
        project_root / "scripts" / "run_dynamic_sequence_gpu.py",
        project_root / "tests" / "test_dynamic_monitors.py",
        project_root / "tests" / "test_gpu_dynamic.py",
        project_root / "tests" / "_gpu_dynamic_probe.py",
        project_root / "tests" / "test_run_dynamic_sequence_gpu.py",
        project_root / "pyproject.toml",
    )
    paths = tuple(source_paths) + extras
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol source files are missing: {missing}")
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in paths
    }


def _read_failed_nis(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"prior NIS aggregate not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    gate = value.get("pre_registered_gate_result")
    if not isinstance(gate, dict) or gate.get("decision") != "FAIL_PRE_REGISTERED_GATE":
        raise ValueError("sequential fallback requires the recorded failed NIS gate")
    fingerprint = value.get("protocol_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("prior NIS result lacks a protocol fingerprint")
    return _sha256_file(path), fingerprint


def _protocol(
    *,
    checksums: dict[str, str],
    source_hashes: dict[str, str],
    nis_sha256: str,
    nis_fingerprint: str,
    batch_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "analysis_status": "sequential_exploratory_development",
        "dataset": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": DATASET_DOI,
        "dataset_md5": checksums,
        "prior_failed_nis": {
            "aggregate_sha256": nis_sha256,
            "protocol_fingerprint": nis_fingerprint,
            "required_decision": "FAIL_PRE_REGISTERED_GATE",
        },
        "source_sha256": source_hashes,
        "execution": {
            "required_device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "cpu_gpu_tolerance": 1e-10,
        },
        "sequence_rule": {
            "stage_a": "always run CVA after design/calibration validation",
            "stage_b": "run fallback iff stage_a fails any gate",
            "stop_search": "stop immediately when a stage passes all gates",
            "if_both_fail": "stop this sequence; do not screen runs 101-400",
        },
        "common_decision": {
            "alpha": 0.05,
            "calibration_runs": 100,
            "calibration_unit": "independent healthy run family-envelope maximum",
            "conformal_rank_one_based": 96,
            "persistence": "minimum of latest 3 within each component",
            "family_fusion": "maximum after component-specific persistence",
            "alarm_comparison": "strictly_greater_than",
            "fault_onset_zero_based": FAULT_START,
            "detection": "alarm episode onset at or after fault onset",
            "pre_fault_episode_continuing_into_fault": "not credited",
            "miss_delay_samples": 800,
            "healthy_gate_interpretation": (
                "pre-registered operational screen; fixed-p=0.05 binomial tail is "
                "a benchmark, not an exact finite-calibration conformal significance"
            ),
            "finite_calibration_predictive_reference": (
                "continuous no-tie exchangeable model gives a Beta-binomial count "
                "with a=5, b=96 for the q96-of-100 threshold"
            ),
        },
        "stage_a_method": {
            "name": "causal_regularized_CVA_signed_vector_MEWMA_family_3of3",
            "features": 52,
            "raw_scaler": "design median / 1.4826*MAD; SD fallback",
            "past_order": 5,
            "past_ordering": "current_to_oldest",
            "future_order_design_only": 5,
            "online_future_samples": 0,
            "ridge": "0.001 * trace(covariance) / dimension",
            "rank": "minimum squared canonical correlation energy >= 0.90",
            "rank_failure": "fail if rank does not split state and residual",
            "signed_vector_mewma_lambda": 0.2,
            "mewma_initial": "zero vector independently per run",
            "components": ["state_T2", "residual_Q"],
            "component_transform": "log1p then design median/MAD normalization",
        },
        "stage_b_method": {
            "name": "causal_multiscale_sparse_dynamic_family_envelope_3of3",
            "features": 52,
            "raw_scaler": "design median / 1.4826*MAD; SD fallback",
            "windows_samples": [5, 20, 60],
            "windows_minutes": [15, 60, 180],
            "statistics": ["mean", "log_variance", "log_difference_energy"],
            "epsilon": 1e-8,
            "channel_normalization": "separate design median/MAD per group",
            "sparse_energy": "mean largest 3 squared standardized channels",
            "component_transform": "log1p then design median/MAD normalization",
            "component_count": 9,
        },
        "splits": {
            "design": ["fault_free_training", list(DESIGN_RUNS)],
            "calibration": ["fault_free_testing", list(CALIBRATION_RUNS)],
            "already_consumed": {
                "healthy": ["fault_free_testing", list(range(121, 221))],
                "faults": ["faulty_testing", list(FAULT_IDS), list(range(1, 11))],
            },
            "stage_a_audit": [
                "fault_free_testing",
                list(STAGE_A_AUDIT_RUNS),
            ],
            "stage_a_fault": [
                "faulty_testing",
                list(FAULT_IDS),
                list(STAGE_A_FAULT_RUNS),
            ],
            "stage_b_audit_if_triggered": [
                "fault_free_testing",
                list(STAGE_B_AUDIT_RUNS),
            ],
            "stage_b_fault_if_triggered": [
                "faulty_testing",
                list(FAULT_IDS),
                list(STAGE_B_FAULT_RUNS),
            ],
            "final_healthy_reserved": ["fault_free_testing", list(FINAL_RUNS)],
            "final_fault_reserved": [
                "faulty_testing",
                list(range(1, 21)),
                list(FINAL_RUNS),
            ],
        },
        "data_access_semantics": {
            "archive_format": "partition-level RData decoded before row selection",
            "model_visible_rows": "only explicit selections returned by load_runs",
            "reserved_claim": "runs 401-500 are not selected, scored, or inspected",
        },
        "gates": {
            "stage_a": {
                "healthy_alarm_runs_maximum": 9,
                "healthy_total": 100,
                "fault_detected_runs_minimum": 60,
                "fault_total": 120,
                "per_fault_detected_runs_minimum": 12,
                "per_fault_total": 40,
                "all_required": True,
            },
            "stage_b": {
                "healthy_alarm_runs_maximum": 7,
                "healthy_total": 80,
                "fault_detected_runs_minimum": 75,
                "fault_total": 150,
                "per_fault_detected_runs_minimum": 15,
                "per_fault_total": 50,
                "all_required": True,
            },
        },
        "forbidden_after_manifest": [
            "method family",
            "past/future order",
            "CVA ridge/rank rule/lambda",
            "fallback windows/statistics/top-k",
            "component transforms/fusion",
            "persistence/alpha/gates",
            "data splits",
        ],
    }


def _load_or_create_manifest(path: Path, protocol: dict[str, object]) -> str:
    fingerprint = _fingerprint(protocol)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        stored = current.pop("protocol_fingerprint", None)
        if stored != _fingerprint(current) or stored != fingerprint:
            raise ValueError("existing output has a corrupt or different protocol")
        return fingerprint
    existing = tuple(path.parent.iterdir())
    if existing:
        raise ValueError(
            "output directory is nonempty but has no manifest; use a fresh directory"
        )
    _atomic_json(path, {**protocol, "protocol_fingerprint": fingerprint})
    return fingerprint


def _verify_completion(output_dir: Path, fingerprint: str) -> bool:
    path = output_dir / "completion.json"
    if not path.exists():
        return False
    completion = json.loads(path.read_text(encoding="utf-8"))
    stored = completion.pop("completion_fingerprint", None)
    if stored != _fingerprint(completion):
        raise ValueError("completion manifest fingerprint is corrupt")
    if completion.get("protocol_fingerprint") != fingerprint:
        raise ValueError("completion manifest belongs to another protocol")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("completion manifest has no artifacts")
    for relative, expected in artifacts.items():
        target = output_dir / relative
        if not target.is_file() or not isinstance(expected, dict):
            raise ValueError(f"completed artifact is missing: {relative}")
        if target.stat().st_size != int(expected.get("size_bytes", -1)):
            raise ValueError(f"completed artifact size changed: {relative}")
        if _sha256_file(target) != expected.get("sha256"):
            raise ValueError(f"completed artifact hash changed: {relative}")
    return True


def _write_completion(
    output_dir: Path,
    fingerprint: str,
    decision: str,
    artifact_names: Iterable[str],
) -> None:
    artifacts: dict[str, object] = {}
    for name in sorted(set(artifact_names)):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"cannot complete: artifact is missing: {name}")
        artifacts[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    value = {
        "schema_version": 1,
        "protocol_fingerprint": fingerprint,
        "decision": decision,
        "artifacts": artifacts,
    }
    _atomic_json(
        output_dir / "completion.json",
        {**value, "completion_fingerprint": _fingerprint(value)},
    )


def _run_ids(
    runs: Iterable[SensorSequence],
    expected: tuple[int, ...],
    *,
    label: str,
) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(int(run.metadata["simulation_run"]) for run in normalized)
    if actual != expected:
        raise RuntimeError(f"{label} run IDs differ from the frozen split")
    return normalized


def _fault_run_ids(
    runs: Iterable[SensorSequence],
    expected_runs: tuple[int, ...],
    *,
    label: str,
) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(
        (int(run.metadata["fault_id"]), int(run.metadata["simulation_run"]))
        for run in normalized
    )
    expected = tuple(
        (fault_id, run_id)
        for fault_id in FAULT_IDS
        for run_id in expected_runs
    )
    if actual != expected:
        raise RuntimeError(f"{label} fault IDs/runs differ from the frozen split")
    return normalized


def _named_matrices(
    runs: Iterable[SensorSequence],
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple((run.name, np.asarray(run.x, dtype=np.float64)) for run in runs)


def _score_split(
    label: str,
    scorer: Callable[..., DynamicGPUResult],
    monitor: object,
    runs: tuple[SensorSequence, ...],
    *,
    device: str,
    batch_size: int,
) -> tuple[DynamicGPUResult, dict[str, object]]:
    started = time.perf_counter()
    result = scorer(monitor, runs, device=device, batch_size=batch_size)
    metadata = {
        "split": label,
        "run_count": len(runs),
        "wall_seconds": time.perf_counter() - started,
        **result.execution,
    }
    print(
        f"{label}: {len(runs)} runs, {result.execution['time_steps']} steps, "
        f"{result.execution['cuda_event_elapsed_seconds']:.3f} CUDA seconds",
        flush=True,
    )
    return result, metadata


def _difference_summary(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(f"CPU/CUDA shapes differ: {left.shape} versus {right.shape}")
    finite_exact = bool(
        np.array_equal(np.isfinite(left), np.isfinite(right))
    )
    nonfinite_exact = bool(
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    )
    finite = np.isfinite(left) & np.isfinite(right)
    maximum = (
        float(np.max(np.abs(left[finite] - right[finite])))
        if finite.any()
        else 0.0
    )
    return {
        "maximum_absolute": maximum,
        "finite_pattern_exact": finite_exact,
        "nonfinite_pattern_exact": nonfinite_exact,
        "exact_equal": bool(np.array_equal(left, right, equal_nan=True)),
    }


def _cpu_path_fields(monitor: object, run: SensorSequence) -> dict[str, np.ndarray]:
    result = monitor.decision_path(run.x)
    fields = {
        "component_score": result.component_score,
        "component_persistence": result.component_persistence,
        "persistence": result.persistence,
    }
    if hasattr(result, "state_energy"):
        fields["state_energy"] = result.state_energy
        fields["residual_energy"] = result.residual_energy
    return fields


def _validate_cuda_path(
    name: str,
    monitor: object,
    scorer: Callable[..., DynamicGPUResult],
    run: SensorSequence,
    production: dict[str, np.ndarray],
    *,
    device: str,
) -> dict[str, object]:
    cpu = _cpu_path_fields(monitor, run)
    comparisons = {
        field: _difference_summary(reference, production[field])
        for field, reference in cpu.items()
    }
    duplicate = replace(run, name=f"{run.name}-duplicate")
    duplicate_result = scorer(
        monitor, (run, duplicate), device=device, batch_size=2
    )
    duplicate_comparisons = {
        field: _difference_summary(
            duplicate_result.by_run[run.name][field],
            duplicate_result.by_run[duplicate.name][field],
        )
        for field in cpu
    }
    single = scorer(monitor, (run,), device=device, batch_size=1)
    batch_comparisons = {
        field: _difference_summary(single.by_run[run.name][field], production[field])
        for field in cpu
    }
    tolerance = 1e-10
    numeric_passed = all(
        bool(item["finite_pattern_exact"])
        and bool(item["nonfinite_pattern_exact"])
        and float(item["maximum_absolute"]) <= tolerance
        for group in (comparisons, duplicate_comparisons, batch_comparisons)
        for item in group.values()
    )
    duplicate_exact = all(
        bool(item["exact_equal"]) for item in duplicate_comparisons.values()
    )
    cpu_alarm = np.isfinite(cpu["persistence"]) & (
        cpu["persistence"] > float(monitor.threshold_)
    )
    gpu_alarm = np.isfinite(production["persistence"]) & (
        production["persistence"] > float(monitor.threshold_)
    )
    alarm_exact = bool(np.array_equal(cpu_alarm, gpu_alarm))
    passed = numeric_passed and duplicate_exact and alarm_exact
    result = {
        "passed": passed,
        "name": name,
        "run": run.name,
        "tolerance": tolerance,
        "cpu_vs_production": comparisons,
        "batch_duplicate": duplicate_comparisons,
        "batch_duplicate_exact": duplicate_exact,
        "batch_size_1_vs_production": batch_comparisons,
        "alarm_exact": alarm_exact,
        "validation_execution": {
            "duplicate": duplicate_result.execution,
            "single": single.execution,
        },
    }
    if not passed:
        raise RuntimeError(f"{name} CPU/CUDA validation failed: {result}")
    return result


def _episode_starts(alarm: np.ndarray) -> list[int]:
    return [start for start, _ in contiguous_regions(np.asarray(alarm, dtype=bool))]


def _health_records(
    monitor: object,
    runs: tuple[SensorSequence, ...],
    scored: DynamicGPUResult,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    threshold = float(monitor.threshold_)
    for run in runs:
        if np.asarray(run.fault, dtype=bool).any():
            raise ValueError("healthy audit contains a faulty run")
        persistence = scored.by_run[run.name]["persistence"]
        eligible = np.isfinite(persistence)
        alarm = eligible & (persistence > threshold)
        starts = _episode_starts(alarm)
        maximum = float(np.max(persistence[eligible]))
        records.append(
            {
                "run": run.name,
                "simulation_run": int(run.metadata["simulation_run"]),
                "any_alarm": bool(starts),
                "alarm_episode_count": len(starts),
                "first_alarm_index": starts[0] if starts else None,
                "maximum_persistence": maximum,
                "maximum_to_threshold_ratio": maximum / threshold,
            }
        )
    return records


def _fault_records(
    monitor: object,
    runs: tuple[SensorSequence, ...],
    scored: DynamicGPUResult,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    threshold = float(monitor.threshold_)
    for run in runs:
        fault = np.asarray(run.fault, dtype=bool)
        if contiguous_regions(fault) != [(FAULT_START, run.x.shape[0])]:
            raise RuntimeError(f"unexpected fault interval for {run.name}")
        persistence = scored.by_run[run.name]["persistence"]
        eligible = np.isfinite(persistence)
        alarm = eligible & (persistence > threshold)
        regions = contiguous_regions(alarm)
        starts = [start for start, _ in regions]
        post_starts = [
            index for index in starts if FAULT_START <= index < run.x.shape[0]
        ]
        first = post_starts[0] if post_starts else None
        credited_alarm = np.zeros_like(alarm)
        for start, end in regions:
            if start >= FAULT_START:
                credited_alarm[start:end] = True
        metrics = event_metrics(
            credited_alarm,
            fault,
            run.sample_period_seconds,
            warning_horizon=0,
            refractory=0,
            decision_mask=eligible,
        ).as_dict()
        maximum = float(np.max(persistence[fault & eligible]))
        records.append(
            {
                "run": run.name,
                "fault_id": int(run.metadata["fault_id"]),
                "simulation_run": int(run.metadata["simulation_run"]),
                "detected": bool(post_starts),
                "first_fault_alarm_episode_index": first,
                "delay_samples": first - FAULT_START if first is not None else None,
                "restricted_delay_samples_miss_as_800": (
                    first - FAULT_START
                    if first is not None
                    else run.x.shape[0] - FAULT_START
                ),
                "pre_fault_alarm_episode_count": sum(
                    start < FAULT_START for start, _ in regions
                ),
                "maximum_post_fault_persistence": maximum,
                "maximum_to_threshold_ratio": maximum / threshold,
                "monitoring_after_episode_onset_rule": metrics,
            }
        )
    return records


def _wilson(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return {
        "proportion": p,
        "lower_95": max(0.0, center - half),
        "upper_95": min(1.0, center + half),
    }


def _binomial_upper_tail(total: int, probability: float, at_least: int) -> float:
    return float(
        sum(
            math.comb(total, count)
            * probability**count
            * (1.0 - probability) ** (total - count)
            for count in range(at_least, total + 1)
        )
    )


def _beta_binomial_upper_tail(
    total: int,
    alpha_shape: float,
    beta_shape: float,
    at_least: int,
) -> float:
    """Predictive count tail after integrating an uncertain Bernoulli rate."""
    log_beta_denominator = (
        math.lgamma(alpha_shape)
        + math.lgamma(beta_shape)
        - math.lgamma(alpha_shape + beta_shape)
    )
    probabilities = []
    for count in range(at_least, total + 1):
        log_combination = (
            math.lgamma(total + 1)
            - math.lgamma(count + 1)
            - math.lgamma(total - count + 1)
        )
        log_beta_numerator = (
            math.lgamma(count + alpha_shape)
            + math.lgamma(total - count + beta_shape)
            - math.lgamma(total + alpha_shape + beta_shape)
        )
        probabilities.append(
            math.exp(log_combination + log_beta_numerator - log_beta_denominator)
        )
    return float(math.fsum(probabilities))


def _fault_aggregate(
    records: list[dict[str, object]],
    fault_id: int | None = None,
) -> dict[str, object]:
    selected = [
        row
        for row in records
        if fault_id is None or int(row["fault_id"]) == fault_id
    ]
    detected = sum(bool(row["detected"]) for row in selected)
    delays = [
        int(row["delay_samples"])
        for row in selected
        if row["delay_samples"] is not None
    ]
    restricted = [
        int(row["restricted_delay_samples_miss_as_800"]) for row in selected
    ]
    return {
        "run_count": len(selected),
        "detected_runs": detected,
        "detection_rate": _wilson(detected, len(selected)),
        "conditional_median_delay_samples": (
            float(np.median(delays)) if delays else None
        ),
        "restricted_mean_delay_samples_miss_as_800": float(np.mean(restricted)),
        "runs_with_pre_fault_alarm": sum(
            int(row["pre_fault_alarm_episode_count"]) > 0 for row in selected
        ),
        "pre_fault_alarm_episodes": sum(
            int(row["pre_fault_alarm_episode_count"]) for row in selected
        ),
    }


def _stage_summary(
    *,
    name: str,
    health_records: list[dict[str, object]],
    fault_records: list[dict[str, object]],
    healthy_maximum: int,
    fault_minimum: int,
    per_fault_minimum: int,
) -> dict[str, object]:
    health_alarms = sum(bool(row["any_alarm"]) for row in health_records)
    health_total = len(health_records)
    overall = _fault_aggregate(fault_records)
    by_fault = {
        f"{fault_id:02d}": _fault_aggregate(fault_records, fault_id)
        for fault_id in FAULT_IDS
    }
    per_fault_counts = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"])
        for fault_id in FAULT_IDS
    ]
    healthy_passed = health_alarms <= healthy_maximum
    total_passed = int(overall["detected_runs"]) >= fault_minimum
    per_fault_passed = min(per_fault_counts) >= per_fault_minimum
    passed = healthy_passed and total_passed and per_fault_passed
    return {
        "stage": name,
        "passed": passed,
        "decision": "PASS_ALL_PRE_REGISTERED_GATES" if passed else "FAIL_GATE",
        "healthy_audit": {
            "run_count": health_total,
            "runs_with_any_alarm": health_alarms,
            "run_alarm_rate": _wilson(health_alarms, health_total),
            "alarm_episodes": sum(
                int(row["alarm_episode_count"]) for row in health_records
            ),
            "gate_maximum_alarm_runs": healthy_maximum,
            "fixed_p_0.05_binomial_benchmark_tail": _binomial_upper_tail(
                health_total, 0.05, healthy_maximum + 1
            ),
            "finite_calibration_beta_binomial_predictive_tail": (
                _beta_binomial_upper_tail(
                    health_total, 5.0, 96.0, healthy_maximum + 1
                )
            ),
            "gate_interpretation": (
                "pre-registered operational screen; benchmark tail is not an exact "
                "finite-calibration conformal significance"
            ),
            "records": health_records,
        },
        "fault_evaluation": {
            "run_count": len(fault_records),
            "overall": overall,
            "by_fault": by_fault,
            "records": fault_records,
        },
        "gate": {
            "healthy_gate_passed": healthy_passed,
            "fault_total_gate_passed": total_passed,
            "per_fault_gate_passed": per_fault_passed,
            "healthy_alarm_runs_maximum": healthy_maximum,
            "fault_detected_runs_minimum": fault_minimum,
            "per_fault_detected_runs_minimum": per_fault_minimum,
            "minimum_detected_runs_across_faults": min(per_fault_counts),
        },
    }


def _pack_scores(
    target: dict[str, np.ndarray],
    result: DynamicGPUResult,
    runs: Iterable[SensorSequence],
    *,
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for run in runs:
        metadata = run.metadata
        if int(metadata["fault_id"]):
            key = (
                f"{prefix}_fault_{int(metadata['fault_id']):02d}_"
                f"run_{int(metadata['simulation_run']):03d}"
            )
        else:
            key = f"{prefix}_{int(metadata['simulation_run']):03d}"
        arrays = result.by_run[run.name]
        flattened: list[np.ndarray] = []
        for field in fields:
            value = np.asarray(arrays[field], dtype=np.float64)
            flattened.append(value if value.ndim == 2 else value[:, None])
        target[key] = np.concatenate(flattened, axis=1)


def _merge_execution(parts: Iterable[dict[str, object]]) -> dict[str, object]:
    records = tuple(parts)
    return {
        "backend": "pytorch_cuda_frozen_dynamic_sequence",
        "device": records[0]["device"],
        "device_name": records[0]["device_name"],
        "dtype": "float64",
        "kernel_calls": sum(int(item["kernel_calls"]) for item in records),
        "sequences": sum(int(item["sequences"]) for item in records),
        "time_steps": sum(int(item["time_steps"]) for item in records),
        "cuda_event_elapsed_seconds": sum(
            float(item["cuda_event_elapsed_seconds"]) for item in records
        ),
        "wall_seconds": sum(float(item["wall_seconds"]) for item in records),
        "fallback_sequences": sum(
            int(item["fallback_sequences"]) for item in records
        ),
        "factorization_failures": sum(
            int(item["factorization_failures"]) for item in records
        ),
        "parameter_fitting_backend": "numpy_cpu_float64",
        "scoring_backend": "pytorch_cuda_float64",
        "decision_backend": "numpy_cpu",
        "tf32": False,
    }


def _report_text(aggregate: dict[str, object]) -> str:
    lines = [
        "# Frozen sequential dynamic GPU experiment",
        "",
        f"Protocol fingerprint: {aggregate['protocol_fingerprint']}.",
        "",
        "Stage B was frozen before Stage A audit data were loaded and was run "
        "only if Stage A failed.",
        "",
    ]
    stages = (
        ("stage_a", "Stage A: causal CVA-MEWMA"),
        ("stage_b", "Stage B: multi-scale sparse fallback"),
    )
    for key, title in stages:
        stage = aggregate.get(key)
        lines.extend([f"## {title}", ""])
        if not isinstance(stage, dict) or stage.get("status") == "not_triggered":
            lines.extend(["Not triggered because Stage A passed.", ""])
            continue
        health = stage["healthy_audit"]
        fault = stage["fault_evaluation"]
        gate = stage["gate"]
        lines.extend(
            [
                "| Gate | Result | Required | Pass |",
                "|---|---:|---:|---:|",
                f"| Healthy runs with alarm | {health['runs_with_any_alarm']}/"
                f"{health['run_count']} | <= {gate['healthy_alarm_runs_maximum']} | "
                f"{str(gate['healthy_gate_passed']).lower()} |",
                f"| Fault runs detected | {fault['overall']['detected_runs']}/"
                f"{fault['run_count']} | >= {gate['fault_detected_runs_minimum']} | "
                f"{str(gate['fault_total_gate_passed']).lower()} |",
                f"| Minimum detected per fault | "
                f"{gate['minimum_detected_runs_across_faults']} | >= "
                f"{gate['per_fault_detected_runs_minimum']} | "
                f"{str(gate['per_fault_gate_passed']).lower()} |",
                "",
                f"Decision: {stage['decision']}",
                "",
                "| Fault | Detected | Conditional median delay | Restricted mean delay |",
                "|---:|---:|---:|---:|",
            ]
        )
        for fault_id in FAULT_IDS:
            item = fault["by_fault"][f"{fault_id:02d}"]
            delay = item["conditional_median_delay_samples"]
            delay_text = "-" if delay is None else f"{delay:.1f}"
            lines.append(
                f"| {fault_id:02d} | {item['detected_runs']}/{item['run_count']} | "
                f"{delay_text} | {item['restricted_mean_delay_samples_miss_as_800']:.1f} |"
            )
        lines.append("")
    execution = aggregate["gpu_execution"]["production_total"]
    lines.extend(
        [
            "## Final sequential decision",
            "",
            f"{aggregate['sequential_decision']}",
            "",
            f"Selected method: {aggregate['selected_method'] or 'none'}.",
            "",
            "## GPU execution",
            "",
            f"- Device: {execution['device_name']}; dtype: float64; TF32: false.",
            f"- Production: {execution['sequences']} sequence evaluations, "
            f"{execution['time_steps']} sequence-steps, {execution['kernel_calls']} CUDA batches.",
            f"- CUDA-event time: {execution['cuda_event_elapsed_seconds']:.3f} seconds.",
            f"- Fallback/factorization failures: "
            f"{execution['fallback_sequences']}/"
            f"{execution['factorization_failures']}.",
            "",
            "Healthy and faulty runs 401-500 were not selected, scored, or inspected; "
            "they remain reserved for one final confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nis_sha256, nis_fingerprint = _read_failed_nis(args.nis_result)
    archive = RiethTEPArchive(args.data_dir, verify_checksums=True)
    checksums = archive.verify(
        ("fault_free_training", "fault_free_testing", "faulty_testing")
    )
    protocol = _protocol(
        checksums=checksums,
        source_hashes=_source_hashes(project_root),
        nis_sha256=nis_sha256,
        nis_fingerprint=nis_fingerprint,
        batch_size=args.batch_size,
    )
    fingerprint = _load_or_create_manifest(
        args.output_dir / "manifest.json", protocol
    )
    if _verify_completion(args.output_dir, fingerprint):
        print("frozen sequential dynamic experiment is already complete", flush=True)
        return 0

    runtime = cuda_runtime_metadata(args.device)
    print("loading design runs 1..20 and calibration runs 21..120", flush=True)
    design = _run_ids(
        archive.load_runs(
            "fault_free_training",
            {0: DESIGN_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        DESIGN_RUNS,
        label="design",
    )
    calibration = _run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: CALIBRATION_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        CALIBRATION_RUNS,
        label="calibration",
    )

    fit_started = time.perf_counter()
    cva = CausalCVAMEWMAMonitor().fit_design(_named_matrices(design))
    cva_fit_seconds = time.perf_counter() - fit_started
    fit_started = time.perf_counter()
    fallback = CausalMultiScaleSparseMonitor().fit_design(
        _named_matrices(design)
    )
    fallback_fit_seconds = time.perf_counter() - fit_started
    print(
        f"design fit: CVA rank={cva.rank_} ({cva_fit_seconds:.2f}s); "
        f"fallback={fallback_fit_seconds:.2f}s",
        flush=True,
    )

    import torch

    torch.cuda.reset_peak_memory_stats()
    executions: list[dict[str, object]] = []
    cva_scores: dict[str, np.ndarray] = {}
    fallback_scores: dict[str, np.ndarray] = {}
    cva_fields = (
        "state_energy",
        "residual_energy",
        "component_score",
        "component_persistence",
        "persistence",
    )
    fallback_fields = (
        "component_score",
        "component_persistence",
        "persistence",
    )

    cva_design, execution = _score_split(
        "cva_design",
        score_cva_runs_cuda,
        cva,
        design,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)
    cva_calibration, execution = _score_split(
        "cva_calibration",
        score_cva_runs_cuda,
        cva,
        calibration,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)
    fallback_design, execution = _score_split(
        "fallback_design",
        score_multiscale_runs_cuda,
        fallback,
        design,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)
    fallback_calibration, execution = _score_split(
        "fallback_calibration",
        score_multiscale_runs_cuda,
        fallback,
        calibration,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)

    cva.calibrate_envelopes(
        (
            run.name,
            cva_calibration.by_run[run.name]["persistence"],
        )
        for run in calibration
    )
    fallback.calibrate_envelopes(
        (
            run.name,
            fallback_calibration.by_run[run.name]["persistence"],
        )
        for run in calibration
    )
    cva_validation = {
        "design": _validate_cuda_path(
            "cva_design",
            cva,
            score_cva_runs_cuda,
            design[0],
            cva_design.by_run[design[0].name],
            device=args.device,
        ),
        "calibration": _validate_cuda_path(
            "cva_calibration",
            cva,
            score_cva_runs_cuda,
            calibration[0],
            cva_calibration.by_run[calibration[0].name],
            device=args.device,
        ),
    }
    fallback_validation = {
        "design": _validate_cuda_path(
            "fallback_design",
            fallback,
            score_multiscale_runs_cuda,
            design[0],
            fallback_design.by_run[design[0].name],
            device=args.device,
        ),
        "calibration": _validate_cuda_path(
            "fallback_calibration",
            fallback,
            score_multiscale_runs_cuda,
            calibration[0],
            fallback_calibration.by_run[calibration[0].name],
            device=args.device,
        ),
    }
    validation = {
        "passed": True,
        "protocol_fingerprint": fingerprint,
        "cva": cva_validation,
        "fallback": fallback_validation,
    }
    _pack_scores(
        cva_scores, cva_design, design, prefix="design", fields=cva_fields
    )
    _pack_scores(
        cva_scores,
        cva_calibration,
        calibration,
        prefix="calibration",
        fields=cva_fields,
    )
    _pack_scores(
        fallback_scores,
        fallback_design,
        design,
        prefix="design",
        fields=fallback_fields,
    )
    _pack_scores(
        fallback_scores,
        fallback_calibration,
        calibration,
        prefix="calibration",
        fields=fallback_fields,
    )
    cva_calibration_summary = {
        **cva.calibration_summary_,
        "protocol_fingerprint": fingerprint,
    }
    fallback_calibration_summary = {
        **fallback.calibration_summary_,
        "protocol_fingerprint": fingerprint,
    }
    _atomic_json(args.output_dir / "cuda_validation.json", validation)
    _atomic_json(
        args.output_dir / "stage_a_calibration.json", cva_calibration_summary
    )
    _atomic_json(
        args.output_dir / "stage_b_calibration.json", fallback_calibration_summary
    )
    _atomic_pickle(args.output_dir / "fitted_stage_a_cva.pkl", cva)
    _atomic_pickle(args.output_dir / "fitted_stage_b_fallback.pkl", fallback)

    del cva_design, cva_calibration, fallback_design, fallback_calibration
    gc.collect()
    print("opening Stage A audit 221..320 and faults 11..50", flush=True)
    stage_a_audit = _run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: STAGE_A_AUDIT_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        STAGE_A_AUDIT_RUNS,
        label="stage A healthy audit",
    )
    stage_a_fault = _fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: STAGE_A_FAULT_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        STAGE_A_FAULT_RUNS,
        label="stage A",
    )
    stage_a_audit_scored, execution = _score_split(
        "stage_a_healthy_audit",
        score_cva_runs_cuda,
        cva,
        stage_a_audit,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)
    stage_a_fault_scored, execution = _score_split(
        "stage_a_fault_evaluation",
        score_cva_runs_cuda,
        cva,
        stage_a_fault,
        device=args.device,
        batch_size=args.batch_size,
    )
    executions.append(execution)
    stage_a = _stage_summary(
        name="stage_a",
        health_records=_health_records(cva, stage_a_audit, stage_a_audit_scored),
        fault_records=_fault_records(cva, stage_a_fault, stage_a_fault_scored),
        healthy_maximum=9,
        fault_minimum=60,
        per_fault_minimum=12,
    )
    _pack_scores(
        cva_scores,
        stage_a_audit_scored,
        stage_a_audit,
        prefix="audit",
        fields=cva_fields,
    )
    _pack_scores(
        cva_scores,
        stage_a_fault_scored,
        stage_a_fault,
        prefix="evaluation",
        fields=cva_fields,
    )
    _atomic_json(args.output_dir / "stage_a_result.json", stage_a)
    print(f"Stage A decision: {stage_a['decision']}", flush=True)

    stage_b: dict[str, object]
    if bool(stage_a["passed"]):
        stage_b = {
            "stage": "stage_b",
            "status": "not_triggered",
            "reason": "Stage A passed every frozen gate",
        }
        selected_method = cva_calibration_summary["method"]
        sequential_decision = "PASS_STAGE_A_STOP_SEQUENCE"
    else:
        del stage_a_audit_scored, stage_a_fault_scored, stage_a_audit, stage_a_fault
        gc.collect()
        print("Stage A failed; opening frozen Stage B audit/fault splits", flush=True)
        stage_b_audit = _run_ids(
            archive.load_runs(
                "fault_free_testing",
                {0: STAGE_B_AUDIT_RUNS},
                fault_start=FAULT_START,
                split_manipulated=False,
            ),
            STAGE_B_AUDIT_RUNS,
            label="stage B healthy audit",
        )
        stage_b_fault = _fault_run_ids(
            archive.load_runs(
                "faulty_testing",
                {fault_id: STAGE_B_FAULT_RUNS for fault_id in FAULT_IDS},
                fault_start=FAULT_START,
                split_manipulated=False,
            ),
            STAGE_B_FAULT_RUNS,
            label="stage B",
        )
        stage_b_audit_scored, execution = _score_split(
            "stage_b_healthy_audit",
            score_multiscale_runs_cuda,
            fallback,
            stage_b_audit,
            device=args.device,
            batch_size=args.batch_size,
        )
        executions.append(execution)
        stage_b_fault_scored, execution = _score_split(
            "stage_b_fault_evaluation",
            score_multiscale_runs_cuda,
            fallback,
            stage_b_fault,
            device=args.device,
            batch_size=args.batch_size,
        )
        executions.append(execution)
        stage_b = _stage_summary(
            name="stage_b",
            health_records=_health_records(
                fallback, stage_b_audit, stage_b_audit_scored
            ),
            fault_records=_fault_records(
                fallback, stage_b_fault, stage_b_fault_scored
            ),
            healthy_maximum=7,
            fault_minimum=75,
            per_fault_minimum=15,
        )
        _pack_scores(
            fallback_scores,
            stage_b_audit_scored,
            stage_b_audit,
            prefix="audit",
            fields=fallback_fields,
        )
        _pack_scores(
            fallback_scores,
            stage_b_fault_scored,
            stage_b_fault,
            prefix="evaluation",
            fields=fallback_fields,
        )
        print(f"Stage B decision: {stage_b['decision']}", flush=True)
        if bool(stage_b["passed"]):
            selected_method = fallback_calibration_summary["method"]
            sequential_decision = "PASS_STAGE_B_AFTER_STAGE_A_FAILURE"
        else:
            selected_method = None
            sequential_decision = "FAIL_BOTH_FROZEN_STAGES_STOP_SEARCH"

    torch.cuda.synchronize()
    execution = {
        "protocol_fingerprint": fingerprint,
        "runtime_environment": runtime,
        "fit_seconds": {
            "stage_a_cva_cpu": cva_fit_seconds,
            "stage_b_fallback_cpu": fallback_fit_seconds,
        },
        "by_split": executions,
        "production_total": _merge_execution(executions),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    aggregate = {
        "schema_version": 2,
        "analysis_status": "sequential_exploratory_development",
        "protocol_fingerprint": fingerprint,
        "dataset_doi": DATASET_DOI,
        "sequential_decision": sequential_decision,
        "selected_method": selected_method,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "calibration": {
            "stage_a": cva_calibration_summary,
            "stage_b_frozen_before_stage_a_audit": fallback_calibration_summary,
        },
        "cuda_validation": validation,
        "gpu_execution": execution,
        "score_archive_schema": {
            "stage_a_cva_scores.npz": list(cva_fields),
            "stage_b_fallback_scores.npz": list(fallback_fields),
            "multi_column_fields_expand_in_declared_order": True,
        },
        "data_boundary": {
            "healthy_121_220_previously_consumed": True,
            "fault_3_9_15_runs_1_10_previously_consumed": True,
            "stage_a_healthy_221_320_consumed": True,
            "stage_a_fault_runs_11_50_consumed": True,
            "stage_b_healthy_321_400_consumed": not bool(stage_a["passed"]),
            "stage_b_fault_runs_51_100_consumed": not bool(stage_a["passed"]),
            "final_healthy_401_500_not_selected_or_scored": True,
            "final_all_faults_401_500_not_selected_or_scored": True,
            "archive_decode_caveat": (
                "pyreadr decodes the complete RData partition before load_runs "
                "returns only explicitly selected rows"
            ),
        },
    }
    _atomic_json(args.output_dir / "stage_b_result.json", stage_b)
    _atomic_json(args.output_dir / "stage_b_status.json", stage_b)
    _atomic_json(args.output_dir / "gpu_execution.json", execution)
    _atomic_json(args.output_dir / "aggregate.json", aggregate)
    _atomic_npz(args.output_dir / "stage_a_cva_scores.npz", cva_scores)
    _atomic_npz(
        args.output_dir / "stage_b_fallback_scores.npz", fallback_scores
    )
    _atomic_text(args.output_dir / "REPORT.md", _report_text(aggregate))
    artifacts = (
        "manifest.json",
        "cuda_validation.json",
        "stage_a_calibration.json",
        "stage_b_calibration.json",
        "fitted_stage_a_cva.pkl",
        "fitted_stage_b_fallback.pkl",
        "stage_a_result.json",
        "stage_b_status.json",
        "gpu_execution.json",
        "aggregate.json",
        "stage_a_cva_scores.npz",
        "stage_b_fallback_scores.npz",
        "REPORT.md",
        "stage_b_result.json",
    )
    _write_completion(
        args.output_dir, fingerprint, sequential_decision, artifacts
    )
    print(f"sequential decision: {sequential_decision}", flush=True)
    print(f"report: {args.output_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
