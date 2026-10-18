"""Run the frozen complete-input log-NIS reflected-EWMA GPU experiment."""

from __future__ import annotations

import argparse
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
from typing import Iterable


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# This Windows CUDA environment must initialize torch DLLs before NumPy/MKL.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc.datasets import RiethTEPArchive, SensorSequence
from mate_spc.gpu_multirun import cuda_runtime_metadata
from mate_spc.gpu_nis import (
    CompleteRunNISResult,
    score_complete_runs_cuda,
    validate_healthy_run_nis_cuda,
)
from mate_spc.metrics import contiguous_regions, event_metrics
from mate_spc.multirun import FittedMultiRunSuite
from mate_spc.nis_ewma import NISEWMAMonitor


DESIGN_RUNS = tuple(range(1, 21))
CALIBRATION_RUNS = tuple(range(21, 121))
AUDIT_RUNS = tuple(range(121, 221))
FAULT_IDS = (3, 9, 15)
FAULT_RUNS = tuple(range(1, 11))
FAULT_START = 160
DATASET_DOI = "10.7910/DVN/6C3JR1"
SOURCE_FILES = (
    "scripts/run_nis_ewma_gpu.py",
    "src/mate_spc/nis_ewma.py",
    "src/mate_spc/gpu_nis.py",
    "src/mate_spc/torch_backend.py",
    "src/mate_spc/state_space.py",
    "src/mate_spc/monitor.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
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
        mode="w", encoding="utf-8", suffix=".json", prefix=f".{path.stem}-",
        dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(
            _clean_json(value), stream, indent=2, ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", prefix=f".{path.stem}-",
        dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=f".{path.stem}-",
        dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen complete-input NIS-EWMA Rieth experiment on CUDA"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-suite", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-only", action="store_true")
    return parser


def _source_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"protocol source file not found: {path}")
        result[relative] = _sha256_file(path)
    return result


def _protocol(
    checksums: dict[str, str],
    suite_sha256: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": "exploratory_development",
        "dataset": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": DATASET_DOI,
        "dataset_md5": checksums,
        "trusted_suite_sha256": suite_sha256,
        "source_sha256": source_hashes,
        "method": {
            "name": "complete_normalized_NIS_log_reflected_EWMA_3of3",
            "nis": "innovation quadratic form divided by 52 observed channels",
            "t0_excluded": True,
            "transformation": "log1p(max(normalized_innovation, 0))",
            "normalization": "design-only median and 1.4826*MAD; SD fallback",
            "ewma": {"lambda": 0.2, "initial": 0.0, "lower_reflection": 0.0},
            "persistence": {"k": 3, "window": 3},
            "alpha": 0.05,
            "calibration_unit": "healthy_run_maximum",
            "conformal_rank_one_based": 96,
            "alarm_comparison": "strictly_greater_than",
        },
        "splits": {
            "design": ["fault_free_training", list(DESIGN_RUNS)],
            "calibration": ["fault_free_testing", list(CALIBRATION_RUNS)],
            "healthy_audit": ["fault_free_testing", list(AUDIT_RUNS)],
            "fault_development": [
                "faulty_testing", list(FAULT_IDS), list(FAULT_RUNS),
            ],
            "fault_validation_reserved": [
                "faulty_testing", list(FAULT_IDS), list(range(11, 51)),
            ],
            "next_healthy_audit_if_failed": [
                "fault_free_testing", list(range(221, 321)),
            ],
            "final_confirmation_reserved": {
                "faults": list(range(1, 21)),
                "runs": list(range(401, 501)),
            },
        },
        "pre_registered_gate": {
            "healthy_alarm_runs_maximum": 9,
            "fault_detected_runs_minimum": 15,
            "per_fault_detected_runs_minimum": 3,
            "all_required": True,
        },
        "forbidden_tuning": [
            "lambda", "VAR_lag", "transform", "initialization",
            "persistence", "alpha", "burn_in",
        ],
    }


def _load_or_create_manifest(path: Path, protocol: dict[str, object]) -> str:
    fingerprint = _fingerprint(protocol)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("protocol_fingerprint") != fingerprint:
            raise ValueError("existing output has a different frozen protocol")
        return fingerprint
    _atomic_json(path, {**protocol, "protocol_fingerprint": fingerprint})
    return fingerprint


def _load_suite(path: Path) -> FittedMultiRunSuite:
    if not path.is_file():
        raise FileNotFoundError(f"trusted fitted suite not found: {path}")
    with path.open("rb") as stream:
        suite = pickle.load(stream)
    if not isinstance(suite, FittedMultiRunSuite):
        raise TypeError("trusted checkpoint is not a FittedMultiRunSuite")
    expected = tuple(
        f"Rieth-fault_free_training-fault-00-run-{run_id:03d}"
        for run_id in DESIGN_RUNS
    )
    if suite.design_run_names != expected:
        raise ValueError("trusted suite does not use design runs 1..20")
    model = suite.proposed.monitor.filter_model
    if (
        int(model.dimension_) != 52
        or int(model.n_exogenous_) != 0
        or int(model.training_run_count_) != 20
    ):
        raise ValueError("trusted suite has incompatible frozen VAR parameters")
    return suite


def _run_ids(
    runs: Iterable[SensorSequence], expected: tuple[int, ...], *, label: str,
) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(int(run.metadata["simulation_run"]) for run in normalized)
    if actual != expected:
        raise RuntimeError(f"{label} IDs differ from the frozen split")
    return normalized


def _fault_run_ids(runs: Iterable[SensorSequence]) -> tuple[SensorSequence, ...]:
    normalized = tuple(runs)
    actual = tuple(
        (int(run.metadata["fault_id"]), int(run.metadata["simulation_run"]))
        for run in normalized
    )
    expected = tuple(
        (fault_id, run_id) for fault_id in FAULT_IDS for run_id in FAULT_RUNS
    )
    if actual != expected:
        raise RuntimeError("fault runs differ from the frozen selection")
    return normalized


def _score_split(
    label: str,
    monitor: object,
    runs: tuple[SensorSequence, ...],
    *,
    device: str,
    batch_size: int,
) -> tuple[CompleteRunNISResult, dict[str, object]]:
    started = time.perf_counter()
    scored = score_complete_runs_cuda(
        monitor, runs, device=device, batch_size=batch_size,
    )
    metadata = {
        "split": label,
        "run_count": len(runs),
        "wall_seconds": time.perf_counter() - started,
        **scored.execution,
    }
    print(
        f"{label}: {len(runs)} runs, {scored.execution['time_steps']} steps, "
        f"{scored.execution['cuda_event_elapsed_seconds']:.3f} CUDA seconds",
        flush=True,
    )
    return scored, metadata


def _merge_execution(parts: Iterable[dict[str, object]]) -> dict[str, object]:
    records = tuple(parts)
    return {
        "backend": records[0]["backend"],
        "device": records[0]["device"],
        "device_name": records[0]["device_name"],
        "dtype": records[0]["dtype"],
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
        "parameter_fitting_backend": "numpy_cpu_frozen_checkpoint",
        "nis_scoring_backend": "pytorch_cuda_compact_batch",
        "decision_backend": "numpy_cpu",
        "tf32": False,
    }


def _finite_maximum(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = np.asarray(values, dtype=float)
    if mask is not None:
        selected = selected[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("no finite eligible statistic")
    return float(np.max(selected))


def _episode_starts(alarm: np.ndarray) -> list[int]:
    return [start for start, _ in contiguous_regions(np.asarray(alarm, dtype=bool))]


def _health_records(
    monitor: NISEWMAMonitor,
    runs: tuple[SensorSequence, ...],
    by_run: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        if np.asarray(run.fault, dtype=bool).any():
            raise ValueError("healthy audit contains a faulty run")
        result = monitor.predict(by_run[run.name])
        starts = _episode_starts(result.alarm)
        maximum = _finite_maximum(result.persistence, result.eligible)
        records.append({
            "run": run.name,
            "simulation_run": int(run.metadata["simulation_run"]),
            "any_alarm": bool(starts),
            "alarm_episode_count": len(starts),
            "first_alarm_index": starts[0] if starts else None,
            "maximum_persistence": maximum,
            "maximum_to_threshold_ratio": maximum / monitor.threshold_,
        })
    return records


def _fault_records(
    monitor: NISEWMAMonitor,
    runs: tuple[SensorSequence, ...],
    by_run: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run in runs:
        fault = np.asarray(run.fault, dtype=bool)
        if contiguous_regions(fault) != [(FAULT_START, run.x.shape[0])]:
            raise RuntimeError(f"unexpected fault interval for {run.name}")
        result = monitor.predict(by_run[run.name])
        metrics = event_metrics(
            result.alarm, fault, run.sample_period_seconds,
            warning_horizon=0, refractory=0,
        ).as_dict()
        starts = _episode_starts(result.alarm)
        post_starts = [
            index for index in starts if FAULT_START <= index < run.x.shape[0]
        ]
        first = post_starts[0] if post_starts else None
        maximum = _finite_maximum(result.persistence, fault & result.eligible)
        record = {
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
            "pre_fault_alarm_episode_count": int(metrics["false_alarm_episodes"]),
            "maximum_post_fault_persistence": maximum,
            "maximum_to_threshold_ratio": maximum / monitor.threshold_,
            "monitoring": metrics,
        }
        if bool(metrics["events_detected"]) != bool(post_starts):
            raise RuntimeError("event metric and episode-onset result disagree")
        records.append(record)
    return records


def _wilson(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "proportion": p,
        "lower_95": max(0.0, center - half),
        "upper_95": min(1.0, center + half),
    }


def _binomial_upper_tail(total: int, probability: float, at_least: int) -> float:
    return float(sum(
        math.comb(total, count) * probability**count
        * (1.0 - probability) ** (total - count)
        for count in range(at_least, total + 1)
    ))


def _fault_aggregate(
    records: list[dict[str, object]], fault_id: int | None = None,
) -> dict[str, object]:
    selected = [
        row for row in records
        if fault_id is None or int(row["fault_id"]) == fault_id
    ]
    detected = sum(bool(row["detected"]) for row in selected)
    delays = [
        int(row["delay_samples"]) for row in selected
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


def _decision_validation(
    monitor: NISEWMAMonitor,
    filter_model: object,
    run: SensorSequence,
    gpu_nis: np.ndarray,
) -> dict[str, object]:
    cpu_nis = np.maximum(
        np.asarray(
            filter_model.filter(
                run.x, np.ones_like(run.x, dtype=bool)
            ).normalized_innovation,
            dtype=float,
        ),
        0.0,
    )
    cpu = monitor.predict(cpu_nis)
    gpu = monitor.predict(gpu_nis)
    fields: dict[str, object] = {}
    for name in ("transformed", "standardized", "ewma", "persistence"):
        left = np.asarray(getattr(cpu, name), dtype=float)
        right = np.asarray(getattr(gpu, name), dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        fields[name] = {
            "maximum_absolute": (
                float(np.max(np.abs(left[finite] - right[finite])))
                if finite.any() else 0.0
            ),
            "finite_pattern_exact": bool(
                np.array_equal(np.isfinite(left), np.isfinite(right))
            ),
        }
    result = {
        "run": run.name,
        "fields": fields,
        "alarm_exact": bool(np.array_equal(cpu.alarm, gpu.alarm)),
    }
    if not result["alarm_exact"] or any(
        not bool(item["finite_pattern_exact"])
        or float(item["maximum_absolute"]) > 1e-9
        for item in fields.values()
    ):
        raise RuntimeError(f"CPU/CUDA decision path disagrees: {result}")
    return result


def _write_report(path: Path, aggregate: dict[str, object]) -> None:
    calibration = aggregate["calibration"]
    audit = aggregate["healthy_audit"]
    fault = aggregate["fault_development"]
    gate = aggregate["pre_registered_gate_result"]
    execution = aggregate["gpu_execution"]["production_total"]
    lines = [
        "# Complete-input log-NIS reflected-EWMA GPU development experiment",
        "",
        f"Protocol fingerprint: {aggregate['protocol_fingerprint']}.",
        "",
        "This is exploratory development. Healthy runs 121-220 were consumed "
        "once as the frozen false-alarm audit; final runs 401-500 remain untouched.",
        "",
        "## Calibration",
        "",
        f"- Design: {calibration['design_run_count']} healthy runs and "
        f"{calibration['design_effective_samples']} effective t>=1 samples.",
        f"- Center / scale: {calibration['center']:.8f} / "
        f"{calibration['scale']:.8f} ({calibration['scale_estimator']}).",
        f"- 100-run alpha=0.05 conformal threshold: "
        f"{calibration['threshold']:.8f}, rank 96.",
        f"- Rank driver: {calibration['rank_driver']['run']}.",
        "",
        "## Pre-registered gate",
        "",
        "| Gate | Result | Required | Pass |",
        "|---|---:|---:|---:|",
        f"| Healthy audit runs with any alarm | "
        f"{audit['runs_with_any_alarm']}/100 | <=9 | "
        f"{str(gate['healthy_gate_passed']).lower()} |",
        f"| Fault runs detected | {fault['overall']['detected_runs']}/30 | >=15 | "
        f"{str(gate['fault_total_gate_passed']).lower()} |",
        f"| Minimum detected within each fault | "
        f"{gate['minimum_detected_runs_across_faults']}/10 | >=3 | "
        f"{str(gate['per_fault_gate_passed']).lower()} |",
        "",
        f"Overall decision: {gate['decision']}",
        "",
        "## Fault results",
        "",
        "| Fault | Detected | Conditional median delay | "
        "Restricted mean delay (miss=800) | Pre-fault alarm runs |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fault_id in FAULT_IDS:
        item = fault["by_fault"][f"{fault_id:02d}"]
        delay = item["conditional_median_delay_samples"]
        delay_text = "-" if delay is None else f"{delay:.1f}"
        lines.append(
            f"| {fault_id:02d} | {item['detected_runs']}/10 | {delay_text} | "
            f"{item['restricted_mean_delay_samples_miss_as_800']:.1f} | "
            f"{item['runs_with_pre_fault_alarm']}/10 |"
        )
    lines.extend([
        "",
        "Only an alarm episode whose onset is at or after sample 160 counts as "
        "detection. A pre-fault episode persisting into the fault is not credited.",
        "",
        "## GPU execution",
        "",
        f"- Device: {execution['device_name']}; dtype: {execution['dtype']}.",
        f"- Production: {execution['sequences']} sequences, "
        f"{execution['time_steps']} sequence-steps, "
        f"{execution['kernel_calls']} CUDA batch calls.",
        f"- CUDA-event time: "
        f"{execution['cuda_event_elapsed_seconds']:.3f} seconds.",
        f"- Factorization failures / fallback sequences: "
        f"{execution['factorization_failures']} / "
        f"{execution['fallback_sequences']}.",
        f"- CPU/CUDA validation passed: "
        f"{str(aggregate['cuda_validation']['passed']).lower()}.",
        "",
        "Fitting, normalization, conformal ranking, metrics, and reporting use "
        "CPU. Repeated posterior/NIS scoring uses CUDA.",
        "",
        "## Frozen next action",
        "",
        gate["next_action"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = RiethTEPArchive(args.data_dir, verify_checksums=True)
    checksums = archive.verify(
        ("fault_free_training", "fault_free_testing", "faulty_testing")
    )
    protocol = _protocol(
        checksums,
        _sha256_file(args.trusted_suite),
        _source_hashes(project_root),
    )
    fingerprint = _load_or_create_manifest(
        args.output_dir / "manifest.json", protocol,
    )
    aggregate_path = args.output_dir / "aggregate.json"
    if aggregate_path.exists() and not args.calibration_only:
        existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") == fingerprint:
            print("frozen NIS-EWMA experiment is already complete", flush=True)
            return 0

    runtime = cuda_runtime_metadata(args.device)
    suite = _load_suite(args.trusted_suite)
    filter_monitor = suite.proposed.monitor
    print("loading healthy design runs 1..20", flush=True)
    design = _run_ids(
        archive.load_runs(
            "fault_free_training", {0: DESIGN_RUNS},
            fault_start=FAULT_START, split_manipulated=False,
        ),
        DESIGN_RUNS,
        label="design",
    )
    testing_ids = (
        CALIBRATION_RUNS
        if args.calibration_only
        else CALIBRATION_RUNS + AUDIT_RUNS
    )
    print(
        f"loading fault-free testing runs {testing_ids[0]}..{testing_ids[-1]}",
        flush=True,
    )
    healthy_testing = _run_ids(
        archive.load_runs(
            "fault_free_testing", {0: testing_ids},
            fault_start=FAULT_START, split_manipulated=False,
        ),
        testing_ids,
        label="fault-free testing",
    )
    calibration = healthy_testing[: len(CALIBRATION_RUNS)]
    audit = (
        healthy_testing[len(CALIBRATION_RUNS):]
        if not args.calibration_only else ()
    )

    import torch

    torch.cuda.reset_peak_memory_stats()
    validation_design = validate_healthy_run_nis_cuda(
        filter_monitor, design[0], device=args.device,
    )
    validation_calibration = validate_healthy_run_nis_cuda(
        filter_monitor, calibration[0], device=args.device,
    )
    design_scored, design_execution = _score_split(
        "design", filter_monitor, design,
        device=args.device, batch_size=args.batch_size,
    )
    calibration_scored, calibration_execution = _score_split(
        "calibration", filter_monitor, calibration,
        device=args.device, batch_size=args.batch_size,
    )
    single = score_complete_runs_cuda(
        filter_monitor, (calibration[0],), device=args.device, batch_size=1,
    )
    batch_position_error = float(np.max(np.abs(
        single.by_run[calibration[0].name]
        - calibration_scored.by_run[calibration[0].name]
    )))
    if batch_position_error > 1e-12:
        raise RuntimeError("NIS differs between batch size 1 and production batch")

    nis_monitor = NISEWMAMonitor().fit(
        (
            (run.name, design_scored.by_run[run.name])
            for run in design
        ),
        (
            (run.name, calibration_scored.by_run[run.name])
            for run in calibration
        ),
    )
    decision_validation = _decision_validation(
        nis_monitor,
        filter_monitor.filter_model,
        calibration[0],
        calibration_scored.by_run[calibration[0].name],
    )
    validation = {
        "passed": True,
        "protocol_fingerprint": fingerprint,
        "design_run": validation_design,
        "calibration_run": validation_calibration,
        "batch_size_1_vs_production_maximum_absolute": batch_position_error,
        "decision_path": decision_validation,
    }
    calibration_summary = {
        **nis_monitor.calibration_summary_,
        "design_effective_samples": sum(run.x.shape[0] - 1 for run in design),
        "protocol_fingerprint": fingerprint,
    }
    calibration_summary["calibration_runs_above_threshold"] = sum(
        item["maximum"] > nis_monitor.threshold_
        for item in calibration_summary["run_maxima"]
    )
    _atomic_json(args.output_dir / "cuda_validation.json", validation)
    _atomic_json(args.output_dir / "calibration.json", calibration_summary)
    _atomic_pickle(args.output_dir / "fitted_nis_ewma.pkl", nis_monitor)

    executions = [design_execution, calibration_execution]
    score_arrays = {
        **{
            f"design_{int(run.metadata['simulation_run']):03d}":
            design_scored.by_run[run.name]
            for run in design
        },
        **{
            f"calibration_{int(run.metadata['simulation_run']):03d}":
            calibration_scored.by_run[run.name]
            for run in calibration
        },
    }
    if args.calibration_only:
        torch.cuda.synchronize()
        execution = {
            "protocol_fingerprint": fingerprint,
            "runtime_environment": runtime,
            "by_split": executions,
            "production_total": _merge_execution(executions),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "calibration_only": True,
        }
        _atomic_json(args.output_dir / "gpu_execution.json", execution)
        _atomic_npz(args.output_dir / "nis_scores.npz", score_arrays)
        print(f"calibration threshold={nis_monitor.threshold_:.8f}", flush=True)
        return 0

    audit = _run_ids(audit, AUDIT_RUNS, label="healthy audit")
    audit_scored, audit_execution = _score_split(
        "healthy_audit", filter_monitor, audit,
        device=args.device, batch_size=args.batch_size,
    )
    executions.append(audit_execution)
    audit_records = _health_records(nis_monitor, audit, audit_scored.by_run)
    for run in audit:
        score_arrays[
            f"audit_{int(run.metadata['simulation_run']):03d}"
        ] = audit_scored.by_run[run.name]

    del healthy_testing, audit_scored
    gc.collect()
    print("loading faults 3/9/15, runs 1..10", flush=True)
    fault_runs = _fault_run_ids(archive.load_runs(
        "faulty_testing",
        {fault_id: FAULT_RUNS for fault_id in FAULT_IDS},
        fault_start=FAULT_START,
        split_manipulated=False,
    ))
    fault_scored, fault_execution = _score_split(
        "fault_development", filter_monitor, fault_runs,
        device=args.device, batch_size=args.batch_size,
    )
    executions.append(fault_execution)
    fault_records = _fault_records(
        nis_monitor, fault_runs, fault_scored.by_run,
    )
    for run in fault_runs:
        score_arrays[
            f"fault_{int(run.metadata['fault_id']):02d}_"
            f"run_{int(run.metadata['simulation_run']):03d}"
        ] = fault_scored.by_run[run.name]

    audit_alarm_runs = sum(bool(row["any_alarm"]) for row in audit_records)
    audit_summary = {
        "run_count": len(audit_records),
        "runs_with_any_alarm": audit_alarm_runs,
        "run_alarm_rate": _wilson(audit_alarm_runs, len(audit_records)),
        "alarm_episodes": sum(
            int(row["alarm_episode_count"]) for row in audit_records
        ),
        "gate_maximum_alarm_runs": 9,
        "binomial_p_0.05_tail_for_10_or_more": (
            _binomial_upper_tail(100, 0.05, 10)
        ),
        "records": audit_records,
    }
    overall_fault = _fault_aggregate(fault_records)
    by_fault = {
        f"{fault_id:02d}": _fault_aggregate(fault_records, fault_id)
        for fault_id in FAULT_IDS
    }
    by_fault_counts = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"])
        for fault_id in FAULT_IDS
    ]
    healthy_gate = audit_alarm_runs <= 9
    total_gate = int(overall_fault["detected_runs"]) >= 15
    per_fault_gate = min(by_fault_counts) >= 3
    passed = healthy_gate and total_gate and per_fault_gate
    if passed:
        decision = "PASS_ALL_PRE_REGISTERED_GATES"
        next_action = (
            "Freeze the NIS channel, integrate it with the missing-aware monitor, "
            "derive mask seeds by run while keeping methods paired, and validate "
            "once on fault 3/9/15 runs 11-50. Keep runs 401-500 untouched."
        )
    else:
        decision = "FAIL_PRE_REGISTERED_GATE"
        next_action = (
            "Do not lower the threshold or tune lambda/window. Stop the NIS-only "
            "branch and move to a richer CVA or higher-order dynamic detector. "
            "Use healthy runs 221-320 for its audit; keep runs 401-500 untouched."
        )
    gate = {
        "decision": decision,
        "passed": passed,
        "healthy_gate_passed": healthy_gate,
        "fault_total_gate_passed": total_gate,
        "per_fault_gate_passed": per_fault_gate,
        "minimum_detected_runs_across_faults": min(by_fault_counts),
        "next_action": next_action,
    }

    torch.cuda.synchronize()
    execution = {
        "protocol_fingerprint": fingerprint,
        "runtime_environment": runtime,
        "by_split": executions,
        "production_total": _merge_execution(executions),
        "validation_execution": {
            "design_run": validation_design["cuda_execution"],
            "calibration_run": validation_calibration["cuda_execution"],
            "single_calibration_run": single.execution,
        },
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "calibration_only": False,
    }
    aggregate = {
        "schema_version": 1,
        "analysis_status": "exploratory_development",
        "protocol_fingerprint": fingerprint,
        "dataset_doi": DATASET_DOI,
        "method": "complete_normalized_NIS_log_reflected_EWMA_3of3",
        "calibration": calibration_summary,
        "healthy_audit": audit_summary,
        "fault_development": {
            "run_count": len(fault_records),
            "overall": overall_fault,
            "by_fault": by_fault,
            "records": fault_records,
        },
        "pre_registered_gate_result": gate,
        "cuda_validation": validation,
        "gpu_execution": execution,
        "data_boundary": {
            "healthy_audit_121_220_consumed": True,
            "fault_3_9_15_runs_1_10_development_only": True,
            "healthy_and_fault_runs_401_500_untouched": True,
        },
    }
    _atomic_json(args.output_dir / "healthy_audit.json", audit_summary)
    _atomic_json(
        args.output_dir / "fault_development.json",
        aggregate["fault_development"],
    )
    _atomic_json(args.output_dir / "gpu_execution.json", execution)
    _atomic_json(args.output_dir / "aggregate.json", aggregate)
    _atomic_npz(args.output_dir / "nis_scores.npz", score_arrays)
    _write_report(args.output_dir / "REPORT.md", aggregate)
    print(f"decision: {decision}", flush=True)
    print(f"report: {args.output_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
