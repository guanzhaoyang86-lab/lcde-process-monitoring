"""Run a frozen, resumable Rieth multi-run TEP pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Iterable

def _cli_requests_cuda(tokens: list[str]) -> bool:
    """Recognize both separated and equals-style CUDA device options."""
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2].startswith("cuda")
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1].startswith("cuda")
    return False


# The local torch5090 environment requires torch's DLLs to load before NumPy.
# This prescan keeps the ordinary CPU installation free of a torch dependency.
if _cli_requests_cuda(sys.argv[1:]):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc.datasets import RiethTEPArchive
from mate_spc.missingness import DEFAULT_SCENARIOS
from mate_spc.model import MAPICConfig
from mate_spc.multirun import (
    FittedMultiRunSuite,
    aggregate_multirun_reports,
    evaluate_multirun_stress,
    fit_multirun_suite,
    multirun_suite_summary,
    write_multirun_markdown,
)


DEFAULT_DESIGN_RUNS = tuple(range(1, 21))
DEFAULT_CALIBRATION_RUNS = tuple(range(21, 41))
DEFAULT_FAULTS = (1, 3, 9, 15, 19)
DEFAULT_TEST_RUNS = tuple(range(1, 11))
DEFAULT_MASK_SEEDS = (11, 29, 47)
DEFAULT_BASELINES = ("median", "locf", "kalman_mean", "linear_offline")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_safe(value), ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fingerprint(protocol: dict[str, object]) -> str:
    canonical = json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _positive_ids(values: Iterable[int], label: str, maximum: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} must be a nonempty list of unique IDs")
    if min(result) < 1 or max(result) > maximum:
        raise ValueError(f"{label} must lie in 1..{maximum}")
    return result


def _protocol(args: argparse.Namespace, checksums: dict[str, str]) -> dict[str, object]:
    config = MAPICConfig(
        alpha=args.alpha,
        block_size=args.block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    protocol = {
        "schema_version": 1,
        "dataset": "Rieth et al. multi-run Tennessee Eastman Process",
        "dataset_doi": "10.7910/DVN/6C3JR1",
        "dataset_md5": checksums,
        "design_partition": "fault_free_training",
        "calibration_partition": "fault_free_testing",
        "test_partition": "faulty_testing",
        "design_runs": list(args.design_runs),
        "calibration_runs": list(args.calibration_runs),
        "faults": list(args.faults),
        "test_runs": list(args.test_runs),
        "fault_start_zero_based": args.fault_start,
        "split_manipulated": args.split_manipulated,
        "mask_seeds": list(args.mask_seeds),
        "missingness_scenarios": [asdict(item) for item in DEFAULT_SCENARIOS],
        "baselines": list(args.baselines),
        "configuration": asdict(config),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "analysis_unit": "independent process run",
    }
    if args.analysis_status is not None:
        protocol["analysis_status"] = args.analysis_status
    return protocol


def _load_manifest(path: Path, expected: dict[str, object]) -> str:
    expected_fingerprint = _fingerprint(expected)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != expected_fingerprint:
            raise ValueError(
                "existing manifest uses a different frozen protocol; choose a "
                "new output directory"
            )
        return expected_fingerprint
    _atomic_json(
        path,
        {**expected, "protocol_fingerprint": expected_fingerprint},
    )
    return expected_fingerprint


def _suite_matches(
    suite: FittedMultiRunSuite, args: argparse.Namespace
) -> bool:
    expected_design = tuple(
        f"Rieth-fault_free_training-fault-00-run-{run_id:03d}"
        for run_id in args.design_runs
    )
    expected_calibration = tuple(
        f"Rieth-fault_free_testing-fault-00-run-{run_id:03d}"
        for run_id in args.calibration_runs
    )
    return (
        suite.design_run_names == expected_design
        and suite.calibration_run_names == expected_calibration
        and tuple(suite.baselines) == tuple(args.baselines)
    )


def _warm_suite_matches(
    suite: FittedMultiRunSuite, args: argparse.Namespace
) -> bool:
    expected_design = tuple(
        f"Rieth-fault_free_training-fault-00-run-{run_id:03d}"
        for run_id in args.design_runs
    )
    expected_config = MAPICConfig(
        alpha=args.alpha,
        block_size=args.block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    return (
        suite.design_run_names == expected_design
        and tuple(suite.baselines) == tuple(args.baselines)
        and asdict(suite.config) == asdict(expected_config)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen and resumable Rieth multi-run TEP pilot"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--design-runs", type=int, nargs="+", default=DEFAULT_DESIGN_RUNS)
    parser.add_argument(
        "--calibration-runs", type=int, nargs="+", default=DEFAULT_CALIBRATION_RUNS
    )
    parser.add_argument("--faults", type=int, nargs="+", default=DEFAULT_FAULTS)
    parser.add_argument("--test-runs", type=int, nargs="+", default=DEFAULT_TEST_RUNS)
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=DEFAULT_MASK_SEEDS)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--fault-start", type=int, default=160)
    parser.add_argument("--split-manipulated", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-batch-size", type=int, default=32)
    parser.add_argument("--warm-start-suite", type=Path)
    parser.add_argument(
        "--analysis-status",
        choices=("exploratory_development", "confirmatory"),
    )
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.design_runs = _positive_ids(args.design_runs, "design runs", 500)
    args.calibration_runs = _positive_ids(
        args.calibration_runs, "calibration runs", 500
    )
    args.faults = _positive_ids(args.faults, "faults", 20)
    args.test_runs = _positive_ids(args.test_runs, "test runs", 500)
    if not args.mask_seeds or len(args.mask_seeds) != len(set(args.mask_seeds)):
        raise ValueError("mask seeds must be a nonempty list of unique integers")
    args.mask_seeds = tuple(int(value) for value in args.mask_seeds)
    allowed = {"median", "locf", "kalman_mean", "linear_offline"}
    args.baselines = tuple(str(value) for value in args.baselines)
    if not args.baselines or not set(args.baselines) <= allowed:
        raise ValueError(f"baselines must be selected from {sorted(allowed)}")
    if len(args.baselines) != len(set(args.baselines)):
        raise ValueError("baselines must be unique")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    minimum_runs = int(np.ceil(1.0 / args.alpha) - 1)
    if len(args.calibration_runs) < minimum_runs:
        raise ValueError(
            f"alpha={args.alpha} requires at least {minimum_runs} independent "
            "healthy calibration runs"
        )
    if args.block_size < 1 or args.calibration_repeats < 1:
        raise ValueError("block size and calibration repeats must be positive")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0 <= args.fault_start < 960:
        raise ValueError("fault start must lie in 0..959 for faulty testing runs")
    if args.cuda_batch_size < 1:
        raise ValueError("CUDA batch size must be positive")
    if args.device == "cuda" and args.split_manipulated:
        raise ValueError("the CUDA backend currently requires all 52 channels as features")
    if args.device == "cuda" and args.warm_start_suite is None:
        checkpoint = args.output_dir / "fitted_suite.pkl"
        if args.no_resume or not checkpoint.exists():
            raise ValueError(
                "a CUDA fit requires --warm-start-suite with trusted frozen "
                "design parameters when no resumable checkpoint is available"
            )


def _fit_or_load_suite(
    archive: RiethTEPArchive,
    args: argparse.Namespace,
    checkpoint: Path,
    summary_path: Path,
    protocol_fingerprint: str,
) -> FittedMultiRunSuite:
    if checkpoint.exists() and not args.no_resume:
        print(f"loading fitted suite from {checkpoint}", flush=True)
        with checkpoint.open("rb") as stream:
            suite = pickle.load(stream)
        if not isinstance(suite, FittedMultiRunSuite) or not _suite_matches(suite, args):
            raise ValueError("fitted checkpoint does not match the frozen manifest")
        return suite

    if args.device == "cuda":
        from mate_spc.gpu_multirun import (
            cuda_runtime_metadata,
            recalibrate_multirun_suite_cuda,
            validate_cuda_equivalence,
        )

        assert args.warm_start_suite is not None
        print(
            f"loading trusted frozen design suite from {args.warm_start_suite}",
            flush=True,
        )
        with args.warm_start_suite.open("rb") as stream:
            suite = pickle.load(stream)
        if not isinstance(suite, FittedMultiRunSuite) or not _warm_suite_matches(
            suite, args
        ):
            raise ValueError(
                "warm-start suite does not match the frozen design/configuration"
            )
        print(
            f"loading {len(args.calibration_runs)} independent calibration runs",
            flush=True,
        )
        calibration = archive.load_runs(
            "fault_free_testing",
            {0: args.calibration_runs},
            fault_start=args.fault_start,
            split_manipulated=False,
        )
        print("validating NumPy CPU versus CUDA float64", flush=True)
        validation = validate_cuda_equivalence(
            suite,
            calibration[0],
            DEFAULT_SCENARIOS[2],
            device="cuda",
        )
        _atomic_json(args.output_dir / "cuda_validation.json", validation)
        print("CPU/CUDA validation passed; recalibrating on CUDA", flush=True)
        suite, diagnostics = recalibrate_multirun_suite_cuda(
            suite,
            calibration,
            device="cuda",
            batch_size=args.cuda_batch_size,
            progress=lambda message: print(message, flush=True),
        )
        diagnostics["runtime_environment"] = cuda_runtime_metadata("cuda")
        _atomic_pickle(checkpoint, suite)
        _atomic_json(
            args.output_dir / "calibration_diagnostics.json", diagnostics
        )
        _atomic_json(
            summary_path,
            {
                "protocol_fingerprint": protocol_fingerprint,
                "pickle_security_note": (
                    "Load this local checkpoint only from a trusted run."
                ),
                "execution_backend": diagnostics["runtime_environment"],
                **multirun_suite_summary(suite),
            },
        )
        print(f"saved CUDA-recalibrated suite to {checkpoint}", flush=True)
        return suite

    print("loading Phase-I design runs", flush=True)
    design = archive.load_runs(
        "fault_free_training",
        {0: args.design_runs},
        fault_start=args.fault_start,
        split_manipulated=args.split_manipulated,
    )
    print("loading independent Phase-I calibration runs", flush=True)
    calibration = archive.load_runs(
        "fault_free_testing",
        {0: args.calibration_runs},
        fault_start=args.fault_start,
        split_manipulated=args.split_manipulated,
    )
    config = MAPICConfig(
        alpha=args.alpha,
        block_size=args.block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    suite = fit_multirun_suite(
        design,
        calibration,
        config,
        baseline_methods=args.baselines,
        progress=lambda message: print(message, flush=True),
    )
    _atomic_pickle(checkpoint, suite)
    _atomic_json(
        summary_path,
        {
            "protocol_fingerprint": protocol_fingerprint,
            "pickle_security_note": "Load this local checkpoint only from a trusted run.",
            **multirun_suite_summary(suite),
        },
    )
    print(f"saved fitted suite to {checkpoint}", flush=True)
    return suite


def _read_result(
    path: Path,
    fingerprint: str,
    required_backend: str | None = None,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol_fingerprint") != fingerprint:
        raise ValueError(f"result checkpoint has a different protocol: {path}")
    if required_backend is not None:
        backend = report.get("execution_backend")
        if not isinstance(backend, dict) or backend.get("backend") != required_backend:
            raise ValueError(f"result checkpoint has a different backend: {path}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = RiethTEPArchive(args.data_dir, verify_checksums=True)
    checksums = archive.verify(
        ("fault_free_training", "fault_free_testing", "faulty_testing")
    )
    protocol = _protocol(args, checksums)
    fingerprint = _load_manifest(args.output_dir / "manifest.json", protocol)
    suite = _fit_or_load_suite(
        archive,
        args,
        args.output_dir / "fitted_suite.pkl",
        args.output_dir / "fitted_suite.json",
        fingerprint,
    )
    required_backend = (
        "pytorch_cuda_compact_batch" if args.device == "cuda" else None
    )
    gpu_tracker = None
    runtime_environment = None
    if args.device == "cuda":
        from mate_spc.gpu_multirun import (
            CUDAExecutionTracker,
            cuda_runtime_metadata,
            evaluate_multirun_stress_cuda,
        )

        gpu_tracker = CUDAExecutionTracker()
        runtime_environment = cuda_runtime_metadata("cuda")
    if args.fit_only:
        print("fit-only requested; Phase-I checkpoint is ready", flush=True)
        return 0

    expected_paths = {
        (fault_id, run_id): (
            args.output_dir / "runs" / f"fault_{fault_id:02d}_run_{run_id:03d}.json"
        )
        for fault_id in args.faults
        for run_id in args.test_runs
    }
    missing = {
        key: path
        for key, path in expected_paths.items()
        if _read_result(path, fingerprint, required_backend) is None
    }
    if missing:
        selections: dict[int, tuple[int, ...]] = {}
        for fault_id in args.faults:
            run_ids = tuple(
                run_id for run_id in args.test_runs if (fault_id, run_id) in missing
            )
            if run_ids:
                selections[fault_id] = run_ids
        print(f"loading {len(missing)} unfinished faulty runs", flush=True)
        test_runs = archive.load_runs(
            "faulty_testing",
            selections,
            fault_start=args.fault_start,
            split_manipulated=args.split_manipulated,
        )
        for index, test in enumerate(test_runs, start=1):
            fault_id = int(test.metadata["fault_id"])
            run_id = int(test.metadata["simulation_run"])
            path = expected_paths[(fault_id, run_id)]
            print(
                f"evaluating fault {fault_id:02d}, run {run_id:03d} "
                f"({index}/{len(test_runs)})",
                flush=True,
            )
            if args.device == "cuda":
                assert gpu_tracker is not None
                report = evaluate_multirun_stress_cuda(
                    suite,
                    test,
                    DEFAULT_SCENARIOS,
                    seeds=args.mask_seeds,
                    device="cuda",
                    batch_size=args.cuda_batch_size,
                    tracker=gpu_tracker,
                    analysis_status=(
                        args.analysis_status or "exploratory_development"
                    ),
                )
            else:
                report = evaluate_multirun_stress(
                    suite,
                    test,
                    DEFAULT_SCENARIOS,
                    seeds=args.mask_seeds,
                )
            report["protocol_fingerprint"] = fingerprint
            _atomic_json(path, report)
            if args.device == "cuda":
                assert gpu_tracker is not None
                _atomic_json(
                    args.output_dir / "cuda_evaluation_session.json",
                    {
                        "runtime_environment": runtime_environment,
                        "session_execution": gpu_tracker.as_dict(),
                        "completed_run_results_this_session": index,
                    },
                )
    else:
        print("all run checkpoints already exist", flush=True)

    reports = []
    for path in expected_paths.values():
        report = _read_result(path, fingerprint, required_backend)
        if report is None:
            raise RuntimeError(f"expected result is missing: {path}")
        if args.device == "cuda":
            expected_status = args.analysis_status or "exploratory_development"
            if report.get("analysis_status") != expected_status:
                report["analysis_status"] = expected_status
                _atomic_json(path, report)
        reports.append(report)
    aggregate = aggregate_multirun_reports(
        reports,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.device == "cuda":
        aggregate["execution_backend"] = runtime_environment
        aggregate["analysis_status"] = args.analysis_status or "exploratory_development"
        aggregate["calibration_diagnostics_file"] = (
            "calibration_diagnostics.json"
        )
    _atomic_json(args.output_dir / "aggregate.json", aggregate)
    write_multirun_markdown(aggregate, args.output_dir / "REPORT.md")
    print(f"completed {len(reports)} independent fault runs", flush=True)
    print(f"report: {args.output_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
