"""Command-line entry points for the executable MAPIC research prototype."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .baselines import CalibratedPointMonitor
from .confirmation import run_batch_confirmation
from .datasets import BatchDistillationArchive, load_tep
from .experiment import evaluate_natural_missingness, run_stress_test
from .missingness import DEFAULT_SCENARIOS
from .model import MAPIC, MAPICConfig
from .synthetic import make_demo_data


BATCH_NAMES = {
    "labels": "00_Batch_Distillation_Plant_M-202210_Timeseries_Label_Anomaly_Metadata.zip",
    "sensors": "01_Batch_Distillation_Plant_M-202210_Timeseries_Sensors.zip",
    "actuators": "02_Batch_Distillation_Plant_M-202210_Timeseries_Actuators.zip",
    "uncertainties": "03_Batch_Distillation_Plant_M-202210_Timeseries_Sensor_Uncertainties.zip",
}

BATCH_OP028_CONFIRMATION = {
    "normal": (
        "Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/"
        "operating_point_028/train_normal_experiment_001.csv"
    ),
    "tests": (
        "Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/"
        "operating_point_028/test_anormal_experiment_001.csv",
        "Operation/batch_dist_ternary_butan-1-ol+propan-2-ol+water/"
        "operating_point_028/test_anormal_experiment_002.csv",
    ),
}

BATCH_V112_MD5 = {
    "labels": "636dcb8188e5eb80f19522e244704e2c",
    "sensors": "165b3d89c5d483392eeb7e7909edb859",
    "actuators": "58b4648b4c37d5d0dc8a678015e054e7",
    "uncertainties": "7d0a19666294c62a2ba4a5474b03b4e4",
}

BATCH_OP028_V1_FIXED = {
    "design_fraction": 0.65,
    "purge_minutes": 10.0,
    "block_minutes": 5.0,
    "alpha": 0.05,
    "calibration_repeats": 2,
    "seed": 20260907,
}
BATCH_OP028_V1_BASELINES = (
    "median",
    "locf",
    "kalman_mean",
    "linear_offline",
)


def _write_json(data: object, output: Path | None) -> None:
    def safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    text = json.dumps(safe(data), ensure_ascii=False, indent=2, allow_nan=False)
    if output is None:
        print(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {output}")


def _batch_dataset(folder: Path) -> BatchDistillationArchive:
    return BatchDistillationArchive(*(folder / BATCH_NAMES[name] for name in BATCH_NAMES))


def _file_md5(path: Path) -> str:
    checksum = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _verify_batch_archives(folder: Path) -> dict[str, str]:
    actual = {
        name: _file_md5(folder / filename)
        for name, filename in BATCH_NAMES.items()
    }
    mismatches = {
        name: {"expected": BATCH_V112_MD5[name], "actual": digest}
        for name, digest in actual.items()
        if digest != BATCH_V112_MD5[name]
    }
    if mismatches:
        details = ", ".join(
            f"{name}: expected {item['expected']}, got {item['actual']}"
            for name, item in mismatches.items()
        )
        raise ValueError(f"Batch Distillation v1.1.2 checksum mismatch ({details})")
    return actual


def _validate_batch_op028_v1(args: argparse.Namespace) -> None:
    mismatches: list[str] = []
    for name, expected in BATCH_OP028_V1_FIXED.items():
        actual = getattr(args, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r} (required {expected!r})")
    if tuple(args.baselines) != BATCH_OP028_V1_BASELINES:
        mismatches.append(
            f"baselines={tuple(args.baselines)!r} "
            f"(required {BATCH_OP028_V1_BASELINES!r})"
        )
    if mismatches:
        raise ValueError(
            "batch-op028-confirmation-v1 is frozen; " + "; ".join(mismatches)
        )


def command_demo(args: argparse.Namespace) -> None:
    train, calibration, test = make_demo_data(
        args.train_rows, args.calibration_rows, args.test_rows, args.dimensions
    )
    config = MAPICConfig(
        alpha=args.alpha,
        block_size=args.block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    report = run_stress_test(
        train,
        calibration,
        test,
        config,
        DEFAULT_SCENARIOS,
        seeds=(args.seed + 1,),
        baseline_methods=tuple(args.baselines),
    )
    _write_json(report, args.output)


def command_batch_list(args: argparse.Namespace) -> None:
    with _batch_dataset(args.folder) as dataset:
        keys = dataset.list_runs(args.contains)
    for key in keys:
        print(key)
    print(f"runs: {len(keys)}", file=sys.stderr)


def command_batch_run(args: argparse.Namespace) -> None:
    keys = (args.train_key, args.calibration_key, args.test_key)
    if len(set(keys)) != 3:
        raise ValueError("train, calibration, and test keys must be distinct")
    dataset_md5 = _verify_batch_archives(args.folder)
    with _batch_dataset(args.folder) as dataset:
        train = dataset.load_run(
            args.train_key, restore_natural_missingness=True, actuator_mode="features"
        )
        calibration = dataset.load_run(
            args.calibration_key, restore_natural_missingness=True, actuator_mode="features"
        )
        test = dataset.load_run(
            args.test_key, restore_natural_missingness=True, actuator_mode="features"
        )
    if train.fault.any() or calibration.fault.any():
        raise ValueError("train/calibration keys contain nonzero expert labels; choose clean runs")
    if (
        train.feature_names != calibration.feature_names
        or train.feature_names != test.feature_names
    ):
        raise ValueError("batch runs have incompatible process-variable schemas")
    if (
        train.covariate_names != calibration.covariate_names
        or train.covariate_names != test.covariate_names
    ):
        raise ValueError("batch runs have incompatible covariate schemas")
    periods = np.array(
        [train.sample_period_seconds, calibration.sample_period_seconds, test.sample_period_seconds]
    )
    if not np.allclose(periods, periods[0], rtol=0.0, atol=1e-9):
        raise ValueError("train, calibration, and test sample periods must match")
    if not np.isfinite(args.block_minutes) or args.block_minutes <= 0.0:
        raise ValueError("block_minutes must be finite and positive")
    block_size = max(
        10, int(round(args.block_minutes * 60.0 / calibration.sample_period_seconds))
    )
    config = MAPICConfig(
        alpha=args.alpha,
        block_size=block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    train_release = np.asarray(train.metadata["released_values"], dtype=float)
    calibration_release = np.asarray(calibration.metadata["released_values"], dtype=float)
    model = MAPIC(config).fit(
        train_release,
        calibration_release,
        train.covariates,
        calibration.covariates,
        train_observed_mask=train.observed_mask,
        calibration_observed_mask=calibration.observed_mask,
    )
    ablation_reports: dict[str, object] = {}
    if not args.skip_ablation:
        integrated_config = replace(config, integrate_uncertainty=True)
        integrated = MAPIC(integrated_config).fit(
            train_release,
            calibration_release,
            train.covariates,
            calibration.covariates,
            train_observed_mask=train.observed_mask,
            calibration_observed_mask=calibration.observed_mask,
        )
        ablation_reports["MAPIC_integrated_expectation"] = {
            "configuration": asdict(integrated_config),
            "calibration_summary": integrated.calibration_summary_,
            "natural_missingness": evaluate_natural_missingness(integrated, test),
            "note": (
                "independently fitted with M + tr(A P) in score scaling, regime "
                "design, and conformal calibration"
            ),
        }
        mask_only_config = replace(
            config,
            integrate_uncertainty=False,
            use_posterior_quality_context=False,
        )
        mask_only = MAPIC(mask_only_config).fit(
            train_release,
            calibration_release,
            train.covariates,
            calibration.covariates,
            train_observed_mask=train.observed_mask,
            calibration_observed_mask=calibration.observed_mask,
        )
        ablation_reports["UD-MAPIC_mask_only_context"] = {
            "configuration": asdict(mask_only_config),
            "calibration_summary": mask_only.calibration_summary_,
            "natural_missingness": evaluate_natural_missingness(mask_only, test),
            "note": (
                "independently fitted with M retained but posterior trace and U "
                "removed from quality governance"
            ),
        }
    baseline_reports: dict[str, object] = {}
    for method in args.baselines:
        baseline = CalibratedPointMonitor(method, config).fit(
            train_release,
            calibration_release,
            train.covariates,
            calibration.covariates,
            train_observed_mask=train.observed_mask,
            calibration_observed_mask=calibration.observed_mask,
        )
        baseline_reports[method] = {
            "causal": method != "linear_offline",
            "quality_context": "mask_only_without_posterior_covariance",
            "calibration_summary": baseline.calibration_summary_,
            "natural_missingness": evaluate_natural_missingness(baseline, test),
        }
    report = {
        "method": "UD-MAPIC",
        "software_version": __version__,
        "dataset": "Batch Distillation v1.1.2",
        "dataset_doi": "10.5281/zenodo.21535243",
        "dataset_md5": dataset_md5,
        "dataset_checksum_verified": True,
        "train_key": args.train_key,
        "calibration_key": args.calibration_key,
        "test_key": args.test_key,
        "process_variables": len(test.feature_names),
        "sample_period_seconds": {
            "train": train.sample_period_seconds,
            "calibration": calibration.sample_period_seconds,
            "test": test.sample_period_seconds,
        },
        "configuration": asdict(config),
        "cli_protocol": {
            "block_minutes": args.block_minutes,
            "seed": args.seed,
            "calibration_repeats": args.calibration_repeats,
            "actuator_mode": "features",
        },
        "phase1_interpolated_rows_excluded": {
            "train": int(np.asarray(train.metadata["data_interpolated"]).sum()),
            "calibration": int(np.asarray(calibration.metadata["data_interpolated"]).sum()),
        },
        "calibration_summary": model.calibration_summary_,
        "natural_missingness": evaluate_natural_missingness(model, test),
        "ablations": ablation_reports,
        "baselines": baseline_reports,
        "label_protocol": {
            "primary": "visible expert states 2 and 3",
            "secondary": (
                "non-normal episode states 1, 2, and 3, including post-cause recovery"
            ),
        },
    }
    _write_json(report, args.output)


def command_batch_confirm(args: argparse.Namespace) -> None:
    """Run the frozen op028 retrospective protocol in two channel modes."""
    _validate_batch_op028_v1(args)
    dataset_md5 = _verify_batch_archives(args.folder)
    analyses: dict[str, object] = {}
    effective_timing: dict[str, float | int] | None = None
    with _batch_dataset(args.folder) as dataset:
        for label, actuator_mode in (
            ("31_process_channels", "features"),
            ("18_sensors_only", "exclude"),
        ):
            normal = dataset.load_run(
                BATCH_OP028_CONFIRMATION["normal"],
                restore_natural_missingness=True,
                actuator_mode=actuator_mode,
            )
            tests = tuple(
                dataset.load_run(
                    key,
                    restore_natural_missingness=True,
                    actuator_mode=actuator_mode,
                )
                for key in BATCH_OP028_CONFIRMATION["tests"]
            )
            block_size = max(
                10,
                int(round(args.block_minutes * 60.0 / normal.sample_period_seconds)),
            )
            purge_samples = int(
                round(args.purge_minutes * 60.0 / normal.sample_period_seconds)
            )
            timing = {
                "effective_block_samples": block_size,
                "effective_block_minutes": (
                    block_size * normal.sample_period_seconds / 60.0
                ),
                "effective_purge_samples": purge_samples,
                "effective_purge_minutes": (
                    purge_samples * normal.sample_period_seconds / 60.0
                ),
            }
            if effective_timing is None:
                effective_timing = timing
            elif effective_timing != timing:
                raise ValueError("confirmation channel modes have inconsistent timing")
            config = MAPICConfig(
                alpha=args.alpha,
                block_size=block_size,
                calibration_repeats=args.calibration_repeats,
                seed=args.seed,
            )
            analysis = run_batch_confirmation(
                normal,
                tests,
                config,
                train_fraction=args.design_fraction,
                purge_samples=purge_samples,
                baseline_methods=tuple(args.baselines),
            )
            analysis["actuator_mode"] = actuator_mode
            analysis["interpretation"] = (
                "deployment-oriented joint state including actuator readouts"
                if actuator_mode == "features"
                else "sensor-response-only sensitivity analysis"
            )
            analyses[label] = analysis
    assert effective_timing is not None
    report = {
        "protocol": "batch-op028-confirmation-v1",
        "software_version": __version__,
        "status": (
            "frozen retrospective reanalysis after public labels and dataset "
            "documentation were inspected"
        ),
        "dataset": "Batch Distillation v1.1.2",
        "dataset_doi": "10.5281/zenodo.21535243",
        "dataset_md5": dataset_md5,
        "dataset_checksum_verified": True,
        "normal_key": BATCH_OP028_CONFIRMATION["normal"],
        "test_keys": BATCH_OP028_CONFIRMATION["tests"],
        "fixed_protocol": {
            "design_fraction_of_nonpurged_rows": args.design_fraction,
            "purge_minutes": args.purge_minutes,
            "block_minutes": args.block_minutes,
            **effective_timing,
            "alpha": args.alpha,
            "calibration_repeats": args.calibration_repeats,
            "seed": args.seed,
            "baseline_methods": list(BATCH_OP028_V1_BASELINES),
            "channel_modes": ["features", "exclude"],
            "primary_labels": "official CSV states 2 and 3 (visible effect)",
            "secondary_labels": (
                "official CSV states 1, 2, and 3 (non-normal episode including recovery)"
            ),
            "test001_role": "confounded real-world stress case",
            "test002_role": "cleaner confirmed-anomaly case",
        },
        "limitations": [
            "one healthy run is time-split, so calibration is not run-level independent",
            "calibration covers the late batch phase and block exchangeability is approximate",
            "natural gaps are mostly isolated one-second gaps and do not validate long blackouts",
            "the four visible events do not support population-level significance claims",
        ],
        "analyses": analyses,
    }
    _write_json(report, args.output)


def command_tep_run(args: argparse.Namespace) -> None:
    """Run a complete controlled-missingness experiment on one TEP fault."""
    train = load_tep(args.folder, fault_id=0, test=False, split_manipulated=False)
    calibration = load_tep(args.folder, fault_id=0, test=True, split_manipulated=False)
    test = load_tep(args.folder, fault_id=args.fault_id, test=True, split_manipulated=False)
    config = MAPICConfig(
        alpha=args.alpha,
        block_size=args.block_size,
        calibration_repeats=args.calibration_repeats,
        seed=args.seed,
    )
    report = run_stress_test(
        train,
        calibration,
        test,
        config,
        DEFAULT_SCENARIOS,
        seeds=tuple(args.mask_seeds),
        baseline_methods=tuple(args.baselines),
    )
    _write_json(report, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mate-spc", description="MAPIC experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a complete synthetic stress test")
    demo.add_argument("--output", type=Path)
    demo.add_argument("--train-rows", type=int, default=500)
    demo.add_argument("--calibration-rows", type=int, default=1200)
    demo.add_argument("--test-rows", type=int, default=600)
    demo.add_argument("--dimensions", type=int, default=6)
    demo.add_argument("--alpha", type=float, default=0.05)
    demo.add_argument("--block-size", type=int, default=30)
    demo.add_argument("--calibration-repeats", type=int, default=2)
    demo.add_argument("--seed", type=int, default=20260907)
    demo.add_argument(
        "--baselines",
        nargs="*",
        default=["median", "locf", "kalman_mean", "linear_offline"],
        choices=sorted(CalibratedPointMonitor.METHODS),
    )
    demo.set_defaults(func=command_demo)

    batch_list = subparsers.add_parser("batch-list", help="list aligned Operation runs")
    batch_list.add_argument("folder", type=Path)
    batch_list.add_argument("--contains")
    batch_list.set_defaults(func=command_batch_list)

    batch_run = subparsers.add_parser("batch-run", help="run the natural-gap batch case")
    batch_run.add_argument("folder", type=Path)
    batch_run.add_argument("--train-key", required=True)
    batch_run.add_argument("--calibration-key", required=True)
    batch_run.add_argument("--test-key", required=True)
    batch_run.add_argument("--output", type=Path)
    batch_run.add_argument("--alpha", type=float, default=0.05)
    batch_run.add_argument("--block-minutes", type=float, default=1.0)
    batch_run.add_argument("--calibration-repeats", type=int, default=2)
    batch_run.add_argument("--seed", type=int, default=20260907)
    batch_run.add_argument(
        "--skip-ablation",
        action="store_true",
        help="skip the independently fitted posterior-expectation ablation",
    )
    batch_run.add_argument(
        "--baselines",
        nargs="*",
        default=["median", "locf", "kalman_mean", "linear_offline"],
        choices=sorted(CalibratedPointMonitor.METHODS),
    )
    batch_run.set_defaults(func=command_batch_run)

    batch_confirm = subparsers.add_parser(
        "batch-confirm",
        help="run the frozen two-mode op028 retrospective reanalysis",
    )
    batch_confirm.add_argument("folder", type=Path)
    batch_confirm.add_argument("--output", type=Path)
    batch_confirm.add_argument(
        "--design-fraction", type=float, default=BATCH_OP028_V1_FIXED["design_fraction"]
    )
    batch_confirm.add_argument(
        "--purge-minutes", type=float, default=BATCH_OP028_V1_FIXED["purge_minutes"]
    )
    batch_confirm.add_argument(
        "--block-minutes", type=float, default=BATCH_OP028_V1_FIXED["block_minutes"]
    )
    batch_confirm.add_argument("--alpha", type=float, default=BATCH_OP028_V1_FIXED["alpha"])
    batch_confirm.add_argument(
        "--calibration-repeats",
        type=int,
        default=BATCH_OP028_V1_FIXED["calibration_repeats"],
    )
    batch_confirm.add_argument("--seed", type=int, default=BATCH_OP028_V1_FIXED["seed"])
    batch_confirm.add_argument(
        "--baselines",
        nargs="*",
        default=list(BATCH_OP028_V1_BASELINES),
        choices=sorted(CalibratedPointMonitor.METHODS),
    )
    batch_confirm.set_defaults(func=command_batch_confirm)

    tep_run = subparsers.add_parser("tep-run", help="run a TEP missingness stress test")
    tep_run.add_argument("folder", type=Path)
    tep_run.add_argument("--fault-id", type=int, required=True, choices=range(1, 22))
    tep_run.add_argument("--output", type=Path)
    tep_run.add_argument("--alpha", type=float, default=0.05)
    tep_run.add_argument("--block-size", type=int, default=20)
    tep_run.add_argument("--calibration-repeats", type=int, default=2)
    tep_run.add_argument("--seed", type=int, default=20260907)
    tep_run.add_argument("--mask-seeds", nargs="+", type=int, default=[11, 29, 47])
    tep_run.add_argument(
        "--baselines",
        nargs="*",
        default=["median", "locf", "kalman_mean", "linear_offline"],
        choices=sorted(CalibratedPointMonitor.METHODS),
    )
    tep_run.set_defaults(func=command_tep_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
