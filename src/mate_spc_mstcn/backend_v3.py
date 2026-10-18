"""Governed CUDA numeric backend for the MSCD-TCN-SPC v3 experiment.

Torch is imported before NumPy/HDF5.  Each complete access plan is validated
before the archive is opened, and all fold preprocessors/monitors are fitted
from training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

if "torch" in sys.modules:
    _torch = sys.modules["torch"]
elif "numpy" in sys.modules:  # pragma: no cover - import-order regression
    _torch = None
else:
    try:  # pragma: no cover - host dependent
        import torch as _torch
    except (ImportError, OSError):  # pragma: no cover - host dependent
        _torch = None

import numpy as np

from .model import MSTCNConfig, MSTCNModel, MSTCNPath, fit_mstcn_cuda, load_mstcn_model
from .monitor import (
    FittedTopTailMonitor,
    causal_fault_attribution,
    first_persistent_exceedance,
    strict_order_statistic,
)


ALL_IDVS = (1, 3, 9, 15)
FAULT_IDVS = (3, 9, 15)
RUNS = tuple(range(1, 101))
SEQUENCE_LENGTH = 500
PROCESS_FEATURE_COUNT = 52
MONITOR_ENDPOINTS = np.arange(160, 500, dtype=np.int64)
SCAN_WINDOWS = (16, 32, 64, 128)
Q_VALUES = (0.125, 0.25)
RHO_VALUES = (0.125, 0.25)
FOLD_COUNT = 5
FOLD_SALT = "mstcn-spc-v3-mode1-mag50-block-folds-2026-09-08"
PARITY_ATOL = 1.0e-9
FLOAT_TOLERANCE = 1.0e-12
TRAINING_THRESHOLD_RANK = 79
MISS_DELAY = 340
BACKEND_SCHEMA = "mstcn_spc_backend_v3"
MODEL_METADATA_SCHEMA = "mstcn_spc_bound_model_metadata_v1"


PROCESSDATA_LABELS = (
    "Time", "A Feed", "D Feed", "E Feed", "A and C Feed", "Recycle Flow",
    "Reactor Feed Rate", "Reactor Pressure", "Reactor Level", "Reactor Temperature",
    "Purge Rate", "Product Sep Temp", "Product Sep Level", "Product Sep Pressure",
    "Product Sep Underflow", "Stripper Level", "Stripper Pressure",
    "Stripper Underflow", "Stripper Temp", "Stripper Steam Flow", "Compressor Work",
    "Reactor Coolant Temp", "Separator Coolant Temp", "Component A to Reactor",
    "Component B to Reactor", "Component C to Reactor", "Component D to Reactor",
    "Component E to Reactor", "Component F to Reactor", "Component A in Purge",
    "Component B in Purge", "Component C in Purge", "Component D in Purge",
    "Component E in Purge", "Component F in Purge", "Component G in Purge",
    "Component H in Purge", "Component D in Product", "Component E in Product",
    "Component F in Product", "Component G in Product", "Component H in Product",
    "D feed", "E Feed", "A Feed", "A and C Feed", "Recycle", "Purge", "Separator",
    "Stripper", "Steam", "Reactor Coolant", "Condenser Coolant", "Agitator",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _plain_json(value: object) -> Any:
    return json.loads(_canonical_json(value))


def _config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _require_cuda_float64(*, device: str, dtype: str) -> None:
    if device != "cuda" or dtype != "float64":
        raise ValueError("v3 numeric phases require device='cuda', dtype='float64'")
    if _torch is None:
        raise RuntimeError("PyTorch must be imported before NumPy in a fresh process")
    if not _torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")


def cuda_preflight() -> Mapping[str, Any]:
    _require_cuda_float64(device="cuda", dtype="float64")
    properties = _torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_type": "cuda",
        "dtype": "float64",
        "torch_version": str(_torch.__version__),
        "cuda_version": str(_torch.version.cuda),
        "device_name": str(properties.name),
    }


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"q{int(q * 1000):03d}-rho{int(rho * 1000):03d}",
            "mil_top_fraction_q": q,
            "scan_tail_fraction_rho": rho,
        }
        for q in Q_VALUES
        for rho in RHO_VALUES
    ]


def _expected_config() -> dict[str, Any]:
    return {
        "schema": "mstcn_spc_complete_model_config",
        "schema_version": 1,
        "feature_transform": {
            "feature_schema": "mstcn_causal_haar_loop_v2", "input_features": 512,
            "commissioning_baseline_half_open": [0, 128],
            "fit_scope": "fold_training_run_prefixes_only", "causal_no_future_rows": True,
        },
        "architecture": {
            "window_length": 128, "channels": 32, "kernel_size": 3,
            "dilations": [1, 2, 4, 8, 16], "pool_scales": [8, 32, 96],
            "pooling": "TCN_endpoint_state_plus_trailing_mean_RMS_of_1x1_pointwise_projection",
            "hidden_size": 64, "detector_output_count": 1,
            "attribution_class_ids": [3, 9, 15],
        },
        "endpoints": {
            "dense_half_open_step": [160, 500, 1],
            "mil_half_open_step": [160, 500, 4],
        },
        "training": {
            "seed": 20260908, "optimizer": "AdamW", "learning_rate": 0.001,
            "weight_decay": 0.0001, "epochs": 30, "bag_batch_size": 8,
            "class_balanced_runs_per_class_per_batch": 2,
            "examples_per_class_per_epoch_cap": 100,
            "fold_training_runs_per_class": 80, "final_refit_runs_per_class": 100,
            "gradient_clip_norm": 1.0, "healthy_detection_loss_weight": 0.5,
            "each_fault_detection_loss_weight": 1.0 / 6.0,
            "attribution_loss_weight": 0.25,
            "mil_top_fractions_q": list(Q_VALUES), "models_per_q_per_fold": 1,
            "validation_early_stopping": False, "device": "cuda", "dtype": "float64",
        },
        "scan": {
            "evidence": "detector_logit", "window_samples_W": list(SCAN_WINDOWS),
            "tail_fractions_rho": list(RHO_VALUES),
            "pointwise_standardizer": "robust_healthy_training_only",
            "per_window_studentizer": "robust_healthy_training_only",
            "aggregation": "maximum_over_W_of_trailing_top_rho_studentized_mean",
            "persistence_samples": 2, "threshold_comparison": "strictly_greater_than",
            "attribution": "argmax_cumulative_three_class_logits_through_first_alarm",
        },
        "candidate_grid": _candidate_grid(),
        "selection_order": [
            "maximize_minimum_fault_detection_rate",
            "maximize_minimum_fold_by_fault_detection_rate",
            "minimize_worst_fault_nearest_rank_p90_delay",
            "minimize_worst_fault_median_delay", "minimize_oof_healthy_alarm_runs",
            "prefer_larger_rho", "prefer_larger_q", "canonical_candidate_id",
        ],
    }


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    try:
        normalized = _plain_json(config)
    except (TypeError, ValueError) as error:
        raise ValueError("config must be finite JSON") from error
    if _canonical_json(normalized) != _canonical_json(_expected_config()):
        raise ValueError("config differs from the exact audited v3 specification")
    return normalized


def _canonical_fold_map() -> dict[int, int]:
    ranked = sorted(
        RUNS,
        key=lambda run: hashlib.sha256(f"{FOLD_SALT}|{run}".encode("ascii")).hexdigest(),
    )
    return dict(sorted((run, index % FOLD_COUNT) for index, run in enumerate(ranked)))


def _validate_fold_map(fold_map: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(fold_map, Mapping):
        raise ValueError("fold_map must be a mapping")
    if any(type(key) is not int or type(value) is not int for key, value in fold_map.items()):
        raise ValueError("fold_map must contain exact integer identities")
    normalized = dict(fold_map)
    if normalized != _canonical_fold_map():
        raise ValueError("fold_map differs from the canonical salted five-fold map")
    return normalized


def _plans() -> dict[str, tuple[dict[str, Any], ...]]:
    development: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for run in RUNS:
        development.append({
            "idv": 1, "magnitude_percent": 50, "simulation_run": run,
            "window_half_open": [100, 600], "role": "healthy_development",
        })
        development.extend({
            "idv": fault, "magnitude_percent": 50, "simulation_run": run,
            "window_half_open": [440, 940], "role": f"fault-{fault}_development",
        } for fault in FAULT_IDVS)
        confirmation.append({
            "idv": 1, "magnitude_percent": 75, "simulation_run": run,
            "window_half_open": [100, 600], "role": "healthy_confirmation",
        })
        confirmation.extend({
            "idv": fault, "magnitude_percent": 75, "simulation_run": run,
            "window_half_open": [440, 940], "role": f"fault-{fault}_confirmation",
        } for fault in FAULT_IDVS)
    calibration = tuple({
        "idv": 1, "magnitude_percent": 25, "simulation_run": run,
        "window_half_open": [100, 600], "role": "healthy_calibration_prefault_only",
    } for run in RUNS)
    return {
        "development": tuple(development),
        "calibrate": calibration,
        "confirm": tuple(confirmation),
    }


_PLANS = _plans()


def _validate_access_plan(
    access_plan: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[dict[str, Any], ...]]:
    if isinstance(access_plan, (str, bytes)) or not isinstance(access_plan, Sequence):
        raise TypeError("access_plan must be a sequence")
    required = {"idv", "magnitude_percent", "simulation_run", "window_half_open", "role"}
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(access_plan):
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"access_plan[{index}] fields differ from v3")
        window = raw["window_half_open"]
        if isinstance(window, (str, bytes)) or not isinstance(window, Sequence) or len(window) != 2:
            raise ValueError(f"access_plan[{index}] window is invalid")
        numeric = (raw["idv"], raw["magnitude_percent"], raw["simulation_run"], *window)
        if any(type(value) is not int for value in numeric):
            raise ValueError(f"access_plan[{index}] numeric fields must be exact integers")
        normalized.append({
            "idv": raw["idv"],
            "magnitude_percent": raw["magnitude_percent"],
            "simulation_run": raw["simulation_run"],
            "window_half_open": [window[0], window[1]],
            "role": str(raw["role"]),
        })
    candidate = tuple(normalized)
    matches = [phase for phase, expected in _PLANS.items() if candidate == expected]
    if len(matches) != 1:
        raise ValueError("access plan is not one exact canonical v3 phase")
    return matches[0], candidate


@dataclass(frozen=True)
class NumericRun:
    idv: int
    simulation_run: int
    magnitude_percent: int
    rng_seed: int
    x: np.ndarray

    def __post_init__(self) -> None:
        if any(
            type(value) is not int
            for value in (self.idv, self.simulation_run, self.magnitude_percent, self.rng_seed)
        ):
            raise ValueError("numeric run identity fields must be exact integers")
        if self.idv not in ALL_IDVS or self.simulation_run not in RUNS:
            raise ValueError("numeric run identity is outside the v3 allocation")
        if self.magnitude_percent not in (25, 50, 75) or self.rng_seed < 1:
            raise ValueError("numeric run magnitude/seed is invalid")
        values = np.asarray(self.x)
        if (
            values.shape != (SEQUENCE_LENGTH, PROCESS_FEATURE_COUNT)
            or values.dtype != np.dtype(np.float64)
            or not np.isfinite(values).all()
        ):
            raise ValueError("numeric run must be finite float64 (500,52)")
        values.setflags(write=False)
        object.__setattr__(self, "x", values)


def _require_h5py() -> Any:
    try:
        import h5py
    except (ImportError, OSError) as error:  # pragma: no cover - host dependent
        raise RuntimeError("Extended TEP loading requires h5py") from error
    return h5py


def _decode_labels(handle: Any, h5py: Any) -> None:
    if "Processdata_Labels" not in handle or not isinstance(
        handle["Processdata_Labels"], h5py.Dataset
    ):
        raise ValueError("HDF5 root lacks the exact Processdata_Labels dataset")
    raw = np.asarray(handle["Processdata_Labels"][...]).reshape(-1)
    labels = tuple(
        value.decode("utf-8", errors="strict")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in raw.tolist()
    )
    if labels != PROCESSDATA_LABELS:
        raise ValueError("Processdata_Labels differ from the official sequence")


def _materialize_row(handle: Any, h5py: Any, row: Mapping[str, Any]) -> NumericRun:
    idv = int(row["idv"])
    run = int(row["simulation_run"])
    magnitude = int(row["magnitude_percent"])
    paths = {
        status: (
            f"Mode1/SingleFault/{status}/IDV{idv}/"
            f"Mode1_IDVInfo_{idv}_{magnitude}/Run{run}"
        )
        for status in ("SimulationCompleted", "SimulationStopped")
    }
    observed = tuple(status for status, path in paths.items() if path in handle)
    if observed != ("SimulationCompleted",):
        raise ValueError(f"IDV{idv} magnitude{magnitude} Run{run} is not uniquely completed")
    group_path = paths["SimulationCompleted"]
    group = handle[group_path]
    if not isinstance(group, h5py.Group) or "seed" not in group.attrs:
        raise ValueError(f"{group_path} is not a seeded run group")
    seed_raw = np.asarray(group.attrs["seed"])
    if seed_raw.shape != (1,) or seed_raw.dtype.kind not in "iuf":
        raise ValueError(f"{group_path} seed attribute is malformed")
    seed_float = float(seed_raw[0])
    if not math.isfinite(seed_float) or seed_float <= 0.0 or not seed_float.is_integer():
        raise ValueError(f"{group_path} seed must be a positive integer")

    process_path = f"{group_path}/processdata"
    init_path = f"{group_path}/idv_init"
    if (
        process_path not in handle
        or init_path not in handle
        or not isinstance(handle[process_path], h5py.Dataset)
        or not isinstance(handle[init_path], h5py.Dataset)
    ):
        raise ValueError(f"{group_path} lacks exact processdata/idv_init datasets")
    process = handle[process_path]
    init = handle[init_path]
    if process.shape != (2000, 54) or np.dtype(process.dtype) != np.dtype(np.float64):
        raise ValueError(f"{process_path} metadata changed")
    if init.shape != (3, 28) or np.dtype(init.dtype) != np.dtype(np.float64):
        raise ValueError(f"{init_path} metadata changed")
    init_values = np.asarray(init[...])
    expected_init = np.zeros((3, 28), dtype=np.float64)
    expected_init[:, idv - 1] = (0.0, magnitude / 100.0, 30.0)
    if not np.array_equal(init_values, expected_init):
        raise ValueError(f"{init_path} values changed")

    start, stop = (int(value) for value in row["window_half_open"])
    values = np.array(process[start:stop, 1:53], dtype=np.float64, copy=True, order="C")
    times = np.asarray(process[start:stop, 0], dtype=np.float64)
    expected_times = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
    if (
        values.shape != (500, 52)
        or not np.isfinite(values).all()
        or times.shape != (500,)
        or not np.isfinite(times).all()
        or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-9)
    ):
        raise ValueError(f"{process_path} authorized numeric window changed")
    return NumericRun(idv, run, magnitude, int(seed_float), values)


def load_numeric_runs(
    source_path: str | Path,
    access_plan: Sequence[Mapping[str, Any]],
) -> tuple[NumericRun, ...]:
    """Validate the complete plan first, then read only authorized slices."""

    _, plan = _validate_access_plan(access_plan)
    source = Path(source_path)
    if source.name != "TEP_Mode1.h5":
        raise ValueError("source file must be the exact TEP_Mode1.h5 name")
    h5py = _require_h5py()
    runs: list[NumericRun] = []
    with h5py.File(source, "r") as handle:
        _decode_labels(handle, h5py)
        for row in plan:
            runs.append(_materialize_row(handle, h5py, row))
    identities = [
        (item.magnitude_percent, item.idv, item.simulation_run, item.rng_seed)
        for item in runs
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("materialized run identities are not unique")
    if len({item.rng_seed for item in runs}) != len(runs):
        raise ValueError("authorized simulation RNG seeds must be globally distinct")
    return tuple(runs)


def _architecture(config: Mapping[str, Any], q: float) -> MSTCNConfig:
    feature = config["feature_transform"]
    architecture = config["architecture"]
    endpoints = config["endpoints"]
    return MSTCNConfig(
        feature_schema=str(feature["feature_schema"]),
        input_features=int(feature["input_features"]),
        window_length=int(architecture["window_length"]),
        channels=int(architecture["channels"]),
        kernel_size=int(architecture["kernel_size"]),
        dilations=tuple(int(value) for value in architecture["dilations"]),
        pool_scales=tuple(int(value) for value in architecture["pool_scales"]),
        hidden_size=int(architecture["hidden_size"]),
        class_ids=(0, *tuple(int(value) for value in architecture["attribution_class_ids"])),
        monitor_start=int(endpoints["dense_half_open_step"][0]),
        endpoint_stride=int(endpoints["mil_half_open_step"][2]),
        mil_top_fraction=float(q),
    )


def _run_name(run: NumericRun) -> str:
    return f"m{run.magnitude_percent:02d}-idv{run.idv:02d}-run{run.simulation_run:03d}"


def _training_maps(
    runs: Sequence[NumericRun],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    raw: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    for run in runs:
        name = _run_name(run)
        if name in raw:
            raise ValueError("training run name collision")
        raw[name] = run.x
        labels[name] = 0 if run.idv == 1 else run.idv
    counts = {label: list(labels.values()).count(label) for label in (0, 3, 9, 15)}
    if len(set(counts.values())) != 1:
        raise ValueError("training partition is not balanced across classes")
    return raw, labels


def _fit_model(runs: Sequence[NumericRun], config: Mapping[str, Any], q: float) -> Any:
    raw, labels = _training_maps(runs)
    training = config["training"]
    return fit_mstcn_cuda(
        raw,
        labels,
        config=_architecture(config, q),
        seed=int(training["seed"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["bag_batch_size"]),
        examples_per_class_per_epoch=int(training["examples_per_class_per_epoch_cap"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        device="cuda",
    )


def _score_runs(
    model: MSTCNModel,
    runs: Sequence[NumericRun],
    *,
    device: str,
    batch_size: int = 8,
) -> tuple[MSTCNPath, ...]:
    """Score each full sequence once; overlapping windows are never re-run."""

    if not runs:
        raise ValueError("runs cannot be empty")
    if device not in ("cpu", "cuda"):
        raise ValueError("score device must be cpu or cuda")
    if _torch is None:
        raise RuntimeError("PyTorch numeric runtime is unavailable")
    network = model._network(device=device)
    target = _torch.device(device)
    endpoint_tensor = _torch.as_tensor(MONITOR_ENDPOINTS, device=target, dtype=_torch.int64)
    paths: list[MSTCNPath] = []
    with _torch.no_grad():
        for start in range(0, len(runs), batch_size):
            selected = runs[start : start + batch_size]
            features = np.stack(
                [model.transform.transform_run(run.x) for run in selected], axis=0
            )
            tensor = _torch.as_tensor(features, device=target, dtype=_torch.float64)
            detection, attribution = network.forward_endpoints(tensor, endpoint_tensor)
            detection_np = detection.detach().cpu().numpy()
            attribution_np = attribution.detach().cpu().numpy()
            paths.extend(
                MSTCNPath(MONITOR_ENDPOINTS, detection_np[index], attribution_np[index])
                for index in range(len(selected))
            )
    return tuple(paths)


def _parity_errors(
    gpu_paths: Sequence[MSTCNPath], cpu_paths: Sequence[MSTCNPath]
) -> list[float]:
    if len(gpu_paths) != len(cpu_paths):
        raise ValueError("CPU/CUDA path counts differ")
    errors: list[float] = []
    for gpu, cpu in zip(gpu_paths, cpu_paths, strict=True):
        if not np.array_equal(gpu.endpoints, cpu.endpoints):
            raise AssertionError("CPU/CUDA endpoints differ")
        error = max(
            float(np.max(np.abs(gpu.detection_logits - cpu.detection_logits), initial=0.0)),
            float(np.max(np.abs(gpu.attribution_logits - cpu.attribution_logits), initial=0.0)),
        )
        if not math.isfinite(error) or error > PARITY_ATOL:
            raise AssertionError(
                f"CPU/CUDA dense dual-head parity failed: {error:.3e} > {PARITY_ATOL:.3e}"
            )
        errors.append(error)
    return errors


def _row_from_path(
    run: NumericRun,
    path: MSTCNPath,
    monitor: FittedTopTailMonitor,
    threshold: float,
    parity_error: float,
) -> tuple[dict[str, Any], tuple[bool, int | None, int | None]]:
    scan = monitor.transform(path.detection_evidence)
    first = first_persistent_exceedance(scan, threshold)
    alarm = first is not None
    attribution = causal_fault_attribution(path.attribution_logits, path.endpoints, first)
    row = {
        "idv": run.idv,
        "simulation_run": run.simulation_run,
        "run_maximum": scan.run_maximum,
        "parity_max_abs_error": float(parity_error),
        "alarm": alarm,
        "first_alarm_index": first,
        "attribution": attribution,
    }
    return row, (alarm, first, attribution)


def _score_candidate_rows(
    runs: Sequence[NumericRun],
    gpu_paths: Sequence[MSTCNPath],
    cpu_paths: Sequence[MSTCNPath],
    monitor: FittedTopTailMonitor,
    threshold: float,
) -> tuple[list[dict[str, Any]], float, bool]:
    errors = _parity_errors(gpu_paths, cpu_paths)
    rows: list[dict[str, Any]] = []
    discrete_equal = True
    for run, gpu, cpu, error in zip(runs, gpu_paths, cpu_paths, errors, strict=True):
        row, gpu_discrete = _row_from_path(run, gpu, monitor, threshold, error)
        _, cpu_discrete = _row_from_path(run, cpu, monitor, threshold, error)
        discrete_equal = discrete_equal and gpu_discrete == cpu_discrete
        rows.append(row)
    return rows, max(errors, default=0.0), discrete_equal


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[max(1, math.ceil(probability * len(ordered))) - 1]


def _candidate_metrics(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for fold in folds for row in fold["rows"]]
    healthy = [row for row in rows if row["idv"] == 1]
    fault_metrics: dict[int, dict[str, float]] = {}
    global_pass = sum(bool(row["alarm"]) for row in healthy) <= 10
    for fault in FAULT_IDVS:
        selected = [row for row in rows if row["idv"] == fault]
        delays = [
            int(row["first_alarm_index"]) - 160 if row["alarm"] else MISS_DELAY
            for row in selected
        ]
        rate = sum(bool(row["alarm"]) for row in selected) / len(selected)
        median = float(statistics.median(delays))
        p90 = _nearest_rank(delays, 0.90)
        fault_metrics[fault] = {"rate": rate, "median": median, "p90": p90}
        global_pass = global_pass and rate >= 0.85 and median <= 140.0 and p90 <= 220.0
    fold_rates: list[float] = []
    fold_pass = True
    for fold in folds:
        fold_rows = fold["rows"]
        fold_pass = fold_pass and sum(
            bool(row["alarm"]) for row in fold_rows if row["idv"] == 1
        ) <= 4
        maxima = sorted(float(value) for value in fold["training_healthy_maxima"])
        tolerance = max(FLOAT_TOLERANCE, float(fold["score_parity_max_abs_error"]))
        distinct = 1 + sum(
            current - previous > tolerance
            for previous, current in zip(maxima, maxima[1:])
        )
        fold_pass = fold_pass and (
            distinct >= 76
            and maxima[78] < maxima[79]
            and maxima[79] - maxima[78] > 100.0 * tolerance
        )
        for fault in FAULT_IDVS:
            chosen = [row for row in fold_rows if row["idv"] == fault]
            rate = sum(bool(row["alarm"]) for row in chosen) / len(chosen)
            fold_rates.append(rate)
            fold_pass = fold_pass and rate >= 0.70
    return {
        "eligible": bool(global_pass and fold_pass),
        "healthy_alarms": sum(bool(row["alarm"]) for row in healthy),
        "fault_metrics": fault_metrics,
        "minimum_fold_detection": min(fold_rates),
    }


def _selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _candidate_metrics(candidate["folds"])
    fault_metrics = metrics["fault_metrics"]
    return (
        -min(fault_metrics[fault]["rate"] for fault in FAULT_IDVS),
        -metrics["minimum_fold_detection"],
        max(fault_metrics[fault]["p90"] for fault in FAULT_IDVS),
        max(fault_metrics[fault]["median"] for fault in FAULT_IDVS),
        metrics["healthy_alarms"],
        -float(candidate["scan_tail_fraction_rho"]),
        -float(candidate["mil_top_fraction_q"]),
        str(candidate["candidate_id"]),
    )


def _model_metadata(
    training_result: Any,
    *,
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    monitor: FittedTopTailMonitor,
) -> dict[str, Any]:
    return _plain_json(
        {
            "schema": MODEL_METADATA_SCHEMA,
            "backend_schema": BACKEND_SCHEMA,
            "complete_model_config_sha256": _config_sha256(config),
            "selected_candidate": {
                key: selected[key]
                for key in (
                    "candidate_id",
                    "mil_top_fraction_q",
                    "scan_tail_fraction_rho",
                )
            },
            "fitted_monitor": monitor.to_document(),
            "training": dict(training_result.metadata),
            "loss_history": list(training_result.loss_history),
            "monitor_fit_scope": "all_100_magnitude_50_development_healthy_runs_only",
            "held_out_25_or_75_percent_accessed": False,
        }
    )


def development_iteration(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    fold_map: Mapping[int, int],
    config: Mapping[str, Any],
    model_path: Path,
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete_config = _validate_config(config)
    folds_by_run = _validate_fold_map(fold_map)
    phase, _ = _validate_access_plan(access_plan)
    if phase != "development":
        raise ValueError("development backend received a non-development plan")
    destination = Path(model_path)
    if destination.exists():
        raise FileExistsError("development model destination already exists")
    runs = load_numeric_runs(source_path, access_plan)
    candidates = [{**identity, "folds": []} for identity in complete_config["candidate_grid"]]
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    for fold in range(FOLD_COUNT):
        train_runs = [run for run in runs if folds_by_run[run.simulation_run] != fold]
        validation_runs = [run for run in runs if folds_by_run[run.simulation_run] == fold]
        training_healthy = [run for run in train_runs if run.idv == 1]
        for q in Q_VALUES:
            print(f"[v3] fold {fold + 1}/5 q={q:g}: training start", flush=True)
            result = _fit_model(train_runs, complete_config, q)
            print(f"[v3] fold {fold + 1}/5 q={q:g}: training complete", flush=True)
            gpu_training_healthy = _score_runs(result.model, training_healthy, device="cuda")
            gpu_validation = _score_runs(result.model, validation_runs, device="cuda")
            cpu_validation = _score_runs(result.model, validation_runs, device="cpu")
            training_evidence = np.stack(
                [path.detection_evidence for path in gpu_training_healthy], axis=0
            )
            for rho in RHO_VALUES:
                monitor = FittedTopTailMonitor.fit(
                    training_evidence,
                    MONITOR_ENDPOINTS,
                    windows=SCAN_WINDOWS,
                    top_fraction=rho,
                )
                maxima = monitor.training_run_maxima(training_evidence)
                threshold = strict_order_statistic(maxima, rank=TRAINING_THRESHOLD_RANK)
                rows, parity, discrete = _score_candidate_rows(
                    validation_runs,
                    gpu_validation,
                    cpu_validation,
                    monitor,
                    threshold,
                )
                if not discrete:
                    raise AssertionError("CPU/CUDA fold alarm/attribution outputs differ")
                candidate_id = f"q{int(q * 1000):03d}-rho{int(rho * 1000):03d}"
                by_id[candidate_id]["folds"].append(
                    {
                        "fold": fold,
                        "training_healthy_maxima": maxima.tolist(),
                        "rows": rows,
                        "score_parity_max_abs_error": parity,
                        "discrete_parity_equal": True,
                    }
                )
            del result
            _torch.cuda.empty_cache()

    eligible = [
        candidate
        for candidate in candidates
        if _candidate_metrics(candidate["folds"])["eligible"]
    ]
    selected = min(eligible, key=_selection_key) if eligible else None
    runtime = dict(cuda_preflight())
    if selected is None:
        return {
            "config": complete_config,
            "candidates": candidates,
            "selected_candidate_id": None,
            "runtime": runtime,
            "model_metadata": None,
            "model_parity_max_abs_error": None,
        }

    print(
        f"[v3] final refit q={float(selected['mil_top_fraction_q']):g}: start",
        flush=True,
    )
    final_result = _fit_model(runs, complete_config, float(selected["mil_top_fraction_q"]))
    final_gpu_paths = _score_runs(final_result.model, runs, device="cuda")
    final_cpu_paths = _score_runs(final_result.model, runs, device="cpu")
    model_parity = max(
        _parity_errors(final_gpu_paths, final_cpu_paths), default=0.0
    )
    healthy_paths = [
        path
        for run, path in zip(runs, final_gpu_paths, strict=True)
        if run.idv == 1
    ]
    evidence = np.stack([path.detection_evidence for path in healthy_paths], axis=0)
    final_monitor = FittedTopTailMonitor.fit(
        evidence,
        MONITOR_ENDPOINTS,
        windows=SCAN_WINDOWS,
        top_fraction=float(selected["scan_tail_fraction_rho"]),
    )
    metadata = _model_metadata(
        final_result,
        config=complete_config,
        selected=selected,
        monitor=final_monitor,
    )
    final_result.model.save_npz(destination, metadata=metadata)
    print("[v3] final refit and bound model save: complete", flush=True)
    return {
        "config": complete_config,
        "candidates": candidates,
        "selected_candidate_id": selected["candidate_id"],
        "runtime": runtime,
        "model_metadata": metadata,
        "model_parity_max_abs_error": model_parity,
    }


def _load_bound_model(
    model_path: Path, config: Mapping[str, Any]
) -> tuple[MSTCNModel, FittedTopTailMonitor, dict[str, Any]]:
    complete_config = _validate_config(config)
    model, metadata = load_mstcn_model(model_path)
    required = {
        "schema", "backend_schema", "complete_model_config_sha256",
        "selected_candidate", "fitted_monitor", "training", "loss_history",
        "monitor_fit_scope", "held_out_25_or_75_percent_accessed",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("model bound metadata fields changed")
    if (
        metadata["schema"] != MODEL_METADATA_SCHEMA
        or metadata["backend_schema"] != BACKEND_SCHEMA
        or metadata["complete_model_config_sha256"] != _config_sha256(complete_config)
        or metadata["monitor_fit_scope"]
        != "all_100_magnitude_50_development_healthy_runs_only"
        or metadata["held_out_25_or_75_percent_accessed"] is not False
    ):
        raise ValueError("model binding metadata differs from v3")
    selected = metadata["selected_candidate"]
    if (
        not isinstance(selected, Mapping)
        or set(selected) != {
            "candidate_id", "mil_top_fraction_q", "scan_tail_fraction_rho"
        }
        or dict(selected) not in complete_config["candidate_grid"]
    ):
        raise ValueError("model selected candidate is not in the frozen grid")
    if model.config != _architecture(
        complete_config, float(selected["mil_top_fraction_q"])
    ):
        raise ValueError("model architecture/q differs from its bound candidate")
    training = metadata["training"]
    if (
        not isinstance(training, Mapping)
        or training.get("requested_device") != "cuda"
        or training.get("dtype") != "float64"
        or int(training.get("seed", -1)) != complete_config["training"]["seed"]
    ):
        raise ValueError("model training metadata is not the frozen CUDA float64 fit")
    losses = metadata["loss_history"]
    if (
        not isinstance(losses, list)
        or len(losses) != complete_config["training"]["epochs"]
        or any(not math.isfinite(float(value)) for value in losses)
    ):
        raise ValueError("model loss history differs from the frozen training length")
    monitor = FittedTopTailMonitor.from_document(metadata["fitted_monitor"])
    if (
        not np.array_equal(monitor.endpoints, MONITOR_ENDPOINTS)
        or monitor.windows != SCAN_WINDOWS
        or monitor.top_fraction != float(selected["scan_tail_fraction_rho"])
        or monitor.persistence != 2
    ):
        raise ValueError("serialized monitor differs from the selected scan candidate")
    return model, monitor, metadata


def calibrate(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    model_path: Path,
    config: Mapping[str, Any],
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete_config = _validate_config(config)
    phase, _ = _validate_access_plan(access_plan)
    if phase != "calibrate":
        raise ValueError("calibration backend received a non-calibration plan")
    model, monitor, _ = _load_bound_model(model_path, complete_config)
    runs = load_numeric_runs(source_path, access_plan)
    gpu_paths = _score_runs(model, runs, device="cuda")
    cpu_paths = _score_runs(model, runs, device="cpu")
    errors = _parity_errors(gpu_paths, cpu_paths)
    rows: list[dict[str, Any]] = []
    gpu_maxima: list[float] = []
    cpu_maxima: list[float] = []
    for run, gpu, cpu, error in zip(runs, gpu_paths, cpu_paths, errors, strict=True):
        gpu_maximum = monitor.transform(gpu.detection_evidence).run_maximum
        cpu_maximum = monitor.transform(cpu.detection_evidence).run_maximum
        gpu_maxima.append(gpu_maximum)
        cpu_maxima.append(cpu_maximum)
        rows.append(
            {
                "idv": run.idv,
                "simulation_run": run.simulation_run,
                "run_maximum": gpu_maximum,
                "parity_max_abs_error": error,
            }
        )
    # Calibration has no alarm label.  Its discrete operation is the stable
    # ordering used by the rank statistic, so both devices must order runs identically.
    gpu_order = np.argsort(np.asarray(gpu_maxima), kind="stable")
    cpu_order = np.argsort(np.asarray(cpu_maxima), kind="stable")
    discrete = bool(np.array_equal(gpu_order, cpu_order))
    if not discrete:
        raise AssertionError("CPU/CUDA calibration rank ordering differs")
    return {
        "config": complete_config,
        "rows": rows,
        "model_parity_max_abs_error": max(errors, default=0.0),
        "discrete_parity_equal": True,
    }


def confirm(
    *,
    source_path: Path,
    access_plan: Sequence[Mapping[str, Any]],
    model_path: Path,
    config: Mapping[str, Any],
    threshold: float,
    device: str,
    dtype: str,
) -> Mapping[str, Any]:
    _require_cuda_float64(device=device, dtype=dtype)
    complete_config = _validate_config(config)
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise ValueError("confirmation threshold must be finite")
    frozen_threshold = float(threshold)
    phase, _ = _validate_access_plan(access_plan)
    if phase != "confirm":
        raise ValueError("confirmation backend received a non-confirmation plan")
    model, monitor, _ = _load_bound_model(model_path, complete_config)
    runs = load_numeric_runs(source_path, access_plan)
    gpu_paths = _score_runs(model, runs, device="cuda")
    cpu_paths = _score_runs(model, runs, device="cpu")
    rows, parity, discrete = _score_candidate_rows(
        runs, gpu_paths, cpu_paths, monitor, frozen_threshold
    )
    if not discrete:
        raise AssertionError("CPU/CUDA confirmation alarm/attribution outputs differ")
    return {
        "config": complete_config,
        "threshold": frozen_threshold,
        "rows": rows,
        "model_parity_max_abs_error": parity,
        "discrete_parity_equal": True,
    }


__all__ = [
    "NumericRun",
    "calibrate",
    "confirm",
    "cuda_preflight",
    "development_iteration",
    "load_numeric_runs",
]
