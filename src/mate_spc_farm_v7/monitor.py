"""Frozen two-sided empirical-CDF CUSUM monitor for FARM v7.

Only healthy fit runs enter the empirical reference.  CPU scoring is the
transparent reference implementation; CUDA scoring performs the same float64
rank lookup and recurrence in batches.  Both return CPU NumPy arrays so the
run-level threshold and decision contract has one representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


CANDIDATE_ID = "farm-rankcusum-v7-fixed"
REFERENCE_SCHEMA = "farm_rankcusum_v7_reference_v1"
THRESHOLD_SCHEMA = "farm_rankcusum_v7_inner_threshold_v1"

INPUT_CHANNEL_COUNT = 52
CHANNEL_INDICES = tuple(range(41)) + (46, 47, 48, 50, 51)
SELECTED_CHANNEL_COUNT = 46
FIT_ROWS_PER_RUN = 500
ALLOWED_FIT_RUN_COUNTS = (60, 100)
DEVELOPMENT_FIT_RUN_COUNT = 60
FINAL_REFIT_RUN_COUNT = 100

MONITOR_START = 160
MONITOR_STOP = 500
MONITOR_ENDPOINTS = tuple(range(MONITOR_START, MONITOR_STOP))
MONITOR_LENGTH = len(MONITOR_ENDPOINTS)
MISS_DELAY = MONITOR_LENGTH

DEFAULT_DRIFT = 1.3
TOP_R = 4
ECDF_SEARCH_SIDE = "left"
CALIBRATION_RUN_COUNT = 20
MINIMUM_DISTINCT_MAXIMA = 18
NUMERIC_TOLERANCE_FLOOR = 1.0e-10


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _strict_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_fit_run_count(value: Any, *, name: str) -> int:
    count = _strict_int(value, name=name)
    if count not in ALLOWED_FIT_RUN_COUNTS:
        raise ValueError(f"{name} must be exactly 60 or 100")
    return count


def _validate_reference_count(value: Any) -> int:
    count = _strict_int(value, name="reference_count")
    allowed = tuple(run_count * FIT_ROWS_PER_RUN for run_count in ALLOWED_FIT_RUN_COUNTS)
    if count not in allowed:
        raise ValueError("reference_count must be exactly 30000 or 50000")
    return count


def _validate_runs(
    value: Any, *, name: str, expected_run_count: int | None = None
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a NumPy float64 array")
    runs = value
    if (
        runs.ndim != 3
        or runs.shape[0] < 1
        or runs.shape[1:] != (FIT_ROWS_PER_RUN, INPUT_CHANNEL_COUNT)
    ):
        raise ValueError(f"{name} must have shape (runs, 500, 52)")
    if expected_run_count is not None and runs.shape[0] != expected_run_count:
        raise ValueError(f"{name} must contain exactly {expected_run_count} runs")
    if not np.isfinite(runs).all():
        raise ValueError(f"{name} must contain only finite values")
    return runs


def theoretical_score_upper_bound(reference_count: int) -> float:
    """Return the conservative finite-horizon maximum of the frozen scan."""

    count = _validate_reference_count(reference_count)
    maximum_increment = max(math.log(count + 2.0) - DEFAULT_DRIFT, 0.0)
    return float(TOP_R * MONITOR_LENGTH * maximum_increment)


@dataclass(frozen=True)
class FarmReference:
    """Per-channel stable-sorted healthy values used by the Bayesian ECDF."""

    sorted_values: np.ndarray
    fit_run_count: int

    def __post_init__(self) -> None:
        run_count = _validate_fit_run_count(self.fit_run_count, name="fit_run_count")
        values = self.sorted_values
        expected_shape = (SELECTED_CHANNEL_COUNT, run_count * FIT_ROWS_PER_RUN)
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype(np.float64)
            or values.shape != expected_shape
        ):
            raise ValueError(
                f"sorted_values must be a NumPy float64 array with shape {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("sorted_values must contain only finite values")
        if np.any(values[:, 1:] < values[:, :-1]):
            raise ValueError("each empirical reference channel must be nondecreasing")
        owned = np.array(values, dtype=np.float64, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "sorted_values", owned)
        object.__setattr__(self, "fit_run_count", run_count)

    @property
    def reference_count(self) -> int:
        return self.fit_run_count * FIT_ROWS_PER_RUN

    @classmethod
    def fit(
        cls, healthy_runs: np.ndarray, *, expected_run_count: int = DEVELOPMENT_FIT_RUN_COUNT
    ) -> "FarmReference":
        """Fit from exactly 60 development or 100 final-refit healthy runs."""

        run_count = _validate_fit_run_count(
            expected_run_count, name="expected_run_count"
        )
        healthy = _validate_runs(
            healthy_runs, name="healthy_runs", expected_run_count=run_count
        )
        selected = healthy[:, :, CHANNEL_INDICES]
        flat = np.reshape(selected, (run_count * FIT_ROWS_PER_RUN, SELECTED_CHANNEL_COUNT))
        # ``kind='stable'`` is part of the frozen tie policy.  searchsorted uses
        # the left boundary for an observation equal to one or more fit values.
        sorted_values = np.sort(flat, axis=0, kind="stable").T
        return cls(sorted_values=np.asarray(sorted_values, dtype=np.float64), fit_run_count=run_count)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "input_channel_count": INPUT_CHANNEL_COUNT,
            "channel_indices": list(CHANNEL_INDICES),
            "selected_channel_count": SELECTED_CHANNEL_COUNT,
            "fit_rows_per_run": FIT_ROWS_PER_RUN,
            "fit_run_count": self.fit_run_count,
            "reference_count": self.reference_count,
            "monitor_endpoints": [MONITOR_START, MONITOR_STOP - 1, 1],
            "ecdf_search_side": ECDF_SEARCH_SIDE,
            "ecdf_formula": "(searchsorted_left_rank + 1) / (N + 2)",
            "drift": DEFAULT_DRIFT,
            "top_r": TOP_R,
            "dtype": "float64",
            "sorted_values": np.array(self.sorted_values, copy=True),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "FarmReference":
        if not isinstance(state, Mapping):
            raise ValueError("reference state must be a mapping")
        expected_keys = {
            "schema",
            "candidate_id",
            "input_channel_count",
            "channel_indices",
            "selected_channel_count",
            "fit_rows_per_run",
            "fit_run_count",
            "reference_count",
            "monitor_endpoints",
            "ecdf_search_side",
            "ecdf_formula",
            "drift",
            "top_r",
            "dtype",
            "sorted_values",
        }
        if set(state) != expected_keys:
            raise ValueError("reference state fields changed")
        frozen_values = {
            "schema": REFERENCE_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "input_channel_count": INPUT_CHANNEL_COUNT,
            "channel_indices": list(CHANNEL_INDICES),
            "selected_channel_count": SELECTED_CHANNEL_COUNT,
            "fit_rows_per_run": FIT_ROWS_PER_RUN,
            "monitor_endpoints": [MONITOR_START, MONITOR_STOP - 1, 1],
            "ecdf_search_side": ECDF_SEARCH_SIDE,
            "ecdf_formula": "(searchsorted_left_rank + 1) / (N + 2)",
            "drift": DEFAULT_DRIFT,
            "top_r": TOP_R,
            "dtype": "float64",
        }
        for key, expected in frozen_values.items():
            if state[key] != expected:
                raise ValueError(f"reference state {key} changed")
        run_count = _validate_fit_run_count(state["fit_run_count"], name="fit_run_count")
        reference_count = _validate_reference_count(state["reference_count"])
        if reference_count != run_count * FIT_ROWS_PER_RUN:
            raise ValueError("reference_count does not match fit_run_count")
        values = state["sorted_values"]
        if not isinstance(values, np.ndarray):
            raise ValueError("sorted_values must remain a NumPy array")
        return cls(sorted_values=values, fit_run_count=run_count)


def save_farm_reference_npz(reference: FarmReference, path: str | os.PathLike[str]) -> None:
    """Atomically save a reference as a no-pickle NPZ archive."""

    if not isinstance(reference, FarmReference):
        raise TypeError("reference must be a FarmReference")
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("reference path must end in .npz")
    if not destination.parent.is_dir():
        raise FileNotFoundError("reference destination directory does not exist")
    state = reference.state_dict()
    sorted_values = state.pop("sorted_values")
    metadata = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(metadata),
                sorted_values=sorted_values,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_farm_reference_npz(path: str | os.PathLike[str]) -> FarmReference:
    """Load a no-pickle archive and re-run every frozen-state validation."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata_json", "sorted_values"}:
            raise ValueError("reference archive fields changed")
        metadata_value = archive["metadata_json"]
        if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
            raise ValueError("reference metadata must be a scalar string")
        metadata_item = metadata_value.item()
        if isinstance(metadata_item, bytes):
            metadata_item = metadata_item.decode("utf-8")
        try:
            state = json.loads(metadata_item)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("reference metadata is not valid JSON") from exc
        if not isinstance(state, dict):
            raise ValueError("reference metadata JSON must be an object")
        state["sorted_values"] = np.array(archive["sorted_values"], copy=True)
    return FarmReference.from_state_dict(state)


@dataclass(frozen=True)
class FarmScoreBatch:
    """Aligned FARM paths and maxima returned by either numeric backend."""

    paths: np.ndarray
    source_device: str
    reference_count: int
    run_maxima: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        paths = self.paths
        if (
            not isinstance(paths, np.ndarray)
            or paths.dtype != np.dtype(np.float64)
            or paths.ndim != 2
            or paths.shape[0] < 1
            or paths.shape[1] != MONITOR_LENGTH
        ):
            raise ValueError("paths must be a NumPy float64 array with shape (runs, 340)")
        if not np.isfinite(paths).all() or np.any(paths < 0.0):
            raise ValueError("paths must be finite and nonnegative")
        valid_cuda = (
            isinstance(self.source_device, str)
            and self.source_device.startswith("cuda:")
            and self.source_device.removeprefix("cuda:").isdigit()
        )
        if not isinstance(self.source_device, str) or not (
            self.source_device in {"cpu", "cuda"} or valid_cuda
        ):
            raise ValueError("source_device must identify cpu or cuda")
        count = _validate_reference_count(self.reference_count)
        owned = np.array(paths, dtype=np.float64, order="C", copy=True)
        maxima = np.max(owned, axis=1)
        owned.setflags(write=False)
        maxima.setflags(write=False)
        object.__setattr__(self, "paths", owned)
        object.__setattr__(self, "run_maxima", maxima)
        object.__setattr__(self, "reference_count", count)

    @property
    def endpoints(self) -> np.ndarray:
        result = np.arange(MONITOR_START, MONITOR_STOP, dtype=np.int64)
        result.setflags(write=False)
        return result


def _rank_paths_numpy(reference: FarmReference, selected: np.ndarray) -> np.ndarray:
    run_count = selected.shape[0]
    ranks = np.empty(selected.shape, dtype=np.int64)
    for channel in range(SELECTED_CHANNEL_COUNT):
        ranks[:, :, channel] = np.searchsorted(
            reference.sorted_values[channel], selected[:, :, channel], side=ECDF_SEARCH_SIDE
        )
    mu = (ranks.astype(np.float64) + 1.0) / (reference.reference_count + 2.0)
    upper = np.zeros((run_count, SELECTED_CHANNEL_COUNT), dtype=np.float64)
    lower = np.zeros_like(upper)
    paths = np.empty((run_count, MONITOR_LENGTH), dtype=np.float64)
    for time in range(MONITOR_LENGTH):
        current = mu[:, time, :]
        upper = np.maximum(upper - np.log1p(-current) - DEFAULT_DRIFT, 0.0)
        lower = np.maximum(lower - np.log(current) - DEFAULT_DRIFT, 0.0)
        channel_scores = np.maximum(upper, lower)
        top = np.partition(
            channel_scores, SELECTED_CHANNEL_COUNT - TOP_R, axis=1
        )[:, -TOP_R:]
        paths[:, time] = np.sum(top, axis=1, dtype=np.float64)
    if not np.isfinite(paths).all():
        raise FloatingPointError("CPU FARM recurrence produced nonfinite scores")
    return paths


def score_runs_cpu(reference: FarmReference, runs: np.ndarray) -> FarmScoreBatch:
    """Score complete 500x52 runs with the transparent NumPy reference."""

    if not isinstance(reference, FarmReference):
        raise TypeError("reference must be a FarmReference")
    values = _validate_runs(runs, name="runs")
    selected = values[:, MONITOR_START:MONITOR_STOP, CHANNEL_INDICES]
    paths = _rank_paths_numpy(reference, selected)
    return FarmScoreBatch(paths=paths, source_device="cpu", reference_count=reference.reference_count)


def score_runs_cuda(
    reference: FarmReference,
    runs: np.ndarray,
    *,
    device: Any = "cuda",
    batch_size: int = 64,
) -> FarmScoreBatch:
    """Authoritatively score complete runs with a float64 CUDA recurrence."""

    if not isinstance(reference, FarmReference):
        raise TypeError("reference must be a FarmReference")
    values = _validate_runs(runs, name="runs")
    size = _strict_int(batch_size, name="batch_size")
    if size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("PyTorch is required for CUDA FARM scoring") from exc
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("score_runs_cuda requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    boundaries = torch.as_tensor(
        np.array(reference.sorted_values, copy=True), dtype=torch.float64, device=target
    )
    result = np.empty((values.shape[0], MONITOR_LENGTH), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, values.shape[0], size):
            stop = min(start + size, values.shape[0])
            selected = np.ascontiguousarray(
                values[start:stop, MONITOR_START:MONITOR_STOP, CHANNEL_INDICES]
            )
            tensor = torch.as_tensor(selected, dtype=torch.float64, device=target)
            ranks = torch.empty(tensor.shape, dtype=torch.int64, device=target)
            for channel in range(SELECTED_CHANNEL_COUNT):
                ranks[:, :, channel] = torch.searchsorted(
                    boundaries[channel], tensor[:, :, channel].contiguous(), right=False
                )
            mu = (ranks.to(torch.float64) + 1.0) / float(reference.reference_count + 2)
            upper = torch.zeros(
                (stop - start, SELECTED_CHANNEL_COUNT), dtype=torch.float64, device=target
            )
            lower = torch.zeros_like(upper)
            batch_paths = torch.empty(
                (stop - start, MONITOR_LENGTH), dtype=torch.float64, device=target
            )
            for time in range(MONITOR_LENGTH):
                current = mu[:, time, :]
                upper = torch.clamp_min(
                    upper - torch.log1p(-current) - DEFAULT_DRIFT, 0.0
                )
                lower = torch.clamp_min(lower - torch.log(current) - DEFAULT_DRIFT, 0.0)
                channel_scores = torch.maximum(upper, lower)
                batch_paths[:, time] = torch.topk(
                    channel_scores, k=TOP_R, dim=1, largest=True, sorted=False
                ).values.sum(dim=1)
            if not bool(torch.isfinite(batch_paths).all().item()):
                raise FloatingPointError("CUDA FARM recurrence produced nonfinite scores")
            result[start:stop] = batch_paths.cpu().numpy()
    return FarmScoreBatch(
        paths=result,
        source_device=str(target),
        reference_count=reference.reference_count,
    )


@dataclass(frozen=True)
class FarmThreshold:
    """A valid rank-20 inner-health threshold with numeric audit fields."""

    threshold: float
    rank19: float
    distinct_count: int
    repeat_error: float
    reference_count: int

    def __post_init__(self) -> None:
        threshold = _strict_float(self.threshold, name="threshold")
        if threshold < 0.0:
            raise ValueError("threshold must be nonnegative")
        rank19 = _strict_float(self.rank19, name="rank19")
        error = _strict_float(self.repeat_error, name="repeat_error")
        if threshold < 0.0 or rank19 < 0.0 or rank19 > threshold:
            raise ValueError("threshold order must satisfy 0 <= rank19 <= threshold")
        if error < 0.0:
            raise ValueError("repeat_error must be nonnegative")
        distinct = _strict_int(self.distinct_count, name="distinct_count")
        if distinct < MINIMUM_DISTINCT_MAXIMA or distinct > CALIBRATION_RUN_COUNT:
            raise ValueError("inner healthy maxima are numerically degenerate")
        count = _validate_reference_count(self.reference_count)
        gap = threshold - rank19
        tolerance = 100.0 * max(error, NUMERIC_TOLERANCE_FLOOR)
        if gap <= tolerance:
            raise ValueError("rank20-rank19 does not clear the numeric tolerance")
        if threshold >= theoretical_score_upper_bound(count):
            raise ValueError("threshold is not strictly below the reachable score bound")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "rank19", rank19)
        object.__setattr__(self, "distinct_count", distinct)
        object.__setattr__(self, "repeat_error", error)
        object.__setattr__(self, "reference_count", count)

    @property
    def top_gap(self) -> float:
        return float(self.threshold - self.rank19)

    @property
    def tolerance(self) -> float:
        return float(100.0 * max(self.repeat_error, NUMERIC_TOLERANCE_FLOOR))

    def to_document(self) -> dict[str, object]:
        return {
            "schema": THRESHOLD_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "calibration_run_count": CALIBRATION_RUN_COUNT,
            "ascending_rank": CALIBRATION_RUN_COUNT,
            "comparison": "strictly_greater_than",
            "threshold": self.threshold,
            "rank19": self.rank19,
            "rank20_minus_rank19": self.top_gap,
            "distinct_count": self.distinct_count,
            "minimum_distinct_count": MINIMUM_DISTINCT_MAXIMA,
            "repeat_error": self.repeat_error,
            "numeric_tolerance_floor": NUMERIC_TOLERANCE_FLOOR,
            "numeric_tolerance": self.tolerance,
            "reference_count": self.reference_count,
            "theoretical_score_upper_bound": theoretical_score_upper_bound(
                self.reference_count
            ),
            "monitor_endpoints": [MONITOR_START, MONITOR_STOP - 1, 1],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "FarmThreshold":
        if not isinstance(document, Mapping):
            raise ValueError("threshold document must be a mapping")
        expected_keys = {
            "schema",
            "candidate_id",
            "calibration_run_count",
            "ascending_rank",
            "comparison",
            "threshold",
            "rank19",
            "rank20_minus_rank19",
            "distinct_count",
            "minimum_distinct_count",
            "repeat_error",
            "numeric_tolerance_floor",
            "numeric_tolerance",
            "reference_count",
            "theoretical_score_upper_bound",
            "monitor_endpoints",
        }
        if set(document) != expected_keys:
            raise ValueError("threshold document fields changed")
        frozen = {
            "schema": THRESHOLD_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "calibration_run_count": CALIBRATION_RUN_COUNT,
            "ascending_rank": CALIBRATION_RUN_COUNT,
            "comparison": "strictly_greater_than",
            "minimum_distinct_count": MINIMUM_DISTINCT_MAXIMA,
            "numeric_tolerance_floor": NUMERIC_TOLERANCE_FLOOR,
            "monitor_endpoints": [MONITOR_START, MONITOR_STOP - 1, 1],
        }
        for key, expected in frozen.items():
            if document[key] != expected:
                raise ValueError(f"threshold document {key} changed")
        restored = cls(
            threshold=document["threshold"],
            rank19=document["rank19"],
            distinct_count=document["distinct_count"],
            repeat_error=document["repeat_error"],
            reference_count=document["reference_count"],
        )
        derived = {
            "rank20_minus_rank19": restored.top_gap,
            "numeric_tolerance": restored.tolerance,
            "theoretical_score_upper_bound": theoretical_score_upper_bound(
                restored.reference_count
            ),
        }
        for key, expected in derived.items():
            if _strict_float(document[key], name=key) != expected:
                raise ValueError(f"threshold document {key} is inconsistent")
        return restored


def calibrate_threshold(
    inner_healthy_maxima: Sequence[float] | np.ndarray,
    *,
    expected_run_count: int = CALIBRATION_RUN_COUNT,
    repeat_error: float = 0.0,
    reference_count: int = DEVELOPMENT_FIT_RUN_COUNT * FIT_ROWS_PER_RUN,
) -> FarmThreshold:
    """Fit the strict rank-20 threshold and fail closed on degeneracy."""

    calibration_count = _strict_int(expected_run_count, name="expected_run_count")
    if calibration_count != CALIBRATION_RUN_COUNT:
        raise ValueError("expected_run_count must be exactly 20")
    raw = np.asarray(inner_healthy_maxima)
    if raw.dtype.kind not in {"i", "u", "f"} or raw.dtype.kind == "b":
        raise ValueError("inner_healthy_maxima must be real numeric values")
    maxima = np.asarray(inner_healthy_maxima, dtype=np.float64)
    if maxima.shape != (CALIBRATION_RUN_COUNT,) or not np.isfinite(maxima).all():
        raise ValueError("inner_healthy_maxima must contain exactly 20 finite values")
    if np.any(maxima < 0.0):
        raise ValueError("inner_healthy_maxima must be nonnegative")
    error = _strict_float(repeat_error, name="repeat_error")
    if error < 0.0:
        raise ValueError("repeat_error must be nonnegative")
    ordered = np.sort(maxima, kind="stable")
    return FarmThreshold(
        threshold=float(ordered[-1]),
        rank19=float(ordered[-2]),
        distinct_count=int(np.unique(maxima).size),
        repeat_error=error,
        reference_count=_validate_reference_count(reference_count),
    )


@dataclass(frozen=True)
class FarmDecisions:
    """Strict-threshold alarms aligned to complete input runs."""

    alarms: np.ndarray
    alarm_endpoints: np.ndarray
    delays: np.ndarray
    run_maxima: np.ndarray
    threshold: float

    def __post_init__(self) -> None:
        alarms = np.asarray(self.alarms)
        endpoints = np.asarray(self.alarm_endpoints)
        delays = np.asarray(self.delays)
        maxima = np.asarray(self.run_maxima)
        run_count = alarms.shape[0] if alarms.ndim == 1 else -1
        if alarms.dtype != np.dtype(np.bool_) or run_count < 1:
            raise ValueError("alarms must be a one-dimensional boolean array")
        if endpoints.dtype != np.dtype(np.int64) or endpoints.shape != (run_count,):
            raise ValueError("alarm_endpoints must be an aligned int64 array")
        if delays.dtype != np.dtype(np.int64) or delays.shape != (run_count,):
            raise ValueError("delays must be an aligned int64 array")
        if maxima.dtype != np.dtype(np.float64) or maxima.shape != (run_count,):
            raise ValueError("run_maxima must be an aligned float64 array")
        if not np.isfinite(maxima).all() or np.any(maxima < 0.0):
            raise ValueError("run_maxima must be finite and nonnegative")
        threshold = _strict_float(self.threshold, name="threshold")
        if np.any(endpoints[alarms] < MONITOR_START) or np.any(
            endpoints[alarms] >= MONITOR_STOP
        ):
            raise ValueError("alarmed endpoints must lie in 160..499")
        if np.any(endpoints[~alarms] != -1):
            raise ValueError("missed runs must use endpoint -1")
        expected_delays = np.where(alarms, endpoints - MONITOR_START, MISS_DELAY)
        if not np.array_equal(delays, expected_delays):
            raise ValueError("delays are inconsistent with alarm endpoints")
        if not np.array_equal(alarms, maxima > threshold):
            raise ValueError("alarms are inconsistent with strict-threshold run maxima")
        for name, value in (
            ("alarms", alarms),
            ("alarm_endpoints", endpoints),
            ("delays", delays),
            ("run_maxima", maxima),
        ):
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        object.__setattr__(self, "threshold", threshold)


def apply_threshold(
    scores: FarmScoreBatch | np.ndarray, threshold: FarmThreshold
) -> FarmDecisions:
    """Apply the frozen strict ``>`` comparison to a batch of score paths."""

    if not isinstance(threshold, FarmThreshold):
        raise TypeError("threshold must be a FarmThreshold")
    if isinstance(scores, FarmScoreBatch):
        if scores.reference_count != threshold.reference_count:
            raise ValueError("score and threshold reference counts differ")
        paths = scores.paths
    else:
        paths = scores
        if (
            not isinstance(paths, np.ndarray)
            or paths.dtype != np.dtype(np.float64)
            or paths.ndim != 2
            or paths.shape[0] < 1
            or paths.shape[1] != MONITOR_LENGTH
            or not np.isfinite(paths).all()
            or np.any(paths < 0.0)
        ):
            raise ValueError("scores must be finite float64 paths with shape (runs, 340)")
    exceedances = paths > threshold.threshold
    alarms = np.any(exceedances, axis=1).astype(np.bool_)
    first_positions = np.argmax(exceedances, axis=1)
    endpoints = np.where(alarms, MONITOR_START + first_positions, -1).astype(np.int64)
    delays = np.where(alarms, first_positions, MISS_DELAY).astype(np.int64)
    maxima = np.max(paths, axis=1).astype(np.float64)
    return FarmDecisions(
        alarms=alarms,
        alarm_endpoints=endpoints,
        delays=delays,
        run_maxima=maxima,
        threshold=threshold.threshold,
    )


__all__ = [
    "ALLOWED_FIT_RUN_COUNTS",
    "CALIBRATION_RUN_COUNT",
    "CANDIDATE_ID",
    "CHANNEL_INDICES",
    "DEFAULT_DRIFT",
    "DEVELOPMENT_FIT_RUN_COUNT",
    "ECDF_SEARCH_SIDE",
    "FINAL_REFIT_RUN_COUNT",
    "FIT_ROWS_PER_RUN",
    "FarmDecisions",
    "FarmReference",
    "FarmScoreBatch",
    "FarmThreshold",
    "INPUT_CHANNEL_COUNT",
    "MINIMUM_DISTINCT_MAXIMA",
    "MISS_DELAY",
    "MONITOR_ENDPOINTS",
    "MONITOR_LENGTH",
    "MONITOR_START",
    "MONITOR_STOP",
    "NUMERIC_TOLERANCE_FLOOR",
    "REFERENCE_SCHEMA",
    "SELECTED_CHANNEL_COUNT",
    "THRESHOLD_SCHEMA",
    "TOP_R",
    "apply_threshold",
    "calibrate_threshold",
    "load_farm_reference_npz",
    "save_farm_reference_npz",
    "score_runs_cpu",
    "score_runs_cuda",
    "theoretical_score_upper_bound",
]
