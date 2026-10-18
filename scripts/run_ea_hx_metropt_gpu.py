"""One-shot RTX evaluation of frozen EA-HX-CUSUM on MetroPT-3.

The model family and score fusion are unchanged from the synthetic study.
MetroPT is resampled by the existing causal one-minute adapter.  The predictor
is fitted on February 1--20, the monitor state on February 21--29, trajectory
thresholds on 70 later healthy calendar days, and the final score is reported
once on the three official confirmation failures plus event-free days.
"""

from __future__ import annotations

import torch  # noqa: F401  # torch-first DLL contract on Windows

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np

from mate_spc_cris_v9.baselines import (
    EnergyCUSUMBaseline,
    PCAT2QBaseline,
    PredictiveResidualCUSUMBaseline,
)
from mate_spc_cris_v9.energy_anchor_monitor import EnergyAnchoredHXMonitor
from mate_spc_cris_v9.model import CRISConfig, CRISModel, CRISTrainingConfig
from mate_spc_metropt.data import (
    MetroPTMinuteInterval,
    MetroPTTimestampAudit,
    load_metropt_interval,
)


PROJECT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT / "data" / "metropt" / "MetroPT3(AirCompressor).csv"
AUDIT_PATH = PROJECT / "outputs" / "metropt_cared_bc_gpu_v0_1" / "timestamp_audit.json"
OUTPUT_ROOT = PROJECT / "outputs" / "ea_hx_metropt"
MODEL_PATH = OUTPUT_ROOT / "ea_hx_metropt_model.npz"
INNOVATION_PATH = OUTPUT_ROOT / "metropt_innovations.npz"
SCORE_PATH = OUTPUT_ROOT / "score_paths_v3.npz"
RESULT_PATH = OUTPUT_ROOT / "complete_result_v3.json"

FIT = ("2020-02-01T00:00:00", "2020-02-21T00:00:00")
REFERENCE = ("2020-02-21T00:00:00", "2020-03-01T00:00:00")
CALIBRATION = (
    ("2020-03-01T00:00:00", "2020-04-11T00:00:00"),
    ("2020-04-22T00:00:00", "2020-05-21T00:00:00"),
)
CONFIRMATION = ("2020-05-21T00:00:00", "2020-09-01T00:00:00")
DEVELOPMENT_EVENT = {
    "event_id": "F1",
    "start": "2020-04-18T00:00:00",
    "end": "2020-04-19T00:00:00",
    "fault_type": "air_leak",
}
CONFIRMATION_EVENTS = (
    {
        "event_id": "F2",
        "start": "2020-05-29T23:30:00",
        "end": "2020-05-30T06:00:00",
        "fault_type": "air_leak",
    },
    {
        "event_id": "F3",
        "start": "2020-06-05T10:00:00",
        "end": "2020-06-07T14:30:00",
        "fault_type": "air_leak",
    },
    {
        "event_id": "F4",
        "start": "2020-07-15T14:30:00",
        "end": "2020-07-15T19:00:00",
        "fault_type": "air_leak",
    },
)
METHODS = (
    "ea_hx_cusum",
    "predictive_residual_cusum",
    "pca_t2_plus_q",
    "energy_cusum",
)
TARGET_DIM = 13
MINIMUM_RUN_LENGTH = 75  # context 60 + largest horizon 15
ALPHA = 0.05
MINIMUM_VALID_SCORES_PER_DAY = 1141
RECOVERY_HOURS = 72
AVAILABILITY_LAG_MINUTES = 1


@dataclass(frozen=True)
class ScoreBlock:
    first: int
    last: int
    date: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    temporary.replace(path)


def _load_saved_audit() -> MetroPTTimestampAudit:
    document = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit = MetroPTTimestampAudit(
        source_path=document["source_path"],
        source_size_bytes=int(document["source_size_bytes"]),
        source_mtime_ns=int(document["source_mtime_ns"]),
        source_sha256=document["source_sha256"],
        header=tuple(document["header"]),
        row_count=int(document["row_count"]),
        first_timestamp=document["first_timestamp"],
        last_timestamp=document["last_timestamp"],
        minimum_interval_seconds=document["minimum_interval_seconds"],
        modal_interval_seconds=document["modal_interval_seconds"],
        modal_interval_count=int(document["modal_interval_count"]),
        maximum_interval_seconds=document["maximum_interval_seconds"],
        chunks_scanned=int(document["chunks_scanned"]),
        partition_row_counts=document["partition_row_counts"],
    )
    if audit.fingerprint != document["audit_fingerprint"]:
        raise ValueError("saved MetroPT timestamp audit fingerprint is invalid")
    return audit


def _bounds(timestamps: np.ndarray, start: str, stop: str) -> tuple[int, int]:
    first = int(np.searchsorted(timestamps, np.datetime64(start, "ns"), side="left"))
    last = int(np.searchsorted(timestamps, np.datetime64(stop, "ns"), side="left"))
    if not 0 <= first < last <= timestamps.size:
        raise ValueError(f"invalid interval bounds: {start}, {stop}")
    return first, last


def _feature_mask(
    interval: MetroPTMinuteInterval, features: np.ndarray
) -> np.ndarray:
    finite = np.isfinite(features)
    return finite & np.asarray(interval.eligible, dtype=bool)[:, None]


def _score_blocks(interval: MetroPTMinuteInterval) -> tuple[ScoreBlock, ...]:
    timestamps = np.asarray(interval.timestamps, dtype="datetime64[ns]")
    dates = timestamps.astype("datetime64[D]")
    boundary = np.asarray(interval.reset_before, dtype=bool).copy()
    boundary[0] = True
    boundary[1:] |= dates[1:] != dates[:-1]
    starts = np.flatnonzero(boundary)
    stops = np.r_[starts[1:], timestamps.size]
    return tuple(
        ScoreBlock(int(first), int(last), str(dates[first]))
        for first, last in zip(starts, stops, strict=True)
    )


def _training_runs(
    interval: MetroPTMinuteInterval,
    features: np.ndarray,
    mask: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    first, last = _bounds(interval.timestamps, *FIT)
    resets = np.flatnonzero(interval.reset_before[first + 1 : last]) + first + 1
    boundaries = np.r_[first, resets, last]
    x_runs: list[np.ndarray] = []
    c_runs: list[np.ndarray] = []
    mask_runs: list[np.ndarray] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < MINIMUM_RUN_LENGTH:
            continue
        local_mask = np.asarray(mask[left_i:right_i], dtype=bool)
        if int(np.count_nonzero(np.all(local_mask, axis=1))) < MINIMUM_RUN_LENGTH:
            continue
        x_runs.append(np.asarray(features[left_i:right_i], dtype=np.float32))
        c_runs.append(np.zeros((right_i - left_i, 1), dtype=np.float32))
        mask_runs.append(local_mask)
    if not x_runs:
        raise ValueError("MetroPT fit interval produced no usable causal runs")
    return tuple(x_runs), tuple(c_runs), tuple(mask_runs)


def _load_or_train_model(
    interval: MetroPTMinuteInterval,
    features: np.ndarray,
    mask: np.ndarray,
) -> CRISModel:
    if MODEL_PATH.exists():
        model = CRISModel.load_npz(MODEL_PATH)
        if model.config.target_dim != TARGET_DIM or model.config.covariate_dim != 1:
            raise ValueError("cached MetroPT model dimensions are incompatible")
        print(f"[model-cache] {MODEL_PATH}", flush=True)
        return model
    x_runs, c_runs, mask_runs = _training_runs(interval, features, mask)
    print(
        f"[train] {len(x_runs)} reset-safe runs, 40 deterministic epochs on CUDA",
        flush=True,
    )
    model = CRISModel.fit(
        x_runs,
        c_runs,
        mask_runs,
        config=CRISConfig(target_dim=TARGET_DIM, covariate_dim=1),
        training_config=CRISTrainingConfig(
            epochs=40,
            batch_size=1024,
            learning_rate=3.0e-4,
            weight_decay=1.0e-4,
            gradient_clip_norm=1.0,
            seed=20260915,
            device="cuda",
        ),
        metadata={
            "dataset": "MetroPT-3",
            "fit_interval": list(FIT),
            "targets": list(interval.feature_names),
            "covariate": "constant_zero_dummy",
            "method_role": "frozen_EA-HX_predictor",
        },
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    model.save_npz(MODEL_PATH)
    print(
        f"[model-saved] loss {model.loss_history[0]:.6f} -> "
        f"{model.loss_history[-1]:.6f}",
        flush=True,
    )
    return model


def _invalidate_cross_reset_predictions(
    innovations: np.ndarray,
    reset_before: np.ndarray,
    model: CRISModel,
) -> np.ndarray:
    output = np.array(innovations, dtype=np.float32, order="C", copy=True)
    for reset in np.flatnonzero(reset_before):
        for horizon_index, horizon in enumerate(model.config.horizons):
            invalid_length = model.config.context_length - 1 + int(horizon)
            output[int(reset) : min(output.shape[0], int(reset) + invalid_length), horizon_index] = np.nan
    return output


def _load_or_predict(
    model: CRISModel,
    interval: MetroPTMinuteInterval,
    features: np.ndarray,
    mask: np.ndarray,
    model_hash: str,
    source_hash: str,
) -> np.ndarray:
    if INNOVATION_PATH.exists():
        with np.load(INNOVATION_PATH, allow_pickle=False) as archive:
            expected = {"innovations", "model_sha256", "source_sha256"}
            if set(archive.files) != expected:
                raise ValueError("invalid MetroPT innovation cache schema")
            if str(np.asarray(archive["model_sha256"]).item()) != model_hash:
                raise ValueError("innovation cache belongs to a different model")
            if str(np.asarray(archive["source_sha256"]).item()) != source_hash:
                raise ValueError("innovation cache belongs to a different MetroPT file")
            result = np.array(archive["innovations"], dtype=np.float32, copy=True)
        if result.shape != (interval.timestamps.size, 3, TARGET_DIM):
            raise ValueError("cached MetroPT innovations have the wrong shape")
        print(f"[innovation-cache] {INNOVATION_PATH}", flush=True)
        return result
    print(f"[predict] {interval.timestamps.size} one-minute rows on CUDA", flush=True)
    raw = model.innovations(
        features,
        np.zeros((interval.timestamps.size, 1), dtype=np.float32),
        mask,
        device="cuda",
        batch_size=4096,
    )
    result = _invalidate_cross_reset_predictions(raw, interval.reset_before, model)
    temporary = INNOVATION_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        innovations=result,
        model_sha256=np.asarray(model_hash),
        source_sha256=np.asarray(source_hash),
    )
    temporary.replace(INNOVATION_PATH)
    print(f"[innovation-saved] {INNOVATION_PATH}", flush=True)
    return result


def _training_fill(
    interval: MetroPTMinuteInterval,
    features: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first, last = _bounds(interval.timestamps, *FIT)
    center = np.empty(TARGET_DIM, dtype=np.float64)
    values = np.asarray(features, dtype=np.float64)
    for feature in range(TARGET_DIM):
        observed = values[first:last, feature][mask[first:last, feature]]
        if observed.size == 0:
            raise ValueError(f"feature {feature} is never observed in model fit")
        center[feature] = float(np.median(observed))
    filled = np.where(mask, values, center[None, :])
    if not np.isfinite(filled).all():
        raise ValueError("filled MetroPT features are not finite")
    return filled, center


def _reference_slices(
    interval: MetroPTMinuteInterval,
    blocks: Iterable[ScoreBlock],
) -> tuple[slice, ...]:
    first, last = _bounds(interval.timestamps, *REFERENCE)
    selected = tuple(
        slice(block.first, block.last)
        for block in blocks
        if block.first >= first and block.last <= last
    )
    if not selected:
        raise ValueError("no reference score blocks")
    return selected


def _fit_methods(
    interval: MetroPTMinuteInterval,
    innovations: np.ndarray,
    filled: np.ndarray,
    blocks: tuple[ScoreBlock, ...],
    score_valid: np.ndarray,
) -> dict[str, Any]:
    slices = _reference_slices(interval, blocks)
    innovation_runs = tuple(innovations[value] for value in slices)
    feature_runs = tuple(
        filled[value][score_valid[value]]
        for value in slices
        if np.any(score_valid[value])
    )
    return {
        "ea_hx_cusum": EnergyAnchoredHXMonitor.fit(innovation_runs),
        "predictive_residual_cusum": PredictiveResidualCUSUMBaseline.fit(
            innovation_runs
        ),
        "pca_t2_plus_q": PCAT2QBaseline.fit(feature_runs),
        "energy_cusum": EnergyCUSUMBaseline.fit(feature_runs),
    }


def _score_all(
    methods: Mapping[str, Any],
    innovations: np.ndarray,
    filled: np.ndarray,
    blocks: tuple[ScoreBlock, ...],
    score_valid: np.ndarray,
    model_hash: str,
    source_hash: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if SCORE_PATH.exists():
        with np.load(SCORE_PATH, allow_pickle=False) as archive:
            expected = {*METHODS, "block_start", "model_sha256", "source_sha256"}
            if set(archive.files) != expected:
                raise ValueError("invalid MetroPT score cache schema")
            if str(np.asarray(archive["model_sha256"]).item()) != model_hash:
                raise ValueError("score cache belongs to a different model")
            if str(np.asarray(archive["source_sha256"]).item()) != source_hash:
                raise ValueError("score cache belongs to a different MetroPT file")
            paths = {
                name: np.array(archive[name], dtype=np.float64, copy=True)
                for name in METHODS
            }
            block_start = np.array(archive["block_start"], dtype=bool, copy=True)
        print(f"[score-cache] {SCORE_PATH}", flush=True)
        return paths, block_start

    paths = {
        name: np.zeros(innovations.shape[0], dtype=np.float64) for name in METHODS
    }
    block_start = np.zeros(innovations.shape[0], dtype=bool)
    for index, block in enumerate(blocks, start=1):
        block_start[block.first] = True
        innovation_run = innovations[block.first : block.last]
        feature_run = filled[block.first : block.last]
        for name, method in methods.items():
            values = (
                innovation_run
                if name in {"ea_hx_cusum", "predictive_residual_cusum"}
                else feature_run
            )
            score = method.score(values)
            local_valid = score_valid[block.first : block.last]
            if name == "energy_cusum":
                state = 0.0
                local_path = np.zeros(local_valid.size, dtype=np.float64)
                for local_index, (increment, valid) in enumerate(
                    zip(score.point_score, local_valid, strict=True)
                ):
                    if valid:
                        state = max(0.0, state + float(increment))
                    local_path[local_index] = state
                paths[name][block.first : block.last] = local_path
            else:
                paths[name][block.first : block.last] = score.path
        if index == 1 or index % 50 == 0 or index == len(blocks):
            print(f"[score] {index}/{len(blocks)} reset/day blocks", flush=True)
    temporary = SCORE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **paths,
        block_start=block_start,
        model_sha256=np.asarray(model_hash),
        source_sha256=np.asarray(source_hash),
    )
    temporary.replace(SCORE_PATH)
    return paths, block_start


def _days(start: str, stop: str) -> np.ndarray:
    return np.arange(np.datetime64(start, "D"), np.datetime64(stop, "D"), dtype="datetime64[D]")


def _daily_maxima(
    path: np.ndarray,
    timestamps: np.ndarray,
    score_valid: np.ndarray,
    ranges: Iterable[tuple[str, str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dates = timestamps.astype("datetime64[D]")
    requested = np.concatenate([_days(start, stop) for start, stop in ranges])
    maxima: list[float] = []
    records: list[dict[str, Any]] = []
    for date in requested:
        selected = (dates == date) & score_valid
        count = int(np.count_nonzero(selected))
        qualified = count >= MINIMUM_VALID_SCORES_PER_DAY
        maximum = None if count == 0 else float(np.max(path[selected]))
        if qualified:
            assert maximum is not None
            maxima.append(maximum)
        records.append(
            {
                "date": str(date),
                "score_count": count,
                "minimum_required": MINIMUM_VALID_SCORES_PER_DAY,
                "qualified": qualified,
                "maximum": maximum,
            }
        )
    if not maxima:
        raise ValueError("no coverage-qualified calibration days")
    return np.asarray(maxima, dtype=np.float64), records


def _threshold(maxima: np.ndarray, alpha: float) -> tuple[float, int]:
    rank = int(math.ceil((maxima.size + 1) * (1.0 - alpha)))
    if rank > maxima.size:
        raise ValueError("not enough calibration days for requested alpha")
    return float(np.sort(maxima, kind="stable")[rank - 1]), rank


def _rising_edges(
    path: np.ndarray,
    threshold: float,
    block_start: np.ndarray,
    score_valid: np.ndarray,
) -> np.ndarray:
    alarm = np.asarray(path > threshold, dtype=bool)
    edges = np.zeros(alarm.size, dtype=bool)
    previous_valid_alarm = False
    for index in range(alarm.size):
        if block_start[index]:
            previous_valid_alarm = False
        if not score_valid[index]:
            continue
        if alarm[index] and not previous_valid_alarm:
            edges[index] = True
        previous_valid_alarm = bool(alarm[index])
    return edges


def _event_metrics(
    path: np.ndarray,
    threshold: float,
    timestamps: np.ndarray,
    block_start: np.ndarray,
    score_valid: np.ndarray,
    event: Mapping[str, str],
) -> dict[str, Any]:
    start = np.datetime64(event["start"], "ns")
    stop = np.datetime64(event["end"], "ns")
    event_mask = (timestamps >= start) & (timestamps < stop)
    edges = _rising_edges(path, threshold, block_start, score_valid)
    availability_grid = timestamps + np.timedelta64(AVAILABILITY_LAG_MINUTES, "m")
    found = np.flatnonzero(edges & event_mask & (availability_grid < stop))
    first = None if found.size == 0 else int(found[0])
    preceding = (timestamps >= start - np.timedelta64(24, "h")) & (timestamps < start)
    valid_event = event_mask & score_valid
    event_max = None if not np.any(valid_event) else float(np.max(path[valid_event]))
    availability = (
        None
        if first is None
        else timestamps[first] + np.timedelta64(AVAILABILITY_LAG_MINUTES, "m")
    )
    return {
        "event_id": event["event_id"],
        "fault_type": event["fault_type"],
        "start": event["start"],
        "end": event["end"],
        "detected": first is not None,
        "first_alarm_score_timestamp": None if first is None else str(timestamps[first]),
        "first_alarm_availability_timestamp": (
            None if availability is None else str(availability)
        ),
        "delay_minutes": (
            None
            if first is None
            else float((availability - start) / np.timedelta64(1, "m"))
        ),
        "valid_score_count": int(np.count_nonzero(valid_event)),
        "alarm_edges_in_preceding_24h": int(np.count_nonzero(edges & preceding)),
        "maximum_score": event_max,
        "maximum_to_threshold_ratio": (
            None if threshold <= 0.0 or event_max is None else event_max / threshold
        ),
    }


def _event_and_recovery_days(events: Iterable[Mapping[str, str]]) -> set[str]:
    output: set[str] = set()
    for event in events:
        start = np.datetime64(event["start"], "ns")
        stop = np.datetime64(event["end"], "ns") + np.timedelta64(
            RECOVERY_HOURS, "h"
        )
        first_day = start.astype("datetime64[D]")
        last_day = (stop - np.timedelta64(1, "ns")).astype("datetime64[D]")
        for date in np.arange(
            first_day, last_day + np.timedelta64(1, "D"), dtype="datetime64[D]"
        ):
            output.add(str(date))
    return output


def _confirmation_healthy_metrics(
    path: np.ndarray,
    threshold: float,
    timestamps: np.ndarray,
    block_start: np.ndarray,
    score_valid: np.ndarray,
) -> dict[str, Any]:
    dates = timestamps.astype("datetime64[D]")
    excluded_days = _event_and_recovery_days(CONFIRMATION_EVENTS)
    candidate_days = [
        date for date in _days(*CONFIRMATION) if str(date) not in excluded_days
    ]
    edges = _rising_edges(path, threshold, block_start, score_valid)
    alarm = (path > threshold) & score_valid
    alarm_days = 0
    episode_count = 0
    alarm_points = 0
    point_count = 0
    qualified_days = 0
    unqualified_days = 0
    for date in candidate_days:
        selected = (dates == date) & score_valid
        score_count = int(np.count_nonzero(selected))
        if score_count < MINIMUM_VALID_SCORES_PER_DAY:
            unqualified_days += 1
            continue
        qualified_days += 1
        has_alarm = bool(np.any(alarm[selected]))
        alarm_days += int(has_alarm)
        episode_count += int(np.count_nonzero(edges[selected]))
        alarm_points += int(np.count_nonzero(alarm[selected]))
        point_count += score_count
    if qualified_days == 0 or point_count == 0:
        raise ValueError("confirmation produced no qualified healthy days")
    return {
        "candidate_event_free_day_count": len(candidate_days),
        "qualified_healthy_day_count": qualified_days,
        "coverage_unqualified_day_count": unqualified_days,
        "alarmed_day_count": alarm_days,
        "day_level_false_alarm_rate": alarm_days / qualified_days,
        "alarm_episode_count": episode_count,
        "point_alarm_occupancy": alarm_points / point_count,
        "excluded_fault_or_72h_recovery_days": sorted(excluded_days),
    }


def _method_result(
    name: str,
    path: np.ndarray,
    timestamps: np.ndarray,
    block_start: np.ndarray,
    score_valid: np.ndarray,
) -> dict[str, Any]:
    dev_maxima, dev_records = _daily_maxima(
        path, timestamps, score_valid, (REFERENCE,)
    )
    dev_threshold, dev_rank = float(np.max(dev_maxima)), int(dev_maxima.size)
    development = _event_metrics(
        path,
        dev_threshold,
        timestamps,
        block_start,
        score_valid,
        DEVELOPMENT_EVENT,
    )
    calibration_maxima, calibration_records = _daily_maxima(
        path, timestamps, score_valid, CALIBRATION
    )
    final_threshold, final_rank = _threshold(calibration_maxima, ALPHA)
    events = [
        _event_metrics(
            path, final_threshold, timestamps, block_start, score_valid, event
        )
        for event in CONFIRMATION_EVENTS
    ]
    delays = [float(item["delay_minutes"]) for item in events if item["detected"]]
    healthy = _confirmation_healthy_metrics(
        path, final_threshold, timestamps, block_start, score_valid
    )
    return {
        "method": name,
        "development": {
            "healthy_day_count": int(dev_maxima.size),
            "threshold_rule": "maximum_of_qualified_development_healthy_days",
            "threshold_rank": dev_rank,
            "threshold": dev_threshold,
            "daily_maxima": dev_records,
            "F1": development,
        },
        "calibration": {
            "healthy_day_count": int(calibration_maxima.size),
            "alpha": ALPHA,
            "rank": final_rank,
            "threshold": final_threshold,
            "daily_maxima": calibration_records,
        },
        "confirmation": {
            "events": events,
            "event_recall": sum(bool(item["detected"]) for item in events) / len(events),
            "median_detected_delay_minutes": None if not delays else float(np.median(delays)),
            "healthy": healthy,
        },
    }


def main() -> int:
    started = time.perf_counter()
    if RESULT_PATH.exists():
        print(f"[complete-cache] {RESULT_PATH}", flush=True)
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = _load_saved_audit()
    print("[data] authenticating and loading one causal MetroPT minute grid", flush=True)
    interval = load_metropt_interval(
        CSV_PATH,
        FIT[0],
        CONFIRMATION[1],
        chunksize=100_000,
        timestamp_audit=audit,
    )
    features = np.asarray(interval.features, dtype=np.float32)
    mask = _feature_mask(interval, features)
    blocks = _score_blocks(interval)
    source_hash = audit.source_sha256
    model = _load_or_train_model(interval, features, mask)
    model_hash = _sha256(MODEL_PATH)
    print(f"[gpu] {model.attestation.cuda_device_name}", flush=True)
    innovations = _load_or_predict(
        model, interval, features, mask, model_hash, source_hash
    )
    filled, fill_center = _training_fill(interval, features, mask)
    score_valid = np.asarray(interval.eligible, dtype=bool) & np.any(
        np.isfinite(innovations), axis=(1, 2)
    )
    methods = _fit_methods(interval, innovations, filled, blocks, score_valid)
    paths, block_start = _score_all(
        methods,
        innovations,
        filled,
        blocks,
        score_valid,
        model_hash,
        source_hash,
    )
    results = {
        name: _method_result(
            name, paths[name], interval.timestamps, block_start, score_valid
        )
        for name in METHODS
    }
    proposed = results["ea_hx_cusum"]["confirmation"]
    anchor = results["predictive_residual_cusum"]["confirmation"]
    document = {
        "schema": "ea_hx_metropt_complete_external_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_seconds": time.perf_counter() - started,
        "device": {
            "requested": model.attestation.requested_device,
            "source": model.attestation.source_device,
            "name": model.attestation.cuda_device_name,
            "torch": model.attestation.torch_version,
            "cuda_runtime": model.attestation.cuda_runtime,
        },
        "data": {
            "dataset": "UCI MetroPT-3 air compressor",
            "source_sha256": source_hash,
            "minute_rows": int(interval.timestamps.size),
            "eligible_rows": int(np.count_nonzero(interval.eligible)),
            "reset_or_day_score_blocks": len(blocks),
            "feature_names": list(interval.feature_names),
            "direct_alarm_channels_used": False,
            "fit_interval": list(FIT),
            "monitor_reference_interval": list(REFERENCE),
            "calibration_intervals": [list(value) for value in CALIBRATION],
            "confirmation_interval": list(CONFIRMATION),
        },
        "frozen_method": {
            "id": "EA-HX-CUSUM",
            "predictor": "two-layer causal GRU Student-t, horizons [1,5,15] minutes",
            "energy_anchor": "predictive_residual_cusum",
            "structure_targets": list(interval.feature_names[:4]),
            "fusion": "S_E + 0.25 * min(S_E, S_HX)",
            "fit_feature_medians_for_missing_raw_baselines": fill_center.tolist(),
        },
        "methods": results,
        "proposed_minus_anchor": {
            "event_recall": proposed["event_recall"] - anchor["event_recall"],
            "healthy_day_false_alarm_rate": (
                proposed["healthy"]["day_level_false_alarm_rate"]
                - anchor["healthy"]["day_level_false_alarm_rate"]
            ),
            "median_detected_delay_minutes": (
                None
                if proposed["median_detected_delay_minutes"] is None
                or anchor["median_detected_delay_minutes"] is None
                else proposed["median_detected_delay_minutes"]
                - anchor["median_detected_delay_minutes"]
            ),
        },
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(PROJECT)),
            "innovations": str(INNOVATION_PATH.relative_to(PROJECT)),
            "score_paths": str(SCORE_PATH.relative_to(PROJECT)),
        },
    }
    _write_json(RESULT_PATH, document)
    print(f"[complete] {RESULT_PATH}", flush=True)
    for name in METHODS:
        confirmation = results[name]["confirmation"]
        print(
            f"[result] {name}: recall={confirmation['event_recall']:.3f}, "
            f"median_delay={confirmation['median_detected_delay_minutes']}, "
            f"healthy_day_FA={confirmation['healthy']['day_level_false_alarm_rate']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
