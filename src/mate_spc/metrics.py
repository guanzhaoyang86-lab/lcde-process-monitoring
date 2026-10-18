"""Point- and event-level monitoring metrics without point-adjustment leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def contiguous_regions(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous True regions."""
    binary = np.asarray(values, dtype=bool)
    if binary.ndim != 1:
        raise ValueError("values must be one-dimensional")
    padded = np.r_[False, binary, False].astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _merge_alarm_episodes(alarm: np.ndarray, refractory: int) -> list[tuple[int, int]]:
    regions = contiguous_regions(alarm)
    if not regions:
        return []
    merged = [regions[0]]
    for start, stop in regions[1:]:
        old_start, old_stop = merged[-1]
        if start - old_stop <= refractory:
            merged[-1] = (old_start, stop)
        else:
            merged.append((start, stop))
    return merged


@dataclass(frozen=True)
class EventMetrics:
    event_recall: float
    events_detected: int
    event_count: int
    median_delay_samples: float
    mean_lead_samples: float
    false_alarm_episodes: int
    false_alarm_episodes_per_100h: float
    normal_alarm_occupancy: float
    point_precision: float
    point_recall: float
    point_f1: float
    decision_coverage: float
    monitored_normal_hours: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "event_recall": self.event_recall,
            "events_detected": self.events_detected,
            "event_count": self.event_count,
            "median_delay_samples": self.median_delay_samples,
            "mean_lead_samples": self.mean_lead_samples,
            "false_alarm_episodes": self.false_alarm_episodes,
            "false_alarm_episodes_per_100h": self.false_alarm_episodes_per_100h,
            "normal_alarm_occupancy": self.normal_alarm_occupancy,
            "point_precision": self.point_precision,
            "point_recall": self.point_recall,
            "point_f1": self.point_f1,
            "decision_coverage": self.decision_coverage,
            "monitored_normal_hours": self.monitored_normal_hours,
        }


def event_metrics(
    alarm: np.ndarray,
    fault: np.ndarray,
    sample_period_seconds: float,
    warning_horizon: int = 0,
    refractory: int = 0,
    decision_mask: np.ndarray | None = None,
) -> EventMetrics:
    """Evaluate alarms as operational episodes and fault events.

    An event is detected when at least one alarm occurs from ``warning_horizon``
    samples before its start through its end.  Positive lead means an early alarm;
    otherwise delay is measured from event onset.  No post-hoc point adjustment is
    applied.
    """
    predicted = np.asarray(alarm, dtype=bool)
    target = np.asarray(fault, dtype=bool)
    if predicted.ndim != 1 or target.shape != predicted.shape:
        raise ValueError("alarm and fault must be aligned one-dimensional arrays")
    monitored = (
        np.ones_like(predicted, dtype=bool)
        if decision_mask is None
        else np.asarray(decision_mask, dtype=bool)
    )
    if monitored.shape != predicted.shape:
        raise ValueError("decision_mask must match alarm")
    if np.any(predicted & ~monitored):
        raise ValueError("alarms cannot occur where the monitor abstained")
    if (
        not np.isfinite(sample_period_seconds)
        or sample_period_seconds <= 0.0
        or isinstance(warning_horizon, (bool, np.bool_))
        or not isinstance(warning_horizon, (int, np.integer))
        or warning_horizon < 0
        or isinstance(refractory, (bool, np.bool_))
        or not isinstance(refractory, (int, np.integer))
        or refractory < 0
    ):
        raise ValueError("invalid timing argument")

    events = contiguous_regions(target)
    alarm_episodes = _merge_alarm_episodes(predicted & monitored, refractory)
    used_alarm_episodes: set[int] = set()
    detected = 0
    delays: list[float] = []
    leads: list[float] = []
    matched_alarm = np.zeros_like(predicted)
    for start, stop in events:
        search_start = max(0, start - warning_horizon)
        candidates: list[tuple[int, int]] = []
        for episode_index, (episode_start, episode_stop) in enumerate(alarm_episodes):
            if episode_index in used_alarm_episodes:
                continue
            # Attribute by episode onset, not by any later point in a nuisance
            # alarm that happened to persist into the event.
            if not search_start <= episode_start < stop:
                continue
            if episode_start < start and target[episode_start]:
                continue
            candidates.append((episode_start, episode_index))
        if candidates:
            first, episode_index = min(candidates)
            used_alarm_episodes.add(episode_index)
            detected += 1
            episode_start, episode_stop = alarm_episodes[episode_index]
            matched_stop = min(episode_stop, stop)
            matched_alarm[episode_start:matched_stop] |= predicted[episode_start:matched_stop]
            if first < start:
                leads.append(float(start - first))
                delays.append(0.0)
            else:
                leads.append(0.0)
                delays.append(float(first - start))

    normal = ~target & monitored
    normal_hours = float(normal.sum()) * sample_period_seconds / 3600.0
    false_alarm_signal = predicted & normal & ~matched_alarm
    episodes = _merge_alarm_episodes(false_alarm_signal, refractory)
    false_rate = len(episodes) * 100.0 / normal_hours if normal_hours > 0 else float("nan")

    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    fn = int(np.sum(~predicted & target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMetrics(
        event_recall=detected / len(events) if events else float("nan"),
        events_detected=detected,
        event_count=len(events),
        median_delay_samples=float(np.median(delays)) if delays else float("nan"),
        mean_lead_samples=float(np.mean(leads)) if leads else float("nan"),
        false_alarm_episodes=len(episodes),
        false_alarm_episodes_per_100h=false_rate,
        normal_alarm_occupancy=float(np.mean(predicted[normal])) if normal.any() else float("nan"),
        point_precision=precision,
        point_recall=recall,
        point_f1=f1,
        decision_coverage=float(np.mean(monitored)),
        monitored_normal_hours=normal_hours,
    )
