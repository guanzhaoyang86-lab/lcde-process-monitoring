"""Frozen deployment and prefix-applicability contracts for PAC monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any, Sequence

import numpy as np


ROBUST_SCALE_FACTOR = 1.4826
CLIP_LIMIT = 8.0
PERSISTENCE_WINDOW = 3
GATE_PASS = "PASS"
GATE_DEFER = "DEFER"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def probability(value: Any, *, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    if 1.0 - result == 1.0:
        raise ValueError(
            f"{name} is below the binary64 range supported for PAC certification"
        )
    return result


def validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def readonly_vector(
    value: float | Sequence[float] | np.ndarray,
    *,
    head_count: int,
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.repeat(array.reshape(1), head_count)
    if array.shape != (head_count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or length-{head_count} vector")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DeploymentSpec:
    """Exact upstream and horizon contract for one deployable monitor."""

    run_length: int
    valid_start: int
    monitor_start: int
    head_ids: tuple[str, ...]
    upstream_artifact_sha256: str
    clip_limit: float = CLIP_LIMIT
    persistence_window: int = PERSISTENCE_WINDOW

    def __post_init__(self) -> None:
        run_length = positive_integer(self.run_length, name="run_length")
        valid_start = nonnegative_integer(self.valid_start, name="valid_start")
        monitor_start = nonnegative_integer(self.monitor_start, name="monitor_start")
        if monitor_start <= valid_start:
            raise ValueError("monitor_start must be strictly greater than valid_start")
        persistence = positive_integer(
            self.persistence_window,
            name="persistence_window",
        )
        if persistence != PERSISTENCE_WINDOW:
            raise ValueError(f"persistence_window is frozen at {PERSISTENCE_WINDOW}")
        if run_length < monitor_start + persistence:
            raise ValueError("run_length must include a complete persistence window")
        clip_limit = float(self.clip_limit)
        if not isfinite(clip_limit) or clip_limit != CLIP_LIMIT:
            raise ValueError(f"clip_limit is frozen at {CLIP_LIMIT:g}")
        if isinstance(self.head_ids, (str, bytes)):
            raise ValueError("head_ids must be an ordered sequence of identifiers")
        head_ids = tuple(self.head_ids)
        if not head_ids or any(not isinstance(item, str) or not item for item in head_ids):
            raise ValueError("head_ids must contain nonempty strings")
        if len(set(head_ids)) != len(head_ids):
            raise ValueError("head_ids must be unique and order-explicit")
        upstream = validate_sha256(
            self.upstream_artifact_sha256,
            name="upstream_artifact_sha256",
        )
        object.__setattr__(self, "run_length", run_length)
        object.__setattr__(self, "valid_start", valid_start)
        object.__setattr__(self, "monitor_start", monitor_start)
        object.__setattr__(self, "head_ids", head_ids)
        object.__setattr__(self, "upstream_artifact_sha256", upstream)
        object.__setattr__(self, "clip_limit", clip_limit)
        object.__setattr__(self, "persistence_window", persistence)

    @property
    def head_count(self) -> int:
        return len(self.head_ids)

    @property
    def fingerprint(self) -> str:
        payload = {
            "clip_limit_hex": self.clip_limit.hex(),
            "head_ids": list(self.head_ids),
            "monitor_start": self.monitor_start,
            "persistence_window": self.persistence_window,
            "run_length": self.run_length,
            "upstream_artifact_sha256": self.upstream_artifact_sha256,
            "valid_start": self.valid_start,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrefixApplicabilityConfig:
    """Predeclared inclusive healthy-reference bounds for prefix statistics."""

    head_count: int
    median_lower: np.ndarray
    median_upper: np.ndarray
    robust_scale_lower: np.ndarray
    robust_scale_upper: np.ndarray
    minimum_prefix_observations: int

    def __post_init__(self) -> None:
        head_count = positive_integer(self.head_count, name="head_count")
        minimum = positive_integer(
            self.minimum_prefix_observations,
            name="minimum_prefix_observations",
        )
        lower = readonly_vector(self.median_lower, head_count=head_count, name="median_lower")
        upper = readonly_vector(self.median_upper, head_count=head_count, name="median_upper")
        scale_lower = readonly_vector(
            self.robust_scale_lower,
            head_count=head_count,
            name="robust_scale_lower",
            nonnegative=True,
        )
        scale_upper = readonly_vector(
            self.robust_scale_upper,
            head_count=head_count,
            name="robust_scale_upper",
            nonnegative=True,
        )
        if np.any(lower > upper):
            raise ValueError("median_lower must not exceed median_upper")
        if np.any(scale_lower > scale_upper):
            raise ValueError("robust_scale_lower must not exceed robust_scale_upper")
        object.__setattr__(self, "head_count", head_count)
        object.__setattr__(self, "median_lower", lower)
        object.__setattr__(self, "median_upper", upper)
        object.__setattr__(self, "robust_scale_lower", scale_lower)
        object.__setattr__(self, "robust_scale_upper", scale_upper)
        object.__setattr__(self, "minimum_prefix_observations", minimum)


def monitor_configuration_sha256(
    deployment: DeploymentSpec,
    prefix_gate: PrefixApplicabilityConfig,
    kappa: np.ndarray,
) -> str:
    """Bind every value that changes the path or its applicability decision."""
    if prefix_gate.head_count != deployment.head_count:
        raise ValueError("prefix gate and deployment head counts differ")
    allowance = readonly_vector(
        kappa,
        head_count=deployment.head_count,
        name="kappa",
        nonnegative=True,
    )
    payload = {
        "deployment_spec_sha256": deployment.fingerprint,
        "kappa_hex": [value.hex() for value in allowance],
        "median_lower_hex": [value.hex() for value in prefix_gate.median_lower],
        "median_upper_hex": [value.hex() for value in prefix_gate.median_upper],
        "minimum_prefix_observations": prefix_gate.minimum_prefix_observations,
        "robust_scale_lower_hex": [
            value.hex() for value in prefix_gate.robust_scale_lower
        ],
        "robust_scale_upper_hex": [
            value.hex() for value in prefix_gate.robust_scale_upper
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrefixGateResult:
    status: str
    reasons: tuple[str, ...]
    prefix_count: int
    median: np.ndarray
    robust_scale: np.ndarray
    bias: np.ndarray
    positive_scale: np.ndarray

    @property
    def passed(self) -> bool:
        return self.status == GATE_PASS


def prefix_gate_result(
    prefix: np.ndarray,
    config: PrefixApplicabilityConfig,
) -> PrefixGateResult:
    median = np.median(prefix, axis=0)
    mad = np.median(np.abs(prefix - median), axis=0)
    robust_scale = ROBUST_SCALE_FACTOR * mad
    bias = np.maximum(0.0, median)
    positive_scale = np.maximum(1.0, robust_scale)
    reasons: list[str] = []
    if prefix.shape[0] < config.minimum_prefix_observations:
        reasons.append(
            f"prefix_too_short:{prefix.shape[0]}<{config.minimum_prefix_observations}"
        )
    checks = (
        (median < config.median_lower, "median_below_reference"),
        (median > config.median_upper, "median_above_reference"),
        (robust_scale < config.robust_scale_lower, "scale_below_reference"),
        (robust_scale > config.robust_scale_upper, "scale_above_reference"),
    )
    for failed, label in checks:
        indices = np.flatnonzero(failed)
        if indices.size:
            reasons.append(f"{label}:{','.join(str(int(index)) for index in indices)}")
    arrays: list[np.ndarray] = []
    for value in (median, robust_scale, bias, positive_scale):
        array = np.array(value, dtype=np.float64, copy=True)
        array.setflags(write=False)
        arrays.append(array)
    return PrefixGateResult(
        status=GATE_PASS if not reasons else GATE_DEFER,
        reasons=tuple(reasons),
        prefix_count=int(prefix.shape[0]),
        median=arrays[0],
        robust_scale=arrays[1],
        bias=arrays[2],
        positive_scale=arrays[3],
    )
