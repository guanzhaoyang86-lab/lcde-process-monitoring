"""Pure-logic portability audit for the MetroPT CARED-BC development run.

This module deliberately has no data-loading or artifact-writing code.  It
accepts complete in-memory CPU and CUDA paths for the development-healthy and
F1 splits, derives the portability-adjusted threshold, and returns one strict
JSON record.  Invalid evidence is rejected before a decision can be produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np


PRIMITIVE_MAX_ABS_TOLERANCE = 1.0e-9
BOUNDARY_DISTANCE_TOLERANCE = 1.0e-9
AVAILABILITY_LAG_MINUTES = 1
F1_MAX_DELAY_MINUTES = 60.0
F1_MIN_PEAK_RATIO = 1.25
FORECAST_HEAD_COUNT = 3
ANALOG_COUNT = 7
PREDICTION_RESIDUAL_COUNT = FORECAST_HEAD_COUNT * ANALOG_COUNT
PHYSICAL_RESIDUAL_COUNT = 10
MONITORING_RESIDUAL_COUNT = PREDICTION_RESIDUAL_COUNT + PHYSICAL_RESIDUAL_COUNT
SCHEMA = "mate_spc_metropt_cared_development_portability_audit"
SCHEMA_VERSION = 1


class PortabilityValidationError(ValueError):
    """Raised when supplied audit evidence is malformed or non-finite."""


@dataclass(frozen=True, init=False)
class CertifiedDevelopmentPortabilityAudit:
    """Opaque, immutable result produced only by the trusted audit function.

    The canonical JSON payload prevents a caller from mutating a returned
    dictionary after certification.  Runners must still recompute this object
    from persisted primitive evidence on completed re-entry; its type is not a
    substitute for semantic verification.
    """

    _canonical_json: str

    @classmethod
    def _from_record(
        cls, record: Mapping[str, Any]
    ) -> "CertifiedDevelopmentPortabilityAudit":
        try:
            payload = json.dumps(
                dict(record),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise PortabilityValidationError(
                "certified audit record is not strict JSON"
            ) from exc
        instance = object.__new__(cls)
        object.__setattr__(instance, "_canonical_json", payload)
        return instance

    @property
    def passed(self) -> bool:
        return bool(self.as_dict()["passed"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True)
class PortabilityPath:
    """Complete portability evidence for one split on one device.

    ``prediction`` may have a different leading length from the score path.
    The score-aligned arrays ``residual``, ``timestamps``, ``indices``,
    ``state_codes``, ``reset_before``, and ``envelope`` must all share their
    first dimension.
    """

    prediction: np.ndarray
    residual: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    state_codes: np.ndarray
    reset_before: np.ndarray
    envelope: np.ndarray


@dataclass(frozen=True)
class _ValidatedPath:
    prediction: np.ndarray
    residual: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    state_codes: np.ndarray
    reset_before: np.ndarray
    envelope: np.ndarray


def _float_array(value: Any, *, name: str, minimum_ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityValidationError(f"{name} must be a numeric array") from exc
    if array.ndim < minimum_ndim or array.size == 0:
        raise PortabilityValidationError(
            f"{name} must be non-empty with at least {minimum_ndim} dimension(s)"
        )
    if not np.all(np.isfinite(array)):
        raise PortabilityValidationError(f"{name} contains a non-finite value")
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _integer_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise PortabilityValidationError(f"{name} must have an integer dtype")
    if raw.shape != (length,):
        raise PortabilityValidationError(f"{name} must have shape ({length},)")
    try:
        result = np.asarray(raw, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityValidationError(f"{name} cannot be represented as int64") from exc
    if not np.array_equal(raw, result):
        raise PortabilityValidationError(f"{name} cannot be represented exactly as int64")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _boolean_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (length,):
        raise PortabilityValidationError(f"{name} must be a boolean vector of length {length}")
    result = np.array(raw, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _timestamp_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    try:
        timestamps = np.asarray(value, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityValidationError(f"{name} must contain valid timestamps") from exc
    if timestamps.shape != (length,):
        raise PortabilityValidationError(f"{name} must have shape ({length},)")
    if np.any(np.isnat(timestamps)):
        raise PortabilityValidationError(f"{name} contains NaT")
    if length > 1 and np.any(np.diff(timestamps) <= np.timedelta64(0, "ns")):
        raise PortabilityValidationError(f"{name} must be strictly increasing")
    result = np.array(timestamps, dtype="datetime64[ns]", copy=True)
    result.setflags(write=False)
    return result


def _validate_path(path: PortabilityPath, *, name: str) -> _ValidatedPath:
    if not isinstance(path, PortabilityPath):
        raise PortabilityValidationError(f"{name} must be a PortabilityPath")
    prediction = _float_array(path.prediction, name=f"{name}.prediction", minimum_ndim=1)
    residual = _float_array(path.residual, name=f"{name}.residual", minimum_ndim=1)
    envelope = _float_array(path.envelope, name=f"{name}.envelope", minimum_ndim=1)
    if prediction.ndim != 3 or prediction.shape[1:] != (
        FORECAST_HEAD_COUNT,
        ANALOG_COUNT,
    ):
        raise PortabilityValidationError(
            f"{name}.prediction must have shape (N, {FORECAST_HEAD_COUNT}, {ANALOG_COUNT})"
        )
    if residual.ndim != 2 or residual.shape[1] != MONITORING_RESIDUAL_COUNT:
        raise PortabilityValidationError(
            f"{name}.residual must have shape (M, {MONITORING_RESIDUAL_COUNT})"
        )
    if envelope.ndim != 1:
        raise PortabilityValidationError(f"{name}.envelope must be one-dimensional")
    count = int(envelope.shape[0])
    if residual.shape[0] != count:
        raise PortabilityValidationError(
            f"{name}.residual first dimension must equal envelope length {count}"
        )
    if np.any(envelope < 0.0):
        raise PortabilityValidationError(f"{name}.envelope must be non-negative")
    timestamps = _timestamp_vector(path.timestamps, name=f"{name}.timestamps", length=count)
    indices = _integer_vector(path.indices, name=f"{name}.indices", length=count)
    state_codes = _integer_vector(
        path.state_codes, name=f"{name}.state_codes", length=count
    )
    reset_before = _boolean_vector(
        path.reset_before, name=f"{name}.reset_before", length=count
    )
    if np.any(indices < 0):
        raise PortabilityValidationError(f"{name}.indices must be non-negative")
    if np.any(state_codes < 0):
        raise PortabilityValidationError(f"{name}.state_codes must be non-negative")
    return _ValidatedPath(
        prediction=prediction,
        residual=residual,
        timestamps=timestamps,
        indices=indices,
        state_codes=state_codes,
        reset_before=reset_before,
        envelope=envelope,
    )


def _finite_positive_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise PortabilityValidationError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityValidationError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise PortabilityValidationError(f"{name} must be a finite positive number")
    return result


def _timestamp_scalar(value: Any, *, name: str) -> np.datetime64:
    try:
        raw = np.asarray(value, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortabilityValidationError(f"{name} must be a valid scalar timestamp") from exc
    if raw.shape != () or np.isnat(raw):
        raise PortabilityValidationError(f"{name} must be a valid scalar timestamp")
    return np.datetime64(raw, "ns")


def _max_abs_error(left: np.ndarray, right: np.ndarray, *, name: str) -> float:
    if left.shape != right.shape:
        raise PortabilityValidationError(
            f"{name} shape mismatch: {left.shape!r} != {right.shape!r}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(left - right)
    if not np.all(np.isfinite(difference)):
        raise PortabilityValidationError(f"{name} difference is non-finite")
    return float(np.max(difference))


def _timestamp_text(timestamp: np.datetime64) -> str:
    return str(np.datetime_as_string(timestamp.astype("datetime64[ns]"), unit="ns"))


def _first_alarm(
    alarm: np.ndarray,
    timestamps: np.ndarray,
    origin: np.datetime64,
) -> tuple[str | None, float | None]:
    locations = np.flatnonzero(alarm)
    if locations.size == 0:
        return None, None
    score_time = timestamps[int(locations[0])]
    availability = score_time + np.timedelta64(AVAILABILITY_LAG_MINUTES, "m")
    delay_ns = int(
        (availability - origin).astype("timedelta64[ns]").astype(np.int64)
    )
    delay_minutes = float(delay_ns / (60.0 * 1.0e9))
    if not np.isfinite(delay_minutes):
        raise PortabilityValidationError("F1 alarm delay is non-finite")
    return _timestamp_text(availability), delay_minutes


def _alignment_flags(
    cpu: _ValidatedPath,
    cuda: _ValidatedPath,
) -> dict[str, bool]:
    return {
        "timestamps_exact": bool(np.array_equal(cpu.timestamps, cuda.timestamps)),
        "indices_exact": bool(np.array_equal(cpu.indices, cuda.indices)),
        "state_codes_exact": bool(np.array_equal(cpu.state_codes, cuda.state_codes)),
        "reset_before_exact": bool(np.array_equal(cpu.reset_before, cuda.reset_before)),
    }


def audit_development_portability(
    *,
    healthy_cpu: PortabilityPath,
    healthy_cuda: PortabilityPath,
    f1_cpu: PortabilityPath,
    f1_cuda: PortabilityPath,
    h_raw: float,
    f1_origin: Any,
) -> dict[str, Any]:
    """Audit complete CPU/CUDA development paths under the frozen rules.

    Malformed evidence raises :class:`PortabilityValidationError`.  Well-formed
    evidence that violates one or more scientific gates returns a strict JSON
    record with ``passed`` false and never silently relaxes a threshold.
    """

    threshold_raw = _finite_positive_scalar(h_raw, name="h_raw")
    origin = _timestamp_scalar(f1_origin, name="f1_origin")
    paths = {
        "dev_healthy": {
            "cpu": _validate_path(healthy_cpu, name="healthy_cpu"),
            "cuda": _validate_path(healthy_cuda, name="healthy_cuda"),
        },
        "dev_fault_f1": {
            "cpu": _validate_path(f1_cpu, name="f1_cpu"),
            "cuda": _validate_path(f1_cuda, name="f1_cuda"),
        },
    }

    split_metrics: dict[str, dict[str, Any]] = {}
    envelope_errors: list[float] = []
    for split_name, devices in paths.items():
        cpu = devices["cpu"]
        cuda = devices["cuda"]
        prediction_error = _max_abs_error(
            cpu.prediction, cuda.prediction, name=f"{split_name}.prediction"
        )
        prediction_residual_error = _max_abs_error(
            cpu.residual[:, :PREDICTION_RESIDUAL_COUNT],
            cuda.residual[:, :PREDICTION_RESIDUAL_COUNT],
            name=f"{split_name}.prediction_residual",
        )
        physical_residual_error = _max_abs_error(
            cpu.residual[:, PREDICTION_RESIDUAL_COUNT:],
            cuda.residual[:, PREDICTION_RESIDUAL_COUNT:],
            name=f"{split_name}.physical_residual",
        )
        physical_residual_exact = bool(
            np.array_equal(
                cpu.residual[:, PREDICTION_RESIDUAL_COUNT:],
                cuda.residual[:, PREDICTION_RESIDUAL_COUNT:],
            )
        )
        envelope_error = _max_abs_error(
            cpu.envelope, cuda.envelope, name=f"{split_name}.envelope"
        )
        envelope_errors.append(envelope_error)
        split_metrics[split_name] = {
            "score_count": int(cpu.envelope.size),
            "prediction_max_abs_error": prediction_error,
            "prediction_residual_max_abs_error": prediction_residual_error,
            "physical_residual_max_abs_error": physical_residual_error,
            "physical_residual_exact": physical_residual_exact,
            "envelope_max_abs_error": envelope_error,
            "alignment": _alignment_flags(cpu, cuda),
        }

    delta = float(max(envelope_errors))
    threshold_sum = threshold_raw + delta + BOUNDARY_DISTANCE_TOLERANCE
    if not np.isfinite(threshold_sum):
        raise PortabilityValidationError("h_raw + delta + boundary tolerance is non-finite")
    threshold_effective = float(np.nextafter(threshold_sum, np.inf))
    if not np.isfinite(threshold_effective) or threshold_effective <= threshold_raw:
        raise PortabilityValidationError("derived h_eff is invalid")

    alarms: dict[str, dict[str, np.ndarray]] = {}
    all_boundary_margins: list[float] = []
    for split_name, devices in paths.items():
        alarms[split_name] = {}
        for device_name, path in devices.items():
            alarm = np.asarray(path.envelope > threshold_effective, dtype=np.bool_)
            alarms[split_name][device_name] = alarm
            distance = np.abs(path.envelope - threshold_effective)
            if not np.all(np.isfinite(distance)):
                raise PortabilityValidationError(
                    f"{split_name}.{device_name} boundary distance is non-finite"
                )
            minimum_distance = float(np.min(distance))
            all_boundary_margins.append(minimum_distance)
            split_metrics[split_name].setdefault("devices", {})[device_name] = {
                "alarm_count": int(np.count_nonzero(alarm)),
                "minimum_boundary_distance": minimum_distance,
                "peak_envelope": float(np.max(path.envelope)),
                "peak_to_h_eff_ratio": float(np.max(path.envelope) / threshold_effective),
            }

    f1_availability: dict[str, str | None] = {}
    f1_delay: dict[str, float | None] = {}
    for device_name in ("cpu", "cuda"):
        availability, delay = _first_alarm(
            alarms["dev_fault_f1"][device_name],
            paths["dev_fault_f1"][device_name].timestamps,
            origin,
        )
        f1_availability[device_name] = availability
        f1_delay[device_name] = delay
        split_metrics["dev_fault_f1"]["devices"][device_name][
            "first_alarm_availability"
        ] = availability
        split_metrics["dev_fault_f1"]["devices"][device_name][
            "first_alarm_delay_minutes"
        ] = delay

    continuous_ok = all(
        split_metrics[name][field] <= PRIMITIVE_MAX_ABS_TOLERANCE
        for name in ("dev_healthy", "dev_fault_f1")
        for field in (
            "prediction_max_abs_error",
            "prediction_residual_max_abs_error",
        )
    )
    physical_residuals_exact = all(
        split_metrics[name]["physical_residual_exact"]
        for name in ("dev_healthy", "dev_fault_f1")
    )
    alignment_ok = all(
        all(split_metrics[name]["alignment"].values())
        for name in ("dev_healthy", "dev_fault_f1")
    )
    alarm_paths_exact = all(
        np.array_equal(alarms[name]["cpu"], alarms[name]["cuda"])
        for name in ("dev_healthy", "dev_fault_f1")
    )
    healthy_zero = all(
        not np.any(alarms["dev_healthy"][device]) for device in ("cpu", "cuda")
    )
    f1_exists = all(
        bool(np.any(alarms["dev_fault_f1"][device])) for device in ("cpu", "cuda")
    )
    first_availability_exact = (
        f1_availability["cpu"] is not None
        and f1_availability["cuda"] is not None
        and f1_availability["cpu"] == f1_availability["cuda"]
    )
    delay_ok = all(
        delay is not None and 0.0 <= delay <= F1_MAX_DELAY_MINUTES
        for delay in f1_delay.values()
    )
    peak_ratio_ok = all(
        split_metrics["dev_fault_f1"]["devices"][device]["peak_to_h_eff_ratio"]
        >= F1_MIN_PEAK_RATIO
        for device in ("cpu", "cuda")
    )
    boundary_ok = all(
        margin > BOUNDARY_DISTANCE_TOLERANCE for margin in all_boundary_margins
    )

    checks = {
        "continuous_primitives_within_1e_9": bool(continuous_ok),
        "physical_residual_last_10_exact": bool(physical_residuals_exact),
        "timestamps_indices_state_reset_exact": bool(alignment_ok),
        "h_eff_alarm_paths_exact": bool(alarm_paths_exact),
        "dev_healthy_zero_alarms_both_devices": bool(healthy_zero),
        "f1_alarm_exists_both_devices": bool(f1_exists),
        "f1_first_alarm_availability_exact": bool(first_availability_exact),
        "f1_delay_at_most_60_minutes": bool(delay_ok),
        "f1_peak_to_h_eff_at_least_1_25_both_devices": bool(peak_ratio_ok),
        "all_envelopes_more_than_1e_9_from_h_eff": bool(boundary_ok),
    }
    passed = bool(all(checks.values()))
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_DEVELOPMENT_PORTABILITY_AUDIT"
        if passed
        else "FAIL_DEVELOPMENT_PORTABILITY_AUDIT",
        "passed": passed,
        "frozen_rules": {
            "primitive_max_abs_tolerance": PRIMITIVE_MAX_ABS_TOLERANCE,
            "prediction_shape_tail": [FORECAST_HEAD_COUNT, ANALOG_COUNT],
            "monitoring_residual_count": MONITORING_RESIDUAL_COUNT,
            "prediction_residual_count": PREDICTION_RESIDUAL_COUNT,
            "physical_residual_count": PHYSICAL_RESIDUAL_COUNT,
            "physical_residual_comparison": "pointwise_exact",
            "boundary_distance_tolerance": BOUNDARY_DISTANCE_TOLERANCE,
            "alarm_comparison": "envelope_strictly_greater_than_h_eff",
            "availability_lag_minutes": AVAILABILITY_LAG_MINUTES,
            "f1_max_delay_minutes": F1_MAX_DELAY_MINUTES,
            "f1_min_peak_ratio": F1_MIN_PEAK_RATIO,
        },
        "threshold": {
            "h_raw": threshold_raw,
            "global_envelope_delta": delta,
            "h_eff": threshold_effective,
            "formula": "nextafter(h_raw + global_envelope_delta + 1e-9, +infinity)",
        },
        "f1_origin": _timestamp_text(origin),
        "splits": split_metrics,
        "checks": checks,
    }
    try:
        json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise PortabilityValidationError("audit record is not strict JSON") from exc
    return record


def certify_development_portability(
    *,
    healthy_cpu: PortabilityPath,
    healthy_cuda: PortabilityPath,
    f1_cpu: PortabilityPath,
    f1_cuda: PortabilityPath,
    h_raw: float,
    f1_origin: Any,
) -> CertifiedDevelopmentPortabilityAudit:
    """Recompute and immutably certify the complete four-path audit.

    This is intentionally a concrete, non-hook boundary for governed runners.
    It delegates all numerical rules to :func:`audit_development_portability`
    so the legacy dictionary API remains backward compatible.
    """

    record = audit_development_portability(
        healthy_cpu=healthy_cpu,
        healthy_cuda=healthy_cuda,
        f1_cpu=f1_cpu,
        f1_cuda=f1_cuda,
        h_raw=h_raw,
        f1_origin=f1_origin,
    )
    return CertifiedDevelopmentPortabilityAudit._from_record(record)


__all__ = [
    "AVAILABILITY_LAG_MINUTES",
    "BOUNDARY_DISTANCE_TOLERANCE",
    "CertifiedDevelopmentPortabilityAudit",
    "F1_MAX_DELAY_MINUTES",
    "F1_MIN_PEAK_RATIO",
    "FORECAST_HEAD_COUNT",
    "ANALOG_COUNT",
    "PREDICTION_RESIDUAL_COUNT",
    "PHYSICAL_RESIDUAL_COUNT",
    "MONITORING_RESIDUAL_COUNT",
    "PRIMITIVE_MAX_ABS_TOLERANCE",
    "PortabilityPath",
    "PortabilityValidationError",
    "audit_development_portability",
    "certify_development_portability",
]
