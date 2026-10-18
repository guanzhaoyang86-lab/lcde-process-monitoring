from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from mate_spc_metropt.portability import (
    BOUNDARY_DISTANCE_TOLERANCE,
    MONITORING_RESIDUAL_COUNT,
    PREDICTION_RESIDUAL_COUNT,
    PRIMITIVE_MAX_ABS_TOLERANCE,
    PortabilityPath,
    PortabilityValidationError,
    audit_development_portability,
)


def _path(
    envelope: list[float] | np.ndarray,
    *,
    start: str = "2020-04-18T00:00",
    prediction_shift: float = 0.0,
    residual_shift: float = 0.0,
) -> PortabilityPath:
    values = np.asarray(envelope, dtype=np.float64)
    count = int(values.size)
    timestamps = np.datetime64(start, "m") + np.arange(count).astype("timedelta64[m]")
    prediction = np.arange(count * 3 * 7, dtype=np.float64).reshape(count, 3, 7)
    residual = (
        np.arange(count * MONITORING_RESIDUAL_COUNT, dtype=np.float64)
        .reshape(count, MONITORING_RESIDUAL_COUNT)
        / 10.0
    )
    residual[:, :PREDICTION_RESIDUAL_COUNT] += residual_shift
    return PortabilityPath(
        prediction=prediction + prediction_shift,
        residual=residual,
        timestamps=timestamps.astype("datetime64[ns]"),
        indices=np.arange(count, dtype=np.int64),
        state_codes=(np.arange(count, dtype=np.int64) % 4),
        reset_before=np.array([True] + [False] * (count - 1), dtype=bool),
        envelope=values,
    )


def _valid_inputs() -> dict[str, object]:
    return {
        "healthy_cpu": _path([1.0, 2.0, 3.0], start="2020-03-01T00:00"),
        "healthy_cuda": _path(
            [1.0005, 2.0005, 3.0005],
            start="2020-03-01T00:00",
            prediction_shift=0.5e-9,
            residual_shift=-0.5e-9,
        ),
        "f1_cpu": _path([8.0, 12.0, 15.0]),
        "f1_cuda": _path(
            [8.0005, 12.0005, 15.0005],
            prediction_shift=-0.5e-9,
            residual_shift=0.5e-9,
        ),
        "h_raw": 10.0,
        "f1_origin": np.datetime64("2020-04-18T00:00", "ns"),
    }


def test_pass_record_uses_complete_global_delta_nextafter_and_strict_json() -> None:
    inputs = _valid_inputs()
    record = audit_development_portability(**inputs)

    expected_delta = max(
        np.max(
            np.abs(
                np.asarray(inputs["healthy_cpu"].envelope)
                - np.asarray(inputs["healthy_cuda"].envelope)
            )
        ),
        np.max(
            np.abs(
                np.asarray(inputs["f1_cpu"].envelope)
                - np.asarray(inputs["f1_cuda"].envelope)
            )
        ),
    )
    expected_h_eff = np.nextafter(10.0 + expected_delta + 1.0e-9, np.inf)
    assert record["threshold"]["global_envelope_delta"] == expected_delta
    assert record["threshold"]["h_eff"] == expected_h_eff
    assert record["passed"] is True
    assert record["decision"] == "PASS_DEVELOPMENT_PORTABILITY_AUDIT"
    assert record["splits"]["dev_healthy"]["devices"]["cpu"]["alarm_count"] == 0
    assert (
        record["splits"]["dev_fault_f1"]["devices"]["cpu"][
            "first_alarm_availability"
        ]
        == "2020-04-18T00:02:00.000000000"
    )
    assert (
        record["splits"]["dev_fault_f1"]["devices"]["cpu"][
            "first_alarm_delay_minutes"
        ]
        == 2.0
    )
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("field", ["prediction", "residual"])
def test_continuous_primitive_above_frozen_tolerance_fails(field: str) -> None:
    inputs = _valid_inputs()
    cuda = inputs["f1_cuda"]
    changed = np.array(getattr(cuda, field), copy=True)
    changed.flat[0] += 2.0 * PRIMITIVE_MAX_ABS_TOLERANCE
    inputs["f1_cuda"] = replace(cuda, **{field: changed})

    record = audit_development_portability(**inputs)

    assert record["passed"] is False
    assert record["checks"]["continuous_primitives_within_1e_9"] is False


def test_physical_residuals_require_bitwise_value_equality() -> None:
    inputs = _valid_inputs()
    cuda = inputs["f1_cuda"]
    residual = np.array(cuda.residual, copy=True)
    residual[0, PREDICTION_RESIDUAL_COUNT] += 1.0e-15
    assert residual[0, PREDICTION_RESIDUAL_COUNT] != cuda.residual[
        0, PREDICTION_RESIDUAL_COUNT
    ]
    inputs["f1_cuda"] = replace(cuda, residual=residual)

    record = audit_development_portability(**inputs)

    assert record["checks"]["continuous_primitives_within_1e_9"] is True
    assert record["checks"]["physical_residual_last_10_exact"] is False
    assert record["splits"]["dev_fault_f1"]["physical_residual_max_abs_error"] > 0.0
    assert record["passed"] is False


@pytest.mark.parametrize(
    "field,replacement",
    [
        (
            "timestamps",
            np.array(
                [
                    "2020-04-18T00:00:00",
                    "2020-04-18T00:01:30",
                    "2020-04-18T00:02:00",
                ],
                dtype="datetime64[ns]",
            ),
        ),
        ("indices", np.array([0, 7, 2], dtype=np.int64)),
        ("state_codes", np.array([0, 9, 2], dtype=np.int64)),
        ("reset_before", np.array([True, True, False], dtype=bool)),
    ],
)
def test_discrete_alignment_must_be_pointwise_exact(
    field: str,
    replacement: np.ndarray,
) -> None:
    inputs = _valid_inputs()
    inputs["f1_cuda"] = replace(inputs["f1_cuda"], **{field: replacement})

    record = audit_development_portability(**inputs)

    assert record["passed"] is False
    assert record["checks"]["timestamps_indices_state_reset_exact"] is False


def test_regenerated_alarm_paths_and_first_availability_must_match() -> None:
    inputs = _valid_inputs()
    inputs["healthy_cpu"] = _path([1.0, 2.0], start="2020-03-01T00:00")
    inputs["healthy_cuda"] = _path([1.2, 2.2], start="2020-03-01T00:00")
    inputs["f1_cpu"] = _path([10.1, 13.0])
    inputs["f1_cuda"] = _path([10.3, 13.2])

    record = audit_development_portability(**inputs)

    assert record["threshold"]["global_envelope_delta"] == pytest.approx(0.2)
    assert record["checks"]["h_eff_alarm_paths_exact"] is False
    assert record["checks"]["f1_first_alarm_availability_exact"] is False
    assert record["passed"] is False


def test_healthy_alarm_is_a_frozen_gate_failure() -> None:
    inputs = _valid_inputs()
    inputs["healthy_cpu"] = _path([1.0, 11.0], start="2020-03-01T00:00")
    inputs["healthy_cuda"] = _path([1.0, 11.0], start="2020-03-01T00:00")
    inputs["f1_cpu"] = _path([9.0, 13.0])
    inputs["f1_cuda"] = _path([9.0, 13.0])

    record = audit_development_portability(**inputs)

    assert record["checks"]["dev_healthy_zero_alarms_both_devices"] is False
    assert record["passed"] is False


def test_f1_availability_delay_includes_one_minute_lag() -> None:
    inputs = _valid_inputs()
    inputs["f1_cpu"] = _path([13.0], start="2020-04-18T01:00")
    inputs["f1_cuda"] = _path([13.0], start="2020-04-18T01:00")

    record = audit_development_portability(**inputs)

    assert record["checks"]["f1_delay_at_most_60_minutes"] is False
    assert (
        record["splits"]["dev_fault_f1"]["devices"]["cpu"][
            "first_alarm_delay_minutes"
        ]
        == 61.0
    )


def test_f1_peak_ratio_is_required_on_both_devices() -> None:
    inputs = _valid_inputs()
    inputs["f1_cpu"] = _path([9.0, 12.4])
    inputs["f1_cuda"] = _path([9.0, 12.4])

    record = audit_development_portability(**inputs)

    assert record["checks"]["f1_alarm_exists_both_devices"] is True
    assert record["checks"]["f1_peak_to_h_eff_at_least_1_25_both_devices"] is False
    assert record["passed"] is False


def test_every_envelope_requires_strict_boundary_margin() -> None:
    inputs = _valid_inputs()
    h_eff = np.nextafter(
        float(inputs["h_raw"]) + BOUNDARY_DISTANCE_TOLERANCE,
        np.inf,
    )
    inputs["healthy_cpu"] = _path([1.0, h_eff], start="2020-03-01T00:00")
    inputs["healthy_cuda"] = _path([1.0, h_eff], start="2020-03-01T00:00")
    inputs["f1_cpu"] = _path([9.0, 14.0])
    inputs["f1_cuda"] = _path([9.0, 14.0])

    record = audit_development_portability(**inputs)

    assert record["checks"]["all_envelopes_more_than_1e_9_from_h_eff"] is False
    assert record["passed"] is False


@pytest.mark.parametrize("field", ["prediction", "residual", "envelope"])
def test_nonfinite_continuous_evidence_is_rejected(field: str) -> None:
    inputs = _valid_inputs()
    cpu = inputs["healthy_cpu"]
    changed = np.array(getattr(cpu, field), copy=True)
    changed.flat[0] = np.nan
    inputs["healthy_cpu"] = replace(cpu, **{field: changed})

    with pytest.raises(PortabilityValidationError, match="non-finite"):
        audit_development_portability(**inputs)


def test_nat_timestamp_is_rejected() -> None:
    inputs = _valid_inputs()
    cpu = inputs["healthy_cpu"]
    timestamps = np.array(cpu.timestamps, copy=True)
    timestamps[1] = np.datetime64("NaT")
    inputs["healthy_cpu"] = replace(cpu, timestamps=timestamps)

    with pytest.raises(PortabilityValidationError, match="NaT"):
        audit_development_portability(**inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda path: replace(path, prediction=np.zeros((4, 2), dtype=np.float64)),
        lambda path: replace(
            path,
            residual=np.zeros((3, MONITORING_RESIDUAL_COUNT - 1), dtype=np.float64),
        ),
        lambda path: replace(path, indices=np.arange(2, dtype=np.int64)),
    ],
)
def test_shape_mismatch_is_rejected(mutation: object) -> None:
    inputs = _valid_inputs()
    inputs["healthy_cuda"] = mutation(inputs["healthy_cuda"])

    with pytest.raises(PortabilityValidationError, match="shape|dimension|length"):
        audit_development_portability(**inputs)


def test_no_f1_alarm_returns_auditable_fail_without_nonfinite_json() -> None:
    inputs = _valid_inputs()
    inputs["f1_cpu"] = _path([8.0, 9.0])
    inputs["f1_cuda"] = _path([8.0, 9.0])

    record = audit_development_portability(**inputs)

    assert record["passed"] is False
    assert record["checks"]["f1_alarm_exists_both_devices"] is False
    assert record["checks"]["f1_first_alarm_availability_exact"] is False
    assert record["checks"]["f1_delay_at_most_60_minutes"] is False
    assert (
        record["splits"]["dev_fault_f1"]["devices"]["cpu"][
            "first_alarm_availability"
        ]
        is None
    )
    json.dumps(record, allow_nan=False)
