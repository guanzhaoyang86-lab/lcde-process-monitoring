from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import mate_spc_metropt.data as metropt_data
from mate_spc_metropt.data import (
    ANALOG_COLUMNS,
    APPROVED_NUMERIC_COLUMNS,
    DIRECT_ALARM_COLUMNS,
    SAFE_CONTROL_COLUMNS,
    MetroPTDataError,
    MetroPTTimestampAudit,
    audit_metropt_timestamps,
    load_metropt_cared_bc_interval,
    load_governed_metropt_partition,
    load_metropt_interval,
)
from mate_spc_metropt.governance import (
    CHARTER_ID,
    DEVELOPMENT_CHARTER_SHA256,
    PARTITIONS,
    PHASE_DEVELOPMENT,
    PORTABILITY_DEVELOPMENT_ID,
    PORTABILITY_DEVELOPMENT_SHA256,
    PORTABILITY_V1_1_DEVELOPMENT_ID,
    PORTABILITY_V1_1_DEVELOPMENT_SHA256,
    SOURCE_FILE_SPECS,
    AccessDeniedError,
    IntegrityError,
    claim_workspace_access_ledger,
    create_fingerprinted_json,
    create_phase_access_intent,
)


SOURCE_HEADER = (
    "Unnamed: 0",
    "timestamp",
    *ANALOG_COLUMNS,
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)


def test_protocol_qualified_loader_name_is_the_governed_entry() -> None:
    assert load_metropt_cared_bc_interval is load_governed_metropt_partition


def _row(
    timestamp: str,
    *,
    analog: float = 1.0,
    state: int = 0,
    impulses: float = 1.0,
) -> dict[str, object]:
    result: dict[str, object] = {
        "Unnamed: 0": 0,
        "timestamp": timestamp,
        "LPS": 0,
        "Oil_level": 1,
    }
    result.update({name: analog for name in ANALOG_COLUMNS})
    result.update({name: state for name in SAFE_CONTROL_COLUMNS[:-1]})
    result[SAFE_CONTROL_COLUMNS[-1]] = impulses
    return result


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    header: tuple[str, ...] = SOURCE_HEADER,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(name, "") for name in header])


def test_timestamp_audit_and_interval_access_are_value_isolated_and_right_open(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic.csv"
    rows = [
        _row("2020-01-01 00:00:00", analog=1.0),
        _row("2020-01-01 00:01:00", analog=2.0, state=0, impulses=2.0),
        _row("2020-01-01 00:02:30", analog=4.0, state=1, impulses=3.0),
        _row("2020-01-01 00:03:00", analog=8.0),
        _row("2020-01-01 00:04:00", analog=9.0),
    ]
    # These approved-field strings are outside the requested right-open range.
    # A loader that converts a full column before filtering would fail here.
    rows[0]["TP2"] = "outside-secret-not-numeric"
    rows[3]["TP3"] = "stop-endpoint-not-numeric"
    rows[4]["COMP"] = "outside-control-not-numeric"
    # Alarm values inside the requested interval are deliberately hostile; the
    # direct alarm columns must never be requested from the CSV parser.
    rows[1]["LPS"] = "=HYPERLINK(\"https://example.invalid\")"
    rows[2]["Oil_level"] = "not-a-number-and-not-for-the-model"
    _write_csv(source, rows)

    audit = audit_metropt_timestamps(source, chunksize=2)
    assert audit.row_count == 5
    assert audit.modal_interval_seconds == 60.0
    assert audit.as_dict()["sensor_numeric_values_parsed"] is False
    assert audit.as_dict()["numeric_columns_parsed"] == []

    interval = load_metropt_interval(
        source,
        "2020-01-01 00:01:00",
        "2020-01-01 00:03:00",
        chunksize=2,
        timestamp_audit=audit,
    )

    np.testing.assert_array_equal(
        interval.timestamps,
        np.asarray(
            ["2020-01-01T00:01:00", "2020-01-01T00:02:00"],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_allclose(interval.analog[:, 0], [2.0, 4.0])
    assert interval.source_row_count_per_bin.tolist() == [1, 1]
    np.testing.assert_allclose(interval.coverage_fraction, [1.0 / 6.0, 1.0 / 6.0])
    assert interval.eligible.tolist() == [False, False]
    assert interval.reset_before.tolist() == [True, True]
    assert interval.features.shape == (2, 13)
    assert interval.feature_names == APPROVED_NUMERIC_COLUMNS
    metadata = interval.access_metadata
    assert metadata["requested_interval"]["notation"] == "[start, stop)"
    assert metadata["requested_source_row_count"] == 2
    assert metadata["unrequested_numeric_values_parsed"] is False
    assert metadata["direct_alarm_values_parsed"] is False
    assert metadata["excluded_direct_alarm_columns"] == list(DIRECT_ALARM_COLUMNS)


def test_minute_resampling_aggregation_gap_masks_and_bounded_causal_fill(
    tmp_path: Path,
) -> None:
    source = tmp_path / "gaps.csv"
    rows = [
        _row("2020-01-01 00:00:10", analog=1.0, state=0, impulses=2.0),
        _row("2020-01-01 00:00:50", analog=3.0, state=1, impulses=3.0),
        _row("2020-01-01 00:07:10", analog=7.0, state=0, impulses=4.0),
    ]
    _write_csv(source, rows)

    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:09:00",
        chunksize=1,
    )

    assert interval.timestamps.shape == (9,)
    assert interval.source_row_count_per_bin.tolist() == [2, 0, 0, 0, 0, 0, 0, 1, 0]
    assert interval.bin_observed_mask.tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    # Median analog, last state, and summed impulses for the first minute.
    np.testing.assert_allclose(interval.analog[0], np.full(7, 2.0))
    np.testing.assert_allclose(interval.controls[0, :5], np.ones(5))
    assert interval.controls[0, 5] == 5.0

    # Five empty minutes receive state only from the past.  The sixth does not.
    np.testing.assert_allclose(interval.controls[1:6, :5], np.ones((5, 5)))
    assert interval.control_imputed_mask[1:6, :5].all()
    assert np.isnan(interval.controls[6, :5]).all()
    assert not interval.control_imputed_mask[6, :5].any()
    # An impulse is a per-bin count and is never copied into an empty minute.
    assert np.isnan(interval.controls[1:7, 5]).all()

    # Empty-bin analog values are safe placeholders accompanied by false masks.
    np.testing.assert_array_equal(interval.analog[1:7], np.zeros((6, 7)))
    assert not interval.analog_observed_mask[1:7].any()
    assert interval.eligible.tolist() == [False] * 9
    assert interval.reset_before.tolist() == [True, False, False, False, False, False, False, True, False]


def test_raw_gap_reset_survives_when_endpoints_land_in_adjacent_minutes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adjacent-minute-raw-gap.csv"
    _write_csv(
        source,
        [
            _row("2020-01-01 00:00:10", analog=1.0),
            _row("2020-01-01 00:01:20", analog=2.0),
            _row("2020-01-01 00:02:20", analog=3.0),
        ],
    )

    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:03:00",
    )

    assert interval.source_row_count_per_bin.tolist() == [1, 1, 1]
    # The completed grid has ordinary one-minute spacing, but the 70-second
    # raw acquisition gap is retained at its right endpoint's minute.
    assert interval.reset_before.tolist() == [True, True, False]


def test_exactly_60_second_raw_gap_does_not_add_a_reset(tmp_path: Path) -> None:
    source = tmp_path / "exactly-sixty.csv"
    _write_csv(
        source,
        [
            _row("2020-01-01 00:00:20"),
            _row("2020-01-01 00:01:20"),
        ],
    )
    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:02:00",
    )
    assert interval.reset_before.tolist() == [True, False]


def test_coverage_fraction_is_capped_and_three_samples_are_eligible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "coverage.csv"
    rows = [
        _row("2020-01-01 00:00:00"),
        _row("2020-01-01 00:00:20"),
        _row("2020-01-01 00:01:00"),
        _row("2020-01-01 00:01:20"),
        _row("2020-01-01 00:01:40"),
        *[_row(f"2020-01-01 00:02:{second:02d}") for second in (0, 5, 10, 20, 30, 40, 50)],
    ]
    _write_csv(source, rows)
    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:03:00",
    )
    assert interval.source_row_count_per_bin.tolist() == [2, 3, 7]
    np.testing.assert_allclose(interval.coverage_fraction, [1.0 / 3.0, 0.5, 1.0])
    assert interval.eligible.tolist() == [False, True, True]


def test_state_fill_never_uses_a_future_observation(tmp_path: Path) -> None:
    source = tmp_path / "future.csv"
    _write_csv(
        source,
        [_row("2020-01-01 00:02:10", analog=2.0, state=1, impulses=1.0)],
    )

    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:09:00",
    )

    assert np.isnan(interval.controls[:2, :5]).all()
    assert not interval.control_imputed_mask[:2, :5].any()
    np.testing.assert_allclose(interval.controls[3:8, :5], np.ones((5, 5)))
    assert np.isnan(interval.controls[8, :5]).all()
    assert interval.eligible.tolist() == [False] * 9


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2020-01-01 00:00:00", "2020-01-01 00:00:00"],
        ["2020-01-01 00:01:00", "2020-01-01 00:00:00"],
        ["2020-01-01 00:00:00", "malicious-timestamp"],
    ],
)
def test_timestamp_audit_rejects_duplicate_unsorted_or_invalid_values(
    tmp_path: Path, timestamps: list[str]
) -> None:
    source = tmp_path / "bad-time.csv"
    _write_csv(source, [_row(value) for value in timestamps])
    with pytest.raises(MetroPTDataError):
        audit_metropt_timestamps(source, chunksize=1)


@pytest.mark.parametrize("schema_problem", ["duplicate", "missing", "unexpected"])
def test_schema_validation_fails_closed(tmp_path: Path, schema_problem: str) -> None:
    source = tmp_path / "bad-schema.csv"
    header = list(SOURCE_HEADER)
    if schema_problem == "duplicate":
        header[header.index("TP3")] = "TP2"
    elif schema_problem == "missing":
        header.remove("Oil_level")
    else:
        header.append("undeclared_sensor")
    _write_csv(source, [_row("2020-01-01 00:00:00")], header=tuple(header))

    with pytest.raises(MetroPTDataError):
        audit_metropt_timestamps(source)


def test_published_empty_first_index_header_is_normalized(tmp_path: Path) -> None:
    source = tmp_path / "published-header.csv"
    published_header = ("", *SOURCE_HEADER[1:])
    _write_csv(
        source,
        [_row("2020-01-01 00:00:00")],
        header=published_header,
    )

    audit = audit_metropt_timestamps(source)
    assert audit.header[0] == "Unnamed: 0"
    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:01:00",
        timestamp_audit=audit,
    )
    assert interval.eligible.tolist() == [False]


def test_empty_header_outside_first_position_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty-signal-header.csv"
    bad_header = list(SOURCE_HEADER)
    bad_header[bad_header.index("TP2")] = ""
    _write_csv(
        source,
        [_row("2020-01-01 00:00:00")],
        header=tuple(bad_header),
    )

    with pytest.raises(MetroPTDataError, match="empty or invalid"):
        audit_metropt_timestamps(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("TP2", "NaN", "non-finite"),
        ("TP2", "Infinity", "non-finite"),
        ("TP2", "malicious-text", "non-numeric"),
        ("COMP", 2, "binary"),
        ("Caudal_impulses", -1, "non-negative"),
    ],
)
def test_requested_values_are_strictly_validated(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source = tmp_path / f"bad-{field}.csv"
    row = _row("2020-01-01 00:00:00")
    row[field] = value
    _write_csv(source, [row])

    with pytest.raises(MetroPTDataError, match=message):
        load_metropt_interval(
            source,
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
        )


def test_precomputed_audit_is_bound_to_immutable_source(tmp_path: Path) -> None:
    source = tmp_path / "mutation.csv"
    rows = [_row("2020-01-01 00:00:00")]
    _write_csv(source, rows)
    audit = audit_metropt_timestamps(source)

    rows[0]["LPS"] = "changed-after-audit"
    _write_csv(source, rows)
    with pytest.raises(MetroPTDataError, match="changed"):
        load_metropt_interval(
            source,
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            timestamp_audit=audit,
        )


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        ("2020-01-01 00:00:30", "2020-01-01 00:02:00"),
        ("2020-01-01 00:00:00", "2020-01-01 00:02:30"),
        ("2020-01-01 00:02:00", "2020-01-01 00:01:00"),
    ],
)
def test_interval_boundaries_must_define_complete_minutes(
    tmp_path: Path, start: str, stop: str
) -> None:
    source = tmp_path / "bounds.csv"
    _write_csv(source, [_row("2020-01-01 00:01:00")])
    with pytest.raises(MetroPTDataError):
        load_metropt_interval(source, start, stop)


def test_returned_arrays_are_read_only(tmp_path: Path) -> None:
    source = tmp_path / "readonly.csv"
    _write_csv(source, [_row("2020-01-01 00:00:00")])
    interval = load_metropt_interval(
        source,
        "2020-01-01 00:00:00",
        "2020-01-01 00:01:00",
    )

    with pytest.raises(ValueError):
        interval.analog[0, 0] = 99.0
    with pytest.raises(ValueError):
        interval.eligible[0] = False
    with pytest.raises(ValueError):
        interval.coverage_fraction[0] = 1.0


def _governed_development_credentials(
    tmp_path: Path,
    *,
    protocol_id: str = CHARTER_ID,
) -> tuple[Path, object, object, Path, str, MetroPTTimestampAudit]:
    output = tmp_path / "governed-output"
    output.mkdir()
    source = tmp_path / "sensor.csv"
    source.write_text("numeric loader spy sentinel\n", encoding="utf-8")
    phase_fingerprint = "1" * 64
    sensor = next(spec for spec in SOURCE_FILE_SPECS if spec.key == "sensor_csv")
    ledger_path = output / "development_access_ledger.json"
    claim = claim_workspace_access_ledger(
        ledger_path,
        phase=PHASE_DEVELOPMENT,
        workspace_relative_path=ledger_path.name,
        protocol_id=protocol_id,
        charter_sha256=(
            PORTABILITY_V1_1_DEVELOPMENT_SHA256
            if protocol_id == PORTABILITY_V1_1_DEVELOPMENT_ID
            else (
                PORTABILITY_DEVELOPMENT_SHA256
                if protocol_id == PORTABILITY_DEVELOPMENT_ID
                else DEVELOPMENT_CHARTER_SHA256
            )
        ),
        final_protocol_sha256=None,
        sensor_csv_sha256=sensor.sha256,
        source_fingerprint="2" * 64,
        phase_manifest_fingerprint=phase_fingerprint,
        output_directory=output,
    )
    preflight_fingerprint = None
    if protocol_id == PORTABILITY_V1_1_DEVELOPMENT_ID:
        preflight = create_fingerprinted_json(
            output / "runtime_preflight.json",
            {
                "schema": "metropt_cared_bc_v1_1_runtime_preflight",
                "schema_version": 1,
                "phase": PHASE_DEVELOPMENT,
                "protocol_id": PORTABILITY_V1_1_DEVELOPMENT_ID,
                "phase_manifest_fingerprint": phase_fingerprint,
                "passed": True,
                "numeric_access_occurred_before_preflight": False,
            },
            fingerprint_field="preflight_fingerprint",
        )
        preflight_fingerprint = preflight.fingerprint
    intent_path = output / "development_access_intent.json"
    intent = create_phase_access_intent(
        intent_path,
        claim,
        phase_manifest_fingerprint=phase_fingerprint,
        runtime_preflight_fingerprint=preflight_fingerprint,
    )
    audit = MetroPTTimestampAudit(
        source_path=str(source.resolve()),
        source_size_bytes=sensor.size_bytes,
        source_mtime_ns=1,
        source_sha256=sensor.sha256,
        header=SOURCE_HEADER,
        row_count=sum(partition.row_count for partition in PARTITIONS),
        first_timestamp=PARTITIONS[0].start,
        last_timestamp="2020-09-01T03:59:50",
        minimum_interval_seconds=10.0,
        modal_interval_seconds=10.0,
        modal_interval_count=1,
        maximum_interval_seconds=300.0,
        chunks_scanned=1,
        partition_row_counts={
            partition.name: partition.row_count for partition in PARTITIONS
        },
    )
    return source, claim, intent, intent_path, phase_fingerprint, audit


@pytest.mark.parametrize(
    "partition_name",
    ["calibration_a", "calibration_b", "confirmation", "fit_subinterval", ""],
)
def test_governed_development_loader_denies_unsealed_or_arbitrary_partition_before_numeric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition_name: str,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(tmp_path)
    )
    numeric_calls: list[object] = []

    def numeric_spy(*args: object, **kwargs: object) -> object:
        numeric_calls.append((args, kwargs))
        raise AssertionError("denied request reached the numeric loader")

    monkeypatch.setattr(metropt_data, "load_metropt_interval", numeric_spy)
    with pytest.raises(AccessDeniedError):
        load_governed_metropt_partition(
            source,
            partition_name,
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )
    assert numeric_calls == []


def test_governed_loader_accepts_only_the_named_portability_development_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NumericLoadReached(RuntimeError):
        pass

    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(
            tmp_path, protocol_id=PORTABILITY_DEVELOPMENT_ID
        )
    )
    monkeypatch.setattr(metropt_data, "_authenticate_audit", lambda *_: None)
    monkeypatch.setattr(
        metropt_data,
        "load_metropt_interval",
        lambda *args, **kwargs: (_ for _ in ()).throw(NumericLoadReached()),
    )
    with pytest.raises(NumericLoadReached):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )


def test_v1_1_loader_rejects_fit_before_any_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(
            tmp_path, protocol_id=PORTABILITY_V1_1_DEVELOPMENT_ID
        )
    )
    assert claim.document.value["authorized_partitions"] == [
        "dev_healthy",
        "dev_fault_f1",
    ]
    assert intent.value["authorized_partitions"] == [
        "dev_healthy",
        "dev_fault_f1",
    ]
    assert claim.document.value["reusable"] is False
    assert intent.value["single_access"] is True
    source_calls: list[object] = []

    def source_spy(*args: object, **kwargs: object) -> object:
        source_calls.append((args, kwargs))
        raise AssertionError("denied fit request inspected the CSV")

    monkeypatch.setattr(metropt_data, "_validated_source_path", source_spy)
    monkeypatch.setattr(metropt_data, "load_metropt_interval", source_spy)
    with pytest.raises(AccessDeniedError, match="not authorized"):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )
    assert source_calls == []


def test_governed_loader_rejects_unknown_development_protocol_before_numeric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(tmp_path, protocol_id="unknown-development")
    )
    numeric_calls: list[object] = []

    def numeric_spy(*args: object, **kwargs: object) -> object:
        numeric_calls.append((args, kwargs))
        raise AssertionError("unknown protocol reached numeric loading")

    monkeypatch.setattr(metropt_data, "load_metropt_interval", numeric_spy)
    with pytest.raises(IntegrityError, match="access contract"):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )
    assert numeric_calls == []


@pytest.mark.parametrize(
    "tamper", ["phase_fingerprint", "intent", "row_count", "ledger"]
)
def test_governed_loader_rejects_tampered_authority_before_numeric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(tmp_path)
    )
    if tamper == "phase_fingerprint":
        phase_fingerprint = "3" * 64
    elif tamper == "intent":
        intent = replace(intent, fingerprint="4" * 64)
    elif tamper == "row_count":
        counts = dict(audit.partition_row_counts)
        counts["fit"] -= 1
        audit = replace(audit, partition_row_counts=counts)
    else:
        ledger_value = dict(claim.document.value)
        ledger_value["reusable"] = False
        claim = replace(
            claim,
            document=replace(claim.document, value=ledger_value),
        )
    numeric_calls: list[object] = []

    def numeric_spy(*args: object, **kwargs: object) -> object:
        numeric_calls.append((args, kwargs))
        raise AssertionError("tampered authority reached the numeric loader")

    monkeypatch.setattr(metropt_data, "load_metropt_interval", numeric_spy)
    with pytest.raises(IntegrityError):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )
    assert numeric_calls == []


def test_governed_loader_authenticates_actual_source_before_numeric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(tmp_path)
    )
    numeric_calls: list[object] = []

    def numeric_spy(*args: object, **kwargs: object) -> object:
        numeric_calls.append((args, kwargs))
        raise AssertionError("unauthenticated source reached the numeric loader")

    monkeypatch.setattr(metropt_data, "load_metropt_interval", numeric_spy)
    with pytest.raises(MetroPTDataError):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
        )
    assert numeric_calls == []


def test_governed_loader_delegates_only_the_exact_frozen_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, claim, intent, intent_path, phase_fingerprint, audit = (
        _governed_development_credentials(tmp_path)
    )
    observed: dict[str, object] = {}
    fit = next(partition for partition in PARTITIONS if partition.name == "fit")

    class NumericLoadReached(RuntimeError):
        pass

    source_identity_checks: list[tuple[Path, MetroPTTimestampAudit]] = []

    def source_identity_spy(
        path: Path, timestamp_audit: MetroPTTimestampAudit
    ) -> None:
        source_identity_checks.append((path, timestamp_audit))

    def numeric_spy(
        csv_path: Path,
        start: str,
        stop: str,
        *,
        chunksize: int,
        timestamp_audit: MetroPTTimestampAudit,
    ) -> object:
        observed.update(
            csv_path=csv_path,
            start=start,
            stop=stop,
            chunksize=chunksize,
            timestamp_audit=timestamp_audit,
        )
        raise NumericLoadReached

    monkeypatch.setattr(metropt_data, "_authenticate_audit", source_identity_spy)
    monkeypatch.setattr(metropt_data, "load_metropt_interval", numeric_spy)
    with pytest.raises(NumericLoadReached):
        load_governed_metropt_partition(
            source,
            "fit",
            claim=claim,
            access_intent=intent,
            access_intent_path=intent_path,
            phase_manifest_fingerprint=phase_fingerprint,
            timestamp_audit=audit,
            chunksize=17,
        )
    assert source_identity_checks == [(source.resolve(), audit)]
    assert observed == {
        "csv_path": source.resolve(),
        "start": fit.start,
        "stop": fit.end,
        "chunksize": 17,
        "timestamp_audit": audit,
    }
