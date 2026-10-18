from __future__ import annotations

import ast
import hashlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import torch
import numpy as np
import pytest

from mate_spc_metropt.data import (
    APPROVED_NUMERIC_COLUMNS,
    DIRECT_ALARM_COLUMNS,
    MetroPTTimestampAudit,
    load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (
    DEVELOPMENT_ABORT,
    DEVELOPMENT_CHARTER_SHA256,
    DEVELOPMENT_PASS,
    PARTITIONS,
    PHASE_CALIBRATION,
    PHASE_CONFIRMATION,
    PHASE_DEVELOPMENT,
    PHASE_PARTITIONS,
    SOURCE_FILE_SPECS,
    AccessDeniedError,
    AuthenticatedDocument,
    DevelopmentInputAuthentication,
    IntegrityError,
    LedgerClaim,
    canonical_fingerprint,
    read_strict_json,
    require_completion_decision,
)
from mate_spc_metropt.model import (
    CaredScorePath,
    align_inference_forecasts_to_score_times,
    build_causal_input_windows,
    dual_page_cusum,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_metropt_cared_bc_gpu.py"
SPEC = importlib.util.spec_from_file_location("run_metropt_cared_bc_gpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


PARTITION_BY_NAME = {partition.name: partition for partition in PARTITIONS}
NAME_BY_START = {
    partition.start: partition.name
    for partition in PARTITIONS
    if partition.name in ("fit", "dev_healthy", "dev_fault_f1")
}


class FakeInterval:
    def __init__(self, name: str, source_sha256: str) -> None:
        partition = PARTITION_BY_NAME[name]
        self.timestamps = np.arange(
            np.datetime64(partition.start, "m"),
            np.datetime64(partition.end, "m"),
            np.timedelta64(1, "m"),
        ).astype("datetime64[ns]")
        count = self.timestamps.size
        self.analog = np.zeros((count, 7), dtype=np.float64)
        self.controls = np.zeros((count, 6), dtype=np.float64)
        self.eligible = np.ones(count, dtype=bool)
        self.reset_before = np.zeros(count, dtype=bool)
        self.reset_before[0] = True
        if count > 100:
            self.reset_before[100] = True
        self.feature_names = APPROVED_NUMERIC_COLUMNS
        self.access_metadata = {
            "requested_source_row_count": partition.row_count,
            "source_sha256": source_sha256,
            "unrequested_numeric_values_parsed": False,
            "direct_alarm_values_parsed": False,
            "numeric_columns_parsed": list(APPROVED_NUMERIC_COLUMNS),
            "excluded_direct_alarm_columns": list(DIRECT_ALARM_COLUMNS),
            "access_fingerprint": hashlib.sha256(name.encode("ascii")).hexdigest(),
        }


class FakeWindows:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inputs = np.zeros((32, 10, 13), dtype=np.float64)
        self.targets = np.zeros((32, 3, 7), dtype=np.float64)
        start = np.datetime64(PARTITION_BY_NAME[name].start, "ns")
        self.anchor_timestamps = start + np.arange(32).astype("timedelta64[m]")
        horizons = np.asarray([1, 3, 5], dtype="timedelta64[m]")
        self.target_timestamps = self.anchor_timestamps[:, None] + horizons

    def subset(self, indices: Any) -> "FakeWindows":
        selected = np.arange(self.inputs.shape[0])[indices]
        result = FakeWindows(self.name)
        result.inputs = self.inputs[selected]
        result.targets = self.targets[selected]
        result.anchor_timestamps = self.anchor_timestamps[selected]
        result.target_timestamps = self.target_timestamps[selected]
        return result


class FakeEnsemble:
    def predict(
        self, inputs: np.ndarray, *, device: str, batch_size: int
    ) -> np.ndarray:
        assert device == "cuda"
        assert batch_size == 1024
        return np.zeros((inputs.shape[0], 3, 7), dtype=np.float64)

    def save_npz(self, path: Path, *, metadata: dict[str, Any]) -> Path:
        assert metadata["development_only"] is True
        np.savez(path, metadata_json=np.asarray("{}"), weight=np.asarray([1.0]))
        return path


class FakeTraining:
    def __init__(self, relative_improvement: float) -> None:
        self.ensemble = FakeEnsemble()
        self.histories: tuple[Any, ...] = ()
        self.persistence_validation_huber_loss = 1.0
        self.ensemble_validation_huber_loss = 1.0 - relative_improvement
        self._improvement = relative_improvement

    @property
    def relative_improvement_over_persistence(self) -> float:
        return self._improvement


class FakeScorer:
    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold

    def with_threshold(self, threshold: float | None) -> "FakeScorer":
        return FakeScorer(threshold)

    def score(
        self,
        residuals: np.ndarray,
        state_codes: np.ndarray,
        timestamps: np.ndarray,
        *,
        reset_before: np.ndarray,
    ) -> CaredScorePath:
        del state_codes
        assert reset_before.shape == timestamps.shape
        assert bool(reset_before[0])
        envelope = np.asarray(residuals[:, 0], dtype=np.float64)
        count = envelope.size
        cusum = np.column_stack((envelope, np.zeros(count)))
        alarm = (
            np.zeros(count, dtype=bool)
            if self.threshold is None
            else envelope > self.threshold
        )
        return CaredScorePath(
            energy=envelope.copy(),
            directional_glr=np.zeros(count),
            standardized_evidence=cusum.copy(),
            cusum=cusum,
            envelope=envelope,
            reset=np.asarray([True, *([False] * (count - 1))], dtype=bool),
            alarm=alarm,
            threshold=self.threshold,
        )

    def save_npz(self, path: Path, *, metadata: dict[str, Any]) -> Path:
        assert self.threshold is None
        assert metadata["threshold_status"] == "uncalibrated"
        np.savez(path, metadata_json=np.asarray("{}"), center=np.asarray([0.0]))
        return path


def _source_records() -> dict[str, dict[str, Any]]:
    return {
        spec.key: {
            "relative_path": spec.relative_path,
            "size_bytes": spec.size_bytes,
            "sha256": spec.sha256,
        }
        for spec in SOURCE_FILE_SPECS
    }


def _authentication() -> DevelopmentInputAuthentication:
    sources = _source_records()
    charter = AuthenticatedDocument(
        value={"synthetic_test_double": True},
        record={
            "filename": "metropt_cared_bc_v0_development_charter.json",
            "size_bytes": 1,
            "sha256": DEVELOPMENT_CHARTER_SHA256,
            "sidecar_filename": "metropt_cared_bc_v0_development_charter.sha256",
            "sidecar_sha256": "a" * 64,
        },
    )
    return DevelopmentInputAuthentication(
        charter=charter,
        sources=sources,
        source_fingerprint=canonical_fingerprint(sources),
    )


def _parent_amendment() -> AuthenticatedDocument:
    return AuthenticatedDocument(
        value={
            "schema": "metropt_cared_bc_preaccess_amendment",
            "schema_version": 1,
            "amendment_id": runner.PARENT_AMENDMENT_ID,
            "status": "FROZEN_BEFORE_ANY_SENSOR_NUMERIC_ACCESS",
            "parent_charter": {"sha256": DEVELOPMENT_CHARTER_SHA256},
            "scope": {
                "sensor_numeric_values_accessed_before_freeze": False,
                "sealed_partitions_remain_sealed": True,
            },
            "governed_numeric_entry": {
                "formal_runner_must_use_authorized_partition_loader": True,
            },
            "complete_and_qualified_days": {
                "minimum_valid_score_count_inclusive": 1141,
            },
        },
        record={
            "filename": "metropt_cared_bc_v0_preaccess_amendment_001.json",
            "size_bytes": 1,
            "sha256": runner.PARENT_AMENDMENT_SHA256,
            "sidecar_filename": "metropt_cared_bc_v0_preaccess_amendment_001.sha256",
            "sidecar_sha256": "b" * 64,
        },
    )


def _amendment() -> AuthenticatedDocument:
    sealed = [
        partition.name
        for partition in PARTITIONS
        if partition.name not in PHASE_PARTITIONS[PHASE_DEVELOPMENT]
    ]
    return AuthenticatedDocument(
        value={
            "schema": "metropt_cared_bc_development_recovery_amendment",
            "schema_version": 1,
            "amendment_id": runner.AMENDMENT_ID,
            "status": (
                "FROZEN_BEFORE_REPEATED_DEVELOPMENT_RUN_AND_BEFORE_ANY_"
                "CALIBRATION_OR_CONFIRMATION_ACCESS"
            ),
            "parent_bindings": {
                "development_charter_sha256": DEVELOPMENT_CHARTER_SHA256,
                "preaccess_amendment_001_sha256": (
                    runner.PARENT_AMENDMENT_SHA256
                ),
            },
            "failed_run_binding": {
                "output_directory": runner.FAILED_DEVELOPMENT_OUTPUT,
                "decision": DEVELOPMENT_ABORT,
                "reason": (
                    "dev_healthy did not produce exactly nine complete "
                    "calendar-day score blocks"
                ),
                "abort_record_sha256": runner.FAILED_ABORT_SHA256,
                "completion_sha256": runner.FAILED_COMPLETION_SHA256,
                "completion_fingerprint": runner.FAILED_COMPLETION_FINGERPRINT,
                "model_artifact_was_saved": False,
                "scorer_or_power_gate_was_evaluated": False,
                "calibration_numeric_access_occurred": False,
                "confirmation_numeric_access_occurred": False,
            },
            "evidence": {
                "timestamp_only_daily_audit_path": (
                    runner.TIMESTAMP_DAILY_AUDIT_PATH
                ),
                "timestamp_only_daily_audit_sha256": (
                    runner.TIMESTAMP_DAILY_AUDIT_SHA256
                ),
                "development_coverage_diagnostic_path": (
                    runner.COVERAGE_DIAGNOSTIC_PATH
                ),
                "development_coverage_diagnostic_sha256": (
                    runner.COVERAGE_DIAGNOSTIC_SHA256
                ),
            },
            "development_threshold_revision": {
                "minimum_qualified_development_days": 6,
                "qualification_rule_unchanged": (
                    "At least 1141 causal scores out of the theoretical 1426."
                ),
                "unqualified_dates": (
                    "Excluded from the provisional maximum and reported with "
                    "their score counts and reason."
                ),
                "insufficient_qualified_days_action": DEVELOPMENT_ABORT,
            },
            "unchanged_requirements": {
                "method_family": "CARED-BC",
                "minimum_score_count_per_qualified_day": 1141,
                "tcn_improvement_gate": 0.05,
                "f1_latency_gate_minutes": 60,
                "f1_peak_to_boundary_ratio_gate": 1.25,
                "sealed_partitions": sealed,
            },
            "recovery_run": {
                "new_output_directory_required": runner.RECOVERY_OUTPUT,
                "old_failed_bundle_must_remain_unchanged": True,
                "development_numeric_partitions_may_be_reused": list(
                    PHASE_PARTITIONS[PHASE_DEVELOPMENT]
                ),
                "calibration_or_confirmation_access_authorized": False,
                "switch_method_family_if_recovery_power_gate_fails": True,
            },
        },
        record={
            "filename": "metropt_cared_bc_v0_preaccess_amendment_002.json",
            "size_bytes": 1,
            "sha256": runner.AMENDMENT_SHA256,
            "sidecar_filename": (
                "metropt_cared_bc_v0_preaccess_amendment_002.sha256"
            ),
            "sidecar_sha256": "c" * 64,
        },
    )


def _recovery_bindings() -> dict[str, Any]:
    return {
        "schema": "metropt_cared_bc_recovery_binding_authentication",
        "schema_version": 1,
        "timestamp_only_daily_audit": {
            "filename": "metropt_dev_healthy_timestamp_daily_audit.json",
            "size_bytes": 1,
            "sha256": runner.TIMESTAMP_DAILY_AUDIT_SHA256,
        },
        "development_coverage_diagnostic": {
            "filename": "development_coverage_diagnostic.json",
            "size_bytes": 1,
            "sha256": runner.COVERAGE_DIAGNOSTIC_SHA256,
        },
        "failed_run": {
            "output_directory": runner.FAILED_DEVELOPMENT_OUTPUT,
            "abort_record": {
                "filename": runner.ABORT_NAME,
                "size_bytes": 1,
                "sha256": runner.FAILED_ABORT_SHA256,
            },
            "completion_record": {
                "filename": runner.COMPLETION_NAME,
                "size_bytes": 1,
                "sha256": runner.FAILED_COMPLETION_SHA256,
            },
            "completion_fingerprint": runner.FAILED_COMPLETION_FINGERPRINT,
            "decision": DEVELOPMENT_ABORT,
            "fitted_artifacts_absent": True,
        },
    }


def _audit(csv_path: Path) -> MetroPTTimestampAudit:
    sensor = next(spec for spec in SOURCE_FILE_SPECS if spec.key == "sensor_csv")
    return MetroPTTimestampAudit(
        source_path=str(csv_path),
        source_size_bytes=sensor.size_bytes,
        source_mtime_ns=1,
        source_sha256=sensor.sha256,
        header=(
            "Unnamed: 0",
            "timestamp",
            *APPROVED_NUMERIC_COLUMNS,
            *DIRECT_ALARM_COLUMNS,
        ),
        row_count=sum(partition.row_count for partition in PARTITIONS),
        first_timestamp=PARTITIONS[0].start,
        last_timestamp="2020-09-01 03:59:50",
        minimum_interval_seconds=10.0,
        modal_interval_seconds=10.0,
        modal_interval_count=1,
        maximum_interval_seconds=300.0,
        chunks_scanned=1,
        partition_row_counts={
            partition.name: partition.row_count for partition in PARTITIONS
        },
    )


def _aligned(
    name: str,
    interval: FakeInterval,
    *,
    healthy_score_counts: list[int] | None = None,
    healthy_day_maxima: list[float] | None = None,
) -> SimpleNamespace:
    if name == "dev_healthy":
        day_timestamps: list[np.ndarray] = []
        day_envelopes: list[np.ndarray] = []
        counts = healthy_score_counts or [
            runner.MINIMUM_VALID_SCORES_PER_HEALTHY_DAY
        ] * runner.REQUIRED_HEALTHY_DAYS
        maxima = healthy_day_maxima or [
            float(day + 1) for day in range(runner.REQUIRED_HEALTHY_DAYS)
        ]
        for day in range(9):
            if counts[day] == 0:
                continue
            beginning = (
                np.datetime64("2020-02-21", "m")
                + np.timedelta64(day, "D")
                + np.timedelta64(15, "m")
            )
            day_timestamps.append(
                beginning
                + np.arange(counts[day]).astype("timedelta64[m]")
            )
            day_envelopes.append(
                np.full(counts[day], maxima[day])
            )
        timestamps = (
            np.concatenate(day_timestamps).astype("datetime64[ns]")
            if day_timestamps
            else np.asarray([], dtype="datetime64[ns]")
        )
        envelope = (
            np.concatenate(day_envelopes)
            if day_envelopes
            else np.asarray([], dtype=np.float64)
        )
    elif name == "dev_fault_f1":
        timestamps = np.asarray(
            ["2020-04-18T00:30:00", "2020-04-18T01:00:00"],
            dtype="datetime64[ns]",
        )
        envelope = np.asarray([12.0, 10.0])
    else:
        timestamps = np.asarray(
            ["2020-02-01T00:30:00", "2020-02-01T01:00:00"],
            dtype="datetime64[ns]",
        )
        envelope = np.asarray([1.0, 2.0])
    residuals = np.zeros((timestamps.size, 31), dtype=np.float64)
    residuals[:, 0] = envelope
    score_indices = np.searchsorted(interval.timestamps, timestamps)
    return SimpleNamespace(
        residuals=residuals,
        state_codes=np.zeros(timestamps.size, dtype=np.int64),
        score_timestamps=timestamps,
        score_indices=score_indices.astype(np.int64),
        source_window_rows=np.zeros((timestamps.size, 3), dtype=np.int64),
        reset_before=np.asarray(
            [True, *([False] * (timestamps.size - 1))], dtype=bool
        ),
    )


def _workspace_and_config(tmp_path: Path) -> tuple[Path, Any]:
    workspace = tmp_path / "workspace"
    csv_path = workspace / "data" / "metropt" / "MetroPT3(AirCompressor).csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("synthetic test sentinel; runner hooks never parse it\n", encoding="utf-8")
    config = runner.DevelopmentRunConfig(
        csv_path=csv_path,
        output_directory=workspace / "outputs" / "experiment",
        workspace_root=workspace,
        charter_path=workspace / "protocols" / "charter.json",
        charter_sidecar_path=workspace / "protocols" / "charter.sha256",
        parent_amendment_path=workspace / "protocols" / "amendment_001.json",
        parent_amendment_sidecar_path=(
            workspace / "protocols" / "amendment_001.sha256"
        ),
        amendment_path=workspace / "protocols" / "amendment_002.json",
        amendment_sidecar_path=workspace / "protocols" / "amendment_002.sha256",
        phase=PHASE_DEVELOPMENT,
        device="cuda",
        dtype="float64",
        batch_size=1024,
        csv_chunksize=7,
    )
    return csv_path, config


def _hooks(
    config: Any,
    events: list[str],
    *,
    improvement: float = 0.10,
    parity_passed: bool = True,
    authentication_error: Exception | None = None,
    audit_error: Exception | None = None,
    healthy_score_counts: list[int] | None = None,
    healthy_day_maxima: list[float] | None = None,
) -> Any:
    authentication = _authentication()
    parent_amendment = _parent_amendment()
    amendment = _amendment()
    intervals: dict[str, FakeInterval] = {}

    def authenticate_inputs(*args: Any) -> DevelopmentInputAuthentication:
        del args
        events.append("authenticate")
        if authentication_error is not None:
            raise authentication_error
        return authentication

    def timestamp_audit(csv_path: Path, *, chunksize: int) -> MetroPTTimestampAudit:
        assert chunksize == 7
        output = config.output_directory
        assert (output / runner.PHASE_MANIFEST_NAME).is_file()
        assert (output / runner.LEDGER_NAME).is_file()
        assert (output / runner.ACCESS_INTENT_NAME).is_file()
        events.append("timestamp_audit")
        if audit_error is not None:
            raise audit_error
        return _audit(csv_path)

    def authenticate_amendment(
        path: Path,
        sidecar: Path,
        *,
        expected_sha256: str,
    ) -> AuthenticatedDocument:
        del path, sidecar
        if expected_sha256 == runner.PARENT_AMENDMENT_SHA256:
            events.append("authenticate_parent_amendment")
            return parent_amendment
        assert expected_sha256 == runner.AMENDMENT_SHA256
        events.append("authenticate_recovery_amendment")
        return amendment

    def authenticate_recovery_bindings(
        workspace: Path,
        supplied_amendment: AuthenticatedDocument,
    ) -> dict[str, Any]:
        assert workspace == config.workspace_root
        assert supplied_amendment is amendment
        events.append("authenticate_recovery_bindings")
        return _recovery_bindings()

    def load_partition(
        csv_path: Path,
        partition_name: str,
        *,
        claim: Any,
        access_intent: AuthenticatedDocument,
        access_intent_path: Path,
        phase_manifest_fingerprint: str,
        chunksize: int,
        timestamp_audit: MetroPTTimestampAudit,
    ) -> FakeInterval:
        del csv_path
        assert chunksize == 7
        assert isinstance(claim, LedgerClaim)
        assert claim.document.value["phase"] == PHASE_DEVELOPMENT
        assert access_intent.value["authorized_partitions"] == list(
            PHASE_PARTITIONS[PHASE_DEVELOPMENT]
        )
        assert access_intent_path == config.output_directory / runner.ACCESS_INTENT_NAME
        assert (
            access_intent.value["phase_manifest_fingerprint"]
            == phase_manifest_fingerprint
        )
        assert timestamp_audit.as_dict()["sensor_numeric_values_parsed"] is False
        name = partition_name
        assert name in PHASE_PARTITIONS[PHASE_DEVELOPMENT], (
            f"sealed partition was requested: {name}"
        )
        events.append(f"load:{name}")
        interval = FakeInterval(
            name,
            authentication.sources["sensor_csv"]["sha256"],
        )
        intervals[name] = interval
        return interval

    def build_training_windows(
        analog: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
        *,
        eligible: np.ndarray,
        reset_before: np.ndarray,
    ) -> FakeWindows:
        del analog, controls
        assert eligible.all()
        np.testing.assert_array_equal(np.flatnonzero(reset_before), [0, 100])
        name = NAME_BY_START[str(timestamps[0]).replace("T", " ")[:19]]
        events.append(f"training_windows:{name}")
        return FakeWindows(name)

    def build_inference_windows(
        analog: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
        *,
        eligible: np.ndarray,
        reset_before: np.ndarray,
    ) -> FakeWindows:
        del analog, controls
        assert eligible.all()
        np.testing.assert_array_equal(np.flatnonzero(reset_before), [0, 100])
        name = NAME_BY_START[str(timestamps[0]).replace("T", " ")[:19]]
        events.append(f"inference_windows:{name}")
        return FakeWindows(name)

    def train_ensemble(
        fit: FakeWindows,
        healthy: FakeWindows,
        *,
        device: str,
        dtype: str,
        batch_size: int,
    ) -> FakeTraining:
        assert (fit.name, healthy.name) == ("fit", "dev_healthy")
        assert (device, dtype, batch_size) == ("cuda", "float64", 1024)
        events.append("train")
        return FakeTraining(improvement)

    def align_inference_forecasts(
        windows: FakeWindows,
        prediction: np.ndarray,
        analog: np.ndarray,
        controls: np.ndarray,
        timestamps: np.ndarray,
        *,
        eligible: np.ndarray,
        reset_before: np.ndarray,
    ) -> SimpleNamespace:
        del prediction, analog, controls
        assert eligible.all()
        np.testing.assert_array_equal(np.flatnonzero(reset_before), [0, 100])
        events.append(f"align:{windows.name}")
        return _aligned(
            windows.name,
            SimpleNamespace(timestamps=timestamps),
            healthy_score_counts=healthy_score_counts,
            healthy_day_maxima=healthy_day_maxima,
        )

    def fit_scorer(*args: Any) -> FakeScorer:
        assert len(args) == 6
        events.append("fit_scorer")
        return FakeScorer()

    def parity_check(**kwargs: Any) -> dict[str, Any]:
        assert set(kwargs["intervals"]) == {
            "fit",
            "dev_healthy",
            "dev_fault_f1",
        }
        events.append("parity")
        return {
            "schema": "synthetic_parity",
            "alarm_decisions_exact": parity_passed,
            "passed": parity_passed,
            "splits": {},
        }

    return runner.DevelopmentHooks(
        authenticate_inputs=authenticate_inputs,
        authenticate_amendment=authenticate_amendment,
        authenticate_recovery_bindings=authenticate_recovery_bindings,
        timestamp_audit=timestamp_audit,
        load_partition=load_partition,
        build_training_windows=build_training_windows,
        build_inference_windows=build_inference_windows,
        train_ensemble=train_ensemble,
        align_inference_forecasts=align_inference_forecasts,
        fit_scorer=fit_scorer,
        runtime_metadata=lambda unused: {
            "backend": "synthetic",
            "device": "cuda",
            "dtype": "float64",
            "tf32": False,
        },
        parity_check=parity_check,
        causality_check=lambda **unused: True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_windows_runner_imports_torch_before_numpy() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert imported.index("torch") < imported.index("numpy")


def test_default_runner_hook_exposes_only_the_governed_partition_loader() -> None:
    hooks = runner.default_development_hooks()
    assert hooks.load_partition is load_governed_metropt_partition


def test_recovery_defaults_bind_amendment_002_and_new_output() -> None:
    parser = runner.build_argument_parser()
    arguments = parser.parse_args(["data/metropt/synthetic.csv"])
    assert arguments.output_dir == runner.RECOVERY_OUTPUT
    assert arguments.parent_amendment == runner.DEFAULT_PARENT_AMENDMENT_PATH
    assert arguments.amendment == runner.DEFAULT_AMENDMENT_PATH


def test_recovery_amendment_rejects_tampered_failed_run_binding() -> None:
    parent = _parent_amendment()
    amendment = _amendment()
    value = dict(amendment.value)
    failed = dict(value["failed_run_binding"])
    failed["completion_fingerprint"] = "0" * 64
    value["failed_run_binding"] = failed
    with pytest.raises(IntegrityError, match="failed-run binding"):
        runner.validate_recovery_amendment(
            replace(amendment, value=value),
            parent,
        )


def _causal_minute_arrays(
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = (
        np.datetime64("2020-02-01T00:00:00", "ns")
        + np.arange(count).astype("timedelta64[m]")
    )
    time_axis = np.arange(count, dtype=np.float64)
    analog = np.column_stack(
        [0.1 * time_axis + float(column) for column in range(7)]
    )
    controls = np.zeros((count, 6), dtype=np.float64)
    controls[:, 5] = 1.0
    eligible = np.ones(count, dtype=bool)
    return analog, controls, timestamps, eligible


def _synthetic_daily_score_path(
    score_counts: list[int],
    maxima: list[float],
) -> tuple[SimpleNamespace, np.ndarray]:
    assert len(score_counts) == len(maxima) == runner.REQUIRED_HEALTHY_DAYS
    timestamp_parts: list[np.ndarray] = []
    envelope_parts: list[np.ndarray] = []
    for day, (count, maximum) in enumerate(zip(score_counts, maxima, strict=True)):
        if count == 0:
            continue
        start = (
            np.datetime64("2020-02-21T00:14:00", "m")
            + np.timedelta64(day, "D")
        )
        timestamp_parts.append(
            start + np.arange(count).astype("timedelta64[m]")
        )
        envelope_parts.append(np.full(count, maximum, dtype=np.float64))
    timestamps = (
        np.concatenate(timestamp_parts).astype("datetime64[ns]")
        if timestamp_parts
        else np.asarray([], dtype="datetime64[ns]")
    )
    envelope = (
        np.concatenate(envelope_parts)
        if envelope_parts
        else np.asarray([], dtype=np.float64)
    )
    return SimpleNamespace(envelope=envelope), timestamps


def test_day_threshold_fails_when_fewer_than_six_days_qualify() -> None:
    path, timestamps = _synthetic_daily_score_path(
        [1141, 1141, 1141, 1141, 1141, 1140, 100, 1, 0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    with pytest.raises(runner.InsufficientQualifiedDaysError) as captured:
        runner._complete_day_threshold(path, timestamps)
    assert captured.value.qualified_day_count == 5
    assert len(captured.value.day_records) == 9
    assert captured.value.day_records[-1]["score_count"] == 0
    assert captured.value.day_records[-1]["unqualified_reason"] == "zero_scores"


def test_day_threshold_accepts_exactly_six_qualified_days() -> None:
    path, timestamps = _synthetic_daily_score_path(
        [1141, 1141, 1141, 1141, 1141, 1141, 1140, 100, 0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    threshold, records = runner._complete_day_threshold(path, timestamps)
    assert threshold == 6.0
    assert sum(record["qualified"] for record in records) == 6
    assert len(records) == 9


def test_day_threshold_excludes_unqualified_day_even_when_its_maximum_is_high() -> None:
    path, timestamps = _synthetic_daily_score_path(
        [1141, 1141, 1141, 1141, 1141, 1141, 1141, 1141, 1140],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 999.0],
    )
    threshold, records = runner._complete_day_threshold(path, timestamps)
    assert threshold == 8.0
    assert records[-1]["qualified"] is False
    assert records[-1]["cusum_maximum"] == 999.0
    assert records[-1]["included_in_provisional_maximum"] is False


def test_input_only_monitoring_scores_do_not_depend_on_future_eligibility() -> None:
    analog, controls, timestamps, eligible = _causal_minute_arrays(80)
    reset_before = np.zeros(80, dtype=bool)
    reset_before[0] = True
    cutoff = 50

    original = build_causal_input_windows(
        analog,
        controls,
        timestamps,
        eligible=eligible,
        reset_before=reset_before,
    )
    changed_eligibility = eligible.copy()
    changed_eligibility[cutoff + 1 :] = False
    changed = build_causal_input_windows(
        analog,
        controls,
        timestamps,
        eligible=changed_eligibility,
        reset_before=reset_before,
    )
    original_prefix = original.anchor_indices <= cutoff
    np.testing.assert_array_equal(
        original.anchor_indices[original_prefix], changed.anchor_indices
    )
    np.testing.assert_allclose(original.inputs[original_prefix], changed.inputs)

    original_prediction = np.zeros((original.inputs.shape[0], 3, 7))
    changed_prediction = np.zeros((changed.inputs.shape[0], 3, 7))
    original_aligned = align_inference_forecasts_to_score_times(
        original,
        original_prediction,
        analog,
        controls,
        timestamps,
        eligible=eligible,
        reset_before=reset_before,
    )
    changed_aligned = align_inference_forecasts_to_score_times(
        changed,
        changed_prediction,
        analog,
        controls,
        timestamps,
        eligible=changed_eligibility,
        reset_before=reset_before,
    )
    original_score_prefix = original_aligned.score_indices <= cutoff
    np.testing.assert_array_equal(
        original_aligned.score_timestamps[original_score_prefix],
        changed_aligned.score_timestamps,
    )
    np.testing.assert_allclose(
        original_aligned.residuals[original_score_prefix],
        changed_aligned.residuals,
    )


def test_raw_gap_reset_blocks_windows_and_resets_page_cusum() -> None:
    analog, controls, timestamps, eligible = _causal_minute_arrays(60)
    reset_before = np.zeros(60, dtype=bool)
    reset_before[[0, 25]] = True
    windows = build_causal_input_windows(
        analog,
        controls,
        timestamps,
        eligible=eligible,
        reset_before=reset_before,
    )
    assert not np.isin(np.arange(25, 34), windows.anchor_indices).any()
    assert 34 in windows.anchor_indices

    aligned = align_inference_forecasts_to_score_times(
        windows,
        np.zeros((windows.inputs.shape[0], 3, 7)),
        analog,
        controls,
        timestamps,
        eligible=eligible,
        reset_before=reset_before,
    )
    assert np.count_nonzero(aligned.reset_before) >= 2
    page = dual_page_cusum(
        np.ones((aligned.score_timestamps.size, 2)),
        aligned.score_timestamps,
        reset_before=aligned.reset_before,
    )
    np.testing.assert_array_equal(page.reset, aligned.reset_before)
    np.testing.assert_allclose(page.cusum[page.reset], 0.5)


def test_development_pass_obeys_intent_order_and_loads_only_allowed_partitions(
    tmp_path: Path,
) -> None:
    _, config = _workspace_and_config(tmp_path)
    events: list[str] = []
    outcome = runner.run_development(config, hooks=_hooks(config, events))

    assert outcome.decision == DEVELOPMENT_PASS
    assert outcome.reused_completed_bundle is False
    assert events[:7] == [
        "authenticate",
        "authenticate_parent_amendment",
        "authenticate_recovery_amendment",
        "authenticate_recovery_bindings",
        "timestamp_audit",
        "load:fit",
        "load:dev_healthy",
    ]
    assert not any(
        sealed in event
        for event in events
        for sealed in (
            "calibration_a",
            "embargo_a",
            "embargo_b",
            "calibration_b",
            "confirmation",
            "post_study_tail",
        )
    )
    assert outcome.result["provisional_threshold"]["value"] == 9.0
    assert len(outcome.result["provisional_threshold"]["days"]) == 9
    assert outcome.result["f1"]["first_alarm_delay_minutes"] == 31.0
    assert outcome.result["f1"]["maximum_to_threshold_ratio"] == pytest.approx(
        12.0 / 9.0
    )
    assert all(record["passed"] for record in outcome.result["gates"].values())
    assert (
        outcome.result["calibration_numeric_access_authorized_by_this_result"]
        is False
    )
    output = config.output_directory
    for name in (
        *runner.DEVELOPMENT_ARTIFACT_NAMES,
        runner.LEDGER_NAME,
        runner.ACCESS_INTENT_NAME,
        runner.ARTIFACT_MANIFEST_NAME,
        runner.COMPLETION_NAME,
    ):
        assert (output / name).is_file(), name


def test_completed_reentry_authenticates_without_touching_csv_or_model_hooks(
    tmp_path: Path,
) -> None:
    csv_path, config = _workspace_and_config(tmp_path)
    first_events: list[str] = []
    first = runner.run_development(config, hooks=_hooks(config, first_events))
    completion_hash = _sha256(config.output_directory / runner.COMPLETION_NAME)
    csv_path.unlink()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("completed re-entry touched a source or numerical hook")

    forbidden_hooks = runner.DevelopmentHooks(
        authenticate_inputs=forbidden,
        authenticate_amendment=forbidden,
        authenticate_recovery_bindings=forbidden,
        timestamp_audit=forbidden,
        load_partition=forbidden,
        build_training_windows=forbidden,
        build_inference_windows=forbidden,
        train_ensemble=forbidden,
        align_inference_forecasts=forbidden,
        fit_scorer=forbidden,
        runtime_metadata=forbidden,
        parity_check=forbidden,
        causality_check=forbidden,
    )
    second = runner.run_development(config, hooks=forbidden_hooks)

    assert second.decision == first.decision == DEVELOPMENT_PASS
    assert second.reused_completed_bundle is True
    assert second.completion.fingerprint == first.completion.fingerprint
    assert _sha256(config.output_directory / runner.COMPLETION_NAME) == completion_hash


@pytest.mark.parametrize(
    ("improvement", "parity_passed", "failed_gate"),
    [
        (0.049, True, "tcn_relative_improvement"),
        (0.10, False, "cpu_cuda_sample_parity"),
    ],
)
def test_failed_power_gate_records_abort_and_cannot_authorize_calibration(
    tmp_path: Path,
    improvement: float,
    parity_passed: bool,
    failed_gate: str,
) -> None:
    _, config = _workspace_and_config(tmp_path)
    events: list[str] = []
    outcome = runner.run_development(
        config,
        hooks=_hooks(
            config,
            events,
            improvement=improvement,
            parity_passed=parity_passed,
        ),
    )

    assert outcome.decision == DEVELOPMENT_ABORT
    assert outcome.result["gates"][failed_gate]["passed"] is False
    assert outcome.result["next_phase_status"] == "CALIBRATION_REMAINS_SEALED"
    with pytest.raises(AccessDeniedError):
        require_completion_decision(
            outcome.completion,
            phase=PHASE_DEVELOPMENT,
            required_decision=DEVELOPMENT_PASS,
        )
    assert set(event.removeprefix("load:") for event in events if event.startswith("load:")) == {
        "fit",
        "dev_healthy",
        "dev_fault_f1",
    }


def test_authentication_hash_failure_is_before_output_and_numeric_access(
    tmp_path: Path,
) -> None:
    _, config = _workspace_and_config(tmp_path)
    events: list[str] = []
    hooks = _hooks(
        config,
        events,
        authentication_error=IntegrityError("synthetic frozen source mismatch"),
    )

    with pytest.raises(IntegrityError, match="source mismatch"):
        runner.run_development(config, hooks=hooks)
    assert events == ["authenticate"]
    assert not config.output_directory.exists()


def test_recovery_runner_refuses_to_write_inside_the_old_failed_bundle(
    tmp_path: Path,
) -> None:
    _, base_config = _workspace_and_config(tmp_path)
    old_output = (
        base_config.workspace_root
        / "outputs"
        / "metropt_cared_bc_gpu_v0"
    )
    old_output.mkdir(parents=True)
    sentinel = old_output / "immutable-sentinel.json"
    sentinel.write_text('{"old_bundle":true}\n', encoding="utf-8")
    before = _sha256(sentinel)
    events: list[str] = []

    with pytest.raises(AccessDeniedError, match="old failed bundle"):
        runner.run_development(
            replace(base_config, output_directory=old_output),
            hooks=_hooks(base_config, events),
        )
    assert events == []
    assert _sha256(sentinel) == before


@pytest.mark.parametrize("phase", [PHASE_CALIBRATION, PHASE_CONFIRMATION])
def test_future_phases_are_rejected_before_authentication_or_numeric_access(
    tmp_path: Path, phase: str
) -> None:
    _, development_config = _workspace_and_config(tmp_path)
    config = runner.DevelopmentRunConfig(
        csv_path=development_config.csv_path,
        output_directory=development_config.output_directory,
        workspace_root=development_config.workspace_root,
        charter_path=development_config.charter_path,
        charter_sidecar_path=development_config.charter_sidecar_path,
        phase=phase,
    )
    events: list[str] = []
    with pytest.raises(AccessDeniedError, match="development-only"):
        runner.run_development(config, hooks=_hooks(config, events))
    assert events == []
    assert not config.output_directory.exists()


def test_numeric_stage_exception_is_closed_with_authenticated_abort_record(
    tmp_path: Path,
) -> None:
    _, config = _workspace_and_config(tmp_path)
    events: list[str] = []
    hooks = _hooks(
        config,
        events,
        audit_error=ValueError("synthetic timestamp audit failure"),
    )

    with pytest.raises(ValueError, match="timestamp audit failure"):
        runner.run_development(config, hooks=hooks)
    assert events == [
        "authenticate",
        "authenticate_parent_amendment",
        "authenticate_recovery_amendment",
        "authenticate_recovery_bindings",
        "timestamp_audit",
    ]
    assert (config.output_directory / runner.ACCESS_INTENT_NAME).is_file()
    assert (config.output_directory / runner.ABORT_NAME).is_file()
    assert (config.output_directory / runner.COMPLETION_NAME).is_file()

    completed = runner.authenticate_completed_development(config)
    assert completed.decision == DEVELOPMENT_ABORT
    assert completed.reused_completed_bundle is True
    assert completed.result["calibration_numeric_access_authorized"] is False


def test_fewer_than_six_qualified_days_aborts_before_saving_model(
    tmp_path: Path,
) -> None:
    _, config = _workspace_and_config(tmp_path)
    events: list[str] = []
    counts = [1141, 1141, 1141, 1141, 1141, 1140, 100, 1, 0]
    with pytest.raises(runner.InsufficientQualifiedDaysError):
        runner.run_development(
            config,
            hooks=_hooks(
                config,
                events,
                healthy_score_counts=counts,
            ),
        )
    abort = read_strict_json(config.output_directory / runner.ABORT_NAME)
    qualification = abort["development_day_qualification"]
    assert qualification["candidate_day_count"] == 9
    assert qualification["qualified_day_count"] == 5
    assert len(qualification["days"]) == 9
    assert qualification["days"][-1]["score_count"] == 0
    assert not (config.output_directory / runner.ENSEMBLE_NAME).exists()
    assert not (config.output_directory / runner.SCORER_NAME).exists()
    assert (config.output_directory / runner.COMPLETION_NAME).is_file()
    assert not any(
        event.startswith("load:calibration") or event == "load:confirmation"
        for event in events
    )
