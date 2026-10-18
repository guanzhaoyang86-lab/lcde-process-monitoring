"""Strict, intent-gated access to the Rieth TEP training partitions.

The upstream archive format cannot provide row-level physical access: ``pyreadr``
decodes a complete RData object before Python can select rows.  This module makes
that limitation explicit and requires a hash-authenticated, pre-access intent for
every decode.  It is intentionally independent of :mod:`mate_spc.datasets` so the
previously frozen dataset adapter does not need to be changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd


DATASET_DOI = "10.7910/DVN/6C3JR1"
INTENT_SCHEMA = "hybrid_pac_rmax_external_access_intent"
INTENT_SCHEMA_VERSION = 1
FULL_DECODE_DISCLOSURE = (
    "pyreadr decodes the complete authorized RData partition before exact "
    "fault/run rows are selected"
)
METADATA_COLUMNS = ("faultNumber", "simulationRun", "sample")
FEATURE_COLUMNS = tuple(
    [f"xmeas_{index}" for index in range(1, 42)]
    + [f"xmv_{index}" for index in range(1, 12)]
)
EXTERNAL_RIETH_FILES = MappingProxyType(
    {
        "fault_free_training": {
            "file_name": "TEP_FaultFree_Training.RData",
            "md5": "ec126484534331f85001d8c4ebce6d17",
            "sha256": "12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940",
            "size_bytes": 24_678_017,
        },
        "faulty_training": {
            "file_name": "TEP_Faulty_Training.RData",
            "md5": "c5f594d54c47e620ff877feb58407fda",
            "sha256": "058df8bd90c5f575f44d2f6085bdd3bc3662e3e9cc25a50e34af42be3748952b",
            "size_bytes": 494_063_194,
        },
    }
)


def _md5_sha256_file(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _authenticated_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("authenticated JSON SHA-256 differs")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _normalize_selection(
    selections: Mapping[int, tuple[int, ...] | list[int]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(selections, Mapping) or not selections:
        raise ValueError("selections must be a nonempty mapping")
    normalized: list[tuple[int, tuple[int, ...]]] = []
    for raw_fault, raw_runs in selections.items():
        if isinstance(raw_fault, (bool, np.bool_)) or not isinstance(
            raw_fault, (int, np.integer)
        ):
            raise ValueError("fault identifiers must be integers")
        fault = int(raw_fault)
        materialized = tuple(raw_runs)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in materialized
        ):
            raise ValueError(f"fault {fault}: run identifiers must be integers")
        runs = tuple(int(value) for value in materialized)
        if not runs or len(runs) != len(set(runs)) or any(value < 1 for value in runs):
            raise ValueError(f"fault {fault}: run identifiers must be unique and positive")
        normalized.append((fault, runs))
    normalized.sort(key=lambda item: item[0])
    if len({fault for fault, _ in normalized}) != len(normalized):
        raise ValueError("duplicate fault identifiers")
    return tuple(normalized)


def _authorization_record(
    *,
    partition: str,
    selections: tuple[tuple[int, tuple[int, ...]], ...],
    expected_length: int,
    fault_start: int,
) -> dict[str, Any]:
    if partition not in EXTERNAL_RIETH_FILES:
        raise ValueError(f"unsupported external partition: {partition}")
    if isinstance(expected_length, (bool, np.bool_)) or not isinstance(
        expected_length, (int, np.integer)
    ):
        raise ValueError("expected_length must be an integer")
    if isinstance(fault_start, (bool, np.bool_)) or not isinstance(
        fault_start, (int, np.integer)
    ):
        raise ValueError("fault_start must be an integer")
    if int(expected_length) < 1 or not 0 <= int(fault_start) <= int(expected_length):
        raise ValueError("expected_length/fault_start bounds are invalid")
    source = EXTERNAL_RIETH_FILES[partition]
    return {
        "partition": partition,
        "file_name": source["file_name"],
        "expected_md5": source["md5"],
        "expected_sha256": source["sha256"],
        "expected_size_bytes": source["size_bytes"],
        "fault_ids": [fault for fault, _ in selections],
        "simulation_runs_by_fault": {
            str(fault): list(runs) for fault, runs in selections
        },
        "expected_run_length": int(expected_length),
        "fault_start_zero_based": int(fault_start),
        "decode_scope": "complete_partition_then_exact_row_selection",
    }


@dataclass(frozen=True)
class ExternalRiethRun:
    """One independent 52-channel Rieth training run."""

    name: str
    x: np.ndarray
    fault: np.ndarray
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        matrix = np.array(self.x, dtype=np.float64, copy=True, order="C")
        labels = np.array(self.fault, dtype=bool, copy=True)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError("external Rieth run must have shape (time, 52)")
        if labels.shape != (matrix.shape[0],):
            raise ValueError("fault mask length does not match the run")
        if not np.isfinite(matrix).all():
            raise ValueError("external Rieth run contains non-finite values")
        if tuple(self.feature_names) != FEATURE_COLUMNS:
            raise ValueError("external Rieth feature order is not canonical")
        matrix.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "x", matrix)
        object.__setattr__(self, "fault", labels)
        object.__setattr__(self, "feature_names", FEATURE_COLUMNS)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


_MINT_SEAL = object()


class RiethAccessGrant:
    """Ordered attempt-once grant minted only from an authenticated intent.

    Cross-process one-shot behavior is enforced by the runner's atomic completed
    phase or retained-partial rule; this object enforces order within one process.
    """

    __slots__ = (
        "_protocol_sha256", "_intent_path", "_intent_sha256",
        "_intent_canonical", "_phase", "_phase_binding_canonical",
        "_authorizations_canonical", "_next_authorization",
        "_invalidated",
    )

    def __init__(self, *, protocol_sha256: str, intent_path: Path,
                 intent_sha256: str, intent: Mapping[str, Any], _seal: object) -> None:
        if _seal is not _MINT_SEAL:
            raise PermissionError("RiethAccessGrant can only be minted from an intent")
        self._protocol_sha256 = _require_sha256(protocol_sha256, label="protocol_sha256")
        self._intent_sha256 = _require_sha256(intent_sha256, label="intent_sha256")
        self._intent_path = Path(intent_path).resolve()
        self._intent_canonical = json.dumps(
            intent, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._phase = str(intent["phase"])
        self._phase_binding_canonical = json.dumps(
            intent["phase_binding"], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._authorizations_canonical = tuple(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for record in intent["authorized_reads"]
        )
        self._next_authorization = 0
        self._invalidated = False

    @property
    def protocol_sha256(self) -> str:
        return self._protocol_sha256

    @property
    def intent_sha256(self) -> str:
        return self._intent_sha256

    @property
    def phase(self) -> str:
        return self._phase

    def _reauthenticate(self) -> dict[str, Any]:
        if not self._intent_path.is_file():
            raise PermissionError("access intent changed after grant minting")
        try:
            current = _authenticated_json(self._intent_path, self._intent_sha256)
        except (OSError, UnicodeError, ValueError) as error:
            raise PermissionError("access intent changed after grant minting") from error
        canonical = json.dumps(current, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical != self._intent_canonical:
            raise PermissionError("access intent content changed after grant minting")
        if str(current.get("phase")) != self._phase:
            raise PermissionError("access intent phase changed after grant minting")
        binding = json.dumps(
            current.get("phase_binding"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if binding != self._phase_binding_canonical:
            raise PermissionError("access intent phase binding changed after grant minting")
        return current

    def authorization_for(
        self,
        partition: str,
        selections: Mapping[int, tuple[int, ...] | list[int]],
        *,
        expected_length: int,
        fault_start: int,
    ) -> dict[str, Any]:
        if self._invalidated:
            raise PermissionError("access grant was invalidated by an earlier failed attempt")
        try:
            expected = _authorization_record(
                partition=partition,
                selections=_normalize_selection(selections),
                expected_length=expected_length,
                fault_start=fault_start,
            )
            current = self._reauthenticate()
        except Exception:
            self._invalidated = True
            raise
        if self._next_authorization >= len(self._authorizations_canonical):
            self._invalidated = True
            raise PermissionError("access grant has already been fully consumed")
        authorized = current["authorized_reads"][self._next_authorization]
        canonical = json.dumps(
            authorized, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected_canonical = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical != expected_canonical:
            self._invalidated = True
            raise PermissionError("requested RData decode is out of order or not exactly authorized")
        if canonical != self._authorizations_canonical[self._next_authorization]:
            self._invalidated = True
            raise PermissionError("nested authorization content changed")
        self._next_authorization += 1
        return expected

    def assert_fully_consumed(self) -> None:
        if self._invalidated:
            raise PermissionError("access grant was invalidated")
        self._reauthenticate()
        if self._next_authorization != len(self._authorizations_canonical):
            raise PermissionError("not every ordered authorization was consumed")

    def _invalidate_after_failed_decode(self) -> None:
        self._invalidated = True


def load_access_grant(
    intent_path: str | Path,
    *,
    expected_protocol_sha256: str,
    expected_intent_sha256: str,
) -> RiethAccessGrant:
    """Authenticate a written intent and mint the only accepted decode grant."""

    path = Path(intent_path).resolve()
    protocol_sha256 = _require_sha256(
        expected_protocol_sha256, label="expected_protocol_sha256"
    )
    intent_sha256 = _require_sha256(
        expected_intent_sha256, label="expected_intent_sha256"
    )
    if not path.is_file():
        raise ValueError("access-intent file is missing or its SHA-256 differs")
    intent = _authenticated_json(path, intent_sha256)
    if set(intent) != {
        "schema",
        "schema_version",
        "phase",
        "protocol_sha256",
        "created_before_external_numeric_decode",
        "whole_partition_decode_acknowledged",
        "authorized_reads",
        "forbidden_actions",
        "phase_binding",
    }:
        raise ValueError("access-intent fields differ from the closed schema")
    if (
        intent["schema"] != INTENT_SCHEMA
        or type(intent["schema_version"]) is not int
        or intent["schema_version"] != INTENT_SCHEMA_VERSION
    ):
        raise ValueError("access-intent schema/version mismatch")
    if intent["protocol_sha256"] != protocol_sha256:
        raise ValueError("access intent belongs to another protocol")
    if intent["created_before_external_numeric_decode"] is not True:
        raise ValueError("access intent does not certify pre-decode creation")
    if intent["whole_partition_decode_acknowledged"] != FULL_DECODE_DISCLOSURE:
        raise ValueError("access intent omits the exact whole-partition disclosure")
    if intent["forbidden_actions"] != [
        "training",
        "refitting",
        "hyperparameter_selection",
        "threshold_selection_from_audit_rows",
        "same_data_retuning_after_result",
    ]:
        raise ValueError("access intent has an unexpected forbidden-action policy")
    phase = intent["phase"]
    reads = intent["authorized_reads"]
    if not isinstance(reads, list) or any(
        not isinstance(record, dict) for record in reads
    ):
        raise ValueError("access intent must contain a list of read records")
    partitions = [record.get("partition") for record in reads]
    expected_partitions = {
        "calibration": ["fault_free_training"],
        "audit": ["fault_free_training", "faulty_training"],
    }
    if phase not in expected_partitions or partitions != expected_partitions[phase]:
        raise ValueError("access intent partition order/scope is invalid")
    binding = intent["phase_binding"]
    if not isinstance(binding, dict):
        raise ValueError("access intent must bind its phase inputs")
    if phase == "calibration":
        if set(binding) != {
            "deployment_spec_sha256",
            "monitor_configuration_sha256",
            "threshold_rule",
            "numeric_threshold_unknown_before_this_access",
        }:
            raise ValueError("calibration phase binding has an open or incomplete schema")
        _require_sha256(binding["deployment_spec_sha256"], label="deployment_spec_sha256")
        _require_sha256(
            binding["monitor_configuration_sha256"],
            label="monitor_configuration_sha256",
        )
        if binding["threshold_rule"] != "PAC q99 second-largest of 100 CPU run maxima":
            raise ValueError("calibration threshold rule changed")
        if binding["numeric_threshold_unknown_before_this_access"] is not True:
            raise ValueError("calibration binding improperly contains a known threshold")
    else:
        if set(binding) != {
            "calibration_completion_sha256",
            "calibration_completion_fingerprint",
            "pac_certificate_sha256",
            "calibration_evidence_sha256",
            "operational_threshold",
            "operational_threshold_binary64_hex",
            "threshold_source",
            "calibration_decision",
        }:
            raise ValueError("audit phase binding has an open or incomplete schema")
        for key in (
            "calibration_completion_sha256",
            "calibration_completion_fingerprint",
            "pac_certificate_sha256",
            "calibration_evidence_sha256",
        ):
            _require_sha256(binding[key], label=key)
        threshold = binding["operational_threshold"]
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise ValueError("audit operational threshold must be numeric")
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("audit operational threshold must be finite and nonnegative")
        if binding["operational_threshold_binary64_hex"] != threshold.hex():
            raise ValueError("audit threshold decimal/hex values differ")
        if binding["threshold_source"] != "CPU_NumPy_reference":
            raise ValueError("audit threshold source changed")
        if binding["calibration_decision"] != (
            "PAC_Q99_CALIBRATION_FROZEN_FOR_EXTERNAL_AUDIT"
        ):
            raise ValueError("audit calibration decision changed")
    return RiethAccessGrant(
        protocol_sha256=protocol_sha256,
        intent_path=path,
        intent_sha256=intent_sha256,
        intent=intent,
        _seal=_MINT_SEAL,
    )


def _exact_integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f"Rieth metadata column {column!r} is not finite integral data")
    return values.astype(np.int64)


def _frame_to_runs(
    frame: pd.DataFrame,
    *,
    partition: str,
    selections: Mapping[int, tuple[int, ...] | list[int]],
    source_name: str,
    source_md5: str,
    expected_length: int,
    fault_start: int,
) -> tuple[ExternalRiethRun, ...]:
    """Convert only authorized rows after pyreadr has completed its full decode."""

    if isinstance(expected_length, (bool, np.bool_)) or not isinstance(
        expected_length, (int, np.integer)
    ):
        raise ValueError("expected_length must be an integer")
    if isinstance(fault_start, (bool, np.bool_)) or not isinstance(
        fault_start, (int, np.integer)
    ):
        raise ValueError("fault_start must be an integer")
    expected_length = int(expected_length)
    fault_start = int(fault_start)
    if expected_length < 1 or not 0 <= fault_start <= expected_length:
        raise ValueError("expected_length/fault_start bounds are invalid")
    if bool(frame.columns.duplicated().any()):
        raise ValueError("Rieth RData object contains duplicate column names")
    normalized = _normalize_selection(selections)
    if partition == "fault_free_training" and tuple(fault for fault, _ in normalized) != (0,):
        raise ValueError("fault-free training access must select only fault 0")
    if partition == "faulty_training" and any(
        fault not in {3, 9, 15} for fault, _ in normalized
    ):
        raise ValueError("faulty training access is restricted to faults 3, 9, and 15")
    required = set(METADATA_COLUMNS + FEATURE_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rieth RData object is missing columns: {missing}")
    fault_values = _exact_integer_column(frame, "faultNumber")
    run_values = _exact_integer_column(frame, "simulationRun")
    sample_values = _exact_integer_column(frame, "sample")
    selected = np.zeros(frame.shape[0], dtype=bool)
    for fault_id, run_ids in normalized:
        selected |= (fault_values == fault_id) & np.isin(run_values, run_ids)
    indices = np.flatnonzero(selected)
    if indices.size != expected_length * sum(len(runs) for _, runs in normalized):
        raise ValueError("authorized selection does not contain the exact expected row count")
    selected_features = frame.iloc[indices][list(FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    feature_values = selected_features.to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("authorized Rieth rows contain non-finite features")
    selected_fault_values = fault_values[indices]
    selected_run_values = run_values[indices]
    selected_sample_values = sample_values[indices]

    result: list[ExternalRiethRun] = []
    for fault_id, run_ids in normalized:
        for run_id in run_ids:
            positions = np.flatnonzero(
                (selected_fault_values == fault_id) & (selected_run_values == run_id)
            )
            if positions.size != expected_length:
                raise ValueError(
                    f"fault {fault_id}, run {run_id}: expected {expected_length} rows"
                )
            order = np.argsort(selected_sample_values[positions], kind="stable")
            ordered_positions = positions[order]
            samples = selected_sample_values[ordered_positions]
            if not np.array_equal(samples, np.arange(1, expected_length + 1)):
                raise ValueError(
                    f"fault {fault_id}, run {run_id}: samples are not exactly 1..{expected_length}"
                )
            matrix = feature_values[ordered_positions]
            labels = np.zeros(expected_length, dtype=bool)
            if fault_id != 0:
                labels[min(fault_start, expected_length) :] = True
            result.append(
                ExternalRiethRun(
                    name=(
                        f"Rieth-{partition}-fault-{fault_id:02d}-run-{run_id:03d}"
                    ),
                    x=matrix,
                    fault=labels,
                    feature_names=FEATURE_COLUMNS,
                    metadata={
                        "dataset": "Rieth et al. multi-run Tennessee Eastman process",
                        "dataset_doi": DATASET_DOI,
                        "partition": partition,
                        "fault_id": fault_id,
                        "simulation_run": run_id,
                        "source_file": source_name,
                        "source_md5": source_md5,
                        "fault_start_zero_based": fault_start,
                        "full_partition_decoded_by_pyreadr": True,
                        "decode_disclosure": FULL_DECODE_DISCLOSURE,
                    },
                )
            )
    return tuple(result)


class SafeRiethTrainingArchive:
    """Checksum-verifying, grant-gated reader for the two training RData files."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Rieth data directory not found: {self.folder}")

    def verify_sources(self) -> dict[str, dict[str, Any]]:
        """Hash raw source bytes without decoding either RData object."""

        records: dict[str, dict[str, Any]] = {}
        for partition, specification in EXTERNAL_RIETH_FILES.items():
            path = self.folder / str(specification["file_name"])
            if not path.is_file():
                raise FileNotFoundError(f"external Rieth file not found: {path}")
            size = path.stat().st_size
            if size != specification["size_bytes"]:
                raise ValueError(f"{path.name}: source byte size differs from protocol")
            digest, sha256 = _md5_sha256_file(path)
            if digest != specification["md5"]:
                raise ValueError(
                    f"{path.name}: expected MD5 {specification['md5']}, observed {digest}"
                )
            if sha256 != specification["sha256"]:
                raise ValueError(
                    f"{path.name}: expected SHA-256 {specification['sha256']}, "
                    f"observed {sha256}"
                )
            records[partition] = {
                "file_name": path.name,
                "size_bytes": size,
                "md5": digest,
                "sha256": sha256,
            }
        return records

    def load_runs(
        self,
        grant: RiethAccessGrant,
        partition: str,
        selections: Mapping[int, tuple[int, ...] | list[int]],
        *,
        expected_length: int,
        fault_start: int,
    ) -> tuple[ExternalRiethRun, ...]:
        """Decode one whole authorized partition and return only exact selections."""

        if type(grant) is not RiethAccessGrant:
            raise PermissionError("a hash-authenticated RiethAccessGrant is required")
        grant.authorization_for(
            partition,
            selections,
            expected_length=expected_length,
            fault_start=fault_start,
        )
        try:
            specification = EXTERNAL_RIETH_FILES[partition]
            path = self.folder / str(specification["file_name"])
            if not path.is_file() or path.stat().st_size != specification["size_bytes"]:
                raise ValueError("authorized Rieth source disappeared or changed size")
            observed_md5, observed_sha256 = _md5_sha256_file(path)
            if observed_md5 != specification["md5"]:
                raise ValueError("authorized Rieth source MD5 changed after preflight")
            if observed_sha256 != specification["sha256"]:
                raise ValueError("authorized Rieth source SHA-256 changed after preflight")
            try:
                import pyreadr
            except (ImportError, OSError) as error:
                raise RuntimeError("external RData access requires pyreadr") from error
            objects = pyreadr.read_r(str(path))
            if len(objects) != 1:
                raise ValueError(f"{path.name}: expected exactly one R object")
            frame = next(iter(objects.values()))
            if not isinstance(frame, pd.DataFrame):
                raise ValueError(f"{path.name}: decoded R object is not a pandas DataFrame")
            return _frame_to_runs(
                frame,
                partition=partition,
                selections=selections,
                source_name=path.name,
                source_md5=str(specification["md5"]),
                expected_length=expected_length,
                fault_start=fault_start,
            )
        except Exception:
            grant._invalidate_after_failed_decode()
            raise


__all__ = [
    "DATASET_DOI",
    "EXTERNAL_RIETH_FILES",
    "ExternalRiethRun",
    "FEATURE_COLUMNS",
    "FULL_DECODE_DISCLOSURE",
    "INTENT_SCHEMA",
    "INTENT_SCHEMA_VERSION",
    "RiethAccessGrant",
    "SafeRiethTrainingArchive",
    "load_access_grant",
]
