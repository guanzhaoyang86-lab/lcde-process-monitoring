"""Run the prospectively frozen Hydraulic PAC-PADR-CUSUM study on CUDA.

The command has two deliberately separated phases.  Development may read all
cycles in the 18 development nuisance groups and only the healthy cycles in
the nine calibration groups.  Confirmation first authenticates the complete
development result and then opens the nine held-out nuisance groups exactly
once.  Numeric sensor parsing is always preceded by a hash-authenticated,
atomically published manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def _requested_device(tokens: Sequence[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# This Windows host requires CUDA to be initialized before NumPy/MKL is loaded.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

from mate_spc_hydraulic.data import (
    CYCLE_SECONDS,
    OUTPUT_CHANNEL_COUNT,
    RAW_CHANNEL_COUNT,
    fit_phasewise_template,
    load_hydraulic_cycles,
    load_phasewise_template,
    save_phasewise_template,
)
from mate_spc_hydraulic.model import (
    DEFAULT_KAPPA,
    HEALTHY_LABEL,
    PERSISTENCE_WINDOW,
    POSITIVE_LABELS,
    RIDGE_LAMBDA_GRID,
    HydraulicDirectionalPath,
    HydraulicDirectionalModel,
    HydraulicPACCalibration,
    calibrate_hydraulic_pac,
    compare_hydraulic_paths,
    fit_hydraulic_grouped_cv_cuda,
    hydraulic_calibration_sha256,
    hydraulic_directional_model_sha256,
    load_hydraulic_model,
    protocol_grouped_fold_ids,
    save_hydraulic_model,
    score_hydraulic_cuda,
    score_hydraulic_numpy,
)
from mate_spc_pac import pac_tolerance_rank


PROTOCOL_ID = "hydraulic_padr_pac_v1_3"
METHOD_NAME = "PAC-PADR-CUSUM"
ANALYSIS_PROTOCOL_SHA256 = (
    "d3d446c4dfc18de676c08f366f1a2d9099928ce7f8d537dd77ddb89ae0f5e115"
)
PARENT_PROTOCOL_ID = "hydraulic_padr_pac_v1_2"
PARENT_ANALYSIS_PROTOCOL_SHA256 = (
    "42003f3f1b394b2284ee82e7558d3d421b6ece7a81c45dae4a8a56e963a58efa"
)
ABORTED_PARENT_RECORD_SHA256 = (
    "8ffe81d25167aaac5344d35607b372e50bf58bd55120de6eb0518ca6e513ea76"
)
PARENT_DEVELOPMENT_MANIFEST_SHA256 = (
    "192ea5af010d3f60d01ad7c837bf5e39ebcc01a848272523759ca474378894ab"
)
PARENT_DEVELOPMENT_MANIFEST_FINGERPRINT = (
    "4aaa478bd9841d2a8a8c7b375e0210da51113854d9ae953e7cef7f05aa13f0eb"
)
PARENT_DEVELOPMENT_COMPLETION_SHA256 = (
    "2c2610dc7f5010312a95c8ea5060f3c91243fcf317f304981373b0db7820dd34"
)
PARENT_DEVELOPMENT_COMPLETION_FINGERPRINT = (
    "3e33b8cf8b8fc30628665bec0b076ecdc0156f23242fc00435c770cfdc4a62f1"
)
PARENT_DEVELOPMENT_ARTIFACT_SHA256 = {
    "calibrated_model.npz": "7f016deef19abc930764981939e852e8db610f4f3d595ed378b27be6916b76d1",
    "calibration_certificate.json": "0606b8d2d76a56b7a9b7a78f08487a4e36072627e93456ee608d064f50ccbbd7",
    "cuda_validation_development.json": "dee9b79b7fa16afc7b2a97b366ef7df58635e9c411ccd0e617414b01ffea6e5b",
    "development_access_intent.json": "15463ec3df837fc3efad175a901e0351a1a8bbb992ab18811f4664549d235abd",
    "development_manifest.json": PARENT_DEVELOPMENT_MANIFEST_SHA256,
    "development_result.json": "85b9aac7d31e39b6006b83b5408f10c39b316419cb5121e2bf0615d32d17f3d5",
    "gpu_execution_development.json": "3a8c26535d1d1b1af090cbbb9f2172fa9335d9ef5b0fc3b3f98a2c5870eeebe1",
    "phase_template.npz": "e71247efaa232f1ae037ff5ccd5401913e4ad5466ab79b78fe7e787e46e272e4",
}
BASE_PROTOCOL_ID = "hydraulic_padr_pac_v1"
BASE_ANALYSIS_PROTOCOL_SHA256 = (
    "24809b4c89cc1cd72c08c2442489d4498e279a82dc9b040b8eaabc8bfe8f3d67"
)
ABORTED_V1_RECORD_SHA256 = (
    "1efa46813ebd8993e3a54710f03207e97dfa853d7e268c346feda5ed4f9359bb"
)
ABORTED_V1_MANIFEST_SHA256 = (
    "cbe0c9adbc1908143a08aa0a7f97139f87bfbfdcb96e933ec5390abc6e4165a7"
)
ABORTED_V1_MANIFEST_FINGERPRINT = (
    "064cae53deba51085d5b47f9d344526371a943ef4cf23365b8b44951de8777f0"
)
ABORTED_V1_ACCESS_SHA256 = (
    "2dcf730296ccfaf9e1699612266362812ca41d895f5358da4e9af5394803b96f"
)
ABORTED_V1_1_MANIFEST_SHA256 = (
    "5754ea7f34916b6ab181d2126b2bc1800f258a0da3c972c8e0232b201e06269a"
)
ABORTED_V1_1_MANIFEST_FINGERPRINT = (
    "567c0743a430aef6d6ab61adc3bf4fd9842ce0aec95491ac7fd4c185f68c9253"
)
ABORTED_V1_1_ACCESS_SHA256 = (
    "e68a95aeef7fb7008e05baaad98171c0cbeae6166d4c08992bbde7139badc74a"
)
ABORTED_V1_1_MODEL_ARTIFACT_SHA256 = (
    "e5882bb6e4fb0283234f7b3e2c62b990c7c7af04de656969bf77a1c81ef0d220"
)
ABORTED_V1_1_TEMPLATE_ARTIFACT_SHA256 = (
    "e71247efaa232f1ae037ff5ccd5401913e4ad5466ab79b78fe7e787e46e272e4"
)
ACQUISITION_PROTOCOL_SHA256 = (
    "25840109b6f75fe2e56ecd0dc259da0bd356205e1bd706596e26c9334f88e74d"
)
DATA_FILE_MANIFEST_SHA256 = (
    "99095794446310d12d918923b30ad533e0af159aa7c645676375b7e0290fdec7"
)
ARCHIVE_SHA256 = "24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2"
PROFILE_SHA256 = "867987ca884b2f0fd607ad3918329eba887d095af2befa21122cea518eaa584a"
LABEL_AUDIT_FINGERPRINT = (
    "2983dea038f1272ddc354a468e1c6427fec7d27ab71e1f73a8703060d0b5114a"
)

EXPECTED_GROUP_COUNTS = {"development": 18, "calibration": 9, "confirmation": 9}
EXPECTED_CYCLE_COUNTS = {
    "development": {100: 180, 90: 180, 80: 180, 73: 180},
    "calibration": {100: 90, 90: 90, 80: 90, 73: 90},
    "confirmation": {100: 99, 90: 90, 80: 90, 73: 90},
}
CPU_CUDA_ATOL = 1.0e-10
NUMERICAL_ALARM_GUARD = CPU_CUDA_ATOL
EFFECTIVE_BOUNDARY_RULE = (
    "nextafter(raw_pac_threshold + numerical_alarm_guard, positive_infinity)"
)
PAC_EPSILON = 0.05
PAC_DELTA = 0.05
PAC_CALIBRATION_COUNT = 90
OPTIMIZER_MAX_ITER = 2000
OPTIMIZER_MAX_EVAL = 2500
OPTIMIZER_TOLERANCE_GRAD = 1.0e-7
OPTIMIZER_TOLERANCE_CHANGE = 1.0e-10
OPTIMIZER_HISTORY_SIZE = 100
HEALTHY_MAXIMUM_ALARMS = 3
HEALTHY_POINT_FAP_MAXIMUM = 0.05
HEALTHY_UCB_MAXIMUM = 0.08
MINIMUM_RECALL_EACH_SEVERITY = 0.80
MINIMUM_OVERALL_FAULT_RECALL = 0.85
PASS_DECISION = "PASS_EXTERNAL_CONFIRMATION"
FAIL_DECISION = "FAIL_EXTERNAL_CONFIRMATION_NO_RETUNING"
DEVELOPMENT_DECISION = "DEVELOPMENT_COMPLETE_METHOD_FROZEN"
ACCESS_LEDGER_RELATIVE = "outputs/hydraulic_padr_pac_v1_3_confirmation_access_ledger.json"

DEVELOPMENT_MANIFEST = "development_manifest.json"
DEVELOPMENT_ACCESS = "development_access_intent.json"
DEVELOPMENT_COMPLETION = "development_completion.json"
CONFIRMATION_MANIFEST = "confirmation_manifest.json"
CONFIRMATION_ACCESS = "confirmation_access_intent.json"
CONFIRMATION_COMPLETION = "completion.json"

DEVELOPMENT_ARTIFACTS = (
    DEVELOPMENT_MANIFEST,
    DEVELOPMENT_ACCESS,
    "phase_template.npz",
    "calibrated_model.npz",
    "development_result.json",
    "calibration_certificate.json",
    "cuda_validation_development.json",
    "gpu_execution_development.json",
)
CONFIRMATION_ARTIFACTS = (
    CONFIRMATION_MANIFEST,
    CONFIRMATION_ACCESS,
    "confirmation_result.json",
    "confirmation_cycles.csv",
    "confirmation_scores.npz",
    "cuda_validation_confirmation.json",
    "gpu_execution_confirmation.json",
    "REPORT.md",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _effective_alarm_boundary(raw_pac_threshold: object) -> float:
    if isinstance(raw_pac_threshold, bool) or not isinstance(
        raw_pac_threshold, (int, float)
    ):
        raise ValueError("raw PAC threshold must be a finite nonnegative number")
    raw = float(raw_pac_threshold)
    if not math.isfinite(raw) or raw < 0.0:
        raise ValueError("raw PAC threshold must be a finite nonnegative number")
    guarded = raw + NUMERICAL_ALARM_GUARD
    effective = math.nextafter(guarded, math.inf)
    if not math.isfinite(effective) or effective <= raw:
        raise ValueError("effective alarm boundary must be finite and exceed raw PAC")
    return effective


def _alarm_boundary_record(raw_pac_threshold: object) -> dict[str, object]:
    effective = _effective_alarm_boundary(raw_pac_threshold)
    raw = float(raw_pac_threshold)
    return {
        "raw_pac_threshold": raw,
        "numerical_alarm_guard": NUMERICAL_ALARM_GUARD,
        "effective_alarm_boundary": effective,
        "effective_boundary_rule": EFFECTIVE_BOUNDARY_RULE,
        "comparison": "strictly_greater_than_effective_alarm_boundary",
    }


def _validate_alarm_boundary_record(
    record: object, *, expected_raw_pac_threshold: object
) -> dict[str, object]:
    expected = _alarm_boundary_record(expected_raw_pac_threshold)
    if not isinstance(record, Mapping) or dict(record) != expected:
        raise ValueError("raw PAC threshold/effective boundary binding is invalid")
    return expected


@dataclass(frozen=True)
class CycleRecord:
    """One stable official cycle selected solely from the label-only audit."""

    index: int
    valve_label: int
    nuisance_group: str
    episode_id: int
    split: str

    def as_dict(self) -> dict[str, object]:
        return {
            "cycle_index_zero_based": self.index,
            "cycle_number_one_based": self.index + 1,
            "valve_condition_percent": self.valve_label,
            "nuisance_group": self.nuisance_group,
            "episode_id": self.episode_id,
            "split": self.split,
        }


@dataclass(frozen=True)
class Preflight:
    analysis_protocol: dict[str, object]
    frozen_inputs: dict[str, object]
    source_sha256: dict[str, str]
    records: dict[str, tuple[CycleRecord, ...]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("refusing to serialize a non-finite JSON number")
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _clean_json(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    """Publish a complete new file and never replace an existing artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Publishing a hard link to the flushed same-directory temporary file
        # is atomic and fails if another process already created the target.
        os.link(temporary_name, path)
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_json_new(path: Path, value: object) -> None:
    _atomic_bytes_new(path, _json_bytes(value))


def _atomic_text_new(path: Path, value: str) -> None:
    _atomic_bytes_new(path, value.encode("utf-8"))


def _atomic_npz_new(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular JSON file is missing: {path}")
    def reject_nonfinite(token: str) -> object:
        raise ValueError(f"{path} contains forbidden non-finite JSON token {token}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_fingerprinted_json(
    path: Path, *, fingerprint_field: str
) -> tuple[dict[str, object], str]:
    value = _read_json(path)
    stored = value.get(fingerprint_field)
    body = dict(value)
    body.pop(fingerprint_field, None)
    if not isinstance(stored, str) or stored != _fingerprint(body):
        raise ValueError(f"{path} fingerprint is corrupt")
    return value, stored


def _parse_sidecar(path: Path, expected_filename: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required checksum sidecar is missing: {path}")
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path} must contain exactly one checksum line")
    parts = lines[0].split()
    if len(parts) != 2 or parts[1].lstrip("*") != expected_filename:
        raise ValueError(f"{path} does not authenticate {expected_filename}")
    digest = parts[0]
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path} contains an invalid SHA-256 digest")
    return digest


def _authenticated_json(
    path: Path,
    sidecar: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    declared = _parse_sidecar(sidecar, path.name)
    observed = _sha256_file(path)
    if declared != expected_sha256 or observed != expected_sha256:
        raise ValueError(f"frozen SHA-256 mismatch for {path.name}")
    return _read_json(path), {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
        "sidecar_filename": sidecar.name,
        "sidecar_sha256": _sha256_file(sidecar),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular file is missing: {path}")
    return {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_aborted_v1_attempt(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate the immutable v1 operational abort without opening sensors."""

    observed = _artifact_record(path)
    if observed["sha256"] != ABORTED_V1_RECORD_SHA256:
        raise ValueError("frozen v1 ABORTED record SHA-256 mismatch")
    aborted = _read_json(path)
    failure = aborted.get("observed_failure")
    diagnostic = aborted.get("development_only_diagnostic")
    governance = aborted.get("governance")
    provenance = aborted.get("provenance")
    if (
        aborted.get("schema") != "hydraulic_padr_pac_aborted_development"
        or aborted.get("schema_version") != 1
        or aborted.get("protocol_id") != BASE_PROTOCOL_ID
        or aborted.get("status") != "ABORTED_BEFORE_METHOD_FREEZE"
        or aborted.get("failed_phase") != "development_grouped_cv"
        or not isinstance(failure, Mapping)
        or failure.get("ridge_lambda") != RIDGE_LAMBDA_GRID[0]
        or failure.get("held_out_fold") != 0
        or failure.get("optimizer_iterations") != 500
        or failure.get("termination_reason") != "maximum_iterations"
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("max_iter") != OPTIMIZER_MAX_ITER
        or diagnostic.get("converged_at_iteration") != 621
        or diagnostic.get("termination_reason") != "optimizer_change_tolerance"
        or not isinstance(governance, Mapping)
        or governance.get("development_sensor_values_accessed") is not True
        or governance.get("healthy_calibration_sensor_values_loaded_and_transformed")
        is not True
        or governance.get("calibration_fault_sensor_values_accessed") is not False
        or governance.get("confirmation_sensor_values_decoded_or_numerically_parsed")
        is not False
        or governance.get("confirmation_access_ledger_created") is not False
        or governance.get("model_or_threshold_frozen") is not False
        or governance.get("v1_may_not_be_resumed") is not True
        or not isinstance(provenance, Mapping)
        or provenance.get("analysis_protocol_sha256")
        != BASE_ANALYSIS_PROTOCOL_SHA256
        or provenance.get("development_manifest_sha256")
        != ABORTED_V1_MANIFEST_SHA256
        or provenance.get("development_manifest_fingerprint")
        != ABORTED_V1_MANIFEST_FINGERPRINT
        or provenance.get("development_access_intent_sha256")
        != ABORTED_V1_ACCESS_SHA256
    ):
        raise ValueError("frozen v1 ABORTED record semantics changed")

    parent_dir = path.parent
    parent_manifest_path = parent_dir / DEVELOPMENT_MANIFEST
    parent_access_path = parent_dir / DEVELOPMENT_ACCESS
    _, parent_manifest_fingerprint = _read_fingerprinted_json(
        parent_manifest_path, fingerprint_field="protocol_fingerprint"
    )
    parent_access = _read_json(parent_access_path)
    if (
        _sha256_file(parent_manifest_path) != ABORTED_V1_MANIFEST_SHA256
        or parent_manifest_fingerprint != ABORTED_V1_MANIFEST_FINGERPRINT
        or _sha256_file(parent_access_path) != ABORTED_V1_ACCESS_SHA256
        or parent_access.get("phase_fingerprint")
        != ABORTED_V1_MANIFEST_FINGERPRINT
        or parent_access.get("confirmation_values_accessed") is not False
    ):
        raise ValueError("v1 partial development artifacts changed after abort")
    forbidden_parent_artifacts = (
        DEVELOPMENT_COMPLETION,
        CONFIRMATION_MANIFEST,
        CONFIRMATION_ACCESS,
        CONFIRMATION_COMPLETION,
    )
    if any((parent_dir / name).exists() for name in forbidden_parent_artifacts):
        raise ValueError("aborted v1 unexpectedly contains a completion/confirmation artifact")
    old_ledger = parent_dir.parent / "hydraulic_padr_pac_v1_confirmation_access_ledger.json"
    if old_ledger.exists():
        raise ValueError("aborted v1 unexpectedly has a confirmation access ledger")
    return aborted, {"filename": path.name, **observed}


def _verify_aborted_parent_attempt(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate the v1.2 pre-confirmation abort and frozen development."""

    observed = _artifact_record(path)
    if observed["sha256"] != ABORTED_PARENT_RECORD_SHA256:
        raise ValueError("frozen v1.2 ABORTED record SHA-256 mismatch")
    aborted = _read_json(path)
    failure = aborted.get("observed_failure")
    governance = aborted.get("governance")
    frozen_method = aborted.get("frozen_method")
    provenance = aborted.get("provenance")
    if (
        aborted.get("schema") != "hydraulic_padr_pac_aborted_confirmation_entry"
        or aborted.get("schema_version") != 3
        or aborted.get("protocol_id") != PARENT_PROTOCOL_ID
        or aborted.get("status") != "ABORTED_BEFORE_CONFIRMATION_SENSOR_ACCESS"
        or aborted.get("failed_phase")
        != "confirmation_development_artifact_authentication"
        or not isinstance(failure, Mapping)
        or failure.get("validator") != "_selected_lambda_from_cv"
        or failure.get("actual_group_encoding") != "string:{raw_group}"
        or failure.get("incorrect_expected_group_encoding") != "{raw_group}"
        or failure.get("affected_computation") != "authentication only"
        or failure.get("model_training_scoring_threshold_or_gate_affected") is not False
        or not isinstance(governance, Mapping)
        or governance.get("development_completed_and_method_frozen") is not True
        or governance.get("confirmation_manifest_created") is not False
        or governance.get("confirmation_access_ledger_created") is not False
        or governance.get("confirmation_sensor_values_decoded_or_numerically_parsed")
        is not False
        or governance.get("confirmation_scoring_performed") is not False
        or governance.get("v1_2_may_not_be_resumed") is not True
        or not isinstance(frozen_method, Mapping)
        or frozen_method.get("selected_ridge_lambda") != 0.1
        or frozen_method.get("raw_pac_threshold") != 250.44148069867225
        or frozen_method.get("numerical_alarm_guard") != NUMERICAL_ALARM_GUARD
        or frozen_method.get("effective_alarm_boundary")
        != _effective_alarm_boundary(250.44148069867225)
        or frozen_method.get("pac_sample_count") != PAC_CALIBRATION_COUNT
        or frozen_method.get("pac_rank_one_based") != PAC_CALIBRATION_COUNT
        or frozen_method.get("cpu_cuda_alarm_exact") is not True
        or not isinstance(provenance, Mapping)
        or provenance.get("analysis_protocol_sha256")
        != PARENT_ANALYSIS_PROTOCOL_SHA256
        or provenance.get("development_manifest_sha256")
        != PARENT_DEVELOPMENT_MANIFEST_SHA256
        or provenance.get("development_manifest_fingerprint")
        != PARENT_DEVELOPMENT_MANIFEST_FINGERPRINT
        or provenance.get("development_completion_sha256")
        != PARENT_DEVELOPMENT_COMPLETION_SHA256
        or provenance.get("development_completion_fingerprint")
        != PARENT_DEVELOPMENT_COMPLETION_FINGERPRINT
    ):
        raise ValueError("frozen v1.2 ABORTED record semantics changed")

    parent_dir = path.parent
    completion_path = parent_dir / DEVELOPMENT_COMPLETION
    completion, completion_fingerprint = _read_fingerprinted_json(
        completion_path, fingerprint_field="completion_fingerprint"
    )
    if (
        _sha256_file(completion_path) != PARENT_DEVELOPMENT_COMPLETION_SHA256
        or completion_fingerprint != PARENT_DEVELOPMENT_COMPLETION_FINGERPRINT
        or completion.get("schema") != "hydraulic_padr_pac_phase_completion"
        or completion.get("schema_version") != 1
        or completion.get("protocol_id") != PARENT_PROTOCOL_ID
        or completion.get("decision") != DEVELOPMENT_DECISION
        or completion.get("phase_fingerprint")
        != PARENT_DEVELOPMENT_MANIFEST_FINGERPRINT
    ):
        raise ValueError("frozen v1.2 development completion changed")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        PARENT_DEVELOPMENT_ARTIFACT_SHA256
    ):
        raise ValueError("frozen v1.2 development artifact set changed")
    parent_artifacts: dict[str, dict[str, object]] = {}
    for name, expected_sha256 in PARENT_DEVELOPMENT_ARTIFACT_SHA256.items():
        actual = _artifact_record(parent_dir / name)
        parent_artifacts[name] = actual
        completion_record = artifacts.get(name)
        if (
            actual["sha256"] != expected_sha256
            or not isinstance(completion_record, Mapping)
            or actual
            != {
                "size_bytes": int(completion_record.get("size_bytes", -1)),
                "sha256": completion_record.get("sha256"),
            }
        ):
            raise ValueError(f"frozen v1.2 development artifact changed: {name}")
    manifest, manifest_fingerprint = _read_fingerprinted_json(
        parent_dir / DEVELOPMENT_MANIFEST,
        fingerprint_field="protocol_fingerprint",
    )
    source_hashes = manifest.get("source_sha256")
    expected_parent_sources = {
        "scripts/run_hydraulic_padr_pac_gpu.py": provenance.get(
            "runner_sha256_at_failure"
        ),
        "tests/test_run_hydraulic_padr_pac_gpu.py": provenance.get(
            "runner_test_sha256_at_failure"
        ),
        "src/mate_spc_hydraulic/model.py": provenance.get(
            "model_source_sha256_at_failure"
        ),
        "src/mate_spc_hydraulic/data.py": provenance.get(
            "data_source_sha256_at_failure"
        ),
    }
    if (
        manifest_fingerprint != PARENT_DEVELOPMENT_MANIFEST_FINGERPRINT
        or manifest.get("protocol_id") != PARENT_PROTOCOL_ID
        or not isinstance(source_hashes, Mapping)
        or any(source_hashes.get(name) != digest for name, digest in expected_parent_sources.items())
        or any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in source_hashes.values()
        )
    ):
        raise ValueError("frozen v1.2 development source hashes changed")
    old_ledger = parent_dir.parent / "hydraulic_padr_pac_v1_2_confirmation_access_ledger.json"
    if old_ledger.exists() or any(
        (parent_dir / name).exists()
        for name in (CONFIRMATION_MANIFEST, CONFIRMATION_ACCESS, CONFIRMATION_COMPLETION)
    ):
        raise ValueError("aborted v1.2 unexpectedly contains a confirmation artifact")
    return aborted, {
        "filename": path.name,
        **observed,
        "frozen_development": {
            "completion": {
                "filename": DEVELOPMENT_COMPLETION,
                **_artifact_record(completion_path),
                "completion_fingerprint": completion_fingerprint,
            },
            "manifest_fingerprint": manifest_fingerprint,
            "artifacts": parent_artifacts,
        },
    }


def _source_hashes(project_root: Path) -> dict[str, str]:
    package_files = sorted((project_root / "src" / "mate_spc_hydraulic").glob("*.py"))
    pac_files = sorted((project_root / "src" / "mate_spc_pac").glob("*.py"))
    test_files = sorted((project_root / "tests").glob("*hydraulic*.py")) + sorted(
        (project_root / "tests").glob("*pac*.py")
    )
    paths = tuple(
        sorted(
            set(
                package_files
                + pac_files
                + test_files
                + [
                    project_root / "scripts" / "run_hydraulic_padr_pac_gpu.py",
                    project_root / "scripts" / "audit_hydraulic_labels.py",
                    project_root / "pyproject.toml",
                ]
            )
        )
    )
    if not package_files or not pac_files:
        raise FileNotFoundError("hydraulic or PAC source package is incomplete")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"protocol source files are missing: {missing}")
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path) for path in paths
    }


def _verify_label_audit(
    audit_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    completion_path = audit_dir / "completion.json"
    completion, completion_fingerprint = _read_fingerprinted_json(
        completion_path,
        fingerprint_field="completion_fingerprint",
    )
    if completion.get("schema") != "hydraulic_label_only_audit_completion":
        raise ValueError("label-audit completion schema changed")
    if completion.get("sensor_values_accessed") is not False:
        raise ValueError("label-audit completion does not certify label-only access")
    if completion.get("audit_fingerprint") != LABEL_AUDIT_FINGERPRINT:
        raise ValueError("label-audit fingerprint is not the frozen value")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"label_audit.json"}:
        raise ValueError("label-audit completion artifact set changed")
    audit_path = audit_dir / "label_audit.json"
    expected = artifacts["label_audit.json"]
    if not isinstance(expected, Mapping) or _artifact_record(audit_path) != {
        "size_bytes": int(expected.get("size_bytes", -1)),
        "sha256": expected.get("sha256"),
    }:
        raise ValueError("label-audit artifact does not match its completion")
    audit = _read_json(audit_path)
    stored_audit_fingerprint = audit.get("audit_fingerprint")
    audit_body = dict(audit)
    audit_body.pop("audit_fingerprint", None)
    if (
        stored_audit_fingerprint != LABEL_AUDIT_FINGERPRINT
        or _fingerprint(audit_body) != LABEL_AUDIT_FINGERPRINT
    ):
        raise ValueError("label-audit canonical fingerprint is corrupt")
    if audit.get("schema") != "hydraulic_label_only_audit":
        raise ValueError("label-audit schema changed")
    scope = audit.get("audit_scope")
    if not isinstance(scope, Mapping) or scope.get("sensor_values_accessed") is not False:
        raise ValueError("label-audit scope is not label-only")
    record = {
        "completion": {
            **_artifact_record(completion_path),
            "completion_fingerprint": completion_fingerprint,
        },
        "label_audit": {
            **_artifact_record(audit_path),
            "audit_fingerprint": LABEL_AUDIT_FINGERPRINT,
        },
    }
    return audit, record


def _verify_archive(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    expected = manifest.get("archive")
    if not isinstance(expected, Mapping):
        raise ValueError("data-file manifest has no archive record")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Hydraulic source archive is missing: {path}")
    observed = _artifact_record(path)
    if (
        path.name != expected.get("filename")
        or observed["size_bytes"] != int(expected.get("size_bytes", -1))
        or observed["sha256"] != expected.get("sha256")
        or observed["sha256"] != ARCHIVE_SHA256
    ):
        raise ValueError("Hydraulic source archive integrity check failed")
    return {"filename": path.name, **observed}


def _verify_data_files(
    dataset_dir: Path, file_records: object
) -> dict[str, dict[str, object]]:
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Hydraulic data directory is missing: {dataset_dir}")
    root = dataset_dir.resolve(strict=True)
    if not isinstance(file_records, list) or len(file_records) != 20:
        raise ValueError("frozen data-file manifest must contain exactly 20 files")
    expected_names: list[str] = []
    result: dict[str, dict[str, object]] = {}
    for record in file_records:
        if not isinstance(record, Mapping):
            raise ValueError("invalid data-file manifest record")
        name = record.get("name")
        if not isinstance(name, str) or Path(name).name != name or not name:
            raise ValueError("data-file manifest contains an unsafe filename")
        if name in expected_names:
            raise ValueError("data-file manifest contains a duplicate filename")
        expected_names.append(name)
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True).parent != root:
            raise ValueError(f"Hydraulic data file is missing or unsafe: {name}")
        observed = _artifact_record(path)
        if (
            observed["size_bytes"] != int(record.get("size_bytes", -1))
            or observed["sha256"] != record.get("sha256")
        ):
            raise ValueError(f"Hydraulic data file integrity check failed: {name}")
        result[name] = observed
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != set(expected_names) or any(path.is_dir() for path in root.iterdir()):
        raise ValueError("Hydraulic extracted directory differs from the frozen 20-file set")
    if result.get("profile.txt", {}).get("sha256") != PROFILE_SHA256:
        raise ValueError("profile.txt is not the frozen label file")
    return dict(sorted(result.items()))


def _group_key(profile: Mapping[str, object]) -> str:
    try:
        cooler = int(profile["cooler_condition_percent"])
        pump = int(profile["internal_pump_leakage"])
        accumulator = int(profile["hydraulic_accumulator_bar"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid nuisance profile in label audit") from error
    return f"C{cooler:03d}-P{pump}-A{accumulator:03d}"


def _build_cycle_records(
    protocol: Mapping[str, object], audit: Mapping[str, object]
) -> dict[str, tuple[CycleRecord, ...]]:
    split = protocol.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("analysis protocol has no split definition")
    split_groups: dict[str, tuple[str, ...]] = {}
    for name in ("development", "calibration", "confirmation"):
        raw = split.get(f"{name}_groups")
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError(f"analysis protocol has invalid {name} groups")
        values = tuple(raw)
        if len(values) != EXPECTED_GROUP_COUNTS[name] or len(set(values)) != len(values):
            raise ValueError(f"analysis protocol {name} group count changed")
        split_groups[name] = values
    all_groups = [group for values in split_groups.values() for group in values]
    if len(all_groups) != 36 or len(set(all_groups)) != 36:
        raise ValueError("analysis split groups overlap or are incomplete")

    episodes = audit.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("label audit has no episode list")
    by_group_label: dict[tuple[str, int], tuple[int, int, int]] = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("invalid episode record")
        profile = episode.get("profile")
        if not isinstance(profile, Mapping) or int(profile.get("stable_flag", -1)) != 0:
            continue
        group = _group_key(profile)
        label = int(profile.get("valve_condition_percent", -1))
        if group not in set(all_groups) or label not in (100, 90, 80, 73):
            raise ValueError("stable label-audit episode is outside the frozen design")
        key = (group, label)
        if key in by_group_label:
            raise ValueError(f"multiple stable episodes found for {group}, valve={label}")
        start = int(episode.get("start_row_1based", 0))
        end = int(episode.get("end_row_1based_inclusive", 0))
        episode_id = int(episode.get("episode_id", 0))
        length = int(episode.get("length_cycles", 0))
        if start < 1 or end < start or end - start + 1 != length or episode_id < 1:
            raise ValueError("invalid stable episode bounds")
        by_group_label[key] = (start - 1, end, episode_id)
    expected_cells = {(group, label) for group in all_groups for label in (100, 90, 80, 73)}
    if set(by_group_label) != expected_cells:
        raise ValueError("stable audit cells do not match the frozen 36-by-4 design")

    result: dict[str, tuple[CycleRecord, ...]] = {}
    for split_name, groups in split_groups.items():
        records: list[CycleRecord] = []
        for group in groups:
            for label in (100, 90, 80, 73):
                start, stop, episode_id = by_group_label[(group, label)]
                records.extend(
                    CycleRecord(index, label, group, episode_id, split_name)
                    for index in range(start, stop)
                )
        counts = {
            label: sum(record.valve_label == label for record in records)
            for label in (100, 90, 80, 73)
        }
        if counts != EXPECTED_CYCLE_COUNTS[split_name]:
            raise ValueError(
                f"{split_name} stable cycle counts changed: {counts} != "
                f"{EXPECTED_CYCLE_COUNTS[split_name]}"
            )
        if len({record.index for record in records}) != len(records):
            raise ValueError(f"{split_name} contains duplicate cycle indices")
        result[split_name] = tuple(records)
    selected = [record.index for records in result.values() for record in records]
    if len(selected) != len(set(selected)):
        raise ValueError("cycle indices overlap across frozen partitions")
    return result


def _validate_protocol(
    protocol: Mapping[str, object],
    parent_protocol: Mapping[str, object],
    aborted_parent: Mapping[str, object],
    base_protocol: Mapping[str, object],
    aborted_v1: Mapping[str, object],
) -> None:
    if (
        protocol.get("schema") != "hydraulic_padr_pac_analysis_protocol"
        or protocol.get("schema_version") != 4
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("protocol_version") != "1.3"
        or protocol.get("frozen_before_sensor_access") is not False
        or protocol.get("base_protocol_frozen_before_sensor_access") is not True
        or protocol.get("frozen_after_development_sensor_access") is not True
        or protocol.get("frozen_after_healthy_calibration_sensor_access") is not True
        or protocol.get("amendment_frozen_before_confirmation_sensor_access")
        is not True
        or protocol.get("status")
        != "prospective_external_confirmation_validator_amendment"
    ):
        raise ValueError("v1.3 analysis protocol identity or freeze timing changed")
    if (
        parent_protocol.get("schema") != "hydraulic_padr_pac_analysis_protocol"
        or parent_protocol.get("schema_version") != 3
        or parent_protocol.get("protocol_id") != PARENT_PROTOCOL_ID
        or parent_protocol.get("protocol_version") != "1.2"
        or parent_protocol.get("amendment_frozen_before_confirmation_sensor_access")
        is not True
        or parent_protocol.get("status")
        != "prospective_external_confirmation_amendment"
    ):
        raise ValueError("parent v1.2 analysis protocol identity changed")
    if (
        base_protocol.get("schema") != "hydraulic_padr_pac_analysis_protocol"
        or base_protocol.get("schema_version") != 1
        or base_protocol.get("protocol_id") != BASE_PROTOCOL_ID
        or base_protocol.get("frozen_before_sensor_access") is not True
        or base_protocol.get("status") != "prospective_external_confirmation"
    ):
        raise ValueError("base v1 analysis protocol identity changed")
    expected_parent = {
        "protocol_id": PARENT_PROTOCOL_ID,
        "filename": "hydraulic_padr_pac_v1_2_analysis.json",
        "sha256": PARENT_ANALYSIS_PROTOCOL_SHA256,
    }
    expected_abort = {
        "record_path": "outputs/hydraulic_padr_pac_gpu_v1_2/ABORTED.json",
        "sha256": ABORTED_PARENT_RECORD_SHA256,
        "status": "ABORTED_BEFORE_CONFIRMATION_SENSOR_ACCESS",
    }
    parent_prior = parent_protocol.get("prior_lineage")
    if not isinstance(parent_prior, Mapping):
        raise ValueError("parent v1.2 prior lineage is missing")
    expected_inherited_lineage = {
        "v1_2_parent_protocol": parent_protocol.get("parent_protocol"),
        "v1_2_aborted_parent_attempt": parent_protocol.get("aborted_parent_attempt"),
        "base_protocol": {
            "protocol_id": BASE_PROTOCOL_ID,
            "filename": "hydraulic_padr_pac_v1_analysis.json",
            "sha256": BASE_ANALYSIS_PROTOCOL_SHA256,
        },
        "aborted_v1_attempt": {
            "record_path": "outputs/hydraulic_padr_pac_gpu_v1/ABORTED.json",
            "sha256": ABORTED_V1_RECORD_SHA256,
            "status": "ABORTED_BEFORE_METHOD_FREEZE",
        },
    }
    if (
        protocol.get("parent_protocol") != expected_parent
        or protocol.get("aborted_parent_attempt") != expected_abort
        or protocol.get("inherited_prior_lineage") != expected_inherited_lineage
        or aborted_parent.get("protocol_id") != PARENT_PROTOCOL_ID
        or aborted_parent.get("status") != expected_abort["status"]
        or aborted_v1.get("protocol_id") != BASE_PROTOCOL_ID
        or aborted_v1.get("status") != "ABORTED_BEFORE_METHOD_FREEZE"
        or parent_prior.get("base_protocol")
        != expected_inherited_lineage["base_protocol"]
        or parent_prior.get("aborted_v1_attempt")
        != expected_inherited_lineage["aborted_v1_attempt"]
    ):
        raise ValueError("v1.3 parent or prior abort lineage changed")
    expected_parent_development = {
        "protocol_id": PARENT_PROTOCOL_ID,
        "output_directory": "outputs/hydraulic_padr_pac_gpu_v1_2",
        "development_manifest_sha256": PARENT_DEVELOPMENT_MANIFEST_SHA256,
        "development_manifest_fingerprint": PARENT_DEVELOPMENT_MANIFEST_FINGERPRINT,
        "development_access_intent_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "development_access_intent.json"
        ],
        "development_completion_sha256": PARENT_DEVELOPMENT_COMPLETION_SHA256,
        "development_completion_fingerprint": PARENT_DEVELOPMENT_COMPLETION_FINGERPRINT,
        "calibrated_model_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "calibrated_model.npz"
        ],
        "phase_template_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "phase_template.npz"
        ],
        "development_result_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "development_result.json"
        ],
        "calibration_certificate_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "calibration_certificate.json"
        ],
        "cuda_validation_development_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "cuda_validation_development.json"
        ],
        "gpu_execution_development_sha256": PARENT_DEVELOPMENT_ARTIFACT_SHA256[
            "gpu_execution_development.json"
        ],
    }
    if protocol.get("frozen_parent_development") != expected_parent_development:
        raise ValueError("v1.3 frozen v1.2 development anchors changed")
    amendment = protocol.get("amendment")
    expected_only_change = {
        "confirmation_authentication.expected_cv_held_out_nuisance_groups": {
            "from": "{raw_protocol_group}",
            "to": "string:{raw_protocol_group}",
            "reason": (
                "mate_spc_hydraulic.model._normalize_nuisance_groups canonically "
                "prefixes nonempty string identifiers with string: before CV results "
                "are serialized"
            ),
            "training_scoring_threshold_gate_or_numeric_result_changed": False,
        }
    }
    if (
        not isinstance(amendment, Mapping)
        or amendment.get("scope")
        != "confirmation_entry_validator_representation_only"
        or amendment.get("only_change") != expected_only_change
        or amendment.get("frozen_v1_2_development_artifacts_bound_as_reference")
        is not True
        or amendment.get("deterministic_development_replay_required_in_new_v1_3_output")
        is not True
        or amendment.get("development_values_previously_accessed") is not True
        or amendment.get(
            "healthy_calibration_values_previously_loaded_transformed_calibrated_and_scored"
        )
        is not True
        or amendment.get("calibration_fault_values_accessed") is not False
        or amendment.get("confirmation_values_decoded_or_numerically_parsed")
        is not False
        or amendment.get("confirmation_manifest_created") is not False
        or amendment.get("confirmation_access_ledger_created") is not False
        or amendment.get(
            "confirmation_fit_selection_calibration_or_scoring_performed"
        )
        is not False
    ):
        raise ValueError("v1.3 validator-only amendment declaration changed")

    # Every scientific and numerical method section is identical to v1.2.
    for section in (
        "inputs",
        "estimand",
        "split",
        "features",
        "directional_model",
        "online_statistic",
        "calibration",
        "confirmation",
    ):
        if protocol.get(section) != parent_protocol.get(section):
            raise ValueError(f"v1.3 changed frozen v1.2 method section: {section}")
    if (
        protocol.get("method_name") != parent_protocol.get("method_name")
        or protocol.get("method_expansion")
        != parent_protocol.get("method_expansion")
    ):
        raise ValueError("v1.3 changed the parent method identity")

    inputs = protocol.get("inputs")
    expected_inputs = {
        "acquisition_protocol_sha256": ACQUISITION_PROTOCOL_SHA256,
        "label_audit_fingerprint": LABEL_AUDIT_FINGERPRINT,
        "archive_sha256": ARCHIVE_SHA256,
        "profile_sha256": PROFILE_SHA256,
    }
    if not isinstance(inputs, Mapping) or dict(inputs) != expected_inputs:
        raise ValueError("analysis protocol input anchors changed")
    model = protocol.get("directional_model")
    online = protocol.get("online_statistic")
    calibration = protocol.get("calibration")
    confirmation = protocol.get("confirmation")
    if not all(
        isinstance(item, Mapping)
        for item in (model, online, calibration, confirmation)
    ):
        raise ValueError("analysis protocol method sections are missing")
    if (
        int(model.get("max_iter", -1)) != OPTIMIZER_MAX_ITER
        or int(model.get("max_eval", -1)) != OPTIMIZER_MAX_EVAL
        or tuple(model.get("heads_in_order", ())) != POSITIVE_LABELS
        or int(model.get("normal_label", -1)) != HEALTHY_LABEL
        or tuple(float(x) for x in model.get("ridge_lambda_grid", ()))
        != RIDGE_LAMBDA_GRID
        or model.get("required_device") != "cuda"
        or model.get("optimizer") != "PyTorch CUDA float64 LBFGS"
        or float(model.get("tolerance_grad", -1.0))
        != OPTIMIZER_TOLERANCE_GRAD
        or float(model.get("tolerance_change", -1.0))
        != OPTIMIZER_TOLERANCE_CHANGE
        or int(model.get("history_size", -1)) != OPTIMIZER_HISTORY_SIZE
        or model.get("tf32") is not False
        or float(online.get("page_kappa", -1.0)) != DEFAULT_KAPPA
        or int(calibration.get("expected_count", -1)) != PAC_CALIBRATION_COUNT
        or float(calibration.get("epsilon", -1.0)) != PAC_EPSILON
        or float(calibration.get("delta", -1.0)) != PAC_DELTA
        or int(confirmation.get("healthy_maximum_alarms", -1))
        != HEALTHY_MAXIMUM_ALARMS
        or float(confirmation.get("healthy_exact_one_sided_95pct_ucb_maximum", -1.0))
        != HEALTHY_UCB_MAXIMUM
        or float(confirmation.get("healthy_point_fap_maximum", -1.0))
        != HEALTHY_POINT_FAP_MAXIMUM
        or float(confirmation.get("minimum_recall_each_severity", -1.0))
        != MINIMUM_RECALL_EACH_SEVERITY
        or float(confirmation.get("minimum_overall_fault_recall", -1.0))
        != MINIMUM_OVERALL_FAULT_RECALL
        or float(confirmation.get("cpu_cuda_max_abs_error", -1.0))
        != CPU_CUDA_ATOL
    ):
        raise ValueError("analysis protocol method constants changed")
    governance = protocol.get("governance")
    if not isinstance(governance, Mapping):
        raise ValueError("analysis protocol governance is missing")
    if (
        governance.get("source_hashes_and_data_file_hashes_locked_before_measurement_loading")
        is not True
        or governance.get("confirmation_may_not_change_any_method_or_gate") is not True
        or governance.get("failed_confirmation_forbids_same_data_retuning") is not True
        or governance.get("parent_v1_attempt_permanently_aborted") is not True
        or governance.get("parent_v1_1_attempt_permanently_aborted") is not True
        or governance.get("parent_v1_2_attempt_permanently_aborted") is not True
        or governance.get("v1_2_completed_development_artifacts_bound_as_reference")
        is not True
        or governance.get("v1_3_development_is_a_deterministic_replay_without_method_change")
        is not True
        or governance.get("v1_3_validator_amendment_frozen_before_confirmation_numeric_access")
        is not True
    ):
        raise ValueError("v1.3 amendment governance changed")


def _preflight(args: argparse.Namespace, project_root: Path) -> Preflight:
    protocol, protocol_record = _authenticated_json(
        args.analysis_protocol,
        args.analysis_protocol_sidecar,
        expected_sha256=ANALYSIS_PROTOCOL_SHA256,
    )
    parent_protocol, parent_protocol_record = _authenticated_json(
        args.parent_analysis_protocol,
        args.parent_analysis_protocol_sidecar,
        expected_sha256=PARENT_ANALYSIS_PROTOCOL_SHA256,
    )
    base_protocol, base_protocol_record = _authenticated_json(
        args.base_analysis_protocol,
        args.base_analysis_protocol_sidecar,
        expected_sha256=BASE_ANALYSIS_PROTOCOL_SHA256,
    )
    aborted_parent, aborted_parent_record = _verify_aborted_parent_attempt(
        args.aborted_parent_record
    )
    aborted_v1, aborted_v1_record = _verify_aborted_v1_attempt(args.aborted_v1_record)
    _validate_protocol(
        protocol,
        parent_protocol,
        aborted_parent,
        base_protocol,
        aborted_v1,
    )
    acquisition, acquisition_record = _authenticated_json(
        args.acquisition_protocol,
        args.acquisition_protocol_sidecar,
        expected_sha256=ACQUISITION_PROTOCOL_SHA256,
    )
    if (
        acquisition.get("schema") != "hydraulic_ws_pac_acquisition_protocol"
        or acquisition.get("protocol_id") != "hydraulic_ws_pac_v1"
        or acquisition.get("frozen_before_download") is not True
    ):
        raise ValueError("acquisition protocol identity changed")
    file_manifest, file_manifest_record = _authenticated_json(
        args.file_manifest,
        args.file_manifest_sidecar,
        expected_sha256=DATA_FILE_MANIFEST_SHA256,
    )
    if (
        file_manifest.get("schema") != "hydraulic_uci_447_file_manifest"
        or file_manifest.get("created_before_numeric_sensor_parsing") is not True
    ):
        raise ValueError("data-file manifest identity changed")
    audit, audit_record = _verify_label_audit(args.label_audit_dir)
    archive_record = _verify_archive(args.archive, file_manifest)
    data_records = _verify_data_files(args.data_dir, file_manifest.get("files"))
    source_hashes = _source_hashes(project_root)
    records = _build_cycle_records(protocol, audit)

    audit_inputs = audit.get("inputs")
    if not isinstance(audit_inputs, Mapping):
        raise ValueError("label audit has no authenticated inputs")
    audit_profile = audit_inputs.get("profile")
    audit_archive = audit_inputs.get("archive")
    if (
        not isinstance(audit_profile, Mapping)
        or audit_profile.get("sha256") != PROFILE_SHA256
        or not isinstance(audit_archive, Mapping)
        or audit_archive.get("sha256") != ARCHIVE_SHA256
    ):
        raise ValueError("label-audit input hashes disagree with the analysis protocol")

    frozen_inputs = {
        "analysis_protocol": protocol_record,
        "parent_analysis_protocol": parent_protocol_record,
        "aborted_parent_attempt": aborted_parent_record,
        "base_analysis_protocol": base_protocol_record,
        "aborted_v1_attempt": aborted_v1_record,
        "acquisition_protocol": acquisition_record,
        "data_file_manifest": file_manifest_record,
        "label_audit": audit_record,
        "source_archive": archive_record,
        "extracted_files": data_records,
    }
    return Preflight(
        analysis_protocol=protocol,
        frozen_inputs=frozen_inputs,
        source_sha256=source_hashes,
        records=records,
    )


def _cycle_plan(records: Sequence[CycleRecord]) -> dict[str, object]:
    return {
        "cycle_count": len(records),
        "cycle_indices_zero_based": [record.index for record in records],
        "cycle_count_by_label": {
            str(label): sum(record.valve_label == label for record in records)
            for label in (100, 90, 80, 73)
        },
        "nuisance_groups": list(dict.fromkeys(record.nuisance_group for record in records)),
        "episode_ids": sorted({record.episode_id for record in records}),
        "records": [record.as_dict() for record in records],
    }


def _calibration_cycle_names(records: Sequence[CycleRecord]) -> tuple[str, ...]:
    if len(records) != PAC_CALIBRATION_COUNT or any(
        record.valve_label != HEALTHY_LABEL for record in records
    ):
        raise ValueError("PAC calibration names require exactly 90 healthy cycles")
    names = tuple(
        f"cycle-{record.index + 1:04d}-group-{record.nuisance_group}"
        for record in records
    )
    if len(set(names)) != len(names):
        raise ValueError("PAC calibration cycle names are not unique")
    return names


def _development_manifest_body(
    preflight: Preflight, *, batch_size: int
) -> dict[str, object]:
    development = preflight.records["development"]
    calibration_healthy = tuple(
        record
        for record in preflight.records["calibration"]
        if record.valve_label == HEALTHY_LABEL
    )
    return {
        "schema": "hydraulic_padr_pac_development_manifest",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": "development_and_healthy_calibration",
        "frozen_inputs": preflight.frozen_inputs,
        "source_sha256": preflight.source_sha256,
        "cycle_plan": {
            "development_all_valve_classes": _cycle_plan(development),
            "calibration_healthy_only": _cycle_plan(calibration_healthy),
            "calibration_fault_cycles_forbidden": True,
            "confirmation_cycles_forbidden": True,
        },
        "method": {
            "name": METHOD_NAME,
            "raw_cycle_shape": [CYCLE_SECONDS, RAW_CHANNEL_COUNT],
            "feature_cycle_shape": [CYCLE_SECONDS, OUTPUT_CHANNEL_COUNT],
            "heads_in_order": list(POSITIVE_LABELS),
            "ridge_lambda_grid": list(RIDGE_LAMBDA_GRID),
            "grouped_cv_folds": 3,
            "kappa": DEFAULT_KAPPA,
            "persistence": "3_of_3",
            "PAC": {
                "healthy_cycle_count": PAC_CALIBRATION_COUNT,
                "epsilon": PAC_EPSILON,
                "delta": PAC_DELTA,
                "comparison": "strictly_greater_than",
                "raw_certificate_threshold_unchanged": True,
                "numerical_alarm_guard": NUMERICAL_ALARM_GUARD,
                "effective_boundary_rule": EFFECTIVE_BOUNDARY_RULE,
            },
        },
        "execution": {
            "required_device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "optimizer": {
                "name": "PyTorch CUDA LBFGS strong_wolfe",
                "max_iter": OPTIMIZER_MAX_ITER,
                "max_eval": OPTIMIZER_MAX_EVAL,
                "tolerance_grad": OPTIMIZER_TOLERANCE_GRAD,
                "tolerance_change": OPTIMIZER_TOLERANCE_CHANGE,
                "history_size": OPTIMIZER_HISTORY_SIZE,
            },
            "cpu_cuda_max_abs_error": CPU_CUDA_ATOL,
        },
    }


def _confirmation_manifest_body(
    preflight: Preflight,
    development_completion: Mapping[str, object],
    development_completion_path: Path,
    *,
    batch_size: int,
    access_ledger: Path,
) -> dict[str, object]:
    if access_ledger.name != Path(ACCESS_LEDGER_RELATIVE).name:
        raise ValueError("confirmation access ledger filename changed")
    return {
        "schema": "hydraulic_padr_pac_confirmation_manifest",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": "single_access_external_confirmation",
        "frozen_inputs": preflight.frozen_inputs,
        "source_sha256": preflight.source_sha256,
        "development": {
            "completion_filename": development_completion_path.name,
            "completion_sha256": _sha256_file(development_completion_path),
            "completion_fingerprint": development_completion["completion_fingerprint"],
            "development_manifest_fingerprint": development_completion["phase_fingerprint"],
            "decision": development_completion["decision"],
            "artifacts": development_completion["artifacts"],
        },
        "cycle_plan": {
            "confirmation_all_valve_classes": _cycle_plan(
                preflight.records["confirmation"]
            ),
            "single_access": True,
        },
        "global_access_ledger": {
            "workspace_relative_path": ACCESS_LEDGER_RELATIVE,
            "scope": "workspace_protocol_and_dataset",
            "atomic_claim": "same_directory_temporary_then_exclusive_hardlink",
            "claim_precedes_sensor_loading": True,
        },
        "execution": {
            "required_device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "single_batched_gpu_scoring_call": True,
            "complete_cpu_cuda_parity": True,
            "cpu_cuda_max_abs_error": CPU_CUDA_ATOL,
        },
        "gates": {
            "healthy_maximum_alarms": HEALTHY_MAXIMUM_ALARMS,
            "healthy_point_fap_maximum": HEALTHY_POINT_FAP_MAXIMUM,
            "healthy_exact_one_sided_95pct_ucb_maximum": HEALTHY_UCB_MAXIMUM,
            "minimum_recall_each_severity": MINIMUM_RECALL_EACH_SEVERITY,
            "minimum_overall_fault_recall": MINIMUM_OVERALL_FAULT_RECALL,
            "pass_decision": PASS_DECISION,
            "failure_decision": FAIL_DECISION,
            "all_required": True,
        },
        "prohibitions": {
            "fit": True,
            "calibration": True,
            "threshold_change": True,
            "hyperparameter_change": True,
            "gate_change": True,
            "same_data_retuning_after_failure": True,
        },
    }


def _load_or_create_manifest(path: Path, body: dict[str, object]) -> str:
    fingerprint = _fingerprint(body)
    if path.exists():
        current, stored = _read_fingerprinted_json(
            path, fingerprint_field="protocol_fingerprint"
        )
        observed_body = dict(current)
        observed_body.pop("protocol_fingerprint", None)
        if stored != fingerprint or observed_body != body:
            raise ValueError(f"{path} contains a different frozen manifest")
        return fingerprint
    _atomic_json_new(path, {**body, "protocol_fingerprint": fingerprint})
    return fingerprint


def _completion_body(
    root: Path,
    *,
    phase_fingerprint: str,
    decision: str,
    artifact_names: Iterable[str],
) -> dict[str, object]:
    artifacts = {
        name: _artifact_record(root / name) for name in sorted(set(artifact_names))
    }
    return {
        "schema": "hydraulic_padr_pac_phase_completion",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase_fingerprint": phase_fingerprint,
        "decision": decision,
        "artifacts": artifacts,
    }


def _write_completion(
    path: Path,
    root: Path,
    *,
    phase_fingerprint: str,
    decision: str,
    artifact_names: Iterable[str],
) -> None:
    body = _completion_body(
        root,
        phase_fingerprint=phase_fingerprint,
        decision=decision,
        artifact_names=artifact_names,
    )
    _atomic_json_new(path, {**body, "completion_fingerprint": _fingerprint(body)})


def _verify_completion(
    path: Path,
    root: Path,
    *,
    expected_artifacts: Sequence[str],
    expected_decisions: set[str],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    completion, _ = _read_fingerprinted_json(
        path, fingerprint_field="completion_fingerprint"
    )
    if (
        completion.get("schema") != "hydraulic_padr_pac_phase_completion"
        or completion.get("schema_version") != 1
        or completion.get("protocol_id") != PROTOCOL_ID
        or completion.get("decision") not in expected_decisions
    ):
        raise ValueError(f"{path} has invalid completion identity")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_artifacts):
        raise ValueError(f"{path} artifact set changed")
    for name, expected in artifacts.items():
        if not isinstance(expected, Mapping):
            raise ValueError(f"invalid completion artifact record for {name}")
        observed = _artifact_record(root / name)
        normalized = {
            "size_bytes": int(expected.get("size_bytes", -1)),
            "sha256": expected.get("sha256"),
        }
        if observed != normalized:
            raise ValueError(f"completed artifact changed: {name}")
    manifest_name = (
        DEVELOPMENT_MANIFEST if path.name == DEVELOPMENT_COMPLETION else CONFIRMATION_MANIFEST
    )
    manifest, manifest_fingerprint = _read_fingerprinted_json(
        root / manifest_name,
        fingerprint_field="protocol_fingerprint",
    )
    if manifest_fingerprint != completion.get("phase_fingerprint"):
        raise ValueError("completion and phase manifest fingerprints disagree")
    expected_manifest_identity = (
        (
            "hydraulic_padr_pac_development_manifest",
            "development_and_healthy_calibration",
        )
        if manifest_name == DEVELOPMENT_MANIFEST
        else (
            "hydraulic_padr_pac_confirmation_manifest",
            "single_access_external_confirmation",
        )
    )
    if (
        manifest.get("schema") != expected_manifest_identity[0]
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("phase") != expected_manifest_identity[1]
    ):
        raise ValueError("completed phase manifest protocol changed")
    return completion


def _verify_development_completion(output_dir: Path) -> dict[str, object] | None:
    return _verify_completion(
        output_dir / DEVELOPMENT_COMPLETION,
        output_dir,
        expected_artifacts=DEVELOPMENT_ARTIFACTS,
        expected_decisions={DEVELOPMENT_DECISION},
    )


def _records_from_confirmation_plan(plan: object) -> tuple[CycleRecord, ...]:
    if not isinstance(plan, Mapping) or set(plan) != {
        "cycle_count",
        "cycle_indices_zero_based",
        "cycle_count_by_label",
        "nuisance_groups",
        "episode_ids",
        "records",
    }:
        raise ValueError("confirmation cycle plan has an invalid schema")
    raw_records = plan.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("confirmation cycle plan has no exact records")
    records: list[CycleRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"confirmation cycle record {index} is invalid")
        values = (
            raw.get("cycle_index_zero_based"),
            raw.get("valve_condition_percent"),
            raw.get("nuisance_group"),
            raw.get("episode_id"),
        )
        if (
            any(isinstance(value, bool) for value in values)
            or not isinstance(values[0], int)
            or not isinstance(values[1], int)
            or not isinstance(values[2], str)
            or not isinstance(values[3], int)
            or raw.get("split") != "confirmation"
        ):
            raise ValueError(f"confirmation cycle record {index} has invalid fields")
        record = CycleRecord(
            index=int(values[0]),
            valve_label=int(values[1]),
            nuisance_group=str(values[2]),
            episode_id=int(values[3]),
            split="confirmation",
        )
        if record.as_dict() != dict(raw):
            raise ValueError(f"confirmation cycle record {index} is noncanonical")
        records.append(record)
    result = tuple(records)
    if _cycle_plan(result) != dict(plan):
        raise ValueError("confirmation cycle plan summary disagrees with its records")
    counts = {
        label: sum(record.valve_label == label for record in result)
        for label in (100, 90, 80, 73)
    }
    if (
        len(result) != 369
        or counts != EXPECTED_CYCLE_COUNTS["confirmation"]
        or len({record.nuisance_group for record in result}) != 9
        or len({record.index for record in result}) != len(result)
    ):
        raise ValueError("confirmation cycle plan does not match the frozen 9-group design")
    return result


def _load_confirmation_score_path(
    path: Path, records: Sequence[CycleRecord]
) -> HydraulicDirectionalPath:
    required = {
        "cycle_indices_zero_based",
        "valve_labels",
        "nuisance_groups",
        "episode_ids",
        "threshold",
        "raw_pac_threshold",
        "numerical_alarm_guard",
        "effective_alarm_boundary",
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
        "envelope",
        "eligible",
        "alarm",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("confirmation score archive schema changed")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("confirmation score archive contains an object array")
    indices, labels, groups, episodes = _records_arrays(records)
    for name, expected in (
        ("cycle_indices_zero_based", indices),
        ("valve_labels", labels),
        ("nuisance_groups", groups),
        ("episode_ids", episodes),
    ):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"confirmation score ordering changed: {name}")
    head_shape = (len(records), CYCLE_SECONDS, len(POSITIVE_LABELS))
    phase_shape = (len(records), CYCLE_SECONDS)
    for name in (
        "logits",
        "standardized_logits",
        "clipped_evidence",
        "cusum",
        "component_persistence",
    ):
        if arrays[name].shape != head_shape or arrays[name].dtype != np.float64:
            raise ValueError(f"confirmation score field {name} has an invalid shape/dtype")
    for name in ("envelope", "eligible", "alarm"):
        if arrays[name].shape != phase_shape:
            raise ValueError(f"confirmation score field {name} has an invalid shape")
    if arrays["envelope"].dtype != np.float64:
        raise ValueError("confirmation envelope must be float64")
    if arrays["eligible"].dtype != np.bool_ or arrays["alarm"].dtype != np.bool_:
        raise ValueError("confirmation eligibility/alarm fields must be boolean")
    for name in (
        "threshold",
        "raw_pac_threshold",
        "numerical_alarm_guard",
        "effective_alarm_boundary",
    ):
        if arrays[name].shape != (1,) or arrays[name].dtype != np.float64:
            raise ValueError(f"confirmation {name} field must be one float64 value")
    threshold = _finite_number(
        arrays["threshold"][0].item(),
        name="confirmation score threshold",
        nonnegative=True,
    )
    raw_pac_threshold = _finite_number(
        arrays["raw_pac_threshold"][0].item(),
        name="confirmation raw PAC threshold",
        nonnegative=True,
    )
    guard = _finite_number(
        arrays["numerical_alarm_guard"][0].item(),
        name="confirmation numerical alarm guard",
        nonnegative=True,
    )
    effective = _finite_number(
        arrays["effective_alarm_boundary"][0].item(),
        name="confirmation effective alarm boundary",
        nonnegative=True,
    )
    if (
        guard != NUMERICAL_ALARM_GUARD
        or effective != _effective_alarm_boundary(raw_pac_threshold)
        or threshold != effective
    ):
        raise ValueError("confirmation score archive boundary binding changed")
    for name in ("logits", "standardized_logits", "clipped_evidence", "cusum"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"confirmation score field {name} is non-finite")
    first = PERSISTENCE_WINDOW - 1
    component = arrays["component_persistence"]
    envelope = arrays["envelope"]
    if (
        not np.isneginf(component[:, :first, :]).all()
        or not np.isfinite(component[:, first:, :]).all()
        or not np.isneginf(envelope[:, :first]).all()
        or not np.isfinite(envelope[:, first:]).all()
        or np.any(arrays["cusum"] < 0.0)
        or np.any(component[:, first:, :] < 0.0)
        or not np.array_equal(envelope, np.max(component, axis=2))
    ):
        raise ValueError("confirmation Page/persistence envelope semantics changed")
    expected_eligible = np.zeros(phase_shape, dtype=bool)
    expected_eligible[:, first:] = True
    expected_alarm = expected_eligible & (envelope > threshold)
    if not np.array_equal(arrays["eligible"], expected_eligible) or not np.array_equal(
        arrays["alarm"], expected_alarm
    ):
        raise ValueError("confirmation alarms do not use the frozen strict threshold")
    return HydraulicDirectionalPath(
        positive_labels=POSITIVE_LABELS,
        logits=arrays["logits"],
        standardized_logits=arrays["standardized_logits"],
        clipped_evidence=arrays["clipped_evidence"],
        cusum=arrays["cusum"],
        component_persistence=component,
        envelope=envelope,
        eligible=arrays["eligible"],
        alarm=arrays["alarm"],
        threshold=threshold,
    )


def _validate_confirmation_cuda_record(
    validation: Mapping[str, object], *, phase_fingerprint: str
) -> None:
    if (
        validation.get("schema_version") != 1
        or validation.get("phase") != "single_access_external_confirmation"
        or validation.get("protocol_fingerprint") != phase_fingerprint
        or validation.get("passed") is not True
        or validation.get("validated_cycle_count") != 369
        or validation.get("atol") != CPU_CUDA_ATOL
        or validation.get("eligible_exact") is not True
        or validation.get("alarm_exact") is not True
    ):
        raise ValueError("confirmation CPU/CUDA validation identity or decision changed")
    for key, value in validation.items():
        if key.endswith("_max_abs_error") and _finite_number(
            value, name=key, nonnegative=True
        ) > CPU_CUDA_ATOL:
            raise ValueError(f"confirmation CUDA parity exceeds tolerance: {key}")


def _verify_confirmation_development_chain(
    output_dir: Path, manifest: Mapping[str, object]
) -> dict[str, object]:
    reference = manifest.get("development")
    if (
        not isinstance(reference, Mapping)
        or set(reference)
        != {
            "completion_filename",
            "completion_sha256",
            "completion_fingerprint",
            "development_manifest_fingerprint",
            "decision",
            "artifacts",
        }
        or reference.get("completion_filename") != DEVELOPMENT_COMPLETION
        or reference.get("decision") != DEVELOPMENT_DECISION
    ):
        raise ValueError("confirmation manifest has an invalid development reference")
    path = output_dir / DEVELOPMENT_COMPLETION
    development, fingerprint = _read_fingerprinted_json(
        path, fingerprint_field="completion_fingerprint"
    )
    if (
        development.get("schema") != "hydraulic_padr_pac_phase_completion"
        or development.get("schema_version") != 1
        or development.get("protocol_id") != PROTOCOL_ID
        or development.get("decision") != DEVELOPMENT_DECISION
        or _sha256_file(path) != reference.get("completion_sha256")
        or fingerprint != reference.get("completion_fingerprint")
        or development.get("phase_fingerprint")
        != reference.get("development_manifest_fingerprint")
        or development.get("artifacts") != reference.get("artifacts")
    ):
        raise ValueError("confirmation-to-development completion chain is corrupt")
    return development


def _validate_confirmation_manifest_semantics(manifest: Mapping[str, object]) -> None:
    expected_gates = {
        "healthy_maximum_alarms": HEALTHY_MAXIMUM_ALARMS,
        "healthy_point_fap_maximum": HEALTHY_POINT_FAP_MAXIMUM,
        "healthy_exact_one_sided_95pct_ucb_maximum": HEALTHY_UCB_MAXIMUM,
        "minimum_recall_each_severity": MINIMUM_RECALL_EACH_SEVERITY,
        "minimum_overall_fault_recall": MINIMUM_OVERALL_FAULT_RECALL,
        "pass_decision": PASS_DECISION,
        "failure_decision": FAIL_DECISION,
        "all_required": True,
    }
    expected_prohibitions = {
        "fit": True,
        "calibration": True,
        "threshold_change": True,
        "hyperparameter_change": True,
        "gate_change": True,
        "same_data_retuning_after_failure": True,
    }
    expected_ledger_spec = {
        "workspace_relative_path": ACCESS_LEDGER_RELATIVE,
        "scope": "workspace_protocol_and_dataset",
        "atomic_claim": "same_directory_temporary_then_exclusive_hardlink",
        "claim_precedes_sensor_loading": True,
    }
    execution = manifest.get("execution")
    cycle_plan = manifest.get("cycle_plan")
    frozen_inputs = manifest.get("frozen_inputs")
    source_hashes = manifest.get("source_sha256")
    if (
        manifest.get("gates") != expected_gates
        or manifest.get("prohibitions") != expected_prohibitions
        or manifest.get("global_access_ledger") != expected_ledger_spec
        or not isinstance(execution, Mapping)
        or execution.get("required_device") != "cuda"
        or execution.get("dtype") != "float64"
        or execution.get("tf32") is not False
        or execution.get("single_batched_gpu_scoring_call") is not True
        or execution.get("complete_cpu_cuda_parity") is not True
        or execution.get("cpu_cuda_max_abs_error") != CPU_CUDA_ATOL
        or isinstance(execution.get("batch_size"), bool)
        or not isinstance(execution.get("batch_size"), int)
        or int(execution["batch_size"]) < 1
        or not isinstance(cycle_plan, Mapping)
        or set(cycle_plan) != {"confirmation_all_valve_classes", "single_access"}
        or cycle_plan.get("single_access") is not True
        or not isinstance(frozen_inputs, Mapping)
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in source_hashes.values()
        )
    ):
        raise ValueError("confirmation manifest frozen semantics changed")
    required_input_hashes = (
        ("analysis_protocol", "sha256", ANALYSIS_PROTOCOL_SHA256),
        (
            "parent_analysis_protocol",
            "sha256",
            PARENT_ANALYSIS_PROTOCOL_SHA256,
        ),
        ("aborted_parent_attempt", "sha256", ABORTED_PARENT_RECORD_SHA256),
        ("base_analysis_protocol", "sha256", BASE_ANALYSIS_PROTOCOL_SHA256),
        ("aborted_v1_attempt", "sha256", ABORTED_V1_RECORD_SHA256),
        ("acquisition_protocol", "sha256", ACQUISITION_PROTOCOL_SHA256),
        ("data_file_manifest", "sha256", DATA_FILE_MANIFEST_SHA256),
        ("source_archive", "sha256", ARCHIVE_SHA256),
    )
    for section, field, expected in required_input_hashes:
        record = frozen_inputs.get(section)
        if not isinstance(record, Mapping) or record.get(field) != expected:
            raise ValueError(f"confirmation manifest input anchor changed: {section}")
    extracted = frozen_inputs.get("extracted_files")
    label_audit = frozen_inputs.get("label_audit")
    if (
        not isinstance(extracted, Mapping)
        or len(extracted) != 20
        or not isinstance(extracted.get("profile.txt"), Mapping)
        or extracted["profile.txt"].get("sha256") != PROFILE_SHA256
        or not isinstance(label_audit, Mapping)
        or not isinstance(label_audit.get("label_audit"), Mapping)
        or label_audit["label_audit"].get("audit_fingerprint")
        != LABEL_AUDIT_FINGERPRINT
    ):
        raise ValueError("confirmation manifest data/audit anchors changed")


def _verify_confirmation_ledger(
    access_ledger: Path,
    *,
    output_dir: Path,
    manifest: Mapping[str, object],
    phase_fingerprint: str,
    development: Mapping[str, object],
    records: Sequence[CycleRecord],
) -> dict[str, object]:
    ledger, ledger_fingerprint = _read_fingerprinted_json(
        access_ledger, fingerprint_field="ledger_fingerprint"
    )
    expected_body = {
        "schema": "hydraulic_padr_pac_confirmation_access_ledger",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "analysis_protocol_sha256": ANALYSIS_PROTOCOL_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "confirmation_manifest_fingerprint": phase_fingerprint,
        "development_completion_fingerprint": development["completion_fingerprint"],
        "output_directory": str(output_dir.resolve()),
        "cycle_records_fingerprint": _fingerprint(
            [record.as_dict() for record in records]
        ),
        "sensor_access_committed": True,
        "reusable": False,
    }
    observed_body = dict(ledger)
    observed_body.pop("ledger_fingerprint", None)
    ledger_spec = manifest.get("global_access_ledger")
    if (
        not isinstance(ledger_spec, Mapping)
        or ledger_spec.get("workspace_relative_path") != ACCESS_LEDGER_RELATIVE
        or observed_body != expected_body
        or ledger_fingerprint != _fingerprint(expected_body)
    ):
        raise ValueError("workspace-level confirmation access ledger is invalid")
    return {
        "path": str(access_ledger.resolve()),
        "size_bytes": access_ledger.stat().st_size,
        "sha256": _sha256_file(access_ledger),
        "ledger_fingerprint": ledger_fingerprint,
    }


def _verify_confirmation_completion(
    output_dir: Path, *, access_ledger: Path | None = None
) -> dict[str, object] | None:
    completion = _verify_completion(
        output_dir / CONFIRMATION_COMPLETION,
        output_dir,
        expected_artifacts=CONFIRMATION_ARTIFACTS,
        expected_decisions={PASS_DECISION, FAIL_DECISION},
    )
    if completion is None:
        return None
    manifest, phase_fingerprint = _read_fingerprinted_json(
        output_dir / CONFIRMATION_MANIFEST,
        fingerprint_field="protocol_fingerprint",
    )
    _validate_confirmation_manifest_semantics(manifest)
    plan = manifest.get("cycle_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("confirmation manifest has no cycle plan")
    records = _records_from_confirmation_plan(
        plan.get("confirmation_all_valve_classes")
    )
    development = _verify_confirmation_development_chain(output_dir, manifest)
    if access_ledger is None:
        access_record = _read_json(output_dir / CONFIRMATION_ACCESS).get(
            "global_access_ledger"
        )
        if not isinstance(access_record, Mapping):
            raise ValueError("confirmation access record has no global ledger")
        candidate = Path(str(access_record.get("path", "")))
        if candidate.name != Path(ACCESS_LEDGER_RELATIVE).name:
            raise ValueError("confirmation access record has an unsafe ledger identity")
        access_ledger = candidate
    ledger_record = _verify_confirmation_ledger(
        access_ledger,
        output_dir=output_dir,
        manifest=manifest,
        phase_fingerprint=phase_fingerprint,
        development=development,
        records=records,
    )
    indices, _, _, _ = _records_arrays(records)
    expected_access = {
        "schema": "hydraulic_padr_pac_access_intent",
        "schema_version": 1,
        "phase": "single_access_external_confirmation",
        "phase_fingerprint": phase_fingerprint,
        "cycle_indices_zero_based": indices.tolist(),
        "cycle_count": 369,
        "single_access": True,
        "fit_calibration_or_selection_permitted": False,
        "global_access_ledger": ledger_record,
    }
    if _read_json(output_dir / CONFIRMATION_ACCESS) != expected_access:
        raise ValueError("confirmation access intent differs from the global ledger")
    path = _load_confirmation_score_path(
        output_dir / "confirmation_scores.npz", records
    )
    metrics, cycle_rows = _confirmation_metrics(records, path)
    result = _read_json(output_dir / "confirmation_result.json")
    development_result = _read_json(output_dir / "development_result.json")
    certificate = _read_json(output_dir / "calibration_certificate.json")
    development_boundary = _validate_alarm_boundary_record(
        development_result.get("alarm_boundary"),
        expected_raw_pac_threshold=certificate.get("threshold"),
    )
    result_boundary = _validate_alarm_boundary_record(
        result.get("alarm_boundary"),
        expected_raw_pac_threshold=certificate.get("threshold"),
    )
    for key, expected in metrics.items():
        if result.get(key) != expected:
            raise ValueError(f"confirmation result metric changed: {key}")
    expected_governance = {
        "fit_performed": False,
        "calibration_performed": False,
        "selection_performed": False,
        "gate_change_performed": False,
        "same_data_retuning_permitted": False,
    }
    if (
        result.get("schema_version") != 1
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("phase") != "single_access_external_confirmation"
        or result.get("protocol_fingerprint") != phase_fingerprint
        or result.get("method") != METHOD_NAME
        or result.get("threshold") != path.threshold
        or result_boundary != development_boundary
        or path.threshold != result_boundary["effective_alarm_boundary"]
        or result.get("governance") != expected_governance
        or result.get("decision") != completion.get("decision")
    ):
        raise ValueError("confirmation result identity or frozen semantics changed")
    data_access = result.get("data_access")
    if not isinstance(data_access, Mapping):
        raise ValueError("confirmation result has no data-access record")
    _validate_loader_metadata(data_access, indices)
    if (output_dir / "confirmation_cycles.csv").read_bytes().decode("utf-8") != (
        _cycles_csv(cycle_rows)
    ):
        raise ValueError("confirmation cycle CSV disagrees with frozen scores")
    validation = _read_json(output_dir / "cuda_validation_confirmation.json")
    _validate_confirmation_cuda_record(
        validation, phase_fingerprint=phase_fingerprint
    )
    execution = _read_json(output_dir / "gpu_execution_confirmation.json")
    if (
        execution.get("schema_version") != 1
        or execution.get("phase") != "single_access_external_confirmation"
        or execution.get("protocol_fingerprint") != phase_fingerprint
        or execution.get("scoring_call_count") != 1
        or execution.get("cpu_cuda_validation") != validation
    ):
        raise ValueError("confirmation GPU execution record changed")
    return completion


def _assert_no_partial_artifacts(
    output_dir: Path,
    *,
    allowed_existing: set[str],
    phase_artifacts: Sequence[str],
    completion_name: str,
) -> None:
    names = {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    relevant = names & (set(phase_artifacts) | {completion_name})
    unexpected = relevant - allowed_existing
    if unexpected:
        raise RuntimeError(
            "phase output is incomplete; preserve it for audit and use a fresh "
            f"output directory: {sorted(unexpected)}"
        )


def _records_arrays(
    records: Sequence[CycleRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray([record.index for record in records], dtype=np.int64)
    labels = np.asarray([record.valve_label for record in records], dtype=np.int64)
    groups = np.asarray([record.nuisance_group for record in records])
    episodes = np.asarray([record.episode_id for record in records], dtype=np.int64)
    return indices, labels, groups, episodes


def _validate_loader_metadata(
    metadata: Mapping[str, object], expected_indices: np.ndarray
) -> None:
    if metadata.get("parsed_cycle_indices") != expected_indices.tolist():
        raise ValueError("Hydraulic loader parsed a different cycle selection")
    if metadata.get("unrequested_numeric_values_parsed") is not False:
        raise ValueError("Hydraulic loader reports parsing unrequested cycle values")
    if (
        metadata.get("unrequested_rows_decoded") is not False
        or metadata.get("unrequested_rows_byte_scanned_for_boundaries") is not True
    ):
        raise ValueError("Hydraulic loader no longer preserves held-out byte isolation")
    if metadata.get("output_shape") != [len(expected_indices), CYCLE_SECONDS, RAW_CHANNEL_COUNT]:
        raise ValueError("Hydraulic loader output shape metadata changed")


def _run_development(
    args: argparse.Namespace, preflight: Preflight
) -> dict[str, object]:
    completed = _verify_development_completion(args.output_dir)
    if completed is not None:
        print("hydraulic development is already complete", flush=True)
        return completed

    manifest_body = _development_manifest_body(preflight, batch_size=args.batch_size)
    manifest_path = args.output_dir / DEVELOPMENT_MANIFEST
    if not manifest_path.exists() and args.output_dir.exists() and tuple(args.output_dir.iterdir()):
        raise RuntimeError(
            "development requires an empty output directory before its manifest is frozen"
        )
    _assert_no_partial_artifacts(
        args.output_dir,
        allowed_existing={DEVELOPMENT_MANIFEST} if manifest_path.exists() else set(),
        phase_artifacts=DEVELOPMENT_ARTIFACTS,
        completion_name=DEVELOPMENT_COMPLETION,
    )
    phase_fingerprint = _load_or_create_manifest(manifest_path, manifest_body)

    development = preflight.records["development"]
    calibration_healthy = tuple(
        record
        for record in preflight.records["calibration"]
        if record.valve_label == HEALTHY_LABEL
    )
    requested = development + calibration_healthy
    indices, labels, groups, _ = _records_arrays(requested)
    access_body = {
        "schema": "hydraulic_padr_pac_access_intent",
        "schema_version": 1,
        "phase": "development_and_healthy_calibration",
        "phase_fingerprint": phase_fingerprint,
        "cycle_indices_zero_based": indices.tolist(),
        "development_cycle_count": len(development),
        "calibration_healthy_cycle_count": len(calibration_healthy),
        "confirmation_values_accessed": False,
    }
    _atomic_json_new(args.output_dir / DEVELOPMENT_ACCESS, access_body)

    print(
        "loading 720 development cycles and 90 healthy calibration cycles",
        flush=True,
    )
    raw, load_metadata = load_hydraulic_cycles(args.data_dir, indices.tolist())
    _validate_loader_metadata(load_metadata, indices)
    if raw.shape != (810, CYCLE_SECONDS, RAW_CHANNEL_COUNT):
        raise RuntimeError("development/calibration raw cycle shape changed")

    dev_count = len(development)
    dev_raw = raw[:dev_count]
    cal_raw = raw[dev_count:]
    dev_labels = labels[:dev_count]
    dev_groups = groups[:dev_count]
    if not np.all(labels[dev_count:] == HEALTHY_LABEL):
        raise RuntimeError("non-healthy calibration cycles entered development")
    template = fit_phasewise_template(dev_raw[dev_labels == HEALTHY_LABEL])
    dev_features = template.transform(dev_raw)
    calibration_features = template.transform(cal_raw)
    folds = protocol_grouped_fold_ids(dev_groups)
    fold_summary = {
        str(fold): {
            "nuisance_group_count": len(set(dev_groups[folds == fold].tolist())),
            "cycle_count": int(np.count_nonzero(folds == fold)),
            "cycle_count_by_label": {
                str(label): int(np.count_nonzero((folds == fold) & (dev_labels == label)))
                for label in (100, 90, 80, 73)
            },
        }
        for fold in range(3)
    }
    expected_fold = {
        "nuisance_group_count": 6,
        "cycle_count": 240,
        "cycle_count_by_label": {"100": 60, "90": 60, "80": 60, "73": 60},
    }
    if any(summary != expected_fold for summary in fold_summary.values()):
        raise RuntimeError("development grouped folds do not match the frozen 6-group design")

    print("running frozen three-fold ridge grid on CUDA float64", flush=True)
    frozen_model, cv_result = fit_hydraulic_grouped_cv_cuda(
        dev_features,
        dev_labels,
        dev_groups,
        fold_by_cycle=folds,
        ridge_lambdas=RIDGE_LAMBDA_GRID,
        kappa=DEFAULT_KAPPA,
        device=args.device,
        max_iter=OPTIMIZER_MAX_ITER,
        max_eval=OPTIMIZER_MAX_EVAL,
        tolerance_grad=OPTIMIZER_TOLERANCE_GRAD,
        tolerance_change=OPTIMIZER_TOLERANCE_CHANGE,
        history_size=OPTIMIZER_HISTORY_SIZE,
        require_convergence=True,
    )
    calibration_names = _calibration_cycle_names(calibration_healthy)
    raw_calibrated_model, calibration = calibrate_hydraulic_pac(
        frozen_model,
        calibration_features,
        cycle_names=calibration_names,
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
        required_count=PAC_CALIBRATION_COUNT,
    )
    if calibration.status != "CERTIFIED":
        raise RuntimeError("healthy calibration did not issue a PAC certificate")
    alarm_boundary = _alarm_boundary_record(calibration.threshold)
    calibrated_model = raw_calibrated_model.with_threshold(
        float(alarm_boundary["effective_alarm_boundary"])
    )

    template_path = args.output_dir / "phase_template.npz"
    model_path = args.output_dir / "calibrated_model.npz"
    if template_path.exists() or model_path.exists():
        raise FileExistsError("refusing to overwrite a development model artifact")
    save_phasewise_template(template, template_path)
    template_record = _artifact_record(template_path)
    model_metadata = {
        "protocol_id": PROTOCOL_ID,
        "phase_fingerprint": phase_fingerprint,
        "selected_ridge_lambda": cv_result.selected_lambda,
        "template_sha256": template_record["sha256"],
        "prethreshold_model_sha256": calibration.model_sha256,
        "calibration_sha256": calibration.calibration_sha256,
        "threshold_source": "raw PAC threshold from 90 healthy calibration cycles only",
        "raw_pac_threshold": alarm_boundary["raw_pac_threshold"],
        "numerical_alarm_guard": alarm_boundary["numerical_alarm_guard"],
        "effective_alarm_boundary": alarm_boundary["effective_alarm_boundary"],
        "effective_boundary_rule": alarm_boundary["effective_boundary_rule"],
    }
    save_hydraulic_model(calibrated_model, model_path, metadata=model_metadata)
    model_record = _artifact_record(model_path)

    # Score the strict on-disk artifacts, not merely the in-memory objects, so
    # successful completion also proves serialization round-trip integrity.
    persisted_template = load_phasewise_template(template_path)
    persisted_model, persisted_metadata = load_hydraulic_model(model_path)
    if (
        persisted_metadata != model_metadata
        or persisted_model.threshold != calibrated_model.threshold
        or persisted_model.threshold != alarm_boundary["effective_alarm_boundary"]
        or hydraulic_directional_model_sha256(persisted_model.with_threshold(None))
        != calibration.model_sha256
        or not np.array_equal(
            persisted_template.transform(dev_raw[:1]), template.transform(dev_raw[:1])
        )
    ):
        raise RuntimeError("saved Hydraulic template/model failed round-trip validation")

    combined_features = np.concatenate((dev_features, calibration_features), axis=0)
    cpu_path = score_hydraulic_numpy(persisted_model, combined_features)
    gpu_path, gpu_execution = score_hydraulic_cuda(
        persisted_model,
        combined_features,
        device=args.device,
        batch_size=args.batch_size,
    )
    parity = compare_hydraulic_paths(cpu_path, gpu_path, atol=CPU_CUDA_ATOL)
    parity_passed = bool(parity["eligible_exact"] and parity["alarm_exact"])
    validation = {
        "schema_version": 1,
        "phase": "development_and_healthy_calibration",
        "protocol_fingerprint": phase_fingerprint,
        "passed": parity_passed,
        "validated_cycle_count": len(requested),
        "atol": CPU_CUDA_ATOL,
        **parity,
    }
    if not parity_passed:
        raise RuntimeError("development CPU/CUDA alarm decisions differ")

    calibration_dict = calibration.as_dict()
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": "development_complete_method_frozen",
        "decision": DEVELOPMENT_DECISION,
        "protocol_fingerprint": phase_fingerprint,
        "counts": {
            "development": _cycle_plan(development),
            "calibration_healthy": _cycle_plan(calibration_healthy),
        },
        "grouped_folds": fold_summary,
        "cross_validation": cv_result.as_dict(),
        "cross_validation_interpretation": (
            "development-only lambda selection criterion; not an unbiased "
            "generalization-performance estimate"
        ),
        "calibration": calibration_dict,
        "alarm_boundary": alarm_boundary,
        "artifacts": {
            "phase_template.npz": template_record,
            "calibrated_model.npz": model_record,
        },
        "data_access": load_metadata,
        "confirmation_values_accessed": False,
        "method_frozen_for_confirmation": True,
    }
    execution = {
        "schema_version": 1,
        "phase": "development_and_healthy_calibration",
        "protocol_fingerprint": phase_fingerprint,
        "fit": cv_result.final_fit,
        "scoring": gpu_execution,
        "cpu_cuda_validation": validation,
    }
    _atomic_json_new(args.output_dir / "development_result.json", result)
    _atomic_json_new(args.output_dir / "calibration_certificate.json", calibration_dict)
    _atomic_json_new(args.output_dir / "cuda_validation_development.json", validation)
    _atomic_json_new(args.output_dir / "gpu_execution_development.json", execution)
    _write_completion(
        args.output_dir / DEVELOPMENT_COMPLETION,
        args.output_dir,
        phase_fingerprint=phase_fingerprint,
        decision=DEVELOPMENT_DECISION,
        artifact_names=DEVELOPMENT_ARTIFACTS,
    )
    verified = _verify_development_completion(args.output_dir)
    if verified is None:
        raise RuntimeError("development completion disappeared during verification")
    print(
        f"development frozen: lambda={cv_result.selected_lambda:g}, "
        f"raw_pac_threshold={calibration.threshold:.15g}, "
        f"effective_alarm_boundary={persisted_model.threshold:.15g}",
        flush=True,
    )
    return verified


def _finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " and nonnegative" if nonnegative else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _rebuild_calibration_certificate(
    certificate: Mapping[str, object],
    calibration_records: Sequence[CycleRecord],
) -> HydraulicPACCalibration:
    expected_rank = pac_tolerance_rank(
        PAC_CALIBRATION_COUNT,
        epsilon=PAC_EPSILON,
        delta=PAC_DELTA,
    )
    if certificate.get("rank") != expected_rank.as_dict():
        raise ValueError("PAC certificate rank does not match the frozen exact rank")
    expected_top_level = {
        "status": expected_rank.status,
        "sample_count": expected_rank.sample_count,
        "epsilon": expected_rank.epsilon,
        "delta": expected_rank.delta,
        "rank_one_based": expected_rank.rank_one_based,
        "achieved_confidence": expected_rank.achieved_confidence,
    }
    if any(certificate.get(key) != value for key, value in expected_top_level.items()):
        raise ValueError("PAC certificate top-level rank fields changed")
    raw_maxima = certificate.get("run_maxima")
    if not isinstance(raw_maxima, list) or len(raw_maxima) != PAC_CALIBRATION_COUNT:
        raise ValueError("PAC certificate must contain exactly 90 run maxima")
    maxima: list[tuple[str, float]] = []
    for index, item in enumerate(raw_maxima):
        if not isinstance(item, Mapping) or set(item) != {"cycle", "maximum"}:
            raise ValueError(f"PAC run maximum {index} has an invalid schema")
        name = item.get("cycle")
        if not isinstance(name, str) or not name:
            raise ValueError(f"PAC run maximum {index} has an invalid cycle name")
        maximum = _finite_number(
            item.get("maximum"),
            name=f"PAC run maximum {name}",
            nonnegative=True,
        )
        maxima.append((name, maximum))
    names = [name for name, _ in maxima]
    expected_names = set(_calibration_cycle_names(calibration_records))
    if len(set(names)) != PAC_CALIBRATION_COUNT or set(names) != expected_names:
        raise ValueError("PAC certificate run names differ from frozen healthy calibration")

    model_sha = certificate.get("model_sha256")
    scorer_sha = certificate.get("scorer_spec_sha256")
    calibration_sha = certificate.get("calibration_sha256")
    driver = certificate.get("rank_driver_cycle")
    if not all(
        isinstance(value, str)
        for value in (model_sha, scorer_sha, calibration_sha, driver)
    ):
        raise ValueError("PAC certificate hash or driver field is invalid")
    threshold = _finite_number(
        certificate.get("threshold"), name="PAC threshold", nonnegative=True
    )
    expected_calibration_sha = hydraulic_calibration_sha256(
        model_sha256=str(model_sha),
        run_maxima=maxima,
        rank=expected_rank,
    )
    if calibration_sha != expected_calibration_sha:
        raise ValueError("PAC certificate calibration fingerprint is corrupt")
    rebuilt = HydraulicPACCalibration(
        status=expected_rank.status,
        threshold=threshold,
        rank=expected_rank,
        rank_driver_cycle=str(driver),
        run_maxima=tuple(maxima),
        scorer_spec_sha256=str(scorer_sha),
        model_sha256=str(model_sha),
        calibration_sha256=str(calibration_sha),
    )
    if rebuilt.as_dict() != dict(certificate):
        raise ValueError("PAC certificate differs from its reconstructed canonical form")
    return rebuilt


def _validate_development_cuda_record(
    validation: Mapping[str, object], *, phase_fingerprint: str
) -> None:
    if (
        validation.get("schema_version") != 1
        or validation.get("phase") != "development_and_healthy_calibration"
        or validation.get("protocol_fingerprint") != phase_fingerprint
        or validation.get("passed") is not True
        or validation.get("validated_cycle_count") != 810
        or validation.get("atol") != CPU_CUDA_ATOL
        or validation.get("eligible_exact") is not True
        or validation.get("alarm_exact") is not True
    ):
        raise ValueError("development CPU/CUDA validation identity or decision changed")
    for key, value in validation.items():
        if key.endswith("_max_abs_error") and _finite_number(
            value, name=key, nonnegative=True
        ) > CPU_CUDA_ATOL:
            raise ValueError(f"development CUDA parity exceeds tolerance: {key}")


def _selected_lambda_from_cv(
    cv: Mapping[str, object],
    *,
    expected_fold_groups: Mapping[int, tuple[str, ...]],
) -> float:
    means = cv.get("mean_validation_loss_by_lambda")
    folds = cv.get("folds")
    if not isinstance(means, list) or len(means) != len(RIDGE_LAMBDA_GRID):
        raise ValueError("development CV must contain all five ridge candidates")
    parsed: list[tuple[float, float]] = []
    for item in means:
        if not isinstance(item, Mapping) or set(item) != {
            "ridge_lambda",
            "mean_validation_log_loss",
        }:
            raise ValueError("development CV mean-loss record has an invalid schema")
        ridge = _finite_number(item["ridge_lambda"], name="CV ridge lambda")
        loss = _finite_number(item["mean_validation_log_loss"], name="CV mean loss")
        parsed.append((ridge, loss))
    if tuple(ridge for ridge, _ in parsed) != RIDGE_LAMBDA_GRID:
        raise ValueError("development CV ridge grid or ordering changed")
    if not isinstance(folds, list) or len(folds) != 3 * len(RIDGE_LAMBDA_GRID):
        raise ValueError("development CV must contain exactly fifteen fold results")
    pairs: list[tuple[float, int]] = []
    loss_by_pair: dict[tuple[float, int], float] = {}
    for item in folds:
        if not isinstance(item, Mapping):
            raise ValueError("development CV fold result is invalid")
        ridge = _finite_number(item.get("ridge_lambda"), name="fold ridge lambda")
        fold = int(item.get("held_out_fold", -1))
        pair = (ridge, fold)
        pairs.append(pair)
        loss_by_pair[pair] = _finite_number(
            item.get("validation_log_loss"), name="fold validation log loss"
        )
        held_groups = item.get("held_out_nuisance_groups")
        convergence = item.get("convergence")
        if (
            item.get("training_cycle_count") != 480
            or item.get("held_out_cycle_count") != 240
            or not isinstance(held_groups, list)
            or tuple(held_groups) != expected_fold_groups.get(fold)
            or not isinstance(convergence, Mapping)
            or convergence.get("converged") is not True
            or convergence.get("dtype") != "float64"
            or convergence.get("tf32") is not False
        ):
            raise ValueError("development CV fold scope or convergence record changed")
    expected_pairs = [
        (ridge, fold) for ridge in RIDGE_LAMBDA_GRID for fold in range(3)
    ]
    if pairs != expected_pairs:
        raise ValueError("development CV fold ordering or coverage changed")
    for ridge, reported_mean in parsed:
        recomputed_mean = float(
            np.mean([loss_by_pair[(ridge, fold)] for fold in range(3)])
        )
        if reported_mean != recomputed_mean:
            raise ValueError("development CV mean is not the mean of its three folds")
    selected = min(parsed, key=lambda item: (item[1], -item[0]))[0]
    if cv.get("selected_lambda") != selected:
        raise ValueError("development selected lambda does not minimize the frozen criterion")
    final_fit = cv.get("final_fit")
    if (
        not isinstance(final_fit, Mapping)
        or final_fit.get("ridge_lambda") != selected
        or final_fit.get("development_cycle_count") != 720
        or final_fit.get("nuisance_group_count") != 18
        or final_fit.get("dtype") != "float64"
        or final_fit.get("tf32") is not False
        or final_fit.get("converged") is not True
    ):
        raise ValueError("development final-fit record is incomplete or inconsistent")
    return selected


def _authenticate_development_for_confirmation(
    output_dir: Path, preflight: Preflight
) -> tuple[dict[str, object], HydraulicDirectionalModel, object, dict[str, object]]:
    completion = _verify_development_completion(output_dir)
    if completion is None:
        raise RuntimeError("confirmation requires a completed development phase")
    manifest, manifest_fingerprint = _read_fingerprinted_json(
        output_dir / DEVELOPMENT_MANIFEST,
        fingerprint_field="protocol_fingerprint",
    )
    if (
        manifest_fingerprint != completion.get("phase_fingerprint")
        or manifest.get("frozen_inputs") != preflight.frozen_inputs
        or manifest.get("source_sha256") != preflight.source_sha256
    ):
        raise ValueError("current protocols, data, audit, or sources differ from development")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("development manifest execution section is invalid")
    expected_manifest = _development_manifest_body(
        preflight, batch_size=int(execution.get("batch_size", 0))
    )
    observed_body = dict(manifest)
    observed_body.pop("protocol_fingerprint", None)
    if observed_body != expected_manifest:
        raise ValueError("development manifest no longer matches the frozen protocol")

    result = _read_json(output_dir / "development_result.json")
    certificate = _read_json(output_dir / "calibration_certificate.json")
    validation = _read_json(output_dir / "cuda_validation_development.json")
    calibration_records = tuple(
        record
        for record in preflight.records["calibration"]
        if record.valve_label == HEALTHY_LABEL
    )
    rebuilt_certificate = _rebuild_calibration_certificate(
        certificate, calibration_records
    )
    alarm_boundary = _validate_alarm_boundary_record(
        result.get("alarm_boundary"),
        expected_raw_pac_threshold=rebuilt_certificate.threshold,
    )
    expected_counts = {
        "development": _cycle_plan(preflight.records["development"]),
        "calibration_healthy": _cycle_plan(calibration_records),
    }
    cv = result.get("cross_validation")
    if not isinstance(cv, Mapping):
        raise ValueError("development result has no cross-validation record")
    dev_records = preflight.records["development"]
    _, dev_labels, dev_groups, _ = _records_arrays(dev_records)
    dev_folds = protocol_grouped_fold_ids(dev_groups)
    expected_fold_groups = {
        fold: tuple(
            sorted(
                f"string:{group}"
                for group in set(dev_groups[dev_folds == fold].tolist())
            )
        )
        for fold in range(3)
    }
    selected_lambda = _selected_lambda_from_cv(
        cv, expected_fold_groups=expected_fold_groups
    )
    expected_grouped_folds = {
        str(fold): {
            "nuisance_group_count": 6,
            "cycle_count": 240,
            "cycle_count_by_label": {
                str(label): int(
                    np.count_nonzero((dev_folds == fold) & (dev_labels == label))
                )
                for label in (100, 90, 80, 73)
            },
        }
        for fold in range(3)
    }
    _validate_development_cuda_record(
        validation, phase_fingerprint=manifest_fingerprint
    )
    template_path = output_dir / "phase_template.npz"
    model_path = output_dir / "calibrated_model.npz"
    actual_result_artifacts = {
        "phase_template.npz": _artifact_record(template_path),
        "calibrated_model.npz": _artifact_record(model_path),
    }
    completion_artifacts = completion.get("artifacts")
    if not isinstance(completion_artifacts, Mapping):
        raise ValueError("development completion artifact map is invalid")
    completion_model_artifacts = {
        name: completion_artifacts.get(name) for name in actual_result_artifacts
    }
    if (
        result.get("schema_version") != 1
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("phase") != "development_complete_method_frozen"
        or result.get("decision") != DEVELOPMENT_DECISION
        or result.get("protocol_fingerprint") != manifest_fingerprint
        or result.get("counts") != expected_counts
        or result.get("grouped_folds") != expected_grouped_folds
        or result.get("calibration") != certificate
        or result.get("artifacts") != actual_result_artifacts
        or completion_model_artifacts != actual_result_artifacts
        or result.get("confirmation_values_accessed") is not False
        or result.get("method_frozen_for_confirmation") is not True
    ):
        raise ValueError("development result semantics or artifact bindings changed")
    data_access = result.get("data_access")
    if not isinstance(data_access, Mapping):
        raise ValueError("development result has no data-access record")
    requested_records = preflight.records["development"] + calibration_records
    requested_indices = np.asarray(
        [record.index for record in requested_records], dtype=np.int64
    )
    _validate_loader_metadata(data_access, requested_indices)

    template = load_phasewise_template(template_path)
    model, metadata = load_hydraulic_model(model_path)
    if not isinstance(metadata, Mapping):
        raise ValueError("calibrated model metadata is invalid")
    prethreshold_sha = hydraulic_directional_model_sha256(model.with_threshold(None))
    if (
        metadata.get("protocol_id") != PROTOCOL_ID
        or metadata.get("phase_fingerprint") != manifest_fingerprint
        or metadata.get("selected_ridge_lambda") != selected_lambda
        or metadata.get("template_sha256")
        != actual_result_artifacts["phase_template.npz"]["sha256"]
        or metadata.get("prethreshold_model_sha256") != prethreshold_sha
        or rebuilt_certificate.model_sha256 != prethreshold_sha
        or rebuilt_certificate.calibration_sha256
        != metadata.get("calibration_sha256")
        or metadata.get("raw_pac_threshold")
        != alarm_boundary["raw_pac_threshold"]
        or metadata.get("numerical_alarm_guard")
        != alarm_boundary["numerical_alarm_guard"]
        or metadata.get("effective_alarm_boundary")
        != alarm_boundary["effective_alarm_boundary"]
        or metadata.get("effective_boundary_rule")
        != alarm_boundary["effective_boundary_rule"]
        or model.threshold != alarm_boundary["effective_alarm_boundary"]
        or not np.array_equal(
            model.kappa,
            np.full(len(POSITIVE_LABELS), DEFAULT_KAPPA, dtype=np.float64),
        )
    ):
        raise ValueError("frozen development model/certificate binding is invalid")
    return completion, model, template, alarm_boundary


def _binomial_cdf_leq(successes: int, trials: int, probability: float) -> float:
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    q = 1.0 - probability
    return math.fsum(
        math.comb(trials, k) * probability**k * q ** (trials - k)
        for k in range(successes + 1)
    )


def clopper_pearson_upper(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided Clopper-Pearson binomial upper confidence bound."""

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise TypeError("successes and trials must be integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials >= 1")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(160):
        middle = (low + high) / 2.0
        if _binomial_cdf_leq(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return min(1.0, math.nextafter(high, math.inf))


def _first_alarm_second(alarm: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(alarm, dtype=bool))
    return None if indices.size == 0 else int(indices[0]) + 1


def _confirmation_metrics(
    records: Sequence[CycleRecord], path: object
) -> tuple[dict[str, object], list[dict[str, object]]]:
    alarms = np.asarray(path.alarm, dtype=bool)
    component = np.asarray(path.component_persistence, dtype=np.float64)
    if alarms.shape != (len(records), CYCLE_SECONDS):
        raise ValueError("confirmation alarm array shape changed")
    if component.shape != (len(records), CYCLE_SECONDS, len(POSITIVE_LABELS)):
        raise ValueError("confirmation component array shape changed")

    cycle_rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        first = _first_alarm_second(alarms[index])
        attributed: int | None = None
        maximum = float(np.max(path.envelope[index]))
        if first is not None:
            attributed = POSITIVE_LABELS[
                int(np.argmax(component[index, first - 1, :]))
            ]
        cycle_rows.append(
            {
                **record.as_dict(),
                "alarm": first is not None,
                "first_alarm_second_one_based": first,
                "attributed_severity_head": attributed,
                "cycle_maximum_envelope": maximum,
            }
        )

    healthy_rows = [row for row in cycle_rows if row["valve_condition_percent"] == 100]
    fault_rows = [row for row in cycle_rows if row["valve_condition_percent"] != 100]
    healthy_alarms = sum(bool(row["alarm"]) for row in healthy_rows)
    healthy_count = len(healthy_rows)
    point_fap = healthy_alarms / healthy_count
    upper = clopper_pearson_upper(healthy_alarms, healthy_count, confidence=0.95)

    by_severity: dict[str, object] = {}
    for severity in POSITIVE_LABELS:
        rows = [row for row in fault_rows if row["valve_condition_percent"] == severity]
        detected = sum(bool(row["alarm"]) for row in rows)
        seconds = [
            int(row["first_alarm_second_one_based"])
            for row in rows
            if row["first_alarm_second_one_based"] is not None
        ]
        matches = sum(
            bool(row["alarm"]) and row["attributed_severity_head"] == severity
            for row in rows
        )
        attribution_counts = {
            str(head): sum(
                bool(row["alarm"]) and row["attributed_severity_head"] == head
                for row in rows
            )
            for head in POSITIVE_LABELS
        }
        by_severity[str(severity)] = {
            "cycle_count": len(rows),
            "detected_cycles": detected,
            "recall": detected / len(rows),
            "conditional_median_first_alarm_second": (
                None if not seconds else float(np.median(seconds))
            ),
            "attribution_matches": matches,
            "attribution_counts_by_head": attribution_counts,
            "attribution_accuracy_among_detected": (
                None if detected == 0 else matches / detected
            ),
        }
    total_detected = sum(bool(row["alarm"]) for row in fault_rows)
    overall_recall = total_detected / len(fault_rows)

    episode_summary: dict[str, object] = {}
    for label in (100, *POSITIVE_LABELS):
        label_rows = [row for row in cycle_rows if row["valve_condition_percent"] == label]
        episode_ids = sorted({int(row["episode_id"]) for row in label_rows})
        alarm_episodes = sum(
            any(
                bool(row["alarm"])
                for row in label_rows
                if int(row["episode_id"]) == episode_id
            )
            for episode_id in episode_ids
        )
        episode_summary[str(label)] = {
            "episode_count": len(episode_ids),
            "episodes_with_any_alarm": alarm_episodes,
            "incidence": alarm_episodes / len(episode_ids),
        }

    group_summary: dict[str, object] = {}
    for group in dict.fromkeys(record.nuisance_group for record in records):
        group_rows = [row for row in cycle_rows if row["nuisance_group"] == group]
        cells: dict[str, object] = {}
        for label in (100, *POSITIVE_LABELS):
            count = sum(row["valve_condition_percent"] == label for row in group_rows)
            detected = sum(
                row["valve_condition_percent"] == label and bool(row["alarm"])
                for row in group_rows
            )
            cells[str(label)] = {
                "cycle_count": count,
                "cycles_with_alarm": detected,
                "alarm_rate": detected / count,
                "interpretation": "false_alarm_probability" if label == 100 else "recall",
            }
        group_summary[group] = cells

    gates = {
        "healthy_alarm_count": {
            "observed": healthy_alarms,
            "maximum": HEALTHY_MAXIMUM_ALARMS,
            "passed": healthy_alarms <= HEALTHY_MAXIMUM_ALARMS,
        },
        "healthy_point_fap": {
            "observed": point_fap,
            "maximum": HEALTHY_POINT_FAP_MAXIMUM,
            "passed": point_fap <= HEALTHY_POINT_FAP_MAXIMUM,
        },
        "healthy_exact_one_sided_95pct_ucb": {
            "observed": upper,
            "maximum": HEALTHY_UCB_MAXIMUM,
            "passed": upper <= HEALTHY_UCB_MAXIMUM,
        },
        "minimum_recall_each_severity": {
            "observed": min(float(item["recall"]) for item in by_severity.values()),
            "minimum": MINIMUM_RECALL_EACH_SEVERITY,
            "passed": all(
                float(item["recall"]) >= MINIMUM_RECALL_EACH_SEVERITY
                for item in by_severity.values()
            ),
        },
        "overall_fault_recall": {
            "observed": overall_recall,
            "minimum": MINIMUM_OVERALL_FAULT_RECALL,
            "passed": overall_recall >= MINIMUM_OVERALL_FAULT_RECALL,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    result = {
        "healthy": {
            "cycle_count": healthy_count,
            "cycles_with_alarm": healthy_alarms,
            "point_false_alarm_probability": point_fap,
            "exact_one_sided_95pct_ucb": upper,
        },
        "faults": {
            "cycle_count": len(fault_rows),
            "detected_cycles": total_detected,
            "overall_recall": overall_recall,
            "by_severity": by_severity,
        },
        "episodes_by_valve_label": episode_summary,
        "by_nuisance_group": group_summary,
        "gates": gates,
        "passed": passed,
        "decision": PASS_DECISION if passed else FAIL_DECISION,
    }
    return result, cycle_rows


def _cycles_csv(rows: Sequence[Mapping[str, object]]) -> str:
    columns = (
        "cycle_index_zero_based",
        "cycle_number_one_based",
        "valve_condition_percent",
        "nuisance_group",
        "episode_id",
        "alarm",
        "first_alarm_second_one_based",
        "attributed_severity_head",
        "cycle_maximum_envelope",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in columns})
    return stream.getvalue()


def _confirmation_report(result: Mapping[str, object], execution: Mapping[str, object]) -> str:
    healthy = result["healthy"]
    faults = result["faults"]
    boundary = result["alarm_boundary"]
    lines = [
        "# PAC-PADR-CUSUM external confirmation",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "The frozen method was applied once to the nine held-out Hydraulic nuisance ",
        "groups. No fit, calibration, threshold change, or gate change occurred in ",
        "this phase.",
        f"Raw PAC threshold: {boundary['raw_pac_threshold']:.17g}.",
        f"Fixed numerical guard: {boundary['numerical_alarm_guard']:.17g}.",
        f"Effective strict alarm boundary: {boundary['effective_alarm_boundary']:.17g}.",
        "",
        "| Gate | Observed | Requirement | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for gate_name, gate in result["gates"].items():
        bound_name = "maximum" if "maximum" in gate else "minimum"
        operator = "<=" if bound_name == "maximum" else ">="
        lines.append(
            f"| {gate_name} | {gate.get('observed')} | "
            f"{operator} {gate[bound_name]} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Healthy alarms: {healthy['cycles_with_alarm']}/{healthy['cycle_count']}; "
            f"one-sided 95% exact UCB={healthy['exact_one_sided_95pct_ucb']:.6f}.",
            "",
            "| Valve severity | Detected cycles | Recall | Median first-alarm second |",
            "|---:|---:|---:|---:|",
        ]
    )
    for severity in POSITIVE_LABELS:
        item = faults["by_severity"][str(severity)]
        median = item["conditional_median_first_alarm_second"]
        lines.append(
            f"| {severity} | {item['detected_cycles']}/{item['cycle_count']} | "
            f"{item['recall']:.6f} | {'NA' if median is None else f'{median:.3f}'} |"
        )
    validation = execution["cpu_cuda_validation"]
    scoring = execution["scoring"]
    lines.extend(
        [
            "",
            "## Execution",
            "",
            f"- GPU: {scoring.get('device_name', scoring.get('device', 'cuda'))}.",
            "- CUDA scoring used float64 with TF32 disabled.",
            f"- CPU/CUDA alarm decisions exact: {validation['alarm_exact']}.",
            "- Development-only CV loss selected ridge lambda and is not an ",
            "  unbiased estimate of external generalization performance.",
            "- Failed confirmation, if any, is terminal for this frozen analysis; ",
            "  the held-out cycles may not be used for same-data retuning.",
            "",
        ]
    )
    return "\n".join(lines)


def _claim_confirmation_ledger(
    path: Path,
    *,
    output_dir: Path,
    phase_fingerprint: str,
    development_completion: Mapping[str, object],
    records: Sequence[CycleRecord],
) -> dict[str, object]:
    if path.exists():
        raise RuntimeError(
            "the workspace-level Hydraulic confirmation access token already exists; "
            "do not open the one-shot split again or use another output directory"
        )
    body = {
        "schema": "hydraulic_padr_pac_confirmation_access_ledger",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "analysis_protocol_sha256": ANALYSIS_PROTOCOL_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "confirmation_manifest_fingerprint": phase_fingerprint,
        "development_completion_fingerprint": development_completion[
            "completion_fingerprint"
        ],
        "output_directory": str(output_dir.resolve()),
        "cycle_records_fingerprint": _fingerprint(
            [record.as_dict() for record in records]
        ),
        "sensor_access_committed": True,
        "reusable": False,
    }
    ledger_fingerprint = _fingerprint(body)
    _atomic_json_new(path, {**body, "ledger_fingerprint": ledger_fingerprint})
    verified, stored = _read_fingerprinted_json(
        path, fingerprint_field="ledger_fingerprint"
    )
    if stored != ledger_fingerprint or verified.get("sensor_access_committed") is not True:
        raise RuntimeError("confirmation access ledger failed post-write verification")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "ledger_fingerprint": ledger_fingerprint,
    }


def _run_confirmation(
    args: argparse.Namespace, preflight: Preflight
) -> dict[str, object]:
    completed = _verify_confirmation_completion(
        args.output_dir, access_ledger=args.access_ledger
    )
    if completed is not None:
        print("hydraulic external confirmation is already complete", flush=True)
        return completed
    (
        development_completion,
        model,
        template,
        alarm_boundary,
    ) = _authenticate_development_for_confirmation(args.output_dir, preflight)
    development_completion_path = args.output_dir / DEVELOPMENT_COMPLETION
    manifest_body = _confirmation_manifest_body(
        preflight,
        development_completion,
        development_completion_path,
        batch_size=args.batch_size,
        access_ledger=args.access_ledger,
    )
    manifest_path = args.output_dir / CONFIRMATION_MANIFEST
    _assert_no_partial_artifacts(
        args.output_dir,
        allowed_existing={CONFIRMATION_MANIFEST} if manifest_path.exists() else set(),
        phase_artifacts=CONFIRMATION_ARTIFACTS,
        completion_name=CONFIRMATION_COMPLETION,
    )
    phase_fingerprint = _load_or_create_manifest(manifest_path, manifest_body)

    records = preflight.records["confirmation"]
    indices, labels, groups, episodes = _records_arrays(records)
    ledger_record = _claim_confirmation_ledger(
        args.access_ledger,
        output_dir=args.output_dir,
        phase_fingerprint=phase_fingerprint,
        development_completion=development_completion,
        records=records,
    )
    access_body = {
        "schema": "hydraulic_padr_pac_access_intent",
        "schema_version": 1,
        "phase": "single_access_external_confirmation",
        "phase_fingerprint": phase_fingerprint,
        "cycle_indices_zero_based": indices.tolist(),
        "cycle_count": len(records),
        "single_access": True,
        "fit_calibration_or_selection_permitted": False,
        "global_access_ledger": ledger_record,
    }
    _atomic_json_new(args.output_dir / CONFIRMATION_ACCESS, access_body)

    print("loading the 369 held-out confirmation cycles exactly once", flush=True)
    raw, load_metadata = load_hydraulic_cycles(args.data_dir, indices.tolist())
    _validate_loader_metadata(load_metadata, indices)
    if raw.shape != (369, CYCLE_SECONDS, RAW_CHANNEL_COUNT):
        raise RuntimeError("confirmation raw cycle shape changed")
    features = template.transform(raw)
    print("scoring all held-out cycles in one CUDA call", flush=True)
    gpu_path, gpu_execution = score_hydraulic_cuda(
        model,
        features,
        device=args.device,
        batch_size=args.batch_size,
    )
    cpu_path = score_hydraulic_numpy(model, features)
    parity = compare_hydraulic_paths(cpu_path, gpu_path, atol=CPU_CUDA_ATOL)
    parity_passed = bool(parity["eligible_exact"] and parity["alarm_exact"])
    validation = {
        "schema_version": 1,
        "phase": "single_access_external_confirmation",
        "protocol_fingerprint": phase_fingerprint,
        "passed": parity_passed,
        "validated_cycle_count": len(records),
        "atol": CPU_CUDA_ATOL,
        **parity,
    }
    if not parity_passed:
        raise RuntimeError("confirmation CPU/CUDA alarm decisions differ")

    metrics, cycle_rows = _confirmation_metrics(records, gpu_path)
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "phase": "single_access_external_confirmation",
        "protocol_fingerprint": phase_fingerprint,
        "method": METHOD_NAME,
        "threshold": model.threshold,
        "alarm_boundary": alarm_boundary,
        **metrics,
        "data_access": load_metadata,
        "governance": {
            "fit_performed": False,
            "calibration_performed": False,
            "selection_performed": False,
            "gate_change_performed": False,
            "same_data_retuning_permitted": False,
        },
    }
    execution = {
        "schema_version": 1,
        "phase": "single_access_external_confirmation",
        "protocol_fingerprint": phase_fingerprint,
        "scoring_call_count": 1,
        "scoring": gpu_execution,
        "cpu_cuda_validation": validation,
    }
    score_arrays = {
        "cycle_indices_zero_based": indices,
        "valve_labels": labels,
        "nuisance_groups": groups,
        "episode_ids": episodes,
        "threshold": np.asarray([model.threshold], dtype=np.float64),
        "raw_pac_threshold": np.asarray(
            [alarm_boundary["raw_pac_threshold"]], dtype=np.float64
        ),
        "numerical_alarm_guard": np.asarray(
            [alarm_boundary["numerical_alarm_guard"]], dtype=np.float64
        ),
        "effective_alarm_boundary": np.asarray(
            [alarm_boundary["effective_alarm_boundary"]], dtype=np.float64
        ),
        **gpu_path.as_arrays(),
    }
    _atomic_json_new(args.output_dir / "confirmation_result.json", result)
    _atomic_text_new(args.output_dir / "confirmation_cycles.csv", _cycles_csv(cycle_rows))
    _atomic_npz_new(args.output_dir / "confirmation_scores.npz", score_arrays)
    _atomic_json_new(args.output_dir / "cuda_validation_confirmation.json", validation)
    _atomic_json_new(args.output_dir / "gpu_execution_confirmation.json", execution)
    _atomic_text_new(args.output_dir / "REPORT.md", _confirmation_report(result, execution))
    _write_completion(
        args.output_dir / CONFIRMATION_COMPLETION,
        args.output_dir,
        phase_fingerprint=phase_fingerprint,
        decision=str(result["decision"]),
        artifact_names=CONFIRMATION_ARTIFACTS,
    )
    verified = _verify_confirmation_completion(
        args.output_dir, access_ledger=args.access_ledger
    )
    if verified is None:
        raise RuntimeError("confirmation completion disappeared during verification")
    print(f"external confirmation decision: {result['decision']}", flush=True)
    return verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen UCI Hydraulic PAC-PADR-CUSUM CUDA experiment"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--phase", choices=("development", "confirmation", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--analysis-protocol", type=Path)
    parser.add_argument("--analysis-protocol-sidecar", type=Path)
    parser.add_argument("--parent-analysis-protocol", type=Path)
    parser.add_argument("--parent-analysis-protocol-sidecar", type=Path)
    parser.add_argument("--aborted-parent-record", type=Path)
    parser.add_argument("--base-analysis-protocol", type=Path)
    parser.add_argument("--base-analysis-protocol-sidecar", type=Path)
    parser.add_argument("--aborted-v1-record", type=Path)
    parser.add_argument("--acquisition-protocol", type=Path)
    parser.add_argument("--acquisition-protocol-sidecar", type=Path)
    parser.add_argument("--file-manifest", type=Path)
    parser.add_argument("--file-manifest-sidecar", type=Path)
    parser.add_argument("--label-audit-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    return parser


def _resolve_paths(args: argparse.Namespace, project_root: Path) -> None:
    defaults = {
        "analysis_protocol": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_3_analysis.json",
        "analysis_protocol_sidecar": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_3_analysis.sha256",
        "parent_analysis_protocol": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_2_analysis.json",
        "parent_analysis_protocol_sidecar": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_2_analysis.sha256",
        "aborted_parent_record": project_root
        / "outputs"
        / "hydraulic_padr_pac_gpu_v1_2"
        / "ABORTED.json",
        "base_analysis_protocol": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_analysis.json",
        "base_analysis_protocol_sidecar": project_root
        / "protocols"
        / "hydraulic_padr_pac_v1_analysis.sha256",
        "aborted_v1_record": project_root
        / "outputs"
        / "hydraulic_padr_pac_gpu_v1"
        / "ABORTED.json",
        "acquisition_protocol": project_root
        / "protocols"
        / "hydraulic_ws_pac_v1_acquisition.json",
        "acquisition_protocol_sidecar": project_root
        / "protocols"
        / "hydraulic_ws_pac_v1_acquisition.sha256",
        "file_manifest": project_root / "protocols" / "hydraulic_uci_447_files.json",
        "file_manifest_sidecar": project_root
        / "protocols"
        / "hydraulic_uci_447_files.sha256",
        "label_audit_dir": project_root / "outputs" / "hydraulic_ws_pac_v1_label_audit",
        "archive": project_root / "data" / "hydraulic_uci_447.zip",
    }
    for name, default in defaults.items():
        value = getattr(args, name)
        setattr(args, name, (default if value is None else value).resolve())
    args.data_dir = args.data_dir.resolve()
    args.output_dir = (
        project_root / "outputs" / "hydraulic_padr_pac_gpu_v1_3"
        if args.output_dir is None
        else args.output_dir
    ).resolve()
    args.access_ledger = (
        project_root
        / "outputs"
        / "hydraulic_padr_pac_v1_3_confirmation_access_ledger.json"
    ).resolve()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if (
        args.output_dir == args.data_dir
        or args.output_dir in args.data_dir.parents
        or args.data_dir in args.output_dir.parents
    ):
        raise ValueError("output directory must not overlap the Hydraulic data directory")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    _resolve_paths(args, project_root)

    # A completed final phase is immutable.  Verify only its own result bundle
    # and return before hashing or opening the data, model, or source tree.
    if args.phase in {"confirmation", "all"} and (
        args.output_dir / CONFIRMATION_COMPLETION
    ).exists():
        completed = _verify_confirmation_completion(
            args.output_dir, access_ledger=args.access_ledger
        )
        if completed is None:
            raise RuntimeError("confirmation completion disappeared")
        print("hydraulic external confirmation is already complete", flush=True)
        return 0
    if args.phase == "development" and (args.output_dir / DEVELOPMENT_COMPLETION).exists():
        completed = _verify_development_completion(args.output_dir)
        if completed is None:
            raise RuntimeError("development completion disappeared")
        print("hydraulic development is already complete", flush=True)
        return 0
    if args.phase == "confirmation" and not (
        args.output_dir / DEVELOPMENT_COMPLETION
    ).exists():
        raise RuntimeError("confirmation requires a completed development phase")

    preflight = _preflight(args, project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in {"development", "all"}:
        _run_development(args, preflight)
    if args.phase in {"confirmation", "all"}:
        _run_confirmation(args, preflight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
