"""Pickle-free evidence persistence and semantic replay for CARED-BC v1.1.

This module is deliberately numerical and side-effect-light.  It never opens
MetroPT source data, trains a model, or chooses a threshold.  Its only write is
the create-new evidence NPZ.  A PASS-capable caller must reload that NPZ with
``allow_pickle=False`` and use :func:`replay_portability_evidence`; the replay
recomputes every scorer stage and calls the concrete trusted certifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .model import AlignedResidualBatch, CaredBCScorer, CaredScorePath
from .portability import (
    CertifiedDevelopmentPortabilityAudit,
    PortabilityPath,
    certify_development_portability,
)


SPLITS = ("dev_healthy", "dev_fault_f1")
DEVICES = ("cpu", "cuda")
F1_ORIGIN = np.datetime64("2020-04-18T00:00:00", "ns")
FORECAST_HEAD_COUNT = 3
ANALOG_COUNT = 7
PREDICTION_RESIDUAL_COUNT = 21
PHYSICAL_FEATURE_COUNT = 10
MONITORING_RESIDUAL_COUNT = 31

PATH_FLOAT_FIELDS = (
    "prediction",
    "residual",
    "physical_features",
    "energy",
    "directional_glr",
    "standardized_evidence",
    "cusum",
    "envelope",
)
PATH_INTEGER_FIELDS = ("timestamps_ns", "indices", "state_codes", "source_window_rows")
PATH_BOOLEAN_FIELDS = ("reset_before", "score_reset", "alarm")
PATH_FIELDS = PATH_FLOAT_FIELDS + PATH_INTEGER_FIELDS + PATH_BOOLEAN_FIELDS
SCALAR_FIELDS = ("h_raw", "global_envelope_delta", "h_eff")
SCORE_FIELDS = (
    "energy",
    "directional_glr",
    "standardized_evidence",
    "cusum",
    "envelope",
)
V1_PROJECTION_PATH_FIELDS = (
    "prediction",
    "residual",
    "timestamps_ns",
    "indices",
    "state_codes",
    "reset_before",
    "envelope",
    "alarm",
)
V1_PROJECTION_KEYS = tuple(
    sorted(
        [
            f"{split}__{device}__{field}"
            for split in SPLITS
            for device in DEVICES
            for field in V1_PROJECTION_PATH_FIELDS
        ]
        + list(SCALAR_FIELDS)
    )
)
EVIDENCE_KEYS = tuple(
    sorted(
        [
            f"{split}__{device}__{field}"
            for split in SPLITS
            for device in DEVICES
            for field in PATH_FIELDS
        ]
        + list(SCALAR_FIELDS)
    )
)


class EvidenceReplayError(ValueError):
    """Raised when persisted portability evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ReplayedPortabilityEvidence:
    paths: Mapping[str, Mapping[str, PortabilityPath]]
    aligned: Mapping[str, Mapping[str, AlignedResidualBatch]]
    scores: Mapping[str, Mapping[str, CaredScorePath]]
    certificate: CertifiedDevelopmentPortabilityAudit
    exact_backend_checks: Mapping[str, Any]
    alarm_array_hashes: Mapping[str, Any]
    full_sequence_max_abs_differences: Mapping[str, Any]
    semantic_manifest: Mapping[str, Any]
    semantic_sha256: str
    v1_projection_semantic_sha256: str


def _canonical_fingerprint(value: object) -> str:
    try:
        payload = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceReplayError("semantic record is not strict canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any) -> str:
    """Hash dtype, compact C-order shape, and bytes exactly as the v1 runner."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def semantic_array_manifest(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {
            "dtype": np.asarray(payload[key]).dtype.str,
            "shape": list(np.asarray(payload[key]).shape),
            "sha256": array_sha256(payload[key]),
        }
        for key in sorted(payload)
    }


def semantic_array_fingerprint(payload: Mapping[str, np.ndarray]) -> str:
    return _canonical_fingerprint(semantic_array_manifest(payload))


def v1_projection_semantic_fingerprint(payload: Mapping[str, np.ndarray]) -> str:
    missing = set(V1_PROJECTION_KEYS) - set(payload)
    if missing:
        raise EvidenceReplayError(
            f"v1 projection is missing arrays: {sorted(missing)!r}"
        )
    projection = {key: np.asarray(payload[key]) for key in V1_PROJECTION_KEYS}
    return semantic_array_fingerprint(projection)


def portability_path(
    prediction: Any, aligned: AlignedResidualBatch, score: CaredScorePath
) -> PortabilityPath:
    if not isinstance(aligned, AlignedResidualBatch):
        raise EvidenceReplayError("aligned evidence must be AlignedResidualBatch")
    if not isinstance(score, CaredScorePath):
        raise EvidenceReplayError("score evidence must be CaredScorePath")
    return PortabilityPath(
        prediction=np.asarray(prediction),
        residual=np.asarray(aligned.residuals),
        timestamps=np.asarray(aligned.score_timestamps),
        indices=np.asarray(aligned.score_indices),
        state_codes=np.asarray(aligned.state_codes),
        reset_before=np.asarray(aligned.reset_before),
        envelope=np.asarray(score.envelope),
    )


def exact_backend_checks(
    aligned: Mapping[str, Mapping[str, AlignedResidualBatch]],
) -> dict[str, Any]:
    fields = (
        "score_timestamps",
        "score_indices",
        "state_codes",
        "reset_before",
        "source_window_rows",
        "physical_features",
    )
    result: dict[str, Any] = {}
    for split in SPLITS:
        devices = aligned.get(split)
        if not isinstance(devices, Mapping) or set(devices) != set(DEVICES):
            raise EvidenceReplayError(f"{split} aligned device set changed")
        checks = {
            f"{field}_exact": bool(
                np.array_equal(
                    np.asarray(getattr(devices["cpu"], field)),
                    np.asarray(getattr(devices["cuda"], field)),
                )
            )
            for field in fields
        }
        result[split] = {"checks": checks, "passed": bool(all(checks.values()))}
    result["passed"] = bool(all(result[split]["passed"] for split in SPLITS))
    return result


def _max_abs(left: Any, right: Any, *, name: str) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size == 0:
        raise EvidenceReplayError(f"{name} shape mismatch or empty sequence")
    difference = np.abs(left_array - right_array)
    if not np.isfinite(difference).all():
        raise EvidenceReplayError(f"{name} difference is non-finite")
    return float(np.max(difference))


def full_sequence_max_abs_differences(
    paths: Mapping[str, Mapping[str, PortabilityPath]],
    scores: Mapping[str, Mapping[str, CaredScorePath]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        cpu_path = paths[split]["cpu"]
        cuda_path = paths[split]["cuda"]
        cpu_score = scores[split]["cpu"]
        cuda_score = scores[split]["cuda"]
        result[split] = {
            "prediction": _max_abs(
                cpu_path.prediction, cuda_path.prediction, name=f"{split}.prediction"
            ),
            "residual": _max_abs(
                cpu_path.residual, cuda_path.residual, name=f"{split}.residual"
            ),
            **{
                field: _max_abs(
                    getattr(cpu_score, field),
                    getattr(cuda_score, field),
                    name=f"{split}.{field}",
                )
                for field in SCORE_FIELDS
            },
        }
    return result


def alarm_evidence(
    paths: Mapping[str, Mapping[str, PortabilityPath]], h_eff: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    threshold = float(h_eff)
    if not np.isfinite(threshold):
        raise EvidenceReplayError("h_eff must be finite")
    records: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        records[split] = {}
        for device in DEVICES:
            alarm = np.asarray(paths[split][device].envelope > threshold, dtype=np.bool_)
            key = f"{split}__{device}__alarm"
            arrays[key] = alarm
            records[split][device] = {
                "count": int(np.count_nonzero(alarm)),
                "shape": list(alarm.shape),
                "dtype": str(alarm.dtype),
                "sha256": array_sha256(alarm),
            }
    return records, arrays


def build_evidence_payload(
    *,
    paths: Mapping[str, Mapping[str, PortabilityPath]],
    aligned: Mapping[str, Mapping[str, AlignedResidualBatch]],
    scores: Mapping[str, Mapping[str, CaredScorePath]],
    alarms: Mapping[str, np.ndarray],
    core_audit: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    threshold = core_audit.get("threshold")
    if not isinstance(threshold, Mapping):
        raise EvidenceReplayError("core audit has no threshold mapping")
    payload: dict[str, np.ndarray] = {}
    for split in SPLITS:
        for device in DEVICES:
            path = paths[split][device]
            batch = aligned[split][device]
            score = scores[split][device]
            prefix = f"{split}__{device}"
            payload[f"{prefix}__prediction"] = np.asarray(path.prediction)
            payload[f"{prefix}__residual"] = np.asarray(path.residual)
            payload[f"{prefix}__timestamps_ns"] = np.asarray(
                path.timestamps, dtype="datetime64[ns]"
            ).astype(np.int64)
            payload[f"{prefix}__indices"] = np.asarray(path.indices)
            payload[f"{prefix}__state_codes"] = np.asarray(path.state_codes)
            payload[f"{prefix}__reset_before"] = np.asarray(path.reset_before)
            payload[f"{prefix}__source_window_rows"] = np.asarray(
                batch.source_window_rows
            )
            payload[f"{prefix}__physical_features"] = np.asarray(
                batch.physical_features
            )
            for field in SCORE_FIELDS:
                payload[f"{prefix}__{field}"] = np.asarray(getattr(score, field))
            payload[f"{prefix}__score_reset"] = np.asarray(score.reset)
            alarm_key = f"{prefix}__alarm"
            if alarm_key not in alarms:
                raise EvidenceReplayError(f"missing effective alarm array: {alarm_key}")
            payload[alarm_key] = np.asarray(alarms[alarm_key])
    payload["h_raw"] = np.asarray(float(threshold["h_raw"]), dtype=np.float64)
    payload["global_envelope_delta"] = np.asarray(
        float(threshold["global_envelope_delta"]), dtype=np.float64
    )
    payload["h_eff"] = np.asarray(float(threshold["h_eff"]), dtype=np.float64)
    if set(payload) != set(EVIDENCE_KEYS):
        raise EvidenceReplayError("constructed evidence key set changed")
    return payload


def save_evidence_npz(path: str | os.PathLike[str], payload: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target}")
    if set(payload) != set(EVIDENCE_KEYS):
        raise EvidenceReplayError("evidence key set is not the frozen v1.1 schema")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            np.savez_compressed(
                stream, **{key: np.asarray(payload[key]) for key in sorted(payload)}
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        Path(temporary).unlink()
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
    return target


def _expected_shape(field: str, windows: int, scores: int) -> tuple[int, ...]:
    return {
        "prediction": (windows, FORECAST_HEAD_COUNT, ANALOG_COUNT),
        "residual": (scores, MONITORING_RESIDUAL_COUNT),
        "timestamps_ns": (scores,),
        "indices": (scores,),
        "state_codes": (scores,),
        "reset_before": (scores,),
        "source_window_rows": (scores, FORECAST_HEAD_COUNT),
        "physical_features": (scores, PHYSICAL_FEATURE_COUNT),
        "energy": (scores,),
        "directional_glr": (scores,),
        "standardized_evidence": (scores, 2),
        "cusum": (scores, 2),
        "envelope": (scores,),
        "score_reset": (scores,),
        "alarm": (scores,),
    }[field]


def load_evidence_npz(
    path: str | os.PathLike[str],
    *,
    expected_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, np.ndarray]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise EvidenceReplayError("evidence must be a regular file")
    result: dict[str, np.ndarray] = {}
    try:
        with np.load(target, allow_pickle=False) as archive:
            if set(archive.files) != set(EVIDENCE_KEYS):
                raise EvidenceReplayError("persisted evidence key set changed")
            for key in EVIDENCE_KEYS:
                result[key] = np.array(archive[key], copy=True)
    except EvidenceReplayError:
        raise
    except Exception as exc:
        raise EvidenceReplayError("failed to load pickle-free evidence") from exc
    for scalar in SCALAR_FIELDS:
        value = result[scalar]
        if value.shape != () or value.dtype != np.dtype("float64") or not np.isfinite(value):
            raise EvidenceReplayError(f"invalid evidence scalar: {scalar}")
    for split in SPLITS:
        record = expected_counts.get(split)
        if not isinstance(record, Mapping):
            raise EvidenceReplayError(f"missing expected counts for {split}")
        windows = int(record["inference_windows"])
        scores = int(record["aligned_scores"])
        for device in DEVICES:
            prefix = f"{split}__{device}"
            for field in PATH_FIELDS:
                key = f"{prefix}__{field}"
                array = result[key]
                if array.shape != _expected_shape(field, windows, scores):
                    raise EvidenceReplayError(f"{key} shape changed")
                if field in PATH_FLOAT_FIELDS:
                    if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
                        raise EvidenceReplayError(f"{key} must be finite float64")
                elif field in PATH_INTEGER_FIELDS:
                    if array.dtype != np.dtype("int64"):
                        raise EvidenceReplayError(f"{key} must be int64")
                elif array.dtype != np.dtype("bool"):
                    raise EvidenceReplayError(f"{key} must be bool")
    return result


def _reconstruct(
    payload: Mapping[str, np.ndarray],
    scorer: CaredBCScorer,
) -> tuple[
    dict[str, dict[str, PortabilityPath]],
    dict[str, dict[str, AlignedResidualBatch]],
    dict[str, dict[str, CaredScorePath]],
]:
    paths: dict[str, dict[str, PortabilityPath]] = {split: {} for split in SPLITS}
    aligned: dict[str, dict[str, AlignedResidualBatch]] = {split: {} for split in SPLITS}
    scores: dict[str, dict[str, CaredScorePath]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for device in DEVICES:
            prefix = f"{split}__{device}"
            residual = payload[f"{prefix}__residual"]
            physical = payload[f"{prefix}__physical_features"]
            if not np.array_equal(residual[:, PREDICTION_RESIDUAL_COUNT:], physical):
                raise EvidenceReplayError(f"{prefix} physical features disagree with residual")
            batch = AlignedResidualBatch(
                residuals=residual,
                prediction_residuals=residual[:, :PREDICTION_RESIDUAL_COUNT],
                physical_features=physical,
                score_indices=payload[f"{prefix}__indices"],
                score_timestamps=payload[f"{prefix}__timestamps_ns"].astype(
                    "datetime64[ns]"
                ),
                state_codes=payload[f"{prefix}__state_codes"],
                source_window_rows=payload[f"{prefix}__source_window_rows"],
                reset_before=payload[f"{prefix}__reset_before"],
            )
            recomputed = scorer.score(
                batch.residuals,
                batch.state_codes,
                batch.score_timestamps,
                reset_before=batch.reset_before,
            )
            for field in SCORE_FIELDS:
                if not np.array_equal(
                    np.asarray(getattr(recomputed, field)),
                    payload[f"{prefix}__{field}"],
                ):
                    raise EvidenceReplayError(
                        f"{prefix} persisted {field} differs from scorer recomputation"
                    )
            if not np.array_equal(
                np.asarray(recomputed.reset), payload[f"{prefix}__score_reset"]
            ):
                raise EvidenceReplayError(
                    f"{prefix} persisted score reset differs from recomputation"
                )
            item = portability_path(payload[f"{prefix}__prediction"], batch, recomputed)
            aligned[split][device] = batch
            scores[split][device] = recomputed
            paths[split][device] = item
    return paths, aligned, scores


def replay_portability_evidence(
    path: str | os.PathLike[str],
    *,
    scorer: CaredBCScorer,
    expected_counts: Mapping[str, Mapping[str, int]],
    expected_h_raw: float,
    f1_origin: Any = F1_ORIGIN,
) -> ReplayedPortabilityEvidence:
    if not isinstance(scorer, CaredBCScorer):
        raise EvidenceReplayError("replay requires a concrete CaredBCScorer")
    if scorer.threshold is not None:
        raise EvidenceReplayError("replay scorer must remain uncalibrated")
    payload = load_evidence_npz(path, expected_counts=expected_counts)
    h_raw = float(payload["h_raw"])
    if h_raw != float(expected_h_raw):
        raise EvidenceReplayError("persisted h_raw differs from the frozen threshold")
    paths, aligned, scores = _reconstruct(payload, scorer)
    certificate = certify_development_portability(
        healthy_cpu=paths["dev_healthy"]["cpu"],
        healthy_cuda=paths["dev_healthy"]["cuda"],
        f1_cpu=paths["dev_fault_f1"]["cpu"],
        f1_cuda=paths["dev_fault_f1"]["cuda"],
        h_raw=h_raw,
        f1_origin=f1_origin,
    )
    if type(certificate) is not CertifiedDevelopmentPortabilityAudit:
        raise EvidenceReplayError("trusted certifier returned an unexpected type")
    core = certificate.as_dict()
    threshold = core["threshold"]
    for name in ("global_envelope_delta", "h_eff"):
        if float(payload[name]) != float(threshold[name]):
            raise EvidenceReplayError(f"persisted {name} differs from trusted audit")
    alarms, alarm_arrays = alarm_evidence(paths, float(threshold["h_eff"]))
    for key, expected in alarm_arrays.items():
        if not np.array_equal(payload[key], expected):
            raise EvidenceReplayError(f"{key} differs from recomputed effective alarm")
    manifest = semantic_array_manifest(payload)
    return ReplayedPortabilityEvidence(
        paths=paths,
        aligned=aligned,
        scores=scores,
        certificate=certificate,
        exact_backend_checks=exact_backend_checks(aligned),
        alarm_array_hashes=alarms,
        full_sequence_max_abs_differences=full_sequence_max_abs_differences(
            paths, scores
        ),
        semantic_manifest=manifest,
        semantic_sha256=_canonical_fingerprint(manifest),
        v1_projection_semantic_sha256=v1_projection_semantic_fingerprint(payload),
    )


__all__ = [
    "DEVICES",
    "EVIDENCE_KEYS",
    "EvidenceReplayError",
    "F1_ORIGIN",
    "ReplayedPortabilityEvidence",
    "SCALAR_FIELDS",
    "SCORE_FIELDS",
    "SPLITS",
    "V1_PROJECTION_KEYS",
    "alarm_evidence",
    "array_sha256",
    "build_evidence_payload",
    "exact_backend_checks",
    "full_sequence_max_abs_differences",
    "load_evidence_npz",
    "portability_path",
    "replay_portability_evidence",
    "save_evidence_npz",
    "semantic_array_fingerprint",
    "semantic_array_manifest",
    "v1_projection_semantic_fingerprint",
]
