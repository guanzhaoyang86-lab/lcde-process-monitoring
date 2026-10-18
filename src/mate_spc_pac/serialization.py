"""Strict, pickle-free serialization for deployment-bound PAC models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .certificate import PACCalibration
from .common import DeploymentSpec, PrefixApplicabilityConfig
from .core import WSPACRMaxModel
from .probability import PACRank


MODEL_SCHEMA = "mate_spc_ws_pac_rmax_cusum"
MODEL_SCHEMA_VERSION = 2


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON safe")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("metadata_json is not strict finite JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model metadata must decode to a JSON object")
    return value


def _exact_int64(archive: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(archive[key])
    if value.dtype != np.dtype(np.int64) or value.shape != shape:
        raise ValueError(f"{key} must have dtype int64 and shape {shape}")
    return value


def _exact_float64(archive: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(archive[key])
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        raise ValueError(f"{key} must have dtype float64 and shape {shape}")
    return value


def _unicode_scalar(archive: Any, key: str) -> str:
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{key} must be a scalar Unicode string")
    return str(value.item())


def _unicode_vector(archive: Any, key: str) -> tuple[str, ...]:
    value = np.asarray(archive[key])
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ValueError(f"{key} must be a one-dimensional Unicode array")
    return tuple(str(item) for item in value.tolist())


def _empty_certificate_arrays() -> dict[str, np.ndarray]:
    return {
        "certificate_status": np.asarray(""),
        "certificate_threshold": np.empty(0, dtype=np.float64),
        "certificate_epsilon": np.empty(0, dtype=np.float64),
        "certificate_delta": np.empty(0, dtype=np.float64),
        "certificate_rank": np.empty(0, dtype=np.int64),
        "certificate_achieved_confidence": np.empty(0, dtype=np.float64),
        "certificate_upper_tail_bound": np.empty(0, dtype=np.float64),
        "certificate_minimum_sample_count": np.empty(0, dtype=np.int64),
        "certificate_rank_driver_run": np.asarray(""),
        "certificate_comparison": np.asarray(""),
        "certificate_deployment_sha256": np.asarray(""),
        "certificate_monitor_configuration_sha256": np.asarray(""),
        "certificate_unconditional_count": np.empty(0, dtype=np.int64),
        "certificate_pass_count": np.empty(0, dtype=np.int64),
        "certificate_defer_count": np.empty(0, dtype=np.int64),
        "certificate_maxima_sha256": np.asarray(""),
        "certificate_maxima_names": np.asarray([], dtype="U1"),
        "certificate_maxima_values": np.empty(0, dtype=np.float64),
        "certificate_deferred_runs": np.asarray([], dtype="U1"),
    }


def _certificate_arrays(certificate: PACCalibration | None) -> dict[str, np.ndarray]:
    if certificate is None:
        return _empty_certificate_arrays()
    return {
        "certificate_status": np.asarray(certificate.status),
        "certificate_threshold": np.asarray([certificate.threshold], dtype=np.float64),
        "certificate_epsilon": np.asarray([certificate.rank.epsilon], dtype=np.float64),
        "certificate_delta": np.asarray([certificate.rank.delta], dtype=np.float64),
        "certificate_rank": np.asarray([certificate.rank.rank_one_based], dtype=np.int64),
        "certificate_achieved_confidence": np.asarray(
            [certificate.rank.achieved_confidence],
            dtype=np.float64,
        ),
        "certificate_upper_tail_bound": np.asarray(
            [certificate.rank.upper_tail_probability_bound],
            dtype=np.float64,
        ),
        "certificate_minimum_sample_count": np.asarray(
            [certificate.rank.minimum_sample_count],
            dtype=np.int64,
        ),
        "certificate_rank_driver_run": np.asarray(certificate.rank_driver_run),
        "certificate_comparison": np.asarray(certificate.comparison),
        "certificate_deployment_sha256": np.asarray(
            certificate.deployment_spec_sha256
        ),
        "certificate_monitor_configuration_sha256": np.asarray(
            certificate.monitor_configuration_sha256
        ),
        "certificate_unconditional_count": np.asarray(
            [certificate.unconditional_sample_count],
            dtype=np.int64,
        ),
        "certificate_pass_count": np.asarray(
            [certificate.pass_conditional_sample_count],
            dtype=np.int64,
        ),
        "certificate_defer_count": np.asarray(
            [certificate.defer_sample_count],
            dtype=np.int64,
        ),
        "certificate_maxima_sha256": np.asarray(
            certificate.calibration_maxima_sha256
        ),
        "certificate_maxima_names": np.asarray(
            [name for name, _ in certificate.run_maxima],
            dtype="U",
        ),
        "certificate_maxima_values": np.asarray(
            [value for _, value in certificate.run_maxima],
            dtype=np.float64,
        ),
        "certificate_deferred_runs": np.asarray(certificate.deferred_runs, dtype="U"),
    }


def save_ws_pac_rmax_model(
    model: WSPACRMaxModel,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    if not isinstance(model, WSPACRMaxModel):
        raise TypeError("model must be a WSPACRMaxModel")
    target = Path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError("model path must end in .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_text = json.dumps(
        {} if metadata is None else dict(metadata),
        default=_json_default,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    spec = model.deployment
    gate = model.prefix_gate
    arrays = {
        "schema": np.asarray(MODEL_SCHEMA),
        "schema_version": np.asarray([MODEL_SCHEMA_VERSION], dtype=np.int64),
        "run_length": np.asarray([spec.run_length], dtype=np.int64),
        "valid_start": np.asarray([spec.valid_start], dtype=np.int64),
        "monitor_start": np.asarray([spec.monitor_start], dtype=np.int64),
        "head_ids": np.asarray(spec.head_ids),
        "clip_limit": np.asarray([spec.clip_limit], dtype=np.float64),
        "persistence_window": np.asarray([spec.persistence_window], dtype=np.int64),
        "upstream_artifact_sha256": np.asarray(spec.upstream_artifact_sha256),
        "deployment_spec_sha256": np.asarray(spec.fingerprint),
        "monitor_configuration_sha256": np.asarray(model.configuration_fingerprint),
        "kappa": model.kappa,
        "median_lower": gate.median_lower,
        "median_upper": gate.median_upper,
        "robust_scale_lower": gate.robust_scale_lower,
        "robust_scale_upper": gate.robust_scale_upper,
        "minimum_prefix_observations": np.asarray(
            [gate.minimum_prefix_observations],
            dtype=np.int64,
        ),
        "certificate_present": np.asarray(
            [int(model.calibration is not None)],
            dtype=np.int64,
        ),
        **_certificate_arrays(model.calibration),
        "metadata_json": np.asarray(metadata_text),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


_REQUIRED_KEYS = {
    "schema",
    "schema_version",
    "run_length",
    "valid_start",
    "monitor_start",
    "head_ids",
    "clip_limit",
    "persistence_window",
    "upstream_artifact_sha256",
    "deployment_spec_sha256",
    "monitor_configuration_sha256",
    "kappa",
    "median_lower",
    "median_upper",
    "robust_scale_lower",
    "robust_scale_upper",
    "minimum_prefix_observations",
    "certificate_present",
    "certificate_status",
    "certificate_threshold",
    "certificate_epsilon",
    "certificate_delta",
    "certificate_rank",
    "certificate_achieved_confidence",
    "certificate_upper_tail_bound",
    "certificate_minimum_sample_count",
    "certificate_rank_driver_run",
    "certificate_comparison",
    "certificate_deployment_sha256",
    "certificate_monitor_configuration_sha256",
    "certificate_unconditional_count",
    "certificate_pass_count",
    "certificate_defer_count",
    "certificate_maxima_sha256",
    "certificate_maxima_names",
    "certificate_maxima_values",
    "certificate_deferred_runs",
    "metadata_json",
}


def _load_certificate(archive: Any, *, present: int) -> PACCalibration | None:
    float_keys = (
        "certificate_threshold",
        "certificate_epsilon",
        "certificate_delta",
        "certificate_achieved_confidence",
        "certificate_upper_tail_bound",
    )
    int_keys = (
        "certificate_rank",
        "certificate_minimum_sample_count",
        "certificate_unconditional_count",
        "certificate_pass_count",
        "certificate_defer_count",
    )
    if present == 0:
        for key in float_keys:
            _exact_float64(archive, key, (0,))
        for key in int_keys:
            _exact_int64(archive, key, (0,))
        if _unicode_vector(archive, "certificate_maxima_names"):
            raise ValueError("uncalibrated archive must have empty maxima names")
        _exact_float64(archive, "certificate_maxima_values", (0,))
        if _unicode_vector(archive, "certificate_deferred_runs"):
            raise ValueError("uncalibrated archive must have empty deferred runs")
        for key in (
            "certificate_status",
            "certificate_rank_driver_run",
            "certificate_comparison",
            "certificate_deployment_sha256",
            "certificate_monitor_configuration_sha256",
            "certificate_maxima_sha256",
        ):
            if _unicode_scalar(archive, key) != "":
                raise ValueError(f"uncalibrated archive must have empty {key}")
        return None

    for key in float_keys:
        _exact_float64(archive, key, (1,))
    for key in int_keys:
        _exact_int64(archive, key, (1,))
    names = _unicode_vector(archive, "certificate_maxima_names")
    values = _exact_float64(archive, "certificate_maxima_values", (len(names),))
    deferred = _unicode_vector(archive, "certificate_deferred_runs")
    rank = PACRank(
        status=_unicode_scalar(archive, "certificate_status"),
        sample_count=int(_exact_int64(archive, "certificate_pass_count", (1,))[0]),
        epsilon=float(_exact_float64(archive, "certificate_epsilon", (1,))[0]),
        delta=float(_exact_float64(archive, "certificate_delta", (1,))[0]),
        rank_one_based=int(_exact_int64(archive, "certificate_rank", (1,))[0]),
        achieved_confidence=float(
            _exact_float64(archive, "certificate_achieved_confidence", (1,))[0]
        ),
        upper_tail_probability_bound=float(
            _exact_float64(archive, "certificate_upper_tail_bound", (1,))[0]
        ),
        minimum_sample_count=int(
            _exact_int64(archive, "certificate_minimum_sample_count", (1,))[0]
        ),
    )
    return PACCalibration(
        status=rank.status,
        threshold=float(_exact_float64(archive, "certificate_threshold", (1,))[0]),
        rank=rank,
        rank_driver_run=_unicode_scalar(archive, "certificate_rank_driver_run"),
        deployment_spec_sha256=_unicode_scalar(
            archive,
            "certificate_deployment_sha256",
        ),
        monitor_configuration_sha256=_unicode_scalar(
            archive,
            "certificate_monitor_configuration_sha256",
        ),
        unconditional_sample_count=int(
            _exact_int64(archive, "certificate_unconditional_count", (1,))[0]
        ),
        pass_conditional_sample_count=rank.sample_count,
        defer_sample_count=int(
            _exact_int64(archive, "certificate_defer_count", (1,))[0]
        ),
        deferred_runs=deferred,
        run_maxima=tuple(zip(names, (float(value) for value in values), strict=True)),
        calibration_maxima_sha256=_unicode_scalar(
            archive,
            "certificate_maxima_sha256",
        ),
        comparison=_unicode_scalar(archive, "certificate_comparison"),
    )


def load_ws_pac_rmax_model(
    path: str | os.PathLike[str],
) -> tuple[WSPACRMaxModel, dict[str, Any]]:
    """Load a strict schema and recompute every deployment/certificate binding."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != _REQUIRED_KEYS:
            missing = sorted(_REQUIRED_KEYS - set(archive.files))
            extra = sorted(set(archive.files) - _REQUIRED_KEYS)
            raise ValueError(f"invalid model archive keys; missing={missing}, extra={extra}")
        if _unicode_scalar(archive, "schema") != MODEL_SCHEMA:
            raise ValueError("unsupported PAC model schema")
        version = _exact_int64(archive, "schema_version", (1,))
        if int(version[0]) != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported PAC model schema version")
        run_length = int(_exact_int64(archive, "run_length", (1,))[0])
        valid_start = int(_exact_int64(archive, "valid_start", (1,))[0])
        monitor_start = int(_exact_int64(archive, "monitor_start", (1,))[0])
        persistence = int(_exact_int64(archive, "persistence_window", (1,))[0])
        prefix_count = int(
            _exact_int64(archive, "minimum_prefix_observations", (1,))[0]
        )
        present = int(_exact_int64(archive, "certificate_present", (1,))[0])
        if present not in (0, 1):
            raise ValueError("certificate_present must be int64 zero or one")
        head_ids = _unicode_vector(archive, "head_ids")
        clip_limit = float(_exact_float64(archive, "clip_limit", (1,))[0])
        spec = DeploymentSpec(
            run_length=run_length,
            valid_start=valid_start,
            monitor_start=monitor_start,
            head_ids=head_ids,
            upstream_artifact_sha256=_unicode_scalar(
                archive,
                "upstream_artifact_sha256",
            ),
            clip_limit=clip_limit,
            persistence_window=persistence,
        )
        stored_spec_fingerprint = _unicode_scalar(archive, "deployment_spec_sha256")
        if stored_spec_fingerprint != spec.fingerprint:
            raise ValueError("deployment spec fingerprint mismatch")
        shape = (spec.head_count,)
        for key in (
            "kappa",
            "median_lower",
            "median_upper",
            "robust_scale_lower",
            "robust_scale_upper",
        ):
            _exact_float64(archive, key, shape)
        gate = PrefixApplicabilityConfig(
            head_count=spec.head_count,
            median_lower=archive["median_lower"],
            median_upper=archive["median_upper"],
            robust_scale_lower=archive["robust_scale_lower"],
            robust_scale_upper=archive["robust_scale_upper"],
            minimum_prefix_observations=prefix_count,
        )
        stored_monitor_fingerprint = _unicode_scalar(
            archive,
            "monitor_configuration_sha256",
        )
        expected_monitor_fingerprint = WSPACRMaxModel(
            deployment=spec,
            prefix_gate=gate,
            kappa=archive["kappa"],
        ).configuration_fingerprint
        if stored_monitor_fingerprint != expected_monitor_fingerprint:
            raise ValueError("monitor configuration fingerprint mismatch")
        certificate = _load_certificate(archive, present=present)
        model = WSPACRMaxModel(
            deployment=spec,
            prefix_gate=gate,
            kappa=archive["kappa"],
            calibration=certificate,
        )
        metadata_text = _unicode_scalar(archive, "metadata_json")
    return model, _load_json_object(metadata_text)
