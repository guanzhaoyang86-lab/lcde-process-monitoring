"""Strict, auditable persistence helpers for the CRIS-v9 study.

This module deliberately keeps protocol JSON, one-shot claims, and numeric
artifacts in distinct byte domains.  JSON files are canonical UTF-8 plus one
LF.  Numeric artifacts are no-pickle NPZ archives whose exact schemas are
validated before monitor objects are reconstructed through their public APIs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import tempfile
from typing import Any
import zipfile

import numpy as np

from mate_spc.hash_contract import (
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_json_sha256,
    require_canonical_json_file,
    sha256_file,
    strict_json_from_bytes,
)
from mate_spc_cris_v9.monitor import (
    ComponentKey,
    MonitorConfig,
    PointCalibrator,
    TrajectoryThreshold,
)


POINT_CALIBRATOR_SCHEMA = "mate_spc_cris_v9_point_calibrator"
POINT_CALIBRATOR_SCHEMA_VERSION = 1
TRAJECTORY_THRESHOLD_SCHEMA = "mate_spc_cris_v9_trajectory_threshold"
TRAJECTORY_THRESHOLD_SCHEMA_VERSION = 1
FREEZE_MANIFEST_SCHEMA = "mate_spc_cris_v9_freeze_manifest"
FREEZE_MANIFEST_SCHEMA_VERSION = 1
CONFIRMATION_CLAIM_SCHEMA = "mate_spc_cris_v9_confirmation_global_claim"
CONFIRMATION_CLAIM_SCHEMA_VERSION = 1
CONFIRMATION_GLOBAL_CLAIM_RELATIVE_PATH = Path(
    "outputs/cris_v9_four_tank/confirmation_global.claim.json"
)
FOUR_TANK_PROTOCOL_SCHEMA = "cris_v9_four_tank_study_charter"

_POINT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_id",
        "config_json",
        "components_json",
        "horizon_count",
        "dimension_count",
        "sample_offsets",
        "sorted_samples_flat",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_id",
        "config_json",
        "horizon_count",
        "dimension_count",
        "threshold_json",
        "calibration_maxima",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "top_k",
        "moment_weights",
        "rolling_scales",
        "eta",
        "hmm_modes",
        "dependence_lag",
        "reset_gap",
    }
)
_COMPONENT_KEYS = frozenset({"moment", "horizon", "scale", "side", "name"})
_THRESHOLD_DOCUMENT_KEYS = frozenset(
    {
        "value",
        "alpha",
        "calibration_run_count",
        "ascending_rank",
        "comparison",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "hash_algorithm",
        "path_mode",
        "sources",
        "artifacts",
        "metadata",
    }
)
_BINDING_KEYS = frozenset({"relative_path", "size_bytes", "file_bytes_sha256"})


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _validate_candidate_id(value: object, *, name: str = "candidate_id") -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty trimmed string")
    return value


def _strict_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _file_record(path: Path, *, relative_path: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.is_symlink():
        raise ValueError(f"bound file must not be a symlink: {path}")
    return {
        "relative_path": path.as_posix() if relative_path is None else relative_path,
        "size_bytes": path.stat().st_size,
        "file_bytes_sha256": sha256_file(path),
    }


def _written_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_bytes_sha256": sha256_file(path),
    }


def read_canonical_json(path: str | os.PathLike[str]) -> Any:
    """Read strict canonical JSON whose exact file form ends in one LF."""
    document, _contract = require_canonical_json_file(Path(path))
    return document


def write_json_atomic(
    path: str | os.PathLike[str], document: object
) -> dict[str, object]:
    """Atomically replace a canonical JSON file using a same-directory temp."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _written_record(target)


def write_json_exclusive(
    path: str | os.PathLike[str], document: object
) -> dict[str, object]:
    """Create a permanent canonical JSON token using kernel ``O_EXCL``.

    Once the target has been created it is never cleaned up by this function,
    even when a later write or fsync fails.  A partial claim therefore remains
    fail-closed evidence and continues to block a replay.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(document)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _written_record(target)


def _atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _text_scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=np.str_)


def _int64_scalar(value: int) -> np.ndarray:
    return np.asarray(value, dtype=np.int64)


def _read_text_scalar(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _read_int64_scalar(value: np.ndarray, *, name: str, minimum: int) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must be a scalar int64 array")
    return _strict_integer(array.item(), name=name, minimum=minimum)


def _parse_canonical_json_text(value: str, *, name: str) -> Any:
    try:
        document = strict_json_from_bytes(value.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain strict JSON") from exc
    if canonical_json_bytes(document).decode("utf-8") != value:
        raise ValueError(f"{name} must contain canonical JSON")
    return document


def _monitor_config_document(config: MonitorConfig) -> dict[str, object]:
    if not isinstance(config, MonitorConfig):
        raise TypeError("config must be MonitorConfig")
    # asdict makes a fresh tree; canonical JSON deliberately turns tuples into
    # arrays, while the loader restores the tuple-valued public constructor API.
    return asdict(config)


def _monitor_config_from_document(value: object) -> MonitorConfig:
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        raise ValueError("config JSON keys do not match MonitorConfig")
    weights = value["moment_weights"]
    rolling = value["rolling_scales"]
    modes = value["hmm_modes"]
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("moment_weights must be a length-3 JSON array")
    if not isinstance(rolling, list) or len(rolling) != 3:
        raise ValueError("rolling_scales must be a length-3 JSON array")
    if not isinstance(modes, list) or any(
        not isinstance(mode, list) or len(mode) != 2 for mode in modes
    ):
        raise ValueError("hmm_modes must be a JSON array of pairs")
    config = MonitorConfig(
        top_k=_strict_integer(value["top_k"], name="top_k", minimum=1),
        moment_weights=tuple(
            _strict_float(item, name=f"moment_weights[{index}]")
            for index, item in enumerate(weights)
        ),
        rolling_scales=tuple(
            _strict_integer(item, name=f"rolling_scales[{index}]", minimum=1)
            for index, item in enumerate(rolling)
        ),
        eta=_strict_float(value["eta"], name="eta"),
        hmm_modes=tuple(
            tuple(
                _strict_float(item, name=f"hmm_modes[{row}][{column}]")
                for column, item in enumerate(mode)
            )
            for row, mode in enumerate(modes)
        ),
        dependence_lag=_strict_integer(
            value["dependence_lag"], name="dependence_lag", minimum=1
        ),
        reset_gap=_strict_integer(value["reset_gap"], name="reset_gap", minimum=1),
    )
    if canonical_json_bytes(_monitor_config_document(config)) != canonical_json_bytes(value):
        raise ValueError("config JSON does not exactly represent MonitorConfig")
    return config


def _components_document(
    components: tuple[ComponentKey, ...]
) -> list[dict[str, object]]:
    return [
        {
            "moment": component.moment,
            "horizon": component.horizon,
            "scale": component.scale,
            "side": component.side,
            "name": component.name,
        }
        for component in components
    ]


def _components_from_document(value: object) -> tuple[ComponentKey, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("components JSON must be a nonempty array")
    result: list[ComponentKey] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _COMPONENT_KEYS:
            raise ValueError(f"component {index} keys do not match the schema")
        if not isinstance(item["moment"], str) or not isinstance(item["side"], str):
            raise ValueError(f"component {index} moment and side must be text")
        if not isinstance(item["name"], str):
            raise ValueError(f"component {index} name must be text")
        component = ComponentKey(
            moment=item["moment"],
            horizon=_strict_integer(
                item["horizon"], name=f"component[{index}].horizon", minimum=0
            ),
            scale=_strict_integer(
                item["scale"], name=f"component[{index}].scale", minimum=1
            ),
            side=item["side"],
        )
        if item["name"] != component.name:
            raise ValueError(f"component {index} name does not match its identity")
        result.append(component)
    if len(set(result)) != len(result):
        raise ValueError("component identities must be unique")
    return tuple(result)


def save_point_calibrator_npz(
    calibrator: PointCalibrator,
    path: str | os.PathLike[str],
    *,
    candidate_id: str,
) -> Path:
    """Save a fitted point calibrator as a strict, no-pickle NPZ artifact."""
    if not isinstance(calibrator, PointCalibrator):
        raise TypeError("calibrator must be PointCalibrator")
    candidate = _validate_candidate_id(candidate_id)
    offsets = np.empty(len(calibrator.sorted_samples) + 1, dtype=np.int64)
    offsets[0] = 0
    for index, sample in enumerate(calibrator.sorted_samples):
        offsets[index + 1] = offsets[index] + sample.size
    flattened = np.concatenate(calibrator.sorted_samples).astype(np.float64, copy=False)
    payload = {
        "schema": _text_scalar(POINT_CALIBRATOR_SCHEMA),
        "schema_version": _int64_scalar(POINT_CALIBRATOR_SCHEMA_VERSION),
        "candidate_id": _text_scalar(candidate),
        "config_json": _text_scalar(
            canonical_json_bytes(_monitor_config_document(calibrator.config)).decode("utf-8")
        ),
        "components_json": _text_scalar(
            canonical_json_bytes(_components_document(calibrator.components)).decode("utf-8")
        ),
        "horizon_count": _int64_scalar(calibrator.horizon_count),
        "dimension_count": _int64_scalar(calibrator.dimension_count),
        "sample_offsets": offsets,
        "sorted_samples_flat": flattened,
    }
    return _atomic_npz(Path(path), payload)


def _load_npz_exact(
    path: Path,
    expected_keys: frozenset[str],
    *,
    expected_file_sha256: str,
) -> dict[str, np.ndarray]:
    wanted = _validate_sha256(expected_file_sha256, name="expected_file_sha256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != wanted:
        raise ValueError("NPZ file SHA-256 mismatch")
    expected_members = {f"{name}.npy" for name in expected_keys}
    with zipfile.ZipFile(io.BytesIO(raw), mode="r") as zipped:
        members = zipped.namelist()
        if (
            len(members) != len(expected_members)
            or len(set(members)) != len(members)
            or set(members) != expected_members
        ):
            raise ValueError("NPZ ZIP members are not exact and unique")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(expected_keys) or len(set(archive.files)) != len(
            archive.files
        ):
            raise ValueError("NPZ array keys must be exact and unique")
        observed = set(archive.files)
        if observed != expected_keys:
            missing = sorted(expected_keys - observed)
            unexpected = sorted(observed - expected_keys)
            raise ValueError(
                f"NPZ keys mismatch; missing={missing}, unexpected={unexpected}"
            )
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def load_point_calibrator_npz(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    expected_candidate_id: str,
    expected_config: MonitorConfig,
    expected_horizon_count: int,
    expected_dimension_count: int,
) -> PointCalibrator:
    """Strictly load and reconstruct a :class:`PointCalibrator`."""
    expected_candidate = _validate_candidate_id(
        expected_candidate_id, name="expected_candidate_id"
    )
    if not isinstance(expected_config, MonitorConfig):
        raise TypeError("expected_config must be MonitorConfig")
    expected_horizons = _strict_integer(
        expected_horizon_count, name="expected_horizon_count", minimum=1
    )
    expected_dimensions = _strict_integer(
        expected_dimension_count, name="expected_dimension_count", minimum=1
    )
    payload = _load_npz_exact(
        Path(path), _POINT_KEYS, expected_file_sha256=expected_file_sha256
    )
    if _read_text_scalar(payload["schema"], name="schema") != POINT_CALIBRATOR_SCHEMA:
        raise ValueError("point-calibrator schema mismatch")
    version = _read_int64_scalar(
        payload["schema_version"], name="schema_version", minimum=1
    )
    if version != POINT_CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("point-calibrator schema_version mismatch")
    candidate = _read_text_scalar(payload["candidate_id"], name="candidate_id")
    if candidate != expected_candidate:
        raise ValueError("point-calibrator candidate identity mismatch")
    horizon_count = _read_int64_scalar(
        payload["horizon_count"], name="horizon_count", minimum=1
    )
    dimension_count = _read_int64_scalar(
        payload["dimension_count"], name="dimension_count", minimum=1
    )
    if horizon_count != expected_horizons or dimension_count != expected_dimensions:
        raise ValueError("point-calibrator horizon/dimension identity mismatch")
    config_document = _parse_canonical_json_text(
        _read_text_scalar(payload["config_json"], name="config_json"),
        name="config_json",
    )
    config = _monitor_config_from_document(config_document)
    if config != expected_config:
        raise ValueError("point-calibrator config identity mismatch")
    component_document = _parse_canonical_json_text(
        _read_text_scalar(payload["components_json"], name="components_json"),
        name="components_json",
    )
    stored_components = _components_from_document(component_document)

    offsets = payload["sample_offsets"]
    flattened = payload["sorted_samples_flat"]
    if offsets.dtype != np.dtype(np.int64) or offsets.ndim != 1:
        raise ValueError("sample_offsets must be a one-dimensional int64 array")
    if flattened.dtype != np.dtype(np.float64) or flattened.ndim != 1:
        raise ValueError("sorted_samples_flat must be a one-dimensional float64 array")
    if not np.isfinite(flattened).all():
        raise ValueError("sorted_samples_flat must contain only finite values")
    if offsets.shape != (len(stored_components) + 1,):
        raise ValueError("sample_offsets length must equal component_count + 1")
    if offsets[0] != 0 or offsets[-1] != flattened.size:
        raise ValueError("sample_offsets endpoints do not match sorted_samples_flat")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("sample_offsets must be strictly increasing")
    samples = tuple(
        np.array(flattened[offsets[index] : offsets[index + 1]], copy=True)
        for index in range(len(stored_components))
    )
    calibrator = PointCalibrator.from_sorted_samples(
        samples,
        horizon_count=horizon_count,
        dimension_count=dimension_count,
        config=config,
    )
    if calibrator.components != stored_components:
        raise ValueError("stored component identities differ from the public monitor ordering")
    if calibrator.config != config:
        raise ValueError("reconstructed point calibrator changed its stored config")
    return calibrator


def _threshold_document(threshold: TrajectoryThreshold) -> dict[str, object]:
    return {
        "value": threshold.value,
        "alpha": threshold.alpha,
        "calibration_run_count": threshold.calibration_run_count,
        "ascending_rank": threshold.ascending_rank,
        "comparison": threshold.comparison,
    }


def save_trajectory_threshold_npz(
    threshold: TrajectoryThreshold,
    path: str | os.PathLike[str],
    *,
    candidate_id: str,
    config: MonitorConfig,
    horizon_count: int,
    dimension_count: int,
) -> Path:
    """Save a trajectory threshold and every calibration maximum without pickle."""
    if not isinstance(threshold, TrajectoryThreshold):
        raise TypeError("threshold must be TrajectoryThreshold")
    candidate = _validate_candidate_id(candidate_id)
    if not isinstance(config, MonitorConfig):
        raise TypeError("config must be MonitorConfig")
    horizons = _strict_integer(horizon_count, name="horizon_count", minimum=1)
    dimensions = _strict_integer(dimension_count, name="dimension_count", minimum=1)
    if config.top_k > dimensions:
        raise ValueError("config.top_k cannot exceed dimension_count")
    payload = {
        "schema": _text_scalar(TRAJECTORY_THRESHOLD_SCHEMA),
        "schema_version": _int64_scalar(TRAJECTORY_THRESHOLD_SCHEMA_VERSION),
        "candidate_id": _text_scalar(candidate),
        "config_json": _text_scalar(
            canonical_json_bytes(_monitor_config_document(config)).decode("utf-8")
        ),
        "horizon_count": _int64_scalar(horizons),
        "dimension_count": _int64_scalar(dimensions),
        "threshold_json": _text_scalar(
            canonical_json_bytes(_threshold_document(threshold)).decode("utf-8")
        ),
        "calibration_maxima": np.array(
            threshold.calibration_maxima, dtype=np.float64, copy=True, order="C"
        ),
    }
    return _atomic_npz(Path(path), payload)


def load_trajectory_threshold_npz(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    expected_candidate_id: str,
    expected_config: MonitorConfig,
    expected_horizon_count: int,
    expected_dimension_count: int,
) -> TrajectoryThreshold:
    """Strictly load and reconstruct a :class:`TrajectoryThreshold`."""
    expected_candidate = _validate_candidate_id(
        expected_candidate_id, name="expected_candidate_id"
    )
    if not isinstance(expected_config, MonitorConfig):
        raise TypeError("expected_config must be MonitorConfig")
    expected_horizons = _strict_integer(
        expected_horizon_count, name="expected_horizon_count", minimum=1
    )
    expected_dimensions = _strict_integer(
        expected_dimension_count, name="expected_dimension_count", minimum=1
    )
    payload = _load_npz_exact(
        Path(path), _THRESHOLD_KEYS, expected_file_sha256=expected_file_sha256
    )
    if _read_text_scalar(payload["schema"], name="schema") != TRAJECTORY_THRESHOLD_SCHEMA:
        raise ValueError("trajectory-threshold schema mismatch")
    version = _read_int64_scalar(
        payload["schema_version"], name="schema_version", minimum=1
    )
    if version != TRAJECTORY_THRESHOLD_SCHEMA_VERSION:
        raise ValueError("trajectory-threshold schema_version mismatch")
    candidate = _read_text_scalar(payload["candidate_id"], name="candidate_id")
    if candidate != expected_candidate:
        raise ValueError("trajectory-threshold candidate identity mismatch")
    config_document = _parse_canonical_json_text(
        _read_text_scalar(payload["config_json"], name="config_json"),
        name="config_json",
    )
    config = _monitor_config_from_document(config_document)
    if config != expected_config:
        raise ValueError("trajectory-threshold config identity mismatch")
    horizons = _read_int64_scalar(
        payload["horizon_count"], name="horizon_count", minimum=1
    )
    dimensions = _read_int64_scalar(
        payload["dimension_count"], name="dimension_count", minimum=1
    )
    if horizons != expected_horizons or dimensions != expected_dimensions:
        raise ValueError("trajectory-threshold horizon/dimension identity mismatch")
    document = _parse_canonical_json_text(
        _read_text_scalar(payload["threshold_json"], name="threshold_json"),
        name="threshold_json",
    )
    if not isinstance(document, dict) or set(document) != _THRESHOLD_DOCUMENT_KEYS:
        raise ValueError("threshold JSON keys do not match the schema")
    if document["comparison"] != "strictly_greater_than":
        raise ValueError("trajectory threshold comparison must be strictly_greater_than")
    maxima = payload["calibration_maxima"]
    if maxima.dtype != np.dtype(np.float64) or maxima.ndim != 1:
        raise ValueError("calibration_maxima must be a one-dimensional float64 array")
    count = _strict_integer(
        document["calibration_run_count"], name="calibration_run_count", minimum=1
    )
    if maxima.shape != (count,) or not np.isfinite(maxima).all():
        raise ValueError("calibration_maxima must be finite and aligned with run count")
    threshold = TrajectoryThreshold(
        value=_strict_float(document["value"], name="value"),
        alpha=_strict_float(document["alpha"], name="alpha"),
        calibration_run_count=count,
        ascending_rank=_strict_integer(
            document["ascending_rank"], name="ascending_rank", minimum=1
        ),
        calibration_maxima=maxima,
    )
    if canonical_json_bytes(_threshold_document(threshold)) != canonical_json_bytes(
        document
    ):
        raise ValueError("reconstructed trajectory threshold differs from stored metadata")
    return threshold


def _normalize_root(root: str | os.PathLike[str]) -> Path:
    value = Path(root).resolve(strict=True)
    if not value.is_dir():
        raise NotADirectoryError(value)
    return value


def _canonical_relative_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be canonical POSIX relative path text")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"{name} must be a canonical root-contained relative path")
    return Path(*pure.parts)


def _resolve_bound_path(path: str | os.PathLike[str], *, root: Path) -> tuple[Path, str]:
    candidate = Path(path)
    unresolved = root / candidate if not candidate.is_absolute() else candidate
    if unresolved.is_symlink():
        raise ValueError(f"bound path must not be a symlink: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bound path must stay within root: {resolved}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"bound path must be a regular non-symlink file: {resolved}")
    return resolved, relative.as_posix()


def _binding_group(
    values: Mapping[str, str | os.PathLike[str]],
    *,
    expected_sha256: Mapping[str, str] | None,
    root: Path,
    group_name: str,
) -> dict[str, dict[str, object]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{group_name} must be a nonempty mapping")
    keys = set(values)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{group_name} labels must be nonempty strings")
    expected = None if expected_sha256 is None else dict(expected_sha256)
    if expected is not None and set(expected) != keys:
        raise ValueError(f"expected {group_name} SHA-256 labels must exactly match paths")
    result: dict[str, dict[str, object]] = {}
    seen_paths: set[str] = set()
    for label in sorted(keys):
        resolved, relative = _resolve_bound_path(values[label], root=root)
        record = _file_record(resolved, relative_path=relative)
        if relative in seen_paths:
            raise ValueError(f"{group_name} paths must be unique")
        seen_paths.add(relative)
        if expected is not None:
            wanted = _validate_sha256(
                expected[label], name=f"expected {group_name}[{label}] SHA-256"
            )
            if record["file_bytes_sha256"] != wanted:
                raise ValueError(f"{group_name}[{label}] SHA-256 mismatch")
        result[label] = record
    return result


def build_freeze_manifest(
    *,
    root: str | os.PathLike[str],
    sources: Mapping[str, str | os.PathLike[str]],
    artifacts: Mapping[str, str | os.PathLike[str]],
    expected_source_sha256: Mapping[str, str] | None = None,
    expected_artifact_sha256: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind named source and artifact files to exact relative paths and hashes."""
    base = _normalize_root(root)
    if set(sources) & set(artifacts):
        raise ValueError("source and artifact labels must be disjoint")
    metadata_document: object = {} if metadata is None else dict(metadata)
    # Round-trip through the strict canonical domain to reject unsupported or
    # nonfinite metadata before any manifest is returned.
    metadata_document = strict_json_from_bytes(canonical_json_bytes(metadata_document))
    source_bindings = _binding_group(
        sources,
        expected_sha256=expected_source_sha256,
        root=base,
        group_name="sources",
    )
    artifact_bindings = _binding_group(
        artifacts,
        expected_sha256=expected_artifact_sha256,
        root=base,
        group_name="artifacts",
    )
    source_paths = {str(record["relative_path"]) for record in source_bindings.values()}
    artifact_paths = {
        str(record["relative_path"]) for record in artifact_bindings.values()
    }
    if source_paths & artifact_paths:
        raise ValueError("source and artifact physical paths must be disjoint")
    manifest = {
        "schema": FREEZE_MANIFEST_SCHEMA,
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "hash_algorithm": "sha256_file_bytes",
        "path_mode": "relative_to_explicit_root",
        "sources": source_bindings,
        "artifacts": artifact_bindings,
        "metadata": metadata_document,
    }
    canonical_json_bytes(manifest)
    return manifest


def _verify_binding_group(
    value: object, *, root: Path, group_name: str
) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"manifest {group_name} must be a nonempty object")
    result: dict[str, dict[str, object]] = {}
    for label, raw_record in value.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"manifest {group_name} labels must be nonempty strings")
        if not isinstance(raw_record, dict) or set(raw_record) != _BINDING_KEYS:
            raise ValueError(f"manifest {group_name}[{label}] keys mismatch")
        relative_path = raw_record["relative_path"]
        relative = _canonical_relative_path(
            relative_path, name=f"manifest {group_name}[{label}] path"
        )
        unresolved = root / relative
        if unresolved.is_symlink():
            raise ValueError(f"manifest {group_name}[{label}] path is a symlink")
        resolved = unresolved.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest {group_name}[{label}] escapes root") from exc
        expected_size = _strict_integer(
            raw_record["size_bytes"],
            name=f"manifest {group_name}[{label}].size_bytes",
            minimum=0,
        )
        expected_hash = _validate_sha256(
            raw_record["file_bytes_sha256"],
            name=f"manifest {group_name}[{label}].file_bytes_sha256",
        )
        observed = _file_record(resolved, relative_path=relative.as_posix())
        if observed["size_bytes"] != expected_size:
            raise ValueError(f"manifest {group_name}[{label}] size mismatch")
        if observed["file_bytes_sha256"] != expected_hash:
            raise ValueError(f"manifest {group_name}[{label}] SHA-256 mismatch")
        if observed["relative_path"] != relative_path:
            raise ValueError(f"manifest {group_name}[{label}] path is not canonical")
        result[label] = observed
    return result


def verify_freeze_manifest(
    manifest: object, *, root: str | os.PathLike[str]
) -> dict[str, object]:
    """Re-hash every binding and fail closed on any path, size, or hash mismatch."""
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("freeze manifest top-level keys mismatch")
    if manifest["schema"] != FREEZE_MANIFEST_SCHEMA:
        raise ValueError("freeze manifest schema mismatch")
    version = _strict_integer(
        manifest["schema_version"], name="freeze manifest schema_version", minimum=1
    )
    if version != FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("freeze manifest schema_version mismatch")
    if manifest["hash_algorithm"] != "sha256_file_bytes":
        raise ValueError("freeze manifest hash_algorithm mismatch")
    if manifest["path_mode"] != "relative_to_explicit_root":
        raise ValueError("freeze manifest path_mode mismatch")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("freeze manifest metadata must be an object")
    canonical_json_bytes(manifest["metadata"])
    base = _normalize_root(root)
    sources = _verify_binding_group(manifest["sources"], root=base, group_name="sources")
    artifacts = _verify_binding_group(
        manifest["artifacts"], root=base, group_name="artifacts"
    )
    if set(sources) & set(artifacts):
        raise ValueError("manifest source and artifact labels overlap")
    source_paths = {str(record["relative_path"]) for record in sources.values()}
    artifact_paths = {str(record["relative_path"]) for record in artifacts.values()}
    if len(source_paths) != len(sources) or len(artifact_paths) != len(artifacts):
        raise ValueError("manifest physical paths must be unique within each group")
    if source_paths & artifact_paths:
        raise ValueError("manifest source and artifact physical paths overlap")
    return {
        "manifest_canonical_json_sha256": canonical_json_sha256(manifest),
        "source_count": len(sources),
        "artifact_count": len(artifacts),
    }


def write_freeze_manifest(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    sources: Mapping[str, str | os.PathLike[str]],
    artifacts: Mapping[str, str | os.PathLike[str]],
    expected_source_sha256: Mapping[str, str] | None = None,
    expected_artifact_sha256: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build, verify, then exclusively persist an immutable freeze manifest."""
    manifest = build_freeze_manifest(
        root=root,
        sources=sources,
        artifacts=artifacts,
        expected_source_sha256=expected_source_sha256,
        expected_artifact_sha256=expected_artifact_sha256,
        metadata=metadata,
    )
    verify_freeze_manifest(manifest, root=root)
    return write_json_exclusive(path, manifest)


def create_confirmation_global_claim(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    candidate_id: str,
    protocol_path: str | os.PathLike[str],
    freeze_manifest_path: str | os.PathLike[str],
    expected_protocol_sha256: str | None = None,
    expected_freeze_manifest_sha256: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create the one global confirmation claim with kernel ``O_EXCL``.

    Both governing JSON files must be canonical.  The freeze manifest is also
    revalidated against every bound source/artifact immediately before claim
    creation.  A second call for the same path raises ``FileExistsError``.
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id must be a nonempty string")
    base = _normalize_root(root)
    protocol, protocol_relative = _resolve_bound_path(protocol_path, root=base)
    freeze, freeze_relative = _resolve_bound_path(freeze_manifest_path, root=base)
    read_canonical_json(protocol)
    freeze_document = read_canonical_json(freeze)
    verify_freeze_manifest(freeze_document, root=base)
    protocol_record = _file_record(protocol, relative_path=protocol_relative)
    freeze_record = _file_record(freeze, relative_path=freeze_relative)
    if expected_protocol_sha256 is not None:
        wanted = _validate_sha256(
            expected_protocol_sha256, name="expected_protocol_sha256"
        )
        if protocol_record["file_bytes_sha256"] != wanted:
            raise ValueError("protocol SHA-256 mismatch")
    if expected_freeze_manifest_sha256 is not None:
        wanted = _validate_sha256(
            expected_freeze_manifest_sha256,
            name="expected_freeze_manifest_sha256",
        )
        if freeze_record["file_bytes_sha256"] != wanted:
            raise ValueError("freeze manifest SHA-256 mismatch")
    metadata_document: object = {} if metadata is None else dict(metadata)
    metadata_document = strict_json_from_bytes(canonical_json_bytes(metadata_document))
    claim = {
        "schema": CONFIRMATION_CLAIM_SCHEMA,
        "schema_version": CONFIRMATION_CLAIM_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "protocol": protocol_record,
        "freeze_manifest": freeze_record,
        "metadata": metadata_document,
    }
    return write_json_exclusive(path, claim)


__all__ = [
    "CONFIRMATION_CLAIM_SCHEMA",
    "CONFIRMATION_CLAIM_SCHEMA_VERSION",
    "FREEZE_MANIFEST_SCHEMA",
    "FREEZE_MANIFEST_SCHEMA_VERSION",
    "POINT_CALIBRATOR_SCHEMA",
    "POINT_CALIBRATOR_SCHEMA_VERSION",
    "TRAJECTORY_THRESHOLD_SCHEMA",
    "TRAJECTORY_THRESHOLD_SCHEMA_VERSION",
    "build_freeze_manifest",
    "canonical_json_bytes",
    "canonical_json_file_bytes",
    "canonical_json_sha256",
    "create_confirmation_global_claim",
    "load_point_calibrator_npz",
    "load_trajectory_threshold_npz",
    "read_canonical_json",
    "save_point_calibrator_npz",
    "save_trajectory_threshold_npz",
    "sha256_file",
    "verify_freeze_manifest",
    "write_freeze_manifest",
    "write_json_atomic",
    "write_json_exclusive",
]
