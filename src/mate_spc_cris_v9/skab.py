"""Strict, label-separated SKAB v0.9 data access for CRIS-v9."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SENSOR_COLUMNS = (
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
)
TIME_COLUMN = "datetime"
LABEL_COLUMNS = ("anomaly", "changepoint")
EXPECTED_COLUMNS = (TIME_COLUMN, *SENSOR_COLUMNS, *LABEL_COLUMNS)
SPLIT_PREFIX = (
    "SKAB-v0.9.0-80bac4a00d9cd2a277a6fda8f6a6d5936b892bd2-split-v1|"
)


@dataclass(frozen=True)
class SKABFeatures:
    run_id: str
    timestamps: np.ndarray
    x: np.ndarray
    feature_names: tuple[str, ...]
    sample_period_seconds: float
    source_sha256: str


@dataclass(frozen=True)
class SKABLabels:
    run_id: str
    anomaly: np.ndarray
    changepoint: np.ndarray
    source_sha256: str


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.array(array, copy=True, order="C")
    value.setflags(write=False)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error as error:
        raise ValueError(f"cannot identify CSV delimiter: {path}") from error
    return dialect.delimiter


def _header(
    path: Path, delimiter: str, *, labels_required: bool
) -> tuple[str, ...]:
    frame = pd.read_csv(path, sep=delimiter, nrows=0, encoding="utf-8-sig")
    columns = tuple(str(column).strip() for column in frame.columns)
    expected = EXPECTED_COLUMNS if labels_required else (
        (TIME_COLUMN, *SENSOR_COLUMNS),
        EXPECTED_COLUMNS,
    )
    valid = columns == expected if labels_required else columns in expected
    if not valid:
        raise ValueError(
            f"SKAB schema mismatch: expected {expected!r}, observed {columns!r}"
        )
    return columns


def _run_id(path: Path, dataset_root: str | Path) -> str:
    root = Path(dataset_root).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("SKAB file lies outside the bound dataset root") from error
    value = relative.as_posix()
    if value.startswith("../") or not value.endswith(".csv"):
        raise ValueError("invalid SKAB relative path")
    return value


def _timestamps(values: pd.Series) -> tuple[np.ndarray, float]:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or parsed.duplicated().any() or not parsed.is_monotonic_increasing:
        raise ValueError("SKAB timestamps must be finite, unique, and strictly increasing")
    raw = parsed.to_numpy(dtype="datetime64[ns]")
    if raw.size < 2:
        raise ValueError("SKAB run must contain at least two rows")
    delta = np.diff(raw).astype("timedelta64[ns]").astype(np.int64) / 1e9
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("SKAB timestamp increments must be positive")
    return _readonly(raw), float(np.median(delta))


def load_features(path: str | Path, *, dataset_root: str | Path) -> SKABFeatures:
    """Read timestamps and eight sensors without decoding either label column."""
    source = Path(path)
    delimiter = _delimiter(source)
    _header(source, delimiter, labels_required=False)
    usecols = (TIME_COLUMN, *SENSOR_COLUMNS)
    frame = pd.read_csv(
        source, sep=delimiter, usecols=list(usecols), encoding="utf-8-sig"
    )
    # pandas may return file order rather than usecols order; enforce the contract.
    frame = frame.loc[:, list(usecols)]
    timestamps, period = _timestamps(frame[TIME_COLUMN])
    numeric = frame.loc[:, list(SENSOR_COLUMNS)].apply(pd.to_numeric, errors="raise")
    x = numeric.to_numpy(dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(SENSOR_COLUMNS) or not np.isfinite(x).all():
        raise ValueError("SKAB sensors must form a finite N x 8 matrix")
    return SKABFeatures(
        run_id=_run_id(source, dataset_root),
        timestamps=timestamps,
        x=_readonly(x),
        feature_names=SENSOR_COLUMNS,
        sample_period_seconds=period,
        source_sha256=sha256_file(source),
    )


def load_labels(path: str | Path, *, dataset_root: str | Path) -> SKABLabels:
    """Decode labels in a separate, auditable call after score persistence."""
    source = Path(path)
    delimiter = _delimiter(source)
    _header(source, delimiter, labels_required=True)
    frame = pd.read_csv(
        source, sep=delimiter, usecols=list(LABEL_COLUMNS), encoding="utf-8-sig"
    )
    arrays: dict[str, np.ndarray] = {}
    for name in LABEL_COLUMNS:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.int64)
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"SKAB {name} values must be binary")
        arrays[name] = _readonly(values.astype(bool, copy=False))
    return SKABLabels(
        run_id=_run_id(source, dataset_root),
        anomaly=arrays["anomaly"],
        changepoint=arrays["changepoint"],
        source_sha256=sha256_file(source),
    )


def bind_scores_to_source(features: SKABFeatures, score_path: str | Path) -> Mapping[str, Any]:
    """Return immutable metadata used beside a persisted score path."""
    score = Path(score_path)
    if not score.is_file():
        raise FileNotFoundError(score)
    return MappingProxyType(
        {
            "run_id": features.run_id,
            "source_sha256": features.source_sha256,
            "row_count": int(features.x.shape[0]),
            "score_path": str(score.resolve()),
            "score_file_sha256": sha256_file(score),
            "labels_decoded": False,
        }
    )


def validate_score_label_binding(
    metadata: Mapping[str, Any], labels: SKABLabels
) -> None:
    if (
        metadata.get("run_id") != labels.run_id
        or metadata.get("source_sha256") != labels.source_sha256
        or metadata.get("row_count") != int(labels.anomaly.size)
        or labels.anomaly.shape != labels.changepoint.shape
    ):
        raise ValueError("persisted score path and later SKAB labels do not match")


def load_preaccess_charter(path: str | Path) -> Mapping[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    split = document.get("split_rule", {})
    development = tuple(split.get("development_files", ()))
    final = tuple(split.get("final_files", ()))
    strata = split.get("strata", {})
    expected_development: list[str] = []
    expected_final: list[str] = []
    valid_strata = isinstance(strata, dict) and bool(strata)
    if valid_strata:
        for names_value in strata.values():
            if not isinstance(names_value, list) or not all(
                isinstance(name, str) for name in names_value
            ):
                valid_strata = False
                break
            ordered = sorted(
                names_value,
                key=lambda name: hashlib.sha256(
                    (SPLIT_PREFIX + name).encode("utf-8")
                ).hexdigest(),
            )
            expected_development.extend(ordered[0::2])
            expected_final.extend(ordered[1::2])
    if (
        document.get("schema") != "cris_v9_skab_preaccess_charter"
        or document.get("state") != "FROZEN_SPLIT_BEFORE_LOCAL_DOWNLOAD_OR_NUMERIC_ACCESS"
        or not valid_strata
        or len(development) != 18
        or len(final) != 16
        or len(set(development + final)) != 34
        or set(development).intersection(final)
        or development != tuple(expected_development)
        or final != tuple(expected_final)
    ):
        raise ValueError("invalid CRIS-v9 SKAB preaccess charter")
    return MappingProxyType(document)


def assigned_paths(
    dataset_root: str | Path,
    charter: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Path, ...]:
    if role not in {"development", "final"}:
        raise ValueError("role must be development or final")
    names: Sequence[str] = charter["split_rule"][f"{role}_files"]
    root = Path(dataset_root).resolve()
    paths = tuple((root / name).resolve() for name in names)
    for path in paths:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("assigned SKAB path escapes dataset root") from error
    return paths


def partition_bounds(length: int) -> Mapping[str, tuple[int, int]]:
    """Frozen 50/20/15/15 chronological partition of the anomaly-free run."""
    if length < 400:
        raise ValueError("anomaly-free SKAB run is too short for frozen partitions")
    boundaries = (0, int(0.50 * length), int(0.70 * length), int(0.85 * length), length)
    names = (
        "healthy_model_fit",
        "point_score_reference",
        "trajectory_threshold_blocks",
        "healthy_development_audit",
    )
    return MappingProxyType(
        {name: (boundaries[index], boundaries[index + 1]) for index, name in enumerate(names)}
    )


__all__ = [
    "EXPECTED_COLUMNS",
    "LABEL_COLUMNS",
    "SENSOR_COLUMNS",
    "SPLIT_PREFIX",
    "SKABFeatures",
    "SKABLabels",
    "assigned_paths",
    "bind_scores_to_source",
    "load_features",
    "load_labels",
    "load_preaccess_charter",
    "partition_bounds",
    "sha256_file",
    "validate_score_label_binding",
]
