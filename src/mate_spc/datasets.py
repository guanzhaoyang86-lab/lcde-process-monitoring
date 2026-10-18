"""Leakage-aware adapters for the recommended public datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd


RIETH_FILES = {
    "fault_free_training": (
        "TEP_FaultFree_Training.RData",
        "ec126484534331f85001d8c4ebce6d17",
    ),
    "fault_free_testing": (
        "TEP_FaultFree_Testing.RData",
        "38ad9810fc871026157086ae2c2f0ee9",
    ),
    "faulty_testing": (
        "TEP_Faulty_Testing.RData",
        "556bdb64c83021bc0c5f92e427753565",
    ),
}
RIETH_METADATA_COLUMNS = ("faultNumber", "simulationRun", "sample")
RIETH_FEATURE_COLUMNS = tuple(
    [f"xmeas_{index}" for index in range(1, 42)]
    + [f"xmv_{index}" for index in range(1, 12)]
)


@dataclass(frozen=True)
class SensorSequence:
    name: str
    x: np.ndarray
    fault: np.ndarray
    timestamps: np.ndarray | None = None
    observed_mask: np.ndarray | None = None
    covariates: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    covariate_names: tuple[str, ...] = ()
    sample_period_seconds: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


METROPT_FAILURE_INTERVALS = (
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59", "air_leak_high_stress"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00", "air_leak_high_stress"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00", "air_leak_high_stress"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00", "air_leak_high_stress"),
)


def phase_basis(
    n_rows: int,
    degree: int = 3,
    sample_period_seconds: float = 1.0,
    scale_seconds: float = 3600.0,
) -> np.ndarray:
    """Causal polynomial elapsed-time basis, excluding the intercept.

    The basis deliberately does not normalize by the final run length: doing so
    would reveal the future batch termination time during online replay.
    """
    if n_rows < 2 or degree < 0:
        raise ValueError("phase_basis requires at least two rows and nonnegative degree")
    if sample_period_seconds <= 0.0 or scale_seconds <= 0.0:
        raise ValueError("time scales must be positive")
    if degree == 0:
        return np.empty((n_rows, 0), dtype=float)
    elapsed = np.arange(n_rows, dtype=float) * sample_period_seconds / scale_seconds
    return np.column_stack([elapsed**power for power in range(1, degree + 1)])


def with_phase_covariates(
    covariates: np.ndarray | None,
    n_rows: int,
    degree: int = 3,
    sample_period_seconds: float = 1.0,
) -> np.ndarray | None:
    """Append a causal elapsed-time basis to reliable operating covariates."""
    phase = phase_basis(n_rows, degree, sample_period_seconds)
    if covariates is None:
        return phase if phase.shape[1] else None
    values = np.asarray(covariates, dtype=float)
    if values.ndim != 2 or values.shape[0] != n_rows:
        raise ValueError("covariates have an incompatible shape")
    return np.column_stack([values, phase])


def _median_period_seconds(timestamps: pd.Series | pd.DatetimeIndex) -> float:
    values = pd.DatetimeIndex(pd.to_datetime(timestamps))
    differences = np.asarray((values[1:] - values[:-1]).total_seconds(), dtype=float)
    positive = differences[differences > 0]
    return float(np.median(positive)) if positive.size else 1.0


def load_metropt(
    csv_path: str | Path,
    resample: str | None = "1min",
    analog_only: bool = False,
) -> SensorSequence:
    """Load MetroPT-3 and derive labels only from the four official failure reports."""
    path = Path(csv_path)
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    timestamp_column = next(
        (column for column in frame.columns if column.lower() in {"timestamp", "time"}), None
    )
    if timestamp_column is None:
        raise ValueError("MetroPT file has no timestamp column")
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="raise")
    frame = frame.sort_values(timestamp_column).drop_duplicates(timestamp_column, keep="first")

    excluded = {timestamp_column, "index", "Unnamed: 0"}
    all_numeric_columns = [column for column in frame.columns if column not in excluded]
    for column in all_numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    covariate_columns: list[str] = []
    if analog_only:
        preferred = [
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Oil_temperature",
            "Motor_current",
        ]
        numeric_columns = [column for column in preferred if column in frame.columns]
        if len(numeric_columns) != len(preferred):
            raise ValueError("MetroPT analogue columns do not match the official schema")
        covariate_columns = [
            column for column in all_numeric_columns if column not in numeric_columns
        ]
    else:
        numeric_columns = all_numeric_columns

    indexed = frame.set_index(timestamp_column)
    selected = numeric_columns + covariate_columns
    data = indexed[selected]
    covariate_gap_policy: str | None = None
    if resample is not None:
        # Preserve the complete regular time grid.  Empty aggregation bins are
        # real telemetry gaps: dropping them would splice distant observations
        # into a fictitious one-step transition and erase the missingness that
        # MAPIC is intended to handle.
        data = data.resample(resample).median()
    elif data.isna().any().any():
        raise ValueError("MetroPT unexpectedly contains missing numeric values")
    timestamps = pd.DatetimeIndex(data.index)
    fault = np.zeros(data.shape[0], dtype=bool)
    for start, stop, _ in METROPT_FAILURE_INTERVALS:
        fault |= (timestamps >= pd.Timestamp(start)) & (timestamps <= pd.Timestamp(stop))
    sensor_frame = data[numeric_columns]
    observed_mask = sensor_frame.notna().to_numpy(dtype=bool)
    # Masked release values are finite placeholders only; fitting and filtering
    # use observed_mask and therefore never treat these zeros as measurements.
    x = sensor_frame.fillna(0.0).to_numpy(dtype=float)
    if covariate_columns:
        covariate_frame = data[covariate_columns]
        if covariate_frame.isna().any().any():
            # Digital operating signals are optional exogenous inputs and have
            # no mask channel in SensorSequence.  Carry them forward causally
            # across empty bins; the corresponding sensor row remains UNKNOWN.
            covariate_frame = covariate_frame.ffill()
            if covariate_frame.isna().any().any():
                raise ValueError("MetroPT covariates are missing before their first observation")
            covariate_gap_policy = "causal_forward_fill"
        covariates = covariate_frame.to_numpy(dtype=float)
    else:
        covariates = None
    return SensorSequence(
        name="MetroPT-3",
        x=x,
        fault=fault,
        timestamps=timestamps.to_numpy(),
        observed_mask=observed_mask,
        covariates=covariates,
        feature_names=tuple(numeric_columns),
        covariate_names=tuple(covariate_columns),
        sample_period_seconds=_median_period_seconds(timestamps),
        metadata={
            "failure_intervals": METROPT_FAILURE_INTERVALS,
            "source": str(path),
            "resample": resample,
            "covariate_gap_policy": covariate_gap_policy,
        },
    )


def _load_tep_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(Path(path), dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TEP data file is not a matrix")
    if matrix.shape[0] == 52 and matrix.shape[1] != 52:
        matrix = matrix.T
    if matrix.shape[1] != 52:
        raise ValueError(f"expected 52 TEP variables, found shape {matrix.shape}")
    return matrix


def load_tep(
    data_path: str | Path,
    fault_id: int = 0,
    test: bool = True,
    fault_start: int = 160,
    split_manipulated: bool = True,
) -> SensorSequence:
    """Load one classic Tennessee Eastman reference run.

    Testing faults are introduced after the first 160 three-minute samples in the
    standard reference files.  For the normal run, ``fault_id=0`` yields no labels.
    """
    if not 0 <= fault_id <= 21:
        raise ValueError("fault_id must lie between 0 and 21")
    suffix = "_te" if test else ""
    filename = f"d{fault_id:02d}{suffix}.dat"
    path = Path(data_path) / filename
    matrix = _load_tep_matrix(path)
    fault = np.zeros(matrix.shape[0], dtype=bool)
    if fault_id:
        start = fault_start if test else 0
        fault[max(0, min(start, matrix.shape[0])) :] = True
    measurement_names = tuple(f"XMEAS_{i}" for i in range(1, 42))
    manipulated_names = tuple(f"XMV_{i}" for i in range(1, 12))
    x = matrix[:, :41] if split_manipulated else matrix
    covariates = matrix[:, 41:] if split_manipulated else None
    names = measurement_names if split_manipulated else measurement_names + manipulated_names
    return SensorSequence(
        name=f"TEP-fault-{fault_id:02d}{'-test' if test else '-train'}",
        x=x,
        fault=fault,
        observed_mask=np.ones_like(x, dtype=bool),
        covariates=covariates,
        feature_names=names,
        covariate_names=manipulated_names if split_manipulated else (),
        sample_period_seconds=180.0,
        metadata={"fault_id": fault_id, "source": str(path)},
    )


def _file_digest(path: Path, algorithm: str = "md5") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def rieth_frame_to_runs(
    frame: pd.DataFrame,
    partition: str,
    selections: dict[int, tuple[int, ...]],
    *,
    source_name: str,
    source_md5: str,
    fault_start: int = 160,
    split_manipulated: bool = False,
) -> tuple[SensorSequence, ...]:
    """Convert selected Rieth rows into separate, boundary-safe process runs."""
    if partition not in RIETH_FILES:
        raise ValueError(f"unknown Rieth partition: {partition}")
    if (
        isinstance(fault_start, (bool, np.bool_))
        or not isinstance(fault_start, (int, np.integer))
        or fault_start < 0
    ):
        raise ValueError("fault_start must be a nonnegative integer")
    required = set(RIETH_METADATA_COLUMNS + RIETH_FEATURE_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rieth table is missing columns: {missing}")
    if not selections:
        raise ValueError("at least one Rieth fault/run selection is required")

    normalized: dict[int, tuple[int, ...]] = {}
    for raw_fault, raw_runs in selections.items():
        fault_id = int(raw_fault)
        run_ids = tuple(int(value) for value in raw_runs)
        if not 0 <= fault_id <= 20:
            raise ValueError("Rieth fault identifiers must lie in [0, 20]")
        if not run_ids or any(value < 1 for value in run_ids):
            raise ValueError("Rieth simulationRun identifiers must be positive")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError(f"fault {fault_id}: duplicate simulationRun identifiers")
        normalized[fault_id] = run_ids
    if partition.startswith("fault_free") and set(normalized) != {0}:
        raise ValueError("fault-free Rieth partitions only support fault 0")
    if partition == "faulty_testing" and 0 in normalized:
        raise ValueError("faulty Rieth testing requires fault identifiers 1-20")

    fault_values = pd.to_numeric(frame["faultNumber"], errors="raise").to_numpy()
    run_values = pd.to_numeric(frame["simulationRun"], errors="raise").to_numpy()
    selected = np.zeros(frame.shape[0], dtype=bool)
    for fault_id, run_ids in normalized.items():
        selected |= (fault_values == fault_id) & np.isin(run_values, run_ids)
    subset = frame.loc[
        selected, list(RIETH_METADATA_COLUMNS + RIETH_FEATURE_COLUMNS)
    ].copy()
    if subset.empty:
        raise ValueError("the requested Rieth selections contain no rows")
    for column in subset.columns:
        subset[column] = pd.to_numeric(subset[column], errors="raise")
    if not np.isfinite(
        subset[list(RIETH_METADATA_COLUMNS + RIETH_FEATURE_COLUMNS)]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Rieth selected rows contain non-finite values")

    measurement_names = RIETH_FEATURE_COLUMNS[:41]
    manipulated_names = RIETH_FEATURE_COLUMNS[41:]
    sequences: list[SensorSequence] = []
    for fault_id in sorted(normalized):
        for run_id in normalized[fault_id]:
            rows = subset[
                (subset["faultNumber"] == fault_id)
                & (subset["simulationRun"] == run_id)
            ].sort_values("sample", kind="mergesort")
            if rows.empty:
                raise ValueError(
                    f"Rieth partition {partition} lacks fault {fault_id}, run {run_id}"
                )
            samples = rows["sample"].to_numpy(dtype=int)
            expected_samples = np.arange(1, samples.size + 1, dtype=int)
            if not np.array_equal(samples, expected_samples):
                raise ValueError(
                    f"fault {fault_id}, run {run_id}: samples are not exactly 1..n"
                )
            features = rows[list(RIETH_FEATURE_COLUMNS)].to_numpy(dtype=float)
            if split_manipulated:
                x = features[:, :41]
                covariates = features[:, 41:]
                feature_names = measurement_names
                covariate_names = manipulated_names
            else:
                x = features
                covariates = None
                feature_names = RIETH_FEATURE_COLUMNS
                covariate_names = ()
            fault = np.zeros(samples.size, dtype=bool)
            if fault_id:
                fault[min(int(fault_start), samples.size) :] = True
            sequences.append(
                SensorSequence(
                    name=(
                        f"Rieth-{partition}-fault-{fault_id:02d}"
                        f"-run-{run_id:03d}"
                    ),
                    x=x,
                    fault=fault,
                    observed_mask=np.ones_like(x, dtype=bool),
                    covariates=covariates,
                    feature_names=feature_names,
                    covariate_names=covariate_names,
                    sample_period_seconds=180.0,
                    metadata={
                        "dataset": "Rieth et al. multi-run TEP",
                        "dataset_doi": "10.7910/DVN/6C3JR1",
                        "partition": partition,
                        "fault_id": fault_id,
                        "simulation_run": run_id,
                        "source_file": source_name,
                        "source_md5": source_md5,
                        "fault_start_zero_based": int(fault_start),
                    },
                )
            )
    return tuple(sequences)


class RiethTEPArchive:
    """Load selected official Rieth RData runs without joining run boundaries."""

    def __init__(self, folder: str | Path, verify_checksums: bool = True) -> None:
        self.folder = Path(folder)
        self.verify_checksums = bool(verify_checksums)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Rieth data directory not found: {self.folder}")

    def verify(self, partitions: tuple[str, ...]) -> dict[str, str]:
        actual: dict[str, str] = {}
        for partition in partitions:
            if partition not in RIETH_FILES:
                raise ValueError(f"unknown Rieth partition: {partition}")
            filename, expected = RIETH_FILES[partition]
            path = self.folder / filename
            if not path.is_file():
                raise FileNotFoundError(f"Rieth file not found: {path}")
            digest = _file_digest(path)
            if self.verify_checksums and digest != expected:
                raise ValueError(
                    f"{filename}: expected MD5 {expected}, observed {digest}"
                )
            actual[partition] = digest
        return actual

    def load_runs(
        self,
        partition: str,
        selections: dict[int, tuple[int, ...]],
        *,
        fault_start: int = 160,
        split_manipulated: bool = False,
    ) -> tuple[SensorSequence, ...]:
        checksums = self.verify((partition,))
        filename, _ = RIETH_FILES[partition]
        path = self.folder / filename
        try:
            import pyreadr
        except ImportError as error:
            raise ImportError(
                "Rieth RData loading requires 'pip install -e .[rdata]'"
            ) from error
        objects = pyreadr.read_r(str(path))
        if len(objects) != 1:
            raise ValueError(f"{filename}: expected one R object, found {len(objects)}")
        frame = next(iter(objects.values()))
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{filename}: R object is not a data frame")
        return rieth_frame_to_runs(
            frame,
            partition,
            selections,
            source_name=filename,
            source_md5=checksums[partition],
            fault_start=fault_start,
            split_manipulated=split_manipulated,
        )


class BatchDistillationArchive:
    """Read aligned v1.1.2 Batch Distillation modalities directly from ZIP files."""

    def __init__(
        self,
        labels_zip: str | Path,
        sensors_zip: str | Path,
        actuators_zip: str | Path,
        uncertainties_zip: str | Path,
    ) -> None:
        self._archives = {
            "labels": ZipFile(labels_zip),
            "sensors": ZipFile(sensors_zip),
            "actuators": ZipFile(actuators_zip),
            "uncertainties": ZipFile(uncertainties_zip),
        }
        self._indexes = {
            name: self._build_index(archive, ".csv")
            for name, archive in self._archives.items()
        }
        common = set.intersection(*(set(index) for index in self._indexes.values()))
        self.common_keys = tuple(sorted(key for key in common if key.parts[0] == "Operation"))
        if not self.common_keys:
            self.close()
            raise ValueError("the four archives contain no aligned Operation CSV runs")

    @staticmethod
    def _build_index(archive: ZipFile, suffix: str) -> dict[PurePosixPath, str]:
        index: dict[PurePosixPath, str] = {}
        for member in archive.namelist():
            path = PurePosixPath(member)
            if path.suffix.lower() == suffix and len(path.parts) >= 5:
                relative = PurePosixPath(*path.parts[1:])
                index[relative] = member
        return index

    @staticmethod
    def _read_csv(archive: ZipFile, member: str) -> pd.DataFrame:
        with archive.open(member) as stream:
            frame = pd.read_csv(stream)
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    def list_runs(self, contains: str | None = None) -> list[str]:
        keys = [str(key) for key in self.common_keys]
        return keys if contains is None else [key for key in keys if contains in key]

    def load_run(
        self,
        key: str | PurePosixPath,
        restore_natural_missingness: bool = True,
        phase_degree: int = 3,
        actuator_mode: str = "covariates",
    ) -> SensorSequence:
        if actuator_mode not in {"covariates", "features", "exclude"}:
            raise ValueError("actuator_mode must be 'covariates', 'features', or 'exclude'")
        relative = PurePosixPath(key)
        if relative not in self.common_keys:
            raise KeyError(f"run is not shared by all four modalities: {relative}")
        frames = {
            name: self._read_csv(self._archives[name], self._indexes[name][relative])
            for name in self._archives
        }
        for name, frame in frames.items():
            if "Time" not in frame:
                raise ValueError(f"{name} run has no Time column")
            frame.dropna(subset=["Time"], inplace=True)
            frame["Time"] = frame["Time"].astype(str).str.strip()
            frame.drop(frame.index[frame["Time"].isin({"", "nan", "NaN"})], inplace=True)
            if not frame["Time"].is_unique:
                raise ValueError(f"{name} timestamps are not unique")
        base_time = frames["sensors"]["Time"].reset_index(drop=True)
        if not base_time.is_unique:
            raise ValueError("sensor timestamps are not unique")
        for name in ("labels", "actuators", "uncertainties"):
            indexed = frames[name].set_index("Time", drop=False)
            missing_times = base_time[~base_time.isin(indexed.index)]
            if not missing_times.empty:
                raise ValueError(f"{name} is missing {missing_times.size} sensor timestamps")
            frames[name] = indexed.loc[base_time].reset_index(drop=True)
            if not base_time.equals(frames[name]["Time"]):
                raise ValueError(f"{name} timestamps do not align exactly with sensors")

        sensor_columns = [column for column in frames["sensors"] if column != "Time"]
        actuator_columns = [column for column in frames["actuators"] if column != "Time"]
        x_release = frames["sensors"][sensor_columns].apply(pd.to_numeric, errors="raise")
        actuators = frames["actuators"][actuator_columns].apply(pd.to_numeric, errors="raise")
        label_candidates = [column for column in frames["labels"] if column != "Time"]
        if len(label_candidates) != 1:
            raise ValueError(f"expected one label column, found {label_candidates}")
        labels = pd.to_numeric(frames["labels"][label_candidates[0]], errors="raise").astype(int)
        if "Data_interpolated" not in frames["uncertainties"]:
            raise ValueError("uncertainty file lacks Data_interpolated")
        interpolated = pd.to_numeric(
            frames["uncertainties"]["Data_interpolated"], errors="raise"
        ).astype(int).eq(1).to_numpy()
        sensor_values = x_release.to_numpy(dtype=float)
        actuator_values = actuators.to_numpy(dtype=float)
        if actuator_mode == "features":
            released_values = np.column_stack([sensor_values, actuator_values])
            feature_names = tuple(sensor_columns) + tuple(actuator_columns)
            base_covariates = None
            covariate_names: tuple[str, ...] = ()
        else:
            released_values = sensor_values
            feature_names = tuple(sensor_columns)
            base_covariates = actuator_values if actuator_mode == "covariates" else None
            covariate_names = tuple(actuator_columns) if actuator_mode == "covariates" else ()
        observed = np.broadcast_to(~interpolated[:, None], released_values.shape).copy()
        x = released_values.copy()
        if restore_natural_missingness:
            x[~observed] = np.nan

        elapsed = pd.to_timedelta(base_time)
        if not elapsed.is_monotonic_increasing:
            raise ValueError("batch timestamps must be strictly increasing")
        sample_period = _median_period_seconds(pd.Timestamp("2000-01-01") + elapsed)
        if restore_natural_missingness and base_covariates is not None:
            base_covariates = base_covariates.copy()
            base_covariates[interpolated, :] = np.nan
        covariates = with_phase_covariates(
            base_covariates, x.shape[0], phase_degree, sample_period
        )
        phase_names = tuple(f"elapsed_hour_power_{power}" for power in range(1, phase_degree + 1))
        return SensorSequence(
            name=str(relative.with_suffix("")),
            x=x,
            fault=labels.to_numpy() > 0,
            timestamps=elapsed.to_numpy(),
            observed_mask=observed,
            covariates=covariates,
            feature_names=feature_names,
            covariate_names=covariate_names + phase_names,
            sample_period_seconds=sample_period,
            metadata={
                "four_state_labels": labels.to_numpy(),
                "visible_fault": labels.isin([2, 3]).to_numpy(),
                "data_interpolated": interpolated,
                "actuator_readouts_restored_as_missing": restore_natural_missingness,
                "released_values": released_values,
                "natural_missing_values_have_no_ground_truth": True,
                "actuator_mode": actuator_mode,
            },
        )

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()

    def __enter__(self) -> "BatchDistillationArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_skab(csv_path: str | Path) -> SensorSequence:
    """Load one SKAB run with its published anomaly and changepoint fields."""
    path = Path(csv_path)
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"datetime", "anomaly", "changepoint"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SKAB file lacks columns {sorted(required - set(frame.columns))}")
    timestamps = pd.to_datetime(frame["datetime"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("SKAB timestamps must be unique and strictly increasing")
    features = [column for column in frame.columns if column not in required]
    x = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    fault = pd.to_numeric(frame["anomaly"], errors="raise").to_numpy(dtype=int) > 0
    return SensorSequence(
        name=f"SKAB-{path.stem}",
        x=x,
        fault=fault,
        timestamps=timestamps.to_numpy(),
        observed_mask=np.ones_like(x, dtype=bool),
        feature_names=tuple(features),
        sample_period_seconds=_median_period_seconds(timestamps),
        metadata={
            "changepoint": pd.to_numeric(frame["changepoint"], errors="raise").to_numpy(),
            "source": str(path),
        },
    )
