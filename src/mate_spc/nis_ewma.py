"""Frozen complete-input normalized-innovation EWMA monitor.

The input statistic is the existing per-observed-dimension normalized
innovation squared produced by :class:`LinearGaussianFilter`.  This module is
deliberately independent of fault labels: normalization uses healthy design
runs and the alarm threshold uses independent healthy calibration runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .monitor import BlockConformalAlarmCalibrator


@dataclass(frozen=True)
class NISEWMAResult:
    """One complete run's frozen NIS-EWMA decision path."""

    normalized_innovation: np.ndarray
    transformed: np.ndarray
    standardized: np.ndarray
    ewma: np.ndarray
    persistence: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    threshold: float | None


@dataclass(frozen=True)
class NISEWMACalibrationSummary:
    """Auditable design normalization and run-conformal calibration summary."""

    design_run_count: int
    healthy_run_count: int
    center: float
    scale: float
    alpha: float
    conformal_rank_one_based: int
    threshold: float
    order_statistics: tuple[tuple[int, float], ...]
    rank_driver_run: str
    rank_driver_score: float
    run_maxima: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""
        return {
            "method": "complete_normalized_NIS_log_reflected_EWMA_3of3",
            "design_run_count": self.design_run_count,
            "healthy_run_count": self.healthy_run_count,
            "t0_excluded": True,
            "transformation": "log1p(max(normalized_innovation, 0))",
            "center": self.center,
            "scale": self.scale,
            "scale_estimator": "1.4826*MAD with sample-SD fallback",
            "ewma_lambda": NISEWMAMonitor.EWMA_LAMBDA,
            "ewma_definition": "max(0, lambda*z_t + (1-lambda)*E_previous)",
            "ewma_initial_state": 0.0,
            "persistence": "minimum of the latest 3 EWMA values",
            "alpha": self.alpha,
            "calibration_unit": "healthy_run_maximum",
            "conformal_rank_one_based": self.conformal_rank_one_based,
            "threshold": self.threshold,
            "alarm_comparison": "strictly_greater_than",
            "order_statistics": {
                str(rank): value for rank, value in self.order_statistics
            },
            "rank_driver": {
                "run": self.rank_driver_run,
                "score": self.rank_driver_score,
            },
            "run_maxima": [
                {"run": name, "maximum": value}
                for name, value in self.run_maxima
            ],
        }


def _validated_runs(
    runs: Iterable[tuple[str, np.ndarray]],
    *,
    label: str,
    required_count: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    records: list[tuple[str, np.ndarray]] = []
    for item in runs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{label} entries must be (run_name, nis_vector) tuples")
        name, values = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} run names must be nonempty strings")
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{label} NIS for {name!r} must be one-dimensional")
        if array.size < 4:
            raise ValueError(
                f"{label} NIS for {name!r} needs t0 plus at least three samples"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{label} NIS for {name!r} must be finite")
        records.append((name, array.copy()))
    if len(records) != required_count:
        raise ValueError(
            f"{label} requires exactly {required_count} independent runs; "
            f"received {len(records)}"
        )
    names = [name for name, _ in records]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} run names must be unique")
    return tuple(records)


_NONNEGATIVE_TOLERANCE = 1e-12


def _transform(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(array < -_NONNEGATIVE_TOLERANCE):
        raise ValueError(
            "normalized_innovation cannot contain values below -1e-12"
        )
    return np.log1p(np.maximum(array, 0.0))


class NISEWMAMonitor:
    """Frozen complete-input NIS detector with honest run-level calibration."""

    DESIGN_RUN_COUNT = 20
    CALIBRATION_RUN_COUNT = 100
    ALPHA = 0.05
    EWMA_LAMBDA = 0.2
    PERSISTENCE_WINDOW = 3

    def __init__(self) -> None:
        self.design_fitted_ = False
        self.fitted_ = False

    def fit_design(
        self,
        design_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "NISEWMAMonitor":
        """Freeze center and scale from 20 healthy design runs, excluding t0."""
        records = _validated_runs(
            design_runs,
            label="NIS design",
            required_count=self.DESIGN_RUN_COUNT,
        )
        pooled = np.concatenate([_transform(values)[1:] for _, values in records])
        center = float(np.median(pooled))
        mad = float(np.median(np.abs(pooled - center)))
        scale = 1.4826 * mad
        self.scale_source_ = "mad"
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(pooled, ddof=1))
            self.scale_source_ = "sample_sd"
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "NIS design scale is degenerate after both MAD and sample-SD "
                "estimation"
            )
        self.center_ = center
        self.scale_ = scale
        self.design_run_names_ = tuple(name for name, _ in records)
        self.design_fitted_ = True
        self.fitted_ = False
        return self

    def _check_design_fitted(self) -> None:
        if not self.design_fitted_:
            raise RuntimeError("NISEWMAMonitor design normalization is not fitted")

    def decision_path(
        self,
        normalized_innovation: np.ndarray,
        *,
        threshold: float | None = None,
    ) -> NISEWMAResult:
        """Compute the frozen one-sided EWMA and 3/3 persistence statistic."""
        self._check_design_fitted()
        nis = np.asarray(normalized_innovation, dtype=float)
        if nis.ndim != 1 or nis.size < 4:
            raise ValueError("normalized_innovation must be a vector of length at least 4")
        if not np.isfinite(nis).all():
            raise ValueError("normalized_innovation must be finite")
        if threshold is not None and (
            not np.isfinite(threshold) or isinstance(threshold, (bool, np.bool_))
        ):
            raise ValueError("threshold must be finite when supplied")

        transformed = _transform(nis)
        standardized = (transformed - self.center_) / self.scale_
        ewma = np.full(nis.size, -np.inf, dtype=float)
        previous = 0.0
        smoothing = self.EWMA_LAMBDA
        # t0 is excluded by protocol.  Each call is one independent run, so
        # the reflected EWMA state always starts from zero here.
        for index in range(1, nis.size):
            previous = max(
                0.0,
                smoothing * float(standardized[index])
                + (1.0 - smoothing) * previous,
            )
            ewma[index] = previous

        persistence = np.full(nis.size, -np.inf, dtype=float)
        window = self.PERSISTENCE_WINDOW
        for index in range(window, nis.size):
            persistence[index] = float(np.min(ewma[index - window + 1 : index + 1]))
        eligible = np.zeros(nis.size, dtype=bool)
        eligible[window:] = True
        alarm = (
            np.zeros(nis.size, dtype=bool)
            if threshold is None
            else eligible & (persistence > float(threshold))
        )
        return NISEWMAResult(
            normalized_innovation=nis.copy(),
            transformed=transformed,
            standardized=standardized,
            ewma=ewma,
            persistence=persistence,
            eligible=eligible,
            alarm=alarm,
            threshold=None if threshold is None else float(threshold),
        )

    def calibrate(
        self,
        calibration_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "NISEWMAMonitor":
        """Calibrate on maxima from exactly 100 independent healthy runs."""
        self._check_design_fitted()
        records = _validated_runs(
            calibration_runs,
            label="NIS calibration",
            required_count=self.CALIBRATION_RUN_COUNT,
        )
        persistence_records: list[tuple[str, str, np.ndarray]] = []
        run_maxima: list[tuple[str, float]] = []
        for name, values in records:
            persistence = self.decision_path(values).persistence
            maximum = float(np.max(persistence))
            if not np.isfinite(maximum):
                raise ValueError(f"NIS calibration run {name!r} has no finite maximum")
            persistence_records.append(("base_observed", name, persistence))
            run_maxima.append((name, maximum))

        calibrator = BlockConformalAlarmCalibrator(
            alpha=self.ALPHA,
            block_size=self.PERSISTENCE_WINDOW,
            min_blocks=self.CALIBRATION_RUN_COUNT,
        )
        calibrator.fit_run_clusters(persistence_records)
        threshold = float(calibrator.threshold_)
        if not np.isfinite(threshold):
            raise RuntimeError("100-run NIS conformal threshold must be finite")

        rank = int(
            np.ceil((self.CALIBRATION_RUN_COUNT + 1) * (1.0 - self.ALPHA))
        )
        ordered = sorted(run_maxima, key=lambda item: (item[1], item[0]))
        driver_name, driver_score = ordered[rank - 1]
        if not np.isclose(driver_score, threshold, rtol=0.0, atol=1e-12):
            raise RuntimeError("NIS conformal rank and calibrator threshold disagree")
        selected_ranks = tuple(
            sorted({1, 90, 95, rank, 99, self.CALIBRATION_RUN_COUNT})
        )
        order_statistics = tuple(
            (position, float(ordered[position - 1][1]))
            for position in selected_ranks
        )
        self.calibrator_ = calibrator
        self.threshold_ = threshold
        self.calibration_run_names_ = tuple(name for name, _ in records)
        self.summary_ = NISEWMACalibrationSummary(
            design_run_count=self.DESIGN_RUN_COUNT,
            healthy_run_count=self.CALIBRATION_RUN_COUNT,
            center=self.center_,
            scale=self.scale_,
            alpha=self.ALPHA,
            conformal_rank_one_based=rank,
            threshold=threshold,
            order_statistics=order_statistics,
            rank_driver_run=driver_name,
            rank_driver_score=float(driver_score),
            run_maxima=tuple(run_maxima),
        )
        self.fitted_ = True
        return self

    def fit(
        self,
        design_runs: Iterable[tuple[str, np.ndarray]],
        calibration_runs: Iterable[tuple[str, np.ndarray]],
    ) -> "NISEWMAMonitor":
        """Fit healthy design normalization followed by healthy calibration."""
        return self.fit_design(design_runs).calibrate(calibration_runs)

    def predict(self, normalized_innovation: np.ndarray) -> NISEWMAResult:
        """Return the calibrated alarm path for one independent complete run."""
        if not self.fitted_:
            raise RuntimeError("NISEWMAMonitor has not been calibrated")
        return self.decision_path(
            normalized_innovation,
            threshold=self.threshold_,
        )

    @property
    def calibration_summary_(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("NISEWMAMonitor has not been calibrated")
        summary = self.summary_.as_dict()
        summary["scale_estimator"] = self.scale_source_
        return summary


__all__ = [
    "NISEWMACalibrationSummary",
    "NISEWMAMonitor",
    "NISEWMAResult",
]
