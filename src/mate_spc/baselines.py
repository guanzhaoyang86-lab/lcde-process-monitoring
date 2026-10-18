"""Fair point-imputation baselines calibrated to the same block alarm risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .datasets import SensorSequence
from .imputation import linear_interpolate_offline, locf_impute, median_impute
from .missingness import apply_mask, gap_lengths, generate_observed_mask, scenario_cycle
from .model import (
    MATESPCConfig,
    _concatenate_phase1_runs,
    _prepare_phase1_runs,
)
from .monitor import (
    BlockConformalAlarmCalibrator,
    PosteriorIntegratedMonitor,
    RegimeStandardizer,
)
from .utils import (
    as_2d_float,
    ewma_with_resets,
    segmented_rolling_kth_largest,
    validate_complete,
)


@dataclass(frozen=True)
class BaselineMonitoringResult:
    method: str
    score: np.ndarray
    standardized_score: np.ndarray
    ewma_score: np.ndarray
    persistence_score: np.ndarray
    alarm: np.ndarray
    unknown: np.ndarray
    reconstructed: np.ndarray


def _mask_only_context_features(
    mask: np.ndarray,
    elapsed_start: int = 0,
) -> np.ndarray:
    """Conventional point-method context without posterior covariance access."""
    observed = np.asarray(mask, dtype=bool)
    if observed.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if (
        isinstance(elapsed_start, (bool, np.bool_))
        or not isinstance(elapsed_start, (int, np.integer))
        or elapsed_start < 0
    ):
        raise ValueError("elapsed_start must be a nonnegative integer")
    missing_fraction = (~observed).mean(axis=1)
    longest_gap = gap_lengths(observed).max(axis=1)
    elapsed = int(elapsed_start) + np.arange(observed.shape[0], dtype=float)
    return np.column_stack(
        [missing_fraction, np.log1p(longest_gap), np.log1p(elapsed)]
    )


class CalibratedPointMonitor:
    """Point-completion comparator with its own honest alarm calibration."""

    METHODS = {"median", "locf", "linear_offline", "kalman_mean"}

    def __init__(self, method: str, config: MATESPCConfig | None = None) -> None:
        if method not in self.METHODS:
            raise ValueError(f"unknown baseline {method!r}")
        self.method = method
        self.config = MATESPCConfig() if config is None else config
        self.monitor = PosteriorIntegratedMonitor(
            variance_explained=self.config.variance_explained,
            max_components=self.config.max_components,
            ridge=self.config.ridge,
            covariance_shrinkage=self.config.covariance_shrinkage,
        )
        self.regime = RegimeStandardizer(
            max_depth=self.config.regime_max_depth,
            min_leaf_fraction=self.config.regime_min_leaf_fraction,
            support_feature_count=2,
        )
        self.calibrator = BlockConformalAlarmCalibrator(
            alpha=self.config.alpha,
            block_size=self.config.block_size,
            min_blocks=self.config.min_calibration_blocks,
        )
        self.fitted_ = False

    def _complete(
        self,
        incomplete: np.ndarray,
        mask: np.ndarray,
        covariates: np.ndarray | None,
        elapsed_start: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        plugin = self.monitor.score(
            incomplete,
            observed_mask=mask,
            exogenous=covariates,
            integrate_uncertainty=False,
            elapsed_start=elapsed_start,
        )
        causal_reference = plugin.filter_result.reconstructed
        context_features = _mask_only_context_features(mask, elapsed_start)
        if self.method == "kalman_mean":
            return plugin.fused, context_features, causal_reference, causal_reference
        if self.method == "median":
            completed = median_impute(incomplete, self.reference_, mask)
        elif self.method == "locf":
            completed = locf_impute(incomplete, self.reference_, mask)
        else:
            completed = linear_interpolate_offline(incomplete, self.reference_, mask)
        scored = self.monitor.score(
            completed,
            exogenous=covariates,
            integrate_uncertainty=False,
            elapsed_start=elapsed_start,
        )
        return scored.fused, context_features, completed, causal_reference

    def _decision_statistic(
        self,
        score: np.ndarray,
        features: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        standardized, _, unsupported = self.regime.transform(features, score)
        unknown = unsupported | ~mask.any(axis=1)
        ewma = ewma_with_resets(
            standardized,
            unknown,
            self.config.ewma_lambda,
        )
        persistence = segmented_rolling_kth_largest(
            ewma,
            unknown,
            self.config.persistence_k,
            self.config.persistence_window,
        )
        return standardized, ewma, unknown, persistence

    def fit(
        self,
        phase1_train: np.ndarray,
        phase1_calibration: np.ndarray,
        train_covariates: np.ndarray | None = None,
        calibration_covariates: np.ndarray | None = None,
        train_observed_mask: np.ndarray | None = None,
        calibration_observed_mask: np.ndarray | None = None,
        train_elapsed_start: int = 0,
        calibration_elapsed_start: int = 0,
    ) -> "CalibratedPointMonitor":
        train = validate_complete(phase1_train, "phase1_train")
        calibration = validate_complete(phase1_calibration, "phase1_calibration")
        if train.shape[1] != calibration.shape[1]:
            raise ValueError("training and calibration feature counts differ")
        train_mask = (
            np.ones_like(train, dtype=bool)
            if train_observed_mask is None
            else np.asarray(train_observed_mask, dtype=bool)
        )
        calibration_mask = (
            np.ones_like(calibration, dtype=bool)
            if calibration_observed_mask is None
            else np.asarray(calibration_observed_mask, dtype=bool)
        )
        if train_mask.shape != train.shape or calibration_mask.shape != calibration.shape:
            raise ValueError("Phase-I observed masks must match their data arrays")
        if train_covariates is not None:
            train_covariates = as_2d_float(train_covariates, "train_covariates")
            if train_covariates.shape[0] != train.shape[0]:
                raise ValueError("train_covariates and phase1_train row counts differ")
        if calibration_covariates is not None:
            calibration_covariates = as_2d_float(
                calibration_covariates, "calibration_covariates"
            )
            if calibration_covariates.shape[0] != calibration.shape[0]:
                raise ValueError("calibration_covariates and phase1_calibration row counts differ")
        if (train_covariates is None) != (calibration_covariates is None):
            raise ValueError("provide covariates for both training and calibration, or neither")
        complete_train_rows = train_mask.all(axis=1)
        # Point-imputation fallbacks are estimated only from genuinely observed
        # complete Phase-I rows, never from values released at masked timestamps.
        self.reference_ = train[complete_train_rows].copy()
        self.monitor.fit(
            train,
            train_covariates,
            train_mask,
            integrate_uncertainty=False,
        )
        design, alarm_cal = train, calibration
        design_base_mask, alarm_base_mask = train_mask, calibration_mask
        design_u, alarm_u = train_covariates, calibration_covariates
        configs = scenario_cycle(self.config.scenarios, self.config.calibration_repeats)

        design_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        base_design = apply_mask(design, design_base_mask)
        clean_score, clean_features, _, design_reference = self._complete(
            base_design, design_base_mask, design_u, train_elapsed_start
        )
        base_design_valid = design_base_mask.any(axis=1)
        design_rows.append(
            (
                "base_observed",
                clean_score[base_design_valid],
                clean_features[base_design_valid],
            )
        )
        for index, scenario in enumerate(configs):
            mask = design_base_mask & generate_observed_mask(
                design_reference, scenario, self.config.seed + 1009 * (index + 1)
            )
            score, features, _, _ = self._complete(
                apply_mask(design, mask), mask, design_u, train_elapsed_start
            )
            valid = mask.any(axis=1)
            design_rows.append(
                (f"base+{scenario.identifier()}", score[valid], features[valid])
            )
        mechanism_counts: dict[str, int] = {}
        for name, _, _ in design_rows:
            mechanism_counts[name] = mechanism_counts.get(name, 0) + 1
        x_context = np.vstack([features for _, _, features in design_rows])
        y_score = np.concatenate([score for _, score, _ in design_rows])
        weights = np.concatenate(
            [
                np.full(score.size, 1.0 / (mechanism_counts[name] * score.size))
                for name, score, _ in design_rows
            ]
        )
        self.regime.fit(x_context, y_score, weights)

        alarm_rows: list[tuple[str, np.ndarray]] = []
        base_alarm = apply_mask(alarm_cal, alarm_base_mask)
        clean_score, clean_features, _, alarm_reference = self._complete(
            base_alarm, alarm_base_mask, alarm_u, calibration_elapsed_start
        )
        _, _, _, clean_persistence = self._decision_statistic(
            clean_score, clean_features, alarm_base_mask
        )
        alarm_rows.append(
            (
                "base_observed",
                clean_persistence,
            )
        )
        for index, scenario in enumerate(configs):
            mask = alarm_base_mask & generate_observed_mask(
                alarm_reference,
                scenario,
                self.config.seed + 1000003 + 1013 * (index + 1),
            )
            score, features, _, _ = self._complete(
                apply_mask(alarm_cal, mask),
                mask,
                alarm_u,
                calibration_elapsed_start,
            )
            _, _, _, persistence = self._decision_statistic(score, features, mask)
            alarm_rows.append(
                (
                    f"base+{scenario.identifier()}",
                    persistence,
                )
            )
        self.calibrator.fit(alarm_rows)
        self.n_features_in_ = train.shape[1]
        self.fitted_ = True
        return self

    def fit_runs(
        self,
        phase1_train_runs: Iterable[SensorSequence],
        phase1_calibration_runs: Iterable[SensorSequence],
    ) -> "CalibratedPointMonitor":
        """Fit the point baseline without joining independent run dynamics."""
        train_runs = _prepare_phase1_runs(
            phase1_train_runs, "baseline training"
        )
        calibration_runs = _prepare_phase1_runs(
            phase1_calibration_runs, "baseline calibration"
        )
        first_train = train_runs[0]
        first_calibration = calibration_runs[0]
        if first_train.x.shape[1] != first_calibration.x.shape[1]:
            raise ValueError("training and calibration feature counts differ")
        if (
            first_train.feature_names
            and first_calibration.feature_names
            and first_train.feature_names != first_calibration.feature_names
        ):
            raise ValueError("training and calibration feature schemas differ")
        if (first_train.covariates is None) != (
            first_calibration.covariates is None
        ):
            raise ValueError(
                "training and calibration covariate usage differs"
            )

        train, train_mask, train_u, starts = _concatenate_phase1_runs(
            train_runs
        )
        self.reference_ = train[train_mask.all(axis=1)].copy()
        self.monitor.fit(
            train,
            train_u,
            train_mask,
            integrate_uncertainty=False,
            run_start_mask=starts,
        )
        configs = scenario_cycle(
            self.config.scenarios, self.config.calibration_repeats
        )
        design_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for run_index, run in enumerate(train_runs):
            clean_score, clean_features, _, reference = self._complete(
                apply_mask(run.x, run.mask),
                run.mask,
                run.covariates,
            )
            valid = run.mask.any(axis=1)
            design_rows.append(
                (
                    "base_observed",
                    clean_score[valid],
                    clean_features[valid],
                )
            )
            for index, scenario in enumerate(configs):
                seed = (
                    self.config.seed
                    + 10000019 * run_index
                    + 1009 * (index + 1)
                )
                mask = run.mask & generate_observed_mask(
                    reference, scenario, seed
                )
                score, features, _, _ = self._complete(
                    apply_mask(run.x, mask),
                    mask,
                    run.covariates,
                )
                valid = mask.any(axis=1)
                design_rows.append(
                    (
                        f"base+{scenario.identifier()}",
                        score[valid],
                        features[valid],
                    )
                )
        mechanism_counts: dict[str, int] = {}
        for name, _, _ in design_rows:
            mechanism_counts[name] = mechanism_counts.get(name, 0) + 1
        x_context = np.vstack(
            [features for _, _, features in design_rows]
        )
        y_score = np.concatenate([score for _, score, _ in design_rows])
        weights = np.concatenate(
            [
                np.full(
                    score.size,
                    1.0 / (mechanism_counts[name] * score.size),
                )
                for name, score, _ in design_rows
            ]
        )
        self.regime.fit(x_context, y_score, weights)

        calibration_records: list[tuple[str, str, np.ndarray]] = []
        for run_index, run in enumerate(calibration_runs):
            clean_score, clean_features, _, reference = self._complete(
                apply_mask(run.x, run.mask),
                run.mask,
                run.covariates,
            )
            _, _, _, persistence = self._decision_statistic(
                clean_score, clean_features, run.mask
            )
            calibration_records.append(
                ("base_observed", run.name, persistence)
            )
            for index, scenario in enumerate(configs):
                seed = (
                    self.config.seed
                    + 1000003
                    + 10000019 * run_index
                    + 1013 * (index + 1)
                )
                mask = run.mask & generate_observed_mask(
                    reference, scenario, seed
                )
                score, features, _, _ = self._complete(
                    apply_mask(run.x, mask),
                    mask,
                    run.covariates,
                )
                _, _, _, persistence = self._decision_statistic(
                    score, features, mask
                )
                calibration_records.append(
                    (
                        f"base+{scenario.identifier()}",
                        run.name,
                        persistence,
                    )
                )
        self.calibrator.fit_run_clusters(calibration_records)
        self.n_features_in_ = train.shape[1]
        self.phase1_design_run_count_ = len(train_runs)
        self.phase1_calibration_run_count_ = len(calibration_runs)
        self.fitted_ = True
        return self

    def predict(
        self,
        x: np.ndarray,
        observed_mask: np.ndarray | None = None,
        covariates: np.ndarray | None = None,
        elapsed_start: int = 0,
    ) -> BaselineMonitoringResult:
        if not self.fitted_:
            raise RuntimeError("CalibratedPointMonitor has not been fitted")
        arr = as_2d_float(x, "x")
        mask = np.isfinite(arr) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("observed_mask must match x")
        score, features, reconstructed, _ = self._complete(
            arr, mask, covariates, elapsed_start
        )
        standardized, ewma, unknown, persistence = self._decision_statistic(
            score, features, mask
        )
        alarm = (persistence > self.calibrator.threshold_) & ~unknown
        return BaselineMonitoringResult(
            method=self.method,
            score=score,
            standardized_score=standardized,
            ewma_score=ewma,
            persistence_score=persistence,
            alarm=alarm,
            unknown=unknown,
            reconstructed=reconstructed,
        )

    @property
    def calibration_summary_(self) -> dict[str, object]:
        if not self.fitted_:
            raise RuntimeError("CalibratedPointMonitor has not been fitted")
        return {
            "threshold": self.calibrator.threshold_,
            "mechanism_thresholds": self.calibrator.mechanism_thresholds_,
            "block_counts": self.calibrator.block_counts_,
            "augmentation_counts": self.calibrator.augmentation_counts_,
            "base_block_count": self.calibrator.base_block_count_,
            "base_run_count": self.calibrator.base_run_count_,
            "calibration_unit": self.calibrator.calibration_unit_,
            "training_run_count": self.monitor.filter_model.training_run_count_,
            "regime_leaf_counts": self.regime.leaf_counts_.tolist(),
            "regime_centers": self.regime.centers_.tolist(),
            "regime_scales": self.regime.scales_.tolist(),
            "n_components": self.monitor.n_components_,
            "component_scales": self.monitor.component_scales_.tolist(),
            "quality_context": "mask_fraction_gap_and_elapsed_only",
        }
