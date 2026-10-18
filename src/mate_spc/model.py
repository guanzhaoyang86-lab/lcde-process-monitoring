"""End-to-end uncertainty-decoupled MAPIC estimator.

UD-MAPIC combines a causal linear-Gaussian sensor model, posterior-location
quadratic evidence, explicit data-quality uncertainty, mask-adaptive robust
normalization, and clustered block-conformal control limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

from .datasets import SensorSequence
from .missingness import (
    DEFAULT_SCENARIOS,
    MissingnessConfig,
    apply_mask,
    generate_observed_mask,
    scenario_cycle,
)
from .monitor import (
    BlockConformalAlarmCalibrator,
    PosteriorIntegratedMonitor,
    RegimeStandardizer,
    ScoreResult,
)
from .utils import (
    as_2d_float,
    ewma_with_resets,
    segmented_rolling_kth_largest,
    validate_complete,
)


@dataclass(frozen=True)
class MATESPCConfig:
    """Configuration retained under the historical package name ``mate_spc``."""

    variance_explained: float = 0.9
    max_components: int | None = None
    ridge: float = 1e-3
    covariance_shrinkage: float = 0.1
    ewma_lambda: float = 0.2
    alpha: float = 0.05
    block_size: int = 60
    min_calibration_blocks: int = 8
    calibration_repeats: int = 3
    persistence_k: int = 3
    persistence_window: int = 3
    regime_max_depth: int = 3
    regime_min_leaf_fraction: float = 0.03
    seed: int = 20260907
    scenarios: tuple[MissingnessConfig, ...] = field(default_factory=lambda: DEFAULT_SCENARIOS)
    # False is the UD-MAPIC primary analysis.  True is the independently fitted
    # posterior-expectation ablation M + tr(A P).
    integrate_uncertainty: bool = False
    use_posterior_quality_context: bool = True


@dataclass(frozen=True)
class MonitoringResult:
    score: np.ndarray
    standardized_score: np.ndarray
    ewma_score: np.ndarray
    persistence_score: np.ndarray
    threshold: np.ndarray
    instantaneous_reference_threshold: np.ndarray
    raw_alarm: np.ndarray
    alarm: np.ndarray
    unknown: np.ndarray
    status: np.ndarray
    context: np.ndarray
    posterior_trace: np.ndarray
    location_energy: np.ndarray
    uncertainty_energy: np.ndarray
    uncertainty_fraction: np.ndarray
    reconstructed: np.ndarray
    component_t2: np.ndarray
    component_spe: np.ndarray
    component_t2_uncertainty: np.ndarray
    component_spe_uncertainty: np.ndarray
    component_innovation: np.ndarray


@dataclass(frozen=True)
class _PreparedPhaseIRun:
    name: str
    x: np.ndarray
    mask: np.ndarray
    covariates: np.ndarray | None
    feature_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    sample_period_seconds: float


def _prepare_phase1_runs(
    sequences: Iterable[SensorSequence],
    role: str,
) -> tuple[_PreparedPhaseIRun, ...]:
    runs: list[_PreparedPhaseIRun] = []
    for sequence in sequences:
        x = validate_complete(sequence.x, f"{role} run {sequence.name}")
        if np.asarray(sequence.fault, dtype=bool).any():
            raise ValueError(f"{role} run {sequence.name} is not healthy")
        mask = (
            np.ones_like(x, dtype=bool)
            if sequence.observed_mask is None
            else np.asarray(sequence.observed_mask, dtype=bool)
        )
        if mask.shape != x.shape:
            raise ValueError(f"{role} run {sequence.name} has an invalid mask")
        if sequence.covariates is None:
            covariates = None
        else:
            covariates = as_2d_float(
                sequence.covariates,
                f"{role} covariates {sequence.name}",
            )
            if covariates.shape[0] != x.shape[0] or not np.isfinite(
                covariates
            ).all():
                raise ValueError(
                    f"{role} run {sequence.name} has invalid covariates"
                )
        period = float(sequence.sample_period_seconds)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(f"{role} run {sequence.name} has invalid timing")
        runs.append(
            _PreparedPhaseIRun(
                name=str(sequence.name),
                x=x,
                mask=mask,
                covariates=covariates,
                feature_names=tuple(sequence.feature_names),
                covariate_names=tuple(sequence.covariate_names),
                sample_period_seconds=period,
            )
        )
    if not runs:
        raise ValueError(f"at least one {role} run is required")
    names = [run.name for run in runs]
    if len(names) != len(set(names)):
        raise ValueError(f"{role} run names must be unique")
    feature_counts = {run.x.shape[1] for run in runs}
    feature_schemas = {run.feature_names for run in runs if run.feature_names}
    covariate_schemas = {
        run.covariate_names for run in runs if run.covariate_names
    }
    periods = {run.sample_period_seconds for run in runs}
    uses_covariates = {run.covariates is not None for run in runs}
    if len(feature_counts) != 1 or len(feature_schemas) > 1:
        raise ValueError(f"{role} runs have incompatible feature schemas")
    if len(covariate_schemas) > 1 or len(uses_covariates) != 1:
        raise ValueError(f"{role} runs have incompatible covariate schemas")
    if len(periods) != 1:
        raise ValueError(f"{role} runs have incompatible sample periods")
    return tuple(runs)


def _concatenate_phase1_runs(
    runs: tuple[_PreparedPhaseIRun, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    x = np.vstack([run.x for run in runs])
    mask = np.vstack([run.mask for run in runs])
    covariates = (
        None
        if runs[0].covariates is None
        else np.vstack(
            [
                run.covariates
                for run in runs
                if run.covariates is not None
            ]
        )
    )
    starts = np.zeros(x.shape[0], dtype=bool)
    offset = 0
    for run in runs:
        starts[offset] = True
        offset += run.x.shape[0]
    return x, mask, covariates, starts


class MATESPC:
    """Missingness-aware, uncertainty-decoupled process monitor.

    ``fit`` accepts healthy Phase-I releases and explicit observed masks.  Fault
    labels are never used for representation learning, filtering, calibration,
    or threshold tuning.
    """

    def __init__(self, config: MATESPCConfig | None = None) -> None:
        self.config = MATESPCConfig() if config is None else config
        if not 1 <= self.config.persistence_k <= self.config.persistence_window:
            raise ValueError("persistence settings require 1 <= k <= window")
        if self.config.block_size < self.config.persistence_window:
            raise ValueError("block_size must be at least persistence_window")
        self.monitor = PosteriorIntegratedMonitor(
            variance_explained=self.config.variance_explained,
            max_components=self.config.max_components,
            ridge=self.config.ridge,
            covariance_shrinkage=self.config.covariance_shrinkage,
        )
        self.regime = RegimeStandardizer(
            max_depth=self.config.regime_max_depth,
            min_leaf_fraction=self.config.regime_min_leaf_fraction,
            support_feature_count=(
                4 if self.config.use_posterior_quality_context else 2
            ),
        )
        self.calibrator = BlockConformalAlarmCalibrator(
            alpha=self.config.alpha,
            block_size=self.config.block_size,
            min_blocks=self.config.min_calibration_blocks,
        )
        self.fitted_ = False

    def _quality_features(self, scored: ScoreResult) -> np.ndarray:
        if self.config.use_posterior_quality_context:
            return scored.context_features
        # Conventional mask-only state: fraction, active gap, and elapsed index.
        return scored.context_features[:, [0, 1, 4]]

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
    ) -> "MATESPC":
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
        self.monitor.fit(
            train,
            train_covariates,
            train_mask,
            integrate_uncertainty=self.config.integrate_uncertainty,
        )
        # All adaptive modeling (including the regime tree) belongs to the
        # training split.  The separate calibration run is reserved in full for
        # the final conformal order statistic.
        design = train
        alarm_calibration = calibration
        design_base_mask = train_mask
        alarm_base_mask = calibration_mask
        design_covariates = train_covariates
        alarm_covariates = calibration_covariates
        configs = scenario_cycle(self.config.scenarios, self.config.calibration_repeats)

        base_design_score = self.monitor.score(
            design,
            observed_mask=design_base_mask,
            exogenous=design_covariates,
            integrate_uncertainty=self.config.integrate_uncertainty,
            elapsed_start=train_elapsed_start,
        )
        design_reference = base_design_score.filter_result.reconstructed
        base_design_valid = base_design_score.filter_result.observed_mask.any(axis=1)
        base_design_context = self._quality_features(base_design_score)
        # Keep only the compact arrays needed by the regime fit.  A ScoreResult
        # also owns the full n x d x d posterior covariance tensor, so retaining
        # every augmented run here can otherwise require several gigabytes for
        # realistic multivariate batches.
        design_rows: list[tuple[str, np.ndarray, np.ndarray]] = [
            (
                "base_observed",
                base_design_score.fused[base_design_valid],
                base_design_context[base_design_valid],
            )
        ]
        del base_design_score
        for index, scenario in enumerate(configs):
            mask = design_base_mask & generate_observed_mask(
                design_reference,
                scenario,
                seed=self.config.seed + 1009 * (index + 1),
            )
            incomplete = apply_mask(design, mask)
            scored = self.monitor.score(
                incomplete,
                mask,
                exogenous=design_covariates,
                integrate_uncertainty=self.config.integrate_uncertainty,
                elapsed_start=train_elapsed_start,
            )
            design_rows.append(
                (
                    f"base+{scenario.identifier()}",
                    scored.fused[mask.any(axis=1)],
                    self._quality_features(scored)[mask.any(axis=1)],
                )
            )
            del scored

        mechanism_counts: dict[str, int] = {}
        for name, _, _ in design_rows:
            mechanism_counts[name] = mechanism_counts.get(name, 0) + 1
        features = np.vstack([context for _, _, context in design_rows])
        scores = np.concatenate([score for _, score, _ in design_rows])
        weights = np.concatenate(
            [
                np.full(score.size, 1.0 / (mechanism_counts[name] * score.size))
                for name, score, _ in design_rows
            ]
        )
        self.regime.fit(features, scores, sample_weight=weights)

        base_alarm_score = self.monitor.score(
            alarm_calibration,
            observed_mask=alarm_base_mask,
            exogenous=alarm_covariates,
            integrate_uncertainty=self.config.integrate_uncertainty,
            elapsed_start=calibration_elapsed_start,
        )
        alarm_reference = base_alarm_score.filter_result.reconstructed
        _, _, _, _, base_persistence = self._decision_statistic(base_alarm_score)
        persistence_runs: list[tuple[str, np.ndarray]] = [
            ("base_observed", base_persistence)
        ]
        del base_alarm_score
        for index, scenario in enumerate(configs):
            mask = alarm_base_mask & generate_observed_mask(
                alarm_reference,
                scenario,
                seed=self.config.seed + 1000003 + 1013 * (index + 1),
            )
            incomplete = apply_mask(alarm_calibration, mask)
            scored = self.monitor.score(
                incomplete,
                mask,
                exogenous=alarm_covariates,
                integrate_uncertainty=self.config.integrate_uncertainty,
                elapsed_start=calibration_elapsed_start,
            )
            _, _, _, _, persistence = self._decision_statistic(scored)
            persistence_runs.append(
                (f"base+{scenario.identifier()}", persistence)
            )
            del scored
        self.calibrator.fit(persistence_runs)
        self.n_features_in_ = train.shape[1]
        self.fitted_ = True
        return self

    def fit_runs(
        self,
        phase1_train_runs: Iterable[SensorSequence],
        phase1_calibration_runs: Iterable[SensorSequence],
    ) -> "MATESPC":
        """Fit on independent healthy runs without creating false transitions."""
        train_runs = _prepare_phase1_runs(phase1_train_runs, "training")
        calibration_runs = _prepare_phase1_runs(
            phase1_calibration_runs, "calibration"
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
        if (
            first_train.covariate_names
            and first_calibration.covariate_names
            and first_train.covariate_names
            != first_calibration.covariate_names
        ):
            raise ValueError("training and calibration covariate schemas differ")
        if (first_train.covariates is None) != (
            first_calibration.covariates is None
        ):
            raise ValueError(
                "training and calibration covariate usage differs"
            )
        if not np.isclose(
            first_train.sample_period_seconds,
            first_calibration.sample_period_seconds,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("training and calibration sample periods differ")

        train, train_mask, train_u, starts = _concatenate_phase1_runs(
            train_runs
        )
        self.monitor.fit(
            train,
            train_u,
            train_mask,
            integrate_uncertainty=self.config.integrate_uncertainty,
            run_start_mask=starts,
        )
        configs = scenario_cycle(
            self.config.scenarios, self.config.calibration_repeats
        )
        design_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for run_index, run in enumerate(train_runs):
            base_score = self.monitor.score(
                run.x,
                observed_mask=run.mask,
                exogenous=run.covariates,
                integrate_uncertainty=self.config.integrate_uncertainty,
            )
            reference = base_score.filter_result.reconstructed
            valid = base_score.filter_result.observed_mask.any(axis=1)
            design_rows.append(
                (
                    "base_observed",
                    base_score.fused[valid],
                    self._quality_features(base_score)[valid],
                )
            )
            del base_score
            for index, scenario in enumerate(configs):
                seed = (
                    self.config.seed
                    + 10000019 * run_index
                    + 1009 * (index + 1)
                )
                mask = run.mask & generate_observed_mask(
                    reference, scenario, seed=seed
                )
                scored = self.monitor.score(
                    apply_mask(run.x, mask),
                    mask,
                    exogenous=run.covariates,
                    integrate_uncertainty=self.config.integrate_uncertainty,
                )
                valid = mask.any(axis=1)
                design_rows.append(
                    (
                        f"base+{scenario.identifier()}",
                        scored.fused[valid],
                        self._quality_features(scored)[valid],
                    )
                )
                del scored

        mechanism_counts: dict[str, int] = {}
        for name, _, _ in design_rows:
            mechanism_counts[name] = mechanism_counts.get(name, 0) + 1
        features = np.vstack([context for _, _, context in design_rows])
        scores = np.concatenate([score for _, score, _ in design_rows])
        weights = np.concatenate(
            [
                np.full(
                    score.size,
                    1.0 / (mechanism_counts[name] * score.size),
                )
                for name, score, _ in design_rows
            ]
        )
        self.regime.fit(features, scores, sample_weight=weights)

        persistence_runs: list[tuple[str, str, np.ndarray]] = []
        for run_index, run in enumerate(calibration_runs):
            base_score = self.monitor.score(
                run.x,
                observed_mask=run.mask,
                exogenous=run.covariates,
                integrate_uncertainty=self.config.integrate_uncertainty,
            )
            reference = base_score.filter_result.reconstructed
            _, _, _, _, persistence = self._decision_statistic(base_score)
            persistence_runs.append(
                ("base_observed", run.name, persistence)
            )
            del base_score
            for index, scenario in enumerate(configs):
                seed = (
                    self.config.seed
                    + 1000003
                    + 10000019 * run_index
                    + 1013 * (index + 1)
                )
                mask = run.mask & generate_observed_mask(
                    reference, scenario, seed=seed
                )
                scored = self.monitor.score(
                    apply_mask(run.x, mask),
                    mask,
                    exogenous=run.covariates,
                    integrate_uncertainty=self.config.integrate_uncertainty,
                )
                _, _, _, _, persistence = self._decision_statistic(scored)
                persistence_runs.append(
                    (f"base+{scenario.identifier()}", run.name, persistence)
                )
                del scored
        self.calibrator.fit_run_clusters(persistence_runs)
        self.n_features_in_ = train.shape[1]
        self.phase1_design_run_count_ = len(train_runs)
        self.phase1_calibration_run_count_ = len(calibration_runs)
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("MATESPC has not been fitted")

    def _decision_statistic(
        self, scored: ScoreResult
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normalize before EWMA and break all temporal state at abstentions."""
        standardized, leaves, unsupported = self.regime.transform(
            self._quality_features(scored), scored.instantaneous
        )
        all_missing = ~scored.filter_result.observed_mask.any(axis=1)
        unknown = unsupported | all_missing
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
        return standardized, ewma, leaves, unknown, persistence

    def predict(
        self,
        x: np.ndarray,
        observed_mask: np.ndarray | None = None,
        covariates: np.ndarray | None = None,
        integrate_uncertainty: bool | None = None,
        elapsed_start: int = 0,
    ) -> MonitoringResult:
        self._check_fitted()
        score_mode = (
            self.config.integrate_uncertainty
            if integrate_uncertainty is None
            else bool(integrate_uncertainty)
        )
        if score_mode != self.config.integrate_uncertainty:
            raise ValueError(
                "the uncertainty score mode is fitted; fit a separate MAPIC model "
                "instead of overriding it during prediction"
            )
        arr = as_2d_float(x, "x")
        if arr.shape[1] != self.n_features_in_:
            raise ValueError("x has a different number of features from the fitted model")
        scored: ScoreResult = self.monitor.score(
            arr,
            observed_mask=observed_mask,
            exogenous=covariates,
            integrate_uncertainty=score_mode,
            elapsed_start=elapsed_start,
        )
        standardized, ewma_score, leaves, unknown, persistence_score = (
            self._decision_statistic(scored)
        )
        threshold = np.full(scored.fused.size, self.calibrator.threshold_, dtype=float)
        instantaneous_reference_threshold = (
            self.regime.centers_[leaves] + self.regime.scales_[leaves] * self.calibrator.threshold_
        )
        raw_alarm = (ewma_score > threshold) & ~unknown
        alarm = (persistence_score > threshold) & ~unknown
        status = np.zeros(scored.fused.size, dtype=np.int8)
        status[alarm] = 1
        status[unknown] = 2
        return MonitoringResult(
            score=scored.fused,
            standardized_score=standardized,
            ewma_score=ewma_score,
            persistence_score=persistence_score,
            threshold=threshold,
            instantaneous_reference_threshold=instantaneous_reference_threshold,
            raw_alarm=raw_alarm,
            alarm=alarm,
            unknown=unknown,
            status=status,
            context=leaves,
            posterior_trace=scored.posterior_trace,
            location_energy=scored.location_energy,
            uncertainty_energy=scored.uncertainty_energy,
            uncertainty_fraction=scored.uncertainty_fraction,
            reconstructed=scored.filter_result.reconstructed,
            component_t2=scored.t2_location,
            component_spe=scored.spe_location,
            component_t2_uncertainty=scored.t2_uncertainty,
            component_spe_uncertainty=scored.spe_uncertainty,
            component_innovation=scored.innovation,
        )

    @property
    def calibration_summary_(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "threshold": self.calibrator.threshold_,
            "mechanism_thresholds": self.calibrator.mechanism_thresholds_,
            "block_counts": self.calibrator.block_counts_,
            "augmentation_counts": self.calibrator.augmentation_counts_,
            "base_block_count": self.calibrator.base_block_count_,
            "base_run_count": self.calibrator.base_run_count_,
            "calibration_unit": self.calibrator.calibration_unit_,
            "training_run_count": self.monitor.filter_model.training_run_count_,
            "pooled_threshold": self.calibrator.pooled_threshold_,
            "regime_leaf_counts": self.regime.leaf_counts_.tolist(),
            "regime_centers": self.regime.centers_.tolist(),
            "regime_scales": self.regime.scales_.tolist(),
            "n_components": self.monitor.n_components_,
            "component_scales": self.monitor.component_scales_.tolist(),
            "score_mode": (
                "posterior_expectation_M_plus_U"
                if self.config.integrate_uncertainty
                else "uncertainty_decoupled_location_M"
            ),
            "quality_context": (
                "mask_gap_posterior_trace_uncertainty_energy_and_elapsed"
                if self.config.use_posterior_quality_context
                else "mask_fraction_gap_and_elapsed_only"
            ),
        }


# Clearer public alias while preserving the initially selected package/API name.
MAPIC = MATESPC
MAPICConfig = MATESPCConfig
UDMAPIC = MATESPC
UDMAPICConfig = MATESPCConfig
