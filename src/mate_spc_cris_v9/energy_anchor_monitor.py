"""Energy-anchored horizon/cross-channel predictive-residual CUSUM.

EA-HX-CUSUM preserves the ordinary predictive-residual energy CUSUM as an
anchor and lets a robust fourth-order structure branch provide only a bounded
acceleration.  The structure branch uses the first four process-state targets;
the two actuator-command targets remain present in the energy anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np

from .baselines import (
    BaselineScore,
    BaselineThreshold,
    CUSUM_ALLOWANCE,
    PredictiveResidualCUSUMBaseline,
    _BaselineInterface,
    _cusum,
    _innovation_collection,
    _innovation_run,
)


STRUCTURE_TARGET_COUNT = 4
ROBUST_MAD_FACTOR = 1.4826
ROBUST_SCALE_FLOOR = 1.0e-3
ENERGY_SCALE_FLOOR = 1.0e-8
STRUCTURE_CLIP = 8.0
FUSION_WEIGHT = 0.25


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def structure_products(
    innovations: Any,
    *,
    include_horizon: bool = True,
    include_cross_channel: bool = True,
    structure_target_count: int = STRUCTURE_TARGET_COUNT,
) -> np.ndarray:
    """Return causal signed H/X products with shape ``(T, components)``.

    H contains all same-state pairs across the three forecast horizons.  X
    contains all same-horizon pairs across the selected process-state targets.
    A row is entirely NaN unless every selected innovation is available, which
    makes the downstream structure CUSUM hold its state on incomplete rows.
    """

    run = _innovation_run(innovations, name="innovations")
    target_count = _positive_integer(
        structure_target_count, name="structure_target_count"
    )
    if target_count > run.shape[2]:
        raise ValueError("structure_target_count exceeds innovation target count")
    if run.shape[1] < 2:
        raise ValueError("at least two prediction horizons are required")
    if not include_horizon and not include_cross_channel:
        raise ValueError("at least one structure product family must be enabled")

    selected = run[:, :, :target_count]
    columns: list[np.ndarray] = []
    if include_horizon:
        for first, second in itertools.combinations(range(run.shape[1]), 2):
            columns.append(selected[:, first, :] * selected[:, second, :])
    if include_cross_channel:
        for first, second in itertools.combinations(range(target_count), 2):
            columns.append(selected[:, :, first] * selected[:, :, second])
    products = np.concatenate(columns, axis=1).astype(np.float64, copy=False)
    complete = np.all(np.isfinite(selected), axis=(1, 2))
    products[~complete] = np.nan
    return products


def _structure_energy(
    products: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    complete = np.all(np.isfinite(products), axis=1)
    energy = np.full(products.shape[0], np.nan, dtype=np.float64)
    if np.any(complete):
        standardized = np.clip(
            (products[complete] - center[None, :]) / scale[None, :],
            -STRUCTURE_CLIP,
            STRUCTURE_CLIP,
        )
        energy[complete] = np.mean(np.square(standardized), axis=1, dtype=np.float64)
    return energy


@dataclass(frozen=True, slots=True)
class EnergyAnchoredHXScore:
    """Component paths and fused score for one trajectory."""

    name: str
    energy_increment: np.ndarray
    energy_path: np.ndarray
    structure_increment: np.ndarray
    structure_path: np.ndarray
    fused_path: np.ndarray

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value)
            for value in (
                self.energy_increment,
                self.energy_path,
                self.structure_increment,
                self.structure_path,
                self.fused_path,
            )
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be nonempty text")
        if any(value.dtype != np.dtype(np.float64) for value in arrays):
            raise ValueError("EA-HX score arrays must be float64")
        if arrays[0].ndim != 1 or arrays[0].size < 1:
            raise ValueError("EA-HX score arrays must be nonempty vectors")
        if any(value.shape != arrays[0].shape for value in arrays):
            raise ValueError("EA-HX score arrays must share one shape")
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("EA-HX score arrays must be finite")
        names = (
            "energy_increment",
            "energy_path",
            "structure_increment",
            "structure_path",
            "fused_path",
        )
        for name, value in zip(names, arrays, strict=True):
            object.__setattr__(self, name, _readonly(value, dtype=np.dtype(np.float64)))

    @property
    def point_score(self) -> np.ndarray:
        return self.energy_increment

    @property
    def path(self) -> np.ndarray:
        return self.fused_path

    @property
    def trajectory_maximum(self) -> float:
        return float(np.max(self.fused_path))


@dataclass(frozen=True, slots=True)
class EnergyAnchoredHXMonitor(_BaselineInterface):
    """Healthy-only EA-HX-CUSUM monitor with a bounded structure boost."""

    energy_baseline: PredictiveResidualCUSUMBaseline
    product_center: np.ndarray
    product_scale: np.ndarray
    structure_energy_center: float
    structure_energy_scale: float
    include_horizon: bool = True
    include_cross_channel: bool = True
    structure_target_count: int = STRUCTURE_TARGET_COUNT
    allowance: float = CUSUM_ALLOWANCE
    fusion_weight: float = FUSION_WEIGHT
    name: str = "ea_hx_cusum"

    def __post_init__(self) -> None:
        if not isinstance(self.energy_baseline, PredictiveResidualCUSUMBaseline):
            raise TypeError("energy_baseline must be PredictiveResidualCUSUMBaseline")
        if type(self.include_horizon) is not bool or type(self.include_cross_channel) is not bool:
            raise TypeError("structure-family flags must be bool")
        if not self.include_horizon and not self.include_cross_channel:
            raise ValueError("at least one structure family must be enabled")
        target_count = _positive_integer(
            self.structure_target_count, name="structure_target_count"
        )
        if target_count > self.energy_baseline.innovation_shape[1]:
            raise ValueError("structure_target_count exceeds fitted target count")
        center = np.asarray(self.product_center)
        scale = np.asarray(self.product_scale)
        if center.dtype != np.dtype(np.float64) or center.ndim != 1 or center.size < 1:
            raise ValueError("product_center must be a nonempty float64 vector")
        if scale.dtype != np.dtype(np.float64) or scale.shape != center.shape:
            raise ValueError("product_scale must match product_center")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("product robust state must be finite with positive scale")
        energy_center = _finite_real(
            self.structure_energy_center, name="structure_energy_center"
        )
        energy_scale = _finite_real(
            self.structure_energy_scale, name="structure_energy_scale"
        )
        allowance = _finite_real(self.allowance, name="allowance")
        fusion_weight = _finite_real(self.fusion_weight, name="fusion_weight")
        if energy_scale <= 0.0:
            raise ValueError("structure_energy_scale must be positive")
        if allowance != CUSUM_ALLOWANCE:
            raise ValueError(f"allowance is frozen at {CUSUM_ALLOWANCE}")
        if fusion_weight != FUSION_WEIGHT:
            raise ValueError(f"fusion_weight is frozen at {FUSION_WEIGHT}")
        expected_name = "ea_"
        expected_name += "h" if self.include_horizon else ""
        expected_name += "x" if self.include_cross_channel else ""
        expected_name += "_cusum"
        if self.name != expected_name:
            raise ValueError(f"name must be {expected_name!r} for this configuration")
        object.__setattr__(self, "product_center", _readonly(center, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "product_scale", _readonly(scale, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "structure_energy_center", energy_center)
        object.__setattr__(self, "structure_energy_scale", energy_scale)
        object.__setattr__(self, "structure_target_count", target_count)
        object.__setattr__(self, "allowance", allowance)
        object.__setattr__(self, "fusion_weight", fusion_weight)

    @classmethod
    def fit(
        cls,
        healthy_fit_runs: Any,
        *,
        include_horizon: bool = True,
        include_cross_channel: bool = True,
        structure_target_count: int = STRUCTURE_TARGET_COUNT,
    ) -> "EnergyAnchoredHXMonitor":
        runs = _innovation_collection(healthy_fit_runs, name="healthy_fit_runs")
        energy = PredictiveResidualCUSUMBaseline.fit(runs)
        products = tuple(
            structure_products(
                run,
                include_horizon=include_horizon,
                include_cross_channel=include_cross_channel,
                structure_target_count=structure_target_count,
            )
            for run in runs
        )
        finite_rows = tuple(value[np.all(np.isfinite(value), axis=1)] for value in products)
        joined = np.concatenate(finite_rows, axis=0)
        if joined.shape[0] < 2:
            raise ValueError("EA-HX fitting needs at least two complete structure rows")
        center = np.median(joined, axis=0)
        mad = np.median(np.abs(joined - center[None, :]), axis=0)
        scale = np.maximum(ROBUST_MAD_FACTOR * mad, ROBUST_SCALE_FLOOR)
        structure_energies = np.concatenate(
            [_structure_energy(value, center, scale) for value in products]
        )
        finite_energy = structure_energies[np.isfinite(structure_energies)]
        if finite_energy.size < 2:
            raise ValueError("EA-HX fitting needs two finite structure energies")
        name = "ea_"
        name += "h" if include_horizon else ""
        name += "x" if include_cross_channel else ""
        name += "_cusum"
        return cls(
            energy_baseline=energy,
            product_center=np.asarray(center, dtype=np.float64),
            product_scale=np.asarray(scale, dtype=np.float64),
            structure_energy_center=float(np.mean(finite_energy)),
            structure_energy_scale=max(float(np.std(finite_energy)), ENERGY_SCALE_FLOOR),
            include_horizon=include_horizon,
            include_cross_channel=include_cross_channel,
            structure_target_count=structure_target_count,
            name=name,
        )

    def score_components(self, values: Any) -> EnergyAnchoredHXScore:
        run = _innovation_run(values, name="values")
        if run.shape[1:] != self.energy_baseline.innovation_shape:
            raise ValueError("innovation shape does not match fitted EA-HX monitor")
        energy_score = self.energy_baseline.score(run)
        products = structure_products(
            run,
            include_horizon=self.include_horizon,
            include_cross_channel=self.include_cross_channel,
            structure_target_count=self.structure_target_count,
        )
        structure_energy = _structure_energy(products, self.product_center, self.product_scale)
        available = np.isfinite(structure_energy)
        increments = np.zeros(structure_energy.shape, dtype=np.float64)
        increments[available] = (
            (structure_energy[available] - self.structure_energy_center)
            / self.structure_energy_scale
            - self.allowance
        )
        structure_path = _cusum(increments)
        fused = energy_score.path + self.fusion_weight * np.minimum(
            energy_score.path, structure_path
        )
        return EnergyAnchoredHXScore(
            name=self.name,
            energy_increment=np.asarray(energy_score.point_score, dtype=np.float64),
            energy_path=np.asarray(energy_score.path, dtype=np.float64),
            structure_increment=increments,
            structure_path=structure_path,
            fused_path=np.asarray(fused, dtype=np.float64),
        )

    def score(self, values: Any) -> BaselineScore:
        components = self.score_components(values)
        return BaselineScore(
            baseline_name=self.name,
            point_score=np.asarray(components.energy_increment, dtype=np.float64),
            path=np.asarray(components.fused_path, dtype=np.float64),
        )

    def calibrate_trajectory_threshold(
        self, healthy_calibration_runs: Any, *, alpha: float = 0.10
    ) -> BaselineThreshold:
        runs = _innovation_collection(
            healthy_calibration_runs, name="healthy_calibration_runs"
        )
        maxima = np.asarray(
            [self.score(run).trajectory_maximum for run in runs], dtype=np.float64
        )
        alpha_value = _finite_real(alpha, name="alpha")
        if not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        rank = int(math.ceil((maxima.size + 1) * (1.0 - alpha_value)))
        if rank > maxima.size:
            raise ValueError(
                "requested trajectory threshold is unreachable for this calibration count"
            )
        threshold = float(np.sort(maxima, kind="stable")[rank - 1])
        return BaselineThreshold(
            baseline_name=self.name,
            alpha=alpha_value,
            rank=rank,
            threshold=threshold,
            calibration_maxima=maxima,
        )


__all__ = [
    "ENERGY_SCALE_FLOOR",
    "EnergyAnchoredHXMonitor",
    "EnergyAnchoredHXScore",
    "FUSION_WEIGHT",
    "ROBUST_MAD_FACTOR",
    "ROBUST_SCALE_FLOOR",
    "STRUCTURE_CLIP",
    "STRUCTURE_TARGET_COUNT",
    "structure_products",
]
