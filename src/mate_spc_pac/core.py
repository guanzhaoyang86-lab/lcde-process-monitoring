"""Causal WS-PAC-RMax-CUSUM scoring for deployment-bound logit runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .certificate import PACCalibration
from .common import (
    GATE_PASS,
    DeploymentSpec,
    PrefixApplicabilityConfig,
    PrefixGateResult,
    monitor_configuration_sha256,
    prefix_gate_result,
    readonly_vector,
    validate_sha256,
)


@dataclass(frozen=True)
class StandardizedLogitRun:
    """Array plus the semantic-order and upstream-artifact identity it claims."""

    values: np.ndarray
    head_ids: tuple[str, ...]
    upstream_artifact_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
            raise ValueError("values must have shape (positive_time, positive_heads)")
        if isinstance(self.head_ids, (str, bytes)):
            raise ValueError("head_ids must be an ordered sequence")
        head_ids = tuple(self.head_ids)
        if values.shape[1] != len(head_ids):
            raise ValueError("head_ids length must match the logit head dimension")
        if any(not isinstance(item, str) or not item for item in head_ids):
            raise ValueError("head_ids must contain nonempty strings")
        upstream = validate_sha256(
            self.upstream_artifact_sha256,
            name="upstream_artifact_sha256",
        )
        object.__setattr__(self, "values", np.array(values, dtype=np.float64, copy=True))
        object.__setattr__(self, "head_ids", head_ids)
        object.__setattr__(self, "upstream_artifact_sha256", upstream)


@dataclass(frozen=True)
class WSPACRMaxModel:
    deployment: DeploymentSpec
    prefix_gate: PrefixApplicabilityConfig
    kappa: np.ndarray
    calibration: PACCalibration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.deployment, DeploymentSpec):
            raise TypeError("deployment must be a DeploymentSpec")
        if not isinstance(self.prefix_gate, PrefixApplicabilityConfig):
            raise TypeError("prefix_gate must be a PrefixApplicabilityConfig")
        if self.prefix_gate.head_count != self.deployment.head_count:
            raise ValueError("prefix gate head count must match deployment head order")
        kappa = readonly_vector(
            self.kappa,
            head_count=self.deployment.head_count,
            name="kappa",
            nonnegative=True,
        )
        if self.calibration is not None:
            if not isinstance(self.calibration, PACCalibration):
                raise TypeError("calibration must be a PACCalibration or None")
            if self.calibration.deployment_spec_sha256 != self.deployment.fingerprint:
                raise ValueError("calibration certificate belongs to another deployment spec")
            if self.calibration.monitor_configuration_sha256 != self.configuration_fingerprint:
                raise ValueError("calibration certificate belongs to another monitor configuration")
        object.__setattr__(self, "kappa", kappa)

    @property
    def head_count(self) -> int:
        return self.deployment.head_count

    @property
    def threshold(self) -> float | None:
        return None if self.calibration is None else self.calibration.threshold

    @property
    def configuration_fingerprint(self) -> str:
        return monitor_configuration_sha256(
            self.deployment,
            self.prefix_gate,
            self.kappa,
        )

    def with_calibration(self, calibration: PACCalibration) -> "WSPACRMaxModel":
        return replace(self, calibration=calibration)

    def save_npz(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        from .serialization import save_ws_pac_rmax_model

        return save_ws_pac_rmax_model(self, path, metadata=metadata)

    @classmethod
    def load_npz(cls, path: str | os.PathLike[str]) -> "WSPACRMaxModel":
        from .serialization import load_ws_pac_rmax_model

        model, _ = load_ws_pac_rmax_model(path)
        return model


@dataclass(frozen=True)
class WSPACRMaxPath:
    standardized_logits: np.ndarray
    transformed_evidence: np.ndarray
    clipped_evidence: np.ndarray
    cusum: np.ndarray
    component_persistence: np.ndarray
    envelope: np.ndarray
    eligible: np.ndarray
    alarm: np.ndarray
    gate: PrefixGateResult
    deployment_spec_sha256: str
    monitor_configuration_sha256: str
    head_ids: tuple[str, ...]
    threshold: float | None

    @property
    def status(self) -> str:
        return self.gate.status

    @property
    def persistence(self) -> np.ndarray:
        return self.envelope

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "standardized_logits": self.standardized_logits,
            "transformed_evidence": self.transformed_evidence,
            "clipped_evidence": self.clipped_evidence,
            "cusum": self.cusum,
            "component_persistence": self.component_persistence,
            "envelope": self.envelope,
            "eligible": self.eligible,
            "alarm": self.alarm,
        }


def page_cusum_numpy(
    evidence: np.ndarray,
    kappa: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("evidence must have shape (positive_time, positive_heads)")
    if not np.isfinite(values).all():
        raise ValueError("evidence must be finite")
    allowance = readonly_vector(
        kappa,
        head_count=values.shape[1],
        name="kappa",
        nonnegative=True,
    )
    partial = np.cumsum(values - allowance, axis=0, dtype=np.float64)
    augmented = np.vstack((np.zeros((1, values.shape[1])), partial))
    running_minimum = np.minimum.accumulate(augmented, axis=0)[1:]
    return np.maximum(partial - running_minimum, 0.0)


def _validated_values(model: WSPACRMaxModel, run: StandardizedLogitRun) -> np.ndarray:
    if not isinstance(run, StandardizedLogitRun):
        raise TypeError("run must be a StandardizedLogitRun with artifact identity")
    spec = model.deployment
    if run.values.shape != (spec.run_length, spec.head_count):
        raise ValueError(
            "logit horizon mismatch: expected exact shape "
            f"({spec.run_length}, {spec.head_count}), received {run.values.shape}"
        )
    if run.head_ids != spec.head_ids:
        raise ValueError("logit head_ids/order does not match the deployment spec")
    if run.upstream_artifact_sha256 != spec.upstream_artifact_sha256:
        raise ValueError("logit upstream artifact does not match the deployment spec")
    if np.isinf(run.values).any():
        raise ValueError("standardized logits must not contain infinity")
    if not np.isfinite(run.values[spec.valid_start:]).all():
        raise ValueError("standardized logits must be finite from deployment valid_start")
    return np.array(run.values, dtype=np.float64, copy=True, order="C")


def score_ws_pac_rmax_numpy(
    model: WSPACRMaxModel,
    run: StandardizedLogitRun,
) -> WSPACRMaxPath:
    """Score exactly the deployment specification frozen into ``model``."""
    if not isinstance(model, WSPACRMaxModel):
        raise TypeError("model must be a WSPACRMaxModel")
    matrix = _validated_values(model, run)
    spec = model.deployment
    gate = prefix_gate_result(
        matrix[spec.valid_start : spec.monitor_start],
        model.prefix_gate,
    )
    transformed = np.full_like(matrix, np.nan)
    centered = matrix[spec.valid_start:] - gate.bias
    transformed[spec.valid_start:] = np.where(
        centered > 0.0,
        centered / gate.positive_scale,
        centered,
    )
    clipped = np.full_like(matrix, np.nan)
    clipped[spec.valid_start:] = np.clip(
        transformed[spec.valid_start:],
        -spec.clip_limit,
        spec.clip_limit,
    )
    cusum = np.full_like(matrix, np.nan)
    cusum[spec.monitor_start:] = page_cusum_numpy(
        clipped[spec.monitor_start:],
        model.kappa,
    )
    component = np.full_like(matrix, -np.inf)
    first = spec.monitor_start + spec.persistence_window - 1
    for index in range(first, spec.run_length):
        component[index] = np.min(
            cusum[index - spec.persistence_window + 1 : index + 1],
            axis=0,
        )
    envelope = np.full(spec.run_length, -np.inf, dtype=np.float64)
    envelope[first:] = np.max(component[first:], axis=1)
    eligible = np.zeros(spec.run_length, dtype=bool)
    if gate.passed:
        eligible[first:] = True
    alarm = np.zeros(spec.run_length, dtype=bool)
    if model.threshold is not None and gate.passed:
        alarm = eligible & (envelope > model.threshold)
    return WSPACRMaxPath(
        standardized_logits=matrix,
        transformed_evidence=transformed,
        clipped_evidence=clipped,
        cusum=cusum,
        component_persistence=component,
        envelope=envelope,
        eligible=eligible,
        alarm=alarm,
        gate=gate,
        deployment_spec_sha256=spec.fingerprint,
        monitor_configuration_sha256=model.configuration_fingerprint,
        head_ids=spec.head_ids,
        threshold=model.threshold,
    )


def score_ws_pac_rmax_runs_numpy(
    model: WSPACRMaxModel,
    runs: Mapping[str, StandardizedLogitRun],
) -> dict[str, WSPACRMaxPath]:
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("runs must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in runs):
        raise ValueError("run names must be nonempty strings")
    return {name: score_ws_pac_rmax_numpy(model, runs[name]) for name in sorted(runs)}
