"""Auditable PASS-conditional PAC calibration certificates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Iterable

from .common import validate_sha256
from .probability import CALIBRATION_CERTIFIED, PACRank, pac_tolerance_rank


STRICT_COMPARISON = "strictly_greater_than"


def calibration_maxima_sha256(
    deployment_spec_sha256: str,
    monitor_configuration_sha256: str,
    run_maxima: Iterable[tuple[str, float]],
    deferred_runs: Iterable[str],
) -> str:
    """Fingerprint names, exact float values, DEFER membership, and deployment."""
    deployment = validate_sha256(
        deployment_spec_sha256,
        name="deployment_spec_sha256",
    )
    monitor = validate_sha256(
        monitor_configuration_sha256,
        name="monitor_configuration_sha256",
    )
    maxima = sorted((str(name), float(value)) for name, value in run_maxima)
    deferred = sorted(str(name) for name in deferred_runs)
    payload = {
        "deferred_runs": deferred,
        "deployment_spec_sha256": deployment,
        "monitor_configuration_sha256": monitor,
        "pass_run_maxima": [
            {"maximum_hex": value.hex(), "run": name} for name, value in maxima
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PACCalibration:
    """Certificate conditional on the predeclared prefix gate returning PASS."""

    status: str
    threshold: float
    rank: PACRank
    rank_driver_run: str
    deployment_spec_sha256: str
    monitor_configuration_sha256: str
    unconditional_sample_count: int
    pass_conditional_sample_count: int
    defer_sample_count: int
    deferred_runs: tuple[str, ...]
    run_maxima: tuple[tuple[str, float], ...]
    calibration_maxima_sha256: str
    comparison: str = STRICT_COMPARISON

    def __post_init__(self) -> None:
        if self.status != CALIBRATION_CERTIFIED:
            raise ValueError(f"calibration status must be {CALIBRATION_CERTIFIED}")
        if self.comparison != STRICT_COMPARISON:
            raise ValueError(f"comparison must be {STRICT_COMPARISON}")
        deployment = validate_sha256(
            self.deployment_spec_sha256,
            name="deployment_spec_sha256",
        )
        monitor = validate_sha256(
            self.monitor_configuration_sha256,
            name="monitor_configuration_sha256",
        )
        fingerprint = validate_sha256(
            self.calibration_maxima_sha256,
            name="calibration_maxima_sha256",
        )
        if not isinstance(self.rank, PACRank):
            raise TypeError("rank must be a PACRank")
        maxima = tuple(sorted((str(name), float(value)) for name, value in self.run_maxima))
        deferred = tuple(sorted(str(name) for name in self.deferred_runs))
        names = [name for name, _ in maxima]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("PASS calibration run names must be nonempty and unique")
        if any(not name for name in deferred) or len(set(deferred)) != len(deferred):
            raise ValueError("DEFER calibration run names must be nonempty and unique")
        if set(names).intersection(deferred):
            raise ValueError("a calibration run cannot be both PASS and DEFER")
        if any(not isfinite(value) or value < 0.0 for _, value in maxima):
            raise ValueError("calibration maxima must be finite and nonnegative")
        pass_count = len(maxima)
        defer_count = len(deferred)
        if self.pass_conditional_sample_count != pass_count:
            raise ValueError("PASS-conditional sample count does not match maxima")
        if self.defer_sample_count != defer_count:
            raise ValueError("DEFER sample count does not match deferred runs")
        if self.unconditional_sample_count != pass_count + defer_count:
            raise ValueError("unconditional sample count must equal PASS plus DEFER")
        if self.rank.sample_count != pass_count:
            raise ValueError("PAC rank sample count must equal PASS-conditional count")
        expected_rank = pac_tolerance_rank(
            pass_count,
            epsilon=self.rank.epsilon,
            delta=self.rank.delta,
        )
        if self.rank != expected_rank:
            raise ValueError("PAC rank fields do not match a fresh conservative calculation")
        ordered = sorted(maxima, key=lambda item: (item[1], item[0]))
        driver, threshold = ordered[self.rank.rank_one_based - 1]
        if self.rank_driver_run != driver or float(self.threshold) != threshold:
            raise ValueError("threshold or rank driver does not match calibration maxima")
        expected_fingerprint = calibration_maxima_sha256(
            deployment,
            monitor,
            maxima,
            deferred,
        )
        if fingerprint != expected_fingerprint:
            raise ValueError("calibration maxima fingerprint mismatch")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "deployment_spec_sha256", deployment)
        object.__setattr__(self, "monitor_configuration_sha256", monitor)
        object.__setattr__(self, "deferred_runs", deferred)
        object.__setattr__(self, "run_maxima", maxima)
        object.__setattr__(self, "calibration_maxima_sha256", fingerprint)

    @property
    def epsilon(self) -> float:
        return self.rank.epsilon

    @property
    def delta(self) -> float:
        return self.rank.delta

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "rank_driver_run": self.rank_driver_run,
            "deployment_spec_sha256": self.deployment_spec_sha256,
            "monitor_configuration_sha256": self.monitor_configuration_sha256,
            "unconditional_sample_count": self.unconditional_sample_count,
            "pass_conditional_sample_count": self.pass_conditional_sample_count,
            "defer_sample_count": self.defer_sample_count,
            "deferred_runs": list(self.deferred_runs),
            "calibration_maxima_sha256": self.calibration_maxima_sha256,
            "rank": self.rank.as_dict(),
            "run_maxima": [
                {"run": name, "maximum": maximum}
                for name, maximum in self.run_maxima
            ],
        }


def build_calibration_certificate(
    *,
    deployment_spec_sha256: str,
    monitor_configuration_sha256: str,
    run_maxima: Iterable[tuple[str, float]],
    deferred_runs: Iterable[str],
    epsilon: float,
    delta: float,
) -> PACCalibration:
    maxima = tuple(sorted((str(name), float(value)) for name, value in run_maxima))
    deferred = tuple(sorted(str(name) for name in deferred_runs))
    rank = pac_tolerance_rank(len(maxima), epsilon=epsilon, delta=delta)
    ordered = sorted(maxima, key=lambda item: (item[1], item[0]))
    driver, threshold = ordered[rank.rank_one_based - 1]
    fingerprint = calibration_maxima_sha256(
        deployment_spec_sha256,
        monitor_configuration_sha256,
        maxima,
        deferred,
    )
    return PACCalibration(
        status=CALIBRATION_CERTIFIED,
        threshold=threshold,
        rank=rank,
        rank_driver_run=driver,
        deployment_spec_sha256=deployment_spec_sha256,
        monitor_configuration_sha256=monitor_configuration_sha256,
        unconditional_sample_count=len(maxima) + len(deferred),
        pass_conditional_sample_count=len(maxima),
        defer_sample_count=len(deferred),
        deferred_runs=deferred,
        run_maxima=maxima,
        calibration_maxima_sha256=fingerprint,
    )
