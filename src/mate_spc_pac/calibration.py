"""Gate-aware PAC calibration from deployment-bound decision paths."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .certificate import PACCalibration, build_calibration_certificate
from .common import GATE_DEFER, GATE_PASS, positive_integer
from .core import WSPACRMaxPath
from .probability import InsufficientCalibrationRunsError, minimum_pac_calibration_runs


def _path_run_maximum(name: str, path: WSPACRMaxPath) -> float:
    values = np.asarray(path.envelope, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise ValueError(f"PASS path {name!r} has no eligible statistic")
    first = int(finite[0])
    if not np.isneginf(values[:first]).all() or not np.isfinite(values[first:]).all():
        raise ValueError(
            f"envelope for {name!r} must contain leading -inf then finite values"
        )
    if not path.eligible[first:].all() or path.eligible[:first].any():
        raise ValueError(f"PASS path {name!r} has inconsistent eligibility")
    return float(np.max(values[first:]))


def calibrate_ws_pac_rmax_paths(
    paths_by_run: Mapping[str, WSPACRMaxPath],
    *,
    epsilon: float = 0.05,
    delta: float = 0.05,
    required_unconditional_count: int | None = None,
) -> PACCalibration:
    """Issue a certificate using only PASS runs and report every DEFER run.

    The distribution-free statement is conditional on applicability.  The
    unconditional count is retained so a high DEFER rate cannot be hidden.
    """
    if not isinstance(paths_by_run, Mapping) or not paths_by_run:
        raise ValueError("paths_by_run must be a nonempty mapping")
    if any(not isinstance(name, str) or not name for name in paths_by_run):
        raise ValueError("calibration run names must be nonempty strings")
    if required_unconditional_count is not None:
        required_unconditional_count = positive_integer(
            required_unconditional_count,
            name="required_unconditional_count",
        )
        if len(paths_by_run) != required_unconditional_count:
            raise ValueError(
                f"expected exactly {required_unconditional_count} calibration runs; "
                f"received {len(paths_by_run)}"
            )

    deployment_fingerprint: str | None = None
    monitor_fingerprint: str | None = None
    maxima: list[tuple[str, float]] = []
    deferred: list[str] = []
    for name in sorted(paths_by_run):
        path = paths_by_run[name]
        if not isinstance(path, WSPACRMaxPath):
            raise TypeError(f"calibration item {name!r} must be a WSPACRMaxPath")
        if deployment_fingerprint is None:
            deployment_fingerprint = path.deployment_spec_sha256
            monitor_fingerprint = path.monitor_configuration_sha256
        elif path.deployment_spec_sha256 != deployment_fingerprint:
            raise ValueError("calibration paths mix different deployment specifications")
        elif path.monitor_configuration_sha256 != monitor_fingerprint:
            raise ValueError("calibration paths mix different monitor configurations")
        if path.status == GATE_PASS:
            maxima.append((name, _path_run_maximum(name, path)))
        elif path.status == GATE_DEFER:
            if path.eligible.any() or path.alarm.any():
                raise ValueError(f"DEFER path {name!r} must never be eligible or alarm")
            deferred.append(name)
        else:
            raise ValueError(f"path {name!r} has unsupported gate status {path.status!r}")

    minimum = minimum_pac_calibration_runs(epsilon=epsilon, delta=delta)
    if len(maxima) < minimum:
        raise InsufficientCalibrationRunsError(
            observed_count=len(maxima),
            minimum_count=minimum,
            epsilon=float(epsilon),
            delta=float(delta),
            unconditional_count=len(paths_by_run),
            deferred_runs=tuple(deferred),
        )
    assert deployment_fingerprint is not None
    assert monitor_fingerprint is not None
    return build_calibration_certificate(
        deployment_spec_sha256=deployment_fingerprint,
        monitor_configuration_sha256=monitor_fingerprint,
        run_maxima=maxima,
        deferred_runs=deferred,
        epsilon=epsilon,
        delta=delta,
    )
