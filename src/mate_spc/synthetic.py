"""Deterministic synthetic process used for tests and executable examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import SensorSequence, phase_basis


@dataclass(frozen=True)
class SyntheticFault:
    start: int
    stop: int
    magnitude: float = 3.0
    kind: str = "step"


def _covariates(n: int) -> np.ndarray:
    progress = phase_basis(n, degree=3)
    time = np.linspace(0.0, 4.0 * np.pi, n)
    actuator = np.column_stack([np.sin(time), (np.cos(time / 2.0) > 0).astype(float)])
    return np.column_stack([actuator, progress])


def simulate_process(
    n: int,
    dimension: int = 6,
    seed: int = 0,
    faults: tuple[SyntheticFault, ...] = (),
) -> SensorSequence:
    """Simulate a correlated, nonstationary process with optional labelled faults."""
    if n < 20 or dimension < 2:
        raise ValueError("simulation requires n >= 20 and dimension >= 2")
    rng = np.random.default_rng(seed)
    u = _covariates(n)
    transition = np.eye(dimension) * 0.62
    for index in range(dimension - 1):
        transition[index + 1, index] = 0.08
    coefficients = np.empty((dimension, u.shape[1]), dtype=float)
    for i in range(dimension):
        coefficients[i] = np.array(
            [0.12 * ((i % 3) - 1), 0.08 * ((i % 2) * 2 - 1), 0.3, -0.25, 0.15]
        )
    correlation = 0.35 ** np.abs(np.subtract.outer(np.arange(dimension), np.arange(dimension)))
    innovation_cholesky = np.linalg.cholesky(0.12 * correlation)
    z = np.zeros((n, dimension), dtype=float)
    for t in range(1, n):
        noise = innovation_cholesky @ rng.normal(size=dimension)
        z[t] = transition @ z[t - 1] + coefficients @ u[t] + noise

    fault = np.zeros(n, dtype=bool)
    for item in faults:
        start = max(0, item.start)
        stop = min(n, item.stop)
        if stop <= start:
            continue
        fault[start:stop] = True
        length = stop - start
        if item.kind == "step":
            z[start:stop, 0] += item.magnitude
            z[start:stop, 1] -= 0.5 * item.magnitude
        elif item.kind == "drift":
            ramp = np.linspace(0.0, item.magnitude, length)
            z[start:stop, 0] += ramp
            z[start:stop, -1] += 0.5 * ramp
        elif item.kind == "variance":
            z[start:stop, :2] += rng.normal(scale=item.magnitude, size=(length, 2))
        else:
            raise ValueError(f"unknown synthetic fault kind: {item.kind}")

    physical_scale = np.linspace(1.0, 3.0, dimension)
    physical_location = np.linspace(5.0, 20.0, dimension)
    x = z * physical_scale + physical_location
    return SensorSequence(
        name=f"synthetic-{seed}",
        x=x,
        fault=fault,
        observed_mask=np.ones_like(x, dtype=bool),
        covariates=u,
        feature_names=tuple(f"sensor_{i + 1}" for i in range(dimension)),
        covariate_names=("actuator_sine", "actuator_state", "phase_1", "phase_2", "phase_3"),
        sample_period_seconds=60.0,
    )


def make_demo_data(
    n_train: int = 800,
    n_calibration: int = 2400,
    n_test: int = 1000,
    dimension: int = 6,
) -> tuple[SensorSequence, SensorSequence, SensorSequence]:
    train = simulate_process(n_train, dimension, seed=10)
    calibration = simulate_process(n_calibration, dimension, seed=20)
    test = simulate_process(
        n_test,
        dimension,
        seed=30,
        faults=(
            SyntheticFault(n_test // 3, n_test // 2, magnitude=2.5, kind="step"),
            SyntheticFault(2 * n_test // 3, 5 * n_test // 6, magnitude=3.0, kind="drift"),
        ),
    )
    return train, calibration, test

