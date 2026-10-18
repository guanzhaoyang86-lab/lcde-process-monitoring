"""Numerically conservative binomial tolerance-rank calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from math import ceil, log, log1p, nextafter, inf

from .common import positive_integer, probability


CALIBRATION_CERTIFIED = "CERTIFIED"
MAX_CERTIFIABLE_SAMPLE_COUNT = 1_000_000
_DECIMAL_PRECISION = 80


class InsufficientCalibrationRunsError(ValueError):
    """Raised when the PASS-conditional sample cannot issue a finite rank."""

    def __init__(
        self,
        *,
        observed_count: int,
        minimum_count: int,
        epsilon: float,
        delta: float,
        unconditional_count: int | None = None,
        deferred_runs: tuple[str, ...] = (),
    ) -> None:
        self.observed_count = observed_count
        self.pass_conditional_count = observed_count
        self.minimum_count = minimum_count
        self.epsilon = epsilon
        self.delta = delta
        self.unconditional_count = unconditional_count
        self.deferred_runs = deferred_runs
        context = ""
        if unconditional_count is not None:
            context = (
                f", unconditional_n={unconditional_count}, "
                f"defer_n={len(deferred_runs)}"
            )
        super().__init__(
            "PAC tolerance threshold unavailable: "
            f"PASS-conditional n={observed_count}, minimum_n={minimum_count} "
            f"for epsilon={epsilon:g}, delta={delta:g}{context}"
        )


@dataclass(frozen=True)
class PACRank:
    status: str
    sample_count: int
    epsilon: float
    delta: float
    rank_one_based: int
    achieved_confidence: float
    upper_tail_probability_bound: float
    minimum_sample_count: int

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rank_one_based": self.rank_one_based,
            "achieved_confidence": self.achieved_confidence,
            "upper_tail_probability_bound": self.upper_tail_probability_bound,
            "minimum_sample_count": self.minimum_sample_count,
        }


def _coverage_power_upper(sample_count: int, epsilon: float) -> Decimal:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        coverage = Decimal(1) - Decimal.from_float(epsilon)
        return +(coverage**sample_count)


def minimum_pac_calibration_runs(*, epsilon: float, delta: float) -> int:
    """Smallest n satisfying the conservative finite-rank condition."""
    epsilon = probability(epsilon, name="epsilon")
    delta = probability(delta, name="delta")
    estimate = max(1, int(ceil(log(delta) / log1p(-epsilon))))
    delta_decimal = Decimal.from_float(delta)
    while estimate > 1 and _coverage_power_upper(estimate - 1, epsilon) <= delta_decimal:
        estimate -= 1
    while _coverage_power_upper(estimate, epsilon) > delta_decimal:
        estimate += 1
    return estimate


def _largest_certified_failure_count(
    sample_count: int,
    *,
    epsilon: float,
    delta: float,
) -> tuple[int, Decimal]:
    """Largest r with an upward-rounded P[Bin(n,epsilon)<=r] <= delta."""
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.rounding = ROUND_CEILING
        epsilon_decimal = Decimal.from_float(epsilon)
        coverage = Decimal(1) - epsilon_decimal
        delta_decimal = Decimal.from_float(delta)
        term = coverage**sample_count
        cumulative = +term
        largest = -1
        certified_cumulative = Decimal(0)
        for failures in range(sample_count + 1):
            if cumulative <= delta_decimal:
                largest = failures
                certified_cumulative = +cumulative
            else:
                break
            if failures == sample_count:
                break
            term = (
                term
                * Decimal(sample_count - failures)
                / Decimal(failures + 1)
                * epsilon_decimal
                / coverage
            )
            cumulative = cumulative + term
    return largest, certified_cumulative


def pac_tolerance_rank(
    sample_count: int,
    *,
    epsilon: float = 0.05,
    delta: float = 0.05,
) -> PACRank:
    """Smallest k with a conservatively verified binomial upper tail <= delta.

    The requested condition
    ``BinomialCDF(k-1; n, 1-epsilon) >= 1-delta`` is evaluated as
    ``P[Bin(n, epsilon) <= n-k] <= delta``.  This avoids forming
    ``1-delta`` and uses upward-rounded Decimal arithmetic, so rounding cannot
    turn a failing tail into a certificate.
    """
    sample_count = positive_integer(sample_count, name="sample_count")
    epsilon = probability(epsilon, name="epsilon")
    delta = probability(delta, name="delta")
    minimum = minimum_pac_calibration_runs(epsilon=epsilon, delta=delta)
    if sample_count < minimum:
        raise InsufficientCalibrationRunsError(
            observed_count=sample_count,
            minimum_count=minimum,
            epsilon=epsilon,
            delta=delta,
        )
    if sample_count > MAX_CERTIFIABLE_SAMPLE_COUNT:
        raise ValueError(
            f"sample_count exceeds the audited limit {MAX_CERTIFIABLE_SAMPLE_COUNT}"
        )
    failures, upper_decimal = _largest_certified_failure_count(
        sample_count,
        epsilon=epsilon,
        delta=delta,
    )
    if failures < 0:
        raise InsufficientCalibrationRunsError(
            observed_count=sample_count,
            minimum_count=minimum,
            epsilon=epsilon,
            delta=delta,
        )
    upper_tail = nextafter(float(upper_decimal), inf)
    confidence = nextafter(max(0.0, 1.0 - upper_tail), -inf)
    return PACRank(
        status=CALIBRATION_CERTIFIED,
        sample_count=sample_count,
        epsilon=epsilon,
        delta=delta,
        rank_one_based=sample_count - failures,
        achieved_confidence=confidence,
        upper_tail_probability_bound=upper_tail,
        minimum_sample_count=minimum,
    )
