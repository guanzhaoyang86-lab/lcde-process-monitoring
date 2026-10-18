"""Strictly causal, mechanism-aware features for the MSCD-TCN monitor.

The feature map is deliberately explicit about the two difficult mechanisms
targeted by the experimental method:

* stochastic changes (notably TEP Fault 9) are represented by signed
  increments, increment energy, and causal multi-scale Haar contrasts;
* valve/controller dynamics (notably TEP Fault 15) distinguish the controlled
  process variable from the actuator's physical flow/temperature response.

For an endpoint ``t``, every value below uses only the fixed commissioning
prefix ``[0, 128)`` and observations with index at most ``t``. No centred
filter, future padding, whole-run normalization, or post-endpoint statistic is
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


RUN_LENGTH = 500
PROCESS_FEATURE_COUNT = 52
BASELINE_STOP = 128
MONITOR_START = 160
FAULT_IDS = (3, 9, 15)
CLASS_IDS = (0, *FAULT_IDS)
MAD_CONSISTENCY = 1.4826
NUMERIC_SCALE_FLOOR = 1.0e-6
HAAR_SCALES = (4, 16, 64)
FEATURE_SCHEMA = "mstcn_causal_haar_loop_v2"

# Zero-based Extended TEP associations. For the first nine flow loops, the
# directly measured response is also the practical controlled variable. The
# two cooling-water loops are different: XMV10 controls reactor temperature
# (XMEAS9) and XMV11 controls separator temperature (XMEAS11).
CONTROL_VARIABLE_PAIRS = (
    (41, 1),
    (42, 2),
    (43, 0),
    (44, 3),
    (45, 4),
    (46, 9),
    (47, 13),
    (48, 16),
    (49, 18),
    (50, 8),   # XMV10 -> reactor temperature, XMEAS9
    (51, 10),  # XMV11 -> separator temperature, XMEAS11
)

# Physical response channels are retained separately. In particular XMEAS21
# and XMEAS22 are cooling-water outlet temperatures; they are not the
# controlled reactor/separator temperatures above.
FLOW_RESPONSE_PAIRS = (
    (41, 1),
    (42, 2),
    (43, 0),
    (44, 3),
    (45, 4),
    (46, 9),
    (47, 13),
    (48, 16),
    (49, 18),
    (50, 20),  # XMV10 -> reactor cooling-water outlet, XMEAS21
    (51, 21),  # XMV11 -> separator cooling-water outlet, XMEAS22
)

# Compatibility name with corrected semantics. New code should prefer the
# explicit CONTROL_VARIABLE_PAIRS or FLOW_RESPONSE_PAIRS name.
CONTROL_PAIRS = CONTROL_VARIABLE_PAIRS


def dynamic_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"signed_z[{index}]" for index in range(PROCESS_FEATURE_COUNT))
    names.extend(f"delta_z[{index}]" for index in range(PROCESS_FEATURE_COUNT))
    names.extend(f"delta_energy_z2[{index}]" for index in range(PROCESS_FEATURE_COUNT))
    for scale in HAAR_SCALES:
        names.extend(
            f"haar_z[scale={scale},variable={index}]"
            for index in range(PROCESS_FEATURE_COUNT)
        )
        names.extend(
            f"haar_energy_z2[scale={scale},variable={index}]"
            for index in range(PROCESS_FEATURE_COUNT)
        )
    names.extend(
        f"control_error[xmv={actuator},controlled_xmeas={measurement}]"
        for actuator, measurement in CONTROL_VARIABLE_PAIRS
    )
    names.extend(
        f"control_hysteresis[xmv={actuator},controlled_xmeas={measurement}]"
        for actuator, measurement in CONTROL_VARIABLE_PAIRS
    )
    names.extend(
        f"flow_response_cross[xmv={actuator},response_xmeas={measurement}]"
        for actuator, measurement in FLOW_RESPONSE_PAIRS
    )
    names.extend(
        f"flow_response_hysteresis[xmv={actuator},response_xmeas={measurement}]"
        for actuator, measurement in FLOW_RESPONSE_PAIRS
    )
    return tuple(names)


DYNAMIC_FEATURE_NAMES = dynamic_feature_names()
DYNAMIC_FEATURE_COUNT = len(DYNAMIC_FEATURE_NAMES)
if DYNAMIC_FEATURE_COUNT != 512:  # pragma: no cover - import-time invariant
    raise RuntimeError("MSCD-TCN dynamic feature geometry changed")


def _validated_run(value: Any, *, name: str = "run") -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (RUN_LENGTH, PROCESS_FEATURE_COUNT):
        raise ValueError(f"{name} must have shape ({RUN_LENGTH}, {PROCESS_FEATURE_COUNT})")
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _readonly_float(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be exact float64")
    array = np.array(source, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float64 with shape {shape}")
    array.setflags(write=False)
    return array


def causal_haar(values: Any, scale: int) -> np.ndarray:
    """Return a trailing two-half Haar contrast without using future rows.

    The value at ``t`` is ``mean(t-h+1:t+1) - mean(t-2h+1:t-h+1)`` where
    ``h = scale / 2``. Rows before a complete support exists are the canonical
    zero value, so window construction never needs future or reflected padding.
    """

    matrix = np.asarray(values)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != RUN_LENGTH
        or matrix.dtype != np.dtype(np.float64)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("values must be a finite exact-float64 500-row matrix")
    if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, (int, np.integer)):
        raise ValueError("scale must be an even positive integer")
    width = int(scale)
    if width < 2 or width % 2 != 0 or width > RUN_LENGTH:
        raise ValueError("scale must be an even positive integer no larger than 500")
    half = width // 2
    prefix = np.vstack(
        (np.zeros((1, matrix.shape[1]), dtype=np.float64), np.cumsum(matrix, axis=0))
    )
    result = np.zeros_like(matrix)
    ends = np.arange(width, RUN_LENGTH + 1, dtype=np.int64)
    recent = (prefix[ends] - prefix[ends - half]) / float(half)
    prior = (prefix[ends - half] - prefix[ends - width]) / float(half)
    result[width - 1 :] = recent - prior
    if not np.isfinite(result).all():
        raise FloatingPointError("causal Haar transform is nonfinite")
    result.setflags(write=False)
    return result


def _oriented_loop_area(
    delta: np.ndarray,
    previous: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return np.column_stack(
        [
            delta[:, actuator] * previous[:, response]
            - delta[:, response] * previous[:, actuator]
            for actuator, response in pairs
        ]
    )


@dataclass(frozen=True)
class RobustRunTransform:
    """Training-only MAD floor plus per-run commissioning normalization."""

    scale_floor: np.ndarray
    floor_quantile: float = 0.10

    def __post_init__(self) -> None:
        floor = _readonly_float(
            self.scale_floor,
            shape=(PROCESS_FEATURE_COUNT,),
            name="scale_floor",
        )
        if isinstance(self.floor_quantile, (bool, np.bool_)):
            raise ValueError("floor_quantile must be finite in (0, 0.5]")
        quantile = float(self.floor_quantile)
        if not np.isfinite(quantile) or not 0.0 < quantile <= 0.5:
            raise ValueError("floor_quantile must be finite in (0, 0.5]")
        if np.any(floor <= 0.0):
            raise ValueError("scale_floor must be strictly positive")
        object.__setattr__(self, "scale_floor", floor)
        object.__setattr__(self, "floor_quantile", quantile)

    @classmethod
    def fit(
        cls,
        runs: Mapping[str, np.ndarray],
        *,
        floor_quantile: float = 0.10,
    ) -> "RobustRunTransform":
        """Fit scale floors from commissioning prefixes of training runs only."""

        if not isinstance(runs, Mapping) or not runs:
            raise ValueError("runs must be a nonempty training-only mapping")
        if isinstance(floor_quantile, (bool, np.bool_)):
            raise ValueError("floor_quantile must be finite in (0, 0.5]")
        quantile = float(floor_quantile)
        if not np.isfinite(quantile) or not 0.0 < quantile <= 0.5:
            raise ValueError("floor_quantile must be finite in (0, 0.5]")
        scales: list[np.ndarray] = []
        for name in sorted(runs):
            run = _validated_run(runs[name], name=f"runs[{name!r}]")
            baseline = run[:BASELINE_STOP]
            center = np.median(baseline, axis=0)
            scales.append(MAD_CONSISTENCY * np.median(np.abs(baseline - center), axis=0))
        floor = np.quantile(np.stack(scales, axis=0), quantile, axis=0)
        floor = np.maximum(floor, NUMERIC_SCALE_FLOOR).astype(np.float64, copy=False)
        return cls(floor, quantile)

    def transform_run(self, run: Any) -> np.ndarray:
        """Return one read-only 500-by-512 strictly causal feature path."""

        values = _validated_run(run)
        baseline = values[:BASELINE_STOP]
        center = np.median(baseline, axis=0)
        run_scale = MAD_CONSISTENCY * np.median(np.abs(baseline - center), axis=0)
        scale = np.maximum(run_scale, self.scale_floor)
        signed = np.arcsinh((values - center) / scale)

        delta = np.zeros_like(signed)
        delta[1:] = signed[1:] - signed[:-1]
        delta_energy = np.square(delta)
        pieces: list[np.ndarray] = [signed, delta, delta_energy]
        for width in HAAR_SCALES:
            contrast = causal_haar(signed, width)
            pieces.extend((contrast, np.square(contrast)))

        previous = np.empty_like(signed)
        previous[0] = signed[0]
        previous[1:] = signed[:-1]
        control_error = np.column_stack(
            [
                signed[:, actuator] - signed[:, controlled]
                for actuator, controlled in CONTROL_VARIABLE_PAIRS
            ]
        )
        control_hysteresis = _oriented_loop_area(
            delta, previous, CONTROL_VARIABLE_PAIRS
        )
        flow_response_cross = np.column_stack(
            [
                signed[:, actuator] * signed[:, response]
                for actuator, response in FLOW_RESPONSE_PAIRS
            ]
        )
        flow_response_hysteresis = _oriented_loop_area(
            delta, previous, FLOW_RESPONSE_PAIRS
        )
        pieces.extend(
            (
                control_error,
                control_hysteresis,
                flow_response_cross,
                flow_response_hysteresis,
            )
        )
        features = np.concatenate(pieces, axis=1)
        if features.shape != (RUN_LENGTH, DYNAMIC_FEATURE_COUNT):
            raise RuntimeError("MSCD-TCN feature construction changed shape")
        if not np.isfinite(features).all():
            raise FloatingPointError("MSCD-TCN feature path is nonfinite")
        result = np.array(features, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "feature_schema": FEATURE_SCHEMA,
                "commissioning_baseline_half_open": (0, BASELINE_STOP),
                "monitor_start": MONITOR_START,
                "normalization": "per_run_baseline_median_MAD_floor_then_asinh",
                "floor_quantile": self.floor_quantile,
                "dynamic_feature_count": DYNAMIC_FEATURE_COUNT,
                "causality": "fixed_0_128_prefix_and_values_at_or_before_endpoint",
                "haar_scales": HAAR_SCALES,
                "haar_incomplete_prefix": "canonical_zero",
                "control_variable_pairs": CONTROL_VARIABLE_PAIRS,
                "flow_response_pairs": FLOW_RESPONSE_PAIRS,
            }
        )


__all__ = [
    "BASELINE_STOP",
    "CLASS_IDS",
    "CONTROL_PAIRS",
    "CONTROL_VARIABLE_PAIRS",
    "DYNAMIC_FEATURE_COUNT",
    "DYNAMIC_FEATURE_NAMES",
    "FAULT_IDS",
    "FEATURE_SCHEMA",
    "FLOW_RESPONSE_PAIRS",
    "HAAR_SCALES",
    "MONITOR_START",
    "PROCESS_FEATURE_COUNT",
    "RUN_LENGTH",
    "RobustRunTransform",
    "causal_haar",
    "dynamic_feature_names",
]
