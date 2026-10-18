"""Counterfactual-ready nonlinear four-tank process simulator for CRIS-v9.

The simulator implements the standard quadruple-tank benchmark with two pumps,
two distribution valves, and decentralized PI control of the two lower tanks.
Every run owns a :class:`numpy.random.Generator`; no module-level random state is
read or modified.  Calling :func:`simulate_run` with the same seed therefore
produces an exactly matched healthy/fault counterfactual before ``onset``.

The fixed public channel schema is

``x = [level_1, level_2, level_3, level_4]`` and
``c = [setpoint_1, setpoint_2, pump_1_command, pump_2_command]``.

``covariate_shift_only`` is deliberately a negative control: it changes the
setpoints, while leaving the plant and sensors healthy.  Consequently its
``fault_mask`` is false, and ``condition_shift_mask`` identifies the shifted
part of the run instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np


DEFAULT_DT = 1.0
DEFAULT_HORIZON = 600
DEFAULT_ONSET = 300
OBSERVATION_NAMES = ("level_1", "level_2", "level_3", "level_4")
COVARIATE_NAMES = (
    "setpoint_1",
    "setpoint_2",
    "pump_1_command",
    "pump_2_command",
)
FAULT_KINDS = (
    "healthy",
    "sensor_bias",
    "intermittent_pump",
    "valve_stiction",
    "dependence_shift",
    "covariate_shift_only",
)

ARCHIVE_SCHEMA = "mate_spc_cris_v9_four_tank_batch"
ARCHIVE_SCHEMA_VERSION = 1

# Johansson-style quadruple-tank parameters, in centimetres and seconds.
_TANK_AREAS = np.asarray((28.0, 32.0, 28.0, 32.0), dtype=np.float64)
_OUTLET_AREAS = np.asarray((0.071, 0.057, 0.071, 0.057), dtype=np.float64)
_PUMP_GAINS = np.asarray((3.33, 3.35), dtype=np.float64)
_NOMINAL_GAMMA = np.asarray((0.70, 0.60), dtype=np.float64)
_GRAVITY = 981.0
_BASE_PUMP_COMMAND = np.asarray((3.0, 3.0), dtype=np.float64)
_PI_KP = np.asarray((0.72, 0.78), dtype=np.float64)
_PI_KI = np.asarray((0.006, 0.006), dtype=np.float64)
_PUMP_LIMITS = (0.20, 8.0)
_STATE_CEILING = 40.0
_MEASUREMENT_NOISE_SD = 0.012
_PROCESS_NOISE_SD = 0.0015


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _validate_seed(value: Any) -> int:
    seed = _validate_integer(value, name="seed", minimum=0)
    if seed > np.iinfo(np.int64).max:
        raise ValueError("seed must fit in a signed 64-bit integer")
    return seed


def _validate_fault_kind(value: Any) -> str:
    if not isinstance(value, str) or value not in FAULT_KINDS:
        raise ValueError(f"fault_kind must be one of {FAULT_KINDS}")
    return value


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CRISRun:
    """One immutable simulated trajectory.

    ``observed_mask`` is a future-proof availability mask and is currently all
    true. ``fault_mask`` marks a genuine plant/sensor fault.  It intentionally
    remains false for the healthy and covariate-only negative controls.
    """

    x: np.ndarray
    c: np.ndarray
    observed_mask: np.ndarray
    fault_mask: np.ndarray
    onset: int
    fault_kind: str
    seed: int

    def __post_init__(self) -> None:
        values = np.asarray(self.x)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 4:
            raise ValueError("x must have shape (T >= 2, 4)")
        horizon = int(values.shape[0])
        x = _readonly_array(
            self.x, dtype=np.float64, shape=(horizon, 4), name="x"
        )
        c = _readonly_array(
            self.c, dtype=np.float64, shape=(horizon, 4), name="c"
        )
        observed_mask = _readonly_array(
            self.observed_mask,
            dtype=np.bool_,
            shape=(horizon, 4),
            name="observed_mask",
        )
        fault_mask = _readonly_array(
            self.fault_mask,
            dtype=np.bool_,
            shape=(horizon,),
            name="fault_mask",
        )
        onset = _validate_integer(self.onset, name="onset", minimum=1)
        if onset >= horizon:
            raise ValueError("onset must be smaller than the trajectory horizon")
        fault_kind = _validate_fault_kind(self.fault_kind)
        seed = _validate_seed(self.seed)
        if np.any(x < 0.0):
            raise ValueError("x must be nonnegative")
        if not np.all(observed_mask):
            raise ValueError("v9 simulator currently requires every observation to be present")
        if np.any(fault_mask[:onset]):
            raise ValueError("fault_mask cannot be true before onset")
        if fault_kind in ("healthy", "covariate_shift_only") and np.any(fault_mask):
            raise ValueError(f"fault_mask must be false for {fault_kind}")
        if fault_kind == "sensor_bias" and not np.all(fault_mask[onset:]):
            raise ValueError("sensor_bias fault_mask must be persistent from onset")
        if fault_kind in ("valve_stiction", "dependence_shift"):
            if fault_mask[onset] or not np.all(fault_mask[onset + 1 :]):
                raise ValueError(
                    f"{fault_kind} fault_mask must begin at the first affected "
                    "observation, onset + 1"
                )
        if fault_kind == "intermittent_pump" and not np.any(fault_mask[onset:]):
            raise ValueError("intermittent_pump must contain at least one active fault sample")

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "observed_mask", observed_mask)
        object.__setattr__(self, "fault_mask", fault_mask)
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "fault_kind", fault_kind)
        object.__setattr__(self, "seed", seed)

    @property
    def horizon(self) -> int:
        return int(self.x.shape[0])

    @property
    def mask(self) -> np.ndarray:
        """Compatibility alias for the observation-availability mask."""
        return self.observed_mask

    @property
    def condition_shift_mask(self) -> np.ndarray:
        """Return the covariate-regime mask for the negative-control run."""
        result = np.zeros(self.horizon, dtype=np.bool_)
        if self.fault_kind == "covariate_shift_only":
            result[self.onset :] = True
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class CRISBatch:
    """Immutable rectangular collection of CRIS-v9 trajectories."""

    x: np.ndarray
    c: np.ndarray
    observed_mask: np.ndarray
    fault_mask: np.ndarray
    onset: np.ndarray
    fault_kind: tuple[str, ...]
    seed: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.x)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2:
            raise ValueError("x must have shape (N >= 1, T >= 2, 4)")
        run_count, horizon, channels = values.shape
        if channels != 4:
            raise ValueError("x must have exactly four observation channels")
        x = _readonly_array(
            self.x,
            dtype=np.float64,
            shape=(run_count, horizon, 4),
            name="x",
        )
        c = _readonly_array(
            self.c,
            dtype=np.float64,
            shape=(run_count, horizon, 4),
            name="c",
        )
        observed_mask = _readonly_array(
            self.observed_mask,
            dtype=np.bool_,
            shape=(run_count, horizon, 4),
            name="observed_mask",
        )
        fault_mask = _readonly_array(
            self.fault_mask,
            dtype=np.bool_,
            shape=(run_count, horizon),
            name="fault_mask",
        )
        onset = _readonly_array(
            self.onset,
            dtype=np.int64,
            shape=(run_count,),
            name="onset",
        )
        seed = _readonly_array(
            self.seed,
            dtype=np.int64,
            shape=(run_count,),
            name="seed",
        )
        kinds = tuple(_validate_fault_kind(value) for value in self.fault_kind)
        if len(kinds) != run_count:
            raise ValueError(f"fault_kind must contain exactly {run_count} values")
        if np.any(x < 0.0):
            raise ValueError("x must be nonnegative")
        for index in range(run_count):
            # Reuse the single-run invariants and obtain independent owned arrays.
            CRISRun(
                x=x[index],
                c=c[index],
                observed_mask=observed_mask[index],
                fault_mask=fault_mask[index],
                onset=int(onset[index]),
                fault_kind=kinds[index],
                seed=int(seed[index]),
            )
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "observed_mask", observed_mask)
        object.__setattr__(self, "fault_mask", fault_mask)
        object.__setattr__(self, "onset", onset)
        object.__setattr__(self, "fault_kind", kinds)
        object.__setattr__(self, "seed", seed)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def run_count(self) -> int:
        return len(self)

    @property
    def horizon(self) -> int:
        return int(self.x.shape[1])

    @property
    def mask(self) -> np.ndarray:
        return self.observed_mask

    def run(self, index: int) -> CRISRun:
        normalized = _validate_integer(index, name="index", minimum=0)
        if normalized >= len(self):
            raise IndexError("run index is out of range")
        return CRISRun(
            x=self.x[normalized],
            c=self.c[normalized],
            observed_mask=self.observed_mask[normalized],
            fault_mask=self.fault_mask[normalized],
            onset=int(self.onset[normalized]),
            fault_kind=self.fault_kind[normalized],
            seed=int(self.seed[normalized]),
        )


def _nominal_equilibrium() -> np.ndarray:
    """Analytic steady state for the nominal pump commands and valve split."""
    u1, u2 = _BASE_PUMP_COMMAND
    gamma1, gamma2 = _NOMINAL_GAMMA
    q3 = (1.0 - gamma2) * _PUMP_GAINS[1] * u2
    q4 = (1.0 - gamma1) * _PUMP_GAINS[0] * u1
    q1 = q3 + gamma1 * _PUMP_GAINS[0] * u1
    q2 = q4 + gamma2 * _PUMP_GAINS[1] * u2
    flows = np.asarray((q1, q2, q3, q4), dtype=np.float64)
    return np.square(flows / _OUTLET_AREAS) / (2.0 * _GRAVITY)


def _intermittent_schedule(uniforms: np.ndarray, onset: int) -> np.ndarray:
    """Generate reproducible block-intermittent pump loss, forcing one pulse."""
    horizon = int(uniforms.shape[0])
    active = np.zeros(horizon, dtype=np.bool_)
    block_length = 18
    post_horizon = horizon - onset
    block_count = (post_horizon + block_length - 1) // block_length
    for block in range(block_count):
        start = onset + block * block_length
        stop = min(horizon, start + block_length)
        # Alternate candidate blocks and randomize whether the candidate fires.
        # The first block is forced on so every legal run contains the fault.
        fires = block == 0 or (block % 2 == 0 and uniforms[start] < 0.72)
        if fires:
            pulse_stop = min(stop, start + 10 + int(5.0 * uniforms[start]))
            active[start:pulse_stop] = True
    return active


def _setpoint_path(rng: np.random.Generator, horizon: int) -> np.ndarray:
    baseline = np.asarray((12.25, 12.80), dtype=np.float64)
    block_length = 75
    block_count = (horizon + block_length - 1) // block_length
    # Healthy roles deliberately span the legitimate operating envelope.  The
    # covariate-only negative control below is therefore a new *trajectory*
    # inside healthy support, not an out-of-distribution extrapolation test.
    offsets = rng.normal(0.0, 0.75, size=(block_count, 2))
    offsets = np.clip(offsets, -1.50, 1.50)
    path = baseline[None, :] + np.repeat(offsets, block_length, axis=0)[:horizon]
    return np.clip(path, (10.5, 10.5), (15.0, 15.0))


def simulate_run(
    fault_kind: str = "healthy",
    *,
    seed: int = 0,
    T: int = DEFAULT_HORIZON,
    onset: int = DEFAULT_ONSET,
    dt: float = DEFAULT_DT,
) -> CRISRun:
    """Simulate one closed-loop nonlinear quadruple-tank trajectory.

    Fault effects are applied only at indices ``onset:``.  All stochastic
    innovations are generated before the fault branch is evaluated, which is
    what makes same-seed healthy/fault prefixes bitwise identical.
    """
    kind = _validate_fault_kind(fault_kind)
    seed_value = _validate_seed(seed)
    horizon = _validate_integer(T, name="T", minimum=2)
    onset_value = _validate_integer(onset, name="onset", minimum=1)
    if onset_value >= horizon:
        raise ValueError("onset must be smaller than T")
    if kind in ("intermittent_pump", "valve_stiction", "dependence_shift") and (
        onset_value + 1 >= horizon
    ):
        raise ValueError("dynamic faults require an observation after the intervention")
    step = float(dt)
    if not np.isfinite(step) or step <= 0.0 or step > 2.0:
        raise ValueError("dt must be finite and in (0, 2]")

    rng = np.random.default_rng(seed_value)
    initial_perturbation = rng.normal(0.0, (0.08, 0.08, 0.025, 0.025))
    measurement_noise = rng.normal(
        0.0, _MEASUREMENT_NOISE_SD, size=(horizon, 4)
    )
    process_noise = rng.normal(0.0, _PROCESS_NOISE_SD, size=(horizon, 4))
    intermittent_uniforms = rng.random(horizon)
    setpoints = _setpoint_path(rng, horizon)
    intermittent_active = _intermittent_schedule(intermittent_uniforms, onset_value)

    if kind == "covariate_shift_only":
        # These offsets remain inside the healthy marginal and transition
        # support while producing a distinct post-onset operating trajectory.
        setpoints[onset_value:, 0] += 0.65
        setpoints[onset_value:, 1] -= 0.55
        setpoints[:, 0] = np.clip(setpoints[:, 0], 10.5, 15.0)
        setpoints[:, 1] = np.clip(setpoints[:, 1], 10.5, 15.0)

    levels = np.clip(
        _nominal_equilibrium() + initial_perturbation,
        0.05,
        _STATE_CEILING,
    )
    integral = np.zeros(2, dtype=np.float64)
    x = np.empty((horizon, 4), dtype=np.float64)
    c = np.empty((horizon, 4), dtype=np.float64)
    observed_mask = np.ones((horizon, 4), dtype=np.bool_)
    fault_mask = np.zeros(horizon, dtype=np.bool_)
    if kind == "sensor_bias":
        fault_mask[onset_value:] = True
    elif kind in ("valve_stiction", "dependence_shift"):
        # The intervention changes the transition from t to t+1, so the first
        # affected observation is onset+1 rather than onset.
        fault_mask[onset_value + 1 :] = True
    elif kind == "intermittent_pump":
        # Pump-efficiency interventions likewise affect the next observation.
        fault_mask[1:] = intermittent_active[:-1]

    for time_index in range(horizon):
        post_onset = time_index >= onset_value
        sensor_bias = (
            np.asarray((1.25, 0.0, 0.0, 0.0), dtype=np.float64)
            if kind == "sensor_bias" and post_onset
            else 0.0
        )
        measurement = np.maximum(levels + measurement_noise[time_index] + sensor_bias, 0.0)
        x[time_index] = measurement

        error = setpoints[time_index] - measurement[:2]
        candidate_integral = np.clip(integral + step * error, -120.0, 120.0)
        raw_command = _BASE_PUMP_COMMAND + _PI_KP * error + _PI_KI * candidate_integral
        command = np.clip(raw_command, _PUMP_LIMITS[0], _PUMP_LIMITS[1])
        # Conditional integration is a deterministic anti-windup rule.
        unsaturated = (raw_command > _PUMP_LIMITS[0]) & (raw_command < _PUMP_LIMITS[1])
        integral = np.where(unsaturated, candidate_integral, integral)
        c[time_index, :2] = setpoints[time_index]
        c[time_index, 2:] = command

        if time_index == horizon - 1:
            continue

        actual_command = command.copy()
        if kind == "intermittent_pump" and intermittent_active[time_index]:
            actual_command[1] *= 0.28

        gamma = _NOMINAL_GAMMA.copy()
        if kind == "valve_stiction" and post_onset:
            # Pump-one distribution valve is stuck far from its nominal split.
            gamma[0] = 0.43

        outlet_flow = _OUTLET_AREAS * np.sqrt(2.0 * _GRAVITY * np.maximum(levels, 0.0))
        q1, q2, q3, q4 = outlet_flow
        lower_from_upper_1 = q3
        lower_from_upper_2 = q4
        if kind == "dependence_shift" and post_onset:
            # Conservatively reroute upper-tank effluent.  This changes the
            # conditional coupling graph without injecting or removing fluid.
            reroute = 0.58
            lower_from_upper_1 = (1.0 - reroute) * q3 + reroute * q4
            lower_from_upper_2 = (1.0 - reroute) * q4 + reroute * q3

        inflow = np.asarray(
            (
                lower_from_upper_1 + gamma[0] * _PUMP_GAINS[0] * actual_command[0],
                lower_from_upper_2 + gamma[1] * _PUMP_GAINS[1] * actual_command[1],
                (1.0 - gamma[1]) * _PUMP_GAINS[1] * actual_command[1],
                (1.0 - gamma[0]) * _PUMP_GAINS[0] * actual_command[0],
            ),
            dtype=np.float64,
        )
        derivative = (inflow - outlet_flow) / _TANK_AREAS
        levels = levels + step * derivative + np.sqrt(step) * process_noise[time_index]
        levels = np.clip(levels, 0.0, _STATE_CEILING)
        if not np.isfinite(levels).all():
            raise FloatingPointError("four-tank integration produced a nonfinite state")

    return CRISRun(
        x=x,
        c=c,
        observed_mask=observed_mask,
        fault_mask=fault_mask,
        onset=onset_value,
        fault_kind=kind,
        seed=seed_value,
    )


def simulate_batch(
    fault_kinds: Sequence[str],
    seeds: Sequence[int],
    *,
    T: int = DEFAULT_HORIZON,
    onset: int = DEFAULT_ONSET,
    dt: float = DEFAULT_DT,
) -> CRISBatch:
    """Generate a rectangular batch using one explicitly seeded RNG per run."""
    if isinstance(fault_kinds, (str, bytes)):
        raise TypeError("fault_kinds must be a sequence, not one string")
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be an integer sequence")
    kinds = tuple(_validate_fault_kind(value) for value in fault_kinds)
    seed_values = tuple(_validate_seed(value) for value in seeds)
    if not kinds:
        raise ValueError("at least one run is required")
    if len(kinds) != len(seed_values):
        raise ValueError("fault_kinds and seeds must have equal length")
    runs = tuple(
        simulate_run(kind, seed=seed, T=T, onset=onset, dt=dt)
        for kind, seed in zip(kinds, seed_values, strict=True)
    )
    return CRISBatch(
        x=np.stack([run.x for run in runs]),
        c=np.stack([run.c for run in runs]),
        observed_mask=np.stack([run.observed_mask for run in runs]),
        fault_mask=np.stack([run.fault_mask for run in runs]),
        onset=np.asarray([run.onset for run in runs], dtype=np.int64),
        fault_kind=tuple(run.fault_kind for run in runs),
        seed=np.asarray([run.seed for run in runs], dtype=np.int64),
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def batch_manifest(batch: CRISBatch) -> dict[str, object]:
    """Return a canonical content manifest with hashes for every stored array."""
    if not isinstance(batch, CRISBatch):
        raise TypeError("batch must be a CRISBatch")
    fault_kind = np.asarray(batch.fault_kind, dtype="<U24")
    arrays = {
        "x": batch.x,
        "c": batch.c,
        "observed_mask": batch.observed_mask,
        "fault_mask": batch.fault_mask,
        "onset": batch.onset,
        "fault_kind": fault_kind,
        "seed": batch.seed,
    }
    return {
        "schema": ARCHIVE_SCHEMA,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "observation_names": list(OBSERVATION_NAMES),
        "covariate_names": list(COVARIATE_NAMES),
        "run_count": batch.run_count,
        "horizon": batch.horizon,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": _array_sha256(value),
            }
            for name, value in arrays.items()
        },
    }


def batch_manifest_sha256(batch: CRISBatch) -> str:
    """Hash the canonical manifest for audit and round-trip verification."""
    return hashlib.sha256(_canonical_json_bytes(batch_manifest(batch))).hexdigest()


def save_batch_npz(batch: CRISBatch, path: str | os.PathLike[str]) -> Path:
    """Atomically save a batch without object arrays or pickle serialization."""
    if not isinstance(batch, CRISBatch):
        raise TypeError("batch must be a CRISBatch")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = batch_manifest(batch)
    manifest_json = _canonical_json_bytes(manifest).decode("ascii")
    manifest_sha256 = hashlib.sha256(manifest_json.encode("ascii")).hexdigest()
    payload = {
        "x": batch.x,
        "c": batch.c,
        "observed_mask": batch.observed_mask,
        "fault_mask": batch.fault_mask,
        "onset": batch.onset,
        "fault_kind": np.asarray(batch.fault_kind, dtype="<U24"),
        "seed": batch.seed,
        "manifest_json": np.asarray(manifest_json),
        "manifest_sha256": np.asarray(manifest_sha256),
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.npz", dir=target.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _strict_scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in ("U", "S"):
        raise ValueError(f"{name} must be a scalar text value")
    normalized = str(array.item())
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def load_batch_npz(path: str | os.PathLike[str]) -> CRISBatch:
    """Load and authenticate a CRIS-v9 archive with ``allow_pickle=False``."""
    source = Path(path)
    expected_keys = {
        "x",
        "c",
        "observed_mask",
        "fault_mask",
        "onset",
        "fault_kind",
        "seed",
        "manifest_json",
        "manifest_sha256",
    }
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError("archive keys do not match the CRIS-v9 schema")
        copied = {name: np.array(archive[name], copy=True) for name in archive.files}

    for name in ("x", "c"):
        if copied[name].dtype.kind != "f":
            raise ValueError(f"{name} must have a floating dtype")
    for name in ("observed_mask", "fault_mask"):
        if copied[name].dtype != np.dtype(np.bool_):
            raise ValueError(f"{name} must have boolean dtype")
    for name in ("onset", "seed"):
        if copied[name].dtype.kind not in ("i", "u"):
            raise ValueError(f"{name} must have an integer dtype")
    fault_kind_array = copied["fault_kind"]
    if fault_kind_array.ndim != 1 or fault_kind_array.dtype.kind not in ("U", "S"):
        raise ValueError("fault_kind must be a one-dimensional text array")

    stored_manifest_json = _strict_scalar_text(
        copied["manifest_json"], name="manifest_json"
    )
    stored_manifest_hash = _strict_scalar_text(
        copied["manifest_sha256"], name="manifest_sha256"
    )
    calculated_stored_hash = hashlib.sha256(stored_manifest_json.encode("ascii")).hexdigest()
    if stored_manifest_hash != calculated_stored_hash:
        raise ValueError("manifest_sha256 does not authenticate manifest_json")
    try:
        stored_manifest = json.loads(stored_manifest_json)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("manifest_json is not valid canonical JSON") from exc
    if _canonical_json_bytes(stored_manifest).decode("ascii") != stored_manifest_json:
        raise ValueError("manifest_json is not canonical JSON")

    batch = CRISBatch(
        x=copied["x"],
        c=copied["c"],
        observed_mask=copied["observed_mask"],
        fault_mask=copied["fault_mask"],
        onset=copied["onset"],
        fault_kind=tuple(str(value) for value in fault_kind_array.tolist()),
        seed=copied["seed"],
    )
    calculated_manifest = batch_manifest(batch)
    if stored_manifest != calculated_manifest:
        raise ValueError("archive contents do not match the authenticated manifest")
    if stored_manifest_hash != batch_manifest_sha256(batch):
        raise ValueError("archive manifest hash does not match the reconstructed batch")
    return batch


# Descriptive aliases for callers that prefer generate/save/load terminology.
generate_batch = simulate_batch
save_cris_batch = save_batch_npz
load_cris_batch = load_batch_npz


__all__ = [
    "ARCHIVE_SCHEMA",
    "ARCHIVE_SCHEMA_VERSION",
    "COVARIATE_NAMES",
    "CRISBatch",
    "CRISRun",
    "DEFAULT_DT",
    "DEFAULT_HORIZON",
    "DEFAULT_ONSET",
    "FAULT_KINDS",
    "OBSERVATION_NAMES",
    "batch_manifest",
    "batch_manifest_sha256",
    "generate_batch",
    "load_batch_npz",
    "load_cris_batch",
    "save_batch_npz",
    "save_cris_batch",
    "simulate_batch",
    "simulate_run",
]
