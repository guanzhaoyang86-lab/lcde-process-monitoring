from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from mate_spc_cris_v9.simulator import (
    COVARIATE_NAMES,
    DEFAULT_DT,
    DEFAULT_HORIZON,
    DEFAULT_ONSET,
    FAULT_KINDS,
    OBSERVATION_NAMES,
    CRISBatch,
    batch_manifest,
    batch_manifest_sha256,
    load_batch_npz,
    save_batch_npz,
    simulate_batch,
    simulate_run,
)


SHORT_T = 180
SHORT_ONSET = 90
PAIR_SEED = 20260909


def _legacy_rng_state_equal(
    first: tuple[object, ...], second: tuple[object, ...]
) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_fixed_schema_defaults_and_readonly_frozen_run() -> None:
    run = simulate_run(seed=7)

    assert DEFAULT_DT == 1.0
    assert DEFAULT_HORIZON == 600
    assert DEFAULT_ONSET == 300
    assert OBSERVATION_NAMES == ("level_1", "level_2", "level_3", "level_4")
    assert COVARIATE_NAMES == (
        "setpoint_1",
        "setpoint_2",
        "pump_1_command",
        "pump_2_command",
    )
    assert run.x.shape == (600, 4)
    assert run.c.shape == (600, 4)
    assert run.observed_mask.shape == (600, 4)
    assert run.fault_mask.shape == (600,)
    assert run.mask is run.observed_mask
    assert np.all(run.observed_mask)
    assert not np.any(run.fault_mask)
    for array in (run.x, run.c, run.observed_mask, run.fault_mask):
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        run.x[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        run.seed = 8  # type: ignore[misc]


def test_determinism_seed_difference_and_no_global_rng_side_effect() -> None:
    np.random.seed(12345)
    before = np.random.get_state()
    first = simulate_run("dependence_shift", seed=11, T=SHORT_T, onset=SHORT_ONSET)
    after = np.random.get_state()
    second = simulate_run("dependence_shift", seed=11, T=SHORT_T, onset=SHORT_ONSET)
    different = simulate_run("dependence_shift", seed=12, T=SHORT_T, onset=SHORT_ONSET)

    assert _legacy_rng_state_equal(before, after)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.c, second.c)
    np.testing.assert_array_equal(first.fault_mask, second.fault_mask)
    assert not np.array_equal(first.x, different.x)
    assert not np.array_equal(first.c, different.c)


@pytest.mark.parametrize("fault_kind", FAULT_KINDS[1:])
def test_same_seed_counterfactual_prefix_is_bitwise_identical(fault_kind: str) -> None:
    healthy = simulate_run("healthy", seed=PAIR_SEED, T=SHORT_T, onset=SHORT_ONSET)
    changed = simulate_run(fault_kind, seed=PAIR_SEED, T=SHORT_T, onset=SHORT_ONSET)

    np.testing.assert_array_equal(changed.x[:SHORT_ONSET], healthy.x[:SHORT_ONSET])
    np.testing.assert_array_equal(changed.c[:SHORT_ONSET], healthy.c[:SHORT_ONSET])
    assert not np.any(changed.fault_mask[:SHORT_ONSET])


@pytest.mark.parametrize("fault_kind", FAULT_KINDS[1:])
def test_every_post_onset_intervention_changes_the_matched_trajectory(
    fault_kind: str,
) -> None:
    healthy = simulate_run("healthy", seed=PAIR_SEED, T=SHORT_T, onset=SHORT_ONSET)
    changed = simulate_run(fault_kind, seed=PAIR_SEED, T=SHORT_T, onset=SHORT_ONSET)

    post_x_change = np.max(np.abs(changed.x[SHORT_ONSET:] - healthy.x[SHORT_ONSET:]))
    post_c_change = np.max(np.abs(changed.c[SHORT_ONSET:] - healthy.c[SHORT_ONSET:]))
    assert max(float(post_x_change), float(post_c_change)) > 1.0e-6
    if fault_kind == "covariate_shift_only":
        assert not np.any(changed.fault_mask)
        assert np.all(changed.condition_shift_mask[SHORT_ONSET:])
        assert not np.any(changed.condition_shift_mask[:SHORT_ONSET])
        assert post_c_change > 0.5
    elif fault_kind == "intermittent_pump":
        assert np.any(changed.fault_mask[SHORT_ONSET:])
        assert not np.all(changed.fault_mask[SHORT_ONSET:])
    elif fault_kind == "sensor_bias":
        assert np.all(changed.fault_mask[SHORT_ONSET:])
    else:
        assert not changed.fault_mask[SHORT_ONSET]
        assert np.all(changed.fault_mask[SHORT_ONSET + 1 :])


@pytest.mark.parametrize("fault_kind", FAULT_KINDS)
def test_all_fault_kinds_are_finite_nonnegative_and_bounded(fault_kind: str) -> None:
    run = simulate_run(fault_kind, seed=31, T=240, onset=120)

    assert np.isfinite(run.x).all()
    assert np.isfinite(run.c).all()
    assert np.all(run.x >= 0.0)
    assert np.max(run.x) < 40.0
    assert np.all((run.c[:, 2:] >= 0.20) & (run.c[:, 2:] <= 8.0))


def test_batch_allows_matched_seeds_and_extracts_runs() -> None:
    # Separate Generator objects may intentionally use the same seed to make a
    # paired healthy/fault counterfactual; they do not share mutable RNG state.
    batch = simulate_batch(
        ("healthy", "sensor_bias", "intermittent_pump"),
        (17, 17, 18),
        T=SHORT_T,
        onset=SHORT_ONSET,
    )

    assert isinstance(batch, CRISBatch)
    assert batch.x.shape == (3, SHORT_T, 4)
    assert batch.c.shape == (3, SHORT_T, 4)
    assert batch.observed_mask.shape == (3, SHORT_T, 4)
    assert batch.fault_mask.shape == (3, SHORT_T)
    assert batch.onset.tolist() == [SHORT_ONSET] * 3
    assert batch.fault_kind == ("healthy", "sensor_bias", "intermittent_pump")
    np.testing.assert_array_equal(batch.x[0, :SHORT_ONSET], batch.x[1, :SHORT_ONSET])
    extracted = batch.run(1)
    np.testing.assert_array_equal(extracted.x, batch.x[1])
    assert extracted.fault_kind == "sensor_bias"
    for array in (
        batch.x,
        batch.c,
        batch.observed_mask,
        batch.fault_mask,
        batch.onset,
        batch.seed,
    ):
        assert not array.flags.writeable


def test_no_pickle_npz_roundtrip_and_manifest_hash(tmp_path: Path) -> None:
    kinds = FAULT_KINDS
    seeds = tuple(range(101, 101 + len(kinds)))
    original = simulate_batch(kinds, seeds, T=150, onset=75)
    path = save_batch_npz(original, tmp_path / "cris-v9-batch.npz")

    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        assert str(archive["manifest_sha256"].item()) == batch_manifest_sha256(original)
        assert set(archive.files) == {
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

    restored = load_batch_npz(path)
    assert restored.fault_kind == original.fault_kind
    for name in ("x", "c", "observed_mask", "fault_mask", "onset", "seed"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
    assert batch_manifest(restored) == batch_manifest(original)
    assert batch_manifest_sha256(restored) == batch_manifest_sha256(original)


def test_loader_rejects_object_archive_without_enabling_pickle(tmp_path: Path) -> None:
    path = tmp_path / "object-payload.npz"
    names = (
        "x",
        "c",
        "observed_mask",
        "fault_mask",
        "onset",
        "fault_kind",
        "seed",
        "manifest_json",
        "manifest_sha256",
    )
    payload = {name: np.asarray(0) for name in names}
    payload["x"] = np.asarray([object()], dtype=object)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_batch_npz(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"fault_kind": "unknown"}, "fault_kind"),
        ({"T": 1}, "T"),
        ({"T": 20, "onset": 20}, "onset"),
        (
            {"fault_kind": "valve_stiction", "T": 20, "onset": 19},
            "observation after",
        ),
        ({"dt": 0.0}, "dt"),
        ({"seed": -1}, "seed"),
    ),
)
def test_invalid_run_arguments_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        simulate_run(**kwargs)  # type: ignore[arg-type]
