from __future__ import annotations

# The supported Windows runtime requires Torch before NumPy, including during
# mixed test collection.
import torch  # noqa: F401
import numpy as np
import pytest

from mate_spc_mstcn.features import (
    BASELINE_STOP,
    CONTROL_VARIABLE_PAIRS,
    DYNAMIC_FEATURE_COUNT,
    DYNAMIC_FEATURE_NAMES,
    FEATURE_SCHEMA,
    FLOW_RESPONSE_PAIRS,
    HAAR_SCALES,
    RobustRunTransform,
    causal_haar,
)


def _run(seed: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(500, 52)).astype(np.float64)


def test_feature_geometry_and_loop_semantics_are_explicit() -> None:
    assert DYNAMIC_FEATURE_COUNT == 512
    assert len(DYNAMIC_FEATURE_NAMES) == len(set(DYNAMIC_FEATURE_NAMES)) == 512
    assert HAAR_SCALES == (4, 16, 64)
    assert CONTROL_VARIABLE_PAIRS[-2:] == ((50, 8), (51, 10))
    assert FLOW_RESPONSE_PAIRS[-2:] == ((50, 20), (51, 21))
    assert (
        "control_error[xmv=51,controlled_xmeas=10]" in DYNAMIC_FEATURE_NAMES
    )
    assert (
        "flow_response_cross[xmv=51,response_xmeas=21]"
        in DYNAMIC_FEATURE_NAMES
    )
    assert FEATURE_SCHEMA == "mstcn_causal_haar_loop_v2"


def test_causal_haar_has_exact_trailing_two_half_definition() -> None:
    values = np.arange(500, dtype=np.float64)[:, None]
    result = causal_haar(values, 4)
    np.testing.assert_array_equal(result[:3], 0.0)
    # mean(2,3) - mean(0,1)
    assert result[3, 0] == 2.0
    # mean(8,9) - mean(6,7)
    assert result[9, 0] == 2.0
    with pytest.raises(ValueError, match="even positive"):
        causal_haar(values, 3)


def test_transform_contains_increment_and_haar_energy_and_is_causal() -> None:
    training = {f"train-{index}": _run(100 + index) for index in range(3)}
    transform = RobustRunTransform.fit(training)
    original = _run(200)
    changed = original.copy()
    changed[251:] += np.linspace(2.0, 20.0, 249)[:, None]
    left = transform.transform_run(original)
    right = transform.transform_run(changed)
    assert left.shape == (500, 512)
    assert left.dtype == np.dtype(np.float64)
    assert not left.flags.writeable
    np.testing.assert_array_equal(left[:251], right[:251])
    assert np.max(np.abs(left[251:] - right[251:])) > 0.0

    delta_index = DYNAMIC_FEATURE_NAMES.index("delta_z[0]")
    delta_energy_index = DYNAMIC_FEATURE_NAMES.index("delta_energy_z2[0]")
    np.testing.assert_allclose(
        left[:, delta_energy_index],
        np.square(left[:, delta_index]),
        rtol=0.0,
        atol=0.0,
    )
    for scale in HAAR_SCALES:
        haar_index = DYNAMIC_FEATURE_NAMES.index(
            f"haar_z[scale={scale},variable=0]"
        )
        energy_index = DYNAMIC_FEATURE_NAMES.index(
            f"haar_energy_z2[scale={scale},variable=0]"
        )
        np.testing.assert_allclose(
            left[:, energy_index],
            np.square(left[:, haar_index]),
            rtol=0.0,
            atol=0.0,
        )

    metadata = transform.metadata
    assert metadata["commissioning_baseline_half_open"] == (0, BASELINE_STOP)
    assert metadata["feature_schema"] == FEATURE_SCHEMA
    assert metadata["control_variable_pairs"][-1] == (51, 10)
    assert metadata["flow_response_pairs"][-1] == (51, 21)


def test_transform_rejects_non_float64_and_copies_floor() -> None:
    with pytest.raises(ValueError, match="exact float64"):
        RobustRunTransform.fit({"bad": np.zeros((500, 52), dtype=np.float32)})
    floor = np.ones(52, dtype=np.float64)
    transform = RobustRunTransform(floor)
    floor[:] = 7.0
    np.testing.assert_array_equal(transform.scale_floor, 1.0)
    with pytest.raises(ValueError, match="exact float64"):
        RobustRunTransform(np.ones(52, dtype=np.float32))
