from __future__ import annotations

import torch  # noqa: F401 -- import first on the supported Windows environment

from pathlib import Path

import numpy as np
import pytest

from mate_spc_metropt.model import (
    AlignedResidualBatch,
    CaredBCScorer,
    DirectionalGLRModel,
    RankNormalTransform,
    StateConditionalReference,
)
from mate_spc_metropt.portability import certify_development_portability
from mate_spc_metropt.portability_replay import (
    EVIDENCE_KEYS,
    F1_ORIGIN,
    V1_PROJECTION_KEYS,
    EvidenceReplayError,
    alarm_evidence,
    build_evidence_payload,
    portability_path,
    replay_portability_evidence,
    save_evidence_npz,
    v1_projection_semantic_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def _scorer() -> CaredBCScorer:
    features = 31
    identity = np.eye(features, dtype=np.float64)
    reference = StateConditionalReference(
        state_codes=np.array([0], dtype=np.int64),
        counts=np.array([100], dtype=np.int64),
        centers=np.zeros((1, features), dtype=np.float64),
        covariances=identity[None, :, :],
        precisions=identity[None, :, :],
        global_center=np.zeros(features, dtype=np.float64),
        global_covariance=identity,
        global_precision=identity,
    )
    direction_vector = np.ones(features, dtype=np.float64) / np.sqrt(features)
    direction = DirectionalGLRModel(
        state_codes=np.array([0], dtype=np.int64),
        fault_counts=np.array([10], dtype=np.int64),
        directions=direction_vector[None, :],
        global_direction=direction_vector,
    )
    rank = RankNormalTransform(
        np.column_stack(
            (
                np.linspace(0.0, 100.0, 64, dtype=np.float64),
                np.linspace(0.0, 20.0, 64, dtype=np.float64),
            )
        )
    )
    return CaredBCScorer(reference, direction, rank, threshold=None)


def _synthetic_payload() -> tuple[dict[str, np.ndarray], CaredBCScorer, dict[str, dict[str, int]]]:
    scorer = _scorer()
    paths: dict[str, dict[str, object]] = {name: {} for name in ("dev_healthy", "dev_fault_f1")}
    aligned: dict[str, dict[str, AlignedResidualBatch]] = {name: {} for name in paths}
    scores: dict[str, dict[str, object]] = {name: {} for name in paths}
    counts = {
        "dev_healthy": {"inference_windows": 4, "aligned_scores": 3},
        "dev_fault_f1": {"inference_windows": 4, "aligned_scores": 3},
    }
    for split in paths:
        start = np.datetime64(
            "2020-04-17T00:00:00" if split == "dev_healthy" else "2020-04-18T00:00:00",
            "ns",
        )
        for device in ("cpu", "cuda"):
            marker = 0.05 if split == "dev_healthy" else 2.0
            residual = np.full((3, 31), marker, dtype=np.float64)
            if device == "cuda":
                residual[:, :21] += 1.0e-12
            timestamps = start + np.arange(3) * np.timedelta64(1, "m")
            batch = AlignedResidualBatch(
                residuals=residual,
                prediction_residuals=residual[:, :21],
                physical_features=residual[:, 21:],
                score_indices=np.arange(3, dtype=np.int64),
                score_timestamps=timestamps,
                state_codes=np.zeros(3, dtype=np.int64),
                source_window_rows=np.arange(9, dtype=np.int64).reshape(3, 3),
                reset_before=np.array([True, False, False], dtype=bool),
            )
            score = scorer.score(
                batch.residuals,
                batch.state_codes,
                batch.score_timestamps,
                reset_before=batch.reset_before,
            )
            prediction = np.full((4, 3, 7), marker, dtype=np.float64)
            if device == "cuda":
                prediction += 1.0e-12
            aligned[split][device] = batch
            scores[split][device] = score
            paths[split][device] = portability_path(prediction, batch, score)
    certificate = certify_development_portability(
        healthy_cpu=paths["dev_healthy"]["cpu"],
        healthy_cuda=paths["dev_healthy"]["cuda"],
        f1_cpu=paths["dev_fault_f1"]["cpu"],
        f1_cuda=paths["dev_fault_f1"]["cuda"],
        h_raw=1.0,
        f1_origin=F1_ORIGIN,
    )
    core = certificate.as_dict()
    _, alarms = alarm_evidence(paths, float(core["threshold"]["h_eff"]))
    payload = build_evidence_payload(
        paths=paths,
        aligned=aligned,
        scores=scores,
        alarms=alarms,
        core_audit=core,
    )
    return payload, scorer, counts


def test_v1_projection_algorithm_reproduces_frozen_reference() -> None:
    path = ROOT / "outputs/metropt_cared_bc_gpu_v1_portability/portability_evidence.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert tuple(sorted(archive.files)) == V1_PROJECTION_KEYS
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    assert v1_projection_semantic_fingerprint(payload) == (
        "59597b715973599186767ca1a68e7a47728fb1ddd4216be8d6d570379297289c"
    )


def test_63_key_pickle_free_roundtrip_recomputes_all_stages(tmp_path: Path) -> None:
    payload, scorer, counts = _synthetic_payload()
    assert len(EVIDENCE_KEYS) == 63
    assert set(payload) == set(EVIDENCE_KEYS)
    target = tmp_path / "evidence.npz"
    save_evidence_npz(target, payload)
    replay = replay_portability_evidence(
        target,
        scorer=scorer,
        expected_counts=counts,
        expected_h_raw=1.0,
    )
    assert set(replay.paths) == {"dev_healthy", "dev_fault_f1"}
    assert set(replay.paths["dev_healthy"]) == {"cpu", "cuda"}
    assert set(replay.full_sequence_max_abs_differences["dev_healthy"]) == {
        "prediction", "residual", "energy", "directional_glr",
        "standardized_evidence", "cusum", "envelope",
    }


def test_stage_tamper_is_rejected_even_in_a_new_self_consistent_npz(tmp_path: Path) -> None:
    payload, scorer, counts = _synthetic_payload()
    tampered = {key: np.array(value, copy=True) for key, value in payload.items()}
    key = "dev_healthy__cpu__energy"
    tampered[key][0] = np.nextafter(tampered[key][0], np.inf)
    target = tmp_path / "tampered.npz"
    save_evidence_npz(target, tampered)
    with pytest.raises(EvidenceReplayError, match="differs from scorer recomputation"):
        replay_portability_evidence(
            target, scorer=scorer, expected_counts=counts, expected_h_raw=1.0
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.array([1.0], dtype=np.float64), "invalid evidence scalar"),
        (np.asarray(1.0, dtype=np.float32), "invalid evidence scalar"),
    ],
)
def test_scalar_shape_and_dtype_are_frozen(
    tmp_path: Path, replacement: np.ndarray, message: str
) -> None:
    payload, scorer, counts = _synthetic_payload()
    payload["h_raw"] = replacement
    target = tmp_path / f"bad-{replacement.dtype}-{replacement.shape}.npz"
    save_evidence_npz(target, payload)
    with pytest.raises(EvidenceReplayError, match=message):
        replay_portability_evidence(
            target, scorer=scorer, expected_counts=counts, expected_h_raw=1.0
        )


def test_missing_key_is_rejected_before_replay(tmp_path: Path) -> None:
    payload, _, _ = _synthetic_payload()
    payload.pop("dev_fault_f1__cuda__cusum")
    with pytest.raises(EvidenceReplayError, match="key set"):
        save_evidence_npz(tmp_path / "missing.npz", payload)
