from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_lcde_skab_gpu as runner
from mate_spc_cris_v9.local_contrast import (
    LocalContrastConfig,
    LocalContrastDualEnergyMonitor,
)
from mate_spc_cris_v9.skab import SENSOR_COLUMNS, SKABFeatures, SKABLabels


def _patch_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    monkeypatch.setattr(runner, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(runner, "MODEL_PATH", tmp_path / "model.npz")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(
        runner, "DEVELOPMENT_RESULT", tmp_path / "output" / "development.json"
    )
    monkeypatch.setattr(runner, "FROZEN_JSON", tmp_path / "output" / "frozen.json")
    monkeypatch.setattr(runner, "FROZEN_NPZ", tmp_path / "output" / "frozen.npz")
    monkeypatch.setattr(runner, "FINAL_RESULT", tmp_path / "output" / "final.json")
    monkeypatch.setattr(
        runner, "FINAL_SCORE_ROOT", tmp_path / "output" / "final_scores"
    )
    if hasattr(runner, "FINAL_SCORE_MANIFEST"):
        monkeypatch.setattr(
            runner,
            "FINAL_SCORE_MANIFEST",
            tmp_path / "output" / "final_score_manifest.json",
        )
    runner.MODEL_PATH.write_bytes(b"synthetic frozen upstream model\n")
    bindings = {
        "runner_sha256": "1" * 64,
        "method_sha256": "2" * 64,
        "charter_sha256": "3" * 64,
        "model_sha256": runner.sha256_file(runner.MODEL_PATH),
    }
    monkeypatch.setattr(runner, "_implementation_hashes", lambda: dict(bindings))


def _frozen_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[
    LocalContrastDualEnergyMonitor,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    dict[str, object],
]:
    _patch_artifact_paths(monkeypatch, tmp_path)
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=(380, 3))
    innovations = rng.normal(size=(380, 2, 3))
    state = LocalContrastDualEnergyMonitor(
        config=LocalContrastConfig(
            current_window=20,
            channel_top_k=1,
            allowance=0.5,
            scan_horizon=60,
        ),
        component_center=np.arange(5, dtype=np.float64) / 10.0,
        component_scale=np.ones(5, dtype=np.float64),
    )
    selected: dict[str, object] = {
        "candidate_id": "w20_k1_a0p5_s60",
        "config": state.config.as_dict(),
        "threshold": float(np.max(state.score(x, innovations, mode="dual"))),
        "healthy_audit_episodes": 0,
        "aggregate": {"detected_runs": 12, "mean_unadjusted_f1": 0.45},
    }
    split_manifest: dict[str, object] = {
        "path_corrections": {},
        "roles": {"development": [], "final": []},
    }
    development: dict[str, object] = {
        "schema": "lcde_skab_development",
        "selected_candidate_id": selected["candidate_id"],
        "selected": selected,
        "go": False,
        "decision": "DEVELOPMENT_NO_GO",
        "artifact_bindings": runner._implementation_hashes(),
        "resolved_split_manifest": split_manifest,
        "resolved_split_manifest_sha256": runner._canonical_sha256(split_manifest),
    }
    monkeypatch.setattr(runner, "load_preaccess_charter", lambda _path: {})
    monkeypatch.setattr(
        runner, "_resolved_split_manifest", lambda _charter: split_manifest
    )
    runner._write_json(runner.DEVELOPMENT_RESULT, development)
    return state, x, innovations, selected, development


def _produce_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> LocalContrastDualEnergyMonitor:
    state, x, innovations, selected, development = _frozen_fixture(
        monkeypatch, tmp_path
    )
    runner._save_frozen(
        state=state,
        selected=selected,
        threshold_x=x,
        threshold_innovations=innovations,
        development=development,
        freeze_basis="UNIT_TEST_COMPARATIVE_FREEZE",
        comparison=None,
    )
    return state


def test_frozen_producer_consumer_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    produced = _produce_frozen(monkeypatch, tmp_path)

    consumed, thresholds, document = runner._load_frozen()

    assert consumed.config == produced.config
    np.testing.assert_array_equal(consumed.component_center, produced.component_center)
    np.testing.assert_array_equal(consumed.component_scale, produced.component_scale)
    assert tuple(thresholds) == runner.MODES
    assert all(np.isfinite(tuple(thresholds.values())))
    assert document["state_npz_sha256"] == runner.sha256_file(runner.FROZEN_NPZ)
    assert document["freeze_basis"] == "UNIT_TEST_COMPARATIVE_FREEZE"


def test_load_frozen_rejects_tampered_recorded_state_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _produce_frozen(monkeypatch, tmp_path)
    document = json.loads(runner.FROZEN_JSON.read_text(encoding="utf-8"))
    document["state_npz_sha256"] = "0" * 64
    runner._write_json(runner.FROZEN_JSON, document)

    with pytest.raises(ValueError, match="state hash mismatch"):
        runner._load_frozen()


def test_final_score_manifest_rejects_score_file_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_artifact_paths(monkeypatch, tmp_path)
    runner.FROZEN_JSON.parent.mkdir(parents=True, exist_ok=True)
    runner.FROZEN_JSON.write_text("{}\n", encoding="ascii")
    runner.FINAL_SCORE_ROOT.mkdir(parents=True, exist_ok=True)
    state_hash = "f" * 64
    run = {
        "run_id": "other/fixture.csv",
        "source_sha256": "a" * 64,
        "timestamps": np.arange(4),
        "x": np.zeros((4, 2), dtype=np.float64),
    }
    score_file = runner.FINAL_SCORE_ROOT / "other__fixture.csv.npz"
    paths = {mode: np.arange(4, dtype=np.float64) for mode in runner.MODES}
    np.savez_compressed(
        score_file,
        run_id=np.asarray(run["run_id"]),
        source_sha256=np.asarray(run["source_sha256"]),
        frozen_state_sha256=np.asarray(state_hash),
        **paths,
    )
    frozen = {"state_npz_sha256": state_hash}
    runner._write_final_score_manifest(
        [{"run": run, "paths": paths, "score_file": str(score_file)}], frozen
    )
    loaded = runner._load_final_score_manifest([run], frozen)
    np.testing.assert_array_equal(loaded[0]["paths"]["dual"], paths["dual"])

    score_file.write_bytes(score_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="final score file hash mismatch"):
        runner._load_final_score_manifest([run], frozen)


def test_final_rejects_score_label_source_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_artifact_paths(monkeypatch, tmp_path)
    runner._write_json(
        runner.DEVELOPMENT_RESULT,
        {"schema": "lcde_skab_development", "go": True},
    )
    runner.FROZEN_JSON.parent.mkdir(parents=True, exist_ok=True)
    runner.FROZEN_JSON.write_text("{}\n", encoding="ascii")
    runner.FROZEN_NPZ.write_bytes(b"synthetic frozen state\n")

    length = 320
    timestamps = np.arange(length).astype("timedelta64[s]") + np.datetime64(
        "2026-01-01"
    )
    x = np.zeros((length, len(SENSOR_COLUMNS)), dtype=np.float64)
    features = SKABFeatures(
        run_id="other/fixture.csv",
        timestamps=timestamps,
        x=x,
        feature_names=SENSOR_COLUMNS,
        sample_period_seconds=1.0,
        source_sha256="a" * 64,
    )
    run = {
        "features": features,
        "run_id": features.run_id,
        "source_sha256": features.source_sha256,
        "timestamps": timestamps,
        "x": x,
        "innovations": np.zeros((length, 1, x.shape[1]), dtype=np.float64),
    }

    class _State:
        def score(self, values, innovations, *, mode="dual"):
            del innovations, mode
            return np.zeros(values.shape[0], dtype=np.float64)

    monkeypatch.setattr(runner, "load_preaccess_charter", lambda _path: {})
    monkeypatch.setattr(runner, "_paths", lambda _charter, _role: (tmp_path / "x",))
    monkeypatch.setattr(runner, "_load_runs", lambda *_args, **_kwargs: [run])
    monkeypatch.setattr(
        runner.CRISModel,
        "load_npz",
        lambda _path: SimpleNamespace(
            attestation=SimpleNamespace(cuda_device_name="synthetic-cuda")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_frozen",
        lambda: (
            _State(),
            {mode: 1.0 for mode in runner.MODES},
            {
                "state_npz_sha256": runner.sha256_file(runner.FROZEN_NPZ),
                "freeze_basis": "UNIT_TEST",
            },
        ),
    )
    anomaly = np.zeros(length, dtype=bool)
    anomaly[310:] = True
    changepoint = np.zeros(length, dtype=bool)
    changepoint[310] = True
    mismatched = SKABLabels(
        run_id=features.run_id,
        anomaly=anomaly,
        changepoint=changepoint,
        source_sha256="b" * 64,
    )
    monkeypatch.setattr(runner, "load_labels", lambda *_args, **_kwargs: mismatched)

    with pytest.raises(
        ValueError, match="persisted score path and later SKAB labels do not match"
    ):
        runner.final()
