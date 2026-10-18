from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mate_spc_cris_v9.skab import (
    EXPECTED_COLUMNS,
    assigned_paths,
    bind_scores_to_source,
    load_features,
    load_labels,
    load_preaccess_charter,
    partition_bounds,
    validate_score_label_binding,
)


def _write_csv(path: Path, *, delimiter: str = ",", bad_label: int | None = None) -> None:
    n = 500
    values: dict[str, object] = {
        "datetime": pd.date_range("2026-01-01", periods=n, freq="s")
    }
    for index, name in enumerate(EXPECTED_COLUMNS[1:-2]):
        values[name] = np.arange(n, dtype=float) + index
    anomaly = np.zeros(n, dtype=int)
    anomaly[350:] = 1
    if bad_label is not None:
        anomaly[400] = bad_label
    values["anomaly"] = anomaly
    changepoint = np.zeros(n, dtype=int)
    changepoint[350] = 1
    values["changepoint"] = changepoint
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False, sep=delimiter)


def test_feature_and_label_access_are_separate_and_bound(tmp_path: Path) -> None:
    root = tmp_path / "SKAB"
    path = root / "valve1" / "1.csv"
    _write_csv(path, delimiter=";")
    features = load_features(path, dataset_root=root)
    assert features.run_id == "valve1/1.csv"
    assert features.x.shape == (500, 8)
    assert features.x.flags.writeable is False
    assert features.timestamps.flags.writeable is False
    assert features.sample_period_seconds == 1.0

    score = tmp_path / "scores.npy"
    with score.open("wb") as stream:
        np.save(stream, np.zeros(500))
    metadata = bind_scores_to_source(features, score)
    assert metadata["labels_decoded"] is False

    labels = load_labels(path, dataset_root=root)
    assert labels.anomaly.sum() == 150
    assert labels.changepoint.sum() == 1
    assert labels.anomaly.flags.writeable is False
    validate_score_label_binding(metadata, labels)


def test_official_anomaly_free_file_may_omit_label_columns(tmp_path: Path) -> None:
    root = tmp_path / "SKAB"
    path = root / "anomaly-free" / "anomaly-free.csv"
    _write_csv(path, delimiter=";")
    frame = pd.read_csv(path, sep=";").drop(columns=["anomaly", "changepoint"])
    frame.to_csv(path, index=False, sep=";")
    features = load_features(path, dataset_root=root)
    assert features.x.shape == (500, 8)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_labels(path, dataset_root=root)


def test_schema_numeric_and_binary_contracts(tmp_path: Path) -> None:
    root = tmp_path / "SKAB"
    path = root / "other" / "1.csv"
    _write_csv(path, bad_label=2)
    load_features(path, dataset_root=root)
    with pytest.raises(ValueError, match="binary"):
        load_labels(path, dataset_root=root)

    frame = pd.read_csv(path)
    frame = frame.drop(columns=["Pressure"])
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_features(path, dataset_root=root)


def test_frozen_charter_and_partitions() -> None:
    project = Path(__file__).resolve().parents[1]
    charter = load_preaccess_charter(
        project / "protocols" / "cris_v9_skab_preaccess_charter.json"
    )
    development = assigned_paths(project / "not_downloaded", charter, role="development")
    final = assigned_paths(project / "not_downloaded", charter, role="final")
    assert len(development) == 18
    assert len(final) == 16
    assert set(development).isdisjoint(final)
    bounds = partition_bounds(1000)
    assert list(bounds.values()) == [(0, 500), (500, 700), (700, 850), (850, 1000)]


def test_charter_recomputes_split_instead_of_trusting_declared_lists(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    source = project / "protocols" / "cris_v9_skab_preaccess_charter.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    development = document["split_rule"]["development_files"]
    final = document["split_rule"]["final_files"]
    development[0], final[0] = final[0], development[0]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid CRIS-v9 SKAB"):
        load_preaccess_charter(tampered)


def test_charter_sidecar_matches() -> None:
    project = Path(__file__).resolve().parents[1]
    path = project / "protocols" / "cris_v9_skab_preaccess_charter.json"
    sidecar = path.with_suffix(".sha256")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split("  ")
    from mate_spc_cris_v9.skab import sha256_file

    assert filename == path.name
    assert sha256_file(path) == expected
