"""Run the one-shot final confirmation of the frozen hybrid GLR-CUSUM.

The script authenticates the completed hybrid experiment, freezes a separate
final protocol, and only then opens Rieth runs 401--500.  It performs no fit,
calibration, threshold selection, or model selection.  A completed invocation
is safely replayable; an incomplete result directory is deliberately not
silently overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence


def _requested_device(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith("--device="):
            return token.partition("=")[2]
        if token == "--device" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


# CUDA initialization must precede NumPy/MKL imports on the target Windows host.
if (_requested_device(sys.argv[1:]) or "").startswith("cuda"):
    import torch as _preloaded_torch  # noqa: F401

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_directional_glr_gpu as base  # noqa: E402
import run_hybrid_energy_glr_gpu as hybrid  # noqa: E402

from mate_spc.datasets import RiethTEPArchive, SensorSequence  # noqa: E402
from mate_spc_directional.glr import (  # noqa: E402
    FAULT_IDS,
    PERSISTENCE_WINDOW,
    load_directional_glr_model,
)
from mate_spc_directional.gpu import (  # noqa: E402
    score_directional_glr_cuda,
    validate_directional_glr_cuda,
)
from mate_spc_hybrid.features import (  # noqa: E402
    ENERGY_RAW_FEATURE_COUNT,
    load_hybrid_feature_transformer,
)


DATASET_DOI = "10.7910/DVN/6C3JR1"
ANALYSIS_STATUS = "one_shot_final_confirmation"
METHOD = "hybrid_directional_energy_density_ratio_page_cusum"
PARENT_DECISION = "PASS_ALL_FROZEN_AUDIT_GATES"
PASS_DECISION = "PASS_FINAL_CONFIRMATION"
FAIL_DECISION = "FAIL_FINAL_CONFIRMATION_NO_RETUNING"

EXPECTED_PARENT_COMPLETION_SHA256 = (
    "d42f7088791ee6cd5ba36be74912e02ea3a1eea2e1bd56894d9231ef9c49087d"
)
EXPECTED_PARENT_COMPLETION_FINGERPRINT = (
    "b29b2ae8d51fd878a63207deb8b58e2c5fc2c9a43e15baae4c6f2c2d77ab5ef1"
)
EXPECTED_PARENT_PHASE_FINGERPRINT = (
    "9e817cd7201176cc27552382ea1b27fe8bfa185e6abc664c9615fdb8fda32efb"
)
EXPECTED_TRANSFORMER_SHA256 = (
    "09cca42496360e3132239150b3bf5f5dcd761df61eb35ae7082465c187fb28ab"
)
EXPECTED_TRANSFORMER_SIZE = 449540
EXPECTED_CALIBRATED_MODEL_SHA256 = (
    "ea799d0eb4d728f6700c922fbf20ee8ace405272a75bed9fa74130ccb89fac57"
)
EXPECTED_CALIBRATED_MODEL_SIZE = 49250

FAULT_START = 160
FINAL_RUNS = tuple(range(401, 501))
SEQUENCE_STOP = 500
EXPECTED_TEST_LENGTH = 960
EXPECTED_DIRECTIONAL_FEATURE_COUNT = 1080
EXPECTED_ENERGY_FEATURE_COUNT = 768
EXPECTED_HYBRID_FEATURE_COUNT = 1848
EXPECTED_VALID_START = 63

FROZEN_RIDGE_LAMBDA = 0.001
FROZEN_KAPPA = 0.75
EXPECTED_THRESHOLD = 165.05745028203916
CPU_GPU_TOLERANCE = 1e-10

FINAL_HEALTH_MAXIMUM = 9
FINAL_PER_FAULT_MINIMUM = 60
FINAL_TOTAL_MINIMUM = 210

PARENT_ARTIFACTS = (
    "DEVELOPMENT_REPORT.md",
    "REPORT.md",
    "aggregate.json",
    "audit_base_manifest.json",
    "audit_open_manifest.json",
    "audit_result.json",
    "audit_scores.npz",
    "calibration.json",
    "calibration_scores.npz",
    "cuda_validation.json",
    "development_aggregate.json",
    "development_candidates.json",
    "development_completion.json",
    "development_fit_metadata.json",
    "development_gpu_execution.json",
    "development_manifest.json",
    "development_scores.npz",
    "gpu_execution.json",
    "hybrid_glr_calibrated.npz",
    "hybrid_glr_uncalibrated.npz",
    "hybrid_transformer.npz",
)

FINAL_ARTIFACTS = (
    "REPORT.md",
    "cuda_validation.json",
    "final_manifest.json",
    "final_result.json",
    "final_scores.npz",
    "gpu_execution.json",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot frozen hybrid GLR-CUSUM final confirmation on CUDA"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hybrid-dir",
        type=Path,
        help="completed outputs/hybrid_energy_glr_gpu_v1 directory",
    )
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _source_hashes(project_root: Path) -> dict[str, str]:
    hybrid_package = sorted((project_root / "src" / "mate_spc_hybrid").glob("*.py"))
    directional_package = sorted(
        (project_root / "src" / "mate_spc_directional").glob("*.py")
    )
    dependencies = (
        project_root / "scripts" / "run_final_confirmation_gpu.py",
        project_root / "scripts" / "run_hybrid_energy_glr_gpu.py",
        project_root / "scripts" / "run_directional_glr_gpu.py",
        project_root / "src" / "mate_spc" / "cva_mewma.py",
        project_root / "src" / "mate_spc" / "dynamic_common.py",
        project_root / "src" / "mate_spc" / "datasets.py",
        project_root / "pyproject.toml",
    )
    final_test = project_root / "tests" / "test_run_final_confirmation_gpu.py"
    optional_tests = [final_test] if final_test.is_file() else []
    paths = tuple(
        sorted(
            set(
                hybrid_package
                + directional_package
                + list(dependencies)
                + optional_tests
            )
        )
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"final protocol source dependency is missing: {missing}")
    return {
        path.relative_to(project_root).as_posix(): base._sha256_file(path)
        for path in paths
    }


def _parent_artifact_record(
    completion: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("hybrid completion has no artifact map")
    record = artifacts.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"hybrid completion does not authenticate {name!r}")
    sha256 = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"hybrid completion has invalid SHA-256 for {name!r}")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"hybrid completion has invalid size for {name!r}")
    return {"size_bytes": size_bytes, "sha256": sha256}


def _require_exact_number(value: object, expected: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    if float(value) != float(expected):
        raise ValueError(f"{label} changed: expected {expected!r}, got {value!r}")


def _verify_parent_completion(hybrid_dir: Path) -> dict[str, object]:
    """Authenticate the complete frozen parent experiment and its freeze fields."""

    root = hybrid_dir.resolve()
    completion_path = root / "completion.json"
    completion, completion_fingerprint = base._read_json_with_fingerprint(
        completion_path,
        fingerprint_field="completion_fingerprint",
    )
    completion_sha256 = base._sha256_file(completion_path)
    if completion_sha256 != EXPECTED_PARENT_COMPLETION_SHA256:
        raise ValueError("hybrid parent completion SHA-256 is not the frozen value")
    if completion_fingerprint != EXPECTED_PARENT_COMPLETION_FINGERPRINT:
        raise ValueError("hybrid parent completion fingerprint is not frozen")
    if completion.get("decision") != PARENT_DECISION:
        raise ValueError("hybrid parent did not pass all frozen audit gates")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(PARENT_ARTIFACTS):
        raise ValueError("hybrid completion artifact set differs from the frozen set")
    phase_fingerprint = completion.get("phase_fingerprint")
    if phase_fingerprint != EXPECTED_PARENT_PHASE_FINGERPRINT:
        raise ValueError("hybrid parent phase fingerprint is not frozen")
    verified = base._verify_completion(
        completion_path,
        root,
        phase_fingerprint,
    )
    if verified is None:
        raise RuntimeError("hybrid completion disappeared during authentication")

    parent_records = {
        name: _parent_artifact_record(completion, name) for name in PARENT_ARTIFACTS
    }
    audit_open, open_fingerprint = base._read_json_with_fingerprint(
        root / "audit_open_manifest.json",
        fingerprint_field="protocol_fingerprint",
    )
    if open_fingerprint != phase_fingerprint:
        raise ValueError("hybrid completion and audit-open fingerprints disagree")
    if audit_open.get("phase") != "audit_open_authorization":
        raise ValueError("hybrid audit-open manifest has the wrong phase")

    audit_base, audit_base_fingerprint = base._read_json_with_fingerprint(
        root / "audit_base_manifest.json",
        fingerprint_field="protocol_fingerprint",
    )
    if audit_open.get("audit_base_fingerprint") != audit_base_fingerprint:
        raise ValueError("hybrid audit-base and audit-open fingerprints disagree")
    parent_sources = audit_base.get("source_sha256")
    if not isinstance(parent_sources, Mapping) or not parent_sources:
        raise ValueError("hybrid audit-base has no frozen source hashes")
    project_root = Path(__file__).resolve().parents[1]
    for relative, expected_sha in parent_sources.items():
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise ValueError("hybrid parent source hash map is malformed")
        source = (project_root / relative).resolve()
        if project_root != source and project_root not in source.parents:
            raise ValueError("hybrid parent source path escapes the project root")
        if not source.is_file() or base._sha256_file(source) != expected_sha:
            raise ValueError(f"hybrid frozen source changed: {relative}")

    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    audit_result = json.loads((root / "audit_result.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (calibration, aggregate, audit_result)):
        raise ValueError("hybrid parent JSON artifacts must contain objects")
    if audit_result.get("decision") != PARENT_DECISION:
        raise ValueError("hybrid audit result and completion decision disagree")
    if aggregate.get("audit", {}).get("decision") != PARENT_DECISION:
        raise ValueError("hybrid aggregate does not record the passing audit")

    selected = aggregate.get("selected_hyperparameters")
    if not isinstance(selected, Mapping):
        raise ValueError("hybrid aggregate has no selected hyperparameters")
    for label, value in (
        ("aggregate ridge lambda", selected.get("ridge_lambda")),
        ("calibration ridge lambda", calibration.get("ridge_lambda")),
    ):
        _require_exact_number(value, FROZEN_RIDGE_LAMBDA, label)
    for label, value in (
        ("aggregate kappa", selected.get("kappa")),
        ("calibration kappa", calibration.get("kappa")),
    ):
        _require_exact_number(value, FROZEN_KAPPA, label)
    for label, value in (
        ("aggregate threshold", selected.get("threshold")),
        ("calibration threshold", calibration.get("threshold")),
        ("audit-open threshold", audit_open.get("threshold")),
    ):
        _require_exact_number(value, EXPECTED_THRESHOLD, label)
    if calibration.get("comparison") != "strictly_greater_than":
        raise ValueError("hybrid calibration comparison is not strict >")
    if audit_open.get("comparison") != "strictly_greater_than":
        raise ValueError("hybrid audit-open comparison is not strict >")
    if int(calibration.get("rank_one_based", -1)) != 96:
        raise ValueError("hybrid calibration is not the frozen q96 threshold")
    if int(audit_open.get("rank_one_based", -1)) != 96:
        raise ValueError("hybrid audit-open is not the frozen q96 threshold")

    linked = {
        "calibration.json": audit_open.get("calibration_sha256"),
        "calibration_scores.npz": audit_open.get("calibration_scores_sha256"),
        "cuda_validation.json": audit_open.get("cuda_validation_sha256"),
        "hybrid_glr_calibrated.npz": audit_open.get("calibrated_model_sha256"),
    }
    for name, expected_sha in linked.items():
        if expected_sha != parent_records[name]["sha256"]:
            raise ValueError(f"hybrid audit-open hash link changed for {name}")

    for name in ("hybrid_transformer.npz", "hybrid_glr_calibrated.npz"):
        path = root / name
        record = parent_records[name]
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"frozen {name} size changed")
        if base._sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen {name} SHA-256 changed")
    transformer_record = parent_records["hybrid_transformer.npz"]
    if transformer_record != {
        "size_bytes": EXPECTED_TRANSFORMER_SIZE,
        "sha256": EXPECTED_TRANSFORMER_SHA256,
    }:
        raise ValueError("hybrid transformer is not the explicitly frozen artifact")
    calibrated_record = parent_records["hybrid_glr_calibrated.npz"]
    if calibrated_record != {
        "size_bytes": EXPECTED_CALIBRATED_MODEL_SIZE,
        "sha256": EXPECTED_CALIBRATED_MODEL_SHA256,
    }:
        raise ValueError("calibrated GLR model is not the explicitly frozen artifact")

    return {
        "root": str(root),
        "completion": completion,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha256,
        "completion_fingerprint": completion_fingerprint,
        "phase_fingerprint": phase_fingerprint,
        "artifacts": parent_records,
        "audit_open": audit_open,
        "audit_base": audit_base,
        "calibration": calibration,
        "aggregate": aggregate,
    }


def _load_and_validate_frozen_method(
    hybrid_dir: Path,
    parent: Mapping[str, object],
):
    transformer_path = hybrid_dir / "hybrid_transformer.npz"
    model_path = hybrid_dir / "hybrid_glr_calibrated.npz"
    transformer, transformer_metadata = load_hybrid_feature_transformer(transformer_path)
    model, model_metadata = load_directional_glr_model(model_path)

    if transformer.directional_feature_count_ != EXPECTED_DIRECTIONAL_FEATURE_COUNT:
        raise ValueError("frozen transformer does not retain 1080 directional features")
    if transformer.energy_feature_count_ != EXPECTED_ENERGY_FEATURE_COUNT:
        raise ValueError("frozen transformer does not retain 768 energy features")
    if ENERGY_RAW_FEATURE_COUNT != EXPECTED_ENERGY_FEATURE_COUNT:
        raise RuntimeError("current energy feature definition is not the frozen 768")
    if transformer.output_feature_count_ != EXPECTED_HYBRID_FEATURE_COUNT:
        raise ValueError("frozen transformer does not output 1848 features")
    if transformer.valid_start_ != EXPECTED_VALID_START:
        raise ValueError("frozen transformer valid_start is not 63")
    if model.feature_count != EXPECTED_HYBRID_FEATURE_COUNT:
        raise ValueError("frozen GLR model does not accept 1848 features")
    if tuple(int(value) for value in model.fault_ids) != tuple(FAULT_IDS):
        raise ValueError("frozen GLR fault heads changed")
    if model.threshold is None:
        raise ValueError("frozen GLR model has no calibrated threshold")
    _require_exact_number(model.threshold, EXPECTED_THRESHOLD, "model threshold")
    if not np.array_equal(
        np.asarray(model.kappa),
        np.full(len(FAULT_IDS), FROZEN_KAPPA, dtype=np.float64),
    ):
        raise ValueError("frozen GLR model kappa changed")
    _require_exact_number(
        model_metadata.get("ridge_lambda"),
        FROZEN_RIDGE_LAMBDA,
        "model metadata ridge lambda",
    )
    if model_metadata.get("feature_transformer_sha256") != parent["artifacts"][
        "hybrid_transformer.npz"
    ]["sha256"]:
        raise ValueError("model metadata points to another hybrid transformer")
    if model_metadata.get("audit_base_fingerprint") != parent["audit_open"].get(
        "audit_base_fingerprint"
    ):
        raise ValueError("model metadata belongs to another hybrid audit")
    if int(transformer_metadata.get("directional_feature_count", -1)) != (
        EXPECTED_DIRECTIONAL_FEATURE_COUNT
    ):
        raise ValueError("transformer metadata directional count changed")
    if int(transformer_metadata.get("retained_energy_feature_count", -1)) != (
        EXPECTED_ENERGY_FEATURE_COUNT
    ):
        raise ValueError("transformer metadata energy count changed")
    if int(transformer_metadata.get("hybrid_feature_count", -1)) != (
        EXPECTED_HYBRID_FEATURE_COUNT
    ):
        raise ValueError("transformer metadata hybrid count changed")
    return transformer, transformer_metadata, model, model_metadata


def _final_protocol(
    *,
    parent: Mapping[str, object],
    checksums: Mapping[str, str],
    source_hashes: Mapping[str, str],
    batch_size: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "phase": "final_confirmation_authorization",
        "method": METHOD,
        "parent_hybrid": {
            "completion_path": parent["completion_path"],
            "completion_sha256": parent["completion_sha256"],
            "completion_fingerprint": parent["completion_fingerprint"],
            "phase_fingerprint": parent["phase_fingerprint"],
            "required_decision": PARENT_DECISION,
            "artifacts": parent["artifacts"],
        },
        "source_sha256": dict(source_hashes),
        "dataset": {
            "name": "Rieth Tennessee Eastman Process simulation archive",
            "doi": DATASET_DOI,
            "dataset_md5": dict(checksums),
            "loader_partition_decode_caveat": (
                "pyreadr decodes a complete RData partition before the loader "
                "returns only the explicitly selected rows"
            ),
        },
        "frozen_method": {
            "ridge_lambda": FROZEN_RIDGE_LAMBDA,
            "kappa_per_head": [FROZEN_KAPPA] * len(FAULT_IDS),
            "threshold": EXPECTED_THRESHOLD,
            "threshold_source": "unchanged healthy FF-testing q96 from runs 301-400",
            "comparison": "strictly_greater_than",
            "fault_heads": list(FAULT_IDS),
            "fault_onset_zero_based": FAULT_START,
            "persistence": (
                f"{PERSISTENCE_WINDOW}_of_{PERSISTENCE_WINDOW}_per_head_then_"
                "family_maximum"
            ),
            "directional_feature_count": EXPECTED_DIRECTIONAL_FEATURE_COUNT,
            "energy_feature_count": EXPECTED_ENERGY_FEATURE_COUNT,
            "total_feature_count": EXPECTED_HYBRID_FEATURE_COUNT,
            "valid_start_zero_based": EXPECTED_VALID_START,
            "first_alarm_eligible_zero_based": (
                EXPECTED_VALID_START + PERSISTENCE_WINDOW - 1
            ),
        },
        "final_split": {
            "healthy": ["fault_free_testing", list(FINAL_RUNS)],
            "faults": ["faulty_testing", list(FAULT_IDS), list(FINAL_RUNS)],
            "sequence_truncation_half_open": [0, SEQUENCE_STOP],
            "expected_source_sequence_length": EXPECTED_TEST_LENGTH,
            "total_sequences": len(FINAL_RUNS) * (1 + len(FAULT_IDS)),
        },
        "execution": {
            "device": "cuda",
            "dtype": "float64",
            "tf32": False,
            "batch_size": batch_size,
            "cpu_cuda_validation_required": True,
            "cpu_cuda_atol": CPU_GPU_TOLERANCE,
        },
        "gates": {
            "healthy_alarm_runs_maximum": FINAL_HEALTH_MAXIMUM,
            "per_fault_detected_runs_minimum": FINAL_PER_FAULT_MINIMUM,
            "total_fault_detected_runs_minimum": FINAL_TOTAL_MINIMUM,
            "all_required": True,
            "pass_decision": PASS_DECISION,
            "failure_decision": FAIL_DECISION,
        },
        "prohibitions": {
            "training": True,
            "calibration": True,
            "threshold_change": True,
            "hyperparameter_selection": True,
            "feature_change": True,
            "gate_change": True,
            "rerun_after_completed_result": True,
        },
        "data_governance": {
            "manifest_written_before_any_401_500_load_runs_call": True,
            "runs_401_500_not_previously_selected_transformed_or_scored": True,
            "runs_401_500_are_one_shot_final_confirmation": True,
            "no_remaining_rieth_data_are_authorized_for_same_dataset_retuning": True,
            "fault_runs_301_400_remain_administratively_sealed": True,
            "post_final_failure_action": "report failure; do not retune on Rieth",
        },
    }


def _decision(
    health_alarm_runs: int,
    per_fault_detected: Sequence[int],
    total_fault_detected: int,
) -> str:
    values = tuple(int(value) for value in per_fault_detected)
    if len(values) != len(FAULT_IDS):
        raise ValueError("one final detection count is required for every fault head")
    if health_alarm_runs < 0 or total_fault_detected < 0 or min(values) < 0:
        raise ValueError("final gate counts cannot be negative")
    passed = (
        int(health_alarm_runs) <= FINAL_HEALTH_MAXIMUM
        and min(values) >= FINAL_PER_FAULT_MINIMUM
        and int(total_fault_detected) >= FINAL_TOTAL_MINIMUM
    )
    return PASS_DECISION if passed else FAIL_DECISION


def _final_summary(
    health_records: Sequence[Mapping[str, object]],
    fault_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary = base._audit_summary(health_records, fault_records)
    by_fault = summary["faults"]["by_fault"]
    per_fault = [
        int(by_fault[f"{fault_id:02d}"]["detected_runs"]) for fault_id in FAULT_IDS
    ]
    decision = _decision(
        int(summary["healthy"]["runs_with_any_alarm"]),
        per_fault,
        int(summary["faults"]["overall"]["detected_runs"]),
    )
    passed = decision == PASS_DECISION
    if bool(summary["passed"]) != passed:
        raise RuntimeError("final gates disagree with the established audit summary")
    return {**summary, "passed": passed, "decision": decision}


def _feature_mapping(transformer, runs: Iterable[SensorSequence]) -> dict[str, np.ndarray]:
    return {
        name: features
        for name, features in transformer.transform_named_runs(
            (run.name, np.asarray(run.x, dtype=np.float64)) for run in runs
        )
    }


def _report(result: Mapping[str, object], execution: Mapping[str, object]) -> str:
    healthy = result["healthy"]
    faults = result["faults"]
    lines = [
        "# Frozen hybrid GLR-CUSUM final confirmation",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "The previously calibrated 1848-feature hybrid model was applied once to ",
        "Rieth runs 401-500. No fit, calibration, threshold selection, feature ",
        "selection, or gate selection was performed in this final phase.",
        "",
        "| Gate | Result | Frozen requirement | Passed |",
        "|---|---:|---:|:---:|",
        (
            f"| Healthy runs with an alarm | {healthy['runs_with_any_alarm']}/100 | "
            f"<= {FINAL_HEALTH_MAXIMUM} | {healthy['passed']} |"
        ),
        (
            "| Total fault runs detected | "
            f"{faults['overall']['detected_runs']}/300 | >= {FINAL_TOTAL_MINIMUM} | "
            f"{faults['total_passed']} |"
        ),
        (
            "| Minimum detected in one fault | "
            f"{faults['minimum_detected_runs_across_faults']}/100 | "
            f">= {FINAL_PER_FAULT_MINIMUM} | {faults['per_fault_passed']} |"
        ),
        "",
        "| Fault | Detected | Median delay (samples) | Attribution matches |",
        "|---:|---:|---:|---:|",
    ]
    for fault_id in FAULT_IDS:
        item = faults["by_fault"][f"{fault_id:02d}"]
        median = item["conditional_median_delay_samples"]
        median_text = "NA" if median is None else f"{float(median):.3f}"
        lines.append(
            f"| {fault_id} | {item['detected_runs']}/100 | {median_text} | "
            f"{item['attribution_matches_fault_runs']}/100 |"
        )
    validation = execution["cuda_validation"]
    score = execution["scoring"]
    lines.extend(
        [
            "",
            "## Execution and interpretation",
            "",
            f"- Device: {score['device_name']}.",
            "- Scoring dtype: CUDA float64; TF32 disabled.",
            f"- Frozen ridge lambda: {FROZEN_RIDGE_LAMBDA}.",
            f"- Frozen kappa: {FROZEN_KAPPA}.",
            f"- Frozen strict threshold: {EXPECTED_THRESHOLD:.15g}.",
            f"- CPU/CUDA alarm equality: {validation['cpu_cuda_alarm_exact']}.",
            "- Every selected sequence was truncated to rows 0:500 before feature ",
            "  transformation; rows 500:960 were not transformed or scored.",
            "- No remaining Rieth data are authorized for same-dataset retuning. ",
            "  Failure must be reported without changing this method.",
            "- Archive caveat: pyreadr decodes a complete RData partition before ",
            "  returning the requested row subset. The selection and scoring boundary ",
            "  is strict, but partition bytes are decoded by the loader.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_final_scores(path: Path) -> None:
    required = {
        "run_names",
        "fault_ids",
        "simulation_runs",
        "logits",
        "standardized_logits",
        "cusum",
        "component_persistence",
        "envelope",
        "eligible",
        "alarm",
        "threshold",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("final score archive fields differ from the frozen schema")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("final score archive contains an object array")
    count = len(FINAL_RUNS) * (1 + len(FAULT_IDS))
    heads = len(FAULT_IDS)
    vector_shapes = {
        "run_names": (count,),
        "fault_ids": (count,),
        "simulation_runs": (count,),
        "envelope": (count, SEQUENCE_STOP),
        "eligible": (count, SEQUENCE_STOP),
        "alarm": (count, SEQUENCE_STOP),
        "threshold": (1,),
    }
    head_shapes = {
        name: (count, SEQUENCE_STOP, heads)
        for name in ("logits", "standardized_logits", "cusum", "component_persistence")
    }
    for name, expected in {**vector_shapes, **head_shapes}.items():
        if arrays[name].shape != expected:
            raise ValueError(f"final score field {name!r} has shape {arrays[name].shape}")
    _require_exact_number(arrays["threshold"][0], EXPECTED_THRESHOLD, "score threshold")
    expected_faults = np.asarray(
        [0] * len(FINAL_RUNS)
        + [fault_id for fault_id in FAULT_IDS for _ in FINAL_RUNS],
        dtype=np.int64,
    )
    expected_runs = np.asarray(list(FINAL_RUNS) * (1 + len(FAULT_IDS)), dtype=np.int64)
    if not np.array_equal(arrays["fault_ids"], expected_faults):
        raise ValueError("final score fault ordering changed")
    if not np.array_equal(arrays["simulation_runs"], expected_runs):
        raise ValueError("final score run ordering changed")
    expected_alarm = arrays["eligible"].astype(bool) & (
        arrays["envelope"] > EXPECTED_THRESHOLD
    )
    if not np.array_equal(arrays["alarm"].astype(bool), expected_alarm):
        raise ValueError("final score alarms do not use the frozen strict threshold")


def _verify_final_completion(
    output_dir: Path,
    protocol_fingerprint: str | None = None,
) -> dict[str, object] | None:
    completion_path = output_dir / "completion.json"
    if not completion_path.exists():
        return None
    completion = base._verify_completion(
        completion_path,
        output_dir,
        protocol_fingerprint,
    )
    if completion is None:
        return None
    if completion.get("decision") not in {PASS_DECISION, FAIL_DECISION}:
        raise ValueError("final completion has an invalid decision")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(FINAL_ARTIFACTS):
        raise ValueError("final completion artifact set differs from the frozen set")
    expected_files = set(FINAL_ARTIFACTS) | {"completion.json"}
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    actual_dirs = [path.name for path in output_dir.iterdir() if path.is_dir()]
    if actual_files != expected_files or actual_dirs:
        raise ValueError("completed final output directory contains unexpected entries")
    result = json.loads((output_dir / "final_result.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_dir / "cuda_validation.json").read_text(encoding="utf-8")
    )
    if not isinstance(result, dict) or result.get("decision") != completion.get("decision"):
        raise ValueError("final result and completion decision disagree")
    if bool(result.get("passed")) != (completion.get("decision") == PASS_DECISION):
        raise ValueError("final result pass flag and decision disagree")
    if not isinstance(validation, dict) or not bool(validation.get("passed")):
        raise ValueError("completed final CUDA validation did not pass")
    _validate_final_scores(output_dir / "final_scores.npz")
    return completion


def _refuse_unsafe_partial_reentry(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    names = {path.name for path in output_dir.iterdir()}
    if names and names != {"final_manifest.json"}:
        raise RuntimeError(
            "final output is incomplete or contains extra files; preserve it for "
            "audit and do not silently rescore the one-shot split"
        )


def _run_final(
    args: argparse.Namespace,
    *,
    project_root: Path,
    hybrid_dir: Path,
) -> dict[str, object]:
    parent = _verify_parent_completion(hybrid_dir)
    source_hashes = _source_hashes(project_root)
    dataset_md5 = parent["audit_base"].get("dataset_md5")
    if not isinstance(dataset_md5, Mapping) or not dataset_md5:
        raise ValueError("authenticated hybrid audit-base has no dataset MD5 map")
    protocol = _final_protocol(
        parent=parent,
        checksums=dataset_md5,
        source_hashes=source_hashes,
        batch_size=args.batch_size,
    )
    manifest_path = args.output_dir / "final_manifest.json"
    protocol_fingerprint = base._load_or_create_manifest(
        manifest_path,
        protocol,
        require_empty_parent=True,
    )

    completed = _verify_final_completion(args.output_dir, protocol_fingerprint)
    if completed is not None:
        print("final confirmation is already complete", flush=True)
        return completed
    _refuse_unsafe_partial_reentry(args.output_dir)

    transformer, transformer_metadata, model, model_metadata = (
        _load_and_validate_frozen_method(hybrid_dir, parent)
    )
    archive = RiethTEPArchive(args.data_dir, verify_checksums=True)
    observed_md5 = archive.verify(("fault_free_testing", "faulty_testing"))
    expected_md5 = {
        name: dataset_md5[name]
        for name in ("fault_free_testing", "faulty_testing")
    }
    if observed_md5 != expected_md5:
        raise ValueError("final archive MD5 values differ from the frozen parent")

    # This is the first point at which any selected 401--500 values are loaded.
    print("loading one-shot final healthy/fault runs 401..500", flush=True)
    healthy_full = base._run_ids(
        archive.load_runs(
            "fault_free_testing",
            {0: FINAL_RUNS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        FINAL_RUNS,
        partition="fault_free_testing",
        expected_length=EXPECTED_TEST_LENGTH,
        label="one-shot final healthy confirmation",
    )
    faults_full = base._fault_run_ids(
        archive.load_runs(
            "faulty_testing",
            {fault_id: FINAL_RUNS for fault_id in FAULT_IDS},
            fault_start=FAULT_START,
            split_manipulated=False,
        ),
        FINAL_RUNS,
        expected_length=EXPECTED_TEST_LENGTH,
        label="one-shot final fault confirmation",
    )
    healthy = tuple(base._truncate_run(run, SEQUENCE_STOP) for run in healthy_full)
    faults = tuple(base._truncate_run(run, SEQUENCE_STOP) for run in faults_full)
    runs = healthy + faults

    transform_started = time.perf_counter()
    features = _feature_mapping(transformer, runs)
    transform_seconds = time.perf_counter() - transform_started
    if len(features) != 400 or any(
        value.shape != (SEQUENCE_STOP, EXPECTED_HYBRID_FEATURE_COUNT)
        for value in features.values()
    ):
        raise RuntimeError("final transformed sequence dimensions changed")

    score_started = time.perf_counter()
    scored = score_directional_glr_cuda(
        model,
        features,
        valid_start=transformer.valid_start_,
        device=args.device,
        batch_size=args.batch_size,
    )
    score_wall_seconds = time.perf_counter() - score_started
    representative_name = healthy[0].name
    validation_started = time.perf_counter()
    validation_raw = validate_directional_glr_cuda(
        model,
        features[representative_name],
        valid_start=transformer.valid_start_,
        device=args.device,
        atol=CPU_GPU_TOLERANCE,
    )
    validation_wall_seconds = time.perf_counter() - validation_started
    validation_passed = bool(
        validation_raw["cpu_cuda_alarm_exact"]
        and validation_raw["duplicate_alarm_exact"]
        and validation_raw["batch_partition_alarm_exact"]
    )
    if not validation_passed:
        raise RuntimeError("final CPU/CUDA or CUDA batch invariance validation failed")
    validation = {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "passed": True,
        "protocol_fingerprint": protocol_fingerprint,
        "representative_run": representative_name,
        "wall_seconds": validation_wall_seconds,
        **validation_raw,
    }

    threshold = float(model.threshold)
    result = {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "phase": "one_shot_final_confirmation",
        "protocol_fingerprint": protocol_fingerprint,
        "method": METHOD,
        "frozen_hyperparameters": {
            "ridge_lambda": FROZEN_RIDGE_LAMBDA,
            "kappa": FROZEN_KAPPA,
            "threshold": threshold,
            "comparison": "strictly_greater_than",
        },
        **_final_summary(
            base._health_records(healthy, scored.by_run, threshold=threshold),
            base._fault_records(faults, scored.by_run, threshold=threshold),
        ),
        "data_boundary": {
            "selected_runs": [401, 500],
            "runs_401_500_not_previously_selected_transformed_or_scored": True,
            "all_selected_sequences_truncated_to_0_500_before_transform": True,
            "rows_500_959_not_transformed_or_scored": True,
            "training_performed": False,
            "calibration_performed": False,
            "selection_performed": False,
            "no_remaining_rieth_data_authorized_for_same_dataset_retuning": True,
            "fault_runs_301_400_administratively_sealed": True,
            "archive_decode_caveat": (
                "pyreadr decodes a complete RData partition before selected rows "
                "are returned"
            ),
        },
    }
    execution = {
        "schema_version": 1,
        "analysis_status": ANALYSIS_STATUS,
        "protocol_fingerprint": protocol_fingerprint,
        "device_name": scored.execution["device_name"],
        "dtype": "float64",
        "tf32": False,
        "feature_transform_seconds": transform_seconds,
        "scoring": {
            "score_wall_seconds": score_wall_seconds,
            **scored.execution,
        },
        "cuda_validation": validation,
        "frozen_artifact_metadata": {
            "transformer": transformer_metadata,
            "model": model_metadata,
        },
    }

    base._atomic_json(args.output_dir / "final_result.json", result)
    base._atomic_json(args.output_dir / "cuda_validation.json", validation)
    base._atomic_json(args.output_dir / "gpu_execution.json", execution)
    base._atomic_npz(
        args.output_dir / "final_scores.npz",
        base._pack_paths(runs, scored.by_run, threshold=threshold),
    )
    base._atomic_text(args.output_dir / "REPORT.md", _report(result, execution))
    base._write_completion(
        args.output_dir / "completion.json",
        args.output_dir,
        protocol_fingerprint,
        str(result["decision"]),
        FINAL_ARTIFACTS,
    )
    verified = _verify_final_completion(args.output_dir, protocol_fingerprint)
    if verified is None:
        raise RuntimeError("final completion disappeared during verification")
    print(f"final confirmation decision: {result['decision']}", flush=True)
    return verified


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = Path(__file__).resolve().parents[1]
    hybrid_dir = (
        args.hybrid_dir
        if args.hybrid_dir is not None
        else project_root / "outputs" / "hybrid_energy_glr_gpu_v1"
    ).resolve()
    args.output_dir = args.output_dir.resolve()
    if (
        args.output_dir == hybrid_dir
        or hybrid_dir in args.output_dir.parents
        or args.output_dir in hybrid_dir.parents
    ):
        raise ValueError("final output directory must not overlap the frozen hybrid root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _run_final(
        args,
        project_root=project_root,
        hybrid_dir=hybrid_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
