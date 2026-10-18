#!/usr/bin/env python
"""Atomic preflight and one-shot Mode2 development runner for CP-ETC v5.1."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # Must precede NumPy and the numeric backend on Windows.

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "cp_etc_scan_v5_mode2_development.json"
AMENDMENT_PATH = PROJECT_ROOT / "protocols" / "cp_etc_scan_v5_1_mode2_development.json"
SOURCE_PATH = PROJECT_ROOT / "data" / "extended_tep_dtu_mode2" / "TEP_Mode2.h5"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cp_etc_scan_v5_1"
BASE_PROTOCOL_SHA256 = "296cacea8ef2b9d2e2bda3aeb5805a9e3627ad8299b564e1ac20b6a0dda5a804"
AMENDMENT_SHA256 = "55cd6dcb09409f53db4f078f2441f18a7659c1ce97b7165876d3f2a891a37923"
SOURCE_SHA256 = "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
SOURCE_SIZE = 23_897_125_720
ACCESS_PLAN_SHA256 = "387ece90e94e7161767374be34ed6d032c10f45a60353c9de74be7b493ca173f"
V5_DATA_CONTRACT_NO_GO_SHA256 = "d8f0578ebae0d0dbb4ae253be38927c8ecfea1df0591ca329a048f8ecb134160"
TEST_PATHS = tuple(
    PROJECT_ROOT / "tests" / name for name in (
        "test_cpetc_monitor_v5.py",
        "test_cpetc_features_v5.py",
        "test_cpetc_model_v5.py",
        "test_cpetc_data_v5.py",
        "test_cpetc_backend_v5.py",
        "test_run_cpetc_gpu_v5_1.py",
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"non-JSON runner value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        _plain_json(value), ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise ValueError(f"nonfinite JSON constant forbidden: {token}")
    value = json.loads(path.read_text("utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> str:
    payload = _canonical_bytes(value)
    _atomic_write(path, payload)
    return _hash_bytes(payload)


def _commit_json_artifact(path: Path, value: Any, *, schema: str) -> dict[str, Any]:
    document_sha256 = _write_json(path, value)
    sidecar = path.with_name(path.name + ".sha256")
    _atomic_write(
        sidecar, f"{document_sha256}  {path.name}\n".encode("ascii")
    )
    sidecar_sha256 = _hash_file(sidecar)
    commit = path.with_name(path.name + ".commit.json")
    commit_document = {
        "schema": schema,
        "schema_version": 1,
        "document_name": path.name,
        "document_sha256": document_sha256,
        "sidecar_name": sidecar.name,
        "sidecar_sha256": sidecar_sha256,
    }
    commit_sha256 = _write_json(commit, commit_document)
    # Re-read all three bytes before any guarded action is allowed.
    if (
        _hash_file(path) != document_sha256
        or sidecar.read_text("utf-8").strip() != f"{document_sha256}  {path.name}"
        or _hash_file(sidecar) != sidecar_sha256
        or _strict_json(commit) != commit_document
        or _hash_file(commit) != commit_sha256
    ):
        raise RuntimeError(f"atomic artifact verification failed for {path.name}")
    return {
        "path": str(path.resolve()),
        "sha256": document_sha256,
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sidecar_sha256,
        "commit_path": str(commit.resolve()),
        "commit_sha256": commit_sha256,
    }


def _load_backend() -> ModuleType:
    source = PROJECT_ROOT / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from mate_spc_cpetc import backend_v5
    return backend_v5


def _sidecar_path(protocol_path: Path) -> Path:
    return protocol_path.with_name(protocol_path.name + ".sha256")


def _commit_path(protocol_path: Path) -> Path:
    return protocol_path.with_name(protocol_path.name + ".commit.json")


def _verify_one_protocol(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _hash_file(path)
    if observed != expected_sha256:
        raise ValueError(f"protocol SHA-256 mismatch for {path.name}")
    sidecar = _sidecar_path(path)
    commit_path = _commit_path(path)
    expected_line = f"{observed}  {path.name}"
    if sidecar.read_text("utf-8").strip() != expected_line:
        raise ValueError(f"protocol sidecar mismatch for {path.name}")
    sidecar_sha = _hash_file(sidecar)
    commit = _strict_json(commit_path)
    expected_commit = {
        "protocol_name": path.name,
        "protocol_sha256": observed,
        "schema": "cp_etc_scan_protocol_atomic_commit",
        "schema_version": 1,
        "sidecar_name": sidecar.name,
        "sidecar_sha256": sidecar_sha,
    }
    if commit != expected_commit:
        raise ValueError(f"atomic protocol commit mismatch for {path.name}")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "sidecar_path": str(sidecar.resolve()),
        "sidecar_sha256": sidecar_sha,
        "commit_path": str(commit_path.resolve()),
        "commit_sha256": _hash_file(commit_path),
        "document": _strict_json(path),
    }


def _verify_protocol_bundle(backend: ModuleType) -> dict[str, Any]:
    base = _verify_one_protocol(BASE_PROTOCOL_PATH, BASE_PROTOCOL_SHA256)
    amendment = _verify_one_protocol(AMENDMENT_PATH, AMENDMENT_SHA256)
    bdoc = base["document"]
    adoc = amendment["document"]
    if (
        bdoc.get("schema") != "cp_etc_scan_preregistered_development_protocol"
        or bdoc.get("schema_version") != 5
        or bdoc.get("protocol_id") != "cp-etc-scan-v5-mode2-development"
        or bdoc.get("candidate", {}).get("candidate_count") != 1
        or bdoc.get("candidate", {}).get("candidate_id") != backend.CANDIDATE_ID
        or bdoc.get("tail_mapping_and_scan", {}).get("operational_loo") is not False
        or bdoc.get("source_files", {}).get("mode2", {}).get("sha256") != SOURCE_SHA256
    ):
        raise ValueError("base protocol semantic binding failed")
    frozen_gates = bdoc.get("development_go_gates", {})
    if (
        frozen_gates.get("healthy", {}).get("total_false_alarms_max") != 10
        or frozen_gates.get("healthy", {}).get("per_fold_false_alarms_max") != 4
        or frozen_gates.get("each_fault_3_9_15", {}).get("aggregate_detection_rate_min") != 0.85
        or frozen_gates.get("each_fault_3_9_15", {}).get("per_fold_detection_rate_min") != 0.7
        or frozen_gates.get("each_fault_3_9_15", {}).get("aggregate_median_delay_samples_max") != 140
        or frozen_gates.get("each_fault_3_9_15", {}).get("aggregate_nearest_rank_p90_delay_samples_max") != 220
    ):
        raise ValueError("base protocol GO gates changed")
    expected_fold_lists = {
        str(key): list(value) for key, value in backend.V51_FOLD_RUN_IDS.items()
    }
    if (
        adoc.get("schema") != "cp_etc_scan_preregistered_development_amendment"
        or adoc.get("schema_version") != 1
        or adoc.get("protocol_id") != backend.PROTOCOL_ID
        or adoc.get("base_protocol", {}).get("sha256") != BASE_PROTOCOL_SHA256
        or adoc.get("replacement_folds", {}).get("fold_run_ids") != expected_fold_lists
        or adoc.get("replacement_folds", {}).get("run_to_fold_canonical_json_sha256")
        != backend.EXPECTED_FOLD_MAP_SHA256
        or adoc.get("v5_disposition", {}).get("decision_record_sha256")
        != V5_DATA_CONTRACT_NO_GO_SHA256
        or adoc.get("candidate_count") != 1
        or adoc.get("method_change") is not False
    ):
        raise ValueError("v5.1 amendment semantic binding failed")
    if backend._fold_map_sha256() != backend.EXPECTED_FOLD_MAP_SHA256:
        raise ValueError("backend fold hash differs from v5.1 amendment")
    base.pop("document")
    amendment.pop("document")
    return {"base": base, "amendment": amendment}


def _source_record() -> dict[str, Any]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    size = SOURCE_PATH.stat().st_size
    if size != SOURCE_SIZE:
        raise ValueError(f"Mode2 source size mismatch: expected {SOURCE_SIZE}, got {size}")
    observed = _hash_file(SOURCE_PATH)
    if observed != SOURCE_SHA256:
        raise ValueError("Mode2 source SHA-256 mismatch")
    return {
        "path": str(SOURCE_PATH.resolve()),
        "filename": SOURCE_PATH.name,
        "size_bytes": size,
        "sha256": observed,
        "raw_byte_hash_verified": True,
        "hash_method": "streaming_sha256_over_entire_file_16MiB_chunks",
        "verified_utc": _now(),
        "hdf5_opened": False,
    }


def _implementation_records() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "src" / "mate_spc_cpetc" / name
        for name in (
            "__init__.py", "data_v5.py", "features.py", "model.py",
            "monitor.py", "backend_v5.py",
        )
    ] + [
        PROJECT_ROOT / "src" / "mate_spc_pclr" / "data.py",
        PROJECT_ROOT / "pyproject.toml",
        Path(__file__).resolve(),
        *TEST_PATHS,
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): {
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        } for path in paths
    }


def _governance_records() -> dict[str, Any]:
    directory = PROJECT_ROOT / "outputs" / "cp_etc_scan_v5" / "governance"
    no_go_path = directory / "v5_data_contract_no_go.json"
    first_intent_path = directory / "mode2_first_access_intent.json"
    no_go = _strict_json(no_go_path)
    first_intent = _strict_json(first_intent_path)
    no_go_sha256 = _hash_file(no_go_path)
    if (
        no_go.get("decision") != "DATA_CONTRACT_NO_GO"
        or no_go.get("protocol_sha256") != BASE_PROTOCOL_SHA256
        or no_go.get("source_sha256") != SOURCE_SHA256
        or no_go.get("decoded_processdata_values") is not False
        or no_go.get("decoded_idv_init_values") is not False
        or first_intent.get("protocol_sha256") != BASE_PROTOCOL_SHA256
        or first_intent.get("source_sha256") != SOURCE_SHA256
        or no_go_sha256 != V5_DATA_CONTRACT_NO_GO_SHA256
    ):
        raise ValueError("old-v5 metadata governance binding is invalid")
    return {
        "v5_data_contract_no_go": {
            "path": str(no_go_path.resolve()),
            "size_bytes": no_go_path.stat().st_size,
            "sha256": no_go_sha256,
            "decision": no_go["decision"],
        },
        "mode2_first_metadata_access_intent": {
            "path": str(first_intent_path.resolve()),
            "size_bytes": first_intent_path.stat().st_size,
            "sha256": _hash_file(first_intent_path),
            "access_scope": first_intent.get("access_scope"),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _hdf5_library_manifest() -> dict[str, Any]:
    import h5py  # Library introspection only; no File object is constructed here.
    return {
        "h5py": str(h5py.__version__),
        "hdf5_library": str(h5py.version.hdf5_version),
        "api_imported_for_version_only": True,
        "file_opened": False,
    }


def _gpu_manifest() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError("unexpected nvidia-smi GPU manifest format")
        rows.append({
            "index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "driver_version": fields[3], "memory_mib": int(fields[4]),
        })
    if not rows or rows[0]["index"] != 0:
        raise RuntimeError("nvidia-smi did not report CUDA device zero")
    properties = torch.cuda.get_device_properties(0)
    return {
        "selected_device_index": 0,
        "selected_device": rows[0],
        "all_visible_devices": rows,
        "torch_total_memory_bytes": int(properties.total_memory),
        "torch_compute_capability": [int(properties.major), int(properties.minor)],
    }


def _pre_numeric_test_evidence() -> dict[str, Any]:
    pytest_arguments = [
        "-q", "-W", "error",
        *(str(path.relative_to(PROJECT_ROOT)) for path in TEST_PATHS)
    ]
    launcher = (
        "import torch,pytest;"
        f"raise SystemExit(pytest.main({pytest_arguments!r}))"
    )
    command = [sys.executable, "-c", launcher]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    combined = completed.stdout + completed.stderr
    def count(label: str) -> int:
        matches = re.findall(rf"(\d+)\s+{label}", combined)
        return int(matches[-1]) if matches else 0
    evidence = {
        "command": command,
        "human_command": subprocess.list2cmdline(command),
        "pytest_arguments": pytest_arguments,
        "torch_imported_before_test_collection": True,
        "returncode": completed.returncode,
        "passed": count("passed"),
        "skipped": count("skipped"),
        "failed": count("failed"),
        "errors": count("errors?"),
        "warnings": count("warnings?"),
        "output_sha256": _hash_bytes(combined.encode("utf-8")),
        "output_tail": combined[-4000:],
    }
    if (
        completed.returncode != 0 or evidence["failed"]
        or evidence["errors"] or evidence["warnings"]
    ):
        raise RuntimeError(
            "pre-numeric synthetic suite failed:\n" + evidence["output_tail"]
        )
    if evidence["passed"] <= 0:
        raise RuntimeError("pre-numeric synthetic suite reported no passing tests")
    return evidence


def _git_revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def preflight(*, backend: ModuleType | None = None) -> dict[str, Any]:
    numeric = _load_backend() if backend is None else backend
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocols = _verify_protocol_bundle(numeric)
    governance = _governance_records()
    source = _source_record()
    runtime = dict(numeric.cuda_preflight())
    if runtime.get("cuda_available") is not True or runtime.get("device_type") != "cuda":
        raise RuntimeError("formal v5.1 development requires a CUDA preflight pass")
    gpu = _gpu_manifest()
    tests = _pre_numeric_test_evidence()
    implementation = _implementation_records()
    hdf5_runtime = _hdf5_library_manifest()
    return {
        "schema": "cp_etc_scan_v5_1_preflight_v1",
        "created_utc": _now(),
        "protocols": protocols,
        "prior_governance": governance,
        "source": source,
        "runtime": runtime,
        "gpu": gpu,
        "pre_numeric_tests": tests,
        "environment": {
            "python": sys.version,
            "sys_executable": sys.executable,
            "sys_prefix": sys.prefix,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "numpy": _package_version("numpy"),
            "h5py": _package_version("h5py"),
            "hdf5_runtime": hdf5_runtime,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "implementation": implementation,
        "git": _git_revision(),
        "hdf5_opened": False,
    }


def _default_iteration_id() -> str:
    return datetime.now(timezone.utc).strftime("v5_1_fixed_%Y%m%dT%H%M%SZ")


def _validate_iteration_id(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError("iteration ID may contain only letters, digits, underscore, and hyphen")
    return value


def _validate_development_result(
    raw: Mapping[str, Any], backend: ModuleType,
) -> dict[str, Any]:
    result = dict(raw)
    if (
        result.get("schema") != backend.BACKEND_SCHEMA
        or result.get("protocol_id") != backend.PROTOCOL_ID
        or result.get("protocol_sha256") != AMENDMENT_SHA256
        or result.get("source_sha256") != SOURCE_SHA256
        or result.get("candidate_id") != backend.CANDIDATE_ID
        or result.get("candidate_count") != 1
        or result.get("fold_map_sha256") != backend.EXPECTED_FOLD_MAP_SHA256
        or result.get("mode1_accessed") is not False
        or result.get("decision") not in ("DEVELOPMENT_GO", "DEVELOPMENT_NO_GO")
    ):
        raise ValueError("backend development identity/result binding is invalid")
    termination = result.get("termination")
    if termination is not None:
        record_path = Path(termination.get("record_path", ""))
        if (
            termination.get("kind") != "FOLD_NUMERIC_NO_GO"
            or result.get("decision") != "DEVELOPMENT_NO_GO"
            or result.get("selected_candidate_id") is not None
            or result.get("formal_mode2_refit") is not None
            or termination.get("outer_data_scored") is not False
            or not record_path.is_file()
            or _hash_file(record_path) != termination.get("record_sha256")
        ):
            raise ValueError("fold numeric NO-GO termination binding is invalid")
        return result
    folds = result.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("development result must contain exactly five folds")
    seen: set[tuple[int, int]] = set()
    for index, fold in enumerate(folds):
        if (
            fold.get("outer_fold") != index
            or fold.get("inner_calibration_fold") != (index + 1) % 5
            or fold.get("inner_threshold", {}).get("operational_loo") is not False
            or len(fold.get("inner_threshold", {}).get("run_maxima", [])) != 20
            or len(fold.get("outer_rows", [])) != 80
        ):
            raise ValueError("fold structure or fixed-ECDF threshold changed")
        outer_artifact = fold.get("outer_result_artifact", {})
        outer_path = Path(outer_artifact.get("path", ""))
        if (
            not outer_path.is_file()
            or _hash_file(outer_path) != outer_artifact.get("sha256")
        ):
            raise ValueError("append-only outer-fold artifact is missing or changed")
        for row in fold["outer_rows"]:
            key = (int(row["idv"]), int(row["simulation_run"]))
            if key in seen:
                raise ValueError("duplicate outer prediction")
            seen.add(key)
            if (
                len(row.get("all_logits", [])) != 93
                or len(row.get("pvalues", [])) != 85
                or len(row.get("evidence", [])) != 85
                or len(row.get("scan_scores", [])) != 85
            ):
                raise ValueError("outer replay path is incomplete")
    if len(seen) != 400:
        raise ValueError("outer predictions do not cover all 400 development runs once")
    decision_artifact = result.get("development_decision_pre_refit", {})
    decision_path = Path(decision_artifact.get("path", ""))
    if (
        not decision_path.is_file()
        or _hash_file(decision_path) != decision_artifact.get("sha256")
    ):
        raise ValueError("pre-refit development decision artifact is missing or changed")
    decision = result["decision"]
    if (
        (decision == "DEVELOPMENT_GO")
        != bool(result.get("gates", {}).get("all_pass"))
        or (decision == "DEVELOPMENT_GO")
        != (result.get("selected_candidate_id") == backend.CANDIDATE_ID)
        or (decision == "DEVELOPMENT_GO")
        != (result.get("formal_mode2_refit") is not None)
    ):
        raise ValueError("decision, gates, selection, and GO-only refit disagree")
    return result


def run_development(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT, iteration_id: str | None = None,
    backend: ModuleType | None = None,
) -> dict[str, Any]:
    numeric = _load_backend() if backend is None else backend
    audit = preflight(backend=numeric)
    identifier = _validate_iteration_id(iteration_id or _default_iteration_id())
    plan = numeric.development_access_plan()
    numeric.validate_development_access_plan(plan)
    plan_sha256 = _hash_bytes(_canonical_bytes(plan))
    if (
        plan_sha256 != ACCESS_PLAN_SHA256
        or plan_sha256 != numeric.EXPECTED_ACCESS_PLAN_SHA256
    ):
        raise ValueError("formal development access-plan hash mismatch")
    directory = Path(output_root) / "development_iterations" / identifier
    if directory.exists():
        raise FileExistsError(directory)
    claim_path = Path(output_root) / "development_access.claim.json"
    global_claim = {
        "schema": "cp_etc_scan_v5_1_once_only_development_claim_v1",
        "claimed_utc": _now(),
        "iteration_id": identifier,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "fold_map_sha256": numeric.EXPECTED_FOLD_MAP_SHA256,
        "access_plan_sha256": plan_sha256,
        "candidate_id": numeric.CANDIDATE_ID,
        "implementation": audit["implementation"],
        "policy": "Any prior claim forbids a new iteration ID; an infrastructure retry requires a separately audited manual continuation and is not automated here.",
    }
    _write_json(claim_path, global_claim)
    directory.mkdir(parents=True, exist_ok=False)
    preflight_path = directory / "preflight.json"
    preflight_sha = _write_json(preflight_path, audit)
    access_intent = {
        "schema": "cp_etc_scan_v5_1_mode2_access_intent_v1",
        "committed_utc": _now(),
        "committed_before_numeric_access": True,
        "operator": getpass.getuser(),
        "phase": "development",
        "protocol_id": numeric.PROTOCOL_ID,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "source": audit["source"],
        "fold_map_sha256": numeric.EXPECTED_FOLD_MAP_SHA256,
        "candidate_id": numeric.CANDIDATE_ID,
        "candidate_count": 1,
        "access_plan": plan,
        "access_plan_sha256": plan_sha256,
        "hdf5_selector": "Mode2/SingleFault/{SimulationCompleted|SimulationStopped}/IDV{idv}/Mode2_IDVInfo_{idv}_50/Run{run_id}",
        "protocol_artifacts": audit["protocols"],
        "prior_v5_metadata_governance": audit["prior_governance"],
        "preflight_sha256": preflight_sha,
        "pre_numeric_tests": audit["pre_numeric_tests"],
        "gpu": audit["gpu"],
        "runtime": audit["runtime"],
        "environment": audit["environment"],
        "before_this_committed_intent": {
            "official_mode2_processdata_labels_values_decoded_by_v5_1_numeric_runner": False,
            "official_mode2_processdata_values_decoded_by_v5_1_numeric_runner": False,
            "official_mode2_idv_init_values_decoded_by_v5_1_numeric_runner": False,
            "official_mode2_hdf5_opened_by_v5_1_numeric_runner": False,
            "prior_metadata_only_v5_access_disclosed_separately": True,
            "synthetic_test_hdf5_fixtures_are_not_the_official_source": True,
        },
        "authorized_contract_reads_after_this_commit": {
            "processdata_labels": "decode and validate the complete label vector",
            "each_selected_idv_init": "decode and validate the complete 3x28 array",
            "each_selected_time_column": "decode and validate all 2000 processdata time values",
            "model_input_only": "project 52 process channels and retain only the authorized 500x52 analytic slice",
        },
        "mode1_access_authorized": False,
        "mode1_hdf5_opens_or_value_reads": 0,
        "mode1_magnitude_25_state": "SEALED_UNTIL_DEVELOPMENT_GO",
        "mode1_magnitude_75_state": "SEALED_UNTIL_FINAL_ARTIFACT_AND_THRESHOLD_FREEZE",
        "implementation": audit["implementation"],
        "git": audit["git"],
    }
    intent_path = directory / "access_intent.json"
    intent_record = _commit_json_artifact(
        intent_path, access_intent,
        schema="cp_etc_scan_v5_1_access_intent_atomic_commit",
    )
    intent_sha = intent_record["sha256"]
    try:
        raw = numeric.development_iteration(
            access_plan=plan,
            fold_map=numeric._canonical_fold_map(),
            artifact_root=directory / "artifacts",
            device="cuda",
        )
        result = _validate_development_result(raw, numeric)
    except BaseException as error:
        artifact_root = directory / "artifacts"
        partial_files = sorted(
            str(path.relative_to(directory)).replace("\\", "/")
            for path in artifact_root.rglob("*")
            if path.is_file()
        ) if artifact_root.exists() else []
        failure = {
            "schema": "cp_etc_scan_v5_1_development_failure_v1",
            "created_utc": _now(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "access_intent_sha256": intent_sha,
            "access_intent_commit_sha256": intent_record["commit_sha256"],
            "metrics_revealed": bool(partial_files),
            "partial_numeric_artifacts": partial_files,
            "automatic_rerun_allowed": False,
            "classification": (
                "PARTIAL_NUMERIC_OUTPUTS_REVEALED_NO_RERUN"
                if partial_files
                else "UNCLASSIFIED_DATA_OR_INFRASTRUCTURE_FAILURE_NO_AUTOMATIC_RERUN"
            ),
        }
        _write_json(directory / "failure.json", failure)
        raise
    result_path = directory / "development_result.json"
    result_sha = _write_json(result_path, result)
    _atomic_write(
        directory / "development_result.json.sha256",
        f"{result_sha}  {result_path.name}\n".encode("ascii"),
    )
    completion = {
        "schema": "cp_etc_scan_v5_1_development_completion_v1",
        "completed_utc": _now(),
        "decision": result["decision"],
        "preflight_sha256": preflight_sha,
        "access_intent_sha256": intent_sha,
        "access_intent_commit_sha256": intent_record["commit_sha256"],
        "development_result_sha256": result_sha,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "mode1_accessed": False,
    }
    completion_sha = _write_json(directory / "completion.json", completion)
    return {
        "iteration_id": identifier,
        "directory": str(directory.resolve()),
        "decision": result["decision"],
        "result_path": str(result_path.resolve()),
        "result_sha256": result_sha,
        "completion_sha256": completion_sha,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "development"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--iteration-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "preflight":
        report = preflight()
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    summary = run_development(
        output_root=args.output_root, iteration_id=args.iteration_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
