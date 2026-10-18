"""Fail-closed Mode2 backend for GPU MiniROCKET Log-Odds CUSUM v8.

Torch is deliberately imported before NumPy on Windows.  h5py is lazy and
reachable only from _load_numeric_blocks, after the runner has committed both
the exclusive claim and exact access intent.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import torch  # must precede NumPy/core imports on the RTX-5090 environment

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import random
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_SITE_PACKAGES = (
    Path(sys.executable).resolve().parent / "Lib" / "site-packages").resolve()
for _trusted_path in (ENVIRONMENT_SITE_PACKAGES, PROJECT_ROOT):
    while str(_trusted_path) in sys.path:
        sys.path.remove(str(_trusted_path))
sys.path[:0] = [str(ENVIRONMENT_SITE_PACKAGES), str(PROJECT_ROOT)]
_HASH_CONTRACT_PATH = PROJECT_ROOT / "src" / "mate_spc" / "hash_contract.py"
_HASH_CONTRACT_NAME = "_mate_spc_minirocket_v8_hash_contract"
_HASH_CONTRACT_SOURCE_SHA256 = "c3dcae32194e892f758c62c99853c0eb15691f13df0ac9ef0b094ecb92a02503"
_HASH_CONTRACT_SOURCE_SIZE = 6_680
_HASH_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    _HASH_CONTRACT_NAME, _HASH_CONTRACT_PATH)
if _HASH_CONTRACT_SPEC is None or _HASH_CONTRACT_SPEC.loader is None:
    raise ImportError(f"cannot load hash contract from {_HASH_CONTRACT_PATH}")
_HASH_CONTRACT = importlib.util.module_from_spec(_HASH_CONTRACT_SPEC)
sys.modules[_HASH_CONTRACT_NAME] = _HASH_CONTRACT
_HASH_CONTRACT_PAYLOAD = _HASH_CONTRACT_PATH.read_bytes()
if (len(_HASH_CONTRACT_PAYLOAD) != _HASH_CONTRACT_SOURCE_SIZE or
        hashlib.sha256(_HASH_CONTRACT_PAYLOAD).hexdigest() != _HASH_CONTRACT_SOURCE_SHA256):
    raise RuntimeError("hash-contract source bytes differ from audited identity")
exec(compile(_HASH_CONTRACT_PAYLOAD, str(_HASH_CONTRACT_PATH), "exec", dont_inherit=True),
     _HASH_CONTRACT.__dict__)
canonical_json_file_bytes = _HASH_CONTRACT.canonical_json_file_bytes
canonical_json_sha256 = _HASH_CONTRACT.canonical_json_sha256
sha256_file = _HASH_CONTRACT.sha256_file
strict_json_from_bytes = _HASH_CONTRACT.strict_json_from_bytes
OFFICIAL_MODE2_PATH = (PROJECT_ROOT / "data" / "extended_tep_dtu_mode2" / "TEP_Mode2.h5").resolve()
EXPECTED_GLOBAL_CLAIM_PATH = (
    PROJECT_ROOT / "outputs" / "gpu_minirocket_logodds_v8" /
    "development_access.claim.json").resolve()
EXPECTED_PROTOCOL_PATH = (
    PROJECT_ROOT / "protocols" / "gpu_minirocket_logodds_v8_mode2_development.json").resolve()
DEVELOPMENT_DIRECTORY_NAME = "v8_fixed_formal"
EXPECTED_FIXED_DIRECTORY = (
    EXPECTED_GLOBAL_CLAIM_PATH.parent / DEVELOPMENT_DIRECTORY_NAME).resolve()
EXPECTED_ARTIFACT_ROOT = (EXPECTED_FIXED_DIRECTORY / "artifacts").resolve()
EXPECTED_FORMAL_INVOCATION = {
    "sys_executable": r"D:\py\anaocnda\envs\torch5090\python.exe",
    "sys_prefix": r"D:\py\anaocnda\envs\torch5090",
    "script_path": str((PROJECT_ROOT / "scripts" /
                        "run_gpu_minirocket_logodds_v8.py").resolve()),
    "argv_tail": ["development"],
    "working_directory": str(PROJECT_ROOT.resolve()),
    "isolated": 1, "no_site": 1, "dont_write_bytecode": 1,
    "stdout_write_through": True, "stderr_write_through": True,
    "conda_prefix": r"D:\py\anaocnda\envs\torch5090",
    "conda_default_env": "torch5090",
    "pythonpath_ignored_by_isolated_mode": True,
}
PROTOCOL_SCHEMA = "gpu_minirocket_logodds_v8_preregistered_development_protocol"
PROTOCOL_ID = "gpu-minirocket-logodds-v8-mode2-development"
BACKEND_RESULT_SCHEMA = "gpu_minirocket_logodds_v8_development_result"
CANDIDATE_ID = "gpu-minirocket-logodds-cusum-v8-fixed"
CANDIDATE_COUNT = 1
MODE2_SHA256 = "8462c0249761ae762a99028c7f8391620b68c69aaea4ecc961fe482799c974b4"
SOURCE_SIZE_BYTES = 23_897_125_720
FOLD_MAP_CANONICAL_JSON_SHA256 = "64a5aee8bc61f0865072e05be3f4cdd25f1e6f39e4061b44add2ab7d1e08eb13"
ACCESS_PLAN_CANONICAL_JSON_SHA256 = "74b19927fee7d31f33c3b64d62a0c5235b58ad41e5d2d7d28b583a6ff4e43466"
ACTIVATION_RESULT_SHA256 = "62411efc8e5d454297de9c66d05d5b96b199e82d95ace19fa588359fd8574d4a"
ACTIVATION_DECISION_SHA256 = "7e1a1c3f131e0d87a9069cb07641b1a4a9744006359cfe45add8f26935e1042d"
UPSTREAM_TAG_OBJECT_SHA = "2bca845b2905aa2cfcb03a2e1615bdb196b821e5"
UPSTREAM_PEELED_COMMIT_SHA = "f9ecc6b6df798946fa5fb290ac2ade9a00815289"
ENVIRONMENT_LOCK = {
    "python_version": "3.11.14", "numpy_version": "2.4.0",
    "h5py_distribution_version": "3.15.1",
    "torch_version": "2.10.0.dev20251124+cu128", "torch_cuda_build": "12.8",
    "cudnn_version": 91002, "nvidia_driver_version": "595.97",
    "gpu_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "gpu_compute_capability": [12, 0], "gpu_total_memory_bytes": 25_650_855_936,
    "nvidia_smi_memory_total_mib_rounded": 24_463,
}


def _attest_pinned_distribution_module(module: ModuleType, package_name: str) -> None:
    """Require a third-party package to originate in the pinned conda env."""

    expected_directory = (ENVIRONMENT_SITE_PACKAGES / package_name).resolve()
    spec = getattr(module, "__spec__", None)
    origin = Path(str(getattr(spec, "origin", ""))).resolve()
    module_file = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        origin.relative_to(expected_directory)
        module_file.relative_to(expected_directory)
    except ValueError as error:
        raise RuntimeError(
            f"{package_name} did not originate in the pinned environment") from error
    if origin != module_file:
        raise RuntimeError(f"{package_name} package origin/file mismatch")


_attest_pinned_distribution_module(torch, "torch")
_attest_pinned_distribution_module(np, "numpy")

GLOBAL_SEED = 20260912
CUDA_REPEAT_ATOL = 1.0e-6
FOLD_COUNT = 5
SEQUENCE_LENGTH = 500
FEATURE_COUNT = 52
FIT_RUNS_PER_CLASS = 60
INNER_HEALTHY_RUNS = 20
OUTER_RUNS_PER_CLASS = 20
FINAL_RUNS_PER_CLASS = 100
MONITOR_LENGTH = 340
CONDITION_IDVS = (1, 3, 9, 15)
FOLD_RUN_IDS: dict[int, tuple[int, ...]] = {
    0: (8, 14, 15, 20, 26, 31, 37, 42, 43, 48, 53, 58, 63, 68, 73, 79, 84, 90, 95, 98),
    1: (3, 6, 10, 16, 21, 27, 32, 38, 44, 49, 54, 59, 64, 69, 74, 76, 80, 85, 91, 96),
    2: (4, 9, 11, 17, 22, 28, 34, 39, 45, 50, 55, 60, 65, 70, 75, 81, 86, 88, 92, 97),
    3: (5, 12, 18, 23, 24, 29, 33, 35, 40, 46, 51, 56, 61, 66, 71, 77, 82, 87, 93, 99),
    4: (1, 2, 7, 13, 19, 25, 30, 36, 41, 47, 52, 57, 62, 67, 72, 78, 83, 89, 94, 100),
}
REQUIRED_UNIQUE_SEED_COUNT = 395
ALLOWED_DUPLICATE_SEED_OWNERS: dict[int, tuple[tuple[int, int], ...]] = {
    1_313_107: ((14, 1), (43, 9)),
    1_314_539: ((9, 9), (88, 15)),
    1_315_245: ((6, 3), (76, 15)),
    1_319_584: ((43, 3), (98, 9)),
    1_319_919: ((24, 3), (33, 15)),
}


def _progress(message: str) -> None:
    """Emit a read-only stage marker; its value is never used for branching."""
    print(f"[v8] {message}", flush=True)


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if hasattr(value, "to_document"):
        return _plain_json(value.to_document())
    if hasattr(value, "to_metadata"):
        return _plain_json(value.to_metadata())
    if is_dataclass(value):
        return {field.name: _plain_json(getattr(value, field.name)) for field in fields(value)}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(_plain_json(value))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _exclusive_json(path: Path, value: Any) -> Mapping[str, Any]:
    """Create a permanent one-file execution claim with kernel O_EXCL."""

    if not path.parent.is_dir():
        raise FileNotFoundError("exclusive JSON parent directory must already exist")
    payload = canonical_json_file_bytes(_plain_json(value))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    # A process crash may leave a partial token.  It is intentionally never
    # removed: any residue permanently blocks replay before HDF5 access.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "size_bytes": len(payload),
            "file_bytes_sha256": hashlib.sha256(payload).hexdigest()}


def _replace_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_file_bytes(_plain_json(value))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _AccessState:
    def __init__(self, path: Path, *, protocol_canonical_json_sha256: str) -> None:
        self.path = path
        self.value: dict[str, Any] = {
            "schema": "gpu_minirocket_logodds_v8_backend_access_state",
            "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
            "hdf5_open_started": False, "hdf5_open_completed": False,
            "processdata_slices_read": 0, "idv_init_slices_read": 0, "numeric_slices_read": 0,
            "processdata_slices_completed": 0, "idv_init_slices_completed": 0,
            "training_or_fit_started": False, "folds_completed": 0,
            "current_fold": None, "decision_persisted": False,
            "final_refit_started": False, "backend_completed": False,
            "mode1_opens": 0, "rieth_opens": 0,
        }
        _replace_json(self.path, self.value)

    def update(self, **changes: Any) -> None:
        if set(changes) - set(self.value):
            raise KeyError(f"unknown access state fields: {sorted(set(changes)-set(self.value))}")
        candidate = dict(self.value)
        candidate.update(changes)
        if candidate["mode1_opens"] != 0 or candidate["rieth_opens"] != 0:
            raise RuntimeError("Mode1 and Rieth opens are forbidden")
        if candidate["numeric_slices_read"] != candidate["processdata_slices_read"] + candidate["idv_init_slices_read"]:
            raise RuntimeError("numeric slice counters disagree")
        if (candidate["processdata_slices_completed"] > candidate["processdata_slices_read"] or
                candidate["idv_init_slices_completed"] > candidate["idv_init_slices_read"]):
            raise RuntimeError("completed numeric slices exceed conservative attempts")
        if not 0 <= candidate["folds_completed"] <= FOLD_COUNT:
            raise RuntimeError("fold completion count invalid")
        self.value = candidate
        _replace_json(self.path, self.value)

    def record_processdata_slice(self) -> None:
        self.update(processdata_slices_read=self.value["processdata_slices_read"] + 1,
                    numeric_slices_read=self.value["numeric_slices_read"] + 1)

    def record_idv_init_slice(self) -> None:
        self.update(idv_init_slices_read=self.value["idv_init_slices_read"] + 1,
                    numeric_slices_read=self.value["numeric_slices_read"] + 1)

    def record_processdata_slice_completed(self) -> None:
        self.update(processdata_slices_completed=self.value["processdata_slices_completed"] + 1)

    def record_idv_init_slice_completed(self) -> None:
        self.update(idv_init_slices_completed=self.value["idv_init_slices_completed"] + 1)


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact missing: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "file_bytes_sha256": sha256_file(path)}


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = strict_json_from_bytes(payload)
    if not isinstance(value, dict) or canonical_json_file_bytes(value) != payload:
        raise ValueError(f"{path.name} is not an exact canonical JSON object plus LF")
    return value, payload


def _validate_run_authorization(*, artifact_root: Path,
                                protocol_canonical_json_sha256: str,
                                protocol_path: str | os.PathLike[str],
                                protocol_file_bytes_sha256: str,
                                protocol_commit_file_bytes_sha256: str,
                                claim_path: str | os.PathLike[str],
                                claim_file_bytes_sha256: str,
                                claim_commit_file_bytes_sha256: str,
                                access_intent_path: str | os.PathLike[str],
                                access_intent_file_bytes_sha256: str,
                                access_intent_commit_file_bytes_sha256: str
                                ) -> Mapping[str, Any]:
    """Validate physical claim -> committed intent evidence before HDF5 import."""
    root = artifact_root.resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol_commit = protocol_file.with_name(protocol_file.name + ".commit.json")
    protocol_sidecar = protocol_file.with_name(protocol_file.name + ".sha256")
    claim_file = Path(claim_path).resolve()
    claim_commit = claim_file.with_name(claim_file.name + ".commit.json")
    claim_sidecar = claim_file.with_name(claim_file.name + ".sha256")
    claim_failure = claim_file.with_name("development_access.claim.failure.json")
    intent_file = Path(access_intent_path).resolve()
    intent_commit = intent_file.with_name(intent_file.name + ".commit.json")
    intent_sidecar = intent_file.with_name(intent_file.name + ".sha256")
    if protocol_file != EXPECTED_PROTOCOL_PATH:
        raise ValueError("backend requires the exact frozen v8 protocol path")
    if claim_file != EXPECTED_GLOBAL_CLAIM_PATH:
        raise ValueError("backend requires the exact global v8 claim path")
    if claim_failure.exists():
        raise RuntimeError("global claim failure evidence forbids numeric execution")
    fixed_directory = intent_file.parent
    if fixed_directory != EXPECTED_FIXED_DIRECTORY or root != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("backend artifact root must be the exact fixed artifacts directory")
    if intent_file != fixed_directory / "access_intent.json":
        raise ValueError("access intent path does not bind the backend artifact root")
    for value, label in (
        (protocol_file_bytes_sha256, "protocol"),
        (protocol_commit_file_bytes_sha256, "protocol commit"),
        (claim_file_bytes_sha256, "claim"),
        (claim_commit_file_bytes_sha256, "claim commit"),
        (access_intent_file_bytes_sha256, "access intent"),
        (access_intent_commit_file_bytes_sha256, "access intent commit"),
    ):
        if (not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"{label} SHA-256 must be an exact lowercase digest")
    if not all(path.is_file() for path in (
            protocol_file, protocol_sidecar, protocol_commit, claim_file, claim_sidecar,
            claim_commit, intent_file, intent_sidecar, intent_commit)):
        raise FileNotFoundError(
            "physical frozen protocol, global claim, and committed access intent are required")

    protocol, protocol_bytes = _read_canonical_json(protocol_file)
    protocol_commit_doc, protocol_commit_bytes = _read_canonical_json(protocol_commit)
    claim, claim_bytes = _read_canonical_json(claim_file)
    claim_commit_doc, claim_commit_bytes = _read_canonical_json(claim_commit)
    intent, intent_bytes = _read_canonical_json(intent_file)
    commit, commit_bytes = _read_canonical_json(intent_commit)
    if (hashlib.sha256(protocol_bytes).hexdigest() != protocol_file_bytes_sha256 or
            hashlib.sha256(protocol_commit_bytes).hexdigest() !=
                protocol_commit_file_bytes_sha256 or
            hashlib.sha256(claim_bytes).hexdigest() != claim_file_bytes_sha256 or
            hashlib.sha256(claim_commit_bytes).hexdigest() != claim_commit_file_bytes_sha256 or
            hashlib.sha256(intent_bytes).hexdigest() != access_intent_file_bytes_sha256 or
            hashlib.sha256(commit_bytes).hexdigest() != access_intent_commit_file_bytes_sha256):
        raise ValueError("physical claim/access-intent identity mismatch")
    expected_protocol_sidecar = (
        f"{protocol_file_bytes_sha256}  {protocol_file.name}\n".encode("ascii"))
    protocol_sidecar_bytes = protocol_sidecar.read_bytes()
    if protocol_sidecar_bytes != expected_protocol_sidecar:
        raise ValueError("frozen-protocol SHA-256 sidecar mismatch")
    if (protocol_commit_doc.get("schema") !=
            "gpu_minirocket_logodds_v8_protocol_atomic_commit" or
            protocol_commit_doc.get("document_name") != protocol_file.name or
            protocol_commit_doc.get("file_bytes_sha256") != protocol_file_bytes_sha256 or
            protocol_commit_doc.get("canonical_json_sha256") !=
                protocol_canonical_json_sha256 or
            protocol_commit_doc.get("file_size_bytes") != len(protocol_bytes) or
            protocol_commit_doc.get("canonical_json_size_bytes") != len(protocol_bytes) - 1 or
            protocol_commit_doc.get("file_bytes_are_canonical_json_plus_lf") is not True or
            protocol_commit_doc.get("sidecar_name") != protocol_sidecar.name or
            protocol_commit_doc.get("sidecar_file_bytes_sha256") !=
                hashlib.sha256(protocol_sidecar_bytes).hexdigest()):
        raise ValueError("frozen-protocol atomic commit evidence mismatch")
    validate_protocol(protocol, expected_canonical_json_sha256=
                      protocol_canonical_json_sha256)
    implementation = protocol.get("bindings", {}).get("implementation")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("frozen protocol has no implementation manifest")
    for relative, record in implementation.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("frozen implementation manifest entry invalid")
        path = (PROJECT_ROOT / relative).resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError as error:
            raise ValueError("implementation path escapes project root") from error
        if (not path.is_file() or record.get("size_bytes") != path.stat().st_size or
                record.get("file_bytes_sha256") != sha256_file(path)):
            raise ValueError(f"frozen implementation mismatch: {relative}")
    method_graph = preclaim_bind_method_graph(implementation)
    if method_graph.get("bound") is not True or method_graph.get("hdf5_loaded") is not False:
        raise RuntimeError("complete method graph was not safely loaded before HDF5")
    expected_claim_sidecar = f"{claim_file_bytes_sha256}  {claim_file.name}\n".encode("ascii")
    claim_sidecar_bytes = claim_sidecar.read_bytes()
    if claim_sidecar_bytes != expected_claim_sidecar:
        raise ValueError("global-claim SHA-256 sidecar mismatch")
    if (claim_commit_doc.get("schema") != "gpu_minirocket_logodds_v8_global_claim_atomic_commit" or
            claim_commit_doc.get("document_name") != claim_file.name or
            claim_commit_doc.get("file_bytes_sha256") != claim_file_bytes_sha256 or
            claim_commit_doc.get("canonical_json_sha256") != canonical_json_sha256(claim) or
            claim_commit_doc.get("file_size_bytes") != len(claim_bytes) or
            claim_commit_doc.get("canonical_json_size_bytes") != len(claim_bytes) - 1 or
            claim_commit_doc.get("file_bytes_are_canonical_json_plus_lf") is not True or
            claim_commit_doc.get("sidecar_name") != claim_sidecar.name or
            claim_commit_doc.get("sidecar_file_bytes_sha256") !=
                hashlib.sha256(claim_sidecar_bytes).hexdigest()):
        raise ValueError("global-claim atomic commit evidence mismatch")
    expected_sidecar = f"{access_intent_file_bytes_sha256}  {intent_file.name}\n".encode("ascii")
    sidecar_bytes = intent_sidecar.read_bytes()
    if sidecar_bytes != expected_sidecar:
        raise ValueError("access-intent SHA-256 sidecar mismatch")
    if (commit.get("schema") != "gpu_minirocket_logodds_v8_access_intent_atomic_commit" or
            commit.get("document_name") != intent_file.name or
            commit.get("file_bytes_sha256") != access_intent_file_bytes_sha256 or
            commit.get("canonical_json_sha256") != canonical_json_sha256(intent) or
            commit.get("file_size_bytes") != len(intent_bytes) or
            commit.get("canonical_json_size_bytes") != len(intent_bytes) - 1 or
            commit.get("file_bytes_are_canonical_json_plus_lf") is not True or
            commit.get("sidecar_name") != intent_sidecar.name or
            commit.get("sidecar_file_bytes_sha256") != hashlib.sha256(sidecar_bytes).hexdigest()):
        raise ValueError("access-intent atomic commit evidence mismatch")

    common = {
        "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID,
        "candidate_count": 1,
        "activation_result_file_bytes_sha256": ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": ACTIVATION_DECISION_SHA256,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
    }
    if any(claim.get(key) != value for key, value in common.items()):
        raise ValueError("global claim semantic binding mismatch")
    claim_attempt_id = claim.get("claim_attempt_id")
    if (claim.get("schema") != "gpu_minirocket_logodds_v8_exclusive_global_development_claim" or
            not isinstance(claim_attempt_id, str) or len(claim_attempt_id) != 32 or
            any(character not in "0123456789abcdef" for character in claim_attempt_id) or
            claim.get("artifact_root") != str(fixed_directory) or
            claim.get("backend_artifact_root") != str(root) or
            claim.get("numeric_execution_claim_path") !=
                str(fixed_directory / "numeric_execution.claim.json") or
            claim.get("formal_invocation") != EXPECTED_FORMAL_INVOCATION or
            claim.get("fixed_directory") != DEVELOPMENT_DIRECTORY_NAME or
            claim.get("rescue_iteration_ids_allowed") is not False):
        raise ValueError("global claim path/run binding mismatch")
    if any(intent.get(key) != value for key, value in common.items()):
        raise ValueError("access intent semantic binding mismatch")
    if (intent.get("schema") != "gpu_minirocket_logodds_v8_mode2_development_access_intent" or
            intent.get("committed_after_read_free_backend_handshake") is not True or
            intent.get("committed_before_hdf5_import_or_open") is not True or
            intent.get("access_plan") != development_access_plan() or
            intent.get("fold_run_ids") != {str(fold): list(runs)
                                           for fold, runs in FOLD_RUN_IDS.items()} or
            intent.get("authorized_reads_after_commit") !=
                "only exact 400 Mode2 magnitude50 selectors" or
            intent.get("wildcards_authorized") is not False or
            intent.get("inner_fault_threshold_values_authorized") is not False or
            intent.get("mode1_access_authorized") is not False or
            intent.get("mode1_hdf5_opens_or_value_reads") != 0 or
            intent.get("rieth_access_authorized") is not False or
            intent.get("backend_module") != "mate_spc_minirocket_v8.backend" or
            intent.get("backend_imported_before_commit") is not True or
            intent.get("production_backend_instance_reused_after_commit") is not True or
            intent.get("h5py_module_loaded_before_commit") is not False or
            intent.get("h5py_import_attempted_before_commit") is not False or
            intent.get("artifact_root") != str(root) or
            intent.get("numeric_execution_claim_path") !=
                str(fixed_directory / "numeric_execution.claim.json") or
            intent.get("formal_invocation") != EXPECTED_FORMAL_INVOCATION or
            intent.get("implementation") != implementation or
            not isinstance(intent.get("source"), Mapping) or
            intent["source"].get("sha256") != MODE2_SHA256 or
            intent["source"].get("size_bytes") != SOURCE_SIZE_BYTES or
            intent["source"].get("hdf5_opened") is not False or
            intent.get("global_claim_path") != str(claim_file) or
            intent.get("global_claim_file_bytes_sha256") != claim_file_bytes_sha256 or
            intent.get("global_claim_canonical_json_sha256") != canonical_json_sha256(claim) or
            intent.get("global_claim_file_size_bytes") != len(claim_bytes) or
            intent.get("global_claim_commit_file_bytes_sha256") !=
                claim_commit_file_bytes_sha256):
        raise ValueError("access intent authorization mismatch")
    preflight_path = fixed_directory / "preflight.json"
    preflight_sha = intent.get("preflight_file_bytes_sha256")
    if (not preflight_path.is_file() or not isinstance(preflight_sha, str) or
            sha256_file(preflight_path) != preflight_sha):
        raise ValueError("access intent does not bind the physical preclaim report")
    return {
        "protocol_path": str(protocol_file),
        "protocol_file_bytes_sha256": protocol_file_bytes_sha256,
        "protocol_commit_file_bytes_sha256": protocol_commit_file_bytes_sha256,
        "claim_path": str(claim_file), "claim_file_bytes_sha256": claim_file_bytes_sha256,
        "claim_commit_file_bytes_sha256": claim_commit_file_bytes_sha256,
        "access_intent_path": str(intent_file),
        "access_intent_file_bytes_sha256": access_intent_file_bytes_sha256,
        "access_intent_commit_file_bytes_sha256": access_intent_commit_file_bytes_sha256,
        "validated_before_hdf5_import_or_open": True,
        "method_graph_bound_before_hdf5": True,
    }


def _configure_determinism() -> None:
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def _golden_handshake() -> Mapping[str, Any]:
    from vendor.tsai_v1_0_1_minirocket.golden_handshake import verify_golden_handshake
    report = _plain_json(verify_golden_handshake(require_cuda=True))
    # The vendored, hash-bound handshake uses an exact categorical contract.
    # Do not silently accept truthy aliases or an absent field here.
    if not isinstance(report, Mapping) or report.get("status") != "PASS":
        raise RuntimeError("vendored MiniROCKET golden handshake failed")
    return report


def cuda_preflight() -> Mapping[str, Any]:
    _configure_determinism()
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else None
    properties = torch.cuda.get_device_properties(0) if available else None
    golden = _golden_handshake() if available and name is not None and "5090" in name.upper() else None
    report = {
        "cuda_available": available, "device_name": name, "device_index": 0 if available else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if available else None,
        "device_total_memory": int(properties.total_memory) if properties is not None else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "h5py_distribution_version": importlib.metadata.version("h5py"),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "rtx_5090": bool(name is not None and "5090" in name.upper()),
        "golden_handshake": golden,
        "golden_handshake_passed": bool(golden is not None and golden.get("status") == "PASS"),
    }
    report["passed"] = bool(report["cuda_available"] and report["rtx_5090"] and
                            report["device_name"] == ENVIRONMENT_LOCK["gpu_name"] and
                            report["device_capability"] == ENVIRONMENT_LOCK["gpu_compute_capability"] and
                            report["device_total_memory"] == ENVIRONMENT_LOCK["gpu_total_memory_bytes"] and
                            report["python_version"] == ENVIRONMENT_LOCK["python_version"] and
                            report["numpy_version"] == ENVIRONMENT_LOCK["numpy_version"] and
                            report["h5py_distribution_version"] ==
                                ENVIRONMENT_LOCK["h5py_distribution_version"] and
                            report["torch_version"] == ENVIRONMENT_LOCK["torch_version"] and
                            report["torch_cuda_version"] == ENVIRONMENT_LOCK["torch_cuda_build"] and
                            report["cudnn_version"] == ENVIRONMENT_LOCK["cudnn_version"] and
                            report["deterministic_algorithms"] and not report["deterministic_warn_only"] and
                            report["cudnn_deterministic"] and not report["cudnn_benchmark"] and
                            report["cublas_workspace_config"] == ":4096:8" and
                            report["golden_handshake_passed"])
    return report


def _require_cuda(device: str) -> Mapping[str, Any]:
    if device != "cuda":
        raise ValueError("formal v8 requires device='cuda'")
    report = dict(cuda_preflight())
    if report["passed"] is not True:
        raise RuntimeError("formal v8 CUDA/golden preflight failed")
    return report


def development_access_plan() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for run_id in range(1, 101):
        for idv in CONDITION_IDVS:
            group = f"Mode2/SingleFault/SimulationCompleted/IDV{idv}/Mode2_IDVInfo_{idv}_50/Run{run_id}"
            window = [100, 600] if idv == 1 else [440, 940]
            result.append({
                "mode": 2, "fault_regime": "SingleFault", "status": "SimulationCompleted",
                "magnitude_percent": 50, "idv": idv, "simulation_run": run_id,
                "group_path": group, "processdata_path": f"{group}/processdata",
                "idv_init_path": f"{group}/idv_init", "window_zero_based_half_open": window,
                "processdata_row_zero_based_half_open": window,
                "processdata_column_zero_based_half_open": [0, 53],
                "idv_init_row_zero_based_half_open": [0, 3],
                "idv_init_column_zero_based_half_open": [0, 28],
                "group_attributes_authorized": ["seed"],
                "dataset_metadata_authorized": ["object_type", "shape", "dtype"],
                "root_dataset_value_reads_authorized": [],
                "processdata_values_outside_row_window_authorized": False,
                "role": "healthy_control" if idv == 1 else f"fault_{idv}",
            })
    return result


_EXPECTED_ACCESS_PLAN = tuple(development_access_plan())
if canonical_json_sha256(_EXPECTED_ACCESS_PLAN) != ACCESS_PLAN_CANONICAL_JSON_SHA256:
    raise RuntimeError("compiled v8 access-plan hash mismatch")


def _validate_access_plan(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("access_plan must be a sequence")
    candidate = _plain_json(list(value))
    if candidate != list(_EXPECTED_ACCESS_PLAN):
        raise ValueError("access_plan differs from exact 400 v8 selectors")
    if canonical_json_sha256(candidate) != ACCESS_PLAN_CANONICAL_JSON_SHA256:
        raise ValueError("access-plan canonical hash changed")
    return candidate


def _canonical_fold_map() -> dict[int, int]:
    return {run: fold for fold, runs in FOLD_RUN_IDS.items() for run in runs}


if canonical_json_sha256({str(run): fold for run, fold in sorted(_canonical_fold_map().items())}) != FOLD_MAP_CANONICAL_JSON_SHA256:
    raise RuntimeError("compiled v8 fold-map hash mismatch")


def _validate_fold_map(value: Mapping[int, int] | Mapping[str, int]) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("fold_map must be a mapping")
    if any(isinstance(k, bool) or isinstance(v, bool) for k, v in value.items()):
        raise ValueError("boolean fold entries forbidden")
    try:
        candidate = {int(k): int(v) for k, v in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("fold-map entries must be integers") from error
    if candidate != _canonical_fold_map():
        raise ValueError("fold_map differs from collision-safe allocation")
    return candidate


def validate_protocol(protocol_doc: Mapping[str, Any], *,
                      expected_canonical_json_sha256: str | None = None) -> Mapping[str, Any]:
    if not isinstance(protocol_doc, Mapping):
        raise ValueError("protocol_doc must be a mapping")
    document = _plain_json(dict(protocol_doc))
    observed_hash = canonical_json_sha256(document)
    if expected_canonical_json_sha256 is not None:
        if (not isinstance(expected_canonical_json_sha256, str) or
                len(expected_canonical_json_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in expected_canonical_json_sha256)):
            raise ValueError("expected_canonical_json_sha256 must be lowercase SHA-256")
        if observed_hash != expected_canonical_json_sha256:
            raise ValueError("protocol canonical_json_sha256 mismatch")
    if (document.get("schema") != PROTOCOL_SCHEMA or document.get("schema_version") != 1 or
            document.get("protocol_id") != PROTOCOL_ID or document.get("candidate_count") != 1 or
            document.get("environment_lock") != ENVIRONMENT_LOCK):
        raise ValueError("v8 protocol identity changed")
    candidate = document.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("candidate_id") != CANDIDATE_ID or candidate.get("candidate_count") != 1:
        raise ValueError("v8 candidate identity changed")
    activation = candidate.get("activation")
    if (not isinstance(activation, Mapping) or
            activation.get("predecessor_candidate_id") != "farm-rankcusum-v7-fixed" or
            activation.get("required_exact_result") != "DEVELOPMENT_NO_GO" or
            activation.get("v7_result_file_bytes_sha256") != ACTIVATION_RESULT_SHA256 or
            activation.get("v7_decision_evidence_file_bytes_sha256") != ACTIVATION_DECISION_SHA256 or
            activation.get("v7_numeric_metrics_used_to_modify_candidate") is not False):
        raise ValueError("v8 activation contract changed")
    evidence = document.get("activation_evidence")
    if (not isinstance(evidence, Mapping) or
            evidence.get("trigger") != {"candidate_id": "farm-rankcusum-v7-fixed", "result": "DEVELOPMENT_NO_GO"} or
            evidence.get("development_result", {}).get("file_bytes_sha256") != ACTIVATION_RESULT_SHA256 or
            evidence.get("decision_before_refit", {}).get("file_bytes_sha256") != ACTIVATION_DECISION_SHA256 or
            evidence.get("numeric_metrics_used_to_modify_v8") is not False):
        raise ValueError("persisted v8 activation evidence changed")
    upstream = candidate.get("upstream")
    if (not isinstance(upstream, Mapping) or upstream.get("tag") != "v1.0.1" or
            upstream.get("tag_kind") != "annotated" or
            upstream.get("annotated_tag_object_sha") != UPSTREAM_TAG_OBJECT_SHA or
            upstream.get("peeled_commit_sha") != UPSTREAM_PEELED_COMMIT_SHA):
        raise ValueError("v8 upstream identity changed")
    data = candidate.get("data")
    if (not isinstance(data, Mapping) or
            (data.get("mode"), data.get("fault_regime"), data.get("magnitude_percent"),
             data.get("idvs"), data.get("runs_per_idv"), data.get("rows_per_run"),
             data.get("process_channel_count"), data.get("healthy_window_zero_based_half_open"),
             data.get("fault_window_zero_based_half_open"), data.get("fault_local_onset_index"),
             data.get("miss_delay_samples")) !=
            (2, "SingleFault", 50, [1, 3, 9, 15], 100, 500, 52,
             [100, 600], [440, 940], 160, 340)):
        raise ValueError("v8 data contract changed")
    normalization = candidate.get("normalization")
    if (not isinstance(normalization, Mapping) or
            (normalization.get("fit_healthy_runs_per_fold"), normalization.get("channel_count"),
             normalization.get("location"), normalization.get("scale"), normalization.get("clip"),
             normalization.get("output_dtype")) !=
            (60, 52, "float64_median", "max(1.4826*MAD,1e-6)", [-10.0, 10.0], "float32")):
        raise ValueError("v8 normalization contract changed")
    mini = candidate.get("minirocket")
    if (not isinstance(mini, Mapping) or
            (mini.get("seed"), mini.get("window_shape"), mini.get("kernel_count"),
             mini.get("requested_feature_count"), mini.get("effective_feature_count"),
             mini.get("max_dilations_per_kernel"), mini.get("dilation_integer_range"),
             mini.get("features_per_dilation_per_kernel"), mini.get("max_num_channels"),
             mini.get("add_lsaz"), mini.get("bias_fit_chunk_size"), mini.get("ppv_comparison"),
             mini.get("training_window_count"), mini.get("kernel_length"),
             mini.get("feature_batch_size"), mini.get("drop_last")) !=
            (GLOBAL_SEED, [52, 128], 84, 10000, 9996, 32, [1, 15],
              [30, 19, 12, 12, 8, 4, 8, 4, 4, 3, 3, 3, 3, 3, 3],
              9, False, 1260, ">", 20400, 9, 128, False)):
        raise ValueError("v8 MiniROCKET contract changed")
    head = candidate.get("linear_head")
    if (not isinstance(head, Mapping) or
            (head.get("shape"), head.get("parameter_count"), head.get("class_order"),
             head.get("initialization"), head.get("class_weights"), head.get("l2"),
             head.get("optimizer"), head.get("max_iter"), head.get("max_eval"),
             head.get("history_size"), head.get("line_search_fn"), head.get("lr"),
             head.get("tolerance_grad"), head.get("tolerance_change")) !=
            ([9996, 4], 39988, ["H", "F3", "F9", "F15"], "all_zero",
              ["85/109", "85/77", "85/77", "85/77"], 1e-4,
              "torch.optim.LBFGS", 100, 125, 20, "strong_wolfe", 1.0, 1e-7, 1e-9)):
        raise ValueError("v8 linear-head contract changed")
    training = candidate.get("training_examples")
    if (not isinstance(training, Mapping) or training != {
            "stable_order": ["class_order_H_F3_F9_F15", "source_or_run_id_ascending",
                             "endpoint_ascending"],
            "total": 20400, "H_total": 6540, "H_healthy_count": 5100,
            "H_fault_prefix_count": 1440, "fault_class_count_each": 4620,
            "healthy_endpoints": [160, 496, 4],
            "fault_prefix_endpoints": [128, 132, 136, 140, 144, 148, 152, 156],
            "mature_fault_endpoints": [192, 496, 4],
            "fault_transition_endpoints_ignored": [160, 164, 168, 172, 176, 180, 184, 188],
    }):
        raise ValueError("v8 training-example contract changed")
    monitor = candidate.get("monitor")
    if (not isinstance(monitor, Mapping) or
            (monitor.get("prefix_endpoints"), monitor.get("dense_monitor_endpoints_inclusive"),
             monitor.get("calibration_fit_healthy_value_count"), monitor.get("cusum_drift"),
             monitor.get("intermittent_first_endpoint"), monitor.get("intermittent_window"),
             monitor.get("intermittent_top_r"), monitor.get("intermittent_coefficient"),
             monitor.get("dtype"), monitor.get("calibration_scale_floor")) !=
            ([128, 132, 136, 140, 144, 148, 152, 156], [160, 499], 20400, 0.25,
              175, 16, 4, 0.5, "cpu_float64", 1e-3)):
        raise ValueError("v8 monitor contract changed")
    threshold = candidate.get("inner_threshold")
    if (not isinstance(threshold, Mapping) or
            (threshold.get("healthy_runs"), threshold.get("fault_values_used"),
             threshold.get("ascending_rank"), threshold.get("comparison"),
             threshold.get("minimum_exact_distinct_maxima"), threshold.get("tail_gap_multiplier"),
             threshold.get("repeat_error_floor")) != (20, False, 20, ">", 18, 100.0, 1e-6)):
        raise ValueError("v8 threshold contract changed")
    determinism = candidate.get("determinism")
    if (not isinstance(determinism, Mapping) or
            determinism.get("seed") != GLOBAL_SEED or
            determinism.get("authoritative_device") != "CUDA RTX 5090" or
            determinism.get("same_cuda_complete_replays") != 2 or
            determinism.get("same_cuda_max_abs_error") != CUDA_REPEAT_ATOL or
            determinism.get("cublas_workspace_config") != ":4096:8" or
            determinism.get("torch_deterministic_algorithms") is not True or
            determinism.get("cudnn_deterministic") is not True or
            determinism.get("cudnn_benchmark") is not False or
            determinism.get("cpu_reference_is_go_gate") is not False):
        raise ValueError("v8 determinism contract changed")
    final_refit = candidate.get("go_only_final_refit")
    if final_refit != {"trigger": "all five development folds pass",
                       "runs_per_class": 100, "mode1_access_authorized": False}:
        raise ValueError("v8 GO-only final-refit contract changed")
    source, access, folds, governance, numeric = (
        document.get("source"), document.get("access_plan"), document.get("folds"),
        document.get("governance"), document.get("numeric_backend_contract"))
    if (not isinstance(source, Mapping) or
            (source.get("sha256"), source.get("size_bytes"), source.get("mode"),
             source.get("magnitude_percent"), source.get("hdf5_opened_during_freeze"),
             source.get("mode1_access_authorized"), source.get("rieth_access_authorized")) !=
            (MODE2_SHA256, SOURCE_SIZE_BYTES, 2, 50, False, False, False)):
        raise ValueError("v8 source binding changed")
    if (not isinstance(access, Mapping) or
            (access.get("record_count"), access.get("selector_count"),
             access.get("canonical_json_sha256"), access.get("wildcards_allowed")) !=
            (400, 400, ACCESS_PLAN_CANONICAL_JSON_SHA256, False)):
        raise ValueError("v8 access-plan binding changed")
    if (not isinstance(folds, Mapping) or
            folds.get("fold_run_ids") != {str(f): list(FOLD_RUN_IDS[f]) for f in range(5)} or
            folds.get("run_to_fold_canonical_json_sha256") != FOLD_MAP_CANONICAL_JSON_SHA256):
        raise ValueError("v8 fold binding changed")
    if (not isinstance(governance, Mapping) or governance.get("mode1_access_authorized") is not False or
            governance.get("rieth_access_authorized") is not False or
            governance.get("inner_fault_numeric_access_authorized") is not False or
            governance.get("development_no_go_terminates_current_pool_search") is not True):
        raise ValueError("v8 governance binding changed")
    if (not isinstance(numeric, Mapping) or numeric.get("module") != "mate_spc_minirocket_v8.backend" or
            numeric.get("required_functions") != [
                "cuda_preflight", "validate_protocol", "preclaim_bind_method_graph",
                "run_development", "load_frozen_fold_pipeline", "load_final_refit_pipeline"] or
            numeric.get("hdf5_import_or_open_before_intent") is not False or
            numeric.get("golden_handshake_before_claim") is not True or
            numeric.get("preclaim_test_count") != 66 or
            numeric.get("result_schema") != BACKEND_RESULT_SCHEMA):
        raise ValueError("v8 numeric backend contract changed")
    execution = document.get("execution")
    if execution != {
            "working_directory": ".",
            "environment_name": "torch5090",
            "output_directory": "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal",
            "completion_path":
                "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/completion.json",
            "numeric_execution_claim_path":
                "outputs/gpu_minirocket_logodds_v8/v8_fixed_formal/numeric_execution.claim.json",
            "command": [
                "conda", "run", "--no-capture-output", "-n", "torch5090",
                "python", "-I", "-S", "-B", "-u", "scripts/run_gpu_minirocket_logodds_v8.py",
                "development",
            ],
            "direct_environment_python_invocation_authorized": False,
            "observable_invocation_contract": {
                "sys_executable": r"D:\py\anaocnda\envs\torch5090\python.exe",
                "sys_prefix": r"D:\py\anaocnda\envs\torch5090",
                "isolated": 1, "no_site": 1, "dont_write_bytecode": 1,
                "working_directory": ".",
                "script_path": "scripts/run_gpu_minirocket_logodds_v8.py",
                "argv_tail": ["development"],
                "stdout_write_through": True, "stderr_write_through": True,
                "conda_prefix": r"D:\py\anaocnda\envs\torch5090",
                "conda_default_env": "torch5090",
            },
            "failure_behavior": {
                "before_claim": "abort_without_hdf5_access; no claim is consumed",
                "partial_or_complete_claim_before_artifact_directory":
                    "persist global claim failure evidence; fail closed; no rerun",
                "after_claim_before_numeric_access":
                    "persist failure and completion evidence; fail closed; no rerun",
                "during_or_after_numeric_access":
                    "persist conservative access state plus failure and completion; no rerun",
                "partial_or_complete_numeric_execution_claim":
                    "permanent execution claim residue; fail closed; no replay",
                "during_go_only_final_refit":
                    "retain decision-before-refit evidence; persist failure and completion; no rerun",
                "same_candidate_rerun_after_any_claim_residue": False,
            },
    }:
        raise ValueError("v8 formal execution command changed")
    gates = document.get("development_go_gates")
    if gates != {
            "logic": "all required", "numeric_folds_valid_required": 5,
            "healthy_false_alarms_overall_max": 10,
            "healthy_false_alarms_per_fold_max": 4,
            "fault_detection_rate_each_overall_min": 0.85,
            "fault_detection_rate_each_per_fold_min": 0.70,
            "pooled_fault_median_delay_max": 140,
            "pooled_fault_nearest_rank_p90_delay_max": 220,
            "pooled_fault_run_count": 300, "miss_delay": 340,
            "authoritative_cuda_repeat_gate": True,
            "cross_device_cpu_cuda_gate": False,
    }:
        raise ValueError("v8 development GO-gate contract changed")
    bindings = document.get("bindings")
    implementation = bindings.get("implementation") if isinstance(bindings, Mapping) else None
    if (not isinstance(implementation, Mapping) or not implementation or
            bindings.get("implementation_tree_canonical_json_sha256") !=
                canonical_json_sha256(implementation)):
        raise ValueError("v8 implementation-tree identity changed")
    return {
        "valid": True, "schema": PROTOCOL_SCHEMA, "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID, "candidate_count": 1,
        "protocol_canonical_json_sha256": observed_hash,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "activation_result_file_bytes_sha256": ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": ACTIVATION_DECISION_SHA256,
        "hash_domain": "canonical_json_without_file_terminator",
        "file_bytes_sha256_compared": False,
    }


@dataclass(frozen=True)
class NumericRun:
    name: str
    x: np.ndarray
    source_idv: int
    simulation_run: int
    rng_seed: int


@dataclass(frozen=True)
class NumericBlock:
    simulation_run: int
    conditions: tuple[NumericRun, ...]


def _read_seed(group: Any, *, path: str) -> int:
    attrs = getattr(group, "attrs", None)
    if attrs is None or "seed" not in attrs:
        raise ValueError(f"{path}: exact seed attribute missing")
    value = np.asarray(attrs["seed"])
    if value.shape != (1,) or value.dtype.kind not in "iuf":
        raise ValueError(f"{path}: seed must be one numeric element")
    scalar = float(value[0])
    if not math.isfinite(scalar) or scalar <= 0 or not scalar.is_integer():
        raise ValueError(f"{path}: seed must be a positive integer")
    return int(scalar)


def _validate_dataset(value: Any, *, path: str, shape: tuple[int, ...],
                      h5py_module: Any | None) -> None:
    if h5py_module is not None and not isinstance(value, h5py_module.Dataset):
        raise ValueError(f"{path}: object must be an HDF5 dataset")
    if tuple(getattr(value, "shape", ())) != shape:
        raise ValueError(f"{path}: exact shape must be {shape}")
    if np.dtype(getattr(value, "dtype", None)) != np.dtype(np.float64):
        raise ValueError(f"{path}: dtype must be exact float64")


def _require_local_hard_link(handle: Any, value: Any, *, path: str,
                             h5py_module: Any) -> None:
    """Prevent an authorized path from escaping the hash-bound Mode2 file."""

    link = handle.get(path, getlink=True)
    if not isinstance(link, h5py_module.HardLink):
        raise ValueError(f"{path}: only an in-file hard link is authorized")
    filename = Path(str(value.file.filename)).resolve()
    if filename != OFFICIAL_MODE2_PATH:
        raise ValueError(f"{path}: HDF5 object escaped the authorized source file")


def _read_exact_blocks(handle: Any, access_plan: Sequence[Mapping[str, Any]], *,
                       h5py_module: Any | None = None,
                       access_state: _AccessState | None = None
                       ) -> tuple[Mapping[str, Any], tuple[NumericBlock, ...]]:
    plan = _validate_access_plan(access_plan)
    by_identity: dict[tuple[int, int], NumericRun] = {}
    seed_owners: dict[int, list[tuple[int, int]]] = {}
    for selector in plan:
        run_id, idv = int(selector["simulation_run"]), int(selector["idv"])
        group_path, process_path, init_path = (
            str(selector["group_path"]), str(selector["processdata_path"]),
            str(selector["idv_init_path"]))
        for path in (group_path, process_path, init_path):
            if path not in handle:
                raise ValueError(f"authorized exact HDF5 path missing: {path}")
        group = handle[group_path]
        if h5py_module is not None and not isinstance(group, h5py_module.Group):
            raise ValueError(f"{group_path}: object must be an HDF5 group")
        if h5py_module is not None:
            _require_local_hard_link(
                handle, group, path=group_path, h5py_module=h5py_module)
        seed = _read_seed(group, path=group_path)
        processdata, declaration_data = handle[process_path], handle[init_path]
        if h5py_module is not None:
            _require_local_hard_link(
                handle, processdata, path=process_path, h5py_module=h5py_module)
            _require_local_hard_link(
                handle, declaration_data, path=init_path, h5py_module=h5py_module)
        _validate_dataset(processdata, path=process_path, shape=(2000, 54), h5py_module=h5py_module)
        _validate_dataset(declaration_data, path=init_path, shape=(3, 28), h5py_module=h5py_module)
        start, stop = (int(item) for item in selector["window_zero_based_half_open"])
        if access_state is not None:
            access_state.record_processdata_slice()
        decoded = np.array(processdata[start:stop, 0:53], dtype=np.float64,
                           copy=True, order="C")
        if access_state is not None and hasattr(access_state, "record_processdata_slice_completed"):
            access_state.record_processdata_slice_completed()
        if decoded.shape != (500, 53) or not np.isfinite(decoded).all():
            raise ValueError(f"{process_path}: authorized slice changed")
        expected_time = (np.arange(start, stop, dtype=np.float64) + 1.0) * 0.05
        if not np.allclose(decoded[:, 0], expected_time, rtol=0.0, atol=1e-12):
            raise ValueError(f"{process_path}: time column changed")
        if access_state is not None:
            access_state.record_idv_init_slice()
        declaration = np.asarray(declaration_data[0:3, 0:28], dtype=np.float64)
        if access_state is not None and hasattr(access_state, "record_idv_init_slice_completed"):
            access_state.record_idv_init_slice_completed()
        expected = np.zeros((3, 28), dtype=np.float64)
        expected[:, idv - 1] = (0.0, 0.5, 30.0)
        if (declaration.shape != (3, 28) or not np.isfinite(declaration).all() or
                not np.allclose(declaration, expected, rtol=0.0, atol=1e-12)):
            raise ValueError(f"{init_path}: fault declaration changed")
        x = np.ascontiguousarray(decoded[:, 1:53], dtype=np.float64)
        x.setflags(write=False)
        identity = (run_id, idv)
        if identity in by_identity:
            raise RuntimeError("duplicate selector identity")
        by_identity[identity] = NumericRun(
            name=f"MiniROCKET-v8-Mode2-IDV{idv}-Run{run_id}", x=x,
            source_idv=idv, simulation_run=run_id, rng_seed=seed)
        seed_owners.setdefault(seed, []).append(identity)
    duplicate_owners = {seed: tuple(sorted(owners)) for seed, owners in seed_owners.items()
                        if len(owners) > 1}
    if (len(by_identity) != 400 or len(seed_owners) != REQUIRED_UNIQUE_SEED_COUNT or
            duplicate_owners != ALLOWED_DUPLICATE_SEED_OWNERS):
        raise ValueError("Mode2 seed-owner topology changed")
    blocks = tuple(NumericBlock(run, tuple(by_identity[(run, idv)] for idv in CONDITION_IDVS))
                   for run in range(1, 101))
    topology = {
        "selected_run_count": 400, "counts_by_status": {"SimulationCompleted": 400},
        "counts_by_idv": {str(idv): 100 for idv in CONDITION_IDVS},
        "unique_seed_count": len(seed_owners),
        "duplicate_seed_value_count": len(duplicate_owners),
        "cross_fold_seed_collision_count": 0,
        "numeric_value_access": {
            "processdata_slices": 400, "processdata_rows_per_slice": 500,
            "processdata_columns_half_open": [0, 53], "idv_init_slices": 400,
            "root_dataset_value_reads": 0,
            "processdata_values_outside_authorized_windows": 0,
        },
    }
    return topology, blocks


def _authenticate_source() -> None:
    path = OFFICIAL_MODE2_PATH
    if not path.is_file() or path.name != "TEP_Mode2.h5":
        raise FileNotFoundError(path)
    if path.stat().st_size != SOURCE_SIZE_BYTES or sha256_file(path) != MODE2_SHA256:
        raise ValueError("official Mode2 source identity changed")


def _load_numeric_blocks(access_plan: Sequence[Mapping[str, Any]], *,
                         access_state: _AccessState | None = None
                         ) -> tuple[Mapping[str, Any], tuple[NumericBlock, ...]]:
    if os.environ.get("MATE_SPC_V8_FORBID_REAL_H5") == "1":
        raise RuntimeError("real HDF5 access forbidden in synthetic-test environment")
    plan = _validate_access_plan(access_plan)
    _authenticate_source()
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("formal v8 loading requires h5py") from error
    _attest_pinned_distribution_module(h5py, "h5py")
    if (importlib.metadata.version("h5py") != ENVIRONMENT_LOCK["h5py_distribution_version"] or
            not all(getattr(value, "__module__", "").startswith("h5py.")
                    for value in (h5py.File, h5py.Group, h5py.Dataset, h5py.HardLink))):
        raise RuntimeError("h5py API/distribution identity changed")
    if access_state is not None:
        access_state.update(hdf5_open_started=True)
    with h5py.File(OFFICIAL_MODE2_PATH, "r") as handle:
        if access_state is not None:
            access_state.update(hdf5_open_completed=True)
        return _read_exact_blocks(handle, plan, h5py_module=h5py, access_state=access_state)


def _validate_blocks(blocks: Sequence[NumericBlock], fold_map: Mapping[int, int]
                     ) -> tuple[NumericBlock, ...]:
    if len(blocks) != 100:
        raise ValueError("v8 requires exactly 100 balanced blocks")
    ordered = tuple(sorted(blocks, key=lambda item: item.simulation_run))
    if tuple(block.simulation_run for block in ordered) != tuple(range(1, 101)):
        raise ValueError("blocks must be exactly Run1..100")
    identities: set[tuple[int, int]] = set()
    seed_folds: dict[int, set[int]] = {}
    for block in ordered:
        if tuple(run.source_idv for run in block.conditions) != CONDITION_IDVS:
            raise ValueError("each block must contain IDV1/3/9/15 in order")
        for run in block.conditions:
            identity = (run.source_idv, run.simulation_run)
            values = np.asarray(run.x)
            if (run.simulation_run != block.simulation_run or identity in identities or
                    values.shape != (500, 52) or values.dtype != np.dtype(np.float64) or
                    not np.isfinite(values).all()):
                raise ValueError("invalid or duplicate run block")
            identities.add(identity)
            seed_folds.setdefault(run.rng_seed, set()).add(fold_map[run.simulation_run])
    if identities != {(idv, run) for idv in CONDITION_IDVS for run in range(1, 101)}:
        raise ValueError("development identities incomplete")
    if any(len(folds) != 1 for folds in seed_folds.values()):
        raise ValueError("repeated seed crosses frozen fold")
    return ordered


def _select_runs(blocks: Sequence[NumericBlock], run_ids: Sequence[int],
                 idvs: Sequence[int]) -> tuple[NumericRun, ...]:
    wanted_runs, wanted_idvs = set(map(int, run_ids)), tuple(map(int, idvs))
    result = tuple(run for block in blocks if block.simulation_run in wanted_runs
                   for run in block.conditions if run.source_idv in wanted_idvs)
    if len(result) != len(wanted_runs) * len(wanted_idvs):
        raise RuntimeError("fold run selection count changed")
    return result


def _values(runs: Sequence[NumericRun]) -> np.ndarray:
    result = np.stack([np.asarray(run.x, dtype=np.float64) for run in runs])
    if result.shape != (len(runs), 500, 52) or not np.isfinite(result).all():
        raise ValueError("run tensor geometry changed")
    return result


class _FoldNumericNoGo(RuntimeError):
    pass


_METHOD_API: Any | None = None
_METHOD_GRAPH_MODULES: dict[str, ModuleType] | None = None

_METHOD_REQUIRED_EXPORTS = (
    "set_determinism", "robust_fit_normalizer", "normalize_runs",
    "build_training_windows", "fit_feature_standardizer", "standardize_features",
    "fit_linear_head", "linear_logits", "fit_minirocket_transform",
    "transform_windows", "save_minirocket_state_npz", "load_minirocket_state_npz",
    "compute_log_odds", "fit_monitor_calibration", "scan_batch", "inner_threshold",
    "apply_threshold", "fold_roles", "evaluate_gates",
    "robust_fit_normalizer_final_refit", "build_final_refit_training_windows",
    "fit_minirocket_transform_final_refit", "fit_feature_standardizer_final_refit",
    "fit_linear_head_final_refit",
)


def _attest_graph_module(module_name: str, expected_path: Path,
                         implementation: Mapping[str, Any],
                         required_callables: Sequence[str] = ()) -> ModuleType:
    module = sys.modules.get(module_name)
    expected = expected_path.resolve()
    relative = expected.relative_to(PROJECT_ROOT).as_posix()
    record = implementation.get(relative)
    if not isinstance(module, ModuleType) or module.__name__ != module_name:
        raise RuntimeError(f"untrusted scoring module object: {module_name}")
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    try:
        loaded_path = Path(loader.get_filename(module_name)).resolve()
    except (AttributeError, ImportError, TypeError, ValueError) as error:
        raise RuntimeError(f"scoring module lacks trusted loader: {module_name}") from error
    if (not isinstance(loader, SourceFileLoader) or spec.name != module_name or
            Path(str(spec.origin)).resolve() != expected or
            Path(str(getattr(module, "__file__", ""))).resolve() != expected or
            getattr(module, "__loader__", None) is not loader or loaded_path != expected):
        raise RuntimeError(f"scoring module source/spec mismatch: {module_name}")
    if (not isinstance(record, Mapping) or not expected.is_file() or
            record.get("size_bytes") != expected.stat().st_size or
            record.get("file_bytes_sha256") != sha256_file(expected)):
        raise RuntimeError(f"scoring module differs from frozen bytes: {relative}")
    for name in required_callables:
        value = getattr(module, name, None)
        source = inspect.getsourcefile(value) if callable(value) else None
        code = getattr(value, "__code__", None)
        code_mismatch = (code is not None and Path(code.co_filename).resolve() != expected)
        if (not callable(value) or getattr(value, "__module__", None) != module_name or
                source is None or Path(source).resolve() != expected or code_mismatch or
                (code is None and not inspect.isclass(value))):
            raise RuntimeError(f"untrusted callable origin: {module_name}.{name}")
    return module


def _prepare_frozen_graph_module(module_name: str, path: Path,
                                 implementation: Mapping[str, Any], *,
                                 is_package: bool) -> tuple[ModuleType, bytes]:
    expected = path.resolve()
    relative = expected.relative_to(PROJECT_ROOT).as_posix()
    record = implementation.get(relative)
    payload = expected.read_bytes()
    if (not isinstance(record, Mapping) or record.get("size_bytes") != len(payload) or
            record.get("file_bytes_sha256") != hashlib.sha256(payload).hexdigest()):
        raise RuntimeError(f"graph source bytes differ from frozen identity: {relative}")
    locations = [str(expected.parent)] if is_package else None
    spec = importlib.util.spec_from_file_location(
        module_name, expected, submodule_search_locations=locations)
    if spec is None or not isinstance(spec.loader, SourceFileLoader):
        raise RuntimeError(f"cannot construct graph source spec: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if not isinstance(parent, ModuleType):
            sys.modules.pop(module_name, None)
            raise RuntimeError(f"graph parent module is missing: {parent_name}")
        setattr(parent, child_name, module)
    return module, payload


def _execute_frozen_graph_module(module: ModuleType, payload: bytes) -> None:
    path = Path(str(module.__file__)).resolve()
    exec(compile(payload, str(path), "exec", dont_inherit=True), module.__dict__)


def _load_one_frozen_graph_module(module_name: str, path: Path,
                                  implementation: Mapping[str, Any], *,
                                  is_package: bool = False) -> ModuleType:
    module, payload = _prepare_frozen_graph_module(
        module_name, path, implementation, is_package=is_package)
    try:
        _execute_frozen_graph_module(module, payload)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_graph_fresh(implementation: Mapping[str, Any]) -> Any:
    disallowed = tuple(
        name for name in sys.modules
        if (name.startswith("mate_spc_minirocket_v8.") and
            name != "mate_spc_minirocket_v8.backend") or
        name == "vendor" or name.startswith("vendor."))
    if disallowed:
        raise RuntimeError(f"v8 scoring graph was preloaded before binding: {sorted(disallowed)}")
    package_root = PROJECT_ROOT / "src" / "mate_spc_minirocket_v8"
    vendor_root = PROJECT_ROOT / "vendor"
    vendor_package = vendor_root / "tsai_v1_0_1_minirocket"
    loaded: list[str] = []
    try:
        for name in ("model", "monitor", "evaluation"):
            module_name = f"mate_spc_minirocket_v8.{name}"
            _load_one_frozen_graph_module(
                module_name, package_root / f"{name}.py", implementation)
            loaded.append(module_name)
        _load_one_frozen_graph_module(
            "vendor", vendor_root / "__init__.py", implementation, is_package=True)
        loaded.append("vendor")
        nested_name = "vendor.tsai_v1_0_1_minirocket"
        nested, nested_payload = _prepare_frozen_graph_module(
            nested_name, vendor_package / "__init__.py", implementation,
            is_package=True)
        loaded.append(nested_name)
        mini_name = f"{nested_name}.minirocket_plus"
        _load_one_frozen_graph_module(
            mini_name, vendor_package / "minirocket_plus.py", implementation)
        loaded.append(mini_name)
        _execute_frozen_graph_module(nested, nested_payload)
        transform_name = "mate_spc_minirocket_v8.transform"
        _load_one_frozen_graph_module(
            transform_name, package_root / "transform.py", implementation)
        loaded.append(transform_name)
        method_name = "mate_spc_minirocket_v8.method"
        method = _load_one_frozen_graph_module(
            method_name, package_root / "method.py", implementation)
        loaded.append(method_name)
        golden_name = f"{nested_name}.golden_handshake"
        _load_one_frozen_graph_module(
            golden_name, vendor_package / "golden_handshake.py", implementation)
        loaded.append(golden_name)
        return method
    except BaseException:
        for module_name in reversed(loaded):
            sys.modules.pop(module_name, None)
        raise


def _method_api() -> Any:
    """Single lazy seam used by production and monkeypatched synthetic tests."""
    global _METHOD_API
    if _METHOD_API is None:
        _METHOD_API = importlib.import_module("mate_spc_minirocket_v8.method")
    return _METHOD_API


def preclaim_bind_method_graph(implementation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load and path/hash-bind the complete scoring graph before any HDF5 access."""
    global _METHOD_API, _METHOD_GRAPH_MODULES
    if not isinstance(implementation, Mapping):
        raise ValueError("implementation manifest mapping required")
    if _METHOD_GRAPH_MODULES is None:
        if _METHOD_API is not None:
            raise RuntimeError("method API was initialized before formal graph binding")
        method = _load_graph_fresh(implementation)
    else:
        method = _METHOD_API
    expected = {
        _HASH_CONTRACT_NAME: _HASH_CONTRACT_PATH,
        "mate_spc_minirocket_v8": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "__init__.py",
        "mate_spc_minirocket_v8.backend": Path(__file__),
        "mate_spc_minirocket_v8.method": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "method.py",
        "mate_spc_minirocket_v8.model": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "model.py",
        "mate_spc_minirocket_v8.transform": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "transform.py",
        "mate_spc_minirocket_v8.monitor": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "monitor.py",
        "mate_spc_minirocket_v8.evaluation": PROJECT_ROOT / "src" / "mate_spc_minirocket_v8" / "evaluation.py",
        "vendor": PROJECT_ROOT / "vendor" / "__init__.py",
        "vendor.tsai_v1_0_1_minirocket":
            PROJECT_ROOT / "vendor" / "tsai_v1_0_1_minirocket" / "__init__.py",
        "vendor.tsai_v1_0_1_minirocket.minirocket_plus":
            PROJECT_ROOT / "vendor" / "tsai_v1_0_1_minirocket" / "minirocket_plus.py",
        "vendor.tsai_v1_0_1_minirocket.golden_handshake":
            PROJECT_ROOT / "vendor" / "tsai_v1_0_1_minirocket" / "golden_handshake.py",
    }
    required = {
        "mate_spc_minirocket_v8.backend": (
            "cuda_preflight", "validate_protocol", "preclaim_bind_method_graph",
            "run_development", "load_frozen_fold_pipeline", "load_final_refit_pipeline"),
        "mate_spc_minirocket_v8.model": (
            "set_determinism", "robust_fit_normalizer", "normalize_runs",
            "build_training_windows", "fit_feature_standardizer", "standardize_features",
            "fit_linear_head", "linear_logits"),
        "mate_spc_minirocket_v8.transform": (
            "fit_minirocket_transform", "transform_windows",
            "save_minirocket_state_npz", "load_minirocket_state_npz"),
        "mate_spc_minirocket_v8.monitor": (
            "compute_log_odds", "fit_monitor_calibration", "scan_batch",
            "inner_threshold", "apply_threshold"),
        "mate_spc_minirocket_v8.evaluation": ("fold_roles", "evaluate_gates"),
        "vendor.tsai_v1_0_1_minirocket.minirocket_plus": ("MiniRocketFeaturesPlus",),
        "vendor.tsai_v1_0_1_minirocket.golden_handshake": ("verify_golden_handshake",),
        _HASH_CONTRACT_NAME: (
            "canonical_json_file_bytes", "canonical_json_sha256", "sha256_file",
            "strict_json_from_bytes"),
    }
    records: dict[str, Any] = {}
    for module_name, expected_path in expected.items():
        module = _attest_graph_module(
            module_name, expected_path, implementation, required.get(module_name, ()))
        records[module_name] = {"path": str(expected_path.resolve()),
                                "file_bytes_sha256": sha256_file(expected_path)}
    method_path_set = {
        expected[name].resolve() for name in (
            "mate_spc_minirocket_v8.model", "mate_spc_minirocket_v8.transform",
            "mate_spc_minirocket_v8.monitor", "mate_spc_minirocket_v8.evaluation")}
    for name in _METHOD_REQUIRED_EXPORTS:
        value = getattr(method, name, None)
        source = inspect.getsourcefile(value) if callable(value) else None
        if not callable(value) or source is None or Path(source).resolve() not in method_path_set:
            raise RuntimeError(f"untrusted method API export: {name}")
    current = {name: sys.modules[name] for name in expected}
    if _METHOD_GRAPH_MODULES is None:
        _METHOD_API = method
        _METHOD_GRAPH_MODULES = current
    elif (method is not _METHOD_API or any(
            _METHOD_GRAPH_MODULES.get(name) is not module for name, module in current.items())):
        raise RuntimeError("formal scoring graph module object was replaced after binding")
    package = sys.modules["mate_spc_minirocket_v8"]
    if getattr(package, "method", None) is not method:
        raise RuntimeError("package method attribute differs from bound module")
    return {"bound": True, "hdf5_loaded": any(
                name == "h5py" or name.startswith("h5py.") for name in sys.modules),
            "modules": records}


@dataclass(frozen=True)
class _FoldPipeline:
    normalizer: Any
    transform: Any
    feature_standardizer: Any
    linear_head: Any


def _rng_state_hashes() -> Mapping[str, Any]:
    def digest_array(value: Any) -> str:
        return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()

    cuda = []
    if torch.cuda.is_available():
        cuda = [digest_array(state.detach().cpu().numpy()) for state in torch.cuda.get_rng_state_all()]
    return {
        "python_state_sha256": hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest(),
        "numpy_state_sha256": hashlib.sha256(repr(np.random.get_state()).encode("utf-8")).hexdigest(),
        "torch_cpu_state_sha256": digest_array(torch.get_rng_state().detach().cpu().numpy()),
        "torch_cuda_state_sha256": cuda,
    }


class _PreserveRng:
    """Prevent state reconstruction during scoring from advancing formal RNG."""

    def __enter__(self) -> "_PreserveRng":
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.cpu = torch.get_rng_state().clone()
        self.cuda = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)


def _causal_windows(normalized_runs: np.ndarray, endpoints: Sequence[int]) -> np.ndarray:
    values = np.asarray(normalized_runs)
    selected = tuple(int(endpoint) for endpoint in endpoints)
    if (values.dtype != np.dtype(np.float32) or values.ndim != 3 or
            values.shape[0] < 1 or values.shape[1:] != (500, 52)):
        raise ValueError("normalized runs must be float32 [n,500,52]")
    if not selected or any(endpoint < 127 or endpoint > 499 for endpoint in selected):
        raise ValueError("window endpoints must lie in 127..499")
    result = np.empty((values.shape[0] * len(selected), 52, 128), dtype=np.float32)
    cursor = 0
    for run in values:
        for endpoint in selected:
            result[cursor] = run[endpoint - 127:endpoint + 1].T
            cursor += 1
    return result


def _fit_pipeline(fit_by_idv: Mapping[int, Sequence[NumericRun]], *, device: str) -> _FoldPipeline:
    api = _method_api()
    if set(fit_by_idv) != set(CONDITION_IDVS):
        raise ValueError("fit mapping must contain H/F3/F9/F15")
    run_ids = np.asarray([run.simulation_run for run in fit_by_idv[1]], dtype=np.int64)
    if run_ids.shape != (60,) or not np.all(run_ids[1:] > run_ids[:-1]):
        raise ValueError("fit run IDs must be 60 strictly ascending identities")
    raw = {idv: _values(fit_by_idv[idv]) for idv in CONDITION_IDVS}
    normalizer = api.robust_fit_normalizer(raw[1])
    normalized = {idv: api.normalize_runs(raw[idv], normalizer) for idv in CONDITION_IDVS}
    training = api.build_training_windows(
        normalized[1], normalized[3], normalized[9], normalized[15],
        fit_run_ids=run_ids)
    transform = api.fit_minirocket_transform(training, device=device)
    features = api.transform_windows(training.windows, transform, device=device, batch_size=128)
    feature_standardizer = api.fit_feature_standardizer(features)
    standardized = api.standardize_features(features, feature_standardizer)
    linear_head = api.fit_linear_head(standardized, training.labels, device=device)
    return _FoldPipeline(normalizer, transform, feature_standardizer, linear_head)


_PREFIX_ENDPOINTS = (128, 132, 136, 140, 144, 148, 152, 156)
_MONITOR_ENDPOINTS = tuple(range(160, 500))
_COMPLETE_ENDPOINTS = _PREFIX_ENDPOINTS + _MONITOR_ENDPOINTS


def _score_complete(pipeline: _FoldPipeline, runs: Sequence[NumericRun], *,
                    device: str) -> np.ndarray:
    """Return q in [run, 8 prefix + 340 dense] order with bounded RAM."""
    api = _method_api()
    output = np.empty((len(runs), len(_COMPLETE_ENDPOINTS)), dtype=np.float64)
    run_batch_size = 4
    for start in range(0, len(runs), run_batch_size):
        selected = tuple(runs[start:start + run_batch_size])
        normalized = api.normalize_runs(_values(selected), pipeline.normalizer)
        windows = _causal_windows(normalized, _COMPLETE_ENDPOINTS)
        # A restored transform may allocate a temporary upstream module.
        # Scoring must not consume the one continuous formal RNG stream.
        with _PreserveRng():
            features = api.transform_windows(windows, pipeline.transform,
                                             device=device, batch_size=128)
        standardized = api.standardize_features(features, pipeline.feature_standardizer)
        logits = api.linear_logits(standardized, pipeline.linear_head,
                                   device=device, batch_size=128)
        output[start:start + len(selected)] = api.compute_log_odds(logits).reshape(
            len(selected), len(_COMPLETE_ENDPOINTS))
    if not np.isfinite(output).all():
        raise FloatingPointError("complete run scoring produced nonfinite q")
    return output


def _replay_inner(pipeline: _FoldPipeline, fit_healthy: Sequence[NumericRun],
                  inner_healthy: Sequence[NumericRun], *, device: str) -> Mapping[str, Any]:
    api = _method_api()
    fit_q = _score_complete(pipeline, fit_healthy, device=device)
    calibration = api.fit_monitor_calibration(fit_q)
    inner_q = _score_complete(pipeline, inner_healthy, device=device)
    scores = api.scan_batch(inner_q[:, 8:], inner_q[:, :8], calibration)
    return {"fit_q": fit_q, "calibration": calibration,
            "inner_q": inner_q, "scores": scores}


def _max_abs(first: Any, second: Any) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("replay arrays invalid")
    return float(np.max(np.abs(a - b), initial=0.0))


def _audit_inner(pipeline: _FoldPipeline, fit_healthy: Sequence[NumericRun],
                 inner_healthy: Sequence[NumericRun], *, device: str
                 ) -> tuple[Any, Any, Mapping[str, Any]]:
    if len(fit_healthy) != 60 or len(inner_healthy) != 20:
        raise ValueError("inner replay requires 60 fit and 20 inner healthy runs")
    api = _method_api()
    first = _replay_inner(pipeline, fit_healthy, inner_healthy, device=device)
    second = _replay_inner(pipeline, fit_healthy, inner_healthy, device=device)
    first_cal, second_cal = first["calibration"], second["calibration"]
    errors = (
        _max_abs(first["fit_q"], second["fit_q"]),
        abs(float(first_cal.center) - float(second_cal.center)),
        abs(float(first_cal.scale) - float(second_cal.scale)),
        _max_abs(first["inner_q"], second["inner_q"]),
        _max_abs(first["scores"].paths, second["scores"].paths),
        _max_abs(first["scores"].run_maxima, second["scores"].run_maxima),
    )
    maximum_error = max(errors)
    try:
        first_threshold = api.inner_threshold(first["scores"].run_maxima, maximum_error)
        second_threshold = api.inner_threshold(second["scores"].run_maxima, maximum_error)
    except (ValueError, FloatingPointError) as error:
        raise _FoldNumericNoGo(str(error)) from error
    first_decisions = api.apply_threshold(first["scores"], first_threshold)
    second_decisions = api.apply_threshold(second["scores"], second_threshold)
    discrete = bool(
        np.array_equal(first_decisions.alarms, second_decisions.alarms)
        and np.array_equal(first_decisions.alarm_endpoints, second_decisions.alarm_endpoints)
        and np.array_equal(first_decisions.delays, second_decisions.delays)
        and first_threshold.to_document() == second_threshold.to_document()
    )
    passes = bool(maximum_error <= CUDA_REPEAT_ATOL and discrete)
    if not passes:
        raise _FoldNumericNoGo("complete CUDA inner replay failed")
    report = {
        "schema": "gpu_minirocket_logodds_v8_authoritative_cuda_repeat",
        "scope": "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay",
        "device": device, "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": maximum_error,
        "component_maximum_absolute_errors": {
            "fit_healthy_log_odds": errors[0], "calibration_center": errors[1],
            "calibration_scale": errors[2], "inner_log_odds": errors[3],
            "scan_paths": errors[4], "run_maxima": errors[5],
        },
        "discrete_equal": discrete, "passes": passes,
        "first_run_maxima": first["scores"].run_maxima.tolist(),
        "second_run_maxima": second["scores"].run_maxima.tolist(),
    }
    return first_cal, first_threshold, report


def _atomic_npz(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _transform_structure_hash(transform: Any) -> str:
    digest = hashlib.sha256()
    items = getattr(transform, "state_items", ())
    selected = [(name, value) for name, value in items
                if name.startswith("channel_combinations_")]
    if len(selected) != 15:
        raise ValueError("transform state lacks 15 channel-combination masks")
    for name, value in selected:
        digest.update(name.encode("ascii"))
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_fold_artifacts(pipeline: _FoldPipeline, calibration: Any, threshold: Any,
                         fold_dir: Path, *, outer_fold: int, roles: Any,
                         protocol_hash: str, rng_hashes: Mapping[str, Any]
                         ) -> Mapping[str, Any]:
    api = _method_api()
    fold_dir.mkdir(parents=True, exist_ok=False)
    transform_path = fold_dir / "minirocket_transform.npz"
    api.save_minirocket_state_npz(transform_path, pipeline.transform)
    core_path = fold_dir / "fold_model_state.npz"
    _atomic_npz(
        core_path,
        metadata_json=np.asarray(json.dumps({
            "schema": "gpu_minirocket_logodds_v8_fold_pipeline_core",
            "candidate_id": CANDIDATE_ID, "outer_fold": outer_fold,
            "training_example_count": 20400,
            "head_source_device": pipeline.linear_head.source_device,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)),
        normalizer_location=np.asarray(pipeline.normalizer.location, dtype=np.float64),
        normalizer_scale=np.asarray(pipeline.normalizer.scale, dtype=np.float64),
        feature_mean=np.asarray(pipeline.feature_standardizer.mean, dtype=np.float64),
        feature_scale=np.asarray(pipeline.feature_standardizer.scale, dtype=np.float64),
        head_weight=np.asarray(pipeline.linear_head.weight, dtype=np.float32),
        head_bias=np.asarray(pipeline.linear_head.bias, dtype=np.float32),
        head_initial_objective=np.asarray(pipeline.linear_head.initial_objective, dtype=np.float64),
        head_final_objective=np.asarray(pipeline.linear_head.final_objective, dtype=np.float64),
        head_final_max_abs_gradient=np.asarray(pipeline.linear_head.final_max_abs_gradient, dtype=np.float64),
        calibration_center=np.asarray(calibration.center, dtype=np.float64),
        calibration_scale=np.asarray(calibration.scale, dtype=np.float64),
    )
    threshold_path = fold_dir / "inner_threshold.json"
    _atomic_json(threshold_path, {
        "schema": "gpu_minirocket_logodds_v8_inner_threshold",
        "outer_fold": outer_fold, "candidate_id": CANDIDATE_ID,
        **_plain_json(threshold.to_document()),
    })
    structure_hash = _transform_structure_hash(pipeline.transform)
    manifest = {
        "schema": "gpu_minirocket_logodds_v8_fold_freeze_manifest",
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": CANDIDATE_ID, "outer_fold": outer_fold,
        "roles": _plain_json(roles), "fit_runs_per_class": 60,
        "inner_threshold_healthy_runs": 20, "inner_threshold_fault_runs": 0,
        "training_examples": 20400, "effective_feature_count": 9996,
        "transform_structure_sha256": structure_hash,
        "rng_state_after_fit_and_inner_replay": _plain_json(rng_hashes),
        "frozen_before_outer_scoring": True,
        "transform": _file_record(transform_path),
        "fold_model_state": _file_record(core_path),
        "threshold": _file_record(threshold_path),
    }
    manifest_path = fold_dir / "freeze_manifest.json"
    _atomic_json(manifest_path, manifest)
    return {
        "transform": manifest["transform"], "fold_model_state": manifest["fold_model_state"],
        "threshold": manifest["threshold"], "freeze_manifest": _file_record(manifest_path),
        "transform_structure_sha256": structure_hash,
    }


def _verified_local_record(value: Any, *, parent: Path, filename: str) -> Path:
    if (not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or
            Path(value["path"]).resolve() != (parent / filename).resolve() or
            not isinstance(value.get("size_bytes"), int) or
            isinstance(value.get("size_bytes"), bool) or value["size_bytes"] <= 0 or
            not isinstance(value.get("file_bytes_sha256"), str)):
        raise ValueError(f"invalid frozen artifact record: {filename}")
    path = (parent / filename).resolve()
    if (not path.is_file() or path.stat().st_size != value["size_bytes"] or
            sha256_file(path) != value["file_bytes_sha256"]):
        raise ValueError(f"frozen artifact bytes changed: {filename}")
    return path


def _load_core_arrays(path: Path, *, expected_schema: str,
                      expected_training_examples: int,
                      include_calibration: bool) -> tuple[Any, Any, Any, Any | None,
                                                          Mapping[str, Any]]:
    api = _method_api()
    base_names = {
        "metadata_json", "normalizer_location", "normalizer_scale", "feature_mean",
        "feature_scale", "head_weight", "head_bias", "head_initial_objective",
        "head_final_objective", "head_final_max_abs_gradient",
    }
    expected_names = base_names | ({"calibration_center", "calibration_scale"}
                                   if include_calibration else set())
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("pipeline NPZ member set changed")
        if any(archive[name].dtype.hasobject for name in archive.files):
            raise ValueError("object arrays are forbidden in pipeline NPZ")
        raw_metadata = archive["metadata_json"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in "US":
            raise ValueError("pipeline metadata_json must be a scalar string")
        metadata_text = str(raw_metadata.item())
        metadata = strict_json_from_bytes(metadata_text.encode("utf-8"))
        expected_metadata_keys = ({"schema", "candidate_id", "outer_fold",
                                   "training_example_count", "head_source_device"}
                                  if include_calibration else
                                  {"schema", "candidate_id", "training_example_count",
                                   "runs_per_class", "target_threshold_fitted",
                                   "head_source_device", "rng_before_final_refit",
                                   "rng_after_final_refit"})
        if (not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_keys or
                json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) != metadata_text or
                metadata.get("schema") != expected_schema or
                metadata.get("candidate_id") != CANDIDATE_ID or
                metadata.get("training_example_count") != expected_training_examples or
                not isinstance(metadata.get("head_source_device"), str) or
                (include_calibration and
                 (not isinstance(metadata.get("outer_fold"), int) or
                  isinstance(metadata.get("outer_fold"), bool) or
                  metadata.get("outer_fold") not in range(5))) or
                (not include_calibration and
                 (metadata.get("runs_per_class") != 100 or
                  metadata.get("target_threshold_fitted") is not False or
                  not isinstance(metadata.get("rng_before_final_refit"), Mapping) or
                  not isinstance(metadata.get("rng_after_final_refit"), Mapping)))):
            raise ValueError("pipeline metadata changed")
        arrays = {name: np.array(archive[name], copy=True)
                  for name in expected_names if name != "metadata_json"}
    specifications = {
        "normalizer_location": ((52,), np.dtype(np.float64)),
        "normalizer_scale": ((52,), np.dtype(np.float64)),
        "feature_mean": ((9996,), np.dtype(np.float64)),
        "feature_scale": ((9996,), np.dtype(np.float64)),
        "head_weight": ((4, 9996), np.dtype(np.float32)),
        "head_bias": ((4,), np.dtype(np.float32)),
        "head_initial_objective": ((), np.dtype(np.float64)),
        "head_final_objective": ((), np.dtype(np.float64)),
        "head_final_max_abs_gradient": ((), np.dtype(np.float64)),
    }
    if include_calibration:
        specifications.update({"calibration_center": ((), np.dtype(np.float64)),
                               "calibration_scale": ((), np.dtype(np.float64))})
    for name, (shape, dtype) in specifications.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"{name} has wrong persisted shape or dtype")
    def scalar(name: str) -> float:
        value = arrays[name]
        if value.shape != () or value.dtype != np.dtype(np.float64):
            raise ValueError(f"{name} must be scalar float64")
        return float(value)
    normalizer = api.RobustNormalizer(
        arrays["normalizer_location"], arrays["normalizer_scale"])
    standardizer = api.FeatureStandardizer(
        arrays["feature_mean"], arrays["feature_scale"])
    head = api.LinearHeadState(
        weight=arrays["head_weight"], bias=arrays["head_bias"],
        initial_objective=scalar("head_initial_objective"),
        final_objective=scalar("head_final_objective"),
        final_max_abs_gradient=scalar("head_final_max_abs_gradient"),
        source_device=str(metadata["head_source_device"]))
    calibration = None
    if include_calibration:
        calibration = api.MonitorCalibration(
            center=scalar("calibration_center"), scale=scalar("calibration_scale"))
    return normalizer, standardizer, head, calibration, metadata


def load_frozen_fold_pipeline(manifest_path: str | os.PathLike[str]
                              ) -> tuple[_FoldPipeline, Any, Any]:
    """Strictly reload one frozen fold bundle with pickle disabled."""
    manifest_file = Path(manifest_path).resolve()
    parent = manifest_file.parent
    manifest, _payload = _read_canonical_json(manifest_file)
    fold = manifest.get("outer_fold")
    api = _method_api()
    if (not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5) or
            manifest.get("schema") != "gpu_minirocket_logodds_v8_fold_freeze_manifest" or
            manifest.get("candidate_id") != CANDIDATE_ID or
            manifest.get("roles") != api.fold_roles(fold).to_document() or
            manifest.get("fit_runs_per_class") != 60 or
            manifest.get("inner_threshold_healthy_runs") != 20 or
            manifest.get("inner_threshold_fault_runs") != 0 or
            manifest.get("training_examples") != 20400 or
            manifest.get("effective_feature_count") != 9996 or
            manifest.get("frozen_before_outer_scoring") is not True):
        raise ValueError("fold freeze manifest semantic mismatch")
    transform_path = _verified_local_record(
        manifest.get("transform"), parent=parent, filename="minirocket_transform.npz")
    core_path = _verified_local_record(
        manifest.get("fold_model_state"), parent=parent, filename="fold_model_state.npz")
    threshold_path = _verified_local_record(
        manifest.get("threshold"), parent=parent, filename="inner_threshold.json")
    transform = api.load_minirocket_state_npz(transform_path)
    if transform.source_device != "cuda:0":
        raise ValueError("formal fold transform source_device must be exact cuda:0")
    if _transform_structure_hash(transform) != manifest.get("transform_structure_sha256"):
        raise ValueError("restored transform structure hash mismatch")
    normalizer, standardizer, head, calibration, metadata = _load_core_arrays(
        core_path, expected_schema="gpu_minirocket_logodds_v8_fold_pipeline_core",
        expected_training_examples=20400, include_calibration=True)
    if metadata.get("outer_fold") != fold:
        raise ValueError("fold core metadata outer_fold mismatch")
    if head.source_device != "cuda":
        raise ValueError("formal fold linear-head source_device must be exact cuda")
    threshold_doc, _ = _read_canonical_json(threshold_path)
    threshold = api.InnerThreshold(
        threshold=threshold_doc.get("threshold"), rank19=threshold_doc.get("rank19"),
        distinct_count=threshold_doc.get("distinct_count"),
        repeat_error=threshold_doc.get("repeat_error"))
    if threshold_doc != {"schema": "gpu_minirocket_logodds_v8_inner_threshold",
                         "outer_fold": fold, "candidate_id": CANDIDATE_ID,
                         **threshold.to_document()}:
        raise ValueError("restored threshold document mismatch")
    assert calibration is not None
    return _FoldPipeline(normalizer, transform, standardizer, head), calibration, threshold


def _evaluation_rows(runs: Sequence[NumericRun], score: Any, decisions: Any
                     ) -> tuple[Any, ...]:
    api = _method_api()
    if len(runs) != len(decisions.alarms):
        raise ValueError("outer decision count changed")
    if not np.array_equal(score.run_maxima, decisions.run_maxima):
        raise FloatingPointError("threshold decisions changed run maxima")
    return tuple(api.RunDecision(
        source_idv=run.source_idv, simulation_run=run.simulation_run,
        run_maximum=float(score.run_maxima[index]),
        alarm_endpoint=(int(decisions.alarm_endpoints[index])
                        if bool(decisions.alarms[index]) else None))
                 for index, run in enumerate(runs))


def _fold_metrics(rows: Sequence[Any]) -> Mapping[str, Any]:
    healthy = tuple(row for row in rows if row.source_idv == 1)
    faults = {}
    for fault in (3, 9, 15):
        selected = tuple(row for row in rows if row.source_idv == fault)
        delays = np.asarray([row.delay for row in selected], dtype=np.float64)
        detected = sum(row.alarm for row in selected)
        faults[str(fault)] = {
            "runs": len(selected), "detections": detected,
            "detection_rate": detected / len(selected),
            "median_delay_samples": float(np.median(delays)),
        }
    return {"healthy": {"runs": len(healthy), "false_alarms": sum(row.alarm for row in healthy)},
            "faults": faults}


def _score_with_calibration(pipeline: _FoldPipeline, runs: Sequence[NumericRun],
                            calibration: Any, *, device: str) -> tuple[np.ndarray, Any]:
    api = _method_api()
    q = _score_complete(pipeline, runs, device=device)
    score = api.scan_batch(q[:, 8:], q[:, :8], calibration)
    return q, score


def _cpu_report(pipeline: _FoldPipeline, runs: Sequence[NumericRun],
                calibration: Any, authoritative_score: Any, threshold: Any,
                authoritative_decisions: Any) -> Mapping[str, Any]:
    api = _method_api()
    try:
        _q, score = _score_with_calibration(pipeline, runs, calibration, device="cpu")
        decisions = api.apply_threshold(score, threshold)
        maximum = _max_abs(authoritative_score.paths, score.paths)
        discrete = bool(np.array_equal(authoritative_decisions.alarms, decisions.alarms)
                        and np.array_equal(authoritative_decisions.alarm_endpoints,
                                           decisions.alarm_endpoints))
        return {"is_go_gate": False, "authoritative_device": "cuda",
                "comparison_device": "cpu", "completed": True,
                "maximum_absolute_error": maximum, "discrete_equal": discrete}
    except Exception as error:
        return {"is_go_gate": False, "authoritative_device": "cuda",
                "comparison_device": "cpu", "completed": False,
                "error_type": type(error).__name__, "error": str(error)}


def _save_outer_result(fold_dir: Path, *, outer_fold: int,
                       rows: Sequence[Any], score: Any) -> Mapping[str, Any]:
    path = fold_dir / "outer_result.json"
    _atomic_json(path, {
        "schema": "gpu_minirocket_logodds_v8_outer_fold_result",
        "outer_fold": outer_fold, "authoritative_device": "cuda",
        "scored_once_after_freeze": True, "metrics": _fold_metrics(rows),
        "decisions": [row.to_document() for row in rows],
        "replay": [{"identity": {"source_idv": row.source_idv,
                                  "simulation_run": row.simulation_run},
                    "score_path_endpoints_160_499": score.paths[index].tolist()}
                   for index, row in enumerate(rows)],
    })
    return _file_record(path)


def _execute_fold(blocks: Sequence[NumericBlock], *, outer_fold: int,
                  artifact_root: Path, protocol_canonical_json_sha256: str,
                  device: str,
                  prior_transform_structure_hashes: Sequence[str] = ()
                  ) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    _progress(f"fold {outer_fold}: start")
    api = _method_api()
    roles = api.fold_roles(outer_fold)
    fit_by_idv = {idv: _select_runs(blocks, roles.fit, (idv,)) for idv in CONDITION_IDVS}
    inner_healthy = _select_runs(blocks, roles.inner_healthy, (1,))
    outer_runs = _select_runs(blocks, roles.outer, CONDITION_IDVS)
    if any(run.source_idv != 1 for run in inner_healthy):
        raise RuntimeError("fault values entered inner threshold")
    pipeline = _fit_pipeline(fit_by_idv, device=device)
    _progress(f"fold {outer_fold}: fit done")
    calibration, threshold, repeat = _audit_inner(
        pipeline, fit_by_idv[1], inner_healthy, device=device)
    _progress(f"fold {outer_fold}: inner audit done")
    rng_hashes = _rng_state_hashes()
    fold_dir = artifact_root / f"fold_{outer_fold}"
    frozen = _save_fold_artifacts(
        pipeline, calibration, threshold, fold_dir, outer_fold=outer_fold,
        roles=roles.to_document(), protocol_hash=protocol_canonical_json_sha256,
        rng_hashes=rng_hashes)
    if frozen["transform_structure_sha256"] in set(prior_transform_structure_hashes):
        raise RuntimeError(
            "fold transform RNG structure repeated before outer scoring; prohibited reseed detected"
        )
    # The authoritative outer score uses only the just-frozen, strictly
    # reloaded no-pickle bundle, proving the artifact is operational.
    pipeline, calibration, threshold = load_frozen_fold_pipeline(
        frozen["freeze_manifest"]["path"])
    _progress(f"fold {outer_fold}: frozen pipeline reload verified")
    _q, score = _score_with_calibration(pipeline, outer_runs, calibration, device=device)
    decisions = api.apply_threshold(score, threshold)
    rows = _evaluation_rows(outer_runs, score, decisions)
    outer_record = _save_outer_result(
        fold_dir, outer_fold=outer_fold, rows=rows, score=score)
    cpu = _cpu_report(pipeline, outer_runs, calibration, score, threshold, decisions)
    _progress(f"fold {outer_fold}: outer and CPU report done")
    return {
        "outer_fold": outer_fold, "roles": roles.to_document(),
        "fit_runs_per_class": 60, "fit_total_run_count": 240,
        "inner_threshold_healthy_run_count": 20,
        "inner_threshold_fault_run_count": 0,
        "outer_run_count_per_class": 20, "training_example_count": 20400,
        "threshold": threshold.to_document(), "monitor_calibration": calibration.to_document(),
        "authoritative_cuda_repeat": repeat, "cross_device_report_only": cpu,
        "numeric_valid": True, "artifacts_frozen_before_outer": frozen,
        "outer_result_append_only": outer_record, "metrics": _fold_metrics(rows),
        "decisions": [row.to_document() for row in rows],
    }, rows


def _persist_development_decision_before_final_refit(
    artifact_root: Path, *, protocol_canonical_json_sha256: str, decision: str,
    folds: Sequence[Mapping[str, Any]], gates: Mapping[str, Any],
    aggregate: Mapping[str, Any] | None, authoritative_cuda_repeat: Mapping[str, Any],
    cross_device_report_only: Mapping[str, Any]) -> Mapping[str, Any]:
    if decision not in {"GO", "DEVELOPMENT_NO_GO"} or len(folds) != 5:
        raise ValueError("pre-refit decision requires one valid decision and five folds")
    document = {
        "schema": "gpu_minirocket_logodds_v8_development_decision_before_final_refit",
        "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID, "candidate_count": 1,
        "activation_result_file_bytes_sha256": ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": ACTIVATION_DECISION_SHA256,
        "source_file_bytes_sha256": MODE2_SHA256, "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "development_decision_determined": True, "decision": decision,
        "selected_candidate_id": CANDIDATE_ID if decision == "GO" else None,
        "folds": _plain_json(folds), "gates": _plain_json(gates),
        "aggregate": _plain_json(aggregate),
        "authoritative_cuda_repeat": _plain_json(authoritative_cuda_repeat),
        "cross_device_report_only": _plain_json(cross_device_report_only),
        "mode1_accessed": False, "mode1_opens": 0,
        "rieth_accessed": False, "rieth_opens": 0,
        "inner_threshold_fault_run_count": 0,
        "final_refit_permitted": decision == "GO",
        "final_refit_started": False, "final_model_usable": False,
        "no_go_terminates_current_development_pool_method_search":
            decision == "DEVELOPMENT_NO_GO",
    }
    path = artifact_root / "development_decision_before_final_refit.json"
    _atomic_json(path, document)
    return _file_record(path)


def _save_complete_pipeline(path: Path, pipeline: _FoldPipeline, *,
                            training_example_count: int,
                            rng_before: Mapping[str, Any],
                            rng_after: Mapping[str, Any]) -> None:
    arrays: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps({
            "schema": "gpu_minirocket_logodds_v8_complete_pipeline",
            "candidate_id": CANDIDATE_ID, "training_example_count": training_example_count,
            "runs_per_class": 100, "target_threshold_fitted": False,
            "head_source_device": pipeline.linear_head.source_device,
            "rng_before_final_refit": _plain_json(rng_before),
            "rng_after_final_refit": _plain_json(rng_after),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)),
        "normalizer_location": pipeline.normalizer.location,
        "normalizer_scale": pipeline.normalizer.scale,
        "feature_mean": pipeline.feature_standardizer.mean,
        "feature_scale": pipeline.feature_standardizer.scale,
        "head_weight": pipeline.linear_head.weight,
        "head_bias": pipeline.linear_head.bias,
        "head_initial_objective": np.asarray(pipeline.linear_head.initial_objective, dtype=np.float64),
        "head_final_objective": np.asarray(pipeline.linear_head.final_objective, dtype=np.float64),
        "head_final_max_abs_gradient": np.asarray(pipeline.linear_head.final_max_abs_gradient, dtype=np.float64),
    }
    _atomic_npz(path, **arrays)


def load_final_refit_pipeline(manifest_path: str | os.PathLike[str]) -> _FoldPipeline:
    """Strictly reload the post-GO no-threshold Mode2 pipeline bundle."""
    manifest_file = Path(manifest_path).resolve()
    parent = manifest_file.parent
    manifest, _payload = _read_canonical_json(manifest_file)
    if (manifest.get("schema") != "gpu_minirocket_logodds_v8_final_mode2_refit" or
            manifest.get("candidate_id") != CANDIDATE_ID or
            manifest.get("runs_per_class") != 100 or
            manifest.get("total_fit_run_count") != 400 or
            manifest.get("training_example_count") != 34_000 or
            manifest.get("created_only_after_development_go") is not True or
            manifest.get("continued_rng_stream_without_reseed") is not True or
            manifest.get("target_domain_threshold_not_fitted") is not True or
            manifest.get("mode1_accessed") is not False or manifest.get("mode1_opens") != 0):
        raise ValueError("final-refit manifest semantic mismatch")
    transform_path = _verified_local_record(
        manifest.get("transform"), parent=parent, filename="minirocket_transform.npz")
    state_path = _verified_local_record(
        manifest.get("state"), parent=parent, filename="final_model_state.npz")
    api = _method_api()
    transform = api.load_minirocket_state_npz(transform_path)
    if transform.source_device != "cuda:0":
        raise ValueError("formal final transform source_device must be exact cuda:0")
    normalizer, standardizer, head, calibration, metadata = _load_core_arrays(
        state_path, expected_schema="gpu_minirocket_logodds_v8_complete_pipeline",
        expected_training_examples=34_000, include_calibration=False)
    if (metadata.get("rng_before_final_refit") != manifest.get("rng_state_before") or
            metadata.get("rng_after_final_refit") != manifest.get("rng_state_after")):
        raise ValueError("final-refit core RNG metadata mismatch")
    if head.source_device != "cuda":
        raise ValueError("formal final linear-head source_device must be exact cuda")
    if calibration is not None:
        raise RuntimeError("final refit must not contain target threshold calibration")
    return _FoldPipeline(normalizer, transform, standardizer, head)


def _pipeline_window_log_odds(pipeline: _FoldPipeline, windows: np.ndarray, *,
                              device: str) -> np.ndarray:
    api = _method_api()
    with _PreserveRng():
        features = api.transform_windows(windows, pipeline.transform, device=device,
                                         batch_size=128)
    standardized = api.standardize_features(features, pipeline.feature_standardizer)
    logits = api.linear_logits(standardized, pipeline.linear_head, device=device,
                               batch_size=128)
    return api.compute_log_odds(logits)


def _run_final_refit(blocks: Sequence[NumericBlock], *, artifact_root: Path,
                     protocol_canonical_json_sha256: str) -> Mapping[str, Any]:
    """Continue the single RNG stream into the post-GO 100-run refit."""
    api = _method_api()
    selected = {idv: _select_runs(blocks, range(1, 101), (idv,)) for idv in CONDITION_IDVS}
    if any(len(runs) != 100 for runs in selected.values()):
        raise RuntimeError("final refit requires exactly 100 runs per class")
    raw = {idv: _values(selected[idv]) for idv in CONDITION_IDVS}
    before = _rng_state_hashes()
    normalizer = api.robust_fit_normalizer_final_refit(raw[1])
    normalized = {idv: api.normalize_runs(raw[idv], normalizer) for idv in CONDITION_IDVS}
    training = api.build_final_refit_training_windows(
        normalized[1], normalized[3], normalized[9], normalized[15],
        fit_run_ids=np.arange(1, 101, dtype=np.int64))
    if training.windows.shape[0] != 34_000:
        raise RuntimeError("final-refit window count changed")
    transform = api.fit_minirocket_transform_final_refit(training, device="cuda")
    features = api.transform_windows(training.windows, transform, device="cuda", batch_size=128)
    standardizer = api.fit_feature_standardizer_final_refit(features)
    standardized = api.standardize_features(features, standardizer)
    head = api.fit_linear_head_final_refit(standardized, training.labels, device="cuda")
    pipeline = _FoldPipeline(normalizer, transform, standardizer, head)
    after = _rng_state_hashes()
    final_dir = artifact_root / "final_refit"
    final_dir.mkdir(parents=True, exist_ok=False)
    transform_path = final_dir / "minirocket_transform.npz"
    api.save_minirocket_state_npz(transform_path, pipeline.transform)
    state_path = final_dir / "final_model_state.npz"
    _save_complete_pipeline(state_path, pipeline, training_example_count=34_000,
                            rng_before=before, rng_after=after)
    manifest = {
        "schema": "gpu_minirocket_logodds_v8_final_mode2_refit",
        "protocol_canonical_json_sha256": protocol_canonical_json_sha256,
        "candidate_id": CANDIDATE_ID, "runs_per_class": 100,
        "total_fit_run_count": 400, "training_example_count": 34_000,
        "created_only_after_development_go": True,
        "continued_rng_stream_without_reseed": True,
        "rng_state_before": before, "rng_state_after": after,
        "target_domain_threshold_not_fitted": True,
        "mode1_accessed": False, "mode1_opens": 0,
        "transform": _file_record(transform_path),
        "state": _file_record(state_path),
    }
    manifest_path = final_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    restored = load_final_refit_pipeline(manifest_path)
    reference_q = _pipeline_window_log_odds(pipeline, training.windows[:2], device="cuda")
    restored_q = _pipeline_window_log_odds(restored, training.windows[:2], device="cuda")
    error = _max_abs(reference_q, restored_q)
    if error != 0.0:
        raise FloatingPointError("final-refit save/load roundtrip changed log-odds")
    return {**manifest, "manifest": _file_record(manifest_path),
            "strict_no_pickle_reload_verified": True,
            "reload_log_odds_max_abs_error": error}


def _validate_protocol_hash(value: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("protocol hash must be lowercase SHA-256")
    return value


def _finalize_access_state(state: _AccessState) -> Mapping[str, Any]:
    state.update(current_fold=None, backend_completed=True)
    return {"final_state": _plain_json(state.value), "artifact": _file_record(state.path)}


def _numeric_no_go_result(*, artifact_root: Path, protocol_hash: str,
                          environment: Mapping[str, Any], topology: Mapping[str, Any],
                          access_state: _AccessState,
                          completed_folds: Sequence[Mapping[str, Any]],
                          failed_fold: int, error: BaseException) -> Mapping[str, Any]:
    folds = list(completed_folds)
    folds.append({"outer_fold": failed_fold, "numeric_valid": False,
                  "status": "FOLD_NUMERIC_NO_GO",
                  "error_type": type(error).__name__, "error": str(error)})
    folds.extend({"outer_fold": fold, "numeric_valid": False,
                  "status": "NOT_RUN_AFTER_NUMERIC_NO_GO"}
                 for fold in range(failed_fold + 1, 5))
    gates = {"all_pass": False, "checks": {"numeric_all_folds": False},
             "aggregate_metrics": None}
    repeat = {
        "scope": "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay",
        "atol": CUDA_REPEAT_ATOL, "maximum_absolute_error": None,
        "discrete_equal": False, "passes": False, "folds": [],
    }
    cpu = {"is_go_gate": False, "folds": []}
    decision_record = _persist_development_decision_before_final_refit(
        artifact_root, protocol_canonical_json_sha256=protocol_hash,
        decision="DEVELOPMENT_NO_GO", folds=folds, gates=gates, aggregate=None,
        authoritative_cuda_repeat=repeat, cross_device_report_only=cpu)
    access_state.update(decision_persisted=True)
    state_record = _finalize_access_state(access_state)
    return {
        "schema": BACKEND_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": CANDIDATE_ID, "candidate_count": 1,
        "activation_result_file_bytes_sha256": ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": ACTIVATION_DECISION_SHA256,
        "source_file_bytes_sha256": MODE2_SHA256, "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "topology": _plain_json(topology), "environment": _plain_json(environment),
        "backend_access_state": state_record, "mode1_accessed": False, "mode1_opens": 0,
        "rieth_accessed": False, "rieth_opens": 0,
        "inner_threshold_fault_run_count": 0,
        "decision": "DEVELOPMENT_NO_GO", "selected_candidate_id": None,
        "folds": folds, "gates": gates, "aggregate": None,
        "authoritative_cuda_repeat": repeat, "cross_device_report_only": cpu,
        "development_decision_before_final_refit": decision_record,
        "final_refit": None,
        "method_search_on_current_pool_terminated": True,
    }


def run_development(*, access_plan: Sequence[Mapping[str, Any]],
                    fold_map: Mapping[int, int] | Mapping[str, int],
                    artifact_root: str | os.PathLike[str],
                    protocol_canonical_json_sha256: str,
                    protocol_path: str | os.PathLike[str],
                    protocol_file_bytes_sha256: str,
                    protocol_commit_file_bytes_sha256: str,
                    claim_path: str | os.PathLike[str],
                    claim_file_bytes_sha256: str,
                    claim_commit_file_bytes_sha256: str,
                    access_intent_path: str | os.PathLike[str],
                    access_intent_file_bytes_sha256: str,
                    access_intent_commit_file_bytes_sha256: str,
                    device: str = "cuda") -> Mapping[str, Any]:
    """Execute the five fixed folds and only then a possible GO refit."""
    validated_access = _validate_access_plan(access_plan)
    validated_folds = _validate_fold_map(fold_map)
    protocol_hash = _validate_protocol_hash(protocol_canonical_json_sha256)
    root = Path(artifact_root).resolve()
    authorization = _validate_run_authorization(
        artifact_root=root, protocol_canonical_json_sha256=protocol_hash,
        protocol_path=protocol_path,
        protocol_file_bytes_sha256=protocol_file_bytes_sha256,
        protocol_commit_file_bytes_sha256=protocol_commit_file_bytes_sha256,
        claim_path=claim_path, claim_file_bytes_sha256=claim_file_bytes_sha256,
        claim_commit_file_bytes_sha256=claim_commit_file_bytes_sha256,
        access_intent_path=access_intent_path,
        access_intent_file_bytes_sha256=access_intent_file_bytes_sha256,
        access_intent_commit_file_bytes_sha256=access_intent_commit_file_bytes_sha256)
    fixed_directory = Path(access_intent_path).resolve().parent
    execution_claim_path = fixed_directory / "numeric_execution.claim.json"
    prior_artifacts = (
        execution_claim_path, root, fixed_directory / "failure.json",
        fixed_directory / "completion.json", fixed_directory / "development_result.json")
    if any(path.exists() for path in prior_artifacts):
        raise FileExistsError("prior numeric execution claim/result residue forbids replay")
    execution_claim = _exclusive_json(execution_claim_path, {
        "schema": "gpu_minirocket_logodds_v8_numeric_execution_claim",
        "protocol_canonical_json_sha256": protocol_hash,
        "protocol_file_bytes_sha256": protocol_file_bytes_sha256,
        "protocol_commit_file_bytes_sha256": protocol_commit_file_bytes_sha256,
        "claim_file_bytes_sha256": claim_file_bytes_sha256,
        "claim_commit_file_bytes_sha256": claim_commit_file_bytes_sha256,
        "access_intent_file_bytes_sha256": access_intent_file_bytes_sha256,
        "access_intent_commit_file_bytes_sha256":
            access_intent_commit_file_bytes_sha256,
        "artifact_root": str(root),
        "permanent_even_after_failure": True,
        "same_authorization_replay_allowed": False,
        "created_before_hdf5_import_or_open": True,
    })
    authorization = {**authorization, "numeric_execution_claim": execution_claim}
    # The parent fixed run directory already exists.  This atomic mkdir is a
    # second, independently visible lease; never accept a pre-created root.
    root.mkdir(parents=False, exist_ok=False)
    access_state = _AccessState(root / "backend_access_state.json",
                                protocol_canonical_json_sha256=protocol_hash)
    environment = _require_cuda(device)
    topology, raw_blocks = _load_numeric_blocks(validated_access, access_state=access_state)
    blocks = _validate_blocks(raw_blocks, validated_folds)
    _progress("authorized Mode2 source loaded")

    # Exactly one formal reset.  Every fold and an optional GO refit consume
    # this one stream in strict fold-0..4 order; scoring is RNG-neutral.
    api = _method_api()
    api.set_determinism(GLOBAL_SEED)
    formal_rng_start = _rng_state_hashes()
    access_state.update(training_or_fit_started=True, current_fold=0)

    fold_documents: list[Mapping[str, Any]] = []
    all_rows: list[Any] = []
    transform_structure_hashes: list[str] = []
    for outer_fold in range(5):
        access_state.update(current_fold=outer_fold)
        try:
            document, rows = _execute_fold(
                blocks, outer_fold=outer_fold, artifact_root=root,
                protocol_canonical_json_sha256=protocol_hash, device=device,
                prior_transform_structure_hashes=tuple(transform_structure_hashes))
        except (_FoldNumericNoGo, FloatingPointError, np.linalg.LinAlgError) as error:
            result = _numeric_no_go_result(
                artifact_root=root, protocol_hash=protocol_hash,
                environment=environment, topology=topology, access_state=access_state,
                completed_folds=fold_documents, failed_fold=outer_fold, error=error)
            result["run_authorization"] = authorization
            _atomic_json(root / "development_result.json", result)
            _progress("decision persisted: DEVELOPMENT_NO_GO")
            return result
        structure_hash = str(document["artifacts_frozen_before_outer"]["transform_structure_sha256"])
        if structure_hash in transform_structure_hashes:
            raise RuntimeError("fold transform RNG structure repeated; prohibited reseed detected")
        transform_structure_hashes.append(structure_hash)
        fold_documents.append(document)
        all_rows.extend(rows)
        access_state.update(folds_completed=outer_fold + 1,
                            current_fold=(outer_fold + 1 if outer_fold < 4 else None))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gate_report = api.evaluate_gates(
        all_rows, [bool(document["numeric_valid"]) for document in fold_documents])
    gates = gate_report.to_document()
    decision = "GO" if gate_report.all_pass else "DEVELOPMENT_NO_GO"
    repeat_folds = [document["authoritative_cuda_repeat"] for document in fold_documents]
    repeat = {
        "scope": "complete_inner_cuda_log_odds_calibration_scan_threshold_decision_replay",
        "atol": CUDA_REPEAT_ATOL,
        "maximum_absolute_error": max(float(item["maximum_absolute_error"]) for item in repeat_folds),
        "discrete_equal": all(bool(item["discrete_equal"]) for item in repeat_folds),
        "passes": all(bool(item["passes"]) for item in repeat_folds),
        "folds": repeat_folds,
    }
    cpu_folds = [document["cross_device_report_only"] for document in fold_documents]
    completed_cpu = [item for item in cpu_folds if item.get("completed") is True]
    cpu = {
        "is_go_gate": False, "completed_fold_count": len(completed_cpu),
        "maximum_absolute_error": (
            max(float(item["maximum_absolute_error"]) for item in completed_cpu)
            if completed_cpu else None),
        "discrete_equal": bool(len(completed_cpu) == 5 and
                               all(bool(item["discrete_equal"]) for item in completed_cpu)),
        "folds": cpu_folds,
    }
    aggregate = gates["aggregate_metrics"]
    decision_record = _persist_development_decision_before_final_refit(
        root, protocol_canonical_json_sha256=protocol_hash, decision=decision,
        folds=fold_documents, gates=gates, aggregate=aggregate,
        authoritative_cuda_repeat=repeat, cross_device_report_only=cpu)
    access_state.update(decision_persisted=True)
    _progress(f"decision persisted: {decision}")
    final_refit = None
    if decision == "GO":
        access_state.update(final_refit_started=True)
        _progress("GO final refit: start")
        final_refit = _plain_json(_run_final_refit(
            blocks, artifact_root=root,
            protocol_canonical_json_sha256=protocol_hash))
        _progress("GO final refit: done")
    state_record = _finalize_access_state(access_state)
    result = {
        "schema": BACKEND_RESULT_SCHEMA, "protocol_id": PROTOCOL_ID,
        "protocol_canonical_json_sha256": protocol_hash,
        "candidate_id": CANDIDATE_ID, "candidate_count": 1,
        "activation_result_file_bytes_sha256": ACTIVATION_RESULT_SHA256,
        "activation_decision_file_bytes_sha256": ACTIVATION_DECISION_SHA256,
        "source_file_bytes_sha256": MODE2_SHA256, "source_size_bytes": SOURCE_SIZE_BYTES,
        "access_plan_canonical_json_sha256": ACCESS_PLAN_CANONICAL_JSON_SHA256,
        "fold_map_canonical_json_sha256": FOLD_MAP_CANONICAL_JSON_SHA256,
        "topology": _plain_json(topology), "environment": _plain_json(environment),
        "run_authorization": authorization,
        "formal_rng": {
            "seeded_exactly_once_after_environment_preflight": True,
            "seed": GLOBAL_SEED, "start_state": formal_rng_start,
            "fold_order": [0, 1, 2, 3, 4],
            "transform_structure_sha256_by_fold": transform_structure_hashes,
            "all_fold_structures_distinct": len(set(transform_structure_hashes)) == 5,
            "go_refit_continues_without_reseed": decision == "GO",
        },
        "backend_access_state": state_record,
        "mode1_accessed": False, "mode1_opens": 0,
        "rieth_accessed": False, "rieth_opens": 0,
        "inner_threshold_fault_run_count": 0,
        "decision": decision,
        "selected_candidate_id": CANDIDATE_ID if decision == "GO" else None,
        "folds": fold_documents, "gates": gates, "aggregate": aggregate,
        "authoritative_cuda_repeat": repeat, "cross_device_report_only": cpu,
        "development_decision_before_final_refit": decision_record,
        "final_refit": final_refit,
        "method_search_on_current_pool_terminated": decision == "DEVELOPMENT_NO_GO",
    }
    _atomic_json(root / "development_result.json", result)
    return result


__all__ = [
    "ACCESS_PLAN_CANONICAL_JSON_SHA256", "BACKEND_RESULT_SCHEMA", "CANDIDATE_ID",
    "FOLD_MAP_CANONICAL_JSON_SHA256", "PROTOCOL_ID", "PROTOCOL_SCHEMA",
    "NumericBlock", "NumericRun", "cuda_preflight", "development_access_plan",
    "load_final_refit_pipeline", "load_frozen_fold_pipeline",
    "preclaim_bind_method_graph", "run_development", "validate_protocol",
]
