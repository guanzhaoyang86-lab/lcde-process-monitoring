from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch
import numpy as np

from vendor.tsai_v1_0_1_minirocket import MiniRocketFeaturesPlus
from vendor.tsai_v1_0_1_minirocket.golden_handshake import verify_golden_handshake


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "tsai_v1_0_1_minirocket"
EXPECTED_CLASS_AST_SHA256 = "3644e12a5af80f25617824c1b34f04af2a665ee5d548b9e696135e5bbc30358d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_ast_sha256(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "MiniRocketFeaturesPlus"
    )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def test_annotated_tag_and_peeled_commit_are_not_conflated() -> None:
    provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    tag = provenance["tag_resolution"]
    assert tag["git_object_type"] == "tag"
    assert tag["annotated_tag_object_git_sha1"] == "2bca845b2905aa2cfcb03a2e1615bdb196b821e5"
    assert tag["full_commit_sha"] == "f9ecc6b6df798946fa5fb290ac2ade9a00815289"
    assert tag["annotated_tag_object_git_sha1"] != tag["full_commit_sha"]


def test_provenance_hashes_every_local_vendor_artifact() -> None:
    provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    for relative_path, expected in provenance["local_files"].items():
        assert _sha256(VENDOR / relative_path) == expected
    assert provenance["canonical_upstream_bytes"]["source"]["git_blob_sha1"] == (
        "16a9357de1ae20d4d58f3534d49b66558d0f01ff"
    )
    assert provenance["canonical_upstream_bytes"]["license"]["git_blob_sha1"] == (
        "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
    )


def test_vendor_class_ast_matches_frozen_official_class_digest() -> None:
    assert _class_ast_sha256(VENDOR / "minirocket_plus.py") == EXPECTED_CLASS_AST_SHA256


def test_clean_subprocess_imports_vendor_without_prior_torch_import() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from vendor.tsai_v1_0_1_minirocket import MiniRocketFeaturesPlus; "
            "print(MiniRocketFeaturesPlus.__name__)"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout.strip() == "MiniRocketFeaturesPlus"
    assert completed.stderr.strip() == ""


def test_no_runtime_tsai_import_or_extra_upstream_class() -> None:
    tree = ast.parse((VENDOR / "minirocket_plus.py").read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    classes: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    assert "tsai" not in imported_roots
    assert classes == ["MiniRocketFeaturesPlus"]


def test_production_constructor_has_frozen_upstream_structure() -> None:
    np.random.seed(20260912)
    torch.manual_seed(20260912)
    model = MiniRocketFeaturesPlus(
        52,
        128,
        num_features=10000,
        max_dilations_per_kernel=32,
        kernel_size=9,
        max_num_channels=9,
        max_num_kernels=84,
        add_lsaz=False,
    )
    assert model.num_features == 9996
    assert model.num_kernels == 84
    assert model.num_dilations == 15
    np.testing.assert_array_equal(model.dilations, np.arange(1, 16, dtype=np.int32))
    np.testing.assert_array_equal(
        model.num_features_per_dilation,
        np.asarray([30, 19, 12, 12, 8, 4, 8, 4, 4, 3, 3, 3, 3, 3, 3], dtype=np.int32),
    )
    assert int(model.num_features_per_dilation.sum()) * model.num_kernels == 9996
    assert int(model.num_channels_per_combination.min()) == 1
    assert int(model.num_channels_per_combination.max()) == 9
    assert len(model.state_dict()) == 32


def test_golden_npz_is_no_pickle_and_deterministically_packaged() -> None:
    manifest = json.loads((VENDOR / "golden" / "manifest.json").read_text(encoding="utf-8"))
    golden = VENDOR / manifest["artifact"]["path"]
    with zipfile.ZipFile(golden) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist())
    with np.load(golden, allow_pickle=False) as arrays:
        assert arrays.files
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)
        assert arrays["ppv_cpu"].shape == (8, 9996)
        assert arrays["synthetic_x"].shape == (8, 52, 128)


def test_complete_cpu_and_available_cuda_golden_handshake() -> None:
    report = verify_golden_handshake(require_cuda=torch.cuda.is_available())
    assert report["status"] == "PASS"
    assert report["cpu_buffer_max_abs_error"] <= 1e-7
    assert report["cpu_output_max_abs_error"] <= 1e-7
    assert report["scalar_fixtures"]["unpadded_ppv"] == 0.0
    if torch.cuda.is_available():
        assert report["cuda_available"] is True
        assert report["cuda_cpu_max_abs_error"] <= 1e-6
        assert report["cuda_replay_max_abs_error"] <= 1e-6

