"""Offline golden verification for the audited tsai MiniROCKET subset."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch
import numpy as np

from .minirocket_plus import MiniRocketFeaturesPlus


SEED = 20260912
PACKAGE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = PACKAGE_DIR / "golden" / "upstream_v1_0_1_synthetic.npz"
MANIFEST_PATH = PACKAGE_DIR / "golden" / "manifest.json"

EXPECTED_GOLDEN_SHA256 = "07c3af6f0972d52d0d9aeed635b68425e697b9fbbed90dfd2e3bb852d6a2f549"
EXPECTED_MANIFEST_SHA256 = "25acef90b0ea188c1d8685c89242ad2a9e48bf0540845b6ea6660034df7a20c6"
EXPECTED_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
EXPECTED_VENDOR_SOURCE_SHA256 = "2d68936955c06d2989f53c10e743adb1d4ca868f7644ccc8c16b396928d41622"
EXPECTED_CLASS_AST_SHA256 = "3644e12a5af80f25617824c1b34f04af2a665ee5d548b9e696135e5bbc30358d"

CONFIG: dict[str, Any] = {
    "add_lsaz": False,
    "c_in": 52,
    "kernel_size": 9,
    "max_dilations_per_kernel": 32,
    "max_num_channels": 9,
    "max_num_kernels": 84,
    "num_features": 10000,
    "seq_len": 128,
}
FIT_CHUNKSIZE = 8
CPU_ATOL_RTOL = 1e-7
CUDA_ATOL_RTOL = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _class_ast_sha256(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "MiniRocketFeaturesPlus"
    )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def _max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def _assert_float_close(name: str, actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    error = _max_abs_error(actual, expected)
    if not np.allclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=False):
        raise AssertionError(f"{name}: exceeds atol=rtol={tolerance}; max_abs_error={error}")
    return error


def _assert_exact(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise AssertionError(f"{name}: exact comparison failed")


def _load_and_validate_artifact() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if _sha256(GOLDEN_PATH) != EXPECTED_GOLDEN_SHA256:
        raise AssertionError("golden NPZ SHA-256 mismatch")
    if _sha256(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("golden manifest SHA-256 mismatch")
    if _sha256(PACKAGE_DIR / "LICENSE") != EXPECTED_LICENSE_SHA256:
        raise AssertionError("vendored LICENSE does not match the official Git blob bytes")
    source_path = PACKAGE_DIR / "minirocket_plus.py"
    if _sha256(source_path) != EXPECTED_VENDOR_SOURCE_SHA256:
        raise AssertionError("vendored MiniROCKET source SHA-256 mismatch")
    if _class_ast_sha256(source_path) != EXPECTED_CLASS_AST_SHA256:
        raise AssertionError("vendored MiniRocketFeaturesPlus AST differs from fixed upstream")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["schema"] != "tsai_v1_0_1_minirocket_synthetic_golden":
        raise AssertionError("unexpected golden manifest schema")
    if manifest["artifact"]["sha256"] != EXPECTED_GOLDEN_SHA256:
        raise AssertionError("manifest does not bind the expected golden NPZ")
    if manifest["configuration"] != CONFIG or manifest["seed"] != SEED:
        raise AssertionError("golden configuration or seed mismatch")

    with np.load(GOLDEN_PATH, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    declared = manifest["array_manifest"]
    if set(arrays) != set(declared):
        raise AssertionError("golden array members differ from manifest")
    for name, array in arrays.items():
        record = declared[name]
        if array.dtype.hasobject:
            raise AssertionError(f"{name}: object dtype is forbidden")
        if array.dtype.str != record["dtype"]:
            raise AssertionError(f"{name}: dtype differs from manifest")
        if list(array.shape) != record["shape"]:
            raise AssertionError(f"{name}: shape differs from manifest")
        if _array_sha256(array) != record["sha256_c_order_bytes"]:
            raise AssertionError(f"{name}: array byte hash differs from manifest")
    return manifest, arrays


def _assert_structure(model: MiniRocketFeaturesPlus, arrays: dict[str, np.ndarray]) -> None:
    if model.num_features != 9996 or model.num_kernels != 84 or model.num_dilations != 15:
        raise AssertionError("production MiniROCKET structure mismatch")
    _assert_exact("indices", model.indices.detach().cpu().numpy(), arrays["indices"])
    _assert_exact("dilations", np.asarray(model.dilations, dtype=np.int32), arrays["dilations"])
    _assert_exact("padding", np.asarray(model.padding, dtype=np.int64), arrays["padding"])
    _assert_exact(
        "num_features_per_dilation",
        np.asarray(model.num_features_per_dilation, dtype=np.int32),
        arrays["num_features_per_dilation"],
    )
    _assert_exact(
        "num_channels_per_combination",
        np.asarray(model.num_channels_per_combination, dtype=np.int32),
        arrays["num_channels_per_combination"],
    )
    if len(model.state_dict()) != 32:
        raise AssertionError("unexpected production state_dict key count")


def _fit_and_compare_cpu(arrays: dict[str, np.ndarray]) -> tuple[MiniRocketFeaturesPlus, float, float]:
    _seed_everything()
    model = MiniRocketFeaturesPlus(**CONFIG).cpu().eval()
    model.fit(arrays["synthetic_x"], chunksize=FIT_CHUNKSIZE)
    _assert_structure(model, arrays)

    buffer_error = _assert_float_close(
        "kernels",
        model.kernels.detach().cpu().numpy(),
        arrays["kernels"],
        CPU_ATOL_RTOL,
    )
    _assert_exact(
        "prefit", model.prefit.detach().cpu().numpy().astype(np.bool_), arrays["prefit"]
    )
    for index in range(model.num_dilations):
        mask = (
            getattr(model, f"channel_combinations_{index}")
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )
        _assert_exact(f"channel_combinations_{index:02d}", mask, arrays[f"channel_combinations_{index:02d}"])
        error = _assert_float_close(
            f"biases_{index:02d}",
            getattr(model, f"biases_{index}").detach().cpu().numpy(),
            arrays[f"biases_{index:02d}"],
            CPU_ATOL_RTOL,
        )
        buffer_error = max(buffer_error, error)

    with torch.no_grad():
        output = model(torch.from_numpy(arrays["synthetic_x"])).detach().cpu().numpy()
    output_error = _assert_float_close("ppv_cpu", output, arrays["ppv_cpu"], CPU_ATOL_RTOL)
    return model, buffer_error, output_error


def _load_golden_state(model: MiniRocketFeaturesPlus, arrays: dict[str, np.ndarray]) -> None:
    state = model.state_dict()
    replacements: dict[str, torch.Tensor] = {}
    for name, current in state.items():
        if name == "kernels" or name == "prefit":
            array_name = name
        elif name.startswith("channel_combinations_"):
            index = int(name.rsplit("_", 1)[1])
            array_name = f"channel_combinations_{index:02d}"
        elif name.startswith("biases_"):
            index = int(name.rsplit("_", 1)[1])
            array_name = f"biases_{index:02d}"
        else:
            raise AssertionError(f"unexpected state key {name}")
        replacements[name] = torch.as_tensor(arrays[array_name], dtype=current.dtype)
    model.load_state_dict(replacements, strict=True)


def _assert_scalar_fixtures() -> dict[str, float]:
    model = MiniRocketFeaturesPlus(1, 9, num_features=84).cpu().eval()
    model.biases_0.zero_()
    model.prefit.fill_(True)
    with torch.no_grad():
        output = model(torch.ones(1, 1, 9, dtype=torch.float32))
    padded = float(output[0, 0])
    unpadded = float(output[0, 42])
    if abs(padded - 4.0 / 9.0) > 1e-7:
        raise AssertionError("hand-calculated padded fixture must equal 4/9")
    if unpadded != 0.0 or float(output[:, 42:].abs().max()) != 0.0:
        raise AssertionError("hand-calculated unpadded fixtures must be strict-tie zeros")

    quantiles = model.get_quantiles(5)
    expected_quantiles = torch.tensor(
        [
            0.6180340051651001,
            0.2360679805278778,
            0.8541019558906555,
            0.4721359610557556,
            0.09016994386911392,
        ],
        dtype=torch.float32,
    )
    if not torch.equal(quantiles, expected_quantiles):
        raise AssertionError("golden-ratio quantile fixture mismatch")

    convolution = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float32)
    bias = torch.tensor([[1.0]], dtype=torch.float32)
    strict_ppv = float(model.get_PPVs(convolution, bias)[0, 0])
    if abs(strict_ppv - 1.0 / 3.0) > 1e-7:
        raise AssertionError("strict > tie fixture must equal 1/3, not 2/3")
    return {"padded_ppv": padded, "strict_tie_ppv": strict_ppv, "unpadded_ppv": unpadded}


def _compare_cuda(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    _seed_everything()
    model = MiniRocketFeaturesPlus(**CONFIG).cpu().eval()
    _load_golden_state(model, arrays)
    model = model.to(torch.device("cuda:0"))
    inputs = torch.from_numpy(arrays["synthetic_x"]).to(torch.device("cuda:0"))
    with torch.no_grad():
        first = model(inputs).detach().cpu().numpy()
        torch.cuda.synchronize()
        second = model(inputs).detach().cpu().numpy()
        torch.cuda.synchronize()
    cuda_cpu_error = _assert_float_close(
        "ppv_cuda_vs_cpu_golden", first, arrays["ppv_cpu"], CUDA_ATOL_RTOL
    )
    replay_error = _assert_float_close("ppv_cuda_replay", second, first, CUDA_ATOL_RTOL)
    return cuda_cpu_error, replay_error


def verify_golden_handshake(*, require_cuda: bool = True) -> dict[str, Any]:
    """Verify provenance, exact buffers, CPU golden, scalar cases, and CUDA replay.

    The function is deliberately dataset-independent and reads only files in
    this vendored package. If ``require_cuda`` is true, missing CUDA fails
    closed. The returned dictionary is JSON-serializable.
    """

    manifest, arrays = _load_and_validate_artifact()
    _, cpu_buffer_error, cpu_output_error = _fit_and_compare_cpu(arrays)
    scalar = _assert_scalar_fixtures()

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required for the formal MiniROCKET golden handshake")
    cuda_cpu_error: float | None = None
    cuda_replay_error: float | None = None
    if cuda_available:
        cuda_cpu_error, cuda_replay_error = _compare_cuda(arrays)

    return {
        "annotated_tag_object_git_sha1": manifest["provenance"]["annotated_tag_object_git_sha1"],
        "cpu_buffer_max_abs_error": cpu_buffer_error,
        "cpu_output_max_abs_error": cpu_output_error,
        "cuda_available": cuda_available,
        "cuda_cpu_max_abs_error": cuda_cpu_error,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_replay_max_abs_error": cuda_replay_error,
        "effective_num_features": 9996,
        "golden_sha256": EXPECTED_GOLDEN_SHA256,
        "peeled_commit_git_sha1": manifest["provenance"]["peeled_commit_git_sha1"],
        "scalar_fixtures": scalar,
        "status": "PASS",
    }


__all__ = ["verify_golden_handshake"]

