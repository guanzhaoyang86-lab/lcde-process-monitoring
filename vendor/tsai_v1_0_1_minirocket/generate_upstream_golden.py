"""Generate the fixed MiniROCKET golden from an official unmodified module.

This is a provenance tool, not runtime code. It refuses any upstream source
whose bytes do not match the canonical Git blob at the peeled v1.0.1 commit.
It never reads project datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import zipfile

import torch
import numpy as np


SEED = 20260912
TAG_OBJECT_GIT_SHA1 = "2bca845b2905aa2cfcb03a2e1615bdb196b821e5"
PEELED_COMMIT_GIT_SHA1 = "f9ecc6b6df798946fa5fb290ac2ade9a00815289"
UPSTREAM_SOURCE_GIT_BLOB_SHA1 = "16a9357de1ae20d4d58f3534d49b66558d0f01ff"
UPSTREAM_SOURCE_SHA256 = "7e3071cebef34b9eb7c1b458c9fc2503acce71de206b65e675ef9c76672001b1"
UPSTREAM_SOURCE_SIZE = 15644

CONFIG = {
    "add_lsaz": False,
    "c_in": 52,
    "kernel_size": 9,
    "max_dilations_per_kernel": 32,
    "max_num_channels": 9,
    "max_num_kernels": 84,
    "num_features": 10000,
    "seq_len": 128,
}
SYNTHETIC_BATCH_SIZE = 8
FIT_CHUNKSIZE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _synthetic_input() -> np.ndarray:
    """Return a platform-stable float32 tensor without consuming any RNG."""

    count = SYNTHETIC_BATCH_SIZE * CONFIG["c_in"] * CONFIG["seq_len"]
    index = np.arange(count, dtype=np.int64)
    values = ((index % 257) - 128).astype(np.float32) / np.float32(64.0)
    values += (((index // 257) % 9) - 4).astype(np.float32) / np.float32(32.0)
    return np.ascontiguousarray(
        values.reshape(SYNTHETIC_BATCH_SIZE, CONFIG["c_in"], CONFIG["seq_len"])
    )


def _load_official_class(source_path: Path):
    if source_path.stat().st_size != UPSTREAM_SOURCE_SIZE:
        raise RuntimeError("official source size mismatch")
    if _sha256(source_path) != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError("official source SHA-256 mismatch; require canonical LF Git blob bytes")
    spec = importlib.util.spec_from_file_location("_tsai_v1_0_1_official", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create import specification for official source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MiniRocketFeaturesPlus


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


def _collect_arrays(model, synthetic_x: np.ndarray, ppv_cpu: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "synthetic_x": synthetic_x,
        "ppv_cpu": ppv_cpu,
        "indices": model.indices.detach().cpu().numpy(),
        "kernels": model.kernels.detach().cpu().numpy(),
        "dilations": np.asarray(model.dilations, dtype=np.int32),
        "padding": np.asarray(model.padding, dtype=np.int64),
        "num_features_per_dilation": np.asarray(
            model.num_features_per_dilation, dtype=np.int32
        ),
        "num_channels_per_combination": np.asarray(
            model.num_channels_per_combination, dtype=np.int32
        ),
        "prefit": model.prefit.detach().cpu().numpy().astype(np.bool_),
    }
    for index in range(model.num_dilations):
        arrays[f"channel_combinations_{index:02d}"] = (
            getattr(model, f"channel_combinations_{index}")
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
        )
        arrays[f"biases_{index:02d}"] = (
            getattr(model, f"biases_{index}").detach().cpu().numpy()
        )
    return arrays


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a no-pickle NPZ with fixed member order and ZIP timestamps."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def generate(source_path: Path, output_path: Path, manifest_path: Path) -> None:
    official_class = _load_official_class(source_path)
    synthetic_x = _synthetic_input()
    _seed_everything()
    model = official_class(**CONFIG).cpu().eval()
    model.fit(synthetic_x, chunksize=FIT_CHUNKSIZE)
    with torch.no_grad():
        ppv_cpu = model(torch.from_numpy(synthetic_x)).detach().cpu().numpy()

    arrays = _collect_arrays(model, synthetic_x, ppv_cpu)
    _write_deterministic_npz(output_path, arrays)
    manifest = {
        "array_manifest": {
            name: {
                "dtype": np.asarray(value).dtype.str,
                "sha256_c_order_bytes": _array_sha256(np.asarray(value)),
                "shape": list(np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "artifact": {
            "format": "npz",
            "member_order": "lexicographic",
            "numpy_allow_pickle": False,
            "path": "golden/upstream_v1_0_1_synthetic.npz",
            "sha256": _sha256(output_path),
            "size_bytes": output_path.stat().st_size,
            "zip_member_timestamp": "1980-01-01T00:00:00",
        },
        "configuration": CONFIG,
        "expected_structure": {
            "dilations": list(range(1, 16)),
            "effective_num_features": 9996,
            "num_dilations": 15,
            "num_features_per_dilation": [30, 19, 12, 12, 8, 4, 8, 4, 4, 3, 3, 3, 3, 3, 3],
            "num_kernels": 84,
            "state_dict_key_count": 32,
        },
        "fit_chunksize": FIT_CHUNKSIZE,
        "generator": {
            "numpy_version": np.__version__,
            "platform": os.name,
            "pytorch_version": torch.__version__,
            "script": "generate_upstream_golden.py",
        },
        "provenance": {
            "annotated_tag_object_git_sha1": TAG_OBJECT_GIT_SHA1,
            "peeled_commit_git_sha1": PEELED_COMMIT_GIT_SHA1,
            "repository": "https://github.com/timeseriesAI/tsai",
            "source_git_blob_sha1": UPSTREAM_SOURCE_GIT_BLOB_SHA1,
            "source_path": "tsai/models/MINIROCKETPlus_Pytorch.py",
            "source_sha256": UPSTREAM_SOURCE_SHA256,
            "source_size_bytes": UPSTREAM_SOURCE_SIZE,
            "tag": "v1.0.1",
        },
        "schema": "tsai_v1_0_1_minirocket_synthetic_golden",
        "schema_version": 1,
        "seed": SEED,
        "source_of_truth": "unmodified_official_tagged_upstream_class",
        "synthetic_batch_size": SYNTHETIC_BATCH_SIZE,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    generate(args.official_source, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

