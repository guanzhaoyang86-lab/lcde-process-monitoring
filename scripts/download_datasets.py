"""Download only the modalities needed for MAPIC experiments.

Examples
--------
python scripts/download_datasets.py batch
python scripts/download_datasets.py metropt tep skab
python scripts/download_datasets.py hydraulic
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile


BATCH_FILES = {
    "00_Batch_Distillation_Plant_M-202210_Timeseries_Label_Anomaly_Metadata.zip": (
        "https://zenodo.org/records/21535243/files/"
        "00_Batch_Distillation_Plant_M-202210_Timeseries_Label_Anomaly_Metadata.zip?download=1",
        "636dcb8188e5eb80f19522e244704e2c",
    ),
    "01_Batch_Distillation_Plant_M-202210_Timeseries_Sensors.zip": (
        "https://zenodo.org/records/21535243/files/"
        "01_Batch_Distillation_Plant_M-202210_Timeseries_Sensors.zip?download=1",
        "165b3d89c5d483392eeb7e7909edb859",
    ),
    "02_Batch_Distillation_Plant_M-202210_Timeseries_Actuators.zip": (
        "https://zenodo.org/records/21535243/files/"
        "02_Batch_Distillation_Plant_M-202210_Timeseries_Actuators.zip?download=1",
        "58b4648b4c37d5d0dc8a678015e054e7",
    ),
    "03_Batch_Distillation_Plant_M-202210_Timeseries_Sensor_Uncertainties.zip": (
        "https://zenodo.org/records/21535243/files/"
        "03_Batch_Distillation_Plant_M-202210_Timeseries_Sensor_Uncertainties.zip?download=1",
        "7d0a19666294c62a2ba4a5474b03b4e4",
    ),
}

METROPT_URL = "https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
HYDRAULIC_URL = (
    "https://archive.ics.uci.edu/static/public/447/"
    "condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
)
TEP_URL = (
    "https://github.com/camaramm/tennessee-eastman-profBraatz/archive/"
    "0643318858ce7c87292bd59820b459e12a13d009.zip"
)
SKAB_URL = "https://github.com/waico/skab/archive/refs/tags/v0.9.0.zip"
RIETH_FILES = {
    "TEP_FaultFree_Training.RData": (
        "https://dataverse.harvard.edu/api/access/datafile/3031241",
        "ec126484534331f85001d8c4ebce6d17",
    ),
    "TEP_FaultFree_Testing.RData": (
        "https://dataverse.harvard.edu/api/access/datafile/3031240",
        "38ad9810fc871026157086ae2c2f0ee9",
    ),
    "TEP_Faulty_Training.RData": (
        "https://dataverse.harvard.edu/api/access/datafile/3031242",
        "c5f594d54c47e620ff877feb58407fda",
    ),
    "TEP_Faulty_Testing.RData": (
        "https://dataverse.harvard.edu/api/access/datafile/3031243",
        "556bdb64c83021bc0c5f92e427753565",
    ),
}


def digest(path: Path, algorithm: str = "md5") -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(url: str, destination: Path, md5: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (md5 is None or digest(destination) == md5):
        print(f"verified: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "MAPIC-research/0.1"})
    print(f"downloading: {destination.name}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if md5 is not None and digest(temporary) != md5:
            raise RuntimeError(f"checksum mismatch for {destination.name}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}") from error
        archive.extractall(destination)


def fetch_batch(root: Path) -> None:
    folder = root / "batch_distillation_v1.1.2"
    for filename, (url, checksum) in BATCH_FILES.items():
        download(url, folder / filename, checksum)


def fetch_rieth(root: Path) -> None:
    """Download only the healthy Phase-I and faulty test partitions."""
    folder = root / "tep_rieth_2017"
    for filename, (url, checksum) in RIETH_FILES.items():
        download(url, folder / filename, checksum)


def fetch_and_extract(name: str, url: str, root: Path) -> None:
    archive = download(url, root / f"{name}.zip")
    destination = root / name
    safe_extract(archive, destination)
    print(f"extracted: {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=("batch", "metropt", "tep", "rieth", "skab", "hydraulic", "all"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    requested = (
        {"batch", "metropt", "tep", "rieth", "skab", "hydraulic"}
        if "all" in args.datasets
        else set(args.datasets)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if "batch" in requested:
        fetch_batch(args.output)
    if "metropt" in requested:
        fetch_and_extract("metropt", METROPT_URL, args.output)
    if "tep" in requested:
        fetch_and_extract("tep_classic_0643318", TEP_URL, args.output)
    if "rieth" in requested:
        fetch_rieth(args.output)
    if "skab" in requested:
        fetch_and_extract("skab_v0.9.0", SKAB_URL, args.output)
    if "hydraulic" in requested:
        fetch_and_extract("hydraulic_uci_447", HYDRAULIC_URL, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
