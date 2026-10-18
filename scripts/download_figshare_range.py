"""Concurrent, resumable HTTP Range downloader with fail-closed validation.

The downloader is intentionally strict: every response must be ``206 Partial
Content`` and its ``Content-Range`` must exactly match the requested interval.
Parts are first written to ``<output>.parts/part-NNNNNN.tmp`` and atomically
renamed after their exact length is verified.  The assembled file is likewise
written to ``<output>.tmp`` and only replaces the destination after both its
declared size and MD5 digest have been authenticated.

Example
-------
python scripts/download_figshare_range.py \
  --url https://ndownloader.figshare.com/files/26003087 \
  --output data/tep_extended_mode1.zip \
  --size 23924710848 \
  --md5 <authoritative-32-character-md5>

Progress is emitted as compact JSON Lines so a long download can be logged and
audited without a progress-bar dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import requests
import urllib3


DEFAULT_WORKERS = 16
DEFAULT_PART_SIZE = 256 * 1024 * 1024
READ_CHUNK_SIZE = 8 * 1024 * 1024
MAX_ATTEMPTS = 5
USER_AGENT = "curl/8.21.0"
MANIFEST_SCHEMA = "mate_spc_http_range_parts_v1"

_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_PRINT_LOCK = threading.Lock()


class DownloadError(RuntimeError):
    """Base class for a safe download failure."""


class ConfigurationError(DownloadError):
    """Raised when the requested download geometry is unsafe or invalid."""


class RangeProtocolError(DownloadError):
    """Raised when a server does not honor the exact requested byte range."""


class TransientTransferError(DownloadError):
    """Raised when a valid Range response ends before all promised bytes arrive."""


class IntegrityError(DownloadError):
    """Raised when a local or assembled file fails size or MD5 authentication."""


@dataclass(frozen=True)
class ByteRange:
    """One inclusive, zero-based byte interval."""

    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def emit(event: str, **fields: object) -> None:
    """Emit one compact, thread-safe JSONL progress record."""

    record = {"event": event, **fields}
    line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with _PRINT_LOCK:
        print(line, flush=True)


def normalize_md5(value: str) -> str:
    digest = value.strip().lower()
    if not _MD5_RE.fullmatch(digest):
        raise ConfigurationError("md5 must contain exactly 32 hexadecimal characters")
    return digest


def plan_ranges(size: int, part_size: int) -> tuple[ByteRange, ...]:
    """Create and validate a complete, non-overlapping partition of a file."""

    if size <= 0:
        raise ConfigurationError("size must be a positive integer")
    if part_size <= 0:
        raise ConfigurationError("part-size must be a positive integer")
    part_count = (size + part_size - 1) // part_size
    if part_count > 1_000_000:
        raise ConfigurationError("download would require more than 1,000,000 parts")
    ranges = tuple(
        ByteRange(index, start, min(start + part_size, size) - 1)
        for index, start in enumerate(range(0, size, part_size))
    )
    validate_ranges(ranges, size)
    return ranges


def validate_ranges(ranges: Sequence[ByteRange], size: int) -> None:
    """Fail closed unless ranges exactly cover ``[0, size)`` without overlap."""

    if size <= 0 or not ranges:
        raise ConfigurationError("ranges must cover a positive-size file")
    expected_start = 0
    seen_indices: set[int] = set()
    for part in ranges:
        if part.index in seen_indices:
            raise ConfigurationError(f"duplicate part index: {part.index}")
        seen_indices.add(part.index)
        if part.start != expected_start:
            relation = "overlap" if part.start < expected_start else "gap"
            raise ConfigurationError(
                f"unsafe range {relation}: expected byte {expected_start}, got {part.start}"
            )
        if part.end < part.start or part.end >= size:
            raise ConfigurationError(
                f"invalid range for part {part.index}: {part.start}-{part.end}"
            )
        expected_start = part.end + 1
    if expected_start != size:
        raise ConfigurationError(
            f"ranges stop at byte {expected_start - 1}, but declared size is {size}"
        )


def file_md5(path: Path) -> str:
    checksum = hashlib.md5()  # noqa: S324 - MD5 is required for dataset identity, not security.
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(READ_CHUNK_SIZE), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _is_authenticated(path: Path, size: int, expected_md5: str) -> bool:
    return path.is_file() and path.stat().st_size == size and file_md5(path) == expected_md5


def _part_paths(parts_dir: Path, part: ByteRange) -> tuple[Path, Path]:
    completed = parts_dir / f"part-{part.index:06d}"
    temporary = completed.with_name(completed.name + ".tmp")
    return completed, temporary


def _manifest_payload(
    *, url: str, size: int, md5: str, part_size: int, part_count: int
) -> dict[str, object]:
    return {
        "schema": MANIFEST_SCHEMA,
        "url": url,
        "size": size,
        "md5": md5,
        "part_size": part_size,
        "part_count": part_count,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ensure_manifest(parts_dir: Path, expected: dict[str, object]) -> None:
    manifest = parts_dir / "manifest.json"
    if manifest.exists():
        if not manifest.is_file():
            raise ConfigurationError(f"parts manifest is not a file: {manifest}")
        try:
            observed = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"cannot authenticate parts manifest: {manifest}") from error
        if observed != expected:
            raise ConfigurationError(
                "existing parts belong to a different URL/size/MD5/part-size; "
                f"refusing to reuse {parts_dir}"
            )
        return
    _write_json_atomic(manifest, expected)


@contextmanager
def _exclusive_download_lock(parts_dir: Path) -> Iterator[None]:
    lock_path = parts_dir / "download.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise DownloadError(
            f"another download may be using {parts_dir}; remove stale {lock_path} only after "
            "confirming no downloader is running"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validate_response(response: requests.Response, part: ByteRange, request_start: int) -> int:
    if response.status_code in {408, 429} or response.status_code >= 500:
        response.raise_for_status()
    if response.status_code != 206:
        if response.status_code == 200:
            detail = "server ignored Range and returned 200"
        else:
            detail = f"expected status 206, got {response.status_code}"
        raise RangeProtocolError(
            f"{detail} for requested bytes {request_start}-{part.end}"
        )

    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE_RE.fullmatch(content_range.strip())
    if match is None:
        raise RangeProtocolError(f"invalid Content-Range: {content_range!r}")
    observed_start, observed_end, observed_total = (int(item) for item in match.groups())
    expected = (request_start, part.end)
    if (observed_start, observed_end) != expected:
        raise RangeProtocolError(
            "Content-Range does not match request: "
            f"expected {expected[0]}-{expected[1]}, got {observed_start}-{observed_end}"
        )
    expected_total = getattr(response, "_declared_total_size", None)
    if expected_total is None:
        raise RangeProtocolError("internal error: declared total size was not bound to response")
    if observed_total != expected_total:
        raise RangeProtocolError(
            f"Content-Range total mismatch: expected {expected_total}, got {observed_total}"
        )

    content_encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        raise RangeProtocolError(
            f"encoded Range response is unsafe to append: Content-Encoding={content_encoding!r}"
        )

    expected_length = part.end - request_start + 1
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            observed_length = int(content_length)
        except ValueError as error:
            raise RangeProtocolError(f"invalid Content-Length: {content_length!r}") from error
        if observed_length != expected_length:
            raise RangeProtocolError(
                f"Content-Length mismatch: expected {expected_length}, got {observed_length}"
            )
    return expected_length


def _request_remaining_range(
    session: requests.Session,
    *,
    url: str,
    total_size: int,
    part: ByteRange,
    temporary: Path,
) -> None:
    already_present = temporary.stat().st_size if temporary.exists() else 0
    if already_present < 0 or already_present >= part.size:
        raise RangeProtocolError(
            f"unsafe temporary part length for part {part.index}: {already_present}"
        )
    request_start = part.start + already_present
    headers = {
        "Accept-Encoding": "identity",
        "Cache-Control": "no-transform",
        "Range": f"bytes={request_start}-{part.end}",
        "User-Agent": USER_AGENT,
    }
    with session.get(
        url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(30, 120),
    ) as response:
        # This private, in-memory binding keeps the validator signature compact;
        # it does not mutate data returned by the server.
        response._declared_total_size = total_size  # type: ignore[attr-defined]
        expected_length = _validate_response(response, part, request_start)
        received = 0
        with temporary.open("ab") as stream:
            for chunk in response.raw.stream(READ_CHUNK_SIZE, decode_content=False):
                if not chunk:
                    continue
                if received + len(chunk) > expected_length:
                    raise RangeProtocolError(
                        f"response exceeded requested length for part {part.index}"
                    )
                stream.write(chunk)
                received += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if received != expected_length:
            raise TransientTransferError(
                f"short response for part {part.index}: expected {expected_length}, got {received}"
            )


def _download_part(
    *,
    url: str,
    total_size: int,
    parts_dir: Path,
    part: ByteRange,
) -> Path:
    completed, temporary = _part_paths(parts_dir, part)
    if completed.exists():
        if not completed.is_file() or completed.stat().st_size != part.size:
            raise IntegrityError(
                f"completed part {part.index} has an invalid length; refusing to overwrite it"
            )
        emit("part_verified", index=part.index, bytes=part.size)
        return completed
    if temporary.exists() and not temporary.is_file():
        raise IntegrityError(f"temporary part is not a file: {temporary}")
    if temporary.exists() and temporary.stat().st_size > part.size:
        raise IntegrityError(
            f"temporary part {part.index} exceeds its planned length; refusing unsafe resume"
        )
    if temporary.exists() and temporary.stat().st_size == part.size:
        os.replace(temporary, completed)
        emit("part_completed", index=part.index, bytes=part.size, resumed=True)
        return completed

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resumed_bytes = temporary.stat().st_size if temporary.exists() else 0
            if resumed_bytes == part.size:
                break
            if resumed_bytes > part.size:
                raise IntegrityError(
                    f"temporary part {part.index} exceeds its planned length during retry"
                )
            try:
                _request_remaining_range(
                    session,
                    url=url,
                    total_size=total_size,
                    part=part,
                    temporary=temporary,
                )
                break
            except RangeProtocolError:
                raise
            except (
                TransientTransferError,
                requests.RequestException,
                urllib3.exceptions.HTTPError,
                OSError,
            ) as error:
                if attempt == MAX_ATTEMPTS:
                    raise DownloadError(
                        f"part {part.index} failed after {MAX_ATTEMPTS} attempts"
                    ) from error
                delay = min(2 ** (attempt - 1), 8)
                emit(
                    "part_retry",
                    index=part.index,
                    attempt=attempt,
                    resumed_bytes=resumed_bytes,
                    delay_seconds=delay,
                    error=type(error).__name__,
                )
                time.sleep(delay)

    if not temporary.is_file() or temporary.stat().st_size != part.size:
        observed = temporary.stat().st_size if temporary.exists() else -1
        raise IntegrityError(
            f"part {part.index} length mismatch after download: expected {part.size}, got {observed}"
        )
    os.replace(temporary, completed)
    emit("part_completed", index=part.index, bytes=part.size, resumed=resumed_bytes > 0)
    return completed


def _merge_parts(
    *,
    output: Path,
    temporary_output: Path,
    parts_dir: Path,
    ranges: Sequence[ByteRange],
    size: int,
    expected_md5: str,
) -> None:
    emit("merge_started", parts=len(ranges), bytes=size)
    checksum = hashlib.md5()  # noqa: S324 - required dataset identity checksum.
    written = 0
    with temporary_output.open("wb") as destination:
        for part in ranges:
            completed, _ = _part_paths(parts_dir, part)
            if not completed.is_file() or completed.stat().st_size != part.size:
                raise IntegrityError(f"part {part.index} changed or disappeared before merge")
            part_written = 0
            with completed.open("rb") as source:
                for chunk in iter(lambda: source.read(READ_CHUNK_SIZE), b""):
                    destination.write(chunk)
                    checksum.update(chunk)
                    written += len(chunk)
                    part_written += len(chunk)
            if part_written != part.size:
                raise IntegrityError(
                    f"part {part.index} changed length during merge: {part_written} != {part.size}"
                )
            emit("merge_part", index=part.index, bytes_written=written)
        destination.flush()
        os.fsync(destination.fileno())

    if written != size or temporary_output.stat().st_size != size:
        raise IntegrityError(f"assembled size mismatch: expected {size}, got {written}")
    observed_md5 = checksum.hexdigest()
    if observed_md5 != expected_md5:
        raise IntegrityError(
            "assembled MD5 mismatch: "
            f"expected {expected_md5}, got {observed_md5}; parts were retained for diagnosis"
        )
    os.replace(temporary_output, output)


def _cleanup_authenticated_parts(parts_dir: Path, ranges: Sequence[ByteRange]) -> None:
    """Remove only downloader-owned files after the final file is authenticated."""

    known_paths: list[Path] = [
        parts_dir / "manifest.json",
        parts_dir / "manifest.json.tmp",
        parts_dir / "download.lock",
    ]
    for part in ranges:
        completed, temporary = _part_paths(parts_dir, part)
        known_paths.extend((completed, temporary))
    for path in known_paths:
        if path.is_file():
            path.unlink()
    try:
        parts_dir.rmdir()
    except OSError:
        # Unknown files are never deleted; leaving the directory is safer.
        emit("parts_retained", path=str(parts_dir), reason="contains_unknown_files")


def download_file(
    *,
    url: str,
    output: Path,
    size: int,
    md5: str,
    workers: int = DEFAULT_WORKERS,
    part_size: int = DEFAULT_PART_SIZE,
) -> Path:
    """Download one authenticated file using concurrent resumable byte ranges."""

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigurationError("url must be an absolute http:// or https:// URL")
    if workers <= 0 or workers > 128:
        raise ConfigurationError("workers must be between 1 and 128")
    expected_md5 = normalize_md5(md5)
    ranges = plan_ranges(size, part_size)
    output = Path(output)
    if output.exists():
        if not output.is_file():
            raise IntegrityError(f"output exists but is not a file: {output}")
        observed_size = output.stat().st_size
        if observed_size != size:
            raise IntegrityError(
                f"existing output size mismatch: expected {size}, got {observed_size}; "
                "refusing to overwrite it"
            )
        observed_md5 = file_md5(output)
        if observed_md5 != expected_md5:
            raise IntegrityError(
                f"existing output MD5 mismatch: expected {expected_md5}, got {observed_md5}; "
                "refusing to overwrite it"
            )
        emit("already_verified", output=str(output), bytes=size, md5=expected_md5)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.with_name(output.name + ".parts")
    temporary_output = output.with_name(output.name + ".tmp")
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(
        url=url,
        size=size,
        md5=expected_md5,
        part_size=part_size,
        part_count=len(ranges),
    )

    completed_successfully = False
    with _exclusive_download_lock(parts_dir):
        _ensure_manifest(parts_dir, manifest)
        emit(
            "download_started",
            output=str(output),
            bytes=size,
            parts=len(ranges),
            workers=min(workers, len(ranges)),
            part_size=part_size,
        )

        if _is_authenticated(temporary_output, size, expected_md5):
            os.replace(temporary_output, output)
            emit("assembled_tmp_recovered", output=str(output), bytes=size, md5=expected_md5)
        else:
            executor = ThreadPoolExecutor(max_workers=min(workers, len(ranges)))
            futures = [
                executor.submit(
                    _download_part,
                    url=url,
                    total_size=size,
                    parts_dir=parts_dir,
                    part=part,
                )
                for part in ranges
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
            _merge_parts(
                output=output,
                temporary_output=temporary_output,
                parts_dir=parts_dir,
                ranges=ranges,
                size=size,
                expected_md5=expected_md5,
            )
        if not _is_authenticated(output, size, expected_md5):
            raise IntegrityError("final file failed post-rename authentication")
        completed_successfully = True

    if completed_successfully:
        _cleanup_authenticated_parts(parts_dir, ranges)
    emit("download_complete", output=str(output), bytes=size, md5=expected_md5)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent resumable HTTP Range download with size and MD5 authentication"
    )
    parser.add_argument("--url", required=True, help="absolute HTTP(S) source URL")
    parser.add_argument("--output", required=True, type=Path, help="final output file")
    parser.add_argument("--size", required=True, type=int, help="authoritative size in bytes")
    parser.add_argument("--md5", required=True, help="authoritative 32-character MD5 digest")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--part-size",
        type=int,
        default=DEFAULT_PART_SIZE,
        help=f"bytes per part (default: {DEFAULT_PART_SIZE})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        download_file(
            url=args.url,
            output=args.output,
            size=args.size,
            md5=args.md5,
            workers=args.workers,
            part_size=args.part_size,
        )
    except DownloadError as error:
        emit("download_failed", error=type(error).__name__, message=str(error))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
