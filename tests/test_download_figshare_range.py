from __future__ import annotations

import hashlib
import importlib.util
import re
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_figshare_range.py"
SPEC = importlib.util.spec_from_file_location("download_figshare_range", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class _RangeServer(ThreadingHTTPServer):
    data: bytes
    mode: str
    requested_ranges: list[tuple[int, int]]
    request_lock: threading.Lock


class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        server = self.server
        assert isinstance(server, _RangeServer)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match is None:
            self.send_error(400, "Range required")
            return
        start, end = (int(value) for value in match.groups())
        with server.request_lock:
            server.requested_ranges.append((start, end))

        if server.mode == "ignore_range":
            body = server.data
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = server.data[start : end + 1]
        content_start, content_end, total = start, end, len(server.data)
        disconnect_once = server.mode == "disconnect_once"
        if disconnect_once:
            server.mode = "correct"
        if server.mode == "wrong_range":
            content_start = start + 1
        elif server.mode == "wrong_total":
            total += 1
        elif server.mode == "wrong_length":
            body = body[:-1]

        self.send_response(206)
        self.send_header("Content-Range", f"bytes {content_start}-{content_end}/{total}")
        declared_length = end - start + 1 if disconnect_once else len(body)
        self.send_header("Content-Length", str(declared_length))
        self.send_header("Content-Encoding", "identity")
        self.end_headers()
        if disconnect_once:
            self.wfile.write(body[: max(1, len(body) // 3)])
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        else:
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class SyntheticRangeFixture:
    def __init__(self, data: bytes, mode: str = "correct") -> None:
        self.server = _RangeServer(("127.0.0.1", 0), _RangeHandler)
        self.server.data = data
        self.server.mode = mode
        self.server.requested_ranges = []
        self.server.request_lock = threading.Lock()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/fixture.bin"

    def __enter__(self) -> "SyntheticRangeFixture":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()  # noqa: S324 - test fixture identity.


class RangeDownloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = bytes((index * 37 + 11) % 256 for index in range(12_345))

    def test_concurrent_download_has_exact_non_overlapping_requests(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            result = downloader.download_file(
                url=fixture.url,
                output=output,
                size=len(self.data),
                md5=_md5(self.data),
                workers=4,
                part_size=1024,
            )

            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), self.data)
            requested = sorted(fixture.server.requested_ranges)
            self.assertEqual(requested[0][0], 0)
            self.assertEqual(requested[-1][1], len(self.data) - 1)
            self.assertTrue(all(left[1] + 1 == right[0] for left, right in zip(requested, requested[1:])))
            self.assertFalse(output.with_name(output.name + ".parts").exists())

    def test_partial_tmp_resumes_at_the_next_unwritten_byte(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            parts_dir = output.with_name(output.name + ".parts")
            parts_dir.mkdir()
            partial_size = 137
            (parts_dir / "part-000001.tmp").write_bytes(
                self.data[1024 : 1024 + partial_size]
            )

            downloader.download_file(
                url=fixture.url,
                output=output,
                size=len(self.data),
                md5=_md5(self.data),
                workers=3,
                part_size=1024,
            )

            self.assertEqual(output.read_bytes(), self.data)
            self.assertIn((1024 + partial_size, 2047), fixture.server.requested_ranges)

    def test_completed_output_is_authenticated_without_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            output.write_bytes(self.data)

            downloader.download_file(
                url=fixture.url,
                output=output,
                size=len(self.data),
                md5=_md5(self.data),
                workers=2,
                part_size=1024,
            )

            self.assertEqual(fixture.server.requested_ranges, [])

    def test_existing_wrong_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            original = b"x" * len(self.data)
            output.write_bytes(original)

            with self.assertRaises(downloader.IntegrityError):
                downloader.download_file(
                    url=fixture.url,
                    output=output,
                    size=len(self.data),
                    md5=_md5(self.data),
                    workers=2,
                    part_size=1024,
                )

            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(fixture.server.requested_ranges, [])

    def test_server_ignoring_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data, mode="ignore_range"
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            with self.assertRaisesRegex(downloader.RangeProtocolError, "ignored Range"):
                downloader.download_file(
                    url=fixture.url,
                    output=output,
                    size=len(self.data),
                    md5=_md5(self.data),
                    workers=1,
                    part_size=len(self.data),
                )
            self.assertFalse(output.exists())

    def test_wrong_content_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data, mode="wrong_range"
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            with self.assertRaisesRegex(downloader.RangeProtocolError, "does not match"):
                downloader.download_file(
                    url=fixture.url,
                    output=output,
                    size=len(self.data),
                    md5=_md5(self.data),
                    workers=1,
                    part_size=len(self.data),
                )
            self.assertFalse(output.exists())

    def test_wrong_response_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data, mode="wrong_length"
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            with self.assertRaisesRegex(downloader.RangeProtocolError, "Content-Length mismatch"):
                downloader.download_file(
                    url=fixture.url,
                    output=output,
                    size=len(self.data),
                    md5=_md5(self.data),
                    workers=1,
                    part_size=len(self.data),
                )
            self.assertFalse(output.exists())

    def test_interrupted_body_retries_from_persisted_tmp_length(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data, mode="disconnect_once"
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            downloader.download_file(
                url=fixture.url,
                output=output,
                size=len(self.data),
                md5=_md5(self.data),
                workers=1,
                part_size=len(self.data),
            )

            self.assertEqual(output.read_bytes(), self.data)
            first, second = fixture.server.requested_ranges[:2]
            self.assertEqual(first, (0, len(self.data) - 1))
            self.assertGreater(second[0], 0)
            self.assertEqual(second[1], len(self.data) - 1)

    def test_overlapping_plan_is_rejected(self) -> None:
        ranges = (
            downloader.ByteRange(0, 0, 99),
            downloader.ByteRange(1, 99, 199),
        )
        with self.assertRaisesRegex(downloader.ConfigurationError, "overlap"):
            downloader.validate_ranges(ranges, 200)

    def test_parts_manifest_prevents_cross_source_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as folder_name, SyntheticRangeFixture(
            self.data
        ) as fixture:
            output = Path(folder_name) / "dataset.bin"
            parts_dir = output.with_name(output.name + ".parts")
            parts_dir.mkdir()
            downloader._ensure_manifest(
                parts_dir,
                downloader._manifest_payload(
                    url="https://example.invalid/other.bin",
                    size=len(self.data),
                    md5=_md5(self.data),
                    part_size=1024,
                    part_count=13,
                ),
            )

            with self.assertRaisesRegex(downloader.ConfigurationError, "different"):
                downloader.download_file(
                    url=fixture.url,
                    output=output,
                    size=len(self.data),
                    md5=_md5(self.data),
                    workers=1,
                    part_size=1024,
                )
            self.assertEqual(fixture.server.requested_ranges, [])


if __name__ == "__main__":
    unittest.main()
