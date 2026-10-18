from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

import mate_spc_hydraulic.data as hydraulic_data
from mate_spc_hydraulic.data import (
    CYCLE_SECONDS,
    DIFFERENCE_FEATURE_NAMES,
    EXPECTED_CYCLES,
    LEVEL_FEATURE_NAMES,
    MINIMUM_SCALE,
    OUTPUT_CHANNEL_COUNT,
    OUTPUT_FEATURE_NAMES,
    RAW_CHANNEL_COUNT,
    RAW_FEATURE_NAMES,
    RESIDUAL_CLIP,
    ROBUST_SCALE_FACTOR,
    SCALE_FLOOR_FRACTION,
    SENSOR_SPECS,
    HydraulicPhasewiseTemplate,
    SensorSpec,
    load_hydraulic_cycles,
    load_phasewise_template,
)


def _cycle_values(rate_hz: int, offset: float) -> np.ndarray:
    return offset + np.arange(CYCLE_SECONDS * rate_hz, dtype=np.float64)


def _write_rows(path: Path, rows: list[np.ndarray], *, mixed_whitespace: bool = False) -> None:
    pieces: list[str] = []
    for row_index, row in enumerate(rows):
        separator = "\t" if mixed_whitespace and row_index % 2 == 0 else "   "
        pieces.append(separator.join(format(float(value), ".17g") for value in row))
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def _healthy_raw_cycles() -> np.ndarray:
    cycle_offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None, None]
    phase = np.linspace(-0.5, 0.5, CYCLE_SECONDS)[None, :, None]
    channel_scale = np.linspace(0.5, 2.0, RAW_CHANNEL_COUNT)[None, None, :]
    cycles = phase + cycle_offsets * channel_scale
    cycles = np.broadcast_to(cycles, (5, CYCLE_SECONDS, RAW_CHANNEL_COUNT)).copy()
    # Exercise both scale-floor branches: a fully constant coordinate and a
    # phase-varying but cycle-constant coordinate.
    cycles[:, :, 0] = 7.0
    cycles[:, :, 1] = np.linspace(-5.0, 5.0, CYCLE_SECONDS)[None, :]
    return cycles


class HydraulicSensorSpecificationTests(unittest.TestCase):
    def test_official_sensor_names_rates_and_channel_counts_are_fixed(self) -> None:
        self.assertEqual(EXPECTED_CYCLES, 2205)
        self.assertEqual(CYCLE_SECONDS, 60)
        self.assertEqual(len(SENSOR_SPECS), 17)
        self.assertEqual(
            [(spec.name, spec.rate_hz) for spec in SENSOR_SPECS],
            [
                ("PS1", 100),
                ("PS2", 100),
                ("PS3", 100),
                ("PS4", 100),
                ("PS5", 100),
                ("PS6", 100),
                ("EPS1", 100),
                ("FS1", 10),
                ("FS2", 10),
                ("TS1", 1),
                ("TS2", 1),
                ("TS3", 1),
                ("TS4", 1),
                ("VS1", 1),
                ("CE", 1),
                ("CP", 1),
                ("SE", 1),
            ],
        )
        self.assertEqual(RAW_CHANNEL_COUNT, 44)
        self.assertEqual(len(RAW_FEATURE_NAMES), 44)
        self.assertEqual(len(set(RAW_FEATURE_NAMES)), 44)
        self.assertEqual(OUTPUT_CHANNEL_COUNT, 88)
        self.assertEqual(OUTPUT_FEATURE_NAMES, LEVEL_FEATURE_NAMES + DIFFERENCE_FEATURE_NAMES)
        self.assertEqual(RAW_FEATURE_NAMES[:4], (
            "PS1_mean",
            "PS1_population_std",
            "PS1_minimum",
            "PS1_maximum",
        ))
        self.assertEqual(RAW_FEATURE_NAMES[-1], "SE_observed_value")

    def test_high_rate_streaming_parser_aggregates_and_preserves_index_order(self) -> None:
        spec = SensorSpec("SYN", 2)
        rows = [_cycle_values(2, offset) for offset in (0.0, 1000.0, 2000.0)]
        with TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            _write_rows(path, rows, mixed_whitespace=True)
            output = hydraulic_data._read_sensor_cycles(
                path, spec, (2, 0), expected_cycles=3
            )

        self.assertEqual(output.shape, (2, 60, 4))
        first_second = np.asarray([2000.0, 2001.0])
        np.testing.assert_allclose(
            output[0, 0],
            [np.mean(first_second), np.std(first_second, ddof=0), 2000.0, 2001.0],
            rtol=0.0,
            atol=0.0,
        )
        last_second = np.asarray([118.0, 119.0])
        np.testing.assert_allclose(
            output[1, 59],
            [np.mean(last_second), np.std(last_second, ddof=0), 118.0, 119.0],
            rtol=0.0,
            atol=0.0,
        )

    def test_one_hz_values_are_kept_without_noncausal_aggregation(self) -> None:
        spec = SensorSpec("ONE", 1)
        rows = [_cycle_values(1, offset) for offset in (0.0, 100.0, 200.0)]
        with TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            _write_rows(path, rows, mixed_whitespace=True)
            output = hydraulic_data._read_sensor_cycles(
                path, spec, (1,), expected_cycles=3
            )
        np.testing.assert_array_equal(output[0, :, 0], rows[1])

    def test_small_end_to_end_loader_streams_each_sensor_into_final_slice(self) -> None:
        high = SensorSpec("HIGH", 2)
        one = SensorSpec("ONE", 1)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            high_rows = [_cycle_values(2, offset) for offset in (0.0, 1000.0, 2000.0)]
            one_rows = [_cycle_values(1, offset) for offset in (0.0, 100.0, 200.0)]
            _write_rows(root / high.filename, high_rows)
            _write_rows(root / one.filename, one_rows)
            output, metadata = hydraulic_data._load_cycles_from_specs(
                root,
                (2, 0),
                specs=(high, one),
                expected_cycles=3,
            )

        self.assertEqual(output.shape, (2, 60, 5))
        np.testing.assert_array_equal(output[:, :, 4], np.stack((one_rows[2], one_rows[0])))
        np.testing.assert_allclose(output[0, 0, :4], [2000.5, 0.5, 2000.0, 2001.0])
        self.assertEqual(metadata["parsed_cycle_indices"], [2, 0])
        self.assertTrue(metadata["unrequested_rows_byte_scanned_for_boundaries"])
        self.assertFalse(metadata["unrequested_rows_decoded"])
        self.assertFalse(metadata["unrequested_numeric_values_parsed"])
        self.assertNotIn("all_rows_and_column_counts_validated", metadata)
        self.assertEqual(metadata["output_shape"], [2, 60, 5])

    def test_each_sensor_file_requires_exact_rows_columns_numeric_and_finite(self) -> None:
        spec = SensorSpec("BAD", 1)
        valid = [_cycle_values(1, offset) for offset in (0.0, 100.0, 200.0)]
        cases: list[tuple[str, list[np.ndarray], str]] = [
            ("too_few", valid[:2], "exactly 3 rows"),
            ("too_many", valid + [valid[0]], "exactly 3 rows"),
            ("wrong_width", [valid[0][:-1], valid[1], valid[2]], "exactly 60"),
        ]
        for name, rows, expected_message in cases:
            with self.subTest(name=name), TemporaryDirectory() as folder:
                path = Path(folder) / spec.filename
                _write_rows(path, rows)
                with self.assertRaisesRegex(ValueError, expected_message):
                    hydraulic_data._read_sensor_cycles(
                        path, spec, (0,), expected_cycles=3
                    )

        with TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            rows = [row.copy() for row in valid]
            rows[0][7] = np.nan
            _write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "only finite"):
                hydraulic_data._read_sensor_cycles(path, spec, (0,), expected_cycles=3)

        with TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            _write_rows(path, valid)
            lines = path.read_text(encoding="utf-8").splitlines()
            first_tokens = lines[0].split()
            first_tokens[7] = "not-a-number"
            lines[0] = " ".join(first_tokens)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-numeric"):
                hydraulic_data._read_sensor_cycles(path, spec, (0,), expected_cycles=3)

    def test_unrequested_rows_are_not_decoded_or_content_validated(self) -> None:
        spec = SensorSpec("GUARDED", 1)
        valid = [_cycle_values(1, offset) for offset in (0.0, 100.0, 200.0)]
        with TemporaryDirectory() as folder:
            path = Path(folder) / spec.filename
            _write_rows(path, valid)
            lines = path.read_bytes().splitlines()
            lines[1] = b"\xff\xfeHELD-OUT-INVALID-UTF8"
            lines[2] = b"1\t2\t3"
            path.write_bytes(b"\n".join(lines) + b"\n")

            development = hydraulic_data._read_sensor_cycles(
                path, spec, (0,), expected_cycles=3
            )
            self.assertEqual(development.shape, (1, 60, 1))
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                hydraulic_data._read_sensor_cycles(path, spec, (1,), expected_cycles=3)
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                hydraulic_data._read_sensor_cycles(path, spec, (2,), expected_cycles=3)

    def test_cycle_indices_are_explicit_zero_based_unique_and_bounded(self) -> None:
        validate = hydraulic_data._validate_cycle_indices
        self.assertEqual(validate((np.int64(2), 0), expected_cycles=3), (2, 0))
        for indices, exception, message in (
            ((), ValueError, "at least one"),
            ((0, 0), ValueError, "duplicates"),
            ((-1,), IndexError, "valid range"),
            ((3,), IndexError, "valid range"),
            ((True,), TypeError, "integers"),
            ((0.0,), TypeError, "integers"),
        ):
            with self.subTest(indices=indices), self.assertRaisesRegex(exception, message):
                validate(indices, expected_cycles=3)

    def test_missing_files_and_symlinks_are_rejected(self) -> None:
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "PS1.txt"):
                load_hydraulic_cycles(folder, (0,))

        with TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch.object(Path, "is_symlink", return_value=True):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    hydraulic_data._checked_sensor_path(root, SENSOR_SPECS[0])


class HydraulicPhasewiseTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.healthy = _healthy_raw_cycles()
        self.template = HydraulicPhasewiseTemplate.fit(self.healthy)

    def test_fit_matches_frozen_phasewise_median_mad_and_scale_floor(self) -> None:
        phase_center = np.median(self.healthy, axis=0)
        phase_mad = np.median(
            np.abs(self.healthy - phase_center[None, :, :]), axis=0
        )
        flattened = self.healthy.reshape(-1, RAW_CHANNEL_COUNT)
        global_center = np.median(flattened, axis=0)
        global_scale = ROBUST_SCALE_FACTOR * np.median(
            np.abs(flattened - global_center[None, :]), axis=0
        )
        floor = np.maximum(MINIMUM_SCALE, SCALE_FLOOR_FRACTION * global_scale)

        np.testing.assert_array_equal(self.template.phase_center, phase_center)
        np.testing.assert_allclose(
            self.template.phase_scale,
            np.maximum(ROBUST_SCALE_FACTOR * phase_mad, floor[None, :]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(self.template.global_center, global_center)
        np.testing.assert_allclose(self.template.global_robust_scale, global_scale)
        np.testing.assert_allclose(self.template.scale_floor, floor)
        self.assertTrue(np.all(self.template.phase_scale > 0.0))
        self.assertEqual(self.template.scale_floor[0], MINIMUM_SCALE)
        self.assertGreater(self.template.scale_floor[1], MINIMUM_SCALE)
        for name in (
            "phase_center",
            "phase_scale",
            "global_center",
            "global_robust_scale",
            "scale_floor",
        ):
            self.assertFalse(getattr(self.template, name).flags.writeable)

    def test_transform_is_causal_clipped_and_has_current_residual_difference(self) -> None:
        raw = self.healthy[[2]].copy()
        raw[0, 10, 3] += 1.0e9
        raw[0, 11, 3] -= 1.0e9
        output = self.template.transform(raw)

        self.assertEqual(output.shape, (1, 60, 88))
        level = output[:, :, :RAW_CHANNEL_COUNT]
        difference = output[:, :, RAW_CHANNEL_COUNT:]
        self.assertLessEqual(float(np.max(level)), RESIDUAL_CLIP)
        self.assertGreaterEqual(float(np.min(level)), -RESIDUAL_CLIP)
        np.testing.assert_array_equal(difference[:, 0, :], 0.0)
        np.testing.assert_array_equal(difference[:, 1:, :], level[:, 1:, :] - level[:, :-1, :])
        self.assertEqual(level[0, 10, 3], RESIDUAL_CLIP)
        self.assertEqual(level[0, 11, 3], -RESIDUAL_CLIP)
        self.assertEqual(difference[0, 11, 3], -2.0 * RESIDUAL_CLIP)

        changed = raw.copy()
        changed[:, 40:, :] += 50.0
        changed_output = self.template.transform(changed)
        np.testing.assert_array_equal(output[:, :40, :], changed_output[:, :40, :])

    def test_input_validation_rejects_wrong_shape_nonfinite_and_empty_fit(self) -> None:
        bad_values = self.healthy.copy()
        bad_values[0, 0, 0] = np.inf
        for action, message in (
            (lambda: HydraulicPhasewiseTemplate.fit(self.healthy[:, :, :-1]), "shape"),
            (lambda: HydraulicPhasewiseTemplate.fit(self.healthy[:0]), "at least one"),
            (lambda: HydraulicPhasewiseTemplate.fit(bad_values), "finite"),
            (lambda: self.template.transform(self.healthy[:, :-1]), "shape"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                action()

        empty = self.template.transform(np.empty((0, 60, 44), dtype=np.float64))
        self.assertEqual(empty.shape, (0, 60, 88))

    def test_safe_npz_round_trip_and_exact_schema_validation(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "phase-template.npz"
            returned = self.template.save_npz(path)
            self.assertEqual(returned, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertFalse(any(archive[key].dtype.kind == "O" for key in archive.files))
            restored = load_phasewise_template(path)
            expected = self.template.transform(self.healthy[[0]])
            np.testing.assert_array_equal(restored.transform(self.healthy[[0]]), expected)

            bad = Path(folder) / "bad.npz"
            np.savez(bad, schema=np.asarray("incomplete"))
            with self.assertRaisesRegex(ValueError, "archive keys mismatch"):
                load_phasewise_template(bad)

    def test_npz_rejects_bad_numeric_fields_before_float_conversion(self) -> None:
        corruptions = (
            (
                "text",
                "phase_center",
                lambda value: value.astype(str),
                "phase_center.*numeric",
            ),
            (
                "object",
                "phase_scale",
                lambda value: value.astype(object),
                "phase_scale.*numeric",
            ),
            (
                "boolean",
                "global_center",
                lambda value: value.astype(bool),
                "global_center.*numeric",
            ),
            (
                "nonfinite",
                "global_robust_scale",
                lambda value: np.full(value.shape, np.nan, dtype=np.float64),
                "global_robust_scale.*finite",
            ),
            (
                "shape",
                "scale_floor",
                lambda value: value[:-1],
                "scale_floor.*shape",
            ),
        )
        with TemporaryDirectory() as folder:
            root = Path(folder)
            valid = root / "valid.npz"
            self.template.save_npz(valid)
            with np.load(valid, allow_pickle=False) as archive:
                payload = {name: np.array(archive[name], copy=True) for name in archive.files}

            for label, field, corrupt, message in corruptions:
                with self.subTest(label=label):
                    damaged_payload = dict(payload)
                    damaged_payload[field] = corrupt(payload[field].copy())
                    damaged = root / f"{label}.npz"
                    np.savez(damaged, **damaged_payload)
                    with self.assertRaisesRegex(ValueError, message):
                        load_phasewise_template(damaged)


if __name__ == "__main__":
    unittest.main()
