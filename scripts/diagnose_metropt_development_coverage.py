#!/usr/bin/env python
"""Report causal score availability for an already-authorized dev partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# This Windows environment must load torch before NumPy/Pandas dependencies.
import torch  # noqa: F401
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSPACE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mate_spc_metropt.data import (  # noqa: E402
    audit_metropt_timestamps,
    load_governed_metropt_partition,
)
from mate_spc_metropt.governance import (  # noqa: E402
    PHASE_DEVELOPMENT,
    PHASE_PARTITIONS,
    authenticate_phase_access_intent,
    authenticate_workspace_access_ledger,
    read_strict_json,
)
from mate_spc_metropt.model import (  # noqa: E402
    align_inference_forecasts_to_score_times,
    build_causal_input_windows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--partition", choices=PHASE_PARTITIONS[PHASE_DEVELOPMENT], required=True
    )
    parser.add_argument("--csv-chunksize", type=int, default=100_000)
    arguments = parser.parse_args()

    output = Path(arguments.output_dir).resolve(strict=True)
    ledger_path = output / "development_access_ledger.json"
    ledger_body = read_strict_json(ledger_path)
    expected_ledger = dict(ledger_body)
    phase_fingerprint = str(expected_ledger["phase_manifest_fingerprint"])
    expected_ledger.pop("ledger_fingerprint")
    claim = authenticate_workspace_access_ledger(
        ledger_path, expected_body=expected_ledger
    )
    intent_path = output / "development_access_intent.json"
    intent = authenticate_phase_access_intent(
        intent_path, claim, phase_manifest_fingerprint=phase_fingerprint
    )

    csv_path = Path(arguments.csv_path).resolve(strict=True)
    audit = audit_metropt_timestamps(csv_path, chunksize=arguments.csv_chunksize)
    interval = load_governed_metropt_partition(
        csv_path,
        arguments.partition,
        claim=claim,
        access_intent=intent,
        access_intent_path=intent_path,
        phase_manifest_fingerprint=phase_fingerprint,
        timestamp_audit=audit,
        chunksize=arguments.csv_chunksize,
    )
    windows = build_causal_input_windows(
        interval.analog,
        interval.controls,
        interval.timestamps,
        eligible=interval.eligible,
        reset_before=interval.reset_before,
    )
    dummy = np.zeros((windows.inputs.shape[0], 3, 7), dtype=np.float64)
    aligned = align_inference_forecasts_to_score_times(
        windows,
        dummy,
        interval.analog,
        interval.controls,
        interval.timestamps,
        eligible=interval.eligible,
        reset_before=interval.reset_before,
    )

    minute_dates = interval.timestamps.astype("datetime64[D]")
    anchor_dates = windows.anchor_timestamps.astype("datetime64[D]")
    score_dates = aligned.score_timestamps.astype("datetime64[D]")
    first_day = minute_dates[0]
    last_day = minute_dates[-1]
    dates = np.arange(first_day, last_day + np.timedelta64(1, "D"), dtype="datetime64[D]")
    days: list[dict[str, object]] = []
    for day in dates:
        selected = minute_dates == day
        days.append(
            {
                "date": str(day),
                "raw_rows": int(interval.source_row_count_per_bin[selected].sum()),
                "observed_minutes": int(interval.bin_observed_mask[selected].sum()),
                "coverage_ge_half_minutes": int(
                    np.count_nonzero(interval.coverage_fraction[selected] >= 0.5)
                ),
                "eligible_minutes": int(np.count_nonzero(interval.eligible[selected])),
                "reset_count": int(np.count_nonzero(interval.reset_before[selected])),
                "inference_window_count": int(np.count_nonzero(anchor_dates == day)),
                "causal_score_count": int(np.count_nonzero(score_dates == day)),
                "qualified_at_1141": bool(np.count_nonzero(score_dates == day) >= 1141),
            }
        )
    result = {
        "schema": "metropt_cared_bc_development_coverage_diagnostic",
        "partition": arguments.partition,
        "sensor_numeric_scope": "development_only",
        "sealed_partition_access": False,
        "minute_count": int(interval.timestamps.size),
        "inference_window_count": int(windows.inputs.shape[0]),
        "causal_score_count": int(aligned.residuals.shape[0]),
        "days": days,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
