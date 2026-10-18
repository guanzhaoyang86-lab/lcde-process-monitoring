"""Aggregate validated classic TEP per-fault stress reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from mate_spc.tep_aggregate import (
    aggregate_tep_fault_files,
    write_tep_aggregate,
    write_tep_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--first-fault", type=int, default=1)
    parser.add_argument("--last-fault", type=int, default=21)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    args = parser.parse_args()
    expected = range(args.first_fault, args.last_fault + 1)
    paths = sorted(args.input_dir.glob("fault_*.json"))
    report = aggregate_tep_fault_files(paths, expected_faults=expected)
    write_tep_aggregate(report, args.output_json, args.output_csv)
    if args.output_markdown is not None:
        write_tep_markdown(report, args.output_markdown)
    print(
        f"aggregated {report['fault_count']} faults and {report['total_cases']} cases "
        f"to {args.output_json} and {args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
