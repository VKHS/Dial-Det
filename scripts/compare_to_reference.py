from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: str):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["result_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local numeric results with article references")
    parser.add_argument("--generated", required=True, help="CSV with result_id,value")
    parser.add_argument("--reference", default="results/reported/paper_results.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generated = read_rows(args.generated)
    reference = read_rows(args.reference)
    output = []
    for result_id in sorted(set(generated) & set(reference)):
        try:
            local = float(generated[result_id]["value"])
            reported = float(reference[result_id]["value"])
        except (KeyError, ValueError):
            continue
        output.append(
            {
                "result_id": result_id,
                "generated_value": local,
                "generated_provenance": "generated_by_local_run",
                "reported_value": reported,
                "reported_provenance": "reported_in_article",
                "difference": local - reported,
            }
        )
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
