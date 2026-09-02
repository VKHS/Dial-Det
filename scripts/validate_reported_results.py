from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ledger = ROOT / "results" / "reported" / "paper_results.csv"
    errors: list[str] = []
    with ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 318:
        errors.append(f"expected 318 reported table values, found {len(rows)}")
    required = {
        "result_id", "paper_location", "claim_family", "dataset", "stream", "rule",
        "metric", "value", "provenance_status",
    }
    if rows and (missing := sorted(required - set(rows[0]))):
        errors.append("missing columns: " + ", ".join(missing))
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        result_id = row.get("result_id", "")
        if not result_id or result_id in seen:
            errors.append(f"line {line}: empty or duplicate result_id")
        seen.add(result_id)
        if row.get("provenance_status") != "reported_in_article":
            errors.append(f"line {line}: invalid provenance status")
        try:
            float(row.get("value", ""))
        except ValueError:
            errors.append(f"line {line}: nonnumeric value")
        text = " ".join(row.values())
        if re.search(r"\b(projected|estimated|placeholder|TBD)\b", text, re.I):
            errors.append(f"line {line}: non-obtained marker")
    figure_rows = 0
    for path in sorted((ROOT / "results" / "figure_data").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            entries = list(csv.DictReader(handle))
        figure_rows += len(entries)
        for line, row in enumerate(entries, start=2):
            if row.get("provenance_status") != "reported_in_article":
                errors.append(f"{path.name}:{line}: invalid provenance status")
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    source = json.loads((ROOT / "results" / "reported" / "SOURCE.json").read_text())
    if source.get("sha256") != digest:
        errors.append("reported result SHA-256 does not match SOURCE.json")
    report = {
        "valid": not errors,
        "reported_table_rows": len(rows),
        "reported_figure_rows": figure_rows,
        "sha256": digest,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
