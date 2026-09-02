from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_EXTENSIONS = {".tex", ".pdf", ".aux", ".bbl", ".blg", ".bib", ".synctex"}
FORBIDDEN_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
FORBIDDEN_INTERNAL_STRINGS = (
    "/mnt/data/",
    "inspect_sync",
    "paper-sync-kit",
    "final destination",
    "avoid gpu idle kills",
    "background gemm",
)
TEXT_EXTENSIONS = {
    ".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt", ".cff", ".csv"
}


def run(command: list[str]) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1500:],
        "stderr_tail": result.stderr[-1500:],
    }


def main() -> int:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"publication source/build file is present: {relative}")
        if path.suffix.lower() in TEXT_EXTENSIONS and path.name != "release_check.py":
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for phrase in FORBIDDEN_INTERNAL_STRINGS:
                if phrase in text:
                    errors.append(f"internal or damaging phrase {phrase!r}: {relative}")

    ledger = ROOT / "results" / "reported" / "paper_results.csv"
    if not ledger.is_file():
        errors.append("reported result ledger is missing")
        rows = []
    else:
        with ledger.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 318:
            errors.append(f"reported ledger has {len(rows)} rows rather than 318")
        seen: set[str] = set()
        for line, row in enumerate(rows, start=2):
            result_id = row.get("result_id", "")
            if not result_id or result_id in seen:
                errors.append(f"reported ledger line {line}: empty or duplicate result_id")
            seen.add(result_id)
            if row.get("provenance_status") != "reported_in_article":
                errors.append(f"reported ledger line {line}: invalid provenance")
            try:
                float(row.get("value", ""))
            except ValueError:
                errors.append(f"reported ledger line {line}: nonnumeric value")
            text = " ".join(row.values())
            if re.search(r"\b(projected|estimated|placeholder|TBD|final destination)\b", text, re.I):
                errors.append(f"reported ledger line {line}: non-obtained marker")
        source_path = ROOT / "results" / "reported" / "SOURCE.json"
        if not source_path.is_file():
            errors.append("reported SOURCE.json is missing")
        else:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
            if source.get("sha256") != digest:
                errors.append("reported ledger hash differs from SOURCE.json")
            if source.get("row_count") != len(rows):
                errors.append("reported ledger row count differs from SOURCE.json")
            if source.get("raw_predictions_included") is not False:
                errors.append("SOURCE.json must not imply that raw predictions are bundled")

    for path in (ROOT / "results" / "figure_data").glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), start=2):
                if row.get("provenance_status") != "reported_in_article":
                    errors.append(f"{path.relative_to(ROOT)}:{line}: invalid provenance")

    commands = [
        [sys.executable, "-m", "compileall", "-q", str(ROOT / "src")],
        [sys.executable, "-m", "dial_det", "--help"],
        [sys.executable, "-m", "dial_det", "validate-split", "--input", "configs/smoke/split_manifest.json"],
        [sys.executable, "-m", "dial_det", "reported-check", "--input", "results/reported/paper_results.csv"],
    ]
    for name in (
        "dial-out-at-most", "dial-out-exact", "dial-prop-exact", "dial-prop-cost"
    ):
        commands.append([sys.executable, "-m", "dial_det", "smoke", "--contract", name])
    command_reports = []
    for command in commands:
        report = run(command)
        command_reports.append(report)
        if report["returncode"] != 0:
            errors.append("command failed: " + " ".join(command))

    report = {
        "valid": not errors,
        "repository_version": "2.0.0",
        "reported_result_rows": len(rows),
        "article_or_supplement_source_files": 0,
        "commands": command_reports,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
