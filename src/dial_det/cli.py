from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .config.rule import load_complete_rule
from .config.splits import load_split_manifest
from .examples import synthetic_complete_rule, synthetic_encoded_stream, write_synthetic_examples
from .io.npz import load_encoded_stream
from .io.run import environment_record, write_run_directory
from .pipeline import run_pipeline
from .version import __version__


def _repository_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "configs" / "smoke").is_dir():
            return candidate
    return Path.cwd()


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_doctor(args) -> int:
    record = environment_record()
    record["version"] = __version__
    _print_json(record) if args.json else [print(f"{key}: {value}") for key, value in sorted(record.items())]
    return 0


def command_validate_rule(args) -> int:
    rule = load_complete_rule(args.config)
    _print_json({"valid": True, "rule_id": rule.rule_id, "sha256": rule.sha256})
    return 0


def command_validate_split(args) -> int:
    manifest = load_split_manifest(args.input)
    _print_json({"valid": True, "sha256": manifest.sha256})
    return 0


def command_hash_stream(args) -> int:
    stream = load_encoded_stream(args.input)
    _print_json({"valid": True, "candidates": len(stream.candidate_ids), "sha256": stream.sha256})
    return 0


def command_run(args) -> int:
    stream = load_encoded_stream(args.input)
    rule = load_complete_rule(args.config)
    result = run_pipeline(stream, rule)
    if args.output:
        path = write_run_directory(
            args.output, result=result, rule=rule, input_stream_path=args.input
        )
        _print_json({"run_directory": str(path), "fallback_triggered": result.fallback_triggered})
    else:
        _print_json(result.to_dict())
    return 0


def command_smoke(args) -> int:
    rule = synthetic_complete_rule(args.contract)
    result = run_pipeline(synthetic_encoded_stream(), rule)
    _print_json(result.to_dict())
    return 2 if result.fallback_triggered else 0


def command_write_examples(args) -> int:
    write_synthetic_examples(args.output)
    print(Path(args.output).resolve())
    return 0


def command_reported_check(args) -> int:
    path = Path(args.input)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "result_id", "paper_location", "dataset", "stream", "rule", "metric", "value",
        "provenance_status",
    }
    errors = []
    if not rows:
        errors.append("result ledger is empty")
    elif missing := sorted(required - set(rows[0])):
        errors.append("missing columns: " + ", ".join(missing))
    seen = set()
    for line, row in enumerate(rows, start=2):
        if not row.get("result_id") or row["result_id"] in seen:
            errors.append(f"line {line}: empty or duplicate result_id")
        seen.add(row.get("result_id"))
        if row.get("provenance_status") != "reported_in_article":
            errors.append(f"line {line}: provenance must be reported_in_article")
        try:
            float(row.get("value", ""))
        except ValueError:
            errors.append(f"line {line}: nonnumeric value")
        combined = " ".join(row.values()).lower()
        if any(term in combined for term in ("projected", "estimated", "placeholder", "tbd")):
            errors.append(f"line {line}: forbidden non-obtained marker")
    _print_json({"valid": not errors, "rows": len(rows), "errors": errors})
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dial-det", description="DIAL-Det standalone implementation and experiment tools"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="show environment and dependency information")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(function=command_doctor)
    rule = sub.add_parser("validate-rule", help="validate a complete-rule JSON")
    rule.add_argument("--config", required=True)
    rule.set_defaults(function=command_validate_rule)
    split = sub.add_parser("validate-split", help="validate actual disjoint sample IDs")
    split.add_argument("--input", required=True)
    split.set_defaults(function=command_validate_split)
    stream_hash = sub.add_parser("hash-stream", help="validate and hash an encoded stream")
    stream_hash.add_argument("--input", required=True)
    stream_hash.set_defaults(function=command_hash_stream)
    run = sub.add_parser("run", help="run DIAL-Out or DIAL-Prop from a frozen stream")
    run.add_argument("--input", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--output")
    run.set_defaults(function=command_run)
    smoke = sub.add_parser("smoke", help="run an end-to-end deterministic example")
    smoke.add_argument(
        "--contract",
        required=True,
        choices=["dial-out-at-most", "dial-out-exact", "dial-prop-exact", "dial-prop-cost"],
    )
    smoke.set_defaults(function=command_smoke)
    examples = sub.add_parser("write-examples", help="write the deterministic NPZ example")
    examples.add_argument("--output", default="examples/data")
    examples.set_defaults(function=command_write_examples)
    reported = sub.add_parser("reported-check", help="validate article-reported result values")
    reported.add_argument("--input", default="results/reported/paper_results.csv")
    reported.set_defaults(function=command_reported_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
