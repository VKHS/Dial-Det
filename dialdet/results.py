"""Strict comparison of explicitly versioned reported and generated results.

The schema deliberately requires protocol and unit identity. A matching number
alone is not evidence that the same experiment was run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .artifacts import relative_artifact, verify_run
from .data import sha256_file

CONTEXT_FIELDS = ("dataset", "split", "detector", "metric", "unit", "budget", "track", "protocol")
HASH = re.compile(r"[0-9a-f]{64}")


def _load(path, kind):
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document.get("schema") != "dialdet-results-v1" or document.get("kind") != kind:
        raise ValueError(f"Expected dialdet-results-v1 kind={kind}")
    source = document.get("source", {})
    if kind == "reported":
        if not all(isinstance(source.get(k), str) and source[k].strip()
                   for k in ("title", "version", "document_sha256")):
            raise ValueError("Reported results need manuscript title, version and document SHA256")
        if not HASH.fullmatch(source["document_sha256"]):
            raise ValueError("Invalid source document SHA256")
    rows = document.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("A result ledger must contain at least one result")
    found = {}
    verified_runs = {}
    for row in rows:
        result_id = row.get("result_id")
        if not isinstance(result_id, str) or not result_id.strip() or result_id in found:
            raise ValueError("Result identifiers must be nonempty and unique")
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Result {result_id} needs a finite numeric value")
        context = row.get("context", {})
        if any(field not in context for field in CONTEXT_FIELDS):
            raise ValueError(f"Result {result_id} is missing its experiment context")
        if any(not isinstance(context[field], str) or not context[field].strip()
               for field in CONTEXT_FIELDS if field != "budget"):
            raise ValueError("Experiment context strings must be nonempty")
        if isinstance(context["budget"], bool) or not isinstance(context["budget"], int) or context["budget"] < 1:
            raise ValueError("Result budget must be a positive integer")
        if kind == "reported":
            if row.get("provenance_status") != "reported_in_article" or not row.get("paper_location"):
                raise ValueError("Reported values need paper_location and reported_in_article status")
        else:
            if row.get("provenance_status") != "generated_by_local_run":
                raise ValueError("Generated values need explicit generated_by_local_run provenance")
            evidence = row.get("evidence", {})
            if not all(evidence.get(key) for key in ("run_manifest", "artifact", "artifact_sha256", "json_pointer")):
                raise ValueError("Generated results need a run manifest, artifact hash and JSON pointer")
            manifest = (path.parent / evidence["run_manifest"]).resolve()
            if manifest not in verified_runs:
                checked = verify_run(manifest)
                if not checked["valid"]:
                    raise ValueError(f"Generated run failed verification: {checked['errors']}")
                if checked.get("synthetic") is not False:
                    raise ValueError("Synthetic or unclassified execution checks cannot be used as benchmark results")
                verified_runs[manifest] = json.loads(manifest.read_text())
            run = verified_runs[manifest]
            artifact = relative_artifact(manifest.parent, evidence["artifact"])
            expected = run["artifacts"].get(evidence["artifact"])
            if expected is None or expected["sha256"] != evidence["artifact_sha256"] or sha256_file(artifact) != evidence["artifact_sha256"]:
                raise ValueError("Result artifact is absent from or differs from the recorded run")
            observed = json.loads(artifact.read_text())
            pointer = evidence["json_pointer"]
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                raise ValueError("Result JSON pointer must begin with /")
            try:
                for token in pointer[1:].split("/"):
                    token = token.replace("~1", "/").replace("~0", "~")
                    observed = observed[int(token)] if isinstance(observed, list) else observed[token]
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise ValueError("Result JSON pointer does not identify an artifact value") from exc
            if isinstance(observed, bool) or not isinstance(observed, (float, int)) or not math.isfinite(observed) or float(observed) != float(value):
                raise ValueError("Ledger value differs from its recorded output artifact")
        found[result_id] = row
    return document, found


def compare_results(generated, reference, allow_partial=False):
    reference_doc, expected = _load(reference, "reported")
    generated_doc, observed = _load(generated, "generated")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if unexpected:
        raise ValueError(f"Generated result IDs are absent from reference: {unexpected}")
    if missing and not allow_partial:
        raise ValueError(f"Generated results are incomplete: {missing}")
    shared = sorted(set(expected) & set(observed))
    if not shared:
        raise ValueError("No result IDs can be compared")
    comparisons = []
    for key in shared:
        wanted, got = expected[key], observed[key]
        differing = [field for field in CONTEXT_FIELDS if wanted["context"][field] != got["context"][field]]
        if differing:
            raise ValueError(f"Experiment context differs for {key}: {differing}")
        comparisons.append({"result_id": key, "context": wanted["context"],
                            "reported_value": wanted["value"], "generated_value": got["value"],
                            "difference": got["value"] - wanted["value"],
                            "paper_location": wanted["paper_location"], "evidence": got["evidence"]})
    return {"schema": "dialdet-comparison-v1", "complete": not missing,
            "compared": len(comparisons), "reference_rows": len(expected), "missing": missing,
            "reference_source": reference_doc["source"], "comparisons": comparisons,
            "reference_sha256": sha256_file(reference), "generated_sha256": sha256_file(generated),
            "note": "Comparison of recorded values under equal declared context; equality is not proof of original-paper reproduction."}
