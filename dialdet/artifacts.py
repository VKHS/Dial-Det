"""Portable run records and strict verification of the files a command produced.

Hashes record file consistency. Scientific claims also depend on the declared
training histories, data populations and experimental protocol.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

from . import __version__
from .data import atomic_json, sha256_file

SCHEMA = "dialdet-run-v1"
_EXCLUDED = {"__pycache__", ".pytest_cache", ".git", "000_MANIFEST"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def relative_artifact(root, relative):
    """Resolve an artifact reference without allowing paths outside its run."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact references must be relative paths inside the run")
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError(f"Missing or external run artifact: {relative}")
    return target


def source_fingerprint(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent.parent
    files = sorted((root / "dialdet").rglob("*.py"))
    for name in ("run_dialdet_arrhenius.py", "SOURCE_PROVENANCE.json", "configs/paper_defaults.json"):
        path = root / name
        if path.is_file():
            files.append(path)
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in files if not path.is_symlink()}
    return {"sha256": _digest(hashes), "files": hashes}


def runtime_description():
    versions = {}
    for package in ("numpy", "scipy", "torch", "torchvision", "pycocotools"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "packages": versions}


def _artifact_files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in _EXCLUDED for part in rel.parts) or path.is_symlink() or not path.is_file():
            continue
        if rel.as_posix() in {"RUN_MANIFEST.json", "RUN_STATUS.json", "FAILURE.json", "JOB_STATUS.json", "JOB_STATUS.txt"}:
            continue
        result[rel.as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def write_run_manifest(output, command, arguments, inputs=None, synthetic=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_records = {}
    for name, path in sorted((inputs or {}).items()):
        if path is None:
            continue
        path = Path(path).resolve()
        if path.is_file():
            input_records[name] = {"path": str(path), "sha256": sha256_file(path),
                                   "bytes": path.stat().st_size}
    payload = {"schema": SCHEMA, "command": command, "arguments": arguments,
               "implementation_version": __version__, "source": source_fingerprint(),
               "runtime": runtime_description(), "inputs": input_records,
               "artifacts": _artifact_files(root), "synthetic": synthetic,
               "provenance_status": "generated_by_local_run",
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Recorded file integrity and execution identity; benchmark reproduction requires matching data, protocol and evidence."}
    payload["run_sha256"] = _digest(payload)
    atomic_json(root / "RUN_MANIFEST.json", payload)
    # A successful retry must not leave an earlier failure as its current status.
    failure = root / "FAILURE.json"
    if failure.is_file():
        failure.unlink()
    atomic_json(root / "RUN_STATUS.json", {"status": "completed", "command": command,
                                           "run_sha256": payload["run_sha256"]})
    return payload


def verify_run(path, check_inputs=False):
    path = Path(path).resolve()
    manifest = path / "RUN_MANIFEST.json" if path.is_dir() else path
    root = manifest.parent
    record = json.loads(manifest.read_text())
    errors = []
    status_path = root / "RUN_STATUS.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed" or status.get("run_sha256") != record.get("run_sha256"):
            errors.append("Run status does not identify a completed matching run")
    if (root / "FAILURE.json").is_file():
        errors.append("The output directory contains a recorded command failure")
    payload = {k: v for k, v in record.items() if k != "run_sha256"}
    if record.get("schema") != SCHEMA:
        errors.append("Unsupported run-record schema")
    if _digest(payload) != record.get("run_sha256"):
        errors.append("Run manifest digest mismatch")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("A completed run must list its output artifacts")
        artifacts = {}
    for name, expected in artifacts.items():
        try:
            target = relative_artifact(root, name)
            if target.stat().st_size != expected["bytes"] or sha256_file(target) != expected["sha256"]:
                errors.append(f"Changed artifact: {name}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid artifact {name}: {exc}")
    if check_inputs:
        for name, expected in record.get("inputs", {}).items():
            target = Path(expected["path"])
            if not target.is_file() or sha256_file(target) != expected["sha256"]:
                errors.append(f"Missing or changed external input: {name}")
    current = set(_artifact_files(root))
    return {"valid": not errors, "errors": errors, "run_sha256": record.get("run_sha256"),
            "command": record.get("command"), "artifacts_checked": len(artifacts),
            "external_inputs_checked": bool(check_inputs),
            "unlisted_files": sorted(current - set(artifacts)),
            "synthetic": record.get("synthetic"), "provenance_status": record.get("provenance_status"),
            "note": "Content hashes detect inconsistency; they are not an external signature or a scientific performance certificate."}


def invocation_inputs(arguments):
    """Collect explicitly supplied file inputs, without dumping environment secrets."""
    keys = ("manifest", "checkpoint", "config", "resume", "annotations", "coco_annotations",
            "crowdhuman_annotations", "development_annotations", "policy_records", "development_scores",
            "provenance", "certificate", "specification", "generated", "reference", "weights")
    # Training's resume names a checkpoint, whereas study's resume is a flag.
    # Only filesystem inputs belong in this mapping.
    inputs = {key: arguments.get(key) for key in keys
              if isinstance(arguments.get(key), (str, Path)) and arguments.get(key)}
    predictions = arguments.get("predictions")
    if predictions:
        path = Path(predictions)
        inputs["predictions"] = str(path / "predictions.jsonl" if path.is_dir() else path)
    for key in ("manifests", "prediction_dirs"):
        for number, value in enumerate(arguments.get(key) or []):
            path = Path(value)
            if key == "prediction_dirs":
                path = path / "predictions.jsonl"
            inputs[f"{key}_{number}"] = str(path)
    return inputs
