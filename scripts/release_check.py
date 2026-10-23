#!/usr/bin/env python3
"""Verify the distributed DIAL-Det package and execute real local checks.

This gate does not install dependencies, fetch datasets, submit jobs, or claim
benchmark reproduction. Its smoke data are explicitly synthetic. Run after
PACKAGE_SHA256.json has been regenerated for the intended release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SHELL_FILES = (
    "run_dialdet_arrhenius.sh",
    "submit_dialdet_arrhenius.sh",
    "setup_dialdet_arrhenius.sh",
)
REQUIRED_FILES = (
    "README.md", "README_ARRHENIUS.txt", "IMPLEMENTATION_NOTES.md",
    "requirements.txt", "requirements_optional.txt", "pyproject.toml",
    ".gitignore", ".gitattributes", "CONTRIBUTING.md",
    ".github/workflows/ci.yml", ".github/ISSUE_TEMPLATE/bug_report.yml",
    "docs/METHOD.md", "docs/IMPLEMENTATION_STATUS.md", "docs/REVIEWER_GUIDE.md",
    "docs/GITHUB_SETUP.md", "docs/EXPERIMENTS.md", "docs/COCO_STUDY.md", "docs/ABLATIONS.md",
    "run_dialdet_arrhenius.py", "dialdet/__init__.py", "dialdet/cli.py",
    "dialdet/__main__.py", "dialdet/study.py", "tests/test_study.py",
    "scripts/release_check.py", "tests/test_release.py",
    "dialdet/data.py", "dialdet/evidence.py", "dialdet/objective.py",
    "dialdet/solver.py", "dialdet/certificate.py", "dialdet/training.py",
    "dialdet/metrics.py", "dialdet/risk.py", "dialdet/risk_runtime.py",
    "configs/paper_defaults.json", "configs/arrhenius.env.example",
    "configs/coco_study.example.json", "configs/ablations/full.json",
    "configs/ablations/no_appearance.json", "configs/ablations/no_rescue.json",
    "configs/ablations/no_mass_loss.json", "configs/ablations/no_transitivity.json",
    *SHELL_FILES,
)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def verify_package_hashes(root):
    """Check byte hashes and complete code, configuration and documentation coverage."""
    root = Path(root).resolve()
    manifest_path = root / "PACKAGE_SHA256.json"
    errors = []
    if not manifest_path.is_file():
        return {"valid": False, "checked_files": 0,
                "errors": ["PACKAGE_SHA256.json is missing"]}
    try:
        document = json.loads(manifest_path.read_text(), object_pairs_hook=_unique_object)
        hashes = document["sha256"]
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("sha256 must be a nonempty path-to-digest object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"valid": False, "checked_files": 0, "errors": [str(exc)]}
    checked = 0
    for relative, expected in hashes.items():
        path_parts = PurePosixPath(relative)
        if (path_parts.is_absolute() or ".." in path_parts.parts or "\\" in relative
                or relative != path_parts.as_posix() or not relative
                or relative == "PACKAGE_SHA256.json"):
            errors.append(f"Unsafe or noncanonical manifest path: {relative}")
            continue
        path = root / relative
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"Invalid SHA-256 value: {relative}")
            continue
        contains_symlink = any((root.joinpath(*path_parts.parts[:index])).is_symlink()
                               for index in range(1, len(path_parts.parts) + 1))
        if contains_symlink or root not in path.resolve().parents or not path.is_file():
            errors.append(f"Missing or nonlocal regular file: {relative}")
            continue
        checked += 1
        if digest_file(path) != expected:
            errors.append(f"Hash mismatch: {relative}")
    source_paths = set()
    for dirname in ("dialdet", "tests", "scripts", "configs", ".github", "docs"):
        directory = root / dirname
        if directory.is_dir():
            source_paths.update(str(path.relative_to(root)) for path in directory.rglob("*")
                                if path.is_file() and "__pycache__" not in path.parts
                                and path.suffix not in {".pyc", ".pyo"})
    source_paths.update(str(path.relative_to(root)) for path in root.iterdir()
                        if path.is_file() and (path.suffix in {".py", ".sh", ".toml", ".yaml", ".yml", ".cfg", ".ini"}
                                             or path.name in {".gitignore", ".gitattributes"}
                                             or path.name.startswith("requirements")))
    for relative in sorted(source_paths - set(hashes)):
        errors.append(f"Executable/configuration source is untracked: {relative}")
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"Required release file is missing: {relative}")
        elif relative not in hashes:
            errors.append(f"Required release file is untracked: {relative}")
    return {"valid": not errors, "checked_files": checked, "errors": errors,
            "manifest_version": document.get("version"),
            "manifest_sha256": digest_file(manifest_path),
            "executable_and_configuration_files": len(source_paths)}


def distributed_modules(root, neural=False):
    """Enumerate package modules without executing the command-line entry point."""
    optional = set() if neural else {"evidence", "training"}
    return ["dialdet"] + [f"dialdet.{path.stem}" for path in sorted((Path(root) / "dialdet").glob("*.py"))
                          if path.stem not in {"__init__", "__main__", *optional}]


def run_check(name, argv, root, output, environment, timeout=1200):
    """Run argv directly, retaining full logs and a truthful exit status."""
    started = time.perf_counter()
    log = Path(output) / f"{name}.log"
    timed_out = False
    with log.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(argv, cwd=root, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       text=True, timeout=timeout, check=False)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
            handle.write(f"\nTimed out after {timeout} seconds.\n")
        except OSError as exc:
            returncode = None
            handle.write(f"{type(exc).__name__}: {exc}\n")
    return {"name": name, "argv": list(map(str, argv)), "returncode": returncode,
            "passed": returncode == 0, "timed_out": timed_out,
            "seconds": time.perf_counter() - started, "log": log.name}


def verify_wheel(root, output, environment, neural=False):
    """Build/install offline, then import and run the wheel outside the source tree.

    Existing dependencies are used; no index, isolated build-environment download
    or dependency installation is permitted. Build in a copy so release checking
    does not add build products to the source being verified.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks, errors = [], []
    result = {"valid": False, "checks": checks, "errors": errors,
              "dependency_resolution": "disabled; existing environment only",
              "wheel": None, "wheel_sha256": None}
    build_environment = dict(environment, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    with tempfile.TemporaryDirectory(prefix="dialdet-wheel-") as temp:
        temporary = Path(temp)
        source = temporary / "source"
        source.mkdir()
        for filename in ("pyproject.toml", "README.md", "setup.cfg", "setup.py", "MANIFEST.in", "LICENSE", "LICENSE.txt"):
            candidate = root / filename
            if candidate.is_file():
                shutil.copy2(candidate, source / filename)
        shutil.copytree(root / "dialdet", source / "dialdet",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        wheels = temporary / "wheels"
        checks.append(run_check("wheel_build", [sys.executable, "-m", "pip", "wheel", "--no-index",
                                "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(source)],
                                temporary, output, build_environment))
        built = sorted(wheels.glob("*.whl"))
        if checks[-1]["passed"] and len(built) == 1:
            saved = output / "wheels" / built[0].name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built[0], saved)
            result.update(wheel=str(saved.relative_to(output)), wheel_sha256=digest_file(saved))
            installed = temporary / "installed"
            checks.append(run_check("wheel_install", [sys.executable, "-m", "pip", "install", "--no-index",
                                    "--no-deps", "--no-compile", "--target", str(installed), str(built[0])],
                                    temporary, output, build_environment))
            if checks[-1]["passed"]:
                # Keep externally supplied dependency paths while removing every
                # checkout path. Verify each module actually resolves to the wheel.
                retained = []
                for entry in environment.get("PYTHONPATH", "").split(os.pathsep):
                    if entry:
                        resolved = Path(entry).resolve()
                        if resolved != root and root not in resolved.parents:
                            retained.append(str(resolved))
                installed_environment = dict(build_environment)
                installed_environment["PYTHONPATH"] = os.pathsep.join([str(installed), *retained])
                modules = distributed_modules(root, neural=neural)
                import_code = "\n".join([
                    "import importlib, json", "from pathlib import Path",
                    "from importlib.metadata import distribution",
                    f"target = Path({str(installed)!r}).resolve()", f"modules = {modules!r}",
                    "locations = {}", "for name in modules:",
                    "    module = importlib.import_module(name)",
                    "    location = Path(module.__file__).resolve()",
                    "    assert target in location.parents, (name, str(location))",
                    "    locations[name] = str(location.relative_to(target))",
                    "dist = distribution('dialdet')",
                    "assert any(e.group == 'console_scripts' and e.name == 'dialdet' and e.value == 'dialdet.cli:main' for e in dist.entry_points)",
                    "print(json.dumps({'version': dist.version, 'imported_modules': locations}, sort_keys=True))",
                ])
                checks.append(run_check("wheel_imports", [sys.executable, "-c", import_code],
                                        temporary, output, installed_environment))
                entrypoint = installed / "bin" / "dialdet"
                checks.append(run_check("wheel_entrypoint", [sys.executable, str(entrypoint), "--help"],
                                        temporary, output, installed_environment))
        elif checks[-1]["passed"]:
            errors.append(f"Expected one built wheel, found {len(built)}")
    errors.extend(f"Check failed: {check['name']} (see {check['log']})" for check in checks if not check["passed"])
    result["valid"] = not errors and len(checks) == 4
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory for the release report and full command logs")
    parser.add_argument("--neural", action="store_true",
                        help="Require PyTorch, run neural tests and the two-epoch CPU neural smoke")
    parser.add_argument("--wheel", action="store_true",
                        help="Also build/install a wheel offline and verify its imports and console entry point; pip/setuptools/wheel must already be available")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = verify_package_hashes(ROOT)
    checks = []
    errors = list(hashes["errors"])
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])])
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    # Import each distributed module; only explicitly optional torch modules
    # are excluded from a core-only check. Undefined imports cannot hide behind
    # successful byte compilation or wheel building.
    modules = distributed_modules(ROOT, neural=args.neural)
    import_code = ("import importlib,json; modules=" + repr(modules)
                   + "; [importlib.import_module(n) for n in modules]; "
                   + "print(json.dumps({'imported_modules':modules}))")
    checks.append(run_check("imports", [sys.executable, "-c", import_code], ROOT, output, environment))
    checks.append(run_check("cli_help", [sys.executable, str(ROOT / "run_dialdet_arrhenius.py"), "--help"],
                            ROOT, output, environment))
    shell = shutil.which("bash")
    if shell:
        for name in SHELL_FILES:
            checks.append(run_check(f"shell_{Path(name).stem}", [shell, "-n", str(ROOT / name)],
                                    ROOT, output, environment))
    else:
        errors.append("bash is required for the shell syntax gate")
    tests_report, smoke_report = None, None
    with tempfile.TemporaryDirectory(prefix="dialdet-release-") as temp:
        temporary = Path(temp)
        runner = [sys.executable, str(ROOT / "run_dialdet_arrhenius.py")]
        tests = [*runner, "self-test", "--output", str(temporary / "tests")]
        smoke = [*runner, "smoke", "--output", str(temporary / "smoke"), "--device", "cpu"]
        if args.neural:
            tests.append("--neural")
            smoke.append("--neural")
        checks.append(run_check("unittest", tests, ROOT, output, environment))
        checks.append(run_check("synthetic_smoke", smoke, ROOT, output, environment))
        for relative, name in (("tests/TEST_REPORT.json", "TEST_REPORT.json"),
                               ("smoke/SMOKE_REPORT.json", "SMOKE_REPORT.json")):
            source = temporary / relative
            if source.is_file():
                document = json.loads(source.read_text())
                (output / name).write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
                if name == "TEST_REPORT.json":
                    tests_report = document
                else:
                    smoke_report = document
    errors.extend(f"Check failed: {check['name']} (see {check['log']})" for check in checks if not check["passed"])
    if tests_report is None:
        errors.append("self-test did not produce TEST_REPORT.json")
    elif tests_report.get("tests_run", 0) <= 0 or not tests_report.get("success"):
        errors.append("self-test did not report a nonempty passing test suite")
    if smoke_report is None:
        errors.append("smoke did not produce SMOKE_REPORT.json")
    wheel_report = verify_wheel(ROOT, output, environment, neural=args.neural) if args.wheel else None
    if wheel_report is not None:
        checks.extend(wheel_report["checks"])
        errors.extend(wheel_report["errors"])
        if not wheel_report["valid"] and not wheel_report["errors"]:
            errors.append("Wheel verification did not complete every required stage")
    report = {"valid": not errors, "neural_requested": args.neural,
              "wheel_requested": args.wheel, "wheel_summary": wheel_report,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "machine": platform.machine(),
              "hashes": hashes, "checks": checks, "test_summary": tests_report,
              "smoke_summary": smoke_report, "errors": errors,
              "scope": "Local executable release verification on synthetic data; no dataset benchmark, Slurm job, Arrhenius GPU execution or manuscript result reproduction was performed."}
    destination = output / "RELEASE_CHECK.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"valid": report["valid"], "report": str(destination),
                      "checks": len(checks), "errors": len(errors)}, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
