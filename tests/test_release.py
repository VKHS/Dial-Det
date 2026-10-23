"""Release integrity and shell routing checks; no Slurm or GPU execution."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dialdet_release_gate", ROOT / "scripts" / "release_check.py")
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class ReleaseHashTests(unittest.TestCase):
    def fixture(self, root):
        for name in GATE.REQUIRED_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture for {name}\n")
        hashes = {str(path.relative_to(root)): GATE.digest_file(path)
                  for path in root.rglob("*") if path.is_file()}
        (root / "PACKAGE_SHA256.json").write_text(json.dumps({"version": "fixture", "sha256": hashes}))
        return hashes

    def test_complete_package_and_byte_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            self.assertTrue(GATE.verify_package_hashes(root)["valid"])
            (root / "dialdet" / "solver.py").write_text("changed scientific code\n")
            report = GATE.verify_package_hashes(root)
            self.assertFalse(report["valid"])
            self.assertIn("Hash mismatch: dialdet/solver.py", report["errors"])

    def test_missing_core_and_untracked_importable_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "dialdet" / "solver.py").unlink()
            (root / "dialdet" / "new_module.py").write_text("pass\n")
            errors = GATE.verify_package_hashes(root)["errors"]
            self.assertTrue(any("solver.py" in error for error in errors))
            self.assertTrue(any("untracked" in error and "new_module.py" in error for error in errors))

    def test_workflow_metadata_and_new_configuration_require_hash_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = self.fixture(root)
            omitted = (".github/workflows/ci.yml", ".gitignore", "pyproject.toml", "docs/IMPLEMENTATION_STATUS.md")
            for name in omitted:
                hashes.pop(name)
            added = (".github/workflows/extra.yaml", ".github/ISSUE_TEMPLATE/extra.yml",
                     "configs/nested/runtime.conf", "requirements_ci.txt")
            for name in added:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("new configuration\n")
            (root / "PACKAGE_SHA256.json").write_text(json.dumps({"sha256": hashes}))
            report = GATE.verify_package_hashes(root)
            self.assertFalse(report["valid"])
            for name in (*omitted, *added):
                self.assertTrue(any("untracked" in error and name in error for error in report["errors"]), name)

    def test_symlinked_directory_does_not_create_a_valid_manifest_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = self.fixture(root)
            (root / "alias").symlink_to(root / "configs", target_is_directory=True)
            hashes["alias/paper_defaults.json"] = hashes["configs/paper_defaults.json"]
            (root / "PACKAGE_SHA256.json").write_text(json.dumps({"sha256": hashes}))
            self.assertIn("Missing or nonlocal regular file: alias/paper_defaults.json",
                          GATE.verify_package_hashes(root)["errors"])

    def test_hash_manifest_rejects_traversal_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hashes = self.fixture(root)
            hashes["../outside.py"] = "0" * 64
            manifest = root / "PACKAGE_SHA256.json"
            manifest.write_text(json.dumps({"sha256": hashes}))
            self.assertTrue(any("Unsafe" in error for error in GATE.verify_package_hashes(root)["errors"]))
            manifest.write_text('{"sha256":{"x":"a","x":"b"}}')
            self.assertTrue(any("Duplicate" in error for error in GATE.verify_package_hashes(root)["errors"]))

    def test_command_runner_keeps_real_failure_and_complete_log(self):
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = GATE.run_check("intentional_failure", [sys.executable, "-c", "print('real output'); raise SystemExit(7)"],
                                    root, root, dict(os.environ), timeout=10)
            self.assertEqual(report["returncode"], 7)
            self.assertFalse(report["passed"])
            self.assertIn("real output", (root / report["log"]).read_text())


@unittest.skipUnless(shutil.which("bash"), "bash is required for shell checks")
class ArrheniusRoutingTests(unittest.TestCase):
    def environment(self, root, **extra):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("DIAL_") and key != "CM_BASE"}
        environment.update(DIAL_BASE=str(root / "server root"), DIAL_CODE_DIR=str(ROOT),
                           DIAL_PLAN="1", DIAL_OUTPUT=str(root / "output with spaces"),
                           DIAL_PYTHON="/existing environment/bin/python")
        environment.update(extra)
        return environment

    def run_plan(self, root, mode, *arguments, **extra):
        result = subprocess.run(["bash", str(ROOT / "run_dialdet_arrhenius.sh"), mode, *arguments],
                                env=self.environment(root, **extra), text=True, capture_output=True, timeout=10)
        command = next((line.removeprefix("[COMMAND]").strip() for line in result.stdout.splitlines()
                        if line.startswith("[COMMAND]")), None)
        return result, shlex.split(command) if command is not None else []

    def test_collect_risk_uses_single_budget_checkpoint_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, args = self.run_plan(root, "collect-risk", DIAL_DATASET="crowdhuman",
                DIAL_MANIFEST="/data with spaces/focal.json", DIAL_CHECKPOINT="/models with spaces/calibrated.pt",
                DIAL_CONFIG="/configs with spaces/rule.json", DIAL_BUDGET="25", DIAL_BUDGETS="10 50 100")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[args.index("--manifest") + 1], "/data with spaces/focal.json")
            self.assertEqual(args[args.index("--checkpoint") + 1], "/models with spaces/calibrated.pt")
            self.assertEqual(args[args.index("--config") + 1], "/configs with spaces/rule.json")
            self.assertEqual(args[args.index("--budget") + 1], "25")
            self.assertNotIn("--budgets", args)
            self.assertFalse((root / "server root").exists())

    def test_tune_local_uses_development_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_plan(Path(temp), "tune-local", DIAL_DATASET="crowdhuman",
                DIAL_MANIFEST="/data/roles.json", DIAL_CHECKPOINT="/data/model.pt",
                DIAL_DEV_ANNOTATIONS="/data/annotation_train.odgt",
                DIAL_EVAL_ANNOTATIONS="/data/annotation_val.odgt", DIAL_CROWDHUMAN_EVALUATOR="/tools/CrowdDetection")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[args.index("--crowdhuman-annotations") + 1], "/data/annotation_train.odgt")
            self.assertNotIn("/data/annotation_val.odgt", args)

    def test_score_family_requires_registered_policies_and_keeps_their_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = {"DIAL_DATASET": "crowdhuman", "DIAL_MANIFEST": "/data with spaces/focal.json",
                        "DIAL_CHECKPOINT": "/models with spaces/calibrated.pt",
                        "DIAL_DEV_ANNOTATIONS": "/data with spaces/annotation_train.odgt",
                        "DIAL_EVAL_ANNOTATIONS": "/data with spaces/annotation_val.odgt",
                        "DIAL_CROWDHUMAN_EVALUATOR": "/tools with spaces/CrowdDetection",
                        "DIAL_BUDGET": "25", "DIAL_CONFIG": "/must not override/solver.json"}
            result, _ = self.run_plan(root, "score-family", **settings)
            self.assertEqual(result.returncode, 64)
            self.assertIn("--policy-records", result.stderr)
            result, args = self.run_plan(root, "score-family", DIAL_POLICY_RECORDS="/risk with spaces/policy_records.json", **settings)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[args.index("--policy-records") + 1], "/risk with spaces/policy_records.json")
            self.assertEqual(args[args.index("--crowdhuman-annotations") + 1], "/data with spaces/annotation_train.odgt")
            self.assertEqual(args[args.index("--checkpoint") + 1], "/models with spaces/calibrated.pt")
            self.assertEqual(args[args.index("--device") + 1], "cuda")
            for flag in ("--budget", "--budgets", "--config", "--quality-mode"):
                self.assertNotIn(flag, args)

    def test_tune_local_rejects_implicit_proxy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, _ = self.run_plan(root, "tune-local", DIAL_MANIFEST="/data/roles.json", DIAL_CHECKPOINT="/data/model.pt")
            self.assertNotEqual(result.returncode, 0)
            result, _ = self.run_plan(root, "tune-local", "--allow-matching-proxy",
                                     DIAL_MANIFEST="/data/roles.json", DIAL_CHECKPOINT="/data/model.pt")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_statistics_has_only_cpu_stage_options(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_plan(Path(temp), "paired-statistics", DIAL_STATISTICS_SPEC="/data with spaces/pairs.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[args.index("--specification") + 1], "/data with spaces/pairs.json")
            for flag in ("--device", "--dataset", "--manifest", "--checkpoint"):
                self.assertNotIn(flag, args)

    def test_study_defaults_to_plan_and_preserves_specification_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specification = "/data with spaces/study;$(not_executed).json"
            result, args = self.run_plan(root, "study", DIAL_STUDY_SPEC=specification,
                DIAL_DATASET="crowdhuman", DIAL_MANIFEST="/ignored/manifest.json",
                DIAL_DEVICE="cuda", DIAL_EPOCHS="999", DIAL_BUDGETS="1 2",
                DIAL_CONFIG="/ignored/config.json", DIAL_RESUME="/ignored/checkpoint.pt")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[2:], ["study", "--output", str(root / "output with spaces"),
                                       "--specification", specification])
            self.assertNotIn("--execute", args)
            self.assertNotIn("--resume", args)
            self.assertFalse((root / "server root").exists())

    def test_study_execution_resume_and_cli_specification_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_plan(Path(temp), "study", "--specification=/override path/study.json",
                                        "--execute", "--resume", DIAL_STUDY_SPEC="/default/study.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[-3:], ["--specification=/override path/study.json", "--execute", "--resume"])
            self.assertEqual(args.count("--execute"), 1)

    def test_study_requires_a_specification_and_rejects_conflicting_stage_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for options in ((), ("--specification=",), ("--specification", "--execute")):
                result, _ = self.run_plan(root, "study", *options)
                self.assertEqual(result.returncode, 64, result.stderr)
            for options in (("--device", "cpu"), ("--epochs", "1"), ("--budgets", "10"),
                            ("--config", "settings.json"), ("--dataset", "crowdhuman"),
                            ("--resume", "checkpoint.pt")):
                result, _ = self.run_plan(root, "study", *options, DIAL_STUDY_SPEC="/data/study.json")
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn("specification", result.stderr)

    def test_study_preflight_uses_spec_device_and_only_execution_needs_neural_coco(self):
        # Stop at the preflight boundary. No site module, CUDA tensor, source
        # snapshot, Slurm command or scientific task is actually executed.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            module = fake_bin / "module"
            module.write_text("#!/bin/sh\nexit 0\n")
            module.chmod(0o755)
            interpreter = fake_bin / "preflight-python"
            interpreter.write_text(f"#!{sys.executable}\n" + "\n".join([
                "import json, os, subprocess, sys", "from pathlib import Path",
                "if len(sys.argv) == 3 and sys.argv[1] == '-':",
                "    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)",
                "Path(os.environ['DIAL_TEST_PREFLIGHT_RECORD']).write_text(json.dumps({",
                "    'device': os.environ['DIAL_DEVICE'],",
                "    'modules': os.environ['DIAL_REQUIRED_MODULES'].split()}))",
                "raise SystemExit(77)", "",
            ]))
            interpreter.chmod(0o755)
            specification = root / "study with spaces.json"
            record = root / "preflight.json"
            for execute, spec_device, expected_device in ((False, "cuda", "cpu"),
                                                         (True, "cpu", "cpu"),
                                                         (True, "cuda", "cuda")):
                specification.write_text(json.dumps({"device": spec_device}))
                environment = self.environment(root, DIAL_PLAN="0", DIAL_PYTHON=str(interpreter),
                    DIAL_STUDY_SPEC=str(specification), DIAL_DEVICE="cuda",
                    DIAL_TEST_PREFLIGHT_RECORD=str(record), PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
                result = subprocess.run(["bash", str(ROOT / "run_dialdet_arrhenius.sh"), "study",
                                         *(["--execute"] if execute else [])], env=environment,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 77, result.stderr)
                observed = json.loads(record.read_text())
                self.assertEqual(observed["device"], expected_device)
                self.assertEqual(set(observed["modules"]),
                                 {"numpy", "scipy", "torch", "pycocotools"} if execute else {"numpy", "scipy"})

    def test_required_values_must_not_be_empty_or_other_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for extra in (("--checkpoint=",), ("--checkpoint", "--device", "cpu")):
                result, _ = self.run_plan(root, "collect-risk", *extra, DIAL_MANIFEST="/data/roles.json")
                self.assertEqual(result.returncode, 64)
            result, _ = self.run_plan(root, "paired-statistics", "--specification=")
            self.assertEqual(result.returncode, 64)

    def test_cli_value_precedence_remains_separate_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_plan(Path(temp), "collect-risk", "--budget", "10", "--checkpoint=/override/model.pt",
                                        DIAL_MANIFEST="/data/roles.json", DIAL_CHECKPOINT="/default/model.pt", DIAL_BUDGET="100")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[-3:], ["--budget", "10", "--checkpoint=/override/model.pt"])

    def test_submit_preserves_account_and_quotes_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = root / "code with spaces"
            code.mkdir()
            shutil.copyfile(ROOT / "run_dialdet_arrhenius.sh", code / "run_dialdet_arrhenius.sh")
            (code / "run_dialdet_arrhenius.py").write_text("# plan-only fixture\n")
            for mode, setting, extra in (("paired-statistics", {"DIAL_STATISTICS_SPEC": "/data/pairs.json"}, []),
                                         ("study", {"DIAL_STUDY_SPEC": "/data with spaces/study.json"}, ["--execute", "--resume"])):
                environment = self.environment(root, DIAL_CODE_DIR=str(code), DIAL_DRY_RUN="1", **setting)
                result = subprocess.run(["bash", str(ROOT / "submit_dialdet_arrhenius.sh"), mode, *extra],
                                        env=environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(line for line in result.stdout.splitlines() if line.startswith("[SUBMIT]"))
                args = shlex.split(line.removeprefix("[SUBMIT]").strip())
                self.assertIn("--account=naiss2025-5-643-gpu", args)
                self.assertIn("--partition=gpu", args)
                self.assertIn(str(code / "run_dialdet_arrhenius.sh"), args)
                self.assertIn(f"--output={root / 'server root' / 'logs'}/DIALDET_%j.out", args)
                self.assertIn("[DRY_RUN] No job submitted.", result.stdout)
                self.assertEqual(args[-len(extra):] if extra else [], extra)

    def test_all_shell_files_parse_and_preserve_server_defaults(self):
        for name in GATE.SHELL_FILES:
            result = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")
        text = (ROOT / "run_dialdet_arrhenius.sh").read_text()
        for required in ("naiss2025-5-643-gpu", "naiss2025-6-428/from-alvis/tnnls",
                         "venv_arrhenius_gh200", "GPU/buildenv-nvhpc/25.9-cu13.0",
                         "GPU/Python/3.13.5-bare-gcc-2025b-eb"):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
