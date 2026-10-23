"""Contract tests for orchestration, registration and verified task reuse."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dialdet.artifacts import invocation_inputs, write_run_manifest
from dialdet.cli import main, make_parser
from dialdet.data import CandidateRecord, atomic_json, load_manifest, make_crossfit_manifests, save_record, sha256_file
from dialdet.study import run_study


def study_fixture(root, seeds=None, comparisons=None):
    """Small invented COCO-format population, always marked synthetic."""
    data = root / "data"
    entries, images, annotations = [], [], []
    for image_id in range(1, 7):
        record = CandidateRecord(
            image_id=str(image_id), ids=np.array([10, 20]),
            boxes=np.array([[8, 8, 28, 28], [9, 9, 29, 29]], dtype=float),
            classes=np.array([1, 1]), scores=np.array([.9, .8]),
            features=np.array([[1., image_id / 10, .2, .3], [.8, .4, image_id / 10, .1]], dtype=np.float32),
            image_size=(64, 64), gt_boxes=np.array([[8, 8, 28, 28]], dtype=float),
            gt_classes=np.array([1]), gt_ids=np.array([image_id]), gt_ignore=np.array([False]),
            metadata={"synthetic": True, "fixture": "study contract test"})
        path = save_record(record, data / f"{image_id}.npz")
        entries.append({"image_id": str(image_id), "path": path.name, "role": "pool", "sha256": sha256_file(path)})
        images.append({"id": image_id, "width": 64, "height": 64, "file_name": f"{image_id}.jpg"})
        annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                            "bbox": [8, 8, 20, 20], "area": 400, "iscrowd": 0})
    # Deliberately omit is_synthetic on the manifest: record metadata must win
    # over the recognized COCO dataset name.
    atomic_json(data / "manifest.json", {"schema_version": 1, "dataset": "coco", "num_classes": 2,
                                         "protocol": "public_checkpoint_exports", "records": entries})
    atomic_json(data / "annotations.json", {"images": images, "annotations": annotations,
                                            "categories": [{"id": 1, "name": "person"}], "info": {}})
    spec = {"schema_version": 1, "source_manifest": "data/manifest.json",
            "coco_annotations": "data/annotations.json", "epochs": 1, "budgets": [10, 100]}
    if seeds is not None:
        spec["seeds"] = seeds
    if comparisons is not None:
        spec["comparisons"] = comparisons
    path = root / "study.json"
    atomic_json(path, spec)
    return path


class FakeCLI:
    """Produce real signed-content run records, without pretending to train."""
    def __init__(self):
        self.calls = []
        self.fail_once = None

    def __call__(self, argv, **kwargs):
        assert isinstance(argv, list) and argv[0] == sys.executable
        assert Path(argv[1]).is_absolute() and kwargs["shell"] is False
        self.calls.append(argv)
        args = make_parser().parse_args(argv[2:])
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        if args.command == self.fail_once:
            self.fail_once = None
            atomic_json(root / "partial.json", {"failed": True})
            return SimpleNamespace(returncode=7)
        if args.command == "make-crossfit":
            make_crossfit_manifests(args.manifest, root, args.seed, args.dev_fraction)
        elif args.command == "pipeline":
            atomic_json(root / "PIPELINE.json", {"is_synthetic": True, "mock": True})
            atomic_json(root / "inference" / "predictions.jsonl", {"mock": True})
            atomic_json(root / "evaluation" / "EVALUATION.json", {"is_synthetic": True, "mock": True})
            atomic_json(root / "evaluation" / "per_image_audits.json", {"mock": True})
        elif args.command == "merge-crossfit":
            document = json.loads(Path(args.manifests[0]).read_text())
            for entry in document["records"]:
                entry["role"] = "audit"
            atomic_json(root / "manifest.json", document)
            atomic_json(root / "MERGE.json", {"is_synthetic": True, "mock": True})
            atomic_json(root / "predictions.jsonl", {"mock": True})
        elif args.command == "evaluate":
            atomic_json(root / "EVALUATION.json", {"is_synthetic": True, "mock": True})
            atomic_json(root / "per_image_audits.json", {"mock": True})
        elif args.command == "paired-statistics":
            document = json.loads(Path(args.specification).read_text())
            for run in document["runs"]:
                assert Path(run["audits"]).is_file()
                assert Path(run["predictions"]).is_file()
                assert Path(run["run_manifest"]).is_file()
            atomic_json(root / "PAIRED_STATISTICS.json", {"is_synthetic": True, "mock": True})
        else:
            raise AssertionError(args.command)
        write_run_manifest(root, args.command, vars(args), invocation_inputs(vars(args)), synthetic=True)
        return SimpleNamespace(returncode=0)


class StudyTests(unittest.TestCase):
    def test_plan_has_five_seeds_two_folds_and_union_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, comparisons=[{"metric": "AP"}, {"metric": "mean_image_shortfall"}])
            atomic_json(root / "model.json", {"solver": {}, "model": {"use_rescue": False}})
            spec = json.loads(path.read_text())
            spec["config"] = "model.json"
            atomic_json(path, spec)
            output = root / "output"
            # The outer public CLI creates this before run_study is dispatched.
            atomic_json(output / "RUN_STATUS.json", {"status": "running", "command": "study"})
            with patch("dialdet.study.subprocess.run") as runner:
                result = run_study(path, output)
            runner.assert_not_called()
            self.assertEqual(result["status"], "planned")
            self.assertTrue(result["is_synthetic"])
            plan = json.loads((output / "STUDY_PLAN.json").read_text())
            self.assertEqual(len(plan["tasks"]), 23)
            pipelines = [task for task in plan["tasks"] if task["command"] == "pipeline"]
            self.assertEqual(len(pipelines), 10)
            self.assertEqual({task["arguments"]["seed"] for task in pipelines}, set(range(1, 6)))
            self.assertEqual({Path(task["arguments"]["manifest"]).name for task in pipelines}, {"fold_0.json", "fold_1.json"})
            for task in pipelines:
                self.assertEqual(task["arguments"]["budgets"], [10, 100])
                self.assertEqual(task["arguments"]["quality_mode"], "coco")
                self.assertEqual(task["arguments"]["config"], str(root / "model.json"))
                self.assertEqual(task["arguments"]["coco_annotations"], str(root / "data" / "annotations.json"))
                self.assertEqual(task["arguments"]["development_annotations"], task["arguments"]["coco_annotations"])
            evaluations = [task for task in plan["tasks"] if task["command"] == "evaluate"]
            self.assertEqual(len(evaluations), 5)
            for task in evaluations:
                self.assertEqual(Path(task["arguments"]["predictions"]).name, "union")
            for spec in plan["generated_specifications"].values():
                self.assertEqual(len(spec["runs"]), 5)
                self.assertEqual(spec["repeats"], 2000)
                self.assertTrue(all("/union/" in run["predictions"] for run in spec["runs"]))
            self.assertFalse((output / "folds").exists())
            self.assertFalse((output / "specifications").exists())

    def test_rejects_bad_configuration_and_unknown_keys(self):
        bad = [({"schema_version": True}, "schema_version"), ({"schema_version": 2}, "schema_version"),
               ({"seeds": [1]}, "at least 2"), ({"seeds": [1, 1]}, "unique"),
               ({"seeds": [True, 2]}, "integer"), ({"epochs": 0}, "integer"),
               ({"device": "cuda:3"}, "device"), ({"dev_fraction": .5}, "dev_fraction"),
               ({"budgets": [10, 10, 100]}, "unique"), ({"budgets": [10]}, "include 100"),
               ({"unknown": 1}, "Unknown"), ({"comparisons": [{"metric": "accuracy"}]}, "metric"),
               ({"comparisons": [{"metric": "AP", "budget": 20}]}, "Comparison budget"),
               ({"comparisons": [{"metric": "AP", "method_b": "dialdet"}]}, "different"),
               ({"comparisons": [{"metric": "AP", "repeats": 1}]}, "repeats"),
               ({"comparisons": [{"metric": "AP"}, {"metric": "AP", "seed": 2}]}, "Duplicate")]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root)
            original = json.loads(path.read_text())
            for changes, message in bad:
                with self.subTest(changes=changes):
                    atomic_json(path, {**original, **changes})
                    with self.assertRaisesRegex(ValueError, message):
                        run_study(path, root / "output")
            path.write_text('{"schema_version": 1, "schema_version": 1}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                run_study(path, root / "output")

    def test_rejects_wrong_dataset_fixed_roles_and_incomplete_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root)
            manifest = root / "data" / "manifest.json"
            original = json.loads(manifest.read_text())
            atomic_json(manifest, {**original, "dataset": "crowdhuman"})
            with self.assertRaisesRegex(ValueError, "COCO"):
                run_study(path, root / "output")
            original["records"][0]["role"] = "audit"
            atomic_json(manifest, original)
            with self.assertRaisesRegex(ValueError, "role=pool"):
                run_study(path, root / "output")
            original["records"][0]["role"] = "pool"
            atomic_json(manifest, original)
            ann_path = root / "data" / "annotations.json"
            annotations = json.loads(ann_path.read_text())
            annotations["images"].pop()
            atomic_json(ann_path, annotations)
            with self.assertRaisesRegex(ValueError, "complete frozen"):
                run_study(path, root / "output")

    def test_declared_semantic_hash_uses_manifest_dataset_metadata(self):
        from dialdet.provenance import candidate_semantic_hash
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[1, 2])
            manifest = root / "data" / "manifest.json"
            document = json.loads(manifest.read_text())
            for entry, record in zip(document["records"], load_manifest(manifest)):
                entry["semantic_sha256"] = candidate_semantic_hash(record)
            atomic_json(manifest, document)
            result = run_study(path, root / "output")
            self.assertEqual(result["status"], "planned")
            self.assertTrue(result["is_synthetic"])

    def test_registered_ablation_configurations_can_be_planned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[1, 2])
            spec = json.loads(path.read_text())
            configurations = sorted((Path(__file__).resolve().parent.parent / "configs" / "ablations").glob("*.json"))
            self.assertGreaterEqual(len(configurations), 5)
            for config in configurations:
                with self.subTest(config=config.name):
                    atomic_json(path, {**spec, "config": str(config)})
                    result = run_study(path, root / config.stem)
                    self.assertEqual(result["status"], "planned")

    def test_rejects_output_containing_inputs_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                run_study(path, root)
            output = root / "output"
            output.mkdir()
            (output / "unrelated.txt").write_text("do not replace")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                run_study(path, output)
            self.assertEqual((output / "unrelated.txt").read_text(), "do not replace")

    def test_plan_then_execute_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[7, 11], comparisons=[{"metric": "AP", "repeats": 2}])
            output = root / "output"
            planned = run_study(path, output)
            fake = FakeCLI()
            with patch("dialdet.study.subprocess.run", side_effect=fake):
                result = run_study(path, output, execute=True)
                again = run_study(path, output, execute=True, resume=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["plan_sha256"], planned["plan_sha256"])
            self.assertEqual(result["tasks_completed"], 10)
            self.assertEqual(again["tasks_completed"], 10)
            self.assertEqual([argv[2] for argv in fake.calls],
                             ["make-crossfit", "pipeline", "pipeline", "merge-crossfit", "evaluate",
                              "pipeline", "pipeline", "merge-crossfit", "evaluate", "paired-statistics"])
            with self.assertRaisesRegex(ValueError, "use --resume"):
                run_study(path, output, execute=True)

    def test_resume_rejects_changed_artifact_and_dependency_record(self):
        for alteration in ("artifact", "dependency", "run_manifest", "extra_artifact"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = study_fixture(root, seeds=[1, 2])
                output = root / "output"
                fake = FakeCLI()
                with patch("dialdet.study.subprocess.run", side_effect=fake):
                    run_study(path, output, execute=True)
                if alteration == "artifact":
                    (output / "seed_1" / "union" / "predictions.jsonl").write_text("tampered")
                elif alteration == "dependency":
                    state_path = output / "STUDY_STATUS.json"
                    state = json.loads(state_path.read_text())
                    state["tasks"]["seed_1_fold_0"]["dependencies"]["crossfit"] = "0" * 64
                    atomic_json(state_path, state)
                elif alteration == "run_manifest":
                    record_path = output / "seed_1" / "evaluation" / "RUN_MANIFEST.json"
                    record = json.loads(record_path.read_text())
                    record["arguments"]["role"] = "dev"
                    atomic_json(record_path, record)
                else:
                    (output / "seed_1" / "evaluation" / "unrecorded.txt").write_text("unrecorded")
                with patch("dialdet.study.subprocess.run") as runner:
                    with self.assertRaisesRegex(ValueError, "Cannot trust"):
                        run_study(path, output, execute=True, resume=True)
                runner.assert_not_called()

    def test_plan_rejects_changed_spec_input_and_implementation(self):
        for alteration in ("spec", "annotations", "candidate", "implementation"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = study_fixture(root, seeds=[1, 2])
                output = root / "output"
                run_study(path, output)
                if alteration == "implementation":
                    with patch("dialdet.study.source_fingerprint", return_value={"sha256": "0" * 64, "files": {}}):
                        with self.assertRaisesRegex(ValueError, "source.*changed"):
                            run_study(path, output, execute=True)
                    continue
                changed = {"spec": path, "annotations": root / "data" / "annotations.json",
                           "candidate": root / "data" / "1.npz"}[alteration]
                with changed.open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaisesRegex(ValueError, "changed|hash mismatch"):
                    run_study(path, output, execute=True)

    def test_failed_partial_task_requires_relocation_then_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[1, 2])
            output = root / "output"
            fake = FakeCLI()
            fake.fail_once = "pipeline"
            with patch("dialdet.study.subprocess.run", side_effect=fake):
                with self.assertRaisesRegex(RuntimeError, "exited 7"):
                    run_study(path, output, execute=True)
                self.assertEqual(json.loads((output / "STUDY_STATUS.json").read_text())["status"], "failed")
                with self.assertRaisesRegex(ValueError, "Move that failed task directory"):
                    run_study(path, output, execute=True, resume=True)
                (output / "seed_1" / "fold_0").rename(root / "failed_fold_0")
                result = run_study(path, output, execute=True, resume=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(sum(argv[2] == "make-crossfit" for argv in fake.calls), 1)
            self.assertEqual(sum(argv[2] == "pipeline" for argv in fake.calls), 5)

    def test_installed_package_plan_uses_module_entry_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[1, 2])
            original = Path.is_file
            def is_file(file):
                return False if file.name == "run_dialdet_arrhenius.py" else original(file)
            with patch.object(Path, "is_file", is_file):
                run_study(path, root / "output")
            plan = json.loads((root / "output" / "STUDY_PLAN.json").read_text())
            self.assertEqual(plan["tasks"][0]["argv"][:4], [sys.executable, "-m", "dialdet", "make-crossfit"])

    def test_rejects_bad_model_config_and_output_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root)
            spec = json.loads(path.read_text())
            spec["config"] = "model.json"
            atomic_json(path, spec)
            atomic_json(root / "model.json", {"solver": {}, "model": {"unexpected": True}})
            with self.assertRaisesRegex(ValueError, "config.model"):
                run_study(path, root / "output")
            atomic_json(root / "model.json", {"solver": {}, "model": {"heads": 3}})
            with self.assertRaisesRegex(ValueError, "divisible"):
                run_study(path, root / "output")
            atomic_json(root / "model.json", {"solver": {}})
            output = root / "output"
            output.mkdir()
            (output / "specifications").symlink_to(root / "data", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                run_study(path, output)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("pycocotools"),
                         "install torch and pycocotools for the actual synthetic study integration")
    def test_real_outer_cli_two_seed_synthetic_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = study_fixture(root, seeds=[1, 2], comparisons=[
                {"metric": "AP", "repeats": 2}, {"metric": "mean_image_shortfall", "repeats": 2}])
            atomic_json(root / "tiny.json", {"solver": {}, "model": {
                "appearance_dim": 4, "class_embedding_dim": 2, "hidden_dim": 8, "heads": 2, "layers": 1}})
            spec = json.loads(path.read_text())
            spec.update(config="tiny.json", budgets=[100])
            atomic_json(path, spec)
            output = root / "output"
            arguments = ["study", "--specification", str(path), "--output", str(output)]
            transcript = io.StringIO()
            with redirect_stdout(transcript), redirect_stderr(transcript):
                self.assertEqual(main(arguments), 0, transcript.getvalue())
                code = main(arguments + ["--execute"])
            if code:
                logs = "\n".join(f"{p.name}\n{p.read_text()[-4000:]}" for p in (output / "logs").glob("*.log"))
                self.fail(transcript.getvalue() + "\n" + logs)
            from dialdet.artifacts import verify_run
            verified = verify_run(output, check_inputs=True)
            self.assertTrue(verified["valid"], verified["errors"])
            self.assertTrue(verified["synthetic"])
            status = json.loads((output / "STUDY_STATUS.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(len(status["tasks"]), 11)
            crossfit_run = json.loads((output / "folds" / "RUN_MANIFEST.json").read_text())
            self.assertTrue(crossfit_run["synthetic"])
            for fold in (0, 1):
                split = json.loads((output / "folds" / f"fold_{fold}.json").read_text())
                self.assertTrue(split["is_synthetic"])
            for seed in (1, 2):
                merged = json.loads((output / f"seed_{seed}" / "union" / "manifest.json").read_text())
                self.assertEqual(len(merged["records"]), 6)
                self.assertEqual({entry["role"] for entry in merged["records"]}, {"audit"})
            statistics = json.loads((output / "statistics" / "comparison_00" / "PAIRED_STATISTICS.json").read_text())
            self.assertEqual(statistics["encoder_seeds"], [1, 2])
            self.assertTrue(statistics["is_synthetic"])
            with patch("dialdet.study.subprocess.run") as runner, redirect_stdout(transcript), redirect_stderr(transcript):
                self.assertEqual(main(arguments + ["--execute", "--resume"]), 0, transcript.getvalue())
            runner.assert_not_called()
            verified = verify_run(output, check_inputs=True)
            self.assertTrue(verified["valid"], verified["errors"])


if __name__ == "__main__":
    unittest.main()
