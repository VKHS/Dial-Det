"""Execution-boundary tests for optional runtime adapters and data provenance."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from dialdet.cli import main, make_parser, train_command
from dialdet.data import atomic_json, make_crossfit_manifests, load_manifest, record_hash
from dialdet.demo import create_synthetic_dataset
from dialdet.metrics import official_coco_evaluation


class DatasetGuardTests(unittest.TestCase):
    def test_detection_dataset_mismatch_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            source = create_synthetic_dataset(Path(tmp) / "input")
            out = Path(tmp) / "output"
            rc = main(["train", "--output", str(out), "--manifest", source, "--dataset", "coco"])
            self.assertEqual(rc, 1)
            failure = json.loads((out / "FAILURE.json").read_text())
            self.assertIn("does not match", failure["message"])
            self.assertFalse((out / "last.pt").exists())

    def test_crossfit_union_contains_each_audit_image_once(self):
        from dialdet.crossfit import merge_runs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_synthetic_dataset(root / "data", counts=(3, 2, 3))
            folds = make_crossfit_manifests(source, root / "folds")
            directories = []
            for fold, manifest in enumerate(folds):
                out = root / f"predictions{fold}"
                out.mkdir()
                records = load_manifest(manifest, "audit")
                with (out / "predictions.jsonl").open("w") as stream:
                    for record in records:
                        stream.write(json.dumps({"image_id": record.image_id, "K": 2, "method": "nms",
                            "selected_indices": [], "selected_ids": [], "boxes": [], "classes": [],
                            "native_scores": [], "candidate_export_hash": record_hash(record),
                            "status": "nms_baseline"}) + "\n")
                directories.append(out)
            with redirect_stdout(io.StringIO()):
                rc = main(["merge-crossfit", "--manifests", *map(str, folds),
                           "--prediction-dirs", *map(str, directories),
                           "--output", str(root / "merged")])
            self.assertEqual(rc, 0)
            result = json.loads((root / "merged" / "MERGE.json").read_text())
            self.assertEqual(result["audited_images"], 8)
            self.assertEqual(len(load_manifest(result["manifest"], "audit")), 8)
            with self.assertRaises(ValueError):
                merge_runs([folds[0], folds[0]], directories, root / "bad")


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional official COCO evaluator absent")
class OfficialEvaluatorTests(unittest.TestCase):
    def test_perfect_and_empty_predictions_use_official_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "annotations.json"
            atomic_json(path, {"info": {}, "images": [{"id": 1, "height": 64, "width": 64}],
                "categories": [{"id": 1, "name": "synthetic_object"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0}]})
            predictions = [{"image_id": 1, "category_id": 1, "bbox": [5, 5, 20, 20], "score": .9}]
            perfect = official_coco_evaluation(path, predictions, [1], 1)
            empty = official_coco_evaluation(path, [], [1], 1)
            self.assertAlmostEqual(perfect["metrics"]["AP"], 1.)
            self.assertAlmostEqual(empty["metrics"]["AP"], 0.)
            self.assertEqual(perfect["cocoeval_per_category_maxDets"], [1, 10, 100])


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime absent")
class RiskRuntimeTests(unittest.TestCase):
    def test_cli_exponent_reaches_mass_training_and_family_keeps_all_images(self):
        import torch
        from dialdet.training import load_evidence_checkpoint
        from dialdet.risk_runtime import collect_policy_records
        from dialdet.solver import SolverConfig
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = Path(create_synthetic_dataset(root / "data", counts=(2, 1, 2)))
            doc = json.loads(manifest.read_text())
            for entry in doc["records"]:
                if entry["role"] == "audit":
                    entry["role"] = "calibration"
            atomic_json(manifest, doc)
            args = make_parser().parse_args(["train", "--manifest", str(manifest), "--output", str(root / "training"),
                                            "--epochs", "1", "--quality-mode", "iou50", "--p", "3"])
            result = train_command(args)
            model, metadata = load_evidence_checkpoint(result["checkpoint"])
            self.assertEqual(metadata["training_config"]["p"], 3.)
            family = collect_policy_records(manifest, result["checkpoint"], root / "risk", budget=3,
                                            config=SolverConfig(p=3.))
            records = json.loads(Path(family["policy_records"]).read_text())
            self.assertEqual(len(records), 24)
            self.assertTrue(all(len(p["records"]) == 2 for p in records))
            self.assertTrue(all(row["role"] == "calibration" and row["K"] == 3
                                for p in records for row in p["records"]))
            with self.assertRaises(ValueError):
                collect_policy_records(manifest, result["checkpoint"], root / "bad", config=SolverConfig(p=2.))


if __name__ == "__main__":
    unittest.main()
