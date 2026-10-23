"""Dataset routing and benchmark population checks for the detection workflows."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dialdet.cli import (make_parser, resolve_quality_mode, validate_dataset_arguments,
                         evaluate_command, train_command, pipeline_command)
from dialdet.data import CandidateRecord, atomic_json, save_record, load_record, record_hash, sha256_file
from dialdet.demo import create_synthetic_dataset


def crowd_fixture(root):
    entries, predictions, annotations = [], [], []
    for index in range(2):
        original_id = f"image-{index}"
        record = CandidateRecord(image_id=f"crowdhuman:val:{original_id}", ids=[10],
                    boxes=[[10, 20, 30, 60]], classes=[1], scores=[.9 - .1 * index],
                    features=np.ones((1, 4)), image_size=(100, 100),
                    gt_boxes=[[10, 20, 30, 60]], gt_classes=[1], gt_ids=[0], gt_ignore=[False],
                    metadata={"original_image_id": original_id, "dataset": "crowdhuman", "person_class_id": 1})
        path = save_record(record, root / f"record-{index}.npz")
        entries.append({"image_id": record.image_id, "path": str(path), "role": "audit", "sha256": sha256_file(path)})
        # Keep one image explicitly empty; it must stay in the benchmark denominator.
        selected = [0] if index == 0 else []
        predictions.append({"image_id": record.image_id, "K": 1, "method": "nms", "status": "nms_baseline",
                    "candidate_export_hash": record_hash(record), "selected_indices": selected,
                    "selected_ids": record.ids[selected].tolist(), "boxes": record.boxes[selected].tolist(),
                    "classes": record.classes[selected].tolist(), "native_scores": record.scores[selected].tolist()})
        annotations.append({"ID": original_id, "gtboxes": [{"tag": "person", "fbox": [10, 20, 20, 40],
                    "hbox": [10, 20, 10, 10], "extra": {"ignore": 0}}]})
    manifest = root / "manifest.json"
    atomic_json(manifest, {"schema_version": 1, "dataset": "crowdhuman", "num_classes": 2,
                          "quality_mode": "iou50", "protocol": "synthetic_interface_test", "records": entries})
    prediction_file, annotation_file = root / "predictions.jsonl", root / "annotation_val.odgt"
    prediction_file.write_text("".join(json.dumps(row) + "\n" for row in predictions))
    annotation_file.write_text("".join(json.dumps(row) + "\n" for row in annotations))
    return manifest, prediction_file, annotation_file


class DetectionWorkflowTests(unittest.TestCase):
    def test_auto_quality_matches_each_papers_dataset_and_explicit_ablation(self):
        from argparse import Namespace
        for dataset, mode in (("coco", "coco"), ("crowdhuman", "iou50"), ("citypersons", "iou50")):
            self.assertEqual(resolve_quality_mode(Namespace(quality_mode="auto"), {"dataset": dataset}), mode)
        self.assertEqual(resolve_quality_mode(Namespace(quality_mode="auto"),
                         {"dataset": "synthetic", "quality_mode": "iou50"}), "iou50")
        self.assertEqual(resolve_quality_mode(Namespace(quality_mode="coco"), {"dataset": "crowdhuman"}), "coco")

    def test_dataset_and_evaluator_mismatch_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, prediction_file, _ = crowd_fixture(root)
            args = make_parser().parse_args(["evaluate", "--manifest", str(manifest), "--dataset", "coco",
                       "--predictions", str(prediction_file), "--output", str(root / "out")])
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_dataset_arguments(args)
            args.dataset = "crowdhuman"
            args.coco_annotations = "incorrect.json"
            with self.assertRaisesRegex(ValueError, "COCOeval"):
                validate_dataset_arguments(args)
            args.coco_annotations = None
            args.crowdhuman_evaluator = tmp
            with self.assertRaisesRegex(ValueError, "requires --crowdhuman-annotations"):
                validate_dataset_arguments(args)

    def test_native_crowdhuman_predictions_export_without_inventing_ap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, prediction_file, _ = crowd_fixture(root)
            args = make_parser().parse_args(["evaluate", "--manifest", str(manifest),
                       "--predictions", str(prediction_file), "--output", str(root / "out")])
            result = evaluate_command(args)
            group = result["results"]["nms_K1"]
            self.assertNotIn("official_crowdhuman", group)
            self.assertNotIn("official_coco", group)
            self.assertIsNone(group["ap"])
            rows = [json.loads(line) for line in Path(group["crowdhuman_predictions"]).read_text().splitlines()]
            self.assertEqual([row["ID"] for row in rows], ["image-0", "image-1"])
            self.assertEqual(rows[0]["dtboxes"][0]["score"], .9)
            self.assertEqual(rows[1]["dtboxes"], [])

    @unittest.skipUnless(os.environ.get("DIAL_CROWDHUMAN_EVALUATOR"), "optional official CrowdHuman checkout absent")
    def test_official_evaluator_receives_original_ids_and_empty_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, prediction_file, annotations = crowd_fixture(root)
            args = make_parser().parse_args(["evaluate", "--manifest", str(manifest), "--dataset", "crowdhuman",
                       "--predictions", str(prediction_file), "--output", str(root / "out"),
                       "--crowdhuman-annotations", str(annotations), "--crowdhuman-evaluator",
                       os.environ["DIAL_CROWDHUMAN_EVALUATOR"]])
            validate_dataset_arguments(args)
            official = evaluate_command(args)["results"]["nms_K1"]["official_crowdhuman"]
            self.assertEqual(official["population_ids"], ["image-0", "image-1"])
            self.assertEqual(official["n_images"], 2)
            self.assertEqual(official["metrics"]["Recall"], .5)
            self.assertEqual(official["matched_detections"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime absent")
    def test_auto_quality_and_declared_label_table_reach_real_training(self):
        from dialdet.training import load_evidence_checkpoint
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = Path(create_synthetic_dataset(root / "data", counts=(2, 1, 1)))
            doc = json.loads(manifest.read_text())
            doc.update(dataset="crowdhuman", num_classes=91)
            atomic_json(manifest, doc)
            args = make_parser().parse_args(["train", "--manifest", str(manifest), "--output", str(root / "train"),
                                           "--epochs", "1", "--dataset", "crowdhuman"])
            validate_dataset_arguments(args)
            trained = train_command(args)
            model, metadata = load_evidence_checkpoint(trained["checkpoint"])
            self.assertEqual(metadata["training_config"]["quality_mode"], "iou50")
            self.assertEqual(model.config.num_classes, 91)

    @unittest.skipUnless(importlib.util.find_spec("torch") and os.environ.get("DIAL_CROWDHUMAN_EVALUATOR"),
                         "PyTorch and the optional official CrowdHuman checkout are required")
    def test_crowdhuman_pipeline_uses_separate_development_and_audit_annotations(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest, _, audit_annotations = crowd_fixture(root)
            document = json.loads(manifest.read_text())
            template = load_record(root / "record-0.npz")
            development_rows = []
            for role in ("train", "dev"):
                original_id = f"fitting-{role}"
                record = replace(template, image_id=f"crowdhuman:train:{original_id}",
                                 metadata={**template.metadata, "original_image_id": original_id})
                path = save_record(record, root / f"{role}.npz")
                document["records"].append({"image_id": record.image_id, "path": str(path),
                                            "role": role, "sha256": sha256_file(path)})
                development_rows.append({"ID": original_id, "gtboxes": [{"tag": "person",
                    "fbox": [10, 20, 20, 40], "hbox": [10, 20, 10, 10], "extra": {"ignore": 0}}]})
            atomic_json(manifest, document)
            development_annotations = root / "annotation_train.odgt"
            development_annotations.write_text("".join(json.dumps(row) + "\n" for row in development_rows))
            args = make_parser().parse_args(["pipeline", "--manifest", str(manifest), "--output", str(root / "run"),
                "--dataset", "crowdhuman", "--epochs", "1", "--budgets", "1",
                "--crowdhuman-annotations", str(audit_annotations), "--development-annotations", str(development_annotations),
                "--crowdhuman-evaluator", os.environ["DIAL_CROWDHUMAN_EVALUATOR"]])
            validate_dataset_arguments(args)
            result = pipeline_command(args)
            self.assertEqual(result["tuning"]["annotations_sha256"], sha256_file(development_annotations))
            self.assertEqual(result["tuning"]["development_image_ids"], ["crowdhuman:train:fitting-dev"])
            self.assertEqual(result["inference"]["quality_mode"], "iou50")
            official = result["evaluation"]["results"]["dialdet_K1"]["official_crowdhuman"]
            self.assertEqual(official["annotation_sha256"], sha256_file(audit_annotations))
            self.assertEqual(official["population_ids"], ["image-0", "image-1"])


if __name__ == "__main__":
    unittest.main()
