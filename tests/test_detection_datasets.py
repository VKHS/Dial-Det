"""Native pedestrian annotations and immutable detector export contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dialdet.detection_datasets import image_path, load_crowdhuman_annotations, load_citypersons_annotations
from dialdet.data import load_manifest

try:
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    HAS_VISION = True
except (ImportError, RuntimeError):
    HAS_VISION = False


def _body(tag="person", ignore=0, head_ignore=0):
    return {"tag": tag, "fbox": [-4, 2, 30, 70], "vbox": [4, 8, 15, 20],
            "hbox": [8, 3, 7, 9], "extra": {"ignore": ignore, "box_id": 12},
            "head_attr": {"ignore": head_ignore}}


def _write_citypersons(path, rows, split="val"):
    from scipy.io import savemat
    cells = np.empty((1, len(rows)), dtype=object)
    for index, (city, name, boxes) in enumerate(rows):
        record = np.empty((1, 1), dtype=[("cityname", "O"), ("im_name", "O"), ("bbs", "O")])
        record[0, 0] = (city, name, np.asarray(boxes, dtype=np.float64).reshape(-1, 10))
        cells[0, index] = record
    savemat(path, {f"anno_{split}_aligned": cells})


class NativeCityPersonsTests(unittest.TestCase):
    def test_native_mat_coordinates_and_explicit_origin_conversion(self):
        # Native class, x,y,w,h, source instance, visible x,y,w,h.
        boxes = [[1, 947, 406, 17, 40, 24000, 950, 407, 14, 39],
                 [0, 1839, 382, 31, 71, 0, 1839, 382, 31, 71]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anno_val.mat"
            _write_citypersons(path, [("frankfurt", "example_leftImg8bit.png", boxes)])
            rows, split = load_citypersons_annotations(path)
            shifted, _ = load_citypersons_annotations(path, coordinate_origin=1)
        self.assertEqual(split, "val")
        self.assertEqual(rows[0].file_name, "frankfurt/example_leftImg8bit.png")
        self.assertEqual(rows[0].ground_truth[0]["bbox"], [947., 406., 17., 40.])
        self.assertEqual(shifted[0].ground_truth[0]["bbox"], [946., 405., 17., 40.])
        self.assertEqual([r["ignore"] for r in rows[0].ground_truth], [False, True])
        self.assertEqual(rows[0].metadata["source_annotation_attributes"][0]["source_instance_id"], 24000)
        self.assertFalse(rows[0].metadata["official_miss_rate_evaluator_applied"])

    def test_reasonable_subset_keeps_ignored_regions_and_source_class_semantics(self):
        boxes = [[1, 2, 3, 20, 60, 1, 2, 3, 20, 60],
                 [1, 2, 3, 20, 40, 2, 2, 3, 20, 40],
                 [1, 2, 3, 20, 60, 3, 2, 3, 10, 30],
                 [2, 2, 3, 20, 60, 4, 2, 3, 20, 60],
                 [5, 2, 3, 20, 60, 0, 2, 3, 20, 60]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anno_train.mat"
            _write_citypersons(path, [("a", "one.png", boxes), ("b", "two.png", [])], "train")
            all_rows, _ = load_citypersons_annotations(path)
            reasonable, _ = load_citypersons_annotations(path, subset="reasonable", person_class_id=2)
            with self.assertRaisesRegex(ValueError, "one matching"):
                load_citypersons_annotations(path, dataset_split="val")
        self.assertEqual([r["ignore"] for r in all_rows[0].ground_truth], [False, False, False, True, True])
        self.assertEqual([r["ignore"] for r in reasonable[0].ground_truth], [False, True, True, True, True])
        self.assertTrue(all(r["category_id"] == 2 for r in reasonable[0].ground_truth))
        self.assertEqual(reasonable[1].ground_truth, [])
        self.assertIn("visible/full area<0.65", reasonable[0].metadata["gt_ignore_convention"])

    def test_invalid_mat_labels_or_duplicate_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anno_val.mat"
            row = ("a", "one.png", [[7, 2, 3, 20, 60, 1, 2, 3, 20, 60]])
            _write_citypersons(path, [row])
            with self.assertRaisesRegex(ValueError, "labels"):
                load_citypersons_annotations(path)
            row = ("a", "one.png", [[1, 2, 3, 20, 60, 1, 2, 3, 20, 60]])
            _write_citypersons(path, [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_citypersons_annotations(path)


class NativeCrowdHumanTests(unittest.TestCase):
    def test_full_body_ignore_is_independent_of_head_and_visible_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation_val.odgt"
            path.write_text(json.dumps({"ID": "image_one", "gtboxes": [
                _body(head_ignore=1), _body(ignore=1), _body(tag="mask")
            ]}) + "\n")
            rows, split = load_crowdhuman_annotations(path)
        self.assertEqual(split, "val")
        self.assertEqual(rows[0].image_id, "crowdhuman:val:image_one")
        self.assertEqual(rows[0].original_image_id, "image_one")
        self.assertEqual(rows[0].file_name, "image_one.jpg")
        self.assertEqual([r["ignore"] for r in rows[0].ground_truth], [False, True, True])
        self.assertEqual(rows[0].ground_truth[0]["bbox"], [-4., 2., 30., 70.])
        self.assertEqual([r["id"] for r in rows[0].ground_truth], [0, 1, 2])
        self.assertFalse(rows[0].metadata["full_body_boxes_clipped"])

    def test_split_namespace_and_empty_ground_truth_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.odgt"
            path.write_text(json.dumps({"ID": "same", "gtboxes": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "dataset_split"):
                load_crowdhuman_annotations(path)
            trains, _ = load_crowdhuman_annotations(path, "train")
            vals, _ = load_crowdhuman_annotations(path, "val")
        self.assertNotEqual(trains[0].image_id, vals[0].image_id)
        self.assertEqual(trains[0].ground_truth, [])

    def test_duplicate_images_unknown_body_flags_and_degenerate_boxes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation_train.odgt"
            source = {"ID": "one", "gtboxes": [_body()]}
            path.write_text((json.dumps(source) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_crowdhuman_annotations(path)
            source["gtboxes"][0]["extra"]["ignore"] = "0"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "extra.ignore"):
                load_crowdhuman_annotations(path)
            source["gtboxes"][0]["extra"]["ignore"] = 0
            source["gtboxes"][0]["fbox"][2] = 0
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "positive size"):
                load_crowdhuman_annotations(path)

    def test_image_paths_cannot_escape_root_and_no_implicit_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.jpg"
            outside.write_bytes(b"test")
            images = root / "images"
            images.mkdir()
            with self.assertRaisesRegex(ValueError, "relative"):
                image_path(images, "../outside.jpg")
            (images / "symlink.jpg").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                image_path(images, "symlink.jpg")
            path = root / "annotation_test.odgt"
            path.write_text(json.dumps({"ID": "one"}))
            with self.assertRaisesRegex(ValueError, "labeled gtboxes"):
                load_crowdhuman_annotations(path)


@unittest.skipUnless(HAS_VISION, "install matching torch/torchvision to exercise the detector exporter")
class PedestrianExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(19)
        cls.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=3,
                                          min_size=64, max_size=64, rpn_pre_nms_top_n_test=30,
                                          rpn_post_nms_top_n_test=12).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.set_num_threads(cls.previous_threads)

    def test_person_filter_precedes_cap_and_preserves_source_scores_ids_features(self):
        from dialdet.exporter import export_image
        image = torch.rand(3, 64, 64)
        full = export_image(self.model, image, "one", score_floor=0, max_candidates=500)
        chosen_class = 2
        expected = np.flatnonzero(full.classes == chosen_class)[:4]
        person = export_image(self.model, image, "one", score_floor=0, max_candidates=4,
                               category_ids=[chosen_class])
        self.assertEqual(len(person.ids), 4)
        np.testing.assert_array_equal(person.ids, full.ids[expected])
        np.testing.assert_array_equal(person.scores, full.scores[expected])
        np.testing.assert_array_equal(person.boxes, full.boxes[expected])
        np.testing.assert_array_equal(person.features, full.features[expected])
        np.testing.assert_array_equal(person.classes, np.full(4, chosen_class))
        self.assertTrue(np.all(person.ids % 3 == chosen_class))
        with self.assertRaisesRegex(ValueError, "category_ids"):
            export_image(self.model, image, "one", category_ids=[3])

    def test_native_odgt_reaches_frozen_manifest_without_gt_clipping(self):
        from PIL import Image
        from dialdet.exporter import export_crowdhuman
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            images = root / "Images"
            images.mkdir()
            Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(images / "one.jpg")
            annotations = root / "annotation_val.odgt"
            annotations.write_text(json.dumps({"ID": "one", "gtboxes": [_body(head_ignore=1), _body(tag="mask")]}) + "\n")
            with patch("dialdet.exporter.build_detector", return_value=(self.model, {"validation_only": True})) as build:
                result = export_crowdhuman(images, annotations, root / "exports", checkpoint=root / "validation-only.pt",
                                          num_classes=3, person_class_id=2, score_floor=0, max_candidates=4)
            self.assertEqual(build.call_args.kwargs["num_classes"], 3)
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["dataset"], "crowdhuman")
            self.assertEqual(manifest["quality_mode"], "iou50")
            self.assertEqual(manifest["num_classes"], 3)
            self.assertFalse(manifest["risk_eligible"])
            record = load_manifest(result["manifest"])[0]
            self.assertEqual(record.image_id, "crowdhuman:val:one")
            self.assertEqual(record.metadata["original_image_id"], "one")
            np.testing.assert_array_equal(record.gt_boxes[0], [-4., 2., 26., 72.])
            np.testing.assert_array_equal(record.gt_ignore, [False, True])
            np.testing.assert_array_equal(record.classes, np.full(4, 2))
            np.testing.assert_array_equal(record.gt_classes, np.full(2, 2))
            with self.assertRaises(FileExistsError):
                export_crowdhuman(images, annotations, root / "exports", checkpoint="none", num_classes=3)
            with self.assertRaisesRegex(ValueError, "local frozen detector"):
                export_crowdhuman(images, annotations, root / "other")

    def test_citypersons_mat_reaches_person_only_frozen_manifest(self):
        from PIL import Image
        from dialdet.exporter import export_citypersons
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            images = root / "leftImg8bit" / "val"
            (images / "a").mkdir(parents=True)
            Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(images / "a" / "one.png")
            annotations = root / "anno_val.mat"
            _write_citypersons(annotations, [("a", "one.png", [[1, 1, 2, 20, 60, 9, 1, 2, 20, 60]])])
            with patch("dialdet.exporter.build_detector", return_value=(self.model, {"validation_only": True})):
                result = export_citypersons(images, annotations, root / "exports", checkpoint="validation-only.pt",
                                            num_classes=3, score_floor=0, max_candidates=4, subset="reasonable")
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["dataset"], "citypersons")
            self.assertEqual(manifest["citypersons_subset"], "reasonable")
            self.assertFalse(manifest["risk_eligible"])
            record = load_manifest(result["manifest"])[0]
            self.assertEqual(record.image_id, "citypersons:val:a:one.png")
            np.testing.assert_array_equal(record.gt_boxes[0], [1., 2., 21., 62.])
            np.testing.assert_array_equal(record.classes, np.ones(4, dtype=np.int64))
            self.assertFalse(record.gt_ignore[0])


if __name__ == "__main__":
    unittest.main()
