"""Native CrowdHuman interface and optional upstream metric integration tests."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dialdet.crowdhuman import (crowdhuman_predictions, write_crowdhuman_predictions,
                               official_crowdhuman_evaluation, _prepare_population)
from dialdet.data import CandidateRecord, record_hash


def candidate_record():
    return CandidateRecord(image_id="crowdhuman:val:a", ids=[20, 2, 10],
                           boxes=[[1, 2, 11, 22], [4, 5, 14, 25], [30, 0, 40, 10]],
                           classes=[1, 1, 1], scores=[.8, .8, .9],
                           features=np.ones((3, 4)), image_size=(100, 200),
                           metadata={"original_image_id": "a", "dataset": "crowdhuman"})


def ground_truth(identifier, ignored=True):
    gt = [{"tag": "person", "fbox": [0, 0, 10, 10], "hbox": [0, 0, 5, 5],
           "extra": {"ignore": 0}}]
    if ignored:
        gt.append({"tag": "mask", "fbox": [30, 0, 10, 10], "hbox": [30, 0, 5, 5],
                   "extra": {"ignore": 1}})
    return {"ID": identifier, "gtboxes": gt}


def predictions(identifier, detections):
    return {"ID": identifier, "width": 100, "height": 100,
            "dtboxes": [{"box": box, "score": score, "tag": 1} for box, score in detections]}


class CrowdHumanInterfaceTests(unittest.TestCase):
    def test_native_id_full_body_xywh_scores_and_deterministic_ties(self):
        record = candidate_record()
        before = record_hash(record)
        row = crowdhuman_predictions(record, [0, 2, 1])
        self.assertEqual(row["ID"], "a")
        self.assertEqual((row["height"], row["width"]), (100, 200))
        self.assertEqual([box["score"] for box in row["dtboxes"]], [.9, .8, .8])
        self.assertEqual(row["dtboxes"][1]["box"], [4., 5., 10., 20.])
        self.assertEqual(record_hash(record), before)
        self.assertEqual(crowdhuman_predictions(record, [])["dtboxes"], [])

    def test_missing_original_id_nonperson_and_bad_selection_rejected(self):
        record = candidate_record()
        del record.metadata["original_image_id"]
        with self.assertRaisesRegex(ValueError, "original_image_id"):
            crowdhuman_predictions(record, [0])
        record.metadata["original_image_id"] = "a"
        record.classes[0] = 2
        with self.assertRaisesRegex(ValueError, "person"):
            crowdhuman_predictions(record, [0])
        with self.assertRaises(ValueError):
            crowdhuman_predictions(record, [.5])

    def test_declared_native_person_class_is_preserved_and_serialized_as_person(self):
        record = candidate_record()
        record.classes[:] = 5
        record.metadata["person_class_id"] = 5
        before = record_hash(record)
        row = crowdhuman_predictions(record, [0, 2])
        self.assertEqual([detection["tag"] for detection in row["dtboxes"]], [1, 1])
        self.assertEqual(record_hash(record), before)
        np.testing.assert_array_equal(record.classes, [5, 5, 5])
        record.classes[1] = 1
        with self.assertRaisesRegex(ValueError, "declared native person"):
            crowdhuman_predictions(record, [1])
        record.metadata["person_class_id"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            crowdhuman_predictions(record, [])

    def test_population_filter_retains_source_tie_order_and_native_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_rows = [ground_truth("b"), ground_truth("a"), ground_truth("unused")]
            path = write_crowdhuman_predictions(Path(tmp) / "gt.odgt", source_rows)
            gt, dt = _prepare_population(path, [predictions("a", []), predictions("b", [])], ["a", "b"], 1)
            self.assertEqual([row["ID"] for row in gt], ["b", "a"])
            self.assertEqual([row["ID"] for row in dt], ["b", "a"])
            self.assertEqual(gt, source_rows[:2])

    def test_budget_population_and_explicit_empty_row_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_crowdhuman_predictions(Path(tmp) / "gt.odgt", [ground_truth("a"), ground_truth("b")])
            detections = [([0, 0, 10, 10], .9), ([30, 0, 10, 10], .8)]
            with self.assertRaisesRegex(ValueError, "at most K"):
                _prepare_population(path, [predictions("a", detections)], ["a"], 1)
            with self.assertRaisesRegex(ValueError, "explicit prediction"):
                _prepare_population(path, [predictions("a", [])], ["a", "b"], 1)
            with self.assertRaisesRegex(ValueError, "outside"):
                _prepare_population(path, [predictions("c", [])], ["a"], 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _prepare_population(path, [predictions("a", [])], ["a", "a"], 1)

    def test_full_body_upstream_requires_original_head_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = ground_truth("a")
            del row["gtboxes"][0]["hbox"]
            path = write_crowdhuman_predictions(Path(tmp) / "gt.odgt", [row])
            with self.assertRaisesRegex(ValueError, "hbox"):
                _prepare_population(path, [predictions("a", [])], ["a"], 1)


@unittest.skipUnless(os.environ.get("DIAL_CROWDHUMAN_EVALUATOR"),
                     "Set DIAL_CROWDHUMAN_EVALUATOR to a real upstream APMRToolkits checkout")
class CrowdHumanUpstreamTests(unittest.TestCase):
    def evaluate(self, ground_truth_rows, prediction_rows, budget=5):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_crowdhuman_predictions(Path(tmp) / "gt.odgt", ground_truth_rows)
            result = official_crowdhuman_evaluation(path, prediction_rows,
                [row["ID"] for row in ground_truth_rows], budget,
                os.environ["DIAL_CROWDHUMAN_EVALUATOR"], Path(tmp) / "metrics")
            saved = json.loads((Path(tmp) / "metrics" / "CROWDHUMAN_EVALUATION.json").read_text())
            self.assertEqual(result, saved)
            self.assertIn("database.py", result["evaluator_source_sha256"])
            return result

    def test_ignore_duplicate_and_dataset_score_order(self):
        gt = [ground_truth("a"), ground_truth("b", False)]
        rows = [predictions("a", [([0, 0, 10, 10], .9), ([0, 0, 10, 10], .8),
                                   ([30, 0, 10, 10], .99)]),
                predictions("b", [([0, 0, 10, 10], .7)])]
        result = self.evaluate(gt, rows)
        self.assertEqual((result["matched_detections"], result["false_positive_detections"],
                          result["ignored_detections"]), (2, 1, 1))
        # Upstream trapezoidal AP, without a fabricated PR origin, is 7/24.
        self.assertAlmostEqual(result["metrics"]["AP50"], 7 / 24)
        self.assertEqual(result["metrics"]["Recall"], 1.)
        moved = copy.deepcopy(rows)
        moved[1]["dtboxes"][0]["score"] = .95
        better = self.evaluate(gt, moved)
        self.assertAlmostEqual(better["metrics"]["AP50"], .5)
        self.assertGreater(better["metrics"]["AP50"], result["metrics"]["AP50"])

    def test_empty_report_convention_is_finite_and_recorded(self):
        result = self.evaluate([ground_truth("a")], [predictions("a", [])])
        self.assertEqual(result["metrics"], {"AP50": 0., "MR_minus2": 1., "Recall": 0.})
        self.assertTrue(any("Empty effective" in note for note in result["edge_case_notes"]))

    def test_nonempty_all_false_positive_report_has_miss_rate_one(self):
        result = self.evaluate([ground_truth("a", False)],
                               [predictions("a", [([60, 60, 10, 10], .9)])])
        self.assertEqual(result["metrics"], {"AP50": 0., "MR_minus2": 1., "Recall": 0.})
        self.assertEqual(result["false_positive_detections"], 1)
        self.assertEqual(result["edge_case_notes"], [])

    def test_positive_match_takes_precedence_over_overlapping_ignore(self):
        gt = ground_truth("a")
        gt["gtboxes"][1]["fbox"] = [0, 0, 20, 20]
        result = self.evaluate([gt], [predictions("a", [([0, 0, 10, 10], .9)])])
        self.assertEqual(result["matched_detections"], 1)
        self.assertEqual(result["ignored_detections"], 0)

    def test_empty_gt_image_is_retained_as_a_negative_image(self):
        gt = [ground_truth("a", False), {"ID": "negative", "gtboxes": []}]
        rows = [predictions("a", [([0, 0, 10, 10], .7)]),
                predictions("negative", [([0, 0, 10, 10], .9)])]
        result = self.evaluate(gt, rows)
        self.assertEqual(result["n_images"], 2)
        self.assertEqual(result["false_positive_detections"], 1)
        self.assertTrue(any("Empty-GT" in note for note in result["edge_case_notes"]))


if __name__ == "__main__":
    unittest.main()
