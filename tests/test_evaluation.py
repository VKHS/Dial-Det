"""Checks for evaluation populations, matching metrics and baseline safeguards."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dialdet.baselines import dpp_kernel, dpp_map, local_rule
from dialdet.data import CandidateRecord, atomic_json
from dialdet.metrics import (audit_record, high_pressure_ids, maximum_matching,
                             official_coco_evaluation, summarize_high_pressure)


def audit(image_id, h, k=10, c=None, method="dialdet"):
    c = min(k, h) if c is None else c
    q = min(k, h)
    return {"image_id": str(image_id), "method": method, "status": "certified", "K": k,
            "G": h, "H": h, "Q": q, "C": c, "N": h, "selected_count": c,
            "shortfall": (q-c)/q if q else 0., "waste": 0., "allocation_shortfall": q-c,
            "full_report": c == k, "capacity_binding": h > k, "role": "audit",
            "matching_iou": .5, "matching_convention": "maximum_class_consistent_nonignored"}


class HighPressureTests(unittest.TestCase):
    def test_subset_uses_attainable_H_and_stable_ids(self):
        rows = [audit(str(i), i % 5) for i in range(10)]
        self.assertEqual(high_pressure_ids(rows), ["4", "9"])
        self.assertEqual(high_pressure_ids(list(reversed(rows))), ["4", "9"])
        rows[0]["G"] = 10000
        self.assertEqual(high_pressure_ids(rows), ["4", "9"])
        self.assertEqual(high_pressure_ids([audit("b", 2), audit("a", 2)]), ["a"])

    def test_one_fixed_population_across_methods_and_budgets(self):
        first = [audit(i, i) for i in range(10)]
        other = [audit(i, i, k=3, method="local") for i in range(10)]
        report = summarize_high_pressure({"dialdet_K10": first, "local_K3": other})
        self.assertEqual(report["image_ids"], ["9", "8"])
        self.assertEqual(report["results"]["local_K3"]["n_images"], 2)
        changed = copy.deepcopy(other)
        changed[0]["H"] = 1
        with self.assertRaisesRegex(ValueError, "identical"):
            summarize_high_pressure({"a": first, "b": changed})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            high_pressure_ids([*first, first[0]])

    def test_matching_threshold_is_validated(self):
        record = CandidateRecord("1", np.array([1]), np.array([[0,0,10,10]]),
                                 np.array([1]), np.array([.9]), np.ones((1,2)), (20,20))
        for threshold in (0, -.1, 1.1, np.nan, np.inf, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                maximum_matching(record, iou_threshold=threshold)
        with self.assertRaises(ValueError):
            audit_record(record, [0], True)


class DPPTests(unittest.TestCase):
    def test_projection_handles_indefinite_symmetric_affinities(self):
        affinity = np.array([[1,.9,.9],[.9,1,0],[.9,0,1]])
        self.assertLess(np.linalg.eigvalsh(affinity).min(), 0)
        kernel = dpp_kernel([.4,.6,.8], affinity, scale=2., ridge=1e-4)
        self.assertGreater(np.linalg.eigvalsh(kernel).min(), 0)
        np.testing.assert_allclose(np.diag(kernel), 2*(np.array([.4,.6,.8])+1e-8)**2)

    def test_at_most_includes_empty_set_and_scale_changes_admission(self):
        self.assertEqual(dpp_map([.5,.5], np.eye(2), 10), [])
        self.assertEqual(dpp_map([.5,.5], np.eye(2), 10, scale=10), [0,1])
        self.assertEqual(dpp_map([.5,.5], np.eye(2), 1, [20,10], scale=10), [1])
        self.assertEqual(dpp_map([], np.empty((0,0)), 10), [])

    def test_bad_probabilities_and_caps_are_rejected(self):
        for q in ([np.nan], [1.5], [-1.]):
            with self.assertRaises(ValueError):
                dpp_map(q, [[1]], 1)
        for budget in (-1, .5, True):
            with self.assertRaises(ValueError):
                local_rule([.8], [[1]], budget)
        self.assertEqual(local_rule([.8], [[1]], 100), [0])


class COCOValidationTests(unittest.TestCase):
    def test_reject_invalid_input_before_import_or_annotation_read(self):
        pred = {"image_id":1,"category_id":1,"bbox":[0,0,10,10],"score":.8}
        for ids in ([], [1,1], [1.5], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                official_coco_evaluation("unused", [pred], ids, 1)
        for field, value in (("bbox", [0,0,0,5]), ("bbox", [0,0,np.nan,5]),
                             ("score", np.inf), ("category_id", 1.5)):
            changed = {**pred, field:value}
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                official_coco_evaluation("unused", [changed], [1], 1)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional COCO evaluator absent")
    def test_annotation_population_category_and_empty_output(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp)/"annotations.json"
            atomic_json(path, {"info":{},"images":[{"id":1,"height":20,"width":20}],
                "categories":[{"id":1,"name":"object"}],
                "annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[0,0,10,10],"area":100,"iscrowd":0}]})
            with self.assertRaisesRegex(ValueError,"absent"):
                official_coco_evaluation(path, [], [2], 10)
            with self.assertRaisesRegex(ValueError,"category"):
                official_coco_evaluation(path,[{"image_id":1,"category_id":2,"bbox":[0,0,10,10],"score":.5}],[1],10)
            predictions=[{"image_id":1,"category_id":1,"bbox":[0,0,10,10],"score":.9}]
            frozen=copy.deepcopy(predictions)
            self.assertAlmostEqual(official_coco_evaluation(path,predictions,[1],10)["metrics"]["AP"],1.)
            self.assertEqual(predictions,frozen)
            empty = official_coco_evaluation(path, [], [1], 10)
            self.assertEqual(empty["metrics"]["AP"], 0.)
            self.assertEqual(empty["cocoeval_per_category_maxDets"], [1,10,100])


if __name__ == "__main__":
    unittest.main()
