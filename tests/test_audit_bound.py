"""Retrospective matching-bound checks; no annotation reaches the allocator."""
import copy
import unittest

import numpy as np

from dialdet.audit_bound import matching_miss_bound
from dialdet.data import CandidateRecord, record_hash
from dialdet.solver import allocate, SolverConfig


def make_record(boxes, gt_boxes, *, ids=None, gt_ids=None):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    n, m = len(boxes), len(gt_boxes)
    return CandidateRecord(image_id="audit-test", ids=np.arange(n) if ids is None else np.asarray(ids),
                           boxes=boxes, classes=np.zeros(n, dtype=np.int64), scores=np.full(n, .8),
                           features=np.zeros((n, 4), dtype=np.float32), image_size=(100, 100),
                           gt_boxes=gt_boxes, gt_classes=np.zeros(m, dtype=np.int64),
                           gt_ids=np.arange(m) if gt_ids is None else np.asarray(gt_ids),
                           gt_ignore=np.zeros(m, dtype=bool), metadata={"role": "audit"})


def solve(record, q, r, k, config=None, mass=None):
    return allocate(q, r, k, candidate_ids=record.ids, config=config, mass=mass,
                    image_id=record.image_id, candidate_export_hash=record_hash(record))


class MatchingBoundTests(unittest.TestCase):
    def test_ideal_reference_has_zero_miss_bound(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
        record = make_record([gt[0], gt[0], gt[1], gt[1], gt[2], [70, 70, 80, 80]], gt)
        labels = np.array([0, 0, 1, 1, 2, 3])
        q = np.array([.9, .8, .85, .7, .65, 0.])
        r = (labels[:, None] == labels[None, :]).astype(float)
        result = solve(record, q, r, 3)
        audit = matching_miss_bound(record, result["selected"], q, r, 3, result)
        self.assertEqual(audit["status"], "checked")
        self.assertEqual(audit["bound"], 0)
        self.assertTrue(audit["nonvacuous"])
        self.assertEqual(audit["Qpositive"], 3)
        self.assertEqual(audit["background_private_block_count"], 1)
        self.assertEqual(audit["realized_misses"], 0)

    def test_price_loss_and_mass_error(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        record = make_record([gt[0], gt[0], gt[1]], gt)
        r = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        q = np.array([.01, .01, .01])
        result = solve(record, q, r, 2)
        audit = matching_miss_bound(record, result["selected"], q, r, 2, result)
        self.assertEqual(audit["Qpositive"], 0)
        self.assertEqual(audit["price_induced_loss"], 2)
        self.assertEqual(audit["bound"], 2)
        q = np.array([.9, .8, .85])
        mass = 2*(r*r).sum(axis=0)
        result = solve(record, q, r, 2, mass=mass)
        audit = matching_miss_bound(record, result["selected"], q, r, 2, result, mass=mass)
        self.assertEqual(audit["status"], "checked")
        self.assertGreater(audit["d_reference_upper"]["value"], .1)
        self.assertGreaterEqual(audit["bound"], audit["realized_misses"])

    def test_fallback_and_evidence_mismatch_are_vacuous(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        record = make_record(gt, gt)
        q, r = np.array([.9, .8]), np.eye(2)
        result = solve(record, q, r, 2)
        missing = matching_miss_bound(record, result["selected"], q, r, 2, None)
        self.assertEqual(missing["bound"], 2)
        self.assertFalse(missing["certificate_valid"])
        wrong_q = matching_miss_bound(record, result["selected"], q*.9, r, 2, result)
        self.assertEqual(wrong_q["status"], "vacuous")
        self.assertIn("differ", wrong_q["reason"])
        changed = copy.deepcopy(record)
        changed.features[0, 0] = 1.
        wrong_export = matching_miss_bound(changed, result["selected"], q, r, 2, result)
        self.assertIn("hash", wrong_export["reason"])
        tampered = copy.deepcopy(result["certificate"])
        tampered["upper_grid"] += 1
        rejected = matching_miss_bound(record, result["selected"], q, r, 2, tampered)
        self.assertEqual(rejected["status"], "vacuous")

    def test_fragmentation_bound_covers_real_duplicate_loss(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
        record = make_record([gt[0], gt[0], gt[1]], gt)
        q, r = np.array([.95, .9, .8]), np.eye(3)
        result = solve(record, q, r, 2)
        self.assertEqual(result["selected"], [0, 1])
        audit = matching_miss_bound(record, result["selected"], q, r, 2, result)
        self.assertEqual(audit["realized_misses"], 1)
        self.assertGreaterEqual(audit["bound"], 1)

    def test_random_ambiguous_matching_bound_dominates_realized_loss(self):
        rng = np.random.default_rng(4072)
        gt = np.array([[0, 0, 10, 10], [3, 0, 13, 10], [30, 0, 40, 10]], dtype=float)
        for repetition in range(15):
            n = 5+repetition % 4
            targets = rng.integers(0, 3, n)
            boxes = gt[targets].copy()
            jitter = rng.uniform(-1., 1., (n, 2))
            boxes[:, :2] += jitter
            boxes[:, 2:] += jitter
            record = make_record(boxes, gt, gt_ids=[30, 10, 20])
            q = rng.uniform(.05, .99, n)
            a = rng.random((n, n)); r = (a+a.T)/2
            np.fill_diagonal(r, 1.)
            k = 1+repetition % 4
            result = solve(record, q, r, k)
            audit = matching_miss_bound(record, result["selected"], q, r, k, result)
            self.assertEqual(audit["status"], "checked")
            self.assertGreaterEqual(audit["bound"], audit["realized_misses"])
            self.assertLessEqual(audit["bound"], audit["Q"])
            self.assertEqual(audit["reference_block_count"], audit["H"])

    def test_empty_population_and_zero_budget(self):
        record = make_record([], [])
        q, r = np.empty(0), np.empty((0, 0))
        result = solve(record, q, r, 0)
        audit = matching_miss_bound(record, [], q, r, 0, result)
        self.assertEqual(audit["bound"], 0)
        self.assertEqual(audit["Qpositive"], 0)
        self.assertFalse(audit["nonvacuous"])


if __name__ == "__main__":
    unittest.main()
