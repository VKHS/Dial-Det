"""Scientific-interface invariants, independent of PyTorch and COCOeval binaries."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dialdet.data import (
    CandidateRecord, assignment_targets, atomic_json, bbox_iou, load_manifest,
    load_record, make_crossfit_manifests, manifest_document, record_hash,
    save_record, sha256_file,
)
from dialdet.metrics import (
    audit_record, maximum_matching, official_coco_evaluation,
    score_order_matching, summarize_audits,
)
from dialdet.risk import configuration_grid, empirical_bernstein, evaluate_family


def record(image_id="101", boxes=None, gt_boxes=None, classes=None,
           gt_classes=None, ignore=None, scores=None, gt_ids=None):
    boxes = np.asarray([[0, 0, 10, 10]] if boxes is None else boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray([[0, 0, 10, 10]] if gt_boxes is None else gt_boxes, dtype=float).reshape(-1, 4)
    n, m = len(boxes), len(gt_boxes)
    return CandidateRecord(
        image_id=str(image_id), ids=np.arange(100, 100 + n), boxes=boxes,
        classes=np.ones(n, dtype=int) if classes is None else classes,
        scores=np.linspace(.9, .5, n) if scores is None else scores,
        features=np.ones((n, 4), dtype=np.float32), image_size=(100, 100),
        gt_boxes=gt_boxes,
        gt_classes=np.ones(m, dtype=int) if gt_classes is None else gt_classes,
        gt_ids=np.arange(m) if gt_ids is None else gt_ids,
        gt_ignore=np.zeros(m, dtype=bool) if ignore is None else ignore,
        metadata={"detector": "test-only", "role": "audit"},
    )


class FrozenDataTests(unittest.TestCase):
    def test_npz_roundtrip_preserves_exact_interface_and_hash(self):
        original = record(boxes=[[0, 0, 10, 10], [5, 6, 15, 18]])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_record(original, Path(tmp) / "frozen.npz")
            recovered = load_record(path)
            self.assertEqual(record_hash(original), record_hash(recovered))
            self.assertEqual(original.metadata, recovered.metadata)
            for key in ("ids", "boxes", "classes", "scores", "features", "gt_boxes", "gt_classes", "gt_ids", "gt_ignore"):
                np.testing.assert_array_equal(getattr(original, key), getattr(recovered, key))
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))

    def test_hash_detects_native_interface_changes(self):
        original = record()
        initial = record_hash(original)
        for key in ("ids", "boxes", "classes", "scores", "features"):
            changed = copy.deepcopy(original)
            getattr(changed, key).flat[0] += .01 if key in ("boxes", "scores", "features") else 10
            self.assertNotEqual(initial, record_hash(changed), key)

    def test_read_only_targets_and_audits_preserve_frozen_values(self):
        original = record()
        before = record_hash(original)
        assignment_targets(original)
        audit_record(original, [0], budget=1)
        maximum_matching(original)
        self.assertEqual(before, record_hash(original))

    def test_manifest_rejects_duplicate_image_or_path_across_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_record(record("101"), root / "a.npz")
            save_record(record("102"), root / "b.npz")
            base = {"schema_version": 1, "dataset": "synthetic_validation", "records": [
                {"image_id": "101", "path": "a.npz", "role": "train"},
                {"image_id": "102", "path": "b.npz", "role": "dev"}]}
            for field, replacement in (("image_id", "101"), ("path", "a.npz")):
                changed = copy.deepcopy(base)
                changed["records"][1][field] = replacement
                atomic_json(root / "manifest.json", changed)
                with self.assertRaises(ValueError):
                    manifest_document(root / "manifest.json")

    def test_manifest_verifies_file_hash_and_record_image_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_record(record("101"), root / "a.npz")
            entry = {"image_id": "101", "path": "a.npz", "role": "train", "sha256": sha256_file(path)}
            doc = {"schema_version": 1, "dataset": "test", "records": [entry]}
            atomic_json(root / "manifest.json", doc)
            self.assertEqual(load_manifest(root / "manifest.json", "train")[0].image_id, "101")
            save_record(record("101", scores=[.2]), path)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_manifest(root / "manifest.json", "train")
            del entry["sha256"]
            entry["image_id"] = "102"
            atomic_json(root / "manifest.json", doc)
            with self.assertRaisesRegex(ValueError, "image-ID"):
                load_manifest(root / "manifest.json", "train")

    def test_public_crossfit_audits_each_image_once_and_disallows_risk_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for i in range(8):
                path = save_record(record(i), root / f"{i}.npz")
                entries.append({"image_id": str(i), "path": path.name, "role": "pool", "sha256": sha256_file(path)})
            source = root / "pool.json"
            atomic_json(source, {"schema_version": 1, "dataset": "test", "records": entries})
            outputs = make_crossfit_manifests(source, root / "folds", seed=4)
            audited = []
            for output in outputs:
                doc = manifest_document(output)
                self.assertFalse(doc["risk_eligible"])
                self.assertEqual(doc["protocol"], "public_checkpoint_crossfit")
                population = {role: {r.image_id for r in load_manifest(output, role)}
                              for role in ("train", "dev", "audit")}
                self.assertFalse(population["train"] & population["dev"])
                self.assertFalse((population["train"] | population["dev"]) & population["audit"])
                audited.extend(population["audit"])
            self.assertEqual(sorted(audited), [str(i) for i in range(8)])

    def test_zero_candidate_targets_matching_and_roundtrip(self):
        empty = record(boxes=[])
        result = assignment_targets(empty)
        self.assertEqual(result["assignments"].shape, (0,))
        self.assertEqual(result["quality_targets"].shape, (0, 10))
        self.assertEqual(result["valid_candidates"].shape, (0,))
        self.assertEqual(maximum_matching(empty), {})
        row = audit_record(empty, [], budget=5)
        self.assertEqual((row["G"], row["H"], row["Q"], row["C"]), (1, 0, 0, 0))
        self.assertEqual(row["shortfall"], 0.)
        self.assertEqual(row["waste"], 0.)
        self.assertEqual(row["candidate_absence"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(record_hash(empty), record_hash(load_record(save_record(empty, Path(tmp) / "empty.npz"))))
        completely_empty = record(boxes=[], gt_boxes=[])
        self.assertEqual(audit_record(completely_empty, [], 1)["G"], 0)
        self.assertEqual(bbox_iou([], []).shape, (0, 0))


class MatchingTests(unittest.TestCase):
    def test_augmenting_path_recovers_match_greedy_misses(self):
        # The high-score first candidate prefers A but can also match B.
        # The second candidate can only match A. Maximum matching reallocates.
        rec = record(boxes=[[1, 0, 11, 10], [0, 0, 8, 10]],
                     gt_boxes=[[0, 0, 10, 10], [4, 0, 14, 10]], scores=[.9, .8])
        self.assertEqual(len(score_order_matching(rec, [0, 1])), 1)
        self.assertEqual(len(maximum_matching(rec, [0, 1])), 2)
        row = audit_record(rec, [0, 1], 2)
        self.assertEqual(row["greedy_assignment_loss"], 1)
        self.assertEqual(row["allocation_shortfall"], 0)
        self.assertEqual(sum(row[k] for k in ("candidate_absence", "capacity_loss", "allocation_shortfall", "greedy_assignment_loss")), row["G"] - row["C_score_greedy"])

    def test_ignore_only_candidates_mask_training_but_consume_report_slots(self):
        rec = record(boxes=[[0, 0, 10, 10], [20, 0, 30, 10], [20, 0, 30, 10], [40, 0, 50, 10]],
                     classes=[1, 1, 2, 1], gt_boxes=[[0, 0, 10, 10], [0, 0, 30, 10]],
                     gt_classes=[1, 1], ignore=[False, True])
        targets = assignment_targets(rec)
        np.testing.assert_array_equal(targets["valid_candidates"], [True, False, True, True])
        self.assertEqual(targets["assignments"][0], 0)
        self.assertEqual(len(set(targets["assignments"][1:])), 3)
        self.assertTrue(np.all(targets["assignments"][1:] < 0))
        row = audit_record(rec, [0, 1, 2, 3], 4)
        self.assertEqual((row["G"], row["H"], row["C"]), (1, 1, 1))
        self.assertAlmostEqual(row["waste"], .75)

    def test_annotation_ties_use_smallest_gt_id_and_classes_must_match(self):
        rec = record(gt_boxes=[[0, 0, 10, 10], [0, 0, 10, 10]], gt_ids=[50, 10])
        self.assertEqual(assignment_targets(rec)["assignments"].tolist(), [1])
        np.testing.assert_array_equal(assignment_targets(rec)["quality_targets"], np.ones((1, 10)))
        rec = record(classes=[2], gt_classes=[1])
        self.assertEqual(maximum_matching(rec), {})
        self.assertEqual(assignment_targets(rec)["quality_targets"].sum(), 0.)

    def test_reject_duplicate_out_of_range_fractional_and_nested_selection(self):
        rec = record(boxes=[[0, 0, 10, 10], [2, 2, 8, 8]])
        for bad in ([0, 0], [-1], [2], [.5], [[0]]):
            with self.subTest(selected=bad), self.assertRaises((ValueError, TypeError)):
                maximum_matching(rec, bad)
        with self.assertRaises(ValueError):
            audit_record(rec, [0, 1], 1)

    def test_summaries_distinguish_image_and_object_weighting_and_do_not_invent_ap(self):
        first = audit_record(record(), [], 1)
        second = dict(first, image_id="102", Q=9, C=9, allocation_shortfall=0,
                      shortfall=0., waste=0., full_report=True)
        summary = summarize_audits([first, second])
        self.assertAlmostEqual(summary["pooled_shortfall"], .1)
        self.assertAlmostEqual(summary["mean_image_shortfall"], .5)
        self.assertIsNone(summary["ap"])

    def test_coco_wrapper_rejects_excess_global_budget_before_engine_import(self):
        prediction = {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .8}
        with self.assertRaises(ValueError):
            official_coco_evaluation("unused.json", [prediction, dict(prediction)], [1], global_budget=1)
        with self.assertRaises(ValueError):
            official_coco_evaluation("unused.json", [prediction], [2], global_budget=1)


def risk_inputs(n=500):
    rows = [{"image_id": f"cal-{i}", "role": "calibration", "K": 5, "Q": 3, "C": 3,
             "H": 3, "G": 3, "selected_count": 3, "waste": 0., "shortfall": 0.,
             "status": "fallback" if i == 0 else "certified"} for i in range(n)]
    grid = configuration_grid()
    family = [{**point, "records": copy.deepcopy(rows)} for point in grid]
    development = [{**point, "score": float(i)} for i, point in enumerate(grid)]
    provenance = {"complete_policy_including_fallback": True, "protocol": "independent_focal_detector",
                  "iid_deployment_population": True, "encoder_fixed_before_calibration": True,
                  "detector_unseen_calibration": True, "policy_family_fixed_before_calibration": True,
                  "budget_fixed_before_calibration": True, "train_ids": ["train-1"],
                  "dev_ids": ["dev-1"], "audit_ids": ["audit-1"],
                  "encoder_sha256": "a" * 64, "detector_sha256": "b" * 64}
    return family, development, provenance


class RiskTests(unittest.TestCase):
    def test_empirical_bernstein_uses_unbiased_variance_and_48_hypotheses(self):
        losses = np.asarray([.1, .3, .7, .9])
        got = empirical_bernstein(losses)
        h = math.log(1920)
        variance = float(np.sum((losses - losses.mean()) ** 2) / 3)
        expected = float(losses.mean()) + math.sqrt(2 * variance * h / 4) + 7 * h / 9
        self.assertAlmostEqual(got["sample_variance"], variance)
        self.assertAlmostEqual(got["raw_upper"], expected)
        self.assertEqual(got["hypotheses"], 48)
        self.assertEqual(got["upper"], 1.)
        self.assertGreater(got["raw_upper"], empirical_bernstein(losses, hypotheses=24)["raw_upper"])
        for bad in ([0.], [0., np.nan], [-.1, .5], [.1, 1.1]):
            with self.assertRaises(ValueError):
                empirical_bernstein(bad)

    def test_legacy_family_is_bounds_only_and_uses_dev_ranking(self):
        family, development, provenance = risk_inputs()
        result = evaluate_family(family, development, provenance)
        self.assertEqual(len(result["policies"]), 24)
        self.assertFalse(result["risk_certificate_eligible"])
        self.assertEqual(result["status"], "bounds_only_unverified_policy_binding")
        self.assertEqual(result["selected"]["development_score"], 23.)
        self.assertTrue(all(p["waste"]["hypotheses"] == 48 for p in result["policies"]))
        # The fallback row is included in the same n=500 population.
        self.assertTrue(all(p["waste"]["n"] == 500 for p in result["policies"]))

    def test_public_checkpoint_protocol_never_claims_risk_certification(self):
        family, development, provenance = risk_inputs()
        provenance["protocol"] = "public_checkpoint_crossfit"
        result = evaluate_family(family, development, provenance)
        self.assertFalse(result["risk_certificate_eligible"])
        self.assertEqual(result["status"], "bounds_only_unverified_policy_binding")

    def test_reject_leakage_incomplete_family_duplicate_dev_and_missing_fallback(self):
        family, development, provenance = risk_inputs()
        with self.assertRaises(ValueError):
            evaluate_family(family[:-1], development, provenance)
        with self.assertRaises(ValueError):
            evaluate_family(family, development + [development[0]], provenance)
        provenance["train_ids"] = ["cal-0"]
        with self.assertRaises(ValueError):
            evaluate_family(family, development, provenance)
        provenance["train_ids"] = []
        provenance["complete_policy_including_fallback"] = False
        with self.assertRaises(ValueError):
            evaluate_family(family, development, provenance)

    def test_reject_changed_calibration_population_budget_or_attainability(self):
        for change in ("population", "budget", "attainability"):
            family, development, provenance = risk_inputs()
            rows = family[-1]["records"]
            if change == "population":
                rows[0]["image_id"] = "different-image"
            elif change == "budget":
                rows[0]["K"] = 6
            else:
                rows[0].update(Q=2, C=2, selected_count=2)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_family(family, development, provenance)

    def test_empty_provenance_hashes_are_ineligible(self):
        family, development, provenance = risk_inputs()
        provenance["encoder_sha256"] = None
        provenance["detector_sha256"] = ""
        try:
            result = evaluate_family(family, development, provenance)
        except ValueError:
            return
        self.assertFalse(result["risk_certificate_eligible"])


if __name__ == "__main__":
    unittest.main()
