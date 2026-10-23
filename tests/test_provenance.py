"""Provenance tests target scientific changes, relocation and input corruption."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dialdet.data import CandidateRecord, atomic_json, load_manifest, record_hash, save_record, sha256_file
from dialdet.provenance import (build_complete_policy, candidate_semantic_hash, canonical_sha256,
                                inspect_manifest, population_descriptor, replace_policy_solver,
                                verify_complete_policy)
from dialdet.solver import SolverConfig


def candidate(image_id="image-1", role="audit"):
    return CandidateRecord(
        image_id=image_id, ids=np.array([101]), boxes=np.array([[0., 0., 10., 10.]]),
        classes=np.array([1]), scores=np.array([.9]), features=np.ones((1, 4), dtype=np.float32),
        image_size=(50, 50), gt_boxes=np.array([[0., 0., 10., 10.]]), gt_classes=np.array([1]),
        gt_ids=np.array([1]), gt_ignore=np.array([False]),
        metadata={"role": role, "dataset": "generic", "checkpoint_sha256": "b" * 64,
                  "checkpoint": "/old/model.pt", "source_path": "/old/record.npz",
                  "preprocessing": {"min_size": 800, "max_size": 1333,
                                    "original_size": [50, 50], "transformed_size": [800, 800]},
                  "appearance_feature": "detector_roi", "max_candidates": 500})


def checkpoint_metadata():
    return {"model_config": {"feature_dim": 4, "num_classes": 2, "overlap_threshold": .25,
                             "rescue_k": 4, "distance_gate": 2., "rescue_same_class": True,
                             "use_appearance": True, "use_rescue": True},
            "temperature": 1.2, "train_image_ids": ["train-1"], "dev_image_ids": ["dev-1"]}


class ProvenanceTests(unittest.TestCase):
    def test_semantic_changes_have_new_identity_even_when_arrays_are_unchanged(self):
        original = candidate()
        changed = copy.deepcopy(original)
        changed.metadata["preprocessing"]["min_size"] = 640
        self.assertEqual(record_hash(original), record_hash(changed))
        self.assertNotEqual(candidate_semantic_hash(original), candidate_semantic_hash(changed))
        changed = copy.deepcopy(original)
        changed.metadata["checkpoint_sha256"] = "c" * 64
        self.assertNotEqual(candidate_semantic_hash(original), candidate_semantic_hash(changed))

    def test_relocation_role_and_manifest_protocol_do_not_change_frozen_stream(self):
        original = candidate()
        moved = copy.deepcopy(original)
        moved.metadata.update(role="calibration", source_path="/new/record.npz", checkpoint="/new/model.pt",
                              image_file="/new/image.jpg", protocol="public_checkpoint_crossfit")
        self.assertEqual(candidate_semantic_hash(original), candidate_semantic_hash(moved))
        self.assertNotEqual(population_descriptor([original])["population_sha256"],
                            population_descriptor([moved])["population_sha256"])

    def test_canonical_json_rejects_nonfinite_and_is_mapping_order_independent(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": [2, 3]}),
                         canonical_sha256({"b": [2, 3], "a": 1}))
        for bad in (np.nan, np.inf, {1: "ambiguous-key"}):
            with self.assertRaises(ValueError):
                canonical_sha256(bad)

    def test_same_policy_different_population_and_changed_solver_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch("dialdet.artifacts.source_fingerprint",
                return_value={"sha256": "e" * 64}):
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"unit-test-checkpoint-content")
            a = candidate("cal-1", "calibration")
            b = candidate("dev-1", "dev")
            b.metadata["preprocessing"]["original_size"] = [60, 60]
            b.metadata["preprocessing"]["transformed_size"] = [900, 900]
            first = build_complete_policy([a], checkpoint, checkpoint_metadata(), SolverConfig(), 100, "coco")
            second = build_complete_policy([b], checkpoint, checkpoint_metadata(), SolverConfig(), 100, "coco")
            self.assertEqual(first["policy_sha256"], second["policy_sha256"])
            self.assertNotEqual(first["binding_sha256"], second["binding_sha256"])
            self.assertTrue(verify_complete_policy(first))
            altered = replace_policy_solver(first, SolverConfig(fallback_threshold=.7))
            self.assertNotEqual(first["policy_sha256"], altered["policy_sha256"])
            checkpoint.write_bytes(b"changed-checkpoint")
            changed = build_complete_policy([a], checkpoint, checkpoint_metadata(), SolverConfig(), 100, "coco")
            self.assertNotEqual(first["policy_sha256"], changed["policy_sha256"])

    def test_identity_covers_actual_source_and_population_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            with patch("dialdet.artifacts.source_fingerprint", return_value={"sha256": "a" * 64}):
                original = build_complete_policy([candidate()], checkpoint, checkpoint_metadata(), SolverConfig(), 10, "coco")
            with patch("dialdet.artifacts.source_fingerprint", return_value={"sha256": "b" * 64}):
                edited_source = build_complete_policy([candidate()], checkpoint, checkpoint_metadata(), SolverConfig(), 10, "coco")
            self.assertNotEqual(original["policy_sha256"], edited_source["policy_sha256"])
            original["population"]["records"][0]["semantic_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Population hash"):
                verify_complete_policy(original)

    def test_preflight_detects_byte_changes_and_reports_unhashed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_record(candidate(), root / "record.npz")
            doc = {"schema_version": 1, "dataset": "generic", "records": [
                {"image_id": "image-1", "role": "audit", "path": path.name}]}
            manifest = root / "manifest.json"
            atomic_json(manifest, doc)
            report = inspect_manifest(manifest)
            self.assertTrue(report["valid"])
            self.assertFalse(report["all_declared_file_hashes_verified"])
            self.assertEqual(report["feature_dimensions"], [4])
            self.assertEqual(report["candidate_count_total"], 1)
            doc["records"][0]["sha256"] = sha256_file(path)
            atomic_json(manifest, doc)
            self.assertTrue(inspect_manifest(manifest)["all_declared_file_hashes_verified"])
            changed = candidate()
            changed.scores[0] = .2
            save_record(changed, path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                inspect_manifest(manifest)

    def test_semantic_manifest_binding_rejects_changed_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = candidate()
            save_record(original, root / "record.npz")
            doc = {"schema_version": 1, "dataset": "generic", "records": [
                {"image_id": original.image_id, "role": "audit", "path": "record.npz",
                 "semantic_sha256": candidate_semantic_hash(original)}]}
            atomic_json(root / "manifest.json", doc)
            self.assertEqual(len(load_manifest(root / "manifest.json")), 1)
            original.metadata["max_candidates"] = 300
            save_record(original, root / "record.npz")
            with self.assertRaisesRegex(ValueError, "semantic_sha256 mismatch"):
                load_manifest(root / "manifest.json")
            with self.assertRaisesRegex(ValueError, "semantic_sha256 mismatch"):
                inspect_manifest(root / "manifest.json")

    def test_preflight_checks_unselected_role_and_refuses_mixed_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [candidate("a", "train"), candidate("b", "dev")]
            records[1].features = np.ones((1, 8), dtype=np.float32)
            entries = []
            for rec in records:
                path = save_record(rec, root / f"{rec.image_id}.npz")
                entries.append({"image_id": rec.image_id, "role": rec.metadata["role"], "path": path.name,
                                "sha256": sha256_file(path)})
            manifest = root / "manifest.json"
            atomic_json(manifest, {"schema_version": 1, "dataset": "generic", "records": entries})
            with self.assertRaisesRegex(ValueError, "feature dimensions"):
                inspect_manifest(manifest)
            self.assertEqual(inspect_manifest(manifest, role="train")["n_images"], 1)
            (root / "b.npz").write_bytes(b"corrupted-unselected-dev-record")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                inspect_manifest(manifest, role="train")


if __name__ == "__main__":
    unittest.main()
