"""Complete-family identity tests; fixtures are not benchmark measurements."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dialdet.data import CandidateRecord, sha256_file
from dialdet.metrics import audit_record
from dialdet.provenance import (build_complete_policy, population_descriptor, replace_policy_solver,
                                seal_development_score, seal_policy_records)
from dialdet.risk import configuration_grid, evaluate_family
from dialdet.solver import SolverConfig


def make_candidate(image_id, role):
    return CandidateRecord(str(image_id), np.array([1]), np.array([[0., 0., 10., 10.]]),
                           np.array([1]), np.array([.9]), np.ones((1, 4), dtype=np.float32), (30, 30),
                           np.array([[0., 0., 10., 10.]]), np.array([1]), np.array([1]), np.array([False]),
                           {"role": role, "dataset": "unit_test_only", "checkpoint_sha256": "b" * 64,
                            "preprocessing": {"scale": 1.}, "max_candidates": 500})


class BoundRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = [make_candidate(f"cal-{i:04d}", "calibration") for i in range(500)]
        dev = [make_candidate("dev-1", "dev")]
        metadata = {"model_config": {"feature_dim": 4, "num_classes": 2, "rescue_k": 4},
                    "temperature": 1., "train_image_ids": ["train-1"], "dev_image_ids": ["dev-1"]}
        with tempfile.TemporaryDirectory() as tmp, patch("dialdet.artifacts.source_fingerprint",
                return_value={"sha256": "e" * 64}):
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"unit-test-metadata-fixture-not-a-trained-checkpoint")
            encoder_hash = sha256_file(checkpoint)
            base = build_complete_policy(records, checkpoint, metadata, SolverConfig(), 5, "coco")
        dev_population = population_descriptor(dev)
        entries = {item["image_id"]: item for item in base["population"]["records"]}
        cls.family, cls.development = [], []
        for index, point in enumerate(configuration_grid()):
            descriptor = replace_policy_solver(base, SolverConfig(**point))
            rows = []
            for record in records:
                row = audit_record(record, [0], 5, status="certified")
                entry = entries[record.image_id]
                row.update(policy_sha256=descriptor["policy_sha256"],
                           candidate_numerical_sha256=entry["numerical_sha256"],
                           candidate_semantic_sha256=entry["semantic_sha256"])
                rows.append(row)
            policy = {**point, "complete_policy": descriptor, "records": rows}
            policy["records_sha256"] = seal_policy_records(policy)
            cls.family.append(policy)
            cls.development.append(seal_development_score(descriptor, index / 100., dev_population, "official_coco_AP"))
        cls.provenance = {
            "complete_policy_including_fallback": True, "protocol": "independent_focal_detector",
            "iid_deployment_population": True, "encoder_fixed_before_calibration": True,
            "detector_unseen_calibration": True, "policy_family_fixed_before_calibration": True,
            "budget_fixed_before_calibration": True, "train_ids": ["train-1"], "dev_ids": ["dev-1"],
            "audit_ids": ["audit-1"], "encoder_sha256": encoder_hash, "detector_sha256": "b" * 64,
            "calibration_population_sha256": base["population"]["population_sha256"],
        }

    def inputs(self):
        return copy.deepcopy((self.family, self.development, self.provenance))

    def test_sealed_matching_family_preserves_24_policies_and_48_hypotheses(self):
        family, development, provenance = self.inputs()
        result = evaluate_family(family, development, provenance)
        self.assertEqual(result["n_feasible"], 24)
        self.assertEqual(result["selected"]["development_score"], .23)
        self.assertTrue(result["policy_binding"]["verified"])
        self.assertTrue(result["policy_binding"]["development_verified"])
        self.assertTrue(result["risk_certificate_eligible"])
        self.assertEqual(result["status"], "conditional_on_declared_provenance")
        self.assertTrue(all(item["waste"]["hypotheses"] == 48 for item in result["policies"]))

    def test_legacy_family_remains_bounds_only(self):
        family, development, provenance = self.inputs()
        for item in family:
            del item["complete_policy"]
            del item["records_sha256"]
        development = [{key: item[key] for key in ("rho", "lam", "score")} for item in development]
        result = evaluate_family(family, development, provenance)
        self.assertFalse(result["risk_certificate_eligible"])
        self.assertEqual(result["status"], "bounds_only_unverified_policy_binding")
        self.assertEqual(result["n_feasible"], 24)

    def test_modified_loss_and_duplicate_policy_are_rejected(self):
        family, development, provenance = self.inputs()
        family[0]["records"][0]["C"] = 0
        with self.assertRaisesRegex(ValueError, "records hash mismatch"):
            evaluate_family(family, development, provenance)
        family, development, provenance = self.inputs()
        family[-1] = family[0]
        with self.assertRaisesRegex(ValueError, "24-policy"):
            evaluate_family(family, development, provenance)

    def test_resealed_but_mixed_fallback_policy_is_rejected(self):
        family, development, provenance = self.inputs()
        last = family[-1]
        config = {**last["complete_policy"]["definition"]["solver"], "fallback_threshold": .9}
        last["complete_policy"] = replace_policy_solver(last["complete_policy"], config)
        for row in last["records"]:
            row["policy_sha256"] = last["complete_policy"]["policy_sha256"]
        last["records_sha256"] = seal_policy_records(last)
        with self.assertRaisesRegex(ValueError, "family mixes"):
            evaluate_family(family, development, provenance)

    def test_wrong_candidate_identity_even_after_resealing_is_rejected(self):
        family, development, provenance = self.inputs()
        family[0]["records"][0]["candidate_semantic_sha256"] = "0" * 64
        family[0]["records_sha256"] = seal_policy_records(family[0])
        with self.assertRaisesRegex(ValueError, "candidate or policy identity"):
            evaluate_family(family, development, provenance)

    def test_changed_development_score_and_wrong_checkpoint_declaration_are_rejected(self):
        family, development, provenance = self.inputs()
        development[0]["score"] = .8
        with self.assertRaisesRegex(ValueError, "Development score hash"):
            evaluate_family(family, development, provenance)
        family, development, provenance = self.inputs()
        provenance["encoder_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "encoder hash"):
            evaluate_family(family, development, provenance)

    def test_unsealed_development_scores_cannot_grant_certificate(self):
        family, development, provenance = self.inputs()
        development = [{key: item[key] for key in ("rho", "lam", "score")} for item in development]
        result = evaluate_family(family, development, provenance)
        self.assertTrue(result["policy_binding"]["verified"])
        self.assertFalse(result["policy_binding"]["development_verified"])
        self.assertFalse(result["risk_certificate_eligible"])

    def test_public_checkpoint_declarations_remain_ineligible_with_bound_records(self):
        family, development, provenance = self.inputs()
        provenance["protocol"] = "public_checkpoint_crossfit"
        result = evaluate_family(family, development, provenance)
        self.assertFalse(result["risk_certificate_eligible"])
        self.assertEqual(result["status"], "bounds_only_unverified_population")

    def test_nonfinite_and_fractional_counts_do_not_pass_resealed_records(self):
        for key, value in (("waste", float("nan")), ("C", .5)):
            family, development, provenance = self.inputs()
            # Legacy inputs still receive numerical validation.
            for item in family:
                del item["complete_policy"]
                del item["records_sha256"]
            development = [{key: item[key] for key in ("rho", "lam", "score")} for item in development]
            family[0]["records"][0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                evaluate_family(family, development, provenance)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime absent")
class DevelopmentFamilyIntegrationTests(unittest.TestCase):
    def test_actual_training_collection_and_development_scoring_preserve_data_roles(self):
        import torch
        from dialdet.data import atomic_json, load_manifest
        from dialdet.demo import create_synthetic_dataset
        from dialdet.development import score_policy_family
        from dialdet.risk_runtime import collect_policy_records
        from dialdet.training import train
        torch.set_num_threads(1)
        fixed_source = {"sha256": "e" * 64}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch("dialdet.artifacts.source_fingerprint", return_value=fixed_source), \
                patch("dialdet.development.source_fingerprint", return_value=fixed_source):
            root = Path(tmp)
            manifest = Path(create_synthetic_dataset(root / "data", counts=(2, 1, 2)))
            document = json.loads(manifest.read_text())
            for entry in document["records"]:
                if entry["role"] == "audit":
                    entry["role"] = "calibration"
            atomic_json(manifest, document)
            training = train(load_manifest(manifest, "train"), load_manifest(manifest, "dev"),
                             root / "training", epochs=1, quality_mode="iou50",
                             model_config={"appearance_dim": 16, "class_embedding_dim": 4,
                                           "hidden_dim": 32, "layers": 1, "heads": 4})
            collected = collect_policy_records(manifest, training["checkpoint"], root / "risk", budget=3)
            family = json.loads(Path(collected["policy_records"]).read_text())
            self.assertTrue(all("records_sha256" in policy for policy in family))
            # A development registry deliberately ignores calibration outcome fields.
            registry = copy.deepcopy(family)
            for policy in registry:
                policy["records"] = [{"this_is_not_a_development_input": 999}]
            atomic_json(root / "registry.json", registry)
            with self.assertRaisesRegex(ValueError, "official evaluator"):
                score_policy_family(manifest, training["checkpoint"], root / "registry.json", root / "blocked")
            result = score_policy_family(manifest, training["checkpoint"], root / "registry.json",
                                         root / "development", allow_matching_proxy=True)
            scores = json.loads(Path(result["development_scores"]).read_text())
            self.assertEqual(len(scores), 24)
            self.assertFalse(result["official_metric"])
            self.assertEqual(result["metric"], "matching_proxy_attainable_recall")
            dev_ids = {record.image_id for record in load_manifest(manifest, "dev")}
            self.assertEqual({item["image_id"] for item in result["development_population"]["records"]}, dev_ids)
            report = evaluate_family(family, scores, {"complete_policy_including_fallback": True,
                                                     "protocol": "synthetic"})
            self.assertTrue(report["policy_binding"]["verified"])
            self.assertTrue(report["policy_binding"]["development_verified"])
            self.assertFalse(report["policy_binding"]["official_development_metric"])
            self.assertFalse(report["risk_certificate_eligible"])


if __name__ == "__main__":
    unittest.main()
