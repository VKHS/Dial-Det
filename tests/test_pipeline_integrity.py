"""Actual neural inference must retain verifiable policy, data and file identities."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from dialdet.cli import make_parser, pipeline_command, evaluate_command, infer_command
from dialdet.data import atomic_json, load_manifest, load_record, save_record
from dialdet.demo import create_synthetic_dataset
from dialdet.provenance import canonical_sha256


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch runtime absent")
class PipelineIntegrityTests(unittest.TestCase):
    def test_real_neural_pipeline_binds_metadata_policies_and_evidence(self):
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            manifest = create_synthetic_dataset(root / "data", counts=(2, 1, 2))
            # Candidate-level synthetic provenance must survive a dataset-name
            # change and reach the pipeline's top-level run classification.
            document = json.loads(Path(manifest).read_text())
            document["dataset"] = "coco"
            atomic_json(manifest, document)
            args = make_parser().parse_args(["pipeline", "--manifest", str(manifest), "--output", str(root / "run"),
                                            "--epochs", "1", "--budgets", "2", "--quality-mode", "iou50"])
            result = pipeline_command(args)
            self.assertIs(result["is_synthetic"], True)
            self.assertEqual(result["evaluation"]["policy_binding"], "verified")
            self.assertEqual(len(result["evaluation"]["high_pressure"]["image_ids"]), 1)
            prediction_dir = root / "run" / "inference"
            prediction_file = prediction_dir / "predictions.jsonl"
            original_predictions = prediction_file.read_text()
            rows = [json.loads(line) for line in original_predictions.splitlines()]
            self.assertTrue(all(row["candidate_semantic_hash"] and row["policy_sha256"] for row in rows))
            evaluation = make_parser().parse_args(["evaluate", "--manifest", str(manifest),
                        "--predictions", str(prediction_dir), "--output", str(root / "rechecked")])
            dial_index = next(i for i,row in enumerate(rows) if row["method"] == "dialdet")
            self.assertNotEqual(rows[dial_index]["status"], "fallback")

            def write_predictions(changed):
                prediction_file.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n"
                                                    for row in changed))

            # A declared policy cannot bypass its evidence checks by omitting
            # artifact references. Check either omission and both together.
            for omitted in (("evidence",), ("certificate",), ("evidence", "certificate")):
                changed = copy.deepcopy(rows)
                for key in omitted:
                    changed[dial_index].pop(key)
                write_predictions(changed)
                with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "require evidence and certificate"):
                    evaluate_command(evaluation)
            prediction_file.write_text(original_predictions)

            # Keep the frozen box/score interface internally correct while
            # changing membership, so only certificate-to-selection binding
            # can reject this input.
            changed = copy.deepcopy(rows)
            changed_row = changed[dial_index]
            records = {r.image_id:r for r in load_manifest(manifest, "audit")}
            record = records[changed_row["image_id"]]
            selection = [] if changed_row["selected_indices"] else [0]
            changed_row.update(selected_indices=selection, selected_ids=record.ids[selection].tolist(),
                               boxes=record.boxes[selection].tolist(), classes=record.classes[selection].tolist(),
                               native_scores=record.scores[selection].tolist())
            write_predictions(changed)
            with self.assertRaisesRegex(ValueError, "Certificate does not bind the selected set"):
                evaluate_command(evaluation)
            prediction_file.write_text(original_predictions)

            wrong_iou = copy.deepcopy(evaluation)
            wrong_iou.iou = .75
            with self.assertRaisesRegex(ValueError, "matching IoU differs"):
                evaluate_command(wrong_iou)

            evidence = prediction_dir / rows[0]["evidence"]
            original = evidence.read_bytes()
            evidence.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "evidence"):
                evaluate_command(evaluation)
            evidence.write_bytes(original)

            # Reuse the trained checkpoint; this performs only one additional
            # forward pass on the existing development image, not retraining.
            dev_dir = root / "dev_inference"
            dev_args = make_parser().parse_args(["infer", "--manifest", str(manifest),
                        "--checkpoint", result["training"]["checkpoint"], "--role", "dev",
                        "--budgets", "2", "--output", str(dev_dir)])
            infer_command(dev_args)
            relabeled = json.loads(Path(manifest).read_text())
            relabeled["records"] = [{**entry, "role":"audit",
                                     "path":str((Path(manifest).parent / entry["path"]).resolve())}
                                    for entry in relabeled["records"] if entry["role"] == "dev"]
            relabeled_path = root / "dev_relabeled_as_audit.json"
            atomic_json(relabeled_path, relabeled)
            dev_evaluation = make_parser().parse_args(["evaluate", "--manifest", str(relabeled_path),
                            "--predictions", str(dev_dir), "--output", str(root / "bad_dev_audit")])
            with self.assertRaisesRegex(ValueError, "population role differs"):
                evaluate_command(dev_evaluation)

            # Even re-sealing the population's declared role must not bypass
            # the independent checkpoint fitting-ID check.
            dev_prediction_file = dev_dir / "predictions.jsonl"
            dev_rows = [json.loads(line) for line in dev_prediction_file.read_text().splitlines()]
            rebound = {}
            for row in dev_rows:
                policy_path = dev_dir / row["policy"]
                if policy_path not in rebound:
                    policy = json.loads(policy_path.read_text())
                    for entry in policy["population"]["records"]:
                        entry["role"] = "audit"
                    policy["population"]["population_sha256"] = canonical_sha256({
                        "schema":policy["population"]["schema"], "records":policy["population"]["records"]})
                    policy["binding_sha256"] = canonical_sha256({"policy_sha256":policy["policy_sha256"],
                        "population_sha256":policy["population"]["population_sha256"]})
                    atomic_json(policy_path, policy)
                    rebound[policy_path] = policy["binding_sha256"]
                row["policy_binding_sha256"] = rebound[policy_path]
            dev_prediction_file.write_text("".join(json.dumps(row)+"\n" for row in dev_rows))
            with self.assertRaisesRegex(ValueError, "overlaps.*fitting population"):
                evaluate_command(dev_evaluation)

            # Numerical arrays stay fixed; semantic export changes must still fail.
            doc = json.loads(Path(manifest).read_text())
            entry = next(e for e in doc["records"] if e["role"] == "audit")
            source = Path(manifest).parent / entry["path"]
            rec = load_record(source)
            rec.metadata["preprocessing"] = {"image_mean": [0., 0., 0.]}
            save_record(rec, source)
            entry.pop("sha256", None)
            atomic_json(manifest, doc)
            with self.assertRaisesRegex(ValueError, "semantic"):
                evaluate_command(evaluation)


if __name__ == "__main__":
    unittest.main()
