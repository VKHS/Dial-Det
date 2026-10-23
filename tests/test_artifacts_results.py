"""Run integrity and result claims must fail on changed or incomplete evidence."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from dialdet.artifacts import relative_artifact, verify_run, write_run_manifest
from dialdet.data import atomic_json, sha256_file
from dialdet.results import compare_results


class ArtifactTests(unittest.TestCase):
    def test_unknown_inputs_remain_unknown_and_synthetic_labels_take_precedence(self):
        from dialdet.cli import classify_synthetic
        from types import SimpleNamespace
        self.assertIsNone(classify_synthetic({"dataset": "generic"}))
        self.assertIsNone(classify_synthetic({"dataset": "custom"}, [SimpleNamespace(metadata={})]))
        self.assertIs(classify_synthetic({"dataset": "coco"}), False)
        self.assertIs(classify_synthetic({"dataset": "coco", "protocol": "synthetic_interface_test"}), True)
        self.assertIs(classify_synthetic({"dataset": "coco"}, [SimpleNamespace(metadata={"synthetic": True})]), True)
        self.assertIs(classify_synthetic({"dataset": "generic"}, [SimpleNamespace(metadata={"synthetic": False})]), False)

    def test_move_is_portable_but_output_and_external_input_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            source.write_text('{"input":1}')
            run = root / "run"
            atomic_json(run / "metrics.json", {"AP": .5})
            write_run_manifest(run, "evaluate", {"dataset": "coco"}, {"input": source})
            moved = root / "moved"
            run.rename(moved)
            self.assertTrue(verify_run(moved, check_inputs=True)["valid"])
            source.write_text('{"input":2}')
            self.assertFalse(verify_run(moved, check_inputs=True)["valid"])
            self.assertTrue(verify_run(moved)["valid"])
            atomic_json(moved / "metrics.json", {"AP": .9})
            self.assertFalse(verify_run(moved)["valid"])

    def test_failure_invalidates_previous_completed_record_and_retry_clears_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "metrics.json", {"AP": .5})
            write_run_manifest(root, "evaluate", {})
            atomic_json(root / "FAILURE.json", {"message": "later failed execution"})
            self.assertFalse(verify_run(root)["valid"])
            write_run_manifest(root, "evaluate", {})
            self.assertTrue(verify_run(root)["valid"])
            atomic_json(root / "RUN_STATUS.json", {"status": "running"})
            self.assertFalse(verify_run(root)["valid"])

    def test_path_escape_and_manifest_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "metrics.json", {"AP": .5})
            document = write_run_manifest(root, "evaluate", {})
            for bad in ("../metrics.json", str(root / "metrics.json")):
                with self.assertRaises(ValueError):
                    relative_artifact(root, bad)
            document["command"] = "different command"
            atomic_json(root / "RUN_MANIFEST.json", document)
            self.assertFalse(verify_run(root)["valid"])

    def test_verify_command_cannot_overwrite_the_run_being_verified(self):
        from dialdet.cli import main
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            root = Path(tmp)
            atomic_json(root / "metrics.json", {"AP": .5})
            write_run_manifest(root, "evaluate", {})
            before = sha256_file(root / "RUN_MANIFEST.json")
            self.assertEqual(main(["verify-run", "--run", tmp, "--output", tmp]), 2)
            self.assertTrue(verify_run(root)["valid"])
            self.assertEqual(before, sha256_file(root / "RUN_MANIFEST.json"))


class ResultComparisonTests(unittest.TestCase):
    def _files(self, root, synthetic=False):
        run = root / "run"
        atomic_json(run / "metrics.json", {"AP": .52})
        write_run_manifest(run, "evaluate", {}, synthetic=synthetic)
        context = {"dataset": "coco", "split": "val2017", "detector": "fixed-checkpoint-a",
                   "metric": "AP", "unit": "fraction", "budget": 100,
                   "track": "membership_native_scores", "protocol": "public_checkpoint_crossfit_union"}
        reference = {"schema": "dialdet-results-v1", "kind": "reported",
                     "source": {"title": "Synthetic fixture manuscript", "version": "fixture", "document_sha256": "a"*64},
                     "results": [{"result_id": "one", "value": .5, "context": context,
                                  "provenance_status": "reported_in_article", "paper_location": "fixture table"}]}
        generated = {"schema": "dialdet-results-v1", "kind": "generated", "results": [
            {"result_id": "one", "value": .52, "context": context,
             "provenance_status": "generated_by_local_run", "evidence": {
                 "run_manifest": "run/RUN_MANIFEST.json", "artifact": "metrics.json",
                 "artifact_sha256": sha256_file(run / "metrics.json"), "json_pointer": "/AP"}}]}
        ref, gen = root / "reference.json", root / "generated.json"
        atomic_json(ref, reference)
        atomic_json(gen, generated)
        return ref, gen, reference, generated

    def test_comparison_reads_and_verifies_actual_value_and_rejects_context_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, gen, reference, generated = self._files(root)
            report = compare_results(gen, ref)
            self.assertEqual(report["compared"], 1)
            self.assertAlmostEqual(report["comparisons"][0]["difference"], .02)
            generated["results"][0]["context"]["unit"] = "percentage_points"
            atomic_json(gen, generated)
            with self.assertRaisesRegex(ValueError, "context"):
                compare_results(gen, ref)

    def test_unknown_duplicate_nan_and_unsupported_values_never_succeed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, gen, reference, generated = self._files(root)
            changes = [lambda d: d["results"][0].update(result_id="unknown"),
                       lambda d: d["results"].append(copy.deepcopy(d["results"][0])),
                       lambda d: d["results"][0].update(value=float("nan")),
                       lambda d: d["results"][0].update(value=.99)]
            for change in changes:
                altered = copy.deepcopy(generated)
                change(altered)
                gen.write_text(json.dumps(altered))
                with self.assertRaises(ValueError):
                    compare_results(gen, ref)

    def test_partial_comparison_is_explicit_and_synthetic_is_not_benchmark_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, gen, reference, generated = self._files(root)
            second = copy.deepcopy(reference["results"][0])
            second["result_id"] = "two"
            reference["results"].append(second)
            atomic_json(ref, reference)
            with self.assertRaises(ValueError):
                compare_results(gen, ref)
            report = compare_results(gen, ref, allow_partial=True)
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing"], ["two"])
            ref, gen, _, _ = self._files(root, synthetic=True)
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                compare_results(gen, ref)


if __name__ == "__main__":
    unittest.main()
