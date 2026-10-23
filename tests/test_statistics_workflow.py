"""Pairing, resampling, and actual COCO dataset-AP regression checks."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dialdet.data import atomic_json, sha256_file
from dialdet.artifacts import write_run_manifest
from dialdet.statistics import crossed_bootstrap, holm
from dialdet.statistical_workflow import run_paired_statistics, _coco_pair, _source_synthetic


def row(image_id, method, covered, h=1):
    return {"image_id":str(image_id),"method":method,"K":2,"G":h,"H":h,"Q":h,"C":covered,
            "selected_count":covered,"shortfall":(h-covered)/h if h else 0.,"waste":0.,
            "allocation_shortfall":h-covered,"role":"audit","matching_iou":.5,
            "matching_convention":"maximum_class_consistent_nonignored"}


def write_spec(root, coco=False):
    runs = []
    for seed in (1,2):
        groups = {"dialdet_K2":[row(1,"dialdet",1),row(2,"dialdet",1)],
                  "local_K2":[row(1,"local",0),row(2,"local",0)]}
        audits = root/f"audit_{seed}.json"
        atomic_json(audits, groups)
        run = {"seed":seed,"audits":audits.name}
        if coco:
            predictions = root/f"predictions_{seed}.jsonl"
            exported = []
            for image_id in (1,2):
                for method in ("dialdet","local"):
                    selected = method == "dialdet"
                    exported.append({"image_id":str(image_id),"method":method,"K":2,
                                     "selected_ids":[1] if selected else [],
                                     "boxes":[[0,0,10,10]] if selected else [],
                                     "classes":[1] if selected else [],"native_scores":[.9] if selected else [],
                                     "candidate_export_hash":str(image_id)*64,"track":"membership_native_scores"})
            predictions.write_text("".join(json.dumps(r)+"\n" for r in exported))
            run["predictions"] = predictions.name
        runs.append(run)
    spec = {"schema_version":1,"method_a":"dialdet","method_b":"local","budget":2,
            "metric":"AP" if coco else "mean_image_shortfall","runs":runs,"repeats":8,"seed":17}
    if coco:
        annotations = root/"coco.json"
        atomic_json(annotations, {"info":{},"images":[{"id":i,"height":20,"width":20} for i in (1,2)],
                                  "categories":[{"id":1,"name":"object"}],
                                  "annotations":[{"id":i,"image_id":i,"category_id":1,"bbox":[0,0,10,10],
                                                  "area":100,"iscrowd":0} for i in (1,2)]})
        spec["coco_annotations"] = annotations.name
    source = root/"spec.json"
    atomic_json(source,spec)
    return source,spec


class CrossedBootstrapTests(unittest.TestCase):
    def test_images_and_seeds_are_shared_for_paired_contrast(self):
        data = np.arange(15,dtype=float).reshape(3,5)/20
        sampled = []
        def metric(images,seeds):
            sampled.append((images.copy(),seeds.copy()))
            a = float(data[np.ix_(seeds,images)].mean())
            return a,a-.25
        result = crossed_bootstrap(5,3,metric,repeats=100,seed=9)
        self.assertAlmostEqual(result["estimate"],.25)
        self.assertLess(result["bootstrap_standard_error"],1e-14)
        self.assertTrue(any(len(set(images))<len(images) for images,seeds in sampled[1:]))
        self.assertTrue(any(len(set(seeds))<len(seeds) for images,seeds in sampled[1:]))

    def test_observed_and_bootstrap_metrics_must_be_finite(self):
        with self.assertRaisesRegex(ValueError,"finite"):
            crossed_bootstrap(2,2,lambda i,s:(np.nan,0),repeats=10)
        for values in ((2,2,1),(True,2,10),(2.5,2,10)):
            with self.assertRaises(ValueError):
                crossed_bootstrap(values[0],values[1],lambda i,s:(0,0),repeats=values[2])
        with self.assertRaises(ValueError):
            holm([.1],alpha=np.nan)


class PairedWorkflowTests(unittest.TestCase):
    def test_diagnostics_recompute_paired_metrics_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,spec=write_spec(root)
            first=run_paired_statistics(path,root/"out")
            second=run_paired_statistics(path)
            self.assertEqual(first,second)
            self.assertEqual(first["estimate"],-1.)
            self.assertEqual(first["normal_95_interval"],[-1.,-1.])
            self.assertEqual(first["metric_kind"],"matching_diagnostic")
            self.assertEqual(first["encoder_seeds"],[1,2])
            self.assertIsNone(first["is_synthetic"])
            self.assertTrue((root/"out"/"PAIRED_STATISTICS.json").exists())
            spec["metric"]="pooled_shortfall"
            spec["subset"]="high_pressure"
            atomic_json(path,spec)
            result=run_paired_statistics(path)
            self.assertEqual(result["image_ids"],["1"])
            self.assertEqual(result["estimate"],-1.)

    def test_synthetic_classification_is_bound_and_uses_three_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,spec=write_spec(root,coco=True)
            spec["metric"]="mean_image_shortfall"
            second_prediction=spec["runs"][1].pop("predictions")
            evaluation=root/"EVALUATION.json"
            atomic_json(evaluation,{"is_synthetic":True,"prediction_sha256":sha256_file(root/spec["runs"][0]["predictions"])})
            atomic_json(path,spec)
            result=run_paired_statistics(path)
            self.assertTrue(result["is_synthetic"])
            self.assertEqual([r["is_synthetic"] for r in result["paired_inputs"]],[True,None])
            atomic_json(evaluation,{"is_synthetic":False,"prediction_sha256":sha256_file(root/spec["runs"][0]["predictions"])})
            self.assertIsNone(run_paired_statistics(path)["is_synthetic"])
            spec["runs"][1]["predictions"]=second_prediction
            atomic_json(path,spec)
            self.assertFalse(run_paired_statistics(path)["is_synthetic"])
            atomic_json(evaluation,{"is_synthetic":False,"prediction_sha256":"0"*64})
            self.assertIsNone(run_paired_statistics(path)["is_synthetic"])

    def test_verified_source_run_manifest_preserves_synthetic_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,spec=write_spec(root)
            write_run_manifest(root,"evaluate",{},synthetic=True)
            value,evidence=_source_synthetic(root/"audit_1.json",spec["runs"][0],root)
            self.assertTrue(value)
            self.assertEqual(evidence[0]["source"],"verified_run_manifest")
            (root/"audit_1.json").write_text("{}")
            value,evidence=_source_synthetic(root/"audit_1.json",spec["runs"][0],root)
            self.assertIsNone(value)

    def test_reject_seed_duplicates_partial_population_and_incompatible_matching(self):
        for mutation in ("seed","missing","duplicate","matching","nonfinite","counts"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                path,spec=write_spec(root)
                if mutation=="seed":
                    spec["runs"][1]["seed"]=1
                    atomic_json(path,spec)
                else:
                    target=root/"audit_2.json"
                    doc=json.loads(target.read_text())
                    if mutation=="missing":doc["local_K2"].pop()
                    if mutation=="duplicate":doc["local_K2"].append(copy.deepcopy(doc["local_K2"][0]))
                    if mutation=="matching":doc["local_K2"][0]["matching_iou"] = .6
                    if mutation=="nonfinite":doc["local_K2"][0]["shortfall"] = float("nan")
                    if mutation=="counts":doc["local_K2"][0]["H"] = 0
                    target.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):run_paired_statistics(path)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"),"optional COCO evaluator absent")
    def test_true_coco_AP_contrast_handles_empty_baseline(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            path,spec=write_spec(root,coco=True)
            result=run_paired_statistics(path)
            self.assertAlmostEqual(result["estimate"],1.)
            self.assertAlmostEqual(result["method_a_estimate"],1.)
            self.assertAlmostEqual(result["method_b_estimate"],0.)
            self.assertEqual(result["metric_kind"],"official_coco_dataset_AP")
            self.assertEqual(result["scale"],"fraction")

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"),"optional COCO evaluator absent")
    def test_AP_bootstrap_preserves_repeated_image_multiplicity(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            path,spec=write_spec(root,coco=True)
            p={"image_id":1,"category_id":1,"bbox":[0,0,10,10],"score":.9}
            a={"1":{"predictions":[p]},"2":{"predictions":[]}}
            b={"1":{"predictions":[]},"2":{"predictions":[]}}
            metric=_coco_pair(root/"coco.json",[(a,b),(a,b)],["1","2"],"AP")
            ordinary,_=metric(np.array([0,1]),np.array([0,1]))
            repeated,_=metric(np.array([0,0,1]),np.array([0,1]))
            self.assertAlmostEqual(ordinary,51/101)
            self.assertAlmostEqual(repeated,67/101)
            self.assertGreater(repeated,ordinary)


if __name__ == "__main__":
    unittest.main()
