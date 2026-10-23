"""Executable paired inference from complete, measured evaluation populations.

Matching diagnostics use per_image_audits.json. COCO AP uses the original
predictions and official COCOeval, with a fresh image ID for every resampled
occurrence. Dataset AP is recomputed per encoder seed, then averaged over the
sampled seeds. Per-image AP is never invented or averaged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path

import numpy as np

from .data import atomic_json, sha256_file
from .metrics import high_pressure_ids, _coco_integer
from .statistics import crossed_bootstrap

DIAGNOSTICS = {"mean_image_shortfall", "mean_image_waste", "pooled_shortfall"}
COCO_METRICS = {"AP", "AP50", "AP75"}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _audit_group(document, key, budget):
    if key not in document or not isinstance(document[key], list) or not document[key]:
        raise ValueError(f"Missing or empty audit group {key}")
    rows = {}
    for row in document[key]:
        image_id = str(row["image_id"])
        if image_id in rows:
            raise ValueError(f"Duplicate image ID in {key}: {image_id}")
        if row.get("method") != key.rsplit("_K", 1)[0] or row.get("K") != budget:
            raise ValueError("Audit method/budget disagrees with its group")
        counts = {name: _integer(row[name], name) for name in ("G", "H", "Q", "C", "selected_count")}
        g, h, q, c, selected = (counts[k] for k in ("G", "H", "Q", "C", "selected_count"))
        if not g >= h >= q >= c or q != min(budget, h) or not c <= selected <= budget:
            raise ValueError("Audit matching counts violate G>=H>=Q>=C or the global budget")
        expected = {"shortfall": (q-c)/q if q else 0., "waste": (selected-c)/budget,
                    "allocation_shortfall": q-c}
        for field, value in expected.items():
            if not np.isfinite(row[field]) or abs(float(row[field])-value) > 1e-12:
                raise ValueError(f"Audit {field} does not equal its matching definition")
        if row.get("role") != "audit":
            raise ValueError("Statistical inference requires held-out role=audit records")
        if row.get("matching_convention") != "maximum_class_consistent_nonignored":
            raise ValueError("Unknown matching convention in audit data")
        if not np.isfinite(row.get("matching_iou", np.nan)) or not 0 < row["matching_iou"] <= 1:
            raise ValueError("Audit matching IoU must lie in (0,1]")
        rows[image_id] = row
    return rows


def _population_signature(rows):
    return {image_id: (row["G"], row["H"], row["Q"], row["K"], row["matching_iou"],
                       row.get("candidate_export_hash"), row.get("candidate_semantic_hash"))
            for image_id, row in rows.items()}


def _source_synthetic(audit_path, run, base):
    """Preserve true/false/unknown execution provenance through aggregation.

    An adjacent evaluation report is usable only with its referenced prediction
    bytes. Otherwise a verified run manifest must actually list this audit file.
    Hand-built audit inputs remain unknown rather than becoming real benchmarks.
    """
    from .artifacts import verify_run
    audit_path, base = Path(audit_path).resolve(), Path(base)
    evidence, classifications = [], []
    prediction = run.get("predictions")
    evaluation_path = audit_path.parent / "EVALUATION.json"
    if prediction and evaluation_path.is_file():
        prediction_path = (base / prediction).resolve()
        if prediction_path.is_dir():
            prediction_path /= "predictions.jsonl"
        evaluation = json.loads(evaluation_path.read_text())
        declared = evaluation.get("is_synthetic")
        if (type(declared) is bool and prediction_path.is_file()
                and evaluation.get("prediction_sha256") == sha256_file(prediction_path)):
            classifications.append(declared)
            evidence.append({"source": "prediction_bound_evaluation", "path": str(evaluation_path),
                             "sha256": sha256_file(evaluation_path), "is_synthetic": declared})
    manifest_paths = ([base / run["run_manifest"]] if run.get("run_manifest") else
                      [audit_path.parent / "RUN_MANIFEST.json", audit_path.parent.parent / "RUN_MANIFEST.json"])
    # If no explicit run manifest was requested, a prediction-bound evaluation
    # report already establishes the recorded declaration without another full
    # scan of potentially large checkpoint/evidence artifacts.
    if not classifications or run.get("run_manifest"):
        for manifest_path in manifest_paths:
            manifest_path = manifest_path.resolve()
            if not manifest_path.is_file():
                continue
            try:
                relative = audit_path.relative_to(manifest_path.parent).as_posix()
                manifest = json.loads(manifest_path.read_text())
                item = manifest.get("artifacts", {}).get(relative)
                if not item or item.get("sha256") != sha256_file(audit_path):
                    continue
                checked = verify_run(manifest_path)
                declared = checked.get("synthetic")
                if checked.get("valid") and type(declared) is bool:
                    classifications.append(declared)
                    evidence.append({"source": "verified_run_manifest", "path": str(manifest_path),
                                     "sha256": sha256_file(manifest_path), "is_synthetic": declared})
            except (ValueError, KeyError, TypeError, OSError):
                continue
    value = (True if any(value is True for value in classifications) else
             False if classifications and all(value is False for value in classifications) else None)
    return value, evidence


def _diagnostic_pair(runs, image_ids, metric):
    arrays = []
    for a, b in runs:
        arrays.append(tuple({field: np.asarray([rows[i][field] for i in image_ids], dtype=float)
                             for field in ("shortfall", "waste", "allocation_shortfall", "Q")}
                            for rows in (a, b)))

    def evaluate(image_indices, seed_indices):
        result = []
        for method in (0, 1):
            per_seed = []
            for s in seed_indices:
                fields = arrays[int(s)][method]
                if metric == "pooled_shortfall":
                    denominator = float(fields["Q"][image_indices].sum())
                    value = float(fields["allocation_shortfall"][image_indices].sum()) / denominator if denominator else 0.
                else:
                    key = "waste" if metric == "mean_image_waste" else "shortfall"
                    value = float(fields[key][image_indices].mean())
                per_seed.append(value)
            result.append(float(np.mean(per_seed)))
        return tuple(result)
    return evaluate


def _prediction_pair(path, methods, budget, image_ids):
    path = Path(path)
    if path.is_dir():
        path = path / "predictions.jsonl"
    groups = {method: {} for method in methods}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        method = row.get("method")
        if method not in groups or row.get("K") != budget:
            continue
        image_id = str(row["image_id"])
        if image_id in groups[method]:
            raise ValueError("Duplicate prediction image/method/budget row")
        if row.get("track") != "membership_native_scores":
            raise ValueError("This comparison requires the recorded native-score membership track")
        ids, boxes = row["selected_ids"], np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
        scores = np.asarray(row["native_scores"], dtype=float)
        classes = row["classes"]
        if len(ids) > budget or len(ids) != len(set(ids)) or boxes.shape != (len(ids), 4):
            raise ValueError("Invalid prediction selection or global image budget")
        if scores.shape != (len(ids),) or len(classes) != len(ids) or not np.isfinite(scores).all():
            raise ValueError("Invalid prediction score/class vectors")
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Prediction boxes must be finite positive-size xyxy coordinates")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Native scores must lie in [0,1]")
        export_hash = row.get("candidate_export_hash")
        if not isinstance(export_hash, str) or len(export_hash) != 64 or any(c not in "0123456789abcdef" for c in export_hash.lower()):
            raise ValueError("Prediction row needs a candidate export SHA-256")
        predictions = [{"image_id": _coco_integer(image_id, "image_id"),
                        "category_id": _coco_integer(label, "category_id"),
                        "bbox": [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                        "score": float(score)} for box, label, score in zip(boxes, classes, scores)]
        groups[method][image_id] = {"predictions": predictions, "hash": export_hash,
                                   "count": len(ids), "semantic_hash": row.get("candidate_semantic_hash")}
    for method, rows in groups.items():
        if set(rows) != set(image_ids):
            raise ValueError(f"Prediction population for {method} differs from complete audit population")
    a, b = (groups[m] for m in methods)
    if any((a[i]["hash"], a[i]["semantic_hash"]) != (b[i]["hash"], b[i]["semantic_hash"]) for i in image_ids):
        raise ValueError("Paired predictions do not share the same frozen candidate exports")
    return (a, b), path


def _coco_pair(annotation_path, predictions_by_seed, image_ids, metric):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    dataset = json.loads(Path(annotation_path).read_text())
    images = {int(image["id"]): image for image in dataset["images"]}
    if len(images) != len(dataset["images"]):
        raise ValueError("COCO annotations contain duplicate image IDs")
    numeric_ids = [_coco_integer(i, "image_id") for i in image_ids]
    if len(set(numeric_ids)) != len(numeric_ids) or not set(numeric_ids) <= set(images):
        raise ValueError("Audit images do not map uniquely to COCO annotations")
    categories = {int(c["id"]) for c in dataset["categories"]}
    annotations = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    for pair in predictions_by_seed:
        for rows in pair:
            for row in rows.values():
                if any(p["category_id"] not in categories for p in row["predictions"]):
                    raise ValueError("Prediction category absent from COCO annotations")

    def evaluate(image_indices, seed_indices):
        # An image may appear several times in a bootstrap sample. Give each
        # occurrence a unique image ID and every copied annotation a fresh ID.
        clone = {"info": copy.deepcopy(dataset.get("info", {})),
                 "images": [], "categories": copy.deepcopy(dataset["categories"]), "annotations": []}
        mapping = []
        for occurrence, index in enumerate(image_indices, start=1):
            original = numeric_ids[int(index)]
            mapping.append((image_ids[int(index)], occurrence))
            clone["images"].append({**images[original], "id": occurrence})
            for annotation in annotations[original]:
                clone["annotations"].append({**copy.deepcopy(annotation), "image_id": occurrence,
                                               "id": len(clone["annotations"])+1})
        values = [[], []]
        with redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = clone
            gt.createIndex()
            for seed_index, multiplicity in Counter(map(int, seed_indices)).items():
                for method in (0, 1):
                    rows = predictions_by_seed[seed_index][method]
                    predictions = [{**prediction, "image_id": fresh}
                                   for original, fresh in mapping for prediction in rows[original]["predictions"]]
                    if predictions:
                        dt = gt.loadRes(predictions)
                    else:
                        dt = COCO()
                        dt.dataset = {"images": clone["images"], "categories": clone["categories"], "annotations": []}
                        dt.createIndex()
                    ev = COCOeval(gt, dt, "bbox")
                    ev.params.imgIds = list(range(1, len(image_indices)+1))
                    ev.params.maxDets = [1, 10, 100]
                    ev.evaluate()
                    ev.accumulate()
                    precision = ev.eval["precision"][:, :, :, 0, 2]
                    if metric != "AP":
                        threshold = .5 if metric == "AP50" else .75
                        precision = precision[np.isclose(ev.params.iouThrs, threshold)]
                    valid = precision[precision >= 0]
                    if valid.size == 0:
                        raise ValueError("COCO AP is undefined for a resample with no evaluable ground truth")
                    values[method].extend([float(valid.mean())]*multiplicity)
        return tuple(float(np.mean(value)) for value in values)
    return evaluate


def run_paired_statistics(specification_path, output=None):
    """Read a version-1 experiment specification and write paired statistics.

    Example specification:
      {"schema_version":1,"method_a":"dialdet","method_b":"local",
       "budget":100,"metric":"mean_image_shortfall","repeats":2000,"seed":1,
       "runs":[{"seed":1,"audits":"seed1/evaluation/per_image_audits.json"},
               {"seed":2,"audits":"seed2/evaluation/per_image_audits.json"}]}

    COCO metrics additionally require coco_annotations at the top level and
    predictions (JSONL file or inference directory) in every run. Relative
    paths are resolved against the specification file. subset="high_pressure"
    fixes the paper's top quintile by H before any bootstrap resampling.
    """
    source = Path(specification_path).resolve()
    spec = json.loads(source.read_text())
    if spec.get("schema_version", 1) != 1:
        raise ValueError("Expected version-1 paired-statistics specification")
    methods = (spec["method_a"], spec["method_b"])
    if any(not isinstance(m, str) or not m for m in methods) or methods[0] == methods[1]:
        raise ValueError("Two different method names are required")
    budget = _integer(spec["budget"], "budget", 1)
    metric = spec["metric"]
    if metric not in DIAGNOSTICS | COCO_METRICS:
        raise ValueError(f"Unsupported paired metric: {metric}")
    runs = spec["runs"]
    seed_ids = [_integer(run["seed"], "encoder seed") for run in runs]
    if len(seed_ids) < 2 or len(set(seed_ids)) != len(seed_ids):
        raise ValueError("At least two unique encoder seeds are required; paper experiments use five")
    paired, predictions, inputs, reference, hashes_reference = [], [], [], None, None
    synthetic_states = []
    for run in sorted(runs, key=lambda r: r["seed"]):
        audit_path = (source.parent / run["audits"]).resolve()
        if audit_path.is_dir():
            audit_path /= "per_image_audits.json"
        document = json.loads(audit_path.read_text())
        pair = tuple(_audit_group(document, f"{method}_K{budget}", budget) for method in methods)
        signature = _population_signature(pair[0])
        if signature != _population_signature(pair[1]) or (reference is not None and signature != reference):
            raise ValueError("Methods/seeds must use exactly the same images, full-pool attainability, matching IoU and budget")
        reference = signature
        paired.append(pair)
        input_record = {"seed": int(run["seed"]), "audits": str(audit_path), "audits_sha256": sha256_file(audit_path)}
        synthetic_state, classification_evidence = _source_synthetic(audit_path, run, source.parent)
        synthetic_states.append(synthetic_state)
        input_record.update(is_synthetic=synthetic_state, classification_evidence=classification_evidence)
        if metric in COCO_METRICS:
            prediction_pair, prediction_path = _prediction_pair(source.parent / run["predictions"], methods, budget, pair[0])
            hashes = {image_id: (row["hash"], row["semantic_hash"]) for image_id, row in prediction_pair[0].items()}
            if hashes_reference is not None and hashes != hashes_reference:
                raise ValueError("Encoder seeds must share one frozen detector candidate pool")
            hashes_reference = hashes
            for rows, audits in zip(prediction_pair, pair):
                for image_id, row in rows.items():
                    if row["count"] != audits[image_id]["selected_count"]:
                        raise ValueError("Prediction count differs from recorded audit")
                    if audits[image_id].get("candidate_export_hash") not in (None, row["hash"]):
                        raise ValueError("Prediction export hash differs from recorded audit")
            predictions.append(prediction_pair)
            input_record.update(predictions=str(prediction_path), predictions_sha256=sha256_file(prediction_path))
        inputs.append(input_record)
    image_ids = sorted(reference)
    subset = spec.get("subset", "all")
    if subset == "high_pressure":
        image_ids = high_pressure_ids(paired[0][0].values())
    elif subset != "all":
        raise ValueError("subset must be all or high_pressure")
    if metric in COCO_METRICS:
        annotations = (source.parent / spec["coco_annotations"]).resolve()
        callback = _coco_pair(annotations, predictions, image_ids, metric)
        annotation_provenance = {"path": str(annotations), "sha256": sha256_file(annotations)}
    else:
        callback = _diagnostic_pair(paired, image_ids, metric)
        annotation_provenance = None
    result = crossed_bootstrap(len(image_ids), len(paired), callback,
                               repeats=_integer(spec.get("repeats", 2000), "repeats", 2),
                               seed=_integer(spec.get("seed", 1), "bootstrap seed"))
    result.update(schema_version=1, metric=metric, metric_kind="official_coco_dataset_AP" if metric in COCO_METRICS else "matching_diagnostic",
                  method_a=methods[0], method_b=methods[1], contrast="method_a - method_b", scale="fraction",
                  budget=budget, subset=subset, image_ids=image_ids, encoder_seeds=sorted(seed_ids),
                  estimand="mean dataset metric over encoder seeds with the frozen detector held fixed",
                  paired_inputs=inputs, coco_annotations=annotation_provenance,
                  specification_sha256=sha256_file(source),
                  is_synthetic=(True if any(value is True for value in synthetic_states) else
                                False if all(value is False for value in synthetic_states) else None),
                  paper_seed_count_matched=len(paired) == 5,
                  provenance_status="computed_from_supplied_run_outputs",
                  limitations="Hashes establish input consistency; detector training history and statistical independence require separate evidence.")
    if output is not None:
        atomic_json(Path(output) / "PAIRED_STATISTICS.json", result)
    return result
