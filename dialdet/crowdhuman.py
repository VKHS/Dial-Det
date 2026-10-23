"""Native CrowdHuman output and an adapter to the upstream APMR toolkit.

No COCO metric is substituted for CrowdHuman AP50 or MR^-2.  The evaluator is
provided as a local checkout of https://github.com/megvii-model/CrowdDetection
and is imported in a separate process.  Its matching and integration code is
neither copied into this package nor modified.  The returned source hashes make
the exact evaluator version inspectable.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .data import atomic_json, sha256_file
from .metrics import _validate_selection


UPSTREAM_URL = "https://github.com/megvii-model/CrowdDetection/tree/master/evaluate/APMRToolkits"


def crowdhuman_predictions(record, selected, own_scores=None):
    """Return one native ODGT prediction row, including empty image reports.

    The immutable candidate's original ODGT identifier is kept in metadata by
    the exporter; namespaced training identifiers must never be parsed back.
    A prediction uses full-body ``box=[x,y,w,h]`` and the native detector score.
    """
    selected = _validate_selection(record, selected)
    original_id = record.metadata.get("original_image_id")
    if original_id is None:
        if str(record.image_id).startswith("crowdhuman:"):
            raise ValueError("Namespaced CrowdHuman records require metadata.original_image_id")
        original_id = record.image_id
    if not isinstance(original_id, str) or not original_id:
        raise ValueError("CrowdHuman's original image ID must be a nonempty string")
    person_class_id = _positive_integer(record.metadata.get("person_class_id", 1),
                                       "The native person class ID")
    if np.any(record.classes[selected] != person_class_id):
        raise ValueError("CrowdHuman full-body evaluation accepts only the declared native person class")
    scores = record.scores if own_scores is None else np.asarray(own_scores, dtype=np.float64)
    if scores.shape != record.scores.shape or not np.isfinite(scores).all():
        raise ValueError("Reporting scores must be a finite vector with one score per candidate")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Reporting scores must lie in [0,1]")
    # A deterministic source-ID tie order is shared by every allocation method.
    order = sorted(selected.tolist(), key=lambda i: (-float(scores[i]), int(record.ids[i])))
    boxes = []
    for index in order:
        box = record.boxes[index]
        boxes.append({"box": [float(box[0]), float(box[1]),
                               float(box[2] - box[0]), float(box[3] - box[1])],
                      "score": float(scores[index]), "tag": 1})
    height, width = record.image_size
    return {"ID": original_id, "width": int(width), "height": int(height), "dtboxes": boxes}


def write_crowdhuman_predictions(path, predictions):
    """Write official-format JSON-lines predictions atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _positive_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _prepare_population(annotation_file, predictions, image_ids, global_budget):
    budget = _positive_integer(global_budget, "The global report budget")
    requested = list(image_ids)
    if not requested or any(not isinstance(i, str) or not i for i in requested):
        raise ValueError("Use nonempty original ODGT image IDs for evaluation")
    if len(set(requested)) != len(requested):
        raise ValueError("The evaluation population contains duplicate image IDs")
    allowed = set(requested)
    by_id = {}
    for row in predictions:
        if not isinstance(row, dict) or row.get("ID") not in allowed:
            raise ValueError("A prediction image lies outside the requested evaluation population")
        key = row["ID"]
        if key in by_id:
            raise ValueError("A prediction image occurs more than once")
        width = _positive_integer(row.get("width"), "Prediction image width")
        height = _positive_integer(row.get("height"), "Prediction image height")
        dets = row.get("dtboxes")
        if not isinstance(dets, list) or len(dets) > budget:
            raise ValueError("A prediction row must contain at most K full-body detections")
        clean = []
        for detection in dets:
            box = np.asarray(detection.get("box"), dtype=np.float64)
            score = detection.get("score")
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
                raise ValueError("CrowdHuman prediction boxes must be finite positive-size xywh")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("CrowdHuman prediction scores must lie in [0,1]")
            if detection.get("tag", 1) not in (1, "person"):
                raise ValueError("Only person detections are valid for CrowdHuman full-body evaluation")
            clean.append({"box": box.tolist(), "score": float(score), "tag": 1})
        by_id[key] = {"ID": key, "width": width, "height": height, "dtboxes": clean}
    if set(by_id) != allowed:
        raise ValueError("Every evaluation image requires an explicit prediction row, even when empty")

    ground_truth, seen = [], set()
    with Path(annotation_file).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("ID")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Invalid ODGT image ID at annotation line {number}")
            if key in seen:
                raise ValueError("The annotation file contains duplicate image IDs")
            seen.add(key)
            if key not in allowed:
                continue
            annotations = row.get("gtboxes")
            if not isinstance(annotations, list):
                raise ValueError("Native CrowdHuman annotations require a gtboxes list")
            for annotation in annotations:
                # The upstream loader reads hbox even in full-body mode.  Keep
                # native annotations, rather than manufacturing head geometry.
                for field in ("fbox", "hbox"):
                    box = np.asarray(annotation.get(field), dtype=np.float64)
                    if (box.shape != (4,) or not np.isfinite(box).all()
                            or (field == "fbox" and np.any(box[2:] <= 0))):
                        raise ValueError(f"Native CrowdHuman annotations require valid {field} xywh boxes")
                if not isinstance(annotation.get("tag"), str):
                    raise ValueError("A native CrowdHuman annotation needs a string tag")
            for field in ("width", "height"):
                if field in row and row[field] != by_id[key][field]:
                    raise ValueError("Annotation and frozen-export image dimensions disagree")
            ground_truth.append(row)
    if {row["ID"] for row in ground_truth} != allowed:
        raise ValueError("Some requested evaluation images are missing from the annotation file")
    # Preserve annotation-file order.  The upstream stable score sort uses this
    # order for cross-image ties, so caller list ordering must not change AP.
    ordered_predictions = [by_id[row["ID"]] for row in ground_truth]
    return ground_truth, ordered_predictions


def _resolve_toolkit(evaluator_root):
    root = Path(evaluator_root).expanduser().resolve()
    candidates = [root, root / "APMRToolkits", root / "evaluate" / "APMRToolkits",
                  root / "lib" / "evaluate" / "APMRToolkits"]
    for path in candidates:
        if path.name == "APMRToolkits" and (path / "database.py").is_file() and (path / "image.py").is_file():
            return path
    raise FileNotFoundError("Expected a CrowdDetection checkout or an APMRToolkits directory containing database.py and image.py")


# Only adapter glue runs here.  Nonempty-image matching and metric integration
# are performed by the imported upstream modules.  Two otherwise undefined
# edge cases are explicitly recorded in the result instead of returning NaN.
_UPSTREAM_WORKER = r'''
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from APMRToolkits.database import Database, PERSON_CLASSES
from APMRToolkits.image import Image
gt_path, dt_path, result_path = sys.argv[2:5]
ground_truth = [json.loads(line) for line in Path(gt_path).read_text().splitlines() if line.strip()]
predictions = {row['ID']: row for row in (json.loads(line) for line in Path(dt_path).read_text().splitlines() if line.strip())}
notes = []
if all(row['gtboxes'] for row in ground_truth):
    database = Database(gt_path, dt_path, 'box', None, 0)
    database.compare(thres=0.5)
else:
    # Upstream load_gt_boxes has an inconsistent empty return shape; construct
    # only those empty images directly, keeping all other upstream operations.
    database = Database.__new__(Database)
    database.eval_mode = 0
    database.images = {}
    for row in ground_truth:
        item = Image(0)
        if row['gtboxes']:
            item.load(row, 'box', None, PERSON_CLASSES, True)
        else:
            item.ID = row['ID']
            item.gtboxes = np.empty((0, 5), dtype=float)
            item._gtNum, item._ignNum = 0, 0
        item.load(predictions[row['ID']], 'box', None, PERSON_CLASSES, False)
        item.clip_all_boader()
        database.images[row['ID']] = item
    database._gtNum = sum(item._gtNum for item in database.images.values())
    database._ignNum = sum(item._ignNum for item in database.images.values())
    database._imageNum = len(database.images)
    database.scorelist = []
    for item in database.images.values():
        if item._gtNum:
            database.scorelist.extend(item.compare_caltech(0.5))
        else:
            database.scorelist.extend((box, 0, item.ID) for box in sorted(item.dtboxes, key=lambda b: b[-1], reverse=True))
    database.scorelist.sort(key=lambda entry: entry[0][-1], reverse=True)
    notes.append('Empty-GT image adapter: retains image and counts every detection as a false positive; upstream empty-array shape bug bypassed.')
positive_count = int(database._gtNum - database._ignNum)
tp = sum(int(item[1] == 1) for item in database.scorelist)
fp = sum(int(item[1] == 0) for item in database.scorelist)
if positive_count == 0:
    metrics = {'AP50': None, 'MR_minus2': None, 'Recall': None}
    notes.append('No nonignored ground-truth people in this population: detection metrics are undefined.')
elif not database.scorelist:
    metrics = {'AP50': 0.0, 'MR_minus2': 1.0, 'Recall': 0.0}
    notes.append('Empty effective report convention: AP50=0, MR_minus2=1, Recall=0; upstream MR otherwise returns NaN.')
else:
    with np.errstate(divide='ignore', invalid='ignore'):
        ap, _ = database.eval_AP()
        mr, _ = database.eval_MR(ref='CALTECH_-2')
    metrics = {'AP50': float(ap), 'MR_minus2': float(mr), 'Recall': float(tp / positive_count)}
result = {'metrics': metrics, 'nonignored_ground_truth': positive_count,
          'matched_detections': tp, 'false_positive_detections': fp,
          'ignored_detections': sum(len(row['dtboxes']) for row in predictions.values()) - tp - fp,
          'edge_case_notes': notes}
Path(result_path).write_text(json.dumps(result, allow_nan=False))
'''


def official_crowdhuman_evaluation(annotation_file, predictions, image_ids, global_budget,
                                  evaluator_root, output_dir=None):
    """Evaluate a fixed image population with upstream full-body AP/MR code.

    ``image_ids`` are original ODGT IDs, not internal namespaced identifiers.
    ``predictions`` has one row from :func:`crowdhuman_predictions` per image.
    ``evaluator_root`` may be the repository root, its ``evaluate`` directory,
    or ``APMRToolkits`` itself.  No network call or automatic checkout occurs.
    If ``output_dir`` is supplied, the exact filtered inputs, evaluator output
    and source fingerprint are retained there for reproducibility.
    """
    ground_truth, predictions = _prepare_population(annotation_file, predictions, image_ids, global_budget)
    toolkit = _resolve_toolkit(evaluator_root)
    fingerprint = {str(path.relative_to(toolkit)): sha256_file(path)
                   for path in sorted(toolkit.rglob("*.py"))}
    temporary = tempfile.TemporaryDirectory(prefix="dialdet_crowdhuman_") if output_dir is None else None
    directory = Path(temporary.name if temporary else output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        gt_path = write_crowdhuman_predictions(directory / "ground_truth.odgt", ground_truth)
        dt_path = write_crowdhuman_predictions(directory / "predictions.odgt", predictions)
        result_path = directory / "upstream_metrics.json"
        completed = subprocess.run([sys.executable, "-c", _UPSTREAM_WORKER, str(toolkit.parent),
                                    str(gt_path), str(dt_path), str(result_path)],
                                   capture_output=True, text=True, check=False)
        if output_dir is not None:
            (directory / "upstream_stdout.log").write_text(completed.stdout)
            (directory / "upstream_stderr.log").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError("CrowdHuman upstream evaluator failed; no benchmark metrics were substituted.\n"
                               + completed.stderr[-6000:])
        result = json.loads(result_path.read_text())
        if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
               for value in result["metrics"].values()):
            raise ValueError("The upstream evaluator produced an invalid metric")
        result.update(evaluator="CrowdDetection_APMRToolkits_upstream", evaluator_url=UPSTREAM_URL,
                      evaluator_source_sha256=fingerprint, evaluator_path=str(toolkit),
                      annotation_sha256=sha256_file(annotation_file),
                      population_ids=[row["ID"] for row in ground_truth],
                      n_images=len(ground_truth), global_candidate_budget=int(global_budget),
                      scale="fraction", target="full_body", matching_iou=0.5,
                      matching="upstream_caltech", mr_reference="CALTECH_-2",
                      score_track="provided_scores", JI=None,
                      JI_note="Jaccard index requires the separate benchmark evaluator and is not computed here.")
        if output_dir is not None:
            atomic_json(directory / "CROWDHUMAN_EVALUATION.json", result)
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()
