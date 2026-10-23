"""Maximum-matching audits and optional official COCOeval; no fabricated AP."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
from .data import bbox_iou


def _iou_threshold(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Matching IoU must be finite and lie in (0,1]")
    return float(value)


def _validate_selection(record, selected):
    raw = np.asarray(selected)
    if raw.ndim != 1 or (raw.size and (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all())):
        raise ValueError("Selected indices must be a one-dimensional integer sequence")
    selected = raw.astype(np.int64)
    if len(set(selected.tolist())) != len(selected) or np.any(selected < 0) or np.any(selected >= len(record.ids)):
        raise ValueError("Selected indices are invalid")
    return selected


def maximum_matching(record, selected=None, iou_threshold=.5):
    """Deterministic maximum cardinality class-consistent matching.

    This is the audit matching nu_tau, not the evaluator's score-order matching.
    Ignored annotations do not contribute to attainable object count.
    """
    iou_threshold = _iou_threshold(iou_threshold)
    selected = np.arange(len(record.ids)) if selected is None else _validate_selection(record, selected)
    selected = selected[np.argsort(record.ids[selected], kind="stable")]
    gt = np.flatnonzero(~record.gt_ignore)
    gt = gt[np.argsort(record.gt_ids[gt], kind="stable")]
    eligible = (bbox_iou(record.boxes[selected], record.gt_boxes[gt]) >= iou_threshold)
    eligible &= record.classes[selected, None] == record.gt_classes[None, gt]
    owners = np.full(len(gt), -1, dtype=int)

    def augment(row, seen):
        for col in np.flatnonzero(eligible[row]):
            if seen[col]:
                continue
            seen[col] = True
            if owners[col] < 0 or augment(int(owners[col]), seen):
                owners[col] = row
                return True
        return False

    for row in range(len(selected)):
        augment(row, np.zeros(len(gt), dtype=bool))
    return {int(selected[row]): int(gt[col]) for col, row in enumerate(owners) if row >= 0}


def score_order_matching(record, selected, iou_threshold=.5, scores=None):
    """Diagnostic greedy assignment; never mislabeled as official COCOeval.

    Official evaluators may have additional ignore, crowd and area rules.
    """
    iou_threshold = _iou_threshold(iou_threshold)
    selected = _validate_selection(record, selected)
    score = record.scores if scores is None else np.asarray(scores)
    if score.shape != (len(record.ids),) or not np.isfinite(score).all():
        raise ValueError("Matching scores must be a finite vector with one score per candidate")
    order = sorted(selected.tolist(), key=lambda i: (-float(score[i]), int(record.ids[i])))
    gt = np.flatnonzero(~record.gt_ignore)
    iou = bbox_iou(record.boxes, record.gt_boxes)
    used, matches = set(), {}
    for i in order:
        valid = [int(g) for g in gt if int(g) not in used and record.classes[i] == record.gt_classes[g]
                 and iou[i, g] >= iou_threshold]
        if valid:
            g = min(valid, key=lambda j: (-float(iou[i, j]), int(record.gt_ids[j])))
            matches[i] = g
            used.add(g)
    return matches


def audit_record(record, selected, budget, method="dialdet", status="unknown", iou_threshold=.5):
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("Risk/audit budgets must be positive integers")
    selected = list(selected)
    if len(selected) > budget:
        raise ValueError("The report violates the global image budget")
    full_match = maximum_matching(record, iou_threshold=iou_threshold)
    report_match = maximum_matching(record, selected, iou_threshold)
    greedy = score_order_matching(record, selected, iou_threshold)
    g, h, c, cev = int((~record.gt_ignore).sum()), len(full_match), len(report_match), len(greedy)
    q = min(budget, h)
    if not g >= h >= q >= c >= cev >= 0:
        raise AssertionError("Matching order invariant failed")
    terms = {"candidate_absence": g - h, "capacity_loss": h - q,
             "allocation_shortfall": q - c, "greedy_assignment_loss": c - cev}
    if sum(terms.values()) != g - cev:
        raise AssertionError("Equation (1) does not telescope")
    return {"image_id": record.image_id, "method": method, "status": status,
            "K": int(budget), "N": len(record.ids), "G": g, "H": h, "Q": q, "C": c,
            "C_score_greedy": cev, "selected_count": len(selected), **terms,
            "waste": (len(selected) - c) / budget, "shortfall": (q - c) / q if q else 0.,
            "full_report": len(selected) == budget, "capacity_binding": h > budget,
            "matching_iou": iou_threshold,
            "matching_convention": "maximum_class_consistent_nonignored",
            "score_assignment_note": "C_score_greedy is a diagnostic, not official evaluator Cev",
            "role": record.metadata.get("role", "unspecified")}


def summarize_audits(rows):
    if not rows:
        return {"n_images": 0}
    denom = sum(row["Q"] for row in rows)
    return {"n_images": len(rows), "attainable_under_budget": denom,
            "matched": sum(row["C"] for row in rows),
            "pooled_shortfall": sum(row["allocation_shortfall"] for row in rows) / denom if denom else 0.,
            "mean_image_shortfall": float(np.mean([row["shortfall"] for row in rows])),
            "mean_image_waste": float(np.mean([row["waste"] for row in rows])),
            "full_report_share": float(np.mean([row["full_report"] for row in rows])),
            "capacity_binding_share": float(np.mean([row["capacity_binding"] for row in rows])),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "ap": None, "ap_note": "Matching counts are diagnostics. Benchmark AP or miss rate requires the dataset-specific evaluator."}


def high_pressure_ids(rows, fraction=.2):
    """Appendix E.10: top quintile by full-pool attainable H, before selection.

    A deterministic string image-ID tie break and ceil(n*fraction) make the
    boundary explicit for populations not divisible by five. Ground-truth
    overlap and the selected report do not determine membership.
    """
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("High-pressure fraction must lie in (0,1]")
    rows = list(rows)
    ids = [str(row["image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("High-pressure population contains duplicate image IDs")
    for row in rows:
        h = row["H"]
        if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 0:
            raise ValueError("Attainable H must be a nonnegative integer")
    ranked = sorted(rows, key=lambda row: (-int(row["H"]), str(row["image_id"])))
    return [str(row["image_id"]) for row in ranked[:math.ceil(len(rows) * fraction)]]


def summarize_high_pressure(groups, fraction=.2):
    """Freeze one H-based subset for all methods/budgets on the same pool."""
    if not groups:
        raise ValueError("At least one audit group is required")
    reference = list(next(iter(groups.values())))
    selected = high_pressure_ids(reference, fraction)
    population = {str(row["image_id"]): int(row["H"]) for row in reference}
    selected_set, summaries = set(selected), {}
    for name, rows in groups.items():
        # This also validates unique IDs and integer H in every group.
        high_pressure_ids(rows, fraction)
        if {str(row["image_id"]): int(row["H"]) for row in rows} != population:
            raise ValueError("HP comparisons require identical image populations and full-pool H")
        summaries[name] = summarize_audits([row for row in rows if str(row["image_id"]) in selected_set])
    return {"definition": "top_quintile_by_full_pool_attainable_H" if fraction == .2 else "top_fraction_by_full_pool_attainable_H",
            "fraction": float(fraction), "image_ids": selected,
            "population_size": len(population), "subset_size": len(selected),
            "tie_break": "descending H, ascending string image_id; ceil(n*fraction)",
            "selection_uses_labels": True, "use": "retrospective audit only; never inference routing",
            "results": summaries}


def coco_predictions(record, selected, own_scores=None):
    selected = _validate_selection(record, selected)
    scores = record.scores if own_scores is None else np.asarray(own_scores)
    if scores.shape != (len(record.ids),) or not np.isfinite(scores).all():
        raise ValueError("COCO scores must be finite and have one entry per candidate")
    predictions = []
    for index in selected:
        box = record.boxes[index]
        predictions.append({"image_id": int(record.image_id), "category_id": int(record.classes[index]),
                            "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                            "score": float(scores[index])})
    return predictions


def official_coco_evaluation(annotation_file, predictions, image_ids, global_budget):
    """COCOeval's standard per-category maxDets=100 follows the global cap."""
    if isinstance(global_budget, (bool, np.bool_)) or not isinstance(global_budget, (int, np.integer)) or global_budget < 1:
        raise ValueError("A positive integer global report budget is required")
    image_ids = list(image_ids)
    converted = [_coco_integer(i, "image_id") for i in image_ids]
    if not converted or len(converted) != len(set(converted)):
        raise ValueError("COCO evaluation requires a nonempty population of unique image IDs")
    allowed = set(converted)
    predictions = list(predictions)
    counts = Counter()
    for pred in predictions:
        image_id = _coco_integer(pred["image_id"], "image_id")
        _coco_integer(pred["category_id"], "category_id")
        box = np.asarray(pred["bbox"], dtype=float)
        score = pred["score"]
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
            raise ValueError("COCO prediction bbox must be finite [x,y,width,height] with positive size")
        if isinstance(score, (bool, np.bool_)) or not np.isscalar(score) or not np.isfinite(score):
            raise ValueError("COCO prediction scores must be finite scalars")
        if image_id not in allowed:
            raise ValueError("A prediction image is outside the requested evaluation population")
        counts[image_id] += 1
        if counts[image_id] > global_budget:
            raise ValueError("Predictions violate the global per-image report budget")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation_file))
    if not allowed <= set(gt.imgs):
        raise ValueError("Requested COCO evaluation images are absent from annotations")
    if any(_coco_integer(pred["category_id"], "category_id") not in gt.cats for pred in predictions):
        raise ValueError("Prediction category is absent from COCO annotation categories")
    if predictions:
        # COCO.loadRes annotates result dictionaries in place. Preserve the
        # caller's frozen export so repeated evaluations do not alter it.
        dt = gt.loadRes([dict(prediction) for prediction in predictions])
    else:
        # COCO.loadRes assumes a nonempty result list in several releases.
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(allowed)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {"metrics": {key: float(val) for key, val in zip(names, evaluator.stats)},
            "scale": "fraction", "global_candidate_budget": int(global_budget),
            "cocoeval_per_category_maxDets": [1, 10, 100], "n_images": len(evaluator.params.imgIds)}


def _coco_integer(value, name):
    """Accept JSON integers and their canonical decimal string representation."""
    if isinstance(value, str):
        if not value or not value.isdecimal():
            raise ValueError(f"COCO {name} must be a nonnegative integer")
        return int(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"COCO {name} must be a nonnegative integer")
    return int(value)

