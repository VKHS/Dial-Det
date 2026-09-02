from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_coco(
    annotation_json: str | Path,
    prediction_json: str | Path,
    *,
    image_ids: list[int] | None = None,
    max_detections: tuple[int, int, int] = (1, 10, 100),
) -> dict[str, float]:
    """Run the official pycocotools evaluator when installed."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("COCO evaluation requires pycocotools") from exc
    ground_truth = COCO(str(annotation_json))
    detections = ground_truth.loadRes(str(prediction_json))
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.maxDets = list(max_detections)
    if image_ids is not None:
        evaluator.params.imgIds = list(map(int, image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    return {name: float(value) for name, value in zip(names, evaluator.stats, strict=True)}
