from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from dial_det.method.risks import evaluate_image_risks


def coco_pressure_score(
    ground_truth_boxes: Sequence[Sequence[float]] | np.ndarray,
    *,
    overlap_threshold: float = 0.1,
) -> float:
    boxes = np.asarray(ground_truth_boxes, dtype=np.float64)
    if boxes.size == 0:
        return 0.0
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("ground_truth_boxes must have shape [G,4]")
    from dial_det.method.risks import pairwise_iou
    iou = pairwise_iou(boxes, boxes)
    overlap_pairs = int(np.triu(iou >= overlap_threshold, k=1).sum())
    return float(len(boxes) + 2 * overlap_pairs)


@dataclass(frozen=True)
class ImageMetrics:
    image_id: str
    selected_count: int
    unique_ground_truth_covered: int
    unique_ground_truth_coverage_percent: float
    duplicate_loss: float
    coverage_loss: float
    unmatched_occupancy: float
    proposal_miss: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_selected_image(
    *,
    image_id: str,
    selected_boxes: np.ndarray,
    ground_truth_boxes: np.ndarray,
    budget_k: int,
    selected_labels: np.ndarray | None = None,
    ground_truth_labels: np.ndarray | None = None,
    ground_truth_ignored: np.ndarray | None = None,
    iou_threshold: float = 0.5,
    proposal_boxes: np.ndarray | None = None,
    proposal_labels: np.ndarray | None = None,
) -> ImageMetrics:
    result = evaluate_image_risks(
        selected_boxes,
        ground_truth_boxes,
        budget_k=budget_k,
        selected_labels=selected_labels,
        ground_truth_labels=ground_truth_labels,
        ground_truth_ignored=ground_truth_ignored,
        iou_threshold=iou_threshold,
        proposal_boxes=proposal_boxes,
        proposal_labels=proposal_labels,
    )
    denominator = min(budget_k, result.ground_truth_count)
    coverage_percent = 100.0 if denominator == 0 else 100.0 * result.covered_ground_truth_count / denominator
    return ImageMetrics(
        image_id=str(image_id),
        selected_count=result.selected_count,
        unique_ground_truth_covered=result.covered_ground_truth_count,
        unique_ground_truth_coverage_percent=coverage_percent,
        duplicate_loss=result.duplicate,
        coverage_loss=result.coverage,
        unmatched_occupancy=result.unmatched,
        proposal_miss=result.proposal_miss,
    )
