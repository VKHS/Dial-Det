"""Transparent same-stream baseline selectors.

Recent published competitor methods are not reimplemented under misleading
names.  Their official implementations should be run separately and converted
to the common prediction schema documented in the repository.
"""
from __future__ import annotations

import numpy as np

from dial_det.method.objective import DIALObjectiveEvidence


def _deterministic_order(score: np.ndarray, candidate_ids: np.ndarray) -> list[int]:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    ids = np.asarray(candidate_ids).astype(str).reshape(-1)
    if values.shape != ids.shape:
        raise ValueError("score and candidate_ids must have the same length")
    return sorted(range(len(values)), key=lambda i: (-float(values[i]), str(ids[i])))


def top_score(score: np.ndarray, candidate_ids: np.ndarray, budget: int) -> tuple[int, ...]:
    if not 0 <= budget <= len(score):
        raise ValueError("budget must lie in [0,N]")
    return tuple(_deterministic_order(score, candidate_ids)[:budget])


def top_value(value: np.ndarray, candidate_ids: np.ndarray, budget: int) -> tuple[int, ...]:
    return top_score(value, candidate_ids, budget)


def hard_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    budget: int,
    iou_threshold: float,
    class_ids: np.ndarray | None = None,
) -> tuple[int, ...]:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    order = _deterministic_order(scores, candidate_ids)
    labels = None if class_ids is None else np.asarray(class_ids).reshape(-1)
    if boxes.shape != (len(order), 4):
        raise ValueError("boxes_xyxy must have shape [N,4]")
    if labels is not None and labels.shape != (len(order),):
        raise ValueError("class_ids must have shape [N]")
    selected: list[int] = []
    while order and len(selected) < budget:
        current = order.pop(0)
        selected.append(current)
        survivors: list[int] = []
        for other in order:
            if labels is not None and labels[current] != labels[other]:
                survivors.append(other)
                continue
            a, b = boxes[current], boxes[other]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou <= iou_threshold:
                survivors.append(other)
        order = survivors
    return tuple(selected)


def greedy_objective(evidence: DIALObjectiveEvidence, budget: int, *, exact: bool) -> tuple[int, ...]:
    if not 0 <= budget <= evidence.number_of_candidates:
        raise ValueError("budget must lie in [0,N]")
    selected: list[int] = []
    remaining = set(range(evidence.number_of_candidates))
    while len(selected) < budget and remaining:
        ranked = sorted(
            ((evidence.marginal_gain(selected, i), -i, i) for i in remaining),
            reverse=True,
        )
        gain, _, chosen = ranked[0]
        if not exact and gain <= 0:
            break
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def dpp_map_greedy(
    quality: np.ndarray,
    similarity: np.ndarray,
    candidate_ids: np.ndarray,
    budget: int,
    *,
    jitter: float = 1e-8,
) -> tuple[int, ...]:
    """Greedy MAP approximation for a quality-weighted DPP kernel."""
    q = np.asarray(quality, dtype=np.float64).reshape(-1)
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.shape != (len(q), len(q)):
        raise ValueError("similarity must have shape [N,N]")
    kernel = q[:, None] * similarity * q[None, :]
    kernel = 0.5 * (kernel + kernel.T) + jitter * np.eye(len(q))
    selected: list[int] = []
    remaining = set(range(len(q)))
    for _ in range(min(budget, len(q))):
        candidates = []
        for i in remaining:
            trial = sorted([*selected, i])
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(trial, trial)])
            objective = float(logdet) if sign > 0 else -np.inf
            candidates.append((objective, -i, i))
        _, _, chosen = max(candidates)
        selected.append(chosen)
        remaining.remove(chosen)
    order = _deterministic_order(q, candidate_ids)
    # Deterministic tie stability for singular kernels.
    if not selected and budget:
        selected = order[:budget]
    return tuple(sorted(selected))
