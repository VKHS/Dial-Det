"""Matched controls on the same frozen candidate pool."""
from __future__ import annotations
import numpy as np
from .data import bbox_iou


def _budget(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("A nonnegative integer report cap is required")
    return int(value)


def local_rule(q, r, budget, threshold=.5, candidate_ids=None):
    budget = _budget(budget)
    q, r = np.asarray(q), np.asarray(r)
    ids = np.arange(len(q)) if candidate_ids is None else np.asarray(candidate_ids)
    selected = []
    for i in sorted(range(len(q)), key=lambda j: (-float(q[j]), int(ids[j]))):
        if len(selected) >= budget:
            break
        if all(r[i, j] < threshold for j in selected):
            selected.append(i)
    return selected


def nms(record, budget, threshold=.5, score_floor=.001):
    budget = _budget(budget)
    iou = bbox_iou(record.boxes, record.boxes)
    order = sorted(range(len(record.ids)), key=lambda i: (-float(record.scores[i]), int(record.ids[i])))
    selected = []
    for i in order:
        if len(selected) >= budget:
            break
        if record.scores[i] < score_floor:
            continue
        if all(record.classes[i] != record.classes[j] or iou[i, j] <= threshold for j in selected):
            selected.append(i)
    return selected


def soft_nms(record, budget, sigma=.5, score_floor=.001):
    budget = _budget(budget)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    iou = bbox_iou(record.boxes, record.boxes)
    working = record.scores.copy()
    pending, selected = set(range(len(record.ids))), []
    while pending and len(selected) < budget:
        i = min(pending, key=lambda j: (-float(working[j]), int(record.ids[j])))
        pending.remove(i)
        if working[i] < score_floor:
            break
        selected.append(i)
        for j in pending:
            if record.classes[i] == record.classes[j]:
                working[j] *= np.exp(-float(iou[i, j] ** 2) / sigma)
    # Membership track restores record.scores; scores are returned separately.
    return selected, working


def greedy(objective, budget, candidate_ids=None):
    budget = _budget(budget)
    n = len(objective.a)
    ids = np.arange(n) if candidate_ids is None else np.asarray(candidate_ids)
    selected = []
    while len(selected) < min(budget, n):
        choices = [(objective.marginal(selected, j), j) for j in range(n) if j not in selected]
        gain, j = max(choices, key=lambda item: (item[0], -int(ids[item[1]])))
        if gain <= 0:
            break
        selected.append(j)
    return selected


def dpp_kernel(quality, similarity, *, scale=1., ridge=1e-4, epsilon=1e-8):
    """Appendix E.4 DPP-MAP* kernel construction.

    Symmetrize affinities, project onto the PSD cone, add ridge, normalize the
    diagonal, then apply zeta * diag(q+epsilon) R+ diag(q+epsilon). The scale
    and ridge must be tuned on development data before a paper comparison.
    Defaults are callable reference settings, not claimed tuned values.
    """
    q = np.asarray(quality, dtype=np.float64)
    r = np.asarray(similarity, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("DPP quality must be a finite probability vector")
    if r.shape != (len(q), len(q)) or not np.isfinite(r).all():
        raise ValueError("DPP similarity must be finite [N,N]")
    if any(not np.isfinite(value) or value <= 0 for value in (scale, ridge, epsilon)):
        raise ValueError("DPP scale, ridge and epsilon must be finite and positive")
    if not len(q):
        return np.empty((0, 0), dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh((r+r.T)*.5)
    projected = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
    projected += float(ridge)*np.eye(len(q))
    diagonal = np.sqrt(np.diag(projected))
    normalized = projected / diagonal[:, None] / diagonal[None, :]
    kernel = float(scale) * (q+epsilon)[:, None] * normalized * (q+epsilon)[None, :]
    return (kernel+kernel.T)*.5


def dpp_map(quality, similarity, budget, candidate_ids=None, *, scale=1., ridge=1e-4, epsilon=1e-8):
    """Greedy log-det MAP under |S|<=K, including the empty-set alternative."""
    budget = _budget(budget)
    kernel = dpp_kernel(quality, similarity, scale=scale, ridge=ridge, epsilon=epsilon)
    n = len(kernel)
    ids = np.arange(n) if candidate_ids is None else np.asarray(candidate_ids)
    if ids.shape != (n,) or len(set(map(str, ids))) != n:
        raise ValueError("DPP candidate IDs must be a unique vector of length N")
    selected, current = [], 0.
    while len(selected) < min(budget, n):
        choices = []
        for j in range(n):
            if j in selected:
                continue
            trial = [*selected, j]
            sign, value = np.linalg.slogdet(kernel[np.ix_(trial, trial)])
            choices.append((float(value) if sign > 0 else -np.inf, j))
        value, chosen = min(choices, key=lambda item: (-item[0], str(ids[item[1]])))
        if not np.isfinite(value) or value <= current:
            break
        selected.append(chosen)
        current = value
    return selected

