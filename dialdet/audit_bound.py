"""Label-only matching-miss audit from Appendix C.8, Theorem C1.

This module is never an inference input. It builds reference identity blocks from
ground-truth matching AFTER decoding and bounds allocation misses, not AP.
All ideal-objective comparisons and tail inversions use exact rational arithmetic.
"""
from __future__ import annotations

from fractions import Fraction
import itertools
import math

import numpy as np

from .certificate import check_certificate
from .data import bbox_iou, record_hash
from .metrics import maximum_matching, _validate_selection
from .objective import GRID_SCALE, build_objective, canonical_hash, parameter


def _fraction(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("The matching audit requires finite qualities and parameters.")
    return Fraction.from_float(value)


def _encoded(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "value": float(value)}


def matching_miss_bound(record, selected, q, r, budget, certificate,
                       config=None, mass=None, iou_threshold=.5) -> dict:
    """Return a conservative integer upper bound on Q-C in Theorem C1.

    `certificate` can be the raw certificate or the complete allocate() result.
    Missing, rejected, mismatched, and fallback certificates produce the vacuous
    bound Q. Reference blocks and F0 use labels only for this retrospective audit.
    A supplied candidate-export hash, candidate IDs, coefficient hash, budget,
    and selected set must match before the certificate is admitted.
    """
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("The matching audit budget must be a nonnegative integer.")
    selected = _validate_selection(record, selected).tolist()
    if len(selected) > budget:
        raise ValueError("Selected report exceeds its global budget.")
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(record.ids),):
        raise ValueError("Audit qualities must align with the frozen candidates.")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU matching threshold must lie in [0,1].")
    full_match = maximum_matching(record, iou_threshold=iou_threshold)
    selected_match = maximum_matching(record, selected, iou_threshold)
    h, count = len(full_match), len(selected_match)
    attainable = min(int(budget), h)
    base = {"image_id": record.image_id, "matching_iou": float(iou_threshold),
            "H": h, "Q": attainable, "C": count, "realized_misses": attainable-count,
            "audit_only": True, "scope": "allocation misses Q-C; not AP or an inference guarantee"}

    def vacuous(reason, extras=None):
        return {**base, "bound": attainable, "nonvacuous": False,
                "certificate_valid": False, "status": "vacuous", "reason": reason,
                "Qpositive": None, "reference_block_count": h,
                **(extras or {})}

    if isinstance(certificate, dict) and "certificate" in certificate:
        if certificate.get("status") == "fallback":
            return vacuous("The deployed allocator used its numerical fallback.")
        certificate = certificate["certificate"]
    if certificate is None:
        return vacuous("No checked objective-gap certificate is available.")
    check = check_certificate(certificate)
    if not check["valid"]:
        return vacuous("Objective certificate rejected: "+"; ".join(check["errors"]))
    if certificate.get("candidate_ids") != [str(i) for i in record.ids.tolist()]:
        return vacuous("Certificate candidate identities do not match the frozen export.")
    if sorted(certificate.get("selected", [])) != sorted(selected):
        return vacuous("Certificate selected set does not match the audited report.")
    if certificate.get("budget") != min(int(budget), len(q)):
        return vacuous("Certificate budget does not match the audited problem.")
    claimed_hash = certificate.get("candidate_export_hash")
    if claimed_hash is not None and claimed_hash != record_hash(record):
        return vacuous("Certificate candidate-export hash does not match this record.")
    if config is None:
        config = certificate["configuration"]
    try:
        obj = build_objective(q, r, config, mass)
    except (ValueError, ArithmeticError) as exc:
        return vacuous("Cannot reconstruct the certified objective: "+str(exc))
    if canonical_hash(obj.to_record()) != certificate["instance_hash"]:
        return vacuous("Qualities, affinities, masses, or parameters differ from the certified objective.")

    # Every reference block is seeded by one edge in a maximum matching.
    # A remaining matchable candidate must touch at least one matched GT: if it
    # touched only an unmatched GT, that edge would augment the matching.
    matched_gt = sorted(set(full_match.values()), key=lambda g: int(record.gt_ids[g]))
    block_by_gt = {g: block for block, g in enumerate(matched_gt)}
    block_of = np.full(len(q), -1, dtype=np.int64)
    for candidate, gt in full_match.items():
        block_of[candidate] = block_by_gt[gt]
    eligible = bbox_iou(record.boxes, record.gt_boxes) >= iou_threshold
    eligible &= record.classes[:, None] == record.gt_classes[None, :]
    eligible &= ~record.gt_ignore[None, :]
    matchable = eligible.any(axis=1)
    next_background = h
    for candidate in range(len(q)):
        if block_of[candidate] >= 0:
            continue
        adjacent = [g for g in matched_gt if eligible[candidate, g]]
        if adjacent:
            block_of[candidate] = block_by_gt[adjacent[0]]
        else:
            if matchable[candidate]:
                raise AssertionError("Maximum-matching reference-block construction failed.")
            block_of[candidate] = next_background
            next_background += 1
    blocks = [[i for i in range(len(q)) if block_of[i] == block] for block in range(next_background)]
    qualities = [_fraction(qi) if matchable[i] else Fraction(0) for i, qi in enumerate(q)]
    eta = _fraction(parameter(config, "eta", .2))
    rho = _fraction(parameter(config, "rho", .08))
    lam = _fraction(parameter(config, "lam", 1.25))

    def ideal_value(chosen):
        chosen = set(chosen)
        value = sum((qualities[i]-rho for i in chosen), Fraction(0))
        for block in blocks:
            present = [i for i in block if i in chosen]
            if present:
                value += eta*max(qualities[i] for i in present)
                value -= lam*sum((min(qualities[i], qualities[j])
                                  for i, j in itertools.combinations(present, 2)), Fraction(0))
        return value

    positive = []
    for block in range(h):
        best = min(blocks[block], key=lambda i: (-qualities[i], int(record.ids[i])))
        value = (1+eta)*qualities[best]-rho
        if value > 0:
            positive.append((value, best, block))
    positive.sort(key=lambda item: (-item[0], int(record.ids[item[1]])))
    qpositive = min(int(budget), len(positive))
    active = positive[:qpositive]
    reference_optimum = sorted(item[1] for item in active)

    # Each |F-F0| is bounded by |Fdelta-F0|+epsilon_g. The returned gap is
    # reconstructed from exact integer certificate fields, never its float view.
    eps2 = int(certificate["epsilon_twice_grid_units"])
    epsilon = Fraction(eps2, 2*GRID_SCALE)
    gap = Fraction(int(certificate["upper_grid"])-int(certificate["lower_grid"])+eps2, GRID_SCALE)
    d_selected = abs(Fraction(obj.value(selected), GRID_SCALE)-ideal_value(selected))+epsilon
    d_reference = abs(Fraction(obj.value(reference_optimum), GRID_SCALE)-ideal_value(reference_optimum))+epsilon
    radius = d_selected+gap+d_reference
    tail, tstar = Fraction(0), 0
    for value, _, _ in reversed(active):
        if tail+value > radius:
            break
        tail += value
        tstar += 1
    bound = min(attainable, attainable-qpositive+tstar)
    if bound < attainable-count:
        raise AssertionError("Computed matching-miss bound is below the realized allocation loss.")
    return {**base, "bound": int(bound), "nonvacuous": bool(bound < attainable),
            "certificate_valid": True, "status": "checked", "reason": None,
            "Qpositive": qpositive, "positive_block_count": len(positive),
            "reference_block_count": h, "background_private_block_count": next_background-h,
            "reference_optimum": reference_optimum,
            "reference_optimum_ids": [int(record.ids[i]) for i in reference_optimum],
            "block_assignment": block_of.tolist(), "matched_gt_ids": [int(record.gt_ids[g]) for g in matched_gt],
            "t_star": tstar, "price_induced_loss": attainable-qpositive,
            "R_upper": _encoded(radius), "d_selected_upper": _encoded(d_selected),
            "d_reference_upper": _encoded(d_reference), "objective_gap": _encoded(gap),
            "epsilon_grid_transfer": _encoded(epsilon),
            "positive_values": [_encoded(value) for value, _, _ in active]}
