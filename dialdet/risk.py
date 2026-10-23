"""Appendix C.9 finite-family image-weighted risk bounds.

Numerical bounds alone are not an exchangeability/provenance certificate.
All branches of the deployed policy, including fallback, must be evaluated.
"""
from __future__ import annotations
from itertools import product
import math
import re
import numpy as np

from .provenance import (canonical_sha256, seal_policy_records, verify_complete_policy,
                         verify_population)

RHO_GRID = (.02, .04, .06, .08, .12, .16)
LAMBDA_GRID = (1., 1.25, 1.5, 2.)


def configuration_grid():
    return [{"rho": rho, "lam": lam} for rho, lam in product(RHO_GRID, LAMBDA_GRID)]


def empirical_bernstein(losses, alpha=.05, hypotheses=48):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1 or len(losses) < 2:
        raise ValueError("At least two independent image losses are required")
    if not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > 1):
        raise ValueError("Every loss must lie in [0,1]")
    if not 0 < alpha < 1 or hypotheses < 1:
        raise ValueError("Invalid confidence/family size")
    n = len(losses)
    mean, variance = float(losses.mean()), float(losses.var(ddof=1))
    h = math.log(2 * hypotheses / alpha)
    raw = mean + math.sqrt(2 * variance * h / n) + 7 * h / (3 * (n - 1))
    return {"n": n, "mean": mean, "sample_variance": variance, "raw_upper": raw,
            "upper": min(1., raw), "alpha": alpha, "hypotheses": hypotheses}


def _verify_policy_bindings(policy_rows, development_scores, provenance):
    """Check the recorded identity chain; no external history is inferred."""
    present = ["complete_policy" in policy or "records_sha256" in policy for policy in policy_rows]
    if any(present) and not all(present):
        raise ValueError("Cannot mix sealed and legacy calibration policies")
    development_has_binding = ["policy_sha256" in score for score in development_scores]
    if any(development_has_binding) and not all(development_has_binding):
        raise ValueError("Cannot mix bound and unbound development policy scores")
    if not any(present):
        if any(development_has_binding):
            raise ValueError("Bound development scores require complete calibration policy identities")
        return {"verified": False, "development_verified": False,
                "status": "legacy_unsealed_records", "population_sha256": None,
                "family_sha256": None}

    scores = {(float(item["rho"]), float(item["lam"])): item for item in development_scores}
    common_definition, common_population, common_development, common_metric = None, None, None, None
    development_verified = all(development_has_binding)
    official_development_metric = True
    policy_hashes = []
    for policy in policy_rows:
        descriptor = policy.get("complete_policy")
        verify_complete_policy(descriptor)
        if policy.get("records_sha256") != seal_policy_records(policy):
            raise ValueError("Calibration policy records hash mismatch")
        definition = descriptor["definition"]
        key = (float(policy["rho"]), float(policy["lam"]))
        if (float(definition["solver"]["rho"]), float(definition["solver"]["lam"])) != key:
            raise ValueError("Calibration policy identity differs from its rho/lambda label")
        fixed = {**definition, "solver": {name: value for name, value in definition["solver"].items()
                                          if name not in {"rho", "lam"}}}
        if common_definition is None:
            common_definition = fixed
            common_population = descriptor["population"]
        elif fixed != common_definition or descriptor["population"] != common_population:
            raise ValueError("Calibration family mixes checkpoint, policy, or candidate population identities")
        population_entries = {item["image_id"]: item for item in descriptor["population"]["records"]}
        rows = policy["records"]
        if len(rows) != len(population_entries) or {str(row["image_id"]) for row in rows} != set(population_entries):
            raise ValueError("Calibration rows differ from the bound candidate population")
        for row in rows:
            image_id = str(row["image_id"])
            entry = population_entries[image_id]
            if entry["role"] != "calibration" or row.get("role") != "calibration":
                raise ValueError("Bound risk records must have calibration role")
            if (row.get("policy_sha256") != descriptor["policy_sha256"]
                    or row.get("candidate_numerical_sha256") != entry["numerical_sha256"]
                    or row.get("candidate_semantic_sha256") != entry["semantic_sha256"]):
                raise ValueError("Calibration row candidate or policy identity mismatch")
            if row.get("K") != definition["budget"]:
                raise ValueError("Calibration row budget differs from its complete policy")
            if row.get("matching_iou") != definition["audit"]["iou"]:
                raise ValueError("Calibration row matcher differs from its complete policy")
            if row.get("matching_convention") != definition["audit"]["matching"]:
                raise ValueError("Calibration row matching convention differs from its policy")
            if row.get("status") not in {"rounded_optimum", "certified", "uncertified", "fallback"}:
                raise ValueError("Calibration row omits a valid deployed-policy status")
        if all(development_has_binding) and scores[key]["policy_sha256"] != descriptor["policy_sha256"]:
            raise ValueError("Development score is bound to a different complete policy")
        score = scores[key]
        if "score_sha256" in score or "development_population" in score:
            if not all(development_has_binding):
                raise ValueError("Sealed development scores require complete policy hashes")
            score_payload = {name: value for name, value in score.items() if name != "score_sha256"}
            if score.get("score_sha256") != canonical_sha256(score_payload):
                raise ValueError("Development score hash mismatch")
            development = score.get("development_population")
            verify_population(development)
            expected_ids = set(definition.get("fitting_population_ids", {}).get("dev", []))
            if (not expected_ids or {item["image_id"] for item in development["records"]} != expected_ids
                    or any(item["role"] != "dev" for item in development["records"])):
                raise ValueError("Development score population differs from checkpoint development IDs")
            if set(population_entries) & expected_ids:
                raise ValueError("Calibration and development populations overlap")
            if common_development is None:
                common_development = development
                common_metric = score.get("metric")
            elif development != common_development:
                raise ValueError("Development scores mix evaluated populations")
            if score.get("metric") != common_metric:
                raise ValueError("Development scores mix selection metrics")
            if score.get("metric") not in {"official_coco_AP", "official_crowdhuman_AP50"}:
                official_development_metric = False
        else:
            development_verified = False
            official_development_metric = False
        policy_hashes.append(descriptor["policy_sha256"])

    encoder_hash = common_definition["checkpoint_sha256"]
    if provenance.get("encoder_sha256") is not None and provenance["encoder_sha256"] != encoder_hash:
        raise ValueError("Declared encoder hash differs from collected policy checkpoint")
    contract = common_definition["candidate_export_protocol"]
    detector_hashes = {contract[key] for key in ("checkpoint_sha256", "detector_checkpoint_sha256", "detector_state_sha256")
                       if isinstance(contract.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", contract[key])}
    detector_verified = bool(detector_hashes and provenance.get("detector_sha256") in detector_hashes)
    if detector_hashes and provenance.get("detector_sha256") is not None and not detector_verified:
        raise ValueError("Declared detector hash differs from the frozen candidate export")
    for role in ("train", "dev"):
        checkpoint_ids = common_definition.get("fitting_population_ids", {}).get(role, [])
        if role + "_ids" in provenance and set(map(str, provenance[role + "_ids"])) != set(checkpoint_ids):
            raise ValueError(f"Declared {role} IDs differ from the evidence checkpoint")
    population_hash = common_population["population_sha256"]
    if (provenance.get("calibration_population_sha256") is not None
            and provenance["calibration_population_sha256"] != population_hash):
        raise ValueError("Declared calibration population hash differs from collected records")
    return {"verified": True, "development_verified": development_verified,
            "official_development_metric": official_development_metric,
            "detector_verified": detector_verified, "status": "sealed_complete_policy_records",
            "synthetic": contract.get("synthetic") is True or contract.get("dataset") == "synthetic",
            "population_sha256": population_hash,
            "family_sha256": canonical_sha256(sorted(policy_hashes))}


def evaluate_family(policy_rows, development_scores, provenance, alpha=.05,
                    waste_target=.05, shortfall_target=.10):
    """Select only from the preregistered 24-pair family; AP supplied from dev.

    policy_rows: list of {rho,lam,records:[per-image audit_record dictionaries]}.
    development_scores: list of {rho,lam,score}, fixed without calibration data.
    provenance: explicit data lineage, never inferred from a dataset name.
    """
    expected = {(float(c["rho"]), float(c["lam"])) for c in configuration_grid()}
    actual = [(float(c["rho"]), float(c["lam"])) for c in policy_rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("The complete predeclared 24-policy family is required")
    scores = {(float(c["rho"]), float(c["lam"])): float(c["score"]) for c in development_scores}
    if len(development_scores) != len(expected) or len(scores) != len(development_scores) or set(scores) != expected or not np.isfinite(list(scores.values())).all():
        raise ValueError("One finite development score per policy is required")
    if not 0 <= waste_target <= 1 or not 0 <= shortfall_target <= 1:
        raise ValueError("Risk targets must lie in [0,1]")
    if not provenance.get("complete_policy_including_fallback", False):
        raise ValueError("Risk evaluation must include uncertified and fallback outputs")
    binding = _verify_policy_bindings(policy_rows, development_scores, provenance)
    populations, results = [], []
    fixed_budget, fixed_attainable = None, None
    for policy in policy_rows:
        rows = policy["records"]
        ids = [str(row["image_id"]) for row in rows]
        if len(set(ids)) != len(ids) or any(row.get("role") != "calibration" for row in rows):
            raise ValueError("Calibration records need unique image IDs and role=calibration")
        populations.append(set(ids))
        budgets = {row["K"] for row in rows}
        if len(budgets) != 1 or any(not isinstance(k, (int, np.integer)) or k < 1 for k in budgets):
            raise ValueError("Every calibration image must use the same positive integer fixed budget")
        budget = next(iter(budgets))
        attainable = {str(row["image_id"]): (row["Q"], row.get("H")) for row in rows}
        if fixed_budget is None:
            fixed_budget, fixed_attainable = budget, attainable
        elif budget != fixed_budget or attainable != fixed_attainable:
            raise ValueError("Budget K and per-image attainable Q must stay fixed across the policy family")
        for row in rows:
            if any(not isinstance(row.get(name), (int, np.integer)) or isinstance(row[name], bool)
                   for name in ("K", "Q", "C", "selected_count")):
                raise ValueError("Calibration counts must be integers")
            if not all(math.isfinite(float(row.get(name, math.nan))) for name in ("waste", "shortfall")):
                raise ValueError("Calibration risks must be finite")
            if "H" in row:
                if (not isinstance(row["H"], (int, np.integer)) or isinstance(row["H"], bool)
                        or row["H"] < 0 or row["Q"] != min(row["K"], row["H"])):
                    raise ValueError("Calibration attainable count must satisfy Q=min(K,H)")
                if "G" in row and (not isinstance(row["G"], (int, np.integer))
                                   or isinstance(row["G"], bool) or row["G"] < row["H"]):
                    raise ValueError("Calibration ground-truth count must satisfy G>=H")
            if not 0 <= row["C"] <= row["Q"] <= row["K"] or not row["C"] <= row["selected_count"] <= row["K"]:
                raise ValueError("Invalid calibration matching/budget record")
            rw = (row["selected_count"] - row["C"]) / row["K"]
            ra = (row["Q"] - row["C"]) / row["Q"] if row["Q"] else 0.
            if abs(rw - row["waste"]) > 1e-12 or abs(ra - row["shortfall"]) > 1e-12:
                raise ValueError("Stored calibration loss differs from its definition")
        waste = empirical_bernstein([r["waste"] for r in rows], alpha, 48)
        shortfall = empirical_bernstein([r["shortfall"] for r in rows], alpha, 48)
        key = (float(policy["rho"]), float(policy["lam"]))
        results.append({"rho": key[0], "lam": key[1], "waste": waste, "shortfall": shortfall,
                        "feasible": waste["upper"] <= waste_target and shortfall["upper"] <= shortfall_target,
                        "development_score": scores[key]})
    if any(s != populations[0] for s in populations):
        raise ValueError("Every configuration must use exactly the same calibration images")
    forbidden = set(map(str, provenance.get("train_ids", []) + provenance.get("dev_ids", []) + provenance.get("audit_ids", [])))
    if populations[0] & forbidden:
        raise ValueError("Calibration overlaps training, development or audit")
    eligible = (binding["verified"] and binding["development_verified"] and binding.get("detector_verified", False)
                and binding.get("official_development_metric", False)
                and not binding.get("synthetic", False)
                and provenance.get("protocol") == "independent_focal_detector"
                and provenance.get("iid_deployment_population") is True
                and provenance.get("encoder_fixed_before_calibration") is True
                and provenance.get("detector_unseen_calibration") is True
                and provenance.get("policy_family_fixed_before_calibration") is True
                and provenance.get("budget_fixed_before_calibration") is True
                and all(key in provenance for key in ("train_ids", "dev_ids", "audit_ids"))
                and all(isinstance(provenance.get(key), str) and re.fullmatch(r"[0-9a-fA-F]{64}", provenance[key])
                        for key in ("encoder_sha256", "detector_sha256")))
    feasible = [r for r in results if r["feasible"]]
    chosen = max(feasible, key=lambda r: (r["development_score"], -r["rho"], -r["lam"])) if feasible else None
    return {"policies": results, "n_feasible": len(feasible), "selected": chosen,
            "targets": {"waste": waste_target, "shortfall": shortfall_target},
            "policy_binding": binding,
            "risk_certificate_eligible": bool(eligible),
            "status": ("conditional_on_declared_provenance" if eligible and chosen else
                       "no_feasible_policy" if not chosen else
                       "bounds_only_unverified_policy_binding" if not (binding["verified"] and binding["development_verified"]) else
                       "bounds_only_unverified_population"),
            "assumptions": "The IID and training-history declarations require external verification; this code does not prove them."}
