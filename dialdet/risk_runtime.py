"""Run all 24 complete deployed policies on a fixed calibration population."""
from __future__ import annotations

from dataclasses import replace, asdict
from pathlib import Path
import numpy as np

from .data import (load_manifest, manifest_document, record_hash, atomic_json, sha256_file,
                   dataset_quality_mode, classify_synthetic)
from .metrics import audit_record
from .risk import configuration_grid
from .solver import allocate, SolverConfig
from .provenance import (build_complete_policy, canonical_sha256,
                         replace_policy_solver, seal_policy_records)


def collect_policy_records(manifest, checkpoint, output, budget=100, device="cpu", config=None,
                           quality_mode="auto"):
    from .training import load_evidence_checkpoint, predict_record
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < 1:
        raise ValueError("Risk calibration uses a positive fixed report budget")
    config = config or SolverConfig()
    records = load_manifest(manifest, "calibration")
    if len(records) < 2:
        raise ValueError("At least two calibration images are required; paper protocol uses 2000")
    model, metadata = load_evidence_checkpoint(checkpoint, device)
    document = manifest_document(manifest)
    if quality_mode == "auto":
        quality_mode = (dataset_quality_mode(document["dataset"]) or document.get("quality_mode")
                        or metadata["training_config"]["quality_mode"])
    if quality_mode not in ("coco", "iou50"):
        raise ValueError("quality_mode must be auto, coco or iou50")
    if config.p != float(metadata["training_config"]["p"]):
        raise ValueError("Calibration exponent differs from the trained mass exponent")
    ids = {r.image_id for r in records}
    fitting = set(metadata["train_image_ids"]) | set(metadata["dev_image_ids"])
    if ids & fitting:
        raise ValueError("Risk calibration images overlap model fitting or temperature tuning")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base_policy = build_complete_policy(records, checkpoint, metadata, config, budget, quality_mode)
    population = {item["image_id"]: item for item in base_policy["population"]["records"]}
    policies = [{**item, "records": [],
                 "complete_policy": replace_policy_solver(base_policy, replace(config, **item))}
                for item in configuration_grid()]
    for image_index, record in enumerate(records):
        ev = predict_record(record, model, temperature=metadata.get("temperature", 1.),
                            p=config.p, quality_mode=quality_mode, device=device)
        q, r = np.asarray(ev["q"], dtype=np.float64), np.asarray(ev["r"], dtype=np.float64)
        mass = np.power(r, config.p).sum(axis=0)
        for policy_index, policy in enumerate(policies):
            setting = replace(config, rho=policy["rho"], lam=policy["lam"])
            result = allocate(q, r, budget, record.ids.tolist(), setting, mass,
                              image_id=record.image_id, candidate_export_hash=record_hash(record))
            row = audit_record(record, result["selected"], budget, status=result["status"])
            row.update(policy_sha256=policy["complete_policy"]["policy_sha256"],
                       candidate_numerical_sha256=population[record.image_id]["numerical_sha256"],
                       candidate_semantic_sha256=population[record.image_id]["semantic_sha256"],
                       certificate_sha256=canonical_sha256(result["certificate"]))
            # Fallback contributes its actual returned membership, never omitted.
            policy["records"].append(row)
            atomic_json(output / "certificates" / f"image{image_index:06d}_policy{policy_index:02d}.json",
                        {"certificate": result["certificate"], "status": result["status"],
                         "fallback_reason": result.get("fallback_reason")})
        print(f"[RISK] {image_index+1}/{len(records)} images, complete 24-policy family", flush=True)
    for policy in policies:
        policy["records_sha256"] = seal_policy_records(policy)
    destination = output / "policy_records.json"
    atomic_json(destination, policies)
    result = {"policy_records": str(destination.resolve()), "n_images": len(records), "K": budget,
              "hypotheses": 48, "checkpoint_sha256": sha256_file(checkpoint),
              "manifest_sha256": sha256_file(manifest), "solver_base": asdict(config),
              "complete_policy_including_fallback": True,
              "calibration_population_sha256": base_policy["population"]["population_sha256"],
              "policy_family_sha256": canonical_sha256(sorted(policy["complete_policy"]["policy_sha256"]
                                                              for policy in policies)),
              "policy_records_sha256": sha256_file(destination),
              "policy_bindings_verified": True,
              "is_synthetic": classify_synthetic(document, records),
              "protocol": document.get("protocol"), "dataset": document["dataset"], "quality_mode": quality_mode,
              "note": "This file records actual calibration losses. Use risk-report with independently fixed development scores and verified population provenance; no guarantee is granted by data collection alone."}
    atomic_json(output / "RISK_COLLECTION.json", result)
    return result

