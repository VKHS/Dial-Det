"""Measure development utility of a fixed, explicitly identified 24-rule family."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import source_fingerprint
from .data import atomic_json, load_manifest, manifest_document, record_hash, sha256_file, classify_synthetic
from .metrics import audit_record, coco_predictions, official_coco_evaluation
from .provenance import (canonical_sha256, export_protocol, population_descriptor,
                         seal_development_score, verify_complete_policy)
from .risk import configuration_grid
from .solver import SolverConfig, allocate


def score_policy_family(manifest, checkpoint, policy_records, output, device="cpu",
                        coco_annotations=None, crowdhuman_annotations=None,
                        crowdhuman_evaluator=None, allow_matching_proxy=False):
    """Execute all registered rules on checkpoint development images.

    ``policy_records`` is a JSON list of {rho,lam,complete_policy}; collection
    files may also contain calibration rows, which this function never uses.
    A registry's creation time cannot prove preregistration. The caller must
    establish the policy family and independence before inspecting risk losses.
    Different out-of-fold development detector exports are explicitly recorded;
    they are not relabeled as the deployed calibration detector.
    """
    from .training import load_evidence_checkpoint, predict_record
    registry_path = Path(policy_records)
    registry = json.loads(registry_path.read_text())
    expected = {(item["rho"], item["lam"]) for item in configuration_grid()}
    if not isinstance(registry, list):
        raise ValueError("Policy registry must be a list")
    keys = [(float(item["rho"]), float(item["lam"])) for item in registry]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Development scoring requires the complete unique 24-policy family")
    descriptors = []
    reference, reference_population = None, None
    for item in registry:
        descriptor = item["complete_policy"]
        verify_complete_policy(descriptor)
        definition = descriptor["definition"]
        if (definition["solver"]["rho"], definition["solver"]["lam"]) != (float(item["rho"]), float(item["lam"])):
            raise ValueError("Registry label differs from complete policy")
        fixed = {**definition, "solver": {key: value for key, value in definition["solver"].items()
                                          if key not in {"rho", "lam"}}}
        if reference is None:
            reference, reference_population = fixed, descriptor["population"]
        elif fixed != reference or descriptor["population"] != reference_population:
            raise ValueError("Registered family differs beyond the predeclared rho/lambda grid")
        descriptors.append(descriptor)
    if reference["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("Development checkpoint differs from the registered deployed policy")
    if reference["implementation_sha256"] != source_fingerprint()["sha256"]:
        raise ValueError("Development source differs from the registered deployed policy")
    document = manifest_document(manifest)
    records = load_manifest(manifest, "dev")
    if not records:
        raise ValueError("Development scoring requires development images")
    model, metadata = load_evidence_checkpoint(checkpoint, device)
    if metadata["model_config"] != reference["model_config"] or float(metadata.get("temperature", 1.)) != reference["identity_temperature"]:
        raise ValueError("Development model or temperature differs from registered policy")
    for role in ("train", "dev"):
        if set(metadata[f"{role}_image_ids"]) != set(reference["fitting_population_ids"][role]):
            raise ValueError("Registered fitting populations differ from the checkpoint")
    if float(metadata["training_config"]["p"]) != float(reference["solver"]["p"]):
        raise ValueError("Development mass exponent differs from trained evidence model")
    dev_ids = {record.image_id for record in records}
    if dev_ids != set(metadata["dev_image_ids"]):
        raise ValueError("Development population must equal the checkpoint's reserved development IDs")
    if dev_ids & set(metadata["train_image_ids"]):
        raise ValueError("Development and training populations overlap")
    if dev_ids & {entry["image_id"] for entry in reference_population["records"]}:
        raise ValueError("Development and registered target populations overlap")
    if coco_annotations and (crowdhuman_annotations or crowdhuman_evaluator):
        raise ValueError("Choose one dataset evaluator")
    if bool(crowdhuman_annotations) != bool(crowdhuman_evaluator):
        raise ValueError("CrowdHuman scoring needs both annotations and official evaluator checkout")
    if coco_annotations and document["dataset"] != "coco":
        raise ValueError("COCO AP requires a COCO development manifest")
    if crowdhuman_annotations and document["dataset"] != "crowdhuman":
        raise ValueError("CrowdHuman AP requires a CrowdHuman development manifest")
    metric = ("official_coco_AP" if coco_annotations else "official_crowdhuman_AP50"
              if crowdhuman_annotations else "matching_proxy_attainable_recall")
    if metric.startswith("matching_proxy") and not allow_matching_proxy:
        raise ValueError("Supply the official evaluator, or explicitly enable a diagnostic matching proxy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    population = population_descriptor(records)
    selections = [[] for _ in descriptors]
    audit_rows = [[] for _ in descriptors]
    budget, quality = reference["budget"], reference["quality_mode"]
    configs = [SolverConfig(**descriptor["definition"]["solver"]) for descriptor in descriptors]
    for index, record in enumerate(records):
        evidence = predict_record(record, model, temperature=reference["identity_temperature"],
                                  p=configs[0].p, quality_mode=quality, device=device)
        q, r = np.asarray(evidence["q"], dtype=np.float64), np.asarray(evidence["r"], dtype=np.float64)
        mass = np.power(r, configs[0].p).sum(axis=0)
        for policy_index, config in enumerate(configs):
            result = allocate(q, r, budget, record.ids.tolist(), config, mass,
                              image_id=record.image_id, candidate_export_hash=record_hash(record))
            selected = result["selected"]
            selections[policy_index].append(np.asarray(selected, dtype=np.int64))
            row = audit_record(record, selected, budget, status=result["status"],
                               iou_threshold=reference["audit"]["iou"])
            row["selected_ids"] = record.ids[selected].tolist()
            audit_rows[policy_index].append(row)
        print(f"[DEVELOPMENT] {index+1}/{len(records)} images, fixed 24-policy family", flush=True)
    scores = []
    for index, descriptor in enumerate(descriptors):
        directory = output / f"policy_{index:02d}"
        atomic_json(directory / "selections.json", audit_rows[index])
        if coco_annotations:
            predictions = [item for record, selected in zip(records, selections[index])
                           for item in coco_predictions(record, selected)]
            evaluation = official_coco_evaluation(coco_annotations, predictions,
                                                  [record.image_id for record in records], budget)
            score = evaluation["metrics"]["AP"]
            atomic_json(directory / "EVALUATION.json", evaluation)
        elif crowdhuman_annotations:
            from .crowdhuman import crowdhuman_predictions, official_crowdhuman_evaluation
            predictions = [crowdhuman_predictions(record, selected)
                           for record, selected in zip(records, selections[index])]
            evaluation = official_crowdhuman_evaluation(crowdhuman_annotations, predictions,
                         [row["ID"] for row in predictions], budget, crowdhuman_evaluator, directory)
            score = evaluation["metrics"]["AP50"]
        else:
            denominator = sum(row["Q"] for row in audit_rows[index])
            score = sum(row["C"] for row in audit_rows[index]) / denominator if denominator else 1.
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Development evaluation has no finite, defined utility in [0,1]")
        scores.append(seal_development_score(descriptor, score, population, metric))
    scores_path = output / "development_scores.json"
    atomic_json(scores_path, scores)
    contracts = {canonical_sha256(export_protocol(record)): export_protocol(record) for record in records}
    result = {"schema": "dialdet-development-family-v1", "development_scores": str(scores_path.resolve()),
              "development_scores_sha256": sha256_file(scores_path), "metric": metric,
              "official_metric": not metric.startswith("matching_proxy"), "n_images": len(records),
              "is_synthetic": classify_synthetic(document, records),
              "policy_count": len(scores), "budget": budget, "quality_mode": quality,
              "checkpoint_sha256": sha256_file(checkpoint), "registry_sha256": sha256_file(registry_path),
              "development_manifest_sha256": sha256_file(manifest), "development_population": population,
              "development_export_protocols": [contracts[key] for key in sorted(contracts)],
              "target_export_protocol": reference["candidate_export_protocol"],
              "annotation_sha256": sha256_file(coco_annotations or crowdhuman_annotations)
                                    if (coco_annotations or crowdhuman_annotations) else None,
              "note": "All 24 scores use development data only. Registry calibration losses are ignored. Predeclaration and detector independence remain externally established conditions. Matching-proxy scores cannot enable risk-certificate eligibility."}
    atomic_json(output / "DEVELOPMENT_FAMILY.json", result)
    return result
