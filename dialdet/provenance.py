"""Canonical experiment identities and read-only candidate preflight.

Hashes detect changed inputs; they do not prove a declared detector training
history or the independence of a population. Policy identity is separate from
the image population, so the same deployed rule can be compared on development,
calibration and audit images without changing its policy hash.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .data import (canonical_dataset, load_record, manifest_document, record_hash,
                   sha256_file, classify_synthetic)


_TRANSIENT_METADATA = {
    "role", "source_path", "protocol", "image_file", "image_root",
    "annotations_file", "checkpoint", "checkpoint_path", "output_path",
}
_PER_IMAGE_PREPROCESSING = {"original_size", "transformed_size", "image_size"}
_EXPORT_FIELDS = {
    "dataset", "synthetic", "exporter", "detector", "detector_id", "architecture",
    "checkpoint_sha256", "detector_checkpoint_sha256", "detector_state_sha256",
    "weights_enum", "torch_version", "torchvision_version", "export_software_version",
    "final_detection_nms_applied", "rpn_nms_preserved", "native_score_floor",
    "score_floor_relation", "max_candidates", "query_class_expansion",
    "exported_category_ids", "candidate_id_definition", "candidate_order",
    "detector_num_classes", "person_class_id", "appearance_feature", "feature_taps",
    "preprocessing", "class_mapping", "annotation_box_type", "gt_ignore_convention",
    "coordinate_origin", "citypersons_subset", "export_contract",
}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical identity dictionaries require string keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Canonical identities require finite JSON-compatible values")


def canonical_sha256(value):
    data = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def semantic_export_metadata(record):
    """Preserve semantic metadata while allowing files to move and roles to change.

    File content hashes remain included. A named checkpoint without a content
    hash remains a declaration, recorded separately rather than interpreted as
    a verified detector identity.
    """
    metadata = {key: value for key, value in record.metadata.items()
                if key not in _TRANSIENT_METADATA}
    if "dataset" in metadata:
        metadata["dataset"] = canonical_dataset(metadata["dataset"])
    if (record.metadata.get("checkpoint") and not any(record.metadata.get(key) for key in
            ("checkpoint_sha256", "detector_checkpoint_sha256", "detector_state_sha256"))):
        metadata["checkpoint_reference_unverified"] = str(record.metadata["checkpoint"])
    return _plain(metadata)


def candidate_semantic_hash(record):
    record.validate()
    return canonical_sha256({"schema": "dialdet-candidate-semantic-v1",
                             "numerical_sha256": record_hash(record),
                             "export_metadata": semantic_export_metadata(record)})


def export_protocol(record):
    """Extract population-independent export settings from known adapter fields.

    Custom adapters should put their complete interface in ``export_contract``.
    All other metadata is still bound by the per-image semantic hash.
    """
    metadata = semantic_export_metadata(record)
    contract = {key: value for key, value in metadata.items() if key in _EXPORT_FIELDS}
    if isinstance(contract.get("preprocessing"), dict):
        contract["preprocessing"] = {key: value for key, value in contract["preprocessing"].items()
                                     if key not in _PER_IMAGE_PREPROCESSING}
    contract["feature_dimension"] = int(record.features.shape[1])
    contract["coordinate_convention"] = "original_image_xyxy"
    contract["native_candidate_values_immutable"] = True
    if "checkpoint_reference_unverified" in metadata:
        contract["checkpoint_reference_unverified"] = metadata["checkpoint_reference_unverified"]
    return contract


def population_descriptor(records):
    entries = []
    seen = set()
    for record in records:
        if record.image_id in seen:
            raise ValueError("Population contains duplicate image IDs")
        seen.add(record.image_id)
        entries.append({"image_id": record.image_id,
                        "role": record.metadata.get("role", "unspecified"),
                        "numerical_sha256": record_hash(record),
                        "semantic_sha256": candidate_semantic_hash(record)})
    entries.sort(key=lambda entry: entry["image_id"])
    payload = {"schema": "dialdet-population-v1", "records": entries}
    return {**payload, "n_images": len(entries), "population_sha256": canonical_sha256(payload)}


def verify_population(population):
    if not isinstance(population, dict) or population.get("schema") != "dialdet-population-v1":
        raise ValueError("Unknown population identity schema")
    entries = population.get("records")
    if not isinstance(entries, list) or population.get("n_images") != len(entries):
        raise ValueError("Population record count differs from its descriptor")
    ids = [entry.get("image_id") for entry in entries]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Population needs unique nonempty string image IDs")
    if entries != sorted(entries, key=lambda entry: entry["image_id"]):
        raise ValueError("Population records must use canonical image-ID order")
    for entry in entries:
        for key in ("numerical_sha256", "semantic_sha256"):
            _require_digest(entry.get(key), key)
    expected = canonical_sha256({"schema": population["schema"], "records": entries})
    if population.get("population_sha256") != expected:
        raise ValueError("Population hash mismatch")
    return True


def inspect_manifest(path, role=None):
    """Validate every source file, streaming one record at a time.

    ``role`` selects the returned population only; all manifest entries are
    inspected. No model, GPU, dataset download or checkpoint loading is needed.
    Missing file hashes are reported explicitly, never silently called verified.
    """
    path = Path(path)
    document = manifest_document(path)
    entries, role_counts, feature_dims, protocols = [], Counter(), set(), {}
    candidate_counts = []
    file_hashes = []
    synthetic = classify_synthetic(document)
    for entry in document["records"]:
        source = (path.parent / entry["path"]).resolve()
        file_digest = sha256_file(source)
        if entry.get("sha256") and file_digest != entry["sha256"]:
            raise ValueError(f"Candidate export hash mismatch: {source}")
        record = load_record(source)
        if classify_synthetic(document, [record]) is True:
            synthetic = True
        if record.image_id != str(entry["image_id"]):
            raise ValueError("Manifest/record image-ID mismatch")
        record.metadata = {**record.metadata, "role": entry["role"], "dataset": document["dataset"],
                           "protocol": document.get("protocol", "unspecified"), "source_path": str(source)}
        semantic = candidate_semantic_hash(record)
        numerical = record_hash(record)
        for key, actual in (("semantic_sha256", semantic), ("numerical_sha256", numerical)):
            if entry.get(key) and entry[key] != actual:
                raise ValueError(f"Candidate {key} mismatch: {record.image_id}")
        role_counts[entry["role"]] += 1
        file_hashes.append({"image_id": record.image_id, "sha256": file_digest,
                            "declared_sha256_verified": bool(entry.get("sha256"))})
        if role is None or entry["role"] == role:
            feature_dims.add(int(record.features.shape[1]))
            candidate_counts.append(len(record.ids))
            contract = export_protocol(record)
            protocols[canonical_sha256(contract)] = contract
            entries.append({"image_id": record.image_id, "role": entry["role"],
                            "numerical_sha256": numerical, "semantic_sha256": semantic})
    if not entries:
        raise ValueError("The selected manifest role has no images")
    if len(feature_dims) != 1:
        raise ValueError("Selected candidate population mixes feature dimensions")
    entries.sort(key=lambda entry: entry["image_id"])
    population = {"schema": "dialdet-population-v1", "records": entries}
    population = {**population, "n_images": len(entries), "population_sha256": canonical_sha256(population)}
    return {"schema": "dialdet-manifest-preflight-v1", "valid": True,
            "is_synthetic": synthetic,
            "manifest_sha256": sha256_file(path), "dataset": document["dataset"],
            "protocol": document.get("protocol", "unspecified"), "selected_role": role,
            "role_counts": dict(role_counts), "n_images": len(entries),
            "feature_dimensions": sorted(feature_dims), "candidate_count_min": min(candidate_counts),
            "candidate_count_max": max(candidate_counts), "candidate_count_total": sum(candidate_counts),
            "empty_candidate_images": candidate_counts.count(0), "file_hashes": file_hashes,
            "all_declared_file_hashes_verified": all(item["declared_sha256_verified"] for item in file_hashes),
            "export_protocols": [protocols[key] for key in sorted(protocols)], "population": population,
            "claim": "Input integrity and shape validation only; detector history and independence are declarations."}


def build_complete_policy(records, checkpoint, checkpoint_metadata, solver_config, budget,
                          quality_mode, implementation_version=None, matching_iou=.5):
    from . import __version__
    from .artifacts import source_fingerprint
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < 1:
        raise ValueError("Policy budget must be a positive integer")
    if quality_mode not in {"coco", "iou50"} or not math.isfinite(matching_iou) or not 0 < matching_iou <= 1:
        raise ValueError("Invalid quality mode or matching threshold")
    records = list(records)
    if not records:
        raise ValueError("A complete policy requires a nonempty candidate population")
    contracts = {canonical_sha256(export_protocol(record)): export_protocol(record) for record in records}
    if len(contracts) != 1:
        raise ValueError("One deployed policy cannot mix candidate export protocols")
    model_config = checkpoint_metadata.get("model_config")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("Checkpoint model_config is required to identify a complete policy")
    temperature = float(checkpoint_metadata.get("temperature", 1.))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Identity temperature must be finite and positive")
    solver = _plain(solver_config)
    if not isinstance(solver, dict):
        raise ValueError("solver_config must be a mapping or dataclass")
    support_keys = ("overlap_threshold", "rescue_k", "distance_gate", "rescue_same_class",
                    "use_appearance", "use_rescue")
    definition = {
        "implementation_version": implementation_version or __version__,
        "implementation_sha256": source_fingerprint()["sha256"],
        "method": "DIAL-Det_ICASSP_identity_mass", "checkpoint_sha256": sha256_file(checkpoint),
        "model_config": model_config, "identity_temperature": temperature,
        "fitting_population_ids": {role: sorted(map(str, checkpoint_metadata.get(f"{role}_image_ids", [])))
                                   for role in ("train", "dev")},
        "quality_mode": quality_mode, "candidate_export_protocol": next(iter(contracts.values())),
        "support": {key: model_config.get(key) for key in support_keys},
        "solver": solver, "budget": int(budget), "constraint": "at_most_global_K",
        "identity_mass": "sum_unthresholded_calibrated_r_power_p_before_allocation_threshold",
        "ranking": {"membership_track": "native_detector_scores",
                    "complete_method_track": "evaluator_matched_quality_q", "tie_break": "candidate_id"},
        "fallback": {"trigger": "numerical_integrity_failure_only",
                     "large_valid_gap": "return_uncertified_incumbent",
                     "rule": "quality_ordered_local_raw_affinity", "threshold": solver.get("fallback_threshold")},
        "audit": {"matching": "maximum_class_consistent_nonignored", "iou": float(matching_iou),
                  "waste": "(selected_count-C)/K", "shortfall": "(min(K,H)-C)/min(K,H);zero_if_Q_zero"},
    }
    definition = _plain(definition)
    population = population_descriptor(records)
    policy_hash = canonical_sha256(definition)
    result = {"schema": "dialdet-complete-policy-v1", "definition": definition,
              "policy_sha256": policy_hash, "population": population,
              "binding_sha256": canonical_sha256({"policy_sha256": policy_hash,
                                                  "population_sha256": population["population_sha256"]})}
    verify_complete_policy(result)
    return result


def verify_complete_policy(descriptor):
    if not isinstance(descriptor, dict) or descriptor.get("schema") != "dialdet-complete-policy-v1":
        raise ValueError("Unknown complete-policy schema")
    definition = descriptor.get("definition")
    if not isinstance(definition, dict) or descriptor.get("policy_sha256") != canonical_sha256(definition):
        raise ValueError("Complete-policy hash mismatch")
    required = {"implementation_version", "implementation_sha256", "checkpoint_sha256", "model_config", "quality_mode",
                "candidate_export_protocol", "support", "solver", "budget", "constraint",
                "identity_temperature", "identity_mass", "ranking", "fallback", "audit", "fitting_population_ids"}
    if not required <= set(definition):
        raise ValueError("Complete-policy definition is missing required fields")
    _require_digest(definition["checkpoint_sha256"], "checkpoint_sha256")
    _require_digest(definition["implementation_sha256"], "implementation_sha256")
    if definition["quality_mode"] not in {"coco", "iou50"} or definition["constraint"] != "at_most_global_K":
        raise ValueError("Unknown policy quality/constraint contract")
    if type(definition["budget"]) is not int or definition["budget"] < 1:
        raise ValueError("Policy budget must be a positive integer")
    fitting = definition["fitting_population_ids"]
    if not isinstance(fitting, dict) or set(fitting) != {"train", "dev"}:
        raise ValueError("Policy fitting population IDs must declare train and dev")
    for ids in fitting.values():
        if (not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Policy fitting populations need unique nonempty string IDs")
    if set(fitting["train"]) & set(fitting["dev"]):
        raise ValueError("Policy training and development populations overlap")
    from .solver import SolverConfig
    solver = definition["solver"]
    if not isinstance(solver, dict) or set(solver) != set(asdict(SolverConfig())):
        raise ValueError("Complete policy must enumerate every solver parameter")
    SolverConfig(**solver)
    fallback = definition["fallback"]
    if (fallback.get("trigger") != "numerical_integrity_failure_only"
            or fallback.get("large_valid_gap") != "return_uncertified_incumbent"
            or fallback.get("threshold") != solver["fallback_threshold"]):
        raise ValueError("Complete policy has an inconsistent fallback contract")
    audit = definition["audit"]
    if (audit.get("matching") != "maximum_class_consistent_nonignored"
            or not isinstance(audit.get("iou"), (int, float))
            or not math.isfinite(audit["iou"]) or not 0 < audit["iou"] <= 1):
        raise ValueError("Complete policy has an invalid matching contract")
    verify_population(descriptor.get("population"))
    expected = canonical_sha256({"policy_sha256": descriptor["policy_sha256"],
                                 "population_sha256": descriptor["population"]["population_sha256"]})
    if descriptor.get("binding_sha256") != expected:
        raise ValueError("Complete-policy population binding mismatch")
    return True


def replace_policy_solver(descriptor, solver_config):
    """Rebind a fixed candidate population to another explicitly fixed solver.

    This avoids re-hashing large candidate arrays for every rho/lambda policy.
    """
    verify_complete_policy(descriptor)
    solver = _plain(solver_config)
    if not isinstance(solver, dict):
        raise ValueError("solver_config must be a mapping or dataclass")
    definition = {**descriptor["definition"], "solver": solver,
                  "fallback": {**descriptor["definition"]["fallback"],
                               "threshold": solver.get("fallback_threshold")}}
    policy_hash = canonical_sha256(definition)
    result = {**descriptor, "definition": definition, "policy_sha256": policy_hash,
              "binding_sha256": canonical_sha256({"policy_sha256": policy_hash,
                            "population_sha256": descriptor["population"]["population_sha256"]})}
    verify_complete_policy(result)
    return result


def seal_policy_records(policy):
    """Seal actual loss rows together with their complete policy and population."""
    descriptor = policy["complete_policy"]
    verify_complete_policy(descriptor)
    payload = {"policy_sha256": descriptor["policy_sha256"], "binding_sha256": descriptor["binding_sha256"],
               "records": policy["records"]}
    return canonical_sha256(payload)


def seal_development_score(policy_descriptor, score, development_population, metric):
    """Bind one measured development score to a predeclared deployed rule.

    The development export may use out-of-fold detector checkpoints, so its
    population is recorded separately from the deployed rule's candidate stream.
    This function seals supplied evidence; it cannot establish when a policy was
    registered or whether a claimed metric was actually evaluated.
    """
    verify_complete_policy(policy_descriptor)
    verify_population(development_population)
    if not math.isfinite(float(score)) or not isinstance(metric, str) or not metric.strip():
        raise ValueError("Development scores require a finite value and named metric")
    entries = development_population["records"]
    if not entries or any(item["role"] != "dev" for item in entries):
        raise ValueError("Development score population must contain development images only")
    expected = set(policy_descriptor["definition"].get("fitting_population_ids", {}).get("dev", []))
    if not expected or {item["image_id"] for item in entries} != expected:
        raise ValueError("Development score population differs from checkpoint development IDs")
    solver = policy_descriptor["definition"]["solver"]
    payload = {"rho": float(solver["rho"]), "lam": float(solver["lam"]), "score": float(score),
               "metric": metric, "policy_sha256": policy_descriptor["policy_sha256"],
               "development_population": development_population}
    return {**payload, "score_sha256": canonical_sha256(payload)}
