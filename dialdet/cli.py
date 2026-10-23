"""Command-line orchestration for DIAL-Det training, allocation and evaluation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import unittest

import numpy as np

from . import __version__
from .data import (atomic_json, load_manifest, manifest_document, make_crossfit_manifests,
                   record_hash, sha256_file, canonical_dataset, dataset_quality_mode,
                   classify_synthetic)
from .metrics import audit_record, summarize_audits, coco_predictions, official_coco_evaluation
from .solver import SolverConfig, allocate, greedy_allocate
from .certificate import check_certificate
from .baselines import local_rule, nms, soft_nms
from .artifacts import (invocation_inputs, relative_artifact, verify_run,
                        write_run_manifest)
from .provenance import (build_complete_policy, candidate_semantic_hash,
                         canonical_sha256, inspect_manifest, verify_complete_policy)


def solver_config(args):
    kwargs = {}
    if getattr(args, "config", None):
        doc = json.loads(Path(args.config).read_text())
        kwargs.update(doc.get("solver", doc))
    for key in ("rho", "eta", "lam", "p", "theta", "exact_limit", "beam_width", "split_limit",
                "lp_time_limit", "max_cliques", "fallback_threshold"):
        if getattr(args, key, None) is not None:
            kwargs[key] = getattr(args, key)
    return SolverConfig(**kwargs)


def _device_check(device):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Select --device cpu for local tests.")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _metadata_value(metadata, key, default):
    if key in metadata:
        return metadata[key]
    return metadata.get("training_config", {}).get(key, default)


def resolve_quality_mode(args, document, fallback="coco"):
    requested = getattr(args, "quality_mode", "auto")
    if requested != "auto":
        return requested
    return (dataset_quality_mode(document["dataset"])
            or document.get("quality_mode") or fallback)


def validate_dataset_arguments(args):
    """Reject a mismatched dataset/evaluator before export or expensive training."""
    document = manifest_document(args.manifest) if getattr(args, "manifest", None) else None
    if getattr(args, "manifests", None):
        merged_datasets = {manifest_document(path)["dataset"] for path in args.manifests}
        if len(merged_datasets) != 1:
            raise ValueError("Cross-fit folds must belong to the same detection dataset")
        actual = merged_datasets.pop()
    elif document:
        actual = document["dataset"]
    elif args.command.startswith("export-"):
        actual = args.command.removeprefix("export-")
    elif args.command in ("smoke", "self-test"):
        actual = "synthetic"
    else:
        actual = None
    requested = getattr(args, "dataset", None)
    if requested and actual and canonical_dataset(requested) != actual:
        raise ValueError(f"--dataset {requested} does not match the {actual} data for this command")
    coco = getattr(args, "coco_annotations", None)
    crowd = getattr(args, "crowdhuman_annotations", None)
    evaluator = getattr(args, "crowdhuman_evaluator", None)
    if coco and crowd:
        raise ValueError("Choose one dataset-specific evaluation annotation file")
    if coco and actual in ("crowdhuman", "citypersons"):
        raise ValueError("COCOeval annotations cannot evaluate this pedestrian benchmark protocol")
    if crowd and actual not in (None, "crowdhuman", "generic", "synthetic"):
        raise ValueError("--crowdhuman-annotations requires a CrowdHuman detection manifest")
    if evaluator and not crowd:
        raise ValueError("--crowdhuman-evaluator requires --crowdhuman-annotations")
    development_annotations = getattr(args, "development_annotations", None)
    if development_annotations and not (coco or crowd):
        raise ValueError("--development-annotations requires a dataset-specific evaluation annotation option")
    for filename in (coco, crowd, development_annotations):
        if filename and not Path(filename).is_file():
            raise FileNotFoundError(filename)
    if evaluator and not Path(evaluator).is_dir():
        raise FileNotFoundError(evaluator)
    return actual


def train_command(args):
    from .training import train
    _device_check(args.device)
    trains, devs = load_manifest(args.manifest, "train"), load_manifest(args.manifest, "dev")
    if not trains or not devs:
        raise ValueError("Training requires disjoint train and dev records in the manifest")
    model_config = None
    training_config = {}
    if getattr(args, "config", None):
        config_doc = json.loads(Path(args.config).read_text())
        model_config = config_doc.get("model")
        training_config = config_doc.get("training", {})
    allowed_training = {"learning_rate", "weight_decay", "beta", "gamma", "assignment_iou", "gradient_clip", "num_classes"}
    if set(training_config) - allowed_training:
        raise ValueError(f"Unknown JSON training keys: {set(training_config) - allowed_training}")
    # For a known dataset the class table is fixed before looking at audit data.
    # A tiny opposite-half fitting set need not contain its largest class ID.
    document = manifest_document(args.manifest)
    dataset = document["dataset"]
    if "num_classes" in document:
        training_config.setdefault("num_classes", int(document["num_classes"]))
    elif dataset == "coco":
        training_config.setdefault("num_classes", 91)
    elif dataset in ("crowdhuman", "citypersons"):
        training_config.setdefault("num_classes", 2)
    quality_mode = resolve_quality_mode(args, document)
    effective_p = solver_config(args).p
    result = train(trains, devs, args.output, epochs=args.epochs, seed=args.seed, device=args.device,
                   resume=args.resume, quality_mode=quality_mode, model_config=model_config,
                   p=effective_p, **training_config)
    result["is_synthetic"] = classify_synthetic(document, [*trains, *devs])
    atomic_json(Path(args.output) / "TRAIN_COMMAND.json", {"manifest_sha256": sha256_file(args.manifest),
                "manifest": str(Path(args.manifest).resolve()), "dataset": dataset,
                "quality_mode": quality_mode, "result": result})
    return result


def infer_command(args):
    from .training import load_evidence_checkpoint, predict_record
    _device_check(args.device)
    records = load_manifest(args.manifest, args.role)
    if not records:
        raise ValueError(f"No records with role={args.role}")
    model, metadata = load_evidence_checkpoint(args.checkpoint, args.device)
    temperature = float(_metadata_value(metadata, "temperature", 1.))
    quality_mode = resolve_quality_mode(args, manifest_document(args.manifest),
                                        fallback=_metadata_value(metadata, "quality_mode", "coco"))
    config = solver_config(args)
    if config.p != float(_metadata_value(metadata, "p", 2.)):
        raise ValueError("Allocation exponent p differs from mass training; supply the matching --p or config")
    train_ids = set(map(str, metadata.get("train_image_ids", [])))
    dev_ids = set(map(str, metadata.get("dev_image_ids", [])))
    if args.role in ("audit", "calibration") and {r.image_id for r in records} & (train_ids | dev_ids):
        raise ValueError("Inference population overlaps the checkpoint's fitting/tuning images")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    policy_files, policies = {}, {}
    for budget in args.budgets:
        base = build_complete_policy(records, args.checkpoint, metadata, config, budget, quality_mode,
                                     matching_iou=getattr(args, "iou", .5))
        for method in ("dialdet", "local", "greedy", "nms", "soft_nms"):
            policy = copy.deepcopy(base)
            if method != "dialdet":
                policy["definition"]["method"] = method + "_control"
                policy["definition"]["control"] = {
                    "local": {"threshold": config.fallback_threshold, "evidence": "q,r"},
                    "greedy": {"objective": "same_F", "stop": "nonpositive_marginal"},
                    "nms": {"iou_threshold": .5, "score_floor": .001, "class_consistent": True},
                    "soft_nms": {"sigma": .5, "score_floor": .001, "class_consistent": True,
                                 "membership_track_scores": "restored_native_scores"},
                }[method]
                policy["policy_sha256"] = canonical_sha256(policy["definition"])
                policy["binding_sha256"] = canonical_sha256({"policy_sha256": policy["policy_sha256"],
                          "population_sha256": policy["population"]["population_sha256"]})
            relative = f"policies/{method}_K{budget}.json"
            atomic_json(output / relative, policy)
            policy_files[(method, budget)], policies[(method, budget)] = relative, policy
    predictions, summaries = [], []
    for number, record in enumerate(records):
        start = time.perf_counter()
        evidence = predict_record(record, model, temperature=temperature, p=config.p,
                                  quality_mode=quality_mode, device=args.device)
        evidence_seconds = time.perf_counter() - start
        q, r = np.asarray(evidence["q"], dtype=np.float64), np.asarray(evidence["r"], dtype=np.float64)
        # Recompute in allocator precision from the calibrated, unthresholded affinities.
        mass = np.power(r, config.p).sum(axis=0)
        tag = f"{number:06d}_{record_hash(record)[:12]}"
        evidence_path = output / "evidence" / f"{tag}.npz"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(evidence_path, q=q, r=r, d=mass, candidate_ids=record.ids,
                            quality_probs=evidence["quality_probs"], support=evidence["support"],
                            overlap=evidence["overlap"], rescue=evidence["rescue"])
        evidence_hash = sha256_file(evidence_path)
        semantic_hash = candidate_semantic_hash(record)
        for budget in args.budgets:
            result = allocate(q, r, int(budget), record.ids.tolist(), config, mass,
                              image_id=record.image_id, candidate_export_hash=record_hash(record))
            cert_path = output / "certificates" / f"{tag}_K{budget}.json"
            atomic_json(cert_path, result["certificate"])
            methods = {"dialdet": (result["selected"], result["status"]),
                       "local": (local_rule(q, r, budget, config.fallback_threshold, record.ids), "local_baseline"),
                       "greedy": (greedy_allocate(q, r, budget, record.ids, config, mass)["selected"], "greedy_baseline"),
                       "nms": (nms(record, budget), "nms_baseline"),
                       "soft_nms": (soft_nms(record, budget)[0], "soft_nms_baseline")}
            for method, (selected, status) in methods.items():
                # Immutable membership interface. Native scores are always restored.
                row = {"image_id": record.image_id, "K": int(budget), "method": method,
                       "selected_indices": [int(i) for i in selected],
                       "selected_ids": record.ids[selected].tolist(),
                       "boxes": record.boxes[selected].tolist(), "classes": record.classes[selected].tolist(),
                       "native_scores": record.scores[selected].tolist(), "status": status,
                       "candidate_export_hash": record_hash(record), "evidence_seconds": evidence_seconds,
                       "candidate_semantic_hash": semantic_hash,
                       "track": "membership_native_scores", "evidence": str(evidence_path.relative_to(output)),
                       "evidence_sha256": evidence_hash, "policy": policy_files[(method, budget)],
                       "policy_sha256": policies[(method, budget)]["policy_sha256"],
                       "policy_binding_sha256": policies[(method, budget)]["binding_sha256"]}
                if method == "dialdet":
                    row.update(objective=result["objective"], gap=result["gap"],
                               certificate=str(cert_path.relative_to(output)),
                               certificate_sha256=sha256_file(cert_path),
                               fallback_reason=result.get("fallback_reason"),
                               solver_seconds=result.get("timings", {}).get("total_seconds", None))
                predictions.append(row)
        print(f"[INFER] {number+1}/{len(records)} image={record.image_id} candidates={len(record.ids)}", flush=True)
    with (output / "predictions.jsonl").open("w") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary = {"images": len(records), "predictions": len(predictions), "budgets": args.budgets,
               "dataset": manifest_document(args.manifest)["dataset"],
               "output": str(output.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
               "manifest_sha256": sha256_file(args.manifest), "solver": asdict(config),
               "quality_mode": quality_mode, "temperature": temperature, "role": args.role,
               "policy_schema": "dialdet-complete-policy-v1", "policies": list(policy_files.values()),
               "is_synthetic": classify_synthetic(manifest_document(args.manifest), records),
               "scores": "native detector scores preserved",
               "baseline_note": "NMS/Soft-NMS use documented default controls; local is tuned only if a tuning file was supplied. No eleven-baseline reproduction is claimed."}
    atomic_json(output / "INFERENCE.json", summary)
    return summary


def evaluate_command(args):
    dataset = manifest_document(args.manifest)["dataset"]
    records = {r.image_id: r for r in load_manifest(args.manifest, args.role)}
    source = Path(args.predictions)
    path = source / "predictions.jsonl" if source.is_dir() else source
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    coco_annotations = getattr(args, "coco_annotations", None)
    crowd_annotations = getattr(args, "crowdhuman_annotations", None)
    crowd_evaluator = getattr(args, "crowdhuman_evaluator", None)
    export_crowd = dataset == "crowdhuman" or bool(crowd_annotations)
    groups, coco_groups, crowd_groups, seen = {}, {}, {}, set()
    checked_policies, checked_artifacts, binding_states = {}, {}, set()
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in records:
            raise ValueError(f"Prediction image is absent from requested manifest role: {image_id}")
        record = records[image_id]
        selected = row["selected_indices"]
        # Validate indices before using them in NumPy indexing.
        audit = audit_record(record, selected, row["K"], row["method"], row["status"], args.iou)
        if row["candidate_export_hash"] != record_hash(record):
            raise ValueError("The prediction was produced from a different frozen export")
        if row.get("candidate_semantic_hash") and row["candidate_semantic_hash"] != candidate_semantic_hash(record):
            raise ValueError("The prediction's semantic export metadata changed")
        bound = bool(row.get("policy"))
        binding_states.add(bound)
        if bound:
            policy_path = relative_artifact(path.parent, row["policy"])
            if policy_path not in checked_policies:
                policy = json.loads(policy_path.read_text())
                verify_complete_policy(policy)
                checked_policies[policy_path] = policy
            policy = checked_policies[policy_path]
            if (row.get("policy_sha256") != policy["policy_sha256"] or
                    row.get("policy_binding_sha256") != policy["binding_sha256"] or
                    row["K"] != policy["definition"]["budget"]):
                raise ValueError("Prediction row differs from its complete policy")
            expected_method = "DIAL-Det_ICASSP_identity_mass" if row["method"] == "dialdet" else row["method"] + "_control"
            if policy["definition"]["method"] != expected_method:
                raise ValueError("Prediction method differs from its recorded selection policy")
            if policy["definition"]["audit"]["iou"] != args.iou:
                raise ValueError("Requested matching IoU differs from the recorded policy audit")
            population = {entry["image_id"]: entry for entry in policy["population"]["records"]}
            entry = population.get(image_id)
            if (entry is None or entry["numerical_sha256"] != row["candidate_export_hash"] or
                    entry["semantic_sha256"] != row.get("candidate_semantic_hash")):
                raise ValueError("Prediction candidate hashes differ from the policy's population binding")
            if entry["role"] != args.role:
                raise ValueError("Prediction population role differs from the requested evaluation role")
            fitting = policy["definition"].get("fitting_population_ids", {})
            if args.role in ("audit", "calibration") and image_id in set(fitting.get("train", []) + fitting.get("dev", [])):
                raise ValueError("Evaluation image overlaps the policy checkpoint's fitting population")
            if row["method"] == "dialdet" and any(not row.get(k) for k in
                    ("evidence", "evidence_sha256", "certificate", "certificate_sha256")):
                raise ValueError("Policy-bound DIAL predictions require evidence and certificate artifacts")
            for artifact in ("evidence", "certificate"):
                if row.get(artifact):
                    artifact_path = relative_artifact(path.parent, row[artifact])
                    if artifact_path not in checked_artifacts:
                        checked_artifacts[artifact_path] = sha256_file(artifact_path)
                    if row.get(artifact + "_sha256") != checked_artifacts[artifact_path]:
                        raise ValueError(f"Changed or unbound {artifact} artifact")
            audit["policy_sha256"] = row["policy_sha256"]
        audit["candidate_export_hash"] = row["candidate_export_hash"]
        audit["candidate_semantic_hash"] = row.get("candidate_semantic_hash")
        audit["policy_binding_verified"] = bound
        for key, observed in (("selected_ids", record.ids[selected].tolist()),
                              ("boxes", record.boxes[selected].tolist()),
                              ("classes", record.classes[selected].tolist()),
                              ("native_scores", record.scores[selected].tolist())):
            if row[key] != observed:
                raise ValueError(f"Frozen membership interface was modified: {key}")
        key = f"{row['method']}_K{row['K']}"
        if not all(character.isalnum() or character in "_-" for character in key):
            raise ValueError("Prediction method names must be letters, numbers, underscores or hyphens")
        identity = (key, image_id)
        if identity in seen:
            raise ValueError("Duplicate image/method/budget prediction")
        seen.add(identity)
        if row["method"] == "dialdet" and row.get("evidence") and row.get("certificate"):
            from .audit_bound import matching_miss_bound
            cert = json.loads(relative_artifact(path.parent, row["certificate"]).read_text())
            with np.load(relative_artifact(path.parent, row["evidence"]), allow_pickle=False) as ev:
                if bound:
                    if not np.array_equal(ev["candidate_ids"], record.ids):
                        raise ValueError("Saved evidence candidate identities differ from the frozen export")
                    if cert is None:
                        if row["status"] != "fallback" or not row.get("fallback_reason") or row.get("gap") is not None:
                            raise ValueError("A missing certificate requires a recorded numerical fallback")
                        from .solver import local_rule as fallback_rule
                        expected_selection = fallback_rule(ev["q"], ev["r"], row["K"], candidate_ids=record.ids,
                                    threshold=policy["definition"]["solver"]["fallback_threshold"])
                        if sorted(selected) != expected_selection:
                            raise ValueError("Fallback membership differs from its registered local rule")
                    elif (cert.get("configuration") != policy["definition"]["solver"] or
                          cert.get("status") != row["status"]):
                        raise ValueError("Certificate configuration or status differs from the recorded policy")
                audit["matching_miss_bound"] = matching_miss_bound(
                    record, selected, ev["q"], ev["r"], row["K"], cert,
                    config=policy["definition"]["solver"] if bound else None,
                    mass=ev["d"], iou_threshold=args.iou)
                if bound and row["status"] != "fallback" and not audit["matching_miss_bound"]["certificate_valid"]:
                    raise ValueError("Certificate does not bind the selected set and saved evidence: " +
                                     str(audit["matching_miss_bound"]["reason"]))
        groups.setdefault(key, []).append(audit)
        if coco_annotations:
            coco_groups.setdefault(key, []).extend(coco_predictions(record, selected))
        if export_crowd:
            from .crowdhuman import crowdhuman_predictions
            crowd_groups.setdefault(key, []).append(crowdhuman_predictions(record, selected))
    if not groups:
        raise ValueError("No predictions to evaluate")
    if len(binding_states) != 1:
        raise ValueError("Cannot mix policy-bound and legacy unbound predictions in one evaluation")
    for group in groups.values():
        if {row["image_id"] for row in group} != set(records):
            raise ValueError("Partial prediction population: evaluation would omit images")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "per_image_audits.json", groups)
    report = {key: summarize_audits(group) for key, group in groups.items()}
    for key, group in groups.items():
        bounded = [row["matching_miss_bound"] for row in group if "matching_miss_bound" in row and row["Q"] > 0]
        if bounded:
            report[key]["matching_bound_nonvacuous_share"] = float(np.mean([r["nonvacuous"] for r in bounded]))
            report[key]["matching_bound_mean"] = float(np.mean([r["bound"] for r in bounded]))
    if coco_annotations:
        for key in groups:
            report[key]["official_coco"] = official_coco_evaluation(coco_annotations,
                         coco_groups[key], list(records), groups[key][0]["K"])
    if export_crowd:
        prediction_dir = output / "crowdhuman_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for key in groups:
            prediction_file = prediction_dir / f"{key}.odgt"
            from .crowdhuman import write_crowdhuman_predictions
            write_crowdhuman_predictions(prediction_file, crowd_groups[key])
            report[key]["crowdhuman_predictions"] = str(prediction_file.resolve())
            if crowd_annotations and crowd_evaluator:
                from .crowdhuman import official_crowdhuman_evaluation
                report[key]["official_crowdhuman"] = official_crowdhuman_evaluation(
                    crowd_annotations, crowd_groups[key], [row["ID"] for row in crowd_groups[key]],
                    groups[key][0]["K"], crowd_evaluator, output / "crowdhuman_evaluation" / key)
            else:
                report[key]["crowdhuman_benchmark_status"] = (
                    "Predictions exported; AP50/MR require --crowdhuman-annotations and --crowdhuman-evaluator")
    result = {"results": report, "prediction_sha256": sha256_file(path),
              "dataset": dataset,
              "policy_binding": "verified" if binding_states == {True} else "legacy_numerical_only",
              "matching_iou": args.iou,
              "is_synthetic": classify_synthetic(manifest_document(args.manifest), records.values()),
              "note": "Pooled shortfall and mean image shortfall have different denominators. No manuscript results are filled in."}
    from .metrics import summarize_high_pressure
    result["high_pressure"] = summarize_high_pressure(groups)
    atomic_json(output / "EVALUATION.json", result)
    return result


def tune_local_command(args):
    from .training import load_evidence_checkpoint, predict_record
    from .metrics import maximum_matching
    _device_check(args.device)
    records = load_manifest(args.manifest, "dev")
    if not records:
        raise ValueError("Local-rule tuning requires development records")
    coco_annotations = getattr(args, "coco_annotations", None)
    crowd_annotations = getattr(args, "crowdhuman_annotations", None)
    crowd_evaluator = getattr(args, "crowdhuman_evaluator", None)
    official_crowd = bool(crowd_annotations and crowd_evaluator)
    if not coco_annotations and not official_crowd and not args.allow_matching_proxy:
        raise ValueError("AP tuning needs --coco-annotations, or --crowdhuman-annotations with --crowdhuman-evaluator; explicitly choose --allow-matching-proxy only for diagnostic tuning")
    model, metadata = load_evidence_checkpoint(args.checkpoint, args.device)
    if {r.image_id for r in records} & set(metadata.get("train_image_ids", [])):
        raise ValueError("Local-rule development tuning overlaps the checkpoint's training population")
    quality_mode = resolve_quality_mode(args, manifest_document(args.manifest),
                                        fallback=_metadata_value(metadata, "quality_mode", "coco"))
    evidence = [predict_record(r, model, temperature=float(_metadata_value(metadata, "temperature", 1.)),
                p=float(_metadata_value(metadata, "p", 2.)),
                quality_mode=quality_mode, device=args.device) for r in records]
    results = []
    for threshold in (.3 + .025 * i for i in range(24)):
        predictions, counts = [], []
        for record, ev in zip(records, evidence):
            chosen = local_rule(ev["q"], ev["r"], args.budget, threshold, record.ids)
            counts.append(len(maximum_matching(record, chosen)))
            if coco_annotations:
                predictions.extend(coco_predictions(record, chosen))
            elif official_crowd:
                from .crowdhuman import crowdhuman_predictions
                predictions.append(crowdhuman_predictions(record, chosen))
        if coco_annotations:
            metrics = official_coco_evaluation(coco_annotations, predictions,
                                              [r.image_id for r in records], args.budget)
            score = metrics["metrics"]["AP"]
        elif official_crowd:
            from .crowdhuman import official_crowdhuman_evaluation
            metrics = official_crowdhuman_evaluation(crowd_annotations, predictions,
                        [row["ID"] for row in predictions], args.budget, crowd_evaluator,
                        Path(args.output) / "crowdhuman_tuning" / f"threshold_{len(results):02d}")
            score = metrics["metrics"]["AP50"]
        else:
            score = float(np.mean(counts))
        if score is None or not np.isfinite(score):
            raise ValueError("Development AP is undefined for this population; provide development images with nonignored targets")
        results.append({"threshold": threshold, "score": score})
    best = max(results, key=lambda x: (x["score"], -x["threshold"]))
    result = {"best_threshold": best["threshold"], "grid": results, "budget": args.budget,
              "metric": ("official_COCO_AP" if coco_annotations else
                         "official_CrowdHuman_AP50" if official_crowd else "mean_maximum_matches_PROXY"),
              "quality_mode": quality_mode,
              "is_synthetic": classify_synthetic(manifest_document(args.manifest), records),
              "annotations_sha256": sha256_file(coco_annotations or crowd_annotations)
                                     if coco_annotations or crowd_annotations else None,
              "checkpoint_sha256": sha256_file(args.checkpoint),
              "development_image_ids": [r.image_id for r in records]}
    atomic_json(Path(args.output) / "LOCAL_TUNING.json", result)
    return result


def pipeline_command(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training = argparse.Namespace(**vars(args))
    training.output = str(output / "training")
    trained = train_command(training)
    checkpoint = trained.get("checkpoint", trained.get("best_checkpoint"))
    if not checkpoint:
        raise RuntimeError("Training did not return a checkpoint path")
    inference = argparse.Namespace(**vars(args))
    inference.checkpoint = checkpoint
    inference.output = str(output / "inference")
    inference.role = "audit"
    tuning = None
    if (getattr(args, "coco_annotations", None) or
            (getattr(args, "crowdhuman_annotations", None) and getattr(args, "crowdhuman_evaluator", None))):
        tuning_args = argparse.Namespace(**vars(args))
        tuning_args.checkpoint = checkpoint
        tuning_args.output = str(output / "tuning")
        tuning_args.budget = 100
        tuning_args.allow_matching_proxy = False
        if getattr(args, "development_annotations", None):
            if getattr(args, "coco_annotations", None):
                tuning_args.coco_annotations = args.development_annotations
            else:
                tuning_args.crowdhuman_annotations = args.development_annotations
        tuning = tune_local_command(tuning_args)
        inference.fallback_threshold = tuning["best_threshold"]
    inferred = infer_command(inference)
    evaluation = argparse.Namespace(**vars(args))
    evaluation.predictions = inference.output
    evaluation.output = str(output / "evaluation")
    evaluation.role = "audit"
    evaluated = evaluate_command(evaluation)
    result = {"training": trained, "tuning": tuning, "inference": inferred,
              "evaluation": evaluated, "paper_results_reproduced": False,
              "is_synthetic": evaluated["is_synthetic"],
              "risk_certificate": "not_requested_or_not_eligible"}
    atomic_json(output / "PIPELINE.json", result)
    return result


def smoke_command(args):
    from .demo import create_synthetic_dataset, analytic_evidence
    output = Path(args.output)
    manifest = create_synthetic_dataset(output / "synthetic")
    records = load_manifest(manifest, "audit")
    config = SolverConfig()
    diagnostic = []
    for record in records:
        q, r = analytic_evidence(record)
        for k in (1, 2, 3, 5):
            result = allocate(q, r, k, record.ids.tolist(), config,
                              image_id=record.image_id, candidate_export_hash=record_hash(record))
            checked = check_certificate(result["certificate"])
            if not checked["valid"]:
                raise AssertionError(checked)
            row = audit_record(record, result["selected"], k, status=result["status"])
            diagnostic.append(row)
            atomic_json(output / "analytic_certificates" / f"{record.image_id}_K{k}.json", result["certificate"])
    result = {"version": __version__, "synthetic_only": True, "analytic_audits": diagnostic,
              "analytical_certificates_checked": len(diagnostic), "neural_pipeline": None,
              "note": "Synthetic execution check. These numbers are not detection-benchmark results."}
    if args.neural:
        neural = argparse.Namespace(manifest=manifest, output=str(output / "neural"), epochs=2,
                  seed=1, device=args.device, resume=None, quality_mode="iou50", config=None,
                  budgets=[2, 3], coco_annotations=None, iou=.5, fallback_threshold=.5)
        result["neural_pipeline"] = pipeline_command(neural)
    atomic_json(output / "SMOKE_REPORT.json", result)
    return result


def self_test_command(args):
    tests = Path(__file__).resolve().parent.parent / "tests"
    if not tests.is_dir():
        raise RuntimeError("Self-test requires the extracted source release, which includes tests/")
    suite = unittest.TestSuite()
    neural_files = {"test_evidence.py", "test_integration.py", "test_detection_datasets.py", "test_pipeline_integrity.py"}
    patterns = [path.name for path in sorted(tests.glob("test_*.py"))
                if args.neural or path.name not in neural_files]
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern=pattern))
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "TEST_LOG.txt").write_text(buffer.getvalue())
    summary = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "success": result.wasSuccessful(), "neural_requested": args.neural,
               "python": platform.python_version(), "numpy": np.__version__, "package_version": __version__}
    atomic_json(output / "TEST_REPORT.json", summary)
    print(buffer.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError("Verification failed; see TEST_LOG.txt")
    return summary


def verify_run_command(args):
    source = Path(args.run).resolve()
    source_dir = source if source.is_dir() else source.parent
    if Path(args.output).resolve() == source_dir:
        raise ValueError("Use a separate output directory for the verification report")
    result = verify_run(source, check_inputs=args.check_inputs)
    atomic_json(Path(args.output) / "RUN_VERIFICATION.json", result)
    if not result["valid"]:
        raise ValueError(f"Run verification failed: {result['errors']}")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description="DIAL-Det object detection: COCO, CrowdHuman and CityPersons")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(p, manifest=False, device=False):
        p.add_argument("--output", required=True)
        p.add_argument("--dataset", type=canonical_dataset,
                       choices=("coco", "crowdhuman", "citypersons", "synthetic", "generic"),
                       default=None, help="Detection dataset; checked against the manifest or export command")
        if manifest:
            p.add_argument("--manifest", required=True)
        if device:
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    def solver(p):
        p.add_argument("--config", help="JSON config; solver field contains SolverConfig keys")
        for key in ("rho", "eta", "lam", "p", "theta", "lp-time-limit", "fallback-threshold"):
            p.add_argument("--" + key, type=float, default=None)
        for key in ("exact-limit", "beam-width", "split-limit", "max-cliques"):
            p.add_argument("--" + key, type=int, default=None)

    def evaluator(p):
        annotations = p.add_mutually_exclusive_group()
        annotations.add_argument("--coco-annotations", help="Native COCO JSON for official COCOeval AP")
        annotations.add_argument("--crowdhuman-annotations", help="Native CrowdHuman ODGT for the requested population")
        p.add_argument("--crowdhuman-evaluator", help="Official CrowdDetection checkout containing APMRToolkits")

    for name in ("self-test", "smoke"):
        p = commands.add_parser(name)
        common(p, device=True)
        p.add_argument("--neural", action="store_true")
    for name in ("train", "pipeline"):
        p = commands.add_parser(name)
        common(p, manifest=True, device=True)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--resume")
        p.add_argument("--quality-mode", choices=("auto", "coco", "iou50"), default="auto",
                       help="auto: COCO ten-threshold mean; CrowdHuman/CityPersons probability at IoU 0.50")
        solver(p)
        if name == "pipeline":
            p.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100, 200])
            evaluator(p)
            p.add_argument("--development-annotations",
                           help="Separate development JSON/ODGT for threshold tuning; defaults to the evaluation annotation file")
            p.add_argument("--iou", type=float, default=.5)
    p = commands.add_parser("infer")
    common(p, manifest=True, device=True)
    solver(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--quality-mode", choices=("auto", "coco", "iou50"), default="auto")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    p.add_argument("--role", choices=("audit", "calibration", "dev", "pool"), default="audit")
    p = commands.add_parser("evaluate")
    common(p, manifest=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--role", default="audit")
    evaluator(p)
    p.add_argument("--iou", type=float, default=.5)
    p = commands.add_parser("tune-local")
    common(p, manifest=True, device=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--quality-mode", choices=("auto", "coco", "iou50"), default="auto")
    evaluator(p)
    p.add_argument("--allow-matching-proxy", action="store_true")
    p.add_argument("--budget", type=int, default=100)
    for name in ("export-coco", "export-crowdhuman", "export-citypersons"):
        p = commands.add_parser(name)
        common(p, device=True)
        p.add_argument("--annotations", required=True)
        p.add_argument("--images", required=True)
        p.add_argument("--weights", required=name != "export-coco")
        if name == "export-coco":
            p.add_argument("--download-weights", action="store_true")
        else:
            p.add_argument("--person-class-id", type=int, default=1)
            p.add_argument("--split", "--dataset-split", dest="dataset_split", default=None,
                           help="Source annotation split; inferred from canonical annotation filenames")
        if name == "export-citypersons":
            p.add_argument("--coordinate-origin", type=int, choices=(0, 1), default=0,
                           help="0 preserves official annotation coordinates; 1 explicitly subtracts one from x/y")
            p.add_argument("--citypersons-subset", choices=("all", "reasonable"), default="all",
                           help="Annotation target filter; the official CityPersons MR evaluator remains external")
        p.add_argument("--num-classes", type=int, default=91 if name == "export-coco" else 2,
                       help="Frozen detector label-table size including background")
        p.add_argument("--max-images", type=int)
        p.add_argument("--role", choices=("pool", "train", "dev", "calibration", "audit"), default="pool")
        p.add_argument("--score-floor", type=float)
    p = commands.add_parser("make-crossfit")
    common(p, manifest=True)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--dev-fraction", type=float, default=.2)
    p = commands.add_parser("merge-crossfit")
    common(p)
    p.add_argument("--manifests", nargs=2, required=True)
    p.add_argument("--prediction-dirs", nargs=2, required=True)
    p = commands.add_parser("check-certificate")
    common(p)
    p.add_argument("--certificate", required=True)
    p = commands.add_parser("risk-report")
    common(p)
    p.add_argument("--policy-records", required=True)
    p.add_argument("--development-scores", required=True)
    p.add_argument("--provenance", required=True)
    p = commands.add_parser("collect-risk")
    common(p, manifest=True, device=True)
    solver(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--quality-mode", choices=("auto", "coco", "iou50"), default="auto")
    p.add_argument("--budget", type=int, default=100)
    p = commands.add_parser("validate-manifest", help="Inspect all candidate files and data roles before training")
    common(p, manifest=True)
    p.add_argument("--role", choices=("train", "dev", "calibration", "audit", "pool"))
    p = commands.add_parser("verify-run", help="Verify recorded output and optionally external input hashes")
    common(p)
    p.add_argument("--run", required=True)
    p.add_argument("--check-inputs", action="store_true")
    p = commands.add_parser("paired-statistics", help="Paired crossed image-seed bootstrap on complete held-out runs")
    common(p)
    p.add_argument("--specification", required=True)
    p = commands.add_parser("study", help="Plan or execute a versioned multi-seed COCO cross-fit study")
    common(p)
    p.add_argument("--specification", required=True,
                   help="Version-1 study JSON; paths resolve relative to this file")
    p.add_argument("--execute", action="store_true",
                   help="Run the recorded tasks; the default only validates and writes a plan")
    p.add_argument("--resume", action="store_true",
                   help="Verify and reuse completed tasks in an unchanged study")
    p = commands.add_parser("compare-results", help="Compare versioned reference values with verified local artifacts")
    common(p)
    p.add_argument("--generated", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--allow-partial", action="store_true")
    p = commands.add_parser("score-family", help="Measure development utility for the registered 24-policy family")
    common(p, manifest=True, device=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy-records", required=True,
                   help="Registered complete-policy definitions; calibration losses are ignored")
    evaluator(p)
    p.add_argument("--allow-matching-proxy", action="store_true",
                   help="Diagnostic development scores only; cannot enable risk certification")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "verify-run":
        source = Path(args.run).resolve()
        if Path(args.output).resolve() == (source if source.is_dir() else source.parent):
            print("Use a separate output directory for the verification report", file=sys.stderr)
            return 2
    try:
        atomic_json(Path(args.output) / "RUN_STATUS.json", {"status": "running", "command": args.command})
        validate_dataset_arguments(args)
        if hasattr(args, "budgets") and any(k < 1 for k in args.budgets):
            raise ValueError("All CLI report budgets must be positive")
        if hasattr(args, "budgets") and len(set(args.budgets)) != len(args.budgets):
            raise ValueError("Budgets must be unique")
        handlers = {"train": train_command, "infer": infer_command, "evaluate": evaluate_command,
                    "pipeline": pipeline_command, "smoke": smoke_command, "self-test": self_test_command,
                    "tune-local": tune_local_command}
        if args.command in handlers:
            result = handlers[args.command](args)
        elif args.command == "export-coco":
            from .exporter import export_coco
            _device_check(args.device)
            result = export_coco(args.images, args.annotations, args.output, checkpoint=args.weights,
                       download_weights=args.download_weights, limit=args.max_images, device=args.device,
                       score_floor=args.score_floor, role=args.role, num_classes=args.num_classes)
        elif args.command in ("export-crowdhuman", "export-citypersons"):
            from . import exporter
            _device_check(args.device)
            export = getattr(exporter, args.command.replace("-", "_"))
            options = {"checkpoint": args.weights, "limit": args.max_images, "device": args.device,
                       "score_floor": args.score_floor, "role": args.role, "num_classes": args.num_classes,
                       "person_class_id": args.person_class_id, "dataset_split": args.dataset_split}
            if args.command == "export-citypersons":
                options["coordinate_origin"] = args.coordinate_origin
                options["subset"] = args.citypersons_subset
            result = export(args.images, args.annotations, args.output, **options)
        elif args.command == "make-crossfit":
            # Inspect one candidate file at a time. A COCO dataset name must
            # never erase an explicit synthetic flag on an exported record.
            inspected = inspect_manifest(args.manifest)
            folds = make_crossfit_manifests(args.manifest, args.output, args.seed, args.dev_fraction)
            for filename in folds:
                fold = json.loads(Path(filename).read_text())
                fold["is_synthetic"] = inspected["is_synthetic"]
                atomic_json(filename, fold)
            result = {"manifests": folds, "is_synthetic": inspected["is_synthetic"]}
        elif args.command == "merge-crossfit":
            from .crossfit import merge_runs
            result = merge_runs(args.manifests, args.prediction_dirs, args.output)
        elif args.command == "check-certificate":
            result = check_certificate(json.loads(Path(args.certificate).read_text()))
            atomic_json(Path(args.output) / "CERTIFICATE_CHECK.json", result)
            if not result["valid"]:
                raise ValueError(f"Certificate rejected: {result.get('errors')}")
        elif args.command == "risk-report":
            from .risk import evaluate_family
            result = evaluate_family(json.loads(Path(args.policy_records).read_text()),
                     json.loads(Path(args.development_scores).read_text()), json.loads(Path(args.provenance).read_text()))
            atomic_json(Path(args.output) / "RISK_REPORT.json", result)
        elif args.command == "collect-risk":
            from .risk_runtime import collect_policy_records
            _device_check(args.device)
            result = collect_policy_records(args.manifest, args.checkpoint, args.output,
                                             args.budget, args.device, solver_config(args), args.quality_mode)
        elif args.command == "validate-manifest":
            result = inspect_manifest(args.manifest, args.role)
            atomic_json(Path(args.output) / "MANIFEST_CHECK.json", result)
        elif args.command == "verify-run":
            result = verify_run_command(args)
        elif args.command == "paired-statistics":
            from .statistical_workflow import run_paired_statistics
            result = run_paired_statistics(args.specification, args.output)
        elif args.command == "study":
            from .study import run_study
            if args.dataset not in (None, "coco"):
                raise ValueError("The study workflow requires --dataset coco, or no dataset override")
            result = run_study(args.specification, args.output,
                               execute=args.execute, resume=args.resume)
        elif args.command == "compare-results":
            from .results import compare_results
            result = compare_results(args.generated, args.reference, args.allow_partial)
            atomic_json(Path(args.output) / "RESULT_COMPARISON.json", result)
        elif args.command == "score-family":
            from .development import score_policy_family
            _device_check(args.device)
            result = score_policy_family(args.manifest, args.checkpoint, args.policy_records, args.output,
                        args.device, args.coco_annotations, args.crowdhuman_annotations,
                        args.crowdhuman_evaluator, args.allow_matching_proxy)
        else:
            raise AssertionError(args.command)
        synthetic = result.get("is_synthetic")
        if args.command in ("smoke", "self-test"):
            synthetic = True
        if getattr(args, "manifest", None):
            declared = classify_synthetic(manifest_document(args.manifest))
            if declared is True:
                synthetic = True
            elif declared is False and synthetic is not True:
                synthetic = False
        write_run_manifest(args.output, args.command, vars(args), invocation_inputs(vars(args)), synthetic=synthetic)
        completion = {"command": args.command, "status": "completed",
                      "output": str(Path(args.output).resolve())}
        if args.command == "study":
            completion["study_status"] = result.get("status")
            completion["execute_requested"] = args.execute
        print(json.dumps(completion))
        return 0
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        atomic_json(Path(args.output) / "FAILURE.json", {"command": args.command, "type": type(exc).__name__, "message": str(exc)})
        atomic_json(Path(args.output) / "RUN_STATUS.json", {"status": "failed", "command": args.command})
        traceback.print_exc()
        return 1

