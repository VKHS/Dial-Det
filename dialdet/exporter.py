"""Frozen Faster R-CNN export before final detection NMS.

This adapter deliberately leaves RPN proposal generation, including its internal
NMS, unchanged. It replaces only the final RoI detection decoder and records
the frozen candidate interface. Weights downloads require an explicit option.

Reference API: torchvision/models/detection/roi_heads.py, postprocess_detections:
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import CandidateRecord, ROLES, atomic_json, save_record, sha256_file
from .detection_datasets import image_path, load_crowdhuman_annotations, load_citypersons_annotations


def build_detector(checkpoint: str | Path | None = None, download_weights: bool = False,
                   device: str = "cpu", num_classes: int = 91,
                   min_size: int = 800, max_size: int = 1333):
    """Construct the official torchvision R50-FPN; explicit weights required."""
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    if checkpoint is not None and download_weights:
        raise ValueError("provide either a local checkpoint or download_weights, not both")
    if checkpoint is None and not download_weights:
        raise ValueError("A frozen detector checkpoint is required; use an explicit download_weights option for official COCO weights")
    if download_weights:
        if num_classes != 91:
            raise ValueError("official COCO pretrained Faster R-CNN uses a 91-entry label space")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights,
                                       min_size=min_size, max_size=max_size)
        provenance = {"checkpoint": "torchvision:FasterRCNN_ResNet50_FPN_Weights.DEFAULT",
                      "weights_enum": str(weights),
                      "download_authorized": True, "training_population": "COCO train2017"}
    else:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes,
                                       min_size=min_size, max_size=max_size)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        model.load_state_dict(payload, strict=True)
        provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                      "download_authorized": False, "training_population": "not supplied; establish out-of-fold provenance separately"}
    # A content identity is recorded for downloaded as well as local weights.
    # Hash before moving to CUDA; no implicit second download or model execution.
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    provenance["detector_state_sha256"] = digest.hexdigest()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().to(device)
    return model, provenance


def export_image(model, image, image_id: str, ground_truth: list[dict[str, Any]] | None = None,
                 score_floor: float | None = None, max_candidates: int = 500,
                 metadata: dict[str, Any] | None = None,
                 category_ids: list[int] | tuple[int, ...] | None = None) -> CandidateRecord:
    """Export one [3,H,W] float RGB image in [0,1] without final RoI NMS.

    IDs are proposal_index*num_classes + class_index before filtering. Features
    are channelwise spatial means of the detector's existing RoIAlign outputs,
    shared by the class expansions of a proposal. They are not image patches or
    extra neural features trained by the evidence model.

    Optional category_ids restrict a dataset to native detector classes before
    applying the global candidate cap. IDs, logits, scores and class labels are
    never reindexed or renormalized by this restriction.
    """
    import torch
    from torchvision.ops.boxes import clip_boxes_to_image
    import torchvision

    if not 1 <= max_candidates <= 500:
        raise ValueError("max_candidates must be in [1,500]")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected an RGB [3,H,W] image tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite in [0,1]")
    if model.training:
        raise ValueError("the detector must be in eval mode")
    native_floor = float(model.roi_heads.score_thresh) if score_floor is None else float(score_floor)
    if not 0 <= native_floor <= 1:
        raise ValueError("score_floor must be in [0,1]")
    original_size = (int(image.shape[-2]), int(image.shape[-1]))
    with torch.inference_mode():
        transformed, _ = model.transform([image], None)
        maps = model.backbone(transformed.tensors)
        if isinstance(maps, torch.Tensor):
            maps = OrderedDict([("0", maps)])
        proposals, _ = model.rpn(transformed, maps, None)
        pooled = model.roi_heads.box_roi_pool(maps, proposals, transformed.image_sizes)
        features = pooled.mean(dim=(-1, -2))
        representation = model.roi_heads.box_head(pooled)
        class_logits, box_regression = model.roi_heads.box_predictor(representation)
        decoded = model.roi_heads.box_coder.decode(box_regression, proposals)
        scores = torch.softmax(class_logits, dim=-1)
        n_proposals, n_classes = scores.shape
        if category_ids is not None:
            if (not category_ids or any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
                                        or not 1 <= int(x) < n_classes for x in category_ids)
                    or len(set(map(int, category_ids))) != len(category_ids)):
                raise ValueError("category_ids must be unique non-background classes in the detector label space")
        if decoded.numel() != n_proposals * n_classes * 4:
            raise ValueError("adapter expects class-specific Faster R-CNN box regression")
        decoded = decoded.reshape(n_proposals, n_classes, 4)
        decoded = clip_boxes_to_image(decoded, transformed.image_sizes[0])
        class_index = torch.arange(n_classes, device=image.device).expand(n_proposals, -1)
        proposal_index = torch.arange(n_proposals, device=image.device)[:, None].expand_as(class_index)
        ids = proposal_index * n_classes + class_index
        boxes = decoded[:, 1:].reshape(-1, 4)
        scores = scores[:, 1:].reshape(-1)
        classes = class_index[:, 1:].reshape(-1)
        source_proposals = proposal_index[:, 1:].reshape(-1)
        ids = ids[:, 1:].reshape(-1)
        sizes = boxes[:, 2:] - boxes[:, :2]
        eligible = (scores > native_floor) & (sizes[:, 0] >= 1e-2) & (sizes[:, 1] >= 1e-2)
        if category_ids is not None:
            allowed = torch.as_tensor(category_ids, dtype=classes.dtype, device=classes.device)
            eligible &= torch.isin(classes, allowed)
        selected = eligible.nonzero(as_tuple=True)[0]
        order = torch.argsort(scores[selected], descending=True, stable=True)[:max_candidates]
        selected = selected[order]
        boxes = boxes[selected].clone()
        h_transformed, w_transformed = transformed.image_sizes[0]
        boxes[:, [0, 2]] *= original_size[1] / w_transformed
        boxes[:, [1, 3]] *= original_size[0] / h_transformed
        values = {"ids": ids[selected].cpu().numpy().astype(np.int64),
                  "boxes": boxes.cpu().numpy().astype(np.float64),
                  "classes": classes[selected].cpu().numpy().astype(np.int64),
                  "scores": scores[selected].cpu().numpy().astype(np.float64),
                  "features": features[source_proposals[selected]].cpu().numpy().astype(np.float32)}
    annotations = sorted(ground_truth or [], key=lambda annotation: int(annotation["id"]))
    gt_boxes, gt_classes, gt_ids, gt_ignore = [], [], [], []
    for annotation in annotations:
        x, y, width, height = (float(v) for v in annotation["bbox"])
        if width <= 0 or height <= 0:
            raise ValueError(f"nonpositive annotation box for annotation {annotation['id']}")
        gt_boxes.append([x, y, x + width, y + height])
        gt_classes.append(int(annotation["category_id"]))
        gt_ids.append(int(annotation["id"]))
        gt_ignore.append(bool(annotation.get("iscrowd", False) or annotation.get("ignore", False)))
    provenance = {**(metadata or {}), "exporter": "torchvision_fasterrcnn_pre_final_nms",
                  "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                  "final_detection_nms_applied": False, "rpn_nms_preserved": True,
                  "native_score_floor": native_floor,
                  "score_floor_relation": "strict greater-than, matching torchvision",
                  "max_candidates": max_candidates, "number_of_proposals": n_proposals,
                  "query_class_expansion": "all non-background class logits after softmax; optional dataset class filter before candidate cap",
                  "exported_category_ids": list(map(int, category_ids)) if category_ids is not None else list(range(1, n_classes)),
                  "candidate_id_definition": "source_proposal_index * detector_num_classes + class_index",
                  "candidate_order": "descending_native_score_then_source_candidate_id",
                  "detector_num_classes": n_classes,
                  "appearance_feature": "spatial mean of existing proposal RoIAlign output; class expansions share source feature",
                  "preprocessing": {"transform": "detector GeneralizedRCNNTransform", "original_size": list(original_size),
                                    "transformed_size": list(transformed.image_sizes[0]),
                                    "min_size": list(model.transform.min_size),
                                    "max_size": int(model.transform.max_size),
                                    "size_divisible": int(model.transform.size_divisible),
                                    "image_mean": list(model.transform.image_mean), "image_std": list(model.transform.image_std)},
                  "gt_ignore_convention": (metadata or {}).get("gt_ignore_convention", "iscrowd or ignore; official annotation file is retained for COCOeval")}
    return CandidateRecord(image_id=str(image_id), image_size=original_size, **values,
                           gt_boxes=np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4),
                           gt_classes=np.asarray(gt_classes, dtype=np.int64),
                           gt_ids=np.asarray(gt_ids, dtype=np.int64), gt_ignore=np.asarray(gt_ignore, dtype=bool),
                           metadata=provenance)


def export_coco(image_root: str | Path, annotations_file: str | Path, output_dir: str | Path,
                checkpoint: str | Path | None = None, download_weights: bool = False,
                limit: int | None = None, device: str = "cpu", score_floor: float | None = None,
                max_candidates: int = 500, role: str = "pool", num_classes: int = 91,
                min_size: int = 800, max_size: int = 1333) -> dict[str, Any]:
    """Export supplied COCO-format images/labels; does not obtain a dataset.

    The default pool role is intentional. The caller must establish the paper's
    out-of-fold or public-checkpoint cross-fitting protocol before fitting.
    """
    import torch
    from PIL import Image

    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    image_root, annotations_file, output_dir = Path(image_root).resolve(), Path(annotations_file).resolve(), Path(output_dir).resolve()
    dataset = json.loads(annotations_file.read_text())
    if not {"images", "annotations", "categories"} <= dataset.keys():
        raise ValueError("COCO-format image, annotation and category arrays are required")
    model, provenance = build_detector(checkpoint, download_weights, device, num_classes, min_size, max_size)
    metadata = {**provenance, "annotations_file": str(annotations_file), "annotations_sha256": sha256_file(annotations_file),
                "image_root": str(image_root), "dataset": "coco", "role": role}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    images = sorted(dataset["images"], key=lambda item: int(item["id"]))
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError("no images to export")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new export directory")
    entries = []
    for image_info in images:
        path = (image_root / image_info["file_name"]).resolve()
        if image_root not in path.parents:
            raise ValueError("image filename escapes the supplied image directory")
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        if (rgb.shape[0], rgb.shape[1]) != (int(image_info["height"]), int(image_info["width"])):
            raise ValueError(f"image dimensions disagree with annotations: {image_info['id']}")
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device)
        image_id = str(image_info["id"])
        record = export_image(model, image_tensor, image_id, by_image[int(image_id)], score_floor,
                              max_candidates, {**metadata, "image_file": str(path)})
        relative_path = f"records/{int(image_id):012d}.npz"
        saved = save_record(record, output_dir / relative_path)
        entries.append({"image_id": image_id, "path": relative_path, "role": role, "sha256": sha256_file(saved)})
        print(json.dumps({"stage": "frozen_export", "image_id": image_id, "candidates": len(record.ids)}), flush=True)
    document = {"schema_version": 1, "dataset": "coco", "protocol": "public_checkpoint_exports",
                "num_classes": num_classes, "quality_mode": "coco",
                "risk_eligible": False, "records": entries,
                "provenance": metadata, "annotations_file": str(annotations_file),
                "note": "Export only. Use opposite-half encoder cross-fitting for public checkpoints. A local checkpoint alone does not establish out-of-fold or independent risk populations."}
    atomic_json(manifest_path, document)
    return {"manifest": str(manifest_path), "manifest_path": str(manifest_path), "images": len(entries),
            "status": "FROZEN_EXPORT_COMPLETE", "dataset": "coco", "risk_eligible": False}


def export_crowdhuman(image_root: str | Path, annotations_file: str | Path, output_dir: str | Path,
                     checkpoint: str | Path | None = None, limit: int | None = None,
                     device: str = "cpu", score_floor: float | None = None,
                     max_candidates: int = 500, role: str = "pool", num_classes: int = 2,
                     min_size: int = 800, max_size: int = 1333,
                     person_class_id: int = 1, dataset_split: str | None = None) -> dict[str, Any]:
    """Export CrowdHuman full-body candidates from native ODGT annotations.

    Supply a torchvision-compatible detector checkpoint. The default label space
    contains background and person; use num_classes explicitly for another
    compatible checkpoint. This adapter never silently downloads a COCO model or
    asserts the paper's focal detector training/independence protocol.
    """
    import torch
    from PIL import Image

    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
    if isinstance(person_class_id, bool) or not isinstance(person_class_id, int) or not 1 <= person_class_id < num_classes:
        raise ValueError("person_class_id must be a non-background class in the detector label space")
    if checkpoint is None:
        raise ValueError("CrowdHuman export requires a local frozen detector --weights checkpoint")
    image_root, annotations_file, output_dir = Path(image_root).resolve(), Path(annotations_file).resolve(), Path(output_dir).resolve()
    images, split = load_crowdhuman_annotations(annotations_file, dataset_split, person_class_id)
    if limit is not None:
        images = images[:limit]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new export directory")
    # Check every source image before allocating/loading a detector.
    paths = [image_path(image_root, item.file_name) for item in images]
    model, provenance = build_detector(checkpoint=checkpoint, device=device, num_classes=num_classes,
                                       min_size=min_size, max_size=max_size)
    metadata = {**provenance, "annotations_file": str(annotations_file),
                "annotations_sha256": sha256_file(annotations_file), "image_root": str(image_root),
                "dataset": "crowdhuman", "dataset_split": split, "role": role,
                "detector_num_classes": num_classes, "person_class_id": person_class_id,
                "exported_category_ids": [person_class_id], "annotation_box_type": "fbox",
                "gt_ignore_convention": "tag=mask or extra.ignore=1; head_attr.ignore is head-only",
                "focal_detector_reproduced": False,
                "checkpoint_population_verified": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, (item, path) in enumerate(zip(images, paths)):
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device)
        record = export_image(model, image_tensor, item.image_id, item.ground_truth,
                              score_floor=score_floor, max_candidates=max_candidates,
                              metadata={**metadata, **item.metadata, "image_file": str(path)},
                              category_ids=[person_class_id])
        relative_path = f"records/{number:08d}.npz"
        saved = save_record(record, output_dir / relative_path)
        entries.append({"image_id": item.image_id, "original_image_id": item.original_image_id,
                        "path": relative_path, "role": role, "sha256": sha256_file(saved)})
        print(json.dumps({"stage": "frozen_export", "dataset": "crowdhuman",
                          "image_id": item.image_id, "candidates": len(record.ids)}), flush=True)
    document = {"schema_version": 1, "dataset": "crowdhuman", "dataset_split": split,
                "num_classes": num_classes, "person_class_id": person_class_id,
                "quality_mode": "iou50", "protocol": "frozen_detector_exports",
                "risk_eligible": False, "records": entries, "provenance": metadata,
                "annotations_file": str(annotations_file), "annotation_format": "crowdhuman_odgt",
                "note": "Native full-body export only. Establish disjoint train/dev/calibration/audit populations and detector provenance before fitting. A local checkpoint does not establish the paper's focal-detector or out-of-fold protocol."}
    atomic_json(manifest_path, document)
    return {"manifest": str(manifest_path), "manifest_path": str(manifest_path), "images": len(entries),
            "status": "FROZEN_EXPORT_COMPLETE", "dataset": "crowdhuman", "risk_eligible": False}


def export_citypersons(image_root: str | Path, annotations_file: str | Path, output_dir: str | Path,
                      checkpoint: str | Path | None = None, limit: int | None = None,
                      device: str = "cpu", score_floor: float | None = None,
                      max_candidates: int = 500, role: str = "pool", num_classes: int = 2,
                      min_size: int = 800, max_size: int = 1333,
                      person_class_id: int = 1, dataset_split: str | None = None,
                      coordinate_origin: int = 0, subset: str = "all") -> dict[str, Any]:
    """Export native CityPersons MAT annotations and person-only candidates.

    image_root points at the selected Cityscapes leftImg8bit split directory,
    which contains one child directory per city. A local torchvision-compatible
    checkpoint is required. Prepared targets retain the requested all/reasonable
    subset; official CityPersons MR evaluation remains a separate operation.
    """
    import torch
    from PIL import Image

    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
    if isinstance(person_class_id, bool) or not isinstance(person_class_id, int) or not 1 <= person_class_id < num_classes:
        raise ValueError("person_class_id must be a non-background class in the detector label space")
    if checkpoint is None:
        raise ValueError("CityPersons export requires a local frozen detector --weights checkpoint")
    image_root, annotations_file, output_dir = Path(image_root).resolve(), Path(annotations_file).resolve(), Path(output_dir).resolve()
    images, split = load_citypersons_annotations(annotations_file, dataset_split, person_class_id,
                                                 coordinate_origin, subset)
    if limit is not None:
        images = images[:limit]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists; choose a new export directory")
    paths = [image_path(image_root, item.file_name) for item in images]
    model, provenance = build_detector(checkpoint=checkpoint, device=device, num_classes=num_classes,
                                       min_size=min_size, max_size=max_size)
    metadata = {**provenance, "annotations_file": str(annotations_file),
                "annotations_sha256": sha256_file(annotations_file), "image_root": str(image_root),
                "dataset": "citypersons", "dataset_split": split, "role": role,
                "detector_num_classes": num_classes, "person_class_id": person_class_id,
                "exported_category_ids": [person_class_id], "annotation_box_type": "aligned_full_body",
                "coordinate_origin": int(coordinate_origin), "citypersons_subset": subset,
                "checkpoint_population_verified": False,
                "official_miss_rate_evaluator_applied": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for number, (item, path) in enumerate(zip(images, paths)):
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).to(device)
        record = export_image(model, image_tensor, item.image_id, item.ground_truth,
                              score_floor=score_floor, max_candidates=max_candidates,
                              metadata={**metadata, **item.metadata, "image_file": str(path)},
                              category_ids=[person_class_id])
        relative_path = f"records/{number:08d}.npz"
        saved = save_record(record, output_dir / relative_path)
        entries.append({"image_id": item.image_id, "original_image_id": item.original_image_id,
                        "path": relative_path, "role": role, "sha256": sha256_file(saved)})
        print(json.dumps({"stage": "frozen_export", "dataset": "citypersons",
                          "image_id": item.image_id, "candidates": len(record.ids)}), flush=True)
    document = {"schema_version": 1, "dataset": "citypersons", "dataset_split": split,
                "num_classes": num_classes, "person_class_id": person_class_id,
                "quality_mode": "iou50", "protocol": "frozen_detector_exports",
                "coordinate_origin": int(coordinate_origin), "citypersons_subset": subset,
                "risk_eligible": False, "records": entries, "provenance": metadata,
                "annotations_file": str(annotations_file), "annotation_format": "citypersons_mat",
                "note": "Native aligned-full-body export. Every non-pedestrian source class is an ignored pedestrian region. Targets honor the recorded subset, but matching diagnostics are not the official CityPersons miss rate. Use a disjoint fitting/audit protocol for the chosen frozen detector."}
    atomic_json(manifest_path, document)
    return {"manifest": str(manifest_path), "manifest_path": str(manifest_path), "images": len(entries),
            "status": "FROZEN_EXPORT_COMPLETE", "dataset": "citypersons", "risk_eligible": False}
