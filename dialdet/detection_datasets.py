"""Native object-detection annotations for the frozen candidate exporters.

CrowdHuman's official ODGT schema is documented at
https://www.crowdhuman.org/download.html .  Full-body ``fbox`` coordinates are
retained, including coordinates outside the image.  Body ignore flags are
independent of head ignore flags; the latter must not discard a valid body.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionImage:
    image_id: str
    original_image_id: str
    file_name: str
    ground_truth: list[dict[str, Any]]
    metadata: dict[str, Any]


def image_path(image_root: str | Path, file_name: str) -> Path:
    """Resolve a relative image filename without leaving the supplied root."""
    root = Path(image_root).resolve()
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("image filename must be relative to the supplied image directory")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ValueError("image filename escapes the supplied image directory")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _crowdhuman_split(annotations_file: str | Path, dataset_split: str | None) -> str:
    if dataset_split is None:
        name = Path(annotations_file).name.lower()
        match = re.fullmatch(r"annotation_(train|val|test)\.odgt", name)
        if match is None:
            raise ValueError("Specify dataset_split for a nonstandard CrowdHuman annotation filename")
        dataset_split = match.group(1)
    if not isinstance(dataset_split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", dataset_split):
        raise ValueError("dataset_split must be a nonempty alphanumeric split name (hyphen/underscore allowed)")
    return dataset_split.lower()


def _body_box(annotation: dict[str, Any], image_id: str, number: int) -> list[float]:
    try:
        box = np.asarray(annotation["fbox"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CrowdHuman image {image_id}, box {number}: a numeric fbox is required") from exc
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= 0):
        raise ValueError(f"CrowdHuman image {image_id}, box {number}: fbox must be finite [x,y,w,h] with positive size")
    return box.tolist()


def load_crowdhuman_annotations(annotations_file: str | Path, dataset_split: str | None = None,
                               person_class_id: int = 1) -> tuple[list[DetectionImage], str]:
    """Read labeled native ODGT, preserving the source ID for official evaluation.

    Decoder image IDs are namespaced as ``crowdhuman:<split>:<source ID>``;
    annotation IDs are the stable zero-based positions within each ODGT row.
    Source ``extra.box_id`` values are retained in metadata, since they need not
    be present or unique.  A ``mask`` region or ``extra.ignore=1`` is ignored.
    All other body boxes keep their full extent; ``hbox`` and ``head_attr`` do
    not change body detection targets.
    """
    if isinstance(person_class_id, bool) or not isinstance(person_class_id, int) or person_class_id < 1:
        raise ValueError("person_class_id must be a positive integer in the detector label space")
    split = _crowdhuman_split(annotations_file, dataset_split)
    images: list[DetectionImage] = []
    seen: set[str] = set()
    with Path(annotations_file).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid CrowdHuman JSON at line {line_number}") from exc
            if not isinstance(source, dict) or not isinstance(source.get("ID"), str) or not source["ID"]:
                raise ValueError(f"CrowdHuman line {line_number} requires a nonempty string ID")
            original_id = source["ID"]
            if original_id in seen:
                raise ValueError(f"Duplicate CrowdHuman image ID: {original_id}")
            seen.add(original_id)
            relative = Path(original_id)
            if relative.is_absolute() or ".." in relative.parts or ":" in original_id or "\\" in original_id:
                raise ValueError(f"Unsafe CrowdHuman image ID: {original_id}")
            if not isinstance(source.get("gtboxes"), list):
                raise ValueError(f"CrowdHuman image {original_id} requires labeled gtboxes")
            annotations, source_attributes = [], []
            for index, box in enumerate(source["gtboxes"]):
                if not isinstance(box, dict) or box.get("tag") not in ("person", "mask"):
                    raise ValueError(f"CrowdHuman image {original_id}, box {index}: tag must be person or mask")
                extra = box.get("extra", {})
                if not isinstance(extra, dict):
                    raise ValueError(f"CrowdHuman image {original_id}, box {index}: extra must be an object")
                ignore = extra.get("ignore", 0)
                if not isinstance(ignore, (int, bool)) or ignore not in (0, 1):
                    raise ValueError(f"CrowdHuman image {original_id}, box {index}: extra.ignore must be 0 or 1")
                annotations.append({"id": index, "category_id": person_class_id,
                                    "bbox": _body_box(box, original_id, index),
                                    "ignore": box["tag"] == "mask" or bool(ignore), "iscrowd": False})
                source_attributes.append({"annotation_id": index, "source_box_id": extra.get("box_id"),
                                          "tag": box["tag"], "extra": extra,
                                          "head_attr": box.get("head_attr", {}),
                                          "vbox": box.get("vbox"), "hbox": box.get("hbox")})
            file_name = original_id if relative.suffix.lower() in (".jpg", ".jpeg", ".png") else original_id + ".jpg"
            images.append(DetectionImage(
                image_id=f"crowdhuman:{split}:{original_id}", original_image_id=original_id,
                file_name=file_name, ground_truth=annotations,
                metadata={"dataset": "crowdhuman", "dataset_split": split,
                          "original_image_id": original_id, "annotation_box_type": "fbox",
                          "source_annotation_attributes": source_attributes,
                          "annotation_id_definition": "zero-based position in the source gtboxes array",
                          "full_body_boxes_clipped": False,
                          "gt_ignore_convention": "tag=mask or extra.ignore=1; head_attr.ignore is head-only"}))
    if not images:
        raise ValueError("No labeled CrowdHuman images in the supplied ODGT file")
    images.sort(key=lambda item: item.original_image_id)
    return images, split


def load_citypersons_annotations(annotations_file: str | Path, dataset_split: str | None = None,
                                person_class_id: int = 1, coordinate_origin: int = 0,
                                subset: str = "all") -> tuple[list[DetectionImage], str]:
    """Read the official CityPersons MATLAB struct/cell annotation format.

    Native arrays store [class,x,y,w,h,instance_id,xvis,yvis,wvis,hvis]. The
    published ``annotations/anno_val.mat`` and ``evaluation/val_gt.json`` retain
    identical numeric x/y/w/h coordinates; origin=0 preserves that interface.
    Set origin=1 only for an explicitly one-based external annotation variant;
    this subtracts one from x/y, never from widths/heights.

    ``all`` retains every pedestrian (source class 1). ``reasonable`` additionally
    ignores pedestrian boxes with height below 50 or visible/full area below
    0.65. Other source classes remain ignored regions for the pedestrian class.
    This target preparation does not implement official CityPersons MR: that
    evaluator also applies detection-range filtering and its matching protocol.

    Primary format reference:
    https://github.com/cvgroup-njust/CityPersons/blob/master/annotations/README.txt
    """
    from scipy.io import loadmat

    if isinstance(person_class_id, bool) or not isinstance(person_class_id, int) or person_class_id < 1:
        raise ValueError("person_class_id must be a positive integer in the detector label space")
    if isinstance(coordinate_origin, bool) or coordinate_origin not in (0, 1):
        raise ValueError("coordinate_origin must be 0 (preserve) or 1 (subtract one from x/y)")
    if subset not in ("all", "reasonable"):
        raise ValueError("CityPersons subset must be all or reasonable")
    if dataset_split is not None and dataset_split not in ("train", "val", "test"):
        raise ValueError("CityPersons dataset_split must be train, val or test")
    document = loadmat(annotations_file, simplify_cells=True)
    keys = [key for key in document if re.fullmatch(r"anno_(train|val|test)(?:_aligned)?", key)]
    if dataset_split is not None:
        keys = [key for key in keys if re.fullmatch(rf"anno_{dataset_split}(?:_aligned)?", key)]
    if len(keys) != 1:
        raise ValueError("CityPersons MAT must contain one matching anno_<split>[_aligned] variable; specify dataset_split if needed")
    key = keys[0]
    split = key.split("_")[1]
    rows = document[key]
    if isinstance(rows, dict):
        rows = [rows]
    elif isinstance(rows, np.ndarray):
        rows = rows.reshape(-1).tolist()
    if not isinstance(rows, list) or not rows:
        raise ValueError("CityPersons annotation variable must contain image structs")
    images: list[DetectionImage] = []
    seen: set[str] = set()
    for source_index, source in enumerate(rows):
        if not isinstance(source, dict) or "bbs" not in source or "im_name" not in source:
            raise ValueError("CityPersons image entries require cityname, im_name and bbs fields")
        city = source.get("cityname", source.get("city_name"))
        name = source["im_name"]
        if (not isinstance(city, str) or not city or Path(city).name != city
                or not isinstance(name, str) or not name or Path(name).name != name
                or city in (".", "..") or name in (".", "..")
                or any(x in city + name for x in (":", "\\"))):
            raise ValueError("CityPersons city/image names must be nonempty safe path components")
        image_id = f"citypersons:{split}:{city}:{name}"
        if image_id in seen:
            raise ValueError(f"Duplicate CityPersons image ID: {image_id}")
        seen.add(image_id)
        try:
            boxes = np.asarray(source["bbs"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CityPersons image {name}: bbs must be a numeric N x 10 matrix") from exc
        if boxes.size == 0:
            boxes = np.empty((0, 10), dtype=np.float64)
        elif boxes.ndim == 1 and boxes.size == 10:
            boxes = boxes.reshape(1, 10)
        if (boxes.ndim != 2 or boxes.shape[1] != 10 or not np.isfinite(boxes).all()
                or np.any(boxes[:, 3:5] <= 0) or np.any(boxes[:, 8:10] < 0)):
            raise ValueError(f"CityPersons image {name}: bbs must be finite N x 10 with positive full-box size")
        if (not np.equal(boxes[:, 0], np.floor(boxes[:, 0])).all()
                or not np.isin(boxes[:, 0], np.arange(6)).all()
                or not np.equal(boxes[:, 5], np.floor(boxes[:, 5])).all()):
            raise ValueError(f"CityPersons image {name}: source labels must be integers 0..5 and instance IDs integers")
        annotations, attributes = [], []
        for number, box in enumerate(boxes):
            full, visible = box[1:5].copy(), box[6:10].copy()
            full[:2] -= coordinate_origin
            visible[:2] -= coordinate_origin
            ratio = float(visible[2] * visible[3] / (full[2] * full[3]))
            ignored = int(box[0]) != 1 or (subset == "reasonable" and (full[3] < 50 or ratio < .65))
            annotations.append({"id": number, "category_id": person_class_id,
                                "bbox": full.tolist(), "ignore": bool(ignored), "iscrowd": False})
            attributes.append({"annotation_id": number, "source_class": int(box[0]),
                               "source_instance_id": int(box[5]), "source_full_bbox": box[1:5].tolist(),
                               "visible_bbox": visible.tolist(), "visible_fraction": ratio})
        metadata = {"dataset": "citypersons", "dataset_split": split,
                    "original_image_id": name, "cityname": city,
                    "source_image_index": source_index, "matlab_variable": key,
                    "annotation_box_type": "aligned_full_body", "coordinate_origin": int(coordinate_origin),
                    "coordinate_transform": "x/y -= coordinate_origin; width/height unchanged",
                    "coordinate_reference": "Published CityPersons anno_val.mat and evaluation/val_gt.json keep the same numeric bbox coordinates; default 0 preserves them.",
                    "annotation_reference": "https://github.com/cvgroup-njust/CityPersons/blob/master/annotations/README.txt",
                    "evaluation_annotation_reference": "https://github.com/cvgroup-njust/CityPersons/blob/master/evaluation/val_gt.json",
                    "source_annotation_attributes": attributes, "full_body_boxes_clipped": False,
                    "citypersons_subset": subset,
                    "gt_ignore_convention": "source class !=1" + (" or height<50 or visible/full area<0.65" if subset == "reasonable" else ""),
                    "official_miss_rate_evaluator_applied": False}
        images.append(DetectionImage(image_id=image_id, original_image_id=name,
                                     file_name=str(Path(city) / name), ground_truth=annotations, metadata=metadata))
    images.sort(key=lambda item: item.image_id)
    return images, split
