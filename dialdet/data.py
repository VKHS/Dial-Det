"""Frozen candidate records, provenance and leakage-resistant split manifests.

All box coordinates are original-image xyxy. NPZ files contain no pickled objects.
The decoder never modifies IDs, classes, boxes or detector scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
from typing import Any

import numpy as np

ROLES = {"train", "dev", "calibration", "audit", "pool"}


def canonical_dataset(name):
    """Normalize known detection dataset names without relabeling custom data."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A detection manifest needs a nonempty dataset name")
    compact = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {"coco": "coco", "coco2017": "coco", "mscoco": "coco",
               "crowdhuman": "crowdhuman", "citypersons": "citypersons",
               "synthetic": "synthetic", "generic": "generic"}
    return aliases.get(compact, name.lower().strip())


def dataset_quality_mode(dataset):
    """Appendix B.4: COCO ten-threshold mean; pedestrian datasets p_0.50."""
    return {"coco": "coco", "crowdhuman": "iou50", "citypersons": "iou50"}.get(
        canonical_dataset(dataset))


def classify_synthetic(document, records=()):
    """Preserve unknown data provenance instead of inferring a benchmark run."""
    flags = [record.metadata.get("synthetic") for record in records]
    if (document["dataset"].startswith("synthetic") or
            str(document.get("protocol", "")).startswith("synthetic") or
            document.get("is_synthetic") is True or any(flag is True for flag in flags)):
        return True
    if (document["dataset"] in ("coco", "crowdhuman", "citypersons") or
            document.get("is_synthetic") is False or
            (flags and all(flag is False for flag in flags))):
        return False
    return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CandidateRecord:
    image_id: str
    ids: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    scores: np.ndarray
    features: np.ndarray
    image_size: tuple[int, int]
    gt_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float64))
    gt_classes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    gt_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    gt_ignore: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.image_id = str(self.image_id)
        for name in ("ids", "classes", "gt_classes", "gt_ids"):
            raw = np.asarray(getattr(self, name))
            if raw.size and (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()):
                raise ValueError(f"{name} must contain finite integer values")
            setattr(self, name, raw.astype(np.int64))
        for name in ("boxes", "scores", "gt_boxes"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.features = np.asarray(self.features, dtype=np.float32)
        self.gt_ignore = np.asarray(self.gt_ignore, dtype=bool)
        self.image_size = tuple(int(x) for x in self.image_size)
        self.validate()

    def validate(self):
        n, m = len(self.ids), len(self.gt_ids)
        if n > 500:
            raise ValueError("DIAL-Det exports at most 500 candidates per image")
        if len(set(self.ids.tolist())) != n or len(set(self.gt_ids.tolist())) != m:
            raise ValueError("Candidate and annotation IDs must each be unique within an image")
        for name, shape in (("boxes", (n, 4)), ("classes", (n,)), ("scores", (n,)),
                            ("gt_boxes", (m, 4)), ("gt_classes", (m,)), ("gt_ignore", (m,))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Invalid {name} shape: expected {shape}")
        if self.features.ndim != 2 or self.features.shape[0] != n or self.features.shape[1] < 1:
            raise ValueError("features must have shape [N,D], D>=1, including empty images")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size is (height,width), both positive")
        for name in ("boxes", "scores", "features", "gt_boxes"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"Non-finite {name}")
        if np.any(self.scores < 0) or np.any(self.scores > 1):
            raise ValueError("Native scores must be probabilities in [0,1]")
        if np.any(self.classes < 0) or np.any(self.gt_classes < 0):
            raise ValueError("Class IDs must be nonnegative")
        for boxes in (self.boxes, self.gt_boxes):
            if np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError("Boxes must have strictly positive width and height")
        return self


ARRAY_FIELDS = ("ids", "boxes", "classes", "scores", "features", "gt_boxes", "gt_classes", "gt_ids", "gt_ignore")


def record_hash(record):
    """Hash the frozen numerical interface, independently of file compression."""
    digest = hashlib.sha256()
    digest.update(json.dumps([record.image_id, record.image_size]).encode())
    for name in ARRAY_FIELDS:
        value = np.ascontiguousarray(getattr(record, name))
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_record(record, path):
    record.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {name: getattr(record, name) for name in ARRAY_FIELDS}
    values.update(image_id=np.asarray(record.image_id), image_size=np.asarray(record.image_size),
                  metadata_json=np.asarray(json.dumps(record.metadata, sort_keys=True, allow_nan=False)))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def load_record(path):
    with np.load(path, allow_pickle=False) as values:
        missing = set(ARRAY_FIELDS) - set(values.files)
        if missing:
            raise ValueError(f"Missing frozen-export arrays: {sorted(missing)}")
        return CandidateRecord(**{name: values[name] for name in ARRAY_FIELDS},
                               image_id=str(values["image_id"].item()),
                               image_size=tuple(values["image_size"].tolist()),
                               metadata=json.loads(str(values["metadata_json"].item())))


def manifest_document(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Expected version-1 JSON detection manifest")
    value["dataset"] = canonical_dataset(value.get("dataset", "generic"))
    entries = value.get("records", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("The manifest must contain records")
    seen_ids, seen_paths = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest record must be an object")
        if entry.get("role") not in ROLES:
            raise ValueError(f"Unknown data role {entry.get('role')}")
        image_id = str(entry["image_id"])
        if not image_id or entry["image_id"] is None:
            raise ValueError("Manifest image IDs must be nonempty")
        for name in ("sha256", "numerical_sha256", "semantic_sha256"):
            if name in entry and (not isinstance(entry[name], str) or re.fullmatch(r"[0-9a-f]{64}", entry[name]) is None):
                raise ValueError(f"Manifest {name} must be a lowercase SHA-256 digest")
        source = (path.parent / entry["path"]).resolve()
        if image_id in seen_ids or str(source) in seen_paths:
            raise ValueError("An image occurs more than once; train/dev/calibration/audit must be disjoint")
        seen_ids.add(image_id)
        seen_paths.add(str(source))
        if not source.is_file():
            raise FileNotFoundError(source)
    return value


def load_manifest(path, role=None):
    doc = manifest_document(path)
    records = []
    for entry in doc["records"]:
        if role is not None and entry["role"] != role:
            continue
        source = (Path(path).parent / entry["path"]).resolve()
        if entry.get("sha256") and sha256_file(source) != entry["sha256"]:
            raise ValueError(f"Candidate export hash mismatch: {source}")
        rec = load_record(source)
        if rec.image_id != str(entry["image_id"]):
            raise ValueError("Manifest/record image-ID mismatch")
        rec.metadata = {**rec.metadata, "role": entry["role"], "dataset": doc.get("dataset", "unknown"),
                        "protocol": doc.get("protocol", "unspecified"), "source_path": str(source)}
        if entry.get("numerical_sha256") and entry["numerical_sha256"] != record_hash(rec):
            raise ValueError(f"Candidate numerical_sha256 mismatch: {rec.image_id}")
        if entry.get("semantic_sha256"):
            from .provenance import candidate_semantic_hash
            if entry["semantic_sha256"] != candidate_semantic_hash(rec):
                raise ValueError(f"Candidate semantic_sha256 mismatch: {rec.image_id}")
        records.append(rec)
    return records


def bbox_iou(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1, 4), np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(hi - lo, 0).prod(axis=-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return np.divide(inter, area_a[:, None] + area_b[None, :] - inter,
                     out=np.zeros_like(inter), where=(area_a[:, None] + area_b[None, :] - inter) > 0)


def assignment_targets(record, iou_threshold=.5, ignore_ioa=.5):
    """Highest-IoU eligible GT; annotation-ID ties, private background identities.

    Ambiguous ignore-only candidates are excluded from training losses. Physical
    report capacity still counts them in the allocation audit; official COCOeval
    independently applies its own crowd/ignore policy.
    """
    n = len(record.ids)
    labels = -1 - np.arange(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    nonignored = np.flatnonzero(~record.gt_ignore)
    nonignored = nonignored[np.argsort(record.gt_ids[nonignored], kind="stable")]
    iou = bbox_iou(record.boxes, record.gt_boxes[nonignored])
    if nonignored.size:
        iou = np.where(record.classes[:, None] == record.gt_classes[nonignored][None, :], iou, -1.)
        local = iou.argmax(axis=1)
        best = iou[np.arange(n), local]
        positive = best >= iou_threshold
        labels[positive] = nonignored[local[positive]]
    else:
        best = np.zeros(n)
        positive = np.zeros(n, dtype=bool)
    ignored = np.flatnonzero(record.gt_ignore)
    if ignored.size:
        lo = np.maximum(record.boxes[:, None, :2], record.gt_boxes[ignored][None, :, :2])
        hi = np.minimum(record.boxes[:, None, 2:], record.gt_boxes[ignored][None, :, 2:])
        inter = np.maximum(hi - lo, 0).prod(axis=-1)
        area = (record.boxes[:, 2:] - record.boxes[:, :2]).prod(axis=1)
        ioa = inter / np.maximum(area[:, None], 1e-12)
        compat = record.classes[:, None] == record.gt_classes[ignored][None, :]
        valid = positive | ~np.any((ioa >= ignore_ioa) & compat, axis=1)
    thresholds = .5 + .05 * np.arange(10)
    quality = (best[:, None] + 1e-12 >= thresholds[None, :]).astype(np.float32)
    return {"assignments": labels, "quality_targets": quality, "valid_candidates": valid,
            "best_iou": np.maximum(best, 0)}


def make_crossfit_manifests(source, output, seed=20261022, dev_fraction=.2):
    """Opposite-half fitting and evaluation for public frozen checkpoints."""
    source, output = Path(source), Path(output)
    doc = manifest_document(source)
    entries = doc["records"]
    if len(entries) < 6 or not 0 < dev_fraction < .5:
        raise ValueError("Cross-fitting needs >=6 images and 0<dev_fraction<.5")
    order = np.random.default_rng(seed).permutation(len(entries))
    halves = np.array_split(order, 2)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in range(2):
        audit, fitting = halves[fold], halves[1 - fold]
        ndev = max(1, int(round(len(fitting) * dev_fraction)))
        dev, train = fitting[:ndev], fitting[ndev:]
        new_entries = []
        for role, indices in (("train", train), ("dev", dev), ("audit", audit)):
            for idx in indices:
                entry = dict(entries[int(idx)])
                entry.update(role=role, path=str((source.parent / entry["path"]).resolve()))
                new_entries.append(entry)
        result = {**doc, "records": new_entries, "protocol": "public_checkpoint_crossfit",
                  "risk_eligible": False, "fold": fold, "split_seed": seed,
                  "source_manifest_sha256": sha256_file(source),
                  "note": "Train/dev is inside the opposite half; each image is audited once across the two folds."}
        target = output / f"fold_{fold}.json"
        atomic_json(target, result)
        paths.append(str(target))
    return paths
