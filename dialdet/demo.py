"""Small synthetic fixtures for execution checks, never benchmark evidence."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .data import CandidateRecord, atomic_json, save_record, sha256_file


def create_synthetic_dataset(output, seed=7, counts=(8, 4, 4)):
    output = Path(output)
    rng = np.random.default_rng(seed)
    entries = []
    for role, count in zip(("train", "dev", "audit"), counts):
        for number in range(count):
            image_id = f"synthetic-{role}-{number:04d}"
            gt = np.array([[20, 20, 75, 130], [50, 24, 108, 137], [155, 40, 205, 145]], dtype=float)
            gt += rng.normal(0, 1., gt.shape)
            features, boxes, scores = [], [], []
            for identity, box in enumerate(gt):
                for duplicate in range(2):
                    boxes.append(box + rng.normal(0, 1.8 + duplicate, 4))
                    feat = np.zeros(16)
                    feat[identity] = 1.
                    features.append(feat + rng.normal(0, .025, 16))
                    scores.append(.92 - .07 * duplicate - .04 * identity)
            boxes.append([220, 190, 250, 240])
            features.append(rng.normal(0, .2, 16))
            scores.append(.12)
            rec = CandidateRecord(image_id=image_id, ids=np.arange(7), boxes=np.asarray(boxes),
                                  classes=np.ones(7, dtype=int), scores=np.asarray(scores),
                                  features=np.asarray(features), image_size=(256, 256), gt_boxes=gt,
                                  gt_classes=np.ones(3, dtype=int), gt_ids=np.arange(3), gt_ignore=np.zeros(3, dtype=bool),
                                  metadata={"synthetic": True, "role": role, "dataset": "synthetic",
                                            "purpose": "mechanical and numerical verification only"})
            path = output / "records" / f"{image_id}.npz"
            save_record(rec, path)
            entries.append({"path": str(path.relative_to(output)), "image_id": image_id,
                            "role": role, "sha256": sha256_file(path)})
    manifest = output / "manifest.json"
    atomic_json(manifest, {"schema_version": 1, "dataset": "synthetic", "protocol": "synthetic",
                          "risk_eligible": False, "quality_mode": "iou50", "seed": seed, "records": entries,
                          "warning": "These synthetic records are execution fixtures, not detection-benchmark results."})
    return str(manifest)


def analytic_evidence(record):
    """Known synthetic identities; prohibited for real inference."""
    if not record.metadata.get("synthetic"):
        raise ValueError("Analytic smoke evidence is restricted to synthetic records")
    q = np.asarray([.95, .85, .9, .8, .85, .75, .01])
    r = np.eye(7)
    for i in (0, 2, 4):
        r[i, i+1] = r[i+1, i] = 1.
    return q, r

