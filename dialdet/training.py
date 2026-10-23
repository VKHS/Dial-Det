"""Evidence training, held-out temperature fitting, and inference utilities.

There are no detector downloads and no evaluation-label fitting in this module.
Calibration images in the paper mean *risk calibration*: they are explicitly
rejected for evidence fitting and temperature selection.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .data import CandidateRecord, assignment_targets, record_hash
from .evidence import EvidenceModel, evidence_losses


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _tensors(record: CandidateRecord, device: str | torch.device) -> dict[str, Any]:
    return {"boxes": torch.as_tensor(record.boxes, dtype=torch.float32, device=device),
            "classes": torch.as_tensor(record.classes, dtype=torch.long, device=device),
            "scores": torch.as_tensor(record.scores, dtype=torch.float32, device=device),
            "features": torch.as_tensor(record.features, dtype=torch.float32, device=device),
            "image_size": tuple(record.image_size)}


def _targets(record: CandidateRecord, device: str | torch.device, iou: float) -> dict[str, torch.Tensor]:
    values = assignment_targets(record, iou_threshold=iou)
    return {"assignments": torch.as_tensor(values["assignments"], dtype=torch.long, device=device),
            "quality_targets": torch.as_tensor(values["quality_targets"], dtype=torch.float32, device=device),
            "valid_candidates": torch.as_tensor(values["valid_candidates"], dtype=torch.bool, device=device)}


def _validate_split(records: Sequence[CandidateRecord], role: str) -> list[str]:
    ids = [str(record.image_id) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate image_id in {role} records")
    for record in records:
        declared = record.metadata.get("role")
        if declared in {"audit", "calibration"}:
            raise ValueError(f"{declared} images cannot be used for {role}")
        if declared not in {None, "pool", role}:
            raise ValueError(f"record {record.image_id} declares role {declared!r}, expected {role!r}")
    return ids


def _split_description(records: Sequence[CandidateRecord], role: str) -> dict[str, Any]:
    entries = [{"image_id": str(r.image_id), "sha256": record_hash(r)} for r in records]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"role": role, "count": len(records), "records": entries,
            "sha256": hashlib.sha256(encoded).hexdigest()}


@torch.no_grad()
def predict_record(record: CandidateRecord, model: EvidenceModel, temperature: float = 1.0,
                   p: float = 2.0, quality_mode: str = "coco",
                   device: str | torch.device | None = None) -> dict[str, np.ndarray]:
    """Compute frozen evidence once; reuse it for every allocation budget."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    output = model(**_tensors(record, device), temperature=temperature, p=p, quality_mode=quality_mode)
    names = ("q", "r", "d", "quality_probs", "support", "overlap", "rescue")
    return {name: output[name].detach().cpu().numpy() for name in names}


def load_evidence_checkpoint(path: str | Path, device: str = "cpu") -> tuple[EvidenceModel, dict[str, Any]]:
    """Read a DIAL-Det evidence checkpoint using PyTorch's restricted loader."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "dialdet-evidence-v1":
        raise ValueError("not a DIAL-Det evidence checkpoint")
    model = EvidenceModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    metadata = {key: value for key, value in payload.items()
                if key not in {"model_state", "optimizer_state", "torch_rng_state", "cuda_rng_state"}}
    # Expose the saved population identities in the CLI's leakage-guard shape.
    # These are derived from the checkpoint's hashed population description,
    # never from the inference manifest being checked.
    for role in ("train", "dev"):
        metadata[f"{role}_image_ids"] = [str(item["image_id"])
                                         for item in payload["populations"][role]["records"]]
    return model, metadata


@torch.no_grad()
def fit_temperature(model: EvidenceModel, dev_records: Sequence[CandidateRecord], device: str = "cpu",
                    assignment_iou: float = 0.5) -> dict[str, Any]:
    """Fit a positive identity-logit temperature on development edges only.

    All natural eligible pairs are included with weight one. The log-temperature
    search interval [.05, 20] and edge-pooled normalization are recorded in the
    output. Calibration adjusts identity logits; quality logits retain their
    trained values.
    """
    from scipy.optimize import minimize_scalar

    _validate_split(dev_records, "dev")
    model.eval()
    logits, targets = [], []
    for record in dev_records:
        output = model(**_tensors(record, device))
        target = _targets(record, device, assignment_iou)
        valid = target["valid_candidates"]
        mask = output["pair_mask"] & valid[:, None] & valid[None, :]
        assignments = target["assignments"]
        if mask.any():
            logits.append(output["pair_logits"][mask].cpu().numpy().astype(np.float64))
            targets.append((assignments[:, None] == assignments[None, :])[mask].cpu().numpy().astype(np.float64))
    if not logits:
        return {"temperature": 1.0, "edges": 0, "status": "NO_ELIGIBLE_PAIRS",
                "loss_before": None, "loss_after": None, "population": "development"}
    x, y = np.concatenate(logits), np.concatenate(targets)

    def objective(log_t: float) -> float:
        z = x / math.exp(log_t)
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    bounds = (math.log(0.05), math.log(20.0))
    optimized = minimize_scalar(objective, method="bounded", bounds=bounds,
                                options={"xatol": 1e-7, "maxiter": 200})
    # Retain T=1 if the optimizer cannot improve the held-out likelihood.
    candidates = [(0.0, objective(0.0)), (bounds[0], objective(bounds[0])),
                  (bounds[1], objective(bounds[1]))]
    if optimized.success and math.isfinite(float(optimized.fun)):
        candidates.append((float(optimized.x), float(optimized.fun)))
    best_log_t, best_value = min(candidates, key=lambda pair: pair[1])
    return {"temperature": math.exp(best_log_t), "edges": int(x.size),
            "status": "FITTED" if optimized.success else "BOUNDARY_OR_IDENTITY_FALLBACK",
            "loss_before": objective(0.0), "loss_after": best_value,
            "temperature_bounds": [0.05, 20.0], "population": "development",
            "normalization": "all eligible unordered support edges, weight one"}


def train(train_records: Sequence[CandidateRecord], dev_records: Sequence[CandidateRecord],
          output_dir: str | Path, epochs: int = 30, seed: int = 1, device: str = "cpu",
          resume: str | Path | None = None, quality_mode: str = "coco",
          model_config: dict[str, Any] | None = None, learning_rate: float = 3e-4,
          weight_decay: float = 1e-4, beta: float = 0.1, gamma: float = 0.05,
          p: float = 2.0, assignment_iou: float = 0.5, gradient_clip: float = 5.0,
          num_classes: int | None = None) -> dict[str, Any]:
    """Train image-by-image, save each completed epoch, then calibrate on dev.

    Epoch order is a deterministic function of (seed, epoch), making completed-
    epoch resume explicit. Fixed 30 epochs is the paper default; best development
    loss chooses the saved evidence model. Batching, hidden width, activations
    and gradient clipping are documented implementation choices.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not train_records or not dev_records:
        raise ValueError("nonempty, disjoint train and development populations are required")
    train_ids = _validate_split(train_records, "train")
    dev_ids = _validate_split(dev_records, "dev")
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and development image IDs overlap")
    feature_dims = {int(record.features.shape[1]) for record in [*train_records, *dev_records]}
    if len(feature_dims) != 1:
        raise ValueError("every candidate record must use one frozen feature dimension")
    largest_class = max((int(record.classes.max()) for record in [*train_records, *dev_records]
                         if len(record.classes)), default=0)
    config = dict(model_config or {})
    config.setdefault("feature_dim", feature_dims.pop())
    config.setdefault("num_classes", num_classes if num_classes is not None else largest_class + 1)
    if config["num_classes"] <= largest_class:
        raise ValueError("num_classes does not cover candidate class IDs")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = EvidenceModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path, calibrated_path = output / "last.pt", output / "best.pt", output / "calibrated.pt"
    if last_path.exists() and resume is None:
        raise FileExistsError(f"{last_path} already exists; use resume or a new output directory")
    populations = {"train": _split_description(train_records, "train"),
                   "dev": _split_description(dev_records, "dev")}
    training_config = {"epochs": epochs, "seed": seed, "quality_mode": quality_mode,
                       "learning_rate": learning_rate, "weight_decay": weight_decay,
                       "beta": beta, "gamma": gamma, "p": p, "assignment_iou": assignment_iou,
                       "gradient_clip": gradient_clip, "image_batch_size": 1,
                       "pair_sampling": "all natural unordered support pairs; P=Q; weight=1",
                       "epoch_selection": "minimum development full evidence loss",
                       "detector_training": "not performed; frozen candidate exports required",
                       "risk_certification": "not performed by evidence training"}
    history: list[dict[str, Any]] = []
    start_epoch, best_loss = 0, math.inf
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=True)
        if payload.get("schema") != "dialdet-evidence-v1":
            raise ValueError("resume checkpoint has an unknown schema")
        if payload["model_config"] != model.config_dict():
            raise ValueError("resume model configuration differs")
        if payload["populations"] != populations:
            raise ValueError("resume candidate populations or contents differ")
        for name in ("seed", "quality_mode", "learning_rate", "weight_decay", "beta", "gamma", "p", "assignment_iou", "gradient_clip"):
            if payload["training_config"][name] != training_config[name]:
                raise ValueError(f"resume training configuration differs at {name}")
        if "optimizer_state" not in payload:
            raise ValueError("resume requires last.pt, which contains optimizer state")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch, best_loss = int(payload["epoch"]), float(payload["best_dev_loss"])
        history = list(payload["history"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
        if not best_path.exists():
            source_best = Path(resume).resolve().parent / "best.pt"
            if not source_best.exists():
                raise FileNotFoundError("resume requires the sibling best.pt to preserve development selection")
            import shutil
            shutil.copy2(source_best, best_path)
    _atomic_json(output / "populations.json", populations)
    _atomic_json(output / "training_config.json", {**training_config, "model": model.config_dict()})

    def process(record: CandidateRecord) -> dict[str, Any]:
        prediction = model(**_tensors(record, device), p=p, quality_mode=quality_mode)
        return evidence_losses(prediction, **_targets(record, device, assignment_iou), beta=beta, gamma=gamma, p=p)

    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(train_records))
        train_sums = {name: 0.0 for name in ("total", "quality", "pair", "mass", "transitivity")}
        nonempty = 0
        for index in order:
            record = train_records[int(index)]
            if len(record.ids) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = process(record)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"nonfinite training loss at image {record.image_id}")
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            nonempty += 1
            for name in train_sums:
                train_sums[name] += float(losses[name].detach().cpu())
        if nonempty == 0:
            raise ValueError("training records contain no candidates")
        model.eval()
        dev_sums = {name: 0.0 for name in train_sums}
        dev_nonempty = 0
        with torch.no_grad():
            for record in dev_records:
                if len(record.ids) == 0:
                    continue
                losses = process(record)
                dev_nonempty += 1
                for name in dev_sums:
                    dev_sums[name] += float(losses[name].cpu())
        if dev_nonempty == 0:
            raise ValueError("development records contain no candidates")
        row = {"epoch": epoch + 1, "train": {k: v / nonempty for k, v in train_sums.items()},
               "dev": {k: v / dev_nonempty for k, v in dev_sums.items()}}
        if not all(math.isfinite(v) for side in (row["train"], row["dev"]) for v in side.values()):
            raise FloatingPointError("nonfinite epoch metrics")
        history.append(row)
        improved = row["dev"]["total"] < best_loss
        best_loss = min(best_loss, row["dev"]["total"])
        checkpoint = {"schema": "dialdet-evidence-v1", "model_config": model.config_dict(),
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "epoch": epoch + 1, "best_dev_loss": best_loss, "training_config": training_config,
                      "history": history, "populations": populations, "temperature": 1.0,
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        if improved:
            _save_checkpoint(best_path, {k: v for k, v in checkpoint.items() if k != "optimizer_state"})
        _save_checkpoint(last_path, checkpoint)
        _atomic_json(output / "history.json", history)
        print(json.dumps({"stage": "evidence_training", **row}), flush=True)
    model, best_metadata = load_evidence_checkpoint(best_path, device)
    temperature = fit_temperature(model, dev_records, device, assignment_iou)
    final_payload = {"schema": "dialdet-evidence-v1", "model_config": model.config_dict(),
                     "model_state": model.state_dict(), **best_metadata,
                     "temperature": temperature["temperature"], "temperature_fit": temperature}
    _save_checkpoint(calibrated_path, final_payload)
    summary = {"checkpoint": str(calibrated_path.resolve()), "best_checkpoint": str(best_path.resolve()),
               "last_checkpoint": str(last_path.resolve()), "epochs_completed": len(history),
               "selected_epoch": best_metadata["epoch"], "best_dev_loss": best_loss,
               "temperature_fit": temperature, "model_config": model.config_dict(),
               "training_config": training_config,
               "measured_scope": "evidence training/development losses only; no benchmark AP or risk certificate"}
    _atomic_json(output / "training_summary.json", summary)
    return summary
