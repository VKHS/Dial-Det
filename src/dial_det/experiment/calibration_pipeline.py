from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dial_det.config.rule import load_complete_rule
from dial_det.config.splits import SplitManifest, load_split_manifest
from dial_det.method.calibration import CalibrationResult, calibrate_complete_rules


@dataclass(frozen=True)
class RuleLossFile:
    rule_id: str
    rule_sha256: str
    sample_ids: tuple[str, ...]
    losses: Mapping[str, np.ndarray]


def load_rule_loss_file(path: str | Path) -> RuleLossFile:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"sample_ids", "rule_id", "rule_sha256"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{source.name} is missing: {', '.join(missing)}")
        sample_ids = tuple(str(item) for item in archive["sample_ids"].astype(str))
        rule_id = str(archive["rule_id"].item())
        rule_sha256 = str(archive["rule_sha256"].item())
        losses = {
            name.removeprefix("loss_"): np.asarray(archive[name], dtype=np.float64)
            for name in archive.files
            if name.startswith("loss_")
        }
    if not losses:
        raise ValueError(f"{source.name} contains no loss_* vectors")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{source.name} contains duplicate sample IDs")
    for name, values in losses.items():
        if values.shape != (len(sample_ids),):
            raise ValueError(f"{source.name}: loss_{name} must have one value per sample")
    return RuleLossFile(rule_id, rule_sha256, sample_ids, losses)


def _require_exact_ids(actual: tuple[str, ...], expected: tuple[str, ...], role: str) -> None:
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise ValueError(f"{role} IDs differ from manifest; missing={missing}, extra={extra}")


def run_calibration_specification(specification_path: str | Path) -> CalibrationResult:
    specification = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    base = Path(specification_path).resolve().parent
    split_path = (base / specification["split_manifest"]).resolve()
    split = load_split_manifest(split_path)
    loss_files: list[RuleLossFile] = []
    complete_hashes: dict[str, str] = {}
    for item in specification["rules"]:
        rule = load_complete_rule((base / item["complete_rule"]).resolve())
        losses = load_rule_loss_file((base / item["calibration_losses"]).resolve())
        if losses.rule_id != rule.rule_id or losses.rule_sha256 != rule.sha256:
            raise ValueError(f"loss file does not match complete rule {rule.rule_id}")
        _require_exact_ids(losses.sample_ids, split.calibration_ids, "calibration")
        loss_files.append(losses)
        complete_hashes[rule.rule_id] = rule.sha256
    if len({entry.sample_ids for entry in loss_files}) != 1:
        raise ValueError("all rules must be evaluated on the same ordered calibration IDs")

    utilities = None
    if "utility_values" in specification:
        utility_path = (base / specification["utility_values"]).resolve()
        with np.load(utility_path, allow_pickle=False) as archive:
            utility_ids = tuple(str(item) for item in archive["sample_ids"].astype(str))
            _require_exact_ids(utility_ids, split.utility_ids, "utility")
            utilities = {}
            for entry in loss_files:
                key = f"utility_{entry.rule_id}"
                if key not in archive.files:
                    raise ValueError(f"utility file is missing {key}")
                values = np.asarray(archive[key], dtype=np.float64)
                if values.shape != (len(utility_ids),):
                    raise ValueError(f"{key} must have one value per utility image")
                utilities[entry.rule_id] = float(values.mean())

    losses_by_rule = {entry.rule_id: entry.losses for entry in loss_files}
    return calibrate_complete_rules(
        losses_by_rule,
        risk_targets=specification["risk_targets"],
        alpha=float(specification["alpha"]),
        utilities=utilities,
        complete_rule_hashes=complete_hashes,
        calibration_split_id=f"{split.dataset}:{split.version}:calibration:{split.sha256}",
        utility_split_id=f"{split.dataset}:{split.version}:utility:{split.sha256}",
        payloads={entry.rule_id: {"loss_file_rule_sha256": entry.rule_sha256} for entry in loss_files},
    )
