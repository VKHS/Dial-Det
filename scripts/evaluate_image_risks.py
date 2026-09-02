from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dial_det.experiment.metrics import evaluate_selected_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DIAL image-level bounded risks")
    parser.add_argument("--input", required=True, help="NPZ with selected/ground-truth boxes")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        kwargs = {
            "image_id": str(archive["image_id"].item()),
            "selected_boxes": archive["selected_boxes"],
            "ground_truth_boxes": archive["ground_truth_boxes"],
            "budget_k": args.budget,
            "iou_threshold": args.iou_threshold,
        }
        for target, key in (
            ("selected_labels", "selected_labels"),
            ("ground_truth_labels", "ground_truth_labels"),
            ("ground_truth_ignored", "ground_truth_ignored"),
            ("proposal_boxes", "proposal_boxes"),
            ("proposal_labels", "proposal_labels"),
        ):
            if key in archive.files:
                kwargs[target] = archive[key]
    result = evaluate_selected_image(**kwargs)
    Path(args.output).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
