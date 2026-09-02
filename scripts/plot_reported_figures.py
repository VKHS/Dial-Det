"""Reconstruct selected plots from article-reported numeric coordinates.

The outputs are reference reconstructions, not plots recomputed from raw
predictions.  Every generated directory contains that provenance label.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires `pip install -e '.[plots]'`") from exc
    root = Path(__file__).resolve().parents[1]
    output = root / "runs" / "reported_reference_plots"
    output.mkdir(parents=True, exist_ok=True)

    data = rows(root / "results" / "figure_data" / "figure_2b_same_evidence.csv")
    names = [entry["stream"] for entry in data]
    values = [float(entry["delta_ap"]) for entry in data]
    errors = [float(entry["half_interval"]) for entry in data]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(values, range(len(names)), xerr=errors, fmt="o")
    ax.axvline(0, linestyle="--")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("DIAL minus matched-local AP")
    ax.set_title("Reference reconstruction from article-reported values")
    fig.tight_layout()
    fig.savefig(output / "figure_2b_reference.png", dpi=200)
    plt.close(fig)

    data = rows(root / "results" / "figure_data" / "figure_7_proposal.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for selector in sorted({entry["selector"] for entry in data if entry["panel"] == "total_latency"}):
        selected = [entry for entry in data if entry["panel"] == "total_latency" and entry["selector"] == selector]
        ax.plot([float(entry["M"]) for entry in selected], [float(entry["value"]) for entry in selected], marker="o", label=selector)
    ax.set_xlabel("proposal budget M")
    ax.set_ylabel("total latency (ms)")
    ax.set_title("Reference reconstruction from article-reported values")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figure_7b_reference.png", dpi=200)
    plt.close(fig)

    (output / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "provenance_status": "reported_in_article_reference_reconstruction",
                "raw_predictions_used": False,
                "claim": "Plots are reconstructed from numeric coordinates shipped in results/figure_data.",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
