from __future__ import annotations

import argparse
import json
from pathlib import Path

from dial_det.training import TrainingConfiguration, train_graph_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the role-routed DIAL evidence encoder")
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--configuration", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.configuration).read_text(encoding="utf-8"))
    payload["relation_types"] = tuple(payload["relation_types"])
    result = train_graph_directory(
        args.data_directory,
        args.output_directory,
        TrainingConfiguration(**payload),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
