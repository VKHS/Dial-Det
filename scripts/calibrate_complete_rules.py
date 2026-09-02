from __future__ import annotations

import argparse
import json
from pathlib import Path

from dial_det.experiment.calibration_pipeline import run_calibration_specification


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a fixed family of complete DIAL rules")
    parser.add_argument("--specification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_calibration_specification(args.specification)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
