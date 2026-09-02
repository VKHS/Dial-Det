from __future__ import annotations

import argparse

from dial_det.io.npz import load_encoded_stream, save_encoded_stream
from dial_det.training.inference import encode_with_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a trained DIAL evidence checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--active-relations",
        help="comma-separated frozen relation names; default: stream metadata",
    )
    args = parser.parse_args()
    stream = load_encoded_stream(args.input)
    active = (
        tuple(item.strip() for item in args.active_relations.split(",") if item.strip())
        if args.active_relations
        else None
    )
    encoded = encode_with_checkpoint(
        stream, args.checkpoint, device=args.device, active_relations=active
    )
    save_encoded_stream(args.output, encoded)
    print(encoded.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
