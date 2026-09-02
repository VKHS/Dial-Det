from pathlib import Path

from dial_det.config.rule import load_complete_rule
from dial_det.examples import synthetic_encoded_stream
from dial_det.pipeline import run_pipeline

root = Path(__file__).resolve().parents[1]
stream = synthetic_encoded_stream()
for name in (
    "dial_out_at_most.json",
    "dial_out_exact.json",
    "dial_prop_exact.json",
    "dial_prop_cost.json",
):
    rule = load_complete_rule(root / "configs" / "smoke" / name)
    result = run_pipeline(stream, rule)
    print(name, result.selected_candidate_ids, result.fallback_triggered)
