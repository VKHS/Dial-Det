from pathlib import Path

from dial_det.config.rule import load_complete_rule
from dial_det.examples import synthetic_complete_rule, synthetic_encoded_stream
from dial_det.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_all_four_contracts_run_without_fallback():
    stream = synthetic_encoded_stream()
    names = (
        "dial_out_at_most.json", "dial_out_exact.json",
        "dial_prop_exact.json", "dial_prop_cost.json",
    )
    for name in names:
        rule = load_complete_rule(ROOT / "configs" / "smoke" / name)
        result = run_pipeline(stream, rule)
        assert not result.fallback_triggered
        assert result.support_exposure == 1.0
        assert result.rule_sha256 == rule.sha256
        assert result.stream_sha256 == stream.sha256


def test_missing_exposure_triggers_registered_fallback():
    stream = synthetic_encoded_stream()
    stream.eligible_identity_pairs[:] = False
    stream.eligible_identity_pairs[0, 4] = True
    stream.eligible_identity_pairs[4, 0] = True
    rule = load_complete_rule(ROOT / "configs" / "smoke" / "dial_out_exact.json")
    result = run_pipeline(stream, rule)
    assert result.fallback_triggered
    assert result.fallback_reasons



def test_in_code_smoke_rules_cover_all_contracts():
    for name in (
        "dial-out-at-most",
        "dial-out-exact",
        "dial-prop-exact",
        "dial-prop-cost",
    ):
        result = run_pipeline(synthetic_encoded_stream(), synthetic_complete_rule(name))
        assert not result.fallback_triggered



def test_identity_threshold_changes_objective_support():
    from copy import deepcopy
    from dataclasses import asdict
    from dial_det.config.rule import CompleteRule

    stream = synthetic_encoded_stream()
    base = synthetic_complete_rule("dial-out-exact")
    payload = asdict(base)
    payload["rule_id"] = "threshold-regression"
    payload["graph"] = deepcopy(payload["graph"])
    payload["graph"]["identity_threshold"] = 0.99
    high = CompleteRule(**payload)
    result = run_pipeline(stream, high)
    matrix = result.identity_affinity
    assert matrix[0][1] == 0.0
    assert matrix[0][0] == 1.0


def test_active_relation_set_must_match_frozen_stream():
    from copy import deepcopy
    from dataclasses import asdict
    import pytest
    from dial_det.config.rule import CompleteRule

    stream = synthetic_encoded_stream()
    base = synthetic_complete_rule("dial-out-exact")
    payload = asdict(base)
    payload["rule_id"] = "relation-regression"
    payload["graph"] = deepcopy(payload["graph"])
    payload["graph"]["active_relations"] = ["geometry", "appearance"]
    rule = CompleteRule(**payload)
    with pytest.raises(ValueError, match="active_relations"):
        run_pipeline(stream, rule)
