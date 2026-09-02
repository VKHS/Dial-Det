import numpy as np

from dial_det.method.values import (
    dial_out_value,
    dial_prop_value,
    softplus,
    symmetrize_identity_logits,
)


def test_dial_out_value_matches_equation():
    raw = np.array([0.0, 1.0])
    uncertainty = np.array([0.0, 0.5])
    result = dial_out_value(
        raw, uncertainty, budget=1, candidate_count=2, uncertainty_discount=2.0
    )
    assert np.allclose(result, softplus(np.array([0.0, 0.0])))


def test_dial_prop_cost_is_not_folded_into_value():
    result = dial_prop_value(
        [0.8], [0.5], [0.75], roi_success_probability=[0.9], beta_roi=0.5
    )
    assert np.allclose(result, [0.8 * 0.5 * 0.75 * 0.95])


def test_identity_averages_logits_before_sigmoid():
    result = symmetrize_identity_logits(np.array([4.0]), np.array([-2.0]))
    assert np.allclose(result, 1.0 / (1.0 + np.exp(-1.0)))
