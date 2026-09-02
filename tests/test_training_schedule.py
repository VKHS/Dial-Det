import numpy as np

from dial_det.models.losses import soft_budget_temperature


def test_temperature_anneals_over_final_forty_percent():
    assert soft_budget_temperature(0.0)==1.0
    assert soft_budget_temperature(.6)==1.0
    assert np.isclose(soft_budget_temperature(1.0), .05)
