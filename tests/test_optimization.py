import numpy as np

from dial_det.method.objective import DIALObjectiveEvidence
from dial_det.method.optimization import (
    optimize_cardinality_with_certificate,
    optimize_measured_cost_with_certificate,
)


def evidence():
    return DIALObjectiveEvidence(
        value=np.array([1.2, 1.0, 0.9, 0.8]),
        coverage=np.array([[1, .8, 0, 0], [.8, 1, 0, 0], [0, 0, 1, .2], [0, 0, .2, 1]], dtype=float),
        redundancy=np.array([[0,.7,0,0],[.7,0,0,0],[0,0,0,.4],[0,0,.4,0]], dtype=float),
        eta=.2, lambda_redundancy=1.0,
    )


def test_exact_cardinality_certificate():
    result = optimize_cardinality_with_certificate(
        evidence(), budget=2, exact_budget=True, contract="DIAL-Out",
        support_exposure=1.0, minimum_support_exposure=.95,
        exact_component_threshold=40, enumeration_threshold=18,
    )
    assert not result.fallback_triggered
    assert result.lower_bound == result.upper_bound
    assert len(result.returned_selection) == 2


def test_at_most_can_choose_less_than_budget():
    ev = DIALObjectiveEvidence(
        value=np.array([.1,.1]), coverage=np.zeros((2,0)),
        redundancy=np.array([[0,10],[10,0]],float), eta=0, lambda_redundancy=1,
    )
    result = optimize_cardinality_with_certificate(
        ev, budget=2, exact_budget=False, support_exposure=1,
        minimum_support_exposure=0, gap_threshold=.1,
        exact_component_threshold=40, enumeration_threshold=18,
    )
    assert len(result.returned_selection) == 1


def test_measured_cost_contract():
    result = optimize_measured_cost_with_certificate(
        evidence(), costs=np.array([1.0,1.1,1.0,1.3]), cost_budget=2.1,
        exact_candidate_threshold=60,
    )
    assert result.solver_status == "optimal"
    assert result.total_cost <= 2.1 + 1e-9
    assert result.lower_bound == result.upper_bound
