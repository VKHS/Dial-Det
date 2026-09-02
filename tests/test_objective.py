import numpy as np

from dial_det.method.objective import (
    DIALObjectiveEvidence,
    evidence_perturbation_bound,
    regret_upper_bound,
    verify_submodularity_by_enumeration,
)


def test_objective_and_submodularity():
    evidence = DIALObjectiveEvidence(
        value=np.array([1.0, 0.9, 0.7]),
        coverage=np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        redundancy=np.array([[0, 0.8, 0], [0.8, 0, 0.1], [0, 0.1, 0]], dtype=float),
        eta=0.4,
        lambda_redundancy=0.5,
    )
    assert np.isclose(evidence.evaluate([0, 2]), 1.0 + 0.7 + 0.4 * 2.0)
    assert verify_submodularity_by_enumeration(evidence)


def test_perturbation_regret_bound():
    delta = evidence_perturbation_bound(
        budget_k=3, number_of_representatives=4, epsilon_value=0.01,
        epsilon_coverage=0.02, epsilon_redundancy=0.03, eta=0.2,
        lambda_redundancy=0.5,
    )
    assert np.isclose(regret_upper_bound(optimization_gap=0.1, perturbation_bound=delta), 0.1 + 2 * delta)
