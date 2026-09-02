import numpy as np
import pytest

from dial_det.method.adapters import (
    identity_matrix_from_directional_edges,
    reverse_edge_positions,
)


def test_identity_matrix_is_symmetric_and_logit_averaged():
    edges = np.array([[0, 1], [1, 0]])
    logits = np.array([4.0, -2.0])
    matrix = identity_matrix_from_directional_edges(2, edges, logits)
    expected = 1.0 / (1.0 + np.exp(-1.0))
    assert np.isclose(matrix[0, 1], expected)
    assert np.isclose(matrix[1, 0], expected)
    assert np.allclose(np.diag(matrix), 1.0)


def test_missing_reverse_edge_is_rejected():
    with pytest.raises(ValueError):
        reverse_edge_positions(np.array([[0], [1]]))
