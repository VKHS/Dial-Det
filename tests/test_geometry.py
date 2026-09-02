import numpy as np

from dial_det.data.geometry import directed_geometry_features, make_bidirectional_edges


def test_reverse_geometry_is_recomputed():
    boxes = np.array([[0, 0, 10, 20], [4, 3, 24, 13]], dtype=float)
    edge_index, _ = make_bidirectional_edges([[0, 1]])
    geometry = directed_geometry_features(boxes, edge_index)
    assert np.isclose(geometry[0, 0], geometry[1, 0])
    assert np.isclose(geometry[0, 1], -geometry[1, 1])
    assert np.isclose(geometry[0, 2], -geometry[1, 2])
    assert np.isclose(geometry[0, 3], -geometry[1, 3])
    assert np.isclose(geometry[0, 4], -geometry[1, 4])
    assert not np.array_equal(geometry[0], geometry[1])
