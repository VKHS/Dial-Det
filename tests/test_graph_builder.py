import numpy as np

from dial_det.adapters.graph_builder import build_typed_candidate_graph
from dial_det.method.adapters import reverse_edge_positions


def test_graph_is_bidirectional_without_duplicate_endpoint_edges():
    boxes=np.array([[0,0,10,10],[1,1,11,11],[30,0,40,10]],float)
    features=np.array([[1,0],[.9,.1],[0,1]],float)
    graph=build_typed_candidate_graph(
        boxes,features,iou_threshold=.2,spatial_neighbors=1,feature_neighbors=1
    )
    reverse=reverse_edge_positions(graph.edge_index)
    assert len(reverse)==graph.edge_index.shape[1]
    assert graph.edge_geometry.shape[1]==8
    assert graph.graph_degree.shape==(3,)
