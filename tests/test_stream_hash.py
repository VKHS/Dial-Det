from copy import deepcopy

import numpy as np

from dial_det.data.stream import EncodedCandidateStream
from dial_det.examples import synthetic_encoded_stream


def clone(stream):
    return EncodedCandidateStream(
        **{
            name: (getattr(stream, name).copy() if hasattr(getattr(stream, name), "copy") else deepcopy(getattr(stream, name)))
            for name in EncodedCandidateStream.__dataclass_fields__
        }
    )


def test_hash_covers_features_graph_values_and_metadata():
    original = synthetic_encoded_stream()
    baseline = original.sha256
    variants = []
    a = clone(original); a.node_features[0, 0] += 1e-4; variants.append(a)
    a = clone(original); a.edge_geometry[0, 1] += 1e-4; variants.append(a)
    a = clone(original); a.directional_identity_logits[0] += 1e-4; variants.append(a)
    a = clone(original); a.raw_benefit_out[0] += 1e-4; variants.append(a)
    a = clone(original); a.measured_cost[0] += 1e-4; variants.append(a)
    a = clone(original); a.metadata["feature_taps"] = ["changed"]; variants.append(a)
    assert all(item.sha256 != baseline for item in variants)
