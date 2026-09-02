from __future__ import annotations

import numpy as np


def eligible_pairs_from_instance_assignments(assignments: np.ndarray) -> np.ndarray:
    """Build an independent duplicate-pair denominator from labeled assignments.

    `assignments[i]` is a physical-instance ID or -1 for an unmatched candidate.
    The result should be computed from audit labels, not from retained affinity
    edges.
    """
    values = np.asarray(assignments, dtype=np.int64).reshape(-1)
    n = len(values)
    matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        if values[i] < 0:
            continue
        for j in range(i + 1, n):
            if values[j] == values[i]:
                matrix[i, j] = matrix[j, i] = True
    return matrix
