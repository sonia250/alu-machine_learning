#!/usr/bin/env python3
"""Calculate intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance."""
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(C, np.ndarray)
        or X.ndim != 2
        or C.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or C.shape[0] == 0
        or C.shape[1] != X.shape[1]
    ):
        return None

    distances = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return np.sum(np.min(distances, axis=1))
