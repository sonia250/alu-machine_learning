#!/usr/bin/env python3
"""Initialize K-means centroids."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or not isinstance(k, int)
        or k <= 0
    ):
        return None

    return np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, X.shape[1]),
    )
