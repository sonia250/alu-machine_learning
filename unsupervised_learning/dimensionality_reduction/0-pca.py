#!/usr/bin/env python3
"""Perform PCA while preserving a requested variance fraction."""

import numpy as np


def pca(X, var=0.95):
    """Return a PCA weight matrix preserving the requested variance."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(var, float)
        or var <= 0
        or var > 1
    ):
        return None

    _, singular_values, vh = np.linalg.svd(X)
    total = np.sum(singular_values)
    if total == 0:
        return vh[:1].T

    cumulative = np.cumsum(singular_values) / total
    dimensions = np.searchsorted(cumulative, var) + 1
    return vh[:dimensions].T
