#!/usr/bin/env python3
"""Perform PCA with a requested output dimensionality."""

import numpy as np


def pca(X, ndim):
    """Return the data projected onto its leading principal components."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(ndim, int)
        or ndim <= 0
        or ndim > X.shape[1]
    ):
        return None

    _, _, vh = np.linalg.svd(X)
    return np.matmul(X, vh[:ndim].T)
