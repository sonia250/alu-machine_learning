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
    ):
        return None

    centered = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(centered)
    return np.matmul(centered, vh[:ndim].T)
