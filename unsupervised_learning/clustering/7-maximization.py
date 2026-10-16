#!/usr/bin/env python3
"""Perform the maximization step for a Gaussian Mixture Model."""

import numpy as np


def maximization(X, g):
    """Calculate updated priors, means, and covariance matrices."""
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(g, np.ndarray)
        or X.ndim != 2
        or g.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or g.shape[0] == 0
        or g.shape[1] != X.shape[0]
        or not np.all(np.isfinite(g))
        or not np.allclose(np.sum(g, axis=0), 1)
    ):
        return None, None, None

    weights = np.sum(g, axis=1)
    if np.any(weights == 0):
        return None, None, None

    pi = weights / X.shape[0]
    m = np.matmul(g, X) / weights[:, np.newaxis]
    differences = X[np.newaxis, :, :] - m[:, np.newaxis, :]
    S = np.einsum(
        'kn,kni,knj->kij',
        g,
        differences,
        differences,
    ) / weights[:, np.newaxis, np.newaxis]
    return pi, m, S
