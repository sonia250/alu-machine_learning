#!/usr/bin/env python3
"""Initialize Gaussian Mixture Model variables."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize priors, means, and covariance matrices."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k <= 0
    ):
        return None, None, None

    means, _ = kmeans(X, k)
    if means is None:
        return None, None, None

    priors = np.full(k, 1 / k)
    covariances = np.tile(np.eye(X.shape[1]), (k, 1, 1))
    return priors, means, covariances
