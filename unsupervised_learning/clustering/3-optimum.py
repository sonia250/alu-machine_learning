#!/usr/bin/env python3
"""Find the optimum number of clusters."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Evaluate K-means variance across a range of cluster sizes."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(kmin, int)
        or kmin <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        return None, None

    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None

    results = []
    variances = []
    for clusters in range(kmin, kmax + 1):
        centroids, classifications = kmeans(X, clusters, iterations)
        results.append((centroids, classifications))
        variances.append(variance(X, centroids))

    d_vars = [variances[0] - current for current in variances]
    return results, d_vars
