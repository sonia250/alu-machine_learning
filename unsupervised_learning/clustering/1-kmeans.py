#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        return None, None

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    C = np.random.uniform(minimum, maximum, (k, X.shape[1]))

    for _ in range(iterations):
        distances = np.linalg.norm(
            X[:, np.newaxis] - C[np.newaxis, :],
            axis=2,
        )
        clss = np.argmin(distances, axis=1)
        previous = C.copy()

        for cluster in range(k):
            points = X[clss == cluster]
            if points.shape[0] == 0:
                C[cluster] = np.random.uniform(
                    minimum,
                    maximum,
                    X.shape[1],
                )
            else:
                C[cluster] = np.mean(points, axis=0)

        if np.array_equal(C, previous):
            break

    return C, clss
