#!/usr/bin/env python3
"""Perform the expectation step for a Gaussian Mixture Model."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculate posterior probabilities and log likelihood."""
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(pi, np.ndarray)
        or not isinstance(m, np.ndarray)
        or not isinstance(S, np.ndarray)
        or X.ndim != 2
        or pi.ndim != 1
        or m.ndim != 2
        or S.ndim != 3
        or X.shape[0] == 0
        or X.shape[1] == 0
        or pi.shape[0] == 0
        or pi.shape[0] != m.shape[0]
        or S.shape[0] != pi.shape[0]
        or m.shape[1] != X.shape[1]
        or S.shape[1:] != (X.shape[1], X.shape[1])
    ):
        return None, None

    probabilities = np.zeros((pi.shape[0], X.shape[0]))
    for cluster in range(pi.shape[0]):
        probabilities[cluster] = pi[cluster] * pdf(
            X,
            m[cluster],
            S[cluster],
        )

    totals = np.sum(probabilities, axis=0)
    g = probabilities / totals
    return g, np.sum(np.log(totals))
