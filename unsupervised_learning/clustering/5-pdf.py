#!/usr/bin/env python3
"""Calculate Gaussian probability density values."""

import numpy as np


def pdf(X, m, S):
    """Calculate the probability density for a Gaussian distribution."""
    if (
        not isinstance(X, np.ndarray)
        or not isinstance(m, np.ndarray)
        or not isinstance(S, np.ndarray)
        or X.ndim != 2
        or m.ndim != 1
        or S.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or m.shape[0] != X.shape[1]
        or S.shape != (X.shape[1], X.shape[1])
    ):
        return None

    sign, log_determinant = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(log_determinant):
        return None

    try:
        difference = X - m
        solved = np.linalg.solve(S, difference.T).T
    except np.linalg.LinAlgError:
        return None

    mahalanobis = np.sum(difference * solved, axis=1)
    log_density = -0.5 * (
        X.shape[1] * np.log(2 * np.pi)
        + log_determinant
        + mahalanobis
    )
    return np.maximum(np.exp(log_density), 1e-300)
