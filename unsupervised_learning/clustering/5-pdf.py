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

    determinant = np.linalg.det(S)
    if determinant <= 0:
        return None

    try:
        inverse = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    difference = X - m
    exponent = -0.5 * np.einsum(
        'ij,jk,ik->i',
        difference,
        inverse,
        difference,
    )
    coefficient = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * determinant)
    return np.maximum(coefficient * np.exp(exponent), 1e-300)
