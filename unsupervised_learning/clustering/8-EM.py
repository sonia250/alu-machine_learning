#!/usr/bin/env python3
"""Perform expectation maximization for a Gaussian Mixture Model."""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(
    X,
    k,
    iterations=1000,
    tol=1e-5,
    verbose=False,
):
    """Fit a Gaussian Mixture Model with the EM algorithm."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
        or not isinstance(tol, float)
        or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    previous = None
    g = None
    likelihood = None
    for iteration in range(iterations):
        g, likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose and (iteration % 10 == 0 or iteration == iterations - 1):
            print(
                'Log Likelihood after {} iterations: {}'.format(
                    iteration,
                    round(likelihood, 5),
                )
            )
        if previous is not None and abs(likelihood - previous) <= tol:
            break
        previous = likelihood
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

    return pi, m, S, g, likelihood
