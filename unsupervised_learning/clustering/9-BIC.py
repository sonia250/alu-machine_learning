#!/usr/bin/env python3
"""Select a Gaussian Mixture Model using the Bayesian Information Criterion."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(
    X,
    kmin=1,
    kmax=None,
    iterations=1000,
    tol=1e-5,
    verbose=False,
):
    """Find the best number of GMM clusters using BIC."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(kmin, int)
        or kmin <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
        or not isinstance(tol, float)
        or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None

    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None, None, None

    likelihoods = []
    bics = []
    best_result = None
    for clusters in range(kmin, kmax + 1):
        pi, means, covariances, _, likelihood = expectation_maximization(
            X,
            clusters,
            iterations,
            tol,
            verbose,
        )
        if pi is None:
            return None, None, None, None
        parameters = clusters * (
            X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2
        ) + clusters - 1
        likelihoods.append(likelihood)
        bics.append(parameters * np.log(X.shape[0]) - 2 * likelihood)
        if best_result is None or bics[-1] < bics[np.argmin(bics[:-1])]:
            best_result = (pi, means, covariances)

    best_index = int(np.argmin(bics))
    best_k = kmin + best_index
    return (
        best_k,
        best_result,
        np.array(likelihoods),
        np.array(bics),
    )
