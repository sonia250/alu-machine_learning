#!/usr/bin/env python3
"""Fit a Gaussian Mixture Model with scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Fit a Gaussian Mixture Model and return its parameters."""
    if (
        not hasattr(X, 'shape')
        or len(X.shape) != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k <= 0
    ):
        return None, None, None, None, None

    try:
        model = sklearn.mixture.GaussianMixture(n_components=k)
        model.fit(X)
    except (TypeError, ValueError):
        return None, None, None, None, None

    return (
        model.weights_,
        model.means_,
        model.covariances_,
        model.predict(X),
        model.bic(X),
    )
