#!/usr/bin/env python3
"""Perform K-means clustering with scikit-learn."""

import sklearn.cluster


def kmeans(X, k):
    """Cluster a data set with scikit-learn KMeans."""
    if (
        not hasattr(X, 'shape')
        or len(X.shape) != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(k, int)
        or k <= 0
    ):
        return None, None

    try:
        model = sklearn.cluster.KMeans(n_clusters=k)
        model.fit(X)
    except (TypeError, ValueError):
        return None, None

    return model.cluster_centers_, model.labels_
