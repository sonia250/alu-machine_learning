#!/usr/bin/env python3
"""Perform agglomerative clustering with Ward linkage."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Cluster a data set and display its dendrogram."""
    if (
        not hasattr(X, 'shape')
        or len(X.shape) != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not isinstance(dist, (int, float))
        or dist <= 0
    ):
        return None

    try:
        linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
        clss = scipy.cluster.hierarchy.fcluster(
            linkage,
            dist,
            criterion='distance',
        )
        scipy.cluster.hierarchy.dendrogram(
            linkage,
            color_threshold=dist,
        )
        plt.show()
    except (TypeError, ValueError):
        return None

    return clss
