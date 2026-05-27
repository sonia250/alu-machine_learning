#!/usr/bin/env python3
"""Module for shuffling data"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Args:
        X: first numpy.ndarray of shape (m, nx) to shuffle
           m is the number of data points
           nx is the number of features in X
        Y: second numpy.ndarray of shape (m, ny) to shuffle
           m is the same number of data points as in X
           ny is the number of features in Y

    Returns:
        the shuffled X and Y matrices
    """
    # Get the number of data points
    m = X.shape[0]

    # Create a random permutation of indices
    permutation = np.random.permutation(m)

    # Shuffle both X and Y using the same permutation
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y
