#!/usr/bin/env python3
"""Module for normalizing data"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
           d is the number of data points
           nx is the number of features
        m: numpy.ndarray of shape (nx,) containing the mean of all features
        s: numpy.ndarray of shape (nx,) containing the standard deviation

    Returns:
        The normalized X matrix
    """
    normalized_X = (X - m) / s
    return normalized_X
