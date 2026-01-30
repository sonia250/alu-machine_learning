#!/usr/bin/env python3
"""Module for concatenating matrices along a specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis

    Args:
        mat1: First matrix (numpy.ndarray)
        mat2: Second matrix (numpy.ndarray)
        axis: Axis along which to concatenate (default=0)

    Returns:
        A new numpy.ndarray with concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
