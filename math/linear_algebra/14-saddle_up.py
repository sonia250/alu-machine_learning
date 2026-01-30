#!/usr/bin/env python3
"""Module for performing matrix multiplication using NumPy"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray

    Returns:
        The result of matrix multiplication
    """
    return np.matmul(mat1, mat2)
