#!/usr/bin/env python3
"""Module for calculating the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Args:
        matrix: A numpy.ndarray of shape (n, n)

    Returns:
        String indicating definiteness, or None if invalid

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is empty or not 2D
    if matrix.size == 0 or len(matrix.shape) != 2:
        return None

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check definiteness based on eigenvalues
    # Use a small tolerance for floating point comparison
    tol = 1e-10

    if np.all(eigenvalues > tol):
        return "Positive definite"
    elif np.all(eigenvalues >= -tol):
        return "Positive semi-definite"
    elif np.all(eigenvalues < -tol):
        return "Negative definite"
    elif np.all(eigenvalues <= tol):
        return "Negative semi-definite"
    else:
        return "Indefinite"
