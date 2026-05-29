#!/usr/bin/env python3
"""Module for batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) to normalize
           m is the number of data points
           n is the number of features
        gamma: numpy.ndarray of shape (1, n) containing scales
        beta: numpy.ndarray of shape (1, n) containing offsets
        epsilon: small number to avoid division by zero

    Returns:
        the normalized Z matrix
    """
    # Calculate mean and variance across data points (axis 0)
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)

    # Normalize Z
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_output = gamma * Z_norm + beta

    return Z_output
