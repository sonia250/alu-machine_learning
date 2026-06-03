#!/usr/bin/env python3
"""Module for Gaussian Process"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize Gaussian Process

        Args:
            X_init: numpy.ndarray of shape (t, 1) with sampled inputs
            Y_init: numpy.ndarray of shape (t, 1) with function outputs
            l: length parameter for the kernel
            sigma_f: standard deviation of the output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculate covariance kernel matrix using RBF

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            covariance kernel matrix of shape (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
