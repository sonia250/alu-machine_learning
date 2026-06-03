#!/usr/bin/env python3
"""Module for Gaussian Process with update"""
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

    def predict(self, X_s):
        """
        Predict mean and standard deviation of points

        Args:
            X_s: numpy.ndarray of shape (s, 1) with sample points

        Returns:
            mu: numpy.ndarray of shape (s,) with mean for each point
            sigma: numpy.ndarray of shape (s,) with variance for each point
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.dot(K_s.T, K_inv).dot(self.Y).flatten()
        sigma = np.diag(K_ss) - np.diag(np.dot(K_s.T, K_inv).dot(K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Update Gaussian Process with new sample

        Args:
            X_new: numpy.ndarray of shape (1,) with new input
            Y_new: numpy.ndarray of shape (1,) with new output
        """
        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)

        self.X = np.vstack([self.X, X_new])
        self.Y = np.vstack([self.Y, Y_new])
        self.K = self.kernel(self.X, self.X)
