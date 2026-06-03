#!/usr/bin/env python3
"""Module for Bayesian Optimization"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization

        Args:
            f: black-box function to optimize
            X_init: numpy.ndarray of shape (t, 1) with sampled inputs
            Y_init: numpy.ndarray of shape (t, 1) with function outputs
            bounds: tuple of (min, max) for search space
            ac_samples: number of samples for acquisition
            l: length parameter for kernel
            sigma_f: standard deviation of output
            xsi: exploration-exploitation factor
            minimize: bool for minimization vs maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize

        # Create evenly spaced acquisition samples
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
