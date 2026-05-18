#!/usr/bin/env python3
"""Module for a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """A single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron

        Args:
            nx: number of input features to the neuron

        Raises:
            TypeError: if nx is not an integer
            ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features
               m is the number of examples

        Returns:
            The activated output (prediction) using sigmoid function
        """
        # Calculate weighted sum: Z = W·X + b
        Z = np.matmul(self.__W, X) + self.__b

        # Apply sigmoid activation function: A = 1 / (1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
