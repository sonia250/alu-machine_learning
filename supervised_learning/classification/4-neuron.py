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

        Returns:
            The activated output using sigmoid function
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels

        Returns:
            The neuron's prediction and the cost of the network
            Prediction shape is (1, m) containing predicted labels (0 or 1)
        """
        # Forward propagation to get predictions
        A = self.forward_prop(X)

        # Convert probabilities to binary predictions (0 or 1)
        # If A >= 0.5, predict 1, otherwise predict 0
        prediction = np.where(A >= 0.5, 1, 0)

        # Calculate cost
        cost = self.cost(Y, A)

        return prediction, cost
