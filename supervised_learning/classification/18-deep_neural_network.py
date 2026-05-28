#!/usr/bin/env python3
"""Module for deep neural network with forward propagation"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer or layers is not a list
            ValueError: if nx is less than 1 or layers is empty/contains non-positive integers
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        # Set private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights using He et al. method
        prev_nodes = nx
        for i in range(len(layers)):
            layer_num = i + 1
            nodes = layers[i]

            # He initialization: variance = 2 / prev_nodes
            self.__weights[f'W{layer_num}'] = np.random.normal(
                0, np.sqrt(2 / prev_nodes), (nodes, prev_nodes)
            )
            self.__weights[f'b{layer_num}'] = np.zeros((nodes, 1))

            prev_nodes = nodes

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            the output of the neural network and the cache, respectively
        """
        self.__cache['A0'] = X

        A = X
        for layer in range(1, self.__L + 1):
            W = self.__weights[f'W{layer}']
            b = self.__weights[f'b{layer}']

            # Z = W·A_prev + b
            Z = np.matmul(W, A) + b

            # A = sigmoid(Z)
            A = 1 / (1 + np.exp(-Z))

            # Store in cache
            self.__cache[f'A{layer}'] = A

        return A, self.__cache
