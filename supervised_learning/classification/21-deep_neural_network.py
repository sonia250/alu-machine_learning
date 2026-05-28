#!/usr/bin/env python3
"""Module for deep neural network with gradient descent"""
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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output

        Returns:
            the cost
        """
        m = Y.shape[1]
        # Use 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) with correct labels

        Returns:
            the neuron's prediction and the cost of the network, respectively
        """
        A, _ = self.forward_prop(X)

        # Convert probabilities to binary predictions (0 or 1)
        prediction = np.where(A >= 0.5, 1, 0)

        # Calculate cost
        cost = self.cost(Y, A)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            cache: dictionary containing all intermediary values of the network
            alpha: learning rate

        Updates:
            __weights using gradient descent
        """
        m = Y.shape[1]

        # Start with the output layer
        dZ = cache[f'A{self.__L}'] - Y

        # Backpropagate through all layers
        for layer in range(self.__L, 0, -1):
            A_prev = cache[f'A{layer - 1}']

            # Calculate gradients
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            self.__weights[f'W{layer}'] -= alpha * dW
            self.__weights[f'b{layer}'] -= alpha * db

            # Calculate dZ for previous layer if not at input
            if layer > 1:
                W = self.__weights[f'W{layer}']
                A_prev_sigmoid = cache[f'A{layer - 1}']
                dZ = np.matmul(W.T, dZ) * A_prev_sigmoid * (1 - A_prev_sigmoid)
