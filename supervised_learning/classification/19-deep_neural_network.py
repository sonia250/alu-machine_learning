#!/usr/bin/env python3
"""Module for deep neural network with cost"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_nodes = nx
        for i in range(len(layers)):
            layer_num = i + 1
            nodes = layers[i]
            self.__weights[f'W{layer_num}'] = np.random.normal(
                0, np.sqrt(2 / prev_nodes), (nodes, prev_nodes)
            )
            self.__weights[f'b{layer_num}'] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache['A0'] = X
        A = X
        for layer in range(1, self.__L + 1):
            W = self.__weights[f'W{layer}']
            b = self.__weights[f'b{layer}']
            Z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{layer}'] = A
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
