#!/usr/bin/env python3
"""Module for L2 regularization gradient descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases using gradient descent with L2 regularization

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases of the network
        cache: dictionary of outputs of each layer of the network
        alpha: the learning rate
        lambtha: the L2 regularization parameter
        L: the number of layers of the network

    The network uses tanh on each layer except the last (softmax)
    Weights and biases are updated in place
    """
    m = Y.shape[1]

    # Backward propagation
    dZ = cache['A{}'.format(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A{}'.format(layer - 1)]

        # Compute weight and bias gradients
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Add L2 regularization term to weight gradient
        dW = dW + (lambtha / m) * weights['W{}'.format(layer)]

        # Update weights and biases
        weights['W{}'.format(layer)] = (
            weights['W{}'.format(layer)] - alpha * dW
        )
        weights['b{}'.format(layer)] = (
            weights['b{}'.format(layer)] - alpha * db
        )

        # Prepare dZ for previous layer (if not at input layer)
        if layer > 1:
            W = weights['W{}'.format(layer)]
            dZ = np.matmul(W.T, dZ)
            # tanh derivative: 1 - A^2
            dZ = dZ * (1 - cache['A{}'.format(layer - 1)] ** 2)