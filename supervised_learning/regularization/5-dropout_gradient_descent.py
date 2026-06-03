#!/usr/bin/env python3
"""Module for gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights using gradient descent with dropout

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) with correct labels
        weights: dictionary of weights and biases of the network
        cache: dictionary of outputs and dropout masks of each layer
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network

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

        # Update weights and biases
        weights['W{}'.format(layer)] = (
            weights['W{}'.format(layer)] - alpha * dW
        )
        weights['b{}'.format(layer)] = (
            weights['b{}'.format(layer)] - alpha * db
        )

        if layer > 1:
            W = weights['W{}'.format(layer)]
            dA_prev = np.matmul(W.T, dZ)

            # Apply dropout mask if it exists (not for last layer)
            if 'D{}'.format(layer - 1) in cache:
                D = cache['D{}'.format(layer - 1)]
                dA_prev = (dA_prev * D) / keep_prob

            # tanh derivative: 1 - A^2
            dZ = dA_prev * (1 - cache['A{}'.format(layer - 1)] ** 2)
