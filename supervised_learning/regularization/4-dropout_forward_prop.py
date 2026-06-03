#!/usr/bin/env python3
"""Module for forward propagation with dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
        weights: dictionary of the weights and biases of the network
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept

    Returns:
        a dictionary containing outputs of each layer and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W{}'.format(layer)]
        b = weights['b{}'.format(layer)]
        A_prev = cache['A{}'.format(layer - 1)]

        # Compute Z and A
        Z = np.matmul(W, A_prev) + b

        # Activation function
        if layer == L:
            # Last layer: softmax
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Hidden layers: tanh
            A = np.tanh(Z)

        cache['A{}'.format(layer)] = A

        # Dropout (only for hidden layers, not last layer)
        if layer < L:
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D{}'.format(layer)] = D
            cache['A{}'.format(layer)] = A

    return cache
