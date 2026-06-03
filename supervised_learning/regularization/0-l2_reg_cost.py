#!/usr/bin/env python3
"""Module for L2 regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: dictionary of weights and biases of the network
        L: the number of layers in the network
        m: the number of data points used

    Returns:
        the cost of the network accounting for L2 regularization
    """
    l2_cost = 0
    for i in range(1, L + 1):
        W = weights['W{}'.format(i)]
        l2_cost += np.sum(W ** 2)

    l2_cost = l2_cost * (lambtha / (2 * m))
    total_cost = cost + l2_cost

    return total_cost