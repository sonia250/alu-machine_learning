#!/usr/bin/env python3
"""Module for momentum gradient descent"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:
        the updated variable and the new moment, respectively
    """
    # Update first moment (velocity)
    v = beta1 * v + (1 - beta1) * grad

    # Update variable
    var = var - alpha * v

    return var, v
