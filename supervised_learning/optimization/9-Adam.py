#!/usr/bin/env python3
"""Module for Adam optimization"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction

    Returns:
        the updated variable, the new first moment, and the new second moment, respectively
    """
    # Update first moment (moving average of gradient)
    v = beta1 * v + (1 - beta1) * grad

    # Update second moment (moving average of squared gradient)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction for first moment
    v_corrected = v / (1 - (beta1 ** t))

    # Bias correction for second moment
    s_corrected = s / (1 - (beta2 ** t))

    # Update variable
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
