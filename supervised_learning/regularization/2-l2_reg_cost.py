#!/usr/bin/env python3
"""Module for L2 regularization cost in TensorFlow"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: a tensor containing the cost without L2 regularization

    Returns:
        a tensor containing the cost with L2 regularization
    """
    reg_losses = tf.losses.get_regularization_losses()
    total_cost = cost + tf.add_n(reg_losses)

    return total_cost
