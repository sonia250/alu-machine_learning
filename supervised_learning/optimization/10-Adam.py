#!/usr/bin/env python3
"""Module for creating Adam training operation in TensorFlow"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation using Adam optimization algorithm

    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero

    Returns:
        the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1=beta1, beta2=beta2,
                                       epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
