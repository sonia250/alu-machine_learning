#!/usr/bin/env python3
"""Module for creating momentum training operation in TensorFlow"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation using gradient descent with momentum

    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: momentum weight

    Returns:
        the momentum optimization operation
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
