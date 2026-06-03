#!/usr/bin/env python3
"""Module for creating a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer with L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function to use on the layer
        lambtha: the L2 regularization parameter

    Returns:
        the output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        n, activation=activation, kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    output = layer(prev)

    return output
