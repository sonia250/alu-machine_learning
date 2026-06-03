#!/usr/bin/env python3
"""Module for creating a layer with dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function to use on the layer
        keep_prob: probability that a node will be kept

    Returns:
        the output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    layer = tf.layers.Dense(
        n, activation=activation, kernel_initializer=initializer
    )

    output = layer(prev)
    output = tf.layers.Dropout(rate=1 - keep_prob)(output)

    return output
