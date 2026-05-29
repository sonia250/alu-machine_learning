#!/usr/bin/env python3
"""Module for batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer
        activation: activation function to use on output

    Returns:
        tensor of the activated output for the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    x = tf.layers.dense(prev, n, activation=None,
                        kernel_initializer=initializer)
    bn = tf.layers.batch_normalization(x, training=True, epsilon=1e-8)
    output = activation(bn)

    return output
