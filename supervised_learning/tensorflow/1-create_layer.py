#!/usr/bin/env python3
"""Module for creating a TensorFlow layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use

    Returns:
        tensor output of the layer
    """
    # He et al. initialization (variance scaling)
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    # Create dense layer with the specified activation
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )

    # Apply the layer to the previous tensor
    return layer(prev)
