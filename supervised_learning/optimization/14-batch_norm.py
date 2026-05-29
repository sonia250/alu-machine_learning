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
    # He initialization for weights
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )

    # Create dense layer without activation
    dense = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=initializer,
        name='dense'
    )

    # Apply dense layer to get Z
    Z = dense(prev)

    # Create trainable parameters for batch normalization
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    # Get mean and variance of Z
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Normalize Z
    Z_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon=1e-8
    )

    # Apply activation function
    output = activation(Z_norm)

    return output
