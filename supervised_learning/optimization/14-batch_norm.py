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

    Z = tf.layers.dense(prev, n, activation=None,
                        kernel_initializer=initializer)

    mean, variance = tf.nn.moments(Z, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    Z_norm = (Z - mean) / tf.sqrt(variance + 1e-8)
    output = activation(gamma * Z_norm + beta)

    return output
