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

    # Dense layer
    x = tf.layers.dense(prev, n, activation=None,
                        kernel_initializer=initializer)

    # Manual batch norm with gamma initialized to 1, beta to 0
    mean, variance = tf.nn.moments(x, [0], keep_dims=False)
    gamma = tf.Variable(tf.ones([1, n]), trainable=True)
    beta = tf.Variable(tf.zeros([1, n]), trainable=True)

    bn = (x - mean) / tf.sqrt(variance + 1e-8)
    output = activation(tf.nn.bias_add(gamma * bn, tf.squeeze(beta)))

    return output
