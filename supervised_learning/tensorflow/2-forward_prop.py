#!/usr/bin/env python3
"""Module for creating forward propagation graph"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        prediction of the network in tensor form
    """
    # Start with input
    prev = x

    # Create each layer sequentially
    for i in range(len(layer_sizes)):
        layer = create_layer(prev, layer_sizes[i], activations[i])
        prev = layer

    return prev
