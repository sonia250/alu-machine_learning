#!/usr/bin/env python3
"""Module for creating TensorFlow placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for a neural network

    Args:
        nx: number of feature columns in our data
        classes: number of classes in our classifier

    Returns:
        x: placeholder for input data (shape: [None, nx])
        y: placeholder for one-hot labels (shape: [None, classes])
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
