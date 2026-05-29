#!/usr/bin/env python3
"""Module for learning rate decay in TensorFlow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine rate at which alpha decays
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes before alpha is decayed

    Returns:
        the learning rate decay operation
    """
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )
    return learning_rate
