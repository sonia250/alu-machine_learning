#!/usr/bin/env python3
"""Module for calculating prediction accuracy"""
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction

    Args:
        y: placeholder for the labels of the input data
        y_pred: tensor containing the network's predictions

    Returns:
        tensor containing the decimal accuracy of the prediction
    """
    predicted_labels = tf.argmax(y_pred, axis=1)
    true_labels = tf.argmax(y, axis=1)
    correct_predictions = tf.equal(predicted_labels, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy