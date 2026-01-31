#!/usr/bin/env python3
"""Module for calculating the shape of a matrix"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix

    Args:
        matrix: A matrix (nested list)

    Returns:
        A list of integers representing the shape
    """
    shape = []
    current = matrix

    while isinstance(current, list):
        shape.append(len(current))
        current = current[0]

    return shape
