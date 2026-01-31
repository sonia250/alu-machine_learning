#!/usr/bin/env python3
"""Module for transposing a 2D matrix"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix

    Args:
        matrix: A 2D matrix (list of lists)

    Returns:
        A new transposed matrix
    """
    # Get dimensions
    rows = len(matrix)
    cols = len(matrix[0])

    # Create transposed matrix
    result = []
    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        result.append(row)

    return result
