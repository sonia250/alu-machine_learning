#!/usr/bin/env python3
"""Module for performing matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)

    Returns:
        A new matrix resulting from multiplication, or None if incompatible
    """
    # Check if matrices can be multiplied
    # mat1 must have same number of columns as mat2 has rows
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    # Result will have rows of mat1 and columns of mat2
    result = []

    # Iterate through rows of mat1
    for i in range(len(mat1)):
        row = []
        # Iterate through columns of mat2
        for j in range(len(mat2[0])):
            # Calculate dot product
            element = 0
            for k in range(len(mat2)):
                element += mat1[i][k] * mat2[k][j]
            row.append(element)
        result.append(row)

    return result
