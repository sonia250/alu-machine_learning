#!/usr/bin/env python3
"""Module for adding two 2D matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)

    Returns:
        A new matrix with element-wise sums, or None if shapes don't match
    """
    # Check if matrices have same number of rows
    if len(mat1) != len(mat2):
        return None

    # Check if matrices have same number of columns
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Perform element-wise addition
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
