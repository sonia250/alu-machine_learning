#!/usr/bin/env python3
"""Module for concatenating two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)
        axis: Axis along which to concatenate (0 for rows, 1 for columns)

    Returns:
        A new concatenated matrix, or None if matrices cannot be concatenated
    """
    if axis == 0:
        # Concatenate along rows (stack vertically)
        # Matrices must have same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        # Create deep copy and concatenate
        result = [row[:] for row in mat1]
        for row in mat2:
            result.append(row[:])
        return result
    elif axis == 1:
        # Concatenate along columns (stack horizontally)
        # Matrices must have same number of rows
        if len(mat1) != len(mat2):
            return None
        # Create deep copy and concatenate
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i][:] + mat2[i][:])
        return result
    else:
        return None
