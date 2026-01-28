#!/usr/bin/env python3
"""Module for calculating the determinant of a matrix"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix: A list of lists representing a matrix

    Returns:
        The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty list (not a list of lists)
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Special case: [[]] represents a 0x0 matrix with determinant 1
    if matrix == [[]]:
        return 1

    # Check if all elements are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: use cofactor expansion along the first row
    det = 0
    for col in range(n):
        # Create the minor matrix (exclude first row and current column)
        minor = []
        for row in range(1, n):
            minor_row = []
            for c in range(n):
                if c != col:
                    minor_row.append(matrix[row][c])
            minor.append(minor_row)

        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor)
        det += cofactor

    return det
