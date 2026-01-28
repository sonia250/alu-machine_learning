#!/usr/bin/env python3
"""Module for calculating the minor matrix of a matrix"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix: A list of lists representing a matrix

    Returns:
        The determinant of the matrix
    """
    # Special case: [[]] represents a 0x0 matrix with determinant 1
    if matrix == [[]]:
        return 1

    n = len(matrix)

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
        minor_matrix = []
        for row in range(1, n):
            minor_row = []
            for c in range(n):
                if c != col:
                    minor_row.append(matrix[row][c])
            minor_matrix.append(minor_row)

        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** col) * matrix[0][col] * determinant(minor_matrix)
        det += cofactor

    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Args:
        matrix: A list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty list
    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if all elements are lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty (contains only empty lists)
    if any(len(row) == 0 for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create submatrix by removing row i and column j
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    submatrix_row = []
                    for col_idx in range(n):
                        if col_idx != j:
                            submatrix_row.append(matrix[row_idx][col_idx])
                    submatrix.append(submatrix_row)

            # Calculate determinant of submatrix
            minor_value = determinant(submatrix)
            minor_row.append(minor_value)

        minor_matrix.append(minor_row)

    return minor_matrix
