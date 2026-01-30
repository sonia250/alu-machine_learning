#!/usr/bin/env python3
"""Module for calculating the inverse of a matrix"""


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
        cofactor_val = ((-1) ** col) * matrix[0][col]
        cofactor_val = cofactor_val * determinant(minor_matrix)
        det += cofactor_val

    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    Args:
        matrix: A list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix
    """
    n = len(matrix)

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


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    Args:
        matrix: A list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix
    """
    n = len(matrix)

    # Get the minor matrix
    minor_matrix = minor(matrix)

    # Apply the sign pattern to get cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Apply sign: (-1)^(i+j)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix

    Args:
        matrix: A list of lists whose adjugate matrix should be calculated

    Returns:
        The adjugate matrix of matrix
    """
    n = len(matrix)

    # Get the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = []
    for j in range(n):
        adjugate_row = []
        for i in range(n):
            adjugate_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix

    Args:
        matrix: A list of lists whose inverse should be calculated

    Returns:
        The inverse of matrix, or None if matrix is singular

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

    # Calculate determinant
    det = determinant(matrix)

    # If determinant is 0, matrix is singular (no inverse)
    if det == 0:
        return None

    # Calculate adjugate matrix
    adj = adjugate(matrix)

    # Calculate inverse: (1/det) * adjugate
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
