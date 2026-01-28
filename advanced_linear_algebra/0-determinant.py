#!/usr/bin/env python3
"""Module to calculate the determinant of a matrix"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    
    Args:
        matrix: list of lists whose determinant should be calculated
    
    Returns:
        The determinant of matrix
    
    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    # Check if it's a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Special case: 0x0 matrix
    if matrix == [[]]:
        return 1
    
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
    
    # Recursive case: use cofactor expansion
    det = 0
    for j in range(n):
        # Create minor matrix (remove first row and jth column)
        minor = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            minor.append(row)
        
        # Calculate cofactor and add to determinant
        cofactor = ((-1) ** j) * matrix[0][j] * determinant(minor)
        det += cofactor
    
    return det