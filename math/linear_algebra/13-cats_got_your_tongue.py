#!/usr/bin/env python3
"""Module for performing element-wise operations on matrices"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division

    Args:
        mat1: First matrix (numpy.ndarray)
        mat2: Second matrix (numpy.ndarray or scalar)

    Returns:
        A tuple containing (sum, difference, product, quotient)
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return (add, sub, mul, div)
