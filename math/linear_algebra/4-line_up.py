#!/usr/bin/env python3
"""Module for adding two arrays element-wise"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise

    Args:
        arr1: First array (list of ints/floats)
        arr2: Second array (list of ints/floats)

    Returns:
        A new list with element-wise sums, or None if shapes don't match
    """
    # Check if arrays have same length
    if len(arr1) != len(arr2):
        return None

    # Perform element-wise addition
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
