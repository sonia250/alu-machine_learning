#!/usr/bin/env python3
"""Module for calculating weighted moving average"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        list containing the moving averages of data with bias correction
    """
    moving_avgs = []
    v = 0  # First moment (moving average)

    for i, val in enumerate(data):
        # Update the moving average
        v = beta * v + (1 - beta) * val

        # Bias correction: divide by (1 - beta^(t+1)) where t is the step number
        bias_correction = 1 - (beta ** (i + 1))
        corrected_avg = v / bias_correction

        moving_avgs.append(corrected_avg)

    return moving_avgs
