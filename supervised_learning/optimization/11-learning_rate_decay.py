#!/usr/bin/env python3
"""Module for learning rate decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay

    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha decays
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent before alpha decays

    Returns:
        the updated value for alpha
    """
    # Calculate which decay step we're in (stepwise fashion using floor)
    step = np.floor(global_step / decay_step)

    # Apply inverse time decay formula
    updated_alpha = alpha / (1 + decay_rate * step)

    return updated_alpha
