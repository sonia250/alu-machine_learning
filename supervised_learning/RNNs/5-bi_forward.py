#!/usr/bin/env python3
"""Bidirectional RNN forward cell."""

import numpy as np


class BidirectionalCell:
    """Represents one direction of a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """Initialize the cell."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the next forward hidden state."""
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)
