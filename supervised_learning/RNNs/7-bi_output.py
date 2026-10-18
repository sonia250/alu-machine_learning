#!/usr/bin/env python3
"""Bidirectional RNN output layer."""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell with an output layer."""

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

    def backward(self, h_next, x_t):
        """Calculate the previous backward hidden state."""
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whb) + self.bhb)

    def output(self, H):
        """Calculate outputs for concatenated hidden states."""
        logits = np.matmul(H, self.Wy) + self.by
        exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
