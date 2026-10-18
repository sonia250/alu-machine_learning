#!/usr/bin/env python3
"""Bidirectional RNN forward propagation."""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation through a bidirectional RNN."""
    t, m, _ = X.shape
    h = h_0.shape[1]
    H = np.zeros((t, m, 2 * h))
    for step in range(t):
        forward = bi_cell.forward(
            h_0 if step == 0 else H[step - 1, :, :h],
            X[step],
        )
        H[step, :, :h] = forward
    for step in range(t - 1, -1, -1):
        backward = bi_cell.backward(
            h_t if step == t - 1 else H[step + 1, :, h:],
            X[step],
        )
        H[step, :, h:] = backward
    return H, bi_cell.output(H)
