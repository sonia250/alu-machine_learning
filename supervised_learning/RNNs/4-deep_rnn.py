#!/usr/bin/env python3
"""Deep RNN forward propagation."""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation through a deep RNN."""
    t, m, _ = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((t + 1, layers, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for step in range(t):
        layer_input = X[step]
        for layer, cell in enumerate(rnn_cells):
            H[step + 1, layer], output = cell.forward(
                H[step, layer],
                layer_input,
            )
            layer_input = H[step + 1, layer]
        Y[step] = output
    return H, Y
