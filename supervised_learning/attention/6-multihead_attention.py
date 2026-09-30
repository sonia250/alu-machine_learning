#!/usr/bin/env python3
"""Multi-head attention layer for Transformer models."""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, dm, h):
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask=None):
        """Compute multi-head attention."""
        batch_size = tf.shape(Q)[0]

        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        attention, weights = sdp_attention(Q, K, V, mask)
        attention = tf.transpose(attention, [0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.dm))

        return self.linear(attention), weights

    def split_heads(self, x, batch_size):
        """Reshape to ``(batch, h, seq_len, depth)``."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])
