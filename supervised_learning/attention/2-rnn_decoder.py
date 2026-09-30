#!/usr/bin/env python3
"""Attention-based GRU decoder."""

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN decoder with additive attention."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the decoder."""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)
        self.batch = batch

    def call(self, x, s_prev, hidden_states):
        """Perform one decoding step."""
        context, _ = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)

        x = self.embedding(x)
        x = tf.concat([context, x], axis=-1)

        output, state = self.gru(x, initial_state=s_prev)
        y = self.F(tf.squeeze(output, axis=1))

        return y, state
