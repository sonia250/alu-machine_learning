#!/usr/bin/env python3
"""Additive (Bahdanau-style) attention for seq2seq decoding."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Additive alignment: Dense W/U plus Dense V on tanh(score)."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Apply additive attention scores over encoder time steps.

        Args:
            s_prev: ``(batch, units)`` prior decoder hidden state.
            hidden_states: ``(batch, sequence_length, encoder_units)``
                encoder outputs.

        Returns:
            Tuple ``(context, weights)``: ``context`` has shape ``(batch,
            units)``, ``weights`` ``(batch, input_seq_len, 1)``.
        """
        s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev) + self.U(hidden_states)
            )
        )
        attention_weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(
            attention_weights * hidden_states,
            axis=1,
        )
        return context, attention_weights
