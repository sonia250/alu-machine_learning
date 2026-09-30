#!/usr/bin/env python3
"""Transformer model for machine translation."""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """Transformer encoder-decoder model."""

    def __init__(
            self, N, dm, h, hidden, input_vocab, target_vocab,
            max_seq_len, drop_rate=0.1):
        """Initialize the Transformer."""
        super().__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_len, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_len, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self, inputs, target, training,
            encoder_mask=None, look_ahead_mask=None, decoder_mask=None):
        """Run the Transformer."""
        enc_output = self.encoder(
            inputs, training=training, mask=encoder_mask
        )
        dec_output = self.decoder(
            target,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask,
        )
        return self.linear(dec_output)
