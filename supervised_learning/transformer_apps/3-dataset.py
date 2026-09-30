#!/usr/bin/env python3
"""Dataset pipeline for Transformer translation."""

import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """Loads, tokenizes, and batches the translation dataset."""

    def __init__(self, batch_size, max_len):
        """Initialize the training and validation pipelines."""
        data = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
        )

        self.data_train = data['train']
        self.data_valid = data['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        self.data_valid = self.data_valid.map(
            self.tf_encode,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        self.data_train = (
            self.data_train
            .filter(
                lambda pt, en: tf.logical_and(
                    tf.size(pt) <= max_len,
                    tf.size(en) <= max_len,
                )
            )
            .cache()
            .shuffle(10000)
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None]),
                padding_values=(0, 0),
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.data_valid = (
            self.data_valid
            .filter(
                lambda pt, en: tf.logical_and(
                    tf.size(pt) <= max_len,
                    tf.size(en) <= max_len,
                )
            )
            .padded_batch(
                batch_size,
                padded_shapes=([None], [None]),
                padding_values=(0, 0),
            )
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        tokenizer_pt = (
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy().decode('utf-8') for pt, _ in data),
                target_vocab_size=2 ** 15,
            )
        )
        tokenizer_en = (
            tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy().decode('utf-8') for _, en in data),
                target_vocab_size=2 ** 15,
            )
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode sentences with start and end tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        pt_tokens += [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy().decode('utf-8'))
        en_tokens += [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Wrap ``encode`` for TensorFlow."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            (pt, en),
            (tf.int64, tf.int64),
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
