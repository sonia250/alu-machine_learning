#!/usr/bin/env python3
"""Dataset preparation and token encoding for translation."""

import numpy as np
import tensorflow_datasets as tfds


class Dataset:
    """Loads and tokenizes the Portuguese-English dataset."""

    def __init__(self):
        """Initialize datasets and tokenizers."""
        data = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
        )

        self.data_train = data['train']
        self.data_valid = data['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers from the dataset."""
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
        """Encode Portuguese and English sentences with start/end tokens."""
        pt_tokens = [self.tokenizer_pt.vocab_size]
        pt_tokens += self.tokenizer_pt.encode(pt.numpy().decode('utf-8'))
        pt_tokens += [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size]
        en_tokens += self.tokenizer_en.encode(en.numpy().decode('utf-8'))
        en_tokens += [self.tokenizer_en.vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)
