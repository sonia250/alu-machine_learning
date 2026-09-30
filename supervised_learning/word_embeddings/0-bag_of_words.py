#!/usr/bin/env python3
"""Creates a bag-of-words embedding matrix."""

import re
from collections import Counter

import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create bag-of-words embeddings."""
    tokenized = []

    for sentence in sentences:
        sentence = re.sub(r"'s\b", "", sentence.lower())
        tokenized.append(re.findall(r"\b\w+\b", sentence))

    if vocab is None:
        features = sorted(
            {word for sentence in tokenized for word in sentence}
        )
    else:
        features = list(vocab)

    embeddings = np.zeros(
        (len(sentences), len(features)),
        dtype=int,
    )

    feature_index = {word: index for index, word in enumerate(features)}

    for row, sentence in enumerate(tokenized):
        counts = Counter(sentence)
        for word, count in counts.items():
            if word in feature_index:
                embeddings[row, feature_index[word]] = count

    return embeddings, features
