#!/usr/bin/env python3
"""Creates TF-IDF embeddings."""

import re

import numpy as np


def tf_idf(sentences, vocab=None):
    """Create normalized TF-IDF embeddings."""
    tokenized = []
    for sentence in sentences:
        sentence = re.sub(r"'s\b", "", sentence.lower())
        tokenized.append(re.findall(r"\b\w+\b", sentence))

    if vocab is None:
        features = sorted({
            word for sentence in tokenized for word in sentence
        })
    else:
        features = list(vocab)

    embeddings = np.zeros(
        (len(sentences), len(features)),
        dtype=float,
    )

    document_count = {
        word: sum(word in sentence for sentence in tokenized)
        for word in features
    }

    idf = {
        word: np.log(
            (1 + len(sentences)) / (1 + document_count[word])
        ) + 1
        for word in features
    }

    for row, sentence in enumerate(tokenized):
        length = len(sentence)
        if length == 0:
            continue

        for column, word in enumerate(features):
            term_frequency = sentence.count(word) / length
            embeddings[row, column] = term_frequency * idf[word]

        norm = np.linalg.norm(embeddings[row])
        if norm:
            embeddings[row] /= norm

    return embeddings, features
