#!/usr/bin/env python3
"""Train a Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a Word2Vec model.

    Args:
        sentences: List of tokenized sentences.
        size: Dimensionality of the word vectors.
        min_count: Minimum frequency of a word.
        window: Maximum distance between the current and predicted word.
        negative: Number of negative samples.
        cbow: If True, use CBOW; otherwise use Skip-gram.
        iterations: Number of training iterations.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        A trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=iterations,
        seed=seed,
        workers=workers
    )

    return model