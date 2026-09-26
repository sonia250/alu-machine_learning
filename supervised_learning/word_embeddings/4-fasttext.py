#!/usr/bin/env python3
"""Train a FastText model."""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a FastText model.

    Args:
        sentences: List of tokenized sentences.
        size: Dimensionality of the word vectors.
        min_count: Minimum frequency of a word.
        negative: Number of negative samples.
        window: Maximum distance between the current and predicted word.
        cbow: If True, use CBOW; otherwise use Skip-gram.
        iterations: Number of training iterations.
        seed: Random seed.
        workers: Number of worker threads.

    Returns:
        A trained FastText model.
    """
    sg = 0 if cbow else 1

    model = FastText(
        sentences=sentences,
        vector_size=size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=sg,
        epochs=iterations,
        seed=seed,
        workers=workers
    )

    return model