#!/usr/bin/env python3
"""Convert a Gensim Word2Vec model to a Keras Embedding layer."""


def gensim_to_keras(model):
    """Create a trainable Keras Embedding layer from a Gensim model.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer.
    """
    return model.wv.get_keras_embedding(True)