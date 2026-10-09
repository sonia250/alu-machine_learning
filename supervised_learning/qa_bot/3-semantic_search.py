#!/usr/bin/env python3
"""Semantic search over a document corpus."""

import glob
import os

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_search(corpus_path, sentence):
    """Return the document most similar to sentence."""
    paths = glob.glob(os.path.join(corpus_path, "*"))
    documents = []

    for path in paths:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as file:
                documents.append(file.read())

    if not documents:
        return None

    query_embedding = model.encode(sentence, convert_to_tensor=True)
    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True,
    )
    similarities = util.cos_sim(query_embedding, document_embeddings)[0]
    index = int(similarities.argmax())

    return documents[index]
