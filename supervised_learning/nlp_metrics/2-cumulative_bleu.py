#!/usr/bin/env python3
"""Cumulative n-gram BLEU score."""

from collections import Counter
from math import exp, log


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score."""
    sentence_length = len(sentence)
    if sentence_length == 0:
        return 0

    reference_length = min(
        (len(reference) for reference in references),
        key=lambda length: (abs(length - sentence_length), length),
    )

    if sentence_length > reference_length:
        brevity_penalty = 1
    else:
        brevity_penalty = exp(
            1 - reference_length / sentence_length
        )

    precisions = []

    for size in range(1, n + 1):
        sentence_ngrams = Counter(
            tuple(sentence[index:index + size])
            for index in range(sentence_length - size + 1)
        )

        reference_ngrams = Counter()
        for reference in references:
            current = Counter(
                tuple(reference[index:index + size])
                for index in range(len(reference) - size + 1)
            )
            for ngram, count in current.items():
                reference_ngrams[ngram] = max(
                    reference_ngrams[ngram], count
                )

        clipped_count = sum(
            min(count, reference_ngrams[ngram])
            for ngram, count in sentence_ngrams.items()
        )
        total_count = sum(sentence_ngrams.values())

        if total_count == 0 or clipped_count == 0:
            return 0

        precisions.append(clipped_count / total_count)

    score = exp(sum(log(precision) for precision in precisions) / n)
    return brevity_penalty * score
