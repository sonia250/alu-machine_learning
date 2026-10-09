#!/usr/bin/env python3
"""Question-answering function using BERT."""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)
model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')


def question_answer(question, reference):
    """Find an answer to question within reference."""
    inputs = tokenizer(
        question,
        reference,
        return_tensors='tf',
    )

    outputs = model([
        inputs['input_ids'],
        inputs['attention_mask'],
        inputs['token_type_ids'],
    ])

    start = int(tf.argmax(outputs[0][0]))
    end = int(tf.argmax(outputs[1][0]))

    if start > end:
        return None

    tokens = inputs['input_ids'][0][start:end + 1]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
    )

    return answer if answer else None
