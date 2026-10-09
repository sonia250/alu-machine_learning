#!/usr/bin/env python3
"""Multi-reference question-answering loop."""

semantic_search = __import__(
    '3-semantic_search'
).semantic_search
qa = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """Answer questions using multiple reference documents."""
    while True:
        question = input("Q: ")

        if question.lower() in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference) if reference else None

        if answer is None:
            answer = "Sorry, I do not understand your question."

        print(f"A: {answer}")
