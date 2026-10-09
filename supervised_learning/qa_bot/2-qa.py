#!/usr/bin/env python3
"""Question-answering loop."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answer questions using the reference text."""
    while True:
        question = input("Q: ")

        if question.lower() in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."

        print(f"A: {answer}")
