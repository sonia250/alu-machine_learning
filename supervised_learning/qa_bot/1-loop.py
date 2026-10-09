#!/usr/bin/env python3
"""Simple question-answer loop."""

while True:
    question = input("Q: ")
    if question.lower() in ("exit", "quit", "goodbye", "bye"):
        print("A: Goodbye")
        break
    print("A:")