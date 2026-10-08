"""SQuAD-style exact match, the outcome reward used by Search-R1."""

from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punc = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punc)
    return " ".join(without_articles.split())


def exact_match(prediction: str | None, golden_answers: list[str]) -> float:
    """1 when the normalized prediction equals any gold answer, else 0."""
    if prediction is None:
        return 0.0
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    return 1.0 if any(normalize_answer(gold) == pred for gold in golden_answers) else 0.0


def contains_answer(text: str, golden_answers: list[str]) -> bool:
    haystack = normalize_answer(text)
    return any(normalize_answer(gold) in haystack for gold in golden_answers if normalize_answer(gold))
