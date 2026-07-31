from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _levenshtein_distance(a: Sequence, b: Sequence) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    previous_row = list(range(m + 1))
    for i in range(1, n + 1):
        current_row = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current_row[j] = min(
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1] + cost,
            )
        previous_row = current_row
    return previous_row[m]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if len(reference_words) == 0:
        return 0.0 if len(hypothesis_words) == 0 else 1.0
    return _levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def character_accuracy(reference: str, hypothesis: str) -> float:
    return max(0.0, min(1.0, 1.0 - character_error_rate(reference, hypothesis)))


def word_accuracy(reference: str, hypothesis: str) -> float:
    return max(0.0, min(1.0, 1.0 - word_error_rate(reference, hypothesis)))


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference == hypothesis


@dataclass(frozen=True)
class EvaluationResult:
    mean_cer: float
    mean_wer: float
    mean_char_accuracy: float
    mean_word_accuracy: float
    exact_match_rate: float
    num_samples: int


def evaluate_predictions(references: list[str], hypotheses: list[str]) -> EvaluationResult:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, got "
            f"{len(references)} and {len(hypotheses)}"
        )
    if len(references) == 0:
        raise ValueError("references and hypotheses must not be empty")

    num_samples = len(references)
    pairs = list(zip(references, hypotheses, strict=True))
    cers = [character_error_rate(r, h) for r, h in pairs]
    wers = [word_error_rate(r, h) for r, h in pairs]
    char_accuracies = [character_accuracy(r, h) for r, h in pairs]
    word_accuracies = [word_accuracy(r, h) for r, h in pairs]
    exact_matches = [exact_match(r, h) for r, h in pairs]

    return EvaluationResult(
        mean_cer=sum(cers) / num_samples,
        mean_wer=sum(wers) / num_samples,
        mean_char_accuracy=sum(char_accuracies) / num_samples,
        mean_word_accuracy=sum(word_accuracies) / num_samples,
        exact_match_rate=sum(exact_matches) / num_samples,
        num_samples=num_samples,
    )
