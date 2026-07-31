import math
from types import SimpleNamespace

import pytest
import torch

from recognition.recognizer import RecognitionResult, Recognizer


def test_recognition_result_is_a_frozen_dataclass():
    result = RecognitionResult(text="hi", confidence=0.9)
    assert result.text == "hi"
    assert result.confidence == 0.9


def test_compute_confidence_matches_hand_computed_softmax():
    scores = (
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 3.0]]),
    )
    sequences = torch.tensor([[999, 0, 2]])  # prefix token, then the 2 generated tokens
    fake_output = SimpleNamespace(scores=scores, sequences=sequences)

    confidence = Recognizer._compute_confidence(fake_output)

    prob0 = math.exp(2.0) / (math.exp(2.0) + 1 + 1)
    prob1 = math.exp(3.0) / (1 + 1 + math.exp(3.0))
    expected = (prob0 + prob1) / 2
    assert confidence == pytest.approx(expected, abs=1e-5)


def test_compute_confidence_returns_zero_for_empty_scores():
    fake_output = SimpleNamespace(scores=(), sequences=torch.tensor([[0]]))
    assert Recognizer._compute_confidence(fake_output) == 0.0
