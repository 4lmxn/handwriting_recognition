import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognizer_does_not_wrap_with_peft_when_no_adapter(mock_processor, mock_vem):
    # Sanity: default construction (adapter_path=None) must NOT import peft
    # or wrap the model. This preserves the pre-Phase-5 code path for
    # everyone who doesn't have an adapter yet.
    with patch("peft.PeftModel.from_pretrained") as mock_peft_load:
        Recognizer(model_name="fake-model", device="cpu")
    mock_peft_load.assert_not_called()


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognizer_wraps_with_peft_when_adapter_path_set(mock_processor, mock_vem, tmp_path):
    base_model = MagicMock()
    mock_vem.from_pretrained.return_value = base_model
    wrapped = MagicMock()

    with patch("peft.PeftModel") as mock_peft_class:
        mock_peft_class.from_pretrained.return_value = wrapped
        Recognizer(model_name="fake-model", device="cpu", adapter_path=tmp_path / "adapter")

    mock_peft_class.from_pretrained.assert_called_once_with(base_model, str(tmp_path / "adapter"))
    wrapped.to.assert_called_once_with("cpu")
    wrapped.eval.assert_called_once()
