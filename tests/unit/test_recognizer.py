import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
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


# ---------- recognize_topk (Phase 7 PR 3) ----------------------------------

def _fake_topk_output(
    texts_after_decode: list[str], per_seq_logprobs: list[list[float]]
):
    """Build a stand-in for `model.generate(..., return_dict_in_generate=True)`
    whose shape matches what the real HF beam-search output looks like.

    `texts_after_decode` sets what the (mocked) processor will decode;
    `per_seq_logprobs` seeds `compute_transition_scores`'s return so
    _compute_topk_confidences gets deterministic per-candidate probs.
    """
    k = len(texts_after_decode)
    # Sequences: k rows of arbitrary token ids (contents don't matter
    # because the processor mock returns preset text).
    sequences = torch.zeros((k, 4), dtype=torch.long)
    # scores/beam_indices don't need real values — they're only fed
    # into compute_transition_scores, which we override on the mocked
    # model to return the transition_scores we want.
    scores = (torch.zeros((k, 5)),)
    beam_indices = torch.zeros((k, 4), dtype=torch.long)
    return SimpleNamespace(
        sequences=sequences, scores=scores, beam_indices=beam_indices
    ), torch.tensor(per_seq_logprobs)


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognize_topk_returns_k_ranked_results(mock_processor_cls, mock_vem_cls):
    output, transition_scores = _fake_topk_output(
        texts_after_decode=["hello", "helllo", "hollo"],
        # Log-probs: three candidates, three tokens each. Exp+mean
        # gives 1st > 2nd > 3rd — the ordering beam search itself
        # already produced.
        per_seq_logprobs=[
            [math.log(0.9), math.log(0.9), math.log(0.9)],
            [math.log(0.7), math.log(0.7), math.log(0.7)],
            [math.log(0.5), math.log(0.5), math.log(0.5)],
        ],
    )

    mock_model = MagicMock()
    mock_model.generate.return_value = output
    mock_model.compute_transition_scores.return_value = transition_scores
    mock_vem_cls.from_pretrained.return_value = mock_model

    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = ["hello", "helllo", "hollo"]
    mock_processor.return_value.pixel_values = torch.zeros((1, 3, 4, 4))
    mock_processor_cls.from_pretrained.return_value = mock_processor

    rec = Recognizer(model_name="fake-model", device="cpu")
    results = rec.recognize_topk(np.zeros((10, 10), dtype=np.uint8), k=3)

    assert len(results) == 3
    assert [r.text for r in results] == ["hello", "helllo", "hollo"]
    assert results[0].confidence == pytest.approx(0.9, abs=1e-5)
    assert results[1].confidence == pytest.approx(0.7, abs=1e-5)
    assert results[2].confidence == pytest.approx(0.5, abs=1e-5)


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognize_topk_calls_generate_with_num_beams_at_least_two(
    mock_processor_cls, mock_vem_cls
):
    # Even for k=1, num_beams must be >=2 for HF to treat generate() as
    # beam search rather than falling back to greedy (which would return
    # the wrong-shaped output).
    output, transition_scores = _fake_topk_output(["hi"], [[math.log(0.5)]])
    mock_model = MagicMock()
    mock_model.generate.return_value = output
    mock_model.compute_transition_scores.return_value = transition_scores
    mock_vem_cls.from_pretrained.return_value = mock_model
    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = ["hi"]
    mock_processor.return_value.pixel_values = torch.zeros((1, 3, 4, 4))
    mock_processor_cls.from_pretrained.return_value = mock_processor

    rec = Recognizer(model_name="fake-model", device="cpu")
    rec.recognize_topk(np.zeros((10, 10), dtype=np.uint8), k=1)

    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["num_beams"] >= 2
    assert kwargs["num_return_sequences"] == 1


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognize_topk_rejects_k_below_one(mock_processor_cls, mock_vem_cls):
    mock_vem_cls.from_pretrained.return_value = MagicMock()
    mock_processor_cls.from_pretrained.return_value = MagicMock()
    rec = Recognizer(model_name="fake-model", device="cpu")
    with pytest.raises(ValueError):
        rec.recognize_topk(np.zeros((10, 10), dtype=np.uint8), k=0)


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognize_topk_masks_out_padding_positions(
    mock_processor_cls, mock_vem_cls
):
    # Real beam search often pads shorter candidates with -inf in
    # compute_transition_scores' output. Those entries must be
    # excluded from the mean so a short candidate isn't penalized
    # for its own padding.
    output, transition_scores = _fake_topk_output(
        texts_after_decode=["a", "b"],
        per_seq_logprobs=[
            [math.log(0.8), math.log(0.8), math.log(0.8)],
            [math.log(0.6), float("-inf"), float("-inf")],
        ],
    )
    mock_model = MagicMock()
    mock_model.generate.return_value = output
    mock_model.compute_transition_scores.return_value = transition_scores
    mock_vem_cls.from_pretrained.return_value = mock_model
    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = ["a", "b"]
    mock_processor.return_value.pixel_values = torch.zeros((1, 3, 4, 4))
    mock_processor_cls.from_pretrained.return_value = mock_processor

    rec = Recognizer(model_name="fake-model", device="cpu")
    results = rec.recognize_topk(np.zeros((10, 10), dtype=np.uint8), k=2)

    assert results[0].confidence == pytest.approx(0.8, abs=1e-5)
    # Second candidate: only one non-inf token, so confidence = 0.6.
    assert results[1].confidence == pytest.approx(0.6, abs=1e-5)


@patch("recognition.recognizer.VisionEncoderDecoderModel")
@patch("recognition.recognizer.TrOCRProcessor")
def test_recognize_topk_all_inf_candidate_returns_zero_confidence(
    mock_processor_cls, mock_vem_cls
):
    output, transition_scores = _fake_topk_output(
        texts_after_decode=[""],
        per_seq_logprobs=[[float("-inf"), float("-inf"), float("-inf")]],
    )
    mock_model = MagicMock()
    mock_model.generate.return_value = output
    mock_model.compute_transition_scores.return_value = transition_scores
    mock_vem_cls.from_pretrained.return_value = mock_model
    mock_processor = MagicMock()
    mock_processor.batch_decode.return_value = [""]
    mock_processor.return_value.pixel_values = torch.zeros((1, 3, 4, 4))
    mock_processor_cls.from_pretrained.return_value = mock_processor

    rec = Recognizer(model_name="fake-model", device="cpu")
    results = rec.recognize_topk(np.zeros((10, 10), dtype=np.uint8), k=1)

    assert results[0].confidence == 0.0
