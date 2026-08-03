"""TrOCR-based recognition pipeline: image -> text + confidence.

Confidence is the mean per-token generation probability — softmax over the
model's logits at each decoding step, evaluated at the token it actually
picked. That's a standard, cheap proxy for how sure the model was while
generating; it says nothing about whether the recognized text is *correct*,
only how confidently the model committed to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float


class Recognizer:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 32,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        adapter_path: Path | None = None,
    ) -> None:
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._repetition_penalty = repetition_penalty
        self._no_repeat_ngram_size = no_repeat_ngram_size
        self._processor = TrOCRProcessor.from_pretrained(model_name)
        # No annotation on `model` on purpose: transformers/peft are
        # ignore_missing_imports in mypy, so leaving this as inferred-Any
        # keeps `.generate()` accessible on both the plain base model and
        # the peft-wrapped version. Annotating as `torch.nn.Module` would
        # make mypy hide `.generate`.
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        # Optional Phase 5 personalization adapter. Wrapped only if a
        # path is passed; a None keeps the base model behavior untouched
        # so tests, CLI eval, and un-personalized users all see the plain
        # recognizer path.
        if adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
        self._model = model
        self._model.to(device)
        self._model.eval()

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                pixel_values,
                max_new_tokens=self._max_new_tokens,
                repetition_penalty=self._repetition_penalty,
                no_repeat_ngram_size=self._no_repeat_ngram_size,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_confidence(output)
        return RecognitionResult(text=text, confidence=confidence)

    def recognize_topk(self, image: np.ndarray, k: int = 5) -> list[RecognitionResult]:
        """Return the top-k beam candidates for `image`, most-likely first.

        Introduced in Phase 7 for LM-assisted rescoring: a separate LM
        can re-rank these k options by combining each candidate's own
        `confidence` with an external log-prob. `.recognize()` still
        returns a single best result via its greedy-friendly path
        (unchanged), so all pre-Phase-7 callers keep working.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1 (got {k})")

        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        # HF beam search requires num_beams >= num_return_sequences and
        # >= 2 to actually be beam search. Using k directly for both
        # keeps each returned candidate an independent beam rather than
        # having beams merge and duplicate outputs.
        num_beams = max(k, 2)

        with torch.no_grad():
            output = self._model.generate(
                pixel_values,
                max_new_tokens=self._max_new_tokens,
                repetition_penalty=self._repetition_penalty,
                no_repeat_ngram_size=self._no_repeat_ngram_size,
                num_beams=num_beams,
                num_return_sequences=k,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self._processor.batch_decode(output.sequences, skip_special_tokens=True)
        confidences = self._compute_topk_confidences(output)
        return [
            RecognitionResult(text=text.strip(), confidence=confidence)
            for text, confidence in zip(texts, confidences, strict=True)
        ]

    def _compute_topk_confidences(self, output) -> list[float]:
        """Per-candidate mean-token-probability under beam search.

        Same semantic as `_compute_confidence`'s single-candidate
        formula (mean per-token generation probability). Uses
        `compute_transition_scores` so each returned sequence gets the
        log-probs of the tokens *it* picked — not the same shared
        scores tensor that `_compute_confidence` reads under greedy.
        """
        transition_scores = self._model.compute_transition_scores(
            output.sequences,
            output.scores,
            output.beam_indices,
            normalize_logits=True,
        )
        # transition_scores is (num_return_sequences, generated_len).
        # Entries for padding positions are -inf; masking them out
        # keeps the mean over genuinely-generated tokens only.
        probs = transition_scores.exp()
        finite_mask = torch.isfinite(transition_scores)
        confidences: list[float] = []
        for seq_probs, seq_mask in zip(probs, finite_mask, strict=True):
            valid = seq_probs[seq_mask]
            if valid.numel() == 0:
                confidences.append(0.0)
            else:
                confidences.append(float(valid.mean().item()))
        return confidences

    @staticmethod
    def _compute_confidence(output) -> float:
        scores = output.scores
        if not scores:
            return 0.0
        generated_token_ids = output.sequences[0, -len(scores) :]
        token_probs = [
            torch.softmax(logits[0], dim=-1)[token_id].item()
            for logits, token_id in zip(scores, generated_token_ids, strict=True)
        ]
        return float(np.mean(token_probs)) if token_probs else 0.0
