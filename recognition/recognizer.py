"""TrOCR-based recognition pipeline: image -> text + confidence.

Confidence is the mean per-token generation probability — softmax over the
model's logits at each decoding step, evaluated at the token it actually
picked. That's a standard, cheap proxy for how sure the model was while
generating; it says nothing about whether the recognized text is *correct*,
only how confidently the model committed to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float


class Recognizer:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 32) -> None:
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
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
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self._processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = self._compute_confidence(output)
        return RecognitionResult(text=text, confidence=confidence)

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
