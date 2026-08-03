"""LM-assisted rescoring on top of a base Recognizer (Phase 7, PR 4).

Pulls the top-K beam candidates from the base recognizer (see
`Recognizer.recognize_topk` added in PR 3), combines each candidate's
own model confidence with a character n-gram log-probability, and
optionally snaps the winner to the nearest dictionary word within a
Levenshtein threshold.

The wrapper preserves the `Recognizer` shape (`.recognize(image) →
RecognitionResult`) so any caller that talks to a base recognizer can
swap in a RescoringRecognizer without other changes. Disabled state
is a pure pass-through so wiring this up unconditionally in the GUI
is safe as long as `RescoringConfig.enabled=False` is the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from language_model.config import RescoringConfig, load_language_model_config
from language_model.dictionary import Dictionary
from language_model.ngram import NGramLM
from recognition.recognizer import RecognitionResult, Recognizer

# Floor for log(model_confidence) — protects against confidence=0
# from an all-padding candidate blowing up the combined score to
# -inf and hiding otherwise-valid rescoring signal.
_MIN_CONF_FOR_LOG = 1e-10


@dataclass(frozen=True)
class RecognizerLike:
    """Minimal protocol RescoringRecognizer needs from its base.

    Kept as a doc-only marker; both `Recognizer` and any mock with
    `.recognize()` + `.recognize_topk()` satisfy it structurally, so
    duck typing rather than nominal inheritance keeps the wrapper
    testable without a heavyweight fake.
    """


class RescoringRecognizer:
    """Wraps a base Recognizer to apply LM + dictionary post-processing."""

    def __init__(
        self,
        base: Recognizer,
        ngram: NGramLM,
        dictionary: Dictionary,
        config: RescoringConfig,
    ) -> None:
        self._base = base
        self._ngram = ngram
        self._dictionary = dictionary
        self._config = config

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        # Fast path: config off → pure pass-through, no beam-search
        # overhead, no LM call, exact same result as the base.
        if not self._config.enabled:
            return self._base.recognize(image)

        candidates = self._base.recognize_topk(image, k=self._config.topk)
        if not candidates:
            return RecognitionResult(text="", confidence=0.0)

        winner = max(candidates, key=self._combined_score)

        # Preserve the model's own confidence on the returned result —
        # rescoring changes text selection, not confidence semantics.
        # Downstream UIs display confidence to the user; keeping the
        # meaning stable across recognizer variants avoids surprising
        # them.
        if (
            self._config.snap_edit_distance > 0
            and len(self._dictionary) > 0
            and winner.text
            and winner.text not in self._dictionary
        ):
            snapped = self._snap_to_dictionary(winner.text)
            if snapped is not None:
                return RecognitionResult(text=snapped, confidence=winner.confidence)
        return winner

    def _combined_score(self, candidate: RecognitionResult) -> float:
        model_log = math.log(max(candidate.confidence, _MIN_CONF_FOR_LOG))
        lm_log = self._ngram.score(candidate.text) if self._ngram.is_fitted else 0.0
        w = self._config.lm_weight
        return (1.0 - w) * model_log + w * lm_log

    def _snap_to_dictionary(self, text: str) -> str | None:
        """Return the nearest dict word if within `snap_edit_distance`.

        Case-normalization mirrors the dictionary's own convention:
        when case-insensitive, we compare and return the lowercase
        form (matching what `Dictionary.words()` stores).
        """
        threshold = self._config.snap_edit_distance
        key = text if self._dictionary.case_sensitive else text.lower()
        best_word: str | None = None
        best_distance = threshold + 1
        for word in self._dictionary.words():
            # Early prune: if lengths differ by more than the threshold,
            # edit distance is at least that difference — skip without
            # running the full DP.
            if abs(len(word) - len(key)) > threshold:
                continue
            distance = _levenshtein(key, word)
            if distance < best_distance:
                best_distance = distance
                best_word = word
                if best_distance == 0:
                    break
        if best_word is not None and best_distance <= threshold:
            return best_word
        return None


def wrap_if_enabled(base: Recognizer) -> Recognizer | RescoringRecognizer:
    """Wrap `base` with LM-assisted rescoring iff configs enable it.

    Called by every place that constructs a base Recognizer for
    interactive use so a single config flip
    (`configs/language_model.yaml: rescoring.enabled: true`) turns
    rescoring on across the whole app without a code change. Returns
    the untouched base when rescoring is off, so tests + callers that
    don't set up a dictionary keep behaving exactly as before.

    The dictionary + n-gram are constructed lazily inside this
    function rather than at import time — a fresh clone with no
    word lists and rescoring off should pay zero cost.
    """
    lm_config = load_language_model_config()
    if not lm_config.rescoring.enabled:
        return base
    dictionary = Dictionary.from_config(lm_config.dictionary)
    ngram = NGramLM(n=lm_config.ngram.n, smoothing_k=lm_config.ngram.smoothing_k)
    if len(dictionary) > 0:
        # No vocab → no useful LM signal; leaving the NGramLM unfitted
        # is fine because the rescorer's _combined_score falls back to
        # pure model log-conf when is_fitted is False.
        ngram.fit(dictionary.words())
    return RescoringRecognizer(
        base=base,
        ngram=ngram,
        dictionary=dictionary,
        config=lm_config.rescoring,
    )


def _levenshtein(a: str, b: str) -> int:
    """Standard iterative-DP Levenshtein distance.

    Kept local rather than importing `training.evaluation._levenshtein_distance`
    to avoid coupling the LM package to the training package (which
    also pulls in the training dataset / torch chain elsewhere).
    """
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[m]
