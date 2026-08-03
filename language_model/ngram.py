"""Character-level n-gram language model for Phase 7 rescoring.

Trains on an iterable of words (typically the merged
`language_model.dictionary.Dictionary`) and scores arbitrary strings
under a Laplace-smoothed n-gram distribution over characters.

Why character-level rather than word-level:
  1. Rescoring targets *misrecognitions* — a wrong letter in a real
     word ("thc" for "the") that a word-LM only sees as OOV. A
     character LM assigns those a low but finite probability that
     still correctly ranks candidates against each other.
  2. Training data is the same vocab we already have; no separate
     text corpus needed.
  3. Small footprint — dict of int counts, no ML deps.

Why Laplace (add-k) smoothing rather than Kneser-Ney or backoff:
  Simplicity. This LM only needs to *rank* candidates from an already-
  small top-K list, not produce well-calibrated probabilities. Add-k
  gives every ngram a floor probability so unseen substrings never
  crash the score to -inf. Upgrading to backoff is a documented
  follow-up if empirical rescoring accuracy falls short.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

# Sentinel characters chosen to (a) be single-code-point so they slot
# into the character n-gram substrings cleanly, and (b) never appear
# in real ASCII/Latin handwriting output — a Private Use Area code
# point can't collide with anything the recognizer produces.
_START_TOKEN = ""
_END_TOKEN = ""


class NGramLM:
    """Character n-gram LM with Laplace smoothing.

    Immutable after fit(): count tables snapshot into frozen dicts so
    a shared model instance is safe to pass across recognition calls
    without accidentally accumulating training data.
    """

    def __init__(self, n: int = 3, smoothing_k: float = 1.0) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        if smoothing_k <= 0:
            raise ValueError(f"smoothing_k must be > 0 (got {smoothing_k})")
        self._n = n
        self._k = smoothing_k
        self._ngram_counts: dict[str, int] = {}
        self._context_counts: dict[str, int] = {}
        self._vocab_size: int = 0
        self._fitted = False

    def fit(self, words: Iterable[str]) -> NGramLM:
        """Accumulate character n-gram counts from `words`.

        Each word is padded with (n-1) start tokens and one end token
        so word-boundary transitions ("first letter given start of
        word", "any letter → end of word") contribute to scores.
        Returns self so training is chainable.
        """
        ngram_counts: Counter[str] = Counter()
        context_counts: Counter[str] = Counter()
        vocab: set[str] = {_END_TOKEN}
        for word in words:
            if not word:
                continue
            padded = _START_TOKEN * (self._n - 1) + word + _END_TOKEN
            vocab.update(word)
            for i in range(len(padded) - self._n + 1):
                context = padded[i : i + self._n - 1]
                target = padded[i + self._n - 1]
                ngram_counts[context + target] += 1
                context_counts[context] += 1
        self._ngram_counts = dict(ngram_counts)
        self._context_counts = dict(context_counts)
        self._vocab_size = len(vocab)
        self._fitted = True
        return self

    def score(self, text: str) -> float:
        """Return the log-probability of `text` under this LM.

        Uses the natural log. Empty strings score 0.0 (log 1) — an
        empty candidate is trivially "seen"; rescoring code that wants
        to penalize empty candidates should do it explicitly rather
        than expecting the LM to.
        """
        if not self._fitted:
            raise RuntimeError("NGramLM.score called before fit()")
        if not text:
            return 0.0
        padded = _START_TOKEN * (self._n - 1) + text + _END_TOKEN
        log_prob = 0.0
        for i in range(len(padded) - self._n + 1):
            context = padded[i : i + self._n - 1]
            target = padded[i + self._n - 1]
            ngram_count = self._ngram_counts.get(context + target, 0)
            context_count = self._context_counts.get(context, 0)
            # Laplace add-k: numerator gets +k, denominator gets +k*|V|.
            # Guarantees a finite probability even for a context / target
            # combination never seen during training.
            prob = (ngram_count + self._k) / (
                context_count + self._k * self._vocab_size
            )
            log_prob += math.log(prob)
        return log_prob

    @property
    def n(self) -> int:
        return self._n

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def is_fitted(self) -> bool:
        return self._fitted
