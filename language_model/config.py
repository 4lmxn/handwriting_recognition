"""Configuration for Phase 7 language-model-assisted decoding.

The dictionary is loaded from up to three sources — base, user, and
domain — merged into one canonical vocab. Every source is optional
(paths may be missing or null) so a fresh clone still works; only
sources with a real file behind them contribute words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import CONFIGS_DIR, REPO_ROOT


@dataclass(frozen=True)
class DictionaryConfig:
    # Three word-list files, merged into one vocab. Kept as separate
    # fields (rather than a single list) so future PRs can weight or
    # prioritize them per source — e.g. boost user-added words over
    # base vocab in the rescoring formula.
    #
    # Each path may be either a .json array of strings or a plain-text
    # file with one word per line. Missing/blank paths are treated as
    # "no words from this source", not as errors — a fresh clone has
    # neither a user list nor a domain list yet.
    base_path: str | None
    user_path: str | None
    domain_path: str | None
    # Whether dictionary lookup is case-sensitive. Handwriting output
    # is inherently case-uncertain (upper vs lower i/l ambiguity, etc.)
    # so False is the sensible default; explicit True is for callers
    # that want proper-noun capitalization preserved.
    case_sensitive: bool = False

    def _resolved(self, path: str | None) -> Path | None:
        if path is None or path == "":
            return None
        return REPO_ROOT / path

    @property
    def base_path_resolved(self) -> Path | None:
        return self._resolved(self.base_path)

    @property
    def user_path_resolved(self) -> Path | None:
        return self._resolved(self.user_path)

    @property
    def domain_path_resolved(self) -> Path | None:
        return self._resolved(self.domain_path)

    @classmethod
    def from_dict(cls, data: dict) -> DictionaryConfig:
        return cls(
            base_path=data.get("base_path"),
            user_path=data.get("user_path"),
            domain_path=data.get("domain_path"),
            case_sensitive=bool(data.get("case_sensitive", False)),
        )


@dataclass(frozen=True)
class RescoringConfig:
    """Tunables for language_model.rescoring.RescoringRecognizer.

    Off by default (`enabled=False`) so a fresh clone doesn't
    silently change recognition results — the caller opts in.
    """
    enabled: bool = False
    # Number of beam candidates to pull from the base recognizer and
    # re-rank. Bigger = more re-ranking headroom, but each candidate
    # requires an LM score, and beam search itself gets slower with
    # more beams. 5 is a reasonable balance.
    topk: int = 5
    # Weight on the LM's log-probability in the combined score:
    #   combined = (1 - lm_weight) * log(model_conf) + lm_weight * lm.score(text)
    # 0.0 disables the LM entirely (equivalent to enabled=False); 1.0
    # ignores model confidence and picks purely by LM score (usually
    # a bad idea — the LM is dumber than the model on well-formed
    # inputs). 0.3 nudges rank order without overriding the model.
    lm_weight: float = 0.3
    # Levenshtein threshold for snap-to-nearest-dict-word correction
    # applied to the winning candidate. 0 = disabled (candidate returned
    # untouched); positive integers snap when there's a dictionary word
    # within that edit distance. Applied only if the winner isn't
    # already in the dictionary, and skipped when the vocab is empty.
    snap_edit_distance: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> RescoringConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            topk=int(data.get("topk", 5)),
            lm_weight=float(data.get("lm_weight", 0.3)),
            snap_edit_distance=int(data.get("snap_edit_distance", 0)),
        )


@dataclass(frozen=True)
class NGramConfig:
    """Tunables for the character n-gram LM training pass."""
    n: int = 3
    smoothing_k: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> NGramConfig:
        return cls(
            n=int(data.get("n", 3)),
            smoothing_k=float(data.get("smoothing_k", 1.0)),
        )


@dataclass(frozen=True)
class LanguageModelConfig:
    dictionary: DictionaryConfig
    ngram: NGramConfig
    rescoring: RescoringConfig

    @classmethod
    def from_dict(cls, data: dict) -> LanguageModelConfig:
        return cls(
            dictionary=DictionaryConfig.from_dict(data.get("dictionary", {})),
            ngram=NGramConfig.from_dict(data.get("ngram", {})),
            rescoring=RescoringConfig.from_dict(data.get("rescoring", {})),
        )


def load_language_model_config() -> LanguageModelConfig:
    path = CONFIGS_DIR / "language_model.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return LanguageModelConfig.from_dict(data)
