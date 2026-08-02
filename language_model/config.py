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
class LanguageModelConfig:
    dictionary: DictionaryConfig

    @classmethod
    def from_dict(cls, data: dict) -> LanguageModelConfig:
        return cls(
            dictionary=DictionaryConfig.from_dict(data.get("dictionary", {})),
        )


def load_language_model_config() -> LanguageModelConfig:
    path = CONFIGS_DIR / "language_model.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return LanguageModelConfig.from_dict(data)
