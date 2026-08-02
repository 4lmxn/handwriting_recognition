"""Vocabulary dictionary for Phase 7 rescoring / correction.

Loads one to three word lists (base, user, domain), merges them, and
exposes membership queries. Kept intentionally minimal — future PRs
add weighting, snap-to-nearest correction, and an n-gram LM on top;
this module just answers "is this word in the vocab?" fast.

Sources may be either:
  - .json → parsed as a top-level array of strings, or
  - anything else (typically .txt) → one word per line, blanks and
    lines starting with '#' ignored so word lists can carry comments.

An empty or missing source contributes zero words rather than raising —
a fresh clone with only a base list should still work.
"""

from __future__ import annotations

import json
from pathlib import Path

from language_model.config import DictionaryConfig


class Dictionary:
    """Case-normalized vocab lookup backed by a single frozenset.

    Immutable after construction: no add/remove — reload from disk if
    the underlying word lists change. Callers that want per-source
    metadata (which source contributed a hit) should keep the source
    files around and query them individually rather than asking
    Dictionary for provenance.
    """

    def __init__(self, words: set[str], case_sensitive: bool) -> None:
        self._case_sensitive = case_sensitive
        self._words: frozenset[str] = frozenset(
            w if case_sensitive else w.lower() for w in words
        )

    @classmethod
    def from_config(cls, config: DictionaryConfig) -> Dictionary:
        words: set[str] = set()
        for path in (
            config.base_path_resolved,
            config.user_path_resolved,
            config.domain_path_resolved,
        ):
            if path is None:
                continue
            words.update(_load_word_list(path))
        return cls(words=words, case_sensitive=config.case_sensitive)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        key = word if self._case_sensitive else word.lower()
        return key in self._words

    def __len__(self) -> int:
        return len(self._words)

    @property
    def case_sensitive(self) -> bool:
        return self._case_sensitive

    def words(self) -> frozenset[str]:
        """The full vocab (normalized to lowercase if not case-sensitive)."""
        return self._words


def _load_word_list(path: Path) -> set[str]:
    """Read a word list from disk. Missing file → empty set (soft).

    Chose "missing → empty, malformed → raise" rather than "missing →
    raise" because user_path / domain_path frequently don't exist on
    a fresh clone; a schema error, on the other hand, is a real config
    problem the user should see immediately.
    """
    if not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"{path} must contain a top-level JSON array of strings"
            )
        return {str(w).strip() for w in data if str(w).strip()}
    with open(path, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }
