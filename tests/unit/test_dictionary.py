"""Unit tests for language_model.dictionary (Phase 7, PR 1)."""

from __future__ import annotations

import json

import pytest

from language_model.config import DictionaryConfig, load_language_model_config
from language_model.dictionary import Dictionary


def _config(**paths) -> DictionaryConfig:
    return DictionaryConfig(
        base_path=paths.get("base"),
        user_path=paths.get("user"),
        domain_path=paths.get("domain"),
        case_sensitive=paths.get("case_sensitive", False),
    )


def test_dictionary_empty_when_all_paths_are_none():
    d = Dictionary.from_config(_config())
    assert len(d) == 0
    assert "anything" not in d


def test_dictionary_reads_plain_text_word_list(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(_config(base=path.name))

    assert len(d) == 3
    assert "alpha" in d
    assert "delta" not in d


def test_dictionary_reads_json_word_list(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(["alpha", "beta", "gamma"]), encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(_config(base=path.name))

    assert "alpha" in d
    assert "gamma" in d


def test_dictionary_merges_all_three_sources(tmp_path):
    (tmp_path / "b.txt").write_text("apple\n", encoding="utf-8")
    (tmp_path / "u.txt").write_text("banana\n", encoding="utf-8")
    (tmp_path / "d.txt").write_text("cherry\n", encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(
            _config(base="b.txt", user="u.txt", domain="d.txt")
        )

    assert len(d) == 3
    assert "apple" in d and "banana" in d and "cherry" in d


def test_dictionary_ignores_missing_source_files(tmp_path):
    # user_path may legitimately not exist yet on a fresh clone —
    # loading must not raise. Only genuine schema errors should surface.
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(_config(user="never_written.txt"))

    assert len(d) == 0


def test_dictionary_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text(
        "# preamble comment\n"
        "apple\n"
        "\n"
        "   \n"
        "  # indented comment (also skipped once stripped)\n"
        "banana\n",
        encoding="utf-8",
    )
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(_config(base=path.name))

    assert len(d) == 2
    assert "apple" in d and "banana" in d


def test_dictionary_case_insensitive_by_default(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("Apple\nBanana\n", encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(_config(base=path.name))

    assert "apple" in d
    assert "APPLE" in d
    assert "Apple" in d


def test_dictionary_case_sensitive_when_enabled(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("Apple\nBanana\n", encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        d = Dictionary.from_config(
            _config(base=path.name, case_sensitive=True)
        )

    assert "Apple" in d
    assert "apple" not in d


def test_dictionary_rejects_malformed_json(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    from unittest.mock import patch

    with patch("language_model.config.REPO_ROOT", tmp_path):
        with pytest.raises(ValueError):
            Dictionary.from_config(_config(base=path.name))


def test_dictionary_contains_rejects_non_str():
    d = Dictionary(words={"foo"}, case_sensitive=False)
    assert 42 not in d  # type: ignore[operator]
    assert None not in d  # type: ignore[operator]


def test_load_language_model_config_reads_yaml():
    config = load_language_model_config()
    # Defaults ship with all three paths as null so a fresh clone works.
    assert config.dictionary.base_path is None
    assert config.dictionary.user_path is None
    assert config.dictionary.domain_path is None
    assert config.dictionary.case_sensitive is False


def test_dictionary_config_path_resolution_relative_to_repo_root():
    cfg = DictionaryConfig(
        base_path="datasets/vocabularies/base.txt",
        user_path=None,
        domain_path=None,
    )
    resolved = cfg.base_path_resolved
    assert resolved is not None
    assert resolved.name == "base.txt"


def test_dictionary_config_none_and_empty_string_both_resolve_to_none():
    cfg = DictionaryConfig(base_path=None, user_path="", domain_path=None)
    assert cfg.base_path_resolved is None
    assert cfg.user_path_resolved is None
