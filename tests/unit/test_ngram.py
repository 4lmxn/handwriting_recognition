"""Unit tests for language_model.ngram (Phase 7, PR 2)."""

from __future__ import annotations

import math

import pytest

from language_model.ngram import NGramLM


def test_constructor_rejects_invalid_n():
    with pytest.raises(ValueError):
        NGramLM(n=0)
    with pytest.raises(ValueError):
        NGramLM(n=-1)


def test_constructor_rejects_invalid_smoothing():
    with pytest.raises(ValueError):
        NGramLM(smoothing_k=0)
    with pytest.raises(ValueError):
        NGramLM(smoothing_k=-0.5)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        NGramLM().score("anything")


def test_fit_returns_self_for_chaining():
    lm = NGramLM()
    assert lm.fit(["hello"]) is lm
    assert lm.is_fitted


def test_empty_text_scores_zero():
    lm = NGramLM().fit(["hello"])
    assert lm.score("") == 0.0


def test_score_returns_finite_log_prob_for_unseen_string():
    # Unseen n-grams must not send the score to -inf — that's the
    # whole point of Laplace smoothing.
    lm = NGramLM(n=3).fit(["abc"])
    score = lm.score("xyz")
    assert math.isfinite(score)
    assert score < 0.0  # log-prob


def test_frequent_string_scores_higher_than_rare_string():
    # Train mostly on words containing "the" so that "the" should
    # dominate over an unseen character combination.
    lm = NGramLM(n=3).fit(
        ["the", "them", "there", "then", "theme", "these", "their"]
    )
    common = lm.score("the")
    unseen = lm.score("qzx")
    assert common > unseen


def test_dictionary_word_scores_higher_than_typo():
    lm = NGramLM(n=3).fit(
        [
            "hello",
            "world",
            "help",
            "held",
            "hero",
            "helm",
            "here",
            "herd",
        ]
    )
    real = lm.score("hello")
    typo = lm.score("hzllo")
    assert real > typo


def test_vocab_size_counts_unique_characters_plus_end_token():
    lm = NGramLM(n=2).fit(["abc"])
    # Unique chars: a, b, c. Plus the end token. Start token doesn't
    # count — it never appears as a target.
    assert lm.vocab_size == 4


def test_n_property_reflects_constructor():
    assert NGramLM(n=4).n == 4


def test_refit_replaces_previous_training():
    lm = NGramLM(n=3).fit(["hello"])
    first_score = lm.score("hello")
    lm.fit(["world"])
    # Vocab now includes "world" chars, "hello" chars are gone → the
    # score for "hello" must change (specifically, drop, since h/e/l/o
    # are no longer in the vocab counts and only smoothing carries them).
    second_score = lm.score("hello")
    assert first_score != second_score


def test_empty_words_in_training_are_skipped():
    lm = NGramLM(n=2).fit(["hi", "", "yo"])
    # Should not raise — empty words are legitimately empty (e.g. from
    # a comment-stripped word list) and must be silently skipped.
    assert lm.is_fitted
    assert math.isfinite(lm.score("hi"))


def test_gibberish_scores_below_training_word_of_same_length():
    # Same-length comparison controls for the length penalty of a
    # product-of-per-position-probabilities model.
    lm = NGramLM(n=3).fit(
        ["hello", "world", "hell", "held", "help", "helm", "here"]
    )
    assert lm.score("hello") > lm.score("qzxvb")
