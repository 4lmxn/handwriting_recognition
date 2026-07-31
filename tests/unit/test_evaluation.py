import pytest

from training.evaluation import (
    EvaluationResult,
    character_accuracy,
    character_error_rate,
    evaluate_predictions,
    exact_match,
    word_accuracy,
    word_error_rate,
)


def test_identical_strings_have_zero_error():
    assert character_error_rate("hello world", "hello world") == 0.0
    assert word_error_rate("hello world", "hello world") == 0.0
    assert character_accuracy("hello world", "hello world") == 1.0
    assert word_accuracy("hello world", "hello world") == 1.0
    assert exact_match("hello world", "hello world") is True


def test_completely_different_strings():
    assert character_error_rate("abc", "xyz") == 1.0
    assert exact_match("abc", "xyz") is False


def test_empty_reference_and_empty_hypothesis():
    assert character_error_rate("", "") == 0.0
    assert word_error_rate("", "") == 0.0
    assert exact_match("", "") is True


def test_empty_reference_with_nonempty_hypothesis():
    assert character_error_rate("", "abc") == 1.0
    assert word_error_rate("", "hello world") == 1.0


def test_substitution_case():
    # cat -> cot: single substitution, distance 1 over 3 reference chars
    assert character_error_rate("cat", "cot") == pytest.approx(1 / 3)


def test_insertion_case():
    # cat -> cats: single insertion, distance 1 over 3 reference chars
    assert character_error_rate("cat", "cats") == pytest.approx(1 / 3)


def test_deletion_case():
    # cats -> cat: single deletion, distance 1 over 4 reference chars
    assert character_error_rate("cats", "cat") == pytest.approx(1 / 4)


def test_word_error_rate_substitution():
    # "the cat sat" -> "the dog sat": one word substituted, distance 1 over 3 words
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_word_error_rate_insertion():
    # "the cat sat" -> "the cat sat down": one word inserted, distance 1 over 3 words
    assert word_error_rate("the cat sat", "the cat sat down") == pytest.approx(1 / 3)


def test_character_accuracy_clamped_when_hypothesis_much_longer():
    # reference="a", hypothesis="abcdef": distance 5, CER = 5.0, would give
    # accuracy -4.0 unclamped
    assert character_error_rate("a", "abcdef") == 5.0
    assert character_accuracy("a", "abcdef") == 0.0


def test_word_accuracy_clamped_when_hypothesis_much_longer():
    assert word_accuracy("a", "a b c d e f") == 0.0


def test_evaluate_predictions_averages_across_samples():
    references = ["cat", "the cat sat"]
    hypotheses = ["cat", "the dog sat"]

    result = evaluate_predictions(references, hypotheses)

    assert isinstance(result, EvaluationResult)
    assert result.num_samples == 2
    # sample 1: CER 0.0, sample 2: CER 0.0 (no char difference in "the ... sat")
    # actually char-level: "the cat sat" -> "the dog sat" has 3 substitutions (c/d, a/o, t/g)
    expected_cer = (0.0 + 3 / 11) / 2
    assert result.mean_cer == pytest.approx(expected_cer)
    expected_wer = (0.0 + 1 / 3) / 2
    assert result.mean_wer == pytest.approx(expected_wer)
    assert result.mean_char_accuracy == pytest.approx(1 - expected_cer)
    assert result.mean_word_accuracy == pytest.approx(1 - expected_wer)
    assert result.exact_match_rate == pytest.approx(0.5)


def test_evaluate_predictions_exact_match_rate():
    references = ["abc", "def", "ghi"]
    hypotheses = ["abc", "xyz", "ghi"]

    result = evaluate_predictions(references, hypotheses)

    assert result.exact_match_rate == pytest.approx(2 / 3)


def test_evaluate_predictions_raises_on_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_predictions(["a", "b"], ["a"])


def test_evaluate_predictions_raises_on_empty_lists():
    with pytest.raises(ValueError):
        evaluate_predictions([], [])
