"""Unit tests for language_model.rescoring (Phase 7, PR 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from language_model.config import RescoringConfig
from language_model.dictionary import Dictionary
from language_model.ngram import NGramLM
from language_model.rescoring import RescoringRecognizer, _levenshtein
from recognition.recognizer import RecognitionResult


def _blank_image() -> np.ndarray:
    return np.zeros((10, 10), dtype=np.uint8)


def _mock_base(single: RecognitionResult, topk: list[RecognitionResult]) -> MagicMock:
    base = MagicMock()
    base.recognize.return_value = single
    base.recognize_topk.return_value = topk
    return base


def _empty_dict() -> Dictionary:
    return Dictionary(words=set(), case_sensitive=False)


def _dict(words: set[str], case_sensitive: bool = False) -> Dictionary:
    return Dictionary(words=words, case_sensitive=case_sensitive)


def _fitted_ngram(words: list[str]) -> NGramLM:
    return NGramLM(n=3).fit(words)


# ---------- disabled → pure pass-through ---------------------------------


def test_disabled_config_skips_lm_and_returns_base_result():
    base = _mock_base(
        single=RecognitionResult(text="hello", confidence=0.9),
        topk=[],  # topk shouldn't even be called
    )
    r = RescoringRecognizer(
        base=base,
        ngram=NGramLM(),  # unfitted — would explode if called
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=False),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"
    base.recognize.assert_called_once()
    base.recognize_topk.assert_not_called()


# ---------- rescoring picks between candidates ---------------------------


def test_rescoring_prefers_higher_lm_score_when_confidences_equal():
    # Two candidates with identical model confidence — the LM must be
    # the tie-breaker, picking the one that looks like real language.
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.6),
        topk=[
            RecognitionResult(text="hzllo", confidence=0.6),
            RecognitionResult(text="hello", confidence=0.6),
        ],
    )
    ngram = _fitted_ngram(
        ["hello", "help", "held", "hero", "hemp", "hence", "heavy", "heart"]
    )
    r = RescoringRecognizer(
        base=base,
        ngram=ngram,
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=2, lm_weight=0.5),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"


def test_rescoring_at_lm_weight_zero_falls_back_to_model():
    # lm_weight=0 → combined score is pure model log-conf → highest
    # confidence wins regardless of LM signal.
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.9),
        topk=[
            RecognitionResult(text="hzllo", confidence=0.9),  # gibberish, high conf
            RecognitionResult(text="hello", confidence=0.6),  # real, low conf
        ],
    )
    ngram = _fitted_ngram(["hello"] * 20)
    r = RescoringRecognizer(
        base=base,
        ngram=ngram,
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=2, lm_weight=0.0),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hzllo"


def test_rescoring_at_lm_weight_one_ignores_model_confidence():
    # lm_weight=1 → pure LM. The LM should easily prefer the real
    # word over the mangled one.
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.9),
        topk=[
            RecognitionResult(text="hzllo", confidence=0.9),
            RecognitionResult(text="hello", confidence=0.1),
        ],
    )
    ngram = _fitted_ngram(["hello"] * 20)
    r = RescoringRecognizer(
        base=base,
        ngram=ngram,
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=2, lm_weight=1.0),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"


def test_empty_topk_returns_empty_result():
    base = _mock_base(
        single=RecognitionResult(text="anything", confidence=0.5),
        topk=[],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=NGramLM(),
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True),
    )
    result = r.recognize(_blank_image())
    assert result.text == ""
    assert result.confidence == 0.0


def test_zero_confidence_candidate_doesnt_crash_log():
    # A candidate with confidence=0 would send log(conf) to -inf and
    # dominate the combined score. The wrapper clamps confidence
    # above zero before taking the log so this stays bounded.
    base = _mock_base(
        single=RecognitionResult(text="", confidence=0.0),
        topk=[
            RecognitionResult(text="", confidence=0.0),
            RecognitionResult(text="hello", confidence=0.8),
        ],
    )
    ngram = _fitted_ngram(["hello"] * 10)
    r = RescoringRecognizer(
        base=base,
        ngram=ngram,
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=2, lm_weight=0.5),
    )
    result = r.recognize(_blank_image())
    # "hello" wins on both signals; "" is heavily penalized.
    assert result.text == "hello"


# ---------- dictionary snap ---------------------------------------------


def test_snap_disabled_at_threshold_zero():
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.9),
        topk=[RecognitionResult(text="hzllo", confidence=0.9)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_dict({"hello"}),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=0),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hzllo"  # unchanged


def test_snap_corrects_typo_within_threshold():
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.9),
        topk=[RecognitionResult(text="hzllo", confidence=0.9)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_dict({"hello"}),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=1),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"
    assert result.confidence == 0.9  # confidence preserved


def test_snap_leaves_word_alone_if_already_in_dictionary():
    base = _mock_base(
        single=RecognitionResult(text="hello", confidence=0.9),
        topk=[RecognitionResult(text="hello", confidence=0.9)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_dict({"hello", "help"}),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=2),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"


def test_snap_does_nothing_when_nothing_within_threshold():
    # Winner is 5 edits from any dict word; snap threshold is 1.
    base = _mock_base(
        single=RecognitionResult(text="qwxyz", confidence=0.5),
        topk=[RecognitionResult(text="qwxyz", confidence=0.5)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_dict({"hello"}),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=1),
    )
    result = r.recognize(_blank_image())
    assert result.text == "qwxyz"


def test_snap_skipped_when_dictionary_empty():
    # Even with snap_edit_distance > 0, an empty vocab has nothing to
    # snap to and must not raise or return None-as-text.
    base = _mock_base(
        single=RecognitionResult(text="hzllo", confidence=0.9),
        topk=[RecognitionResult(text="hzllo", confidence=0.9)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=3),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hzllo"


def test_snap_respects_case_insensitive_dictionary():
    base = _mock_base(
        single=RecognitionResult(text="HZLLO", confidence=0.9),
        topk=[RecognitionResult(text="HZLLO", confidence=0.9)],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=_fitted_ngram(["hello"]),
        dictionary=_dict({"hello"}, case_sensitive=False),
        config=RescoringConfig(enabled=True, topk=1, snap_edit_distance=1),
    )
    result = r.recognize(_blank_image())
    assert result.text == "hello"


# ---------- levenshtein helper -------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("hello", "hzllo", 1),
        ("cat", "hat", 1),
    ],
)
def test_levenshtein(a, b, expected):
    assert _levenshtein(a, b) == expected


# ---------- unfitted-lm graceful fallback --------------------------------


def test_unfitted_ngram_falls_back_to_model_scores():
    # If the LM was never fitted (fresh clone, empty dictionary), the
    # rescorer shouldn't crash — it should just use model log-conf.
    base = _mock_base(
        single=RecognitionResult(text="a", confidence=0.6),
        topk=[
            RecognitionResult(text="a", confidence=0.4),
            RecognitionResult(text="b", confidence=0.7),
        ],
    )
    r = RescoringRecognizer(
        base=base,
        ngram=NGramLM(),  # never .fit()
        dictionary=_empty_dict(),
        config=RescoringConfig(enabled=True, topk=2, lm_weight=0.5),
    )
    result = r.recognize(_blank_image())
    assert result.text == "b"  # highest model conf wins


# ---------- config loading -----------------------------------------------


def test_language_model_config_defaults_have_rescoring_disabled():
    from language_model.config import load_language_model_config

    config = load_language_model_config()
    assert config.rescoring.enabled is False
    assert config.rescoring.topk >= 1
    assert 0.0 <= config.rescoring.lm_weight <= 1.0
    assert config.rescoring.snap_edit_distance >= 0
    assert config.ngram.n >= 1


# ---------- wrap_if_enabled wiring helper (Phase 7 PR 5) ---------------


def test_wrap_if_enabled_returns_base_when_config_disables_rescoring():
    from unittest.mock import patch

    from language_model.config import (
        DictionaryConfig,
        LanguageModelConfig,
        NGramConfig,
    )
    from language_model.rescoring import wrap_if_enabled

    base = MagicMock(spec_set=["recognize", "recognize_topk"])
    disabled_config = LanguageModelConfig(
        dictionary=DictionaryConfig(base_path=None, user_path=None, domain_path=None),
        ngram=NGramConfig(),
        rescoring=RescoringConfig(enabled=False),
    )
    with patch(
        "language_model.rescoring.load_language_model_config",
        return_value=disabled_config,
    ):
        result = wrap_if_enabled(base)
    assert result is base  # exact object, no wrapping


def test_wrap_if_enabled_wraps_when_rescoring_enabled_even_with_empty_vocab(tmp_path):
    # A user could legitimately turn rescoring on before adding any
    # dictionary sources — the wrapper must still return a
    # RescoringRecognizer (fallback behavior lives inside it), not
    # crash on the empty vocab.
    from unittest.mock import patch

    from language_model.config import (
        DictionaryConfig,
        LanguageModelConfig,
        NGramConfig,
    )
    from language_model.rescoring import RescoringRecognizer, wrap_if_enabled

    base = MagicMock(spec_set=["recognize", "recognize_topk"])
    enabled_config = LanguageModelConfig(
        dictionary=DictionaryConfig(base_path=None, user_path=None, domain_path=None),
        ngram=NGramConfig(),
        rescoring=RescoringConfig(enabled=True, topk=3, lm_weight=0.3),
    )
    with patch(
        "language_model.rescoring.load_language_model_config",
        return_value=enabled_config,
    ):
        result = wrap_if_enabled(base)
    assert isinstance(result, RescoringRecognizer)


def test_wrap_if_enabled_fits_ngram_when_vocab_present(tmp_path):
    from unittest.mock import patch

    from language_model.config import (
        DictionaryConfig,
        LanguageModelConfig,
        NGramConfig,
    )
    from language_model.rescoring import wrap_if_enabled

    words_path = tmp_path / "vocab.txt"
    words_path.write_text("hello\nworld\n", encoding="utf-8")
    with patch("language_model.config.REPO_ROOT", tmp_path):
        enabled_config = LanguageModelConfig(
            dictionary=DictionaryConfig(
                base_path="vocab.txt", user_path=None, domain_path=None
            ),
            ngram=NGramConfig(n=3),
            rescoring=RescoringConfig(enabled=True, topk=3, lm_weight=0.3),
        )
        base = MagicMock(spec_set=["recognize", "recognize_topk"])
        base.recognize_topk.return_value = [
            RecognitionResult(text="hello", confidence=0.5),
            RecognitionResult(text="xyzab", confidence=0.5),
        ]
        with patch(
            "language_model.rescoring.load_language_model_config",
            return_value=enabled_config,
        ):
            wrapper = wrap_if_enabled(base)
        # Round-trip through recognize() to force the LM to actually
        # score — if fit didn't happen this would fall back to model
        # confidence only (both = 0.5) and pick arbitrarily; with a
        # fitted LM trained on ["hello", "world"], "hello" wins.
        result = wrapper.recognize(np.zeros((10, 10), dtype=np.uint8))
        assert result.text == "hello"
