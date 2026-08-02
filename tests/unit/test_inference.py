"""Unit tests for documents.inference (Phase 6, PR 4).

The real Recognizer loads a heavy transformer model; here it's
replaced with a MagicMock whose .recognize() returns deterministic
RecognitionResults. That lets the orchestrator's sequencing, cropping,
and result-tree shape be verified in milliseconds without touching
torch or the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from documents.inference import (
    PageResult,
    RecognizedLine,
    RecognizedWord,
    recognize_page,
)
from documents.layout import LineLayout, PageLayout, WordBox
from recognition.recognizer import RecognitionResult


def _make_recognizer(results: list[RecognitionResult]) -> MagicMock:
    recognizer = MagicMock()
    recognizer.recognize.side_effect = results
    return recognizer


def _blank_page(h: int = 100, w: int = 200) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def test_recognize_page_produces_result_tree_matching_layout():
    page = _blank_page()
    layout = PageLayout(
        lines=(
            LineLayout(
                top=10,
                bottom=40,
                words=(
                    WordBox(top=10, bottom=40, left=5, right=30),
                    WordBox(top=10, bottom=40, left=50, right=90),
                ),
            ),
            LineLayout(
                top=60,
                bottom=90,
                words=(WordBox(top=60, bottom=90, left=20, right=100),),
            ),
        ),
        skew_angle=0.0,
    )
    recognizer = _make_recognizer(
        [
            RecognitionResult(text="hello", confidence=0.9),
            RecognitionResult(text="world", confidence=0.8),
            RecognitionResult(text="foo", confidence=0.7),
        ]
    )

    result = recognize_page(page, layout, recognizer)

    assert isinstance(result, PageResult)
    assert len(result.lines) == 2
    assert len(result.lines[0].words) == 2
    assert result.lines[0].words[0].text == "hello"
    assert result.lines[0].words[0].confidence == 0.9
    assert result.lines[0].words[1].text == "world"
    assert result.lines[1].words[0].text == "foo"


def test_recognize_page_crops_working_page_by_word_box():
    # Paint distinct intensity blocks so we can verify the crop passed
    # to recognizer.recognize came from the right box.
    page = _blank_page(h=100, w=200)
    page[10:40, 5:30] = 111
    page[10:40, 50:90] = 222

    layout = PageLayout(
        lines=(
            LineLayout(
                top=10,
                bottom=40,
                words=(
                    WordBox(top=10, bottom=40, left=5, right=30),
                    WordBox(top=10, bottom=40, left=50, right=90),
                ),
            ),
        ),
        skew_angle=0.0,
    )
    recognizer = _make_recognizer(
        [
            RecognitionResult(text="a", confidence=1.0),
            RecognitionResult(text="b", confidence=1.0),
        ]
    )

    recognize_page(page, layout, recognizer)

    call_args = recognizer.recognize.call_args_list
    assert len(call_args) == 2
    first_crop = call_args[0].args[0]
    second_crop = call_args[1].args[0]
    assert first_crop.shape == (30, 25)
    assert second_crop.shape == (30, 40)
    assert first_crop.mean() == 111
    assert second_crop.mean() == 222


def test_recognize_page_empty_layout_returns_empty_result():
    page = _blank_page()
    layout = PageLayout(lines=(), skew_angle=0.0)
    recognizer = _make_recognizer([])

    result = recognize_page(page, layout, recognizer)

    assert result.lines == ()
    assert result.skew_angle == 0.0
    recognizer.recognize.assert_not_called()


def test_recognize_page_propagates_skew_angle():
    layout = PageLayout(lines=(), skew_angle=1.75)
    recognizer = _make_recognizer([])
    result = recognize_page(_blank_page(), layout, recognizer)
    assert result.skew_angle == 1.75


def test_recognized_line_text_joins_words_with_space():
    line = RecognizedLine(
        top=0,
        bottom=10,
        words=(
            RecognizedWord(text="the", confidence=1.0, box=WordBox(0, 10, 0, 20)),
            RecognizedWord(text="quick", confidence=1.0, box=WordBox(0, 10, 25, 60)),
            RecognizedWord(text="fox", confidence=1.0, box=WordBox(0, 10, 65, 90)),
        ),
    )
    assert line.text == "the quick fox"


def test_page_result_text_joins_lines_with_newline():
    def _line(word: str) -> RecognizedLine:
        return RecognizedLine(
            top=0,
            bottom=10,
            words=(
                RecognizedWord(text=word, confidence=1.0, box=WordBox(0, 10, 0, 20)),
            ),
        )

    result = PageResult(lines=(_line("hello"), _line("world")), skew_angle=0.0)
    assert result.text == "hello\nworld"
