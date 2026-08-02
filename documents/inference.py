"""Batch inference over a laid-out page (Phase 6, PR 4).

Takes the `(working_page, PageLayout)` returned by
`documents.layout.analyze_page` and runs each word crop through a
Recognizer, emitting a parallel `PageResult` tree that pairs each
word box with the recognized text + confidence.

Iteration is sequential — one recognizer.recognize() call per word —
mirroring the drawing-tab pipeline. True model-level batching (a single
`generate()` call over all word crops on the page) is a worthwhile
follow-up but requires widening the Recognizer API, so it lives in its
own PR rather than being smuggled in here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from documents.layout import PageLayout, WordBox
from recognition.recognizer import Recognizer


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    confidence: float
    box: WordBox


@dataclass(frozen=True)
class RecognizedLine:
    top: int
    bottom: int
    words: tuple[RecognizedWord, ...]

    @property
    def text(self) -> str:
        """Space-joined word transcripts — the human-readable line."""
        return " ".join(w.text for w in self.words)


@dataclass(frozen=True)
class PageResult:
    lines: tuple[RecognizedLine, ...]
    # Carried through from PageLayout so the caller can distinguish
    # "flat page" from "we rotated by X degrees to get here" when
    # rendering results back over the original scan.
    skew_angle: float

    @property
    def text(self) -> str:
        """Newline-joined line transcripts — the human-readable page."""
        return "\n".join(line.text for line in self.lines)


def recognize_page(
    working_page: np.ndarray, layout: PageLayout, recognizer: Recognizer
) -> PageResult:
    """Run the recognizer over every word box in `layout`.

    `working_page` must be the array returned alongside `layout` by
    `analyze_page` — the box coordinates are relative to it, not to the
    original scan.
    """
    lines: list[RecognizedLine] = []
    for line in layout.lines:
        recognized_words: list[RecognizedWord] = []
        for box in line.words:
            crop = working_page[box.top : box.bottom, box.left : box.right]
            result = recognizer.recognize(crop)
            recognized_words.append(
                RecognizedWord(
                    text=result.text,
                    confidence=result.confidence,
                    box=box,
                )
            )
        lines.append(
            RecognizedLine(
                top=line.top,
                bottom=line.bottom,
                words=tuple(recognized_words),
            )
        )
    return PageResult(lines=tuple(lines), skew_angle=layout.skew_angle)
