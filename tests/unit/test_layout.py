"""Unit tests for documents.layout (Phase 6, PR 3)."""

from __future__ import annotations

import numpy as np
import pytest

from documents.config import LayoutConfig
from documents.layout import analyze_page


def _blank_page(h: int = 200, w: int = 400) -> np.ndarray:
    # ink-dark-on-light-background convention: paper is 255, ink is 0.
    # Matches what documents.loader.load_image / pdf_loader.load_pdf_pages
    # both produce.
    return np.full((h, w), 255, dtype=np.uint8)


def _paint_word(page: np.ndarray, top: int, bottom: int, left: int, right: int) -> None:
    page[top:bottom, left:right] = 0


def _make_config(**overrides) -> LayoutConfig:
    base = dict(
        deskew=False,  # tests default to deskew off to keep coordinates
        # exactly predictable — the deskew branch is exercised by its own test
        binarize_block_size=35,
        binarize_c=11,
        min_line_height=5,
        min_line_gap=3,
        min_word_gap=8,
        min_word_width=3,
    )
    base.update(overrides)
    return LayoutConfig(**base)


def test_analyze_page_returns_expected_lines_and_words():
    page = _blank_page(h=200, w=400)
    # Line 1: two words
    _paint_word(page, top=20, bottom=45, left=20, right=90)
    _paint_word(page, top=20, bottom=45, left=150, right=220)
    # Line 2: one word
    _paint_word(page, top=100, bottom=125, left=50, right=180)

    working, layout = analyze_page(page, _make_config())

    assert working.shape == page.shape
    assert len(layout.lines) == 2
    assert len(layout.lines[0].words) == 2
    assert len(layout.lines[1].words) == 1
    # Rough coordinate sanity: word 1's left edge is where we painted it.
    assert layout.lines[0].words[0].left == 20
    # Line vertical bounds cover the painted region (may extend by ±1 pixel
    # due to adaptive threshold's neighborhood).
    assert layout.lines[0].top <= 25 and layout.lines[0].bottom >= 40


def test_analyze_page_empty_for_blank_page():
    page = _blank_page()
    working, layout = analyze_page(page, _make_config())
    assert layout.lines == ()
    assert layout.skew_angle == 0.0
    assert working.shape == page.shape


def test_analyze_page_word_boxes_expose_page_coordinates():
    page = _blank_page(h=100, w=300)
    _paint_word(page, top=30, bottom=60, left=40, right=120)

    _, layout = analyze_page(page, _make_config())

    word = layout.lines[0].words[0]
    # Vertical bounds come from the line (top/bottom apply line-wide);
    # horizontal bounds are the word's own extent on the page x-axis.
    assert word.top == layout.lines[0].top
    assert word.bottom == layout.lines[0].bottom
    assert word.left == 40
    assert word.right == 120
    assert word.width == 80


def test_analyze_page_rejects_non_grayscale_input():
    color = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        analyze_page(color, _make_config())


def test_analyze_page_rejects_wrong_dtype():
    page = np.full((100, 100), 1.0, dtype=np.float32)
    with pytest.raises(ValueError):
        analyze_page(page, _make_config())


def test_analyze_page_deskew_disabled_reports_zero_angle():
    # A slightly-skewed painted line — with deskew off, the layout still
    # runs (segmentation just gives a taller merged band), and the
    # reported angle is 0 regardless.
    page = _blank_page(h=200, w=400)
    for i in range(60):
        page[30 + i // 10, 20 + i * 5 : 30 + i * 5] = 0

    _, layout = analyze_page(page, _make_config(deskew=False))

    assert layout.skew_angle == 0.0


def test_analyze_page_returns_working_page_on_deskew_path():
    # Deskew turned on — the working_page comes back the same shape as
    # the input (deskew keeps canvas size) and the skew angle is reported.
    page = _blank_page(h=200, w=400)
    _paint_word(page, top=40, bottom=70, left=20, right=200)

    working, layout = analyze_page(page, _make_config(deskew=True))

    assert working.shape == page.shape
    # For a horizontal painted rectangle skew_angle should be ~0.
    assert abs(layout.skew_angle) < 5.0


def test_analyze_page_config_word_gap_controls_merging():
    page = _blank_page(h=100, w=400)
    _paint_word(page, top=30, bottom=60, left=20, right=90)
    _paint_word(page, top=30, bottom=60, left=100, right=170)  # gap of 10 px

    # min_word_gap=8 (< 10): two words.
    _, layout_split = analyze_page(page, _make_config(min_word_gap=8))
    assert len(layout_split.lines[0].words) == 2

    # min_word_gap=20 (> 10): merges into one.
    _, layout_merged = analyze_page(page, _make_config(min_word_gap=20))
    assert len(layout_merged.lines[0].words) == 1
