"""GUI tests for PagePreviewWidget (Phase 6, PR 5)."""

from __future__ import annotations

import numpy as np
import pytest

from app.gui.widgets.page_preview_widget import PagePreviewWidget
from documents.layout import LineLayout, PageLayout, WordBox


def _blank_page(h: int = 100, w: int = 200) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def test_widget_starts_empty(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    assert not w.has_page()


def test_set_page_marks_widget_populated(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    w.set_page(_blank_page())
    assert w.has_page()


def test_set_page_rejects_non_grayscale(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        w.set_page(color)


def test_set_page_rejects_wrong_dtype(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    with pytest.raises(ValueError):
        w.set_page(np.zeros((10, 10), dtype=np.float32))


def test_set_page_accepts_non_contiguous_slice(qtbot):
    # A non-contiguous slice of a page must still render without
    # reading garbage rows — the widget copies to a contiguous
    # buffer internally.
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    parent = np.arange(200 * 100, dtype=np.uint8).reshape(200, 100)
    slice_view = parent[::2, ::2]
    assert not slice_view.flags["C_CONTIGUOUS"]
    w.set_page(slice_view)
    assert w.has_page()


def test_set_page_clears_previous_layout(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    layout = PageLayout(
        lines=(LineLayout(top=0, bottom=10, words=(WordBox(0, 10, 0, 20),)),),
        skew_angle=0.0,
    )
    w.set_page(_blank_page())
    w.set_layout(layout)
    # Loading a new page must retire the old overlay so it doesn't
    # get drawn in the wrong coordinate system.
    w.set_page(_blank_page(h=50, w=50))
    assert w._layout is None


def test_clear_empties_widget(qtbot):
    w = PagePreviewWidget()
    qtbot.addWidget(w)
    w.set_page(_blank_page())
    w.clear()
    assert not w.has_page()
