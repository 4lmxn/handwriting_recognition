"""Page layout / text-region detection (Phase 6, PR 3).

Takes a page ndarray (image or rasterized PDF page — same convention)
and returns a structured `PageLayout` describing where lines and words
sit on the page, ready for the batch-inference orchestrator in the
next PR to feed each word crop through the recognizer.

Composition, not new logic: binarize + optional deskew from
`preprocessing.image_ops`, then `segment_lines` + `segment_words` from
`segmentation/`. Kept here rather than folded into `segmentation/`
because the segmentation package operates on already-binary inputs by
design (see its module docstrings) — the "start from a raw page" glue
is Phase 6 pipeline concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from documents.config import LayoutConfig
from preprocessing.image_ops import adaptive_threshold, deskew
from segmentation.line_segmentation import segment_lines
from segmentation.word_segmentation import segment_words


@dataclass(frozen=True)
class WordBox:
    """Bounding box of a single word on the working page. Coordinates
    are inclusive-top / exclusive-bottom (numpy slice convention)."""
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass(frozen=True)
class LineLayout:
    top: int
    bottom: int
    words: tuple[WordBox, ...]


@dataclass(frozen=True)
class PageLayout:
    lines: tuple[LineLayout, ...]
    # Degrees the page was rotated to horizontal, 0.0 if deskew was
    # disabled or the page was already flat. Reported for diagnostics
    # / GUI overlays; downstream inference doesn't need it.
    skew_angle: float


def analyze_page(
    page: np.ndarray, config: LayoutConfig
) -> tuple[np.ndarray, PageLayout]:
    """Detect line + word regions on `page`.

    Returns `(working_page, layout)` where `working_page` is the
    possibly-deskewed grayscale ndarray whose coordinate system the
    layout's boxes refer to. Callers that need the original coordinates
    should set `config.deskew = False`; otherwise, crop from
    `working_page` rather than from the input.
    """
    if page.ndim != 2 or page.dtype != np.uint8:
        raise ValueError(
            "analyze_page expects a (H, W) uint8 grayscale ndarray "
            f"(got shape={page.shape} dtype={page.dtype})"
        )

    binary = adaptive_threshold(
        page, block_size=config.binarize_block_size, c=config.binarize_c
    )

    if config.deskew:
        # Deskew both binary (used for segmentation) and grayscale (the
        # working_page returned to the caller) using the same angle so
        # box coordinates line up with both. Rotate grayscale with a
        # white border since paper is light; binary with 0 since ink is
        # 255 in the binary convention.
        binary, angle = deskew(binary, border_value=0)
        if abs(angle) > 1e-3:
            working_page, _ = deskew(page, border_value=255)
        else:
            working_page = page
    else:
        angle = 0.0
        working_page = page

    line_regions = segment_lines(
        binary,
        min_line_height=config.min_line_height,
        min_gap=config.min_line_gap,
    )

    lines: list[LineLayout] = []
    for line in line_regions:
        line_binary = binary[line.top : line.bottom, :]
        word_regions = segment_words(
            line_binary,
            min_gap=config.min_word_gap,
            min_word_width=config.min_word_width,
        )
        # Word segmentation runs on the vertically-cropped line strip,
        # so its .left/.right values are already page-x-coordinates
        # (the horizontal axis wasn't touched by the crop). Only the
        # vertical top/bottom need lifting back onto the page.
        words = tuple(
            WordBox(
                top=line.top,
                bottom=line.bottom,
                left=w.left,
                right=w.right,
            )
            for w in word_regions
        )
        lines.append(LineLayout(top=line.top, bottom=line.bottom, words=words))

    return working_page, PageLayout(lines=tuple(lines), skew_angle=angle)
