"""Text line segmentation via horizontal projection profile.

This is a classical, non-learned segmentation step — fast, dependency-free, and
good enough to hand line crops to a recognition model. It struggles on strongly
skewed or overlapping-line handwriting; preprocessing.image_ops.deskew should be
applied first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.image_ops import horizontal_projection_profile


@dataclass(frozen=True)
class LineRegion:
    top: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top


def segment_lines(
    binary_image: np.ndarray, min_line_height: int = 5, min_gap: int = 3
) -> list[LineRegion]:
    """Ink rows (profile > 0) are grouped into bands; gaps of fewer than
    `min_gap` blank rows are merged into the same band (handles rows with a
    stray dot from an i/j or noise). Bands shorter than `min_line_height` are
    dropped as noise."""
    profile = horizontal_projection_profile(binary_image)
    ink_rows = profile > 0

    raw_bands: list[list[int]] = []
    row = 0
    height = len(ink_rows)
    while row < height:
        if ink_rows[row]:
            start = row
            while row < height and ink_rows[row]:
                row += 1
            raw_bands.append([start, row])
        else:
            row += 1

    merged_bands: list[list[int]] = []
    for band in raw_bands:
        if merged_bands and band[0] - merged_bands[-1][1] < min_gap:
            merged_bands[-1][1] = band[1]
        else:
            merged_bands.append(band)

    return [
        LineRegion(top=start, bottom=end)
        for start, end in merged_bands
        if (end - start) >= min_line_height
    ]
