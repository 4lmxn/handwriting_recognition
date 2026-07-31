"""Word segmentation within a single text line, via vertical projection profile.

The gap threshold is the key parameter: small gaps are inter-letter spacing,
larger gaps are inter-word spacing. `min_gap` needs tuning per dataset/writer
(see docs/ROADMAP.md Phase 4 for confusion analysis feeding parameter tuning).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.image_ops import vertical_projection_profile


@dataclass(frozen=True)
class WordRegion:
    left: int
    right: int

    @property
    def width(self) -> int:
        return self.right - self.left


def segment_words(
    line_image: np.ndarray, min_gap: int = 8, min_word_width: int = 3
) -> list[WordRegion]:
    profile = vertical_projection_profile(line_image)
    ink_cols = profile > 0

    raw_bands: list[list[int]] = []
    col = 0
    width = len(ink_cols)
    while col < width:
        if ink_cols[col]:
            start = col
            while col < width and ink_cols[col]:
                col += 1
            raw_bands.append([start, col])
        else:
            col += 1

    merged_bands: list[list[int]] = []
    for band in raw_bands:
        if merged_bands and band[0] - merged_bands[-1][1] < min_gap:
            merged_bands[-1][1] = band[1]
        else:
            merged_bands.append(band)

    return [
        WordRegion(left=start, right=end)
        for start, end in merged_bands
        if (end - start) >= min_word_width
    ]
