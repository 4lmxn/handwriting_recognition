"""Character segmentation within a word, via connected components.

Connected-component segmentation only works for non-touching glyphs — cursive
or connected handwriting will merge multiple characters into one component.
This is expected: the recognition pipeline (Phase 3+) uses a CTC/attention
decoder over whole word or line images and does not depend on character-level
segmentation to function. This module exists for the GUI's optional character
segmentation visualization, not as a recognition prerequisite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.image_ops import connected_components


@dataclass(frozen=True)
class CharacterBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def segment_characters(word_image: np.ndarray, min_component_area: int = 10) -> list[CharacterBox]:
    num_labels, _labels, stats, _centroids = connected_components(word_image)

    boxes = []
    for label in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[label]
        if area < min_component_area:
            continue
        boxes.append(CharacterBox(x=int(x), y=int(y), width=int(w), height=int(h)))

    return sorted(boxes, key=lambda box: box.x)
