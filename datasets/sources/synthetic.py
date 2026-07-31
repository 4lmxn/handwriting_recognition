"""Synthetic handwriting-adjacent dataset: procedurally rendered character and
word images using system fonts. Always available (no download, no license),
used to exercise the full pipeline end to end and to seed coverage of the
confusable classes described in docs/ROADMAP.md Phase 4. Rendered glyphs are
printed text, not genuine handwriting — real datasets (EMNIST, IAM, CVL, ...)
are what give the model real handwriting variation.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from datasets.config import SyntheticConfig
from datasets.manifest import DatasetSample, LabelType, Split
from datasets.sources.base import DatasetSource


def _existing_fonts(font_paths: list[str]) -> list[str]:
    return [p for p in font_paths if Path(p).exists()]


FontLike = ImageFont.FreeTypeFont | ImageFont.ImageFont


def _load_font(font_path: str | None, size: int) -> FontLike:
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(font_path, size)


def _render_text(
    text: str,
    font: FontLike,
    image_height: int,
    jitter: int,
    rng: random.Random,
) -> Image.Image:
    scratch = ImageDraw.Draw(Image.new("L", (10, 10), color=255))
    left, top, right, bottom = scratch.textbbox((0, 0), text, font=font)
    text_w, text_h = int(right - left), int(bottom - top)

    padding = max(4, image_height // 8)
    canvas_w = text_w + 2 * padding

    image = Image.new("L", (canvas_w, image_height), color=255)
    draw = ImageDraw.Draw(image)
    x = padding - left + rng.randint(-jitter, jitter)
    y = (image_height - text_h) // 2 - top + rng.randint(-jitter, jitter)
    draw.text((x, y), text, font=font, fill=0)
    return image


def _assign_split(key: str, split_ratios: dict[str, float]) -> Split:
    # Hash-based rather than the RNG stream, so re-running generation with the
    # same vocabulary keeps each item in the same split even if sample counts
    # or ordering change.
    digest = hashlib.sha256(key.encode()).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF

    cumulative = 0.0
    for split_name in ("train", "val", "test"):
        cumulative += split_ratios.get(split_name, 0.0)
        if fraction <= cumulative:
            return split_name  # type: ignore[return-value]
    return "train"


class SyntheticDatasetSource(DatasetSource):
    name = "synthetic"

    def __init__(self, config: SyntheticConfig, seed: int = 0) -> None:
        self._config = config
        self._rng = random.Random(seed)

    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        del raw_dir  # unused: synthetic data has no raw source files

        fonts = _existing_fonts(self._config.fonts)
        font_pool: list[str | None]
        if fonts:
            font_pool = list(fonts)
        else:
            font_pool = [None]

        vocabulary: list[tuple[str, LabelType]] = [
            (ch, "character") for ch in self._config.characters
        ] + [(word, "word") for word in self._config.words]

        samples: list[DatasetSample] = []

        for text, label_type in vocabulary:
            for i in range(self._config.samples_per_item):
                font_path = self._rng.choice(font_pool)
                font_size = self._rng.choice(self._config.font_sizes)
                font = _load_font(font_path, font_size)
                jitter = max(1, font_size // 12)
                image = _render_text(text, font, self._config.image_height, jitter, self._rng)

                split = _assign_split(f"{text}:{i}", self._config.split_ratios)

                safe_text = "".join(c if c.isalnum() else f"_{ord(c)}_" for c in text)
                relative_path = Path(self.name) / split / label_type / f"{safe_text}_{i}.png"
                full_path = processed_dir / relative_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(full_path)

                samples.append(
                    DatasetSample(
                        image_path=str(relative_path),
                        transcript=text,
                        source=self.name,
                        split=split,
                        label_type=label_type,
                        writer_id=None,
                    )
                )

        return samples
