"""CVL Database — requires manual download (free, but requires accepting the
provider's terms; see datasets/registry.py). Expects the extracted archive
under raw_dir/cvl/cvl-database-1-1/ as:
    raw_dir/cvl/cvl-database-1-1/{trainset,testset}/{words,lines}/.../<file>.png

CVL embeds the ground-truth transcription directly in each image filename —
no separate XML/txt annotation file — as the last '-'-separated segment
before the extension. This parsing logic (including skipping files whose
second segment is '6', and skipping labels containing German umlauts) mirrors
the reference implementation in amzn/convolutional-handwriting-gan
(data/create_text_data.py), verified against that published source rather
than guessed.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from datasets.manifest import DatasetSample, LabelType, Split
from datasets.sources.base import DatasetSource

_MODE_SPLITS: list[tuple[str, Split]] = [("trainset", "train"), ("testset", "test")]


def extract_label(filename: str) -> str | None:
    """Returns None for filenames CVL's own convention (and the reference
    parser above) excludes: the '-6-' sample marker, umlauts, or empty labels."""
    segments = Path(filename).stem.split("-")
    if len(segments) < 2:
        return None
    if segments[1] == "6":
        return None
    label = segments[-1]
    if not label or "ä" in label or "ü" in label:
        return None
    return label


class CvlDatasetSource(DatasetSource):
    name = "cvl"

    def __init__(self, use_words: bool = True) -> None:
        self._use_words = use_words

    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        cvl_root = raw_dir / self.name / "cvl-database-1-1"
        if not cvl_root.exists():
            raise FileNotFoundError(
                f"{cvl_root} not found. See datasets/registry.py for CVL acquisition "
                "instructions — this dataset requires a manual download."
            )

        images_subdir = "words" if self._use_words else "lines"
        label_type: LabelType = "word" if self._use_words else "line"

        samples = []
        for mode_dir, split in _MODE_SPLITS:
            images_dir = cvl_root / mode_dir / images_subdir
            if not images_dir.exists():
                continue
            for source_image_path in sorted(images_dir.rglob("*")):
                if not source_image_path.is_file():
                    continue
                label = extract_label(source_image_path.name)
                if label is None:
                    continue

                relative_path = Path(self.name) / split / label_type / source_image_path.name
                full_path = processed_dir / relative_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                image = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                cv2.imwrite(str(full_path), image)

                samples.append(
                    DatasetSample(
                        image_path=str(relative_path),
                        transcript=label,
                        source=self.name,
                        split=split,
                        label_type=label_type,
                        writer_id=None,
                    )
                )
        return samples
