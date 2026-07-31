"""MNIST digit dataset (0-9), 28x28 grayscale, auto-downloaded — no
registration required. See datasets/registry.py for provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from datasets.config import MnistConfig
from datasets.manifest import DatasetSample, Split
from datasets.sources.base import DatasetSource
from datasets.sources.idx_utils import download, read_idx_gzip


class MnistDatasetSource(DatasetSource):
    name = "mnist"

    def __init__(
        self,
        config: MnistConfig,
        val_fraction: float = 0.1,
        max_samples: int | None = None,
    ) -> None:
        self._config = config
        self._val_fraction = val_fraction
        self._max_samples = max_samples

    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        mnist_raw = raw_dir / self.name
        raw_files = {
            "train_images": self._config.train_images,
            "train_labels": self._config.train_labels,
            "test_images": self._config.test_images,
            "test_labels": self._config.test_labels,
        }
        local_paths: dict[str, Path] = {}
        for key, filename in raw_files.items():
            dest = mnist_raw / filename
            download(self._config.base_url + filename, dest)
            local_paths[key] = dest

        train_images = read_idx_gzip(local_paths["train_images"])
        train_labels = read_idx_gzip(local_paths["train_labels"])
        test_images = read_idx_gzip(local_paths["test_images"])
        test_labels = read_idx_gzip(local_paths["test_labels"])

        samples = self._write_split(train_images, train_labels, processed_dir, has_val=True)
        samples += self._write_split(test_images, test_labels, processed_dir, has_val=False)
        return samples

    def _write_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        processed_dir: Path,
        has_val: bool,
    ) -> list[DatasetSample]:
        n = len(images)
        if self._max_samples is not None:
            n = min(n, self._max_samples)
        val_cutoff = int(n * (1 - self._val_fraction)) if has_val else n

        samples = []
        for i in range(n):
            split: Split = "test"
            if has_val:
                split = "train" if i < val_cutoff else "val"

            # MNIST stores ink as high pixel values on a dark background; invert
            # so ink is dark on a light background, matching every other source.
            image = Image.fromarray((255 - images[i]).astype(np.uint8))
            relative_path = Path(self.name) / split / "character" / f"{labels[i]}_{i}.png"
            full_path = processed_dir / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(full_path)

            samples.append(
                DatasetSample(
                    image_path=str(relative_path),
                    transcript=str(int(labels[i])),
                    source=self.name,
                    split=split,
                    label_type="character",
                    writer_id=None,
                )
            )
        return samples
