"""EMNIST (Extended MNIST): digits + uppercase + lowercase letters, 28x28
grayscale. Auto-downloaded from NIST — no registration required, but the
archive is ~550MB, so this only runs when explicitly requested (see
scripts/prepare_dataset.py) rather than as a side effect of anything else.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.config import EmnistConfig
from datasets.manifest import DatasetSample, Split
from datasets.sources.base import DatasetSource
from datasets.sources.idx_utils import download, read_idx_gzip


def _fix_orientation(image: np.ndarray) -> np.ndarray:
    """EMNIST images are stored transposed relative to normal row-major
    orientation — an artifact of the original MATLAB conversion NIST used.
    See https://www.nist.gov/itl/products-and-services/emnist-dataset."""
    return image.T


def parse_mapping(path: Path) -> dict[int, str]:
    """Parses emnist-<split>-mapping.txt: 'label ascii_code' per line."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, code_str = line.split()
            mapping[int(label_str)] = chr(int(code_str))
    return mapping


class EmnistDatasetSource(DatasetSource):
    name = "emnist"

    def __init__(
        self,
        config: EmnistConfig,
        val_fraction: float = 0.1,
        max_samples: int | None = None,
    ) -> None:
        self._config = config
        self._val_fraction = val_fraction
        self._max_samples = max_samples

    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        emnist_raw = raw_dir / self.name
        archive_path = emnist_raw / "gzip.zip"
        download(self._config.archive_url, archive_path)

        split = self._config.split
        member_names = {
            "train_images": f"gzip/emnist-{split}-train-images-idx3-ubyte.gz",
            "train_labels": f"gzip/emnist-{split}-train-labels-idx1-ubyte.gz",
            "test_images": f"gzip/emnist-{split}-test-images-idx3-ubyte.gz",
            "test_labels": f"gzip/emnist-{split}-test-labels-idx1-ubyte.gz",
            "mapping": f"gzip/emnist-{split}-mapping.txt",
        }
        local_paths = self._extract_members(archive_path, emnist_raw, member_names)

        mapping = parse_mapping(local_paths["mapping"])
        train_images = read_idx_gzip(local_paths["train_images"])
        train_labels = read_idx_gzip(local_paths["train_labels"])
        test_images = read_idx_gzip(local_paths["test_images"])
        test_labels = read_idx_gzip(local_paths["test_labels"])

        samples = self._write_split(
            train_images, train_labels, mapping, processed_dir, has_val=True
        )
        samples += self._write_split(
            test_images, test_labels, mapping, processed_dir, has_val=False
        )
        return samples

    @staticmethod
    def _extract_members(
        archive_path: Path, extract_root: Path, member_names: dict[str, str]
    ) -> dict[str, Path]:
        extract_root.mkdir(parents=True, exist_ok=True)
        local_paths = {}
        with zipfile.ZipFile(archive_path) as zf:
            for key, member in member_names.items():
                dest = extract_root / member
                if not dest.exists():
                    zf.extract(member, path=extract_root)
                local_paths[key] = dest
        return local_paths

    def _write_split(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mapping: dict[int, str],
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

            oriented = _fix_orientation(images[i])
            image = Image.fromarray((255 - oriented).astype(np.uint8))
            character = mapping[int(labels[i])]
            safe_char = character if character.isalnum() else f"_{ord(character)}_"

            relative_path = Path(self.name) / split / "character" / f"{safe_char}_{i}.png"
            full_path = processed_dir / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(full_path)

            samples.append(
                DatasetSample(
                    image_path=str(relative_path),
                    transcript=character,
                    source=self.name,
                    split=split,
                    label_type="character",
                    writer_id=None,
                )
            )
        return samples
