"""Common interface every dataset source implements, so scripts/prepare_dataset.py
and training code can treat them uniformly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from datasets.manifest import DatasetSample


class DatasetSource(ABC):
    name: str

    @abstractmethod
    def prepare(self, raw_dir: Path, processed_dir: Path) -> list[DatasetSample]:
        """Convert raw files under raw_dir into normalized images under
        processed_dir and return the corresponding DatasetSample records.
        Does not write the manifest file — callers decide where/how to persist it."""
        raise NotImplementedError
