"""Unified format every dataset source is normalized into. Training code only
ever reads DatasetSample records from a manifest — it never needs to know
whether a sample originally came from EMNIST, IAM, or a synthetic renderer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

LabelType = Literal["character", "word", "line"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetSample:
    image_path: str  # relative to datasets/processed/
    transcript: str
    source: str  # e.g. "emnist", "iam", "synthetic"
    split: Split
    label_type: LabelType
    writer_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetSample:
        return cls(**data)


def write_manifest(samples: list[DatasetSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def read_manifest(path: Path) -> list[DatasetSample]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(DatasetSample.from_dict(json.loads(line)))
    return samples


def append_manifest(samples: list[DatasetSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
