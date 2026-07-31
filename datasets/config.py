"""Loader for configs/datasets.yaml — dataset acquisition and synthetic
generation parameters. Mirrors the pattern in app/config.py."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class SyntheticConfig:
    fonts: list[str]
    font_sizes: list[int]
    image_height: int
    samples_per_item: int
    split_ratios: dict[str, float]
    characters: str
    words: list[str]


@dataclass(frozen=True)
class MnistConfig:
    base_url: str
    train_images: str
    train_labels: str
    test_images: str
    test_labels: str


@dataclass(frozen=True)
class EmnistConfig:
    archive_url: str
    split: str


@dataclass(frozen=True)
class DatasetsConfig:
    synthetic: SyntheticConfig
    mnist: MnistConfig
    emnist: EmnistConfig


def load_datasets_config() -> DatasetsConfig:
    path = CONFIGS_DIR / "datasets.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    return DatasetsConfig(
        synthetic=SyntheticConfig(**data["synthetic"]),
        mnist=MnistConfig(**data["mnist"]),
        emnist=EmnistConfig(**data["emnist"]),
    )
