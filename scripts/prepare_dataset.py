"""Acquire and normalize a dataset into datasets/processed/ plus a manifest
under datasets/manifests/. See datasets/registry.py for what's available and
how each one is acquired (auto-download vs. manual/registered download).

Usage:
    uv run python scripts/prepare_dataset.py synthetic
    uv run python scripts/prepare_dataset.py mnist
    uv run python scripts/prepare_dataset.py emnist --limit 5000
    uv run python scripts/prepare_dataset.py iam
"""

from __future__ import annotations

import argparse
import logging

from app.config import load_config
from app.logging_config import setup_logging
from datasets.config import load_datasets_config
from datasets.manifest import write_manifest
from datasets.registry import get_source
from datasets.sources.base import DatasetSource
from datasets.sources.cvl import CvlDatasetSource
from datasets.sources.emnist import EmnistDatasetSource
from datasets.sources.iam import IamDatasetSource
from datasets.sources.mnist import MnistDatasetSource
from datasets.sources.synthetic import SyntheticDatasetSource

logger = logging.getLogger(__name__)


def build_source(name: str, max_samples: int | None) -> DatasetSource:
    datasets_config = load_datasets_config()
    if name == "synthetic":
        return SyntheticDatasetSource(datasets_config.synthetic)
    if name == "mnist":
        return MnistDatasetSource(datasets_config.mnist, max_samples=max_samples)
    if name == "emnist":
        return EmnistDatasetSource(datasets_config.emnist, max_samples=max_samples)
    if name == "iam":
        return IamDatasetSource()
    if name == "cvl":
        return CvlDatasetSource()
    raise NotImplementedError(
        f"'{name}' is registered but has no implemented DatasetSource yet — see docs/ROADMAP.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and normalize a handwriting dataset.")
    parser.add_argument("name", help="Dataset name — see datasets/registry.py")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per split (mnist/emnist only) — useful for a quick smoke run.",
    )
    args = parser.parse_args()

    app_config = load_config()
    app_config.paths.ensure_exist()
    setup_logging(app_config.paths.logs, app_config.log_level, filename="prepare_dataset.log")

    info = get_source(args.name)
    if info.acquisition == "manual":
        logger.info("'%s' requires manual acquisition: %s", args.name, info.instructions)

    source = build_source(args.name, args.limit)
    logger.info("Preparing dataset '%s'...", args.name)
    samples = source.prepare(
        raw_dir=app_config.paths.datasets_raw,
        processed_dir=app_config.paths.datasets_processed,
    )

    manifest_path = app_config.paths.datasets_manifests / f"{args.name}.jsonl"
    write_manifest(samples, manifest_path)
    logger.info("Wrote %d samples to %s", len(samples), manifest_path)


if __name__ == "__main__":
    main()
