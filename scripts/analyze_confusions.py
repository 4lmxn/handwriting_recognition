"""Run the recognizer over a manifest's character-level samples, build a
confusion matrix, and save it for hard-negative mining in a follow-up
training round (see training/hard_negative_mining.py).

Usage:
    uv run python scripts/analyze_confusions.py synthetic --split test
"""

from __future__ import annotations

import argparse
import logging

import cv2

from app.config import load_config
from app.logging_config import setup_logging
from datasets.manifest import read_manifest
from recognition.config import load_recognition_config
from recognition.recognizer import Recognizer
from training.confusion_matrix import (
    ConfusionMatrix,
    analyze_ambiguous_classes,
    hard_negative_classes,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a confusion matrix from recognizer output.")
    parser.add_argument(
        "dataset_name", help="Manifest name under datasets/manifests/ (no extension)"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--label-type", default="character", choices=["character", "word", "line"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--min-count", type=int, default=3, help="Threshold for hard_negative_classes"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override recognition_config.model_name — HF hub id OR path to a "
        "local fine-tuned checkpoint (e.g. weights/trocr-cvl/step-2675). Lets "
        "you build the confusion matrix from a specific fine-tuned run.",
    )
    args = parser.parse_args()

    app_config = load_config()
    setup_logging(app_config.paths.logs, app_config.log_level, filename="analyze_confusions.log")

    manifest_path = app_config.paths.datasets_manifests / f"{args.dataset_name}.jsonl"
    samples = [
        s
        for s in read_manifest(manifest_path)
        if s.split == args.split and s.label_type == args.label_type
    ][: args.limit]
    if not samples:
        raise ValueError(f"No matching samples found in {manifest_path}")

    recognition_config = load_recognition_config()
    model_name = args.model_name or recognition_config.model_name
    logger.info("Loading recognizer from %s", model_name)
    recognizer = Recognizer(
        model_name,
        device=recognition_config.resolved_device(),
        max_new_tokens=recognition_config.max_new_tokens,
    )

    matrix = ConfusionMatrix()
    for sample in samples:
        image_path = app_config.paths.datasets_processed / sample.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        result = recognizer.recognize(image)
        matrix.record(sample.transcript, result.text)

    output_path = app_config.paths.experiments / f"confusion_matrix_{args.dataset_name}.json"
    matrix.save_json(output_path)

    logger.info(
        "Analyzed %d samples, %d substitutions total", len(samples), sum(matrix.counts.values())
    )
    logger.info("Most confused pairs: %s", matrix.most_confused(10))
    logger.info("Ambiguous-class report: %s", analyze_ambiguous_classes(matrix))
    logger.info(
        "Hard-negative classes (min_count=%d): %s",
        args.min_count,
        hard_negative_classes(matrix, args.min_count),
    )
    logger.info("Saved confusion matrix to %s", output_path)


if __name__ == "__main__":
    main()
