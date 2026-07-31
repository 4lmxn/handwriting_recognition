"""Run the recognizer over a dataset manifest and report CER/WER/accuracy.

Usage:
    uv run python scripts/evaluate_model.py synthetic --split test --limit 100

Note: the synthetic dataset is procedurally rendered *printed* text (see
datasets/sources/synthetic.py), not real handwriting — useful as an end-to-end
pipeline smoke test, but not a substitute for evaluating against IAM/CVL once
those are downloaded (datasets/registry.py has acquisition instructions).
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
from training.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the recognizer against a manifest.")
    parser.add_argument(
        "dataset_name", help="Manifest name under datasets/manifests/ (no extension)"
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=100, help="Max samples to evaluate")
    args = parser.parse_args()

    app_config = load_config()
    setup_logging(app_config.paths.logs, app_config.log_level, filename="evaluate_model.log")

    manifest_path = app_config.paths.datasets_manifests / f"{args.dataset_name}.jsonl"
    samples = [s for s in read_manifest(manifest_path) if s.split == args.split][: args.limit]
    if not samples:
        raise ValueError(f"No '{args.split}' samples found in {manifest_path}")

    recognition_config = load_recognition_config()
    recognizer = Recognizer(
        recognition_config.model_name,
        device=recognition_config.resolved_device(),
        max_new_tokens=recognition_config.max_new_tokens,
    )

    references, hypotheses = [], []
    for sample in samples:
        image_path = app_config.paths.datasets_processed / sample.image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        result = recognizer.recognize(image)
        references.append(sample.transcript)
        hypotheses.append(result.text)

    metrics = evaluate_predictions(references, hypotheses)
    logger.info(
        "Evaluated %d samples from %s (split=%s)", metrics.num_samples, manifest_path, args.split
    )
    logger.info(
        "CER=%.3f WER=%.3f char_acc=%.3f word_acc=%.3f exact_match=%.3f",
        metrics.mean_cer,
        metrics.mean_wer,
        metrics.mean_char_accuracy,
        metrics.mean_word_accuracy,
        metrics.exact_match_rate,
    )


if __name__ == "__main__":
    main()
