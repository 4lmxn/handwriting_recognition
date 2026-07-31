"""Fine-tuning loop for the TrOCR recognizer, with checkpointing and resume.

Checkpoints are versioned by step count under checkpoint_dir/step-<N>/ — never
overwritten in place — so a bad run can't destroy a previous good checkpoint.
Resume picks up the highest step directory found and continues training from
there, but only the model+processor are restored, not optimizer state (Adam
momentum resets on resume) — a known, documented simplification for Phase 4.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import load_config
from app.logging_config import setup_logging
from preprocessing.augmentation import build_augmentation_pipeline, load_augmentation_config
from training.config import TrainingConfig, load_training_config
from training.confusion_matrix import ConfusionMatrix, hard_negative_classes
from training.dataset import HandwritingDataset, load_samples
from training.hard_negative_mining import oversample_hard_negatives
from training.logging_utils import TrainingLogger

logger = logging.getLogger(__name__)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    step_dirs = sorted(
        (d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step-")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return step_dirs[-1] if step_dirs else None


def save_checkpoint(model, processor: TrOCRProcessor, checkpoint_dir: Path, step: int) -> Path:
    checkpoint_path = checkpoint_dir / f"step-{step}"
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def train(config: TrainingConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    app_config = load_config()
    device = config.resolved_device()
    checkpoint_dir = config.checkpoint_dir_path

    resume_checkpoint = find_latest_checkpoint(checkpoint_dir) if config.resume else None
    model_source = str(resume_checkpoint) if resume_checkpoint else config.model_name
    logger.info("Loading model from %s", model_source)

    processor = TrOCRProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_source)
    # Required for VisionEncoderDecoderModel to compute a training loss from
    # `labels` — it shifts them into decoder_input_ids internally and needs
    # both of these to know how.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # type: ignore[attr-defined]
    model.config.pad_token_id = processor.tokenizer.pad_token_id  # type: ignore[attr-defined]
    model.to(device)
    model.train()

    samples = load_samples(
        config.manifests,
        app_config.paths.datasets_manifests,
        config.splits,
        max_samples=config.max_samples,
    )
    if not samples:
        raise ValueError(
            f"No training samples found for manifests={config.manifests} splits={config.splits} "
            "— run scripts/prepare_dataset.py first."
        )

    matrix_path = config.confusion_matrix_full_path
    if matrix_path is not None and matrix_path.exists():
        matrix = ConfusionMatrix.load_json(matrix_path)
        hard_classes = hard_negative_classes(matrix, config.hard_negative_min_count)
        before = len(samples)
        samples = oversample_hard_negatives(
            samples, hard_classes, config.hard_negative_oversample_factor
        )
        logger.info(
            "Hard-negative mining: %d classes %s, oversampled %d -> %d training samples",
            len(hard_classes),
            sorted(hard_classes),
            before,
            len(samples),
        )

    logger.info("Training on %d samples", len(samples))

    augmentation_pipeline = None
    if config.augment:
        augmentation_pipeline = build_augmentation_pipeline(load_augmentation_config())

    dataset = HandwritingDataset(
        samples,
        app_config.paths.datasets_processed,
        processor,
        config.max_target_length,
        augmentation_pipeline,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    step = int(resume_checkpoint.name.split("-")[1]) if resume_checkpoint else 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with TrainingLogger(config.log_dir_path, config.run_name) as train_logger:
        for epoch in range(config.num_epochs):
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                train_logger.log_scalar("train/loss", loss.item(), step)
                logger.info("epoch=%d step=%d loss=%.4f", epoch, step, loss.item())

                if step % config.save_every_n_steps == 0:
                    save_checkpoint(model, processor, checkpoint_dir, step)

    return save_checkpoint(model, processor, checkpoint_dir, step)


def main() -> None:
    training_config = load_training_config()
    app_config = load_config()
    setup_logging(app_config.paths.logs, app_config.log_level, filename="train.log")
    train(training_config)


if __name__ == "__main__":
    main()
