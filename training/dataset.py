"""PyTorch Dataset wrapping our manifest format for TrOCR fine-tuning.

When an augmentation pipeline is supplied, each __getitem__ call re-augments
the clean image on disk (see preprocessing/augmentation.py) — the same sample
looks different across epochs rather than being pre-augmented once.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from datasets.manifest import DatasetSample, read_manifest


def load_samples(
    manifest_names: list[str],
    manifests_dir: Path,
    splits: list[str],
    max_samples: int | None = None,
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for name in manifest_names:
        manifest_path = manifests_dir / f"{name}.jsonl"
        if not manifest_path.exists():
            continue
        samples.extend(s for s in read_manifest(manifest_path) if s.split in splits)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


class HandwritingDataset(Dataset):
    def __init__(
        self,
        samples: list[DatasetSample],
        processed_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int = 32,
        augmentation_pipeline=None,
    ) -> None:
        self._samples = samples
        self._processed_dir = processed_dir
        self._processor = processor
        self._max_target_length = max_target_length
        self._augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._samples[idx]
        image = cv2.imread(str(self._processed_dir / sample.image_path), cv2.IMREAD_GRAYSCALE)
        if self._augmentation_pipeline is not None:
            image = self._augmentation_pipeline(image=image)["image"]

        pil_image = Image.fromarray(image).convert("RGB")
        pixel_values = self._processor(images=pil_image, return_tensors="pt").pixel_values[0]

        tokenizer = self._processor.tokenizer  # type: ignore[attr-defined]
        token_ids = tokenizer(
            sample.transcript,
            padding="max_length",
            max_length=self._max_target_length,
            truncation=True,
        ).input_ids
        labels = torch.tensor([tid if tid != tokenizer.pad_token_id else -100 for tid in token_ids])

        return {"pixel_values": pixel_values, "labels": labels}
