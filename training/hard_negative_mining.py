"""Oversamples training data for characters the model is currently confusing,
using a `ConfusionMatrix` from a prior evaluation run (see
training/confusion_matrix.py). This is intentionally the only piece of actual
oversampling logic — confusion_matrix.py stays pure analysis, training/train.py
decides when to apply this.
"""

from __future__ import annotations

from datasets.manifest import DatasetSample


def oversample_hard_negatives(
    samples: list[DatasetSample],
    hard_classes: set[str],
    oversample_factor: int = 3,
) -> list[DatasetSample]:
    """Any sample whose transcript contains at least one hard-negative
    character is repeated `oversample_factor` times in total (not appended
    an extra `oversample_factor` times) in the returned list; every other
    sample appears once. `oversample_factor < 1` is treated as 1 (a no-op)."""
    factor = max(1, oversample_factor)
    if not hard_classes:
        return list(samples)

    oversampled: list[DatasetSample] = []
    for sample in samples:
        repeats = factor if any(c in hard_classes for c in sample.transcript) else 1
        oversampled.extend([sample] * repeats)
    return oversampled
