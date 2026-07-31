"""Replay buffer for Phase 5 incremental adapter training.

The user's accumulated corrections drive personalization, but training
the adapter on corrections alone would let it drift away from the base
model's general handwriting distribution (catastrophic forgetting).
Every update mixes a slice of the original training data ("replay") with
the corrections, so the adapter learns the user's writing style without
losing what the base model already knows.

`replay_ratio` controls the mix: 0.7 means 70% base samples, 30% new
corrections. If the correction pool is smaller than the target, we
include all of it and fill the remainder from base. Deterministic under
`seed` so a rejected update can be reproduced from the logged config.
"""

from __future__ import annotations

import random
from pathlib import Path

from datasets.manifest import DatasetSample


def build_replay_batch(
    feedback_samples: list[DatasetSample],
    base_samples: list[DatasetSample],
    replay_ratio: float,
    max_total: int,
    seed: int,
) -> list[DatasetSample]:
    """Return a shuffled mix of feedback + base samples.

    Args:
        feedback_samples: all accumulated user corrections (pre-shuffle).
        base_samples: candidate base-manifest samples to draw the replay
            slice from.
        replay_ratio: fraction of the final batch that should be BASE
            data (0.0 to 1.0). 0.7 = 70% base + 30% corrections.
        max_total: hard cap on total batch size.
        seed: RNG seed. Same inputs + seed always produce the same batch.

    Never mutates the input lists. Handles all four edge cases (either
    input empty, either input smaller than its share of the target)
    without raising; the batch just ends up smaller than max_total.
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError(f"replay_ratio must be in [0, 1], got {replay_ratio}")
    if max_total < 0:
        raise ValueError(f"max_total must be non-negative, got {max_total}")

    rng = random.Random(seed)

    target_base = int(max_total * replay_ratio)
    target_feedback = max_total - target_base

    n_feedback = min(len(feedback_samples), target_feedback)
    n_base = min(len(base_samples), target_base)

    sampled_feedback = rng.sample(feedback_samples, n_feedback) if n_feedback else []
    sampled_base = rng.sample(base_samples, n_base) if n_base else []

    combined = sampled_feedback + sampled_base
    rng.shuffle(combined)
    return combined


def load_replay_base(
    manifests: tuple[str, ...] | list[str],
    splits: tuple[str, ...] | list[str],
    manifests_dir: Path,
    max_samples: int | None = None,
) -> list[DatasetSample]:
    """Thin wrapper around `training.dataset.load_samples` for named clarity.

    Kept in this module so incremental.py imports read as
    "load_replay_base(...)" rather than "load_samples(...)" — the former
    signals intent, the latter reads like a duplicate of the Phase 4
    training loop.
    """
    from training.dataset import load_samples

    return load_samples(list(manifests), manifests_dir, list(splits), max_samples)
