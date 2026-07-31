from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import CONFIGS_DIR, REPO_ROOT, resolve_device


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    device: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_target_length: int
    checkpoint_dir: str
    log_dir: str
    run_name: str
    manifests: list[str]
    splits: list[str]
    resume: bool
    save_every_n_steps: int
    seed: int
    augment: bool
    max_samples: int | None
    confusion_matrix_path: str | None
    hard_negative_min_count: int
    hard_negative_oversample_factor: int

    def resolved_device(self) -> str:
        return resolve_device(self.device)

    @property
    def checkpoint_dir_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_dir

    @property
    def log_dir_path(self) -> Path:
        return REPO_ROOT / self.log_dir

    @property
    def confusion_matrix_full_path(self) -> Path | None:
        if self.confusion_matrix_path is None:
            return None
        return REPO_ROOT / self.confusion_matrix_path


def load_training_config() -> TrainingConfig:
    path = CONFIGS_DIR / "training.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**data)
