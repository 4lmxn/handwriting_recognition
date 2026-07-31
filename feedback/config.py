from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import CONFIGS_DIR, REPO_ROOT


@dataclass(frozen=True)
class FeedbackConfig:
    storage_dir: str
    image_dir: str

    @property
    def storage_dir_path(self) -> Path:
        return REPO_ROOT / self.storage_dir

    @property
    def image_dir_path(self) -> Path:
        return REPO_ROOT / self.image_dir


def load_feedback_config() -> FeedbackConfig:
    path = CONFIGS_DIR / "feedback.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return FeedbackConfig(**data)
