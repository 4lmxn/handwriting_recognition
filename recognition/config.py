from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR, resolve_device


@dataclass(frozen=True)
class RecognitionConfig:
    model_name: str
    device: str
    max_new_tokens: int

    def resolved_device(self) -> str:
        return resolve_device(self.device)


def load_recognition_config() -> RecognitionConfig:
    path = CONFIGS_DIR / "recognition.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return RecognitionConfig(**data)
