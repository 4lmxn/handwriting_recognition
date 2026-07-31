from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR, resolve_device


@dataclass(frozen=True)
class RecognitionConfig:
    model_name: str
    device: str
    max_new_tokens: int
    # Generation-time repetition guards. repetition_penalty > 1.0 softly
    # discourages repeat tokens; no_repeat_ngram_size > 0 hard-forbids any
    # n-gram from occurring twice. 1.0 and 0 disable them (transformers
    # defaults). See configs/recognition.yaml for the tuned defaults and
    # docs/ROADMAP.md Phase 4 for why they exist.
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def resolved_device(self) -> str:
        return resolve_device(self.device)


def load_recognition_config() -> RecognitionConfig:
    path = CONFIGS_DIR / "recognition.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return RecognitionConfig(**data)
