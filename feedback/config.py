from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import CONFIGS_DIR, REPO_ROOT, resolve_device
from models.adapters.config import LoraAdapterConfig


@dataclass(frozen=True)
class ReplayConfig:
    base_manifests: tuple[str, ...]
    base_splits: tuple[str, ...]
    replay_ratio: float
    max_total_samples: int
    min_pending_corrections: int
    max_corrections: int
    seed: int

    @classmethod
    def from_dict(cls, data: dict) -> ReplayConfig:
        return cls(
            base_manifests=tuple(data.get("base_manifests", ("cvl",))),
            base_splits=tuple(data.get("base_splits", ("train",))),
            replay_ratio=float(data.get("replay_ratio", 0.7)),
            max_total_samples=int(data.get("max_total_samples", 512)),
            min_pending_corrections=int(data.get("min_pending_corrections", 5)),
            max_corrections=int(data.get("max_corrections", 200)),
            seed=int(data.get("seed", 42)),
        )


@dataclass(frozen=True)
class IncrementalTrainingConfig:
    device: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_target_length: int
    use_amp: bool
    augment: bool
    log_dir: str
    run_name_prefix: str

    def resolved_device(self) -> str:
        return resolve_device(self.device)

    @property
    def log_dir_path(self) -> Path:
        return REPO_ROOT / self.log_dir

    @classmethod
    def from_dict(cls, data: dict) -> IncrementalTrainingConfig:
        return cls(
            device=str(data.get("device", "auto")),
            batch_size=int(data.get("batch_size", 4)),
            num_epochs=int(data.get("num_epochs", 1)),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            max_target_length=int(data.get("max_target_length", 32)),
            use_amp=bool(data.get("use_amp", True)),
            augment=bool(data.get("augment", True)),
            log_dir=str(data.get("log_dir", "logs/tensorboard")),
            run_name_prefix=str(data.get("run_name_prefix", "phase5-adapter")),
        )


@dataclass(frozen=True)
class IncrementalEvalConfig:
    manifest: str
    split: str
    limit: int
    max_cer_regression: float

    @classmethod
    def from_dict(cls, data: dict) -> IncrementalEvalConfig:
        return cls(
            manifest=str(data.get("manifest", "cvl")),
            split=str(data.get("split", "test")),
            limit=int(data.get("limit", 100)),
            max_cer_regression=float(data.get("max_cer_regression", 0.02)),
        )


@dataclass(frozen=True)
class FeedbackConfig:
    storage_dir: str
    image_dir: str
    adapter: LoraAdapterConfig
    adapter_dir: str
    replay: ReplayConfig = field(
        default_factory=lambda: ReplayConfig.from_dict({})
    )
    training: IncrementalTrainingConfig = field(
        default_factory=lambda: IncrementalTrainingConfig.from_dict({})
    )
    eval: IncrementalEvalConfig = field(
        default_factory=lambda: IncrementalEvalConfig.from_dict({})
    )

    @property
    def storage_dir_path(self) -> Path:
        return REPO_ROOT / self.storage_dir

    @property
    def image_dir_path(self) -> Path:
        return REPO_ROOT / self.image_dir

    @property
    def adapter_dir_path(self) -> Path:
        return REPO_ROOT / self.adapter_dir


def load_feedback_config() -> FeedbackConfig:
    path = CONFIGS_DIR / "feedback.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return FeedbackConfig(
        storage_dir=data["storage_dir"],
        image_dir=data["image_dir"],
        adapter=LoraAdapterConfig.from_dict(data.get("adapter", {})),
        adapter_dir=data["adapter_dir"],
        replay=ReplayConfig.from_dict(data.get("replay", {})),
        training=IncrementalTrainingConfig.from_dict(data.get("training", {})),
        eval=IncrementalEvalConfig.from_dict(data.get("eval", {})),
    )
