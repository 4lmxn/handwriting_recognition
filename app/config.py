"""Central configuration loader.

All paths and tunable parameters live in configs/*.yaml. Nothing in the
application should hardcode a path or hyperparameter — add a field to the
relevant YAML file and read it through this module instead.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda', 'mps' or 'cpu'. Never raises if torch is absent.

    CUDA is preferred over Apple Silicon's MPS backend: MPS accelerates inference
    but has patchier operator coverage, so it is only chosen when no CUDA device
    exists. Any explicit value is passed through untouched, which is how a run can
    be forced onto "cpu" if an MPS operator gap ever shows up.
    """
    if device != "auto":
        return device
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class PathsConfig:
    datasets_raw: Path
    datasets_processed: Path
    datasets_manifests: Path
    weights: Path
    outputs: Path
    logs: Path
    experiments: Path

    @classmethod
    def from_dict(cls, data: dict) -> PathsConfig:
        return cls(**{k: REPO_ROOT / v for k, v in data.items()})

    def ensure_exist(self) -> None:
        for f in (
            self.datasets_raw,
            self.datasets_processed,
            self.datasets_manifests,
            self.weights,
            self.outputs,
            self.logs,
            self.experiments,
        ):
            f.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class WindowConfig:
    title: str = "Handwriting Recognition"
    width: int = 1280
    height: int = 800


@dataclass(frozen=True)
class CanvasConfig:
    default_pen_width: int = 4
    min_pen_width: int = 1
    max_pen_width: int = 40
    background_color: str = "#FFFFFF"
    pen_color: str = "#000000"
    grid_enabled: bool = False
    grid_spacing: int = 20
    undo_stack_depth: int = 50


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    device: str
    log_level: str
    window: WindowConfig
    canvas: CanvasConfig
    paths: PathsConfig = field(repr=False)

    def resolved_device(self) -> str:
        return resolve_device(self.device)


def load_config() -> AppConfig:
    app_data = _load_yaml("app.yaml")
    paths_data = _load_yaml("paths.yaml")

    return AppConfig(
        name=app_data["name"],
        version=app_data["version"],
        device=app_data["device"],
        log_level=app_data["log_level"],
        window=WindowConfig(**app_data["window"]),
        canvas=CanvasConfig(**app_data["canvas"]),
        paths=PathsConfig.from_dict(paths_data),
    )
