# Handwriting Recognition System

An offline, continually-learning handwriting recognition application: draw with a
mouse/stylus, upload images, or scan pages — the system recognizes text and improves
from your corrections over time without forgetting what it already knew.

This project is built in phases; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full
plan and current status.

## Hardware target

This project is developed and tested on **CPU-only hardware** (no discrete/NVIDIA GPU
available on the reference machine — Intel integrated graphics only). All code must run
correctly on CPU. Device selection is automatic (`configs/app.yaml: device: auto`) so a
CUDA GPU is used transparently if one becomes available later, but nothing in the
codebase may *require* a GPU.

Modeling strategy: fine-tune compact **pretrained** handwriting-recognition backbones
(e.g. TrOCR-small, CRNN+CTC baselines) rather than training a large custom architecture
from scratch — see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the rationale.

## Requirements

- Ubuntu 22.04+ (or compatible Linux)
- Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync Phase 1 dependencies (GUI + dev tooling)
uv sync --extra gui --extra dev

# Later phases will additionally need:
#   uv sync --extra gui --extra dev --extra cv --extra ml --extra data --extra log
```

## Running the app

```bash
uv run python -m app.main
```

## Running tests

```bash
uv run pytest
```

GUI tests run headless via the Qt `offscreen` platform plugin automatically (see
`tests/conftest.py`).

## Project layout

```
app/                 PySide6 GUI application (main window, tabs, widgets)
configs/             YAML configuration (paths, app, datasets, augmentation, model, training)
datasets/            Manifest schema, dataset registry, dataset sources; raw/processed data (not committed)
docs/                Documentation, including the phase-by-phase ROADMAP
experiments/         Training run outputs, kept out of git
feedback/            User-correction storage and incremental training scheduling (later phase)
language_model/      Language-model-assisted decoding and correction (later phase)
logs/                Application and training logs (not committed)
models/              Model backbones and personalization adapters (later phase)
outputs/             Inference outputs (not committed)
preprocessing/       Image preprocessing (deskew, denoise, normalization) + augmentation pipeline
recognition/         Recognition pipeline glue code (later phase)
scripts/             prepare_dataset.py and other operational scripts
segmentation/        Line/word/character segmentation
tests/               Unit, integration, and GUI tests
training/            Training loops, dataset adapters, evaluation (later phase)
weights/             Model checkpoints (not committed; see .gitignore)
```

## Configuration

All paths and hyperparameters live in `configs/*.yaml`, loaded through `app/config.py`
(app/paths), `datasets/config.py` (dataset acquisition/synthetic generation), and
`preprocessing/augmentation.py` (augmentation). Do not hardcode paths or hyperparameters
in code — add a field to the relevant config file instead.

## Preparing datasets

```bash
uv sync --extra gui --extra dev --extra cv   # add cv extras for dataset/preprocessing work
uv run python scripts/prepare_dataset.py synthetic   # always available, no download
uv run python scripts/prepare_dataset.py mnist        # auto-downloads (~11MB)
uv run python scripts/prepare_dataset.py emnist       # auto-downloads (~550MB) — run deliberately
uv run python scripts/prepare_dataset.py iam          # requires manual download first, see datasets/registry.py
uv run python scripts/prepare_dataset.py cvl          # requires manual download first, see datasets/registry.py
```

Each run writes normalized images under `datasets/processed/<name>/` and a manifest at
`datasets/manifests/<name>.jsonl` (see `datasets/manifest.py`).
