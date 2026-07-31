# Handwriting Recognition System

An offline, continually-learning handwriting recognition application: draw with a
mouse/stylus, upload images, or scan pages — the system recognizes text and improves
from your corrections over time without forgetting what it already knew.

This project is built in phases; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full
plan and current status.

## Hardware target

**All code must run correctly on CPU** — nothing in the codebase may *require* a GPU.
Device selection is automatic (`configs/app.yaml: device: auto`), resolving in order:

| Device | Used for | Notes |
|--------|----------|-------|
| `cuda` | training + inference | Preferred. fp16 mixed precision enabled automatically (`configs/training.yaml: use_amp`). |
| `mps` | Apple Silicon | Works for training and inference; ~1.3x faster than CPU on single-image TrOCR-small inference. AMP stays off (GradScaler is CUDA-only). |
| `cpu` | fallback | Always supported. |

Set `device` explicitly to any of these to override `auto` — the escape hatch if an
MPS operator gap ever appears.

Modeling strategy: fine-tune compact **pretrained** handwriting-recognition backbones
(e.g. TrOCR-small, CRNN+CTC baselines) rather than training a large custom architecture
from scratch — see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the rationale.

## Requirements

- Linux (Ubuntu 22.04+), macOS (Apple Silicon) or Windows 10/11
- Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

The torch build is selected per-platform in `pyproject.toml` (`[tool.uv.sources]`):
Windows pulls the **CUDA (cu126)** build, everything else pulls the default build
(which already includes MPS on Apple Silicon). `uv sync` is all that's needed.

Do **not** try to swap the build with `uv pip install torch --index-url ...` — `uv run`
re-syncs the environment against `uv.lock` on every invocation and will revert it. To
change the CUDA version, edit the `pytorch-cu126` index in `pyproject.toml` and re-run
`uv lock`. Check what the index actually carries first; `cu124`, for instance, stops at
torch 2.6.0 and would silently downgrade the pin.

Verify the GPU is live:

```bash
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
uv run python -c "from app.config import resolve_device; print(resolve_device('auto'))"
```

Expect a CUDA version + `True`, then `cuda`.

## Setup

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Full sync (GUI + dev + preprocessing/datasets + ML/recognition)
uv sync --extra gui --extra dev --extra cv --extra ml

# Later phases will additionally need:
#   uv sync --extra gui --extra dev --extra cv --extra ml --extra data --extra log --extra export
```

`ml` pulls in CPU-build `torch` + `transformers` (~200MB) for the recognition pipeline —
first run of anything in `recognition/` also downloads the `microsoft/trocr-small-handwritten`
model weights (~250MB) from Hugging Face on first use.

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
recognition/         TrOCR-based recognition pipeline (image -> text + confidence)
scripts/             prepare_dataset.py and other operational scripts
segmentation/        Line/word/character segmentation
tests/               Unit, integration, and GUI tests
training/            Fine-tuning loop, dataset adapter, evaluation, confusion analysis, hard-negative mining
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

## Fine-tuning (Phase 4)

```bash
uv sync --extra gui --extra dev --extra cv --extra ml --extra log   # adds tensorboard

uv run python scripts/prepare_dataset.py mnist --limit 400   # or any other manifest
uv run python scripts/prepare_dataset.py synthetic
uv run python -m training.train   # reads configs/training.yaml

# After training, find what the model confuses and feed it back in:
uv run python scripts/analyze_confusions.py synthetic --split test
# then set configs/training.yaml: confusion_matrix_path to the saved
# experiments/confusion_matrix_<name>.json and rerun training.
```

Checkpoints land under `weights/<checkpoint_dir>/step-<N>/`, never overwritten — training
auto-resumes from the latest one unless `resume: false` in the config. Loss curves go to
TensorBoard, CSV, and JSONL simultaneously under `logs/tensorboard/<run_name>/`.

## Evaluating recognition

```bash
uv run python scripts/evaluate_model.py synthetic --split test --limit 100
```

Reports CER/WER/character/word accuracy against any prepared dataset's manifest. The
`synthetic` dataset is printed text, not real handwriting — useful as a pipeline smoke
test, not a substitute for evaluating against IAM/CVL.
