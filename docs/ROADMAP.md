# Roadmap

This project is built incrementally. Each phase must be fully working — app runs,
tests pass, lint/type-check clean — before the next one starts.

## Decisions on record

- **Hardware**: this machine is CPU-only (Intel integrated graphics, no discrete/NVIDIA
  GPU). Confirmed with the user on 2026-07-31 — this is their only machine, no separate
  GPU box or cloud instance currently in use. All code auto-detects a CUDA device via
  `AppConfig.resolved_device()` and uses it if one becomes available, but nothing may
  *require* a GPU.
- **Modeling strategy**: fine-tune compact **pretrained** handwriting-recognition
  backbones (e.g. TrOCR-small, or a CRNN+CTC baseline) plus a lightweight personalization
  adapter, rather than training the full CNN→ViT→BiLSTM→CTC→Transformer-LM pipeline from
  scratch across 8+ datasets. Training a large custom architecture from scratch on CPU
  is not practical (realistically weeks of compute for worse results than fine-tuning a
  pretrained model). This was an explicit user decision, not a default.
- **Scope**: the original spec is extremely broad (writer identification, REST API, web
  interface, mobile support, TensorRT, KHATT/CASIA/RIMES access which require licensing
  agreements, etc.). These are kept on the backlog as optional stretch goals rather than
  committed phases — see "Explicitly deferred" below. Committed phases below are scoped
  to what's realistically buildable and testable on this hardware.

## Phase 1 — Project architecture & environment setup ✅ (complete)

- Repository structure: `app/`, `datasets/`, `training/`, `models/`, `preprocessing/`,
  `segmentation/`, `recognition/`, `language_model/`, `feedback/`, `tests/`, `configs/`,
  `experiments/`, `scripts/`, `docs/`, `weights/`, `outputs/`, `logs/`.
- Dependency management via `uv`, dependency groups (`gui`, `cv`, `ml`, `data`, `log`,
  `dev`) so later phases add weight incrementally instead of one giant install.
  `torch`/`torchvision` pinned to the CPU wheel index.
- YAML configuration system (`configs/app.yaml`, `configs/paths.yaml`, loaded via
  `app/config.py` dataclasses) — no hardcoded paths or hyperparameters in code.
- Central logging (`app/logging_config.py`) writing to `logs/`.
- PySide6 GUI shell (`app/main.py`, `app/gui/main_window.py`) with a working
  **Drawing Canvas** tab: mouse drawing, undo/redo (bounded stack), clear, pen width and
  color controls, optional grid, save-to-PNG. Recognition is explicitly *not* wired up
  yet — the tab says so rather than faking a result.
- Unit tests (config loader) and GUI tests (canvas drawing/undo/redo, main window
  construction) running headless via `QT_QPA_PLATFORM=offscreen`. `ruff` and `mypy`
  clean.

## Phase 2 — Preprocessing & dataset pipeline

- Unified preprocessing: grayscale, adaptive thresholding, deskew, denoise, connected
  component analysis, baseline detection, normalization.
- Line / word / character segmentation.
- Dataset acquisition + unified manifest format (image path, transcript, source dataset,
  writer id where available). Start with datasets that are freely downloadable without a
  separate licensing agreement: **EMNIST, MNIST, IAM** (registration required, user must
  supply credentials — not auto-bypassed), **CVL**. NIST SD19, RIMES, KHATT, CASIA,
  Bentham are added opportunistically since several require registration/licensing that
  can't be automated.
- Augmentation pipeline (Albumentations): rotation, skew, slant, noise, blur, perspective,
  brightness/contrast, ink fading, pen-thickness simulation, paper texture, compression
  artifacts, cropping, scaling, elastic transforms.
- Dataset tests: manifest schema validation, augmentation determinism/seeding.

## Phase 3 — Baseline recognition (pretrained backbone)

- Integrate a pretrained HTR backbone (TrOCR-small via `transformers`, CPU inference) —
  benchmark against a lighter CRNN+CTC baseline for CPU latency.
- Inference pipeline: image → preprocessing → model → text → confidence score.
- Wire a "Recognize" action into the Drawing Canvas tab and a new Upload Image tab.
- Evaluation harness: CER, WER, character/word/sentence accuracy on a held-out split.

## Phase 4 — Fine-tuning, confusion analysis, hard-negative mining

- Fine-tuning loop on CPU-appropriate batch sizes with checkpointing and resume.
- TensorBoard + CSV/JSON logging of loss/accuracy curves.
- Confusion matrix generation after every training run; automated analysis of commonly
  confused classes (0/O, 1/l/I, 5/S, 2/Z, 8/B, 6/G, 9/g, rn/m, cl/d, vv/w, O/Q, C/G,
  punctuation).
- Hard-negative mining: oversample/re-augment frequently-confused classes in subsequent
  training rounds.

## Phase 5 — Feedback loop & personalization (continual learning)

- Correction UI: user edits a wrong prediction; corrected pairs are stored under
  `feedback/`.
- A **personalization adapter** (LoRA-style, layered on the frozen fine-tuned backbone)
  is incrementally updated from corrections — the base model is never overwritten.
- Replay buffer mixing original training data with new corrections to avoid catastrophic
  forgetting.
- Versioned checkpoints; before/after accuracy logged per incremental update so
  regressions are visible immediately.

## Phase 6 — Document & upload pipeline

- Image upload (PNG/JPEG/TIFF) and PDF (incl. multi-page) via PyMuPDF.
- Automatic text-region detection feeding the Phase 2 segmentation pipeline.
- Batch inference across a full page with structured (line/word) output.

## Phase 7 — Language-model-assisted decoding

- Beam search decoding with a lightweight LM (n-gram or small transformer) rescoring
  candidates.
- Dictionary support: base vocabulary + user-custom vocabulary + technical/domain
  vocabulary.

## Phase 8 — Structured / mathematical notation (scoped down)

- Layout-based detection of superscript/subscript positioning and simple structures
  (fractions, roots) built on top of segmentation output, feeding a LaTeX-fragment
  exporter. This is **not** a full math-OCR research effort — complex nested expressions,
  integrals/summations, and chemical formulas are explicitly out of scope unless later
  reprioritized.

## Phase 9 — Remaining GUI surfaces

- Upload Image, Training Dashboard, Model Statistics, Confusion Matrix, Dataset Manager,
  Settings, Inference History, Feedback Manager tabs — each added once its backend
  (Phases 2–7) exists, never as a placeholder pretending to work.

## Phase 10 — Performance & export

- ONNX export and quantization for faster CPU inference.
- Benchmark suite: CER, WER, character/word/sentence accuracy, precision/recall/F1,
  top-1/top-5, run after every training phase.
- CUDA/mixed-precision paths remain auto-detected but are not the optimization target
  given the current hardware.

## Phase 11 — Docs & packaging

- Installation guide, training guide, inference guide, architecture diagrams, developer
  docs.

## Explicitly deferred (stretch goals, not committed)

Writer identification/clustering, active learning, self-supervised pretraining,
few-shot personalization beyond the Phase 5 adapter, REST API, web interface, mobile
support, cloud sync, TensorRT. These stay on the backlog; pull one in only when there's a
concrete reason to prioritize it over the phases above.
