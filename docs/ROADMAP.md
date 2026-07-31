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

## Phase 2 — Preprocessing & dataset pipeline ✅ (complete)

- Unified preprocessing (`preprocessing/image_ops.py`): grayscale, adaptive
  thresholding, denoise, deskew (with skew-angle estimation), connected component
  analysis, baseline estimation, resize+pad normalization, intensity normalization.
- Segmentation (`segmentation/`): line and word segmentation via projection profiles,
  character segmentation via connected components. Character segmentation is a
  best-effort heuristic — it can't split touching/cursive glyphs, which is fine because
  the planned CTC/attention recognizer (Phase 3+) doesn't depend on it; it exists for the
  GUI's optional segmentation visualization.
- Unified dataset manifest format (`datasets/manifest.py`: `DatasetSample` — image path,
  transcript, source, split, label type, writer id) plus a registry
  (`datasets/registry.py`) documenting every dataset from the original spec and how each
  is acquired.
- Five working dataset sources (`datasets/sources/`), run via
  `scripts/prepare_dataset.py <name>`:
  - **synthetic** — procedurally rendered chars/words from system fonts; always
    available, no license, seeds coverage of the Phase 4 confusable classes.
  - **mnist** / **emnist** — auto-downloaded (IDX format), no registration. EMNIST's
    known transpose-orientation quirk is corrected; label mapping is read from the
    archive's own mapping file rather than assumed.
  - **iam** — parses `ascii/lines.txt` + line images once the user has manually
    downloaded and registered for IAM (see `datasets/registry.py`). All samples
    currently land in the "train" split — IAM's official writer-independent split files
    are a documented follow-up, not guessed at here.
  - **cvl** — parses filename-embedded transcriptions once the user has manually
    downloaded CVL. This format isn't officially documented, so the parsing rules
    (skip second-segment `-6-`, skip umlauts) were verified against the published
    reference implementation in `amzn/convolutional-handwriting-gan`, not guessed.
  - NIST SD19, RIMES, KHATT, Bentham, CASIA remain registry entries with acquisition
    instructions only — not implemented (out of scope until prioritized; see
    "Explicitly deferred" note below).
- Augmentation pipeline (`preprocessing/augmentation.py`, Albumentations-based, fully
  config-driven via `configs/augmentation.yaml`): rotation, shear/slant, scaling,
  perspective, elastic transform, blur, noise, brightness/contrast, JPEG compression
  artifacts, crop/pad, pen-thickness (erosion/dilation), ink fading, and a synthetic
  paper-texture blend. Random word spacing is intentionally not a pixel-level transform
  here — it's exercised at synthetic-render time instead, since it only makes sense
  before text is flattened into an image.
- 91 tests total across Phase 1+2 (added: image ops, segmentation, manifest/registry,
  every dataset source, augmentation determinism), all offline/network-free except one
  manual smoke test against the real MNIST URL. `ruff` and `mypy` clean.

Known gaps carried forward rather than fixed here: IAM/CVL writer IDs aren't populated
(`writer_id=None`), IAM's official train/val/test split isn't wired up, and CVL word vs.
line writer-level directory structure hasn't been exploited for writer identification.
None of these block Phase 3.

## Phase 3 — Baseline recognition (pretrained backbone) ✅ (complete)

- Integrated `microsoft/trocr-small-handwritten` via `transformers` (CPU inference,
  config in `configs/recognition.yaml` / `recognition/config.py`). A CRNN+CTC baseline
  was not built for comparison: it would need training data/time we don't have yet,
  whereas TrOCR-small is already pretrained specifically for handwriting, so it's the
  only fair baseline available right now — revisit the comparison once Phase 4 fine-tuning
  exists. Required pinning `transformers<5` and adding `protobuf` — the bleeding-edge
  v5 line couldn't load this (older-format) model repo's tokenizer.
- Measured CPU latency: ~10s one-time model load, ~0.12s per single-image inference —
  fine for interactive use.
- `recognition/recognizer.py`: `Recognizer.recognize(image) -> RecognitionResult(text,
  confidence)`. Confidence is mean per-token generation probability (softmax over
  logits at each decoding step, at the chosen token) — a measure of the model's own
  certainty, not of correctness.
- Wired a "Recognize" button into the Drawing Canvas tab (`app/gui/tabs/drawing_canvas_tab.py`):
  guards on a blank canvas, lazily constructs the (slow-to-load) recognizer on first use,
  converts the canvas `QImage` to a grayscale numpy array, and shows
  `"Recognized: {text} ({confidence:.0%} confidence)"`. Runs synchronously — no threading
  yet, acceptable at ~0.12s/call but worth revisiting if slower models get used later.
  Visually verified: a meaningless scribble correctly produced a low-confidence (16%)
  nonsense result rather than a falsely-confident one.
- Evaluation harness (`training/evaluation.py`): CER, WER, character/word accuracy,
  exact-match rate, via a generic Levenshtein distance. `scripts/evaluate_model.py` runs
  it against any dataset manifest. Smoke-tested against 40 synthetic-dataset samples
  (CER 0.75) — expected to be mediocre since synthetic data is procedurally-rendered
  *printed* text, not real handwriting, and TrOCR-small wasn't trained on isolated
  single characters (a third of the synthetic vocabulary). Real accuracy validation
  needs IAM/CVL, which require the user's manual download (Phase 2).
- No Upload Image tab yet — deferred to Phase 6 (document/upload pipeline) rather than
  duplicated here ahead of segmentation/multi-region support.
- 114 tests total (up from 91), all mocking the heavy model in GUI/unit tests — nothing
  in the automated suite downloads a model or requires network. `ruff`/`mypy` clean.

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
