# Roadmap

This project is built incrementally. Each phase must be fully working — app runs,
tests pass, lint/type-check clean — before the next one starts.

## Decisions on record

- **Hardware**: development is currently on a Windows 11 laptop with an NVIDIA GeForce
  RTX 3050 Laptop GPU (torch cu126 build, verified 2026-07-31). This is a modest GPU,
  not a training rig — enough to fine-tune TrOCR-small with AMP, not enough to train a
  large model from scratch. The project's earlier development on a CPU-only machine
  (Intel iGPU, no discrete GPU) is the reason the modeling strategy avoids
  training-from-scratch approaches. `AppConfig.resolved_device()` auto-selects
  `cuda` → `mps` → `cpu`, so the same codebase runs correctly on all three; **nothing
  may *require* a GPU** (CPU fallback must stay functional and tested).
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

## Phase 4 — Fine-tuning, confusion analysis, hard-negative mining ✅ (complete)

- Fine-tuning loop (`training/train.py`, config in `configs/training.yaml` /
  `training/config.py`): loads TrOCR-small (or a checkpoint, if resuming), fine-tunes on
  any combination of manifests/splits, checkpoints every `save_every_n_steps` under
  `weights/<checkpoint_dir>/step-<N>/` (never overwritten in place) plus a final save.
  Required setting `model.config.decoder_start_token_id` /
  `pad_token_id` explicitly — `VisionEncoderDecoderModel` doesn't infer these for loss
  computation, a real bug caught by actually running training, not just unit tests.
  **Verified for real**: ran two sequential training passes on 200 synthetic+MNIST
  samples (batch size 4) — loss dropped ~4.9 → ~1.0–1.6 over 50 steps in run 1; run 2
  correctly resumed from `step-50` (no "newly initialized" warning, confirming real
  weights loaded) and continued to `step-100` without resetting the step counter or
  duplicating log rows. Known simplification: only model+processor are checkpointed, not
  optimizer state, so resume restarts Adam's momentum — documented in `train.py`, not
  hidden.
- `training/dataset.py`: wraps the Phase 2 manifest format as a PyTorch `Dataset`,
  applying the Phase 2 augmentation pipeline on the fly per `__getitem__` call (not
  baked into stored images), with `-100`-masked padding for the loss.
- `training/logging_utils.py`: `TrainingLogger` writes every scalar to TensorBoard, a
  CSV, and a JSONL file simultaneously — CSV/JSONL never depend on TensorBoard
  succeeding (wrapped in its own try/except). Added the `log` dependency group
  (`tensorboard`) to the synced extras.
- `training/confusion_matrix.py`: full Levenshtein *alignment* (not just distance) to
  attribute substitutions to specific (reference-char, hypothesis-char) pairs;
  `analyze_ambiguous_classes` reports the single-character pairs from this phase's
  original bullet list (0/O, 1/l, 1/I, 5/S, 2/Z, 8/B, 6/G, 9/g, O/Q, C/G). Multi-character
  pairs (rn/m, cl/d, vv/w) are **not** covered — a character-aligned matrix can't express
  a 2-char-vs-1-char substitution; that needs substring alignment, an explicitly
  documented gap for a future iteration, not silently dropped.
- `training/hard_negative_mining.py` + `scripts/analyze_confusions.py`: the latter runs
  the recognizer over a manifest split and saves a real confusion matrix to
  `experiments/`; the former oversamples training samples containing frequently-confused
  characters. **Verified for real**, not just unit-tested: ran `analyze_confusions.py`
  against 40 real synthetic character samples, then fed that matrix into a training run
  with `hard_negative_min_count=1` — it correctly identified 26 flagged classes and
  oversampled 60 → 135 training samples before training proceeded. With the default
  `min_count=3` and this little data, no pairs currently cross the threshold (too few
  samples for repeated confusions) — that's correct behavior given the data, not a bug;
  the mechanism is proven, its usefulness will show once real handwriting data (IAM/CVL)
  feeds larger evaluation runs.
- 150 tests total (up from 114). GUI/unit tests never load the real model (mocked); the
  training/confusion/logging pieces above were additionally validated with real runs
  documented above, beyond what's captured in the automated suite. `ruff`/`mypy` clean
  across 44 source files.
- **Verified on real handwriting (CVL, RTX 3050, 2026-07-31)** — closing the two
  previously-open gaps. Downloaded CVL (word-level, 10,702 train / 72,576 test), fine-tuned
  TrOCR-small for 1 epoch on CVL train (CUDA + AMP, batch 4, ~15 min), then ran the same
  training with hard-negative mining feeding a confusion matrix built from the plain
  fine-tune's train-split errors (raised `hard_negative_min_count` to 10 to keep the
  oversampling selective — flagged `{a, f, o, t}` from the top confusion pairs, oversampled
  10,702→27,734 samples, ~30 min training). Results on a fixed 500-sample CVL test slice:

  | Metric | Baseline (pretrained) | Plain FT | HN-mined FT |
  |---|---|---|---|
  | CER | 0.482 | 1.578 | **0.314** |
  | WER | 0.740 | 1.280 | **0.354** |
  | word_acc | 0.376 | 0.434 | **0.646** |
  | exact_match | 0.376 | 0.434 | **0.646** |

  Plain FT improved exact-match (+5.8pp) but degraded CER catastrophically — spot-checks
  revealed **token-repetition degeneracy** on uncertain inputs (e.g. `on` → 32× "on"),
  which mathematically pushes CER above 1.0 via runaway insertions. HN mining reversed this:
  by oversampling the top confused characters it apparently regularized the model's
  calibration on uncertain inputs, replacing 32-token repetitions with either empty strings
  (safe give-up) or milder trailing-char repetition (`Triangles` → `Trianglesssss`). Net
  result: HN mining beats the pretrained baseline on all five metrics.
  `configs/recognition.yaml: model_name` now points at the HN-mined checkpoint
  (`weights/trocr-cvl-hn/step-6934`); the weights themselves are gitignored — reproduce
  by re-running `uv run python -m training.train` after preparing CVL. The remaining
  trailing-char repetition is a candidate for an inference-time
  `no_repeat_ngram_size` fix (not applied yet — deferred as a Phase 4.5 followup).

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
