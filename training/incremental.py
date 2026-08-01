"""Incremental adapter training loop for Phase 5 continual learning.

Given the user's accumulated corrections in the `FeedbackStore`, this
module:

  1. loads the base recognizer (from `recognition_config.model_name`, i.e.
     the current committed default — the Phase 4 HN-mined CVL checkpoint);
  2. wraps it in a LoRA adapter (fresh, or loaded from the newest existing
     version so personalization compounds across updates);
  3. evaluates the pre-update recognizer on a held-out eval slice —
     `before` metrics;
  4. mixes corrections with a random slice of base-manifest data via the
     replay buffer to prevent catastrophic forgetting;
  5. runs a short training pass (adapter params only — base stays frozen);
  6. saves the new adapter as `<adapter_dir>/v-<epoch>-<uuid>/` (never
     overwritten);
  7. evaluates the post-update recognizer — `after` metrics;
  8. gates on `after.cer - before.cer > max_cer_regression`. Regression
     → save as `v-...-REJECTED/`, do NOT mark corrections applied, so
     a later attempt can retry them;
  9. on acceptance: calls `store.mark_applied(pending_ids, version)`;
  10. appends the `IncrementalUpdateResult` to `feedback/updates.jsonl`.

Runs on `resolve_device()` — CPU fallback is functional (slower) so tests
that mock out the model can run anywhere.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import AppConfig
from datasets.manifest import DatasetSample, read_manifest
from feedback.config import FeedbackConfig
from feedback.store import FeedbackStore
from models.adapters.lora import apply_lora_to_trocr, load_adapter, save_adapter
from models.adapters.resolver import find_latest_adapter
from preprocessing.augmentation import build_augmentation_pipeline, load_augmentation_config
from recognition.config import RecognitionConfig
from recognition.recognizer import Recognizer
from training.dataset import HandwritingDataset
from training.incremental_eval import (
    IncrementalUpdateResult,
    append_update_log,
    evaluate_recognizer,
)
from training.logging_utils import TrainingLogger
from training.replay_buffer import build_replay_batch, load_replay_base
from training.train import amp_enabled_for

logger = logging.getLogger(__name__)


def _make_version_name() -> str:
    return f"v-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _train_adapter(
    peft_model: torch.nn.Module,
    processor: TrOCRProcessor,
    samples: list[DatasetSample],
    processed_dir: Path,
    training_config,
    device: str,
    train_logger: TrainingLogger,
) -> None:
    """Mini training loop — mirrors the shape of `training/train.py` but
    only touches adapter params (base is frozen)."""
    augmentation_pipeline = None
    if training_config.augment:
        augmentation_pipeline = build_augmentation_pipeline(load_augmentation_config())

    dataset = HandwritingDataset(
        samples,
        processed_dir,
        processor,
        training_config.max_target_length,
        augmentation_pipeline,
    )
    dataloader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=training_config.learning_rate)

    amp_enabled = amp_enabled_for(device, training_config.use_amp)
    amp_device = "cuda" if amp_enabled else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)
    logger.info("Adapter train: device=%s, mixed precision (fp16)=%s", device, amp_enabled)

    peft_model.train()
    step = 0
    for epoch in range(training_config.num_epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=amp_enabled):
                outputs = peft_model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            train_logger.log_scalar("adapter/loss", loss.item(), step)
            logger.info("adapter epoch=%d step=%d loss=%.4f", epoch, step, loss.item())
    peft_model.eval()


def _cap_corrections(samples: list[DatasetSample], cap: int) -> list[DatasetSample]:
    """Keep the most-recent `cap` samples. `store.to_dataset_samples()`
    preserves insertion order, so the tail is the newest — see
    `feedback/store.py::all()`."""
    if cap <= 0 or len(samples) <= cap:
        return samples
    return samples[-cap:]


def _load_eval_samples(
    manifest_name: str,
    split: str,
    limit: int,
    manifests_dir: Path,
) -> list[DatasetSample]:
    manifest_path = manifests_dir / f"{manifest_name}.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Eval manifest {manifest_path} not found — run scripts/prepare_dataset.py first."
        )
    return [s for s in read_manifest(manifest_path) if s.split == split][:limit]


def train_adapter_increment(
    feedback_config: FeedbackConfig,
    recognition_config: RecognitionConfig,
    app_config: AppConfig,
    feedback_store: FeedbackStore,
) -> IncrementalUpdateResult | None:
    """Run one adapter update. Returns the result dataclass, or `None` if
    we skipped because fewer than `min_pending_corrections` were pending.

    Every path (skip, accept, reject) is logged; only the accept path
    modifies FeedbackStore state.
    """
    pending = feedback_store.pending()
    if len(pending) < feedback_config.replay.min_pending_corrections:
        logger.info(
            "Skipping adapter update: only %d pending corrections (min=%d)",
            len(pending),
            feedback_config.replay.min_pending_corrections,
        )
        return None

    replay_cfg = feedback_config.replay
    training_cfg = feedback_config.training
    eval_cfg = feedback_config.eval
    device = training_cfg.resolved_device()

    random.seed(replay_cfg.seed)
    np.random.seed(replay_cfg.seed)
    torch.manual_seed(replay_cfg.seed)

    # ---- 1. Load base + wrap in adapter ---------------------------------
    logger.info("Loading base model from %s", recognition_config.model_name)
    processor = TrOCRProcessor.from_pretrained(recognition_config.model_name)
    base_model = VisionEncoderDecoderModel.from_pretrained(recognition_config.model_name)
    # Same explicit token-id setup as training/train.py — VisionEncoderDecoderModel
    # can't infer these on its own for loss computation.
    base_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # type: ignore[attr-defined]
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id  # type: ignore[attr-defined]

    latest_adapter = find_latest_adapter(feedback_config.adapter_dir_path)
    if latest_adapter is not None:
        logger.info("Continuing from adapter %s", latest_adapter.name)
        peft_model = load_adapter(base_model, latest_adapter)
        # `load_adapter` may leave lora params requires_grad=False (peft
        # loads in inference mode by default) — flip them back on.
        for name, param in peft_model.named_parameters():
            param.requires_grad = "lora_" in name
    else:
        logger.info("No prior adapter — starting fresh LoRA layer")
        peft_model = apply_lora_to_trocr(base_model, feedback_config.adapter)
    peft_model.to(device)

    # ---- 2. Build eval + training splits --------------------------------
    eval_samples = _load_eval_samples(
        eval_cfg.manifest, eval_cfg.split, eval_cfg.limit, app_config.paths.datasets_manifests
    )
    if not eval_samples:
        raise ValueError(
            f"No eval samples for manifest={eval_cfg.manifest} split={eval_cfg.split}"
        )

    all_correction_samples = _cap_corrections(
        feedback_store.to_dataset_samples(), replay_cfg.max_corrections
    )
    pending_ids = {p.id for p in pending}
    base_samples = load_replay_base(
        replay_cfg.base_manifests,
        replay_cfg.base_splits,
        app_config.paths.datasets_manifests,
    )
    training_samples = build_replay_batch(
        feedback_samples=all_correction_samples,
        base_samples=base_samples,
        replay_ratio=replay_cfg.replay_ratio,
        max_total=replay_cfg.max_total_samples,
        seed=replay_cfg.seed,
    )
    if not training_samples:
        raise ValueError("Replay buffer produced an empty batch — nothing to train on.")

    # ---- 3. BEFORE eval -------------------------------------------------
    logger.info("Running BEFORE eval on %d samples", len(eval_samples))
    before_recognizer = _build_recognizer_from_peft(
        peft_model, processor, recognition_config, device
    )
    before_metrics = evaluate_recognizer(
        before_recognizer, eval_samples, app_config.paths.datasets_processed
    )
    logger.info("BEFORE: %s", before_metrics)

    # ---- 4. Train the adapter -------------------------------------------
    run_name = f"{training_cfg.run_name_prefix}-{int(time.time())}"
    with TrainingLogger(training_cfg.log_dir_path, run_name) as train_logger:
        _train_adapter(
            peft_model=peft_model,
            processor=processor,
            samples=training_samples,
            processed_dir=app_config.paths.datasets_processed,
            training_config=training_cfg,
            device=device,
            train_logger=train_logger,
        )

    # ---- 5. Save the new version (name may get REJECTED suffix later) ---
    version = _make_version_name()
    adapter_path = feedback_config.adapter_dir_path / version
    save_adapter(peft_model, adapter_path)

    # ---- 6. AFTER eval + regression gate --------------------------------
    logger.info("Running AFTER eval on %d samples", len(eval_samples))
    after_recognizer = _build_recognizer_from_peft(
        peft_model, processor, recognition_config, device
    )
    after_metrics = evaluate_recognizer(
        after_recognizer, eval_samples, app_config.paths.datasets_processed
    )
    logger.info("AFTER: %s", after_metrics)

    cer_delta = after_metrics.cer - before_metrics.cer
    rejected = cer_delta > eval_cfg.max_cer_regression
    rejection_reason: str | None = None
    if rejected:
        rejection_reason = (
            f"CER regressed by {cer_delta:.4f} (limit {eval_cfg.max_cer_regression:.4f})"
        )
        rejected_path = adapter_path.with_name(adapter_path.name + "-REJECTED")
        adapter_path.rename(rejected_path)
        adapter_path = rejected_path
        logger.warning(
            "Adapter REJECTED: %s. Saved for inspection at %s",
            rejection_reason,
            adapter_path,
        )
    else:
        # Only mark applied on acceptance. Rejected corrections stay
        # pending so a later, better attempt can retry them.
        feedback_store.mark_applied(list(pending_ids), version=adapter_path.name)
        logger.info("Adapter accepted (cer_delta=%.4f). Marked %d corrections applied.",
                    cer_delta, len(pending_ids))

    result = IncrementalUpdateResult(
        version=adapter_path.name,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        correction_ids=tuple(s.image_path for s in all_correction_samples),
        corrections_new=len(pending_ids),
        corrections_replayed=len(all_correction_samples) - len(pending_ids),
        base_samples=len(training_samples) - len(all_correction_samples),
        before=before_metrics,
        after=after_metrics,
        cer_delta=cer_delta,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )
    append_update_log(feedback_config.storage_dir_path / "updates.jsonl", result)
    return result


def _build_recognizer_from_peft(
    peft_model: torch.nn.Module,
    processor: TrOCRProcessor,
    recognition_config: RecognitionConfig,
    device: str,
) -> Recognizer:
    """Wrap the currently-in-memory (base+adapter) model in a Recognizer
    for eval, without re-loading from disk.

    Recognizer's __init__ always loads from a `model_name` string, but
    here we already have a live PeftModel. Assembling the recognizer
    "by hand" via __new__ + attribute assignment sidesteps the reload —
    important because reloading base+adapter for every before/after eval
    would add ~20s per update on a fresh cache.
    """
    peft_model.eval()
    recognizer = Recognizer.__new__(Recognizer)
    recognizer._device = device  # noqa: SLF001
    recognizer._max_new_tokens = recognition_config.max_new_tokens  # noqa: SLF001
    recognizer._repetition_penalty = recognition_config.repetition_penalty  # noqa: SLF001
    recognizer._no_repeat_ngram_size = recognition_config.no_repeat_ngram_size  # noqa: SLF001
    recognizer._processor = processor  # noqa: SLF001
    recognizer._model = peft_model  # noqa: SLF001
    return recognizer
