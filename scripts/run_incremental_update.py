"""Run one adapter update from accumulated feedback (Phase 5, PR 3).

Loads the FeedbackStore, checks whether enough pending corrections have
accumulated, runs `train_adapter_increment` if so, and reports the
before/after metrics + accept/reject status.

Usage:
    uv run python scripts/run_incremental_update.py

Unlike Phase 4 full fine-tuning, this is safe to run frequently — the
base model isn't touched, each new adapter version is a few MB, and a
regressing adapter is silently rejected rather than made active. See
docs/ROADMAP.md Phase 5 for the design.
"""

from __future__ import annotations

import logging

from app.config import load_config
from app.logging_config import setup_logging
from feedback.config import load_feedback_config
from feedback.store import FeedbackStore
from recognition.config import load_recognition_config
from training.incremental import train_adapter_increment

logger = logging.getLogger(__name__)


def main() -> None:
    app_config = load_config()
    app_config.paths.ensure_exist()
    setup_logging(
        app_config.paths.logs, app_config.log_level, filename="incremental_update.log"
    )

    feedback_config = load_feedback_config()
    recognition_config = load_recognition_config()

    store = FeedbackStore(
        storage_dir=feedback_config.storage_dir_path,
        image_dir=feedback_config.image_dir_path,
    )

    result = train_adapter_increment(
        feedback_config=feedback_config,
        recognition_config=recognition_config,
        app_config=app_config,
        feedback_store=store,
    )

    if result is None:
        logger.info("No adapter update run this invocation.")
        return

    verdict = "REJECTED" if result.rejected else "ACCEPTED"
    logger.info(
        "Adapter version %s %s. CER %.4f -> %.4f (delta %+.4f). "
        "%d corrections in mix (%d new, %d replayed), %d base samples.",
        result.version,
        verdict,
        result.before.cer,
        result.after.cer,
        result.cer_delta,
        result.corrections_new + result.corrections_replayed,
        result.corrections_new,
        result.corrections_replayed,
        result.base_samples,
    )
    if result.rejected:
        logger.warning("Rejection reason: %s", result.rejection_reason)


if __name__ == "__main__":
    main()
