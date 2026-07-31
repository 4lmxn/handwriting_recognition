"""Storage layer for user-provided corrections (Phase 5, PR 1).

Each correction pairs a source image with the transcript the user actually
meant, along with what the recognizer originally predicted. Records are
JSONL under `<storage_dir>/corrections.jsonl`; images are separate PNGs
under `<image_dir>/<uuid>.png`. Kept apart so the JSONL stays cheap to
inspect (`cat`, `wc -l`, `jq`) even after thousands of corrections.

`to_dataset_samples()` shapes stored records as manifest-compatible
`DatasetSample`s (same schema every other dataset source produces),
letting the future incremental training loop feed corrections through
the existing HandwritingDataset with no branching.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from datasets.manifest import DatasetSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    timestamp: str  # ISO-8601 UTC, seconds precision is fine
    image_path: str  # relative to the datasets/processed dir (so a
    # DatasetSample built from this record works with HandwritingDataset)
    original_prediction: str
    original_confidence: float
    corrected_transcript: str
    source: str  # "drawing_canvas" for now; "upload" etc. added later
    applied_in_version: str | None = None  # None = pending; the adapter
    # version dir name (e.g. "v-1785500000-a1b2c3d4") once the record has
    # been consumed by an incremental training run


class FeedbackStore:
    """JSONL-backed store for user corrections.

    Not thread-safe: two concurrent `add()` calls could interleave lines
    or two `mark_applied()` calls could race on the rewrite. Phase 5's
    correction flow is single-writer (drawing-canvas main thread, CLI
    training run) so this is fine; if the Phase 9 Feedback Manager tab
    ever grows a background thread we'll revisit.
    """

    def __init__(
        self,
        storage_dir: Path,
        image_dir: Path,
        image_relative_prefix: str = "feedback",
    ) -> None:
        self._storage_dir = storage_dir
        self._image_dir = image_dir
        # image_path in each record is stored as
        #     "<image_relative_prefix>/<uuid>.png"
        # so that when the training pipeline joins it against
        # datasets_processed the result resolves to <image_dir>/<uuid>.png.
        # This assumes image_dir sits under datasets_processed with a
        # matching final component — the default configs/feedback.yaml
        # enforces that (image_dir = datasets/processed/feedback).
        self._image_relative_prefix = image_relative_prefix
        self._corrections_path = storage_dir / "corrections.jsonl"

    def add(
        self,
        image: np.ndarray,
        prediction: str,
        confidence: float,
        corrected: str,
        source: str = "drawing_canvas",
    ) -> FeedbackRecord:
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        image_filename = f"{record_id}.png"

        self._image_dir.mkdir(parents=True, exist_ok=True)
        image_target = self._image_dir / image_filename
        # cv2.imwrite can either return False (unsupported ext, permission
        # denied) or raise cv2.error (empty ndarray, malformed shape). Either
        # way we surface it as OSError so callers can catch one exception type.
        try:
            write_ok = cv2.imwrite(str(image_target), image)
        except cv2.error as exc:
            raise OSError(
                f"cv2.imwrite failed for {image_target} — "
                "check image dtype/shape (expected grayscale uint8 ndarray)."
            ) from exc
        if not write_ok:
            raise OSError(
                f"cv2.imwrite failed for {image_target} — "
                "check image dtype/shape (expected grayscale uint8 ndarray)."
            )

        record = FeedbackRecord(
            id=record_id,
            timestamp=timestamp,
            image_path=f"{self._image_relative_prefix}/{image_filename}",
            original_prediction=prediction,
            original_confidence=confidence,
            corrected_transcript=corrected,
            source=source,
        )
        self._append(record)
        return record

    def all(self) -> list[FeedbackRecord]:
        if not self._corrections_path.exists():
            return []
        records: list[FeedbackRecord] = []
        with open(self._corrections_path) as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(FeedbackRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        "Skipping malformed correction at %s:%d — %s",
                        self._corrections_path,
                        line_num,
                        e,
                    )
        return records

    def pending(self) -> list[FeedbackRecord]:
        return [r for r in self.all() if r.applied_in_version is None]

    def mark_applied(self, ids: list[str], version: str) -> None:
        """Flip `applied_in_version` on the records whose id is in `ids`.

        Rewrites the full JSONL via a tmp file + rename so a crash mid-
        write can't leave a half-updated corrections log. Full-file
        rewrite is fine at expected scale (< a few thousand records).
        """
        id_set = set(ids)
        records = self.all()
        updated = [
            replace(r, applied_in_version=version) if r.id in id_set else r
            for r in records
        ]
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._corrections_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w") as f:
            for r in updated:
                f.write(json.dumps(asdict(r)) + "\n")
        tmp_path.replace(self._corrections_path)

    def to_dataset_samples(self) -> list[DatasetSample]:
        """Every stored correction, shaped for training/dataset.py.

        Returns all corrections (applied + pending) so the incremental
        training loop can replay the full personalization history each
        run — see the "replay all corrections" decision recorded for
        Phase 5. Callers wanting only new corrections can filter via
        `pending()` themselves.
        """
        return [
            DatasetSample(
                image_path=r.image_path,
                transcript=r.corrected_transcript,
                source="feedback",
                split="train",
                label_type="word",
                writer_id=None,
            )
            for r in self.all()
        ]

    def _append(self, record: FeedbackRecord) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self._corrections_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
