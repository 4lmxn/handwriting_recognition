"""Before/after evaluation + result dataclasses for Phase 5 incremental
adapter updates.

Every adapter increment is scored on a held-out slice of the base
manifest twice — once with the pre-update recognizer, once with the
post-update recognizer. If the CER regression exceeds
`max_cer_regression`, the new adapter is REJECTED: saved for
inspection under `v-<ts>-REJECTED/` but never made active, and the
corrections that fed it stay `pending` so a later, better attempt can
retry them.

Kept in its own module so `training/incremental.py` can import both
the metrics-collection helper and the result schema without pulling in
peft / transformers just to read a JSONL entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from datasets.manifest import DatasetSample
from training.evaluation import evaluate_predictions

if TYPE_CHECKING:
    from recognition.recognizer import Recognizer


@dataclass(frozen=True)
class EvaluationMetrics:
    cer: float
    wer: float
    char_acc: float
    word_acc: float
    exact_match: float
    num_samples: int

    @classmethod
    def zero(cls) -> EvaluationMetrics:
        return cls(cer=0.0, wer=0.0, char_acc=0.0, word_acc=0.0, exact_match=0.0, num_samples=0)


@dataclass(frozen=True)
class IncrementalUpdateResult:
    version: str  # e.g. "v-1785500000-a1b2c3d4"; REJECTED versions append "-REJECTED"
    timestamp: str  # ISO-8601 UTC
    correction_ids: tuple[str, ...]  # ids replayed in this update
    corrections_new: int  # of the above, how many were pending before this run
    corrections_replayed: int  # how many were already-applied (replay-all policy)
    base_samples: int  # base-manifest samples in the training mix
    before: EvaluationMetrics
    after: EvaluationMetrics
    cer_delta: float  # after.cer - before.cer (positive = worse)
    rejected: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["correction_ids"] = list(self.correction_ids)
        return d


def evaluate_recognizer(
    recognizer: Recognizer,
    samples: list[DatasetSample],
    processed_dir: Path,
) -> EvaluationMetrics:
    """Score `recognizer` against `samples`, returning aggregate metrics.

    Mirrors the loop in `scripts/evaluate_model.py` — single-image
    inference is fine at 100-sample scale (~10-20 sec on GPU) and keeps
    the code path identical to what users see when running the eval
    script by hand.
    """
    if not samples:
        return EvaluationMetrics.zero()

    refs: list[str] = []
    hyps: list[str] = []
    for sample in samples:
        image = cv2.imread(str(processed_dir / sample.image_path), cv2.IMREAD_GRAYSCALE)
        result = recognizer.recognize(image)
        refs.append(sample.transcript)
        hyps.append(result.text)

    m = evaluate_predictions(refs, hyps)
    return EvaluationMetrics(
        cer=m.mean_cer,
        wer=m.mean_wer,
        char_acc=m.mean_char_accuracy,
        word_acc=m.mean_word_accuracy,
        exact_match=m.exact_match_rate,
        num_samples=m.num_samples,
    )


def append_update_log(log_path: Path, result: IncrementalUpdateResult) -> None:
    """Append `result` as a single JSONL row to `log_path`.

    Creates parent dirs if needed. Each incremental update writes exactly
    one line here — rejected + accepted both get logged so the update
    history is complete.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")
