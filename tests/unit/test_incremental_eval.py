import json
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from datasets.manifest import DatasetSample
from recognition.recognizer import RecognitionResult
from training.incremental_eval import (
    EvaluationMetrics,
    IncrementalUpdateResult,
    append_update_log,
    evaluate_recognizer,
)


def _sample(name: str) -> DatasetSample:
    return DatasetSample(
        image_path=f"feedback/{name}.png",
        transcript=name,
        source="feedback",
        split="train",
        label_type="word",
        writer_id=None,
    )


def test_evaluation_metrics_zero_has_no_samples():
    m = EvaluationMetrics.zero()
    assert m.num_samples == 0
    assert m.cer == 0.0


def test_incremental_update_result_to_dict_is_json_serializable():
    result = IncrementalUpdateResult(
        version="v-1785500000-a1b2c3d4",
        timestamp="2026-07-31T17:00:00+00:00",
        correction_ids=("feedback/a.png", "feedback/b.png"),
        corrections_new=2,
        corrections_replayed=0,
        base_samples=8,
        before=EvaluationMetrics(
            cer=0.3, wer=0.4, char_acc=0.7, word_acc=0.6, exact_match=0.5, num_samples=10
        ),
        after=EvaluationMetrics(
            cer=0.25, wer=0.35, char_acc=0.75, word_acc=0.65, exact_match=0.55, num_samples=10
        ),
        cer_delta=-0.05,
        rejected=False,
    )
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    # correction_ids must round-trip as a list (JSON has no tuples)
    assert decoded["correction_ids"] == ["feedback/a.png", "feedback/b.png"]
    assert decoded["cer_delta"] == pytest.approx(-0.05)
    assert decoded["rejected"] is False
    assert decoded["rejection_reason"] is None


def test_evaluate_recognizer_returns_zero_metrics_for_empty_sample_list():
    fake_recognizer = MagicMock()
    m = evaluate_recognizer(fake_recognizer, [], processed_dir=MagicMock())
    assert m == EvaluationMetrics.zero()
    fake_recognizer.recognize.assert_not_called()


def test_evaluate_recognizer_scores_per_sample(tmp_path):
    # Write two tiny PNGs so cv2.imread can read them back
    img = np.full((8, 32), 255, dtype=np.uint8)
    (tmp_path / "feedback").mkdir()
    cv2.imwrite(str(tmp_path / "feedback" / "a.png"), img)
    cv2.imwrite(str(tmp_path / "feedback" / "b.png"), img)

    samples = [_sample("a"), _sample("b")]

    # Fake recognizer: predicts exactly for the first sample, wrong for the second
    fake_recognizer = MagicMock()
    fake_recognizer.recognize.side_effect = [
        RecognitionResult(text="a", confidence=0.9),
        RecognitionResult(text="z", confidence=0.4),
    ]

    metrics = evaluate_recognizer(fake_recognizer, samples, processed_dir=tmp_path)
    assert metrics.num_samples == 2
    # exact_match: 1/2 = 0.5; char_acc: (1.0 + 0.0) / 2 = 0.5
    assert metrics.exact_match == pytest.approx(0.5)
    assert metrics.char_acc == pytest.approx(0.5)


def test_append_update_log_writes_jsonl_and_creates_parents(tmp_path):
    log_path = tmp_path / "nested" / "updates.jsonl"
    r = IncrementalUpdateResult(
        version="v-1-abc",
        timestamp="t",
        correction_ids=(),
        corrections_new=0,
        corrections_replayed=0,
        base_samples=0,
        before=EvaluationMetrics.zero(),
        after=EvaluationMetrics.zero(),
        cer_delta=0.0,
        rejected=True,
        rejection_reason="test",
    )
    append_update_log(log_path, r)
    append_update_log(log_path, r)  # append is idempotent-safe

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["version"] == "v-1-abc"
    assert json.loads(lines[1])["rejection_reason"] == "test"
