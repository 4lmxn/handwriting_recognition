"""Unit tests for training/incremental.py.

The orchestrator loads real models + runs real training in production;
these tests patch every heavy dependency so we can verify the sequencing
(skip vs accept vs reject) and the FeedbackStore side effects in
milliseconds without a GPU or a HF download.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datasets.manifest import DatasetSample
from feedback.config import (
    FeedbackConfig,
    IncrementalEvalConfig,
    IncrementalTrainingConfig,
    ReplayConfig,
)
from feedback.store import FeedbackRecord, FeedbackStore
from models.adapters.config import LoraAdapterConfig
from recognition.config import RecognitionConfig
from training.incremental import (
    _cap_corrections,
    _make_version_name,
    train_adapter_increment,
)
from training.incremental_eval import EvaluationMetrics

# ---------- helper unit tests --------------------------------------------
# find_latest_adapter tests moved to tests/unit/test_adapter_resolver.py
# when the function moved out of training/incremental.py — see that file.


def test_make_version_name_format():
    name = _make_version_name()
    parts = name.split("-")
    assert parts[0] == "v"
    assert parts[1].isdigit()
    # short uuid is 8 hex chars
    assert len(parts[2]) == 8
    assert all(c in "0123456789abcdef" for c in parts[2])


def test_cap_corrections_returns_input_when_below_cap():
    samples = [MagicMock() for _ in range(5)]
    assert _cap_corrections(samples, cap=100) is samples


def test_cap_corrections_keeps_most_recent():
    # Insertion order is oldest -> newest; we want the tail.
    samples = [MagicMock(name=f"s{i}") for i in range(10)]
    capped = _cap_corrections(samples, cap=3)
    assert len(capped) == 3
    assert capped == samples[-3:]


def test_cap_corrections_returns_input_when_cap_zero_or_negative():
    samples = [MagicMock()]
    assert _cap_corrections(samples, cap=0) is samples
    assert _cap_corrections(samples, cap=-1) is samples


# ---------- orchestration tests ------------------------------------------


def _sample(name: str, source: str = "cvl") -> DatasetSample:
    return DatasetSample(
        image_path=f"{source}/{name}.png",
        transcript=name,
        source=source,
        split="train",
        label_type="word",
        writer_id=None,
    )


def _build_feedback_config(
    tmp_path, min_pending: int = 5, max_regression: float = 0.02
) -> FeedbackConfig:
    return FeedbackConfig(
        storage_dir=str(tmp_path / "feedback"),
        image_dir=str(tmp_path / "images"),
        adapter=LoraAdapterConfig(),
        adapter_dir=str(tmp_path / "weights" / "adapters"),
        replay=ReplayConfig(
            base_manifests=("cvl",),
            base_splits=("train",),
            replay_ratio=0.7,
            max_total_samples=100,
            min_pending_corrections=min_pending,
            max_corrections=200,
            seed=42,
        ),
        training=IncrementalTrainingConfig(
            device="cpu",
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-4,
            max_target_length=32,
            use_amp=False,
            augment=False,
            log_dir=str(tmp_path / "logs"),
            run_name_prefix="test-adapter",
        ),
        eval=IncrementalEvalConfig(
            manifest="cvl",
            split="test",
            limit=5,
            max_cer_regression=max_regression,
        ),
    )


def _fake_pending_records(n: int) -> list[FeedbackRecord]:
    return [
        FeedbackRecord(
            id=f"id-{i}",
            timestamp=f"t{i}",
            image_path=f"feedback/{i}.png",
            original_prediction="x",
            original_confidence=0.5,
            corrected_transcript="y",
            source="drawing_canvas",
        )
        for i in range(n)
    ]


def _build_recognition_config() -> RecognitionConfig:
    return RecognitionConfig(
        model_name="microsoft/trocr-small-handwritten",
        device="cpu",
        max_new_tokens=32,
    )


def _build_app_config(tmp_path) -> MagicMock:
    app = MagicMock()
    app.paths.datasets_manifests = tmp_path / "manifests"
    app.paths.datasets_processed = tmp_path / "processed"
    return app


def test_skips_when_pending_below_min(tmp_path):
    feedback_cfg = _build_feedback_config(tmp_path, min_pending=5)
    store = MagicMock(spec=FeedbackStore)
    store.pending.return_value = _fake_pending_records(3)  # < min

    result = train_adapter_increment(
        feedback_config=feedback_cfg,
        recognition_config=_build_recognition_config(),
        app_config=_build_app_config(tmp_path),
        feedback_store=store,
    )
    assert result is None
    store.mark_applied.assert_not_called()


def _save_adapter_that_creates_dir(_model, path):
    """save_adapter mock side_effect — creates the target dir on disk so
    the follow-up rename (on reject) has something to rename. Matches the
    real save_adapter's behavior of ensuring the dir exists."""
    path.mkdir(parents=True, exist_ok=True)


@patch("training.incremental.append_update_log")
@patch("training.incremental._build_recognizer_from_peft")
@patch("training.incremental.evaluate_recognizer")
@patch("training.incremental._train_adapter")
@patch("training.incremental.save_adapter", side_effect=_save_adapter_that_creates_dir)
@patch("training.incremental.load_replay_base")
@patch("training.incremental.read_manifest")
@patch("training.incremental.apply_lora_to_trocr")
@patch("training.incremental.VisionEncoderDecoderModel")
@patch("training.incremental.TrOCRProcessor")
def test_accept_path_marks_applied_and_logs(
    mock_processor,
    mock_vem,
    mock_apply_lora,
    mock_read_manifest,
    mock_load_base,
    mock_save_adapter,
    mock_train,
    mock_eval,
    mock_build_recognizer,
    mock_append_log,
    tmp_path,
):
    feedback_cfg = _build_feedback_config(tmp_path, max_regression=0.02)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "cvl.jsonl").write_text("")  # existence check

    pending = _fake_pending_records(5)
    store = MagicMock(spec=FeedbackStore)
    store.pending.return_value = pending
    store.to_dataset_samples.return_value = [_sample(f"c{i}", "feedback") for i in range(5)]

    mock_read_manifest.return_value = [
        DatasetSample(f"cvl/e{i}.png", f"e{i}", "cvl", "test", "word", None) for i in range(5)
    ]
    mock_load_base.return_value = [_sample(f"b{i}") for i in range(50)]

    mock_apply_lora.return_value = MagicMock()

    # BEFORE=0.30, AFTER=0.29 -> delta=-0.01 <= 0.02 -> accepted
    mock_eval.side_effect = [
        EvaluationMetrics(0.30, 0.35, 0.70, 0.65, 0.60, 5),
        EvaluationMetrics(0.29, 0.34, 0.71, 0.66, 0.61, 5),
    ]
    mock_build_recognizer.return_value = MagicMock()

    result = train_adapter_increment(
        feedback_config=feedback_cfg,
        recognition_config=_build_recognition_config(),
        app_config=_build_app_config(tmp_path),
        feedback_store=store,
    )

    assert result is not None
    assert result.rejected is False
    assert result.rejection_reason is None
    assert result.cer_delta == pytest.approx(-0.01, abs=1e-6)
    assert result.corrections_new == 5
    assert result.corrections_replayed == 0

    # mark_applied is the load-bearing side-effect of the accept path
    store.mark_applied.assert_called_once()
    called_ids, called_kwargs = store.mark_applied.call_args
    marked_ids = called_ids[0] if called_ids else called_kwargs["ids"]
    assert set(marked_ids) == {p.id for p in pending}

    # Update log written
    mock_append_log.assert_called_once()

    # Adapter dir was saved and NOT renamed to REJECTED
    save_call_path: Path = mock_save_adapter.call_args[0][1]
    assert save_call_path.exists()
    assert not save_call_path.name.endswith("REJECTED")


@patch("training.incremental.append_update_log")
@patch("training.incremental._build_recognizer_from_peft")
@patch("training.incremental.evaluate_recognizer")
@patch("training.incremental._train_adapter")
@patch("training.incremental.save_adapter", side_effect=_save_adapter_that_creates_dir)
@patch("training.incremental.load_replay_base")
@patch("training.incremental.read_manifest")
@patch("training.incremental.apply_lora_to_trocr")
@patch("training.incremental.VisionEncoderDecoderModel")
@patch("training.incremental.TrOCRProcessor")
def test_reject_path_renames_adapter_and_leaves_corrections_pending(
    mock_processor,
    mock_vem,
    mock_apply_lora,
    mock_read_manifest,
    mock_load_base,
    mock_save_adapter,
    mock_train,
    mock_eval,
    mock_build_recognizer,
    mock_append_log,
    tmp_path,
):
    feedback_cfg = _build_feedback_config(tmp_path, max_regression=0.02)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "cvl.jsonl").write_text("")

    pending = _fake_pending_records(5)
    store = MagicMock(spec=FeedbackStore)
    store.pending.return_value = pending
    store.to_dataset_samples.return_value = [_sample(f"c{i}", "feedback") for i in range(5)]

    mock_read_manifest.return_value = [
        DatasetSample(f"cvl/e{i}.png", f"e{i}", "cvl", "test", "word", None) for i in range(5)
    ]
    mock_load_base.return_value = [_sample(f"b{i}") for i in range(50)]
    mock_apply_lora.return_value = MagicMock()

    # BEFORE=0.30, AFTER=0.40 -> delta=+0.10 > 0.02 -> rejected
    mock_eval.side_effect = [
        EvaluationMetrics(0.30, 0.35, 0.70, 0.65, 0.60, 5),
        EvaluationMetrics(0.40, 0.45, 0.60, 0.55, 0.50, 5),
    ]
    mock_build_recognizer.return_value = MagicMock()

    result = train_adapter_increment(
        feedback_config=feedback_cfg,
        recognition_config=_build_recognition_config(),
        app_config=_build_app_config(tmp_path),
        feedback_store=store,
    )

    assert result is not None
    assert result.rejected is True
    assert result.rejection_reason is not None
    assert "0.10" in result.rejection_reason or "0.1000" in result.rejection_reason
    assert result.cer_delta == pytest.approx(0.10, abs=1e-6)

    # LOAD-BEARING: mark_applied MUST NOT be called on a rejected update,
    # so the corrections stay pending and a later attempt can retry them.
    store.mark_applied.assert_not_called()

    # Update log still written (we log every attempt, accepted or not)
    mock_append_log.assert_called_once()

    # Adapter dir was created but renamed with REJECTED suffix
    save_call_path: Path = mock_save_adapter.call_args[0][1]
    rejected_path = save_call_path.with_name(save_call_path.name + "-REJECTED")
    assert not save_call_path.exists()
    assert rejected_path.exists()
    assert result.version.endswith("-REJECTED")


@patch("training.incremental.load_adapter")
@patch("training.incremental.append_update_log")
@patch("training.incremental._build_recognizer_from_peft")
@patch("training.incremental.evaluate_recognizer")
@patch("training.incremental._train_adapter")
@patch("training.incremental.save_adapter", side_effect=_save_adapter_that_creates_dir)
@patch("training.incremental.load_replay_base")
@patch("training.incremental.read_manifest")
@patch("training.incremental.apply_lora_to_trocr")
@patch("training.incremental.VisionEncoderDecoderModel")
@patch("training.incremental.TrOCRProcessor")
def test_existing_adapter_gets_continued_not_replaced(
    mock_processor,
    mock_vem,
    mock_apply_lora,
    mock_read_manifest,
    mock_load_base,
    mock_save_adapter,
    mock_train,
    mock_eval,
    mock_build_recognizer,
    mock_append_log,
    mock_load_adapter,
    tmp_path,
):
    feedback_cfg = _build_feedback_config(tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "cvl.jsonl").write_text("")
    # Pre-existing adapter version — a mock model comes out of load_adapter
    adapter_dir = Path(feedback_cfg.adapter_dir)
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "v-1000-existing").mkdir()

    fake_peft = MagicMock()
    fake_peft.named_parameters.return_value = [
        ("model.lora_A.default.weight", MagicMock(requires_grad=False)),
        ("model.base.weight", MagicMock(requires_grad=False)),
    ]
    mock_load_adapter.return_value = fake_peft

    store = MagicMock(spec=FeedbackStore)
    store.pending.return_value = _fake_pending_records(5)
    store.to_dataset_samples.return_value = [_sample("c0", "feedback")]
    mock_read_manifest.return_value = [
        DatasetSample("cvl/e0.png", "e0", "cvl", "test", "word", None)
    ]
    mock_load_base.return_value = [_sample("b0")]
    mock_eval.side_effect = [
        EvaluationMetrics(0.30, 0.35, 0.70, 0.65, 0.60, 1),
        EvaluationMetrics(0.29, 0.34, 0.71, 0.66, 0.61, 1),
    ]
    mock_build_recognizer.return_value = MagicMock()

    train_adapter_increment(
        feedback_config=feedback_cfg,
        recognition_config=_build_recognition_config(),
        app_config=_build_app_config(tmp_path),
        feedback_store=store,
    )

    # load_adapter was used (continued from existing), NOT apply_lora
    mock_load_adapter.assert_called_once()
    assert mock_load_adapter.call_args[0][1].name == "v-1000-existing"
    mock_apply_lora.assert_not_called()
