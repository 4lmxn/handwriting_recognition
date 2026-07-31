import json

import cv2
import numpy as np
import pytest

from feedback.store import FeedbackRecord, FeedbackStore


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(
        storage_dir=tmp_path / "feedback",
        image_dir=tmp_path / "datasets" / "processed" / "feedback",
    )


@pytest.fixture
def gray_image():
    return np.full((32, 128), 255, dtype=np.uint8)


def test_add_writes_record_and_image_file(store, gray_image):
    record = store.add(
        image=gray_image,
        prediction="Imagines",
        confidence=0.72,
        corrected="Imagine",
    )
    assert isinstance(record, FeedbackRecord)
    assert record.corrected_transcript == "Imagine"
    assert record.original_prediction == "Imagines"
    assert record.original_confidence == 0.72
    assert record.applied_in_version is None
    assert record.source == "drawing_canvas"

    # Image landed at <image_dir>/<uuid>.png and is readable back
    image_file = store._image_dir / f"{record.id}.png"  # noqa: SLF001
    assert image_file.exists()
    reloaded = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    assert reloaded is not None
    assert reloaded.shape == gray_image.shape

    # image_path is relative and matches the "feedback/<uuid>.png" convention
    assert record.image_path == f"feedback/{record.id}.png"


def test_add_raises_on_invalid_image(store):
    # Wrong dtype — cv2.imwrite returns False, we raise rather than silently
    # writing a broken file.
    with pytest.raises(OSError, match="cv2.imwrite failed"):
        store.add(
            image=np.zeros((0, 0), dtype=np.uint8),
            prediction="x",
            confidence=0.5,
            corrected="y",
        )


def test_all_returns_records_in_insertion_order(store, gray_image):
    r1 = store.add(gray_image, "a", 0.1, "A")
    r2 = store.add(gray_image, "b", 0.2, "B")
    r3 = store.add(gray_image, "c", 0.3, "C")
    got = store.all()
    assert [r.id for r in got] == [r1.id, r2.id, r3.id]
    assert [r.corrected_transcript for r in got] == ["A", "B", "C"]


def test_all_returns_empty_list_when_no_corrections_yet(store):
    assert store.all() == []
    assert store.pending() == []


def test_all_skips_malformed_lines(store, gray_image, caplog):
    store.add(gray_image, "a", 0.1, "A")
    # Corrupt: append junk + a partial record
    corrections_path = store._storage_dir / "corrections.jsonl"  # noqa: SLF001
    with open(corrections_path, "a") as f:
        f.write("this is not json\n")
        f.write(json.dumps({"partial": "record"}) + "\n")
    store.add(gray_image, "b", 0.2, "B")

    with caplog.at_level("WARNING"):
        records = store.all()

    # The two valid records survived; malformed ones logged + skipped
    assert len(records) == 2
    assert [r.corrected_transcript for r in records] == ["A", "B"]
    assert any("malformed correction" in msg for msg in caplog.messages)


def test_pending_excludes_applied_records(store, gray_image):
    r1 = store.add(gray_image, "a", 0.1, "A")
    r2 = store.add(gray_image, "b", 0.2, "B")
    r3 = store.add(gray_image, "c", 0.3, "C")

    store.mark_applied([r1.id, r3.id], version="v-100-abc")

    pending = store.pending()
    assert [r.id for r in pending] == [r2.id]

    all_records = store.all()
    versions = {r.id: r.applied_in_version for r in all_records}
    assert versions[r1.id] == "v-100-abc"
    assert versions[r2.id] is None
    assert versions[r3.id] == "v-100-abc"


def test_mark_applied_is_idempotent_and_extensible(store, gray_image):
    r1 = store.add(gray_image, "a", 0.1, "A")
    r2 = store.add(gray_image, "b", 0.2, "B")

    store.mark_applied([r1.id], version="v-1")
    # Re-applying the same id overwrites the version (later run wins) —
    # useful for the "reject an adapter, re-run with a new adapter" flow.
    store.mark_applied([r1.id], version="v-2")

    versions = {r.id: r.applied_in_version for r in store.all()}
    assert versions[r1.id] == "v-2"
    assert versions[r2.id] is None


def test_mark_applied_leaves_no_tmp_file_after_success(store, gray_image):
    r1 = store.add(gray_image, "a", 0.1, "A")
    store.mark_applied([r1.id], version="v-1")
    tmp = store._corrections_path.with_suffix(".jsonl.tmp")  # noqa: SLF001
    assert not tmp.exists()


def test_to_dataset_samples_produces_manifest_compatible_records(store, gray_image):
    r1 = store.add(gray_image, "wrong", 0.4, "right")
    r2 = store.add(gray_image, "off", 0.5, "on")

    samples = store.to_dataset_samples()
    assert len(samples) == 2

    for sample, record in zip(samples, [r1, r2], strict=True):
        assert sample.image_path == record.image_path
        assert sample.transcript == record.corrected_transcript
        assert sample.source == "feedback"
        assert sample.split == "train"
        assert sample.label_type == "word"
        assert sample.writer_id is None


def test_to_dataset_samples_includes_applied_records(store, gray_image):
    # Replay-all policy (Phase 5 decision): every incremental update trains
    # on ALL corrections, applied and pending alike. If this test starts
    # failing because to_dataset_samples() started filtering by pending,
    # revisit memory:phase5_decisions.md first — the policy may have changed.
    r1 = store.add(gray_image, "a", 0.1, "A")
    r2 = store.add(gray_image, "b", 0.2, "B")
    store.mark_applied([r1.id], version="v-1")

    ids = {s.image_path for s in store.to_dataset_samples()}
    assert ids == {r1.image_path, r2.image_path}
