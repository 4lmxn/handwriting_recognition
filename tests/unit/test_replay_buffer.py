import pytest

from datasets.manifest import DatasetSample
from training.replay_buffer import build_replay_batch


def _sample(name: str, source: str) -> DatasetSample:
    return DatasetSample(
        image_path=f"{source}/{name}.png",
        transcript=name,
        source=source,
        split="train",
        label_type="word",
        writer_id=None,
    )


@pytest.fixture
def feedback():
    return [_sample(f"fb{i}", "feedback") for i in range(50)]


@pytest.fixture
def base():
    return [_sample(f"b{i}", "cvl") for i in range(1000)]


def test_ratio_split_at_default_70_30(feedback, base):
    batch = build_replay_batch(feedback, base, replay_ratio=0.7, max_total=100, seed=42)
    n_feedback = sum(1 for s in batch if s.source == "feedback")
    n_base = sum(1 for s in batch if s.source == "cvl")
    assert n_base == 70
    assert n_feedback == 30
    assert len(batch) == 100


def test_ratio_zero_means_no_base(feedback, base):
    batch = build_replay_batch(feedback, base, replay_ratio=0.0, max_total=20, seed=1)
    assert all(s.source == "feedback" for s in batch)
    assert len(batch) == 20


def test_ratio_one_means_no_feedback(feedback, base):
    batch = build_replay_batch(feedback, base, replay_ratio=1.0, max_total=20, seed=1)
    assert all(s.source == "cvl" for s in batch)
    assert len(batch) == 20


def test_determinism_under_seed(feedback, base):
    a = build_replay_batch(feedback, base, replay_ratio=0.7, max_total=50, seed=42)
    b = build_replay_batch(feedback, base, replay_ratio=0.7, max_total=50, seed=42)
    assert [s.image_path for s in a] == [s.image_path for s in b]


def test_different_seed_yields_different_selection(feedback, base):
    a = build_replay_batch(feedback, base, replay_ratio=0.7, max_total=50, seed=42)
    b = build_replay_batch(feedback, base, replay_ratio=0.7, max_total=50, seed=43)
    assert [s.image_path for s in a] != [s.image_path for s in b]


def test_empty_feedback_falls_back_to_base(base):
    batch = build_replay_batch([], base, replay_ratio=0.7, max_total=50, seed=1)
    # With no feedback, only the base slice is drawable.
    assert all(s.source == "cvl" for s in batch)
    # replay_ratio=0.7 means 35 base slots targeted, feedback slot unused
    assert len(batch) == 35


def test_empty_base_falls_back_to_feedback(feedback):
    batch = build_replay_batch(feedback, [], replay_ratio=0.7, max_total=50, seed=1)
    assert all(s.source == "feedback" for s in batch)
    # replay_ratio=0.7 means 15 feedback slots targeted (max_total - int(0.7*50))
    assert len(batch) == 15


def test_both_empty_returns_empty():
    assert build_replay_batch([], [], replay_ratio=0.5, max_total=10, seed=1) == []


def test_small_pools_below_target_are_used_entirely():
    small_fb = [_sample(f"fb{i}", "feedback") for i in range(3)]
    small_base = [_sample(f"b{i}", "cvl") for i in range(3)]
    batch = build_replay_batch(small_fb, small_base, replay_ratio=0.7, max_total=100, seed=1)
    # Both pools smaller than their share -> full inclusion of both.
    assert len(batch) == 6
    assert sum(1 for s in batch if s.source == "feedback") == 3
    assert sum(1 for s in batch if s.source == "cvl") == 3


def test_max_total_cap_is_respected(feedback, base):
    batch = build_replay_batch(feedback, base, replay_ratio=0.5, max_total=10, seed=1)
    assert len(batch) == 10


def test_invalid_ratio_raises():
    with pytest.raises(ValueError, match="replay_ratio must be in"):
        build_replay_batch([], [], replay_ratio=1.5, max_total=10, seed=1)


def test_negative_max_total_raises():
    with pytest.raises(ValueError, match="max_total must be non-negative"):
        build_replay_batch([], [], replay_ratio=0.5, max_total=-1, seed=1)


def test_batch_is_shuffled(feedback, base):
    # A large-enough batch drawn from ordered pools should NOT come out
    # in insertion order — that would defeat the "shuffled mix" contract.
    batch = build_replay_batch(feedback, base, replay_ratio=0.5, max_total=40, seed=1)
    names = [s.transcript for s in batch]
    assert names != sorted(names)
