from datasets.manifest import DatasetSample
from training.hard_negative_mining import oversample_hard_negatives


def _sample(transcript: str) -> DatasetSample:
    return DatasetSample(
        image_path=f"{transcript}.png",
        transcript=transcript,
        source="synthetic",
        split="train",
        label_type="character",
        writer_id=None,
    )


def test_oversamples_only_matching_samples():
    samples = [_sample("0"), _sample("a"), _sample("O")]
    result = oversample_hard_negatives(samples, hard_classes={"0", "O"}, oversample_factor=3)

    assert result.count(samples[0]) == 3
    assert result.count(samples[1]) == 1
    assert result.count(samples[2]) == 3
    assert len(result) == 7


def test_empty_hard_classes_is_a_noop():
    samples = [_sample("0"), _sample("a")]
    result = oversample_hard_negatives(samples, hard_classes=set(), oversample_factor=5)
    assert result == samples


def test_oversample_factor_below_one_is_treated_as_one():
    samples = [_sample("0")]
    result = oversample_hard_negatives(samples, hard_classes={"0"}, oversample_factor=0)
    assert result == samples


def test_matches_substring_within_multi_character_transcript():
    samples = [_sample("turn"), _sample("arm")]
    result = oversample_hard_negatives(samples, hard_classes={"t"}, oversample_factor=2)

    assert result.count(samples[0]) == 2
    assert result.count(samples[1]) == 1


def test_preserves_sample_identity_not_just_equality():
    samples = [_sample("0")]
    result = oversample_hard_negatives(samples, hard_classes={"0"}, oversample_factor=2)
    assert all(item is samples[0] for item in result)
