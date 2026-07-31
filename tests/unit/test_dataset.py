from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from datasets.manifest import DatasetSample, write_manifest
from training.dataset import HandwritingDataset, load_samples


class _FakeTokenizer:
    pad_token_id = 1

    def __call__(self, text, padding, max_length, truncation):
        ids = [ord(c) % 50 + 2 for c in text][:max_length]
        ids += [self.pad_token_id] * (max_length - len(ids))
        return SimpleNamespace(input_ids=ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def __call__(self, images, return_tensors):
        array = np.array(images.convert("RGB"))
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
        return SimpleNamespace(pixel_values=tensor)


def _sample(image_path: str, transcript: str, split: str = "train") -> DatasetSample:
    return DatasetSample(
        image_path=image_path,
        transcript=transcript,
        source="synthetic",
        split=split,
        label_type="word",
        writer_id=None,
    )


def test_load_samples_filters_by_split(tmp_path):
    manifests_dir = tmp_path / "manifests"
    write_manifest(
        [_sample("a.png", "hi", "train"), _sample("b.png", "bye", "test")],
        manifests_dir / "foo.jsonl",
    )

    samples = load_samples(["foo"], manifests_dir, splits=["train"])

    assert len(samples) == 1
    assert samples[0].transcript == "hi"


def test_load_samples_merges_multiple_manifests(tmp_path):
    manifests_dir = tmp_path / "manifests"
    write_manifest([_sample("a.png", "hi")], manifests_dir / "foo.jsonl")
    write_manifest([_sample("b.png", "bye")], manifests_dir / "bar.jsonl")

    samples = load_samples(["foo", "bar"], manifests_dir, splits=["train"])

    assert {s.transcript for s in samples} == {"hi", "bye"}


def test_load_samples_skips_missing_manifest(tmp_path):
    manifests_dir = tmp_path / "manifests"
    write_manifest([_sample("a.png", "hi")], manifests_dir / "foo.jsonl")

    samples = load_samples(["foo", "does_not_exist"], manifests_dir, splits=["train"])

    assert len(samples) == 1


def test_load_samples_respects_max_samples(tmp_path):
    manifests_dir = tmp_path / "manifests"
    write_manifest(
        [_sample(f"{i}.png", str(i)) for i in range(10)],
        manifests_dir / "foo.jsonl",
    )

    samples = load_samples(["foo"], manifests_dir, splits=["train"], max_samples=3)

    assert len(samples) == 3


def test_handwriting_dataset_getitem_shapes(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    Image.fromarray(np.full((32, 64), 255, dtype=np.uint8)).save(processed_dir / "a.png")

    samples = [_sample("a.png", "hi")]
    dataset = HandwritingDataset(samples, processed_dir, _FakeProcessor(), max_target_length=8)

    item = dataset[0]

    assert item["pixel_values"].shape == (3, 32, 64)
    assert item["labels"].shape == (8,)
    assert len(dataset) == 1


def test_handwriting_dataset_masks_padding_with_ignore_index(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    Image.fromarray(np.full((10, 10), 255, dtype=np.uint8)).save(processed_dir / "a.png")

    samples = [_sample("a.png", "hi")]  # 2 real tokens, rest padding
    dataset = HandwritingDataset(samples, processed_dir, _FakeProcessor(), max_target_length=8)

    labels = dataset[0]["labels"]

    assert (labels[2:] == -100).all()
    assert (labels[:2] != -100).all()


def test_handwriting_dataset_applies_augmentation_pipeline(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    Image.fromarray(np.full((32, 64), 255, dtype=np.uint8)).save(processed_dir / "a.png")

    calls = []

    def fake_pipeline(image):
        calls.append(image.shape)
        return {"image": image}

    samples = [_sample("a.png", "hi")]
    dataset = HandwritingDataset(
        samples,
        processed_dir,
        _FakeProcessor(),
        max_target_length=8,
        augmentation_pipeline=fake_pipeline,
    )

    dataset[0]

    assert len(calls) == 1
