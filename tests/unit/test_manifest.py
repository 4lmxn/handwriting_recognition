import pytest

from datasets.manifest import DatasetSample, append_manifest, read_manifest, write_manifest
from datasets.registry import REGISTRY, get_source


def _sample(image_path="a.png", transcript="hello") -> DatasetSample:
    return DatasetSample(
        image_path=image_path,
        transcript=transcript,
        source="synthetic",
        split="train",
        label_type="word",
        writer_id=None,
    )


def test_write_then_read_manifest_roundtrips(tmp_path):
    samples = [_sample("a.png", "hello"), _sample("b.png", "world")]
    manifest_path = tmp_path / "manifest.jsonl"

    write_manifest(samples, manifest_path)
    loaded = read_manifest(manifest_path)

    assert loaded == samples


def test_write_manifest_creates_parent_dirs(tmp_path):
    manifest_path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    write_manifest([_sample()], manifest_path)
    assert manifest_path.exists()


def test_append_manifest_adds_to_existing_file(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest([_sample("a.png")], manifest_path)
    append_manifest([_sample("b.png")], manifest_path)

    loaded = read_manifest(manifest_path)
    assert len(loaded) == 2
    assert loaded[1].image_path == "b.png"


def test_read_manifest_skips_blank_lines(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"image_path": "a.png", "transcript": "hi", '
        '"source": "synthetic", "split": "train", '
        '"label_type": "word", "writer_id": null}\n\n'
    )
    loaded = read_manifest(manifest_path)
    assert len(loaded) == 1


def test_registry_has_expected_sources():
    for expected in ("synthetic", "mnist", "emnist", "iam", "cvl"):
        assert expected in REGISTRY


def test_get_source_returns_info():
    info = get_source("mnist")
    assert info.acquisition == "auto"


def test_get_source_raises_for_unknown():
    with pytest.raises(KeyError):
        get_source("not_a_real_dataset")


def test_manual_sources_have_instructions():
    for info in REGISTRY.values():
        if info.acquisition == "manual":
            assert info.instructions
            assert info.homepage_url
