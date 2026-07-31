import numpy as np
import pytest
from PIL import Image

from datasets.sources.cvl import CvlDatasetSource, extract_label


def _write_image(path, fill=200):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 60), fill, dtype=np.uint8)).save(path)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("0001-1-1-the.png", "the"),
        ("0001-6-1-skip.png", None),  # second segment '6' is excluded
        ("0001-2-1-är.png", None),  # contains umlaut
        ("0001-3-1-.png", None),  # empty label
        ("nodash.png", None),  # fewer than 2 '-'-separated segments
    ],
)
def test_extract_label(filename, expected):
    assert extract_label(filename) == expected


def test_prepare_raises_helpful_error_when_not_downloaded(tmp_path):
    source = CvlDatasetSource()
    with pytest.raises(FileNotFoundError, match="datasets/registry.py"):
        source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)


def test_prepare_produces_samples_with_correct_split_and_label(tmp_path):
    cvl_root = tmp_path / "cvl" / "cvl-database-1-1"
    _write_image(cvl_root / "trainset" / "words" / "writer1" / "0001-1-1-the.png")
    _write_image(cvl_root / "trainset" / "words" / "writer1" / "0001-6-1-skip.png")
    _write_image(cvl_root / "testset" / "words" / "writer2" / "0002-1-1-and.png")

    source = CvlDatasetSource()
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 2
    by_transcript = {s.transcript: s for s in samples}
    assert by_transcript["the"].split == "train"
    assert by_transcript["and"].split == "test"
    assert all(s.label_type == "word" for s in samples)
    for s in samples:
        assert (tmp_path / s.image_path).exists()


def test_prepare_uses_lines_when_use_words_false(tmp_path):
    cvl_root = tmp_path / "cvl" / "cvl-database-1-1"
    _write_image(cvl_root / "trainset" / "lines" / "writer1" / "0001-1-1-hello.png")

    source = CvlDatasetSource(use_words=False)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 1
    assert samples[0].label_type == "line"
    assert samples[0].transcript == "hello"
