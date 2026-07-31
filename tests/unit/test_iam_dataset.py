import numpy as np
import pytest
from PIL import Image

from datasets.sources.iam import IamDatasetSource, parse_lines_txt

_LINES_TXT = """\
#--- lines.txt ---------------------------------------------------------#
#
# line id, ok/err, graylevel, components, x, y, w, h, transcription
#
a01-000u-00 ok 154 19 408 746 1663 91 A|MOVE|to|stop|Mr.|Gaitskell|from
a01-000u-01 err 154 19 408 746 1663 91 This|line|failed|segmentation
"""


def _write_fixture(raw_dir, include_image_for=("a01-000u-00",)):
    iam_dir = raw_dir / "iam"
    ascii_dir = iam_dir / "ascii"
    ascii_dir.mkdir(parents=True)
    (ascii_dir / "lines.txt").write_text(_LINES_TXT)

    for line_id in include_image_for:
        parts = line_id.split("-")
        image_dir = iam_dir / "lines" / parts[0] / f"{parts[0]}-{parts[1]}"
        image_dir.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.full((30, 100), 255, dtype=np.uint8))
        img.save(image_dir / f"{line_id}.png")


def test_parse_lines_txt_skips_comments_and_err_by_default(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text(_LINES_TXT)
    entries = parse_lines_txt(path)

    assert len(entries) == 1
    assert entries[0][0] == "a01-000u-00"
    assert entries[0][1] == "A MOVE to stop Mr. Gaitskell from"


def test_parse_lines_txt_can_include_err(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text(_LINES_TXT)
    entries = parse_lines_txt(path, include_err=True)
    assert len(entries) == 2


def test_prepare_raises_helpful_error_when_not_downloaded(tmp_path):
    source = IamDatasetSource()
    with pytest.raises(FileNotFoundError, match="datasets/registry.py"):
        source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)


def test_prepare_produces_sample_for_available_image(tmp_path):
    _write_fixture(tmp_path)
    source = IamDatasetSource()
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.transcript == "A MOVE to stop Mr. Gaitskell from"
    assert sample.label_type == "line"
    assert sample.split == "train"
    assert (tmp_path / sample.image_path).exists()


def test_prepare_skips_entries_with_missing_images(tmp_path):
    _write_fixture(tmp_path, include_image_for=())  # lines.txt entry, but no image
    source = IamDatasetSource()
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)
    assert samples == []
