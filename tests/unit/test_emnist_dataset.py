import gzip
import struct
import zipfile

import numpy as np
from PIL import Image

from datasets.config import load_datasets_config
from datasets.sources.emnist import EmnistDatasetSource, parse_mapping


def _idx_gzip_bytes(data: np.ndarray) -> bytes:
    header = bytes([0, 0, 0x08, data.ndim])
    shape_bytes = b"".join(struct.pack(">I", d) for d in data.shape)
    return gzip.compress(header + shape_bytes + data.tobytes())


def _oriented_image(high_row: bool) -> np.ndarray:
    """Row 0 set to 200, everything else 0 — asymmetric so the transpose
    orientation fix is verifiable."""
    img = np.zeros((28, 28), dtype=np.uint8)
    if high_row:
        img[0, :] = 200
    return img


def _write_fake_emnist_raw(raw_dir, config):
    emnist_dir = raw_dir / "emnist"
    emnist_dir.mkdir(parents=True)
    split = config.split

    train_images = np.stack([_oriented_image(True), _oriented_image(False)])
    train_labels = np.array([0, 1], dtype=np.uint8)  # '0' -> 48, 'A' -> 65
    test_images = np.stack([_oriented_image(True)])
    test_labels = np.array([2], dtype=np.uint8)  # 'a' -> 97

    mapping_text = "0 48\n1 65\n2 97\n"

    archive_path = emnist_dir / "gzip.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            f"gzip/emnist-{split}-train-images-idx3-ubyte.gz", _idx_gzip_bytes(train_images)
        )
        zf.writestr(
            f"gzip/emnist-{split}-train-labels-idx1-ubyte.gz", _idx_gzip_bytes(train_labels)
        )
        zf.writestr(f"gzip/emnist-{split}-test-images-idx3-ubyte.gz", _idx_gzip_bytes(test_images))
        zf.writestr(f"gzip/emnist-{split}-test-labels-idx1-ubyte.gz", _idx_gzip_bytes(test_labels))
        zf.writestr(f"gzip/emnist-{split}-mapping.txt", mapping_text)


def test_prepare_uses_pre_downloaded_archive_without_network(tmp_path):
    config = load_datasets_config().emnist
    _write_fake_emnist_raw(tmp_path, config)

    source = EmnistDatasetSource(config, val_fraction=0.5)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 3  # 2 train_val + 1 test


def test_prepare_uses_mapping_for_transcripts(tmp_path):
    config = load_datasets_config().emnist
    _write_fake_emnist_raw(tmp_path, config)

    source = EmnistDatasetSource(config, val_fraction=0.5)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    transcripts = sorted(s.transcript for s in samples)
    assert transcripts == ["0", "A", "a"]
    assert all(s.label_type == "character" for s in samples)


def test_prepare_applies_orientation_fix(tmp_path):
    config = load_datasets_config().emnist
    _write_fake_emnist_raw(tmp_path, config)

    source = EmnistDatasetSource(config, val_fraction=0.5)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    sample = next(s for s in samples if s.transcript == "0")
    with Image.open(tmp_path / sample.image_path) as img:
        # Row 0 (high_row=True) transposed becomes column 0: pixel (0,0) is ink.
        top_left = img.getpixel((0, 0))
        top_right = img.getpixel((5, 0))
    assert top_left == 255 - 200
    assert top_right == 255


def test_parse_mapping_converts_ascii_codes(tmp_path):
    mapping_path = tmp_path / "mapping.txt"
    mapping_path.write_text("0 48\n1 65\n2 97\n")

    mapping = parse_mapping(mapping_path)

    assert mapping == {0: "0", 1: "A", 2: "a"}
