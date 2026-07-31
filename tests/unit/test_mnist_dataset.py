import gzip
import struct

import numpy as np
from PIL import Image

from datasets.config import load_datasets_config
from datasets.sources.mnist import MnistDatasetSource


def _idx_gzip_bytes(data: np.ndarray) -> bytes:
    header = bytes([0, 0, 0x08, data.ndim])
    shape_bytes = b"".join(struct.pack(">I", d) for d in data.shape)
    return gzip.compress(header + shape_bytes + data.tobytes())


def _write_fake_mnist_raw(raw_dir, config):
    mnist_dir = raw_dir / "mnist"
    mnist_dir.mkdir(parents=True)

    train_images = np.stack([np.full((28, 28), v, dtype=np.uint8) for v in (10, 20, 30, 40, 50)])
    train_labels = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    test_images = np.stack([np.full((28, 28), v, dtype=np.uint8) for v in (60, 70)])
    test_labels = np.array([5, 6], dtype=np.uint8)

    (mnist_dir / config.train_images).write_bytes(_idx_gzip_bytes(train_images))
    (mnist_dir / config.train_labels).write_bytes(_idx_gzip_bytes(train_labels))
    (mnist_dir / config.test_images).write_bytes(_idx_gzip_bytes(test_images))
    (mnist_dir / config.test_labels).write_bytes(_idx_gzip_bytes(test_labels))

    return train_images, test_images


def test_prepare_uses_pre_downloaded_files_without_network(tmp_path):
    config = load_datasets_config().mnist
    _write_fake_mnist_raw(tmp_path, config)

    source = MnistDatasetSource(config, val_fraction=0.4)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 7  # 5 train_val + 2 test


def test_prepare_splits_train_val_by_val_fraction(tmp_path):
    config = load_datasets_config().mnist
    _write_fake_mnist_raw(tmp_path, config)

    source = MnistDatasetSource(config, val_fraction=0.4)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    splits = [s.split for s in samples]
    assert splits.count("train") == 3
    assert splits.count("val") == 2
    assert splits.count("test") == 2


def test_prepare_sets_transcript_to_digit_label(tmp_path):
    config = load_datasets_config().mnist
    _write_fake_mnist_raw(tmp_path, config)

    source = MnistDatasetSource(config, val_fraction=0.4)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    transcripts = sorted(s.transcript for s in samples)
    assert transcripts == [str(d) for d in range(7)]
    assert all(s.label_type == "character" for s in samples)


def test_prepare_inverts_pixel_values(tmp_path):
    config = load_datasets_config().mnist
    _write_fake_mnist_raw(tmp_path, config)

    source = MnistDatasetSource(config, val_fraction=0.4)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    sample = next(s for s in samples if s.transcript == "0")
    with Image.open(tmp_path / sample.image_path) as img:
        pixel = img.getpixel((0, 0))
    assert pixel == 255 - 10


def test_max_samples_caps_output(tmp_path):
    config = load_datasets_config().mnist
    _write_fake_mnist_raw(tmp_path, config)

    source = MnistDatasetSource(config, val_fraction=0.4, max_samples=2)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    assert len(samples) == 4  # 2 from train_val split + 2 from test split
