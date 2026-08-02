"""Unit tests for documents.loader (Phase 6, PR 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from documents.config import DocumentsConfig, load_documents_config
from documents.loader import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageFormatError,
    load_image,
)


def _make_config(
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    max_bytes: int = 10_000_000,
) -> DocumentsConfig:
    return DocumentsConfig(
        allowed_image_extensions=extensions,
        max_image_bytes=max_bytes,
    )


def _write_png(path: Path, width: int = 8, height: int = 6, color=(0, 0, 0)) -> None:
    Image.new("RGB", (width, height), color=color).save(path, format="PNG")


def test_load_image_returns_grayscale_uint8_hw(tmp_path):
    src = tmp_path / "sample.png"
    _write_png(src, width=8, height=6, color=(200, 200, 200))
    arr = load_image(src, _make_config())
    assert arr.shape == (6, 8)
    assert arr.dtype == np.uint8


def test_load_image_ink_dark_on_light_convention(tmp_path):
    # A pure-white PIL image ("L") should decode to 255, matching the
    # rest of the pipeline's ink-dark-on-light convention.
    src = tmp_path / "white.png"
    Image.new("L", (4, 4), color=255).save(src, format="PNG")
    arr = load_image(src, _make_config())
    assert arr.min() == 255 and arr.max() == 255


def test_load_image_converts_rgba_to_grayscale(tmp_path):
    src = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4), color=(10, 20, 30, 255)).save(src, format="PNG")
    arr = load_image(src, _make_config())
    assert arr.shape == (4, 4)
    assert arr.dtype == np.uint8


def test_load_image_rejects_unsupported_extension(tmp_path):
    src = tmp_path / "not_an_image.gif"
    src.write_bytes(b"GIF89a")
    with pytest.raises(UnsupportedImageFormatError):
        load_image(src, _make_config())


def test_load_image_extension_check_is_case_insensitive(tmp_path):
    src = tmp_path / "upper.PNG"
    _write_png(src)
    load_image(src, _make_config())  # must not raise


def test_load_image_rejects_missing_file(tmp_path):
    src = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        load_image(src, _make_config())


def test_load_image_rejects_oversized_file(tmp_path):
    src = tmp_path / "big.png"
    _write_png(src, width=64, height=64)
    tiny_config = _make_config(max_bytes=50)
    with pytest.raises(ImageTooLargeError):
        load_image(src, tiny_config)


def test_load_image_raises_invalid_for_garbage_bytes(tmp_path):
    src = tmp_path / "corrupt.png"
    src.write_bytes(b"not actually a png")
    with pytest.raises(InvalidImageError):
        load_image(src, _make_config())


def test_load_image_extension_check_precedes_existence_check(tmp_path):
    # A missing file with an unsupported extension should surface as
    # UnsupportedImageFormatError, not FileNotFoundError — extension
    # is validated before the filesystem is touched.
    src = tmp_path / "missing.gif"
    with pytest.raises(UnsupportedImageFormatError):
        load_image(src, _make_config())


def test_load_documents_config_reads_yaml():
    config = load_documents_config()
    assert ".png" in config.allowed_image_extensions
    assert ".pdf" not in config.allowed_image_extensions  # PDFs land in PR 2
    assert config.max_image_bytes > 0
    # Every entry must already be lowercase with a leading dot — that
    # invariant is what makes the case-normalized suffix check in
    # load_image work.
    for ext in config.allowed_image_extensions:
        assert ext.startswith(".")
        assert ext == ext.lower()
