"""Unit tests for documents.pdf_loader (Phase 6, PR 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from documents.config import DocumentsConfig
from documents.loader import ImageTooLargeError, UnsupportedImageFormatError
from documents.pdf_loader import InvalidPdfError, PdfTooLongError, load_pdf_pages


def _make_config(
    max_pdf_bytes: int = 100_000_000,
    max_pdf_pages: int = 50,
    dpi: int = 72,
) -> DocumentsConfig:
    # dpi defaults to 72 in tests (not 150) so tests render 4x fewer
    # pixels — still exercises every code path but keeps the suite fast.
    return DocumentsConfig(
        allowed_image_extensions=(".png",),
        max_image_bytes=10_000_000,
        max_pdf_bytes=max_pdf_bytes,
        max_pdf_pages=max_pdf_pages,
        pdf_render_dpi=dpi,
    )


def _write_pdf(path: Path, num_pages: int = 1) -> None:
    doc = pymupdf.open()
    for _ in range(num_pages):
        doc.new_page(width=612, height=792)  # Letter size in points
    doc.save(path)
    doc.close()


def test_load_pdf_pages_returns_list_of_grayscale_uint8_arrays(tmp_path):
    src = tmp_path / "one.pdf"
    _write_pdf(src, num_pages=1)
    pages = load_pdf_pages(src, _make_config())
    assert len(pages) == 1
    assert pages[0].dtype == np.uint8
    assert pages[0].ndim == 2  # (H, W), no channel axis


def test_load_pdf_pages_preserves_page_order(tmp_path):
    src = tmp_path / "three.pdf"
    _write_pdf(src, num_pages=3)
    pages = load_pdf_pages(src, _make_config())
    assert len(pages) == 3
    # A blank Letter page at 72 DPI is 612x792 points → 612x792 pixels.
    for page in pages:
        assert page.shape == (792, 612)


def test_load_pdf_pages_arrays_are_independent(tmp_path):
    # Regression guard: the loader must copy each Pixmap's samples before
    # the Pixmap goes out of scope, otherwise mutating one page could
    # bleed into another (or into freed memory).
    src = tmp_path / "two.pdf"
    _write_pdf(src, num_pages=2)
    pages = load_pdf_pages(src, _make_config())
    pages[0][:] = 0
    assert pages[1].max() > 0  # untouched page still has its ink


def test_load_pdf_pages_rejects_non_pdf_extension(tmp_path):
    src = tmp_path / "notpdf.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(UnsupportedImageFormatError):
        load_pdf_pages(src, _make_config())


def test_load_pdf_pages_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf_pages(tmp_path / "missing.pdf", _make_config())


def test_load_pdf_pages_rejects_oversized_file(tmp_path):
    src = tmp_path / "big.pdf"
    _write_pdf(src, num_pages=1)
    tiny = _make_config(max_pdf_bytes=10)
    with pytest.raises(ImageTooLargeError):
        load_pdf_pages(src, tiny)


def test_load_pdf_pages_rejects_too_many_pages(tmp_path):
    src = tmp_path / "long.pdf"
    _write_pdf(src, num_pages=5)
    with pytest.raises(PdfTooLongError):
        load_pdf_pages(src, _make_config(max_pdf_pages=2))


def test_load_pdf_pages_raises_invalid_for_garbage_bytes(tmp_path):
    src = tmp_path / "corrupt.pdf"
    src.write_bytes(b"definitely not a pdf")
    with pytest.raises(InvalidPdfError):
        load_pdf_pages(src, _make_config())


def test_load_pdf_pages_extension_check_precedes_existence_check(tmp_path):
    with pytest.raises(UnsupportedImageFormatError):
        load_pdf_pages(tmp_path / "missing.jpg", _make_config())


def test_load_pdf_pages_render_dpi_controls_resolution(tmp_path):
    src = tmp_path / "dpi.pdf"
    _write_pdf(src, num_pages=1)
    lo = load_pdf_pages(src, _make_config(dpi=72))
    hi = load_pdf_pages(src, _make_config(dpi=144))
    assert hi[0].shape[0] == 2 * lo[0].shape[0]
    assert hi[0].shape[1] == 2 * lo[0].shape[1]
