"""Load PDF documents into per-page grayscale ndarrays (Phase 6, PR 2).

Each page is rendered at `DocumentsConfig.pdf_render_dpi` and returned
in the same (H, W) uint8 grayscale convention as `documents.loader`.
Downstream layout/segmentation (Phase 6, PR 3) can then treat a PDF
page and an uploaded image identically.

pymupdf (`import pymupdf` is the modern name; `import fitz` is the
legacy alias for the same package) does the heavy lifting: rendering
to a Pixmap in the grayscale colorspace is a single call and avoids
a per-page RGB→L conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf

from documents.config import DocumentsConfig
from documents.loader import ImageTooLargeError, UnsupportedImageFormatError

_PDF_EXTENSION = ".pdf"


class InvalidPdfError(ValueError):
    """Raised when pymupdf cannot decode the file as a PDF."""


class PdfTooLongError(ValueError):
    """Raised when a PDF's page count exceeds `DocumentsConfig.max_pdf_pages`."""


def load_pdf_pages(path: Path, config: DocumentsConfig) -> list[np.ndarray]:
    """Read a PDF and return one (H, W) uint8 grayscale array per page.

    Ordering is preserved (page 1 → index 0). Raises `FileNotFoundError`,
    `UnsupportedImageFormatError` (wrong extension), `ImageTooLargeError`
    (byte cap), `PdfTooLongError` (page cap), or `InvalidPdfError`
    (decode failure).
    """
    if path.suffix.lower() != _PDF_EXTENSION:
        raise UnsupportedImageFormatError(
            f"{path.suffix!r} is not a PDF (expected {_PDF_EXTENSION!r})"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > config.max_pdf_bytes:
        raise ImageTooLargeError(
            f"{path} is {size} bytes, exceeds max_pdf_bytes={config.max_pdf_bytes}"
        )
    try:
        # pymupdf.open needs `filename=` for str/Path inputs; the positional
        # form is fine but the kwarg makes it obvious we're not passing raw
        # bytes (which would need `stream=`).
        doc = pymupdf.open(filename=str(path))
    except pymupdf.FileDataError as exc:
        raise InvalidPdfError(f"pymupdf could not open {path}") from exc

    try:
        if doc.page_count > config.max_pdf_pages:
            raise PdfTooLongError(
                f"{path} has {doc.page_count} pages, "
                f"exceeds max_pdf_pages={config.max_pdf_pages}"
            )
        pages: list[np.ndarray] = []
        # Index-based iteration rather than `for page in doc` because
        # pymupdf's type stubs don't declare Document.__iter__ (the
        # runtime supports it via __getitem__, but mypy can't see that).
        for i in range(doc.page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=config.pdf_render_dpi, colorspace=pymupdf.csGRAY)
            # pix.samples is a bytes object of length width*height for
            # grayscale (no alpha, no stride padding). Reshape directly —
            # no per-pixel Python loop, no PIL round-trip.
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width
            )
            # Copy so the array outlives the Pixmap (pymupdf frees the
            # underlying buffer when the Pixmap goes out of scope, leaving
            # np.frombuffer views dangling).
            pages.append(arr.copy())
        return pages
    finally:
        doc.close()
