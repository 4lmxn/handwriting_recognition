"""Load image documents (PNG/JPEG/TIFF) into the project's canonical
grayscale ndarray convention (Phase 6, PR 1).

Convention matches `app.gui.tabs.drawing_canvas_tab.qimage_to_grayscale_array`
and everything in `preprocessing/image_ops.py`: uint8, shape (H, W),
ink-dark-on-light-background. Downstream (Phase 6 PR 3/4) will run the
existing segmentation + recognition stack over the returned array, so
keeping the shape/dtype identical to what those already accept means no
per-source branching later.

PDF loading lives in a sibling module in the next PR — this one is
image-only so it can ship in isolation without pulling PyMuPDF into the
smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from documents.config import DocumentsConfig


class UnsupportedImageFormatError(ValueError):
    """Raised when a file's extension isn't in the configured allowlist."""


class ImageTooLargeError(ValueError):
    """Raised when a file exceeds `DocumentsConfig.max_image_bytes`."""


class InvalidImageError(ValueError):
    """Raised when Pillow cannot decode the file as an image."""


def load_image(path: Path, config: DocumentsConfig) -> np.ndarray:
    """Read an image file and return it as an (H, W) uint8 grayscale array.

    Raises `FileNotFoundError`, `UnsupportedImageFormatError`,
    `ImageTooLargeError`, or `InvalidImageError`. Extension check runs
    before the size check so a hostile file with an unsupported
    extension is rejected without ever stat-ing.
    """
    if path.suffix.lower() not in config.allowed_image_extensions:
        raise UnsupportedImageFormatError(
            f"{path.suffix!r} is not in allowed extensions "
            f"{config.allowed_image_extensions}"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size > config.max_image_bytes:
        raise ImageTooLargeError(
            f"{path} is {size} bytes, exceeds max_image_bytes={config.max_image_bytes}"
        )
    try:
        with Image.open(path) as pil_image:
            # Pillow's "L" mode: 8-bit pixels, black-to-white, ink-dark-on-
            # light for typical scanned documents — same convention used by
            # QImage.Format_Grayscale8 in the drawing tab.
            gray = pil_image.convert("L")
            return np.array(gray, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise InvalidImageError(f"Pillow could not decode {path}") from exc
