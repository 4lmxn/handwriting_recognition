from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class DocumentsConfig:
    # Image extensions accepted by documents.loader.load_image. Kept as a
    # config field rather than hardcoded so a user who scans TIFFs from a
    # specific device can add ".tif" without touching code.
    allowed_image_extensions: tuple[str, ...]
    # Hard cap on file size, enforced before Pillow ever opens the file.
    # Pillow's decompression-bomb protection kicks in on pixel count, not
    # bytes — a hostile 100MB PNG can still be well under the pixel-bomb
    # threshold but will happily eat RAM if we let it decode. This is the
    # cheap belt on top of that suspenders.
    max_image_bytes: int
    # PDF ingest (Phase 6, PR 2). Byte cap is separate from images since
    # multi-page PDFs are legitimately larger than any single scanned
    # page; page cap is the second guard so a small-but-adversarial
    # 100k-page PDF can't force the loader into a huge loop.
    max_pdf_bytes: int
    max_pdf_pages: int
    # DPI at which each page is rendered to pixels. 150 is a reasonable
    # balance: TrOCR-small ingests images at 384x384 anyway, and a Letter
    # page at 150 DPI (~1275x1650) leaves enough resolution for
    # line/word segmentation to find text regions without ballooning
    # per-page memory (~2 MB grayscale) or render time. Bump for very
    # small handwriting, drop for scans of large-print documents.
    pdf_render_dpi: int

    @classmethod
    def from_dict(cls, data: dict) -> DocumentsConfig:
        return cls(
            allowed_image_extensions=tuple(
                ext.lower() for ext in data["allowed_image_extensions"]
            ),
            max_image_bytes=int(data["max_image_bytes"]),
            max_pdf_bytes=int(data["max_pdf_bytes"]),
            max_pdf_pages=int(data["max_pdf_pages"]),
            pdf_render_dpi=int(data["pdf_render_dpi"]),
        )


def load_documents_config() -> DocumentsConfig:
    path = CONFIGS_DIR / "documents.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return DocumentsConfig.from_dict(data)
