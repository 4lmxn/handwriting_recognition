from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class LayoutConfig:
    """Tunables for documents.layout.analyze_page (Phase 6, PR 3).

    Wraps the parameters of adaptive_threshold + deskew + segment_lines +
    segment_words so a caller can adjust ingest behavior per document
    type (dense scanned form vs. sparse handwritten note) without
    passing eight kwargs. See the individual functions in
    preprocessing/image_ops.py and segmentation/ for the semantics.
    """
    deskew: bool = True
    binarize_block_size: int = 35
    binarize_c: int = 11
    min_line_height: int = 5
    min_line_gap: int = 3
    min_word_gap: int = 8
    min_word_width: int = 3

    @classmethod
    def from_dict(cls, data: dict) -> LayoutConfig:
        return cls(
            deskew=bool(data.get("deskew", True)),
            binarize_block_size=int(data.get("binarize_block_size", 35)),
            binarize_c=int(data.get("binarize_c", 11)),
            min_line_height=int(data.get("min_line_height", 5)),
            min_line_gap=int(data.get("min_line_gap", 3)),
            min_word_gap=int(data.get("min_word_gap", 8)),
            min_word_width=int(data.get("min_word_width", 3)),
        )


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
    layout: LayoutConfig = field(default_factory=LayoutConfig)

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
            layout=LayoutConfig.from_dict(data.get("layout", {})),
        )


def load_documents_config() -> DocumentsConfig:
    path = CONFIGS_DIR / "documents.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return DocumentsConfig.from_dict(data)
