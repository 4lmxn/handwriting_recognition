from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class DocumentsConfig:
    # Image extensions accepted by documents.loader.load_image. Kept as a
    # config field rather than hardcoded so a user who scans TIFFs from a
    # specific device can add ".tif" without touching code, and so a
    # future PDF extension lands next to it in one place.
    allowed_image_extensions: tuple[str, ...]
    # Hard cap on file size, enforced before Pillow ever opens the file.
    # Pillow's decompression-bomb protection kicks in on pixel count, not
    # bytes — a hostile 100MB PNG can still be well under the pixel-bomb
    # threshold but will happily eat RAM if we let it decode. This is the
    # cheap belt on top of that suspenders.
    max_image_bytes: int

    @classmethod
    def from_dict(cls, data: dict) -> DocumentsConfig:
        return cls(
            allowed_image_extensions=tuple(
                ext.lower() for ext in data["allowed_image_extensions"]
            ),
            max_image_bytes=int(data["max_image_bytes"]),
        )


def load_documents_config() -> DocumentsConfig:
    path = CONFIGS_DIR / "documents.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return DocumentsConfig.from_dict(data)
