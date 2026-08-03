"""Configuration for Phase 8 math-aware recognition.

Tunables cluster around the two hard detection problems this phase
takes on: sub/superscript positioning (ratios relative to a line's
baseline) and horizontal fraction-bar geometry (length/thickness in
pixels). All parameters have safe defaults; the entire feature is
off unless `enabled: true` is set explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class MathNotationConfig:
    # Off by default so a fresh clone / anyone not doing math OCR
    # sees no behavior change from Phase 7. Turning it on wires the
    # math-aware path into the batch inference pipeline (PR 4).
    enabled: bool = False

    # -- Sub / superscript detection ---------------------------------
    # A character is a candidate superscript if its top sits above
    # (baseline_top - super_top_ratio * line_height). 0.2 = 20% of
    # line height above the line-content top. Tuned generously so
    # a slightly-raised character still qualifies; the size check
    # below is what actually distinguishes ascenders from super-
    # scripts.
    super_top_ratio: float = 0.2
    # Symmetric threshold for subscripts.
    sub_bottom_ratio: float = 0.2
    # Even if positioned correctly, a char must be SMALLER than
    # max_script_size_ratio * median_char_height to count as script
    # — an "l" that happens to be tall isn't a superscript.
    max_script_size_ratio: float = 0.75

    # -- Fraction bar detection --------------------------------------
    # Min length of a horizontal bar to qualify as a fraction bar,
    # measured as a fraction of the enclosing region's width. 0.3 =
    # bar must span at least 30% of the region — filters out
    # incidental horizontal ink like the middle of a "z".
    min_fraction_bar_length_ratio: float = 0.3
    # Max thickness in pixels — a fraction bar is thin. Anything
    # thicker than this is probably a struck-through letter or a
    # decoration, not a division symbol.
    max_fraction_bar_thickness: int = 4

    @classmethod
    def from_dict(cls, data: dict) -> MathNotationConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            super_top_ratio=float(data.get("super_top_ratio", 0.2)),
            sub_bottom_ratio=float(data.get("sub_bottom_ratio", 0.2)),
            max_script_size_ratio=float(data.get("max_script_size_ratio", 0.75)),
            min_fraction_bar_length_ratio=float(
                data.get("min_fraction_bar_length_ratio", 0.3)
            ),
            max_fraction_bar_thickness=int(
                data.get("max_fraction_bar_thickness", 4)
            ),
        )


def load_math_notation_config() -> MathNotationConfig:
    path = CONFIGS_DIR / "math_notation.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return MathNotationConfig.from_dict(data)
