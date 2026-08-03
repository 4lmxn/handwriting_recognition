"""Structural primitives for Phase 8 math-aware recognition.

Deliberately narrow scope (see docs/ROADMAP.md Phase 8): sub /
superscript character positioning, simple fractions (a horizontal
bar with content above and below), and simple square roots. Complex
nested expressions, integrals, sums, matrices, and chemistry are
explicitly out of scope — the dataclasses below don't try to model
them.

All coordinate fields are inclusive-top / exclusive-bottom / inclusive-
left / exclusive-right, matching the WordBox convention from
`documents.layout`. Callers can slice a page directly with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScriptPosition(Enum):
    """Where a character sits relative to the surrounding baseline.

    BASELINE is the common case (normal letter). SUPERSCRIPT and
    SUBSCRIPT are the two flagged positions that turn into `^{...}`
    and `_{...}` in the LaTeX emitter of a later PR.
    """

    BASELINE = "baseline"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"


@dataclass(frozen=True)
class ScriptedChar:
    """A single character within a word, tagged with vertical position.

    Emitted by the sub/superscript detector (PR 2). `text` is the
    recognized single-character transcript; layout coordinates are
    the character's bbox on the working page (same coordinate system
    as `documents.layout.WordBox`).
    """

    text: str
    position: ScriptPosition
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass(frozen=True)
class Fraction:
    """A horizontal fraction bar with content above and below.

    numerator / denominator are the LaTeX-ready transcripts of the
    two regions (produced by a downstream recognizer). Bar coordinates
    are page-relative and are what the fraction detector (PR 3) hands
    off so the emitter knows which content belongs to which slot.
    """

    numerator: str
    denominator: str
    bar_top: int
    bar_bottom: int
    bar_left: int
    bar_right: int


@dataclass(frozen=True)
class Root:
    """A √ symbol followed by its argument.

    Just the argument transcript + the whole-glyph bounding box.
    The √ symbol itself isn't kept as separate metadata — the emitter
    hard-codes `\\sqrt{argument}`.
    """

    argument: str
    top: int
    bottom: int
    left: int
    right: int


@dataclass(frozen=True)
class MathFragment:
    """The unit returned to callers wanting math-aware recognition.

    `latex` is the composed LaTeX string (e.g. `x^2`, `\\frac{a}{b}`,
    `\\sqrt{x+1}`). The bbox is on the working page so a UI can draw
    an overlay around exactly the region this fragment came from —
    same coordinate contract as `documents.inference.RecognizedWord`.
    """

    latex: str
    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left
