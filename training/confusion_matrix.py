"""Character-level confusion analysis from (reference, hypothesis) string pairs.

Feeds Phase 4 hard-negative mining: `ConfusionMatrix` accumulates substitution
counts across an evaluation run, `analyze_ambiguous_classes` reports the
known-confusable single-character pairs from docs/ROADMAP.md, and
`hard_negative_classes` surfaces the characters a training loop should
oversample next round.

Gap: multi-character confusions (rn/m, cl/d, vv/w) are NOT covered here.
Those involve one side spanning two characters and the other one, so they
can't be expressed as a (ref_char, hyp_char) pair in a character-aligned
confusion matrix. Detecting them requires aligning and comparing *substrings*
rather than characters — a documented gap for a future iteration, not
something `analyze_ambiguous_classes` handles.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Single-character ambiguous pairs from docs/ROADMAP.md's Phase 4 bullet list.
# rn/m, cl/d, vv/w are deliberately excluded (see module docstring).
_AMBIGUOUS_PAIRS: list[tuple[str, str]] = [
    ("0", "O"),
    ("1", "l"),
    ("1", "I"),
    ("5", "S"),
    ("2", "Z"),
    ("8", "B"),
    ("6", "G"),
    ("9", "g"),
    ("O", "Q"),
    ("C", "G"),
]


def align_strings(reference: str, hypothesis: str) -> list[tuple[str | None, str | None]]:
    n, m = len(reference), len(hypothesis)
    # dp[i][j] = edit distance between reference[:i] and hypothesis[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # match or substitution
            )

    alignment: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)
        ):
            alignment.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((reference[i - 1], None))
            i -= 1
        else:
            alignment.append((None, hypothesis[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


@dataclass
class ConfusionMatrix:
    counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    def record(self, reference: str, hypothesis: str) -> None:
        for ref_char, hyp_char in align_strings(reference, hypothesis):
            if ref_char is not None and hyp_char is not None and ref_char != hyp_char:
                self.counts[(ref_char, hyp_char)] += 1

    def most_confused(self, top_n: int = 20) -> list[tuple[tuple[str, str], int]]:
        return self.counts.most_common(top_n)

    def to_dict(self) -> dict:
        return {
            "confusions": {f"{ref}->{hyp}": count for (ref, hyp), count in self.counts.items()},
            "total_substitutions": sum(self.counts.values()),
        }

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> ConfusionMatrix:
        with open(path) as f:
            data = json.load(f)
        counts: Counter[tuple[str, str]] = Counter()
        for key, count in data["confusions"].items():
            ref, hyp = key.split("->", 1)
            counts[(ref, hyp)] = count
        return cls(counts=counts)


def analyze_ambiguous_classes(matrix: ConfusionMatrix) -> dict[str, int]:
    result: dict[str, int] = {}
    for a, b in _AMBIGUOUS_PAIRS:
        label = f"{a}/{b}"
        result[label] = matrix.counts[(a, b)] + matrix.counts[(b, a)]
    return result


def hard_negative_classes(matrix: ConfusionMatrix, min_count: int = 3) -> set[str]:
    classes: set[str] = set()
    for (ref_char, hyp_char), count in matrix.counts.items():
        if count >= min_count:
            classes.add(ref_char)
            classes.add(hyp_char)
    return classes
