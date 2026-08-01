"""Resolve an on-disk LoRA adapter path (Phase 5, PR 5).

Two functions:

- `find_latest_adapter`: newest accepted adapter version dir under a
  root, or None if none exist. Rejected versions (`-REJECTED` suffix)
  are skipped so a bad increment can't poison the next lookup.

- `resolve_adapter_path`: turns whatever the user wrote in
  configs/recognition.yaml into a concrete path (or None). Accepts
  None / "latest" / a filesystem path.

Kept out of `training/` on purpose: the runtime Recognizer needs to
resolve adapter paths at startup, and `training/incremental.py`
imports torch, transformers, peft, etc. Splitting the resolver into a
zero-heavy-deps module means importing it doesn't drag the full
training stack into the GUI process.
"""

from __future__ import annotations

from pathlib import Path


def find_latest_adapter(adapter_dir: Path) -> Path | None:
    if not adapter_dir.exists():
        return None
    candidates = [
        d
        for d in adapter_dir.iterdir()
        if d.is_dir() and d.name.startswith("v-") and not d.name.endswith("-REJECTED")
    ]
    if not candidates:
        return None
    # Directory name after the "v-" prefix starts with an epoch-seconds
    # integer, so alphabetical sort matches chronological order for the
    # first ~292 million years.
    return max(candidates, key=lambda d: d.name)


def resolve_adapter_path(raw: str | None, adapter_dir: Path) -> Path | None:
    """Turn a config value into a concrete adapter dir, or None.

    - None / empty string  -> None (no adapter, run the plain base model).
    - "latest"             -> newest accepted adapter under `adapter_dir`,
                              or None if the dir is empty (silent fallback
                              is intentional: a fresh install with no
                              corrections yet should still recognize).
    - anything else        -> Path(raw), no existence check here —
                              downstream peft.from_pretrained raises a
                              clearer error than we could.
    """
    if raw is None or raw == "":
        return None
    if raw == "latest":
        return find_latest_adapter(adapter_dir)
    return Path(raw)
