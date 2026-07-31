"""LoRA adapter for TrOCR (Phase 5, PR 2).

Layered on top of the frozen fine-tuned backbone — the base model's
weights are NEVER modified by adapter training. Corrections captured
via the feedback loop train only the adapter's low-rank matrices,
preserving the base model's calibration. Each adapter version is saved
under `weights/adapters/v-<epoch>-<uuid>/` (never overwritten in place)
so a bad update can be rolled back by pointing at an earlier version.

The API is minimal on purpose: apply, save, load. The training loop
(Phase 5 PR 3) drives the fit/eval/version cycle; this module just
provides the wrap/unwrap primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from models.adapters.config import LoraAdapterConfig

_BiasLiteral = Literal["none", "all", "lora_only"]
_VALID_BIAS: tuple[_BiasLiteral, ...] = ("none", "all", "lora_only")


def _to_peft_config(config: LoraAdapterConfig) -> LoraConfig:
    # No `task_type` on purpose. Setting SEQ_2_SEQ_LM triggers peft's
    # task-specific wrapping (input preparation, generate proxies) which
    # expects `prepare_inputs_for_generation` on the base — real TrOCR has
    # it, tiny test-only fakes do not. The wrapped PeftModel still proxies
    # `.generate()` back to the base via __getattr__, so the recognizer
    # code path continues to work; only peft's task-flavored input munging
    # is skipped, which we don't rely on.
    if config.bias not in _VALID_BIAS:
        raise ValueError(
            f"LoraAdapterConfig.bias must be one of {_VALID_BIAS!r}, got {config.bias!r}"
        )
    return LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=cast(_BiasLiteral, config.bias),
        target_modules=list(config.target_modules),
    )


def apply_lora_to_trocr(model: torch.nn.Module, config: LoraAdapterConfig) -> PeftModel:
    """Wrap ``model`` in a LoRA adapter targeting only the modules named in
    ``config.target_modules``.

    All base-model parameters are frozen up front — belt-and-braces so any
    module the peft target list *doesn't* catch also can't accidentally
    train. peft.get_peft_model() then flips requires_grad=True only on the
    newly-inserted lora_A/lora_B matrices.

    Typed as `torch.nn.Module` (not `PreTrainedModel`) so tests can pass
    minimal fakes without dragging in the full transformers class hierarchy.
    The cast on the return is because peft.get_peft_model() is declared as
    returning `PeftModel | PeftMixedModel`; without task_type we always get
    the plain PeftModel branch.
    """
    for param in model.parameters():
        param.requires_grad = False
    return cast(PeftModel, get_peft_model(model, _to_peft_config(config)))  # type: ignore[arg-type]


def save_adapter(peft_model: PeftModel, path: Path) -> None:
    """Persist just the adapter weights (a few MB per version) to ``path``.

    The base model is not saved — reloading requires calling
    ``load_adapter(base, path)`` with the same base weights the adapter
    was trained against.
    """
    path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(path))


def load_adapter(base_model: torch.nn.Module, path: Path) -> PeftModel:
    """Attach a previously-saved adapter at ``path`` to a fresh copy of the
    base model. Returns a PeftModel that behaves like the wrapped base
    model for generate/forward, with the adapter's contribution active.
    """
    return PeftModel.from_pretrained(base_model, str(path))


def trainable_parameter_count(peft_model: PeftModel) -> int:
    """Sum of requires_grad=True parameter counts. Useful for logging and
    for tests that verify the base model stayed frozen (the count should
    be a tiny fraction of the base model's total parameter count)."""
    return sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
