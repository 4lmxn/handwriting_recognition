from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoraAdapterConfig:
    """LoRA adapter hyperparameters (see models/adapters/lora.py).

    Defaults match the Phase 5 decision recorded in the ROADMAP: r=4,
    target only decoder v_proj — conservative rank + single-projection
    targeting to guard against re-introducing the Phase 4 fine-tune's
    catastrophic calibration drift. Encoder stays fully frozen because
    ViT-style encoders use 'query'/'value' naming, so a target_modules
    list of ['v_proj'] naturally matches only the decoder without needing
    regex tricks. Override target_modules if a future backbone uses
    different attention naming.
    """

    r: int = 4
    alpha: int = 8
    dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: ("v_proj",))
    bias: str = "none"

    @classmethod
    def from_dict(cls, data: dict) -> LoraAdapterConfig:
        return cls(
            r=int(data.get("r", 4)),
            alpha=int(data.get("alpha", 8)),
            dropout=float(data.get("dropout", 0.05)),
            target_modules=tuple(data.get("target_modules", ("v_proj",))),
            bias=str(data.get("bias", "none")),
        )
