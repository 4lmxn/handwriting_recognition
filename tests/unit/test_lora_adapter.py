import torch
import torch.nn as nn
from peft import PeftModel

from models.adapters.config import LoraAdapterConfig
from models.adapters.lora import (
    _to_peft_config,
    apply_lora_to_trocr,
    load_adapter,
    save_adapter,
    trainable_parameter_count,
)


class FakeTrOCR(nn.Module):
    """Miniature TrOCR-shaped module for adapter tests.

    Encoder uses ViT-style "query"/"value" naming; decoder uses TrOCR-style
    "q_proj"/"v_proj". Naming matches the real backbones so
    `target_modules=["v_proj"]` behaves exactly as it would on the real
    model — that behavior is the whole reason we don't need a regex.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "attn": nn.ModuleDict(
                    {"query": nn.Linear(16, 16), "value": nn.Linear(16, 16)}
                )
            }
        )
        self.decoder = nn.ModuleDict(
            {
                "self_attn": nn.ModuleDict(
                    {"q_proj": nn.Linear(16, 16), "v_proj": nn.Linear(16, 16)}
                )
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Route through both blocks so all linears are on the trace path
        # (peft ignores unreached modules for gradient purposes).
        return self.decoder["self_attn"]["v_proj"](
            self.decoder["self_attn"]["q_proj"](
                self.encoder["attn"]["value"](self.encoder["attn"]["query"](x))
            )
        )


def test_lora_adapter_config_defaults_match_phase5():
    config = LoraAdapterConfig()
    assert config.r == 4
    assert config.alpha == 8
    assert config.target_modules == ("v_proj",)
    assert config.bias == "none"


def test_lora_adapter_config_from_dict_falls_back_to_defaults():
    partial = LoraAdapterConfig.from_dict({"r": 8})
    assert partial.r == 8
    # Everything else came from the dataclass defaults
    assert partial.alpha == 8
    assert partial.target_modules == ("v_proj",)


def test_to_peft_config_rejects_invalid_bias():
    import pytest

    bad = LoraAdapterConfig(bias="everything")
    with pytest.raises(ValueError, match="bias must be one of"):
        _to_peft_config(bad)


def test_to_peft_config_passes_fields_through():
    src = LoraAdapterConfig(r=8, alpha=16, dropout=0.1, target_modules=("v_proj", "q_proj"))
    peft_config = _to_peft_config(src)
    assert peft_config.r == 8
    assert peft_config.lora_alpha == 16
    assert peft_config.lora_dropout == 0.1
    # peft normalizes target_modules to a set internally, so compare unordered.
    assert set(peft_config.target_modules) == {"v_proj", "q_proj"}


def test_apply_lora_wraps_only_target_modules():
    model = FakeTrOCR()
    peft_model = apply_lora_to_trocr(model, LoraAdapterConfig(r=2))

    trainable_names = sorted(n for n, p in peft_model.named_parameters() if p.requires_grad)
    # Every trainable param should live under a v_proj module, and every
    # v_proj module should have both A and B matrices.
    assert trainable_names, "Expected at least one trainable LoRA param"
    for name in trainable_names:
        assert "v_proj" in name and ("lora_A" in name or "lora_B" in name), name

    # Encoder "value" (ViT naming) MUST stay frozen — that's the invariant
    # that lets us keep target_modules=['v_proj'] instead of needing regex.
    for name in trainable_names:
        assert "encoder" not in name, f"Encoder param unexpectedly trainable: {name}"


def test_apply_lora_freezes_all_base_params():
    model = FakeTrOCR()
    peft_model = apply_lora_to_trocr(model, LoraAdapterConfig(r=2))
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"Adapter param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Base param {name} unexpectedly trainable"


def test_trainable_parameter_count_is_small_fraction_of_base():
    model = FakeTrOCR()
    total_before = sum(p.numel() for p in model.parameters())
    peft_model = apply_lora_to_trocr(model, LoraAdapterConfig(r=2))
    adapter_count = trainable_parameter_count(peft_model)
    # r=2 LoRA on a single 16-dim linear = 2*(2*16) = 64 params. Much
    # smaller than the ~1120-param fake base. Assert < 20% to keep the
    # test resilient to changes in FakeTrOCR's exact param count.
    assert 0 < adapter_count < total_before * 0.2


def test_save_and_load_adapter_round_trip(tmp_path):
    torch.manual_seed(0)
    model = FakeTrOCR()
    # Snapshot the base weights so we can re-load them into a fresh model
    # before attaching the adapter — real usage always pairs an adapter
    # with the exact base weights it was trained against.
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # dropout=0 to remove LoRA-branch dropout randomness between the two
    # forward passes; eval() below is belt-and-braces for the same reason.
    peft_model = apply_lora_to_trocr(model, LoraAdapterConfig(r=2, dropout=0.0))

    # Move adapter matrices to non-zero values — they start near-zero by
    # design, so the round-trip has to preserve real numbers to be meaningful.
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            with torch.no_grad():
                param.copy_(torch.randn_like(param))

    peft_model.eval()
    x = torch.randn(1, 16)
    with torch.no_grad():
        expected = peft_model(x)

    adapter_dir = tmp_path / "v-1-abc"
    save_adapter(peft_model, adapter_dir)
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_config.json").exists()

    reloaded_base = FakeTrOCR()
    reloaded_base.load_state_dict(base_state)
    reloaded = load_adapter(reloaded_base, adapter_dir)
    reloaded.eval()
    assert isinstance(reloaded, PeftModel)
    with torch.no_grad():
        got = reloaded(x)
    assert torch.allclose(expected, got, atol=1e-5)


def test_save_creates_missing_parent_dirs(tmp_path):
    model = FakeTrOCR()
    peft_model = apply_lora_to_trocr(model, LoraAdapterConfig(r=2))
    nested = tmp_path / "deep" / "nested" / "v-1"
    save_adapter(peft_model, nested)
    assert nested.exists()
