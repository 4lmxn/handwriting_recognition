"""Mixed-precision gating for the fine-tuning loop.

The training loop is written so that a single config (use_amp: true) is safe on
every machine this project runs on — fp16 on CUDA, plain fp32 everywhere else.
"""

from training.train import amp_enabled_for


def test_amp_enabled_on_cuda_when_requested():
    assert amp_enabled_for("cuda", use_amp=True) is True


def test_amp_disabled_on_cuda_when_not_requested():
    assert amp_enabled_for("cuda", use_amp=False) is False


def test_amp_disabled_on_non_cuda_devices_even_when_requested():
    # GradScaler has no MPS/CPU implementation, so a config carried over from the
    # CUDA machine must degrade to fp32 rather than crash.
    assert amp_enabled_for("mps", use_amp=True) is False
    assert amp_enabled_for("cpu", use_amp=True) is False
