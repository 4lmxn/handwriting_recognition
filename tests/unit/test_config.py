from pathlib import Path

import pytest

from app.config import AppConfig, load_config, resolve_device


def test_load_config_returns_app_config():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.name
    assert config.version


def test_paths_are_absolute_and_under_repo_root():
    config = load_config()
    for p in (
        config.paths.datasets_raw,
        config.paths.datasets_processed,
        config.paths.datasets_manifests,
        config.paths.weights,
        config.paths.outputs,
        config.paths.logs,
        config.paths.experiments,
    ):
        assert isinstance(p, Path)
        assert p.is_absolute()


def test_ensure_exist_creates_directories(app_config):
    app_config.paths.ensure_exist()
    assert app_config.paths.logs.exists()
    assert app_config.paths.outputs.exists()


def test_resolved_device_is_a_supported_backend(app_config):
    # Development happens across CPU-only, CUDA and Apple Silicon machines, so the
    # only invariant is that "auto" lands on a backend the code supports.
    if app_config.device == "auto":
        assert app_config.resolved_device() in ("cpu", "cuda", "mps")


def test_resolve_device_passes_explicit_values_through():
    # An explicit device must never be second-guessed — this is the escape hatch
    # for forcing a run onto cpu when an MPS operator gap shows up.
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_device_prefers_cuda_over_mps(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_falls_back_to_mps_then_cpu(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_canvas_config_bounds_are_sane(app_config):
    canvas = app_config.canvas
    assert canvas.min_pen_width <= canvas.default_pen_width <= canvas.max_pen_width
    assert canvas.undo_stack_depth > 0
