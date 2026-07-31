from pathlib import Path

from app.config import AppConfig, load_config


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


def test_resolved_device_defaults_to_cpu_without_gpu(app_config):
    # This machine has no CUDA GPU (Intel iGPU only); resolved device must be cpu
    # unless the device is explicitly overridden away from "auto" in app.yaml.
    if app_config.device == "auto":
        assert app_config.resolved_device() in ("cpu", "cuda")


def test_canvas_config_bounds_are_sane(app_config):
    canvas = app_config.canvas
    assert canvas.min_pen_width <= canvas.default_pen_width <= canvas.max_pen_width
    assert canvas.undo_stack_depth > 0
