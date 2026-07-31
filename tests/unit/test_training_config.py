from pathlib import Path

from training.config import load_training_config


def test_load_training_config_returns_expected_fields():
    config = load_training_config()
    assert config.model_name
    assert config.batch_size > 0
    assert config.num_epochs > 0
    assert config.resolved_device() in ("cpu", "cuda")


def test_checkpoint_and_log_dir_paths_are_absolute():
    config = load_training_config()
    assert isinstance(config.checkpoint_dir_path, Path)
    assert config.checkpoint_dir_path.is_absolute()
    assert isinstance(config.log_dir_path, Path)
    assert config.log_dir_path.is_absolute()


def test_confusion_matrix_full_path_is_none_by_default():
    config = load_training_config()
    assert config.confusion_matrix_path is None
    assert config.confusion_matrix_full_path is None


def test_confusion_matrix_full_path_resolves_when_set():
    from dataclasses import replace

    config = replace(load_training_config(), confusion_matrix_path="experiments/foo.json")
    assert config.confusion_matrix_full_path is not None
    assert config.confusion_matrix_full_path.is_absolute()
    assert config.confusion_matrix_full_path.name == "foo.json"
