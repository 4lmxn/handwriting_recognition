from pathlib import Path

from training.config import load_training_config


def test_load_training_config_returns_expected_fields():
    config = load_training_config()
    assert config.model_name
    assert config.batch_size > 0
    assert config.num_epochs > 0
    assert config.resolved_device() in ("cpu", "cuda", "mps")
    assert isinstance(config.use_amp, bool)


def test_checkpoint_and_log_dir_paths_are_absolute():
    config = load_training_config()
    assert isinstance(config.checkpoint_dir_path, Path)
    assert config.checkpoint_dir_path.is_absolute()
    assert isinstance(config.log_dir_path, Path)
    assert config.log_dir_path.is_absolute()


def test_confusion_matrix_full_path_matches_configured_default():
    # The committed default now points at the Phase 4 hard-negative-mining
    # confusion matrix built from the CVL fine-tune — see docs/ROADMAP.md.
    config = load_training_config()
    assert config.confusion_matrix_path == "experiments/confusion_matrix_cvl.json"
    assert config.confusion_matrix_full_path is not None
    assert config.confusion_matrix_full_path.is_absolute()
    assert config.confusion_matrix_full_path.name == "confusion_matrix_cvl.json"


def test_confusion_matrix_full_path_is_none_when_unset():
    from dataclasses import replace

    config = replace(load_training_config(), confusion_matrix_path=None)
    assert config.confusion_matrix_full_path is None


def test_confusion_matrix_full_path_resolves_arbitrary_paths():
    from dataclasses import replace

    config = replace(load_training_config(), confusion_matrix_path="experiments/foo.json")
    assert config.confusion_matrix_full_path is not None
    assert config.confusion_matrix_full_path.is_absolute()
    assert config.confusion_matrix_full_path.name == "foo.json"
