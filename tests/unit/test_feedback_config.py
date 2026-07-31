from feedback.config import FeedbackConfig, load_feedback_config


def test_load_feedback_config_returns_expected_fields():
    config = load_feedback_config()
    assert config.storage_dir
    assert config.image_dir


def test_storage_and_image_dir_paths_are_absolute():
    config = load_feedback_config()
    assert config.storage_dir_path.is_absolute()
    assert config.image_dir_path.is_absolute()


def test_image_dir_defaults_under_datasets_processed():
    # Load-bearing: HandwritingDataset joins datasets_processed + image_path
    # to find each sample's PNG. Feedback records store image_path as
    # "feedback/<uuid>.png", so image_dir MUST live inside datasets/processed
    # with a trailing "feedback" component. If this ever diverges,
    # `feedback/store.py` needs an updated `image_relative_prefix`.
    config = load_feedback_config()
    assert config.image_dir_path.name == "feedback"
    assert config.image_dir_path.parent.name == "processed"


def test_feedback_config_can_be_constructed_directly():
    # Enables tests + one-off scripts to build a temp config without going
    # through YAML.
    config = FeedbackConfig(storage_dir="a", image_dir="b/c")
    assert config.storage_dir == "a"
    assert config.image_dir == "b/c"
