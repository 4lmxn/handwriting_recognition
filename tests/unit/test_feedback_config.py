from feedback.config import FeedbackConfig, load_feedback_config
from models.adapters.config import LoraAdapterConfig


def test_load_feedback_config_returns_expected_fields():
    config = load_feedback_config()
    assert config.storage_dir
    assert config.image_dir
    assert config.adapter_dir
    assert isinstance(config.adapter, LoraAdapterConfig)


def test_storage_image_and_adapter_dir_paths_are_absolute():
    config = load_feedback_config()
    assert config.storage_dir_path.is_absolute()
    assert config.image_dir_path.is_absolute()
    assert config.adapter_dir_path.is_absolute()


def test_image_dir_defaults_under_datasets_processed():
    # Load-bearing: HandwritingDataset joins datasets_processed + image_path
    # to find each sample's PNG. Feedback records store image_path as
    # "feedback/<uuid>.png", so image_dir MUST live inside datasets/processed
    # with a trailing "feedback" component. If this ever diverges,
    # `feedback/store.py` needs an updated `image_relative_prefix`.
    config = load_feedback_config()
    assert config.image_dir_path.name == "feedback"
    assert config.image_dir_path.parent.name == "processed"


def test_adapter_defaults_match_phase5_decision():
    # Locked-in defaults per Phase 5: conservative rank + decoder-only
    # targeting to preserve base-model calibration. If you're changing
    # these, revisit memory:phase5_decisions.md and docs/ROADMAP.md first.
    config = load_feedback_config()
    assert config.adapter.r == 4
    assert config.adapter.target_modules == ("v_proj",)
    assert config.adapter.bias == "none"


def test_feedback_config_can_be_constructed_directly():
    # Enables tests + one-off scripts to build a temp config without going
    # through YAML.
    config = FeedbackConfig(
        storage_dir="a",
        image_dir="b/c",
        adapter=LoraAdapterConfig(),
        adapter_dir="weights/adapters",
    )
    assert config.storage_dir == "a"
    assert config.image_dir == "b/c"
    assert config.adapter_dir == "weights/adapters"
    assert config.adapter.r == 4
