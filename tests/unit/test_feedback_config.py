from feedback.config import (
    FeedbackConfig,
    IncrementalEvalConfig,
    IncrementalTrainingConfig,
    ReplayConfig,
    load_feedback_config,
)
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
    # through YAML. Nested configs default to sensible YAML-load defaults.
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
    # Nested configs should have safe defaults so tests don't need to
    # spell out every field.
    assert isinstance(config.replay, ReplayConfig)
    assert isinstance(config.training, IncrementalTrainingConfig)
    assert isinstance(config.eval, IncrementalEvalConfig)


def test_replay_config_defaults_match_phase5_decision():
    config = load_feedback_config()
    # Locked-in per memory:phase5_decisions.md — max_corrections=200 is
    # part of the "replay all + cap" policy. Don't quietly drop the cap.
    assert config.replay.max_corrections == 200
    assert 0.0 <= config.replay.replay_ratio <= 1.0
    assert config.replay.min_pending_corrections > 0


def test_eval_config_has_regression_gate():
    config = load_feedback_config()
    # Load-bearing per Phase 4 catastrophic-drift evidence: adapter
    # updates MUST be gated on max_cer_regression. Never remove this
    # field or set it to something huge without revisiting the ROADMAP.
    assert config.eval.max_cer_regression > 0.0
