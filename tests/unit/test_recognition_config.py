from recognition.config import load_recognition_config


def test_load_recognition_config_returns_expected_fields():
    config = load_recognition_config()
    assert config.model_name
    assert config.max_new_tokens > 0
    assert config.resolved_device() in ("cpu", "cuda")
