from recognition.config import load_recognition_config


def test_load_recognition_config_returns_expected_fields():
    config = load_recognition_config()
    assert config.model_name
    assert config.max_new_tokens > 0
    assert config.resolved_device() in ("cpu", "cuda", "mps")
    # Repetition guards: committed defaults are >1.0 and >=3 respectively.
    # N=3 (not 2) preserves legitimate words with repeated bigrams like "onion".
    assert config.repetition_penalty >= 1.0
    assert config.no_repeat_ngram_size == 0 or config.no_repeat_ngram_size >= 3


def test_recognition_config_defaults_disable_repetition_guards():
    from recognition.config import RecognitionConfig

    config = RecognitionConfig(model_name="x", device="cpu", max_new_tokens=32)
    assert config.repetition_penalty == 1.0
    assert config.no_repeat_ngram_size == 0
