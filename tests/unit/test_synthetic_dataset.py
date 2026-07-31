from PIL import Image

from datasets.config import SyntheticConfig, load_datasets_config
from datasets.sources.synthetic import SyntheticDatasetSource, _assign_split, _existing_fonts


def _small_config(**overrides) -> SyntheticConfig:
    real_config = load_datasets_config().synthetic
    defaults = dict(
        fonts=real_config.fonts,
        font_sizes=[24],
        image_height=48,
        samples_per_item=2,
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        characters="0Ol",
        words=["turn", "arm"],
    )
    defaults.update(overrides)
    return SyntheticConfig(**defaults)


def test_prepare_generates_expected_sample_count(tmp_path):
    source = SyntheticDatasetSource(_small_config(), seed=1)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    # 3 characters + 2 words, samples_per_item=2 each
    assert len(samples) == (3 + 2) * 2


def test_prepare_writes_valid_images(tmp_path):
    source = SyntheticDatasetSource(_small_config(), seed=1)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    for sample in samples:
        full_path = tmp_path / sample.image_path
        assert full_path.exists()
        with Image.open(full_path) as img:
            assert img.height == 48
            assert img.mode == "L"


def test_prepare_assigns_correct_label_types(tmp_path):
    source = SyntheticDatasetSource(_small_config(), seed=1)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)

    char_samples = [s for s in samples if s.transcript in "0Ol"]
    word_samples = [s for s in samples if s.transcript in ("turn", "arm")]

    assert all(s.label_type == "character" for s in char_samples)
    assert all(s.label_type == "word" for s in word_samples)
    assert len(char_samples) == 3 * 2
    assert len(word_samples) == 2 * 2


def test_prepare_sets_source_name(tmp_path):
    source = SyntheticDatasetSource(_small_config(), seed=1)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)
    assert all(s.source == "synthetic" for s in samples)


def test_split_assignment_is_deterministic():
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    first = _assign_split("hello:0", ratios)
    second = _assign_split("hello:0", ratios)
    assert first == second


def test_split_assignment_respects_ratios_roughly():
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    splits = [_assign_split(f"item{i}", ratios) for i in range(2000)]
    train_fraction = splits.count("train") / len(splits)
    assert 0.7 < train_fraction < 0.9


def test_existing_fonts_filters_missing_paths(tmp_path):
    # A real file on disk, not a system font path: the configured font list spans
    # Linux/macOS/Windows locations, so none of them exists on every dev machine.
    real_font = tmp_path / "exists.ttf"
    real_font.touch()
    fake_font = tmp_path / "does_not_exist.ttf"

    result = _existing_fonts([str(real_font), str(fake_font)])

    assert str(real_font) in result
    assert str(fake_font) not in result


def test_configured_fonts_include_some_for_this_platform():
    # Guards the cross-platform font list: if every configured path is missing,
    # synthetic generation silently degrades to PIL's default bitmap font.
    assert _existing_fonts(load_datasets_config().synthetic.fonts)


def test_prepare_falls_back_to_default_font_when_none_exist(tmp_path):
    config = _small_config(fonts=["/nonexistent/font.ttf"])
    source = SyntheticDatasetSource(config, seed=1)
    samples = source.prepare(raw_dir=tmp_path, processed_dir=tmp_path)
    assert len(samples) > 0
