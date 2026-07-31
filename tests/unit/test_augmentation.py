import random
from dataclasses import replace

import numpy as np

from preprocessing.augmentation import (
    AugmentationConfig,
    augment_image,
    build_augmentation_pipeline,
    load_augmentation_config,
)


def _sample_image(h=64, w=128) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)
    img[20:40, 30:90] = 0  # a block of "ink"
    return img


def _all_probability_one(config: AugmentationConfig) -> AugmentationConfig:
    """Force every transform to apply, so a single run exercises the whole pipeline."""
    return replace(
        config,
        rotate=replace(config.rotate, p=1.0),
        shear=replace(config.shear, p=1.0),
        scale=replace(config.scale, p=1.0),
        perspective=replace(config.perspective, p=1.0),
        elastic=replace(config.elastic, p=1.0),
        blur=replace(config.blur, p=1.0),
        noise=replace(config.noise, p=1.0),
        brightness_contrast=replace(config.brightness_contrast, p=1.0),
        compression=replace(config.compression, p=1.0),
        crop_pad=replace(config.crop_pad, p=1.0),
        pen_thickness=replace(config.pen_thickness, p=1.0),
        ink_fading=replace(config.ink_fading, p=1.0),
        paper_texture=replace(config.paper_texture, p=1.0),
    )


def test_load_augmentation_config_returns_all_sections():
    config = load_augmentation_config()
    assert 0 <= config.rotate.p <= 1
    assert 0 <= config.fill_value <= 255


def test_pipeline_runs_end_to_end_with_every_transform_forced_on():
    config = _all_probability_one(load_augmentation_config())
    pipeline = build_augmentation_pipeline(config)
    image = _sample_image()

    result = augment_image(image, pipeline)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_pipeline_is_a_noop_when_all_probabilities_zero():
    config = load_augmentation_config()
    zeroed = replace(
        config,
        rotate=replace(config.rotate, p=0.0),
        shear=replace(config.shear, p=0.0),
        scale=replace(config.scale, p=0.0),
        perspective=replace(config.perspective, p=0.0),
        elastic=replace(config.elastic, p=0.0),
        blur=replace(config.blur, p=0.0),
        noise=replace(config.noise, p=0.0),
        brightness_contrast=replace(config.brightness_contrast, p=0.0),
        compression=replace(config.compression, p=0.0),
        crop_pad=replace(config.crop_pad, p=0.0),
        pen_thickness=replace(config.pen_thickness, p=0.0),
        ink_fading=replace(config.ink_fading, p=0.0),
        paper_texture=replace(config.paper_texture, p=0.0),
    )
    pipeline = build_augmentation_pipeline(zeroed)
    image = _sample_image()

    result = augment_image(image, pipeline)

    np.testing.assert_array_equal(result, image)


def test_ink_fading_lightens_pure_black_ink():
    config = load_augmentation_config()
    only_fading = _all_probability_one(config)
    only_fading = replace(
        only_fading,
        rotate=replace(only_fading.rotate, p=0.0),
        shear=replace(only_fading.shear, p=0.0),
        scale=replace(only_fading.scale, p=0.0),
        perspective=replace(only_fading.perspective, p=0.0),
        elastic=replace(only_fading.elastic, p=0.0),
        blur=replace(only_fading.blur, p=0.0),
        noise=replace(only_fading.noise, p=0.0),
        brightness_contrast=replace(only_fading.brightness_contrast, p=0.0),
        compression=replace(only_fading.compression, p=0.0),
        crop_pad=replace(only_fading.crop_pad, p=0.0),
        pen_thickness=replace(only_fading.pen_thickness, p=0.0),
        paper_texture=replace(only_fading.paper_texture, p=0.0),
    )
    pipeline = build_augmentation_pipeline(only_fading)
    image = _sample_image()

    result = augment_image(image, pipeline)

    assert result[30, 60] > 0  # ink pixel is no longer pure black


def test_reproducible_with_same_seed_and_global_rng_state():
    config = replace(_all_probability_one(load_augmentation_config()), seed=1234)
    image = _sample_image()

    random.seed(0)
    np.random.seed(0)
    result_a = augment_image(image, build_augmentation_pipeline(config))

    random.seed(0)
    np.random.seed(0)
    result_b = augment_image(image, build_augmentation_pipeline(config))

    np.testing.assert_array_equal(result_a, result_b)
