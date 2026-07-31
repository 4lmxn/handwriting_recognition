"""Augmentation pipeline (Albumentations-based), applied on the fly at
training/personalization time — datasets/processed/ images themselves stay
clean, so the same sample can be re-augmented differently every epoch.

Random spacing (mentioned in docs/ROADMAP.md alongside these transforms) is
not implemented here: it only makes sense at render/layout time, before text
is flattened into a single image, so it's exercised by
datasets/sources/synthetic.py's jitter rather than as a pixel-level transform.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
import yaml

from app.config import CONFIGS_DIR


@dataclass(frozen=True)
class RotateConfig:
    limit: float
    p: float


@dataclass(frozen=True)
class ShearConfig:
    limit: float
    p: float


@dataclass(frozen=True)
class ScaleConfig:
    limit: float
    p: float


@dataclass(frozen=True)
class PerspectiveConfig:
    scale: list[float]
    p: float


@dataclass(frozen=True)
class ElasticConfig:
    alpha: float
    sigma: float
    p: float


@dataclass(frozen=True)
class BlurConfig:
    limit: int
    p: float


@dataclass(frozen=True)
class NoiseConfig:
    std_range: list[float]
    p: float


@dataclass(frozen=True)
class BrightnessContrastConfig:
    brightness_limit: float
    contrast_limit: float
    p: float


@dataclass(frozen=True)
class CompressionConfig:
    quality_range: list[int]
    p: float


@dataclass(frozen=True)
class CropPadConfig:
    percent: float
    p: float


@dataclass(frozen=True)
class PenThicknessConfig:
    scale: list[int]
    p: float


@dataclass(frozen=True)
class InkFadingConfig:
    min_alpha: float
    max_alpha: float
    p: float


@dataclass(frozen=True)
class PaperTextureConfig:
    intensity: float
    p: float


@dataclass(frozen=True)
class AugmentationConfig:
    seed: int | None
    fill_value: int
    rotate: RotateConfig
    shear: ShearConfig
    scale: ScaleConfig
    perspective: PerspectiveConfig
    elastic: ElasticConfig
    blur: BlurConfig
    noise: NoiseConfig
    brightness_contrast: BrightnessContrastConfig
    compression: CompressionConfig
    crop_pad: CropPadConfig
    pen_thickness: PenThicknessConfig
    ink_fading: InkFadingConfig
    paper_texture: PaperTextureConfig


def load_augmentation_config() -> AugmentationConfig:
    path = CONFIGS_DIR / "augmentation.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    return AugmentationConfig(
        seed=data["seed"],
        fill_value=data["fill_value"],
        rotate=RotateConfig(**data["rotate"]),
        shear=ShearConfig(**data["shear"]),
        scale=ScaleConfig(**data["scale"]),
        perspective=PerspectiveConfig(**data["perspective"]),
        elastic=ElasticConfig(**data["elastic"]),
        blur=BlurConfig(**data["blur"]),
        noise=NoiseConfig(**data["noise"]),
        brightness_contrast=BrightnessContrastConfig(**data["brightness_contrast"]),
        compression=CompressionConfig(**data["compression"]),
        crop_pad=CropPadConfig(**data["crop_pad"]),
        pen_thickness=PenThicknessConfig(**data["pen_thickness"]),
        ink_fading=InkFadingConfig(**data["ink_fading"]),
        paper_texture=PaperTextureConfig(**data["paper_texture"]),
    )


def _ink_fading(min_alpha: float, max_alpha: float):
    def apply(image: np.ndarray, **kwargs) -> np.ndarray:
        alpha = random.uniform(min_alpha, max_alpha)
        background = np.full_like(image, 255)
        faded = cv2.addWeighted(image, alpha, background, 1 - alpha, 0)
        return faded.astype(image.dtype)

    return apply


def _paper_texture(intensity: float):
    def apply(image: np.ndarray, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        low_res = np.random.normal(0, 1, size=(h // 8 + 1, w // 8 + 1)).astype(np.float32)
        noise = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6) * 255.0
        blended = image.astype(np.float32) * (1 - intensity) + noise * intensity
        return np.clip(blended, 0, 255).astype(image.dtype)

    return apply


def build_augmentation_pipeline(config: AugmentationConfig) -> A.Compose:
    fill = float(config.fill_value)
    blur_limit = config.blur.limit if config.blur.limit % 2 == 1 else config.blur.limit + 1

    transforms = [
        A.Affine(rotate=(-config.rotate.limit, config.rotate.limit), fill=fill, p=config.rotate.p),
        A.Affine(shear=(-config.shear.limit, config.shear.limit), fill=fill, p=config.shear.p),
        A.Affine(
            scale=(1 - config.scale.limit, 1 + config.scale.limit), fill=fill, p=config.scale.p
        ),
        A.Perspective(scale=tuple(config.perspective.scale), fill=fill, p=config.perspective.p),
        A.ElasticTransform(
            alpha=config.elastic.alpha, sigma=config.elastic.sigma, fill=fill, p=config.elastic.p
        ),
        A.GaussianBlur(blur_limit=(3, blur_limit), p=config.blur.p),
        A.GaussNoise(std_range=tuple(config.noise.std_range), p=config.noise.p),
        A.RandomBrightnessContrast(
            brightness_limit=config.brightness_contrast.brightness_limit,
            contrast_limit=config.brightness_contrast.contrast_limit,
            p=config.brightness_contrast.p,
        ),
        A.ImageCompression(
            quality_range=tuple(config.compression.quality_range), p=config.compression.p
        ),
        A.CropAndPad(
            percent=(-config.crop_pad.percent, config.crop_pad.percent),
            fill=fill,
            p=config.crop_pad.p,
        ),
        A.OneOf(
            [
                A.Morphological(
                    scale=tuple(config.pen_thickness.scale), operation="dilation", p=1.0
                ),
                A.Morphological(
                    scale=tuple(config.pen_thickness.scale), operation="erosion", p=1.0
                ),
            ],
            p=config.pen_thickness.p,
        ),
        A.Lambda(
            image=_ink_fading(config.ink_fading.min_alpha, config.ink_fading.max_alpha),
            p=config.ink_fading.p,
        ),
        A.Lambda(
            image=_paper_texture(config.paper_texture.intensity),
            p=config.paper_texture.p,
        ),
    ]
    return A.Compose(transforms, seed=config.seed)


def augment_image(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    return pipeline(image=image)["image"]
