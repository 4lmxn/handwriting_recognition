import cv2
import numpy as np
import pytest

from preprocessing.image_ops import (
    adaptive_threshold,
    connected_components,
    denoise,
    deskew,
    estimate_baseline,
    estimate_skew_angle,
    horizontal_projection_profile,
    normalize_intensity,
    resize_and_pad,
    to_grayscale,
    vertical_projection_profile,
)


def _blank(h=100, w=200, channels=None) -> np.ndarray:
    shape = (h, w) if channels is None else (h, w, channels)
    return np.zeros(shape, dtype=np.uint8)


def _binary_rect(h=100, w=200, rect=(50, 30, 150, 70)) -> np.ndarray:
    img = _blank(h, w)
    x0, y0, x1, y1 = rect
    img[y0:y1, x0:x1] = 255
    return img


def test_to_grayscale_passes_through_2d():
    gray = _blank()
    assert to_grayscale(gray) is gray


def test_to_grayscale_converts_bgr():
    color = np.full((50, 60, 3), 128, dtype=np.uint8)
    result = to_grayscale(color)
    assert result.shape == (50, 60)
    assert result.dtype == np.uint8


def test_to_grayscale_converts_bgra():
    color = np.full((50, 60, 4), 128, dtype=np.uint8)
    result = to_grayscale(color)
    assert result.shape == (50, 60)


def test_adaptive_threshold_marks_dark_text_as_ink():
    gray = np.full((100, 100), 255, dtype=np.uint8)
    gray[40:60, 40:60] = 0  # dark "text" block on light background
    binary = adaptive_threshold(gray)
    assert binary[50, 50] == 255  # ink where text was
    assert binary.dtype == np.uint8


def test_adaptive_threshold_handles_even_block_size():
    gray = np.full((60, 60), 200, dtype=np.uint8)
    # should not raise despite even block_size
    adaptive_threshold(gray, block_size=10)


def test_denoise_preserves_shape_and_dtype():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, size=(80, 80), dtype=np.uint8)
    result = denoise(noisy)
    assert result.shape == noisy.shape
    assert result.dtype == np.uint8


def test_estimate_skew_angle_zero_for_blank_image():
    assert estimate_skew_angle(_blank()) == 0.0


def test_estimate_skew_angle_recovers_known_rotation():
    img = _binary_rect(h=200, w=200, rect=(60, 90, 140, 110))
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))

    angle = estimate_skew_angle(rotated)
    assert abs(abs(angle) - 15) < 3


def test_deskew_reduces_skew_of_rotated_image():
    img = _binary_rect(h=200, w=200, rect=(60, 90, 140, 110))
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h))

    corrected, angle = deskew(rotated)
    assert abs(angle) > 0
    residual_angle = estimate_skew_angle(corrected)
    assert abs(residual_angle) < abs(angle)


def test_deskew_noop_on_already_straight_image():
    img = _binary_rect()
    corrected, angle = deskew(img)
    assert angle == 0.0
    np.testing.assert_array_equal(corrected, img)


def test_connected_components_counts_separate_blobs():
    img = _blank(h=100, w=200)
    img[10:30, 10:30] = 255
    img[10:30, 100:130] = 255
    num_labels, labels, stats, centroids = connected_components(img)
    assert num_labels == 3  # background + 2 blobs
    assert labels.shape == img.shape
    assert stats.shape[0] == 3


def test_projection_profiles_match_known_rectangle():
    img = _binary_rect(h=100, w=200, rect=(50, 30, 150, 70))
    h_profile = horizontal_projection_profile(img)
    v_profile = vertical_projection_profile(img)
    assert h_profile[50] == 100  # rect width
    assert h_profile[0] == 0
    assert v_profile[100] == 40  # rect height
    assert v_profile[0] == 0


def test_estimate_baseline_within_bounds():
    img = _binary_rect(h=100, w=200, rect=(20, 30, 180, 70))
    baseline = estimate_baseline(img)
    assert 0 <= baseline < img.shape[0]


def test_estimate_baseline_blank_image_returns_midpoint():
    img = _blank(h=100, w=200)
    assert estimate_baseline(img) == 50


@pytest.mark.parametrize("target_size", [(64, 64), (32, 128), (128, 32)])
def test_resize_and_pad_produces_exact_target_size(target_size):
    img = _binary_rect(h=50, w=150, rect=(10, 10, 140, 40))
    result = resize_and_pad(img, target_size, pad_value=255)
    assert result.shape == target_size


def test_resize_and_pad_preserves_color_channels():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    result = resize_and_pad(img, (60, 60), pad_value=255)
    assert result.shape == (60, 60, 3)


def test_normalize_intensity_scales_to_unit_range():
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = normalize_intensity(gray)
    assert normalized.dtype == np.float32
    assert normalized[0, 0] == pytest.approx(0.0)
    assert normalized[0, 2] == pytest.approx(1.0)
