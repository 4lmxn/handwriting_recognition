"""Core image preprocessing operations shared by every dataset source and the
inference pipeline. Canonical types: grayscale images are uint8 HxW arrays;
binary images are uint8 HxW arrays valued {0, 255} with ink as 255.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_threshold(gray: np.ndarray, block_size: int = 35, c: int = 11) -> np.ndarray:
    """Binarize with ink as 255. block_size must be odd and >= 3."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def denoise(gray: np.ndarray, strength: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=strength)


def estimate_skew_angle(binary_image: np.ndarray) -> float:
    """Estimate rotation (degrees) needed to make text horizontal, via the
    minimum-area bounding rectangle of ink pixels. Returns 0.0 for blank input."""
    coords = cv2.findNonZero(binary_image)
    if coords is None:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def deskew(binary_image: np.ndarray, border_value: int = 0) -> tuple[np.ndarray, float]:
    angle = estimate_skew_angle(binary_image)
    if abs(angle) < 1e-3:
        return binary_image.copy(), 0.0
    h, w = binary_image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(float(border_value),),
    )
    return rotated, angle


def connected_components(
    binary_image: np.ndarray, connectivity: int = 8
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (num_labels, labels, stats, centroids) — see cv2.connectedComponentsWithStats.
    num_labels includes the background label (0)."""
    return cv2.connectedComponentsWithStats(binary_image, connectivity=connectivity)


def horizontal_projection_profile(binary_image: np.ndarray) -> np.ndarray:
    return np.sum(binary_image > 0, axis=1)


def vertical_projection_profile(binary_image: np.ndarray) -> np.ndarray:
    return np.sum(binary_image > 0, axis=0)


def estimate_baseline(binary_image: np.ndarray) -> int:
    """Estimate the baseline row for a single text line image: the row with the
    steepest drop in ink density immediately after the profile's peak (i.e. where
    most glyphs stop, ignoring descenders). Returns the vertical midpoint if the
    image is blank."""
    profile = horizontal_projection_profile(binary_image)
    if profile.max() == 0:
        return binary_image.shape[0] // 2
    peak_row = int(np.argmax(profile))
    tail = profile[peak_row:]
    if len(tail) < 2:
        return peak_row
    gradient = np.diff(tail.astype(np.int32))
    steepest_drop_offset = int(np.argmin(gradient))
    return peak_row + steepest_drop_offset


def resize_and_pad(
    image: np.ndarray, target_size: tuple[int, int], pad_value: int = 255
) -> np.ndarray:
    """Resize preserving aspect ratio, then letterbox-pad to exactly target_size (h, w)."""
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas_shape = (target_h, target_w) if image.ndim == 2 else (target_h, target_w, image.shape[2])
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def normalize_intensity(gray: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] to float32 [0.0, 1.0]."""
    return gray.astype(np.float32) / 255.0
