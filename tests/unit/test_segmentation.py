import numpy as np

from segmentation.character_segmentation import segment_characters
from segmentation.line_segmentation import segment_lines
from segmentation.word_segmentation import segment_words


def _blank(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def test_segment_lines_finds_separated_bands():
    img = _blank(200, 100)
    img[10:30, :] = 255  # line 1
    img[60:80, :] = 255  # line 2
    img[150:170, :] = 255  # line 3

    lines = segment_lines(img, min_line_height=5, min_gap=3)

    assert len(lines) == 3
    assert lines[0].top == 10 and lines[0].bottom == 30
    assert lines[1].top == 60 and lines[1].bottom == 80
    assert lines[2].top == 150 and lines[2].bottom == 170


def test_segment_lines_merges_small_gaps_within_a_line():
    img = _blank(100, 100)
    img[10:20, :] = 255
    img[22:30, :] = 255  # gap of 2 rows — smaller than min_gap, should merge

    lines = segment_lines(img, min_line_height=5, min_gap=3)

    assert len(lines) == 1
    assert lines[0].top == 10 and lines[0].bottom == 30


def test_segment_lines_drops_noise_bands_shorter_than_min_height():
    img = _blank(100, 100)
    img[10:12, :] = 255  # 2px tall — noise
    img[50:70, :] = 255  # real line

    lines = segment_lines(img, min_line_height=5, min_gap=3)

    assert len(lines) == 1
    assert lines[0].top == 50


def test_segment_lines_empty_for_blank_image():
    assert segment_lines(_blank(100, 100)) == []


def test_segment_words_finds_word_gaps():
    img = _blank(50, 300)
    img[:, 10:40] = 255  # word 1
    img[:, 60:100] = 255  # word 2 (gap of 20 — larger than min_gap)
    img[:, 250:280] = 255  # word 3

    words = segment_words(img, min_gap=8, min_word_width=3)

    assert len(words) == 3
    assert words[0].left == 10 and words[0].right == 40
    assert words[1].left == 60 and words[1].right == 100
    assert words[2].left == 250 and words[2].right == 280


def test_segment_words_merges_letter_spacing_within_a_word():
    img = _blank(50, 200)
    img[:, 10:20] = 255
    img[:, 24:34] = 255  # gap of 4 — smaller than min_gap, same word

    words = segment_words(img, min_gap=8, min_word_width=3)

    assert len(words) == 1
    assert words[0].left == 10 and words[0].right == 34


def test_segment_characters_returns_sorted_non_touching_boxes():
    img = _blank(50, 200)
    img[10:30, 100:120] = 255  # rightmost char drawn first
    img[10:30, 10:30] = 255  # leftmost char

    boxes = segment_characters(img, min_component_area=10)

    assert len(boxes) == 2
    assert boxes[0].x < boxes[1].x
    assert boxes[0].x == 10
    assert boxes[1].x == 100


def test_segment_characters_filters_tiny_noise_components():
    img = _blank(50, 200)
    img[10:30, 10:30] = 255  # real character, area 400
    img[5, 150] = 255  # single-pixel noise

    boxes = segment_characters(img, min_component_area=10)

    assert len(boxes) == 1
    assert boxes[0].x == 10
