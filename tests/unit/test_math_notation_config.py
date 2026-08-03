"""Unit tests for math_notation.config and math_notation.types
(Phase 8, PR 1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from math_notation.config import MathNotationConfig, load_math_notation_config
from math_notation.types import (
    Fraction,
    MathFragment,
    Root,
    ScriptedChar,
    ScriptPosition,
)

# -- config -----------------------------------------------------------


def test_config_defaults():
    config = MathNotationConfig()
    assert config.enabled is False
    assert config.super_top_ratio > 0
    assert config.sub_bottom_ratio > 0
    assert 0 < config.max_script_size_ratio < 1
    assert 0 < config.min_fraction_bar_length_ratio <= 1
    assert config.max_fraction_bar_thickness >= 1


def test_config_from_dict_reads_all_keys():
    config = MathNotationConfig.from_dict(
        {
            "enabled": True,
            "super_top_ratio": 0.5,
            "sub_bottom_ratio": 0.4,
            "max_script_size_ratio": 0.6,
            "min_fraction_bar_length_ratio": 0.25,
            "max_fraction_bar_thickness": 6,
        }
    )
    assert config.enabled is True
    assert config.super_top_ratio == 0.5
    assert config.sub_bottom_ratio == 0.4
    assert config.max_script_size_ratio == 0.6
    assert config.min_fraction_bar_length_ratio == 0.25
    assert config.max_fraction_bar_thickness == 6


def test_config_from_dict_uses_defaults_for_missing_keys():
    config = MathNotationConfig.from_dict({"enabled": True})
    assert config.enabled is True
    # Everything else falls back to the dataclass defaults.
    assert config.super_top_ratio == 0.2


def test_load_math_notation_config_reads_yaml():
    # The shipped configs/math_notation.yaml exists and is well-formed;
    # the default state must be "off" so no user is surprised.
    config = load_math_notation_config()
    assert config.enabled is False


# -- ScriptPosition ---------------------------------------------------


def test_script_position_enum_values():
    assert ScriptPosition.BASELINE.value == "baseline"
    assert ScriptPosition.SUPERSCRIPT.value == "superscript"
    assert ScriptPosition.SUBSCRIPT.value == "subscript"


# -- ScriptedChar -----------------------------------------------------


def test_scripted_char_height_and_width_derived():
    c = ScriptedChar(
        text="2",
        position=ScriptPosition.SUPERSCRIPT,
        top=10,
        bottom=25,
        left=100,
        right=115,
    )
    assert c.height == 15
    assert c.width == 15


def test_scripted_char_is_frozen():
    c = ScriptedChar(
        text="x",
        position=ScriptPosition.BASELINE,
        top=0,
        bottom=10,
        left=0,
        right=10,
    )
    with pytest.raises(FrozenInstanceError):
        c.text = "y"  # type: ignore[misc]


# -- Fraction ---------------------------------------------------------


def test_fraction_carries_numerator_denominator_and_bar_bbox():
    f = Fraction(
        numerator="a",
        denominator="b",
        bar_top=50,
        bar_bottom=52,
        bar_left=10,
        bar_right=40,
    )
    assert f.numerator == "a"
    assert f.denominator == "b"
    assert f.bar_bottom - f.bar_top == 2


# -- Root -------------------------------------------------------------


def test_root_holds_argument_and_bbox():
    r = Root(argument="x+1", top=10, bottom=30, left=100, right=180)
    assert r.argument == "x+1"
    assert r.bottom - r.top == 20


# -- MathFragment -----------------------------------------------------


def test_math_fragment_height_and_width():
    f = MathFragment(latex=r"x^2", top=0, bottom=20, left=100, right=140)
    assert f.height == 20
    assert f.width == 40


def test_math_fragment_carries_arbitrary_latex():
    # The dataclass doesn't validate LaTeX — that's the emitter's job.
    # Verify raw pass-through so future PRs know they can put whatever
    # string they've built here.
    f = MathFragment(
        latex=r"\frac{\sqrt{a+b}}{c^2}", top=0, bottom=40, left=0, right=100
    )
    assert f.latex == r"\frac{\sqrt{a+b}}{c^2}"
