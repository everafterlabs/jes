"""Character properties from the vendored Unicode tables, and the coarse folding fallbacks."""

from __future__ import annotations

import re
import unicodedata

import pytest

from jes.text import folding, nfkc
from jes.text.unicode import (
    DEFAULT_IGNORABLE_CODES,
    DEFAULT_IGNORABLE_RE,
    char_class,
    is_bidi_control,
    is_c0_c1,
    is_default_ignorable,
    is_extra_format,
    is_joiner,
    is_kept_control,
    is_registered_variation,
    is_variation_selector,
    ranges_of,
)


def test_ignorable_regex_matches_exactly_the_ignorable_table() -> None:
    matched = {
        code
        for code in range(0x110000)
        if not 0xD800 <= code <= 0xDFFF and DEFAULT_IGNORABLE_RE.fullmatch(chr(code))
    }
    assert matched == DEFAULT_IGNORABLE_CODES
    assert all(is_default_ignorable(code) for code in (0x200B, 0x00AD, 0xFEFF, 0xE0041))
    assert not is_default_ignorable(ord("a"))


def test_character_predicates() -> None:
    assert is_bidi_control(0x202E) and is_bidi_control(0x2066)
    assert not is_bidi_control(ord("a"))
    assert is_variation_selector(0xFE0F) and is_variation_selector(0xE0100)
    assert not is_variation_selector(0xFE10)
    assert is_joiner(0x200D) and is_joiner(0x200C) and not is_joiner(0x200B)
    assert is_c0_c1(0x00) and is_c0_c1(0x1B) and is_c0_c1(0x85) and not is_c0_c1(0xA0)
    assert is_kept_control(0x09) and is_kept_control(0x0A) and is_kept_control(0x0D)
    assert not is_kept_control(0x0B)
    assert is_registered_variation("\u2764\ufe0f")  # heavy black heart, emoji style
    assert not is_registered_variation("a\ufe0f")
    assert is_extra_format("\u2060") and is_extra_format("\ue000")
    assert not is_extra_format("a")


def test_char_class_and_ranges() -> None:
    assert ranges_of([5, 1, 2, 3, 3, 9]) == [(1, 3), (5, 5), (9, 9)]
    pattern = f"[{char_class([(ord('-'), ord('-')), (ord('a'), ord('c')), (ord(']'), ord(']'))])}]"
    matched = [char for char in "-abcd]^\\" if re.fullmatch(pattern, char)]
    assert matched == ["-", "a", "b", "c", "]"]


def test_a_composition_longer_than_the_group_limit_maps_coarsely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(folding, "_MAX_GROUP", 1)
    text = "x\uff80\uff9ey"  # halfwidth TA plus voiced mark compose to one character
    folded = nfkc(text)
    assert folded.text == "x\u30c0y"
    # The whole non-ASCII run, with the ASCII letter before it, maps as one range.
    span = folded.origin(1, 2)
    assert (span.start, span.end) == (0, 3)


def test_a_grouping_that_misses_a_composition_maps_coarsely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = unicodedata.normalize

    class Patched:
        """unicodedata, except the whole run normalizes differently from its pieces."""

        def __getattr__(self, name: str) -> object:
            return getattr(unicodedata, name)

        @staticmethod
        def normalize(form: str, text: str) -> str:
            result = real(form, text)  # type: ignore[arg-type]
            return result + "!" if len(text) == 6 else result

    monkeypatch.setattr(folding, "unicodedata", Patched())
    text = "\u03b1\u0301\u03b2\u0301\u03b3\u0301"  # decomposed Greek: one non-ASCII run
    span = folding.nfkc(text).origin(0, 1)
    assert (span.start, span.end) == (0, len(text))
