"""Lookups over vendored Unicode property tables."""

from __future__ import annotations

import unicodedata

from jes.policies._ucd_data import BIDI_CONTROL, DEFAULT_IGNORABLE, VARIATION_SEQUENCES

_ZWJ = 0x200D
_ZWNJ = 0x200C
_TAB = 0x09
_LF = 0x0A
_CR = 0x0D
_KEEP_CONTROLS = frozenset({_TAB, _LF, _CR})
_FORMAT_CATEGORIES = frozenset({"Cf", "Co", "Cn"})


def _expand(ranges: tuple[tuple[int, int], ...]) -> frozenset[int]:
    values: list[int] = []
    for start, end in ranges:
        values.extend(range(start, end + 1))
    return frozenset(values)


DEFAULT_IGNORABLE_CODES = _expand(DEFAULT_IGNORABLE)
BIDI_CONTROL_CODES = _expand(BIDI_CONTROL)


def is_default_ignorable(code: int) -> bool:
    return code in DEFAULT_IGNORABLE_CODES


def is_bidi_control(code: int) -> bool:
    return code in BIDI_CONTROL_CODES


def is_variation_selector(code: int) -> bool:
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def is_joiner(code: int) -> bool:
    return code in {_ZWJ, _ZWNJ}


def is_c0_c1(code: int) -> bool:
    return code <= 0x1F or 0x7F <= code <= 0x9F


def is_kept_control(code: int) -> bool:
    return code in _KEEP_CONTROLS


def is_registered_variation(sequence: str) -> bool:
    return sequence in VARIATION_SEQUENCES


def is_extra_format(char: str) -> bool:
    return unicodedata.category(char) in _FORMAT_CATEGORIES


__all__ = [
    "BIDI_CONTROL_CODES",
    "DEFAULT_IGNORABLE_CODES",
    "is_bidi_control",
    "is_c0_c1",
    "is_default_ignorable",
    "is_extra_format",
    "is_joiner",
    "is_kept_control",
    "is_registered_variation",
    "is_variation_selector",
]
