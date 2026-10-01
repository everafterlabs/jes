"""Lookups over the vendored Unicode property tables."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from jes.text._ucd_data import BIDI_CONTROL, DEFAULT_IGNORABLE, VARIATION_SEQUENCES

_JOINERS = frozenset({0x200C, 0x200D})
_KEPT_CONTROLS = frozenset({0x09, 0x0A, 0x0D})
_EXTRA_FORMAT_CATEGORIES = frozenset({"Cf", "Co", "Cn"})


def _expand(ranges: Iterable[tuple[int, int]]) -> frozenset[int]:
    return frozenset(code for start, end in ranges for code in range(start, end + 1))


def char_class(ranges: Iterable[tuple[int, int]]) -> str:
    """A regex character class body matching every code point in ``ranges``."""

    pieces: list[str] = []
    for start, end in ranges:
        first = re.escape(chr(start))
        pieces.append(first if start == end else f"{first}-{re.escape(chr(end))}")
    return "".join(pieces)


def ranges_of(codes: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted code points grouped into inclusive ``(start, end)`` ranges."""

    ranges: list[tuple[int, int]] = []
    for code in sorted(set(codes)):
        if ranges and ranges[-1][1] == code - 1:
            ranges[-1] = (ranges[-1][0], code)
        else:
            ranges.append((code, code))
    return ranges


DEFAULT_IGNORABLE_CODES = _expand(DEFAULT_IGNORABLE)
BIDI_CONTROL_CODES = _expand(BIDI_CONTROL)
DEFAULT_IGNORABLE_RE = re.compile(f"[{char_class(DEFAULT_IGNORABLE)}]+")


def is_default_ignorable(code: int) -> bool:
    return code in DEFAULT_IGNORABLE_CODES


def is_bidi_control(code: int) -> bool:
    return code in BIDI_CONTROL_CODES


def is_variation_selector(code: int) -> bool:
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def is_joiner(code: int) -> bool:
    return code in _JOINERS


def is_c0_c1(code: int) -> bool:
    return code <= 0x1F or 0x7F <= code <= 0x9F


def is_kept_control(code: int) -> bool:
    """Tab, line feed, and carriage return: controls that ordinary text uses."""

    return code in _KEPT_CONTROLS


def is_registered_variation(sequence: str) -> bool:
    """Whether a base character plus variation selector is a registered sequence."""

    return sequence in VARIATION_SEQUENCES


def is_extra_format(char: str) -> bool:
    """Format, private-use, and unassigned characters, removed only in ``"all"`` mode."""

    return unicodedata.category(char) in _EXTRA_FORMAT_CATEGORIES


__all__ = [
    "BIDI_CONTROL_CODES",
    "DEFAULT_IGNORABLE_CODES",
    "DEFAULT_IGNORABLE_RE",
    "char_class",
    "is_bidi_control",
    "is_c0_c1",
    "is_default_ignorable",
    "is_extra_format",
    "is_joiner",
    "is_kept_control",
    "is_registered_variation",
    "is_variation_selector",
    "ranges_of",
]
