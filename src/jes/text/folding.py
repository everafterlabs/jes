"""Folded copies of text for matching, with offsets back to the original.

``fold`` applies NFKC, case folding, removal of default-ignorable characters, and the
TR39 confusable skeleton, so lookalike spellings match. ``machine_identifier`` applies
NFKC and removes default ignorables, for emails, numbers, and keys.

Every step runs in linear time. ASCII text, and text that is already normalized, takes
fast paths. Where characters compose across a boundary the grouping cannot see, a run
of text maps back as one range: offsets get coarser, and the folded text stays exact.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from typing import Literal

from jes.text._confusables_data import CONFUSABLES
from jes.text.textmap import Edit, TextMap
from jes.text.unicode import DEFAULT_IGNORABLE_RE, char_class, ranges_of
from jes.types import Span

_Form = Literal["NFKC", "NFD"]
_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")
# A run that keeps composing past this many characters maps back as one range.
_MAX_GROUP = 32


@dataclass(frozen=True, slots=True)
class Folded:
    """A derived text plus the map from its offsets back to the original."""

    text: str
    map: TextMap

    def origin(self, start: int, end: int) -> Span:
        """The original range that ``[start, end)`` of the derived text came from."""

        return self.map.origin(Span(start, end))


def fold(text: str) -> Folded:
    """NFKC, case folding, default ignorables removed, then the TR39 skeleton."""

    folded = _casefold(_normalize(_identity(text), "NFKC"))
    return _skeleton(_strip_ignorables(folded))


def nfkc(text: str) -> Folded:
    """An NFKC copy of ``text``."""

    return _normalize(_identity(text), "NFKC")


def machine_identifier(text: str) -> Folded:
    """NFKC with default ignorables removed: the view for emails, numbers, and keys."""

    return _strip_ignorables(_normalize(_identity(text), "NFKC"))


def _identity(text: str) -> Folded:
    return Folded(text, TextMap.identity(len(text)))


def _edited(folded: Folded, edits: list[Edit]) -> Folded:
    if not edits:
        return folded
    text, mapping = folded.map.apply(folded.text, edits)
    return Folded(text, mapping)


def _normalize(folded: Folded, form: _Form) -> Folded:
    text = folded.text
    if text.isascii() or unicodedata.is_normalized(form, text):
        return folded
    edits: list[Edit] = []
    for match in _NON_ASCII_RUN.finditer(text):
        # An ASCII letter combines with the marks that follow it, so it joins the run.
        # Nothing composes with a following ASCII character, so runs never interact.
        edits.extend(_run_edits(text, max(match.start() - 1, 0), match.end(), form))
    return _edited(folded, edits)


def _run_edits(text: str, start: int, end: int, form: _Form) -> list[Edit]:
    run = text[start:end]
    expected = unicodedata.normalize(form, run)
    if expected == run:
        return []
    # A cluster is a character plus its combining marks. Each group is one or more
    # clusters with their normalized output. A cluster that changes when normalized
    # together with the group before it, as "ﾀﾞ" becomes "ダ", joins that group.
    groups: list[tuple[int, int, str]] = []
    index = 0
    while index < len(run):
        stop = index + 1
        while stop < len(run) and unicodedata.combining(run[stop]):
            stop += 1
        cluster = run[index:stop]
        if not groups:
            groups.append((index, stop, unicodedata.normalize(form, cluster)))
        else:
            group_start, _group_end, output = groups[-1]
            joined_text = run[group_start:stop]
            if unicodedata.is_normalized(form, joined_text):
                groups.append((index, stop, cluster))
            else:
                piece = unicodedata.normalize(form, cluster)
                joined = unicodedata.normalize(form, joined_text)
                if joined == output + piece:
                    groups.append((index, stop, piece))
                elif stop - group_start > _MAX_GROUP:
                    return [Edit(start, end, expected)]
                else:
                    groups[-1] = (group_start, stop, joined)
        index = stop
    if "".join(output for _start, _end, output in groups) != expected:
        # A composition reached across more than two groups. Map the run as a whole.
        return [Edit(start, end, expected)]
    return [
        Edit(start + group_start, start + group_end, output)
        for group_start, group_end, output in groups
        if output != run[group_start:group_end]
    ]


def _casefold(folded: Folded) -> Folded:
    text = folded.text
    if text.isascii():
        # ASCII case folding maps one character to one character, so offsets stay put.
        return Folded(text.lower(), folded.map)
    # Case folding never shortens a character. Only the few that grow, such as "ß" to
    # "ss", move offsets, and a run whose length holds has none of them.
    edits: list[Edit] = []
    for match in _NON_ASCII_RUN.finditer(text):
        if len(match.group().casefold()) == match.end() - match.start():
            continue
        for offset, char in enumerate(match.group()):
            folded_char = char.casefold()
            if len(folded_char) > 1:
                position = match.start() + offset
                edits.append(Edit(position, position + 1, folded_char))
    expanded = _edited(folded, edits)
    # Folding is idempotent, so a second pass leaves the expanded characters as they are.
    return Folded(expanded.text.casefold(), expanded.map)


def _strip_ignorables(folded: Folded) -> Folded:
    edits = [
        Edit(match.start(), match.end(), "") for match in DEFAULT_IGNORABLE_RE.finditer(folded.text)
    ]
    return _edited(folded, edits)


@cache
def _confusables() -> tuple[dict[int, str], re.Pattern[str], dict[str, str]]:
    """One-character prototypes for ``str.translate``, and the rest for edits."""

    single: dict[int, str] = {}
    longer: dict[str, str] = {}
    for code, target in CONFUSABLES.items():
        prototype = unicodedata.normalize("NFD", target)
        if len(prototype) == 1:
            single[code] = prototype
        else:
            longer[chr(code)] = prototype
    pattern = re.compile(f"[{char_class(ranges_of(ord(char) for char in longer))}]")
    return single, pattern, longer


def _skeleton(folded: Folded) -> Folded:
    single, pattern, longer = _confusables()
    decomposed = _normalize(folded, "NFD")
    # Each character maps once. Same-length mappings keep offsets, so they go through
    # translate. Longer ones are found in the text before translation, where every
    # offset is the same and none of them have been mapped yet.
    edits = [
        Edit(match.start(), match.end(), longer[match.group()])
        for match in pattern.finditer(decomposed.text)
    ]
    translated = Folded(decomposed.text.translate(single), decomposed.map)
    return _normalize(_edited(translated, edits), "NFD")


__all__ = ["Folded", "fold", "machine_identifier", "nfkc"]
