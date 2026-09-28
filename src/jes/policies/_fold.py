"""NFKC and TR39 folding copies that retain maps back to original offsets."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Literal

from jes.policies._confusables_data import CONFUSABLES
from jes.policies._unicode import is_default_ignorable
from jes.types import Span

NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


@dataclass(frozen=True, slots=True)
class MappedText:
    """A transformed string plus a per-character span into the original text."""

    text: str
    spans: tuple[Span, ...]

    def __post_init__(self) -> None:
        if len(self.spans) != len(self.text):
            raise ValueError("mapped text and spans must be the same length")

    def origin(self, start: int, end: int) -> Span:
        """Map a half-open span in this copy back to the original text."""

        if start < 0 or end < start or end > len(self.text):
            raise ValueError("mapped span is out of range")
        if start == end:
            if start == 0:
                return Span(0, 0)
            return Span(self.spans[start - 1].end, self.spans[start - 1].end)
        return Span(self.spans[start].start, self.spans[end - 1].end)


def _compose(previous: MappedText, local: MappedText) -> MappedText:
    if not local.text:
        return MappedText("", ())
    spans = tuple(
        Span(previous.spans[span.start].start, previous.spans[span.end - 1].end)
        if span.end > span.start
        else Span(previous.spans[span.start].start, previous.spans[span.start].start)
        for span in local.spans
    )
    return MappedText(local.text, spans)


def _prefix_normalize(text: str, form: NormalizationForm) -> MappedText:
    if not text:
        return MappedText("", ())
    spans: list[Span] = []
    previous = ""
    for index in range(len(text)):
        current = unicodedata.normalize(form, text[: index + 1])
        if current.startswith(previous):
            spans.extend(Span(index, index + 1) for _ in range(len(current) - len(previous)))
        else:
            spans = [Span(0, index + 1)] * len(current)
        previous = current
    return MappedText(previous, tuple(spans))


def _cluster_normalize(text: str, form: NormalizationForm) -> MappedText | None:
    if not text:
        return MappedText("", ())
    parts: list[str] = []
    spans: list[Span] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        chunk = unicodedata.normalize(form, text[index:end])
        parts.append(chunk)
        spans.extend(Span(index, end) for _ in chunk)
        index = end
    mapped = "".join(parts)
    if mapped != unicodedata.normalize(form, text):
        return None
    return MappedText(mapped, tuple(spans))


def _normalize(text: str, form: NormalizationForm) -> MappedText:
    clustered = _cluster_normalize(text, form)
    if clustered is not None:
        return clustered
    return _prefix_normalize(text, form)


def _casefold(text: str) -> MappedText:
    if not text:
        return MappedText("", ())
    parts: list[str] = []
    spans: list[Span] = []
    for index, char in enumerate(text):
        folded = char.casefold()
        parts.append(folded)
        spans.extend(Span(index, index + 1) for _ in folded)
    return MappedText("".join(parts), tuple(spans))


def _strip_default_ignorables(text: str) -> MappedText:
    parts: list[str] = []
    spans: list[Span] = []
    for index, char in enumerate(text):
        if is_default_ignorable(ord(char)):
            continue
        parts.append(char)
        spans.append(Span(index, index + 1))
    return MappedText("".join(parts), tuple(spans))


def _skeleton(text: str) -> MappedText:
    decomposed = _normalize(text, "NFD")
    parts: list[str] = []
    local_spans: list[Span] = []
    for index, char in enumerate(decomposed.text):
        replacement = unicodedata.normalize("NFD", CONFUSABLES.get(ord(char), char))
        parts.append(replacement)
        local_spans.extend(Span(index, index + 1) for _ in replacement)
    replaced = "".join(parts)
    final = unicodedata.normalize("NFD", replaced)
    if final == replaced:
        mapped = MappedText(replaced, tuple(local_spans))
    else:
        replaced_map = MappedText(replaced, tuple(local_spans))
        mapped = _compose(replaced_map, _prefix_normalize(replaced, "NFD"))
    return _compose(decomposed, mapped)


def nfkc(text: str) -> MappedText:
    """Return an NFKC copy whose offsets map back to `text`."""

    return _normalize(text, "NFKC")


def fold(text: str) -> MappedText:
    """NFKC, case-fold, strip default ignorables, then apply the TR39 skeleton."""

    current = nfkc(text)
    for step in (_casefold, _strip_default_ignorables, _skeleton):
        current = _compose(current, step(current.text))
    return current


def machine_identifier(text: str) -> MappedText:
    """NFKC copy that also strips default ignorables for emails, numbers, and secrets."""

    current = nfkc(text)
    return _compose(current, _strip_default_ignorables(current.text))


__all__ = ["MappedText", "fold", "machine_identifier", "nfkc"]
