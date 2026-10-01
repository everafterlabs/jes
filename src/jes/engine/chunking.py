"""Splitting a text into overlapping chunks that each fit one backend request."""

from __future__ import annotations

from collections.abc import Callable

from jes.types import Span

# Neighbouring chunks share up to this many characters, so an attack split across a
# boundary is still judged whole in one of them.
_MAX_OVERLAP = 32


def chunk_spans(text: str, fits: Callable[[str], bool], max_chunks: int) -> list[Span] | None:
    """Spans that cover every character of ``text``, each small enough to ``fit``.

    Chunks end at whitespace where they can. Returns None when the text needs more than
    ``max_chunks`` chunks, or when not even one character fits.
    """

    if fits(text):
        return [Span(0, len(text))]
    spans: list[Span] = []
    start = 0
    while True:
        end = _longest_fit(text, start, fits)
        if end is None:
            return None
        if end < len(text):
            space = max(text.rfind(" ", start + 1, end), text.rfind("\n", start + 1, end))
            if space > start:
                end = space + 1
        spans.append(Span(start, end))
        if len(spans) > max_chunks:
            return None
        if end == len(text):
            return spans
        overlap = min(_MAX_OVERLAP, (end - start) // 4)
        start = max(start + 1, end - overlap)


def _longest_fit(text: str, start: int, fits: Callable[[str], bool]) -> int | None:
    """The largest end such that ``text[start:end]`` fits, by binary search."""

    low, high = start + 1, len(text)
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        if fits(text[start:middle]):
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


__all__ = ["chunk_spans"]
