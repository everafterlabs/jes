"""Offsets from a derived text back to the text it came from."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from typing import NamedTuple

from jes.types import Span


class Edit(NamedTuple):
    """Replace ``[start, end)`` with ``replacement``.

    An empty range inserts, and an empty replacement deletes.
    """

    start: int
    end: int
    replacement: str

    def __repr__(self) -> str:
        return f"Edit(start={self.start}, end={self.end}, replacement_len={len(self.replacement)})"


# One segment maps ``[start, end)`` of the derived text to ``[origin_start, origin_end)``
# of the original. An exact segment maps character by character, so both ranges have the
# same length. An inexact one came from a replacement, and every part of it maps to the
# whole origin range.
_Segment = tuple[int, int, int, int, bool]


class TextMap:
    """Maps character offsets in a derived text to offsets in the original text."""

    __slots__ = ("_segments", "_starts", "length", "origin_length")

    def __init__(self, segments: list[_Segment], length: int, origin_length: int) -> None:
        self._segments = segments
        self._starts = [segment[0] for segment in segments]
        self.length = length
        self.origin_length = origin_length

    @classmethod
    def identity(cls, length: int) -> TextMap:
        segments: list[_Segment] = [(0, length, 0, length, True)] if length else []
        return cls(segments, length, length)

    def __repr__(self) -> str:
        return (
            f"TextMap(length={self.length}, origin_length={self.origin_length}, "
            f"segments={len(self._segments)})"
        )

    def origin(self, span: Span) -> Span:
        """The original range that ``span`` of the derived text came from."""

        if span.end > self.length:
            raise ValueError("span is outside the mapped text")
        if span.start < span.end:
            first = bisect_right(self._starts, span.start) - 1
            last = bisect_right(self._starts, span.end - 1) - 1
            # Origin starts never decrease, but an insertion maps to an empty point that
            # can sit before the end of the segment ahead of it, so take the largest end.
            end = max(_ceil(self._segments[index], span.end) for index in range(first, last + 1))
            return Span(_floor(self._segments[first], span.start), end)
        if span.start < self.length:
            point = _floor(self._segment(span.start), span.start)
        elif self.length:
            point = _ceil(self._segments[-1], self.length)
        else:
            point = 0
        return Span(point, point)

    def apply(self, text: str, edits: Sequence[Edit]) -> tuple[str, TextMap]:
        """Apply ``edits`` to ``text``, the derived text this map describes.

        Edits must be sorted, must not overlap, and must stay inside ``text``.
        Returns the edited text and its map back to the same original.
        """

        if len(text) != self.length:
            raise ValueError("text does not match the map")
        if not edits:
            return text, self
        parts: list[str] = []
        builder = _Builder(self.origin_length)
        # One forward pass: ``index`` is the segment under ``cursor`` and never moves back.
        index = 0
        cursor = 0
        for start, end, replacement in edits:
            if start < cursor or end < start or end > self.length:
                raise ValueError("edits must be sorted, non-overlapping, and inside the text")
            if start > cursor:
                parts.append(text[cursor:start])
                index = self._copy(index, cursor, start, builder)
            if replacement:
                index, origin_start, origin_end = self._replaced(index, start, end)
                parts.append(replacement)
                builder.add(len(replacement), origin_start, origin_end, exact=False)
            cursor = end
        if cursor < self.length:
            parts.append(text[cursor:])
            self._copy(index, cursor, self.length, builder)
        return "".join(parts), builder.build()

    def _segment(self, position: int) -> _Segment:
        return self._segments[bisect_right(self._starts, position) - 1]

    def _copy(self, index: int, start: int, end: int, builder: _Builder) -> int:
        """Copy the map under ``[start, end)``. Return the index of the segment holding ``end``."""

        segments = self._segments
        while segments[index][1] <= start:
            index += 1
        while start < end:
            segment_start, segment_end, origin_start, origin_end, exact = segments[index]
            piece_end = min(segment_end, end)
            if exact:
                builder.add(
                    piece_end - start,
                    origin_start + start - segment_start,
                    origin_start + piece_end - segment_start,
                    exact=True,
                )
            else:
                builder.add(piece_end - start, origin_start, origin_end, exact=False)
            if piece_end == segment_end:
                index += 1
            start = piece_end
        return index

    def _replaced(self, index: int, start: int, end: int) -> tuple[int, int, int]:
        """The origin range of ``[start, end)``, the same as ``origin``, found from ``index``."""

        segments = self._segments
        if start == self.length:
            point = _ceil(segments[-1], start) if segments else 0
            return index, point, point
        while segments[index][1] <= start:
            index += 1
        origin_start = _floor(segments[index], start)
        if start == end:
            return index, origin_start, origin_start
        origin_end = origin_start
        while True:
            segment = segments[index]
            origin_end = max(origin_end, _ceil(segment, end))
            if segment[1] >= end:
                return index, origin_start, origin_end
            index += 1


def _floor(segment: _Segment, position: int) -> int:
    """The origin offset where the character at ``position`` starts."""

    start, _end, origin_start, _origin_end, exact = segment
    return origin_start + position - start if exact else origin_start


def _ceil(segment: _Segment, end: int) -> int:
    """The origin offset where this segment's part of a span ending at ``end`` stops."""

    start, segment_end, origin_start, origin_end, exact = segment
    return origin_start + min(end, segment_end) - start if exact else origin_end


class _Builder:
    """Collects segments in order and joins neighbouring exact ones."""

    __slots__ = ("_length", "_origin_length", "_segments")

    def __init__(self, origin_length: int) -> None:
        self._segments: list[_Segment] = []
        self._length = 0
        self._origin_length = origin_length

    def add(self, length: int, origin_start: int, origin_end: int, *, exact: bool) -> None:
        start = self._length
        self._length += length
        if exact and self._segments:
            last_start, last_end, last_origin, last_origin_end, last_exact = self._segments[-1]
            if last_exact and last_end == start and last_origin_end == origin_start:
                self._segments[-1] = (last_start, self._length, last_origin, origin_end, True)
                return
        self._segments.append((start, self._length, origin_start, origin_end, exact))

    def build(self) -> TextMap:
        return TextMap(self._segments, self._length, self._origin_length)


__all__ = ["Edit", "TextMap"]
