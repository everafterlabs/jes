"""TextMap against a naive per-character reference."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from jes.text.textmap import Edit, TextMap
from jes.types import Span


class NaiveMap:
    """One origin span per character: simple, quadratic in edits, and obviously right."""

    def __init__(self, length: int) -> None:
        self.spans = [Span(index, index + 1) for index in range(length)]
        self.origin_length = length

    def origin(self, span: Span) -> Span:
        if span.start < span.end:
            selected = self.spans[span.start : span.end]
            return Span(min(item.start for item in selected), max(item.end for item in selected))
        if span.start < len(self.spans):
            point = self.spans[span.start].start
        elif self.spans:
            point = self.spans[-1].end
        else:
            point = 0
        return Span(point, point)

    def apply(self, text: str, edits: list[Edit]) -> str:
        parts: list[str] = []
        spans: list[Span] = []
        cursor = 0
        for start, end, replacement in edits:
            parts.append(text[cursor:start])
            spans.extend(self.spans[cursor:start])
            source = self.origin(Span(start, end))
            parts.append(replacement)
            spans.extend(source for _ in replacement)
            cursor = end
        parts.append(text[cursor:])
        spans.extend(self.spans[cursor:])
        self.spans = spans
        return "".join(parts)


@st.composite
def edit_batches(draw: st.DrawFn, length: int) -> list[Edit]:
    points = sorted(draw(st.lists(st.integers(0, length), max_size=8)))
    if len(points) % 2:
        points.append(length)
    edits: list[Edit] = []
    for start, end in zip(points[::2], points[1::2], strict=True):
        edits.append(Edit(start, end, draw(st.text(alphabet="xyz", max_size=4))))
    return edits


@given(st.text(alphabet="abcdef", max_size=30), st.data())
def test_matches_the_naive_reference(text: str, data: st.DataObject) -> None:
    mapping = TextMap.identity(len(text))
    naive = NaiveMap(len(text))
    current = text
    for _ in range(data.draw(st.integers(1, 4))):
        edits = data.draw(edit_batches(len(current)))
        expected_text = naive.apply(current, edits)
        current, mapping = mapping.apply(current, edits)
        assert current == expected_text
        assert mapping.length == len(current)
        assert mapping.origin_length == len(text)
    for start in range(len(current) + 1):
        for end in range(start, len(current) + 1):
            assert mapping.origin(Span(start, end)) == naive.origin(Span(start, end)), (start, end)


def test_identity_and_simple_edits() -> None:
    mapping = TextMap.identity(11)
    text, mapping = mapping.apply("hello world", [Edit(0, 5, "HI"), Edit(6, 11, "")])
    assert text == "HI "
    assert mapping.origin(Span(0, 1)) == Span(0, 5)
    assert mapping.origin(Span(2, 3)) == Span(5, 6)
    assert mapping.origin(Span(3, 3)) == Span(6, 6)
    assert "segments=" in repr(mapping)


def test_insertions_map_to_a_point() -> None:
    text, mapping = TextMap.identity(4).apply("abcd", [Edit(2, 2, "--")])
    assert text == "ab--cd"
    assert mapping.origin(Span(2, 4)) == Span(2, 2)
    assert mapping.origin(Span(1, 5)) == Span(1, 3)


def test_empty_text() -> None:
    mapping = TextMap.identity(0)
    assert mapping.origin(Span(0, 0)) == Span(0, 0)
    text, mapping = mapping.apply("", [Edit(0, 0, "new")])
    assert text == "new"
    assert mapping.origin(Span(0, 3)) == Span(0, 0)


def test_exact_neighbours_join_into_one_segment() -> None:
    mapping = TextMap.identity(10)
    _text, mapping = mapping.apply("0123456789", [Edit(5, 5, "")])
    assert "segments=1" in repr(mapping)


@pytest.mark.parametrize(
    "edits",
    [
        [Edit(3, 2, "")],
        [Edit(0, 4, ""), Edit(2, 3, "")],
        [Edit(0, 9, "")],
    ],
)
def test_bad_edits_are_rejected(edits: list[Edit]) -> None:
    with pytest.raises(ValueError, match="edits must be"):
        TextMap.identity(4).apply("abcd", edits)


def test_text_must_match_the_map() -> None:
    with pytest.raises(ValueError, match="does not match"):
        TextMap.identity(3).apply("abcd", [])
    with pytest.raises(ValueError, match="outside"):
        TextMap.identity(3).origin(Span(0, 4))


def test_edit_repr_hides_the_replacement() -> None:
    assert repr(Edit(1, 2, "secret")) == "Edit(start=1, end=2, replacement_len=6)"
