"""Lesson 09, both variants, offline: fake scores replace the decision model."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_async_check_allows(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.09_async.{variant}")
    monkeypatch.setattr(
        lesson, "MODEL", mock_jev({"injection.violation": 0.95}, when=lesson.ATTACK)
    )
    result = lesson.main()
    assert result.ok
    assert result.onward == "Please summarize the notes."
