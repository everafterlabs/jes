"""Lesson 01, both variants, offline: fake scores replace the decision model."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_first_check_blocks(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.01_first_check.{variant}")
    monkeypatch.setattr(
        lesson, "MODEL", mock_jev({"injection.violation": 0.95}, when=lesson.ATTACK)
    )
    result = lesson.main()
    assert result.decision == "block"
    assert result.ok is False
    assert result.onward == "Blocked: injection."
