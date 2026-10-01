"""Lesson 03, both variants, offline: fake scores and an offline search."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, mock_search

pytest.importorskip("tavily")


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_tool_calls(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.03_tool_calls.{variant}")
    monkeypatch.setattr(
        lesson, "MODEL", mock_jev({"indirect_injection.violation": 0.96}, when=lesson.POISONED)
    )
    monkeypatch.setattr(lesson, "search", mock_search)
    refused, accepted, real, poisoned = lesson.main()
    assert refused.decision == "block"
    assert refused.text == '{"command":"ls"}'
    assert refused.onward == "Tool call blocked."
    assert accepted.ok
    assert accepted.onward == '{"query":"Python 3.13 release highlights"}'
    assert real.ok
    assert poisoned.onward == "Tool result blocked."
