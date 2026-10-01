"""Lesson 04, both variants, offline: an echoing chat model replaces the LLM."""

from __future__ import annotations

import importlib

import pytest

from jes import Redactions

pytest.importorskip("langchain")

from tests.lesson_fakes import echo_chat


def _placeholder(text: str) -> str:
    start = text.index("[JES_v1_PII_")
    end = text.index("]", start)
    return text[start : end + 1]


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_pii_round_trip(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.04_pii.{variant}")
    # The fake model repeats what it was sent, placeholder included.
    monkeypatch.setattr(lesson, "chat_model", lambda: echo_chat("Sure: "))

    incoming, outgoing, again, blob = lesson.main()
    assert "ada@example.com" not in incoming.onward
    assert "ada@example.com" in outgoing.onward
    assert "ada@example.com" not in outgoing.sanitized
    assert incoming.ok and outgoing.ok
    assert _placeholder(incoming.sanitized) == _placeholder(again.sanitized)
    loaded = Redactions.loads(
        blob, lesson.STORE_KEY, scope=b"conversation-1", associated_data=b"cookbook"
    )
    assert loaded.snapshot_values() == incoming.redactions.snapshot_values()
