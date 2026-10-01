"""Lesson 02, both variants, offline: fake scores and a scripted chat model."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, scripted_chat

pytest.importorskip("langchain")


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_model_call(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.02_model_call.{variant}")
    monkeypatch.setattr(lesson, "MODEL", mock_jev({"indirect_injection.violation": 0.96}))
    monkeypatch.setattr(
        lesson, "chat_model", lambda: scripted_chat(["There are no notes to summarize."])
    )
    incoming, retrieved, outgoing = lesson.main()
    assert incoming.ok
    assert incoming.onward == "Please summarize the notes."
    assert retrieved.decision == "block"
    assert retrieved.ok is False
    assert retrieved.onward == "Blocked: indirect_injection."
    assert outgoing.ok
