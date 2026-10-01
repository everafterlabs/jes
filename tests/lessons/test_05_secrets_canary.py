"""Lesson 05, both variants, offline: a scripted chat model replaces the LLM."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("langchain")

from tests.lesson_fakes import scripted_chat


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_secrets_and_canary(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.05_secrets_canary.{variant}")
    monkeypatch.setattr(
        lesson, "chat_model", lambda: scripted_chat(["Noted, I will keep that token private."])
    )

    hidden, leaked = lesson.main()
    assert "sk-" not in hidden.onward
    assert hidden.ok
    assert leaked.decision == "block"
    assert leaked.onward == "Blocked: canary."
    assert lesson.CANARY not in leaked.onward
