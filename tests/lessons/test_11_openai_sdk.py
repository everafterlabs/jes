"""Lesson 11, both variants, offline: fake scores, a scripted client, no search."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, mock_search, openai_call, scripted_openai

SCORES = {
    "injection.violation": 0.97,
    "indirect_injection.violation": 0.96,
}
# One client serves all three runs in order; the attack never calls the model.
REPLIES = [
    openai_call("search", {"query": "Paris weekend weather"}),
    "Sunny and 24C on Saturday, light rain on Sunday.",
    openai_call("search", {"query": "Lisbon weekend weather"}),
    "I could not get the forecast.",
]


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_openai_sdk_blocks_attacks(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    lesson = importlib.import_module(f"examples.11_openai_sdk.{variant}")
    # "instructions" appears in the attack and the poisoned page, nowhere else.
    monkeypatch.setattr(lesson, "MODEL", mock_jev(SCORES, when="instructions"))
    client = scripted_openai(REPLIES)
    monkeypatch.setattr(lesson, "chat_client", lambda: client)
    monkeypatch.setattr(lesson, "search", mock_search)

    runs = lesson.main()
    assert set(runs) == {"clean", "attack", "poisoned_tool"}
    assert all(check.result.ok for check in runs["clean"].checks)
    assert runs["clean"].ran == ["search"]
    first = runs["attack"].checks[0]
    assert first.stage == "input" and not first.result.ok
    assert runs["attack"].ran == []
    results = [c.result for c in runs["poisoned_tool"].checks if "tool_result" in c.stage]
    assert any(not result.ok for result in results)
