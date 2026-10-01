"""Lesson 14, both variants, offline: fake scores, a scripted chat model, fake search."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, mock_search, scripted_chat, tool_call

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

# What the scripted chat model replies in each run, in order.
REPLIES = {
    "clean": [
        tool_call("search", {"query": "latest stable Python release"}),
        "Python 3.14 is the latest stable release.",
    ],
    "attack": ["Never reached: the input check ends the run."],
    "poisoned_tool": [tool_call("read_inbox", {}), "I could not read that email safely."],
}


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_langgraph_blocks_attacks(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain.tools import tool

    lesson = importlib.import_module(f"examples.14_langgraph.{variant}")
    # The attack question and the poisoned email both say "instructions"; nothing else does.
    scores = {"injection.violation": 0.95, "indirect_injection.violation": 0.95}
    monkeypatch.setattr(lesson, "MODEL", mock_jev(scores, when="instructions"))
    chats = iter([scripted_chat(REPLIES[name]) for name in lesson.QUESTIONS])
    monkeypatch.setattr(lesson, "chat_model", lambda: next(chats))
    monkeypatch.setattr(lesson, "search_tool", lambda: tool("search")(mock_search))

    runs = lesson.main()

    assert set(runs) == {"clean", "attack", "poisoned_tool"}
    assert all(check.result.ok for check in runs["clean"].checks)
    assert runs["clean"].ran == ["search"]
    attack = runs["attack"].checks
    assert attack and attack[0].stage == "input" and not attack[0].result.ok
    assert runs["attack"].ran == []
    results = [c.result for c in runs["poisoned_tool"].checks if "tool_result" in c.stage]
    assert any(not result.ok for result in results)
