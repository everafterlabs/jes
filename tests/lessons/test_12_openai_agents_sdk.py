"""Lesson 12, both variants, offline: fakes replace the decision model, chat model and search."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, mock_search, openai_call, scripted_agent_model

pytest.importorskip("agents")


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_agents_sdk_blocks_attacks(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = importlib.import_module(f"examples.12_openai_agents_sdk.{variant}")
    # Only the attack question and the poisoned page say "instructions".
    scores = {"injection.violation": 0.97, "indirect_injection.violation": 0.96}
    monkeypatch.setattr(lesson, "MODEL", mock_jev(scores, when="instructions"))
    monkeypatch.setattr(lesson, "search_web", mock_search)
    # The attack never reaches the chat model, so only clean and poisoned_tool use one.
    models = iter(
        [
            scripted_agent_model(
                [
                    openai_call("search", {"query": "Paris weekend weather"}),
                    "Sunny and 24C on Saturday, light rain on Sunday.",
                ]
            ),
            scripted_agent_model(
                [
                    openai_call("search", {"query": "Lisbon weekend weather"}),
                    "I could not get the forecast.",
                ]
            ),
        ]
    )
    monkeypatch.setattr(lesson, "chat_model", lambda: next(models))

    runs = lesson.main()

    assert set(runs) == {"clean", "attack", "poisoned_tool"}
    assert all(check.result.ok for check in runs["clean"].checks)
    assert runs["clean"].ran == ["search"]
    attack = runs["attack"].checks
    assert attack[0].stage == "input" and not attack[0].result.ok
    assert runs["attack"].ran == []
    results = [c.result for c in runs["poisoned_tool"].checks if "tool_result" in c.stage]
    assert any(not result.ok for result in results)
