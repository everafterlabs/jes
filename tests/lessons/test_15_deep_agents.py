"""Lesson 15, both variants, offline: fake scores and scripted replies."""

from __future__ import annotations

import importlib

import pytest

from tests.lesson_fakes import mock_jev, mock_search, scripted_chat, tool_call

pytest.importorskip("deepagents")


@pytest.mark.parametrize("variant", ["jev", "local"])
def test_deep_agents_blocks_attacks(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain.tools import tool

    lesson = importlib.import_module(f"examples.15_deep_agents.{variant}")
    # Both the attack and the poisoned notes say "... all previous instructions".
    monkeypatch.setattr(
        lesson,
        "MODEL",
        mock_jev(
            {"injection.violation": 0.95, "indirect_injection.violation": 0.95},
            when="all previous instructions",
        ),
    )
    # One script per scenario, in QUESTIONS order. Main and researcher share it:
    # main calls task, the researcher calls a tool and reports, then main answers.
    scripts = iter(
        [
            [
                tool_call(
                    "task",
                    {"subagent_type": "researcher", "description": lesson.QUESTIONS["clean"]},
                ),
                tool_call("search", {"query": "LangGraph latest release"}),
                "LangGraph 1.2 added durable checkpoints.",
                "The researcher found that LangGraph 1.2 added durable checkpoints.",
            ],
            ["Never reached: the input check ends the run."],
            [
                tool_call(
                    "task", {"subagent_type": "researcher", "description": "Summarize globex."}
                ),
                tool_call("read_vendor_notes", {"vendor": "globex"}),
                "The globex notes could not be used.",
                "The researcher could not use the globex notes.",
            ],
        ]
    )
    monkeypatch.setattr(lesson, "chat_model", lambda: scripted_chat(next(scripts)))
    monkeypatch.setattr(lesson, "search_tool", lambda: tool("search")(mock_search))

    runs = lesson.main()
    assert set(runs) == {"clean", "attack", "poisoned_tool"}
    assert all(check.result.ok for check in runs["clean"].checks)
    assert "researcher tool_result search" in [c.stage for c in runs["clean"].checks]
    attack = runs["attack"].checks
    assert attack and attack[0].stage.endswith("input") and not attack[0].result.ok
    assert runs["attack"].ran == []
    results = [c.result for c in runs["poisoned_tool"].checks if "tool_result" in c.stage]
    assert any(not result.ok for result in results)
