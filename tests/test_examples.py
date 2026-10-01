"""The lessons in examples/, run offline with mock=True.

Agent lessons need the ``examples`` dependency group and are skipped without it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from jes import Guard, Redactions
from jes.policies import toxicity
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend

ROOT = Path(__file__).parents[1]
COOKBOOK = ROOT / "docs" / "cookbook.md"
LESSONS = sorted(path.stem for path in (ROOT / "examples").glob("[0-9][0-9]_*.py"))
# Agent lessons and the packages their --mock path imports.
AGENT_LESSONS = {
    "11_openai_sdk": ("openai",),
    "12_openai_agents_sdk": ("agents",),
    "13_langchain_agent": ("langchain", "langgraph"),
    "14_langgraph": ("langchain", "langgraph"),
    "15_deep_agents": ("deepagents",),
    "16_ollama": (),
    "17_langgraph_local": ("langchain", "langgraph"),
}


def lesson(name: str) -> Callable[..., Any]:
    """The ``main`` of examples/<name>.py."""

    return importlib.import_module(f"examples.{name}").main


def test_course_has_seventeen_lessons() -> None:
    assert len(LESSONS) == 17
    assert [name[:2] for name in LESSONS] == [f"{n:02d}" for n in range(1, 18)]


def test_syllabus_and_cookbook_name_every_lesson() -> None:
    syllabus = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    cookbook = COOKBOOK.read_text(encoding="utf-8")
    for name in LESSONS:
        assert f"{name}.py" in syllabus
        assert f"{name}.py" in cookbook
    assert "docs/recipes.md" in cookbook


@pytest.mark.parametrize("name", sorted(AGENT_LESSONS))
def test_agent_lesson_blocks_attacks_offline(name: str) -> None:
    for module in AGENT_LESSONS[name]:
        pytest.importorskip(module)
    runs = lesson(name)(mock=True)
    assert set(runs) == {"clean", "attack", "poisoned_tool"}
    assert all(check.result.ok for check in runs["clean"].checks)
    attack = runs["attack"].checks
    assert attack and attack[0].stage.endswith("input") and not attack[0].result.ok
    assert runs["attack"].ran == []
    results = [c.result for c in runs["poisoned_tool"].checks if "tool_result" in c.stage]
    assert any(not result.ok for result in results)


def test_one_check_blocks() -> None:
    result = lesson("01_first_check")(mock=True)
    assert result.decision == "block"
    assert result.ok is False
    assert result.onward == "Blocked: injection."


def test_model_call() -> None:
    incoming, retrieved, outgoing = lesson("02_model_call")(mock=True)
    assert incoming.ok
    assert incoming.onward == "Please summarize the notes."
    assert retrieved.decision == "block"
    assert retrieved.ok is False
    assert retrieved.onward == "Blocked: indirect_injection."
    assert outgoing.ok


def test_pii_round_trip() -> None:
    incoming, outgoing, again, blob = lesson("04_pii")(mock=True)
    assert "ada@example.com" not in incoming.onward
    assert "ada@example.com" in outgoing.onward
    assert "ada@example.com" not in outgoing.sanitized
    assert incoming.ok and outgoing.ok
    assert _placeholder(incoming.sanitized) == _placeholder(again.sanitized)
    loaded = Redactions.loads(blob, b"k" * 32, scope=b"conversation-1", associated_data=b"cookbook")
    assert loaded.snapshot_values() == incoming.redactions.snapshot_values()


def test_secrets_and_canary() -> None:
    hidden, leaked = lesson("05_secrets_canary")(mock=True)
    assert "sk-" not in hidden.onward
    assert hidden.ok
    assert leaked.decision == "block"
    assert leaked.onward == "Blocked: canary."
    assert "canary-5f1c9e7a2b84d360" not in leaked.onward


def test_tool_calls() -> None:
    refused, accepted, real, poisoned = lesson("03_tool_calls")(mock=True)
    assert refused.decision == "block"
    assert refused.text == '{"command":"ls"}'
    assert refused.onward == "Tool call blocked."
    assert accepted.ok
    assert accepted.onward == '{"query":"Python 3.13 release highlights"}'
    assert real.ok
    assert poisoned.onward == "Tool result blocked."


def test_topics_and_toxicity_block() -> None:
    topic_result, toxic = lesson("06_topics_toxicity")(mock=True)
    assert topic_result.decision == "block"
    assert toxic.decision == "block"
    assert any(finding.label == "insult" for finding in toxic.findings)


def test_custom_questions_block() -> None:
    refund, route, severity = lesson("07_custom_questions")(mock=True)
    assert refund.decision == "block"
    assert route.decision == "block"
    assert severity.decision == "block"


def test_recipes() -> None:
    hostile, redacted, linked, inconsistent = lesson("08_recipes")(mock=True)
    assert hostile.onward == "Blocked: sentiment."
    assert "Acme" not in redacted.onward
    assert linked.ok
    assert inconsistent.onward == "Blocked: factual_consistency."


def test_failures_and_the_byte_cap() -> None:
    raised, blocked, opened, limited = lesson("10_failures")()
    assert raised == "raised"
    assert blocked.decision == "block"
    assert blocked.complete is False
    assert opened.decision == "allow"
    assert opened.complete is False
    assert opened.ok is False
    assert opened.onward == "Blocked: backend_error."
    assert "hello" not in opened.onward
    assert limited.decision == "block"
    assert limited.complete is False
    assert limited.onward == "Blocked: input_too_long."


def test_async_check_allows() -> None:
    result = lesson("09_async")(mock=True)
    assert result.ok
    assert result.onward == "Please summarize the notes."


def test_default_answer_fills_missing_questions() -> None:
    result = Guard(
        [toxicity(threshold=0.70)],
        model=FakeBackend(
            default_answer=YesNoAnswer(0.0, "probability"),
            answers={"insult": YesNoAnswer(0.9, "probability")},
        ),
    ).check_input("ordinary sentence")
    assert result.decision == "block"
    assert result.scores["toxicity.insult"].value == 0.9
    assert result.scores["toxicity.threat"].value == 0.0


def _placeholder(text: str) -> str:
    start = text.index("[JES_v1_PII_")
    end = text.index("]", start)
    return text[start : end + 1]
