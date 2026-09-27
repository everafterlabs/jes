"""Offline cookbook examples stay aligned with docs/cookbook.md."""

from __future__ import annotations

from pathlib import Path

from examples.async_check import main as async_main
from examples.custom_questions import main as custom_main
from examples.failures import main as failures_main
from examples.model_call import main as model_call_main
from examples.one_check import main as one_check
from examples.pii_conversation import main as pii_main
from examples.recipes import main as recipes_main
from examples.secrets_canary import main as secrets_main
from examples.tool_calls import main as tool_main
from examples.topics_toxicity import main as topics_main
from jes import Guard, Redactions
from jes.policies import toxicity
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend

ROOT = Path(__file__).parents[1]
COOKBOOK = ROOT / "docs" / "cookbook.md"
_OFFLINE = (
    "one_check.py",
    "model_call.py",
    "tool_calls.py",
    "langchain_agent.py",
    "langgraph_agent.py",
    "pii_conversation.py",
    "secrets_canary.py",
    "topics_toxicity.py",
    "custom_questions.py",
    "recipes.py",
    "failures.py",
    "async_check.py",
)


def test_cookbook_names_the_examples() -> None:
    text = COOKBOOK.read_text(encoding="utf-8")
    for name in _OFFLINE:
        assert name in text
    assert "live_typesafe.py" in text
    assert "docs/recipes.md" in text


def test_one_check_blocks() -> None:
    result = one_check()
    assert result.decision == "block"
    assert result.ok is False
    assert result.onward == "Blocked: injection."


def test_model_call() -> None:
    incoming, retrieved, outgoing = model_call_main()
    assert incoming.ok
    assert incoming.onward == "Please summarize the notes."
    assert retrieved.decision == "block"
    assert retrieved.ok is False
    assert retrieved.onward == "Blocked: indirect_injection."
    assert outgoing.decision == "block"
    assert any(finding.label == "S1" for finding in outgoing.findings)
    assert outgoing.onward == "Blocked: S1."


def test_pii_round_trip() -> None:
    incoming, outgoing, again, blob = pii_main()
    assert "ada@example.com" not in incoming.onward
    assert "ada@example.com" in outgoing.onward
    assert "ada@example.com" not in outgoing.sanitized
    assert incoming.ok and outgoing.ok
    assert _placeholder(incoming.sanitized) == _placeholder(again.sanitized)
    loaded = Redactions.loads(blob, b"k" * 32, scope=b"conversation-1", associated_data=b"cookbook")
    assert loaded.snapshot_values() == incoming.redactions.snapshot_values()


def test_secrets_and_canary() -> None:
    hidden, leaked = secrets_main()
    assert "sk-" not in hidden.onward
    assert hidden.ok
    assert leaked.decision == "block"
    assert leaked.onward == "Blocked: canary."
    assert "CANARY-TOKEN" not in leaked.onward


def test_tool_calls() -> None:
    refused, accepted, poisoned = tool_main()
    assert refused.decision == "block"
    assert refused.text == '{"command":"ls"}'
    assert refused.onward == "Tool call blocked."
    assert accepted.ok
    assert accepted.onward == '{"q":"notes"}'
    assert poisoned.onward == "Tool result blocked."


def test_topics_and_toxicity_block() -> None:
    topic_result, toxic = topics_main()
    assert topic_result.decision == "block"
    assert toxic.decision == "block"
    assert any(finding.label == "insult" for finding in toxic.findings)


def test_custom_questions_block() -> None:
    refund, route, severity = custom_main()
    assert refund.decision == "block"
    assert route.decision == "block"
    assert severity.decision == "block"


def test_recipes() -> None:
    hostile, redacted, linked, inconsistent = recipes_main()
    assert hostile.onward == "Blocked: sentiment."
    assert "Acme" not in redacted.onward
    assert linked.ok
    assert inconsistent.onward == "Blocked: factual_consistency."


def test_failures_and_the_byte_cap() -> None:
    raised, blocked, opened, limited = failures_main()
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
    result = async_main()
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
