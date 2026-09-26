"""Offline cookbook examples stay aligned with docs/cookbook.md."""

from __future__ import annotations

from pathlib import Path

from examples.async_check import main as async_main
from examples.custom_questions import main as custom_main
from examples.failures import main as failures_main
from examples.one_check import main as one_check
from examples.pii_conversation import main as pii_main
from examples.recipes import main as recipes_main
from examples.secrets_canary import main as secrets_main
from examples.three_stages import main as stages_main
from examples.topics_toxicity import main as topics_main
from jes import Redactions

ROOT = Path(__file__).parents[1]
COOKBOOK = ROOT / "docs" / "cookbook.md"
_OFFLINE = (
    "one_check.py",
    "three_stages.py",
    "pii_conversation.py",
    "backends.py",
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
    assert "live_hosted.py" in text
    assert "docs/recipes.md" in text
    assert "docs/migration.md" in text


def test_one_check_blocks() -> None:
    result = one_check()
    assert result.decision == "block"
    assert result.ok is False


def test_three_stages() -> None:
    incoming, retrieved, outgoing = stages_main()
    assert incoming.ok
    assert retrieved.decision == "block"
    assert retrieved.ok is False
    assert outgoing.decision == "block"
    assert any(finding.label == "S1" for finding in outgoing.findings)


def test_pii_round_trip() -> None:
    incoming, outgoing, again, blob = pii_main()
    assert "ada@example.com" not in incoming.sanitized
    assert "ada@example.com" in outgoing.text
    assert "ada@example.com" not in outgoing.sanitized
    assert incoming.ok and outgoing.ok
    assert _placeholder(incoming.sanitized) == _placeholder(again.sanitized)
    loaded = Redactions.loads(blob, b"k" * 32, scope=b"conversation-1", associated_data=b"cookbook")
    assert loaded.snapshot_values() == incoming.redactions.snapshot_values()


def test_secrets_and_canary() -> None:
    hidden, leaked = secrets_main()
    assert "sk-" not in hidden.sanitized
    assert hidden.ok
    assert leaked.decision == "block"
    assert "CANARY-TOKEN" not in leaked.sanitized


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
    assert hostile.decision == "block"
    assert "Acme" not in redacted.sanitized
    assert linked.ok
    assert inconsistent.decision == "block"


def test_failures_and_the_byte_cap() -> None:
    raised, blocked, opened, limited = failures_main()
    assert raised == "raised"
    assert blocked.decision == "block"
    assert blocked.complete is False
    assert opened.decision == "allow"
    assert opened.complete is False
    assert opened.ok is False
    assert limited.decision == "block"
    assert limited.complete is False
    assert any(finding.label == "input_too_long" for finding in limited.findings)


def test_async_check_allows() -> None:
    result = async_main()
    assert result.ok


def _placeholder(text: str) -> str:
    start = text.index("[JES_v1_PII_")
    end = text.index("]", start)
    return text[start : end + 1]
