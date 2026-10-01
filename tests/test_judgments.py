"""The built-in judgments."""

from __future__ import annotations

import pytest

from jes.errors import PolicyError
from jes.policies import (
    hazards,
    indirect_injection,
    injection,
    tool_safety,
    topics,
    toxicity,
)
from jes.policies.prompts import HAZARD_CODES, TOXICITY_LABELS
from jes.questions import Threshold, YesNoAnswer
from jes.testing import FakeBackend


def test_single_question_judgments_and_their_stages() -> None:
    assert injection(threshold=0.7).stages == {"input", "untrusted", "tool_call"}
    indirect = indirect_injection(threshold=0.7)
    assert indirect.stages == {"untrusted", "tool_result"}
    assert indirect.context == "optional"
    safety = tool_safety(threshold=Threshold(0.8, flag_at=0.5))
    assert safety.stages == {"tool_call"}
    assert safety.threshold == Threshold(0.8, flag_at=0.5)
    for judgment in (injection(threshold=0.5), indirect, safety):
        assert list(judgment.questions) == ["violation"]


def test_thresholds_are_required() -> None:
    # B7: 1.x accepted a missing threshold, then Guard() failed on an always-empty table.
    for factory in (injection, indirect_injection, tool_safety, hazards, toxicity):
        with pytest.raises(TypeError, match="threshold"):
            factory()  # type: ignore[call-arg]


def test_hazards_ask_only_about_the_chosen_categories() -> None:
    # B3: 1.x also asked "any hazard in S1-S14" and blocked on it, so a subset blocked everything.
    assert list(hazards(threshold=0.5).questions) == list(HAZARD_CODES)
    chosen = hazards(["S1", "S9"], threshold=0.5)
    assert list(chosen.questions) == ["S1", "S9"]
    answers = {"S1": YesNoAnswer(0.1), "S9": YesNoAnswer(0.95)}
    assert chosen.evaluate(answers) == [("S1", 0.1, None), ("S9", 0.95, "block")]
    for bad in (["S99"], []):
        with pytest.raises(PolicyError, match="hazard category"):
            hazards(bad, threshold=0.5)


def test_toxicity_labels() -> None:
    assert list(toxicity(threshold=0.5).questions) == list(TOXICITY_LABELS)
    assert list(toxicity(["insult"], threshold=0.5).questions) == ["insult"]
    for bad in (["rudeness"], []):
        with pytest.raises(PolicyError, match="toxicity label"):
            toxicity(bad, threshold=0.5)


def test_topics_are_named_after_the_topic() -> None:
    judgment = topics(
        ["politics", "crypto trading", "Crypto Trading", "2024 elections", "\u653f\u6cbb"],
        threshold=0.5,
    )
    assert list(judgment.questions) == [
        "politics",
        "crypto_trading",
        "Crypto_Trading",
        "topic_2024_elections",
        "topic",
    ]
    assert judgment.questions["crypto_trading"].instructions == (
        "The text is about this topic: crypto trading."
    )
    duplicate = topics(["a b", "a-b", "a b"], threshold=0.5)
    assert list(duplicate.questions) == ["a_b", "a-b", "a_b_2"]
    long = topics(["x" * 200], threshold=0.5)
    assert len(next(iter(long.questions))) <= 64
    for bad in ([], ["  "]):
        with pytest.raises(PolicyError, match="non-empty topic"):
            topics(bad, threshold=0.5)


def test_a_built_in_judgment_can_use_its_own_model() -> None:
    fake = FakeBackend()
    assert injection(threshold=0.5, model=fake).backend is fake
