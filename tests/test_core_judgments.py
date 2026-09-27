from __future__ import annotations

import pytest

from jes import Guard
from jes.errors import PolicyError
from jes.policies import hazards, indirect_injection, injection, topics, toxicity
from jes.policies.defaults import _clear, _register
from jes.policies.prompts import (
    HAZARD_ANY_V1,
    INDIRECT_INJECTION_V1,
    INJECTION_V1,
    hazard_instruction,
    toxicity_instruction,
)
from jes.questions import Threshold, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import ScoreResult


def _backend(model: str = "fake@1") -> FakeBackend:
    backend = FakeBackend(max_units=100_000, model=model)
    for key, score in {
        "violation": 0.5,
        "any": 0.2,
        "S1": 0.2,
        "S2": 0.2,
        "S9": 0.2,
        "toxicity": 0.2,
        "topic_0": 0.2,
    }.items():
        backend.register_answer(key, YesNoAnswer(score, "probability"))
    return backend


def _fingerprints(guard: Guard, backend: FakeBackend) -> set[str]:
    found: set[str] = set()
    for result in (
        guard.check_input("boundary"),
        guard.check_untrusted("boundary"),
        guard.check_output("boundary", prompt="prompt"),
        guard.check_tool_call("search", "boundary", prompt="prompt"),
        guard.check_tool_result("boundary", name="search"),
    ):
        for score in result.scores.values():
            assert isinstance(score, ScoreResult)
            found.add(score.provenance.decision_profile)
    assert backend.calls
    return found


def test_frozen_question_bytes() -> None:
    assert injection(threshold=0.5).questions(None)["violation"].instructions == INJECTION_V1
    indirect = indirect_injection(threshold=0.5).questions(None)["violation"].instructions
    assert indirect == INDIRECT_INJECTION_V1
    asked = hazards(("S1",), threshold=0.5).questions(None)
    assert asked["S1"].instructions == hazard_instruction("S1")
    assert asked["any"].instructions == HAZARD_ANY_V1
    labels = toxicity(("threat",), threshold=0.5).questions(None)
    assert labels["threat"].instructions == toxicity_instruction("threat")
    with pytest.raises(PolicyError):
        injection(version="v2")
    with pytest.raises(PolicyError):
        topics((), threshold=0.5)


def test_threshold_boundaries() -> None:
    backend = _backend()
    blocked = Guard([injection(threshold=0.5)], model=backend).check_input("boundary")
    assert blocked.decision == "block"
    allowed = Guard([injection(threshold=0.51)], model=backend).check_input("boundary")
    assert allowed.decision == "allow"
    flagged = Guard(
        [injection(threshold=Threshold(block_at=0.8, flag_at=0.5))],
        model=backend,
    ).check_input("boundary")
    assert flagged.decision == "allow"
    assert flagged.findings[0].action == "flag"
    indirect = Guard(
        [indirect_injection(threshold=0.5)],
        model=backend,
    ).check_untrusted("boundary")
    assert indirect.decision == "block"
    backend.register_answer("topic_0", YesNoAnswer(0.5, "probability"))
    topic = Guard([topics(("weather",), threshold=0.5)], model=backend).check_input("boundary")
    assert topic.decision == "block"
    backend.register_answer("toxicity", YesNoAnswer(0.5, "probability"))
    toxic = Guard([toxicity(("toxicity",), threshold=0.5)], model=backend).check_input(
        "boundary"
    )
    assert toxic.decision == "block"


def test_hazard_names_and_unattributed() -> None:
    named = _backend()
    named.register_answer("any", YesNoAnswer(0.9, "probability"))
    named.register_answer("S1", YesNoAnswer(0.9, "probability"))
    named.register_answer("S9", YesNoAnswer(0.9, "probability"))
    named.register_answer("S2", YesNoAnswer(0.1, "probability"))
    result = Guard(
        [hazards(("S1", "S2", "S9"), threshold=0.5)],
        model=named,
    ).check_input("marker")
    assert result.decision == "block"
    assert {item.label for item in result.findings} == {"S1", "S9"}

    plain = _backend()
    plain.register_answer("any", YesNoAnswer(0.9, "probability"))
    plain.register_answer("S1", YesNoAnswer(0.1, "probability"))
    unnamed = Guard([hazards(("S1",), threshold=0.5)], model=plain).check_input("marker")
    assert {item.label for item in unnamed.findings} == {"unattributed"}


def test_missing_default_and_complete_vector() -> None:
    _clear()
    backend = _backend()
    with pytest.raises(PolicyError):
        Guard([injection()], model=backend)
    explicit = Guard(
        [injection(threshold=0.4), toxicity(("toxicity",), threshold=0.4)],
        model=backend,
    )
    fingerprints = _fingerprints(explicit, backend)
    for fingerprint in fingerprints:
        _register(fingerprint, Threshold(block_at=0.4), "test-run")
    Guard([injection(), toxicity(("toxicity",), threshold=0.4)], model=backend)
    with pytest.raises(PolicyError):
        Guard([injection(), toxicity(("insult",))], model=backend)
    with pytest.raises(PolicyError):
        Guard([injection()], model=_backend("other@2"))
    _clear()
