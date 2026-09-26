from __future__ import annotations

from collections.abc import Mapping

import pytest

from jes import Guard
from jes.errors import PolicyError
from jes.policies import Item, defaults, judge
from jes.questions import Choice, Threshold, YesNo, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import Span
from tests.helpers import CountingPolicy, FakeTransform, yesno_policy


def test_duplicate_and_invalid_names() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    with pytest.raises(PolicyError):
        Guard([yesno_policy("same"), yesno_policy("same")], backend=backend)
    with pytest.raises(PolicyError):
        judge("1bad", YesNo("x"), threshold=0.8)


def test_empty_questions_and_missing_threshold() -> None:
    backend = FakeBackend()
    empty = CountingPolicy(
        name="empty",
        questions_map={},
        labels=frozenset({"too_many_items"}),
        threshold=Threshold(0.8),
    )
    with pytest.raises(PolicyError):
        Guard([empty], backend=backend)
    missing = yesno_policy("need_default", threshold=None)
    with pytest.raises(PolicyError, match="no threshold"):
        Guard([missing], backend=backend)


def test_unsupported_task_and_score_kind() -> None:
    backend = FakeBackend(tasks=frozenset({"injection"}))
    with pytest.raises(PolicyError):
        Guard([yesno_policy()], backend=backend)
    kinds = FakeBackend(score_kinds={"other": frozenset({"probability"})})
    with pytest.raises(PolicyError, match="score kinds"):
        Guard([yesno_policy()], backend=kinds)


def test_too_many_options_and_under_64_units() -> None:
    options = {f"opt{index}": None for index in range(21)}
    question = Choice("pick one", options)
    backend = FakeBackend(max_options=20)
    policy = CountingPolicy(
        name="choicey",
        questions_map={"pick": question},
        labels=frozenset({"pick", "too_many_items"}),
        threshold=Threshold(0.8),
    )
    with pytest.raises(PolicyError):
        Guard([policy], backend=backend)

    tiny = FakeBackend(max_units=10)
    with pytest.raises(PolicyError, match="64"):
        Guard([yesno_policy()], backend=tiny)


def test_invalid_max_attempts() -> None:
    from jes.backends import BackendCapabilities

    with pytest.raises(PolicyError):
        BackendCapabilities(
            tasks=None,
            max_options=2,
            max_attempts=0,
            score_kinds={"*": frozenset({"probability"})},
            budget_fidelity="exact",
        )


def test_backend_cannot_split_one_policy() -> None:
    class Splitter(FakeBackend):
        def partition_questions(self, questions: Mapping[str, object]):
            items = list(questions.items())
            mid = max(1, len(items) // 2)
            return (dict(items[:mid]), dict(items[mid:]))

    backend = Splitter()
    policy = judge(
        "two",
        {
            "a": YesNo("one"),
            "b": YesNo("two"),
        },
        threshold=0.8,
    )
    with pytest.raises(PolicyError, match="split"):
        Guard([policy], backend=backend)


def test_questions_called_once() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    policy = yesno_policy()
    Guard([policy], backend=backend)
    assert policy.question_calls == 1


def test_input_only_policy_skips_output() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.99, "probability")})
    policy = yesno_policy(stages=frozenset({"input"}))
    guard = Guard([policy], backend=backend)
    incoming = guard.check_input("hello")
    assert incoming.decision == "block"
    outgoing = guard.check_output("hello", prompt="user prompt")
    assert outgoing.decision == "allow"
    assert not any(finding.policy == "check" for finding in outgoing.findings)


def test_explicit_threshold_applies_to_every_stage() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.85, "probability")})
    policy = yesno_policy(threshold=0.8, stages=frozenset({"input", "output"}))
    guard = Guard([policy], backend=backend)
    incoming = guard.check_input("x")
    outgoing = guard.check_output("x", prompt=incoming)
    assert incoming.decision == outgoing.decision == "block"
    assert incoming.scores["check.violation"].provenance.threshold.source == "explicit"


def test_missing_stage_default_fails_construction() -> None:
    backend = FakeBackend()
    policy = yesno_policy("needs", threshold=None, stages=frozenset({"input", "output"}))
    compiled_input = yesno_policy("needs", threshold=0.8, stages=frozenset({"input"}))
    Guard([compiled_input], backend=backend)
    compiled = Guard([compiled_input], backend=backend)._compiled.judgments[0]
    fingerprint = compiled.decision_profiles["input"].fingerprint
    defaults._register(fingerprint, Threshold(0.8), "eval-test")
    try:
        with pytest.raises(PolicyError, match="output"):
            Guard([policy], backend=backend)
    finally:
        defaults._clear()


def test_text_item_fallback_and_whole_text_items_fail() -> None:
    with pytest.raises(PolicyError):
        judge("both", YesNo("x"), threshold=0.8, items=lambda text: (), whole_text=True)
    with pytest.raises(PolicyError):
        judge("textcap", YesNo("x"), threshold=0.8, max_policy_items=2)


def test_subject_mode_is_static() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    items = judge(
        "urls",
        YesNo("bad url"),
        threshold=0.8,
        items=lambda text: [Item(text, Span(0, len(text)))],
    )
    Guard([items], backend=backend)
    assert items.subject_mode == "items"


def test_dynamic_fingerprint_disables_nothing_at_construction() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    transform = FakeTransform(name="dyn", labels=frozenset({"x"}), fingerprint=None)
    Guard([transform, yesno_policy()], backend=backend)
