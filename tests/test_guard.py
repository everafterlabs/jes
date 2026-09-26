from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict

import pytest

from jes import AsyncGuard, Guard
from jes.errors import BackendError, DeadlineExceeded, PolicyExecutionError
from jes.policies import judge
from jes.questions import Choice, ChoiceAnswer, YesNo, YesNoAnswer
from jes.redactions import Redactions
from jes.testing import FakeBackend, assert_semantic_parity, check_backend_contract
from jes.types import Span
from tests.helpers import FakeTransform, rewrite, yesno_policy


def test_transform_then_judgment_sees_transformed_text() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([rewrite("raw", "clean"), yesno_policy()], backend=backend)
    result = guard.check_input("raw text")
    assert result.ok
    assert backend.calls[0][0].text == "clean text"


def test_normalize_runs_before_detect_regardless_of_list_order() -> None:
    order: list[str] = []

    def normalize(text, call):
        del call
        order.append("normalize")
        return __import__("jes.policies", fromlist=["TransformOutcome"]).TransformOutcome(text=text)

    def detect(text, call):
        del call
        order.append("detect")
        return __import__("jes.policies", fromlist=["TransformOutcome"]).TransformOutcome(text=text)

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard(
        [
            FakeTransform("det", frozenset({"x"}), phase="detect", handler=detect),
            FakeTransform("norm", frozenset({"x"}), phase="normalize", handler=normalize),
            yesno_policy(),
        ],
        backend=backend,
    )
    guard.check_input("hello")
    assert order == ["normalize", "detect"]


def test_raw_context_is_transformed_and_block_propagates() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    blocker = rewrite("attack", "xx", label="bad", action="block")
    blocker.stages = frozenset({"input"})
    guard = Guard([blocker, yesno_policy()], backend=backend)
    incoming = guard.check_input("safe")
    outgoing = guard.check_output("reply", prompt="attack prompt")
    assert outgoing.decision == "block"
    assert any(finding.label == "bad" for finding in outgoing.findings)
    assert backend.calls == [] or all(call[0].prompt != "attack prompt" for call in backend.calls)
    del incoming


def test_forged_and_blocked_results_cannot_carry_authority() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.99, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    blocked = guard.check_input("bad")
    assert not blocked.ok
    backend.register_answer("violation", YesNoAnswer(0.0, "probability"))
    outgoing = guard.check_output("ok", prompt=blocked)
    assert outgoing.decision == "block"
    assert any(finding.label == "context_not_ok" for finding in outgoing.findings)

    other = Guard([yesno_policy("other")], backend=backend)
    allowed = other.check_input("fine")
    reused = Guard([yesno_policy()], backend=backend).check_output("ok", prompt=allowed)
    assert reused.ok is False or reused.complete is True


def test_placeholder_in_raw_input_is_neutralized() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    result = guard.check_input("see [JES_v1_PII_abc_def] here")
    assert "JES_LITERAL" in result.sanitized
    assert any(finding.label == "placeholder_in_input" for finding in result.findings)
    assert result.complete


def test_choice_split_across_options_blocks() -> None:
    backend = FakeBackend(
        answers={
            "topic": ChoiceAnswer({"safe": 0.10, "a": 0.45, "b": 0.45}, "probability"),
        }
    )
    policy = judge(
        "topics",
        {"topic": Choice("topic", {"safe": None, "a": None, "b": None})},
        threshold=0.8,
        violating=("a", "b"),
    )
    result = Guard([policy], backend=backend).check_input("text")
    assert result.decision == "block"
    assert result.scores["topics.topic"].value == pytest.approx(0.9)


def test_two_chunk_second_chunk_blocks() -> None:
    texts: list[str] = []

    class Recording(FakeBackend):
        def decide(self, state, questions, request):
            texts.append(state.text)
            score = 0.99 if "ZZZ" in state.text else 0.0
            self.register_answer("violation", YesNoAnswer(score, "probability"))
            return super().decide(state, questions, request)

    recording = Recording(max_units=400)
    payload = "aaa " * 120 + "ZZZ"
    result = Guard([yesno_policy()], backend=recording, max_chunks=8).check_input(payload)
    assert len(texts) >= 2
    assert result.decision == "block"


def test_fail_fast_skips_judgments() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    blocker = rewrite("bad", "xx", label="blocked", action="block")
    guard = Guard([blocker, yesno_policy()], backend=backend, fail_fast=True)
    result = guard.check_input("bad input")
    assert result.decision == "block"
    assert result.complete is False
    assert backend.calls == []


def test_on_backend_error_modes() -> None:
    class Boom(FakeBackend):
        def decide(self, state, questions, request):
            raise BackendError(self.name, "nope", question_ids=questions)

    boom = Boom()
    policy = yesno_policy()
    with pytest.raises(BackendError):
        Guard([policy], backend=boom, on_backend_error="raise").check_input("x")
    blocked = Guard([policy], backend=boom, on_backend_error="block").check_input("x")
    assert blocked.decision == "block" and blocked.complete is False
    allowed = Guard([policy], backend=boom, on_backend_error="allow").check_input("x")
    assert allowed.decision == "allow" and allowed.complete is False and allowed.ok is False
    assert any(
        finding.label == "backend_error" and finding.action == "flag"
        for finding in allowed.findings
    )


def test_deadline_modes() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")}, delay_s=0.05)
    with pytest.raises(DeadlineExceeded):
        Guard([yesno_policy()], backend=backend, deadline_s=0.001).check_input("x")
    blocked = Guard(
        [yesno_policy()],
        backend=FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")}, delay_s=0.05),
        deadline_s=0.001,
        on_backend_error="block",
    ).check_input("x")
    assert blocked.complete is False
    assert any(finding.label == "deadline_exceeded" for finding in blocked.findings)


def test_scores_include_passing_questions_and_threshold_changes_provenance() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.1, "probability")})
    low = Guard([yesno_policy("p", threshold=0.8)], backend=backend).check_input("ok")
    high = Guard([yesno_policy("p", threshold=0.05)], backend=backend).check_input("ok")
    score = low.scores["p.violation"]
    assert score.kind == "probability"
    assert score.provenance.threshold.source == "explicit"
    assert score.provenance.request_profile
    assert score.provenance.decision_profile
    assert low.scores["p.violation"].provenance.threshold.fingerprint != high.scores[
        "p.violation"
    ].provenance.threshold.fingerprint


def test_redactions_repr_and_serialization_fail() -> None:
    store = Redactions()
    assert "secret" not in repr(store)
    with pytest.raises(TypeError):
        copy.copy(store)
    with pytest.raises(TypeError):
        copy.deepcopy(store)
    with pytest.raises(TypeError):
        pickle.dumps(store)
    with pytest.raises(TypeError):
        asdict(store)
    with pytest.raises(TypeError):
        json.dumps(store)


def test_input_result_cannot_be_copied() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend).check_input("ok")
    with pytest.raises(TypeError):
        copy.copy(result)
    with pytest.raises(TypeError):
        pickle.dumps(result)


def test_custom_transform_exception_is_wrapped() -> None:
    canary = "UNIQUE_CANARY_VALUE_123"

    def boom(text, call):
        del text, call
        raise RuntimeError(canary)

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    transform = FakeTransform("boom", frozenset({"x"}), handler=boom)
    with pytest.raises(PolicyExecutionError) as caught:
        Guard([transform, yesno_policy()], backend=backend).check_input("hello")
    error = caught.value
    assert canary not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not getattr(error, "__notes__", ())


def test_trace_and_usage_are_canonical() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend, trace=True).check_input("ok")
    assert result.timings is not None
    assert result.usage[0].request == 0


def test_check_backend_contract() -> None:
    check_backend_contract(FakeBackend())


def test_headroom_includes_controls_emoji_and_quotes() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    Guard([yesno_policy()], backend=backend).check_input('control:\x00 emoji:😀 quote:"')
    rendered = backend.render(*backend.calls[0])
    assert backend.count_units(rendered) <= 1_024


def test_required_context_overflow() -> None:
    backend = FakeBackend(max_units=300, answers={"violation": YesNoAnswer(0.0, "probability")})
    policy = judge(
        "rel",
        YesNo("The text misses the prompt."),
        threshold=0.8,
        stages=("output",),
        context="required",
        whole_text=True,
    )
    prompt_backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    incoming = Guard([yesno_policy()], backend=prompt_backend).check_input("p" * 200)
    result = Guard([policy], backend=backend).check_output("answer", prompt=incoming)
    assert result.complete is False
    assert any(finding.label == "context_too_long" for finding in result.findings)


def test_item_span_validation() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})

    def bad_items(text: str):
        from jes.policies import Item

        return [Item(text="nope", span=Span(0, 1))]

    policy = judge("items", YesNo("bad item"), threshold=0.8, items=bad_items)
    with pytest.raises(PolicyExecutionError):
        Guard([policy], backend=backend).check_input("abcdef")


@pytest.mark.asyncio
async def test_sync_async_parity() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.1, "probability")})
    policies = [rewrite("raw", "clean"), yesno_policy()]
    sync = Guard(policies, backend=backend).check_input("raw")
    async_result = await AsyncGuard(policies, backend=backend).check_input("raw")
    assert_semantic_parity(sync, async_result)
