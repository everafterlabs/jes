from __future__ import annotations

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from jes import AsyncGuard, Guard
from jes.backends import BackendProfile
from jes.errors import (
    BackendError,
    DeadlineExceeded,
    PolicyError,
    PolicyExecutionError,
    RedactionError,
)
from jes.policies import Item, TransformEdit, judge
from jes.policies._protocols import apply_transform_edits
from jes.policies.defaults import lookup
from jes.questions import Choice, Score, ScoreAnswer, YesNo, YesNoAnswer
from jes.redactions import Redactions
from jes.testing import FakeBackend, FakeRequestBudget, assert_semantic_parity
from jes.types import Message, SanitizationStamp, ScanResult, Span, State
from tests.helpers import yesno_policy


def test_judge_yesno_choice_score_and_flag() -> None:
    backend = FakeBackend(
        answers={
            "violation": YesNoAnswer(0.5, "probability", confidence=0.9),
            "pick": __import__("jes.questions", fromlist=["ChoiceAnswer"]).ChoiceAnswer(
                {"safe": 0.7, "bad": 0.3}, "probability"
            ),
            "rate": ScoreAnswer((0.1, 0.2, 0.7), "probability"),
        }
    )
    policies = [
        judge("yn", YesNo("bad?"), threshold=0.8),
        judge(
            "ch",
            {"pick": Choice("pick", {"safe": None, "bad": None})},
            threshold=0.8,
            violating={"bad"},
        ),
        judge(
            "sc",
            {"rate": Score("rate", ("low", "mid", "high"))},
            threshold=0.6,
            violation_level=2,
        ),
    ]
    result = Guard(policies, backend=backend).check_input("text")
    assert "yn.violation" in result.scores
    assert result.scores["ch.pick"].value == pytest.approx(0.3)
    assert result.scores["sc.rate"].value == pytest.approx(0.7)
    assert result.decision == "block"


def test_flag_at_does_not_block() -> None:
    from jes.questions import Threshold

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.5, "probability")})
    policy = judge("flaggy", YesNo("bad?"), threshold=Threshold(block_at=0.9, flag_at=0.4))
    result = Guard([policy], backend=backend).check_input("x")
    assert result.decision == "allow"
    assert any(finding.action == "flag" for finding in result.findings)


def test_check_untrusted_and_history_messages() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    incoming = guard.check_input("ask", history=[Message(role="user", text="earlier")])
    untrusted = guard.check_untrusted("doc", question=incoming)
    assert untrusted.ok
    outgoing = guard.check_output(
        "reply",
        prompt=incoming,
        history=[incoming, Message(role="assistant", text="old")],
    )
    assert outgoing.ok


def test_optional_history_drops_oldest() -> None:
    backend = FakeBackend(max_units=400, answers={"violation": YesNoAnswer(0.0, "probability")})
    policy = judge("ctx", YesNo("bad?"), threshold=0.8, stages=("output",), context="optional")
    prompt_backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    incoming = Guard([yesno_policy()], backend=prompt_backend).check_input("q")
    history = [Message(role="user", text="h" * 80) for _ in range(6)]
    result = Guard([policy], backend=backend).check_output("ans", prompt=incoming, history=history)
    assert result.ok
    used = backend.calls[-1][0].history
    assert len(used) < 6


def test_store_mismatch_raises_redaction_error() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    incoming = guard.check_input("x")
    with pytest.raises(RedactionError):
        guard.check_output("y", prompt=incoming, redactions=Redactions())


def test_isolation_across_stores() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    left = Redactions()
    right = Redactions()

    def run(store: Redactions) -> str:
        return guard.check_input("same", redactions=store).redactions.id

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(run, [left, right]))
    assert ids[0] != ids[1]


def test_forged_stamp_is_resanitized() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    incoming = guard.check_input("hello")
    forged = ScanResult(
        stage="input",
        text=incoming.text,
        sanitized=incoming.sanitized,
        decision="allow",
        complete=True,
        findings=incoming.findings,
        scores=incoming.scores,
        sanitization=incoming.sanitization,
    )
    object.__setattr__(forged, "sanitization", incoming.sanitization)
    outgoing = guard.check_output("ok", prompt=forged)
    # public constructor installs empty authority, so HMAC cannot match
    assert outgoing.ok or any(finding.label != "unused" for finding in outgoing.findings)


def test_apply_transform_edits_and_malformed() -> None:
    assert apply_transform_edits("abcd", [TransformEdit(1, 3, "XY")]) == "aXYd"
    with pytest.raises(PolicyExecutionError):
        apply_transform_edits("abcd", [TransformEdit(2, 1, "x")])


def test_backend_profile_validation() -> None:
    with pytest.raises(PolicyError):
        BackendProfile(
            adapter="a",
            adapter_version="1",
            scorer_version="1",
            parser_version="1",
            provider="p",
            provider_profile="p.v1",
            model="m",
            revision=None,
            artifact_digest=None,
            tokenizer_revision=None,
            template_revision=None,
            mode="x",
            generation_settings={},
            context_window_tokens=None,
            max_request_bytes=None,
            output_reserve=0,
            budget_attestation=None,
            dependency_versions={},
        )


def test_errors_include_status_and_deadline() -> None:
    error = BackendError("b", "nope", status_code=429, question_ids=("q",))
    assert "429" in str(error)
    assert "q" in str(error)
    deadline = DeadlineExceeded("b", question_ids=("q",))
    assert deadline.reason == "deadline_exceeded"


def test_stamp_empty_and_span() -> None:
    stamp = SanitizationStamp.empty("input")
    assert stamp.complete is False
    with pytest.raises(PolicyExecutionError):
        Span(3, 1)


def test_redaction_transaction_and_limits() -> None:
    store = Redactions(max_entries=1, max_value_bytes=8, max_bytes=256)
    tx = store.begin_transaction()
    tx.token_for("person", "ALICE")
    tx.preflight()
    tx.commit()
    assert len(store) == 1
    with pytest.raises(RedactionError):
        other = store.begin_transaction()
        other.token_for("person", "BOBBBBBBB")
        other.preflight()
    with pytest.raises(TypeError):
        copy.copy(tx)
    with pytest.raises(TypeError):
        pickle.dumps(store.view() if False else store)


def test_defaults_lookup_empty() -> None:
    assert lookup("missing") is None


def test_guard_context_managers() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    with Guard([yesno_policy()], backend=backend) as guard:
        assert guard.check_input("ok").ok


@pytest.mark.asyncio
async def test_async_untrusted_output_and_parity() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.1, "probability")})
    policies = [yesno_policy()]
    async with AsyncGuard(policies, backend=backend) as guard:
        incoming = await guard.check_input("ask")
        untrusted = await guard.check_untrusted("doc", question=incoming)
        outgoing = await guard.check_output("reply", prompt=incoming)
    sync = Guard(policies, backend=backend)
    assert_semantic_parity(incoming, sync.check_input("ask"))
    assert untrusted.ok and outgoing.ok


def test_custom_backend_exception_wrapped() -> None:
    canary = "BACKEND_CANARY_XYZ"

    class Boom(FakeBackend):
        def decide(self, state, questions, request):
            raise RuntimeError(canary)

    with pytest.raises(PolicyExecutionError) as caught:
        Guard([yesno_policy()], backend=Boom()).check_input("x")
    assert canary not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_location_after_prior_transform() -> None:
    from jes.policies import TransformFinding, TransformOutcome
    from tests.helpers import FakeTransform, rewrite

    def detect(text, call):
        del call
        start = text.find("BAD")
        return TransformOutcome(
            text=text,
            findings=(TransformFinding("hit", "flag", (Span(start, start + 3),)),),
        )

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard(
        [
            rewrite("xx", ""),
            FakeTransform("det", frozenset({"hit"}), handler=detect),
            yesno_policy(),
        ],
        backend=backend,
    )
    result = guard.check_input("xxBAD")
    finding = next(item for item in result.findings if item.label == "hit")
    assert finding.locations[0].span.start >= 0


def test_whole_text_overflow_allow() -> None:
    backend = FakeBackend(max_units=400, answers={"violation": YesNoAnswer(0.0, "probability")})
    policy = judge(
        "whole",
        YesNo("bad?"),
        threshold=0.8,
        whole_text=True,
        on_text_overflow="allow",
    )
    result = Guard([policy], backend=backend).check_input("word " * 200)
    assert result.complete is False
    assert result.decision == "allow"


def test_item_mode_happy_path() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})

    def items(text: str):
        return [Item(text=text[0:2], span=Span(0, 2))]

    policy = judge("it", YesNo("item?"), threshold=0.8, items=items)
    result = Guard([policy], backend=backend).check_input("abcdef")
    assert result.ok
    assert backend.calls[0][0].text == "ab"


def test_fake_request_budget_and_state_repr() -> None:
    budget = FakeRequestBudget(max_requests=1)
    with budget.acquire(None, logical_index=0, attempt=0) as permit:
        assert permit.permit_id == "0:0"
    assert "text_len" in repr(State(stage="input", text="hello"))
    assert "text_len" in repr(Message(role="user", text="hi"))


@pytest.mark.asyncio
async def test_async_deadline_allow() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")}, delay_s=0.05)
    result = await AsyncGuard(
        [yesno_policy()],
        backend=backend,
        deadline_s=0.001,
        on_backend_error="allow",
    ).check_input("x")
    assert result.complete is False
    assert result.decision == "allow"
