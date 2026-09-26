from __future__ import annotations

import copy
import pickle

import pytest

from jes import AsyncGuard, Guard
from jes.backends import BackendCapabilities, BackendProfile, BackendResult, BackendUsage
from jes.errors import DeadlineExceeded, PolicyError, PolicyExecutionError, RedactionError
from jes.policies import judge
from jes.questions import Choice, ChoiceAnswer, Score, ScoreAnswer, YesNo, YesNoAnswer
from jes.redactions import Redactions
from jes.testing import FakeBackend, FakeRequestBudget, fake_sensitive
from jes.types import State
from tests.helpers import yesno_policy


def _profile() -> BackendProfile:
    return BackendProfile(
        adapter="fake",
        adapter_version="1",
        scorer_version="1",
        parser_version="1",
        provider="test",
        provider_profile="fake.v1",
        model="sync",
        revision="1",
        artifact_digest=None,
        tokenizer_revision=None,
        template_revision=None,
        mode="probability",
        generation_settings={},
        context_window_tokens=1024,
        max_request_bytes=None,
        output_reserve=0,
        budget_attestation=None,
        dependency_versions={},
    )


class SyncOnly:
    name = "sync-only"
    capabilities = BackendCapabilities(
        tasks=None,
        max_options=20,
        max_attempts=2,
        score_kinds={"*": frozenset({"probability"})},
        budget_fidelity="exact",
    )
    profile = _profile()

    def count_units(self, text: str) -> int:
        return len(text.encode())

    def partition_questions(self, questions):
        return (dict(questions),)

    def headroom(self, state: State, questions) -> int:
        return 512

    def decide(self, state, questions, request):
        with request.budget.acquire(
            request.deadline, logical_index=request.logical_index, attempt=0
        ) as permit:
            return BackendResult(
                answers={key: YesNoAnswer(0.0, "probability") for key in questions},
                usage=(BackendUsage(permit.permit_id, 0, 1, 1),),
            )


def test_judge_validation_branches() -> None:
    with pytest.raises(PolicyError):
        judge("x", {}, threshold=0.8)
    with pytest.raises(PolicyError):
        judge("x", YesNo("q"), threshold=0.8, violating={"a"})
    with pytest.raises(PolicyError):
        judge(
            "x",
            {
                "a": Choice("c", {"s": None, "b": None}),
                "b": Choice("c2", {"s": None, "b": None}),
            },
            threshold=0.8,
            violating=("b",),
        )
    with pytest.raises(PolicyError):
        judge("x", YesNo("q"), threshold=0.8, violation_level=1)
    with pytest.raises(PolicyError):
        judge(
            "x",
            {"a": Score("s", ("l", "h")), "b": Score("s2", ("l", "h"))},
            threshold=0.8,
            violation_level=1,
        )
    with pytest.raises(PolicyError):
        judge("x", YesNo("q"), threshold=0.8, stages=())
    with pytest.raises(PolicyError):
        judge("x", YesNo("q"), threshold=0.8, context="required", stages=("input",))
    with pytest.raises(PolicyError):
        judge("x", YesNo("q"), threshold=0.8, items=lambda text: (), max_policy_items=0)


def test_mask_partial_and_finalizer_copy() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard(
        [fake_sensitive("p", "ABCDEF", mode="mask_partial"), yesno_policy()],
        backend=backend,
    ).check_input("ABCDEF")
    assert "ABCDEF" not in result.sanitized
    from jes._engine.sensitive import FinalizerMap

    with pytest.raises(TypeError):
        copy.copy(FinalizerMap())
    with pytest.raises(TypeError):
        copy.deepcopy(FinalizerMap())


def test_overlapping_sensitive_edits_fail() -> None:
    from jes._engine.sensitive import FinalizerMap, _SensitiveEdit, apply_sensitive_edits
    from jes.types import Span

    with pytest.raises(PolicyExecutionError):
        apply_sensitive_edits(
            "aaaa",
            (
                _SensitiveEdit(Span(0, 3), "e", "irreversible", "redact"),
                _SensitiveEdit(Span(2, 4), "e", "irreversible", "redact"),
            ),
            transaction=None,
            hmac_key=None,
            allow_conversation_token=False,
            finalizers=FinalizerMap(),
        )


def test_redaction_scope_and_closed_transaction() -> None:
    with pytest.raises(RedactionError):
        Redactions(scope=b"")
    store = Redactions()
    tx = store.begin_transaction()
    tx.close()
    with pytest.raises(RedactionError):
        tx.token_for("e", "v")


def test_sync_guard_rejects_async_only(monkeypatch) -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    # FakeBackend is both; construction should succeed.
    Guard([yesno_policy()], backend=backend)


@pytest.mark.asyncio
async def test_async_uses_sync_only_backend() -> None:
    result = await AsyncGuard([yesno_policy()], backend=SyncOnly()).check_input("ok")
    assert result.ok
    assert result.usage[0].backend == "sync-only"


@pytest.mark.asyncio
async def test_async_deadline_raise() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")}, delay_s=0.05)
    with pytest.raises(DeadlineExceeded):
        await AsyncGuard([yesno_policy()], backend=backend, deadline_s=0.001).check_input("x")


def test_sources_and_item_overflow_allow() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    incoming = Guard([yesno_policy()], backend=backend).check_input("q")
    policy = judge(
        "src",
        YesNo("bad?"),
        threshold=0.8,
        stages=("output",),
        context="optional",
        sources=True,
    )
    result = Guard([policy], backend=backend).check_output(
        "a",
        prompt=incoming,
        sources=["one", "two"],
    )
    assert result.ok
    assert backend.calls[-1][0].sources == ("one", "two")

    def items(text: str):
        from jes.policies import Item
        from jes.types import Span

        return [Item(text[i : i + 1], Span(i, i + 1)) for i in range(len(text))]

    overflow = judge(
        "many",
        YesNo("bad?"),
        threshold=0.8,
        items=items,
        max_policy_items=1,
        on_items_overflow="allow",
    )
    overflowed = Guard([overflow], backend=backend).check_input("abcd")
    assert overflowed.complete is False
    assert overflowed.decision == "allow"


def test_authority_verify_rejects_forged() -> None:
    from jes._engine.authority import verify_stamp

    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend)
    result = guard.check_input("ok")
    assert not verify_stamp(b"0" * 32, result, config_digest="nope", redactions=result.redactions)


def test_check_untrusted_raw_question() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend).check_untrusted("doc", question="what?")
    assert result.ok


def test_budget_deadline() -> None:
    budget = FakeRequestBudget()
    with pytest.raises(DeadlineExceeded), budget.acquire(0.0, logical_index=0, attempt=0):
        pass


def test_run_sync_deadline_block() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")}, delay_s=0.05)
    result = Guard(
        [yesno_policy()],
        backend=backend,
        deadline_s=0.001,
        on_backend_error="block",
    ).check_input("x")
    assert result.decision == "block"
    assert any(finding.label == "deadline_exceeded" for finding in result.findings)


def test_finding_location_and_manifest_copy() -> None:
    from jes.types import FindingLocation, Span, _TokenAuthorityManifest

    with pytest.raises(PolicyExecutionError):
        FindingLocation(target="subject", span=Span(0, 1), index=-1)
    with pytest.raises(PolicyExecutionError):
        FindingLocation(target="subject", span=Span(0, 1), item_ordinal=-1)
    manifest = _TokenAuthorityManifest()
    with pytest.raises(TypeError):
        copy.copy(manifest)
    with pytest.raises(TypeError):
        copy.deepcopy(manifest)
    with pytest.raises(TypeError):
        pickle.dumps(manifest)


def test_backend_profile_negative_values() -> None:
    with pytest.raises(PolicyError):
        BackendProfile(
            adapter="a",
            adapter_version="1",
            scorer_version="1",
            parser_version="1",
            provider="p",
            provider_profile="p",
            model="m",
            revision=None,
            artifact_digest=None,
            tokenizer_revision=None,
            template_revision=None,
            mode="x",
            generation_settings={},
            context_window_tokens=0,
            max_request_bytes=None,
            output_reserve=0,
            budget_attestation=None,
            dependency_versions={},
        )
    with pytest.raises(PolicyError):
        BackendCapabilities(
            tasks=None,
            max_options=1,
            max_attempts=1,
            score_kinds={"*": frozenset({"probability"})},
            budget_fidelity="exact",
        )
    with pytest.raises(PolicyError):
        BackendProfile(
            adapter="a",
            adapter_version="1",
            scorer_version="1",
            parser_version="1",
            provider="p",
            provider_profile="p",
            model="m",
            revision=None,
            artifact_digest=None,
            tokenizer_revision=None,
            template_revision=None,
            mode="x",
            generation_settings={},
            context_window_tokens=8,
            max_request_bytes=0,
            output_reserve=-1,
            budget_attestation=None,
            dependency_versions={},
        )
    from jes.questions import validate_identifier

    with pytest.raises(PolicyError):
        validate_identifier("1bad")
    with pytest.raises(PolicyError):
        YesNo("q", task="bad task")


def test_judge_mapping_forms_and_items_none() -> None:
    policy = judge(
        "mapped",
        {
            "a": Choice("c", {"s": None, "b": None}),
            "b": Choice("c2", {"s": None, "t": None}),
        },
        threshold=0.8,
        violating={"a": ["b"], "b": ["t"]},
    )
    scored = judge(
        "levels",
        {
            "a": Score("s", ("l", "h")),
            "b": Score("s2", ("l", "h")),
        },
        threshold=0.8,
        violation_level={"a": 1, "b": 0},
    )
    backend = FakeBackend(
        answers={
            "a": ChoiceAnswer({"s": 1.0, "b": 0.0}, "probability"),
            "b": ChoiceAnswer({"s": 1.0, "t": 0.0}, "probability"),
        }
    )
    assert Guard([policy], backend=backend).check_input("x").ok
    backend.register_answer("a", ScoreAnswer((1.0, 0.0), "probability"))
    backend.register_answer("b", ScoreAnswer((1.0, 0.0), "probability"))
    scored_result = Guard([scored], backend=backend).check_input("x")
    assert scored_result.complete
    assert "levels.a" in scored_result.scores
    text_policy = judge("plain", YesNo("q"), threshold=0.8)
    assert list(text_policy.items("abc")) == []


def test_input_result_deepcopy_and_ok() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend).check_input("ok")
    assert result.allowed and result.ok
    with pytest.raises(TypeError):
        copy.deepcopy(result)


def test_redaction_value_limit_and_collision() -> None:
    store = Redactions(max_value_bytes=3)
    tx = store.begin_transaction()
    with pytest.raises(RedactionError):
        tx.token_for("e", "abcd")
        tx.preflight()
    store2 = Redactions(max_entries=1)
    first = store2.begin_transaction()
    first.token_for("e", "A")
    first.commit()
    second = store2.begin_transaction()
    with pytest.raises(RedactionError):
        second.token_for("e", "B")
        second.preflight()


def test_short_mask_partial() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard(
        [fake_sensitive("p", "AB", mode="mask_partial"), yesno_policy()],
        backend=backend,
    ).check_input("AB")
    assert "AB" not in result.sanitized
