"""Testing helpers for policy and backend implementations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import AsyncGenerator, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from typing import Literal, cast

from jes._engine.sensitive import (
    _SensitiveEdit,
    _SensitiveMode,
    _SensitiveOutcome,
    register_sensitive,
)
from jes.errors import BackendError, DeadlineExceeded, PolicyError
from jes.judge import (
    Backend,
    BackendCapabilities,
    BackendProfile,
    BackendResult,
    BackendUsage,
    RequestContext,
    RequestPermit,
)
from jes.policies._protocols import (
    CallContext,
    Phase,
    TransformFinding,
    TransformOutcome,
    TransformPolicy,
)
from jes.policies.defaults import _clear as clear_defaults, _register as register_default
from jes.questions import Answer, Question, ScoreKind, YesNo, YesNoAnswer
from jes.redactions import RedactionTransaction
from jes.types import Action, InputResult, ScanResult, Span, Stage, State, Usage


class FakeRequestBudget:
    """A deterministic per-check request budget for tests."""

    def __init__(self, max_requests: int = 128) -> None:
        self._max_requests = max_requests
        self._slots: set[tuple[int, int]] = set()
        self._lock = threading.Lock()

    def _permit(
        self,
        deadline: float | None,
        logical_index: int,
        attempt: int,
    ) -> RequestPermit:
        if deadline is not None and time.monotonic() >= deadline:
            raise DeadlineExceeded("request_budget")
        slot = (logical_index, attempt)
        with self._lock:
            if slot not in self._slots and len(self._slots) >= self._max_requests:
                raise BackendError("request_budget", "request_limit")
            self._slots.add(slot)
        return RequestPermit(permit_id=f"{logical_index}:{attempt}", deadline=deadline)

    @contextmanager
    def acquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> Generator[RequestPermit, None, None]:
        yield self._permit(deadline, logical_index, attempt)

    @asynccontextmanager
    async def aacquire(
        self,
        deadline: float | None,
        *,
        logical_index: int,
        attempt: int,
    ) -> AsyncGenerator[RequestPermit, None]:
        yield self._permit(deadline, logical_index, attempt)


class FakeBackend:
    """A deterministic backend that records every state and question batch.

    An answer registered under a namespaced id such as ``injection.violation``
    is used before an answer registered under the bare local id ``violation``.
    ``default_answer`` fills any question that still has no registered id.
    ``None`` leaves that question as ``missing_fake_answer``.
    A key such as ``output:S1`` is used on that stage before the bare ``S1``.
    """

    name = "fake"

    def __init__(
        self,
        answers: Mapping[str, Answer] | None = None,
        *,
        max_units: int | None = 1_024,
        budget_fidelity: Literal["exact", "conservative"] = "exact",
        tasks: frozenset[str] | None = None,
        max_options: int | None = 20,
        max_attempts: int = 1,
        score_kinds: Mapping[str, frozenset[ScoreKind]] | None = None,
        delay_s: float = 0.0,
        model: str = "fake@1",
        default_answer: Answer | None = None,
    ) -> None:
        self.capabilities = BackendCapabilities(
            tasks=tasks,
            max_options=max_options,
            max_attempts=max_attempts,
            score_kinds=score_kinds or {"*": frozenset({"probability"})},
            budget_fidelity=budget_fidelity,
        )
        self.profile = BackendProfile(
            adapter="fake",
            adapter_version="1",
            scorer_version="1",
            parser_version="1",
            provider="test",
            provider_profile="fake.v1",
            model=model,
            revision="1",
            artifact_digest=hashlib.sha256(model.encode()).hexdigest(),
            tokenizer_revision="fake-bytes.v1",
            template_revision="fake-json.v1",
            mode="probability",
            generation_settings={},
            context_window_tokens=max_units if budget_fidelity == "exact" else None,
            max_request_bytes=max_units if budget_fidelity == "conservative" else None,
            output_reserve=0,
            budget_attestation="fake",
            dependency_versions={},
        )
        self._answers = dict(answers or {})
        self._default_answer = default_answer
        self._max_units = max_units
        self._delay_s = delay_s
        self._lock = threading.Lock()
        self.calls: list[tuple[State, Mapping[str, Question]]] = []

    def register_answer(self, question_id: str, answer: Answer) -> None:
        self._answers[question_id] = answer

    @staticmethod
    def render(state: State, questions: Mapping[str, Question]) -> str:
        state_payload: dict[str, object] = {
            "stage": state.stage,
            "text": state.text,
            "prompt": state.prompt,
            "question": state.question,
            "sources": state.sources,
            "history": [{"role": message.role, "text": message.text} for message in state.history],
        }
        if state.tool is not None:
            state_payload["tool"] = state.tool
        payload = {
            "state": state_payload,
            "questions": {
                key: {
                    "type": type(question).__name__,
                    "instructions": question.instructions,
                }
                for key, question in questions.items()
            },
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def count_units(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def partition_questions(
        self,
        questions: Mapping[str, Question],
    ) -> tuple[Mapping[str, Question], ...]:
        return (dict(questions),)

    def headroom(self, state: State, questions: Mapping[str, Question]) -> int | None:
        if self._max_units is None:
            return None
        return self._max_units - self.count_units(self.render(state, questions))

    def _result(
        self,
        state: State,
        questions: Mapping[str, Question],
        permit: RequestPermit,
        attempt: int,
    ) -> BackendResult:
        room = self.headroom(state, questions)
        if room is not None and room < 0:
            raise BackendError(self.name, "request_does_not_fit", question_ids=questions)
        selected: dict[str, Answer] = {}
        missing: set[str] = set()
        for question_id in questions:
            local = question_id.rsplit(".", 1)[-1]
            staged = f"{state.stage}:{question_id}"
            staged_local = f"{state.stage}:{local}"
            if staged in self._answers:
                selected[question_id] = self._answers[staged]
            elif question_id in self._answers:
                selected[question_id] = self._answers[question_id]
            elif staged_local in self._answers:
                selected[question_id] = self._answers[staged_local]
            elif local in self._answers:
                selected[question_id] = self._answers[local]
            elif self._default_answer is not None:
                selected[question_id] = self._default_answer
            else:
                missing.add(question_id)
        if missing:
            raise BackendError(self.name, "missing_fake_answer", question_ids=missing)
        with self._lock:
            self.calls.append((state, dict(questions)))
        return BackendResult(
            answers=selected,
            usage=(
                BackendUsage(
                    permit_id=permit.permit_id,
                    attempt=attempt,
                    input_tokens=self.count_units(self.render(state, questions)),
                    output_tokens=len(selected),
                ),
            ),
        )

    def decide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        with request.budget.acquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            if self._delay_s:
                time.sleep(self._delay_s)
            if request.deadline is not None and time.monotonic() >= request.deadline:
                raise DeadlineExceeded(self.name, question_ids=questions)
            return self._result(state, questions, permit, 0)

    async def adecide(
        self,
        state: State,
        questions: Mapping[str, Question],
        request: RequestContext,
    ) -> BackendResult:
        async with request.budget.aacquire(
            request.deadline,
            logical_index=request.logical_index,
            attempt=0,
        ) as permit:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            if request.deadline is not None and time.monotonic() >= request.deadline:
                raise DeadlineExceeded(self.name, question_ids=questions)
            return self._result(state, questions, permit, 0)


def _contract_questions(backend: Backend) -> Mapping[str, Question]:
    tasks = backend.capabilities.tasks
    task = "custom"
    if tasks is not None and "custom" not in tasks:
        if "injection" in tasks:
            task = "injection"
        elif "hazard.any" in tasks:
            task = "hazard.any"
        else:
            task = sorted(tasks)[0]
    return {"contract": YesNo("The text violates the contract.", task=task)}


def check_backend_contract(backend: Backend) -> None:
    """Run fast, network-free structural backend checks."""

    questions = _contract_questions(backend)
    partitions = backend.partition_questions(questions)
    if not partitions or set().union(*(set(partition) for partition in partitions)) != set(
        questions
    ):
        raise PolicyError("backend partitions must cover every question exactly")
    for partition in partitions:
        if len(partition) != len(set(partition)):
            raise PolicyError("backend partition contains duplicate question ids")
        state = State(stage="input", text='control:\x00 emoji:😀 quote:"')
        headroom = backend.headroom(state, partition)
        if headroom is not None and headroom < 0:
            # A small contract request must fit every usable backend profile.
            raise PolicyError("backend leaves no room for a contract request")

    from jes.judge import SyncBackend

    if not isinstance(backend, SyncBackend):
        return
    if isinstance(backend, FakeBackend):
        backend.register_answer("contract", YesNoAnswer(0.0, "probability"))
    budget = FakeRequestBudget()
    try:
        result = backend.decide(
            State(stage="input", text="contract"),
            questions,
            RequestContext(
                logical_index=0,
                deadline=None,
                budget=budget,
                max_response_bytes=8_192,
            ),
        )
    except BackendError as error:
        if error.reason in {"connection_error", "http_error"}:
            return
        raise PolicyError(f"backend contract decide failed: {error.reason}") from error
    if set(result.answers) != set(questions):
        raise PolicyError("backend did not return exactly one answer per question")
    answer = result.answers["contract"]
    task = questions["contract"].task
    if answer.kind not in backend.capabilities.supported_kinds(task):
        raise PolicyError("backend returned an unsupported score kind")
    if not result.usage:
        raise PolicyError("backend must report per-call usage")
    if any(item.permit_id == "" for item in result.usage):
        raise PolicyError("backend usage must carry the acquired permit")


class FakeSensitiveTransform:
    """Private-channel sensitive transform used by Milestone 1 engine tests."""

    phase: Phase = "detect"

    def __init__(
        self,
        name: str,
        marker: str,
        *,
        entity: str = "secret",
        mode: _SensitiveMode = "irreversible",
        action: Action = "redact",
        stages: frozenset[Stage] = frozenset({"input", "untrusted", "output"}),
        hmac_key: bytes | None = None,
    ) -> None:
        self.name = name
        self.labels = frozenset({entity})
        self.stages = stages
        self.fingerprint: str | None = f"fake-sensitive.{name}"
        self.marker = marker
        self.entity = entity
        self.mode: _SensitiveMode = mode
        self.action: Action = action
        register_sensitive(cast(TransformPolicy, self), hmac_key=hmac_key)

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        del call
        return TransformOutcome(text=text)

    def _apply_sensitive(
        self,
        text: str,
        call: CallContext,
        transaction: RedactionTransaction,
    ) -> _SensitiveOutcome:
        del call, transaction
        edits: list[_SensitiveEdit] = []
        findings: list[TransformFinding] = []
        start = 0
        while True:
            index = text.find(self.marker, start)
            if index < 0:
                break
            span = Span(index, index + len(self.marker))
            edits.append(_SensitiveEdit(span, self.entity, self.mode, self.action))
            findings.append(TransformFinding(self.entity, self.action, (span,)))
            start = span.end
        return _SensitiveOutcome(tuple(edits), tuple(findings))


def fake_sensitive(
    name: str,
    marker: str,
    *,
    entity: str = "secret",
    mode: _SensitiveMode = "irreversible",
    action: Action = "redact",
    stages: frozenset[Stage] = frozenset({"input", "untrusted", "output"}),
    hmac_key: bytes | None = None,
) -> FakeSensitiveTransform:
    return FakeSensitiveTransform(
        name,
        marker,
        entity=entity,
        mode=mode,
        action=action,
        stages=stages,
        hmac_key=hmac_key,
    )


def assert_semantic_parity(left: ScanResult, right: ScanResult) -> None:
    """Compare sync/async results while ignoring instance-specific identifiers."""

    assert left.stage == right.stage
    assert left.decision == right.decision
    assert left.complete == right.complete
    assert left.text == right.text
    assert left.sanitized == right.sanitized
    assert [(item.policy, item.label, item.action, item.question) for item in left.findings] == [
        (item.policy, item.label, item.action, item.question) for item in right.findings
    ]
    assert {key: (score.value, score.kind) for key, score in left.scores.items()} == {
        key: (score.value, score.kind) for key, score in right.scores.items()
    }

    def usage_key(item: Usage) -> tuple[object, ...]:
        return (
            item.backend,
            item.model,
            item.request,
            item.attempt,
            item.input_tokens,
            item.output_tokens,
        )

    assert [usage_key(item) for item in left.usage] == [usage_key(item) for item in right.usage]
    if isinstance(left, InputResult) and isinstance(right, InputResult):
        assert left.redactions.snapshot_values() == right.redactions.snapshot_values()


__all__ = [
    "FakeBackend",
    "FakeRequestBudget",
    "FakeSensitiveTransform",
    "assert_semantic_parity",
    "check_backend_contract",
    "clear_defaults",
    "fake_sensitive",
    "register_default",
]
