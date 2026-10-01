"""The check pipeline: transforms, planned backend requests, and the result.

Nothing here does I/O. ``Guard`` and ``AsyncGuard`` run ``prepare`` and ``plan``, send
the planned requests their own way, and hand the outcomes to ``finish``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import islice
from typing import Literal, TypeAlias, TypeVar, cast

from jes.backend import AsyncBackend, Reply, SyncBackend
from jes.engine.chunking import chunk_spans
from jes.engine.redact import Redactor
from jes.engine.restore import (
    LocalMarkers,
    Origin,
    UrlMode,
    escape_placeholders,
    restore_tokens,
)
from jes.errors import (
    BackendError,
    DeadlineExceeded,
    PolicyError,
    PolicyExecutionError,
    RedactionError,
)
from jes.limits import LimitExceeded, Limits, require_size, utf8_size
from jes.policies.base import (
    PHASES,
    ItemExtractor,
    Judgment,
    Policy,
    SensitiveHit,
    SensitivePolicy,
    Transform,
    TransformContext,
    TransformFinding,
    TransformOutcome,
    merge_spans,
)
from jes.policies.sensitive import Pii
from jes.questions import Answer, Question
from jes.redactions import TOKEN_RE, Redactions
from jes.result import Result
from jes.text.textmap import TextMap
from jes.types import (
    Action,
    Finding,
    Message,
    Role,
    ScoreResult,
    Span,
    Stage,
    State,
    Target,
    Usage,
    validate_identifier,
)

Backend: TypeAlias = SyncBackend | AsyncBackend
Outcome: TypeAlias = Reply | BackendError
OnBackendError: TypeAlias = Literal["raise", "block", "allow"]
_T = TypeVar("_T")

_MAX_CONTEXT_ITEMS = 1_024
# Normalized and restored text may grow, but not past this multiple of the input limit.
_GROWTH = 4
# A request must leave this much room, and context may take at most half of it.
_MIN_ROOM = 64
_RANK = {"flag": 0, "redact": 1, "block": 2}
_ROLE_STAGES: dict[Role, Stage] = {"user": "input", "tool": "tool_result", "assistant": "output"}


@dataclass(frozen=True, slots=True)
class Check:
    """One check's input. ``prompt`` is the user's request for output and tool checks."""

    stage: Stage
    text: str
    tool: str | None = None
    prompt: str | Result | None = None
    question: str | Result | None = None
    sources: tuple[str | Result, ...] = ()
    history: tuple[Result | Message, ...] = ()
    redactions: Redactions | None = None


@dataclass(slots=True)
class Context:
    """The context judges may see, already sanitized, and the tokens it carries."""

    prompt: str | None = None
    question: str | None = None
    sources: tuple[str, ...] = ()
    history: tuple[Message, ...] = ()
    tokens: frozenset[str] = frozenset()


@dataclass(slots=True)
class Prepared:
    check: Check
    store: Redactions
    started: float
    text: str
    map: TextMap
    findings: list[Finding]
    context: Context
    redactor: Redactor


@dataclass(frozen=True, slots=True, repr=False)
class Planned:
    """One backend request: the state, the questions, and which judgments asked them."""

    backend: Backend
    state: State
    questions: Mapping[str, Question]
    judgments: tuple[tuple[Judgment, Mapping[str, str]], ...]
    span: Span

    def __repr__(self) -> str:
        return f"Planned(backend={self.backend.name!r}, questions={sorted(self.questions)})"


@dataclass(slots=True)
class Plan:
    requests: list[Planned]
    findings: list[Finding]
    complete: bool


class Pipeline:
    """A guard's validated policies, and the steps of a check."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        backend: Backend | None,
        limits: Limits,
        on_backend_error: OnBackendError,
        fail_fast: bool,
        sync_only: bool,
    ) -> None:
        if on_backend_error not in ("raise", "block", "allow"):
            raise PolicyError("on_backend_error must be raise, block, or allow")
        self.limits = limits
        self.on_backend_error: OnBackendError = on_backend_error
        self.fail_fast = fail_fast
        transforms: list[Transform | SensitivePolicy] = []
        self.judgments: list[tuple[Judgment, Backend]] = []
        names: set[str] = set()
        # Typed as object: a guard must reject anything that is not a policy at runtime.
        for policy in cast(Sequence[object], policies):
            if not isinstance(policy, (Judgment, SensitivePolicy, Transform)):
                raise PolicyError(f"a {type(policy).__name__} is not a transform or a judgment")
            name = validate_identifier(policy.name, field="policy name")
            if name in names:
                raise PolicyError(f"two policies are named {name}")
            names.add(name)
            if isinstance(policy, Judgment):
                self.judgments.append((policy, _judge_backend(policy, backend, sync_only)))
                continue
            if policy.phase not in PHASES:
                raise PolicyError(f"policy {name} has an unknown phase")
            transforms.append(policy)
        self.transforms = sorted(transforms, key=lambda item: PHASES.index(item.phase))
        pii = [policy for policy in transforms if isinstance(policy, Pii)]
        self.restore = all(policy.restore for policy in pii)
        origins: set[Origin] = set()
        for policy in pii:
            origins |= policy.restore_origins
        self.origins = frozenset(origins)
        self.on_url: UrlMode = (
            "allow"
            if pii and all(item.on_placeholder_in_url == "allow" for item in pii)
            else "block"
        )

    # prepare: resolve the store, check sizes, and run transforms on context and text.

    def prepare(self, check: Check, deadline: float | None) -> Prepared:
        started = time.perf_counter()
        store = resolve_store(check)
        redactor = Redactor(store, LocalMarkers(), self.limits.max_redactions)
        require_size(check.text, self.limits.max_input_bytes, "input_too_long")
        findings: list[Finding] = []
        context = self._context(check, store, redactor, deadline, findings)
        text, mapping = self._transform(
            check.text,
            check=check,
            origin=check.stage,
            target="text",
            index=None,
            redactor=redactor,
            deadline=deadline,
            findings=findings,
        )
        if redactor.staged:
            try:
                store._check_room(redactor.staged)
            except RedactionError:
                raise LimitExceeded("redaction_store_full") from None
        return Prepared(check, store, started, text, mapping, findings, context, redactor)

    def _context(
        self,
        check: Check,
        store: Redactions,
        redactor: Redactor,
        deadline: float | None,
        findings: list[Finding],
    ) -> Context:
        entries: list[tuple[str | Result | Message, Stage, Target, int | None]] = []
        if check.prompt is not None:
            entries.append((check.prompt, "input", "prompt", None))
        if check.question is not None:
            entries.append((check.question, "input", "question", None))
        entries.extend(
            (source, "untrusted", "source", index) for index, source in enumerate(check.sources)
        )
        for index, entry in enumerate(check.history):
            entries.append((entry, _history_origin(entry), "history", index))
        if len(entries) > _MAX_CONTEXT_ITEMS:
            raise LimitExceeded("too_many_context_items")
        total = 0
        texts: list[str] = []
        tokens: set[str] = set()
        for value, origin, target, index in entries:
            content = value.text if isinstance(value, Message) else value
            if isinstance(content, Result) and content.redactions is store:
                # A result from this conversation's store is already sanitized.
                if not content.ok:
                    findings.append(
                        Finding("jes", "context_not_ok", "block", target=target, index=index)
                    )
                sanitized = content.sanitized
            else:
                raw = content.original if isinstance(content, Result) else content
                sanitized, _map = self._transform(
                    raw,
                    check=check,
                    origin=origin,
                    target=target,
                    index=index,
                    redactor=redactor,
                    deadline=deadline,
                    findings=findings,
                )
            total += utf8_size(sanitized)
            if total > self.limits.max_context_bytes:
                raise LimitExceeded("context_too_long")
            tokens.update(TOKEN_RE.findall(sanitized))
            texts.append(sanitized)
        context = Context(tokens=frozenset(tokens))
        history: list[Message] = []
        sources: list[str] = []
        for (value, origin, target, _index), text in zip(entries, texts, strict=True):
            if target == "prompt":
                context.prompt = text
            elif target == "question":
                context.question = text
            elif target == "source":
                sources.append(text)
            else:
                history.append(Message(_origin_role(value, origin), text))
        context.sources, context.history = tuple(sources), tuple(history)
        return context

    def _transform(
        self,
        text: str,
        *,
        check: Check,
        origin: Stage,
        target: Target,
        index: int | None,
        redactor: Redactor,
        deadline: float | None,
        findings: list[Finding],
    ) -> tuple[str, TextMap]:
        subject = target == "text"
        limit = _GROWTH * (
            self.limits.max_input_bytes if subject else self.limits.max_context_bytes
        )
        # The arguments of a tool call pass on unchanged, so a redaction there must block.
        editable = not (subject and check.stage == "tool_call")
        mapping = TextMap.identity(len(text))
        current = text
        if not (subject and check.stage == "output"):
            escapes = escape_placeholders(current)
            if escapes:
                spans = tuple(Span(edit.start, edit.end) for edit in escapes)
                findings.append(
                    Finding(
                        "jes",
                        "placeholder_in_text",
                        "flag",
                        spans=spans,
                        target=target,
                        index=index,
                    )
                )
                current, mapping = mapping.apply(current, escapes)
        context = TransformContext(
            stage=check.stage,
            origin=origin,
            target=target,
            tool=check.tool if subject else None,
            deadline=deadline,
        )
        for policy in self.transforms:
            if origin not in policy.stages:
                continue
            context.check_deadline()
            if isinstance(policy, SensitivePolicy):
                hits = _guarded(policy.name, policy.detect, current, context)
                _check_hits(policy.name, hits, len(current))
                edits = redactor.edits(
                    current,
                    hits,
                    tokens=origin != "output" and not (subject and check.stage == "tool_call"),
                    local=subject and check.stage == "output",
                    key=policy.hmac_key,
                )
                reported = _grouped(hits)
            else:
                outcome = _guarded(policy.name, policy.apply, current, context)
                _check_outcome(policy.name, outcome, len(current))
                edits, reported = list(outcome.edits), outcome.findings
            for item in reported:
                action: Action = (
                    "block" if item.action == "redact" and not editable else item.action
                )
                spans = tuple(mapping.origin(span) for span in item.spans)
                findings.append(
                    Finding(
                        policy.name, item.label, action, spans=spans, target=target, index=index
                    )
                )
            try:
                current, mapping = mapping.apply(current, edits)
            except (TypeError, ValueError):
                raise PolicyExecutionError(f"policy {policy.name} returned invalid edits") from None
            require_size(current, limit, "normalized_too_long")
        return current, mapping

    # plan: which judgments run, with what context, in how many chunks.

    def plan(self, prepared: Prepared) -> Plan:
        plan = Plan([], [], complete=True)
        if self.fail_fast and any(item.action == "block" for item in prepared.findings):
            plan.complete = False
            return plan
        stage = prepared.check.stage
        groups: dict[tuple[int, str, bool, bool], list[tuple[Judgment, Backend]]] = {}
        item_judgments: list[tuple[Judgment, Backend]] = []
        for judgment, backend in self.judgments:
            if stage not in judgment.stages:
                continue
            if judgment.items is not None:
                item_judgments.append((judgment, backend))
                continue
            key = (id(backend), judgment.context, judgment.sources, judgment.whole_text)
            groups.setdefault(key, []).append((judgment, backend))
        for group in groups.values():
            self._plan_text(prepared, group, plan)
        items_left = self.limits.max_items
        for judgment, backend in item_judgments:
            items_left = self._plan_items(prepared, judgment, backend, plan, items_left)
        if len(plan.requests) > self.limits.max_requests:
            plan.requests = []
            plan.findings.append(Finding("jes", "too_many_requests", "block"))
            plan.complete = False
        return plan

    def _plan_text(
        self, prepared: Prepared, group: list[tuple[Judgment, Backend]], plan: Plan
    ) -> None:
        first, backend = group[0]
        questions, judgments = _namespaced(judgment for judgment, _backend in group)
        state = self._fit(prepared, backend, questions, first, plan)
        if state is None:
            for judgment, _backend in group:
                _overflow(judgment, "context_too_long", plan)
            return
        text = prepared.text
        spans = self._chunks(text, state, backend, questions, whole_text=first.whole_text)
        if spans is None:
            label = "text_too_long" if first.whole_text else "too_many_chunks"
            for judgment, _backend in group:
                _overflow(judgment, label, plan)
            return
        for span in spans:
            plan.requests.append(
                Planned(
                    backend=backend,
                    state=replace(state, text=text[span.start : span.end]),
                    questions=questions,
                    judgments=judgments,
                    span=prepared.map.origin(span),
                )
            )

    def _plan_items(
        self,
        prepared: Prepared,
        judgment: Judgment,
        backend: Backend,
        plan: Plan,
        items_left: int,
    ) -> int:
        # Only judgments with an extractor reach this method.
        extractor = cast(ItemExtractor, judgment.items)
        cap = items_left if judgment.max_items is None else min(items_left, judgment.max_items)
        text = prepared.text
        extracted = list(islice(_guarded(judgment.name, extractor, text), cap + 1))
        if len(extracted) > cap:
            _overflow(judgment, "too_many_items", plan)
            return items_left
        if not extracted:
            return items_left
        questions, judgments = _namespaced((judgment,))
        state = self._fit(prepared, backend, questions, judgment, plan)
        if state is None:
            _overflow(judgment, "context_too_long", plan)
            return items_left - len(extracted)
        for item in extracted:
            span = item.span
            if span.end > len(text) or text[span.start : span.end] != item.text:
                raise PolicyExecutionError(
                    f"policy {judgment.name} returned an item outside the text"
                )
            spans = self._chunks(item.text, state, backend, questions, whole_text=False)
            if spans is None:
                _overflow(judgment, "too_many_chunks", plan)
                continue
            for chunk in spans:
                located = Span(span.start + chunk.start, span.start + chunk.end)
                plan.requests.append(
                    Planned(
                        backend=backend,
                        state=replace(state, text=item.text[chunk.start : chunk.end]),
                        questions=questions,
                        judgments=judgments,
                        span=prepared.map.origin(located),
                    )
                )
        return items_left - len(extracted)

    def _fit(
        self,
        prepared: Prepared,
        backend: Backend,
        questions: Mapping[str, Question],
        judgment: Judgment,
        plan: Plan,
    ) -> State | None:
        """The request state without text: the context that fits, or None when it must."""

        check, context = prepared.check, prepared.context
        base = State(stage=check.stage, text="", tool=check.tool)
        if judgment.context == "none":
            return base
        budget = backend.headroom(base, questions)

        def fits(state: State) -> bool:
            room = backend.headroom(state, questions)
            if room is None or budget is None:
                return room is None or room >= _MIN_ROOM
            return room >= _MIN_ROOM and budget - room <= budget // 2

        request = context.question if check.stage == "untrusted" else context.prompt
        candidate = replace(
            base,
            prompt=None if check.stage == "untrusted" else request,
            question=request if check.stage == "untrusted" else None,
            sources=context.sources if judgment.sources else (),
        )
        if not fits(candidate):
            if judgment.context == "required":
                return None
            target: Target = "question" if check.stage == "untrusted" else "prompt"
            plan.findings.append(Finding("jes", "context_dropped", "flag", target=target))
            candidate = base
        history = context.history
        # Keep the newest messages that fit, without gaps: binary search the count.
        low, high = 0, len(history)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(replace(candidate, history=history[len(history) - middle :])):
                low = middle
            else:
                high = middle - 1
        if low < len(history):
            plan.findings.append(Finding("jes", "history_truncated", "flag", target="history"))
        return replace(candidate, history=history[len(history) - low :]) if low else candidate

    def _chunks(
        self,
        text: str,
        state: State,
        backend: Backend,
        questions: Mapping[str, Question],
        *,
        whole_text: bool,
    ) -> list[Span] | None:
        def fits(chunk: str) -> bool:
            room = backend.headroom(replace(state, text=chunk), questions)
            return room is None or room >= 0

        if whole_text:
            return [Span(0, len(text))] if fits(text) else None
        return chunk_spans(text, fits, self.limits.max_chunks)

    # finish: read answers, restore the reply, and commit new tokens.

    def finish(self, prepared: Prepared, plan: Plan, outcomes: Sequence[Outcome]) -> Result:
        check = prepared.check
        findings = [*prepared.findings, *plan.findings]
        complete = plan.complete
        scores: dict[str, ScoreResult] = {}
        usage: list[Usage] = []
        for planned, outcome in zip(plan.requests, outcomes, strict=True):
            try:
                if isinstance(outcome, BackendError):
                    raise outcome
                reply = outcome
                found = self._read(planned, reply)
            except BackendError as error:
                if self.on_backend_error == "raise":
                    raise
                label = (
                    "deadline_exceeded" if isinstance(error, DeadlineExceeded) else "backend_error"
                )
                action: Action = "block" if self.on_backend_error == "block" else "flag"
                findings.append(Finding("jes", label, action))
                complete = False
                continue
            usage.append(Usage(planned.backend.model, reply.input_tokens, reply.output_tokens))
            for judgment, question_id, score, action_or_none, confidence in found:
                key = f"{judgment.name}.{question_id}"
                if key not in scores or score > scores[key].value:
                    scores[key] = ScoreResult(score, planned.backend.model, confidence)
                if action_or_none is not None:
                    findings.append(
                        Finding(
                            judgment.name,
                            question_id,
                            action_or_none,
                            score=score,
                            spans=(planned.span,),
                        )
                    )
        merged = merge_findings(findings)
        decision: Literal["allow", "block"] = (
            "block" if any(item.action == "block" for item in merged) else "allow"
        )
        onward = check.text if check.stage == "tool_call" else prepared.text
        if check.stage == "output" and decision == "allow" and complete:
            onward, restore_findings = self._restore(prepared)
            if restore_findings:
                merged = merge_findings([*merged, *restore_findings])
                if any(item.action == "block" for item in restore_findings):
                    decision = "block"
        if decision == "allow" and complete and prepared.redactor.staged:
            try:
                prepared.store._commit(prepared.redactor.staged)
            except RedactionError:
                merged = merge_findings([*merged, Finding("jes", "redaction_store_full", "block")])
                decision, complete = "block", False
        return Result(
            stage=check.stage,
            decision=decision,
            complete=complete,
            original=check.text,
            sanitized=prepared.text,
            findings=merged,
            scores=scores,
            usage=tuple(usage),
            duration_ms=(time.perf_counter() - prepared.started) * 1_000,
            redactions=prepared.store,
            _onward=onward,
        )

    def _read(
        self, planned: Planned, reply: Reply
    ) -> list[tuple[Judgment, str, float, Action | None, float | None]]:
        if not set(planned.questions) <= set(reply.answers):
            raise BackendError(
                planned.backend.name, "malformed_reply", question_ids=planned.questions
            )
        found: list[tuple[Judgment, str, float, Action | None, float | None]] = []
        for judgment, ids in planned.judgments:
            answers: dict[str, Answer] = {local: reply.answers[full] for local, full in ids.items()}
            try:
                evaluated = judgment.evaluate(answers)
            except BackendError:
                raise BackendError(
                    planned.backend.name, "malformed_answer", question_ids=ids.values()
                ) from None
            for question_id, score, action in evaluated:
                found.append(
                    (judgment, question_id, score, action, answers[question_id].confidence)
                )
        return found

    def _restore(self, prepared: Prepared) -> tuple[str, list[Finding]]:
        text = prepared.redactor.markers.restore(prepared.text)
        findings: list[Finding] = []
        if self.restore:
            text, findings = restore_tokens(
                text,
                lookup=prepared.store._value,
                authorized=prepared.context.tokens,
                origins=self.origins,
                on_url=self.on_url,
            )
        if utf8_size(text) > _GROWTH * self.limits.max_input_bytes:
            return text, [*findings, Finding("jes", "restored_output_too_large", "block")]
        return text, findings

    def failed(
        self,
        check: Check,
        store: Redactions,
        label: str,
        action: Action,
        started: float,
    ) -> Result:
        """A result for a check that stopped before its judgments finished."""

        return Result(
            stage=check.stage,
            decision="block" if action == "block" else "allow",
            complete=False,
            original=check.text,
            sanitized="",
            findings=(Finding("jes", label, action),),
            scores={},
            usage=(),
            duration_ms=(time.perf_counter() - started) * 1_000,
            redactions=store,
        )


def resolve_store(check: Check) -> Redactions:
    """The conversation store: given, carried by the prompt or question result, or new."""

    carried = [
        value.redactions for value in (check.prompt, check.question) if isinstance(value, Result)
    ]
    stores = ([check.redactions] if check.redactions is not None else []) + carried
    if any(store is not stores[0] for store in stores):
        raise RedactionError("redactions= and the prompt result belong to different stores")
    return stores[0] if stores else Redactions()


def merge_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """One finding per policy, label, and place, with the strongest action and score."""

    merged: dict[tuple[str, str, Target, int | None], Finding] = {}
    for finding in findings:
        key = (finding.policy, finding.label, finding.target, finding.index)
        previous = merged.get(key)
        if previous is None:
            merged[key] = finding
            continue
        action = max(previous.action, finding.action, key=_RANK.__getitem__)
        scores = [score for score in (previous.score, finding.score) if score is not None]
        merged[key] = replace(
            previous,
            action=action,
            score=max(scores) if scores else None,
            spans=merge_spans((*previous.spans, *finding.spans)),
        )
    return tuple(merged.values())


def _judge_backend(judgment: Judgment, default: Backend | None, sync_only: bool) -> Backend:
    backend = judgment.backend if judgment.backend is not None else default
    if backend is None:
        raise PolicyError(f"judgment {judgment.name} has no model; pass model= to the guard")
    if sync_only and not isinstance(backend, SyncBackend):
        raise PolicyError(f"the backend for {judgment.name} is async-only; use AsyncGuard")
    questions, _ids = _namespaced((judgment,))
    for stage in judgment.stages:
        room = backend.headroom(State(stage=stage, text=""), questions)
        if room is not None and room < _MIN_ROOM:
            raise PolicyError(f"the questions of {judgment.name} leave no room for text")
    return backend


def _namespaced(
    judgments: Iterable[Judgment],
) -> tuple[dict[str, Question], tuple[tuple[Judgment, Mapping[str, str]], ...]]:
    questions: dict[str, Question] = {}
    references: list[tuple[Judgment, Mapping[str, str]]] = []
    for judgment in judgments:
        ids = {local: f"{judgment.name}.{local}" for local in judgment.questions}
        questions.update({ids[local]: question for local, question in judgment.questions.items()})
        references.append((judgment, ids))
    return questions, tuple(references)


def _overflow(judgment: Judgment, label: str, plan: Plan) -> None:
    action: Action = "block" if judgment.on_overflow == "block" else "flag"
    plan.findings.append(Finding(judgment.name, label, action))
    plan.complete = False


def _history_origin(entry: Result | Message) -> Stage:
    if isinstance(entry, Message):
        return _ROLE_STAGES[entry.role]
    return entry.stage if entry.stage in ("input", "tool_result") else "output"


def _origin_role(value: str | Result | Message, origin: Stage) -> Role:
    if isinstance(value, Message):
        return value.role
    return "user" if origin == "input" else "tool" if origin == "tool_result" else "assistant"


def _grouped(hits: Sequence[SensitiveHit]) -> list[TransformFinding]:
    """One finding per entity and action, spanning every hit."""

    spans: dict[tuple[str, Action], list[Span]] = {}
    for hit in hits:
        spans.setdefault((hit.entity, hit.action), []).append(hit.span)
    return [
        TransformFinding(entity, action, merge_spans(found))
        for (entity, action), found in spans.items()
    ]


def _guarded(name: str, function: Callable[..., _T], *args: object) -> _T:
    """Run policy code. Anything but a jes error becomes PolicyExecutionError, text-free."""

    try:
        return function(*args)
    except (DeadlineExceeded, LimitExceeded, PolicyError):
        raise
    except Exception:
        raise PolicyExecutionError(f"policy {name} failed") from None


def _check_outcome(name: str, outcome: object, length: int) -> None:
    if not isinstance(outcome, TransformOutcome):
        raise PolicyExecutionError(f"policy {name} did not return a TransformOutcome")
    for finding in outcome.findings:
        _check_finding(name, finding.label, finding.action, finding.spans, length)


def _check_hits(name: str, hits: object, length: int) -> None:
    if not isinstance(hits, list):
        raise PolicyExecutionError(f"policy {name} did not return a list of hits")
    for hit in cast(list[object], hits):
        if (
            not isinstance(hit, SensitiveHit)
            or hit.span.end > length
            or hit.span.start == hit.span.end
        ):
            raise PolicyExecutionError(f"policy {name} returned a hit outside the text")
        _check_finding(name, hit.entity, hit.action, (hit.span,), length)


def _check_finding(name: str, label: str, action: str, spans: Iterable[Span], length: int) -> None:
    try:
        validate_identifier(label, field="finding label")
    except PolicyError:
        raise PolicyExecutionError(f"policy {name} returned an invalid finding label") from None
    if action not in _RANK:
        raise PolicyExecutionError(f"policy {name} returned an unknown action")
    if any(span.end > length for span in spans):
        raise PolicyExecutionError(f"policy {name} returned a span outside the text")


__all__ = [
    "Check",
    "Context",
    "OnBackendError",
    "Outcome",
    "Pipeline",
    "Plan",
    "Planned",
    "Prepared",
]
