"""Shared guard planning, interpretation, and result construction."""

from __future__ import annotations

import hashlib
import math
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from jes.errors import (
    BackendError,
    DeadlineExceeded,
    PolicyError,
    PolicyExecutionError,
    RedactionError,
)
from jes.judge import (
    Backend,
    BackendResult,
    BackendUsage,
    DecisionProfile,
    RequestProfile,
)
from jes.policies._protocols import InterpretationContext, Item, Policy
from jes.questions import Question
from jes.redactions import Redactions, RedactionTransaction
from jes.types import (
    Finding,
    FindingLocation,
    History,
    InputResult,
    Message,
    Provenance,
    Role,
    SanitizationStamp,
    ScanResult,
    ScoreResult,
    Span,
    Stage,
    State,
    ThresholdProvenance,
    Timings,
    Usage,
    _TokenAuthorityManifest,
)

from .authority import (
    attach_authority,
    extract_generation_tokens,
    findings_digest,
    make_manifest,
    sign_stamp,
    verify_stamp,
)
from .limits import CheckRequestBudget, GuardLimits, ResourceLimit, SharedAdmission, utf8_size
from .merge import merge_findings, merge_scores
from .plan import (
    CompiledGuard,
    CompiledJudgment,
    PlannedRequest,
    RequestPolicy,
    SanitizedContexts,
    TransformResult,
    TransformSession,
    chunk_text,
    compile_guard,
    group_judgments,
    original_span,
    run_transforms,
)
from .restore import finalize_output
from .sensitive import FinalizerMap, UriPolicy, sensitive_settings

OnBackendError: TypeAlias = Literal["raise", "block", "allow"]
ContextValue: TypeAlias = str | ScanResult


def _history_projection(entry: History) -> tuple[Role, Stage, ContextValue]:
    """Map a history entry to the role and origin stage used for context."""

    if isinstance(entry, Message):
        if entry.role == "user":
            return "user", "input", entry.text
        if entry.role == "tool":
            return "tool", "tool_result", entry.text
        return "assistant", "output", entry.text
    if entry.stage == "input":
        return "user", "input", entry
    if entry.stage == "tool_result":
        return "tool", "tool_result", entry
    return "assistant", "output", entry


@dataclass(slots=True)
class PreparedCheck:
    stage: Stage
    original_text: str
    transformed: TransformResult
    contexts: SanitizedContexts
    redactions: Redactions
    transaction: RedactionTransaction
    findings: list[Finding]
    complete: bool
    transform_ms: float
    manifest: _TokenAuthorityManifest
    finalizers: FinalizerMap
    generation_tokens: frozenset[str]
    tool: str | None = None


@dataclass(slots=True)
class CheckPlan:
    requests: tuple[PlannedRequest, ...]
    findings: list[Finding]
    complete: bool


@dataclass(frozen=True, slots=True)
class RequestExecution:
    request: PlannedRequest
    result: BackendResult | None = None
    error: BackendError | None = None


def _hash_strings(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class BaseGuard:
    """Shared, I/O-agnostic guard behavior."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        backend: Backend | None,
        fail_fast: bool,
        on_backend_error: OnBackendError,
        limits: GuardLimits,
        deadline_s: float | None,
        trace: bool,
    ) -> None:
        if on_backend_error not in {"raise", "block", "allow"}:
            raise PolicyError("invalid on_backend_error mode")
        if deadline_s is not None and (
            isinstance(deadline_s, bool) or not math.isfinite(deadline_s) or deadline_s <= 0
        ):
            raise PolicyError("deadline_s must be positive or None")
        self._compiled: CompiledGuard = compile_guard(
            policies,
            backend,
            max_chunks=limits.max_chunks,
        )
        self._fail_fast = fail_fast
        self._on_backend_error = on_backend_error
        self._limits = limits
        self._deadline_s = deadline_s
        self._trace = trace
        self._stamp_key = secrets.token_bytes(32)
        self._admission = SharedAdmission(limits)

    def _deadline(self) -> float | None:
        return None if self._deadline_s is None else time.monotonic() + self._deadline_s

    @property
    def admission(self) -> SharedAdmission:
        return self._admission

    def close(self) -> None:
        self._admission.close()

    def _resolve_redactions(
        self,
        stage: Stage,
        explicit: Redactions | None,
        primary: ContextValue | None,
    ) -> Redactions:
        derived = primary.redactions if isinstance(primary, InputResult) else None
        if explicit is not None and derived is not None and explicit.id != derived.id:
            raise RedactionError("explicit and result redaction stores differ")
        if explicit is not None:
            return explicit
        if derived is not None:
            return derived
        # Input always returns its store; other stages use an isolated in-memory store
        # when no conversation store can be derived.
        del stage
        return Redactions()

    def _make_stamp(
        self,
        *,
        stage: Stage,
        sanitized: str,
        redactions: Redactions,
        decision: Literal["allow", "block"],
        complete: bool,
        findings: Sequence[Finding],
        manifest: _TokenAuthorityManifest,
    ) -> SanitizationStamp:
        unsigned = SanitizationStamp(
            result_id=uuid.uuid4().hex,
            stage=stage,
            config_digest=self._compiled.config_digest,
            text_digest=hashlib.sha256(sanitized.encode()).hexdigest(),
            store_id=redactions.id,
            scope_id=redactions.scope_id,
            decision=decision,
            complete=complete,
            findings_digest=findings_digest(findings),
            authority_digest=manifest.digest,
            tag="",
        )
        return replace(unsigned, tag=sign_stamp(self._stamp_key, unsigned))

    def _valid_result_context(
        self,
        result: ScanResult,
        *,
        origin_stage: Stage,
        redactions: Redactions,
    ) -> bool:
        if not result.ok or result.sanitization.stage != origin_stage:
            return False
        return verify_stamp(
            self._stamp_key,
            result,
            config_digest=self._compiled.config_digest,
            redactions=redactions,
        )

    def _sanitize_context_value(
        self,
        value: ContextValue,
        *,
        call_stage: Stage,
        origin_stage: Stage,
        redactions: Redactions,
        deadline: float | None,
        target: Literal["prompt", "question", "source", "history"],
        index: int | None = None,
        role: Role | None = None,
        session: TransformSession | None = None,
    ) -> tuple[str, tuple[Finding, ...]]:
        if isinstance(value, ScanResult):
            if not value.ok:
                finding = Finding(
                    policy="jes",
                    label="context_not_ok",
                    action="block",
                    locations=(
                        FindingLocation(
                            target=target,
                            span=Span(0, len(value.sanitized)),
                            index=index,
                            role=role,
                        ),
                    ),
                )
                return value.sanitized, (finding,)
            if self._valid_result_context(
                value,
                origin_stage=origin_stage,
                redactions=redactions,
            ):
                if session is not None:
                    session.tracked_tokens.update(extract_generation_tokens(value))
                return value.sanitized, ()
            raw = value.sanitized
        else:
            raw = value
        context_session = None
        if session is not None:
            context_session = TransformSession(
                transaction=session.transaction,
                tracked_tokens=set(),
                authorities=[],
                finalizers=FinalizerMap(),
                allow_conversation_token=origin_stage != "output",
                allow_output_local=False,
            )
        try:
            transformed = run_transforms(
                raw,
                stage=call_stage,
                origin_stage=origin_stage,
                target="context",
                policies=self._compiled.transforms,
                redactions=redactions.view(),
                deadline=deadline,
                limits=self._limits,
                location_target=target,
                location_index=index,
                location_role=role,
                run_limit_phase=False,
                neutralize=True,
                session=context_session,
            )
        except ResourceLimit:
            raise ResourceLimit("context_too_long") from None
        if session is not None and context_session is not None:
            session.tracked_tokens.update(context_session.tracked_tokens)
        return transformed.text, transformed.findings

    def _sanitize_contexts(
        self,
        *,
        stage: Stage,
        redactions: Redactions,
        deadline: float | None,
        prompt: str | InputResult | None,
        question: str | InputResult | None,
        sources: Sequence[str | ScanResult],
        history: Sequence[History],
        session: TransformSession | None = None,
    ) -> SanitizedContexts:
        count = int(prompt is not None) + int(question is not None) + len(sources) + len(history)
        if count > self._limits.max_context_items:
            raise ResourceLimit("too_many_context_items")

        findings: list[Finding] = []
        total_bytes = 0

        def add_size(text: str) -> None:
            nonlocal total_bytes
            total_bytes += utf8_size(text, limit=self._limits.max_context_bytes)
            if total_bytes > self._limits.max_context_bytes:
                raise ResourceLimit("context_too_long")

        sanitized_prompt: str | None = None
        if prompt is not None:
            sanitized_prompt, current = self._sanitize_context_value(
                prompt,
                call_stage=stage,
                origin_stage="input",
                redactions=redactions,
                deadline=deadline,
                target="prompt",
                session=session,
            )
            add_size(sanitized_prompt)
            findings.extend(current)

        sanitized_question: str | None = None
        if question is not None:
            sanitized_question, current = self._sanitize_context_value(
                question,
                call_stage=stage,
                origin_stage="input",
                redactions=redactions,
                deadline=deadline,
                target="question",
                session=session,
            )
            add_size(sanitized_question)
            findings.extend(current)

        sanitized_sources: list[str] = []
        for index, source in enumerate(sources):
            text, current = self._sanitize_context_value(
                source,
                call_stage=stage,
                origin_stage="untrusted",
                redactions=redactions,
                deadline=deadline,
                target="source",
                index=index,
                session=session,
            )
            add_size(text)
            sanitized_sources.append(text)
            findings.extend(current)

        sanitized_history: list[Message] = []
        for index, entry in enumerate(history):
            role, origin_stage, value = _history_projection(entry)
            text, current = self._sanitize_context_value(
                value,
                call_stage=stage,
                origin_stage=origin_stage,
                redactions=redactions,
                deadline=deadline,
                target="history",
                index=index,
                role=role,
                session=session,
            )
            add_size(text)
            sanitized_history.append(Message(role=role, text=text))
            findings.extend(current)

        return SanitizedContexts(
            prompt=sanitized_prompt,
            question=sanitized_question,
            sources=tuple(sanitized_sources),
            history=tuple(sanitized_history),
            findings=tuple(findings),
        )

    def prepare(
        self,
        *,
        stage: Stage,
        text: str,
        redactions: Redactions | None,
        prompt: str | InputResult | None = None,
        question: str | InputResult | None = None,
        sources: Sequence[str | ScanResult] = (),
        history: Sequence[History] = (),
        deadline: float | None,
        tool: str | None = None,
    ) -> PreparedCheck:
        started = time.perf_counter()
        primary: ContextValue | None = prompt if prompt is not None else question
        store = self._resolve_redactions(stage, redactions, primary)
        transaction = store.begin_transaction()
        session = TransformSession(
            transaction=transaction,
            tracked_tokens=set(),
            authorities=[],
            finalizers=FinalizerMap(),
            allow_conversation_token=stage not in {"output", "tool_call"},
            allow_output_local=stage == "output",
        )
        try:
            contexts = self._sanitize_contexts(
                stage=stage,
                redactions=store,
                deadline=deadline,
                prompt=prompt,
                question=question,
                sources=sources,
                history=history,
                session=session,
            )
            transformed = run_transforms(
                text,
                stage=stage,
                origin_stage=stage,
                target="subject",
                policies=self._compiled.transforms,
                redactions=transaction.view,
                deadline=deadline,
                limits=self._limits,
                location_target="subject",
                run_limit_phase=True,
                neutralize=stage not in {"output", "tool_call"},
                rewrite_placeholders=stage != "tool_call",
                tool=tool,
                session=session,
            )
            try:
                transaction.preflight()
            except RedactionError:
                raise ResourceLimit("redaction_limit_exceeded") from None
        except BaseException:
            transaction.close()
            raise
        findings = [*transformed.findings, *contexts.findings]
        complete = True
        return PreparedCheck(
            stage=stage,
            original_text=text,
            transformed=transformed,
            contexts=contexts,
            redactions=store,
            transaction=transaction,
            findings=findings,
            complete=complete,
            transform_ms=(time.perf_counter() - started) * 1_000,
            manifest=make_manifest(transformed.authorities),
            finalizers=session.finalizers,
            generation_tokens=frozenset(session.tracked_tokens),
            tool=tool,
        )

    def _fit_context(
        self,
        *,
        stage: Stage,
        contexts: SanitizedContexts,
        mode: Literal["none", "optional", "required"],
        include_sources: bool,
        backend: Backend,
        questions: Mapping[str, Question],
    ) -> State | None:
        empty = State(stage=stage, text="")
        if mode == "none":
            return empty

        primary_prompt = contexts.prompt if stage in {"output", "tool_call"} else None
        primary_question = contexts.question if stage in {"untrusted", "tool_result"} else None
        source_values = contexts.sources if include_sources else ()
        if mode == "required":
            candidate = State(
                stage=stage,
                text="",
                prompt=primary_prompt,
                question=primary_question,
                sources=source_values,
            )
            room = backend.headroom(candidate, questions)
            return candidate if room is None or room >= 64 else None

        baseline = backend.headroom(empty, questions)
        candidate = State(
            stage=stage,
            text="",
            prompt=primary_prompt,
            question=primary_question,
            sources=source_values,
        )
        room = backend.headroom(candidate, questions)
        overflowed = room is not None and (
            room < 64 or (baseline is not None and baseline - room > baseline // 2)
        )
        if overflowed:
            candidate = empty
            room = baseline

        selected: list[Message] = []
        for message in reversed(contexts.history):
            proposed = [message, *selected]
            next_state = replace(candidate, history=tuple(proposed))
            next_room = backend.headroom(next_state, questions)
            if next_room is not None and (
                next_room < 64
                or (baseline is not None and baseline - next_room > baseline // 2)
            ):
                continue
            candidate = next_state
            selected = proposed
            room = next_room
        del room
        return candidate

    def _request_profile(self, compiled: CompiledJudgment, stage: Stage) -> RequestProfile:
        return compiled.request_profiles[stage]

    def _overflow_finding(
        self,
        *,
        policy: CompiledJudgment,
        label: str,
        allow: bool,
    ) -> Finding:
        return Finding(
            policy=policy.policy.name,
            label=label,
            action="flag" if allow else "block",
        )

    def plan(self, prepared: PreparedCheck) -> CheckPlan:
        findings = list(prepared.findings)
        complete = prepared.complete
        if prepared.contexts.blocked:
            return CheckPlan(requests=(), findings=findings, complete=False)
        if prepared.transformed.blocked and self._fail_fast:
            return CheckPlan(requests=(), findings=findings, complete=False)

        applicable = [
            compiled
            for compiled in self._compiled.judgments
            if prepared.stage in compiled.policy.stages
        ]
        direct = [item for item in applicable if item.policy.subject_mode == "text"]
        item_policies = [item for item in applicable if item.policy.subject_mode == "items"]
        requests: list[PlannedRequest] = []
        logical_index = 0

        for group in group_judgments(direct):
            backend = group[0].backend
            combined: dict[str, Question] = {}
            id_maps: dict[str, dict[str, str]] = {}
            for compiled in group:
                local_map: dict[str, str] = {}
                for local_id, question_value in compiled.questions.items():
                    full_id = f"{compiled.policy.name}.{local_id}"
                    combined[full_id] = question_value
                    local_map[local_id] = full_id
                id_maps[compiled.policy.name] = local_map

            for partition in backend.partition_questions(combined):
                refs: list[RequestPolicy] = []
                for compiled in group:
                    mapping = id_maps[compiled.policy.name]
                    included = {local: full for local, full in mapping.items() if full in partition}
                    if included and len(included) != len(mapping):
                        raise PolicyError("backend split one policy across partitions")
                    if included:
                        refs.append(RequestPolicy(compiled=compiled, question_ids=included))
                if not refs:
                    continue
                first = refs[0].compiled
                base_state = self._fit_context(
                    stage=prepared.stage,
                    contexts=prepared.contexts,
                    mode=first.policy.context,
                    include_sources=first.policy.sources,
                    backend=backend,
                    questions=partition,
                )
                if base_state is None:
                    for reference in refs:
                        allow = reference.compiled.policy.on_context_overflow == "allow"
                        findings.append(
                            self._overflow_finding(
                                policy=reference.compiled,
                                label="context_too_long",
                                allow=allow,
                            )
                        )
                    complete = False
                    continue
                if prepared.tool is not None:
                    base_state = replace(base_state, tool=prepared.tool)
                try:
                    chunks = chunk_text(
                        backend=backend,
                        base_state=base_state,
                        questions=partition,
                        text=prepared.transformed.text,
                        max_chunks=self._limits.max_chunks,
                        whole_text=first.policy.whole_text,
                    )
                except ResourceLimit as error:
                    for reference in refs:
                        allow = (
                            error.label == "text_too_long"
                            and reference.compiled.policy.on_text_overflow == "allow"
                        )
                        findings.append(
                            self._overflow_finding(
                                policy=reference.compiled,
                                label=error.label,
                                allow=allow,
                            )
                        )
                    complete = False
                    continue

                profile = self._request_profile(first, prepared.stage)
                for chunk_index, (chunk, span) in enumerate(chunks):
                    location = FindingLocation(
                        target="subject",
                        span=original_span(
                            prepared.transformed.mapping,
                            span,
                            len(prepared.original_text),
                        ),
                    )
                    requests.append(
                        PlannedRequest(
                            logical_index=logical_index,
                            backend=backend,
                            state=replace(base_state, text=chunk),
                            questions=dict(partition),
                            policies=tuple(refs),
                            chunk=chunk_index,
                            location=location,
                            request_profile=profile,
                        )
                    )
                    logical_index += 1

        total_items = 0
        for compiled in item_policies:
            policy_cap = compiled.policy.max_policy_items
            effective_cap = (
                self._limits.max_items
                if policy_cap is None
                else min(self._limits.max_items, policy_cap)
            )
            extracted: list[Item] = []
            overflow = False
            for item in compiled.policy.items(prepared.transformed.text):
                if len(extracted) >= effective_cap:
                    overflow = True
                    break
                extracted.append(item)
            if overflow:
                is_global = effective_cap == self._limits.max_items
                allow = not is_global and compiled.policy.on_items_overflow == "allow"
                findings.append(
                    self._overflow_finding(
                        policy=compiled,
                        label=(
                            "too_many_items"
                            if is_global
                            else compiled.policy.item_overflow_label
                        ),
                        allow=allow,
                    )
                )
                complete = False
                continue
            total_items += len(extracted)
            if total_items > self._limits.max_items:
                findings.append(
                    self._overflow_finding(
                        policy=compiled,
                        label="too_many_items",
                        allow=False,
                    )
                )
                complete = False
                continue

            namespace = {
                f"{compiled.policy.name}.{local}": question_value
                for local, question_value in compiled.questions.items()
            }
            partitions = compiled.backend.partition_questions(namespace)
            if len(partitions) != 1:
                raise PolicyError("item policy questions must remain in one partition")
            partition = partitions[0]
            id_map = {
                local: f"{compiled.policy.name}.{local}" for local in compiled.questions
            }
            base_state = self._fit_context(
                stage=prepared.stage,
                contexts=prepared.contexts,
                mode=compiled.policy.context,
                include_sources=compiled.policy.sources,
                backend=compiled.backend,
                questions=partition,
            )
            if base_state is None:
                allow = compiled.policy.on_context_overflow == "allow"
                findings.append(
                    self._overflow_finding(
                        policy=compiled,
                        label="context_too_long",
                        allow=allow,
                    )
                )
                complete = False
                continue
            if prepared.tool is not None:
                base_state = replace(base_state, tool=prepared.tool)
            profile = self._request_profile(compiled, prepared.stage)
            for item_ordinal, item in enumerate(extracted):
                if (
                    item.span.end > len(prepared.transformed.text)
                    or prepared.transformed.text[item.span.start : item.span.end] != item.text
                ):
                    raise PolicyExecutionError("item span does not describe item text")
                chunks = chunk_text(
                    backend=compiled.backend,
                    base_state=base_state,
                    questions=partition,
                    text=item.text,
                    max_chunks=self._limits.max_chunks,
                    whole_text=False,
                )
                for chunk_index, (chunk, chunk_span) in enumerate(chunks):
                    transformed_span = Span(
                        item.span.start + chunk_span.start,
                        item.span.start + chunk_span.end,
                    )
                    location = FindingLocation(
                        target="subject",
                        span=original_span(
                            prepared.transformed.mapping,
                            transformed_span,
                            len(prepared.original_text),
                        ),
                        item_ordinal=item_ordinal,
                    )
                    requests.append(
                        PlannedRequest(
                            logical_index=logical_index,
                            backend=compiled.backend,
                            state=replace(base_state, text=chunk),
                            questions=dict(partition),
                            policies=(
                                RequestPolicy(
                                    compiled=compiled,
                                    question_ids=id_map,
                                ),
                            ),
                            chunk=chunk_index,
                            location=location,
                            request_profile=profile,
                        )
                    )
                    logical_index += 1

        slots = sum(
            request.backend.capabilities.max_attempts for request in requests
        )
        if slots > self._limits.max_requests:
            findings.append(
                Finding(policy="jes", label="too_many_requests", action="block")
            )
            return CheckPlan(requests=(), findings=findings, complete=False)
        return CheckPlan(requests=tuple(requests), findings=findings, complete=complete)

    def request_budget(self, requests: Sequence[PlannedRequest]) -> CheckRequestBudget:
        slots = (
            (request.logical_index, attempt)
            for request in requests
            for attempt in range(request.backend.capabilities.max_attempts)
        )
        return CheckRequestBudget(
            slots=slots,
            max_requests=self._limits.max_requests,
            backend_semaphore=self._admission.backend_calls,
        )

    def _decision_profile(
        self,
        request: PlannedRequest,
        reference: RequestPolicy,
    ) -> DecisionProfile:
        return reference.compiled.decision_profiles[request.state.stage]

    def interpret(
        self,
        executions: Sequence[RequestExecution],
    ) -> tuple[tuple[Finding, ...], Mapping[str, ScoreResult], tuple[Usage, ...], bool]:
        findings: list[Finding] = []
        score_maps: list[Mapping[str, ScoreResult]] = []
        usage_records: list[tuple[int, Backend, BackendUsage]] = []
        complete = True

        for execution in executions:
            request = execution.request
            if execution.error is not None:
                if execution.error.reason == "response_too_large":
                    findings.append(
                        Finding(policy="jes", label="response_too_large", action="block")
                    )
                    complete = False
                    continue
                if self._on_backend_error == "raise":
                    raise execution.error
                action = "block" if self._on_backend_error == "block" else "flag"
                label = (
                    "deadline_exceeded"
                    if isinstance(execution.error, DeadlineExceeded)
                    else "backend_error"
                )
                findings.append(Finding(policy="jes", label=label, action=action))
                complete = False
                continue
            if execution.result is None:
                raise AssertionError("request execution has neither result nor error")
            if set(execution.result.answers) != set(request.questions):
                raise BackendError(
                    request.backend.name,
                    "answer ids do not match request",
                    question_ids=request.questions,
                )
            for backend_usage in execution.result.usage:
                usage_records.append((request.logical_index, request.backend, backend_usage))

            vector_fingerprint = _hash_strings(
                *(
                    f"{reference.compiled.policy.name}:"
                    f"{reference.compiled.thresholds[request.state.stage].block_at}:"
                    f"{reference.compiled.thresholds[request.state.stage].flag_at}"
                    for reference in request.policies
                )
            )
            for reference in request.policies:
                local_answers = {
                    local: execution.result.answers[full]
                    for local, full in reference.question_ids.items()
                }
                for local, answer in local_answers.items():
                    question = reference.compiled.questions[local]
                    supported = request.backend.capabilities.supported_kinds(question.task)
                    if answer.kind not in supported:
                        raise BackendError(
                            request.backend.name,
                            "unsupported score kind",
                            question_ids=(local,),
                        )
                decision_profile = self._decision_profile(request, reference)
                stage = request.state.stage
                threshold = reference.compiled.thresholds[stage]
                threshold_fingerprint = _hash_strings(
                    str(threshold.block_at),
                    str(threshold.flag_at),
                )
                provenance = Provenance(
                    backend=request.backend.name,
                    model=request.backend.profile.model_identity,
                    request_profile=request.request_profile.fingerprint,
                    decision_profile=decision_profile.fingerprint,
                    prompt_version=reference.compiled.policy.version,
                    threshold=ThresholdProvenance(
                        source=reference.compiled.threshold_sources[stage],
                        block_at=threshold.block_at,
                        flag_at=threshold.flag_at,
                        fingerprint=threshold_fingerprint,
                        vector_fingerprint=vector_fingerprint,
                        evaluation_run=reference.compiled.evaluation_runs[stage],
                    ),
                )
                context = InterpretationContext(
                    provenance=provenance,
                    chunk=request.chunk,
                    item_ordinal=request.location.item_ordinal,
                    location=request.location,
                )
                outcome = reference.compiled.policy.interpret(
                    local_answers,
                    threshold,
                    context,
                )
                if set(outcome.scores) != set(reference.compiled.questions):
                    raise PolicyExecutionError(
                        f"policy {reference.compiled.policy.name} returned incomplete scores"
                    )
                for finding in outcome.findings:
                    if finding.label not in reference.compiled.policy.labels:
                        raise PolicyExecutionError("policy emitted an undeclared label")
                findings.extend(outcome.findings)
                score_maps.append(
                    {
                        f"{reference.compiled.policy.name}.{local}": score
                        for local, score in outcome.scores.items()
                    }
                )

        public_usage: list[Usage] = []
        for public_index, (logical_index, backend, record) in enumerate(
            sorted(usage_records, key=lambda value: (value[0], value[2].attempt))
        ):
            expected = f":{logical_index}:{record.attempt}"
            if not record.permit_id.endswith(expected):
                raise BackendError(backend.name, "usage permit does not match request")
            public_usage.append(
                Usage(
                    backend=backend.name,
                    model=backend.profile.model_identity,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    request=public_index,
                    attempt=record.attempt,
                )
            )
        return (
            merge_findings(findings),
            merge_scores(score_maps),
            tuple(public_usage),
            complete,
        )

    def _metadata_size(self, findings: Sequence[Finding]) -> int:
        total = 0
        for finding in findings:
            total += len(finding.policy.encode("utf-8")) + len(finding.label.encode("utf-8"))
            if finding.question is not None:
                total += len(finding.question.encode("utf-8"))
        return total

    def _limit_findings(
        self,
        findings: Sequence[Finding],
        *,
        authorities: int,
    ) -> tuple[tuple[Finding, ...], bool]:
        locations = sum(len(finding.locations) for finding in findings)
        label: str | None = None
        if authorities > self._limits.max_authorities:
            label = "too_many_authorities"
        elif locations > self._limits.max_locations:
            label = "too_many_locations"
        elif self._metadata_size(findings) > self._limits.max_metadata_bytes:
            label = "metadata_too_large"
        elif len(findings) >= self._limits.max_findings:
            label = "too_many_findings"
        if label is None:
            return tuple(findings), True
        kept = list(findings[: max(self._limits.max_findings - 1, 0)])
        kept.append(Finding(policy="jes", label=label, action="block"))
        return tuple(kept[: self._limits.max_findings]), False

    def result(
        self,
        *,
        prepared: PreparedCheck,
        plan: CheckPlan,
        interpretation: tuple[
            tuple[Finding, ...],
            Mapping[str, ScoreResult],
            tuple[Usage, ...],
            bool,
        ],
        judgment_ms: float,
    ) -> ScanResult:
        interpreted_findings, scores, usage, execution_complete = interpretation
        merged = merge_findings((*plan.findings, *interpreted_findings))
        findings, within_limit = self._limit_findings(
            merged,
            authorities=len(prepared.manifest.entries),
        )
        complete = plan.complete and execution_complete and within_limit
        decision: Literal["allow", "block"] = (
            "block" if any(item.action == "block" for item in findings) else "allow"
        )
        output_text = prepared.transformed.text
        if prepared.stage == "output" and decision == "allow" and complete:
            uri_policy, restore_context = self._restore_settings()
            restored, uri_findings = finalize_output(
                original=prepared.original_text,
                sanitized=prepared.transformed.text,
                store_values=prepared.redactions.snapshot_values(),
                generation_tokens=prepared.generation_tokens,
                finalizers=prepared.finalizers,
                uri_policy=uri_policy,
                restore_context=restore_context,
                max_restorations=self._limits.max_restorations,
                max_restored_output_bytes=self._limits.max_restored_output_bytes,
            )
            findings = (*findings, *uri_findings)
            if restored is None:
                decision = "block"
                complete = False
            else:
                output_text = restored
                if any(item.action == "block" for item in uri_findings):
                    decision = "block"
        if decision == "allow" and complete:
            try:
                prepared.transaction.commit()
            except Exception:
                prepared.transaction.close()
                findings = (*findings, Finding("jes", "redaction_limit_exceeded", "block"))
                decision = "block"
                complete = False
                output_text = prepared.transformed.text
        else:
            prepared.transaction.close()
            if not complete:
                output_text = prepared.transformed.text
        stamp = self._make_stamp(
            stage=prepared.stage,
            sanitized=prepared.transformed.text,
            redactions=prepared.redactions,
            decision=decision,
            complete=complete,
            findings=findings,
            manifest=prepared.manifest,
        )
        timings = (
            Timings(
                transforms_ms=prepared.transform_ms,
                judgments_ms=judgment_ms,
                finalizers_ms=0.0,
            )
            if self._trace
            else None
        )
        result = self._scan_result(
            stage=prepared.stage,
            text=output_text,
            sanitized=prepared.transformed.text,
            decision=decision,
            complete=complete,
            findings=findings,
            scores=scores,
            sanitization=stamp,
            usage=usage,
            timings=timings,
            redactions=prepared.redactions,
        )
        attach_authority(result, prepared.manifest)
        return result

    def _restore_settings(self) -> tuple[UriPolicy | None, bool]:
        uri_policy: UriPolicy | None = None
        restore_context = True
        for policy in self._compiled.transforms:
            settings = sensitive_settings(policy)
            if settings is None:
                continue
            if settings.uri_policy is not None:
                uri_policy = settings.uri_policy
            if not settings.restore_context:
                restore_context = False
        return uri_policy, restore_context

    def empty_interpretation(
        self,
    ) -> tuple[tuple[Finding, ...], Mapping[str, ScoreResult], tuple[Usage, ...], bool]:
        return (), {}, (), True

    def failure_result(
        self,
        *,
        stage: Stage,
        redactions: Redactions,
        label: str,
        action: Literal["flag", "block"] = "block",
    ) -> ScanResult:
        finding = Finding(policy="jes", label=label, action=action)
        decision: Literal["allow", "block"] = "block" if action == "block" else "allow"
        findings = (finding,)
        stamp = self._make_stamp(
            stage=stage,
            sanitized="",
            redactions=redactions,
            decision=decision,
            complete=False,
            findings=findings,
            manifest=make_manifest(()),
        )
        return self._scan_result(
            stage=stage,
            text="",
            sanitized="",
            decision=decision,
            complete=False,
            findings=findings,
            scores={},
            sanitization=stamp,
            usage=(),
            timings=None,
            redactions=redactions,
        )

    @staticmethod
    def _scan_result(
        *,
        stage: Stage,
        text: str,
        sanitized: str,
        decision: Literal["allow", "block"],
        complete: bool,
        findings: tuple[Finding, ...],
        scores: Mapping[str, ScoreResult],
        sanitization: SanitizationStamp,
        usage: tuple[Usage, ...],
        timings: Timings | None,
        redactions: Redactions,
    ) -> ScanResult:
        if stage == "input":
            return InputResult(
                stage=stage,
                text=text,
                sanitized=sanitized,
                decision=decision,
                complete=complete,
                findings=findings,
                scores=scores,
                sanitization=sanitization,
                usage=usage,
                timings=timings,
                redactions=redactions,
            )
        return ScanResult(
            stage=stage,
            text=text,
            sanitized=sanitized,
            decision=decision,
            complete=complete,
            findings=findings,
            scores=scores,
            sanitization=sanitization,
            usage=usage,
            timings=timings,
        )

