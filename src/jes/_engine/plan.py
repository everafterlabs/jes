"""Pure policy compilation, transformation, grouping, and chunk planning."""

from __future__ import annotations

import base64
import hashlib
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from jes.errors import DeadlineExceeded, PolicyError, PolicyExecutionError
from jes.judge import Backend, DecisionProfile, RequestProfile, attestation_digest_for
from jes.policies import defaults
from jes.policies._protocols import (
    CallContext,
    JudgmentPolicy,
    Policy,
    TransformEdit,
    TransformOutcome,
    TransformPolicy,
    validate_transform_outcome,
)
from jes.questions import Choice, Question, Score, Threshold, validate_identifier
from jes.redactions import RedactionTransaction, RedactionView
from jes.types import Finding, FindingLocation, Message, Role, Span, Stage, State, _TokenAuthority

from .limits import GuardLimits, ResourceLimit, utf8_size
from .sensitive import (
    FinalizerMap,
    _SensitiveEdit,
    _SensitiveOutcome,
    _SensitiveTransform,
    apply_sensitive_edits,
    is_sensitive,
    sensitive_key,
)

_PHASES = ("normalize", "detect", "limit")
_PLACEHOLDER_RE = re.compile(r"\[JES_v[0-9]+_[^\]\r\n]{1,256}\]")


@dataclass(frozen=True, slots=True)
class CompiledJudgment:
    policy: JudgmentPolicy
    backend: Backend
    questions: Mapping[str, Question]
    thresholds: Mapping[Stage, Threshold]
    threshold_sources: Mapping[Stage, Literal["explicit", "default"]]
    evaluation_runs: Mapping[Stage, str | None]
    request_profiles: Mapping[Stage, RequestProfile]
    decision_profiles: Mapping[Stage, DecisionProfile]

    @property
    def threshold(self) -> Threshold:
        return next(iter(self.thresholds.values()))


@dataclass(frozen=True, slots=True)
class CompiledGuard:
    transforms: tuple[TransformPolicy, ...]
    judgments: tuple[CompiledJudgment, ...]
    config_digest: str


@dataclass(slots=True)
class TransformSession:
    transaction: RedactionTransaction | None
    tracked_tokens: set[str]
    authorities: list[_TokenAuthority]
    finalizers: FinalizerMap
    allow_conversation_token: bool
    allow_output_local: bool


@dataclass(frozen=True, slots=True)
class TransformResult:
    text: str
    mapping: tuple[Span, ...]
    findings: tuple[Finding, ...]
    authorities: tuple[_TokenAuthority, ...] = ()
    restored_text: str | None = None

    @property
    def blocked(self) -> bool:
        return any(finding.action == "block" for finding in self.findings)


@dataclass(frozen=True, slots=True)
class SanitizedContexts:
    prompt: str | None = None
    question: str | None = None
    sources: tuple[str, ...] = ()
    history: tuple[Message, ...] = ()
    findings: tuple[Finding, ...] = ()

    @property
    def blocked(self) -> bool:
        return any(finding.action == "block" for finding in self.findings)


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    compiled: CompiledJudgment
    question_ids: Mapping[str, str]  # local id -> namespaced id


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    logical_index: int
    backend: Backend
    state: State
    questions: Mapping[str, Question]
    policies: tuple[RequestPolicy, ...]
    chunk: int
    location: FindingLocation
    request_profile: RequestProfile


def _policy_is_judgment(policy: Policy) -> bool:
    return hasattr(policy, "questions") and hasattr(policy, "interpret")


def _digest(parts: Iterable[str]) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


_PLANNER_VERSION = "m1.v1"
_RENDERER_VERSION = "backend.v1"
_ENGINE_VERSION = "m1.v1"
_MERGE_VERSION = "m1.v1"


def build_request_profile(
    *,
    backend: Backend,
    transform_digest: str,
    stage: Stage,
    context_mode: Literal["none", "optional", "required"],
    sources: bool,
    subject_mode: Literal["text", "items"],
    questions: Mapping[str, Question],
    max_chunks: int,
) -> RequestProfile:
    backend_fingerprint = _digest((repr(backend.profile), repr(backend.capabilities)))
    question_digest = _digest(
        f"{key}:{question.task}:{question.instructions}" for key, question in questions.items()
    )
    chunk_digest = _digest((str(max_chunks),))
    budget_digest = _digest(
        (
            str(backend.profile.context_window_tokens),
            str(backend.profile.max_request_bytes),
            str(backend.profile.output_reserve),
        )
    )
    attestation = attestation_digest_for(backend)
    fingerprint = _digest(
        (
            backend_fingerprint,
            attestation or "",
            transform_digest,
            stage,
            context_mode,
            str(sources),
            subject_mode,
            question_digest,
            chunk_digest,
            budget_digest,
            _PLANNER_VERSION,
            _RENDERER_VERSION,
        )
    )
    return RequestProfile(
        backend_fingerprint=backend_fingerprint,
        attestation_digest=attestation,
        transform_digest=transform_digest,
        engine_dependency_digest=_ENGINE_VERSION,
        transform_asset_digest="m1.none",
        renderer_version=_RENDERER_VERSION,
        planner_version=_PLANNER_VERSION,
        stage=stage,
        context_mode=context_mode,
        sources=sources,
        subject_mode=subject_mode,
        question_partition_digest=question_digest,
        chunk_config_digest=chunk_digest,
        budget_digest=budget_digest,
        fingerprint=fingerprint,
    )


def build_decision_profile(
    *,
    request: RequestProfile,
    stage: Stage,
    policy: JudgmentPolicy,
    questions: Mapping[str, Question],
    backend: Backend,
) -> DecisionProfile:
    kinds = _digest(
        f"{question.task}:" + ",".join(sorted(backend.capabilities.supported_kinds(question.task)))
        for question in questions.values()
    )
    subset = _digest(sorted(questions))
    fingerprint = _digest(
        (
            request.fingerprint,
            stage,
            policy.kind,
            policy.version,
            subset,
            policy.interpretation_version,
            _MERGE_VERSION,
            kinds,
        )
    )
    return DecisionProfile(
        request_profile=request.fingerprint,
        stage=stage,
        policy_kind=policy.kind,
        policy_version=policy.version,
        subset_digest=subset,
        interpretation_version=policy.interpretation_version,
        merge_version=_MERGE_VERSION,
        score_kinds_digest=kinds,
        fingerprint=fingerprint,
    )


def compile_guard(
    policies: Sequence[Policy],
    default_backend: Backend | None,
    *,
    max_chunks: int = 32,
) -> CompiledGuard:
    """Validate and freeze the static guard configuration."""

    names: set[str] = set()
    transforms: list[TransformPolicy] = []
    pending: list[tuple[JudgmentPolicy, Backend, Mapping[str, Question]]] = []
    digest_parts: list[str] = []

    for candidate in policies:
        validate_identifier(candidate.name, field="policy name")
        if candidate.name in names:
            raise PolicyError(f"duplicate policy name: {candidate.name}")
        names.add(candidate.name)
        for label in candidate.labels:
            validate_identifier(label, field="finding label")

        if not _policy_is_judgment(candidate):
            transform = cast(TransformPolicy, candidate)
            if transform.phase not in _PHASES:
                raise PolicyError(f"invalid transform phase: {transform.phase}")
            transforms.append(transform)
            digest_parts.append(
                f"t:{transform.name}:{transform.phase}:{transform.fingerprint or 'dynamic'}"
            )
            continue

        policy = cast(JudgmentPolicy, candidate)
        backend = policy.backend or default_backend
        if backend is None:
            raise PolicyError(f"judgment {policy.name} has no model")
        if backend.capabilities.max_attempts < 1:
            raise PolicyError("max_attempts must be at least one")
        if policy.context == "required" and policy.stages != frozenset({"output"}):
            raise PolicyError("required-context policies may run only on output")
        if policy.subject_mode == "items" and policy.whole_text:
            raise PolicyError("item policies cannot be whole_text")
        if policy.subject_mode == "text" and policy.max_policy_items is not None:
            raise PolicyError("text policies cannot set max_policy_items")
        if policy.max_policy_items is not None and policy.max_policy_items < 1:
            raise PolicyError("max_policy_items must be positive")

        question_map = dict(policy.questions(backend.capabilities.tasks))
        if not question_map:
            raise PolicyError(f"judgment {policy.name} has no questions")
        for question_id, question in question_map.items():
            validate_identifier(question_id, field="question id")
            if not backend.capabilities.supports_task(question.task):
                raise PolicyError(f"backend {backend.name} does not support task {question.task}")
            if not backend.capabilities.supported_kinds(question.task):
                raise PolicyError(
                    f"backend {backend.name} does not support score kinds for {question.task}"
                )
            option_count = (
                len(question.options)
                if isinstance(question, Choice)
                else len(question.levels)
                if isinstance(question, Score)
                else 2
            )
            maximum = backend.capabilities.max_options
            if maximum is not None and option_count > maximum:
                raise PolicyError(f"question {question_id} exceeds backend option limit")

        namespaced = {f"{policy.name}.{key}": value for key, value in question_map.items()}
        partitions = backend.partition_questions(namespaced)
        if len(partitions) != 1 or set(partitions[0]) != set(namespaced):
            raise PolicyError("a backend may not split one policy across partitions")
        for stage in policy.stages:
            room = backend.headroom(State(stage=stage, text=""), namespaced)
            if room is not None and room < 64:
                raise PolicyError(f"policy {policy.name} leaves fewer than 64 units")
        pending.append((policy, backend, question_map))
        digest_parts.append(
            f"j:{policy.name}:{policy.version}:{policy.interpretation_version}:"
            f"{backend.profile.model_identity}:{','.join(sorted(question_map))}"
        )

    config_digest = _digest(digest_parts)
    judgments: list[CompiledJudgment] = []
    for policy, backend, question_map in pending:
        thresholds: dict[Stage, Threshold] = {}
        sources: dict[Stage, Literal["explicit", "default"]] = {}
        runs: dict[Stage, str | None] = {}
        request_profiles: dict[Stage, RequestProfile] = {}
        decision_profiles: dict[Stage, DecisionProfile] = {}
        for stage in policy.stages:
            request = build_request_profile(
                backend=backend,
                transform_digest=config_digest,
                stage=stage,
                context_mode=policy.context,
                sources=policy.sources,
                subject_mode=policy.subject_mode,
                questions=question_map,
                max_chunks=max_chunks,
            )
            decision = build_decision_profile(
                request=request,
                stage=stage,
                policy=policy,
                questions=question_map,
                backend=backend,
            )
            request_profiles[stage] = request
            decision_profiles[stage] = decision
            if policy.threshold is not None:
                thresholds[stage] = policy.threshold
                sources[stage] = "explicit"
                runs[stage] = None
                continue
            looked_up = defaults.lookup(decision.fingerprint)
            if looked_up is None:
                raise PolicyError(f"judgment {policy.name} has no threshold for stage {stage}")
            thresholds[stage] = looked_up[0]
            sources[stage] = "default"
            runs[stage] = looked_up[1]
        judgments.append(
            CompiledJudgment(
                policy=policy,
                backend=backend,
                questions=question_map,
                thresholds=thresholds,
                threshold_sources=sources,
                evaluation_runs=runs,
                request_profiles=request_profiles,
                decision_profiles=decision_profiles,
            )
        )

    return CompiledGuard(
        transforms=tuple(transforms),
        judgments=tuple(judgments),
        config_digest=config_digest,
    )


def _map_span(mapping: Sequence[Span], span: Span, original_length: int) -> Span:
    if span.end > len(mapping):
        raise PolicyExecutionError("span is outside transform input")
    if span.start == span.end:
        if span.start < len(mapping):
            offset = mapping[span.start].start
        elif mapping:
            offset = mapping[-1].end
        else:
            offset = min(span.start, original_length)
        return Span(offset, offset)
    selected = mapping[span.start : span.end]
    return Span(min(value.start for value in selected), max(value.end for value in selected))


def _apply_mapping(
    text: str,
    mapping: Sequence[Span],
    edits: Sequence[TransformEdit],
    original_length: int,
) -> tuple[str, tuple[Span, ...]]:
    position = 0
    text_parts: list[str] = []
    mapped: list[Span] = []
    for edit in edits:
        text_parts.append(text[position : edit.start])
        mapped.extend(mapping[position : edit.start])
        source = _map_span(mapping, Span(edit.start, edit.end), original_length)
        text_parts.append(edit.replacement)
        mapped.extend(source for _ in edit.replacement)
        position = edit.end
    text_parts.append(text[position:])
    mapped.extend(mapping[position:])
    return "".join(text_parts), tuple(mapped)


def placeholder_edits(
    text: str,
    *,
    tracked: frozenset[str] = frozenset(),
) -> tuple[tuple[TransformEdit, ...], tuple[str, ...]]:
    edits: list[TransformEdit] = []
    preserved: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        token = match.group()
        if token in tracked:
            preserved.append(token)
            continue
        raw = token.encode("utf-8")
        replacement = f"⟦JES_LITERAL:{base64.urlsafe_b64encode(raw).decode('ascii')}⟧"
        edits.append(TransformEdit(match.start(), match.end(), replacement))
    return tuple(edits), tuple(preserved)


def neutralize_placeholders(
    text: str,
    mapping: Sequence[Span],
    *,
    original_length: int,
    tracked: frozenset[str] = frozenset(),
) -> tuple[str, tuple[Span, ...], tuple[Span, ...], tuple[str, ...]]:
    edits, preserved = placeholder_edits(text, tracked=tracked)
    spans = tuple(_map_span(mapping, Span(edit.start, edit.end), original_length) for edit in edits)
    if not edits:
        return text, tuple(mapping), (), preserved
    rewritten, rewritten_mapping = _apply_mapping(text, mapping, edits, original_length)
    return rewritten, rewritten_mapping, spans, preserved


def _location(
    *,
    target: Literal["subject", "prompt", "question", "source", "history"],
    span: Span,
    index: int | None,
    role: Role | None,
) -> FindingLocation:
    return FindingLocation(target=target, span=span, index=index, role=role)


def _effective_sensitive_edit(
    edit: _SensitiveEdit,
    *,
    allow_conversation_token: bool,
    allow_output_local: bool,
) -> _SensitiveEdit:
    mode = edit.mode
    if mode == "conversation_token" and not allow_conversation_token:
        mode = "irreversible"
    if mode == "output_local" and not allow_output_local:
        mode = "irreversible"
    if mode == edit.mode:
        return edit
    return _SensitiveEdit(span=edit.span, entity=edit.entity, mode=mode, action=edit.action)


def run_transforms(
    text: str,
    *,
    stage: Stage,
    origin_stage: Stage,
    target: Literal["subject", "context"],
    policies: Sequence[TransformPolicy],
    redactions: RedactionView | None,
    deadline: float | None,
    limits: GuardLimits,
    location_target: Literal["subject", "prompt", "question", "source", "history"],
    location_index: int | None = None,
    location_role: Role | None = None,
    run_limit_phase: bool = True,
    neutralize: bool = True,
    rewrite_placeholders: bool = True,
    tool: str | None = None,
    session: TransformSession | None = None,
) -> TransformResult:
    """Run phase-ordered transforms and map local spans to original text."""

    byte_limit = limits.max_input_bytes if target == "subject" else limits.max_context_bytes
    utf8_size(text, limit=byte_limit)
    original_length = len(text)
    mapping: tuple[Span, ...] = tuple(Span(index, index + 1) for index in range(len(text)))
    findings: list[Finding] = []
    tracked: set[str] = set() if session is None else set(session.tracked_tokens)
    finalizers = FinalizerMap() if session is None else session.finalizers

    def record_placeholders(label: str, spans: Sequence[Span]) -> None:
        if not spans:
            return
        findings.append(
            Finding(
                policy="jes",
                label=label,
                action="flag",
                locations=tuple(
                    _location(
                        target=location_target,
                        span=span,
                        index=location_index,
                        role=location_role,
                    )
                    for span in spans
                ),
            )
        )

    if neutralize and rewrite_placeholders:
        text, mapping, neutralized, _preserved = neutralize_placeholders(
            text,
            mapping,
            original_length=original_length,
        )
        record_placeholders("placeholder_in_input", neutralized)
    utf8_size(
        text,
        limit=limits.max_normalized_bytes
        if target == "subject"
        else limits.max_normalized_context_bytes,
    )

    allow_conversation = bool(
        session is not None and session.allow_conversation_token and session.transaction is not None
    )
    allow_output_local = bool(session is not None and session.allow_output_local)

    def post_scan(*, label: str) -> None:
        nonlocal text, mapping
        if not rewrite_placeholders:
            return
        edits, _preserved = placeholder_edits(text, tracked=frozenset(tracked))
        if not edits:
            return
        finalizers.compose(tuple((edit.start, edit.end, edit.replacement) for edit in edits))
        spans = tuple(
            _map_span(mapping, Span(edit.start, edit.end), original_length) for edit in edits
        )
        text, mapping = _apply_mapping(text, mapping, edits, original_length)
        record_placeholders(label, spans)

    phases = _PHASES if run_limit_phase else _PHASES[:2]
    for phase in phases:
        for policy in policies:
            if policy.phase != phase or origin_stage not in policy.stages:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise DeadlineExceeded("transform")
            call = CallContext(
                call_stage=stage,
                origin_stage=origin_stage,
                target=target,
                redactions=redactions,
                deadline=deadline,
                tool=tool,
            )
            before_mapping = mapping
            before_text = text
            failure: PolicyExecutionError | None = None
            sensitive_outcome: _SensitiveOutcome | None = None
            public_outcome: TransformOutcome | None = None
            if is_sensitive(policy):
                transaction = None if session is None else session.transaction
                if transaction is None:
                    raise PolicyExecutionError("sensitive transform requires a transaction")
                sensitive_policy = cast(_SensitiveTransform, policy)
                try:
                    sensitive_outcome = sensitive_policy._apply_sensitive(
                        before_text,
                        call,
                        transaction,
                    )
                except PolicyExecutionError:
                    raise
                except Exception:
                    failure = PolicyExecutionError("custom transform raised")
                if failure is not None:
                    raise failure
            else:
                try:
                    public_outcome = policy.apply(before_text, call)
                except PolicyExecutionError:
                    raise
                except Exception:
                    failure = PolicyExecutionError("custom transform raised")
                if failure is not None:
                    raise failure
                if public_outcome is None:
                    raise PolicyExecutionError("transform returned no outcome")
                validate_transform_outcome(before_text, public_outcome, labels=policy.labels)

            if sensitive_outcome is not None:
                effective = tuple(
                    _effective_sensitive_edit(
                        edit,
                        allow_conversation_token=allow_conversation,
                        allow_output_local=allow_output_local,
                    )
                    for edit in sensitive_outcome.edits
                )
                rewritten, public_edits, issued = apply_sensitive_edits(
                    before_text,
                    effective,
                    transaction=None if session is None else session.transaction,
                    hmac_key=sensitive_key(policy),
                    allow_conversation_token=allow_conversation,
                    finalizers=finalizers,
                )
                tracked.update(issued)
                text, mapping = _apply_mapping(
                    before_text,
                    before_mapping,
                    tuple(
                        TransformEdit(start, end, replacement)
                        for start, end, replacement in public_edits
                    ),
                    original_length,
                )
                if text != rewritten:
                    raise PolicyExecutionError("mapped sensitive text differs from engine rewrite")
                outcome_findings = sensitive_outcome.findings
            else:
                assert public_outcome is not None
                text, mapping = _apply_mapping(
                    before_text,
                    before_mapping,
                    public_outcome.edits,
                    original_length,
                )
                if text != public_outcome.text:
                    raise PolicyExecutionError("mapped transform text differs from outcome")
                outcome_findings = public_outcome.findings
                finalizers.compose(
                    tuple((edit.start, edit.end, edit.replacement) for edit in public_outcome.edits)
                )
            for finding in outcome_findings:
                findings.append(
                    Finding(
                        policy=policy.name,
                        label=finding.label,
                        action=finding.action,
                        locations=tuple(
                            _location(
                                target=location_target,
                                span=_map_span(before_mapping, span, original_length),
                                index=location_index,
                                role=location_role,
                            )
                            for span in finding.spans
                        ),
                    )
                )
            post_scan(label="invalid_placeholder")
            utf8_size(
                text,
                limit=limits.max_normalized_bytes
                if target == "subject"
                else limits.max_normalized_context_bytes,
            )
            if deadline is not None and time.monotonic() >= deadline:
                raise DeadlineExceeded("transform")

    if rewrite_placeholders and not neutralize:
        post_scan(label="invalid_placeholder")

    collected: list[_TokenAuthority] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        token = match.group()
        if token not in tracked:
            continue
        collected.append(
            _TokenAuthority(
                token=token,
                span=_map_span(mapping, Span(match.start(), match.end()), original_length),
                origin_result="",
                reusable=True,
            )
        )
    if session is not None:
        session.tracked_tokens.update(tracked)
        session.authorities[:] = collected
    return TransformResult(
        text=text,
        mapping=mapping,
        findings=tuple(findings),
        authorities=tuple(collected),
        restored_text=None,
    )


def original_span(mapping: Sequence[Span], span: Span, original_length: int) -> Span:
    return _map_span(mapping, span, original_length)


def group_key(compiled: CompiledJudgment) -> tuple[object, ...]:
    policy = compiled.policy
    return (
        id(compiled.backend),
        policy.context,
        policy.sources,
        policy.subject_mode,
        policy.whole_text,
        policy.on_context_overflow,
        policy.on_text_overflow,
    )


def group_judgments(
    judgments: Iterable[CompiledJudgment],
) -> tuple[tuple[CompiledJudgment, ...], ...]:
    groups: dict[tuple[object, ...], list[CompiledJudgment]] = {}
    order: list[tuple[object, ...]] = []
    for compiled in judgments:
        key = group_key(compiled)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(compiled)
    return tuple(tuple(groups[key]) for key in order)


def chunk_text(
    *,
    backend: Backend,
    base_state: State,
    questions: Mapping[str, Question],
    text: str,
    max_chunks: int,
    whole_text: bool,
) -> tuple[tuple[str, Span], ...]:
    """Create covering chunks whose complete rendered requests fit."""

    if backend.headroom(replace(base_state, text=text), questions) is None:
        return ((text, Span(0, len(text))),)
    whole_room = backend.headroom(replace(base_state, text=text), questions)
    if whole_room is not None and whole_room >= 0:
        return ((text, Span(0, len(text))),)
    if whole_text:
        raise ResourceLimit("text_too_long")

    chunks: list[tuple[str, Span]] = []
    start = 0
    while start < len(text):
        low = start + 1
        high = len(text)
        best: int | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = text[start:middle]
            room = backend.headroom(replace(base_state, text=candidate), questions)
            if room is None or room >= 0:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ResourceLimit("input_too_long")

        end = best
        if end < len(text):
            whitespace = max(text.rfind(" ", start + 1, end), text.rfind("\n", start + 1, end))
            if whitespace > start:
                end = whitespace + 1
        if end <= start:
            raise ResourceLimit("input_too_long")
        chunks.append((text[start:end], Span(start, end)))
        if len(chunks) > max_chunks:
            raise ResourceLimit("input_too_long")
        if end == len(text):
            break

        chunk_units = max(1, backend.count_units(text[start:end]))
        overlap_units = min(32, chunk_units // 4)
        next_start = end
        while (
            next_start > start + 1
            and backend.count_units(text[next_start - 1 : end]) < overlap_units
        ):
            next_start -= 1
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return tuple(chunks)
