"""Judgment policy helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from jes.errors import BackendError, PolicyError
from jes.judge import Backend, ModelSpec, resolve_judge
from jes.policies._candidates import (
    _Candidate,
    hazards_candidate,
    indirect_injection_candidate,
    injection_candidate,
    tool_safety_candidate,
    topics_candidate,
    toxicity_candidate,
)
from jes.policies._protocols import (
    ContextMode,
    InterpretationContext,
    Item,
    JudgmentOutcome,
    OverflowMode,
    SubjectMode,
)
from jes.questions import (
    Answer,
    Choice,
    Question,
    Score,
    Threshold,
    YesNo,
    validate_answer,
    validate_identifier,
    violation_score,
)
from jes.types import Action, Finding, ScoreResult, Stage

ItemExtractor = Callable[[str], Iterable[Item]]


def _freeze_questions(
    questions: Question | Mapping[str, Question],
) -> Mapping[str, Question]:
    if isinstance(questions, (YesNo, Choice, Score)):
        values: Mapping[str, Question] = {"violation": questions}
    else:
        values = questions
    if not values:
        raise PolicyError("judge requires at least one question")
    copied = dict(values)
    for question_id in copied:
        validate_identifier(question_id, field="question id")
    return MappingProxyType(copied)


def _normalize_violating(
    questions: Mapping[str, Question],
    violating: Collection[str] | Mapping[str, Collection[str]] | None,
) -> Mapping[str, frozenset[str]]:
    choice_ids = {key for key, question in questions.items() if isinstance(question, Choice)}
    if not choice_ids:
        if violating is not None:
            raise PolicyError("violating is valid only for choice questions")
        return MappingProxyType({})
    if violating is None:
        raise PolicyError("choice questions require violating labels")
    if isinstance(violating, Mapping):
        violating_map = cast(Mapping[str, Collection[str]], violating)
        normalized = {key: frozenset(value) for key, value in violating_map.items()}
    else:
        if isinstance(violating, str) or len(choice_ids) != 1:
            raise PolicyError("one violating collection is valid only for one choice question")
        normalized = {next(iter(choice_ids)): frozenset(violating)}
    if set(normalized) != choice_ids:
        raise PolicyError("violating must cover every choice question exactly")
    for key, labels in normalized.items():
        question = cast(Choice, questions[key])
        if not labels or not labels <= set(question.options):
            raise PolicyError("violating labels must be non-empty choice options")
    return MappingProxyType(normalized)


def _normalize_levels(
    questions: Mapping[str, Question],
    violation_level: int | Mapping[str, int] | None,
) -> Mapping[str, int]:
    score_ids = {key for key, question in questions.items() if isinstance(question, Score)}
    if not score_ids:
        if violation_level is not None:
            raise PolicyError("violation_level is valid only for score questions")
        return MappingProxyType({})
    if violation_level is None:
        raise PolicyError("score questions require violation_level")
    if isinstance(violation_level, Mapping):
        normalized = dict(violation_level)
    else:
        if len(score_ids) != 1:
            raise PolicyError("one violation_level is valid only for one score question")
        normalized = {next(iter(score_ids)): violation_level}
    if set(normalized) != score_ids:
        raise PolicyError("violation_level must cover every score question exactly")
    for key, level in normalized.items():
        question = cast(Score, questions[key])
        if isinstance(level, bool) or not 0 <= level < len(question.levels):
            raise PolicyError("violation_level is out of range")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class _JudgePolicy:
    kind: str
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    subject_mode: SubjectMode
    max_policy_items: int | None
    item_overflow_label: str
    on_items_overflow: OverflowMode
    context: ContextMode
    on_context_overflow: OverflowMode
    sources: bool
    whole_text: bool
    on_text_overflow: OverflowMode
    backend: Backend | None
    threshold: Threshold
    version: str
    interpretation_version: str
    fingerprint: str
    _questions: Mapping[str, Question]
    _violating: Mapping[str, frozenset[str]]
    _violation_levels: Mapping[str, int]
    _extractor: ItemExtractor | None

    def questions(self, tasks: frozenset[str] | None) -> Mapping[str, Question]:
        if tasks is not None and "custom" not in tasks:
            raise PolicyError("backend does not support custom questions")
        return self._questions

    def items(self, text: str) -> Iterable[Item]:
        if self._extractor is None:
            return ()
        return self._extractor(text)

    def interpret(
        self,
        answers: Mapping[str, Answer],
        threshold: Threshold,
        context: InterpretationContext,
    ) -> JudgmentOutcome:
        if set(answers) != set(self._questions):
            raise BackendError(
                context.provenance.backend,
                "answer ids do not match question ids",
                question_ids=self._questions,
            )

        findings: list[Finding] = []
        scores: dict[str, ScoreResult] = {}
        for question_id, question in self._questions.items():
            answer = answers[question_id]
            validate_answer(question, answer)
            value = violation_score(
                question,
                answer,
                violating=self._violating.get(question_id),
                violation_level=self._violation_levels.get(question_id),
            )
            score = ScoreResult(
                value=value,
                kind=answer.kind,
                confidence=answer.confidence,
                provenance=context.provenance,
            )
            scores[question_id] = score
            action: Action | None = None
            if value >= threshold.block_at:
                action = "block"
            elif threshold.flag_at is not None and value >= threshold.flag_at:
                action = "flag"
            if action is not None:
                findings.append(
                    Finding(
                        policy=self.name,
                        label=question_id,
                        action=action,
                        question=question_id,
                        score=score,
                        locations=(context.location,),
                        chunks=(context.chunk,),
                    )
                )
        return JudgmentOutcome(findings=tuple(findings), scores=scores)


def judge(
    name: str,
    questions: Question | Mapping[str, Question],
    *,
    threshold: Threshold | float,
    violating: Collection[str] | Mapping[str, Collection[str]] | None = None,
    violation_level: int | Mapping[str, int] | None = None,
    stages: Iterable[Stage] = ("input", "output"),
    context: ContextMode = "none",
    on_context_overflow: OverflowMode = "block",
    sources: bool = False,
    whole_text: bool = False,
    on_text_overflow: OverflowMode = "block",
    items: ItemExtractor | None = None,
    max_policy_items: int | None = None,
    item_overflow_label: str = "too_many_items",
    on_items_overflow: OverflowMode = "block",
    model: ModelSpec | None = None,
    version: str = "v1",
    interpretation_version: str = "v1",
) -> _JudgePolicy:
    """Create a custom judgment policy."""

    validate_identifier(name, field="policy name")
    validate_identifier(item_overflow_label, field="item overflow label")
    question_map = _freeze_questions(questions)
    stage_set = frozenset(stages)
    if not stage_set:
        raise PolicyError("policy stages must not be empty")
    if context == "required" and stage_set != frozenset({"output"}):
        raise PolicyError("required context policies may run only on output")
    if items is None:
        if max_policy_items is not None:
            raise PolicyError("text mode cannot set max_policy_items")
        subject_mode = "text"
    else:
        subject_mode = "items"
        if whole_text:
            raise PolicyError("item mode cannot be whole_text")
        if max_policy_items is not None and max_policy_items < 1:
            raise PolicyError("max_policy_items must be positive")

    violating_map = _normalize_violating(question_map, violating)
    level_map = _normalize_levels(question_map, violation_level)
    threshold_value = Threshold.coerce(threshold)
    labels = frozenset({*question_map, item_overflow_label})
    digest_source = "|".join(
        [
            name,
            version,
            interpretation_version,
            ",".join(sorted(question_map)),
            ",".join(sorted(stage_set)),
            subject_mode,
            context,
        ]
    )
    fingerprint = hashlib.sha256(digest_source.encode()).hexdigest()
    return _JudgePolicy(
        kind=name,
        name=name,
        labels=labels,
        stages=stage_set,
        subject_mode=subject_mode,
        max_policy_items=max_policy_items,
        item_overflow_label=item_overflow_label,
        on_items_overflow=on_items_overflow,
        context=context,
        on_context_overflow=on_context_overflow,
        sources=sources,
        whole_text=whole_text,
        on_text_overflow=on_text_overflow,
        backend=resolve_judge(model),
        threshold=threshold_value,
        version=version,
        interpretation_version=interpretation_version,
        fingerprint=fingerprint,
        _questions=question_map,
        _violating=violating_map,
        _violation_levels=level_map,
        _extractor=items,
    )


def _checked_version(version: str, canonical: str) -> None:
    if version not in {"v1", canonical}:
        raise PolicyError(f"unknown policy version {version}")


def _applied_threshold(value: float | Threshold | None) -> Threshold | None:
    if value is None:
        return None
    return Threshold.coerce(value)


def _publish(
    policy: _Candidate,
    *,
    threshold: float | Threshold | None,
    model: ModelSpec | None,
) -> _Candidate:
    return replace(policy, threshold=_applied_threshold(threshold), backend=resolve_judge(model))


def injection(
    *,
    threshold: float | Threshold | None = None,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen injection.v1 question. Omit threshold only for an audited profile."""

    _checked_version(version, "injection.v1")
    return _publish(injection_candidate(), threshold=threshold, model=model)


def indirect_injection(
    *,
    threshold: float | Threshold | None = None,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen indirect_injection.v1 question. Omit threshold only for an audited profile."""

    _checked_version(version, "indirect_injection.v1")
    return _publish(indirect_injection_candidate(), threshold=threshold, model=model)


def hazards(
    categories: Iterable[str] | None = None,
    *,
    threshold: float | Threshold | None = None,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen hazards.v1 questions. A category subset is a different decision profile."""

    _checked_version(version, "hazards.v1")
    return _publish(hazards_candidate(categories), threshold=threshold, model=model)


def tool_safety(
    *,
    threshold: float | Threshold,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen tool_safety.v1 question. The threshold is required until it is measured."""

    _checked_version(version, "tool_safety.v1")
    return _publish(tool_safety_candidate(), threshold=threshold, model=model)


def topics(
    deny: Iterable[str],
    *,
    threshold: float | Threshold,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen topics.v1 template. Caller-defined topics always require a threshold."""

    _checked_version(version, "topics.v1")
    return _publish(topics_candidate(deny, threshold=1.0), threshold=threshold, model=model)


def toxicity(
    labels: Iterable[str] | None = None,
    *,
    threshold: float | Threshold | None = None,
    version: str = "v1",
    model: ModelSpec | None = None,
) -> _Candidate:
    """Frozen toxicity.v1 questions. A label subset is a different decision profile."""

    _checked_version(version, "toxicity.v1")
    return _publish(toxicity_candidate(labels), threshold=threshold, model=model)


__all__ = [
    "hazards",
    "indirect_injection",
    "injection",
    "judge",
    "tool_safety",
    "topics",
    "toxicity",
]
