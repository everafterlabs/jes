"""Private evaluation candidates. Not part of the public policy API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from jes.errors import BackendError, PolicyError
from jes.judge import Backend
from jes.policies._protocols import (
    ContextMode,
    InterpretationContext,
    Item,
    JudgmentOutcome,
    OverflowMode,
    SubjectMode,
)
from jes.policies.prompts import (
    HAZARD_ANY_V1,
    HAZARD_CODES,
    INDIRECT_INJECTION_V1,
    INJECTION_V1,
    TOXICITY_LABELS,
    hazard_instruction,
    topic_instruction,
    toxicity_instruction,
)
from jes.questions import Answer, Question, Threshold, YesNo, validate_answer, violation_score
from jes.types import Action, Finding, ScoreResult, Stage

_OVERFLOW = "too_many_items"


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    context: ContextMode
    version: str
    interpretation_version: str
    threshold: Threshold | None
    _questions: Mapping[str, Question]
    _summary: str | None = None
    subject_mode: SubjectMode = "text"
    max_policy_items: int | None = None
    item_overflow_label: str = _OVERFLOW
    on_items_overflow: OverflowMode = "block"
    on_context_overflow: OverflowMode = "block"
    sources: bool = False
    whole_text: bool = False
    on_text_overflow: OverflowMode = "block"
    backend: Backend | None = None

    def questions(self, tasks: frozenset[str] | None) -> Mapping[str, Question]:
        selected: dict[str, Question] = {}
        for key, question in self._questions.items():
            if key == "any" and tasks is not None and "hazard.any" not in tasks:
                continue
            if tasks is not None and not _task_supported(tasks, question.task):
                raise PolicyError(f"backend does not support task {question.task}")
            selected[key] = question
        if not selected:
            raise PolicyError(f"{self.name} has no questions for this backend")
        return selected

    def items(self, text: str) -> Iterable[Item]:
        del text
        return ()

    def interpret(
        self,
        answers: Mapping[str, Answer],
        threshold: Threshold,
        context: InterpretationContext,
    ) -> JudgmentOutcome:
        unknown = set(answers) - set(self._questions)
        if unknown:
            raise BackendError(
                context.provenance.backend,
                "answer ids do not match question ids",
                question_ids=unknown,
            )
        scores: dict[str, ScoreResult] = {}
        for question_id, question in self._questions.items():
            if question_id not in answers:
                continue
            answer = answers[question_id]
            validate_answer(question, answer)
            value = violation_score(question, answer)
            scores[question_id] = ScoreResult(
                value=value,
                kind=answer.kind,
                confidence=answer.confidence,
                provenance=context.provenance,
            )
        summary_id = self._summary if self._summary in scores else None
        summary = (
            scores[summary_id].value
            if summary_id is not None
            else max(score.value for score in scores.values())
        )
        action = _action(summary, threshold)
        findings: list[Finding] = []
        if action is not None:
            named = [
                question_id
                for question_id, score in scores.items()
                if question_id != "any" and score.value >= _cutoff(action, threshold)
            ]
            labels = tuple(named) if named else (("unattributed",) if self._summary else ())
            if not labels:
                labels = tuple(scores)
            for label in labels:
                findings.append(
                    Finding(
                        policy=self.name,
                        label=label,
                        action=action,
                        question=None if label == "unattributed" else label,
                        score=scores.get(label, scores.get(summary_id or label)),
                        locations=(context.location,),
                        chunks=(context.chunk,),
                    )
                )
        return JudgmentOutcome(findings=tuple(findings), scores=scores)


def _task_supported(tasks: frozenset[str], task: str) -> bool:
    return task in tasks


def _cutoff(action: Action, threshold: Threshold) -> float:
    if action == "flag" and threshold.flag_at is not None:
        return threshold.flag_at
    return threshold.block_at


def _action(summary: float, threshold: Threshold) -> Action | None:
    if summary >= threshold.block_at:
        return "block"
    if threshold.flag_at is not None and summary >= threshold.flag_at:
        return "flag"
    return None


def _policy(
    name: str,
    questions: Mapping[str, Question],
    *,
    stages: tuple[Stage, ...],
    context: ContextMode,
    version: str,
    threshold: float | None,
    summary: str | None,
    extra_labels: tuple[str, ...] = (),
) -> _Candidate:
    labels = frozenset({*questions, _OVERFLOW, *extra_labels})
    return _Candidate(
        kind=name,
        name=name,
        labels=labels,
        stages=frozenset(stages),
        context=context,
        version=version,
        interpretation_version="v1",
        threshold=None if threshold is None else Threshold(block_at=threshold),
        _questions=dict(questions),
        _summary=summary,
    )


def injection_candidate(*, threshold: float | None = None) -> _Candidate:
    question = YesNo(INJECTION_V1, task="injection")
    return _policy(
        "injection",
        {"violation": question},
        stages=("input", "untrusted", "tool_call"),
        context="none",
        version="injection.v1",
        threshold=threshold,
        summary=None,
    )


def indirect_injection_candidate(*, threshold: float | None = None) -> _Candidate:
    question = YesNo(INDIRECT_INJECTION_V1, task="indirect_injection")
    return _policy(
        "indirect_injection",
        {"violation": question},
        stages=("untrusted", "tool_result"),
        context="optional",
        version="indirect_injection.v1",
        threshold=threshold,
        summary=None,
    )


def hazards_candidate(
    categories: Iterable[str] | None = None,
    *,
    threshold: float | None = None,
) -> _Candidate:
    codes = tuple(categories) if categories is not None else HAZARD_CODES
    if not codes or any(code not in HAZARD_CODES for code in codes):
        raise PolicyError("unknown hazard category")
    questions: dict[str, Question] = {
        code: YesNo(hazard_instruction(code), task=f"hazard.{code}") for code in codes
    }
    questions["any"] = YesNo(HAZARD_ANY_V1, task="hazard.any")
    return _policy(
        "hazards",
        questions,
        stages=("input", "output"),
        context="optional",
        version="hazards.v1",
        threshold=threshold,
        summary="any",
        extra_labels=("unattributed",),
    )


def topics_candidate(
    deny: Iterable[str],
    *,
    threshold: float,
) -> _Candidate:
    topics = tuple(deny)
    if not topics:
        raise PolicyError("topics requires at least one topic")
    questions = {
        f"topic_{index}": YesNo(topic_instruction(topic), task="topic")
        for index, topic in enumerate(topics)
    }
    return _policy(
        "topics",
        questions,
        stages=("input", "output"),
        context="none",
        version="topics.v1",
        threshold=threshold,
        summary=None,
    )


def toxicity_candidate(
    labels: Iterable[str] | None = None,
    *,
    threshold: float | None = None,
) -> _Candidate:
    selected = tuple(labels) if labels is not None else TOXICITY_LABELS
    if not selected or any(label not in TOXICITY_LABELS for label in selected):
        raise PolicyError("unknown toxicity label")
    questions = {
        label: YesNo(toxicity_instruction(label), task=f"toxicity.{label}") for label in selected
    }
    return _policy(
        "toxicity",
        questions,
        stages=("input", "output"),
        context="none",
        version="toxicity.v1",
        threshold=threshold,
        summary=None,
    )


__all__ = [
    "hazards_candidate",
    "indirect_injection_candidate",
    "injection_candidate",
    "topics_candidate",
    "toxicity_candidate",
]
