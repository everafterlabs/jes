"""Shared test doubles for Milestone 1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from jes.judge import Backend
from jes.policies import (
    CallContext,
    ContextMode,
    InterpretationContext,
    Item,
    JudgmentOutcome,
    OverflowMode,
    Phase,
    SubjectMode,
    TransformEdit,
    TransformFinding,
    TransformOutcome,
)
from jes.questions import Answer, Question, Threshold, YesNo
from jes.types import Action, Finding, ScoreResult, Span, Stage


@dataclass
class FakeTransform:
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage] = frozenset({"input", "untrusted", "output"})
    phase: Phase = "detect"
    fingerprint: str | None = "fake-transform"
    handler: Callable[[str, CallContext], TransformOutcome] | None = None
    calls: list[str] = field(default_factory=list)

    def apply(self, text: str, call: CallContext) -> TransformOutcome:
        self.calls.append(text)
        if self.handler is not None:
            return self.handler(text, call)
        return TransformOutcome(text=text)


def rewrite(
    old: str,
    new: str,
    *,
    label: str = "rewrite",
    action: Action = "redact",
) -> FakeTransform:
    def handler(text: str, call: CallContext) -> TransformOutcome:
        del call
        start = text.find(old)
        if start < 0:
            return TransformOutcome(text=text)
        end = start + len(old)
        return TransformOutcome(
            text=text[:start] + new + text[end:],
            findings=(TransformFinding(label, action, (Span(start, end),)),),
            edits=(TransformEdit(start, end, new),),
        )

    return FakeTransform(
        name=f"rewrite_{label}",
        labels=frozenset({label}),
        handler=handler,
    )


@dataclass
class CountingPolicy:
    name: str
    questions_map: Mapping[str, Question]
    labels: frozenset[str]
    stages: frozenset[Stage] = frozenset({"input", "output"})
    kind: str = "custom"
    subject_mode: SubjectMode = "text"
    max_policy_items: int | None = None
    item_overflow_label: str = "too_many_items"
    on_items_overflow: OverflowMode = "block"
    context: ContextMode = "none"
    on_context_overflow: OverflowMode = "block"
    sources: bool = False
    whole_text: bool = False
    on_text_overflow: OverflowMode = "block"
    backend: Backend | None = None
    threshold: Threshold | None = None
    version: str = "v1"
    interpretation_version: str = "v1"
    question_calls: int = 0
    extractor: Callable[[str], Iterable[Item]] | None = None

    def questions(self, tasks: frozenset[str] | None) -> Mapping[str, Question]:
        del tasks
        self.question_calls += 1
        return self.questions_map

    def items(self, text: str) -> Iterable[Item]:
        if self.extractor is None:
            return ()
        return self.extractor(text)

    def interpret(
        self,
        answers: Mapping[str, Answer],
        threshold: Threshold,
        context: InterpretationContext,
    ) -> JudgmentOutcome:
        from jes.questions import violation_score

        findings: list[Finding] = []
        scores: dict[str, ScoreResult] = {}
        for question_id, question in self.questions_map.items():
            answer = answers[question_id]
            value = violation_score(question, answer)
            score = ScoreResult(
                value=value,
                kind=answer.kind,
                confidence=answer.confidence,
                provenance=context.provenance,
            )
            scores[question_id] = score
            if value >= threshold.block_at:
                findings.append(
                    Finding(
                        policy=self.name,
                        label=question_id,
                        action="block",
                        question=question_id,
                        score=score,
                        locations=(context.location,),
                        chunks=(context.chunk,),
                    )
                )
        return JudgmentOutcome(tuple(findings), scores)


def yesno_policy(
    name: str = "check",
    *,
    threshold: float | None = 0.8,
    stages: frozenset[Stage] = frozenset({"input", "output"}),
    **kwargs: object,
) -> CountingPolicy:
    question = YesNo("The text violates the policy.")
    return CountingPolicy(
        name=name,
        questions_map={"violation": question},
        labels=frozenset({"violation", "too_many_items"}),
        stages=stages,
        threshold=None if threshold is None else Threshold(block_at=threshold),
        **kwargs,  # type: ignore[arg-type]
    )
