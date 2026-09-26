"""Public policy contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from jes.backends import Backend
from jes.errors import PolicyExecutionError
from jes.questions import Answer, Question, Threshold
from jes.redactions import RedactionView
from jes.types import Action, Finding, FindingLocation, Provenance, ScoreResult, Span, Stage

Phase: TypeAlias = Literal["normalize", "detect", "limit"]
ContextTarget: TypeAlias = Literal["subject", "context"]
ContextMode: TypeAlias = Literal["none", "optional", "required"]
OverflowMode: TypeAlias = Literal["block", "allow"]
SubjectMode: TypeAlias = Literal["text", "items"]


@dataclass(frozen=True, slots=True)
class CallContext:
    call_stage: Stage
    origin_stage: Stage
    target: ContextTarget
    redactions: RedactionView | None
    deadline: float | None


@dataclass(frozen=True, slots=True, repr=False)
class TransformEdit:
    start: int
    end: int
    replacement: str

    def __repr__(self) -> str:
        return (
            f"TransformEdit(start={self.start}, end={self.end}, "
            f"replacement_len={len(self.replacement)})"
        )


@dataclass(frozen=True, slots=True)
class TransformFinding:
    label: str
    action: Action
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class TransformOutcome:
    text: str
    findings: tuple[TransformFinding, ...] = ()
    edits: tuple[TransformEdit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "edits", tuple(self.edits))

    def __repr__(self) -> str:
        return (
            f"TransformOutcome(text_len={len(self.text)}, findings={len(self.findings)}, "
            f"edits={len(self.edits)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class Item:
    text: str
    span: Span

    def __repr__(self) -> str:
        return f"Item(text_len={len(self.text)}, span={self.span!r})"


@dataclass(frozen=True, slots=True)
class InterpretationContext:
    provenance: Provenance
    chunk: int
    item_ordinal: int | None
    location: FindingLocation


@dataclass(frozen=True, slots=True)
class JudgmentOutcome:
    findings: tuple[Finding, ...]
    scores: Mapping[str, ScoreResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@runtime_checkable
class TransformPolicy(Protocol):
    name: str
    labels: frozenset[str]
    stages: frozenset[Stage]
    phase: Phase
    fingerprint: str | None

    def apply(self, text: str, call: CallContext) -> TransformOutcome: ...


@runtime_checkable
class JudgmentPolicy(Protocol):
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
    threshold: Threshold | None
    version: str
    interpretation_version: str

    def questions(self, tasks: frozenset[str] | None) -> Mapping[str, Question]: ...

    def items(self, text: str) -> Iterable[Item]: ...

    def interpret(
        self,
        answers: Mapping[str, Answer],
        threshold: Threshold,
        context: InterpretationContext,
    ) -> JudgmentOutcome: ...


Policy: TypeAlias = TransformPolicy | JudgmentPolicy


def apply_transform_edits(text: str, edits: Iterable[TransformEdit]) -> str:
    """Apply sorted, non-overlapping edits to text."""

    ordered = tuple(edits)
    position = 0
    parts: list[str] = []
    for edit in ordered:
        if edit.start < position or edit.end < edit.start or edit.end > len(text):
            raise PolicyExecutionError("transform edits must be sorted and non-overlapping")
        parts.append(text[position : edit.start])
        parts.append(edit.replacement)
        position = edit.end
    parts.append(text[position:])
    return "".join(parts)


def validate_transform_outcome(
    original: str,
    outcome: TransformOutcome,
    *,
    labels: frozenset[str],
) -> None:
    """Validate a custom transform before its output reaches another component."""

    if apply_transform_edits(original, outcome.edits) != outcome.text:
        raise PolicyExecutionError("transform edits do not reproduce outcome text")
    for finding in outcome.findings:
        if finding.label not in labels:
            raise PolicyExecutionError("transform emitted an undeclared label")
        for span in finding.spans:
            if span.end > len(original):
                raise PolicyExecutionError("transform finding span is out of range")


__all__ = [
    "CallContext",
    "ContextMode",
    "ContextTarget",
    "InterpretationContext",
    "Item",
    "JudgmentOutcome",
    "JudgmentPolicy",
    "OverflowMode",
    "Phase",
    "Policy",
    "SubjectMode",
    "TransformEdit",
    "TransformFinding",
    "TransformOutcome",
    "TransformPolicy",
]
