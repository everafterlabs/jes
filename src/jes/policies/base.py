"""Policy building blocks.

A transform edits or inspects text locally, before any backend sees it. A judgment asks a
backend questions and compares the answers with a threshold. ``judge`` builds judgments,
and every built-in judgment is one ``judge`` call.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal, Protocol, TypeAlias, cast, runtime_checkable

from jes.backend import AsyncBackend, ModelSpec, SyncBackend, resolve_model
from jes.errors import DeadlineExceeded, PolicyError
from jes.questions import (
    Answer,
    Choice,
    Question,
    Score,
    Threshold,
    YesNo,
    violation_score,
)
from jes.text.textmap import Edit
from jes.types import STAGES, Action, Span, Stage, Target, validate_identifier

Phase: TypeAlias = Literal["normalize", "detect", "limit"]
ContextMode: TypeAlias = Literal["none", "optional", "required"]
OverflowMode: TypeAlias = Literal["block", "allow"]

PHASES: tuple[Phase, ...] = ("normalize", "detect", "limit")
_OUTPUT_ONLY: frozenset[Stage] = frozenset({"output"})


def freeze_stages(stages: Iterable[Stage]) -> frozenset[Stage]:
    """Validated, non-empty stages."""

    frozen = frozenset(stages)
    if not frozen:
        raise PolicyError("a policy needs at least one stage")
    unknown = frozen - set(STAGES)
    if unknown:
        raise PolicyError(f"unknown stage: {sorted(unknown)[0]}")
    return frozen


@dataclass(frozen=True, slots=True)
class TransformContext:
    """What a transform knows about the text in front of it.

    ``stage`` is the check's stage. ``origin`` is where this text came from: the check's
    own stage for the checked text, or the stage a context value came from, such as
    ``"input"`` for a prompt. ``target`` says which of the two it is.
    """

    stage: Stage
    origin: Stage
    target: Target
    tool: str | None = None
    deadline: float | None = None

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise DeadlineExceeded("transform")


@dataclass(frozen=True, slots=True)
class TransformFinding:
    """A finding from a transform. Spans index into the text the transform was given."""

    label: str
    action: Action
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class TransformOutcome:
    """Edits to apply, sorted and not overlapping, and findings. Both may be empty."""

    edits: tuple[Edit, ...] = ()
    findings: tuple[TransformFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edits", tuple(self.edits))
        object.__setattr__(self, "findings", tuple(self.findings))

    def __repr__(self) -> str:
        return f"TransformOutcome(edits={len(self.edits)}, findings={len(self.findings)})"


@runtime_checkable
class Transform(Protocol):
    """A local policy. It runs in its phase, in list order, on text from its stages."""

    name: str
    stages: frozenset[Stage]
    phase: Phase

    def apply(self, text: str, context: TransformContext) -> TransformOutcome: ...


@dataclass(frozen=True, slots=True, repr=False)
class Item:
    """One piece of a text that an item judgment judges on its own, such as a URL."""

    text: str
    span: Span

    def __repr__(self) -> str:
        return f"Item(text_len={len(self.text)}, span={self.span!r})"


ItemExtractor: TypeAlias = Callable[[str], Iterable[Item]]


@dataclass(frozen=True, slots=True)
class Judgment:
    """Questions for a backend, plus how to read the answers. Build one with ``judge``."""

    name: str
    questions: Mapping[str, Question]
    threshold: Threshold
    stages: frozenset[Stage]
    context: ContextMode
    sources: bool
    whole_text: bool
    on_overflow: OverflowMode
    items: ItemExtractor | None
    max_items: int | None
    backend: SyncBackend | AsyncBackend | None
    violating: Mapping[str, frozenset[str]]
    violation_levels: Mapping[str, int]

    def evaluate(self, answers: Mapping[str, Answer]) -> list[tuple[str, float, Action | None]]:
        """``(question id, violation score, action)`` for each question, in order."""

        results: list[tuple[str, float, Action | None]] = []
        for question_id, question in self.questions.items():
            score = violation_score(
                question,
                answers[question_id],
                violating=self.violating.get(question_id, ()),
                violation_level=self.violation_levels.get(question_id),
            )
            results.append((question_id, score, self.threshold.action(score)))
        return results


SensitiveMode: TypeAlias = Literal["token", "mask", "partial", "hmac", "local", "remove"]


@dataclass(frozen=True, slots=True, repr=False)
class SensitiveHit:
    """A value a sensitive policy found, and how the engine should replace it.

    - ``token``: a conversation token that a reply can restore.
    - ``mask``: ``******``. ``partial``: the first and last two characters.
    - ``hmac``: a keyed hash, stable across conversations.
    - ``local``: a marker for judges only, put back in the reply.
    - ``remove``: ``[REDACTED_<ENTITY>]``, for good.
    """

    span: Span
    entity: str
    mode: SensitiveMode
    action: Action

    def __repr__(self) -> str:
        return (
            f"SensitiveHit(span={self.span!r}, entity={self.entity!r}, "
            f"mode={self.mode!r}, action={self.action!r})"
        )


class SensitivePolicy(abc.ABC):
    """A jes-owned policy that finds values the engine must replace: pii, secrets, canary.

    These policies only report where values are. The engine replaces every standalone
    occurrence of each value, so no backend receives one, and it decides where
    conversation tokens are allowed.
    """

    __slots__ = ()
    phase: ClassVar[Phase] = "detect"
    name: str
    stages: frozenset[Stage]

    @property
    def hmac_key(self) -> bytes | None:
        return None

    @abc.abstractmethod
    def detect(self, text: str, context: TransformContext) -> list[SensitiveHit]: ...


Policy: TypeAlias = Transform | Judgment | SensitivePolicy


def judge(
    name: str,
    questions: Question | Mapping[str, Question],
    *,
    threshold: Threshold | float,
    violating: Collection[str] | Mapping[str, Collection[str]] | None = None,
    violation_level: int | Mapping[str, int] | None = None,
    stages: Iterable[Stage] = ("input", "output"),
    context: ContextMode = "none",
    sources: bool = False,
    whole_text: bool = False,
    on_overflow: OverflowMode = "block",
    items: ItemExtractor | None = None,
    max_items: int | None = None,
    model: ModelSpec | None = None,
) -> Judgment:
    """A judgment that asks your own questions.

    - ``questions``: one question, whose id is ``"violation"``, or a mapping of ids.
    - ``threshold``: block at or above this violation score, or a ``Threshold``.
    - ``violating``: for choice questions, the options that count as a violation.
    - ``violation_level``: for score questions, the lowest level that counts.
    - ``context``: ``"optional"`` adds the prompt (or retrieval question) when it fits,
      and ``"required"`` blocks without it. ``sources`` adds the sources too.
    - ``whole_text``: judge the text in one request instead of in chunks.
    - ``on_overflow``: what to do when the text, context, or items do not fit.
    - ``items``: judge each item this function extracts, such as each URL.
    - ``model``: a backend for this judgment instead of the guard's.
    """

    validate_identifier(name, field="policy name")
    question_map = _question_map(questions)
    stage_set = freeze_stages(stages)
    if context not in ("none", "optional", "required"):
        raise PolicyError("context must be none, optional, or required")
    if context == "required" and stage_set != _OUTPUT_ONLY:
        raise PolicyError("a judgment that requires context runs on output only")
    if on_overflow not in ("block", "allow"):
        raise PolicyError("on_overflow must be block or allow")
    if items is None:
        if max_items is not None:
            raise PolicyError("max_items needs items")
    else:
        if whole_text:
            raise PolicyError("an item judgment cannot be whole_text")
        if max_items is not None and (isinstance(max_items, bool) or max_items < 1):
            raise PolicyError("max_items must be positive")
    return Judgment(
        name=name,
        questions=question_map,
        threshold=Threshold.coerce(threshold),
        stages=stage_set,
        context=context,
        sources=sources,
        whole_text=whole_text,
        on_overflow=on_overflow,
        items=items,
        max_items=max_items,
        backend=resolve_model(model),
        violating=_violating(question_map, violating),
        violation_levels=_levels(question_map, violation_level),
    )


def _question_map(questions: Question | Mapping[str, Question]) -> Mapping[str, Question]:
    if isinstance(questions, (YesNo, Choice, Score)):
        return MappingProxyType({"violation": questions})
    if not questions:
        raise PolicyError("a judgment needs at least one question")
    for question_id in questions:
        validate_identifier(question_id, field="question id")
    return MappingProxyType(dict(questions))


def _violating(
    questions: Mapping[str, Question],
    violating: Collection[str] | Mapping[str, Collection[str]] | None,
) -> Mapping[str, frozenset[str]]:
    choice_ids = [key for key, question in questions.items() if isinstance(question, Choice)]
    if not choice_ids:
        if violating is not None:
            raise PolicyError("violating applies only to choice questions")
        return MappingProxyType({})
    if violating is None:
        raise PolicyError("choice questions need violating options")
    if isinstance(violating, Mapping):
        by_question = cast(Mapping[str, Collection[str]], violating)
        normalized = {key: frozenset(value) for key, value in by_question.items()}
    elif isinstance(violating, str) or len(choice_ids) != 1:
        raise PolicyError("pass a mapping of violating options when there are several choices")
    else:
        normalized = {choice_ids[0]: frozenset(violating)}
    if set(normalized) != set(choice_ids):
        raise PolicyError("violating must name every choice question")
    for key, labels in normalized.items():
        options = cast(Choice, questions[key])
        if not labels or not labels <= set(options.options):
            raise PolicyError("violating options must be non-empty options of the question")
    return MappingProxyType(normalized)


def _levels(
    questions: Mapping[str, Question],
    violation_level: int | Mapping[str, int] | None,
) -> Mapping[str, int]:
    score_ids = [key for key, question in questions.items() if isinstance(question, Score)]
    if not score_ids:
        if violation_level is not None:
            raise PolicyError("violation_level applies only to score questions")
        return MappingProxyType({})
    if violation_level is None:
        raise PolicyError("score questions need a violation_level")
    if isinstance(violation_level, Mapping):
        normalized = dict(violation_level)
    elif len(score_ids) != 1:
        raise PolicyError("pass a mapping of violation levels when there are several scores")
    else:
        normalized = {score_ids[0]: violation_level}
    if set(normalized) != set(score_ids):
        raise PolicyError("violation_level must name every score question")
    for key, level in normalized.items():
        scale = cast(Score, questions[key])
        if isinstance(level, bool) or not 0 <= level < len(scale.levels):
            raise PolicyError("violation_level must name one of the levels")
    return MappingProxyType(normalized)


def merge_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    """Sorted spans with overlapping and touching ones joined."""

    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


__all__ = [
    "PHASES",
    "ContextMode",
    "Item",
    "ItemExtractor",
    "Judgment",
    "OverflowMode",
    "Phase",
    "Policy",
    "SensitiveHit",
    "SensitiveMode",
    "SensitivePolicy",
    "Transform",
    "TransformContext",
    "TransformFinding",
    "TransformOutcome",
    "freeze_stages",
    "judge",
    "merge_spans",
]
