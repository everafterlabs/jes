"""Questions a backend answers, its answers, and the thresholds that turn answers into actions."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from jes.errors import BackendError, PolicyError
from jes.types import validate_identifier

# Probabilities from a backend may be rounded, so sums within this tolerance count as one.
_SUM_TOLERANCE = 1e-3


def _is_unit(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _check_confidence(confidence: float | None) -> None:
    if confidence is not None and not _is_unit(confidence):
        raise BackendError("answer", "confidence must be a number in [0, 1]")


@dataclass(frozen=True, slots=True)
class Threshold:
    """Block at or above ``block_at``. Flag at or above ``flag_at`` when it is set."""

    block_at: float
    flag_at: float | None = None

    def __post_init__(self) -> None:
        if not _is_unit(self.block_at):
            raise PolicyError("block_at must be a number in [0, 1]")
        object.__setattr__(self, "block_at", float(self.block_at))
        if self.flag_at is not None:
            if not _is_unit(self.flag_at):
                raise PolicyError("flag_at must be a number in [0, 1]")
            if self.flag_at > self.block_at:
                raise PolicyError("flag_at must not be above block_at")
            object.__setattr__(self, "flag_at", float(self.flag_at))

    @classmethod
    def coerce(cls, value: Threshold | float) -> Threshold:
        """A threshold as given, or one that blocks at ``value``."""

        return value if isinstance(value, Threshold) else cls(block_at=value)

    def action(self, score: float) -> Literal["flag", "block"] | None:
        """``"block"``, ``"flag"``, or None for a violation score."""

        if score >= self.block_at:
            return "block"
        if self.flag_at is not None and score >= self.flag_at:
            return "flag"
        return None


def _require_instructions(instructions: str, kind: str) -> None:
    if not instructions.strip():
        raise PolicyError(f"{kind} instructions must not be empty")


@dataclass(frozen=True, slots=True)
class YesNo:
    """A yes/no question. "Yes" means the text violates the policy.

    ``true`` and ``false`` optionally describe what each answer means.
    """

    instructions: str
    true: str | None = None
    false: str | None = None

    def __post_init__(self) -> None:
        _require_instructions(self.instructions, "yes/no")


@dataclass(frozen=True, slots=True)
class Choice:
    """Pick one option. ``options`` maps each label to an optional description."""

    instructions: str
    options: Mapping[str, str | None]

    def __post_init__(self) -> None:
        _require_instructions(self.instructions, "choice")
        if len(self.options) < 2:
            raise PolicyError("a choice question needs at least two options")
        for label in self.options:
            validate_identifier(label, field="choice label")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class Score:
    """An ordered scale of 2 to 10 levels, lowest first."""

    instructions: str
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_instructions(self.instructions, "score")
        levels = tuple(self.levels)
        if not 2 <= len(levels) <= 10:
            raise PolicyError("a score question needs between 2 and 10 levels")
        if len(set(levels)) != len(levels) or not all(level.strip() for level in levels):
            raise PolicyError("score levels must be unique and non-empty")
        object.__setattr__(self, "levels", levels)


Question: TypeAlias = YesNo | Choice | Score


@dataclass(frozen=True, slots=True)
class YesNoAnswer:
    """The probability that the answer is "yes"."""

    score: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not _is_unit(self.score):
            raise BackendError("answer", "a yes/no score must be a number in [0, 1]")
        _check_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A probability for each option. They sum to one."""

    scores: Mapping[str, float]
    confidence: float | None = None

    def __post_init__(self) -> None:
        values = dict(self.scores)
        if not values or not all(_is_unit(value) for value in values.values()):
            raise BackendError("answer", "choice scores must be numbers in [0, 1]")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=_SUM_TOLERANCE):
            raise BackendError("answer", "choice scores must sum to one")
        _check_confidence(self.confidence)
        object.__setattr__(self, "scores", MappingProxyType(values))


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A probability for each level, lowest first. They sum to one."""

    scores: tuple[float, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        values = tuple(self.scores)
        if not values or not all(_is_unit(value) for value in values):
            raise BackendError("answer", "level scores must be numbers in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=_SUM_TOLERANCE):
            raise BackendError("answer", "level scores must sum to one")
        _check_confidence(self.confidence)
        object.__setattr__(self, "scores", values)


Answer: TypeAlias = YesNoAnswer | ChoiceAnswer | ScoreAnswer


def violation_score(
    question: Question,
    answer: Answer,
    *,
    violating: Collection[str] = (),
    violation_level: int | None = None,
) -> float:
    """The probability that the text violates the policy.

    A yes/no question uses P(yes). A choice question sums the ``violating`` options.
    A score question sums ``violation_level`` and every level above it.
    """

    if isinstance(question, YesNo) and isinstance(answer, YesNoAnswer):
        return answer.score
    if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
        if set(answer.scores) != set(question.options):
            raise BackendError("answer", "choice answer labels do not match the options")
        if not violating or not set(violating) <= set(question.options):
            raise PolicyError("violating labels must be a non-empty subset of the options")
        return min(1.0, sum(answer.scores[label] for label in set(violating)))
    if isinstance(question, Score) and isinstance(answer, ScoreAnswer):
        if len(answer.scores) != len(question.levels):
            raise BackendError("answer", "score answer length does not match the levels")
        if (
            violation_level is None
            or isinstance(violation_level, bool)
            or not 0 <= violation_level < len(question.levels)
        ):
            raise PolicyError("violation_level must name one of the levels")
        return min(1.0, sum(answer.scores[violation_level:]))
    raise BackendError("answer", "the answer type does not match the question")


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "Threshold",
    "YesNo",
    "YesNoAnswer",
    "violation_score",
]
