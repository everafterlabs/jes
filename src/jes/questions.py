"""Typed questions, answers, and thresholds."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, TypeVar

from jes.errors import BackendError, PolicyError

ScoreKind: TypeAlias = Literal["probability", "label", "verbalized"]

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_TASK_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SUM_TOLERANCE = 1e-3
_K = TypeVar("_K")
_V = TypeVar("_V")


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    """Validate a bounded public metadata identifier."""

    if _ID_RE.fullmatch(value) is None:
        raise PolicyError(f"invalid {field}: {value!r}")
    return value


def _validate_task(value: str) -> str:
    if _TASK_RE.fullmatch(value) is None:
        raise PolicyError(f"invalid task: {value!r}")
    return value


def _freeze_mapping(value: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(value))


def _validate_unit(value: float, *, field: str, policy: bool = False) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        message = f"{field} must be a finite number in [0, 1]"
        if policy:
            raise PolicyError(message)
        raise BackendError("answer", message)
    return value


def _validate_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    return _validate_unit(value, field="confidence")


@dataclass(frozen=True, slots=True)
class YesNo:
    """A question whose true score means violation."""

    instructions: str
    true: str | None = None
    false: str | None = None
    task: str = "custom"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise PolicyError("yes/no instructions must not be empty")
        _validate_task(self.task)


@dataclass(frozen=True, slots=True)
class Choice:
    """A categorical question."""

    instructions: str
    options: Mapping[str, str | None]
    task: str = "custom"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise PolicyError("choice instructions must not be empty")
        if len(self.options) < 2:
            raise PolicyError("choice questions require at least two options")
        for label in self.options:
            validate_identifier(label, field="choice label")
        _validate_task(self.task)
        object.__setattr__(self, "options", _freeze_mapping(self.options))


@dataclass(frozen=True, slots=True)
class Score:
    """An ordered score question."""

    instructions: str
    levels: tuple[str, ...]
    task: str = "custom"

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise PolicyError("score instructions must not be empty")
        if not 2 <= len(self.levels) <= 10:
            raise PolicyError("score questions require between 2 and 10 levels")
        if len(set(self.levels)) != len(self.levels):
            raise PolicyError("score levels must be unique")
        _validate_task(self.task)


Question: TypeAlias = YesNo | Choice | Score


@dataclass(frozen=True, slots=True)
class YesNoAnswer:
    score: float
    kind: ScoreKind
    confidence: float | None = None

    def __post_init__(self) -> None:
        _validate_unit(self.score, field="yes/no score")
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    scores: Mapping[str, float]
    kind: ScoreKind
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            raise BackendError("answer", "choice answer has no scores")
        checked = {
            key: _validate_unit(value, field=f"choice score {key}")
            for key, value in self.scores.items()
        }
        if not math.isclose(sum(checked.values()), 1.0, abs_tol=_SUM_TOLERANCE):
            raise BackendError("answer", "choice scores do not sum to one")
        _validate_confidence(self.confidence)
        object.__setattr__(self, "scores", _freeze_mapping(checked))

    @property
    def top(self) -> str:
        return max(self.scores, key=self.scores.__getitem__)


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    scores: tuple[float, ...]
    kind: ScoreKind
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            raise BackendError("answer", "score answer has no levels")
        checked = tuple(_validate_unit(value, field="level score") for value in self.scores)
        if not math.isclose(sum(checked), 1.0, abs_tol=_SUM_TOLERANCE):
            raise BackendError("answer", "level scores do not sum to one")
        _validate_confidence(self.confidence)
        object.__setattr__(self, "scores", checked)

    @property
    def expected_level(self) -> float:
        return sum(index * value for index, value in enumerate(self.scores))


Answer: TypeAlias = YesNoAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, slots=True)
class Threshold:
    """Flag and block boundaries for a violation score."""

    block_at: float
    flag_at: float | None = None

    def __post_init__(self) -> None:
        _validate_unit(self.block_at, field="block_at", policy=True)
        if self.flag_at is not None:
            _validate_unit(self.flag_at, field="flag_at", policy=True)
            if self.flag_at > self.block_at:
                raise PolicyError("flag_at must be less than or equal to block_at")

    @classmethod
    def coerce(cls, value: Threshold | float) -> Threshold:
        if isinstance(value, Threshold):
            return value
        return cls(block_at=float(value))


def validate_answer(question: Question, answer: Answer) -> None:
    """Validate that an answer matches its question schema."""

    if isinstance(question, YesNo):
        if not isinstance(answer, YesNoAnswer):
            raise BackendError("answer", "yes/no question returned the wrong answer type")
        return
    if isinstance(question, Choice):
        if not isinstance(answer, ChoiceAnswer):
            raise BackendError("answer", "choice question returned the wrong answer type")
        if set(answer.scores) != set(question.options):
            raise BackendError("answer", "choice answer keys do not match options")
        return
    if not isinstance(answer, ScoreAnswer):
        raise BackendError("answer", "score question returned the wrong answer type")
    if len(answer.scores) != len(question.levels):
        raise BackendError("answer", "score answer length does not match levels")


def violation_score(
    question: Question,
    answer: Answer,
    *,
    violating: Collection[str] | None = None,
    violation_level: int | None = None,
) -> float:
    """Compute the value compared with a policy threshold."""

    validate_answer(question, answer)
    if isinstance(question, YesNo) and isinstance(answer, YesNoAnswer):
        return answer.score
    if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
        if violating is None:
            raise PolicyError("choice questions require violating labels")
        unknown = set(violating) - set(question.options)
        if unknown:
            raise PolicyError("violating labels must be choice options")
        return sum(answer.scores[label] for label in violating)
    if not isinstance(question, Score) or not isinstance(answer, ScoreAnswer):
        raise AssertionError("validated question and answer types diverged")
    if violation_level is None or not 0 <= violation_level < len(question.levels):
        raise PolicyError("score questions require a valid violation_level")
    return sum(answer.scores[violation_level:])
