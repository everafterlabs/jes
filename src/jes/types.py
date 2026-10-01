"""Stages, spans, findings, and the other values a check reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from jes.errors import PolicyError

Stage: TypeAlias = Literal["input", "untrusted", "tool_call", "tool_result", "output"]
Action: TypeAlias = Literal["flag", "redact", "block"]
Decision: TypeAlias = Literal["allow", "block"]
Role: TypeAlias = Literal["user", "assistant", "tool"]
Target: TypeAlias = Literal["text", "prompt", "question", "source", "history"]

STAGES: tuple[Stage, ...] = ("input", "untrusted", "tool_call", "tool_result", "output")
ROLES: tuple[Role, ...] = ("user", "assistant", "tool")

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


def validate_identifier(value: str, *, field: str) -> str:
    """Return ``value`` if it is a short ASCII identifier, like every policy name and label."""

    if _IDENTIFIER.fullmatch(value) is None:
        raise PolicyError(f"invalid {field}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open ``[start, end)`` range of character offsets."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("a span needs 0 <= start <= end")


@dataclass(frozen=True, slots=True, repr=False)
class Message:
    """One earlier turn of the conversation, passed as ``history``."""

    role: Role
    text: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError("message role must be user, assistant, or tool")

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, text_len={len(self.text)})"


@dataclass(frozen=True, slots=True, repr=False)
class State:
    """What a backend judges: the text, plus the context its policies asked for."""

    stage: Stage
    text: str
    prompt: str | None = None
    question: str | None = None
    sources: tuple[str, ...] = ()
    history: tuple[Message, ...] = ()
    tool: str | None = None

    def __repr__(self) -> str:
        return (
            f"State(stage={self.stage!r}, text_len={len(self.text)}, "
            f"prompt={self.prompt is not None}, question={self.question is not None}, "
            f"sources={len(self.sources)}, history={len(self.history)}, tool={self.tool!r})"
        )


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One question's violation score: the value its policy compared with the threshold."""

    value: float
    model: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Finding:
    """Something a policy found.

    ``spans`` index into the checked text, or into the context value that ``target``
    names. ``index`` picks the source or history entry.
    """

    policy: str
    label: str
    action: Action
    score: float | None = None
    spans: tuple[Span, ...] = ()
    target: Target = "text"
    index: int | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens one backend request used, when the backend reports them."""

    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


__all__ = [
    "ROLES",
    "STAGES",
    "Action",
    "Decision",
    "Finding",
    "Message",
    "Role",
    "ScoreResult",
    "Span",
    "Stage",
    "State",
    "Target",
    "Usage",
    "validate_identifier",
]
