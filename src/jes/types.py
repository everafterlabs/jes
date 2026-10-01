"""Public result and backend state types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Never, SupportsIndex, TypeAlias, TypeVar

from jes.errors import PolicyExecutionError
from jes.questions import ScoreKind

if TYPE_CHECKING:
    from jes.redactions import Redactions

Stage: TypeAlias = Literal["input", "untrusted", "output", "tool_call", "tool_result"]
Action: TypeAlias = Literal["flag", "redact", "block"]
Decision: TypeAlias = Literal["allow", "block"]
Role: TypeAlias = Literal["user", "assistant", "tool"]
LocationTarget: TypeAlias = Literal["subject", "prompt", "question", "source", "history"]

_K = TypeVar("_K")
_V = TypeVar("_V")


def _freeze_mapping(value: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise PolicyExecutionError("invalid span")


@dataclass(frozen=True, slots=True)
class ThresholdProvenance:
    source: Literal["explicit", "default"]
    block_at: float
    flag_at: float | None
    fingerprint: str
    vector_fingerprint: str
    evaluation_run: str | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    backend: str
    model: str
    request_profile: str
    decision_profile: str
    prompt_version: str
    threshold: ThresholdProvenance


@dataclass(frozen=True, slots=True)
class ScoreResult:
    value: float
    kind: ScoreKind
    confidence: float | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class SanitizationStamp:
    result_id: str
    stage: Stage
    config_digest: str
    text_digest: str
    store_id: str | None
    scope_id: str | None
    decision: Decision
    complete: bool
    findings_digest: str
    authority_digest: str
    tag: str

    @classmethod
    def empty(cls, stage: Stage) -> SanitizationStamp:
        return cls(
            result_id="",
            stage=stage,
            config_digest="",
            text_digest="",
            store_id=None,
            scope_id=None,
            decision="block",
            complete=False,
            findings_digest="",
            authority_digest="",
            tag="",
        )


@dataclass(frozen=True, slots=True, repr=False)
class _TokenAuthority:
    token: str
    span: Span
    origin_result: str
    reusable: bool

    def __repr__(self) -> str:
        return f"_TokenAuthority(span={self.span!r}, reusable={self.reusable})"


@dataclass(frozen=True, slots=True, repr=False)
class _TokenAuthorityManifest:
    entries: tuple[_TokenAuthority, ...] = ()
    digest: str = ""

    def __repr__(self) -> str:
        return f"_TokenAuthorityManifest(entries={len(self.entries)})"

    def __copy__(self) -> _TokenAuthorityManifest:
        raise TypeError("token authority manifests cannot be copied")

    def __deepcopy__(self, memo: object) -> _TokenAuthorityManifest:
        del memo
        raise TypeError("token authority manifests cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("token authority manifests cannot be pickled")


_EMPTY_AUTHORITY = _TokenAuthorityManifest()


@dataclass(frozen=True, slots=True)
class FindingLocation:
    target: LocationTarget
    span: Span
    index: int | None = None
    role: Role | None = None
    item_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.index is not None and self.index < 0:
            raise PolicyExecutionError("finding location index must be non-negative")
        if self.item_ordinal is not None and self.item_ordinal < 0:
            raise PolicyExecutionError("item ordinal must be non-negative")


@dataclass(frozen=True, slots=True)
class Finding:
    policy: str
    label: str
    action: Action
    question: str | None = None
    score: ScoreResult | None = None
    locations: tuple[FindingLocation, ...] = ()
    chunks: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    backend: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    request: int
    attempt: int


@dataclass(frozen=True, slots=True)
class Timings:
    transforms_ms: float
    judgments_ms: float
    finalizers_ms: float


@dataclass(frozen=True, slots=True, repr=False)
class Message:
    role: Role
    text: str

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, text_len={len(self.text)})"


@dataclass(frozen=True, slots=True, repr=False)
class State:
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
            f"sources={len(self.sources)}, history={len(self.history)}, "
            f"tool={self.tool is not None})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ScanResult:
    stage: Stage
    text: str
    sanitized: str
    decision: Decision
    complete: bool
    findings: tuple[Finding, ...]
    scores: Mapping[str, ScoreResult]
    sanitization: SanitizationStamp
    usage: tuple[Usage, ...] = ()
    timings: Timings | None = None
    _authority: _TokenAuthorityManifest = field(
        default=_EMPTY_AUTHORITY,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "scores", _freeze_mapping(self.scores))
        object.__setattr__(self, "usage", tuple(self.usage))

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def ok(self) -> bool:
        return self.allowed and self.complete

    @property
    def onward(self) -> str:
        """Text safe to pass to the next hop. A refusal when the check is not ok."""

        if self.ok:
            return self.text if self.stage == "output" else self.sanitized
        if self.stage == "tool_call":
            return "Tool call blocked."
        if self.stage == "tool_result":
            return "Tool result blocked."
        names = sorted(
            {
                finding.policy if finding.label == "violation" else finding.label
                for finding in self.findings
            }
        )
        if not names:
            return "Blocked."
        return f"Blocked: {', '.join(names)}."

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(stage={self.stage!r}, decision={self.decision!r}, "
            f"complete={self.complete}, text_len={len(self.text)}, "
            f"sanitized_len={len(self.sanitized)}, findings={len(self.findings)}, "
            f"scores={len(self.scores)}, usage={len(self.usage)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class InputResult(ScanResult):
    redactions: Redactions = field(kw_only=True)

    def __copy__(self) -> InputResult:
        raise TypeError("InputResult cannot be copied")

    def __deepcopy__(self, memo: object) -> InputResult:
        del memo
        raise TypeError("InputResult cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise TypeError("InputResult cannot be pickled")


History: TypeAlias = InputResult | ScanResult | Message
