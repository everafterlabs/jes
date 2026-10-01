"""The result of one check."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from jes.redactions import Redactions
from jes.types import Decision, Finding, ScoreResult, Stage, Usage

_STAGE_REFUSALS: dict[Stage, str] = {
    "tool_call": "Tool call blocked.",
    "tool_result": "Tool result blocked.",
}


def finding_name(finding: Finding) -> str:
    """How a refusal names a finding: the policy for a one-question judgment, else the label."""

    return finding.policy if finding.label == "violation" else finding.label


def refusal(stage: Stage, findings: Iterable[Finding]) -> str:
    """The text to pass on in place of a blocked one."""

    if stage in _STAGE_REFUSALS:
        return _STAGE_REFUSALS[stage]
    findings = tuple(findings)
    blocking = [item for item in findings if item.action == "block"] or list(findings)
    names = sorted({finding_name(item) for item in blocking})
    return f"Blocked: {', '.join(names)}." if names else "Blocked."


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Result:
    """What a check decided, and the text to pass on.

    ``ok`` is the one flag to act on: the check allowed the text and finished every
    judgment. ``onward`` is then the text to send next, with redactions applied and, for
    a reply, placeholders restored. When ``ok`` is false, ``onward`` is a short refusal.

    ``original`` is the checked text. ``sanitized`` is what the judges saw.
    """

    stage: Stage
    decision: Decision
    complete: bool
    original: str
    sanitized: str
    findings: tuple[Finding, ...]
    scores: Mapping[str, ScoreResult]
    usage: tuple[Usage, ...]
    duration_ms: float
    redactions: Redactions
    _onward: str = field(default="")
    _issuer: object = field(default=None)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def ok(self) -> bool:
        return self.allowed and self.complete

    @property
    def onward(self) -> str:
        return self._onward if self.ok else refusal(self.stage, self.findings)

    def __repr__(self) -> str:
        return (
            f"Result(stage={self.stage!r}, decision={self.decision!r}, "
            f"complete={self.complete}, findings={len(self.findings)}, "
            f"original_len={len(self.original)}, sanitized_len={len(self.sanitized)})"
        )


__all__ = ["Result", "finding_name", "refusal"]
