"""Results, refusals, and how findings are named."""

from __future__ import annotations

from jes.redactions import Redactions
from jes.result import Result, finding_name, refusal
from jes.types import Finding, Stage


def _result(
    stage: Stage = "input",
    *,
    decision: str = "allow",
    complete: bool = True,
    findings: tuple[Finding, ...] = (),
) -> Result:
    return Result(
        stage=stage,
        decision=decision,  # type: ignore[arg-type]
        complete=complete,
        original="my card is 4111 1111 1111 1111",
        sanitized="my card is [REDACTED]",
        findings=findings,
        scores={},
        usage=(),
        duration_ms=1.0,
        redactions=Redactions(),
        _onward="my card is [REDACTED]",
    )


def test_ok_results_pass_on_their_onward_text() -> None:
    result = _result()
    assert result.ok and result.allowed
    assert result.onward == "my card is [REDACTED]"
    assert "4111" not in repr(result)
    assert "original_len=30" in repr(result)


def test_incomplete_or_blocked_results_pass_on_a_refusal() -> None:
    backend = Finding("jes", "backend_error", "flag")
    incomplete = _result(complete=False, findings=(backend,))
    assert incomplete.allowed and not incomplete.ok
    assert incomplete.onward == "Blocked: backend_error."

    injection = Finding("injection", "violation", "block")
    flagged = Finding("toxicity", "insult", "flag")
    blocked = _result(decision="block", findings=(flagged, injection))
    assert blocked.onward == "Blocked: injection."
    assert _result("tool_call", decision="block").onward == "Tool call blocked."
    assert _result("tool_result", decision="block").onward == "Tool result blocked."
    assert _result(decision="block").onward == "Blocked."


def test_finding_names() -> None:
    assert finding_name(Finding("injection", "violation", "block")) == "injection"
    assert finding_name(Finding("hazards", "S1", "block")) == "S1"
    findings = [Finding("hazards", "S2", "block"), Finding("hazards", "S1", "block")]
    assert refusal("output", findings) == "Blocked: S1, S2."
