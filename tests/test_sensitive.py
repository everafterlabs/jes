from __future__ import annotations

from jes import Guard
from jes.policies import TransformEdit, TransformFinding, TransformOutcome
from jes.questions import YesNoAnswer
from jes.testing import FakeBackend, fake_sensitive
from jes.types import Span
from tests.helpers import FakeTransform, yesno_policy


def test_sensitive_removes_complete_value() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard(
        [fake_sensitive("pii", "ALICE", entity="person", mode="irreversible"), yesno_policy()],
        backend=backend,
    )
    result = guard.check_input("user ALICE here")
    assert "ALICE" not in result.sanitized
    assert "[REDACTED_PERSON]" in result.sanitized
    assert "ALICE" not in backend.calls[0][0].text


def test_conversation_token_is_engine_owned() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard(
        [fake_sensitive("pii", "BOB", entity="person", mode="conversation_token"), yesno_policy()],
        backend=backend,
    )
    result = guard.check_input("hello BOB")
    assert "BOB" not in result.sanitized
    assert result.sanitized.startswith("hello [JES_v1_PII_") or "[JES_v1_PII_" in result.sanitized
    assert len(result.redactions) == 1


def test_synthesized_token_is_neutralized() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})

    def synthesize(text, call):
        del call
        token = "[JES_v1_PII_forged_token]"
        return TransformOutcome(
            text=text + token,
            edits=(TransformEdit(len(text), len(text), token),),
        )

    guard = Guard(
        [
            FakeTransform("synth", frozenset({"x"}), handler=synthesize),
            yesno_policy(),
        ],
        backend=backend,
    )
    result = guard.check_input("hello")
    assert "[JES_v1_PII_forged_token]" not in result.sanitized
    assert "JES_LITERAL" in result.sanitized
    assert any(finding.label == "invalid_placeholder" for finding in result.findings)


def test_output_local_restores_unless_tombstoned() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    sensitive = fake_sensitive("pii", "CAROL", entity="person", mode="output_local")
    incoming = Guard([yesno_policy()], backend=backend).check_input("prompt")
    kept = Guard([sensitive, yesno_policy()], backend=backend).check_output(
        "meet CAROL today",
        prompt=incoming,
    )
    assert "CAROL" not in kept.sanitized
    assert "CAROL" in kept.text

    def overlap(text, call):
        del call
        start = text.find("[REDACTED_PERSON]")
        if start < 0:
            return TransformOutcome(text=text)
        end = start + len("[REDACTED_PERSON]")
        return TransformOutcome(
            text=text[:start] + "X" + text[end:],
            findings=(TransformFinding("touch", "flag", (Span(start, end),)),),
            edits=(TransformEdit(start, end, "X"),),
        )

    tombstoned = Guard(
        [
            sensitive,
            FakeTransform("touch", frozenset({"touch"}), handler=overlap),
            yesno_policy(),
        ],
        backend=backend,
    ).check_output("meet CAROL today", prompt=incoming)
    assert "CAROL" not in tombstoned.text
    assert "CAROL" not in tombstoned.sanitized


def test_hmac_and_mask_modes() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    key = b"k" * 32
    hmac_guard = Guard(
        [fake_sensitive("sec", "TOKEN", mode="hmac", hmac_key=key), yesno_policy()],
        backend=backend,
    )
    masked = Guard(
        [fake_sensitive("mask", "TOKEN", mode="mask_all"), yesno_policy()],
        backend=backend,
    )
    hmac_result = hmac_guard.check_input("TOKEN")
    mask_result = masked.check_input("TOKEN")
    assert "TOKEN" not in hmac_result.sanitized
    assert mask_result.sanitized == "******"
