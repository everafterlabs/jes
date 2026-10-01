"""Value types, identifiers, errors, and the package surface."""

from __future__ import annotations

from importlib.metadata import version

import pytest

import jes
from jes.errors import BackendError, DeadlineExceeded, JesError, PolicyError
from jes.types import Finding, Message, Span, State, validate_identifier


def test_package_version_comes_from_the_installed_metadata() -> None:
    assert jes.__version__ == version("jes")
    for name in jes.__all__:
        assert hasattr(jes, name), name


def test_span_rejects_negative_and_reversed_ranges() -> None:
    assert Span(0, 0) == Span(0, 0)
    for start, end in ((-1, 2), (3, 2)):
        with pytest.raises(ValueError, match="span"):
            Span(start, end)


def test_message_and_state_reprs_hide_text() -> None:
    message = Message("user", "my card is 4111 1111 1111 1111")
    assert "4111" not in repr(message)
    assert repr(message) == "Message(role='user', text_len=30)"
    state = State(stage="input", text="secret text", prompt="secret prompt", history=(message,))
    assert "secret" not in repr(state)
    assert "text_len=11" in repr(state)
    with pytest.raises(ValueError, match="role"):
        Message("system", "hi")  # type: ignore[arg-type]


def test_finding_defaults_to_the_checked_text() -> None:
    finding = Finding("pii", "EMAIL_ADDRESS", "redact", spans=(Span(1, 4),))
    assert finding.target == "text"
    assert finding.index is None
    assert finding.score is None


@pytest.mark.parametrize("value", ["a", "injection", "EMAIL_ADDRESS", "S14", "zh-Hans", "x" * 64])
def test_identifiers_accept_short_ascii_names(value: str) -> None:
    assert validate_identifier(value, field="label") == value


@pytest.mark.parametrize("value", ["", "1abc", "_x", "has space", "dot.name", "x" * 65, "é"])
def test_identifiers_reject_everything_else(value: str) -> None:
    with pytest.raises(PolicyError, match="invalid label"):
        validate_identifier(value, field="label")


def test_backend_errors_carry_metadata_only() -> None:
    error = BackendError("typesafe", "rate_limited", status_code=429, question_ids=["a", "b"])
    assert isinstance(error, JesError)
    assert str(error) == (
        "backend error (backend=typesafe; reason=rate_limited; status=429; questions=a,b)"
    )
    assert error.question_ids == ("a", "b")
    deadline = DeadlineExceeded("typesafe")
    assert isinstance(deadline, BackendError)
    assert deadline.reason == "deadline_exceeded"
    assert str(deadline) == "backend error (backend=typesafe; reason=deadline_exceeded)"
