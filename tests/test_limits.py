from __future__ import annotations

import pytest

from jes import Guard
from jes.policies import Item, judge
from jes.questions import YesNo, YesNoAnswer
from jes.testing import FakeBackend
from jes.types import Span
from tests.helpers import yesno_policy


def test_invalid_surrogate_blocks_without_unicode_encode_error() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend).check_input("ok\ud800bad")
    assert result.decision == "block"
    assert result.complete is False
    assert any(finding.label == "invalid_unicode" for finding in result.findings)


def test_input_byte_limit() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend, max_input_bytes=3).check_input("abcd")
    assert result.decision == "block"
    assert any(finding.label == "input_too_long" for finding in result.findings)


def test_max_findings_reserves_terminal_slot() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.99, "probability")})
    policies = [yesno_policy(f"p{index}") for index in range(3)]
    result = Guard(policies, backend=backend, max_findings=2).check_input("x")
    assert len(result.findings) == 2
    assert result.findings[-1].label == "too_many_findings"
    assert result.complete is False


def test_max_items_and_max_requests() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})

    def many_items(text: str):
        return [
            Item(text=text[index : index + 1], span=Span(index, index + 1))
            for index in range(len(text))
        ]

    policy = judge("items", YesNo("item?"), threshold=0.8, items=many_items)
    items = Guard([policy], backend=backend, max_items=2).check_input("abcd")
    assert items.complete is False
    assert any(finding.label == "too_many_items" for finding in items.findings)

    from jes.errors import PolicyError

    with pytest.raises(PolicyError):
        Guard([yesno_policy()], backend=backend, max_requests=0)


def test_max_requests_blocks_before_io() -> None:
    backend = FakeBackend(max_units=400, answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard(
        [yesno_policy()],
        backend=backend,
        max_chunks=8,
        max_requests=1,
    ).check_input("word " * 120)
    if result.complete:
        pytest.skip("payload fit in one request")
    assert result.complete is False
    labels = {finding.label for finding in result.findings}
    assert labels & {"too_many_requests", "input_too_long"}


def test_guard_busy() -> None:
    backend = FakeBackend(answers={"violation": YesNoAnswer(0.0, "probability")})
    guard = Guard([yesno_policy()], backend=backend, max_active_checks=1)
    with guard.admission.check():
        result = guard.check_input("x")
    assert result.decision == "block"
    assert any(finding.label == "guard_busy" for finding in result.findings)


def test_max_chunks_overflow() -> None:
    backend = FakeBackend(max_units=400, answers={"violation": YesNoAnswer(0.0, "probability")})
    result = Guard([yesno_policy()], backend=backend, max_chunks=1).check_input("abcdefghij " * 80)
    assert result.complete is False
