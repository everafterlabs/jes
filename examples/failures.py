"""Backend errors and a byte cap. Incomplete results are not ok."""

from jes import Guard
from jes.errors import BackendError
from jes.policies import injection
from jes.testing import FakeBackend
from jes.types import InputResult


def main() -> tuple[str, InputResult, InputResult, InputResult]:
    empty = FakeBackend()
    try:
        Guard(
            [injection(threshold=0.50)],
            backend=empty,
            on_backend_error="raise",
        ).check_input("hello")
        raised = "no-error"
    except BackendError:
        raised = "raised"
    blocked = Guard(
        [injection(threshold=0.50)],
        backend=FakeBackend(),
        on_backend_error="block",
    ).check_input("hello")
    opened = Guard(
        [injection(threshold=0.50)],
        backend=FakeBackend(),
        on_backend_error="allow",
    ).check_input("hello")
    limited = Guard(
        [injection(threshold=0.50)],
        backend=FakeBackend(answers={}),
        max_input_bytes=4,
    ).check_input("too long for the cap")
    # opened.decision is allow, but opened.onward is the refusal.
    # limited.onward names input_too_long and does not include the text.
    return raised, blocked, opened, limited
