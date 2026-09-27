"""Conversation-scoped PII placeholders and complete-reply restoration."""

import re

from jes import Guard, Redactions
from jes.policies import pii
from jes.testing import FakeBackend
from jes.types import InputResult, ScanResult

_TOKEN = re.compile(r"\[JES_v1_PII_[A-Za-z0-9_-]+\]")


def main() -> tuple[InputResult, ScanResult, InputResult, bytes]:
    backend = FakeBackend()
    store = Redactions(scope=b"conversation-1")
    guard = Guard([pii()], model=backend)
    incoming = guard.check_input("email me at ada@example.com", redactions=store)
    reply = f"I will write to {_token(incoming.onward)}."
    outgoing = guard.check_output(reply, prompt=incoming, redactions=store)
    # The user sees outgoing.onward, with the address restored.
    again = guard.check_input("ada@example.com again", redactions=store)
    blob = store.dumps(b"k" * 32, associated_data=b"cookbook")
    return incoming, outgoing, again, blob


def _token(sanitized: str) -> str:
    match = _TOKEN.search(sanitized)
    if match is None:
        raise RuntimeError("placeholder was not issued")
    return match.group(0)
