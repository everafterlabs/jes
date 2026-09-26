"""Secrets on input and a canary on output."""

from jes import Guard
from jes.policies import canary, secrets
from jes.testing import FakeBackend
from jes.types import InputResult, ScanResult

_SECRET = "sk-" + ("a" * 20)


def main() -> tuple[InputResult, ScanResult]:
    backend = FakeBackend()
    hidden = Guard([secrets()], backend=backend).check_input(f"token {_SECRET}")
    prompt = Guard([], backend=backend).check_input("write a note")
    leaked = Guard(
        [canary("CANARY-TOKEN")],
        backend=backend,
        fail_fast=False,
    ).check_output("leaked CANARY-TOKEN here", prompt=prompt)
    return hidden, leaked
