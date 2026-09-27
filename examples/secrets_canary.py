"""Secrets on input and a canary on output."""

from jes import Guard
from jes.policies import canary, secrets
from jes.testing import FakeBackend
from jes.types import InputResult, ScanResult

_SECRET = "sk-" + ("a" * 20)


def main() -> tuple[InputResult, ScanResult]:
    backend = FakeBackend()
    hidden = Guard([secrets()], model=backend).check_input(f"token {_SECRET}")
    # The model receives hidden.onward, which has no sk- token.
    prompt = Guard([], model=backend).check_input("write a note")
    leaked = Guard(
        [canary("CANARY-TOKEN")],
        model=backend,
        fail_fast=False,
    ).check_output("leaked CANARY-TOKEN here", prompt=prompt)
    # leaked.onward is the refusal. The canary is not in it.
    return hidden, leaked
