"""Lesson 10 (Jev): When the decision model fails.

on_backend_error picks what a failed check does: raise, block, or allow.
max_input_bytes blocks oversized text before any model call. Prints each case.

Run:   uv run --group examples python -m examples.10_failures.jev
Needs: nothing (the backend is down on purpose)
"""

from langchain_typesafe import TypeSafeClassifier

from examples._common import Check, show
from jes import Guard
from jes.errors import BackendError
from jes.policies import injection
from jes.types import InputResult

# Nothing listens on port 9, so every check fails like a backend that is down.
MODEL = TypeSafeClassifier(
    model="jev-latest", base_url="http://localhost:9", api_key="unused", timeout=5
)
THRESHOLD = 0.72  # An application choice; lesson 01 explains it.
POLICIES = [injection(threshold=THRESHOLD)]


def main() -> tuple[str, InputResult, InputResult, InputResult]:
    # raise (the default): check_input raises BackendError. Fail closed, loudly.
    try:
        Guard(POLICIES, model=MODEL, on_backend_error="raise").check_input("hello")
        raised = "no error"
    except BackendError:
        raised = "raised"

    # complete=False means some question went unanswered, so the text was not fully checked.
    # block: decision is "block", so nothing goes on while the backend is down.
    blocked = Guard(POLICIES, model=MODEL, on_backend_error="block").check_input("hello")
    # allow: decision is "allow" but complete=False and ok=False, so your code decides what to do.
    opened = Guard(POLICIES, model=MODEL, on_backend_error="allow").check_input("hello")

    # Text longer than max_input_bytes is blocked before any model call.
    capped = Guard(POLICIES, model=MODEL, max_input_bytes=4).check_input("too long for the cap")

    print(f"== raise\n  {raised}")
    for name, result in (("block", blocked), ("allow", opened), ("byte cap", capped)):
        show(name, [Check(f"input complete={result.complete}", result)])
    return raised, blocked, opened, capped


if __name__ == "__main__":
    main()
