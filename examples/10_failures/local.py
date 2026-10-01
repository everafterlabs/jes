"""Lesson 10 (local): When the decision model fails.

on_backend_error picks what a failed check does: raise, block, or allow.
max_input_bytes blocks oversized text before any model call.

Run:   uv run --group examples python -m examples.10_failures.local
Needs: nothing (the backend is down on purpose)
"""

from langchain_typesafe import TypeSafeClassifier

from examples._common import print_check
from jes import Guard
from jes.errors import BackendError
from jes.policies import injection

# Nothing listens on port 9, so every check fails like a backend that is down.
MODEL = TypeSafeClassifier(model="tev1", base_url="http://localhost:9", api_key="ollama", timeout=5)
THRESHOLD = 0.5
POLICIES = [injection(threshold=THRESHOLD)]


def main() -> None:
    print("== raise (the default)")
    try:
        Guard(POLICIES, model=MODEL, on_backend_error="raise").check_input("hello")
    except BackendError as error:
        print(f"  BackendError: {error}")

    # complete=False means the text was not fully checked.
    print("== block")
    blocked = Guard(POLICIES, model=MODEL, on_backend_error="block").check_input("hello")
    print_check("input", blocked)
    print(f"  complete={blocked.complete} ok={blocked.ok}")

    # The decision is "allow", but ok is False, so your code chooses what to do.
    print("== allow")
    allowed = Guard(POLICIES, model=MODEL, on_backend_error="allow").check_input("hello")
    print_check("input", allowed)
    print(f"  complete={allowed.complete} ok={allowed.ok}")

    print("== byte cap")
    capped = Guard(POLICIES, model=MODEL, max_input_bytes=4).check_input("too long for the cap")
    print_check("input", capped)


if __name__ == "__main__":
    main()
